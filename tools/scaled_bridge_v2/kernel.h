// Scaled bridge v2 reduction kernel, shared by the C++ and CUDA backends.
// One output element is one sequential reduction in input-channel,
// kernel-row, kernel-column order. The accumulator is a policy type with
// step(a,b) and finish(); new policies are added here and in the dispatch
// tables of native.cpp/native.cu without touching the reduction or the graph
// code. Operands are code-grid integers; the caller certifies the ranges.
// Policies 2 and 3 take parameters and report a per-output event word.
#include <cstdint>
struct Shape { int n,ci,h,w,co,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw,groups; };

// Round-to-nearest-even conversion of sign * (hi*2^64 + lo) to binary64.
// Exact whenever the magnitude is below 2^53.
PE_DEVICE double rne_magnitude(bool negative, uint64_t hi, uint64_t lo) {
    if(hi==0 && lo==0) return 0.0;
    int top = hi ? 127-PE_CLZ64(hi) : 63-PE_CLZ64(lo);
    uint64_t mantissa=lo; int drop=0;
    if(top>52) {
        drop=top-52; bool round_bit, sticky;
        if(drop<64) {
            mantissa=(hi<<(64-drop))|(lo>>drop);
            round_bit=(lo>>(drop-1))&1; sticky=(lo&((uint64_t(1)<<(drop-1))-1))!=0;
        } else if(drop==64) {
            mantissa=hi; round_bit=(lo>>63)&1; sticky=(lo<<1)!=0;
        } else {
            int d=drop-64; mantissa=hi>>d;
            round_bit=(hi>>(d-1))&1; sticky=((hi&((uint64_t(1)<<(d-1))-1))!=0)||lo!=0;
        }
        if(round_bit && (sticky || (mantissa&1))) ++mantissa;
    }
    double value=PE_LDEXP(double(mantissa),drop);
    return negative ? -value : value;
}

// Policy 0, narrow operands: exact signed 64-bit sum (certified |prefix| < 2^63).
struct WideI64 {
    int64_t sum;
    PE_DEVICE void start() { sum=0; }
    PE_DEVICE void step(int64_t a,int64_t b) { sum+=a*b; }
    PE_DEVICE double finish() const {
        bool negative=sum<0; uint64_t magnitude=negative ? uint64_t(0)-uint64_t(sum) : uint64_t(sum);
        return rne_magnitude(negative,0,magnitude);
    }
};
// Policy 0, wide operands: exact two-limb sum. |a|,|b| < 2^32 so each product
// magnitude fits one unsigned limb; certified |prefix| < 2^127.
struct WideI128 {
    uint64_t lo; int64_t hi;
    PE_DEVICE void start() { lo=0; hi=0; }
    PE_DEVICE void step(int64_t a,int64_t b) {
        bool negative=(a<0)!=(b<0);
        uint64_t ua=a<0 ? uint64_t(0)-uint64_t(a) : uint64_t(a);
        uint64_t ub=b<0 ? uint64_t(0)-uint64_t(b) : uint64_t(b);
        uint64_t product=ua*ub, old=lo;
        if(negative) { lo-=product; hi-=(old<product); }
        else { lo+=product; hi+=(lo<old); }
    }
    PE_DEVICE double finish() const {
        bool negative=hi<0; uint64_t h=uint64_t(hi), l=lo;
        if(negative) { l=~l+1; h=~h+(l==0); }
        return rne_magnitude(negative,h,l);
    }
};
// Policy 1: sequential binary32 RNE fused multiply-add from +0. Operands are
// exactly representable in binary32 (checked by the host); integer-grid FMA is
// power-of-two equivariant to code-level FMA in the admitted range.
struct FmaF32 {
    float sum;
    PE_DEVICE void start() { sum=0.0f; }
    PE_DEVICE void step(int64_t a,int64_t b) { sum=PE_FMA(float(a),float(b),sum); }
    PE_DEVICE double finish() const { return double(sum); }
};

// Policy 2: signed saturating integer of width W (2..63) on the product grid.
// After every add the exact sum is clamped to [-2^(W-1), 2^(W-1)-1]; it never
// wraps. up/down count the adds whose exact sum lay outside the range.
struct SatInt {
    int64_t sum, lo, hi; uint32_t up, down;
    PE_DEVICE void start(const int64_t* p) { hi=(int64_t(1)<<(p[0]-1))-1; lo=-hi-1; sum=0; up=0; down=0; }
    PE_DEVICE void step(int64_t a,int64_t b) {
        bool negative=(a<0)!=(b<0);
        uint64_t ua=a<0 ? uint64_t(0)-uint64_t(a) : uint64_t(a);
        uint64_t ub=b<0 ? uint64_t(0)-uint64_t(b) : uint64_t(b);
        uint64_t product=ua*ub;
        if(product==0) return;
        if(!negative) {
            uint64_t room=uint64_t(hi-sum);
            if(product>room) { sum=hi; if(up<65535u) ++up; } else sum+=int64_t(product);
        } else {
            uint64_t room=uint64_t(sum-lo);
            if(product>room) { sum=lo; if(down<65535u) ++down; } else sum-=int64_t(product);
        }
    }
    PE_DEVICE double finish() const {
        bool negative=sum<0; uint64_t magnitude=negative ? uint64_t(0)-uint64_t(sum) : uint64_t(sum);
        return rne_magnitude(negative,0,magnitude);
    }
    PE_DEVICE uint32_t events() const { return up|(down<<16); }
};
// Policy 3: binary floating-point accumulator with P significand bits, one
// round-to-nearest-even after every exact multiply-add, gradual subnormals and
// overflow to infinity. All arithmetic is exact integer arithmetic in product
// grid units: the state is sign * m * 2^e units with m < 2^P. emin is the unit
// exponent of the subnormal spacing, emax the unit exponent of the leading bit
// of the largest binade. Products are finite, so NaN is unreachable and an
// infinite accumulator stays infinite. bad counts the steps that end infinite.
struct SoftFloat {
    uint64_t m; int e, P, emin, emax; bool neg, inf; uint32_t bad;
    PE_DEVICE void start(const int64_t* p) { P=int(p[0]); emin=int(p[1]); emax=int(p[2]); m=0; e=0; neg=false; inf=false; bad=0; }
    PE_DEVICE void round_in(bool negative,uint64_t hi,uint64_t lo) {
        if(!(hi|lo)) { m=0; e=0; neg=false; return; }
        int t=hi ? 127-PE_CLZ64(hi) : 63-PE_CLZ64(lo);
        int er=t-P+1; if(er<emin) er=emin;
        uint64_t mm;
        if(er<=0) { mm=lo; er=0; }
        else {
            bool round_bit, sticky;
            if(er<64) {
                mm=(hi<<(64-er))|(lo>>er);
                round_bit=(lo>>(er-1))&1; sticky=(lo&((uint64_t(1)<<(er-1))-1))!=0;
            } else if(er==64) {
                mm=hi; round_bit=(lo>>63)&1; sticky=(lo<<1)!=0;
            } else {
                int d=er-64; mm=hi>>d;
                round_bit=(hi>>(d-1))&1; sticky=((hi&((uint64_t(1)<<(d-1))-1))!=0)||lo!=0;
            }
            if(round_bit && (sticky || (mm&1))) ++mm;
            if(mm==(uint64_t(1)<<P)) { mm>>=1; ++er; }
            if(mm==0) { m=0; e=0; neg=false; return; }
        }
        if(63-PE_CLZ64(mm)+er>emax) { inf=true; neg=negative; m=0; e=0; return; }
        m=mm; e=er; neg=negative;
    }
    PE_DEVICE void step(int64_t a,int64_t b) {
        if(inf) { ++bad; return; }
        bool negative=(a<0)!=(b<0);
        uint64_t ua=a<0 ? uint64_t(0)-uint64_t(a) : uint64_t(a);
        uint64_t ub=b<0 ? uint64_t(0)-uint64_t(b) : uint64_t(b);
        uint64_t product=ua*ub;
        if(product==0) return;
        if(m==0) { round_in(negative,0,product); if(inf) ++bad; return; }
        // |state| >= 2^66 units and normalised: |product| < 2^64 is below a
        // quarter of the spacing, so the rounded sum is the state itself.
        if(e>=66) return;
        uint64_t hi, lo;
        if(e==0) { hi=0; lo=m; } else if(e<64) { hi=m>>(64-e); lo=m<<e; } else { hi=m<<(e-64); lo=0; }
        if(negative==neg) {
            uint64_t old=lo; lo+=product; hi+=(lo<old);
            round_in(neg,hi,lo);
        } else if(hi || lo>=product) {
            uint64_t old=lo; lo-=product; hi-=(old<product);
            round_in(neg,hi,lo);
        } else round_in(negative,0,product-lo);
        if(inf) ++bad;
    }
    PE_DEVICE double finish() const {
        if(inf) return neg ? -PE_INF : PE_INF;
        double value=PE_LDEXP(double(m),e);
        return neg ? -value : value;
    }
    PE_DEVICE uint32_t events() const { return bad; }
};

template<class T,class Acc>
PE_DEVICE double reduce_stat(const T* x,const T* w,Shape g,int64_t output,int shift,const int64_t* params,uint32_t* events) {
    int64_t idx=output;
    int ox=idx%g.ow; idx/=g.ow;
    int oy=idx%g.oh; idx/=g.oh;
    int oc=idx%g.co, n=idx/g.co;
    int cg=g.ci/g.groups, first=(oc/(g.co/g.groups))*cg;
    Acc acc; acc.start(params);
    for(int c=0;c<cg;++c) for(int ky=0;ky<g.kh;++ky) for(int kx=0;kx<g.kw;++kx) {
        int iy=oy*g.sh-g.ph+ky*g.dh, ix=ox*g.sw-g.pw+kx*g.dw;
        T a=0;
        if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)
            a=x[((int64_t(n)*g.ci+first+c)*g.h+iy)*g.w+ix];
        T b=w[((int64_t(oc)*cg+c)*g.kh+ky)*g.kw+kx];
        acc.step(int64_t(a),int64_t(b));
    }
    events[output]=acc.events();
    return PE_LDEXP(acc.finish(),-shift);
}

template<class T,class Acc>
PE_DEVICE double reduce_one(const T* x,const T* w,Shape g,int64_t output,int shift) {
    int64_t idx=output;
    int ox=idx%g.ow; idx/=g.ow;
    int oy=idx%g.oh; idx/=g.oh;
    int oc=idx%g.co, n=idx/g.co;
    int cg=g.ci/g.groups, first=(oc/(g.co/g.groups))*cg;
    Acc acc; acc.start();
    for(int c=0;c<cg;++c) for(int ky=0;ky<g.kh;++ky) for(int kx=0;kx<g.kw;++kx) {
        int iy=oy*g.sh-g.ph+ky*g.dh, ix=ox*g.sw-g.pw+kx*g.dw;
        T a=0;
        if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)
            a=x[((int64_t(n)*g.ci+first+c)*g.h+iy)*g.w+ix];
        T b=w[((int64_t(oc)*cg+c)*g.kh+ky)*g.kw+kx];
        acc.step(int64_t(a),int64_t(b));
    }
    return PE_LDEXP(acc.finish(),-shift);
}
