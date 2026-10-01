#include <cstdint>
struct Shape { int n,ci,h,w,co,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw,groups; };
PE_DEVICE double reduce_one(const int16_t* x, const int16_t* w, Shape g,
                            int64_t output, int shift, int mode) {
    int64_t idx=output;
    int ox=idx%g.ow; idx/=g.ow;
    int oy=idx%g.oh; idx/=g.oh;
    int oc=idx%g.co, n=idx/g.co;
    int cg=g.ci/g.groups, first=(oc/(g.co/g.groups))*cg;
    int64_t wide=0;
    float control=0.0f;
    for(int c=0;c<cg;++c) for(int ky=0;ky<g.kh;++ky) for(int kx=0;kx<g.kw;++kx) {
        int iy=oy*g.sh-g.ph+ky*g.dh, ix=ox*g.sw-g.pw+kx*g.dw;
        int16_t a=0;
        if(iy>=0 && iy<g.h && ix>=0 && ix<g.w)
            a=x[((int64_t(n)*g.ci+first+c)*g.h+iy)*g.w+ix];
        int16_t b=w[((int64_t(oc)*cg+c)*g.kh+ky)*g.kw+kx];
        if(mode==0) wide += int64_t(a)*b;
        else control=PE_FMA(float(a),float(b),control);
    }
    return PE_LDEXP(mode==0 ? double(wide) : double(control), -2*shift);
}
