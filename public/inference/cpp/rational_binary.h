#pragma once
// Exact rational-domain Model C. Operands have common denominators Dx, Dw;
// D=lcm(Dx*Dw, minimum accumulator quantum denominator). Integer products,
// decoded accumulator values and every tie decision are exact in this grid.
#include "exact_binary.h"
#include <cstdint>

#ifdef __CUDACC__
#define PE_WIDE __host__ __device__ __noinline__
#else
#define PE_WIDE inline
#endif

namespace pe {
struct RationalFormat { int32_t bits, mantissa, exponent, bias, integer, fractional, product_shift, size; };
template<int S> struct Wide {
    uint32_t word[S]{};
    PE_HD static Wide read(const uint32_t* data) { Wide a; for(int i=0;i<S;++i)a.word[i]=data[i]; return a; }
    PE_WIDE int high() const { for(int i=S-1;i>=0;--i)if(word[i]){for(int b=31;b>=0;--b)if((word[i]>>b)&1u)return i*32+b;}return -1; }
    PE_WIDE int compare(const Wide& b) const { for(int i=S-1;i>=0;--i)if(word[i]!=b.word[i])return word[i]>b.word[i]?1:-1;return 0; }
    PE_WIDE Wide shift(int bits) const {
        Wide out;
        if(bits>=0){int q=bits/32,r=bits%32;for(int i=0;i<S-q;++i){out.word[i+q]|=word[i]<<r;if(r&&i+q+1<S)out.word[i+q+1]|=word[i]>>(32-r);}}
        else {int q=(-bits)/32,r=(-bits)%32;for(int i=0;i<S-q;++i){out.word[i]=word[i+q]>>r;if(r&&i+q+1<S)out.word[i]|=word[i+q+1]<<(32-r);}}
        return out;
    }
    PE_HD Wide plus(const Wide& b) const {Wide out;uint64_t carry=0;for(int i=0;i<S;++i){uint64_t p=uint64_t(word[i])+b.word[i]+carry;out.word[i]=uint32_t(p);carry=p>>32;}return out;}
    PE_HD Wide minus(const Wide& b) const {Wide out;uint64_t borrow=0;for(int i=0;i<S;++i){uint64_t p=uint64_t(b.word[i])+borrow;out.word[i]=uint32_t(uint64_t(word[i])-p);borrow=uint64_t(word[i])<p;}return out;}
    PE_WIDE Wide times(const Wide& b) const {
        Wide out;int an=high()/32+1,bn=b.high()/32+1;
        for(int i=0;i<an;++i){uint64_t carry=0;int j=0;for(;j<bn&&i+j<S;++j){uint64_t p=uint64_t(word[i])*b.word[j]+out.word[i+j]+carry;out.word[i+j]=uint32_t(p);carry=p>>32;}
            while(carry&&i+j<S){uint64_t p=uint64_t(out.word[i+j])+carry;out.word[i+j]=uint32_t(p);carry=p>>32;++j;}}
        return out;
    }
    PE_HD Wide times_small(uint64_t b) const {Wide value;value.word[0]=uint32_t(b);value.word[1]=uint32_t(b>>32);return times(value);}
};

struct Quotient {uint64_t value;bool overflow;};
template<int S> PE_WIDE Quotient divide_round(Wide<S> numerator,const Wide<S>& denominator){
    int h=numerator.high()-denominator.high();
    if(h>63)return {0,true};
    uint64_t q=0;
    for(int i=h;i>=0;--i){Wide<S> d=denominator.shift(i);if(numerator.compare(d)>=0){numerator=numerator.minus(d);q|=uint64_t(1)<<i;}}
    int tie=numerator.shift(1).compare(denominator);
    if(tie>0||(tie==0&&(q&1))){if(q==~uint64_t(0))return {0,true};++q;}
    return {q,false};
}

PE_HD inline uint64_t rational_special(int kind,RationalFormat f){
    uint64_t inf=((uint64_t(1)<<f.exponent)-1)<<f.mantissa,sign=uint64_t(1)<<(f.bits-1);
    return kind==3?inf|1:kind==2?sign|inf:kind==4?sign:inf;
}

template<int S> PE_HD uint64_t rational_round(const Wide<S>& magnitude,bool negative,const Wide<S>& denominator,RationalFormat f){
    uint64_t sign=negative?uint64_t(1)<<(f.bits-1):0;
    if(f.integer){
        Quotient q=divide_round(magnitude.shift(f.fractional),denominator);
        uint64_t limit=uint64_t(1)<<(f.bits-1),mask=f.bits==64?~uint64_t(0):(uint64_t(1)<<f.bits)-1;
        uint64_t bound=negative?limit:limit-1;
        if(q.overflow||q.value>bound)q.value=bound;
        return (negative?uint64_t(0)-q.value:q.value)&mask;
    }
    if(magnitude.high()<0)return sign;
    int exponent=magnitude.high()-denominator.high();
    if((exponent>=0?magnitude.compare(denominator.shift(exponent)):magnitude.shift(-exponent).compare(denominator))<0)--exponent;
    int minimum=1-f.bias,step=(exponent<minimum?minimum:exponent)-f.mantissa;
    Quotient q=step>=0?divide_round(magnitude,denominator.shift(step)):divide_round(magnitude.shift(-step),denominator);
    if(exponent<minimum)return sign|q.value;
    if(q.value==(uint64_t(1)<<(f.mantissa+1))){++exponent;q.value>>=1;}
    int encoded=exponent+f.bias;
    if(q.overflow||encoded>=(1<<f.exponent)-1)return rational_special(negative?2:1,f);
    return sign|(uint64_t(encoded)<<f.mantissa)|(q.value-(uint64_t(1)<<f.mantissa));
}

template<int S> PE_HD Wide<S> rational_decode(uint64_t code,const Wide<S>& denominator,RationalFormat f,bool& negative,int& kind){
    negative=(code>>(f.bits-1))!=0;kind=0;
    uint64_t mantissa;int exponent;
    if(f.integer){uint64_t mask=f.bits==64?~uint64_t(0):(uint64_t(1)<<f.bits)-1;mantissa=negative?((~code)+1)&mask:code;exponent=-f.fractional;}
    else {
        uint64_t fraction=code&((uint64_t(1)<<f.mantissa)-1),e=(code>>f.mantissa)&((uint64_t(1)<<f.exponent)-1);
        if(e==(uint64_t(1)<<f.exponent)-1){kind=fraction?3:(negative?2:1);return {};}
        if(!e&&!fraction){kind=negative?4:0;return {};}
        mantissa=e?(uint64_t(1)<<f.mantissa)+fraction:fraction;
        exponent=(e?int(e):1)-f.bias-f.mantissa;
    }
    return denominator.times_small(mantissa).shift(exponent);
}

template<int S> PE_HD uint64_t rational_fma(uint64_t state,const uint32_t* x,const uint32_t* w,const Wide<S>& denominator,RationalFormat f){
    bool old_negative=false;int old_kind=0;
    Wide<S> old=rational_decode(state,denominator,f,old_negative,old_kind);
    int xkind=x[1],wkind=w[1];bool product_negative=bool(x[0])!=bool(w[0]);
    bool xi=xkind==1||xkind==2,wi=wkind==1||wkind==2;
    Wide<S> a=Wide<S>::read(x+2),b=Wide<S>::read(w+2);
    if(xkind==3||wkind==3||old_kind==3||(xi&&!wi&&b.high()<0)||(wi&&!xi&&a.high()<0))return rational_special(3,f);
    if(xi||wi){if((old_kind==1||old_kind==2)&&old_negative!=product_negative)return rational_special(3,f);return rational_special(product_negative?2:1,f);}
    if(old_kind==1||old_kind==2)return state;
    Wide<S> product=a.times(b).shift(f.product_shift);
    if(old_negative==product_negative)return rational_round(old.plus(product),product_negative,denominator,f);
    int comparison=old.compare(product);
    if(!comparison)return 0;
    return comparison>0?rational_round(old.minus(product),old_negative,denominator,f):rational_round(product.minus(old),product_negative,denominator,f);
}

template<int S> PE_HD uint64_t rational_output(const uint32_t* x,const uint32_t* w,const uint32_t* denominator,int64_t output,int channels,int k,RationalFormat f){
    int64_t n=output/channels,c=output%channels;
    Wide<S> d=Wide<S>::read(denominator);uint64_t state=0;
    for(int i=0;i<k;++i)state=rational_fma(state,x+(n*k+i)*(S+2),w+(c*k+i)*(S+2),d,f);
    return state;
}
} // namespace pe
