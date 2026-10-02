#include <cmath>
#include <cfenv>
#include <omp.h>
#include <limits>
#define PE_DEVICE inline
#define PE_FMA std::fma
#define PE_LDEXP std::ldexp
#define PE_CLZ64 __builtin_clzll
#define PE_INF std::numeric_limits<double>::infinity()
#include "kernel.h"
template<class T,class Acc>
static void run(const void* x,const void* w,double* y,Shape g,int shift) {
    int64_t count=int64_t(g.n)*g.co*g.oh*g.ow;
    const T* xs=static_cast<const T*>(x); const T* ws=static_cast<const T*>(w);
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<count;++i) y[i]=reduce_one<T,Acc>(xs,ws,g,i,shift);
}
template<class T,class Acc>
static void run_stat(const void* x,const void* w,double* y,uint32_t* events,Shape g,int shift,const int64_t* params) {
    int64_t count=int64_t(g.n)*g.co*g.oh*g.ow;
    const T* xs=static_cast<const T*>(x); const T* ws=static_cast<const T*>(w);
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<count;++i) y[i]=reduce_stat<T,Acc>(xs,ws,g,i,shift,params,events);
}
extern "C" int bridge2_devices() { return 1; }
// Parameterised policies: 2 saturating integer {W}, 3 float {P, emin, emax} in grid units.
extern "C" int bridge2_conv_stat(const void* x,const void* w,double* y,uint32_t* events,Shape g,int shift,int policy,int operand,const int64_t* params) {
    if(std::fegetround()!=FE_TONEAREST || !x || !w || !y || !events || !params) return 1;
    omp_set_num_threads(4);
    if(operand==32 && policy==2) run_stat<int32_t,SatInt>(x,w,y,events,g,shift,params);
    else if(operand==64 && policy==2) run_stat<int64_t,SatInt>(x,w,y,events,g,shift,params);
    else if(operand==32 && policy==3) run_stat<int32_t,SoftFloat>(x,w,y,events,g,shift,params);
    else if(operand==64 && policy==3) run_stat<int64_t,SoftFloat>(x,w,y,events,g,shift,params);
    else return 2;
    return 0;
}
// operand: 32 (int32 grid integers) or 64 (int64, magnitudes below 2^32).
// policy: 0 exact wide integer, 1 sequential binary32 FMA.
extern "C" int bridge2_conv(const void* x,const void* w,double* y,Shape g,int shift,int policy,int operand) {
    if(std::fegetround()!=FE_TONEAREST || !x || !w || !y) return 1;
    omp_set_num_threads(4);
    if(operand==32 && policy==0) run<int32_t,WideI64>(x,w,y,g,shift);
    else if(operand==64 && policy==0) run<int64_t,WideI128>(x,w,y,g,shift);
    else if(operand==32 && policy==1) run<int32_t,FmaF32>(x,w,y,g,shift);
    else if(operand==64 && policy==1) run<int64_t,FmaF32>(x,w,y,g,shift);
    else return 2;
    return 0;
}
