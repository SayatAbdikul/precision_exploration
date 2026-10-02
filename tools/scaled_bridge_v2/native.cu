#include <cuda_runtime.h>
#include <cmath>
#include <math_constants.h>
#define PE_DEVICE __device__ inline
#define PE_FMA __fmaf_rn
#define PE_LDEXP ldexp
#define PE_CLZ64 __clzll
#define PE_INF CUDART_INF
#include "kernel.h"
template<class T,class Acc>
__global__ void kernel(const T* x,const T* w,double* y,Shape g,int shift) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(g.n)*g.co*g.oh*g.ow) y[i]=reduce_one<T,Acc>(x,w,g,i,shift);
}
template<class T,class Acc>
static int launch(const void* x,const void* w,double* y,Shape g,int shift) {
    T *dx=nullptr,*dw=nullptr; double* dy=nullptr;
    size_t xn=size_t(g.n)*g.ci*g.h*g.w, wn=size_t(g.co)*(g.ci/g.groups)*g.kh*g.kw;
    size_t yn=size_t(g.n)*g.co*g.oh*g.ow;
    cudaError_t e=cudaMalloc(&dx,xn*sizeof(T));
    if(e==cudaSuccess)e=cudaMalloc(&dw,wn*sizeof(T));
    if(e==cudaSuccess)e=cudaMalloc(&dy,yn*sizeof(double));
    if(e==cudaSuccess)e=cudaMemcpy(dx,x,xn*sizeof(T),cudaMemcpyHostToDevice);
    if(e==cudaSuccess)e=cudaMemcpy(dw,w,wn*sizeof(T),cudaMemcpyHostToDevice);
    if(e==cudaSuccess){kernel<T,Acc><<<(yn+127)/128,128>>>(dx,dw,dy,g,shift);e=cudaGetLastError();}
    if(e==cudaSuccess)e=cudaMemcpy(y,dy,yn*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dw);cudaFree(dy);return int(e);
}
struct Params { int64_t v[4]; };
template<class T,class Acc>
__global__ void kernel_stat(const T* x,const T* w,double* y,uint32_t* events,Shape g,int shift,Params p) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(g.n)*g.co*g.oh*g.ow) y[i]=reduce_stat<T,Acc>(x,w,g,i,shift,p.v,events);
}
template<class T,class Acc>
static int launch_stat(const void* x,const void* w,double* y,uint32_t* events,Shape g,int shift,const int64_t* params) {
    T *dx=nullptr,*dw=nullptr; double* dy=nullptr; uint32_t* de=nullptr;
    Params p; for(int i=0;i<4;++i) p.v[i]=params[i];
    size_t xn=size_t(g.n)*g.ci*g.h*g.w, wn=size_t(g.co)*(g.ci/g.groups)*g.kh*g.kw;
    size_t yn=size_t(g.n)*g.co*g.oh*g.ow;
    cudaError_t e=cudaMalloc(&dx,xn*sizeof(T));
    if(e==cudaSuccess)e=cudaMalloc(&dw,wn*sizeof(T));
    if(e==cudaSuccess)e=cudaMalloc(&dy,yn*sizeof(double));
    if(e==cudaSuccess)e=cudaMalloc(&de,yn*sizeof(uint32_t));
    if(e==cudaSuccess)e=cudaMemcpy(dx,x,xn*sizeof(T),cudaMemcpyHostToDevice);
    if(e==cudaSuccess)e=cudaMemcpy(dw,w,wn*sizeof(T),cudaMemcpyHostToDevice);
    if(e==cudaSuccess){kernel_stat<T,Acc><<<(yn+127)/128,128>>>(dx,dw,dy,de,g,shift,p);e=cudaGetLastError();}
    if(e==cudaSuccess)e=cudaMemcpy(y,dy,yn*sizeof(double),cudaMemcpyDeviceToHost);
    if(e==cudaSuccess)e=cudaMemcpy(events,de,yn*sizeof(uint32_t),cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dw);cudaFree(dy);cudaFree(de);return int(e);
}
extern "C" int bridge2_conv_stat(const void* x,const void* w,double* y,uint32_t* events,Shape g,int shift,int policy,int operand,const int64_t* params) {
    if(!x || !w || !y || !events || !params) return -1;
    if(operand==32 && policy==2) return launch_stat<int32_t,SatInt>(x,w,y,events,g,shift,params);
    if(operand==64 && policy==2) return launch_stat<int64_t,SatInt>(x,w,y,events,g,shift,params);
    if(operand==32 && policy==3) return launch_stat<int32_t,SoftFloat>(x,w,y,events,g,shift,params);
    if(operand==64 && policy==3) return launch_stat<int64_t,SoftFloat>(x,w,y,events,g,shift,params);
    return -2;
}
extern "C" int bridge2_devices() {int n=0;return cudaGetDeviceCount(&n)==cudaSuccess?n:0;}
extern "C" int bridge2_conv(const void* x,const void* w,double* y,Shape g,int shift,int policy,int operand) {
    if(!x || !w || !y) return -1;
    if(operand==32 && policy==0) return launch<int32_t,WideI64>(x,w,y,g,shift);
    if(operand==64 && policy==0) return launch<int64_t,WideI128>(x,w,y,g,shift);
    if(operand==32 && policy==1) return launch<int32_t,FmaF32>(x,w,y,g,shift);
    if(operand==64 && policy==1) return launch<int64_t,FmaF32>(x,w,y,g,shift);
    return -2;
}
