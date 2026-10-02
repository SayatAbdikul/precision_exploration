// Lane S1 fast path: the archived scaled bridge v2 reduction kernels (kernel.h, byte-identical copy of the
// 1f75c923/7c6344af archives, hash asserted by native.py) launched on device pointers that the caller owns
// (torch CUDA tensors), on the caller's stream, with no allocation and no host copy. The kernel templates
// below are the archive's native.cu templates verbatim; only the launchers differ. Built with the archive's
// nvcc flags (--fmad=false --ftz=false, no -arch).
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
struct Params { int64_t v[4]; };
template<class T,class Acc>
__global__ void kernel_stat(const T* x,const T* w,double* y,uint32_t* events,Shape g,int shift,Params p) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(g.n)*g.co*g.oh*g.ow) y[i]=reduce_stat<T,Acc>(x,w,g,i,shift,p.v,events);
}
template<class T,class Acc>
static int launch(const void* x,const void* w,double* y,Shape g,int shift,cudaStream_t s) {
    size_t yn=size_t(g.n)*g.co*g.oh*g.ow;
    kernel<T,Acc><<<(yn+127)/128,128,0,s>>>(static_cast<const T*>(x),static_cast<const T*>(w),y,g,shift);
    return int(cudaGetLastError());
}
template<class T,class Acc>
static int launch_stat(const void* x,const void* w,double* y,uint32_t* events,Shape g,int shift,const int64_t* params,cudaStream_t s) {
    Params p; for(int i=0;i<4;++i) p.v[i]=params[i];
    size_t yn=size_t(g.n)*g.co*g.oh*g.ow;
    kernel_stat<T,Acc><<<(yn+127)/128,128,0,s>>>(static_cast<const T*>(x),static_cast<const T*>(w),y,events,g,shift,p);
    return int(cudaGetLastError());
}
extern "C" int fast2_devices() {int n=0;return cudaGetDeviceCount(&n)==cudaSuccess?n:0;}
extern "C" int fast2_conv(const void* x,const void* w,double* y,Shape g,int shift,int policy,int operand,void* stream) {
    if(!x || !w || !y) return -1;
    cudaStream_t s=static_cast<cudaStream_t>(stream);
    if(operand==32 && policy==0) return launch<int32_t,WideI64>(x,w,y,g,shift,s);
    if(operand==64 && policy==0) return launch<int64_t,WideI128>(x,w,y,g,shift,s);
    if(operand==32 && policy==1) return launch<int32_t,FmaF32>(x,w,y,g,shift,s);
    if(operand==64 && policy==1) return launch<int64_t,FmaF32>(x,w,y,g,shift,s);
    return -2;
}
extern "C" int fast2_conv_stat(const void* x,const void* w,double* y,uint32_t* events,Shape g,int shift,int policy,int operand,const int64_t* params,void* stream) {
    if(!x || !w || !y || !events || !params) return -1;
    cudaStream_t s=static_cast<cudaStream_t>(stream);
    if(operand==32 && policy==2) return launch_stat<int32_t,SatInt>(x,w,y,events,g,shift,params,s);
    if(operand==64 && policy==2) return launch_stat<int64_t,SatInt>(x,w,y,events,g,shift,params,s);
    if(operand==32 && policy==3) return launch_stat<int32_t,SoftFloat>(x,w,y,events,g,shift,params,s);
    if(operand==64 && policy==3) return launch_stat<int64_t,SoftFloat>(x,w,y,events,g,shift,params,s);
    return -2;
}
