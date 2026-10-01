#include <cuda_runtime.h>
#include <cmath>
#define PE_DEVICE __device__ inline
#define PE_FMA __fmaf_rn
#define PE_LDEXP ldexp
#include "kernel.h"
__global__ void kernel(const int16_t* x,const int16_t* w,double* y,Shape g,int shift,int mode) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(g.n)*g.co*g.oh*g.ow) y[i]=reduce_one(x,w,g,i,shift,mode);
}
extern "C" int bridge_devices() {int n=0;return cudaGetDeviceCount(&n)==cudaSuccess?n:0;}
extern "C" int bridge_conv(const int16_t* x,const int16_t* w,double* y,Shape g,int shift,int mode) {
    int16_t *dx=nullptr,*dw=nullptr; double*dy=nullptr;
    size_t xn=size_t(g.n)*g.ci*g.h*g.w, wn=size_t(g.co)*(g.ci/g.groups)*g.kh*g.kw;
    size_t yn=size_t(g.n)*g.co*g.oh*g.ow;
    cudaError_t e=cudaMalloc(&dx,xn*sizeof(int16_t));
    if(e==cudaSuccess)e=cudaMalloc(&dw,wn*sizeof(int16_t));
    if(e==cudaSuccess)e=cudaMalloc(&dy,yn*sizeof(double));
    if(e==cudaSuccess)e=cudaMemcpy(dx,x,xn*sizeof(int16_t),cudaMemcpyHostToDevice);
    if(e==cudaSuccess)e=cudaMemcpy(dw,w,wn*sizeof(int16_t),cudaMemcpyHostToDevice);
    if(e==cudaSuccess){kernel<<<(yn+127)/128,128>>>(dx,dw,dy,g,shift,mode);e=cudaGetLastError();}
    if(e==cudaSuccess)e=cudaMemcpy(y,dy,yn*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(dx);cudaFree(dw);cudaFree(dy);return int(e);
}
