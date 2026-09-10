#include "../../inference/cpp/exact_binary.h"
#include "../../inference/cpp/rational_binary.h"
#include <cuda_runtime.h>

__global__ void gemm_kernel(const pe::Value* inputs, const pe::Value* weights,
                            uint32_t* outputs, int batch, int channels, int k, pe::Format format) {
    int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(batch) * channels) return;
    int n = int(index / channels), c = int(index % channels);
    uint32_t state = 0;
    for (int i = 0; i < k; ++i)
        state = pe::fma(state, inputs[int64_t(n) * k + i], weights[int64_t(c) * k + i], format);
    outputs[index] = state;
}

template<bool depthwise>
__global__ void conv_kernel(const pe::Value* inputs, const pe::Value* weights,
                            uint32_t* outputs, pe::Conv geometry, pe::Format format) {
    int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < int64_t(geometry.n) * geometry.co * geometry.oh * geometry.ow)
        outputs[i] = pe::convolution_output<depthwise>(inputs, weights, i, geometry, format);
}

__global__ void encoded_kernel(const uint8_t* x, const uint8_t* w, const pe::Value* xt,
                                const pe::Value* wt, uint32_t* y, int batch, int channels, int k,
                                pe::Operand xf, pe::Operand wf, pe::Format acc, int lookup) {
    int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < int64_t(batch)*channels) y[i] = pe::encoded_output(x, w, xt, wt, i, channels, k, xf, wf, acc, lookup);
}

extern "C" int pe_device_count() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}
extern "C" const char* pe_error(int code) { return cudaGetErrorString(cudaError_t(code)); }

extern "C" int pe_gemm(const pe::Value* inputs, const pe::Value* weights,
                       uint32_t* outputs, int batch, int channels, int k, pe::Format format) {
    if (batch <= 0 || channels <= 0 || k <= 0) return int(cudaErrorInvalidValue);
    pe::Value *device_inputs = nullptr, *device_weights = nullptr;
    uint32_t* device_outputs = nullptr;
    const size_t input_bytes = size_t(batch) * k * sizeof(pe::Value);
    const size_t weight_bytes = size_t(channels) * k * sizeof(pe::Value);
    const size_t output_bytes = size_t(batch) * channels * sizeof(uint32_t);
    cudaError_t error;
    // Every allocation is released on every path, including partial setup.
    error = cudaMalloc(&device_inputs, input_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&device_weights, weight_bytes);
    if (error == cudaSuccess) error = cudaMalloc(&device_outputs, output_bytes);
    if (error == cudaSuccess) error = cudaMemcpy(device_inputs, inputs, input_bytes, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error = cudaMemcpy(device_weights, weights, weight_bytes, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
        gemm_kernel<<<(size_t(batch) * channels + 63) / 64, 64>>>(device_inputs, device_weights, device_outputs, batch, channels, k, format);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaMemcpy(outputs, device_outputs, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(device_inputs); cudaFree(device_weights); cudaFree(device_outputs);
    return int(error);
}

extern "C" int pe_conv2d(const pe::Value* inputs, const pe::Value* weights,
                         uint32_t* outputs, pe::Conv g, pe::Format format) {
    pe::Value *x = nullptr, *w = nullptr;
    uint32_t* y = nullptr;
    size_t xb = size_t(g.n) * g.ci * g.h * g.w * sizeof(pe::Value);
    size_t wb = size_t(g.co) * (g.ci / g.groups) * g.kh * g.kw * sizeof(pe::Value);
    size_t count = size_t(g.n) * g.co * g.oh * g.ow;
    cudaError_t error = cudaMalloc(&x, xb);
    if (error == cudaSuccess) error = cudaMalloc(&w, wb);
    if (error == cudaSuccess) error = cudaMalloc(&y, count * sizeof(uint32_t));
    if (error == cudaSuccess) error = cudaMemcpy(x, inputs, xb, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error = cudaMemcpy(w, weights, wb, cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
        if (g.groups == g.ci) conv_kernel<true><<<(count + 63) / 64, 64>>>(x, w, y, g, format);
        else conv_kernel<false><<<(count + 63) / 64, 64>>>(x, w, y, g, format);
        error = cudaGetLastError();
    }
    if (error == cudaSuccess) error = cudaMemcpy(outputs, y, count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(x); cudaFree(w); cudaFree(y);
    return int(error);
}

extern "C" int pe_encoded_gemm(const uint8_t* inputs, const uint8_t* weights, const pe::Value* input_table,
                               const pe::Value* weight_table, uint32_t* outputs, int batch, int channels, int k,
                               pe::Operand xf, pe::Operand wf, pe::Format acc, int lookup) {
    uint8_t *x=nullptr, *w=nullptr;
    pe::Value *xt=nullptr, *wt=nullptr;
    uint32_t* y=nullptr;
    size_t xb=size_t(batch)*k, wb=size_t(channels)*k, yb=size_t(batch)*channels*sizeof(uint32_t);
    size_t xtb=(size_t(1)<<xf.bits)*sizeof(pe::Value), wtb=(size_t(1)<<wf.bits)*sizeof(pe::Value);
    cudaError_t error=cudaMalloc(&x,xb);
    if (error==cudaSuccess) error=cudaMalloc(&w,wb);
    if (error==cudaSuccess) error=cudaMalloc(&xt,xtb);
    if (error==cudaSuccess) error=cudaMalloc(&wt,wtb);
    if (error==cudaSuccess) error=cudaMalloc(&y,yb);
    if (error==cudaSuccess) error=cudaMemcpy(x,inputs,xb,cudaMemcpyHostToDevice);
    if (error==cudaSuccess) error=cudaMemcpy(w,weights,wb,cudaMemcpyHostToDevice);
    if (error==cudaSuccess) error=cudaMemcpy(xt,input_table,xtb,cudaMemcpyHostToDevice);
    if (error==cudaSuccess) error=cudaMemcpy(wt,weight_table,wtb,cudaMemcpyHostToDevice);
    if (error==cudaSuccess) {
        encoded_kernel<<<(size_t(batch)*channels+63)/64,64>>>(x,w,xt,wt,y,batch,channels,k,xf,wf,acc,lookup);
        error=cudaGetLastError();
    }
    if (error==cudaSuccess) error=cudaMemcpy(outputs,y,yb,cudaMemcpyDeviceToHost);
    cudaFree(x);cudaFree(w);cudaFree(xt);cudaFree(wt);cudaFree(y);
    return int(error);
}

template<int S> __global__ void rational_kernel(const uint32_t* x,const uint32_t* w,const uint32_t* denominator,uint64_t* y,int batch,int channels,int k,pe::RationalFormat f){
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<int64_t(batch)*channels)y[i]=pe::rational_output<S>(x,w,denominator,i,channels,k,f);
}

extern "C" int pe_rational_gemm(const uint32_t* inputs,const uint32_t* weights,const uint32_t* denominator,uint64_t* outputs,int batch,int channels,int k,pe::RationalFormat f){
    if(batch<=0||channels<=0||k<=0||(f.size!=32&&f.size!=64&&f.size!=128&&f.size!=320))return int(cudaErrorInvalidValue);
    uint32_t *x=nullptr,*w=nullptr,*d=nullptr;uint64_t* y=nullptr;
    size_t xb=size_t(batch)*k*(f.size+2)*sizeof(uint32_t),wb=size_t(channels)*k*(f.size+2)*sizeof(uint32_t),db=size_t(f.size)*sizeof(uint32_t),yb=size_t(batch)*channels*sizeof(uint64_t);
    cudaError_t error=cudaMalloc(&x,xb);
    if(error==cudaSuccess)error=cudaMalloc(&w,wb);
    if(error==cudaSuccess)error=cudaMalloc(&d,db);
    if(error==cudaSuccess)error=cudaMalloc(&y,yb);
    if(error==cudaSuccess)error=cudaMemcpy(x,inputs,xb,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(w,weights,wb,cudaMemcpyHostToDevice);
    if(error==cudaSuccess)error=cudaMemcpy(d,denominator,db,cudaMemcpyHostToDevice);
    if(error==cudaSuccess){
        size_t blocks=(size_t(batch)*channels+31)/32;
        if(f.size==32)rational_kernel<32><<<blocks,32>>>(x,w,d,y,batch,channels,k,f);
        else if(f.size==64)rational_kernel<64><<<blocks,32>>>(x,w,d,y,batch,channels,k,f);
        else if(f.size==128)rational_kernel<128><<<blocks,32>>>(x,w,d,y,batch,channels,k,f);
        else rational_kernel<320><<<blocks,32>>>(x,w,d,y,batch,channels,k,f);
        error=cudaGetLastError();
    }
    if(error==cudaSuccess)error=cudaMemcpy(outputs,y,yb,cudaMemcpyDeviceToHost);
    cudaFree(x);cudaFree(w);cudaFree(d);cudaFree(y);
    return int(error);
}
