#include "exact_binary.h"
#include "rational_binary.h"
#include <cstddef>

extern "C" int pe_gemm(const pe::Value* inputs, const pe::Value* weights,
                       uint32_t* outputs, int batch, int channels, int k, pe::Format format) {
    if (batch <= 0 || channels <= 0 || k <= 0) return 1;
    #pragma omp parallel for schedule(static)
    for (int64_t output = 0; output < int64_t(batch) * channels; ++output) {
        int n = int(output / channels), c = int(output % channels);
        uint32_t state = 0;
        for (int i = 0; i < k; ++i)
            state = pe::fma(state, inputs[int64_t(n) * k + i], weights[int64_t(c) * k + i], format);
        outputs[output] = state;
    }
    return 0;
}

extern "C" int pe_fma_batch(const uint32_t* states, const pe::Value* left, const pe::Value* right,
                            uint32_t* outputs, int count, pe::Format format) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) outputs[i] = pe::fma(states[i], left[i], right[i], format);
    return 0;
}

extern "C" int pe_conv2d(const pe::Value* inputs, const pe::Value* weights,
                         uint32_t* outputs, pe::Conv geometry, pe::Format format) {
    const int64_t count = int64_t(geometry.n) * geometry.co * geometry.oh * geometry.ow;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < count; ++i)
        outputs[i] = geometry.groups == geometry.ci
            ? pe::convolution_output<true>(inputs, weights, i, geometry, format)
            : pe::convolution_output<false>(inputs, weights, i, geometry, format);
    return 0;
}

extern "C" int pe_encoded_gemm(const uint8_t* x, const uint8_t* w, const pe::Value* xt,
                               const pe::Value* wt, uint32_t* y, int batch, int channels, int k,
                               pe::Operand xf, pe::Operand wf, pe::Format acc, int lookup) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < int64_t(batch)*channels; ++i)
        y[i] = pe::encoded_output(x, w, xt, wt, i, channels, k, xf, wf, acc, lookup);
    return 0;
}

extern "C" int pe_rational_gemm(const uint32_t* x,const uint32_t* w,const uint32_t* denominator,uint64_t* y,int batch,int channels,int k,pe::RationalFormat f){
    if(batch<=0||channels<=0||k<=0)return 1;
    if(f.size!=32&&f.size!=64&&f.size!=128&&f.size!=320)return 2;
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<int64_t(batch)*channels;++i){
        if(f.size==32)y[i]=pe::rational_output<32>(x,w,denominator,i,channels,k,f);
        else if(f.size==64)y[i]=pe::rational_output<64>(x,w,denominator,i,channels,k,f);
        else if(f.size==128)y[i]=pe::rational_output<128>(x,w,denominator,i,channels,k,f);
        else y[i]=pe::rational_output<320>(x,w,denominator,i,channels,k,f);
    }
    return 0;
}
