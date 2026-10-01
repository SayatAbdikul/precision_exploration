// Independently built version-2 exact dyadic-grid and sequential FP64 kernels.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

struct GridConv {
    int32_t n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw, groups;
};

__device__ __forceinline__ uint64_t grid_bits(int64_t units, int shift) {
    const double value = __dmul_rn(__ll2double_rn(units), ldexp(1.0, -2 * shift));
    return uint64_t(__double_as_longlong(value));
}

__global__ void grid_gemm_kernel(const int16_t* x, const int16_t* w, uint64_t* y,
                                 int batch, int channels, int k, int shift) {
    const int64_t output = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output >= int64_t(batch) * channels) return;
    const int n = int(output / channels), c = int(output % channels);
    int64_t units = 0;
    for (int i = 0; i < k; ++i)
        units += int64_t(x[int64_t(n) * k + i]) * w[int64_t(c) * k + i];
    y[output] = grid_bits(units, shift);
}

__global__ void grid_conv_kernel(const int16_t* x, const int16_t* w, uint64_t* y,
                                 GridConv g, int shift) {
    const int64_t output = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output >= int64_t(g.n) * g.co * g.oh * g.ow) return;
    int64_t index = output;
    const int ox = int(index % g.ow); index /= g.ow;
    const int oy = int(index % g.oh); index /= g.oh;
    const int oc = int(index % g.co), batch = int(index / g.co);
    const int per_group = g.ci / g.groups;
    const int first_channel = (oc / (g.co / g.groups)) * per_group;
    int64_t units = 0;
    for (int ic = 0; ic < per_group; ++ic)
        for (int kr = 0; kr < g.kh; ++kr)
            for (int kc = 0; kc < g.kw; ++kc) {
                const int iy = oy * g.sh - g.ph + kr * g.dh;
                const int ix = ox * g.sw - g.pw + kc * g.dw;
                if (0 <= iy && iy < g.h && 0 <= ix && ix < g.w) {
                    const int64_t xi = ((int64_t(batch) * g.ci + first_channel + ic) * g.h + iy) * g.w + ix;
                    const int64_t wi = ((int64_t(oc) * per_group + ic) * g.kh + kr) * g.kw + kc;
                    units += int64_t(x[xi]) * w[wi];
                }
            }
    y[output] = grid_bits(units, shift);
}

__global__ void seq_gemm_kernel(const double* x, const double* w, uint64_t* y,
                                int batch, int channels, int k) {
    const int64_t output = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output >= int64_t(batch) * channels) return;
    const int n = int(output / channels), c = int(output % channels);
    double state = 0.0;
    for (int i = 0; i < k; ++i)
        state = __fma_rn(x[int64_t(n) * k + i], w[int64_t(c) * k + i], state);
    y[output] = uint64_t(__double_as_longlong(state));
}

__global__ void seq_conv_kernel(const double* x, const double* w, uint64_t* y,
                                GridConv g) {
    const int64_t output = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output >= int64_t(g.n) * g.co * g.oh * g.ow) return;
    int64_t index = output;
    const int ox = int(index % g.ow); index /= g.ow;
    const int oy = int(index % g.oh); index /= g.oh;
    const int oc = int(index % g.co), batch = int(index / g.co);
    const int per_group = g.ci / g.groups;
    const int first_channel = (oc / (g.co / g.groups)) * per_group;
    double state = 0.0;
    for (int ic = 0; ic < per_group; ++ic)
        for (int kr = 0; kr < g.kh; ++kr)
            for (int kc = 0; kc < g.kw; ++kc) {
                const int iy = oy * g.sh - g.ph + kr * g.dh;
                const int ix = ox * g.sw - g.pw + kc * g.dw;
                const double value = 0 <= iy && iy < g.h && 0 <= ix && ix < g.w
                    ? x[((int64_t(batch) * g.ci + first_channel + ic) * g.h + iy) * g.w + ix]
                    : 0.0;
                const double weight = w[((int64_t(oc) * per_group + ic) * g.kh + kr) * g.kw + kc];
                state = __fma_rn(value, weight, state);
            }
    y[output] = uint64_t(__double_as_longlong(state));
}

template<class T, class Launch>
static int run(const T* x, size_t xn, const T* w, size_t wn, uint64_t* y, size_t yn, Launch launch) {
    T *dx = nullptr, *dw = nullptr;
    uint64_t* dy = nullptr;
    cudaError_t error = cudaMalloc(&dx, xn * sizeof(T));
    if (error == cudaSuccess) error = cudaMalloc(&dw, wn * sizeof(T));
    if (error == cudaSuccess) error = cudaMalloc(&dy, yn * sizeof(uint64_t));
    if (error == cudaSuccess) error = cudaMemcpy(dx, x, xn * sizeof(T), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error = cudaMemcpy(dw, w, wn * sizeof(T), cudaMemcpyHostToDevice);
    if (error == cudaSuccess) { launch(dx, dw, dy); error = cudaGetLastError(); }
    if (error == cudaSuccess) error = cudaMemcpy(y, dy, yn * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dw); cudaFree(dy);
    return int(error);
}

extern "C" int pe_grid_v2_device_count() { int count = 0; return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0; }
extern "C" const char* pe_grid_v2_error(int code) { return cudaGetErrorString(cudaError_t(code)); }

extern "C" int pe_grid_v2_gemm(const int16_t* x, const int16_t* w, uint64_t* y,
                                int batch, int channels, int k, int shift) {
    if (!x || !w || !y || batch <= 0 || channels <= 0 || k <= 0 || shift < 0 || shift > 30) return int(cudaErrorInvalidValue);
    const size_t count = size_t(batch) * channels;
    return run(x, size_t(batch) * k, w, size_t(channels) * k, y, count,
               [=](const int16_t* dx, const int16_t* dw, uint64_t* dy) {
                   grid_gemm_kernel<<<(count + 127) / 128, 128>>>(dx, dw, dy, batch, channels, k, shift);
               });
}

extern "C" int pe_grid_v2_conv2d(const int16_t* x, const int16_t* w, uint64_t* y,
                                  GridConv g, int shift) {
    if (!x || !w || !y || g.n <= 0 || g.ci <= 0 || g.co <= 0 || g.h <= 0 || g.w <= 0 ||
        g.kh <= 0 || g.kw <= 0 || g.oh <= 0 || g.ow <= 0 || g.sh <= 0 || g.sw <= 0 ||
        g.dh <= 0 || g.dw <= 0 || g.groups <= 0 || g.ph < 0 || g.pw < 0 ||
        g.ci % g.groups || g.co % g.groups || shift < 0 || shift > 30) return int(cudaErrorInvalidValue);
    const size_t count = size_t(g.n) * g.co * g.oh * g.ow;
    return run(x, size_t(g.n) * g.ci * g.h * g.w,
               w, size_t(g.co) * (g.ci / g.groups) * g.kh * g.kw, y, count,
               [=](const int16_t* dx, const int16_t* dw, uint64_t* dy) {
                   grid_conv_kernel<<<(count + 127) / 128, 128>>>(dx, dw, dy, g, shift);
               });
}

extern "C" int pe_seq_v2_gemm(const double* x, const double* w, uint64_t* y,
                               int batch, int channels, int k) {
    if (!x || !w || !y || batch <= 0 || channels <= 0 || k <= 0) return int(cudaErrorInvalidValue);
    const size_t count = size_t(batch) * channels;
    return run(x, size_t(batch) * k, w, size_t(channels) * k, y, count,
               [=](const double* dx, const double* dw, uint64_t* dy) {
                   seq_gemm_kernel<<<(count + 127) / 128, 128>>>(dx, dw, dy, batch, channels, k);
               });
}

extern "C" int pe_seq_v2_conv2d(const double* x, const double* w, uint64_t* y,
                                 GridConv g) {
    if (!x || !w || !y || g.n <= 0 || g.ci <= 0 || g.co <= 0 || g.h <= 0 || g.w <= 0 ||
        g.kh <= 0 || g.kw <= 0 || g.oh <= 0 || g.ow <= 0 || g.sh <= 0 || g.sw <= 0 ||
        g.dh <= 0 || g.dw <= 0 || g.groups <= 0 || g.ph < 0 || g.pw < 0 ||
        g.ci % g.groups || g.co % g.groups) return int(cudaErrorInvalidValue);
    const size_t count = size_t(g.n) * g.co * g.oh * g.ow;
    return run(x, size_t(g.n) * g.ci * g.h * g.w,
               w, size_t(g.co) * (g.ci / g.groups) * g.kh * g.kw, y, count,
               [=](const double* dx, const double* dw, uint64_t* dy) {
                   seq_conv_kernel<<<(count + 127) / 128, 128>>>(dx, dw, dy, g);
               });
}
