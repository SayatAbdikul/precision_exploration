// Version 2 exact dyadic-grid reductions. This library is built separately from
// the deployed phase-2 backend and never changes its ABI or loaded image.
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>

struct GridConv {
    int32_t n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw, groups;
};

static inline uint64_t bits_of(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// All admitted partial sums are bounded below 2^53 integer units. Conversion
// to binary64 and multiplication by the power-of-two product quantum are exact.
static inline uint64_t grid_bits(int64_t units, int shift) {
    return bits_of(std::ldexp(static_cast<double>(units), -2 * shift));
}

extern "C" int pe_grid_v2_gemm(const int16_t* x, const int16_t* w, uint64_t* y,
                                int batch, int channels, int k, int shift) {
    if (!x || !w || !y || batch <= 0 || channels <= 0 || k <= 0 || shift < 0 || shift > 30)
        return 1;
    #pragma omp parallel for schedule(static)
    for (int64_t output = 0; output < int64_t(batch) * channels; ++output) {
        const int n = int(output / channels), c = int(output % channels);
        int64_t units = 0;
        for (int i = 0; i < k; ++i)
            units += int64_t(x[int64_t(n) * k + i]) * w[int64_t(c) * k + i];
        y[output] = grid_bits(units, shift);
    }
    return 0;
}

extern "C" int pe_grid_v2_conv2d(const int16_t* x, const int16_t* w, uint64_t* y,
                                  GridConv g, int shift) {
    if (!x || !w || !y || g.n <= 0 || g.ci <= 0 || g.co <= 0 || g.h <= 0 || g.w <= 0 ||
        g.kh <= 0 || g.kw <= 0 || g.oh <= 0 || g.ow <= 0 || g.sh <= 0 || g.sw <= 0 ||
        g.dh <= 0 || g.dw <= 0 || g.groups <= 0 || g.ph < 0 || g.pw < 0 ||
        g.ci % g.groups || g.co % g.groups || shift < 0 || shift > 30)
        return 1;
    const int per_group = g.ci / g.groups;
    const int output_group = g.co / g.groups;
    const int64_t count = int64_t(g.n) * g.co * g.oh * g.ow;
    #pragma omp parallel for schedule(static)
    for (int64_t output = 0; output < count; ++output) {
        int64_t index = output;
        const int ox = int(index % g.ow); index /= g.ow;
        const int oy = int(index % g.oh); index /= g.oh;
        const int oc = int(index % g.co), batch = int(index / g.co);
        const int first_channel = (oc / output_group) * per_group;
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
    return 0;
}

// A sequential FP64 reference path for finite compact dyadics outside a
// certified grid. It leaves bias and output stores to the original executor.
// std::fma is one RNE operation per term; build flags disable reassociation.
extern "C" int pe_seq_v2_gemm(const double* x, const double* w, uint64_t* y,
                               int batch, int channels, int k) {
    if (!x || !w || !y || batch <= 0 || channels <= 0 || k <= 0 || std::fegetround() != FE_TONEAREST) return 1;
    #pragma omp parallel for schedule(static)
    for (int64_t output = 0; output < int64_t(batch) * channels; ++output) {
        const int n = int(output / channels), c = int(output % channels);
        double state = 0.0;
        for (int i = 0; i < k; ++i)
            state = std::fma(x[int64_t(n) * k + i], w[int64_t(c) * k + i], state);
        y[output] = bits_of(state);
    }
    return 0;
}

extern "C" int pe_seq_v2_conv2d(const double* x, const double* w, uint64_t* y,
                                 GridConv g) {
    if (!x || !w || !y || g.n <= 0 || g.ci <= 0 || g.co <= 0 || g.h <= 0 || g.w <= 0 ||
        g.kh <= 0 || g.kw <= 0 || g.oh <= 0 || g.ow <= 0 || g.sh <= 0 || g.sw <= 0 ||
        g.dh <= 0 || g.dw <= 0 || g.groups <= 0 || g.ph < 0 || g.pw < 0 ||
        g.ci % g.groups || g.co % g.groups || std::fegetround() != FE_TONEAREST) return 1;
    const int per_group = g.ci / g.groups, output_group = g.co / g.groups;
    const int64_t count = int64_t(g.n) * g.co * g.oh * g.ow;
    #pragma omp parallel for schedule(static)
    for (int64_t output = 0; output < count; ++output) {
        int64_t index = output;
        const int ox = int(index % g.ow); index /= g.ow;
        const int oy = int(index % g.oh); index /= g.oh;
        const int oc = int(index % g.co), batch = int(index / g.co);
        const int first_channel = (oc / output_group) * per_group;
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
                    state = std::fma(value, weight, state);
                }
        y[output] = bits_of(state);
    }
    return 0;
}
