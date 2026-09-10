#pragma once
// Shared CPU/CUDA Model C implementation. A fixed integer represents the exact
// sum in units of 2^-192; rounding occurs only when encoding the accumulator.
#include <cstdint>

#ifdef __CUDACC__
#define PE_HD __host__ __device__
#else
#define PE_HD
#endif

namespace pe {
struct Value { int64_t mantissa; int32_t exponent; int32_t kind; };
// kind: 0 finite, 1 +inf, 2 -inf, 3 canonical NaN, 4 negative zero.
struct Format { int32_t mantissa, exponent, bias, integer; };
struct Conv { int32_t n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw, dh, dw, groups; };
struct Operand { int32_t bits, is_signed, numeric, fractional, mantissa, exponent, bias, subnormal, nan, infinity; };
constexpr int limbs = 12;
constexpr int base_exponent = -192;

struct Big {
    uint32_t word[limbs]{};
    PE_HD bool bit(int n) const { return n >= 0 && n < limbs * 32 && ((word[n / 32] >> (n % 32)) & 1u); }
    PE_HD int highest() const {
        for (int i = limbs * 32 - 1; i >= 0; --i) if (bit(i)) return i;
        return -1;
    }
    PE_HD bool below(int n) const {
        for (int i = 0; i < n; ++i) if (bit(i)) return true;
        return false;
    }
    PE_HD uint64_t shifted(int n) const {
        uint64_t result = 0;
        for (int i = 0; i < 64; ++i) if (bit(n + i)) result |= uint64_t(1) << i;
        return result;
    }
};

PE_HD inline Big magnitude(uint64_t mantissa, int exponent) {
    Big value;
    for (int i = 0; i < 64; ++i) {
        const int target = i + exponent - base_exponent;
        if ((mantissa >> i) & 1u) value.word[target / 32] |= uint32_t(1) << (target % 32);
    }
    return value;
}
PE_HD inline int compare(const Big& a, const Big& b) {
    for (int i = limbs - 1; i >= 0; --i) {
        if (a.word[i] != b.word[i]) return a.word[i] > b.word[i] ? 1 : -1;
    }
    return 0;
}
PE_HD inline Big add(const Big& a, const Big& b) {
    Big value; uint64_t carry = 0;
    for (int i = 0; i < limbs; ++i) {
        uint64_t part = uint64_t(a.word[i]) + b.word[i] + carry;
        value.word[i] = uint32_t(part); carry = part >> 32;
    }
    return value;
}
PE_HD inline Big subtract(const Big& a, const Big& b) {
    Big value; uint64_t borrow = 0;
    for (int i = 0; i < limbs; ++i) {
        uint64_t part = uint64_t(b.word[i]) + borrow;
        value.word[i] = uint32_t(uint64_t(a.word[i]) - part);
        borrow = uint64_t(a.word[i]) < part;
    }
    return value;
}

PE_HD inline Value decode(uint32_t code, Format format) {
    if (format.integer) return {int64_t(int32_t(code)), 0, 0};
    const int width = 1 + format.exponent + format.mantissa;
    const bool negative = (code >> (width - 1)) != 0;
    const uint32_t fraction = code & ((uint32_t(1) << format.mantissa) - 1);
    const uint32_t e = (code >> format.mantissa) & ((uint32_t(1) << format.exponent) - 1);
    if (e == (uint32_t(1) << format.exponent) - 1) return {0, 0, fraction ? 3 : (negative ? 2 : 1)};
    if (!e && !fraction) return {0, 0, negative ? 4 : 0};
    int64_t m = e ? (uint32_t(1) << format.mantissa) + fraction : fraction;
    return {negative ? -m : m, (e ? int(e) : 1) - format.bias - format.mantissa, 0};
}
PE_HD inline bool negative(Value value) { return value.mantissa < 0 || value.kind == 2 || value.kind == 4; }
PE_HD inline Value decode_operand(uint8_t code, Operand f) {
    bool sign = f.is_signed && (code >> (f.bits - 1));
    if (f.numeric) return {sign ? int64_t(code) - (int64_t(1) << f.bits) : int64_t(code), -f.fractional, 0};
    uint32_t fraction = code & ((1u << f.mantissa) - 1);
    uint32_t e = (code >> f.mantissa) & ((1u << f.exponent) - 1);
    if (e == (1u << f.exponent) - 1) {
        if (f.infinity && fraction == 0) return {0, 0, sign ? 2 : 1};
        if (f.nan && (f.infinity || fraction == (1u << f.mantissa) - 1)) return {0, 0, 3};
    }
    if (e == 0 && (fraction == 0 || !f.subnormal)) return {0, 0, sign ? 4 : 0};
    int64_t m = e ? (1u << f.mantissa) + fraction : fraction;
    return {sign ? -m : m, (e ? int(e) : 1) - f.bias - f.mantissa, 0};
}
PE_HD inline uint32_t special(int kind, Format format) {
    const uint32_t inf = ((uint32_t(1) << format.exponent) - 1) << format.mantissa;
    const uint32_t sign = uint32_t(1) << (format.exponent + format.mantissa);
    return kind == 3 ? inf | 1 : kind == 2 ? sign | inf : kind == 4 ? sign : inf;
}

PE_HD inline uint32_t rounded(const Big& value, bool negative_sign, Format format) {
    const int high = value.highest();
    if (format.integer) {
        int shift = -base_exponent;
        uint64_t q = value.shifted(shift);
        if (value.bit(shift - 1) && (value.below(shift - 1) || (q & 1))) ++q;
        if (high + base_exponent >= 32 || q > (negative_sign ? uint64_t(2147483648u) : uint64_t(2147483647u)))
            return negative_sign ? 0x80000000u : 0x7fffffffu;
        return negative_sign ? uint32_t(0 - q) : uint32_t(q);
    }
    const uint32_t sign = negative_sign ? uint32_t(1) << (format.exponent + format.mantissa) : 0;
    if (high < 0) return sign;
    const int minimum = 1 - format.bias;
    int exponent = high + base_exponent;
    int shift = (exponent < minimum ? minimum : exponent) - format.mantissa - base_exponent;
    uint64_t q = value.shifted(shift);
    if (value.bit(shift - 1) && (value.below(shift - 1) || (q & 1))) ++q;
    if (exponent < minimum) return sign | uint32_t(q);
    if (q == uint64_t(1) << (format.mantissa + 1)) { ++exponent; q >>= 1; }
    int encoded_exponent = exponent + format.bias;
    if (encoded_exponent >= (1 << format.exponent) - 1) return special(negative_sign ? 2 : 1, format);
    return sign | (uint32_t(encoded_exponent) << format.mantissa) | uint32_t(q - (uint64_t(1) << format.mantissa));
}

PE_HD inline uint32_t fma(uint32_t state, Value x, Value w, Format format) {
    Value a = decode(state, format);
    const bool product_negative = negative(x) != negative(w);
    bool x_inf = x.kind == 1 || x.kind == 2;
    bool w_inf = w.kind == 1 || w.kind == 2;
    if (x.kind == 3 || w.kind == 3 || a.kind == 3 ||
        (x_inf && !w_inf && w.mantissa == 0) || (w_inf && !x_inf && x.mantissa == 0)) return special(3, format);
    if (x_inf || w_inf) {
        if ((a.kind == 1 || a.kind == 2) && negative(a) != product_negative) return special(3, format);
        return special(product_negative ? 2 : 1, format);
    }
    if (a.kind == 1 || a.kind == 2) return state;
    int64_t product = x.mantissa * w.mantissa;
    Big p = magnitude(uint64_t(product < 0 ? -product : product), x.exponent + w.exponent);
    Big previous = magnitude(uint64_t(a.mantissa < 0 ? -a.mantissa : a.mantissa), a.exponent);
    if (negative(a) == product_negative) return rounded(add(previous, p), product_negative, format);
    int ordering = compare(previous, p);
    if (ordering == 0) return 0;
    return ordering > 0 ? rounded(subtract(previous, p), negative(a), format)
                        : rounded(subtract(p, previous), product_negative, format);
}

template<bool depthwise>
PE_HD inline uint32_t convolution_output(const Value* inputs, const Value* weights,
                                          int64_t index, Conv g, Format format) {
    const int ox = int(index % g.ow); index /= g.ow;
    const int oy = int(index % g.oh); index /= g.oh;
    const int oc = int(index % g.co), batch = int(index / g.co);
    const int group_channels = depthwise ? 1 : g.ci / g.groups;
    const int first_channel = depthwise ? oc / (g.co / g.ci) : (oc / (g.co / g.groups)) * group_channels;
    uint32_t state = 0;
    for (int ic = 0; ic < group_channels; ++ic)
        for (int kr = 0; kr < g.kh; ++kr)
            for (int kc = 0; kc < g.kw; ++kc) {
                int iy = oy * g.sh - g.ph + kr * g.dh, ix = ox * g.sw - g.pw + kc * g.dw;
                Value x = (iy >= 0 && iy < g.h && ix >= 0 && ix < g.w)
                    ? inputs[((int64_t(batch) * g.ci + first_channel + ic) * g.h + iy) * g.w + ix]
                    : Value{0, 0, 0};
                Value w = weights[((int64_t(oc) * group_channels + ic) * g.kh + kr) * g.kw + kc];
                state = fma(state, x, w, format);
            }
    return state;
}

PE_HD inline uint32_t encoded_output(const uint8_t* x, const uint8_t* w, const Value* xt,
                                      const Value* wt, int64_t index, int channels, int k,
                                      Operand xf, Operand wf, Format acc, int lookup) {
    int n = int(index / channels), c = int(index % channels);
    uint32_t state = 0;
    for (int i = 0; i < k; ++i) {
        uint8_t xc = x[int64_t(n)*k+i], wc = w[int64_t(c)*k+i];
        Value left = lookup ? xt[xc] : decode_operand(xc, xf);
        Value right = lookup ? wt[wc] : decode_operand(wc, wf);
        state = fma(state, left, right, acc);
    }
    return state;
}
} // namespace pe
