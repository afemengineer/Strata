// engine/src/inkling2/ink_host_avx2.cpp - Per-TU AVX2+FMA host dot kernels for InkHybridEngine
// Compiled ONLY with -mavx2 -mfma -mf16c (MSVC /arch:AVX2) when STRATA_ISA_FLOOR allows AVX2;
// guarded at runtime by strata::quant::caps().avx2 so non-AVX2 CPUs (e.g. Ivy Bridge Xeon E5 v2, #6618)
// never execute an AVX2/FMA instruction.
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_host_avx2.h"
#include "strata/quant/isa.hpp"

#include <cstdlib>
#include <cstring>

#if defined(INK_BUILD_HOST_AVX2) && defined(__AVX2__)
#include <immintrin.h>
#define INK_TU_HAS_AVX2 1
#endif

namespace strata::ink::host_avx2 {

void sk_act_planar(const int8_t* qs, float d, float s, SkActG64& out) noexcept {
    for (int j = 0; j < 16; ++j)
        for (int k = 0; k < 4; ++k) out.qs[k * 16 + j] = qs[4 * j + k];
    out.d = d;
    out.s = s;
}

bool available() noexcept {
#if defined(INK_TU_HAS_AVX2)
    static const bool ok = strata::quant::caps().avx2;
    return ok;
#else
    return false;
#endif
}

#if defined(INK_TU_HAS_AVX2)

namespace {

// Format constants matching InkFmt in ink_cuda.cuh
constexpr int kFmtS1_5 = 1;
constexpr int kFmtS2_K = 2;
constexpr int kFmtS3_K = 3;
constexpr int kFmtS4_K = 4;
constexpr int kFmtS5_K = 5;
constexpr int kFmtS6_K = 6;
constexpr int kFmtS8_K = 8;

inline float avx2_hsum_ps(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline int32_t avx2_hsum_epi32(__m256i v) noexcept {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}

inline int32_t avx2_dot_u8_i8_64(__m256i u0, __m256i u1, const int8_t* qs) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
    const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(u0, q0), ones);
    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(u1, q1), ones);
    return avx2_hsum_epi32(_mm256_add_epi32(p0, p1));
}

inline void s15_unpack64_local(const uint8_t* pl, int g, int8_t out[64], const int8_t (*lut)[5]) noexcept {
    const int base = g * 64;
    int i = 0;
    int byte_idx = base / 5;
    int rem = base % 5;
    if (rem != 0) {
        const int8_t* t = lut[pl[byte_idx++]];
        for (int k = rem; k < 5 && i < 64; ++k) out[i++] = t[k];
    }
    while (i + 5 <= 64) {
        const int8_t* t = lut[pl[byte_idx++]];
        out[i + 0] = t[0]; out[i + 1] = t[1]; out[i + 2] = t[2]; out[i + 3] = t[3]; out[i + 4] = t[4];
        i += 5;
    }
    if (i < 64) {
        const int8_t* t = lut[pl[byte_idx]];
        for (int k = 0; i < 64; ++k) out[i++] = t[k];
    }
}

}  // namespace

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept {
    const int nb = cols >> 4;
    const __m128i lut2x  = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i mask0f = _mm_set1_epi8(0x0F);
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p + b * 8));
        const __m128i lo  = _mm_and_si128(raw, mask0f);
        const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
        const __m128i nib = _mm_unpacklo_epi8(lo, hi);
        const __m128i w8  = _mm_shuffle_epi8(lut2x, nib);
        const __m256 wf0  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(w8));
        const __m256 wf1  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(w8, 8)));
        const float* xb   = x + b * 16;
        const __m256 prod = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(xb), _mm256_mul_ps(wf1, _mm256_loadu_ps(xb + 8)));
        vsum = _mm256_fmadd_ps(prod, _mm256_set1_ps(0.5f * e4_lut[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum) * w_outer;
}

float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* e4_lut, const float* u8_lut) noexcept {
    const int nb = cols >> 5;
    __m256 vsum = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t* wb = w + b * 32;
        const float* xb = x + b * 32;
        __m256 acc = _mm256_setzero_ps();
        for (int k = 0; k < 32; k += 8) {
            const __m256 wf = _mm256_setr_ps(
                e4_lut[wb[k + 0]], e4_lut[wb[k + 1]], e4_lut[wb[k + 2]], e4_lut[wb[k + 3]],
                e4_lut[wb[k + 4]], e4_lut[wb[k + 5]], e4_lut[wb[k + 6]], e4_lut[wb[k + 7]]);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xb + k), acc);
        }
        vsum = _mm256_fmadd_ps(acc, _mm256_set1_ps(u8_lut[sc[b]]), vsum);
    }
    return avx2_hsum_ps(vsum);
}

float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept {
    int32_t isum = 0;
    if (fmt == kFmtS1_5) {
        alignas(32) int8_t tr[64];
        s15_unpack64_local(pl, g, tr, s15_lut);
        const __m256i bias = _mm256_set1_epi8(1);
        const __m256i u0 = _mm256_add_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(tr)), bias);
        const __m256i u1 = _mm256_add_epi8(_mm256_load_si256(reinterpret_cast<const __m256i*>(tr + 32)), bias);
        isum = avx2_dot_u8_i8_64(u0, u1, qs);
        return eff * (x_d * (float)isum - x_s);
    } else if (fmt == kFmtS4_K) {
        const uint8_t* p = pl + g * 32;
        const __m128i m0f = _mm_set1_epi8(0x0F);
        const __m128i v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
        const __m128i lo0 = _mm_and_si128(v0, m0f), hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m0f);
        const __m128i lo1 = _mm_and_si128(v1, m0f), hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m0f);
        const __m256i u0 = _mm256_set_m128i(_mm_unpackhi_epi8(lo0, hi0), _mm_unpacklo_epi8(lo0, hi0));
        const __m256i u1 = _mm256_set_m128i(_mm_unpackhi_epi8(lo1, hi1), _mm_unpacklo_epi8(lo1, hi1));
        isum = avx2_dot_u8_i8_64(u0, u1, qs);
        return eff * (x_d * (float)isum - 7.5f * x_s);
    } else if (fmt == kFmtS8_K) {
        const int8_t* p = reinterpret_cast<const int8_t*>(pl + g * 64);
        const __m256i p0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
        const __m256i p1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 32));
        const __m256i q0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs));
        const __m256i q1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qs + 32));
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i a0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p0), _mm256_sign_epi8(q0, p0)), ones);
        const __m256i a1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(p1), _mm256_sign_epi8(q1, p1)), ones);
        isum = avx2_hsum_epi32(_mm256_add_epi32(a0, a1));
        return eff * x_d * (float)isum;
    } else {
        alignas(32) uint8_t u[64];
        if (fmt == kFmtS2_K) {
            const uint8_t* p = pl + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t b = p[i];
                u[4 * i + 0] = (uint8_t)(b & 3u);
                u[4 * i + 1] = (uint8_t)((b >> 2) & 3u);
                u[4 * i + 2] = (uint8_t)((b >> 4) & 3u);
                u[4 * i + 3] = (uint8_t)(b >> 6);
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (x_d * (float)isum - 1.5f * x_s);
        } else if (fmt == kFmtS3_K) {
            const uint8_t* p = pl + g * 16;
            const uint8_t* hp = pl + 64 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t b0 = p[2 * i], b1 = p[2 * i + 1], h = hp[i];
                u[8 * i + 0] = (uint8_t)((b0 & 3u) | ((h & 1u) << 2));
                u[8 * i + 1] = (uint8_t)(((b0 >> 2) & 3u) | (((h >> 1) & 1u) << 2));
                u[8 * i + 2] = (uint8_t)(((b0 >> 4) & 3u) | (((h >> 2) & 1u) << 2));
                u[8 * i + 3] = (uint8_t)((b0 >> 6) | (((h >> 3) & 1u) << 2));
                u[8 * i + 4] = (uint8_t)((b1 & 3u) | (((h >> 4) & 1u) << 2));
                u[8 * i + 5] = (uint8_t)(((b1 >> 2) & 3u) | (((h >> 5) & 1u) << 2));
                u[8 * i + 6] = (uint8_t)(((b1 >> 4) & 3u) | (((h >> 6) & 1u) << 2));
                u[8 * i + 7] = (uint8_t)((b1 >> 6) | (((h >> 7) & 1u) << 2));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (x_d * (float)isum - 3.5f * x_s);
        } else if (fmt == kFmtS5_K) {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 8;
            for (int i = 0; i < 8; ++i) {
                const uint8_t h = hp[i];
                for (int k = 0; k < 4; ++k) {
                    const uint8_t b = p[4 * i + k];
                    u[8 * i + 2 * k + 0] = (uint8_t)((b & 0xFu) | (((h >> (2 * k)) & 1u) << 4));
                    u[8 * i + 2 * k + 1] = (uint8_t)((b >> 4) | (((h >> (2 * k + 1)) & 1u) << 4));
                }
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (x_d * (float)isum - 15.5f * x_s);
        } else if (fmt == kFmtS6_K) {
            const uint8_t* p = pl + g * 32;
            const uint8_t* hp = pl + 128 + g * 16;
            for (int i = 0; i < 16; ++i) {
                const uint8_t h = hp[i], b0 = p[2 * i], b1 = p[2 * i + 1];
                u[4 * i + 0] = (uint8_t)((b0 & 0xFu) | ((h & 3u) << 4));
                u[4 * i + 1] = (uint8_t)((b0 >> 4) | (((h >> 2) & 3u) << 4));
                u[4 * i + 2] = (uint8_t)((b1 & 0xFu) | (((h >> 4) & 3u) << 4));
                u[4 * i + 3] = (uint8_t)((b1 >> 4) | ((h >> 6) << 4));
            }
            isum = avx2_dot_u8_i8_64(_mm256_load_si256(reinterpret_cast<const __m256i*>(u)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(u + 32)), qs);
            return eff * (x_d * (float)isum - 31.5f * x_s);
        }
    }
    return 0.0f;
}

namespace {

constexpr int kSkMaxT = 16;

// S2_K / S3_K low 2 bits of one 64-value group (16 bytes, value 4j+k in bits 2k..2k+1 of byte j),
// unpacked in planar order: w0 = values k=0 (bytes 0..15) and k=1 (16..31), w1 = k=2 and k=3.
// srlv on 32-bit lanes shifts neighbour bits into each byte's top, which the mask drops.
inline void sk_lo2_planar(const uint8_t* p, __m256i& w0, __m256i& w1) noexcept {
    const __m256i b  = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    const __m256i m3 = _mm256_set1_epi8(3);
    w0 = _mm256_and_si256(_mm256_srlv_epi32(b, _mm256_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2)), m3);
    w1 = _mm256_and_si256(_mm256_srlv_epi32(b, _mm256_setr_epi32(4, 4, 4, 4, 6, 6, 6, 6)), m3);
}

// S3_K third bit (8 bytes, value e in bit e%8 of byte e/8) added as bit 2 of the planar values.
// Value 4j+k lives in byte j/2, bit 4*(j%2)+k: expand each byte to two (low/high nibble), then
// pick bit k per planar quarter.
inline void sk_hi1_planar(const uint8_t* hp, __m256i& w0, __m256i& w1) noexcept {
    const __m128i h8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(hp));
    const __m128i h2  = _mm_shuffle_epi8(h8, _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7));
    const __m128i odd = _mm_setr_epi8(0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
    const __m128i hx  = _mm_blendv_epi8(h2, _mm_srli_epi16(h2, 4), odd);
    const __m256i hb  = _mm256_broadcastsi128_si256(hx);
    const __m256i one = _mm256_set1_epi8(1);
    const __m256i h01 = _mm256_and_si256(_mm256_srlv_epi32(hb, _mm256_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1)), one);
    const __m256i h23 = _mm256_and_si256(_mm256_srlv_epi32(hb, _mm256_setr_epi32(2, 2, 2, 2, 3, 3, 3, 3)), one);
    w0 = _mm256_or_si256(w0, _mm256_slli_epi16(h01, 2));
    w1 = _mm256_or_si256(w1, _mm256_slli_epi16(h23, 2));
}

inline __m256i sk_dot_u8_i8(__m256i w0, __m256i w1, const int8_t* qs) noexcept {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i q0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qs));
    const __m256i q1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(qs + 32));
    return _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(w0, q0), ones),
                            _mm256_madd_epi16(_mm256_maddubs_epi16(w1, q1), ones));
}

template <int FMT>
void sk_rows_dot_fmt(const uint8_t* rows, size_t row_stride, int n_rows, int n_groups,
                     const SkActG64* act, int act_groups, int T, float* out, size_t out_stride) noexcept {
    constexpr int sb = (FMT == kFmtS2_K) ? 70 : 102;
    constexpr float off = (FMT == kFmtS2_K) ? 1.5f : 3.5f;
    const int n_out = n_groups / act_groups;
    const int n_sb = n_groups / 4;
    for (int r = 0; r < n_rows; ++r) {
        const uint8_t* row = rows + (size_t)r * row_stride;
        __m256 acc[kSkMaxT][2];
        float corr[kSkMaxT][2];
        for (int t = 0; t < T; ++t) {
            acc[t][0] = acc[t][1] = _mm256_setzero_ps();
            corr[t][0] = corr[t][1] = 0.0f;
        }
        int ai = 0, o = 0;  // k % act_groups and k / act_groups, stepped instead of divided per group
        for (int s = 0; s < n_sb; ++s) {
            const uint8_t* blk = row + (size_t)s * sb;
            uint16_t d16;
            std::memcpy(&d16, blk, 2);
            const float d = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d16))) * (1.0f / 255.0f);
            const uint8_t* pl = blk + 6;
            for (int gi = 0; gi < 4; ++gi, ++ai) {
                if (ai == act_groups) { ai = 0; ++o; }
                const float eff = d * (float)blk[2 + gi];
                __m256i w0, w1;
                sk_lo2_planar(pl + gi * 16, w0, w1);
                if (FMT == kFmtS3_K) sk_hi1_planar(pl + 64 + gi * 8, w0, w1);
                for (int t = 0; t < T; ++t) {
                    const SkActG64& a = act[(size_t)t * act_groups + ai];
                    const __m256 f = _mm256_cvtepi32_ps(sk_dot_u8_i8(w0, w1, a.qs));
                    acc[t][o] = _mm256_fmadd_ps(f, _mm256_set1_ps(eff * a.d), acc[t][o]);
                    corr[t][o] += eff * a.s;
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            float* dst = out + (size_t)t * out_stride + (size_t)r * n_out;
            for (int o = 0; o < n_out; ++o) dst[o] = avx2_hsum_ps(acc[t][o]) - off * corr[t][o];
        }
    }
}

}  // namespace

bool sk_rows_supported(int fmt) noexcept {
    // STRATA_INK_HOST_SK_OLD=1 keeps the per-group path, to compare the two on one build.
    static const bool on = available() && !std::getenv("STRATA_INK_HOST_SK_OLD");
    return on && (fmt == kFmtS2_K || fmt == kFmtS3_K);
}

void sk_rows_dot(const uint8_t* rows, size_t row_stride, int n_rows, int fmt, int n_groups,
                 const SkActG64* act, int act_groups, int T, float* out, size_t out_stride) noexcept {
    T = T < kSkMaxT ? T : kSkMaxT;
    if (fmt == kFmtS2_K) sk_rows_dot_fmt<kFmtS2_K>(rows, row_stride, n_rows, n_groups, act, act_groups, T, out, out_stride);
    else if (fmt == kFmtS3_K) sk_rows_dot_fmt<kFmtS3_K>(rows, row_stride, n_rows, n_groups, act, act_groups, T, out, out_stride);
}

namespace {

// exp(x) for 8 lanes, |rel err| < 2e-7 on [-87, 88] (Cephes polynomial); x below -87 gives 0.
inline __m256 exp8(__m256 x) noexcept {
    const __m256 hi = _mm256_set1_ps(88.3762626647949f), lo = _mm256_set1_ps(-87.3365447505531f);
    const __m256 under = _mm256_cmp_ps(x, lo, _CMP_LT_OQ);
    x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_andnot_ps(under, _mm256_mul_ps(y, _mm256_castsi256_ps(e)));
}

inline float hsum8(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline float hmax8(__m256 v) noexcept {
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline __m256 ld_h8(const uint16_t* p) noexcept { return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))); }

}  // namespace

void attn_f16_partial(const float* q, const float* rel, const uint16_t* k, const uint16_t* v, int n, int dist0,
                      float* m, float* l, float* o) noexcept {
    constexpr int HD = 128, KVH = 8, B = 16;
    constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128), as on the GPU
    for (int h = 0; h < 24; ++h) { m[h] = -1e30f; l[h] = 0.0f; }
    std::memset(o, 0, sizeof(float) * 24 * HD);
    alignas(32) float s[3][B];
    for (int kh = 0; kh < KVH; ++kh) {
        const float* q3 = q + (size_t)3 * kh * HD;
        for (int j0 = 0; j0 < n; j0 += B) {
            const int nb = (n - j0) < B ? (n - j0) : B;
            for (int j = 0; j < nb; ++j) {
                const uint16_t* kp = k + ((size_t)(j0 + j) * KVH + kh) * HD;
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0;
                for (int d = 0; d < HD; d += 8) {
                    const __m256 kv = ld_h8(kp + d);
                    a0 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q3 + d), a0);
                    a1 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q3 + HD + d), a1);
                    a2 = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q3 + 2 * HD + d), a2);
                }
                int dist = dist0 - (j0 + j);
                dist = dist < 0 ? 0 : (dist > 1024 ? 1024 : dist);
                const size_t rb = (size_t)3 * kh * 1025 + dist;
                s[0][j] = hsum8(a0) * kScale + rel[rb];
                s[1][j] = hsum8(a1) * kScale + rel[rb + 1025];
                s[2][j] = hsum8(a2) * kScale + rel[rb + 2 * 1025];
            }
            for (int j = nb; j < B; ++j) s[0][j] = s[1][j] = s[2][j] = -1e30f;
            for (int r = 0; r < 3; ++r) {
                const int h = 3 * kh + r;
                const __m256 s0 = _mm256_load_ps(s[r]), s1 = _mm256_load_ps(s[r] + 8);
                const float mb = hmax8(_mm256_max_ps(s0, s1));
                const float mn = mb > m[h] ? mb : m[h];
                const __m256 mnv = _mm256_set1_ps(mn);
                const __m256 p0 = exp8(_mm256_sub_ps(s0, mnv)), p1 = exp8(_mm256_sub_ps(s1, mnv));
                const __m256 al = exp8(_mm256_set1_ps(m[h] - mn));
                const float alpha = _mm256_cvtss_f32(al);
                alignas(32) float p[B];
                _mm256_store_ps(p, p0);
                _mm256_store_ps(p + 8, p1);
                l[h] = l[h] * alpha + hsum8(_mm256_add_ps(p0, p1));
                m[h] = mn;
                float* oh = o + (size_t)h * HD;
                __m256 acc[HD / 8];
                for (int d = 0; d < HD / 8; ++d) acc[d] = _mm256_mul_ps(_mm256_loadu_ps(oh + 8 * d), al);
                for (int j = 0; j < nb; ++j) {
                    const uint16_t* vp = v + ((size_t)(j0 + j) * KVH + kh) * HD;
                    const __m256 pj = _mm256_set1_ps(p[j]);
                    for (int d = 0; d < HD / 8; ++d) acc[d] = _mm256_fmadd_ps(ld_h8(vp + 8 * d), pj, acc[d]);
                }
                for (int d = 0; d < HD / 8; ++d) _mm256_storeu_ps(oh + 8 * d, acc[d]);
            }
        }
    }
}

#else

bool sk_rows_supported(int) noexcept { return false; }
void attn_f16_partial(const float*, const float*, const uint16_t*, const uint16_t*, int, int, float* m, float* l, float*) noexcept {
    for (int h = 0; h < 24; ++h) { m[h] = -1e30f; l[h] = 0.0f; }
}

void sk_rows_dot(const uint8_t*, size_t, int, int, int, const SkActG64*, int, int, float*, size_t) noexcept {}

float dot_sf4_row(const uint8_t*, const uint8_t*, float, const float*, int, const float*) noexcept { return 0.0f; }
float dot_mx8_row(const uint8_t*, const uint8_t*, const float*, int, const float*, const float*) noexcept { return 0.0f; }
float dot_sk_sub64(const uint8_t*, int, int, const int8_t*, float, float, float, const int8_t (*)[5]) noexcept { return 0.0f; }

#endif

}  // namespace strata::ink::host_avx2
