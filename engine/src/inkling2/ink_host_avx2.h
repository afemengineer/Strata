// engine/src/inkling2/ink_host_avx2.h - Runtime-dispatched AVX2+FMA host dot kernels for InkHybridEngine
// Compiled in a separate TU (ink_host_avx2.cpp) with -mavx2 -mfma -mf16c (/arch:AVX2) and ONLY called
// after strata::quant::caps().avx2 is verified at runtime (fixes #6618 SIGILL on non-AVX2 CPUs).
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdint>

namespace strata::ink::host_avx2 {

// Returns true if this binary includes the AVX2+FMA host TU AND the running CPU supports AVX2+FMA+F16C.
bool available() noexcept;

float dot_sf4_row(const uint8_t* p, const uint8_t* sc, float w_outer, const float* x, int cols,
                  const float* e4_lut) noexcept;
float dot_mx8_row(const uint8_t* w, const uint8_t* sc, const float* x, int cols,
                  const float* e4_lut, const float* u8_lut) noexcept;
float dot_sk_sub64(const uint8_t* pl, int fmt, int g, const int8_t* qs, float x_d, float x_s,
                   float eff, const int8_t (*s15_lut)[5]) noexcept;

// One q8_1 activation group for sk_rows_dot: 64 int8 values in the kernels' planar order
// (planar[k * 16 + j] = natural[4 * j + k]), with scale d and s = d * sum(values).
struct alignas(32) SkActG64 {
    int8_t qs[64];
    float d;
    float s;
    float pad_[6];
};

// Fills `out` from a natural-order q8_1 group (qs[64], d, s). Plain C++: callable without AVX2.
void sk_act_planar(const int8_t* qs, float d, float s, SkActG64& out) noexcept;

// True when sk_rows_dot has a whole-row kernel for `fmt` on this CPU (S2_K and S3_K).
bool sk_rows_supported(int fmt) noexcept;

// Dot products of n_rows rows of an S*_K matrix with T activation vectors.
// A row is n_groups groups of 64 values in superblocks of 4 groups (header: f16 d, 4 u8 sub-scales).
// Group k of a row uses act[t * act_groups + k % act_groups] and is summed into
// out[t * out_stride + row * n_out + k / act_groups], where n_out = n_groups / act_groups.
// For the S2_K/S3_K experts that 8 GB cards leave in RAM: gate/up rows give one output per row
// (act_groups = n_groups = 36), down "pair rows" two (n_groups = 36, act_groups = 18).
// The old per-group dot_sk_sub64 unpacked S2_K/S3_K one byte at a time and reduced every group.
void sk_rows_dot(const uint8_t* rows, size_t row_stride, int n_rows, int fmt, int n_groups,
                 const SkActG64* act, int act_groups, int T, float* out, size_t out_stride) noexcept;


// Global attention over a run of FP16 KV tokens kept in host RAM, for one query token: the CPU part of a decode step
// whose context no longer fits in VRAM (the GPU attends the VRAM part and merges both partial softmaxes).
// q: [24][128] queries; rel: [24][1025] relative bias by distance (min(dist, 1024)); k, v: [n][8][128] FP16 tokens
// whose first token is dist0 positions before the query. Writes, per query head, the partial softmax of these tokens:
// running max m[24], sum l[24] and unnormalised output o[24][128]. 24 query heads share 8 KV heads (3:1).
void attn_f16_partial(const float* q, const float* rel, const uint16_t* k, const uint16_t* v, int n, int dist0,
                      float* m, float* l, float* o) noexcept;

}  // namespace strata::ink::host_avx2
