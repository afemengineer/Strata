// engine/tests/core/ink_host_sk_test.cpp - strata-inkling's S2_K/S3_K CPU expert kernels (no GPU)
// MIT License - Copyright (c) 2026 Strata contributors
//
// Checks host_avx2::sk_rows_dot against a scalar reference written from the format layout, for the
// shapes the hybrid engine uses (gate/up rows, down pair rows, 1 and 9 tokens). With --bench it also
// times whole experts against the per-group dot_sk_sub64 path the engine used before.
#include "ink_host_avx2.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace strata::ink::host_avx2;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

constexpr int kS2 = 2, kS3 = 3;
constexpr int kD = 2304, kI = 1152;   // Inkling-2-Mini hidden and expert intermediate sizes
constexpr int kGroups = 36;           // 64-value groups per row (9 superblocks of 4)

int sb_bytes(int fmt) { return fmt == kS2 ? 70 : 102; }
float offset(int fmt) { return fmt == kS2 ? 1.5f : 3.5f; }

float f16_to_f32(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const float v = std::ldexp(1.0f + m / 1024.0f, e - 15);
    return (h & 0x8000) ? -v : v;
}

struct Q8 {
    int8_t qs[64];
    float d, s;
};

// Natural-order value e of group gi in a superblock payload.
int sk_value(int fmt, const uint8_t* pl, int gi, int e) {
    int v = (pl[gi * 16 + e / 4] >> (2 * (e % 4))) & 3;
    if (fmt == kS3) v |= ((pl[64 + gi * 8 + e / 8] >> (e % 8)) & 1) << 2;
    return v;
}

// Scalar reference: group k of a row adds eff * x.d * sum((w - offset) * q) to output k / act_groups.
void ref_rows(const uint8_t* rows, size_t stride, int n_rows, int fmt, const std::vector<Q8>& act, int act_groups,
              int T, std::vector<double>& out) {
    const int n_out = kGroups / act_groups;
    out.assign((size_t)T * n_rows * n_out, 0.0);
    for (int t = 0; t < T; ++t)
        for (int r = 0; r < n_rows; ++r)
            for (int k = 0; k < kGroups; ++k) {
                const uint8_t* blk = rows + (size_t)r * stride + (size_t)(k / 4) * sb_bytes(fmt);
                uint16_t d16;
                std::memcpy(&d16, blk, 2);
                const double eff = f16_to_f32(d16) * (blk[2 + k % 4] / 255.0);
                const Q8& a = act[(size_t)t * act_groups + k % act_groups];
                double sum = 0.0;
                for (int e = 0; e < 64; ++e) sum += (sk_value(fmt, blk + 6, k % 4, e) - offset(fmt)) * a.qs[e];
                out[((size_t)t * n_rows + r) * n_out + k / act_groups] += eff * a.d * sum;
            }
}

void fill_rows(std::mt19937& rng, uint8_t* rows, size_t stride, int n_rows, int fmt) {
    static const uint16_t kScales[] = {0x2C00, 0x2E66, 0x3000, 0x1F00};
    for (int r = 0; r < n_rows; ++r)
        for (int s = 0; s < kGroups / 4; ++s) {
            uint8_t* blk = rows + (size_t)r * stride + (size_t)s * sb_bytes(fmt);
            const uint16_t d16 = kScales[rng() % 4];
            std::memcpy(blk, &d16, 2);
            for (int i = 2; i < sb_bytes(fmt); ++i) blk[i] = (uint8_t)rng();
        }
}

std::vector<Q8> make_act(std::mt19937& rng, int n) {
    std::vector<Q8> v(n);
    for (auto& a : v) {
        int sum = 0;
        for (auto& q : a.qs) { q = (int8_t)((int)(rng() % 255) - 127); sum += q; }
        a.d = 0.001f + (rng() % 1000) * 1e-5f;
        a.s = a.d * (float)sum;
    }
    return v;
}

std::vector<SkActG64> planar(const std::vector<Q8>& v) {
    std::vector<SkActG64> p(v.size());
    for (size_t i = 0; i < v.size(); ++i) sk_act_planar(v[i].qs, v[i].d, v[i].s, p[i]);
    return p;
}

// Runs one shape and compares against the reference; `stride` > row bytes models interleaved gate/up rows.
void check_shape(std::mt19937& rng, int fmt, int n_rows, int act_groups, int T, size_t stride) {
    const size_t row_b = (size_t)(kGroups / 4) * sb_bytes(fmt);
    std::vector<uint8_t> rows(stride * n_rows + row_b, 0);
    fill_rows(rng, rows.data(), stride, n_rows, fmt);
    const auto act = make_act(rng, T * act_groups);
    const auto pa = planar(act);
    const int n_out = kGroups / act_groups;
    std::vector<float> got((size_t)T * n_rows * n_out, -1.0f);
    sk_rows_dot(rows.data(), stride, n_rows, fmt, kGroups, pa.data(), act_groups, T, got.data(), (size_t)n_rows * n_out);
    std::vector<double> want;
    ref_rows(rows.data(), stride, n_rows, fmt, act, act_groups, T, want);
    // Error relative to the typical output size: single outputs can nearly cancel, so a per-output
    // relative error measures float32 rounding of the 36 group terms rather than correctness.
    double rms = 0.0, worst = 0.0;
    for (double w : want) rms += w * w;
    rms = std::sqrt(rms / (double)want.size());
    for (size_t i = 0; i < want.size(); ++i) worst = std::fmax(worst, std::fabs(got[i] - want[i]));
    worst /= rms;
    std::printf("  fmt=S%d_K rows=%d act_groups=%d T=%d stride=%zu: worst err %.2e of rms output\n", fmt, n_rows, act_groups, T, stride, worst);
    CHECK(worst < 1e-5);
}

// ---- --bench: whole experts, new kernel vs the old per-group path, single and multi thread ----

struct Expert {
    std::vector<uint8_t> gu, dn;  // gate/up rows (2*kI rows, S3_K) and down pair rows (kD/2, S2_K)
};

Expert make_expert(std::mt19937& rng) {
    Expert e;
    const size_t gu_row = 9 * 102, dn_row = 9 * 70;
    e.gu.resize(gu_row * 2 * kI);
    e.dn.resize(dn_row * (kD / 2));
    fill_rows(rng, e.gu.data(), gu_row, 2 * kI, kS3);
    fill_rows(rng, e.dn.data(), dn_row, kD / 2, kS2);
    return e;
}

volatile float g_sink = 0.0f;

void run_old(const Expert& e, const std::vector<Q8>& xq, const std::vector<Q8>& hq, float* y) {
    const size_t gu_row = 9 * 102, dn_row = 9 * 70;
    float act[kI];
    for (int j = 0; j < kI; ++j) {
        float g = 0.0f, u = 0.0f;
        for (int k = 0; k < kGroups; ++k) {
            const uint8_t* bg = e.gu.data() + (size_t)(2 * j) * gu_row + (k / 4) * 102;
            const uint8_t* bu = bg + gu_row;
            uint16_t dg, du;
            std::memcpy(&dg, bg, 2); std::memcpy(&du, bu, 2);
            const Q8& a = xq[k];
            g += dot_sk_sub64(bg + 6, kS3, k % 4, a.qs, a.d, a.s, f16_to_f32(dg) * (bg[2 + k % 4] / 255.0f), nullptr);
            u += dot_sk_sub64(bu + 6, kS3, k % 4, a.qs, a.d, a.s, f16_to_f32(du) * (bu[2 + k % 4] / 255.0f), nullptr);
        }
        act[j] = g * u;
    }
    for (int pr = 0; pr < kD / 2; ++pr) {
        float y0 = 0.0f, y1 = 0.0f;
        for (int k = 0; k < kGroups; ++k) {
            const uint8_t* blk = e.dn.data() + (size_t)pr * dn_row + (k / 4) * 70;
            uint16_t d16;
            std::memcpy(&d16, blk, 2);
            const Q8& a = hq[k % 18];
            const float v = dot_sk_sub64(blk + 6, kS2, k % 4, a.qs, a.d, a.s, f16_to_f32(d16) * (blk[2 + k % 4] / 255.0f), nullptr);
            if (k < 18) y0 += v; else y1 += v;
        }
        y[2 * pr] = y0; y[2 * pr + 1] = y1;
    }
    g_sink = g_sink + act[0];
}

void run_new(const Expert& e, const std::vector<SkActG64>& xq, const std::vector<SkActG64>& hq, float* y) {
    std::vector<float> g(kI), u(kI);
    sk_rows_dot(e.gu.data(), 2 * 9 * 102, kI, kS3, kGroups, xq.data(), kGroups, 1, g.data(), kI);
    sk_rows_dot(e.gu.data() + 9 * 102, 2 * 9 * 102, kI, kS3, kGroups, xq.data(), kGroups, 1, u.data(), kI);
    sk_rows_dot(e.dn.data(), 9 * 70, kD / 2, kS2, kGroups, hq.data(), 18, 1, y, kD);
    g_sink = g_sink + g[0] * u[0];
}

int bench() {
    std::mt19937 rng(7);
    const int n_exp = 64;  // ~180 MB: beyond any L3, like the engine's 5 GB of RAM experts
    std::vector<Expert> ex;
    for (int i = 0; i < n_exp; ++i) ex.push_back(make_expert(rng));
    const auto xq = make_act(rng, kGroups), hq = make_act(rng, 18);
    const auto pxq = planar(xq), phq = planar(hq);
    const double mb = (ex[0].gu.size() + ex[0].dn.size()) / 1048576.0;
    std::printf("expert = %.2f MiB (S3_K gate/up + S2_K down), %d experts cycled\n", mb, n_exp);

    auto time_it = [&](bool use_new, int threads, int per_thread) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int th = 0; th < threads; ++th)
            pool.emplace_back([&, th]() {
                std::vector<float> y(kD);
                for (int i = 0; i < per_thread; ++i) {
                    const Expert& e = ex[(th * 7 + i) % n_exp];
                    if (use_new) run_new(e, pxq, phq, y.data());
                    else run_old(e, xq, hq, y.data());
                }
            });
        for (auto& p : pool) p.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const int n = threads * per_thread;
        std::printf("  %-3s threads=%-2d %6.0f us/expert  %6.1f GB/s\n", use_new ? "new" : "old", threads,
                    1e6 * s / n, n * mb / 1024.0 / s);
    };
    // Several tokens per expert (DSpark verify, prompt batches): weights read once, T dot products per group.
    for (int T : {1, 4, 8, 16}) {
        const auto xt = planar(make_act(rng, T * kGroups)), ht = planar(make_act(rng, T * 18));
        const int threads = (int)std::thread::hardware_concurrency(), per = 32;
        auto t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> pool;
        for (int th = 0; th < threads; ++th)
            pool.emplace_back([&, th]() {
                std::vector<float> g((size_t)T * kI), u((size_t)T * kI), y((size_t)T * kD);
                for (int i = 0; i < per; ++i) {
                    const Expert& e = ex[(th * 7 + i) % n_exp];
                    sk_rows_dot(e.gu.data(), 2 * 9 * 102, kI, kS3, kGroups, xt.data(), kGroups, T, g.data(), kI);
                    sk_rows_dot(e.gu.data() + 9 * 102, 2 * 9 * 102, kI, kS3, kGroups, xt.data(), kGroups, T, u.data(), kI);
                    sk_rows_dot(e.dn.data(), 9 * 70, kD / 2, kS2, kGroups, ht.data(), 18, T, y.data(), kD);
                }
            });
        for (auto& p : pool) p.join();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("  T=%-2d threads=%d %6.1f us/expert (all threads)  %5.1f us per token-expert\n", T, threads,
                    1e6 * s / (threads * per), 1e6 * s / (threads * per) / T);
    }
    time_it(false, 1, 16);
    time_it(true, 1, 128);
    const int hw = (int)std::thread::hardware_concurrency();
    for (int th : {4, 8, hw / 2, hw}) {
        if (th < 2) continue;
        time_it(false, th, 8);
        time_it(true, th, 64);
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (!sk_rows_supported(kS2) || !sk_rows_supported(kS3)) {
        std::printf("ink_host_sk_test: SKIP (no AVX2 host kernels in this build or CPU)\n");
        return 77;
    }
    if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) return bench();

    std::mt19937 rng(1234);
    for (int fmt : {kS2, kS3}) {
        const size_t row_b = (size_t)(kGroups / 4) * sb_bytes(fmt);
        check_shape(rng, fmt, 64, kGroups, 1, row_b);      // gate/up rows, one token
        check_shape(rng, fmt, 64, kGroups, 1, 2 * row_b);  // every other row (gate/up interleaved)
        check_shape(rng, fmt, 64, 18, 1, row_b);           // down pair rows: two outputs per row
        check_shape(rng, fmt, 32, kGroups, 9, row_b);      // verify batch of 9 tokens
        check_shape(rng, fmt, 32, 18, 16, row_b);          // the batch maximum
    }
    // Planar order is a fixed transpose of the natural order.
    {
        int8_t nat[64];
        for (int i = 0; i < 64; ++i) nat[i] = (int8_t)i;
        SkActG64 p{};
        sk_act_planar(nat, 1.0f, 2.0f, p);
        CHECK(p.qs[0] == 0 && p.qs[1] == 4 && p.qs[16] == 1 && p.qs[63] == 63 && p.d == 1.0f && p.s == 2.0f);
    }
    if (failed != 0) {
        std::fprintf(stderr, "ink_host_sk_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("ink_host_sk_test: OK\n");
    return 0;
}
