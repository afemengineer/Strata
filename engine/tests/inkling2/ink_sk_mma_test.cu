// engine/tests/inkling2/ink_sk_mma_test.cu - strata-inkling's small-M S_K tensor-core linear against an FP64 reference
// MIT License - Copyright (c) 2026 Strata contributors
//
// launch_sk_mma runs S5_K / S6_K / S8_K matrices for 1..16 rows of X on FP16 tensor cores (the DSpark drafter's layers,
// the DSpark verify head). Checks values against FP64 for each format, M = 1..16, split and unsplit grids, odd N.
// --bench times the SQ-10 drafter and unembed shapes, and the S6_K GEMV the verify head used before, for comparison.
#include "ink_cuda.cuh"
#include "ink_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::ink;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

double f16d(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = std::ldexp(1.0 + m / 1024.0, e - 15);
    return (h & 0x8000) ? -v : v;
}

// FP64 reference with the layouts of k_ink_sk_to_bf16.
double ref_dot(int fmt, const uint8_t* row, const float* x, int K) {
    const int SB = ink_sb_bytes(fmt);
    const double half = ink_sk_half(fmt);
    double acc = 0.0;
    for (int c = 0; c < K; ++c) {
        const uint8_t* blk = row + (c >> 8) * SB;
        const int idx = c & 255, g = idx >> 6;
        uint16_t d16;
        std::memcpy(&d16, blk, 2);
        const double eff = f16d(d16) * (blk[2 + g] / 255.0);
        const uint8_t* pl = blk + 6;
        double q;
        if (fmt == FMT_S8_K) {
            q = (double)(int8_t)pl[idx];
        } else {
            const int lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15;
            const int hi = (fmt == FMT_S5_K) ? ((pl[128 + (idx >> 3)] >> (idx & 7)) & 1)
                                             : ((pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3);
            q = (lo | (hi << 4)) - half;
        }
        acc += q * eff * x[c];
    }
    return acc;
}

std::vector<uint8_t> make_w(std::mt19937& rng, int fmt, int N, int K) {
    const int SB = ink_sb_bytes(fmt);
    std::vector<uint8_t> w((size_t)N * (K / 256) * SB);
    static const uint16_t kScales[] = {0x2C00, 0x2E66, 0x3000, 0x1F00};
    for (size_t b = 0; b < w.size() / SB; ++b) {
        uint8_t* blk = w.data() + b * SB;
        const uint16_t d = kScales[rng() % 4];
        std::memcpy(blk, &d, 2);
        for (int i = 2; i < SB; ++i) blk[i] = (uint8_t)rng();
    }
    return w;
}

const char* fname(int fmt) { return fmt == FMT_S5_K ? "S5_K" : fmt == FMT_S6_K ? "S6_K" : "S8_K"; }
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("ink_sk_mma_test: SKIP (no CUDA device)\n");
        return 77;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8) {
        std::printf("ink_sk_mma_test: SKIP (needs sm_80+)\n");
        return 77;
    }
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::mt19937 rng(7);
    float* dScr = nullptr;
    const size_t scr_floats = (size_t)16 * 16 * 8192;
    INK_CK(cudaMalloc(&dScr, scr_floats * 4));

    if (bench) {
        struct Shape { int fmt, N, K; const char* what; };
        const Shape shapes[] = {
            {FMT_S5_K, 5120, 2304, "drafter qkv"}, {FMT_S5_K, 18432, 2304, "drafter gate+up"},
            {FMT_S5_K, 2304, 3072, "drafter o_proj"}, {FMT_S5_K, 2304, 9216, "drafter down"},
            {FMT_S6_K, 200058, 2304, "unembed"}};
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (const auto& sh : shapes) {
            const auto w = make_w(rng, sh.fmt, sh.N, sh.K);
            std::vector<float> x((size_t)16 * sh.K);
            for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
            // Enough copies to cycle through 128 MB, so the L2 (24 MB on a 4060) cannot hold the weights between runs.
            const int copies = (int)std::max<size_t>(1, (128ull << 20) / w.size() + 1);
            uint8_t* dW = nullptr;
            float *dX = nullptr, *dY = nullptr;
            INK_CK(cudaMalloc(&dW, w.size() * copies));
            INK_CK(cudaMalloc(&dX, x.size() * 4));
            INK_CK(cudaMalloc(&dY, (size_t)16 * sh.N * 4));
            for (int c = 0; c < copies; ++c) INK_CK(cudaMemcpy(dW + w.size() * c, w.data(), w.size(), cudaMemcpyHostToDevice));
            INK_CK(cudaMemcpy(dX, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            for (int M : {1, 5, 8, 16}) {
                launch_sk_mma(sh.fmt, dW, dX, dY, M, sh.N, sh.K, sh.N, dScr, scr_floats, 0);
                cudaEventRecord(a);
                for (int i = 0; i < 20; ++i)
                    launch_sk_mma(sh.fmt, dW + w.size() * (i % copies), dX, dY, M, sh.N, sh.K, sh.N, dScr, scr_floats, 0);
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, a, b);
                float ms_old = 0.0f;
                if (sh.fmt == FMT_S6_K) {
                    const int grid = (sh.N + 7) / 8;
                    cudaEventRecord(a);
                    for (int i = 0; i < 20; ++i) {
                        if (M == 1)      k_ink_gemv_s6k_multi<1><<<grid, 256>>>(dW, dX, dY, M, sh.N, sh.K, sh.N);
                        else if (M <= 8) k_ink_gemv_s6k_multi<8><<<grid, 256>>>(dW, dX, dY, M, sh.N, sh.K, sh.N);
                        else             k_ink_gemv_s6k_multi<16><<<grid, 256>>>(dW, dX, dY, M, sh.N, sh.K, sh.N);
                    }
                    cudaEventRecord(b);
                    cudaEventSynchronize(b);
                    cudaEventElapsedTime(&ms_old, a, b);
                }
                std::printf("  %-16s %s M=%-2d %.3f ms  %.0f GB/s of weights", sh.what, fname(sh.fmt), M, ms / 20,
                            w.size() / (ms / 20 * 1e-3) / 1e9);
                if (ms_old > 0.0f) std::printf("  (S6_K GEMV %.3f ms)", ms_old / 20);
                std::printf("\n");
            }
            cudaFree(dW);
            cudaFree(dX);
            cudaFree(dY);
        }
        return 0;
    }

    for (int fmt : {FMT_S5_K, FMT_S6_K, FMT_S8_K}) {
        for (const auto& nk : {std::pair<int, int>{1003, 2304}, std::pair<int, int>{37, 9216}}) {
            const int N = nk.first, K = nk.second;
            const auto w = make_w(rng, fmt, N, K);
            std::vector<float> x((size_t)16 * K);
            for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
            uint8_t* dW = nullptr;
            float *dX = nullptr, *dY = nullptr;
            INK_CK(cudaMalloc(&dW, w.size()));
            INK_CK(cudaMalloc(&dX, x.size() * 4));
            INK_CK(cudaMalloc(&dY, (size_t)16 * N * 4));
            INK_CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice));
            INK_CK(cudaMemcpy(dX, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            for (int M : {1, 2, 5, 8, 9, 16}) {
                for (int split : {0, 1}) {
                    INK_CK(cudaMemset(dY, 0, (size_t)16 * N * 4));
                    CHECK(launch_sk_mma(fmt, dW, dX, dY, M, N, K, N, split ? dScr : nullptr, split ? scr_floats : 0, 0));
                    INK_CK(cudaDeviceSynchronize());
                    std::vector<float> y((size_t)M * N);
                    INK_CK(cudaMemcpy(y.data(), dY, y.size() * 4, cudaMemcpyDeviceToHost));
                    double rms = 0.0, worst = 0.0;
                    std::vector<double> ref((size_t)M * N);
                    for (int m = 0; m < M; ++m)
                        for (int n = 0; n < N; ++n) {
                            ref[(size_t)m * N + n] = ref_dot(fmt, w.data() + (size_t)n * (K / 256) * ink_sb_bytes(fmt), x.data() + (size_t)m * K, K);
                            rms += ref[(size_t)m * N + n] * ref[(size_t)m * N + n];
                        }
                    rms = std::sqrt(rms / ref.size());
                    int nbad = 0, first_bad = -1;
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double e = std::fabs(y[i] - ref[i]);
                        worst = std::fmax(worst, e);
                        if (e > 1e-5 * rms) { if (first_bad < 0) first_bad = (int)i; ++nbad; }
                    }
                    if (nbad && std::getenv("SKMMA_DEBUG")) std::printf("    bad %d of %zu, first at m=%d n=%d (row bits %d)\n", nbad, ref.size(),
                                                                       first_bad / N, first_bad % N, (int)(((size_t)(first_bad % N) * (K / 256) * ink_sb_bytes(fmt)) & 3));
                    if (M == 1 || M == 16 || worst / rms >= 1e-5)
                        std::printf("  %s N=%d K=%d M=%-2d split=%d worst err %.2e of rms output\n", fname(fmt), N, K, M, split, worst / rms);
                    CHECK(worst / rms < 1e-5);
                }
            }
            cudaFree(dW);
            cudaFree(dX);
            cudaFree(dY);
        }
    }
    cudaFree(dScr);
    if (failed) {
        std::fprintf(stderr, "ink_sk_mma_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("ink_sk_mma_test: OK\n");
    return 0;
}
