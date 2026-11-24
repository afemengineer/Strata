// engine/tests/inkling2/ink_s6k_gemv_test.cu - strata-inkling's small-M S6_K GEMV against an FP64 reference (CUDA)
// MIT License - Copyright (c) 2026 Strata contributors
//
// k_ink_gemv_s6k_multi reads the S6_K unembed once per decode token / DSpark verify instead of expanding it to BF16.
// Checks: values against FP64 for M = 1..16, and that a row's result does not depend on M (bit-identical), which keeps
// greedy speculative decoding equal to plain greedy decoding. --bench times the SQ-10 unembed shape.
#include "ink_cuda.cuh"
#include "ink_kernels.cuh"

#include <cmath>
#include <cstdio>
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

constexpr int SB = ink_sb_bytes(FMT_S6_K);

double f16d(uint16_t h) {
    const int e = (h >> 10) & 31, m = h & 1023;
    const double v = std::ldexp(1.0 + m / 1024.0, e - 15);
    return (h & 0x8000) ? -v : v;
}

// FP64 reference with the same S6_K layout as k_ink_sk_to_bf16 (without its BF16 rounding).
double ref_dot(const uint8_t* row, const float* x, int K) {
    double acc = 0.0;
    for (int c = 0; c < K; ++c) {
        const uint8_t* blk = row + (c >> 8) * SB;
        const int idx = c & 255, g = idx >> 6;
        uint16_t d16;
        std::memcpy(&d16, blk, 2);
        const double eff = f16d(d16) * (blk[2 + g] / 255.0);
        const uint8_t* pl = blk + 6;
        const int lo = (pl[idx >> 1] >> ((idx & 1) * 4)) & 15;
        const int hi = (pl[128 + (idx >> 2)] >> ((idx & 3) * 2)) & 3;
        acc += ((lo | (hi << 4)) - 31.5) * eff * x[c];
    }
    return acc;
}

void run(const uint8_t* dW, const float* dX, float* dY, int M, int N, int K) {
    const int grid = (N + 7) / 8;
    if (M == 1)      k_ink_gemv_s6k_multi<1><<<grid, 256>>>(dW, dX, dY, M, N, K, N);
    else if (M <= 4) k_ink_gemv_s6k_multi<4><<<grid, 256>>>(dW, dX, dY, M, N, K, N);
    else if (M <= 8) k_ink_gemv_s6k_multi<8><<<grid, 256>>>(dW, dX, dY, M, N, K, N);
    else             k_ink_gemv_s6k_multi<16><<<grid, 256>>>(dW, dX, dY, M, N, K, N);
}

std::vector<uint8_t> make_w(std::mt19937& rng, int N, int K) {
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
}  // namespace

int main(int argc, char** argv) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("ink_s6k_gemv_test: SKIP (no CUDA device)\n");
        return 77;
    }
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    std::mt19937 rng(99);
    const int K = 2304, N = bench ? 200058 : 1003;  // odd N: a partial last block
    const auto w = make_w(rng, N, K);
    std::vector<float> x((size_t)16 * K);
    for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);

    uint8_t* dW = nullptr;
    float *dX = nullptr, *dY = nullptr;
    INK_CK(cudaMalloc(&dW, w.size()));
    INK_CK(cudaMalloc(&dX, x.size() * 4));
    INK_CK(cudaMalloc(&dY, (size_t)16 * N * 4));
    INK_CK(cudaMemcpy(dW, w.data(), w.size(), cudaMemcpyHostToDevice));
    INK_CK(cudaMemcpy(dX, x.data(), x.size() * 4, cudaMemcpyHostToDevice));

    if (bench) {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        for (int M : {1, 5, 9, 16}) {
            run(dW, dX, dY, M, N, K);
            cudaEventRecord(a);
            for (int i = 0; i < 20; ++i) run(dW, dX, dY, M, N, K);
            cudaEventRecord(b);
            cudaEventSynchronize(b);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, a, b);
            std::printf("  M=%-2d %.3f ms  %.0f GB/s of weights\n", M, ms / 20, w.size() / (ms / 20 * 1e-3) / 1e9);
        }
        return 0;
    }

    std::vector<float> y0;  // row results with M = 1, to compare against larger M
    for (int M : {1, 2, 4, 5, 8, 9, 12, 16}) {
        INK_CK(cudaMemset(dY, 0, (size_t)16 * N * 4));
        run(dW, dX, dY, M, N, K);
        INK_CK(cudaDeviceSynchronize());
        std::vector<float> y((size_t)M * N);
        INK_CK(cudaMemcpy(y.data(), dY, y.size() * 4, cudaMemcpyDeviceToHost));
        double rms = 0.0, worst = 0.0;
        std::vector<double> ref((size_t)M * N);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                ref[(size_t)m * N + n] = ref_dot(w.data() + (size_t)n * (K / 256) * SB, x.data() + (size_t)m * K, K);
                rms += ref[(size_t)m * N + n] * ref[(size_t)m * N + n];
            }
        rms = std::sqrt(rms / ref.size());
        for (size_t i = 0; i < ref.size(); ++i) worst = std::fmax(worst, std::fabs(y[i] - ref[i]));
        std::printf("  M=%-2d worst err %.2e of rms output\n", M, worst / rms);
        CHECK(worst / rms < 1e-5);
        if (M == 1) y0.assign(y.begin(), y.begin() + N);
        else CHECK(std::memcmp(y0.data(), y.data(), (size_t)N * 4) == 0);  // token 0's row does not depend on M
    }
    cudaFree(dW);
    cudaFree(dX);
    cudaFree(dY);
    if (failed) {
        std::fprintf(stderr, "ink_s6k_gemv_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("ink_s6k_gemv_test: OK\n");
    return 0;
}
