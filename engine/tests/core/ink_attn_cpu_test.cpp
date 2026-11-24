// CPU attention over host-RAM FP16 KV (host_avx2::attn_f16_partial) against an FP64 reference: the partial softmax
// of each of the 24 query heads, normalised, for runs of keys near the query (relative bias by distance) and far
// from it (the 1024 bucket), and two partials merged as k_ink_flash_gqa_merge does.
#include "ink_host_avx2.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int e = (int)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (e <= 0) return (uint16_t)sign;  // tests use |f| >= 2^-10
    return (uint16_t)(sign | ((uint32_t)e << 10) | (mant >> 13));
}

double from_half(uint16_t h) {
    const int e = (h >> 10) & 0x1f;
    const int mant = h & 0x3ff;
    const double v = (e == 0) ? std::ldexp((double)mant, -24) : std::ldexp(1.0 + mant / 1024.0, e - 15);
    return (h & 0x8000) ? -v : v;
}

constexpr int H = 24, D = 128, KVH = 8, R = 1025;

// Normalised output of head h over keys [0, n) whose first key is dist0 before the query.
void reference(const std::vector<float>& q, const std::vector<float>& rel, const std::vector<uint16_t>& k,
               const std::vector<uint16_t>& v, int n, int dist0, std::vector<double>& out) {
    out.assign((size_t)H * D, 0.0);
    for (int h = 0; h < H; ++h) {
        const int kh = h / 3;
        std::vector<double> s(n);
        double mx = -1e300;
        for (int j = 0; j < n; ++j) {
            double dot = 0;
            for (int d = 0; d < D; ++d) dot += (double)q[(size_t)h * D + d] * from_half(k[((size_t)j * KVH + kh) * D + d]);
            int dist = dist0 - j;
            dist = dist < 0 ? 0 : (dist > 1024 ? 1024 : dist);
            s[j] = dot * 0.08838834764831845 + rel[(size_t)h * R + dist];
            mx = std::max(mx, s[j]);
        }
        double l = 0;
        for (int j = 0; j < n; ++j) {
            const double p = std::exp(s[j] - mx);
            l += p;
            for (int d = 0; d < D; ++d) out[(size_t)h * D + d] += p * from_half(v[((size_t)j * KVH + kh) * D + d]);
        }
        for (int d = 0; d < D; ++d) out[(size_t)h * D + d] /= l;
    }
}

}  // namespace

int main() {
    using namespace strata::ink;
    if (!host_avx2::available()) {
        std::printf("SKIP: no AVX2 host kernels\n");
        return 77;
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto nz = [&](float scale) { float f = nd(rng) * scale; return std::fabs(f) < 1e-3f ? 1e-3f : f; };
    int fails = 0;
    const int cases[][2] = {{1, 0}, {37, 36}, {1000, 999}, {3000, 2999}, {3000, 9000}, {16, 1500}};
    for (const auto& c : cases) {
        const int n = c[0], dist0 = c[1];
        std::vector<float> q((size_t)H * D), rel((size_t)H * R);
        std::vector<uint16_t> k((size_t)n * KVH * D), v((size_t)n * KVH * D);
        for (auto& x : q) x = nz(3.0f);  // sharp attention: scores spread over ~10 units
        for (auto& x : rel) x = nz(1.0f);
        for (auto& x : k) x = to_half(nz(1.0f));
        for (auto& x : v) x = to_half(nz(1.0f));
        std::vector<double> ref;
        reference(q, rel, k, v, n, dist0, ref);

        // Whole run, then the same run as two partials merged.
        std::vector<float> m(H), l(H), o((size_t)H * D);
        host_avx2::attn_f16_partial(q.data(), rel.data(), k.data(), v.data(), n, dist0, m.data(), l.data(), o.data());
        const int a = n / 3;
        std::vector<float> m1(H), l1(H), o1((size_t)H * D), m2(H), l2(H), o2((size_t)H * D);
        host_avx2::attn_f16_partial(q.data(), rel.data(), k.data(), v.data(), a, dist0, m1.data(), l1.data(), o1.data());
        host_avx2::attn_f16_partial(q.data(), rel.data(), k.data() + (size_t)a * KVH * D, v.data() + (size_t)a * KVH * D,
                                    n - a, dist0 - a, m2.data(), l2.data(), o2.data());
        double err = 0, err2 = 0, scale = 0;
        for (int h = 0; h < H; ++h) {
            const double mg = std::max(l1[h] > 0 ? (double)m1[h] : -1e300, l2[h] > 0 ? (double)m2[h] : -1e300);
            const double w1 = l1[h] > 0 ? std::exp(m1[h] - mg) : 0, w2 = l2[h] > 0 ? std::exp(m2[h] - mg) : 0;
            const double lt = l1[h] * w1 + l2[h] * w2;
            for (int d = 0; d < D; ++d) {
                const double r = ref[(size_t)h * D + d];
                scale = std::max(scale, std::fabs(r));
                err = std::max(err, std::fabs(o[(size_t)h * D + d] / l[h] - r));
                err2 = std::max(err2, std::fabs((o1[(size_t)h * D + d] * w1 + o2[(size_t)h * D + d] * w2) / lt - r));
            }
        }
        const bool ok = err <= 2e-5 * scale && err2 <= 2e-5 * scale;
        std::printf("%s n=%d dist0=%d max|out|=%.3f err=%.2e merged err=%.2e\n", ok ? "ok  " : "FAIL", n, dist0, scale, err, err2);
        fails += ok ? 0 : 1;
    }
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
