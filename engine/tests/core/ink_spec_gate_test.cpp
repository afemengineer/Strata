// engine/tests/core/ink_spec_gate_test.cpp - When strata-inkling drafts with DSpark and when it decodes plainly (no GPU)
// MIT License - Copyright (c) 2026 Strata contributors
#include "ink_spec_gate.h"

#include <cstdio>

using namespace ink2;

namespace {
int failed = 0;
#define CHECK(expr)                                                                         \
    do {                                                                                    \
        if (!(expr)) {                                                                      \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr);            \
            ++failed;                                                                       \
        }                                                                                   \
    } while (0)

SpecGate fresh() {
    SpecGate g;
    g.enabled = true;  // ignore STRATA_INK_SPEC_GATE in the test environment
    return g;
}
}  // namespace

int main() {
    // Plain tokens are measured first, then drafting is tried.
    {
        SpecGate g = fresh();
        CHECK(!g.want_spec());
        g.on_plain(0.026);
        CHECK(!g.want_spec());
        g.on_plain(0.026);
        CHECK(g.want_spec());
    }
    // A GPU-resident model: a step costs ~1.2 plain tokens and accepts 4 -> drafting stays on.
    {
        SpecGate g = fresh();
        g.on_plain(0.010); g.on_plain(0.010);
        for (int i = 0; i < 20; ++i) CHECK(!g.on_spec(0.012, 4));
        CHECK(g.want_spec());
    }
    // The 8 GB case that motivated the gate: a step costs ~3.2 plain tokens and accepts ~2.8 -> plain decoding.
    {
        SpecGate g = fresh();
        g.on_plain(0.0267); g.on_plain(0.0267);
        CHECK(g.on_spec(0.086, 3));
        CHECK(!g.want_spec());
        // ... for probe_every plain tokens, then one probe.
        for (int i = 0; i < g.probe_every - 1; ++i) { g.on_plain(0.0267); CHECK(!g.want_spec()); }
        g.on_plain(0.0267);
        CHECK(g.want_spec());
        // When acceptance rises (code, tables, repeated text), drafting comes back and stays.
        for (int i = 0; i < 8; ++i) g.on_spec(0.086, 7);
        CHECK(g.spec_rate() > g.plain_rate());
        CHECK(g.want_spec());
    }
    // Disabled: always draft, as before the gate.
    {
        SpecGate g = fresh();
        g.enabled = false;
        CHECK(g.want_spec());
        g.on_plain(0.01); g.on_plain(0.01);
        CHECK(!g.on_spec(1.0, 1));
        CHECK(g.want_spec());
    }
    if (failed) {
        std::fprintf(stderr, "ink_spec_gate_test: %d check(s) failed\n", failed);
        return 1;
    }
    std::printf("ink_spec_gate_test: OK\n");
    return 0;
}
