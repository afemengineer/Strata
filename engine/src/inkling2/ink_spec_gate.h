// engine/src/inkling2/ink_spec_gate.h - Whether a DSpark speculative step pays off on this machine (no GPU)
// MIT License - Copyright (c) 2026 Strata contributors
#pragma once

#include <cstdlib>

namespace ink2 {

// A speculative step drafts tokens and verifies them in one batch. On a GPU that holds every expert, verifying
// 8 tokens costs about one decode step, so drafting always wins. When experts live in RAM (8 GB cards), the verify
// batch touches several times more distinct experts than one token does: measured on an RTX 4060 with SQ-10, a step
// cost ~3.2 plain tokens while a short synthetic prompt accepted ~2.8 tokens per step, so drafting ran at 32.5 tok/s
// against 37.4 tok/s without it. The gate keeps running averages of both rates from the steps the server actually
// runs and drafts only while drafting is faster, probing again every `probe_every` plain tokens because acceptance
// changes with the text.
struct SpecGate {
    double alpha = 0.25;         // weight of the newest sample in each running average
    double margin = 1.03;        // drafting must beat plain decoding by this factor to be chosen
    int probe_every = 32;        // plain tokens between two drafting probes once drafting lost
    int warmup_plain = 2;        // plain tokens measured before the first comparison
    bool enabled = true;         // STRATA_INK_SPEC_GATE=0: always draft, as before

    double plain_s = 0.0;        // seconds per plain decode token
    double spec_s = 0.0;         // seconds per speculative step
    double spec_tokens = 0.0;    // tokens emitted per speculative step
    int n_plain = 0, n_spec = 0;
    int plain_left = 0;          // plain tokens still to run before the next probe

    SpecGate() {
        if (const char* e = std::getenv("STRATA_INK_SPEC_GATE")) enabled = (e[0] != '0');
    }

    // Tokens per second of each choice (0 when not measured yet).
    double plain_rate() const { return plain_s > 0.0 ? 1.0 / plain_s : 0.0; }
    double spec_rate() const { return spec_s > 0.0 ? spec_tokens / spec_s : 0.0; }

    // True when the next decode step should be speculative.
    bool want_spec() const {
        if (!enabled) return true;
        if (n_plain < warmup_plain) return false;  // a plain rate to compare against first
        return plain_left <= 0;
    }

    void on_plain(double seconds) {
        plain_s = n_plain == 0 ? seconds : plain_s + alpha * (seconds - plain_s);
        ++n_plain;
        if (plain_left > 0) --plain_left;
    }

    // Returns true when this step switched drafting off (for the log).
    bool on_spec(double seconds, int tokens) {
        const bool first = (n_spec == 0);
        spec_s = first ? seconds : spec_s + alpha * (seconds - spec_s);
        spec_tokens = first ? tokens : spec_tokens + alpha * ((double)tokens - spec_tokens);
        ++n_spec;
        if (enabled && n_plain >= warmup_plain && spec_rate() * 1.0 < plain_rate() * margin) {
            plain_left = probe_every;
            return true;
        }
        return false;
    }
};

}  // namespace ink2
