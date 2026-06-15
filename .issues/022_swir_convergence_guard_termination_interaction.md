# Issue 022 — SwiR Convergence Guard Blocks Termination on Synthetic Schedules

**Date:** 2026-06-15
**Status:** ✅ RESOLVED (fixed during Plan 275 Phase 3)
**Plan:** [275 SwiR Switch-Thinking](../.plans/275_swir_switch_thinking.md) Phase 3
**Benchmark:** [.benchmarks/275_swir_switch_thinking_goat.md](../.benchmarks/275_swir_switch_thinking_goat.md)

## Summary

When `switch_count` entered the convergence window `[ceil(½·c_max), c_max]`, the
controller enqueued `CloseThink` on **every** Explicit step. The inject-queue
drain (step 1 of `step()`) preempted the mode-switch logic (step 3) by returning
early, so the controller could never switch back to Latent → never did another
Latent→Explicit switch → `switch_count` froze → termination
(`ForceAnswerPrefix` at `switch_count > c_max`) never fired.

## Resolution

**Fixed in `src/swir/controller.rs` step (4)** by changing the convergence /
termination guard condition from:

```rust
// BEFORE (buggy): fired on EVERY Explicit step in the convergence window
if self.mode == ThinkMode::Explicit {
    if self.switch_count >= conv_at && self.switch_count <= self.config.c_max {
        self.try_enqueue(ControlToken::CloseThink);
    }
    ...
}
```

to:

```rust
// AFTER (fixed): fires only on the step where the Latent→Explicit switch
// JUST happened (one-shot trigger per switch event)
if switched_to == Some(ThinkMode::Explicit) {
    if self.switch_count >= conv_at && self.switch_count <= self.config.c_max {
        self.try_enqueue(ControlToken::CloseThink);
    }
    ...
}
```

This matches the paper's intent (§3.4 describes switch-count thresholds, not
continuous conditions) and fires each guard exactly once per switch event.

## Verification

- **Before fix:** G2p ran 1024 steps without terminating (0% reduction).
- **After fix:** G2p with `c_convergence_fraction = 0.5` (paper default) terminates at step 33 — **31× fewer steps** (97% reduction).

The G2p test in `tests/bench_275_swir_goat.rs` now uses the realistic
`c_convergence_fraction = 0.5` (no workaround needed) and documents the fix.

## Original Reproduction (for posterity)

```rust
use katgpt_rs::swir::{SwiRConfig, SwiRController, ThinkMode};

let cfg = SwiRConfig {
    w_e_to_l: 1, w_l_to_e: 0, c_max: 4, c_convergence_fraction: 0.5,
    answer_budget_b: 16, alpha_0: 0.6, beta_0: 0.7, max_steps: 1024,
    kurtosis_escape_threshold: f32::INFINITY,
};
let mut c = SwiRController::new(cfg);
// Alternating HIGH/LOW entropy every step.
for i in 0..1024 {
    let entropy = if i == 0 { 5.0 } else if i % 2 == 1 { 1.0 } else { 5.0 };
    c.step(entropy, i);
}
// BEFORE fix: switch_count froze at 2 (the convergence threshold).
// AFTER fix: switch_count climbs past c_max=4 → ForceAnswerPrefix → Terminate.
```

## References

- Fix: `src/swir/controller.rs` step (4), the `if switched_to == Some(ThinkMode::Explicit)` guard
- G2p test: `tests/bench_275_swir_goat.rs` `g2p_efficiency_proxy_swir_terminates_earlier_than_fixed_budget`
- Benchmark report: `.benchmarks/275_swir_switch_thinking_goat.md`
- `src/swir/BENCHMARKS.md` (T3.10 deliverable)
