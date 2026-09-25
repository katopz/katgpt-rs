# Issue 887: `ladder_gate` — streak-gated difficulty-ladder advancement + corrective backtracking FSM (katgpt-core)

**Status:** LANDED 2026-09-25 — T1–T7 complete ([Bench 891](../.benchmarks/891_ladder_gate_goat.md): G1 17/17 + G2 0.8–1.2 ns/eval, k64/k8 paired 0.977 linear + G3 flag-off clean (default suite 2063 unchanged) + G4 0 allocs). **GOAT PASS, stays OPT-IN** (`ladder_gate = []`) — no production consumer yet; promotion waits on the first consumer's GOAT (the 873/874/875 precedent). Consumers wire later (out of scope here; see the Consumers section). Filed from Research 589 / arXiv:2609.19717.

## Context

ATC (arXiv:2609.19717, Research 589) ships a stage controller our stack does not have anywhere: **advance a difficulty stage iff held-out accuracy ≥ τ for m consecutive evaluations (any sub-τ eval resets the streak), and on advancement re-probe every passed stage at its own budget — if any regressed, retreat to the *shallowest failing* stage and retrain there** (corrective), beside a preventive λ=0.1 rehearsal mix.

Measured value (paper's own ablations, our priors — re-measure per surface):
- Dwell-dominance: (τ=0.9, m=5) → 99.9% beats (τ=0.98, m=1) → 98.5% beats (τ=0.9, m=1) → 91.8%. **Raising m beats raising τ.** A threshold below the ceiling admits a half-learned stage and the run stops there.
- Backtracking under a bounded window: 51.0% → 97.4% (−46pp is the largest single ablation in the paper).
- λ=0 (no rehearsal): the ladder never passes stage 2 (8/8 arms).
- The collapse warning: windowing/eviction without retention re-verification is unsafe.

Workspace grep (Research 589 §3): every shipped streak gate runs the inverse direction (halt/suppress/demote on consecutive failures — `gain_cost_halt::inversion_streak`, riir-clippy `pair_fail_streak`, `EvidenceTier::Withdrawn`); every fade mechanism is detect-only or trigger-only (riir-clippy FADED/STALLED report-only, `auto_readmit` re-admits lints, doesn't retreat a pointer). No stage pointer, no advance-on-streak, no argmin-retreat ships.

## Proposal

New opt-in module `crates/katgpt-core/src/ladder_gate.rs`, feature `ladder_gate` (default-off; promote only on GOAT):

```rust
pub struct LadderGateConfig {
    pub tau: f32,            // paper prior 0.9 (arithmetic 0.98)
    pub streak_needed: u32,  // paper prior 5
    pub rehearsal_frac: f32, // λ, paper prior 0.1
}
#[repr(u8)]
pub enum LadderAction { Hold, Advance, Retreat { to_stage: u32 } }

pub struct LadderGate { stage: u32, streak: u32, cfg: LadderGateConfig }
impl LadderGate {
    /// Probe-free path: O(1). A single accuracy < tau resets the streak.
    pub fn on_eval(&mut self, accuracy: f32) -> LadderAction;
    /// Retention path: O(k) over earlier-stage accuracies (re-probed at own budget).
    /// Retreat selects the SHALLOWEST failing stage (argmin j), not the most recent.
    pub fn on_eval_with_retention(&mut self, accuracy_now: f32, earlier: &[f32]) -> LadderAction;
}
```

Zero-alloc (no heap in either path; `earlier` is a caller-owned slice), `f32`-free where exactness matters (streak counters are u32). Keep the module under ~200 lines — it is a state machine, not a framework.

## Tasks

- [x] T1 Implement `ladder_gate.rs` behind feature `ladder_gate` (config, both paths, `#[repr(u8)]` actions) — **DONE 2026-09-25**: `crates/katgpt-core/src/ladder_gate.rs` (~380 lines incl. tests); paper-prior defaults (0.9/5/0.1); retention gates the ADVANCE (reading B, paper's "on advancement re-probe" — pinned in code); malformed probe slices fail closed to stage 1 (the collapse warning mechanised); non-finite/out-of-range evals are not confirmations; `rehearsal_frac` consumer-facing (the gate detects decay, the trainer prevents it); no auto-demotion anywhere.
- [x] T2 G1 property tests: advance iff m-consecutive ≥τ; single sub-τ resets (pin: this is where 91.8→99.9 lives); retreat = argmin shallowest-failing; Retreat at stage 1 is Hold — **DONE** (17 tests incl. the reset pin, three failing-pattern retreat arms, structural stage-1 Hold, malformed-slice fail-closed, discriminant stability)
- [x] T3 G1 negative controls (the paper ships them free): (a) a synthetic 2-stage ladder with λ=0 MUST fail stage-2 retention; (b) a synthetic windowed ladder without the retention path MUST show the collapse class (the 51% arm) — a gate that cannot red is worthless — **DONE** (deterministic toy learner, no RNG: (a) Retreat{1} emitted + never passes stage 2; (b) probe-free gate climbs 0.95→<0.5 stage-1 skill with no possible red)
- [x] T4 Dwell-dominance pin: on the synthetic ladder, (0.9, 5) outcome ≥ (0.98, 1) outcome ≥ (0.9, 1) — the ordering inequality as a test — **DONE with a recorded toy-ceiling caveat**: the decisive inequality (0.9,5) > (0.9,1) is pinned (final-foundation metric); the (0.98,1) middle arm is EXCLUDED because the toy's 0.95 training cap makes τ=0.98 unmeetable (that arm would measure the toy's ceiling, not the gate — the paper's own 0.98 setting trains to a ceiling above its τ; recorded in Bench 891 + in-test comment)
- [x] T5 G2 µs-scale bench (`bench_ladder_gate`), O(1)/O(k) asserted; G4 alloc-free canary — **DONE**: `crates/katgpt-core/tests/bench_887_ladder_gate_goat.rs` (harness=false, the 886 pattern: shared `ab_median_ratio` paired interleave + `best_of_us` black_box'd) — 0.8/1.2 ns/eval, k64/k8 0.977 (linear), 0 allocs; `[[test]]` required-features row in the same commit
- [x] T6 G3 flag-off byte-identical pin; doc block cites Research 589 + the paper's ablation table — **DONE**: feature-off `cargo check`/`clippy` clean at 0 warnings (module cfg'd to nothing); default-features lib suite 2063 passed / 0 failed, floor unchanged; module doc carries the full paper-ablation table + the substrate-first verdict
- [x] T7 GOAT verdict → promote to default or record the refutation (per Feature Flag Discipline) — **RECORDED 2026-09-25** ([Bench 891](../.benchmarks/891_ladder_gate_goat.md)): G1–G4 ALL PASS, **stays opt-in** — the modelless-gain rule needs a first consumer measuring the gain on a real ladder before any default promotion (the mechanism gain is the paper's ablation table until then)

## Consumers (wire later — NOT this issue's scope)

1. **riir-clippy corpus staging** — doctrine-gated: the Issue-102 ≥3-generation fade-predictive clock must elapse first ("nothing demotes automatically" stands; when the clock fires, the corrective layer should be shallowest-failing retreat + λ-mixed rehearsal, not demotion).
2. **riir-reflex calibration cadence** — dwell-dominance applied to `cal_min_obs`: prefer raising the consecutive-confirmation count over tightening the gate threshold (config + one test, no new code needed).
3. **riir-ai CGSP zone-difficulty unlock** — advance zone tier iff competence probe ≥ τ for m ticks (the `cgsp_runtime` difficulty ladder).
4. **riir-train ATC rig** — if the Plan 352 ATC row ever fires, its τ/m/λ controller is this FSM server-side (one implementation, not a training-loop re-derivation).

## Non-goals

- **KV eviction from attention-mass ordering** (last-thought sufficiency): separate, blocked on a trained sequential-chain latent artifact that does not exist (every trained-latent attempt is a recorded null) + FlashLoop (arXiv:2609.29812) already occupies the looped-model family. Recorded in Research 589 §6.4.
- Theorem 1's 99% concentration is NOT a shippable guarantee (correlation-one assumption).
- No auto-demotion anywhere — retreat re-trains; it never retires.

## References

- Research 589 (`.research/589_Abstract_Token_Curriculum_ATC.md`) — Path-0 table, prior art, signal-diffs
- arXiv:2609.19717 §2/§6.3/App H.2 (gate sensitivity), Table 2 (backtracking ablation)
- Cousins: `hint_regret` (the regulator without a generator — fusion candidate Research 589 §6.1), riir-clippy `frontier_report`/`gate_calibration`/`active_set` (detect-only halves)
