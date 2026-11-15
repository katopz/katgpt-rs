# Issue 1222: `katgpt-core::compounding` — the coverage-factor bound substrate POC (Research 674)
**Status:** POC LANDED 2026-11-16 (zed-m3-i1222@m3, this commit) — T1–T6 complete; opt-in under the `estimators` family umbrella; consumer wiring stays the advisory list below, each in its own repo.
**Lease closed:** zed-m3-i1222@m3, 1794767142, was expires 2026-11-17T18:30:00+00:00 — T1–T6 LANDED under it the same session; the claim line folded per the closing convention. (The claim line was first pushed with a wrong epoch, 1794822000, and a >48h expiry — corrected here to the real claim epoch/expiry in the landing commit.)
**Unclaimed** — routable to any katgpt-rs primitive session. Fixture + falsification arm are fully specified below; implementation is ~150 LOC + tests.

Source: arXiv:2609.19465 (Research 674) — Lemma 5.1 / Prop B.2 / §5.2, distilled modelless by the no-GD advocate and coordinator-verified. Every input is a histogram the consumers already persist; zero gradients involved.

## Tasks

- [x] T1 — LANDED: `crates/katgpt-core/src/compounding.rs` + the `estimators`-umbrella exports in `lib.rs`. Gated under the `estimators` PARENT cfg — NO new `[features]` key: the ratchet holds katgpt-core at ceiling 278 + 12 live grants = 290 exactly (zero headroom, `feature_cap_gate.py` re-run green post-landing); the pins file's own "gated under the parent's cfg" landing shape. Naming per spec (`CoverageFactor`; `coverage_ratio` untouched). Signatures exactly as sketched; additions: `FoldScratch`/`coverage_factor_with`/`scratch_capacity_hint` (the zero-alloc steady fold, O(|C|) memory via a two-pointer sorted merge — never a T×|C| materialization), a private `MergeOverflow` mapped to `Undecidable` (u64 count overflow genuinely undecides the ratio), and `CoverageFactor::{factor, is_vacuous}` accessors. Determinism (Issue 940): f64 over u64 counts, key-ascending walks, sequential folds, no HashMap iteration; `direct_bound`'s reduction order is (steps in input order, keys ascending) — pinned by the API.
- [x] T2 — LANDED: G1 fixture asserted to 1e-12 (C_T 3.2 / bound 0.16 / direct 0.15 ≤ 0.16); property test 100 seeds (xorshift, zero new deps); monotonicity arms (ε_atom strictly ↑ over [0.01, 0.05, 0.1, 0.3]; T ↑ via repeated steps: 0.1/0.2/0.3); horizon-floor arm (matched uniform → C_T = 3.0 exactly); clamp arm (0.4·3.2 = 1.28 → 1.0). ⚠ ONE SPEC TIGHTENING, recorded: the issue's "ALWAYS ≤ C_T·ε_atom" holds as a THEOREM only under matched per-step mass — with M = per-step total and Q = Σq, C_T ≥ T·M/Q, so M = Q gives C_T ≥ T and direct ≤ T·ε ≤ C_T·ε. The property test generates matched-mass rollouts only, and the module doc carries the derivation plus the under-sampled (M < Q) regime warning. The fixture numbers are insensitive to the reading; the theorem is not — the code documents the honest one.
- [x] T3 — LANDED: μ2[c4] = 2 with q[c4] = 0 → `Vacuous{[c4]}`; `compound_failure_bound` refuses with the keys; ε_atom = 0 STILL Vacuous (perfect atomic accuracy buys no Level-III guarantee); `support_gap` independently reports the same gap.
- [x] T4 — LANDED, bar met: 10^7 adds (10^4 contexts × 10^3 steps, full coverage per step) folded in **16.32 ms** release single-thread (bar 25 ms). Box state (all latency claims): aarch64 M3, AC power, battery 100% charged, load 2.22/2.43/2.59, system memory 91% free; the `powermode` enum was NOT recorded — this macOS build rejects `pmset powermode`/`-g powermode` (disclosed, not fabricated). Paired interleaved A/B (5×5 alternating, medians): merge-fold 16.32 ms vs naive HashMap-accumulate-then-sort 68.27 ms = **0.24×** — the naive arm is legal under the Issue-940 law (the final sort pins the read order) and the "determinism tax" measured as a determinism DIVIDEND: the sorted merge is 4.2× faster than rehashing. The perf cell is `#[ignore]` + `#[cfg(not(debug_assertions))]` (the `nuisance_fraction` g2 posture) — run with `cargo test -p katgpt-core --features estimators --release --lib -- --ignored g2_perf -- --nocapture`.
- [x] T5 — LANDED: G4 canary `g4_zero_alloc_steady_fold` (the crate's `any(debug_assertions, alloc_tracking)` alloc gate): reset → `coverage_factor_with` at the hinted capacity → reuse the SAME scratch → second fold → assert 0 allocs (passing in the dev posture, debug_assertions ON). The `Vacuous` arm may allocate by contract — naming the gap is the report.
- [x] T6 — LANDED: the estimator-design notes live in the module doc (context binning = modeling choice, contrastive_scope precedent; μ_t survivorship conditioning + the disclose-beside-any-bound rule; the pre-registered TIGHTNESS readout law — bound/measured ratio per class ≫ 1 is a finding, low non-vacuous fraction means the binning is broken, "bound ≥ measured" alone is never an audit; `Finite(0.0)` = no data, never safety).

## Landing record (2026-11-16, zed-m3-i1222@m3)

- Tests: 15 in-module (fixture, falsification ×2 arms, clamp, monotonicity ×2, horizon floor, min-support abstain + touch-only semantics, direct refusals ×3 classes, first-step inverse, error-budget refusal matrix, epsilon validation, support-gap empty, bit-determinism repeat, 100-seed property, G4 canary, G2 ignored) — 15/15 green under `--features estimators`; full umbrella suite 2788 passed / 0 failed / 16 ignored (14.4 s dev, M3).
- Gate coverage: `scripts/test_gate.sh` ROWS gains `katgpt-core:2788:estimators` (floor, downward-firing — member growth never reds). Arch-invariance checked by inspection: no estimators member carries `cfg(target_arch)` (the arch-gated files — tropical, bf16_convert, cumprodsum, mi, newton_schulz, peira, simd_lut_dequant, ane_roofline, diversity/temp — belong to other feature families).
- Clippy `-D warnings` clean at three postures: `--features estimators`, default (module compiled out), `--all-features` (the root re-export collision check). Default lib suite untouched: 2522 passed / 0 failed (the 2060 floor holds — floors fire downward only). `feature_cap_gate.py`: katgpt-core still 290/300, ratchet held.
- Lint lane: mechanical findings went to `cargo refine fix` first (78 spans seen, 1 edit landed); its one miss (the unused `std::cmp::Ordering` import the fixer declined) was removed manually — session-recorded miss, no refine issue filed for a one-line class.

```rust
pub struct ContextKey(u64);                       // blake3-folded context signature
/// COUNT histograms (integers, not probabilities) — min_support is a count floor.
pub struct CtxHist(/* context -> count */);
pub struct StepHist(/* per-step context -> count, one per rollout step */);
/// Per-context error counts: (errors, trials). A rollout context with NO error
/// counter REFUSES (BoundRefusal::Undecidable) — never treated as error-free.
pub struct CtxErr(/* context -> (errors, trials) */);
pub enum CoverageFactor { Finite(f64), Vacuous { uncovered: Vec<ContextKey> }, Undecidable }
pub fn coverage_factor(mu: &[StepHist], q: &CtxHist, min_support: u32) -> CoverageFactor; // Lemma 5.1's C_T
pub enum BoundRefusal { Vacuous(Vec<ContextKey>), Undecidable }
pub fn compound_failure_bound(eps_atom: f64, c: &CoverageFactor) -> Result<f64, BoundRefusal>; // Lemma 5.1; CLAMPED to [0,1] (a loose C_T can push the raw product past 1 — the clamp is disclosed, never silent)
pub fn direct_bound(mu: &[StepHist], err: &CtxErr) -> Result<f64, BoundRefusal>;   // Σ_t Σ_c μ_t(c)·e(c), e(c) = errors/trials per context — needs per-context counters; a context in μ with no counter REFUSES (never treated as 0); ALWAYS ≤ C_T·ε_atom (Σ_t μ_t(c) ≤ C_T·q(c) by the sup's definition); use it when per-context counters exist
pub fn first_step_bound(eps_comp: f64, c1: f64) -> f64;      // §5.2 reverse bound
pub fn horizon_floor(t_steps: usize) -> f64;                 // C_T >= T
pub fn error_budget(success_target: f64, c: &CoverageFactor) -> Result<f64, BoundRefusal>; // eps <= (1−S)/C_T; refuses Vacuous/Undecidable like the bound — no horizon arg (C_T already encodes it)
pub fn support_gap(mu: &[StepHist], q: &CtxHist) -> Vec<ContextKey>; // mu>0 && q==0
```

  - The `Vacuous` variant is the Level-III structural-gap theorem AS A TYPE: a rollout context with q-mass 0 refuses loud naming the keys — never folds into 0.0; `Undecidable` for thin-support cells under `min_support` (decline-first; esc_audit precedent).
  - Stream the fold: accumulate per-context sums across steps (O(|C|) memory), never materialize T×|C|.
- [x] T2 — LANDED (details in the checked T1 row + the landing record below)
- [x] T3 — LANDED (details in the landing record below)
- [x] T4 — LANDED (details in the landing record below)
- [x] T5 — LANDED (details in the landing record below)
- [x] T6 — LANDED (the notes live in the module doc; the law restated in the landing record below)

## Consumer wiring order (advisory, each in its own repo)

1. **riir-refine** `campaign_bound` — report-only readout on score-bench/`--misses` tails from `.refine/trajectories.kv` rows (context signature `{code}:{strategy_id}:{file-shape}`); the T6 tightness readout (bound/measured ratio + non-vacuous fraction) runs against stored campaigns.
2. **riir-rethink** `context_coverage_ratio` — per-bin μ-vs-q monitor over ESC calibration/serving contexts; formalizes the Bench-089 out-of-window rate transfer as an observable (Wilson-floor flags; observer beside the rate guard, never an actuator).
3. **seal-std** `reach_profile` — ready-set width statics over interaction books (max ready width, merge-tuple count) — tells content authors which quests put behavior systems on uncovered multi-invocation contexts.

## Doctrine (free, binds proposal/plan reviews)

1. Any compound-reliability claim carries a horizon-scaled error budget (C_T ≥ T).
2. Any atomic→composed transfer claim carries a coverage analysis or is refused (Prop B.2).
3. Fit/calibrate on composed slices first — the reverse direction transfers down for free (§5.2).
