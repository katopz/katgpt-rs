# Issue 1222: `katgpt-core::compounding` — the coverage-factor bound substrate POC (Research 674)
**Status:** Open — POC tier. Self-contained; no consumer is blocked on it. Escalates to a GOAT plan the day a consumer wires it (refine campaign bound is the first candidate).
**Unclaimed** — routable to any katgpt-rs primitive session. Fixture + falsification arm are fully specified below; implementation is ~150 LOC + tests.

Source: arXiv:2609.19465 (Research 674) — Lemma 5.1 / Prop B.2 / §5.2, distilled modelless by the no-GD advocate and coordinator-verified. Every input is a histogram the consumers already persist; zero gradients involved.

## Tasks

- [ ] T1 — `crates/katgpt-core/src/compounding.rs`, feature-gated: land under an EXISTING feature-count family umbrella (the `estimators` or `diagnostics` grant rows — check which the gate accepts) or add a reasoned grant row to `scripts/feature_count_pins.txt` in the SAME commit — the ratchet is live (`ceiling katgpt-core 278`, count 290/300, `feature_cap_gate.py` reds a new top-level key). Deterministic per the Issue-940 law: f64, sorted-key sequential folds, HashMap iteration order banned. Naming: **`coverage_factor`/`CoverageFactor`** (the paper's term) — `coverage_ratio` is ALREADY TAKEN by `coverage_manifest.rs:409` (covered/census fraction, a different signal); one crate must not carry two meanings.

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
- [ ] T2 — G1 hand fixture (COUNT-scaled; assert to 1e-12): q = {c1: 5, c2: 3, c3: 2} (Σ=10); μ1 = {c1: 10}, μ2 = {c1: 4, c2: 6}, μ3 = {c1: 2, c2: 3, c3: 5} (Σ=10 each). Per-key sums 16 / 9 / 5 → ratios 3.2 / 3.0 / 2.5 → **C_T = 3.2**; ε_atom = 0.05 → **bound = 0.16**. **direct_bound arm**: uniform e(c) = 0.05 → Σ_t Σ_c μ_t(c)·e(c) = 0.15, assert `direct ≤ C_T·ε` (0.15 ≤ 0.16) and assert the inequality holds on randomized histograms (property test, 100 seeds). Plus: monotonicity arms (bound ↑ in ε_atom, ↑ in T); horizon-floor arm (all-supported uniform case → C_T ≥ 3); clamp arm (ε_atom = 0.4 with C_T = 3.2 → raw 1.28, bound returns 1.0).
- [ ] T3 — falsification arm (the paper's theorem pinned in code): inject μ2[c4] = 2 with q[c4] = 0 → `coverage_factor` returns `Vacuous{[c4]}` and the bound REFUSES; second arm with ε_atom = 0 must STILL be Vacuous (perfect atomic accuracy gives no Level-III guarantee).
- [ ] T4 — G2 perf bar: 10^4 contexts × 10^3 steps (10^7 adds) fold ≤ 25 ms single-thread release; paired interleaved A/B vs a naive fold documents the sorted-key determinism tax.
- [ ] T5 — G4: zero-alloc steady fold (preallocated scratch), counting-allocator canary armed.
- [ ] T6 — estimator-design notes in the module doc (before any consumer wiring): context binning is a modeling choice (contrastive_scope disclosure precedent); μ_t is survivorship-conditioned (mistake-free prefixes) — record context-at-step on failing rollouts too and disclose the conditioning. **The consumer-side audit is a pre-registered TIGHTNESS readout, not a dominance check**: report (a) bound/measured-failure ratio per campaign class (a ratio ≫ 1 means the bound is loose — a finding, never a pass) and (b) the non-vacuous fraction of campaigns (a low fraction means binning is broken — the C_T-instrument is unusable on that surface, said loud). "Bound ≥ measured failure" alone is NOT an audit — any loose bound passes it.

## Consumer wiring order (advisory, each in its own repo)

1. **riir-refine** `campaign_bound` — report-only readout on score-bench/`--misses` tails from `.refine/trajectories.kv` rows (context signature `{code}:{strategy_id}:{file-shape}`); the T6 tightness readout (bound/measured ratio + non-vacuous fraction) runs against stored campaigns.
2. **riir-rethink** `context_coverage_ratio` — per-bin μ-vs-q monitor over ESC calibration/serving contexts; formalizes the Bench-089 out-of-window rate transfer as an observable (Wilson-floor flags; observer beside the rate guard, never an actuator).
3. **seal-std** `reach_profile` — ready-set width statics over interaction books (max ready width, merge-tuple count) — tells content authors which quests put behavior systems on uncovered multi-invocation contexts.

## Doctrine (free, binds proposal/plan reviews)

1. Any compound-reliability claim carries a horizon-scaled error budget (C_T ≥ T).
2. Any atomic→composed transfer claim carries a coverage analysis or is refused (Prop B.2).
3. Fit/calibrate on composed slices first — the reverse direction transfers down for free (§5.2).
