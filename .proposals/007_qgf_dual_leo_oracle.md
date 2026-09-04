# Proposal 007 — QGF DualLeoOracle (Test-Time LEO+UVFA Q-Gradient Fusion)

Status: **shipped Phase 1-2 (Plan 467); G5 measured FAIL on synthetic data (Bench 553 in riir-ai, 2026-07-18); POST-PLAN-500 re-run (2026-07-18) confirmed the same 0.00%/0.50% numbers are a REAL negative finding (DualLeo α-mix hurts on synthetic), not the vacuous 0/0 implied by the pre-fix frozen-noise LEO behavior. Real-network G5 measurement on civ ([Bench 558](../../riir-ai/.benchmarks/558_civ_qgf_dual_leo_oracle_g5_real_network.md), 2026-07-19): dual beats single by **+2.69%** (35.68% → 36.64% solve rate) but misses the ≥3% gate by 0.31pp — fourth axis exhausted. The QGF gradient-time axis is empirically **less wrong** than the postmax flow-field axis on civ, but not positive enough to clear the gate. The remaining unmeasured cell (paired trained LEO + UVFA on civ real data) is deferred.**
Branch: `develop`
Owner: unassigned
Fusion of: Plan 268 (QGF LeoHeadOracle) × Plan 155 (DualLeoMixer) × Plan 460 (postmax lesson — linear-in-grad mix)
Related:
- [Plan 268](../.plans/268_qgf_test_time_q_guided_flow.md) — QGF substrate, ships `LeoHeadOracle` + `FlowFieldOracle`
- [Plan 460 root-cause](../.benchmarks/460_flow_field_dual_leo_postmax_goat.md) — the pre-vs-post-max nonlinearity lesson this proposal carries forward
- `Plan 459 pre-max dual fusion` — the OG attempt; lesson carrier
- Research 003 §"Cognitive/Reasoning — the New Moat" — basic public / GOAT private

## TL;DR

**Should QGF ship a `DualLeoOracle` — a third `QGradientOracle` impl that fuses a LEO teacher head + a UVFA student head via `DualLeoMixer` at the gradient (not the action-selection) level?** Today QGF ships `LeoHeadOracle` (single LEO head) and `FlowFieldOracle` (FFT flow field). When a UVFA student is available, there is no QGF path that uses both — the dual fusion only exists in the `FlowFieldCache::get_or_compute_dual_postmax` substrate (Plan 460), which is a different pipeline. This proposal closes that gap.

**This is a katgpt-rs primitive** — generic, no game semantics, no chain semantics, lives in `katgpt-core` behind `qgf_oracle + dual_leo` feature gates. The private composition layer (which civ UVFA net, which α per goal category) is a riir-ai concern and explicitly out of scope.

The **Plan 460 lesson** directly informs this proposal: pre-max Q-slice fusion was washed out by `max_a(·)` nonlinearity. QGF gradients are similarly nonlinear in the policy-tilt direction — but the gradient itself is linear in the Q-values (`∇_a Q(s, a)[i] = Q(s, a_i)` per `LeoHeadOracle`'s doc), so a **gradient-level mix avoids the washout** by construction. The proposal carries the Plan 460 root-cause analysis forward as a design constraint.

## The problem this solves

Plan 268's QGF ships two `QGradientOracle` impls in `katgpt-core/src/qgf/oracles.rs`:

| Oracle | Feature | What it wraps |
|---|---|---|
| `LeoHeadOracle<H: LeoHead>` | `leo_all_goals` | A single LEO head; emits the per-action Q-slice for the selected goal as the gradient. |
| `FlowFieldOracle` | `flow_field_nav` | An owned `FlowField`; emits the `(dx, dy)` flow vector at the queried cell. |

The five-tier routing table in `qgf/mod.rs:39-41` shows the gap:

```
| Plasma/Hot | FlowFieldOracle              | flow_field_nav | 1.0 |
| Hot        | LeoHeadOracle                | leo_all_goals  | 1.0 |
| Freeze     | BfnProxyOracle               | (always)       | 0.3 |
```

When both a LEO teacher AND a UVFA student are available, **there is no QGF oracle that uses both.** The consumer must pick one. The dual-fusion machinery exists (`DualLeoMixer` trait, shipped Plan 155) but only `FlowFieldCache::get_or_compute_dual_postmax` consumes it, and that's a different pipeline (potential-field navigation, not Q-gradient guidance).

For QGF consumers that want test-time teacher-student fusion (the paper's headline use case), the gap is real: the paper's `QGF Alg 1` permits ANY Q-gradient oracle, and the dual-LEO setup (LEO teacher + UVFA student, mixed at α) is a natural Q-gradient source that the current QGF implementation cannot express.

## The proposed design

### The `DualLeoOracle` struct

```rust
#[cfg(all(feature = "leo_all_goals", feature = "dual_leo"))]
pub struct DualLeoOracle<H1, H2, M>
where
    H1: LeoHead,
    H2: LeoHead,
    M: DualLeoMixer,
{
    head_leo: H1,
    head_uvfa: H2,
    mixer: M,
    alpha: f32,
    goal_idx: usize,
}
```

### The `QGradientOracle` impl — gradient-level mix (carries Plan 460 lesson)

```rust
impl<H1, H2, M> QGradientOracle for DualLeoOracle<H1, H2, M>
where
    H1: LeoHead,
    H2: LeoHead,
    M: DualLeoMixer,
{
    type State = Vec<f32>;
    type Action = ();

    fn q_gradient_at(&self, state: &Self::State, _action: &Self::Action) -> Vec<f32> {
        // Plan 460 lesson: mix at the gradient level (linear in Q), NOT at the
        // action-selection level (nonlinear in policy). The gradient IS the
        // per-action Q-slice, so a DualLeoMixer::combine_into on the two
        // Q-slices produces a gradient that is a linear combination of the
        // two heads' Q-values. No max-pool washout — the gradient is never
        // max-pooled.
        let q_leo_all = self.head_leo.all_goals_q(state);
        let q_leo = self.head_leo.q_for_goal(&q_leo_all, self.goal_idx);

        let q_uvfa_all = self.head_uvfa.all_goals_q(state);
        let q_uvfa = self.head_uvfa.q_for_goal(&q_uvfa_all, self.goal_idx);

        let mut grad = vec![0.0f32; q_leo.len()];
        self.mixer.combine_into(&mut grad, q_leo, q_uvfa, self.alpha);
        grad
    }

    fn confidence(&self, _state: &Self::State) -> f32 {
        // Both heads are deterministic cached lookups → confidence 1.0.
        // Matches LeoHeadOracle's contract.
        1.0
    }
}
```

### Why this avoids the Plan 460 pre-max failure

Plan 459 (pre-max) failed because the FFT pipeline applies `max_a(·)` to the per-cell Q-slice AFTER the α-mix — and `max_a(α·x + (1-α)·y) ≠ α·max_a(x) + (1-α)·max_a(y)`. The α-weighting was washed out by the max-pool.

QGF's `LeoHeadOracle` has no max-pool. The gradient IS the Q-slice (`∇_a Q(s, a)[i] = Q(s, a_i)`, per the existing doc-comment in `LeoHeadOracle`). So the gradient-level mix is a pure linear combination:

```
grad_mix[i] = α · Q_leo(s, a_i) + (1-α) · Q_uvfa(s, a_i)
            = α · grad_leo[i] + (1-α) · grad_uvfa[i]
```

No nonlinearity between the mix and the consumer. **The Plan 460 lesson is encoded as a design invariant: no operator sits between the mix and the consumer.**

### Routing entry

Add a new row to the QGF tier table:

```
| Hot        | DualLeoOracle                | dual_leo       | 1.0 |
```

Sits alongside `LeoHeadOracle` (single-head); consumer picks based on whether a UVFA student is available. The adaptive guidance weight saturates to ~1.0 because both heads are deterministic cached lookups (same rationale as `LeoHeadOracle`).

## Domain classification (per global AGENTS.md + Research 003)

| State | Domain | Treatment |
|---|---|---|
| `state: Vec<f32>` (input) | **Consumer-supplied** | The oracle does not own it; treats as read-only. Consumer's responsibility to classify (physical / semantic / etc.). |
| `grad: Vec<f32>` (output) | **Semantic / latent** | The gradient is a Q-value vector, not a committed action. It feeds QGF's marginal tilt; it does NOT cross any sync boundary. |
| `α` (mixing coefficient) | **Designer scalar** | Authored constant per use case. |

This oracle does NOT cross the sync boundary. It produces a latent gradient that QGF consumes locally. Bridge to raw (committed action) happens downstream in QGF's drafter, not here.

## Honest caveats — READ BEFORE IMPLEMENTING

1. **G5 MEASURED FAIL on synthetic data (Bench 553, 2026-07-18).** The T7 Go puzzle harness in riir-ai was extended (Issue 553) with a T9 test comparing QGF+DualLeoOracle vs QGF+LeoHeadOracle. Result: dual scored **0.00%** vs single **0.50%** — dual is WORSE. The correctness invariant (QGF+LeoHeadOracle ≡ baseline argmax(Q_LEO)) held bit-identically (diff 0.0000), confirming the mechanism is correct. **POST-ISSUE 554 CORRECTION (2026-07-18):** the original root cause "synthetic data produces near-flat Q-fields" was mis-attributed; the actual root cause is riir-ai Issue 554 — `DualLeoTrainer::apply_leo_last_layer_update` is a `let _ = (grad, lr);` no-op stub, meaning **LEO never updates during training**. The 0.50% baseline is argmax over frozen-noise LEO, not argmax over learned LEO. The T11 diagnostic test (riir-ai, lands with Issue 554) proves this. Re-run T9 after Issue 554 lands — that is the actual unblock, not Issue 552's GPU access (which is necessary but not sufficient). **Original pre-2026-07-18 caveat preserved below for reference; the empirical evidence supersedes it.**

   **G5 REAL-NETWORK CIV MEASUREMENT (Bench 558, 2026-07-19):** with trained CivLeoNet (Plan 505 v7) + untrained CivLeoUvfa + real civ trajectories via `CivTrajectoryEnv` (Plan 501), the T10 test measures dual at **36.64%** vs single **35.68%** action-prediction solve rate — dual beats single by **+2.69%**, missing the ≥3% gate by 0.31pp. The correctness invariant (b ≡ a) held bit-identically (diff 0.0000) on real trained weights. This is the closest the dual-LEO mechanism has come to clearing G5; the QGF gradient-time axis is empirically less wrong than the postmax flow-field axis on civ (+2.69% vs Bench 557's +3.6% stuck-reduction vs a 30% gate). The 0.31pp miss is within the noise floor (n=2500 transitions). The remaining unmeasured cell (paired trained LEO + UVFA on civ real data) would require extending `leo::freeze` to `CivUvfaFrozen` — deferred per [Bench 557](../../riir-ai/.benchmarks/557_plan502_t41_trained_weights_bench550_rerun.md) §"Honest caveat". The fourth-axis stop rule is triggered.

2. *(Original caveat 1)* **No measurement yet that dual Q-gradient beats single-head Q-gradient on a downstream task.** Plan 268 itself deferred downstream task-quality gates (Sudoku/DDTree/Bomber) to riir-ai. This proposal adds a new oracle but does not prove the gain — that's the plan's job (G5 below). [**Update 2026-07-18**: the measurement is now DONE — see caveat 1 above.]
3. *(Original caveat 2)* **The `combine_into` default uses `mix_into` (Lc mode), which is a linear α-blend.** For Max / Min modes, the mixer applies element-wise max/min — but this may interact badly with QGF's `(1/β)·g` tilt if the resulting gradient has discontinuities. Document the Lc mode as the recommended default; mark Max/Min as experimental.
4. *(Original caveat 3)* **`goal_idx` is fixed at construction.** Unlike `LeoHeadOracle::set_goal`, this proposal does NOT ship a `set_goal` method on `DualLeoOracle` initially — the use case is "one goal, dual heads" per construction. Multi-goal switching can be added if a consumer needs it; defer until measured demand.
5. *(Original caveat 4 — now resolved in bench-local form)* **The UVFA head must also be `LeoHead`-shaped** (all-goals Q-tensor). Real UVFA nets are single-goal by definition; the consumer must wrap them. Bench 553 ships `UvfaAsLeoHead` (bench-local, runs UVFA forward per goal slot) as the consumer-side reference impl. A generic version could move to `katgpt-core` if more consumers need it.
6. *(Original caveat 5)* **Confidence always = 1.0 is the same lie `LeoHeadOracle` tells.** A deterministic cached lookup is confidence-1.0 by QGF's contract, but two heads mixed at α=0.5 are not "twice as confident" as one head — the contract conflates determinism with quality. Inherited from Plan 268; not made worse here. Document.

## Fusion lineage

Three existing primitives combine:

1. **Plan 268 `LeoHeadOracle`** (`QGradientOracle` for `LeoHead`) — the substrate. The new oracle mirrors its API and its confidence contract.
2. **Plan 155 `DualLeoMixer`** (LEO+UVFA α-mix trait) — the fusion mechanism. Already shipped; has 3 consumers (QuestLeoScorer, `get_or_compute_dual`, `get_or_compute_dual_postmax`). This proposal adds a 4th.
3. **Plan 460 root-cause lesson** (max-pool washout) — the design constraint. Encoded as the "no operator between mix and consumer" invariant in the doc-comment.

The combination produces: **test-time teacher-student Q-gradient fusion** for QGF consumers — something none of the three alone delivers. Plan 268 alone is single-head; Plan 155 alone has no QGF consumer; Plan 460's lesson alone is just a warning.

## GOAT gate

The new oracle ships behind `qgf_oracle + dual_leo` (both already exist; no new feature flag). Promotion to "documented as the recommended dual path" requires:

- **G1** (correctness): with `α=1.0` (LeoOnly mode), `DualLeoOracle` produces bit-identical gradients to `LeoHeadOracle` with the same head + goal. With `α=0.0` (UvfaOnly mode), bit-identical to a `LeoHeadOracle` wrapping the UVFA head.
- **G2** (perf): one Q-gradient query ≤ 1.5× a single `LeoHeadOracle` query (the cost is 2 forward passes + one `combine_into`, both heads are O(state_dim × hidden × actions)).
- **G3** (no-regression): all existing katgpt-core tests pass. The Plan 268 QGF bench must still pass.
- **G4** (alloc-free hot path): `q_gradient_into` variant that takes a pre-allocated `&mut [f32]` scratch buffer. Steady-state zero allocation.
- **G5** (downstream task gain — **MEASURED FAIL on synthetic data (Bench 553, 2026-07-18); MEASURED FAIL on civ real data (Bench 558, 2026-07-19): +2.69% vs ≥3% gate, the closest the mechanism has come to clearing G5; the QGF gradient-time axis is empirically less wrong than the postmax axis on civ, but not positive enough to clear the gate**): ≥3% first-attempt accuracy gain on Sudoku 9×9 OR ≥5% speculative acceptance rate gain on a dual-LEO consumer, vs single-head `LeoHeadOracle`. Mirrors Plan 268's deferred gate. **Measurement attempted 2026-07-18 on T7 Go puzzle harness (riir-ai Bench 553): dual 0.00% vs single 0.50%, dual WORSE.** Real-network measurement on civ (Bench 558, 2026-07-19): dual 36.64% vs single 35.68%, dual +2.69% but misses the ≥3% gate by 0.31pp. The correctness invariant (b ≡ a) held bit-identically on both synthetic and real weights — the mechanism is verified correct end-to-end on CivLeoNet; the quality gate FAILs on both regimes. The one remaining unmeasured cell (paired trained LEO + UVFA on civ real data) is deferred. Until a positive measurement lands, the oracle ships as opt-in and is documented as "mechanism complete, downstream gain unproven (synthetic measurement negative; civ real-network measurement marginally negative)."

## What ships now (katgpt-rs) vs deferred

### Ships now — katgpt-core
- `DualLeoOracle<H1, H2, M>` struct in `crates/katgpt-core/src/qgf/oracles.rs` (new module `dual_leo_oracle`)
- `QGradientOracle` impl with `q_gradient_at` + `q_gradient_into` + `confidence`
- Routing entry in `qgf/mod.rs` tier table
- Unit tests mirroring `LeoHeadOracle`'s test suite (gradient extraction, confidence=1.0, α=1.0 bit-identity, α=0.0 bit-identity, Max/Min modes documented)
- Doc-comment encoding the Plan 460 "no operator between mix and consumer" invariant

### Deferred — riir-ai
- Consumer-side adapter wrapping real UVFA nets as `LeoHead` (caveat 4) — **DONE bench-local in Bench 553 as `UvfaAsLeoHead`** (synthetic Go) and in Bench 558 as `CivLeoUvfaAsLeoHead` (civ real).
- Downstream task-quality gate G5 (Sudoku / DDTree / Bomber) — **MEASURED FAIL on synthetic Go puzzles (Bench 553); MEASURED FAIL on civ real data (Bench 558, +2.69% vs ≥3% gate — fourth axis exhausted)**.
- Tuning α per consumer.
- Promotion from opt-in to "documented as recommended" — **BLOCKED on positive G5 measurement** (would require paired trained LEO + UVFA weights — deferred per Bench 557 §"Honest caveat").

### Explicitly NOT shipped by this proposal
- **Civ flow-field navigation** — that's Proposal 028 (Option A). This proposal does not touch flow fields or civ.
- **UVFA network architecture / training** — riir-train + riir-games-civ scope.
- **Per-goal switching (`set_goal`)** — defer until consumer demand (caveat 3).

## Phased rollout (sketch — a plan would expand this)

### Phase 1 — Primitive (katgpt-core)
- [ ] T1.1 `DualLeoOracle` struct + `QGradientOracle` impl
- [ ] T1.2 `q_gradient_into` zero-alloc variant
- [ ] T1.3 Routing entry in `qgf/mod.rs`
- [ ] T1.4 Unit tests (G1 bit-identity, G2 perf, G4 alloc-free)

### Phase 2 — GOAT gate
- [ ] T2.1 G1 bit-identity at α=1.0 and α=0.0
- [ ] T2.2 G2 perf ≤ 1.5× single-head (median-of-3 per Plan 460 lesson)
- [ ] T2.3 G3 no-regression (Plan 268 QGF bench still passes)
- [ ] T2.4 GOAT report `.benchmarks/007_qgf_dual_leo_oracle_goat.md`

### Phase 3 — Deferred to riir-ai
- [-] T3.1 Consumer adapter (UVFA-as-LeoHead wrapper)
- [-] T3.2 Downstream task gain G5
- [-] T3.3 Promotion decision

## Risks

1. **Gain may not materialize** (caveat 1). QGF's `(1/β)·g` tilt is small; the difference between single-head and dual-head gradients may be washed out by the tilt magnitude. Mitigation: G5 is deferred not skipped — if the gain doesn't show, the oracle ships as opt-in and the lesson is documented.
2. **Shape-mismatch friction** (caveat 4). Real UVFA nets are single-goal; wrapping as `LeoHead` is non-trivial. Mitigation: ship a `UvfaAsLeoHead` adapter helper in katgpt-core if the pattern is generic enough; otherwise leave to riir-ai.
3. **Max/Min mixer modes may produce discontinuous gradients** (caveat 2). The `combine_into` for Max/Min does element-wise max/min, which has kinks at the swap points. QGF's marginal tilt may not converge cleanly. Mitigation: document Lc as recommended; ship Max/Min behind a `dual_leo_oracle_max_min` sub-feature flag.
4. **API surface bloat**. QGF now has 3 oracles (Leo, FlowField, Dual). Future oracles (trio, etc.) may proliferate. Mitigation: the `QGradientOracle` trait is the right abstraction; the proliferation is at the impl level, not the API level. Acceptable.
5. **Confidence-1.0 lie** (caveat 5). Inherited from Plan 268. Not made worse; documented.

## Out of scope

- **Civ flow-field navigation** (Proposal 028).
- **UVFA network architecture / training** (riir-train / riir-games-civ).
- **Per-goal switching at runtime** (deferred; no consumer needs it today).
- **Replacing `LeoHeadOracle`** — it stays as the single-head path. `DualLeoOracle` is a sibling, not a replacement.
- **Continuous-action QGF** — Plan 268's substrate is discrete-action; this proposal stays discrete.

## References

1. **Q-Guided Flow (QGF)** — Zhou et al., 2026. arXiv:2606.11087. Source paper for Plan 268; the test-time Q-gradient guidance framework this proposal extends.
2. **Q-VGM: Q-Guided Value-Gradient Matching for Flow-Matching VLA** — arXiv:2606.08015. Sibling prior art: test-time Q-selection/Q-guidance with critic. Relevant to the dual-head confidence contract (caveat 5).
3. ** Matthews et al. (2026) — "Goal-Conditioned Agents that Learn Everything All at Once"** (LEO paper, arxiv 2605.23551). Source of `DualLeoMixer`.
4. **Plan 460 root-cause analysis** — the max-pool washout lesson encoded as this proposal's design invariant. `katgpt-rs/.benchmarks/460_flow_field_dual_leo_postmax_goat.md` §"Root cause (confirmed)".
5. **Plan 268** — the QGF substrate this proposal extends. `katgpt-rs/.plans/268_qgf_test_time_q_guided_flow.md`.

## Post-stop-rule investigation (Research 322, 2026-07-19)

After Bench 558's fourth-axis stop rule, [Research 322](../../riir-ai/.research/322_civ_alternative_critic_post_stop_rule_verdict.md) investigated the "single-head LEO + a different critic type" axis that Bench 558 line 141 named as the natural next step. **Verdict: the hint does not survive scrutiny.** The four UQ primitives Bench 558 named (conformal-naive floor, BoM, Sleep-Time Anticipator, Best-Belief Beta Selector) all produce **state forecasts** (1-D scalar or D-channel vector), **not per-action Q-gradients**. The `QGradientOracle` trait requires per-action gradients of length = action-space width; none of the named UQ primitives natively fits that shape. Using them as critics would be category confusion that the G5 quality gate exists to catch (treating a 1-step reward forecast as a Bellman-consistent Q is the single most studied trap in RL).

The only realistic alternative is paired-trained UVFA (the unmeasured cell): ~360 LOC freeze/thaw extension (Path A in Research 322). Tractable but **not-worth-it-now** — the three prior civ axes cluster at 3–4% gain, Bench 553's only paired-trained data point was −100%, and Bench 557's §"Honest caveat" already judged the expected gain as "likely 3–5%, not the 30% the postmax axis was targeting".

**The civ dual-LEO investigation is genuinely closed.** Proposal 007's stance ("ship as opt-in, document G5 as unproven, do NOT promote to recommended until a positive G5 measurement lands") is vindicated a fourth time, now across two measurement regimes AND the post-investigation finding that the named alternative axis is itself a dead end. Re-open only on concrete consumer demand (seal integration reporting a gain, or a new game domain measuring positive G5).

## Fusion-mechanism investigation (Issue 188 / Bench 559, 2026-07-21)

After Research 322 closed the civ investigation, Issue 188 asked the orthogonal question: **is the *linear α-blend* the wrong fusion mechanism?** The hypothesis was that `argmax(α·Q_t + (1-α)·Q_s)` admits "compromise actions" neither head prefers, and a sigmoid confidence gate (`σ(β·Δconf)`) picking the more-confident head's argmax would structurally avoid that failure mode. A new sibling oracle `CommitDualLeoOracle` (katgpt-rs commit `de060d7f`, opt-in `leo_all_goals + dual_leo`) was shipped with a β parameter sweeping from β=0 (linear blend sanity check) to β=∞ (hard commit-then-fuse).

**Measurement verdict ([Bench 559](../../riir-ai/.benchmarks/559_commit_dual_leo_oracle_beta_sweep.md), 2026-07-21): hypothesis REFUTED on civ axis.**

- **T11 synthetic Go:** all β ∈ {0,1,4,16,64} produce 0.00% or 0.50% — flat Q-field, no mechanism helps. Matches Bench 553.
- **T12 civ real-network:** gain **monotonically decreases** as β grows. Hard commit (β=64) is the WORST variant (−0.12%, matches baseline), not the best. This is the *opposite* of Issue 188's prediction.
- **Critical caveat:** β=0 is bit-identical to `DualLeoOracle` at α=0.5 (proven by katgpt-rs unit test). The +3.30% gate clearance at β=0 is therefore the **α=0.5-vs-α=0.3 effect** (T10/Bench 558 used α=0.3 → +2.69%), NOT the commit-then-fuse mechanism. The commit mechanism contributes zero or negative.

**Cross-cutting lesson:** the Plan 459 `ActingMode::Max` washout generalizes. Any non-linear fusion that picks one head's full Q-slice over the other's (max-pool, hard commit, high-β sigmoid gate) loses the gradient signal that α-blending preserves. Linear α-blend is **not** structurally broken — the "compromise action" failure mode is a theoretical concern that the civ real-network evidence does not validate. Bench 550 (postmax untrained), Bench 557 (postmax trained-LEO), and Bench 559 (commit-then-fuse) are three failed attempts to beat linear α-blend on civ. **The α-blend is the right mechanism; only α itself matters.**

**Actionable orthogonal finding:** α=0.5 beats α=0.3 on civ by ~0.6pp. T10/Bench 558 should be re-run at α=0.5 to confirm — this is a one-line tweak to the default α, NOT a new mechanism. Recorded as a follow-up; not credited to Issue 188.

**Ship state:** `CommitDualLeoOracle` stays opt-in (valid alternative API, no measurable gain). `DualLeoOracle` stays canonical (NOT deprecated). The civ dual-LEO mechanism investigation is now closed across **five** measurement axes (Bench 553, 550, 557, 558, 559). Re-open only on concrete consumer demand.

## TL;DR

Ship `DualLeoOracle` as QGF's third oracle — LEO+UVFA Q-gradient fusion at the gradient level, sidestepping the Plan 460 max-pool washout by construction. The primitive is small (~80 LOC + tests), the boundary is clean (katgpt-core, no game semantics), the GOAT gate's G1-G4 are mechanistic and **G5 measured FAIL on synthetic data (Bench 553, 2026-07-18): dual 0.00% vs single 0.50% — mechanism correct (b ≡ a invariant holds bit-identically), quality gate FAILs because synthetic data has no real signal to fuse.** **G5 measured FAIL on civ real data (Bench 558, 2026-07-19): dual 36.64% vs single 35.68% (+2.69%) — misses the ≥3% gate by 0.31pp; closest the mechanism has come to clearing G5; fourth axis exhausted.** **Post-stop-rule investigation (Research 322, 2026-07-19): the named alternative-critic axis is itself category-confused — the four UQ primitives produce state forecasts not per-action Q-gradients; the civ dual-LEO investigation is genuinely closed.** **Fusion-mechanism investigation (Issue 188 / Bench 559, 2026-07-21): the orthogonal 'is linear α-blend the wrong mechanism?' question is REFUTED — `CommitDualLeoOracle`'s sigmoid-gated commit-then-fuse is WORSE than linear α-blend at every β>0 on civ. Linear α-blend is the right mechanism; only α itself matters (α=0.5 beats α=0.3 by ~0.6pp on civ). The civ dual-LEO investigation is now closed across FIVE measurement axes.** The oracle ships as opt-in with documented unproven G5.
