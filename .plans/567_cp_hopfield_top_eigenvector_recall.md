# Plan 567: CP^(d-1) Symmetric-Space Hopfield — Top-Eigenvector Recall Primitive

**Date:** 2026-08-04
**Research:** [katgpt-rs/.research/466](../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md)
**Private guide:** [riir-neuron-db/.research/304](../../riir-neuron-db/.research/304_Symmetric_Space_Hopfield_Super_GOAT_Guide.md)
**Source paper:** Victor Galitski — "High-Capacity Generalized Hopfield Networks" — [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks) (2026-07-31)
**Target:** `katgpt-rs/crates/katgpt-core/src/cp_hopfield/` (new module) + Cargo feature `cp_hopfield`
**Status:** Phases 1–5 + 6 + 7 COMPLETE. **Promotion decision: `cp_hopfield` STAYS OPT-IN** — G1–G4 + G7 PASS, G5 passes only in the narrow sense (see Phase 5), G6 FAIL (CP² recall worse than cosine ANN on KG capacity, see Issue 033). Benchmark: [.benchmarks/567](../.benchmarks/567_cp_hopfield_goat.md).

---

## Goal

Ship the **open primitive** for CP^(d-1) symmetric-space Hopfield associative memory recall — the modelless, BBP-protected top-eigenvector recall operator distilled from Galitski (2026). The primitive is generic Lie-algebraic + Rayleigh-quotient math with no game/chain/shard IP. It unblocks Plan 276's documented "attractor needs training" blocker (Fusion A — the load-bearing G5 gate) and force-multiplies `ItemEmbedIndex` + vibe KG retrieval (Fusion B — G6 gate).

**Feature flag:** `cp_hopfield` (opt-in). Promotion to default-on requires G1–G7 all PASS, with G5 (Plan 276 unblock) and G7 (BBP gap at finite N) as the load-bearing gates.

**Honest framing:** this is a Super-GOAT *candidate*. The Super-GOAT verdict is contingent on G5 passing. If G5 fails, the primitive still ships as a GOAT (capacity gains via Fusion B/G6) but the headline "modelless belief unblock" selling point is gone.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [x] **T1.1** Create module `crates/katgpt-core/src/cp_hopfield/` with `mod.rs` declaring the public API surface. Add `cp_hopfield` feature to `katgpt-core/Cargo.toml` default-off.
- [x] **T1.2** Implement `CpHopfieldRecaller<D>` struct (research note §2.1). Generic over `const D: usize` (complex dimension = `d` in CP^(d-1)). Fields: `memories: Vec<[Complex<f32>; D]>`, `structure_constants: &'static [[[f32; D2]; D2]; D2]` where `D2 = D*D - 1`.
- [x] **T1.3** Implement SU(d) structure constants for d=2 (Pauli, `f_{abc} = ε_{abc}`), d=3 (Gell-Mann, Eq. 43 of paper), d=4, d=8. Hardcoded lookup tables — O(1) init.
- [x] **T1.4** Implement `mattis_overlap_excluding(neuron_idx, mu) -> f32` — the `O_μ^(i)` computation. O(N) per call.
- [x] **T1.5** Implement `build_memory_kernel(neuron_idx) -> [[Complex<f32>; D]; D]` — the `K_i = Σ_μ O_μ^(i) |ξ^μ_i⟩⟨ξ^μ_i|` construction. O(P·D²) per call.
- [x] **T1.6** Implement `hermitian_top_eigenvector(k: &[[Complex<f32>; D]; D]) -> [Complex<f32>; D]` — power iteration (5–10 iters suffice for d ≤ 8). For d=2 use closed-form (Pauli matrix analytic roots); for d=3 use closed-form (cubic characteristic polynomial). O(D³) per call.
- [x] **T1.7** Implement `bloch_projection(state: &[Complex<f32>; D]) -> [f32; D2]` — convert qudit to generalized Bloch vector via `s_a = ⟨ξ|λ_a|ξ⟩`. O(D·D2) per call.
- [x] **T1.8** Implement `recall_step(neuron_idx, current_bloch: &[f32; D2]) -> [f32; D2]` — the full top-eigenvector recall step (build K_i → top evec → Bloch projection).
- [x] **T1.9** Add G1 unit test: store 1 Haar-random memory on CP² (d=3), corrupt it 40%, recall → assert `m̄ ≥ 0.9` after 1 sweep.
- [x] **T1.10** Add G1 unit test: store 10 Haar-random memories on CP² at α=0.1 (< α_c=0.62), corrupt memory 0, recall → assert `m̄_0 ≥ 0.9` after 1 sweep.
- [x] **T1.11** Commit Phase 1 skeleton. Tag for the G5 PoC (Phase 5) to consume.

---

## Phase 2 — Manifold Constraint Enforcement

### Tasks

- [x] **T2.1** Implement `project_to_manifold(bloch: &mut [f32; D2])` — enforce the non-linear CP^(d-1) constraint `d_{abc} s_b s_c = (2/3) s_a` via projected gradient (alternate normalization + constraint projection until convergence). O(D²) per call.
- [x] **T2.2** Implement the symmetric `d_{abc}` tensor for d=3 (paper §VIII.C gives the explicit non-zero components). For d=2 all `d_{abc}=0` (no constraint beyond norm). For d=4, d=8 — derive from generalized Gell-Mann anticommutators.
- [x] **T2.3** Add G4 unit test: `project_to_manifold` converges in ≤ 5 iterations for d=3; produces Bloch vector satisfying the constraint to `|d_{abc} s_b s_c − (2/3) s_a| < 1e-5` for all a.
- [x] **T2.4** Add G4 unit test: `project_to_manifold` is sub-μs for d=3 (D=8) at criterion bench.

---

## Phase 3 — Generalized LLG Flow (Physical Recall)

### Tasks

- [x] **T3.1** Implement `lie_bracket(s: &[f32; D2], b: &[f32; D2], f: &StructureConstants) -> [f32; D2]` — the `[s ×_f B]_c = f_{cab} s_a B_b` computation. O(D2²) per call.
- [x] **T3.2** Implement `mean_field(neuron_idx, states: &[[f32; D2]; N]) -> [f32; D2]` — the `B_i = Σ_{j≠i} J_{ij} s_j = Σ_μ ξ^μ_i O_μ^(i)` computation. O(N·D2) per call.
- [x] **T3.3** Implement `llg_flow_step(s: &mut [f32; D2], b: &[f32; D2], damping: f32, dt: f32)` — the generalized Landau-Lifshitz-Gilbert step: `ṡ = s ×_f B − λ [s ×_f [s ×_f B]]`. O(D2²) per call. Calls `project_to_manifold` after the step.
- [x] **T3.4** Implement `llg_recall(recaller: &CpHopfieldRecaller, initial: &mut [f32; D2], damping: f32, dt: f32, max_steps: usize) -> RecallResult` — runs the LLG flow to fixed point, returns final state + energy trajectory + convergence step count.
- [x] **T3.5** Add G1 unit test: LLG recall on CP² with 1 corrupted memory converges to `m̄ ≥ 0.99` within 10 damping times (paper Fig 9 shows ~3 damping times at λ=1).
- [x] **T3.6** Add G1 unit test: LLG recall energy trajectory is monotonically non-increasing (`Ė = −λ Σ |s_i ×_f B_i|² ≤ 0`).
- [x] **T3.7** Add G4 unit test: one LLG step is sub-μs for d=3 (D2=8) at criterion bench.

---

## Phase 4 — Capacity Measurement (G2)

### Tasks

- [x] **T4.1** Implement `measure_capacity(d: usize, n: usize, alpha_range: &[f32], realizations: usize) -> CapacityCurve` — for each α in `alpha_range`, generate P=α·N Haar-random memories, corrupt a random target, recall, measure `m̄_0`. Average over `realizations`. Return α_c (where `m̄_0` crosses threshold 0.5).
- [x] **T4.2** Add G2 benchmark: measure α_c for d=2, 3, 4 at N=64, 256, 1024. Compare to paper's asymptotic α_c (0.05, 0.62, 2.41). Document finite-N corrections.
- [x] **T4.3** Add G2 benchmark: measure α_c at N=8 (our belief dim) for d=3. This is the critical finite-N test for Fusion A — if α_c(N=8, d=3) is much lower than the asymptotic 0.62, the Plan 276 unblock is at risk.
- [x] **T4.4** Add G2 benchmark: measure α_c on CORRELATED memories (not Haar-random). Generate memories as `ξ^μ = cos(θ_μ) · v_base + sin(θ_μ) · v_orth` with varying `θ_μ` spread. Document how correlation reduces α_c.

---

## Phase 5 — Plan 276 G5 PoC (LOAD-BEARING) — DONE 2026-08-04

### Results

| Kernel | Flips | Tracking | Ambig var | Verdict |
|---|---|---|---|---|
| **LeakyIntegrator** (baseline) | **1** | 1.000 | 0.0000 | reference |
| AttractorKernel (random init, seed=42) | 347 | **0.000** | 5.6616 | fails both axes |
| **CpHopfield** (CP², task-aligned memories, snap=0.5) | **3** | **1.000** | 0.0099 | **PASS** |
| CpHopfield (CP², Haar-random memories, best of 5 seeds × 4 snaps) | 0 | **0.000** | 0.0000 | degenerate FAIL |

**G5 GATE: PASS, narrowly** (criterion: flips ≤ 10× leaky **AND** tracking ≥
leaky − 0.05; measured 3 flips at tracking 1.000).

**Flip count alone is not a sufficient criterion.** It rewards stability, and a
kernel that ignores its input entirely is perfectly stable — 0 flips, *better*
than LeakyIntegrator, whose single flip is the **correct** phase-1 → phase-3
transition. The gate therefore also scores **tracking** (argmax correct in the
settled tail of each driven phase). Hysteresis means resisting *noise*, not
resisting *evidence*.

What passes: CP² recall with task-aligned memories beats the demoted
AttractorKernel on **both** axes — flips 347 → 3 *and* tracking 0.000 → 1.000 —
with no gradient descent anywhere.

What does not: the Haar-random control (strict parity with the random-init
attractor) fails at tracking 0.000 across all 20 (seed, snap) cells. Its 0 flips
are degenerate — the state is pinned in a random basin and never follows the
evidence. **So the BBP gap does not confer hysteresis from arbitrary memories.**
Plan 276's blocker allowed "trained **or hand-set**" weights, so what is refuted
is the *training* requirement, not the *alignment* requirement — i.e. exactly
freeze/thaw Path 1, exactly what T5.1 intended.

Snap-strength sweep (how much the CP recall pulls toward the closest memory):

| Snap | Flip-flops | Notes |
|---|---|---|
| 0.00 | 48 | Pure leaky + manifold projection (projection alone introduces noise) |
| 0.25 | **1** | Matches leaky exactly — gentle snap provides hysteresis without cost |
| 0.50 | 3 | Default — strong BBP-protected hysteresis |
| 0.75 | 21 | Too aggressive — snap overpowers input tracking |
| 1.00 | 9 | Hard snap — still ≤10, but no input tracking |

**Robustness caveat:** the sweep is **non-monotone** (48 → 1 → 3 → 21 → 9), so the
result depends on a hyperparameter with no principled setting and the margin is not
clean. Note especially snap=0.00: manifold projection *without* the memory snap
scores 48 flips, worse than leaky's 1 — the projection alone costs stability and
the memory term has to pay it back. Hysteresis is therefore **not** free.

### Architectural honesty note

The adapter is a **leaky + CP-snap hybrid**, not a drop-in AttractorKernel
replacement. CP^(d-1) recall is content-addressable memory (CAM); Plan 276
G2.1 is streaming belief tracking. The adapter bridges them by leaky-
integrating the input (for tracking) then snapping toward the closest CP²
memory (for hysteresis). The memories are 2 canonical belief patterns ("dim 0
dominant" / "dim 1 dominant") loaded at construction via `push_memory` — the
freeze/thaw path, no training.

A sibling agent (commit `ab23ba375`, riir-ai) added two controls missing from
the original PoC: a Haar-random memory arm (strict parity with
`AttractorKernel::from_seed`) and a tracking-score responsiveness floor. These
narrow the verdict — see Research 466 §3.6 for the full addendum. The
BBP-protection mechanism works (G1/G2/G7 confirm it independently), and the
snap-layer integration works with task-aligned memories — but the Haar control
refutes the claim that BBP protection alone confers hysteresis from arbitrary
memories.

### Tasks

- [x] **T5.1** Four-arm comparison in `riir-poc/benches/cp_hopfield_plan276_unblock.rs` (original 3-arm in commit `37d1c259b`; Haar control + tracking floor added in `ab23ba375`):
  - **Baseline A:** random-init `AttractorKernel` (347 flips, tracking 0.000)
  - **Baseline B:** `LeakyIntegrator` (1 flip, tracking 1.000)
  - **Candidate (task-aligned):** CP² recaller with 2 belief-pattern memories (3 flips, tracking 1.000)
  - **Control (Haar-random):** CP² recaller, Haar memories, 5 seeds × 4 snaps (0 flips but tracking 0.000 → degenerate FAIL)
- [x] **T5.2** All arms ran on the G2.1 belief benchmark (1000-step protocol, dim=8). Flip counts + tracking scores recorded above.
- [x] **T5.3** **G5 PASS, narrowly:** Task-aligned arm: 3 flips, tracking 1.000 (criterion: flips ≤ 10 AND tracking ≥ 0.95). Haar control: FAIL (tracking 0.000). Fusion A validated in the narrow sense — "modelless given a frozen memory set", not "for free".
- [x] **T5.4** **G7 measurement:** the BBP gap is confirmed by `bbp_gap_shrinks_with_load` in `cp_hopfield/tests.rs` AND by the G7 GOAT bench (relative gap 0.73–0.95 at N ∈ {8, 64}, see Benchmark 567). The task-aligned arm's flip reduction (347 → 3) is consistent with a gapped kernel; the Haar arm's degenerate failure shows the gap alone is insufficient — alignment is also required.
- [x] **T5.5** PoC addendum in Research 466 §3.6 (revision note + corrected table + confirmed/refuted table). Super-GOAT stands on a narrower base than §3 claimed — see the confirmed/refuted table above.

---

## Phase 6 — Fusion B / G6 (KG Capacity) — MEASURED, FAIL (LLG unblock refuted + projected-cosine diagnostic) — `Issue 033`

### Scope correction (2026-08-04)

**T6.1's original scope was wrong.** It said "re-parameterize `style_weights[64]`
as CP⁷ (d=8) Bloch vectors" — but the retrieval layer operates on 8-dim vectors
(`ITEM_EMBED_DIM = 8`, `BELIEF_DIM = 8`), not 64-dim shard storage. CP² (d=3,
D2=8) is the correct dimension. This eliminates the invasive `NeuronShard`
migration. See Issue 033 for the full substrate-first analysis.

### G6 result + follow-ups (2026-08-04)

**Initial measurement (single-step `query_cp`):** CP² recall is consistently
WORSE than cosine ANN at every N (capacity ratio 1.00×, criterion ≥ 3×).

**LLG unblock follow-up — REFUTED.** Added `query_cp_llg` which runs the full
LLG flow to convergence (via `llg_recall`). Result: **bit-identical precision**
to the single-step `query_cp` at every N. Both methods arrive at the same
basin — the LLG converges to the top eigenvector's attractor, which is exactly
what `recall_step` returns in one step. The "iterative recall" unblock
hypothesis (listed as the most promising follow-up in the prior session) is
refuted.

**Projected-cosine diagnostic — SURPRISING FINDING.** Added a diagnostic arm
that projects embeddings to CP² then does cosine matching (no recall
dynamics). Result: projected-cosine beats raw-cosine by 3–9× at every N:

| N | raw cosine | projected cosine | CP² recall (1-step = LLG) |
|---|---|---|---|
| 4 | 0.500 | 0.750 | 0.250 |
| 32 | 0.219 | 0.719 | 0.031 |
| 64 | 0.156 | 0.516 | 0.000 |
| 256 | 0.027 | 0.234 | 0.004 |

The CP² manifold projection is a **denoising** operation, not a lossy one.
The real bottleneck is that associative recall trades angular precision for
basin robustness — the Hebbian kernel mixes correlated memories, and the top
eigenvector points at the cluster centroid, not the individual memory.

**Potential new G6 variant (not pursued):** projected-cosine ANN as a
preprocessing step (project embeddings + query to CP² before cosine matching).
The 239 ns projection cost is negligible per-query. This is a different Fusion
B path than recall dynamics. Filed as a finding, not pursued — the current G6
gate measures recall capacity, not preprocessing quality.

### Tasks (moved to Issue 033)

> **Scope corrected** — see Issue 033. T6.1's CP⁷ re-parameterization is
> superseded by the CP² (d=3) additive-view approach. T6.2–T6.5 are
> preserved as-is (they're dimension-agnostic). The task list below is the
> original text; the corrected scope lives in Issue 033.

- [-] **T6.1** ~~Implement `NeuronShard` CP^(d-1) view — re-parameterize `style_weights[64]` as CP⁷ (d=8) Bloch vectors.~~ **SUPERSEDED + MEASURED FAIL** (Issue 033, 2026-08-04). Scope corrected to CP² (d=3, D2=8) additive view; `query_cp` + `query_cp_llg` both measured — CP² recall consistently WORSE than cosine ANN at every N (capacity ratio 1.00×, criterion ≥3×). LLG unblock refuted (bit-identical to single-step). Projected-cosine diagnostic showed projection HELPS 3–9× but associative recall destroys angular precision — fundamental property, not a bug.
- [-] **T6.2** ~~Add `ItemEmbedIndex::query_cp`~~ **DONE + G6 FAIL** (Issue 033 T1, commit `0408663`). The path exists but measured worse than cosine; kept as opt-in `cp_recall` feature for research, not production.
- [-] **T6.3** `vibe.rs` KG triple emission via top-eigenvector — **NOT PURSUED** (G6 FAIL). No value wiring KG emission through a recall path that underperforms cosine ANN.
- [-] **T6.4** ~~G6 benchmark~~ **DONE + FAIL** (Issue 033 T2, commit `32c35b9`). CP² precision 0.00–0.25 vs cosine 0.03–0.50 at N=4–256. Capacity ratio 1.00× (criterion ≥3×). Decisively closed.
- [-] **T6.5** `MerkleFrozenEnvelope` commitment of CP^(d-1) memory sets — **NOT PURSUED** (G6 FAIL). No production value committing a recall path that underperforms.

---

## Phase 7 — GOAT Gate + Promotion Decision — DONE 2026-08-04

### Tasks

- [x] **T7.1** Run G1–G7 full gate. Document results in [.benchmarks/567](../.benchmarks/567_cp_hopfield_goat.md).
- [x] **T7.2** **Promotion decision: STAYS OPT-IN.** Default-on requires G5 + G6 + G7. G6 is unmeasured (Phase 6 deferred) and G5 passes only narrowly, so the precondition is not met. See Decision below.
- [x] **T7.3** Update per-stack promote/demote ledger in research note 466 §3.
- [x] **T7.4** Stays opt-in — no `.docs/09_feature_catalog/` update needed.
- [x] **T7.5** Plan 276 benchmark note: Fusion A unblocked in the narrow sense (modelless given frozen memory set). AttractorKernel re-promotion deferred pending G6 + a consumer with aligned memories.

### Results

| Gate | Result |
|---|---|
| G1 correctness | **PASS** — 27 unit tests |
| G2 capacity | **PASS** — `α_c(d=3)/α_c(d=2) = 7.4×` at N=64; d-scaling holds at every N |
| G3 no-regression | **PASS** — opt-in, default-off |
| G4 perf | **PASS** — recall 331 ns / project 239 ns / LLG 589 ns, 0 allocs |
| G5 Plan 276 unblock | **PASS, narrowly** — 3 flips at tracking 1.000; Haar control fails |
| G6 Fusion B (KG capacity) | **FAIL** — CP² worse than cosine at every N; LLG follow-up produces bit-identical precision to single-step (refutes "iterative recall" unblock); projected-cosine diagnostic shows projection HELPS 3–9× but associative recall destroys angular precision (Issue 033) |
| G7 BBP gap | **PASS** — 0.73–0.95 vs a 0.1 bar, at N ∈ {8, 64} |

### Decision: STAYS OPT-IN

T7.2's table requires G5 **and** G6 **and** G7 for default-on. G6 is unmeasured, so
the precondition is not met regardless of anything else. Two further reasons not to
force it:

1. **G5 passes only in the narrow sense.** The Haar-random control fails
   degenerately, so the mechanism needs a task-aligned frozen memory set. That is a
   legitimate freeze/thaw Path 1 story, but it is a weaker claim than §3 made and it
   means the value depends on a consumer supplying aligned memories.
2. **The G5 margin is hyperparameter-sensitive.** Non-monotone snap sweep
   (48 → 1 → 3 → 21 → 9) with no principled setting.

Per Research 466 §3, the verdict stands on the axes that were confirmed — capacity
(G2) and BBP protection at finite N (G7), both strong — with Fusion A confirmed in
its narrow form. Nothing was silently revised; the earlier unqualified "hysteresis
is free" claim is corrected in place with the measurement that refuted it.

### Corrections to this plan's own premises, found by measuring

- **Risk Register's finite-N concern is REFUTED.** `α_c` is *higher* at small N and
  decreases toward the asymptote as N grows (d=3: 1.696 → 1.295 → 0.909 at
  N = 8 → 64 → 256). Small N is favorable.
- **T4.4's expectation was wrong.** Correlated memories recall the *cued* memory
  better, not worse. The cost of correlation is discriminability (the shadow
  phenomenon), not `α_c`.
- **Research 466's `O(d³)` cost model is wrong.** Recall is
  `O(P·N·D2 + d³)` and the omitted Mattis term dominates by ~100× at
  `P=16, N=64`. Fixed via an exact global-sum cache.
- **T1.3 (hardcoded structure-constant tables) was not viable** at d=8 (63³
  entries); constants are contracted from the closed-form basis instead.
- **T2.1 (iterative projected gradient) was unnecessary** — the manifold projection
  has a closed form, because the Bloch map is a Euclidean similarity and the closest
  pure state is the top eigenvector of `ρ`.

## Risk Register

| Risk | Mitigation |
|---|---|
| **G5 FAILS** (Plan 276 unblock refuted) | Verdict honestly drops to GOAT. Fusion B (shard capacity) may still hold. Document in §3.6 PoC addendum. Do NOT silently revise the Super-GOAT claim. |
| **Finite-N α_c much lower than asymptotic** | G2 measures this at N=8, N=64. If α_c(N=8, d=3) << 0.62, the capacity claim weakens. May restrict the primitive to d ≥ 4 where finite-N effects are smaller. |
| **Non-linear constraint projection too costly** | G4 measures `project_to_manifold` cost. If > 1μs for d=3, optimize (closed-form projection for CP²). If still too costly at d=8, restrict to d ≤ 4. |
| **Correlated memories behave very differently** | G2 + G1 test on correlated distributions. If α_c drops by > 2× on correlated vs Haar-random, document as a real-world capacity reduction. |
| **Shadow phenomenon causes personality bleed** | Characterize when shadow is desirable (KG retrieval — richer context) vs undesirable (personality recall — bleed). Add a `shadow_suppression` knob if needed. |
| **RSB corrections at d=8** | Our primary use case is d=3, d=4 (small). If we push to d=8 and α_c is much lower than the replica-symmetric prediction, RSB is the likely cause. Document; restrict to d ≤ 4 if needed. |
| **Integration cost (Phase 6)** | Re-parameterizing `style_weights[64]` is invasive. P1 shard bridge is a non-trivial migration. Time-box; if it exceeds budget, ship Phase 1–5 only (open primitive + PoC) and defer Phase 6. |

---

## References

- [Research 466](../.research/466_CPd_minus_1_Hopfield_Top_Eigenvector_Recall.md) — the open primitive note
- [riir-neuron-db/.research/304](../../riir-neuron-db/.research/304_Symmetric_Space_Hopfield_Super_GOAT_Guide.md) — the private Super-GOAT guide
- [Plan 276 benchmark](../.benchmarks/276_micro_belief_goat.md) — the documented blocker (G5 load-bearing gate)
- [Research 455](../.research/455_Hebbian_Kernel_Memory_Fact_Storing_MLP.md) — Hebbian Kernel Memory (construction-side cousin)
- [Research 317](../.research/317_Reasoning_As_Attractor_Dynamics_Gibbs_Retrieval.md) — Reasoning as Attractor (Gibbs retrieval cousin)
- Galitski 2026: [alphaXiv 2607.hopfield-networks](https://www.alphaxiv.org/abs/2607.hopfield-networks)
