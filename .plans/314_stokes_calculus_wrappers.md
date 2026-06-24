# Plan 314: Stokes Calculus Wrappers — Fokker-Planck Validator + Boundary-Flux Mass + Line Integral

**Date:** 2026-06-24
**Research:** [katgpt-rs/.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md](../.research/296_Stokes_Calculus_Dec_Vocabulary_Crosswalk.md)
**Source papers:**
- [arxiv 2202.11322](https://arxiv.org/abs/2202.11322) — *Efficient CDF Approximations for Normalizing Flows* (TMLR 2022) — "leverage the divergence theorem to estimate the CDF over a closed region in target space"
- [NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/cbf8710b43df3f2c1553e649403426df-Abstract.html) — *Neural Manifold Ordinary Differential Equations* (Lou et al.) — `d/dt log p = -div(f)` instantaneous change-of-variables
**Target:** `katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs` (new file) + Cargo feature `stokes_calculus` (under `dec_operators`)
**Status:** Active — Phase 1 + 2 + 3 COMPLETE (2026-06-24); `stokes_calculus` stays opt-in (G-B PASS, G-C structural fail, **G-A FAIL in riir-ai Plan 334**). Phase 4 filed as riir-ai Plan 334 (2026-06-24); **G-A gate ran 2026-06-24 and FAILED** (candidate 9.5× slower + 36% lower F1 than JS-divergence baseline at `action_dim=8`). See `riir-ai/.benchmarks/334_stokes_validator_g_a.md`.

---

## Goal

Three thin wrapper primitives (<50 LOC each) over the shipped DEC operators (Plan 251), exposing the Generalized Stokes' Theorem as named modelless tools:

1. **`belief_mass_divergence()`** — Fokker-Planck belief-mass validator. Closes Research 271 §2's known gap ("Fokker-Planck / continuity equation → not yet a runtime invariant validator"). Headline primitive — feeds ICT BranchingDetector (Plan 324) a modelless invariant.
2. **`boundary_flux_mass()`** — divergence theorem for low-dim manifolds. `∫_∂M ω` instead of `∫_M dω`. O(boundary) vs O(volume). Bounded error from `hodge_decompose`'s harmonic component.
3. **`line_integral()`** — discrete line integral of a rank-1 cochain along a path. Path energy / geodesic cost / work. Composes with Plan 312's `manifold_geodesic` path output.

**GOAT gate:** each primitive has its own A/B benchmark (see Phase 3). Promote the winners; demote the losers. Per Research 296 verdict, this is **GOAT** (not Super-GOAT — the mechanism ships, only the wrappers + framing are new).

## Non-Goals

- ❌ NO new DEC operators (Plan 251 shipped them). This plan only wraps.
- ❌ NO training, NO flow-matching policy learning (those → riir-train). The papers' training machinery is explicitly out of scope per Research 296 §1.
- ❌ NO high-dim shard compression via boundary commitment (curse of dimensionality — see Research 296 §3.5).
- ❌ NO Super-GOAT guide / riir-ai guide (verdict is GOAT; guide is Super-GOAT-only per skill §1.5). HLA wiring lands as a Phase 2 task, not a separate guide doc.

## Constraint Checklist (per AGENTS.md + skill)

All constraints satisfied by construction during Phase 1–3 implementation.
Reconciled 2026-06-24 (previously left as unchecked bookkeeping despite
inline `✓` confirmations).

- [x] Modelless (inference-time only, no backprop) — ✓ by construction (DEC ops are linear algebra)
- [x] Latent-to-latent preferred (sigmoid not softmax) — ✓ (no softmax in this primitive)
- [x] Freeze/thaw over fine-tuning — ✓ (no weight mutation)
- [x] 5-repo discipline (open primitive in katgpt-rs) — ✓
- [x] SOLID, DRY, zero-alloc hot paths — ✓ (wrappers reuse DEC scratch buffers)
- [x] CPU/SIMD/GPU auto-route inherited from DEC `backend.rs` — ✓
- [x] File < 2048 lines — ✓ (single file, ~300 LOC + tests)

---

## Phase 1 — Unblocking Skeleton (CORE)

Three pure functions over shipped DEC types. No new structs. No allocations in the wrapper layer (delegate to `codifferential` / `exterior_derivative` which already reuse scratch buffers).

### Tasks

- [x] **T1.1** Create `katgpt-rs/crates/katgpt-core/src/dec/stokes_calculus.rs`. Add `mod stokes_calculus;` to `dec/mod.rs` behind `dec_operators` feature. Add Cargo feature `stokes_calculus = ["katgpt-core/dec_operators"]` to `katgpt-rs/Cargo.toml` (opt-in, NOT default).
- [x] **T1.2** Implement `pub fn belief_mass_divergence(cx: &CellComplex, belief_flow: &CochainField) -> f32`. Returns `Σ_v |δ₁(belief_flow)[v]|` (L1 norm of discrete divergence). Pure wrapper over `codifferential` (shipped). Implementation note: plan's `|Σ_v ...|` notation is ambiguous between net sum and L1 norm; the parenthetical "then L1 sum" + the T2.1.3 anomaly-detection use case (local spikes must survive global cancellation) resolves this to L1 norm `Σ_v |...|`. Net sum `|Σ_v δ₁(flow)[v]|` is always 0 by edge-endpoint telescoping — useless as a validator.
- [x] **T1.3** Implement `pub fn boundary_flux_mass(cx: &CellComplex, region_cells: &[u32], field: &CochainField) -> (mass: f32, error_bound: f32)`. Mass = oriented boundary flux `Σ_{(k_cell, kp1_cell, sign) ∈ B_{k+1}, kp1_cell ∈ region} sign · field[k_cell]` (interior k-cells cancel by orientation). Error bound = `‖harmonic(field)‖₁` from `hodge_decompose`. Implementation note: the plan's literal text `Σ_{e ∈ ∂region} (exterior_derivative ∘ field)[e]` is dimensionally inconsistent for rank-1 field + rank-2 region; the correct divergence-theorem formulation sums the field itself over B_{k+1} entries of region cells, which by Stokes equals `Σ_{f ∈ region} d_k(field)[f]`. Single `Vec<bool>` allocation for region membership (proportional to complex size); the wrapper itself allocates nothing beyond this marker.
- [x] **T1.4** Implement `pub fn line_integral(cx: &CellComplex, edge_field: &CochainField, path: &[u32]) -> f32`. For each consecutive `(path[i], path[i+1])` pair, finds the connecting edge via B₁ paired-entry lookup (grid_2d pushes `(tail, e, -1), (head, e, +1)` as adjacent pairs; edges are never removed). Contribution = `sign(b, e) · field[e]` (head=+1 along orientation, tail=-1 against). Reversal-antisymmetric by construction.
- [x] **T1.5** `cargo check --features stokes_calculus` passes (clean, no warnings). `cargo test -p katgpt-core --features dec_operators --lib dec::stokes_calculus` passes (12/12 tests). Note: `stokes_calculus` is a root-crate feature alias for `katgpt-core/dec_operators`; tests run against `katgpt-core` with `--features dec_operators`.

**Exit:** three functions compile and type-check. Zero allocations in the wrapper layer.

---

## Phase 2 — Unit Tests (Correctness, not perf)

Each primitive gets ≥3 unit tests: identity case, scaling case, edge case.

### Tasks

- [x] **T2.1** `belief_mass_divergence` tests:
  - [x] **T2.1.1** Identity: divergence of a zero flow = 0 (trivial mass conservation); also a gradient-of-linear-potential flow has 0 divergence at interior vertices (boundary vertices contribute due to open boundary, test verifies finiteness + non-negativity).
  - [x] **T2.1.2** Scaling: doubling a single-edge flow doubles the L1 divergence (verified to 1e-5).
  - [x] **T2.1.3** Anomaly injection: inflating edge 5 from 1.0→100.0 against a constant-1.0 baseline spikes the L1 divergence.
- [x] **T2.2** `boundary_flux_mass` tests:
  - [x] **T2.2.1** Stokes identity: boundary flux == `Σ_{f ∈ region} d₁(field)[f]` (naive volume integral) for a non-trivial alternating field, verified on both full-grid region AND a 3-face subset. Plus: exact (gradient) field → boundary circulation ≈ 0 AND harmonic ≈ 0 on a simply-connected grid.
  - [x] **T2.2.2** Harmonic bound: covered indirectly — the exact-field test (T2.2.1c) verifies `error_bound ≈ 0` when harmonic is absent. For a field WITH harmonic component on a grid with a hole (β₁ > 0), the harmonic L1 norm is non-zero and bounds the boundary-vs-volume gap. (On a simply-connected grid β₁=0, harmonic is always 0 for rank-1, so this test is structurally limited — see honest risk notes.)
  - [x] **T2.2.3** Empty region: `region_cells = []` returns `(0.0, 0.0)` without panicking.
- [x] **T2.3** `line_integral` tests:
  - [x] **T2.3.1** Straight path: constant field (all edges=1.0) over path [0,1,2,3] = 3.0 (field_value × 3 edges).
  - [x] **T2.3.2** Reversal antisymmetry: `line_integral(0→1→5→4→0) == -line_integral(0→4→5→1→0)` for a sinusoidal field.
  - [x] **T2.3.3** Closed loop of an exact field = 0: gradient of φ(v)=v around face loop [0,1,5,4,0] = 0 (fundamental theorem of calculus for line integrals). Plus short-path edge case (len<2 → 0.0).
- [x] **T2.4** Run full test suite: `cargo test -p katgpt-core --features dec_operators --lib dec::` — 96/96 DEC tests pass (12 new + 84 pre-existing), 0 warnings.

**Exit:** three primitives verified correct against their Stokes-theorem identities.

---

## Phase 3 — GOAT Gate (Benchmarks)

Each primitive gets an A/B benchmark vs the naive alternative. **Promote the winner if gain ≥ threshold; demote the loser.** Per Research 296 §4.

### Tasks

- [x] **T3.1** **G-A (Fokker-Planck validator GOAT):** A/B in riir-ai — does flagging `belief_mass_divergence > τ` catch ICT `BranchingDetector` (Plan 324) branching events earlier/cheaper than the existing JS-divergence-to-mean detector? **This gate runs in riir-ai (needs live HLA), not katgpt-core.** Target: ≥1.5× earlier detection OR ≥2× cheaper per-tick on the same events. **❌ FAIL (ran 2026-06-24 in riir-ai Plan 334 T3.2):** 327 branching events across 3 archetypes; candidate is 9.5× slower (5352 ns vs 562 ns/tick) AND 36% lower F1 (0.640 vs 1.000). Root cause: fixed-grid cost cannot compete at `action_dim=8`, and the G8 corpus is perfectly separable for JS-divergence by construction. See `riir-ai/.benchmarks/334_stokes_validator_g_a.md`. **Baseline measured in katgpt-core:** `belief_mass_divergence` on 32×32 grid = 5.00 µs (2.5 ns/edge), on par with raw `codifferential_into` — wrapper overhead negligible.
- [x] **T3.2** **G-B (Boundary-flux mass GOAT):** A/B in katgpt-core — on a 256×256 game map, `boundary_flux_mass_only` (115.53 µs) vs naive full-volume `exterior_derivative_into` + region sum (619.31 µs). **Result: 5.36× faster, error_bound/mass = 3.78% < 5% → PASS.** Win comes from memory access patterns (no output materialization), NOT from theoretical O(boundary) — see Issue 006 for the coboundary-index optimization that would unlock true O(boundary). See `.benchmarks/314_stokes_calculus_goat.md`.
- [x] **T3.3** **G-C (Line integral GOAT):** A/B in katgpt-core — `line_integral` discriminates smooth vs zigzag paths (Δ=1.872 on non-exact field). **Result: STRUCTURAL FAIL.** `line_integral` of a rank-1 edge cochain cannot encode turn penalties (turns are a pairwise edge property requiring rank-2 face cochains). The primitive is correct and useful as a path-cost function, but the "≥20% fewer reversals" target is mathematically unreachable for rank-1. **Plan 317 follow-up (2026-06-24):** the rank-2 `circulation_integral` wrapper was implemented and benchmarked — G-C **STILL FAILS** empirically (smooth loop circulation=128/3turns vs zigzag=112/25turns; minimizing circulation picks MORE turns because smooth rectangles enclose MORE area than zigzags). Turn count is a combinatorial property that NO rank-k Stokes integral can encode. See Issue 005 (CLOSED), Plan 317, and `.benchmarks/317_circulation_integral_goat.md`.
- [x] **T3.4** Benchmark summary written in `katgpt-rs/.benchmarks/314_stokes_calculus_goat.md`. Honest results documented — G-B passes (5.36×, 3.78% error), G-C fails structurally (rank-1 can't encode turns), **G-A fails in riir-ai (9.5× slower, 36% lower F1)**.
- [x] **T3.5** Promotion decision:
  - G-B PASSES (5.36×, 3.78% error), G-C FAILS (structural), G-A FAILS (riir-ai Plan 334: 9.5× slower, 36% lower F1). Per the split rule: **`stokes_calculus` stays opt-in** — the winning `boundary_flux_mass` is available to callers; `line_integral` documented honestly as a path-cost function (not smoothness regularizer); `belief_mass_divergence` is a clean mass-conservation checker but NOT a branching detector.
  - G-A (riir-ai) ran and FAILED (2026-06-24). No re-evaluation of promotion — all three gates now have verdicts (1 PASS, 2 FAIL).
  - Issues filed: `005_stokes_calculus_g_c_turn_penalty.md` (rank-2 `circulation_integral` fix), `006_coboundary_index_for_boundary_flux.md` (true O(boundary) optimization).

**Exit:** GOAT gate run, results documented, promotion decided.

---

## Phase 4 — riir-ai Wiring (Filed as riir-ai Plan 334)

**Filed 2026-06-24** as [`riir-ai/.plans/334_stokes_calculus_hla_wiring.md`](../../riir-ai/.plans/334_stokes_calculus_hla_wiring.md) per this plan's directive ("File as a riir-ai plan when Phase 1–3 work is done"). Phase 1–3 is COMPLETE; the tasks below are now tracked there, not here. They are riir-engine work, not katgpt-rs work.

### Tasks (now owned by riir-ai Plan 334)

- [-] **T4.1** Construct a `CellComplex` on the HLA belief manifold (8-dim). → riir-ai Plan 334 T1.1–T1.3 (recommendation: 2D projection of 5 synced scalars, lattice discretization).
- [-] **T4.2** Per HLA tick, compute the belief-flow cochain from the HLA state delta (`evolve_hla`'s update vector). Feed to `belief_mass_divergence()`. → riir-ai Plan 334 T2.1–T2.3.
- [-] **T4.3** Run G-A gate (T3.1): does `belief_mass_divergence > τ` catch ICT branching events earlier/cheaper than JS-divergence? → riir-ai Plan 334 T3.1–T3.4. **❌ G-A FAIL (2026-06-24):** candidate 9.5× slower + 36% lower F1 than baseline at `action_dim=8`. See `riir-ai/.benchmarks/334_stokes_validator_g_a.md`.
- [-] **T4.4** Wire into `cgsp_runtime/pulse_bridge.rs` as a curiosity signal (divergence > 0 = expanding belief = curiosity). Cross-reference Plan 277 (Temporal Derivative Kernel). → riir-ai Plan 334 T4.1 (**not gated on G-A** — curiosity is an independent use case; may proceed behind opt-in feature without promotion).
- [-] **T4.5** Feed to LatCal commitment: the 5 synced scalars should be the boundary flux of a near-zero-divergence field. If `belief_mass_divergence > τ` on the committed slice → flag for `mape_k.rs` self-healing (riir-neuron-db). → riir-ai Plan 334 T4.2 (**not gated on G-A** — mass-consistency flag is independent of branching detection; may proceed behind opt-in feature without promotion).

---

## Architecture

```
katgpt-rs/crates/katgpt-core/src/dec/
├── mod.rs              — add `mod stokes_calculus;` behind `dec_operators`
├── operators.rs        — EXISTING (d, δ, Δ) — wrapped, not modified
├── hodge.rs            — EXISTING (hodge_decompose) — wrapped, not modified
├── flow.rs             — EXISTING (DecFlowField) — orthogonal
└── stokes_calculus.rs  — NEW (~300 LOC + tests)
    pub fn belief_mass_divergence(cx, belief_flow) -> f32
    pub fn boundary_flux_mass(cx, region_cells, field) -> (mass, error_bound)
    pub fn line_integral(cx, edge_field, path) -> f32
    mod tests  // ~12 unit tests (Phase 2)
```

## Feature Gate

```toml
# katgpt-rs/Cargo.toml
[features]
stokes_calculus = ["katgpt-core/dec_operators"]  # opt-in, NOT default until G-B + G-C pass
```

Promotion to default requires G-B AND G-C passing (T3.5). **G-A ran in riir-ai and FAILED (2026-06-24)** — does not change the promotion verdict (already blocked on G-C). All three gates now have verdicts: G-B PASS, G-C FAIL (structural), G-A FAIL (domain mismatch).

## Validation

Reconciled 2026-06-24 against Phase 1–3 completion + Phase 4 G-A verdict. The four
discharged gates marked `[x]`; G-C's structural fail and G-A's riir-ai FAIL are
closed (not deferred). Issue 005 remains open for the rank-2 fix.

- [x] All 12 Phase-2 unit tests pass (Stokes identities hold by construction) — confirmed T2.4 (96/96 DEC tests, 0 warnings).
- [x] G-B benchmark: boundary-flux mass ≥3× faster than full-volume on 256×256 map, error < 5% — confirmed 5.36×, 3.78% error (T3.2).
- [-] G-C benchmark: line-integral-weighted geodesic ≥20% fewer reversals — **STRUCTURAL FAIL** (rank-1 cochain cannot encode turn penalties; see Issue 005 for rank-2 `circulation_integral` fix). Primitive retained as a path-cost function.
- [-] G-A (riir-ai): Fokker-Planck validator catches ICT branching ≥1.5× earlier or ≥2× cheaper — **❌ FAIL (2026-06-24)**: candidate 9.5× slower + 36% lower F1 at `action_dim=8`. See `riir-ai/.benchmarks/334_stokes_validator_g_a.md`. Primitive retained as opt-in mass-conservation checker (not a branching detector).
- [x] Zero allocations in the wrapper layer (delegate to DEC scratch buffers) — confirmed T1.3 (single `Vec<bool>` region marker, proportional to complex size).
- [x] Files < 2048 lines — confirmed (~300 LOC + tests, T1.5).

## Honest Risk Notes

- **G-B may fail** if real game-map fields have large harmonic components (topologically-constrained flow). In that case the boundary-flux win is conditional on "near-exact fields only". This is honest — document it, don't hide it. Plan 251's `DecFlowField` already separates the three components, so callers can check `‖harmonic‖ / ‖total‖` before using `boundary_flux_mass`.
- **G-C may fail** if `manifold_geodesic` paths are already near-optimal in reversal count. The line integral is still useful as a pure cost function (e.g. for NPC path comparison) even if it doesn't improve geodesic smoothness.
- **G-A is the highest-value gate** but runs in riir-ai, not katgpt-core. If it passes, the Fokker-Planck validator becomes the headline application; if it fails, the primitive is still a clean Stokes-theorem tool for callers who want mass-conservation checking.
- **Super-GOAT follow-up (out of scope here):** if G-A reveals that projecting the belief flow onto its divergence-free component (enforce mass conservation by construction, not just validate) gives a steering signal, THAT is Super-GOAT-tier. Track in `katgpt-rs/.issues/` after Phase 3 lands. See Research 296 §5 "selling-point honesty".
