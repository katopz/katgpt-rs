# Bench 058: FUNCATTN GOAT Gate — Status

**Date:** 2026-06-17
**Plan:** [286_functional_attention_spectral_transport](../.plans/286_functional_attention_spectral_transport.md)
**Research:** [257_Functional_Attention_Spectral_Transport_Operator](../.research/257_Functional_Attention_Spectral_Transport_Operator.md)
**Reference impl:** [`.raw/FUNCATTN/PDE-StandardBenchmark/model/Functional_attention.py`](../.raw/FUNCATTN/PDE-StandardBenchmark/model/Functional_attention.py)
**Feature flag:** `funcattn` (opt-in, in `full` aggregation, **not** in default features)
**Status:** Phase 1 + G1 PASS; G2–G5 pending

---

## Summary

Shipped the FUNCATTN primal operator as a Gain-tier open primitive in
`crates/katgpt-core/src/funcattn.rs`, matching the reference implementation's
**dual form** (d×d convex-combo regularization `(1-α)·K̃ᵀK̃ + α·I_d`, column-
normalized slice tokens, per-slice-token to_q/to_k/to_v linear projections).
All 13 unit tests pass against a scalar reference. **Not promoted to default
features** — Gain-tier, awaiting G2/G3 accuracy evidence per Plan 286 Phase 4.

---

## Gate Status

| Gate | Description | Status | Notes |
|------|-------------|--------|-------|
| **G1** | Mechanics: finite output, no NaN/Inf, Lipschitz bounded | ✅ PASS | 3 tests: `g1_finite_output_random_inputs`, `g1_sweep_input_norm_and_alpha` (B ∈ {1,10,100} × α ∈ {0.01,0.5,0.99}), `g1_lipschitz_bounded`. Convex combo α∈(0,1) guarantees PD for any input scale — strictly more stable than additive λI. |
| **G2** | Beats Parallax on regression (paper §5.1 setup) | ⏳ DEFERRED | Requires training basis weights W_Φ, W_Ψ via AdamW. Plan 286 T3.2 specifies the Few-Shot-Regression reference (`.raw/FUNCATTN/Few-Shot-Regression/models.py::FuncAttn` L123-176) — different algorithm (primal k×k reg, no to_q/k/v) from the PDE-path we shipped. Either port the few-shot variant or run training externally (Python) and import weights. |
| **G3** | Sigmoid-basis ≈ softmax-basis on PDE proxy | ⏳ DEFERRED | Requires trained basis weights. Cross-check `matches_reference_sigmoid` + `matches_reference_softmax` verify the two basis kinds produce self-consistent output vs. a scalar reference, but not relative accuracy on a task. |
| **G4** | Linear-in-n scaling at n ∈ {512, 2048, 8192} | ⏳ DEFERRED | `forward_large_n_smoke` confirms n=2048 completes without NaN, but no timing/bench yet. Complexity is `O(n·d·k + k·d² + d³)` → linear in n for k≪n; should pass trivially. Needs a criterion bench (`.benchmarks/058_funcattn_goat.md` already reserves this slot for the timing table). |
| **G5** | Zero-alloc hot path | ⏳ NOT FORMALLY CHECKED | All scratch is pre-allocated in `FuncAttnScratch`, all `_into` APIs, in-place Cholesky (no `.clone()`). Allocator-counting bench not yet wired. Should pass by inspection. |

---

## Phase 1 Deliverables (DONE)

- ✅ T1.1 — `funcattn` feature in both `Cargo.toml` files, in `full`, not in default.
- ✅ T1.2 — `FuncAttnBasis` (Softmax/Sigmoid), `FuncAttnConfig` (d, k, basis, alpha, temperature, cholesky_jitter), `FuncAttnScratch` (11 pre-allocated buffers).
- ✅ T1.3 — `compute_basis_into(x, w, bias, n, d, k, kind, temperature, out)` — zero-alloc, partition-of-unity verified for both basis kinds × τ ∈ {0.1, 0.5, 1.0, 5.0}.
- ✅ T1.4 — `funcattn_forward(x_basis, x_value, w_basis, w_q, w_k, w_v, cfg, scratch, out)` matching reference L50-89: basis → column-normalized slice tokens → to_q/k/v linear → dual-form convex-combo Tikhonov solve → inverse projection.
- ✅ T1.5 — `pub fn solve_convex_combo_dual(k_slice, alpha, d, k, reg, y_buf, z_op_t, jitter)` helper. Vendored ~40-line in-place Cholesky (`cholesky_inplace`, `cholesky_solve_into`) — MIT-compatible, exploits PSD structure, faster than LU.

## Phase 2 Status (in progress)

- ✅ T2.1 (G1) — Mechanics gate passes (3 tests).
- ⏳ T2.2 (G4) — Needs criterion bench harness.
- ⏳ T2.3 (G5) — Needs allocator-counting bench.

## Phase 3 Status (deferred)

- ⏳ T3.1 (G3), T3.2 (G2) — Require trained basis weights. The Few-Shot-Regression
  reference (paper §5.1 setup) uses a different algorithm variant than the PDE
  path we shipped; see "Algorithm variant mismatch" below.

---

## Key implementation notes

### Primal vs dual form

The paper (Eq. 7) writes the additive primal form `K̃·K̃ᵀ + λI_k` (k×k).
**The reference implementation uses the dual form** (reference L71-76):
regularize the d×d matrix `(1-α)·K̃ᵀ·K̃ + α·I_d`. We follow the reference
because (1) convex combo guarantees bounded spectrum for any α ∈ (0,1),
strictly more stable than additive λI for rank-deficient K̃; (2) the
d×d form matches the reference's empirical results verbatim.

### Convex-combo vs additive regularization

- Paper Eq. 7: `reg = K̃K̃ᵀ + λI` (additive, unbounded spectrum as λ→0)
- Reference L74: `reg = (1-α)·K̃ᵀK̃ + α·I` (convex combo, bounded for α∈(0,1))

The convex combo makes Cholesky PD-guaranteed — we never trigger
`NotPositiveDefinite` for α > 0, even with degenerate (all-zero) w_k.
This is a strict robustness improvement.

### Column-normalized slice tokens

The reference (L62-64) divides `Φᵀ · x_value` by the column sums of Φ
(`slice_norm[g] = Σ_n Φ[n,g]`), producing weighted **averages** per basis
partition, not raw sums. This is what `slice_token` represents. The inverse
projection (L87) reuses the same Φ without normalization.

### to_q / to_k / to_v on slice_token

The PDE reference applies three separate linear projections (`to_q`, `to_k`,
`to_v`, each d×d) to slice_token AFTER basis projection. This is absent from
the paper's primal formulation but present in all shipped PDE variants. Our
implementation requires these three weight matrices as forward-pass inputs.

### Orthogonal init for w_basis

Reference L20-21 calls `torch.nn.init.orthogonal_(self.in_project_basis.weight)`.
**Caller responsibility** in our inference-time primitive — we don't initialize
weights. Training-side code must apply orthogonal init before the first forward.

---

## Algorithm variant mismatch (Plan 286 T3.2 issue)

Plan 286 T3.2 specifies the Few-Shot-Regression reference (`.raw/FUNCATTN/
Few-Shot-Regression/models.py::FuncAttn` L123-176) for the G2 regression gate.
**That code path uses a different algorithm than what we shipped:**

| Aspect | PDE code (shipped) | Few-shot code (T3.2 target) |
|--------|--------------------|-----------------------------|
| Q source | `to_q(slice_token)` | `encoder(xq)` (separate query input) |
| K source | `to_k(slice_token)` | `encoder(xc)` (same encoder as Q) |
| V source | `to_v(slice_token)` | `yc` (raw y-values at context) |
| Slice-token Q/K/V projections | Yes (to_q/k/v applied) | No (encoders applied to raw x) |
| Regularization form | `(1-α)·K̃ᵀK̃ + α·I_d` (d×d dual) | `(1-ridge)·kkH + ridge·I` (k×k primal, L173) |
| Output projection | `Φ · out_slice` | Direct (C_mat · v, no Φ) |

To run G2 against the few-shot benchmark verbatim, we'd need to ship a second
forward function (`funcattn_forward_fewshot`?) that implements the few-shot
algorithm. **Alternative:** train basis weights externally (Python) and import
them into our PDE-path implementation; the comparison would then be "our
PDE-path FUNCATTN vs. our Parallax at matched parameter count", which is a
valid G2 even if it doesn't reproduce the paper's headline result exactly.

**Recommendation:** defer G2 to riir-ai Plan 318 (the rank-k latent functor
upgrade is the primary value path anyway) and run a simpler synthetic G2 in
katgpt-rs once a small training loop is available.

---

## Files

| File | Role |
|------|------|
| `crates/katgpt-core/src/funcattn.rs` | Module (1344 lines including tests) |
| `crates/katgpt-core/src/lib.rs` | `pub mod funcattn;` + re-exports |
| `crates/katgpt-core/Cargo.toml` | `funcattn = []` feature |
| `Cargo.toml` | `funcattn = ["tiled_attention", "katgpt-core/funcattn"]`, added to `full` |

## Test results

```
running 13 tests
test funcattn::tests::forward_zero_weights_alpha_positive_succeeds ... ok
test funcattn::tests::cholesky_inplace_indefinite_fails ... ok
test funcattn::tests::basis_rows_partition_of_unity ... ok
test funcattn::tests::cholesky_inplace_basic_spd ... ok
test funcattn::tests::cholesky_solve_known_system ... ok
test funcattn::tests::matches_reference_extreme_alpha ... ok
test funcattn::tests::matches_reference_temperature_sweep ... ok
test funcattn::tests::matches_reference_sigmoid ... ok
test funcattn::tests::matches_reference_softmax ... ok
test funcattn::tests::g1_finite_output_random_inputs ... ok
test funcattn::tests::g1_sweep_input_norm_and_alpha ... ok
test funcattn::tests::g1_lipschitz_bounded ... ok
test funcattn::tests::forward_large_n_smoke ... ok

test result: ok. 13 passed; 0 failed
```

## Verdict (Phase 4 — pending)

**Do NOT promote `funcattn` to default features.** G1 passes (mechanics verified)
but G2 (regression accuracy vs. Parallax) is the actual GOAT decision and it's
deferred. Per Plan 286 T4.4: do not promote until LLM-domain token-prediction
evidence exists, which is itself a separate gate deferred per Research 257 §5 Q2.

The primitive is shipped and usable via `--features funcattn`. The convex-combo
dual form gives strict numerical-stability improvements over the paper's
additive primal form (PD-guaranteed for any α∈(0,1)), which is a useful
contribution independent of the accuracy gate outcome.
