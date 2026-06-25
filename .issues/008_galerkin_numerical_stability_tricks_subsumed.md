# Issue 008 — Galerkin Transformer Numerical-Stability Tricks: Subsumed by FUNCATTN α-Regularization

[← Index](../README.md) · **Type:** optimization-analysis · **Priority:** low · **Status:** CLOSED (2026-06-25) — all three tricks subsumed or N/A; no benchmark warranted.

## Origin

Research 306 (`.research/306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md`)
verdict'd the Galerkin Transformer (arXiv:2105.14995) as **Gain** — it is FUNCATTN's
λ=0, identity-basis grandparent predecessor. The note §1.3 flagged three
numerical-stability tricks from the paper as "worth recording" / "alternative to
benchmark" for Plan 286 (FUNCATTN). Plan 286 is now fully complete (all gates
G1–G6 run, `funcattn` shipped opt-in behind the `full` aggregation). This issue
evaluates whether those three tricks offer any benefit to the **shipped** FUNCATTN
primitive, and closes them out.

## The three tricks

### 1. Diagonal-dominant rescaled initialization: `W_init ← ηU + δI`

**Galerkin's purpose.** Stabilize training in the absence of softmax by adding a
small diagonal perturbation δ to the weight initialization. Paper Appendix C.2
claims up to 50% eval-accuracy boost. Needed because Galerkin has λ=0 — no
runtime diagonal floor exists.

**FUNCATTN's coverage.** FUNCATTN's regularization matrix is
`(1-α)·K̃ᵀK̃ + α·I_d` (see `funcattn.rs::solve_convex_combo_dual`, L450-504, and
the call site in `funcattn_forward` at L692-702). The `α·I_d` term is a diagonal
floor added at **runtime on every forward pass**, not just at init. For any
`α ∈ (0,1)` the matrix is positive definite regardless of K̃'s rank or
conditioning — Cholesky cannot fail (the `cholesky_jitter` fallback at L701 is a
defense against float drift only).

**Verdict: SUBSUMED.** Galerkin's init-time `δI` ≈ FUNCATTN's runtime `α·I_d`,
but weaker (init-time hope vs. runtime guarantee). Mathematical identity — no
benchmark can distinguish them.

### 2. Galerkin projection-type layer normalization (pre-dot-product, scale-preserving)

**Galerkin's purpose.** Apply LN to Q,K (Galerkin variant) or K,V (Fourier
variant) AFTER the linear projection, BEFORE the dot-product attention. Bounds
`‖Q·Kᵀ‖` magnitudes. Paper Table 8 ablation: regular (post-attention,
scale-eliminating) LN fails to converge for the Galerkin Transformer under
1cycle scheduling without 0.1 attention dropout. This trick exists **because
Galerkin has λ=0** — nothing else bounds the dot-product magnitudes.

**FUNCATTN's coverage — K-side.** FUNCATTN does not compute raw `Q·Kᵀ`. It
computes the ridge solve `C = Q̃ · reg⁻¹ · K̃ᵀ` where
`reg = (1-α)·K̃ᵀK̃ + α·I_d`. Eigenvalues of `reg` are `(1-α)·σ_i² + α` (σ_i =
singular values of K̃), so `λ_min(reg) ≥ α` and `‖reg⁻¹‖ ≤ 1/α`. The K-side of
the solve is well-conditioned by construction — pre-LN on K̃ is redundant with α.

**FUNCATTN's coverage — Q-side.** FUNCATTN does not explicitly regularize Q̃'s
magnitude. However, Stage 8 reconstructs the output as a partition-of-unity
convex combination:
`out[n,:] = Σ_g Φ[n,g] · out_slice[g,:]`, with `Φ[n,g] ≥ 0` and
`Σ_g Φ[n,g] = 1` (verified by `basis_rows_partition_of_unity` test, L1304-1334,
holds for both `Sigmoid` and `Softmax` bases). By convexity of the norm:
`‖out[n,:]‖ ≤ max_g ‖out_slice[g,:]‖`, independent of Q̃'s magnitude. Pre-LN on
Q̃ would destroy scale information the ridge solve currently uses, with no
corresponding stability gain (the output is already bounded by the POU).

**Verdict: SUBSUMED.** K-side by α-regularization, Q-side by POU convex
bounding. Benchmark prior: null-to-negative. The Galerkin paper needed pre-LN
precisely because it lacks **both** α-regularization (λ=0) **and** partition-of-
unity structure (identity basis). FUNCATTN has both.

### 3. Energy-decay scale preservation

**Galerkin's purpose.** The scale-preserving property of the pre-LN (trick #2)
lets learnable scale propagate through encoder layers, enabling the model to
learn PDE energy-decay laws (e.g., Burgers' `d‖u‖²/dt = −ν‖∂_x u‖²`).

**FUNCATTN's domain mismatch.** FUNCATTN is a general attention primitive (LLM
token routing, latent-space projection, cross-resolution spectral transport per
Plan 310). It does not solve PDEs or target energy-decay properties. The
energy-decay benefit is PDE-specific and does not transfer.

**Verdict: N/A.** Domain mismatch. Not applicable to FUNCATTN's use cases.

## Why this is closed, not deferred

Per repo rules: "Create issue at ./issues for optimization task, do not create
plan." This issue **is** the optimization analysis. The conclusion is that the
three tricks are each addressed by an existing FUNCATTN mechanism:

| Trick | Galerkin needs it because | FUNCATTN has | Verdict |
|-------|--------------------------|-------------|---------|
| Diagonal-dominant init `δI` | λ=0, no runtime diagonal floor | `α·I_d` at runtime (stronger) | Subsumed |
| Pre-dot-product LN | λ=0, unbounded `‖Q·Kᵀ‖` | α-ridge solve (K-side) + POU convex bound (Q-side) | Subsumed |
| Energy-decay preservation | PDE solver target | Not a PDE primitive | N/A |

This is a structural / mathematical argument, not an empirical guess:

- `λ_min((1-α)·K̃ᵀK̃ + α·I_d) ≥ α` is an eigenvalue identity.
- `‖out[n,:]‖ ≤ max_g ‖out_slice[g,:]‖` follows from Φ being a partition of unity
  (tested at L1304-1334 for both basis variants).
- Energy-decay is a PDE-domain claim; FUNCATTN's domain is attention/routing.

Per the modelless-first mandate, the fix for any future numerical instability in
FUNCATTN is **increasing α** (the runtime diagonal floor) — a one-line config
change, no new primitive, no new feature flag, no training. Re-open this issue
only if empirical evidence shows α-regularization insufficient on a real input
distribution (at which point the fix path is α-tuning, not Galerkin pre-LN).

## Non-goals

- Not modifying `funcattn.rs` — the primitive is shipped, GOAT-passed (G1–G5),
  G6-honestly-failed (LLM-domain), and this analysis shows no modification is
  warranted.
- Not running a benchmark — the prior is null-to-negative by the structural
  argument above. The primitive already passed G3 (sigmoid vs softmax) and the
  verdict on G6 (LLM-domain) is closed.
- Not backfilling the dangling `.issues/033_funcattn_t5_1_eigenbasis_no_benefit.md`
  reference from Plan 286 T5.1 — pre-existing tracking gap, unrelated to this
  issue.

## References

- Research 306: `.research/306_Galerkin_Transformer_FUNCATTN_Grandparent_Predecessor.md`
  — the Gain verdict and trick catalog (§1.3, §2.2).
- Plan 286: `.plans/286_functional_attention_spectral_transport.md` — FUNCATTN
  primitive, all phases COMPLETE.
- Benchmark 058: `.benchmarks/058_funcattn_goat.md` — G1–G6 results.
- `crates/katgpt-core/src/funcattn.rs:450-504` — `solve_convex_combo_dual`
  (the α-regularized ridge solve).
- `crates/katgpt-core/src/funcattn.rs:609-745` — `funcattn_forward` (8-stage
  pipeline; Stage 5 = ridge solve, Stage 8 = POU reconstruction).
- `crates/katgpt-core/src/funcattn.rs:1304-1334` — `basis_rows_partition_of_unity`
  test (verifies the POU property the Q-side bound depends on).

## TL;DR

**CLOSED.** All three Galerkin numerical-stability tricks (diagonal-dominant
init, pre-dot-product LN, energy-decay preservation) are subsumed by FUNCATTN's
existing α-regularization (K-side), partition-of-unity convex bounding (Q-side),
or are domain-mismatched (energy-decay is PDE-specific). No benchmark warranted
— the argument is structural, not empirical. The primitive stays as shipped. If
a future caller hits numerical instability, the fix is increasing α (one-line
config), not adding Galerkin pre-LN.
