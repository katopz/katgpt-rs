# Issue 1251: Constrained (RoPE-commuting) orthogonal Procrustes for katgpt-spectral

**Status:** Open — triggered by katgpt-rs Issue 1250 arm B T6 (the substrate gap check confirmed `procrustes.rs` ships only the unconstrained rotation group; the lawful constrained variant is this issue's deliverable).
**Claimed:** zed-7bab0f17@amo, 1795832900, expires 2026-11-30T02:35:00+00:00
**Date:** 2026-11-28
**Lane:** katgpt-spectral `procrustes.rs` (same module, additive)

## Why the constraint exists

The unconstrained solution `R* = polar(BᵀA)` is a free member of O(d). The
KV-sharing init recipe (Research 698 / Issue 1250 arm B; YASA appendix) rotates
a layer's key space so a consumer layer can share the producer's K/V cache. On
qwen3.8-class full-attention layers nothing may sit between the projection and
the logit dot: `W_k → RMSNorm(gain) → RoPE → dot`. A free orthogonal `R` does
NOT commute with RoPE (RoPE rotates coordinate pair `i` by angle `p·θ_i`, a
different angle per frequency), so applying it pre-RoPE is not equivalent to
applying it post-RoPE and the measured loss delta would be that corruption,
not the recipe (Research 698 §5 applicability condition).

The lawful family: **per-frequency 2×2 block-diagonal rotations**
`R = block_diag(rot(φ_0), …, rot(φ_{d/2−1}))`. Each 2×2 block acts in one
RoPE plane, and planar rotations commute — so `R·RoPE_p = RoPE_p·R` exactly,
at every position `p`, for every frequency set. `d` must be even.

## The deliverable

`rope_commuting_procrustes` in `katgpt-spectral/src/procrustes.rs`, same
shape contract as `orthogonal_procrustes` (anchors `a`,`b` `(n, d)` row-major,
caller-owned `out_rotation` `(d, d)` + `ProcrustesScratch`, `ProcrustesConfig`,
`ProcrustesReport`). Closed form, no iteration:

- The Frobenius objective `‖B − A Rᵀ‖²_F` decouples per frequency pair. For
  pair `i`, with `M_i = (BᵢᵀAᵢ)` the 2×2 cross-covariance of columns
  `(2i, 2i+1)` (row-major `M_i[r][c] = Σ_t b[t,2i+r]·a[t,2i+c]`), the optimal
  planar angle is

  ```
  θ_i = atan2(m₁₀ − m₀₁, m₀₀ + m₁₁)
  ```

  maximizing `trace(M_iᵀ… )` — the classical planar Kabsch angle. `O(n·d)`
  total (cheaper than the unconstrained `O(n·d²)`).
- `config.center` supported via the same rank-1 correction
  (`M_i −= n·μ_a,i ⊗ μ_b,i`).
- Residual path mirrors the unconstrained fold (Issue-940 ordered reductions;
  `dot_f32_ordered` for the uncentered dots; sequential scalar loops
  elsewhere). `det(R) = +1` always (product of planar dets).
- `m_norm` = Frobenius norm of the block-diagonal stack
  `sqrt(Σᵢ ‖M_i‖²_F)` (the analog of the unconstrained `‖M‖_F`).

## Gates (fixture battery, mirror the module's test style)

- [x] G1 planted-recovery: plant per-pair rotations `A → B = R·A`; recovered
      angles within 1e-4; off-block entries EXACTLY zero.
- [x] G2 RoPE-commuting pin: for the planted `R`,
      `R·rot(p·θ_i) == rot(p·θ_i)·R` per pair, bit-exact, at several
      positions/frequencies — the load-bearing invariant.
- [x] G3 coheres with the unconstrained solve: when the truth IS
      block-diagonal, the constrained residual ≤ the unconstrained residual
      + 1e-4 (both should recover it).
- [x] G4 refusals: odd `d` → new `ProcrustesError::OddDim`; degenerate
      (zero-norm) anchors → `DegenerateAnchors`; shape mismatches.
- [x] G5 determinism: same inputs → identical output bits (replay pin).
- [x] G6 centered anchors recover a planted rotation.
- [x] G7 **rotate-half pairing** (LANDED same session, the geometry catch):
      the first real consumer checkpoint (Ternary-Bonsai-2.5-27B, `qwen35`
      arch) rotates only the first `rope.dimension_count = 64` of 256 head
      dims, paired `(i, i + 32)` — the NeoX / llama.cpp rotate-half
      convention (riir-infer `apply_partial_rope_with_freq`, Issue 594), NOT
      adjacent pairs. The adjacent-only form could not express it, so the
      substrate grew the generalization: `rope_commuting_procrustes_paired`
      takes an EXPLICIT pair list (empty → `EmptyPairs`; overlapping
      coordinates → `OverlappingPairs`; out-of-range → `PairOutOfRange`;
      odd `d` legal — unlisted dims stay exactly identity), accumulating
      four scalar cells per pair directly (no d×d M buffer);
      `rope_commuting_procrustes` delegates with the adjacent list (G8 pins
      the delegation bit-exact). The pair list covers partial-RoPE heads:
      the harness composes the paired solve on the rotary dims with
      `orthogonal_procrustes` on the non-rotary dim slice (the issue-text
      "non-RoPE part of a partial-rotary head" form). Commutation is pinned
      BIT-EXACT at the 2×2 block level (two-term sums of commutative
      products — the strongest floating-point form) and to 1e-4 on real
      vectors (sequential application rounds intermediates; bit-exact
      vector commutativity is not a property of f32, the structure test is
      the carrier).
- [x] G9 paired-form refusals + odd-d legality (unlisted dims identity).
- [x] Clippy `-D warnings` at the crate postures (gated + default +
      no-default); bare `rustfmt` on the touched file. 113/113 at the
      `orthogonal_procrustes` posture.

## Non-goals

- No harness / GGUF / forward code here — the measurement arm lives in
  Issue 1250 arm B's harness (riir-infer side, cite-only consumer).
- No general block-structured Procrustes beyond the RoPE plane partition —
  the constraint family is exactly the RoPE-commuting subgroup.
- Not a serving path: load-time weight transform primitive, offline.

## References

- Research 698 (`.research/698_YASA_Tied_Sparse_Attention_KV_Sharing.md`) §1/§5 — the recipe + the applicability condition.
- katgpt-rs Issue 1250 arm B T6 — the trigger + the "never a local copy" routing.
- CLA arXiv:2405.12981 (sharing class), xKV arXiv:2503.18893 (cache-side aligned-SVD prior art).
