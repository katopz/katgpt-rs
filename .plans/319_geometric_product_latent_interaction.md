# Plan 319: Channel-Wise Geometric Product — Latent Interaction Primitive

**Date:** 2026-06-25
**Research:** [katgpt-rs/.research/299_Clifford_Geometric_Product_Latent_Interaction.md](../.research/299_Clifford_Geometric_Product_Latent_Interaction.md)
**Source paper:** [arXiv:2601.06793](https://arxiv.org/abs/2601.06793) — CliffordNet: All You Need is Geometric Algebra (Ji, Feb 2026)
**Target:** `katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs` (new module) + Cargo feature `geometric_product`
**Status:** Active — Phase 1 (skeleton)

---

## Goal

Ship the **channel-wise geometric product** `uv = u·v + u∧v` as a modelless, zero-allocation latent-interaction primitive behind the `geometric_product` feature flag. The primitive produces two output vectors per call: a **coherence** term (Hadamard + SiLU, the familiar dot-product-like signal) and a **structure** term (anti-symmetric wedge via cyclic shifts — the bivector signal currently missing from every latent op in the codebase). Run the GOAT gate to prove the wedge carries information the dot product misses. **If G1 passes** (wedge signal is not redundant with dot product on a representative latent substrate), the primitive promotes toward default and unlocks the riir-ai / riir-neuron-db fusion guides (deferred per Research 299).

**Why modelless:** the primitive is Hadamard + cyclic shift + subtract + sigmoid. No backprop, no training, no learned projection. The paper's trained projection `P` and backbone architecture are out of scope (→ riir-train if ever needed); we ship only the deterministic math op.

**Why `linalg/`:** the geometric product is a generic linear-algebra primitive (two vectors in, two vectors out). It has no game/chain/shard semantics. `linalg/` already houses `ridge_solve.rs`; the geometric product is a peer.

---

## Phase 1 — Unblocking Skeleton (CORE)

### Tasks

- [ ] **T1.1** Add `geometric_product` feature to `katgpt-rs/crates/katgpt-core/Cargo.toml` (empty deps, opt-in).
- [ ] **T1.2** Create `katgpt-rs/crates/katgpt-core/src/linalg/geometric_product.rs` with:
  - `pub fn cyclic_shift_into(src: &[f32], dim: usize, shift: usize, out: &mut [f32])` — zero-alloc cyclic channel shift `T_s`. Handles wrap-around. Documented with the anti-symmetric sign caveat (Research 299 §5 Q4).
  - `pub fn geometric_product_into(u, v, dim, shifts, dot_out, wedge_out, scratch_u, scratch_v)` — accumulates `Σ_s SiLU(u ⊙ T_s(v))` into `dot_out` and `Σ_s (u ⊙ T_s(v) − T_s(u) ⊙ v)` into `wedge_out`. Zero alloc after scratch init.
  - SIMD chunking hint (4-wide) on inner channel loop, mirroring `dec/operators.rs::exterior_derivative_into` pattern.
- [ ] **T1.3** Gate the module behind `#[cfg(feature = "geometric_product")]` and re-export from `linalg/mod.rs`.
- [ ] **T1.4** Unit tests (same file, `#[cfg(test)]`):
  - `wedge_is_antisymmetric`: `geometric_product_into(u, v, ...) == -geometric_product_into(v, u, ...)` on the wedge output.
  - `wedge_self_is_zero`: `u ∧ u = 0` (anti-symmetry implies `x∧x=0`).
  - `dot_is_symmetric`: `u·v == v·u` on the dot output.
  - `shift_zero_is_hadamard`: with `shifts = &[0]`, `dot_out[c] = SiLU(u[c]·v[c])` and `wedge_out[c] = 0` (since `u_c v_c − u_c v_c = 0`).
  - `shift_s_extracts_diagonal`: with `shifts = &[s]`, `wedge_out[c] = u[c]·v[(c+s)%dim] − u[(c+s)%dim]·v[c]` — matches paper Eq. 11.
- [ ] **T1.5** `cargo test -p katgpt-core --features geometric_product --lib` passes.

---

## Phase 2 — GOAT Gate (Prove the Wedge Carries Orthogonal Info)

The core question from Research 299 §5 Q1: **does the wedge signal carry information that the dot product misses on a representative latent substrate?** If yes → strong GOAT, promote. If no → demote to opt-in curiosity, do not promote.

### G1 — Orthogonal Information (correctness/quality gate)

- [ ] **T2.1** Construct a synthetic latent-pair dataset where dot product and wedge disagree by construction:
  - **Class A (coherent pairs):** `v = u + small_noise`. High dot product, low wedge.
  - **Class B (orthogonal pairs):** `v` = random unit vector orthogonal to `u`. Near-zero dot product, high wedge magnitude.
  - **Class C (anti-correlated pairs):** `v = -u + small_noise`. Strongly negative dot product, low wedge.
  - **Class D (rotated pairs):** `v = R_θ · u` for θ ∈ (10°, 80°). Moderate dot product (`cos θ`), moderate wedge (`sin θ`).
  - D = 8 (HLA-sized), D = 64 (shard-sized), 1000 pairs per class.
- [ ] **T2.2** Compute `dot_score = Σ dot_out` and `wedge_score = Σ |wedge_out|` per pair for `shifts = &[1, 2, 4]` (D=8) or `&[1, 2, 4, 8, 16, 32]` (D=64).
- [ ] **T2.3** **G1 pass criterion:** a linear classifier on `[dot_score, wedge_score]` achieves ≥ 95% accuracy on the 4-class problem, AND `wedge_score` alone achieves ≥ 75% on Class B vs Class A (where dot product is uninformative). If wedge is redundant with dot → G1 FAILS, the primitive carries no new signal.
- [ ] **T2.4** Document the G1 result in `katgpt-rs/.benchmarks/319_geometric_product_goat.md`.

### G2 — Rotational Recovery (the wedge's reason to exist)

- [ ] **T2.5** Construct 1000 pairs where `v = R_θ · u` for θ uniformly in [0°, 180°]. Compute the correlation between `wedge_score` and `sin(θ)` (the true rotational component). **G2 pass criterion:** Pearson correlation ≥ 0.9. This proves the wedge recovers the rotational angle the dot product collapses (`dot = cos θ`, losing the sign and the orthogonal magnitude).

### G3 — No Regression

- [ ] **T2.6** `cargo check --all-features` clean (no combo-only breakage, per the `merkle_root` lesson).
- [ ] **T2.7** `cargo check --no-default-features` clean (zero-dep baseline intact).
- [ ] **T2.8** No allocation in hot path: the benchmark (`G4` below) must show zero `Vec` allocations per call (use `#[cfg(feature = "alloc_tracker")]` or a manual drop-check).

### G4 — Performance

- [ ] **T2.9** Add `benches/geometric_product_bench.rs` (criterion):
  - `geometric_product_D8_S3` — HLA-sized (D=8, |S|=3). Target: < 50 ns/call (sub-microsecond, plasma tier).
  - `geometric_product_D64_S6` — shard-sized (D=64, |S|=6). Target: < 200 ns/call (plasma tier).
  - Compare vs naive `O(D²)` full wedge (the upper bound). Sparse rolling must be ≥ 4× faster at D=64.
- [ ] **T2.10** Document G4 result in `katgpt-rs/.benchmarks/319_geometric_product_goat.md`.

---

## Phase 3 — Promotion Decision

- [ ] **T3.1** If G1 AND G2 pass: the wedge carries orthogonal info (G1) and recovers rotation (G2). This is a genuine GOAT — the primitive adds a new signal dimension. **Promote `geometric_product` to default-on** after a 1-week bake behind opt-in. Create the riir-ai + riir-neuron-db fusion guides (Research 299 §2 Fusion #2 and #4) and elevate Research 299 to Super-GOAT.
- [ ] **T3.2** If G1 FAILS (wedge redundant with dot on this substrate): keep opt-in, document the null result, do NOT promote. The primitive still ships for experimentation but is not a GOAT. Demote Research 299 verdict to "Gain" in the note.
- [ ] **T3.3** If G1 passes but G2 fails (wedge is informative but not specifically rotational): investigate what the wedge IS capturing before promoting. May still be a GOAT for a different reason than expected.

---

## Phase 4 — Fusion Hooks (deferred until Phase 3 promotion)

Only execute if Phase 3 promotes to default. These land in the PRIVATE repos and create the Super-GOAT guides.

- [ ] **T4.1** `riir-ai/.research/155_clifford_wedge_npc_emotional_complementarity_guide.md` — HLA fusion selling point (formation-quality scoring via `h_NPC1 ∧ h_NPC2`).
- [ ] **T4.2** `riir-neuron-db/.research/007_shard_structural_retrieval_guide.md` — shard retrieval selling point (manifold-spanning ensemble selection via `∧`).
- [ ] **T4.3** Wire `geometric_product_into` into the HLA evolve path (riir-engine `hla/`) as an opt-in complementarity signal alongside the existing dot-product projection. **Respect the raw-vs-latent boundary**: the wedge operates on HLA latents locally; only the resulting scalar (complementarity score) crosses the sync boundary.
- [ ] **T4.4** Wire into NeuronShard retrieval (riir-neuron-db `index.rs`) as an opt-in `retrieve_diverse(k)` that maximizes total wedge span instead of dot-product similarity.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| G1 fails — wedge redundant with dot on HLA/shard substrate | Medium | Primitive still ships opt-in for experimentation. Research 299 demoted to Gain. No promotion. |
| Anti-symmetric wrap-around sign corrupts wedge at low D | Medium | T1.4 `shift_s_extracts_diagonal` test catches this. If sign corruption is systematic, use zero-padded (non-wrapping) shifts instead of cyclic. Document the choice. |
| Shift set S not expressive enough at D=8 (HLA) | Low | G1 uses `&[1,2,4]` which covers all 7 non-trivial shifts mod 8. If G1 fails, try exhaustive `&[1,2,3,4,5,6,7]`. |
| Wedge magnitude scale mismatch with dot in the GGR gate | Low | Sigmoid gate absorbs scale. G1 uses raw `Σ` scores; if scale is an issue, normalize wedge by `1/|S|` before comparison. |
| SIMD auto-vectorization doesn't trigger on the inner loop | Low | Mirror the explicit 4-wide chunking in `dec/operators.rs::exterior_derivative_into` (T1.2). G4 bench will reveal if vectorization landed. |
| Fusion guides (T4.1/T4.2) created before G1 passes | High | **Hard block**: T4.x tasks are gated on T3.1 promotion. Do NOT create riir-ai/riir-neuron-db guides until the GOAT gate passes — per skill rule, no "Super-GOAT candidate" escape hatch. |

---

## GOAT Gate Summary

| Gate | Criterion | Target |
|------|-----------|--------|
| **G1** | Wedge carries info dot misses (4-class linear separability) | ≥ 95% acc on `[dot, wedge]`; ≥ 75% on wedge-only Class B vs A |
| **G2** | Wedge recovers rotational angle | Pearson(wedge_score, sin θ) ≥ 0.9 |
| **G3** | No regression | `--all-features` + `--no-default-features` clean; zero alloc in hot path |
| **G4** | Performance | D=8 < 50ns; D=64 < 200ns; ≥ 4× faster than O(D²) naive |

**Promotion rule (AGENTS.md):** G1 + G2 + G3 + G4 all pass AND gain is modelless → promote `geometric_product` to default. Then create riir-ai + riir-neuron-db fusion guides (T4.1, T4.2) and elevate Research 299 to Super-GOAT.

**Demotion rule:** if G1 fails, keep opt-in, document null result, demote Research 299 to Gain.

---

## References

- Source paper: https://arxiv.org/abs/2601.06793 (CliffordNet, Ji 2026)
- Research note: `katgpt-rs/.research/299_Clifford_Geometric_Product_Latent_Interaction.md`
- Closest shipped cousin (spatial, NOT channel): `katgpt-rs/crates/katgpt-core/src/dec/operators.rs::exterior_derivative` (Plan 251)
- Closest shipped cousin (orthogonal construction, NOT interaction): RotorQuant (Research 65, Plan 100)
- Closest shipped cousin (batch cross-product, NOT per-point): Latent Functor rank-k (Plan 318)
- Canonical plan example: `katgpt-rs/.plans/271_attention_matching_compaction.md`
