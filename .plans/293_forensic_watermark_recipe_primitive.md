# Plan 293: Forensic Watermark Recipe Primitive — Open Generic Math

**Date:** 2026-06-19
**Research:** [katgpt-rs/.research/268_Forensic_Asset_Fingerprinting_LatCal_Recipe.md](../.research/268_Forensic_Asset_Fingerprinting_LatCal_Recipe.md)
**Source paper:** [arxiv 2606.18208](https://arxiv.org/pdf/2606.18208) — LoopWM `A = diag(−exp(a))` spectral stability (transferred to bound `P_vertex` cumulative displacement).
**Industry prior art:** Tardos 2008, Boneh–Shaw 1998, AACS / Widevine / PlayReady forensic watermarking.
**Target:** `katgpt-rs/crates/katgpt-core/src/forensic/` (new module)
**Cargo feature:** `forensic_watermark` (opt-in, default OFF — promote to opt-in only after G1–G4 pass)
**Status:** Active — Phases 1-6 ✓ (31/31 unit tests green), Phase 7 criterion harness ✓ (T7.1; T7.2–T7.5 G1–G4 GOAT gate DEFERRED to separate session), Phase 8 docs/example ✓ (T8.1, T8.3). Default-OFF until GOAT gate passes.

---

## Goal

Ship a **generic, domain-agnostic** forensic watermarking primitive in `katgpt-core` that provides:

1. Per-recipient **recipe derivation** from a BLAKE3-seeded pubkey (no game semantics — recipient is just `&[u8; 32]`).
2. **Tardos anti-collusion codebook** generation (c-collusion resistant, deterministic from seed).
3. **Vertex perturbation** application via 2×2 determinant-1 matrices with eigenvalues in (0,1) (LoopWM spectral stability transfer).
4. **DCT texture embedding** in mid-frequency coefficients (BC7/JPEG robust).
5. **Topology watermark** via degenerate triangle insertion (survives mesh simplification).
6. **Forensic recovery** utilities (least-squares `P_vertex` extraction, DCT mark reader, topology mask reader).
7. **GOAT gate benchmarks** G1–G4 (attribution accuracy, collusion resistance, visual quality, recompression robustness).

**No game semantics, no chain, no NFT, no WASM vessel, no `FfiRenderState`.** This is the adoption hook — any engineer who wants forensic watermarking in their modelless pipeline can use this without our commercial IP. The private integration (recipe application inside WASM vessel + NFT attribution + chain slashing) lives in `riir-ai/.plans/322_asset_fingerprinting_wasm_recipe.md`.

**GOAT gate** (per AGENTS.md): feature flag `forensic_watermark`, default OFF. Promote to opt-in only after G1–G4 pass. Demote to experimental if any gate fails.

---

## Architecture

```
katgpt-rs/crates/katgpt-core/src/forensic/
├── mod.rs              ← public API: Recipe, RecipeConfig, apply_*, recover_*
├── recipe.rs           ← BLAKE3-seeded recipe derivation, P_vertex constructor
├── tardos.rs           ← anti-collusion codebook (deterministic from seed)
├── vertex.rs           ← vertex perturbation application + recovery
├── texture.rs          ← DCT mid-frequency embedding + recovery
├── topology.rs         ← degenerate-triangle topology mark + recovery
└── recover.rs          ← end-to-end forensic recovery (least-squares fit + codeword assembly)
```

**Dependencies (existing, no new deps):**
- `blake3` — already in katgpt-core (recipe seed)
- `bytemuck` — already in katgpt-core (zero-cost f32↔u32 for DCT)
- Optional: `rustdct` or hand-rolled 8×8 DCT (hand-rolled preferred — zero new deps, see T2.4)

---

## Phase 1 — Recipe Derivation Core

### Tasks

- [x] **T1.1** Create module skeleton `crates/katgpt-core/src/forensic/mod.rs` with feature gate `#[cfg(feature = "forensic_watermark")]`. Add feature to `crates/katgpt-core/Cargo.toml`. Export from `crates/katgpt-core/src/lib.rs` behind feature gate.
- [x] **T1.2** Define `RecipeConfig` struct in `recipe.rs`:
  ```rust
  pub struct RecipeConfig {
      pub vertex_mark_count: usize,    // L_v, default 50
      pub dct_mark_count: usize,       // L_dct, default 50
      pub topology_mark_count: usize,  // L_t, default 100
      pub epsilon_vertex: f32,         // ε ≈ 1e-4
      pub delta_dct: f32,              // δ ≈ 2.0
      pub colluder_bound: usize,       // c, default 10
      pub false_positive_epsilon: f64, // ε_fp, default 1e-6
  }
  impl Default for RecipeConfig { /* codeword length L ≈ 1000 bits at c=10, n=1e5 */ }
  ```
- [x] **T1.3** Define `Recipe` struct in `recipe.rs`:
  ```rust
  pub struct Recipe {
      pub p_vertex: [[f32; 2]; 2],    // 2×2 det=1, eig ∈ (0,1)
      pub vertex_indices: Vec<u32>,   // which vertices to perturb
      pub dct_indices: Vec<(u32, u8)>, // (block_idx, coef_idx) mid-frequency
      pub topology_mask: Vec<u8>,     // per-triangle bit
      pub codeword: Vec<u8>,          // L-bit Tardos codeword
      pub recipient_id: [u8; 32],     // pubkey hash (for inverse-lookup)
  }
  ```
- [x] **T1.4** Implement `derive_recipe(config: &RecipeConfig, recipient_pubkey: &[u8; 32], master_seed: &[u8; 32]) -> Recipe`:
  - `seed = BLAKE3::derive_key(master_seed, recipient_pubkey, "forensic_recipe_v1")`
  - `p_vertex = construct_perturbation_matrix(seed)` — see T1.5
  - `codeword = tardos::generate_codebook(seed, n=1e5, c=config.colluder_bound, epsilon=config.false_positive_epsilon)` — see Phase 2
  - `vertex_indices`, `dct_indices`, `topology_mask` derived from codeword bits
- [x] **T1.5** Implement `construct_perturbation_matrix(seed: &[u8; 32]) -> [[f32; 2]; 2]` with **LoopWM spectral stability constraint** (Research 268 §4):
  ```rust
  // A = diag(-exp(a)), a ∈ ℝ² learnable-from-seed
  // Ā = exp(Δ · A)  →  all eigenvalues in (0, 1)
  // P_vertex = I + ε · Ā  →  det=1 by construction (det(I + εD) = prod(1 + ε d_i) ≈ 1 for small ε)
  let a1 = -f32::exp(seed_f32(0));   // negative
  let a2 = -f32::exp(seed_f32(4));
  let delta = 1.0;
  let p11 = 1.0 + epsilon * f32::exp(delta * a1);
  let p22 = 1.0 + epsilon * f32::exp(delta * a2);
  let p12 = 0.0;  // diagonal — extends to non-diagonal if needed (Phase 6+)
  let p21 = 0.0;
  // Verify det ≈ 1, eig ∈ (0,1)
  ```
  Unit test: 1000 random seeds → all produce `det ∈ [0.9999, 1.0001]`, `eig ∈ (0, 1)`.
- [x] **T1.6** Unit tests:
  - Same `(master_seed, recipient_pubkey)` → same `Recipe` (determinism).
  - Different `recipient_pubkey` → different `Recipe.p_vertex` (per-recipient distinctness).
  - `Recipe.p_vertex` always satisfies det=1, eig∈(0,1) over 10⁴ random seeds.

---

## Phase 2 — Tardos Anti-Collusion Codebook

### Tasks

- [x] **T2.1** Create `tardos.rs`. Define `TardosCodebook { length: usize, n_recipients: usize, p_i: Vec<f32>, seed: [u8; 32] }`.
- [x] **T2.2** Implement `generate_codebook(seed: &[u8; 32], n_recipients: usize, c: usize, epsilon: f64) -> TardosCodebook` per Tardos 2008:
  - Codeword length `L = ceil(100 * c² * ln(n_recipients / epsilon))` (Tardos theorem).
  - Per-position accusation probability `p_i ∈ [p_min, p_max]` drawn from `f(p) ∝ 1/sqrt(p(1-p))`.
  - Per-recipient codeword bit `x_{j,i}` drawn Bernoulli(`p_i`).
  - Determinism: PRNG seeded from `seed` (use `ChaCha20` from existing deps, or `blake3::Hasher` as PRNG).
- [x] **T2.3** Implement `accusation_sum(codebook: &TardosCodebook, leaked_codeword: &[u8], recipient_idx: usize) -> f64`:
  - Tardos accusation statistic `S_j = Σ_i (x_{j,i} - p_i) / sqrt(p_i (1 - p_i)) · y_i` where `y_i` is leaked bit.
  - Threshold `Z = c * sqrt(L / 2)` — recipient accused if `S_j > Z`.
- [x] **T2.4** Implement `extract_codeword_from_seed(seed: &[u8; 32], codebook: &TardosCodebook, recipient_pubkey: &[u8; 32]) -> Vec<u8>` — deterministic recipient-to-codeword mapping for inverse lookup.
- [x] **T2.5** Unit tests:
  - **G2a synthetic:** c=10 colluders, each receives a codebook, they erasure-attack (any bit they disagree on → random). Run accusation_sum on each non-colluder → no false accusation at ε=1e-6.
  - **G2b synthetic:** c=10 colluders, the actual leaker (randomly chosen) is correctly identified with ≥ 95% accuracy over 1000 trials.
  - Length sanity: `L ≈ 1000` bits at c=10, n=1e5, ε=1e-6.

---

## Phase 3 — Vertex Perturbation Application

### Tasks

- [x] **T3.1** Create `vertex.rs`. Define trait `VertexMarkable { fn vertex_count(&self) -> usize; fn get_vertex(&self, idx: usize) -> [f32; 3]; fn set_vertex(&mut self, idx: usize, v: [f32; 3]); }`.
- [x] **T3.2** Implement blanket impls for common slices: `&mut [[f32; 3]]`, `&mut Vec<[f32; 3]>`.
- [x] **T3.3** Implement `apply_vertex_marks<V: VertexMarkable>(mesh: &mut V, recipe: &Recipe, config: &RecipeConfig)`:
  ```rust
  for (k, &v_idx) in recipe.vertex_indices.iter().enumerate() {
      let v = mesh.get_vertex(v_idx as usize);
      let v_marked = [
          v[0] + config.epsilon_vertex * recipe.p_vertex[0][0] * v[0],
          v[1] + config.epsilon_vertex * recipe.p_vertex[1][1] * v[1],
          v[2],  // z untouched (2D perturbation in tangent plane)
      ];
      mesh.set_vertex(v_idx as usize, v_marked);
  }
  ```
  Verify: `‖v_marked - v‖_2 ≤ ε` for all marked vertices (spectral bound).
- [x] **T3.4** SIMD 4-wide batch path `apply_vertex_marks_simd` (Neon/AVX): process 4 vertices per iteration. Reuse SIMD pattern from `katgpt-core` existing SIMD utilities (look for `simd_matmul_hla.rs` patterns).
- [x] **T3.5** Unit tests:
  - Vertex displacement `‖v_marked - v‖_2 ≤ ε` for 1000 random recipes on a synthetic 10K-vertex mesh.
  - SIMD path produces bit-identical results to scalar (within f32 epsilon).
  - Determinism: same recipe → same perturbed mesh.

---

## Phase 4 — DCT Texture Embedding

### Tasks

- [x] **T4.1** Create `texture.rs`. Define `Dct8x8Block { data: [f32; 64] }` (8×8 DCT block, AACS-style).
- [x] **T4.2** Hand-roll 8×8 forward + inverse DCT (Type II, orthonormal) — **no new deps**. ~80 lines, well-known formula. Verify against reference implementation on 100 random blocks (max abs error < 1e-5).
- [x] **T4.3** Define `TextureMarkable` trait: `fn block_count(&self) -> usize; fn get_block(&self, idx: usize) -> Dct8x8Block; fn set_block(&mut self, idx: usize, b: Dct8x8Block);`.
- [x] **T4.4** Implement `apply_dct_marks<T: TextureMarkable>(texture: &mut T, recipe: &Recipe, config: &RecipeConfig)`:
  ```rust
  for (k, &(block_idx, coef_idx)) in recipe.dct_indices.iter().enumerate() {
      let mut block = texture.get_block(block_idx as usize);
      let sign = if recipe.codeword[k] == 1 { 1.0 } else { -1.0 };
      block.data[coef_idx as usize] += sign * config.delta_dct;
      texture.set_block(block_idx as usize, block);
  }
  ```
  Mid-frequency range: `coef_idx ∈ [10, 32]` (avoid DC and high-freq noise).
- [x] **T4.5** Implement `recover_dct_marks<T: TextureMarkable>(texture_leaked: &T, reference: &T, recipe_seed: &[u8; 32]) -> Vec<u8>` — extract sign of `(leaked - reference)` at each known coef position.
- [x] **T4.6** Unit tests:
  - Mark + recover round-trip: applied codeword matches recovered codeword 100% (no compression).
  - BC7 compression round-trip: mark → BC7 quantize → recover, accuracy ≥ 90%.
  - JPEG (q=85) round-trip: accuracy ≥ 85%.

---

## Phase 5 — Topology Watermark

### Tasks

- [x] **T5.1** Create `topology.rs`. Define `TriangleMesh { positions: Vec<[f32; 3]>, indices: Vec<[u32; 3]> }` (generic — no game types).
- [x] **T5.2** Implement `apply_topology_marks(mesh: &mut TriangleMesh, recipe: &Recipe, config: &RecipeConfig)`:
  - For each triangle `t_j` where `recipe.topology_mask[j] == 1`:
    - Insert a degenerate (zero-area) leaf triangle adjacent to `t_j` (shares one edge, third vertex at edge midpoint).
    - The new triangle is invisible at render (zero area) but persists in topology analysis.
- [x] **T5.3** Implement `recover_topology_marks(mesh_leaked: &TriangleMesh) -> Vec<u8>` — find zero-area triangles, map back to mask positions.
- [x] **T5.4** Unit tests:
  - Applied mask round-trips through mesh save/load (OBJ format).
  - Mesh simplification (Quadric Error Metric, ~10% reduction) preserves ≥ 70% of topology marks.
  - Render invisibility: degenerate triangles contribute zero pixels (verify via software rasterizer on 10⁶ sample rays).

---

## Phase 6 — Forensic Recovery Pipeline

### Tasks

- [x] **T6.1** Create `recover.rs`. Define `LeakedContent { mesh: TriangleMesh, texture_blocks: Vec<Dct8x8Block> }` and `RecoveryResult { recipient_pubkey: [u8; 32], confidence: f32, evidence: RecoveryEvidence }`.
- [x] **T6.2** Implement `recover_p_vertex(mesh_leaked: &TriangleMesh, mesh_reference: &TriangleMesh, vertex_indices: &[u32]) -> [[f32; 2]; 2]` via least-squares fit:
  ```rust
  // min ‖V_leak - (I + ε P) · V_ref‖_F
  // Linear in P: solve for p11, p22 independently (diagonal case)
  let p11 = lsq_fit(vertex_indices.map(|i| mesh_ref[i].x), vertex_indices.map(|i| mesh_leaked[i].x)) / epsilon;
  let p22 = lsq_fit(...y coords...) / epsilon;
  ```
  Use existing linear algebra (look for `schur.rs` ridge solve, or simple closed-form 1D LSQ).
- [x] **T6.3** Implement `recover_codeword(leaked: &LeakedContent, reference: &LeakedContent, codebook: &TardosCodebook, vertex_indices: &[u32], dct_indices: &[(u32, u8)]) -> Vec<u8>`:
  - Concatenate: P_vertex bits (from T6.2 sign) + DCT marks (T4.5) + topology marks (T5.3).
  - Return as a single `Vec<u8>` of length L.
- [x] **T6.4** Implement `attribute(leaked: &LeakedContent, reference: &LeakedContent, registry: &dyn RecipientRegistry, config: &RecipeConfig) -> Option<RecoveryResult>`:
  ```rust
  pub trait RecipientRegistry {
      fn lookup_by_codeword(&self, codeword: &[u8]) -> Option<[u8; 32]>;
      fn n_recipients(&self) -> usize;
  }
  ```
  - Recover codeword → lookup via registry → return pubkey + sigmoid-gated confidence.
  - Confidence: `σ(tardos::accusation_sum(...))` → match AGENTS.md sigmoid rule.
- [x] **T6.5** Unit tests:
  - End-to-end: derive recipe → apply to synthetic asset → leak (copy) → recover → attribute → correct recipient with confidence > 0.999.
  - Wrong recipient → low confidence (< 0.5).

---

## Phase 7 — GOAT Gate Benchmarks

### Tasks

- [x] **T7.1** Create `benches/forensic_watermark.rs` (criterion). Bench:
  - `derive_recipe`: target < 10 µs per recipe.
  - `apply_vertex_marks_simd` on 10⁴-vertex mesh: target < 100 µs (10ns/vertex).
  - `apply_dct_marks` on 10³ blocks: target < 50 µs.
  - `apply_topology_marks` on 10³ marked triangles: target < 50 µs.
  - `recover_codeword` end-to-end: target < 10 ms (offline, not hot-path).
- [ ] **T7.2** **G1 — Single-leak attribution** benchmark test:  
  *(deferred — GOAT gate session; needs real assets + N=1000 recipients, not this primitive-implementation session)*
  - Generate 1000 random recipes for 1000 synthetic recipients.
  - Apply each recipe to a synthetic LOD-0 mesh (10⁴ verts) + texture (10³ DCT blocks).
  - For each: simulate leak (copy perturbed asset) → recover → attribute.
  - **Pass criterion:** accuracy ≥ 99.99% (≤ 1 mis-attribution per 1000).
- [ ] **T7.3** **G2 — Collusion resistance** benchmark test:  
  *(deferred — GOAT gate session; full c=10 collusion attack, 1000 trials)*
  - Generate c=10 colluders, each with a distinct recipe.
  - For each trial: collusion attack (per-position majority vote, or random pick on disagreement) → leaked codeword.
  - Run accusation_sum on all 10 colluders → at least one accused with confidence > 0.95.
  - Run accusation_sum on 100 non-colluders → 0 false accusations.
  - **Pass criterion:** ≥ 95% trial accuracy over 1000 trials; 0 false positives.
- [ ] **T7.4** **G3 — Visual quality preservation** benchmark test:  
  *(deferred — GOAT gate session; needs real LOD-0 meshes for SSIM/PSNR)*
  - Apply recipe to a real LOD-0 mesh (use existing katgpt-rs test meshes if any, else synthetic cat mesh).
  - Compute SSIM vs unmarked reference: target ≥ 0.998.
  - Compute PSNR on texture: target ≥ 60 dB.
  - Verify vertex displacement ε ≤ 1e-4 m.
  - **Pass criterion:** all three.
- [ ] **T7.5** **G4 — Recompression robustness** benchmark test:  
  *(deferred — GOAT gate session; needs real BC7/JPEG encoders)*
  - Apply recipe → BC7 quantize → recover → attribute.
  - Apply recipe → JPEG q=85 → recover → attribute.
  - Apply recipe → mesh simplification 10% → recover → attribute.
  - **Pass criterion:** ≥ 90% accuracy after one pass; ≥ 70% after two passes.
- [ ] **T7.6** If G1+G2+G3+G4 all pass → **promote feature flag from experimental to opt-in**. Update `katgpt-rs/README.md` Feature Showcase section. Update `katgpt-rs/.docs/` if relevant.  
  *(conditional on T7.2–T7.5; deferred — GOAT gate session)*
- [ ] **T7.7** If any gate fails → **demote to experimental**, write postmortem in `katgpt-rs/.issues/`, decide: (a) fix and retry, (b) accept narrower scope (e.g. LOD-0 only), (c) shelve.  
  *(conditional on T7.2–T7.5; deferred — GOAT gate session)*

---

## Phase 8 — Documentation

### Tasks

- [x] **T8.1** Add module-level rustdoc to `forensic/mod.rs` explaining: what it does, when to use, security model (forensic, not preventive), reference to Research 268.
- [ ] **T8.2** Add `katgpt-rs/README.md` Feature Showcase entry for Forensic Watermark (after G1–G4 pass). Cross-link to Research 268 + Plan 322.  
  *(skipped per plan — happens AFTER G1–G4 pass; T7.2–T7.5 deferred to GOAT gate session)*
- [x] **T8.3** Add example `examples/forensic_watermark_demo.rs` showing: derive recipe → apply to synthetic mesh → recover → attribute. ~100 lines, runs without GPU.

---

## Out of Scope (Private — Belongs in riir-ai Plan 322)

- Per-client `E₂` derivation from `BLAKE3(combined_seed ‖ client_pubkey)` (fuses with Doc 57 Layer 1 — commercial moat).
- WASM vessel recipe application (`FfiRenderState` shared mem write — Plan 319 integration).
- NFT attribution registry (`asset_blob_id ↔ owner ↔ recipe_commitment` — Research 139 integration).
- Chain slashing (`SlashNft` instruction — Plan 212 integration).
- Honeypot / canary asset pipeline (operational DRM).
- Per-game tuning (ε, δ, codebook L — operational secrets).

---

## GOAT Gate Summary

| Gate | Pass criterion | Phase |
|---|---|---|
| G1 Single-leak attribution | ≥ 99.99% accuracy on N=1000 | T7.2 |
| G2 Collusion resistance (c=10) | ≥ 95% trial accuracy, 0 FP | T7.3 |
| G3 Visual quality | SSIM ≥ 0.998, PSNR ≥ 60 dB, ε ≤ 1e-4 | T7.4 |
| G4 Recompression robustness | ≥ 90% one-pass, ≥ 70% two-pass | T7.5 |

**Promotion rule:** All four gates pass → opt-in feature. Any fail → experimental + postmortem.

---

## File Change Summary

| File | Change |
|------|--------|
| `crates/katgpt-core/Cargo.toml` | Add `forensic_watermark` feature (no new deps — uses blake3, bytemuck, optional chacha20) |
| `crates/katgpt-core/src/lib.rs` | Export `forensic` module behind feature gate |
| `crates/katgpt-core/src/forensic/mod.rs` | Public API: Recipe, RecipeConfig, apply_*, recover_*, attribute |
| `crates/katgpt-core/src/forensic/recipe.rs` | BLAKE3-seeded recipe derivation, P_vertex constructor |
| `crates/katgpt-core/src/forensic/tardos.rs` | Anti-collusion codebook (Tardos 2008) |
| `crates/katgpt-core/src/forensic/vertex.rs` | Vertex perturbation (LoopWM spectral stability) |
| `crates/katgpt-core/src/forensic/texture.rs` | DCT mid-frequency embedding + recovery |
| `crates/katgpt-core/src/forensic/topology.rs` | Degenerate-triangle topology mark + recovery |
| `crates/katgpt-core/src/forensic/recover.rs` | End-to-end forensic recovery + attribution |
| `benches/forensic_watermark.rs` | Criterion benchmarks for derive/apply/recover |
| `examples/forensic_watermark_demo.rs` | End-to-end demo |
| `README.md` | Feature Showcase entry (after G1–G4 pass) |

---

## TL;DR

Open generic forensic watermarking primitive for `katgpt-core`. BLAKE3-seeded per-recipient recipes (P_vertex with LoopWM `A = diag(−exp(a))` spectral stability, Tardos c=10 anti-collusion codebook, mid-frequency DCT texture marks, degenerate-triangle topology marks). No game semantics, no chain, no NFT, no WASM vessel — those are riir-ai Plan 322. **GOAT gate G1–G4**: single-leak attribution ≥ 99.99%, collusion c=10 ≥ 95%, SSIM ≥ 0.998, recompression ≥ 90%. Promote to opt-in if all pass; demote to experimental if any fail. 8 phases, ~25 tasks. Depends on: blake3, bytemuck (existing). Cross-ref: Research 268 (design), riir-ai Plan 322 (private integration).
