# Proposal 009: Canonical Intent Space — Plug-and-Play Any Base Model

**Date:** 2026-07-25
**Research:** [katgpt-rs/.research/459_canonical_intent_space_plug_and_play.md](../.research/459_canonical_intent_space_plug_and_play.md)
**Training-side counterpart:** [riir-train/.research/406_git_rebasin_universal_subspace.md](../../riir-train/.research/406_git_rebasin_universal_subspace.md)
**Source papers:**
- Git Re-Basin (arxiv 2209.04836) — permutation alignment
- Universal Weight Subspace Hypothesis (arxiv 2512.05117) — shared spectral subspaces
- Lottery Ticket Hypothesis (arxiv 1803.03635) — sparse winning subnetworks

**Status:** Proposal — **P2 RAN 2026-07-26, G5 FAILED (Bench 422) for square Procrustes; P1 RAN 2026-07-26, G5 PASSED (Bench 423) for joint-SVD SubspaceAdapter at k ∈ {2,4}; P3 RAN 2026-07-26, G6a FAILED (Bench 424) for modelless centroid construction; P3b RAN 2026-07-26 (Bench 425) layer 0 best — Git Re-Basin contradicted; P3c RAN 2026-07-26 (Bench 426) — expanded corpus + length-detrending: d_diff agreement stuck at +0.48 (not a noise issue), length detrending REVERSES Python discrimination (the apparent "Rust-idiom" signal was substantially a prompt-length artifact); Recipe D RAN 2026-07-27 (Bench 427) — length-matched contrastive direction across k ∈ {2,4,8,16}: cross-arch agreement never crosses +0.01 (best +0.009 at k=16, threshold ≥ 0.5), length-detrend PASSES for all k (corpus construction is sound — failure is structural, not length).** Cross-arch modelless canonical direction **PERMANENTLY EXHAUSTED** — four converging failure lines (cross-arch agreement ceiling, length confound, asymmetric prose margin, structural cross-arch disagreement after length control). P1 G5 still holds (shared subspace preserves pairwise alignment — a real result about cross-model covariance, independent of canonical direction existence). Super-GOAT cross-arch claim **PERMANENTLY DEMOTED** — moved from "demoted (modelless exhausted)" to "permanently demoted (hidden-state construction exhausted)." Reopens only on a **non-hidden-state construction** (AST/clippy/ownership-graph features) — NOT on Recipe E (gradient descent): the failure pattern is cross-arch disagreement, not non-linearity, so a richer parameterization cannot help. Intra-arch claim unaffected. See [riir-train/.benchmarks/427_canon_p4_recipe_d_length_matched.md](../../riir-train/.benchmarks/427_canon_p4_recipe_d_length_matched.md). Plan remains unopened.
**Target repos:** `katgpt-rs` (primary, public) + `riir-train` (secondary, private)
**User constraint:** "this should be in katgpt-rs and riir-train as possible bc other riir-* is focus on game" — no files in riir-ai/chain/neuron-db for this work.

---

## Goal

Ship a **canonical intent space** primitive in `katgpt-rs` that lets any frozen base model (Gemma, MiniCPM5, Llama, Qwen) consume the same direction-vector overlays (Rust idioms, NPC personality, emotion vectors, style) without retraining. Each base model carries a deterministic `ModelAdapter` projecting canonical directions into its specific latent space.

This generalizes the existing use case ("we loaded Gemma in our latent space and play game") to "we loaded **any model** in our latent space."

The design fuses three foundational papers:
- **Git Re-Basin** — same-architecture model alignment via permutation symmetries
- **Universal Weight Subspace** — empirical evidence for cross-model shared spectral subspaces
- **Lottery Ticket** — sparse mask transfer across aligned models

---

## Architecture

```text
                    ┌─────────────────────────────────────────┐
                    │  Canonical Intent Space                 │
                    │  (architecture-neutral, owned by katgpt)│
                    │                                         │
                    │  d_Rust_idiom, d_curiosity, d_valence,  │
                    │  d_NPC_personality, d_style, ...        │
                    └────────────────┬────────────────────────┘
                                     │
                ┌────────────────────┼────────────────────┐
                ▼                    ▼                    ▼
        ProcrustesAdapter    SubspaceAdapter       MaskAdapter
        (same-arch swap)     (cross-arch joint)    (lottery ticket)
        substrate:           extends:              substrate:
        procrustes.rs        spectral_rewire.rs    spectral_flatness.rs
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     ▼
                          model_specific_latent
                                     │
                                     ▼
                       frozen_base_model.decode()
                                     │
                                     ▼
                              tokens / actions
```

**Functor semantics (user's framing):** `F: CanonicalIntent × ModelAdapter → ModelSpecificLatent`. Linear adapters (Procrustes, Subspace) preserve canonical-space operations (sum, scale, sigmoid-gate). The mask adapter is elementwise, also commuting through linear ops.

---

## Open primitive spec (katgpt-rs)

### Location

New module `crates/katgpt-core/src/canon/` (sibling of `sense/`, `dec/`, `closure/`). Not a new crate — follows existing substrate-module pattern. If the surface grows >1000 LOC, split to `crates/katgpt-canon/` (deferred until needed).

### Surface (~400 LOC P0)

```rust
// crates/katgpt-core/src/canon/mod.rs

/// Architecture-neutral intent direction.
/// Unit-norm f32 vector + BLAKE3 tag for sync/commit.
#[derive(Clone, Debug)]
pub struct CanonicalIntent {
    pub tag: [u8; 32],        // BLAKE3 of label
    pub direction: Vec<f32>,  // unit-norm in canonical space
}

impl CanonicalIntent {
    pub fn new(label: &str, direction: Vec<f32>) -> Self { /* normalize + blake3 */ }
    pub fn dim(&self) -> usize { self.direction.len() }
    pub fn dot(&self, other: &CanonicalIntent) -> f32 { /* cosine since unit */ }
}

/// Projects a canonical intent into a specific base model's latent space.
/// Modelless: zero training, deterministic given adapter state.
pub trait ModelAdapter: Send + Sync {
    /// Apply adapter; write into `out` (len = target_dim).
    /// Zero-alloc hot path: caller-owned buffer.
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]);

    /// Inverse projection for diagnostics ("what intent is this latent expressing?").
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32>;

    fn target_dim(&self) -> usize;

    /// BLAKE3 of adapter state — for freeze/thaw attestation + cross-node verify.
    fn commitment(&self) -> [u8; 32];
}

// crates/katgpt-core/src/canon/procrustes_adapter.rs
pub struct ProcrustesAdapter {
    rotation: Vec<f32>,  // row-major d×d, from procrustes.rs
    target_dim: usize,
    commitment: [u8; 32],
}

// crates/katgpt-core/src/canon/subspace_adapter.rs
pub struct SubspaceAdapter {
    basis: Vec<f32>,     // row-major d_target × d_canonical, top-k SVD
    target_dim: usize,
    commitment: [u8; 32],
}

// crates/katgpt-core/src/canon/mask_adapter.rs
pub struct MaskAdapter {
    mask: Vec<u32>,      // bit-packed
    target_dim: usize,
    commitment: [u8; 32],
}
```

### Feature gates

- `canon` (P0, opt-in) — `CanonicalIntent` + `ModelAdapter` trait + `ProcrustesAdapter`
- `canon_subspace` (P1, opt-in) — `SubspaceAdapter` (implies `spectral_rewire`)
- `canon_mask` (P4, opt-in) — `MaskAdapter`

Promotion to default-on gated on G1–G6 (§GOAT gate below).

### Tests

- G1 correctness: `project_into` preserves ranking — canonical directions with higher dot-product produce model-latent directions with higher dot-product (Pearson > 0.95 on synthetic).
- G2 perf: `project_into` < 50µs at d=64 on Apple Silicon (criterion bench).
- G4 alloc-free: `project_into` does zero heap allocations after adapter construction (asserted via `assert_no_alloc` in debug builds).
- Determinism (G3 foundation): bit-identical output across x86_64/aarch64/wasm32 (mirrors `procrustes_determinism.rs`).

---

## Training-side counterpart (riir-train)

Lives in `riir-train/.research/406_git_rebasin_universal_subspace.md`. Three training-only pieces:

1. **STE permutation discovery** — Git Re-Basin's straight-through estimator. Only needed if activation/weight matching (which are modelless) underperform.
2. **Iterative Magnitude Pruning (IMP)** — lottery ticket mask discovery. Feeds `MaskAdapter`.
3. **Stitching layer fallback** — if P2 G5 fails (Procrustes loses too much cross-arch), train a small stitching layer per model pair. The adapter then becomes `TrainedStitchingAdapter` (new variant, riir-train hosts training).

Activation collection for activation matching also lives in riir-train (run N models on a shared prompt set, dump activation matrices).

---

## GOAT gate

Per Research 459 §6. The two star gates decide Super-GOAT vs GOAT:

| Gate | Floor | Target | Decision |
|---|---|---|---|
| G1 correctness | baseline + sysprompt | canonical steering ≥ floor on Rust eval | required to ship at all |
| G2 perf | baseline latency | < 50µs project_into | required |
| G3 no-regression | baseline on MMLU-lite | no capability loss | required |
| G4 alloc-free | hot path | 0 alloc after construction | required |
| **G5 cross-model preservation** | cosine sim floor 0.5 | **> 0.7 on held-out prompts across N≥2 same-arch models** | **GO: Super-GOAT path. NO-GO: fall back to trained stitching (riir-train).** **P2 RESULT (2026-07-26, Bench 422): G5 FAILED for square Procrustes + random projection.** Mean cos = -0.27 (proj_dim=16) and -0.08 (proj_dim=64) on Gemma2-2B ↔ MiniCPM5-1B held-out. Three structural blockers surfaced: (1) dim mismatch 2304 vs 1536, (2) O(d³) Newton-Schulz infeasible at d=1536, (3) underdetermined with n=40 ≪ 2·d. **P1 RESULT (2026-07-26, Bench 423): G5 PASSED for joint-SVD SubspaceAdapter.** Same models, same corpus, same n_train/n_test. Replaced random projection with joint SVD: top-k right singular vectors of M=[A\|B] define the shared subspace. Mean cos = +0.87 (k=2), +0.75 (k=4), +0.68 (k=8), +0.64 (k=16). GO at k ∈ {2, 4}; the cross-arch shared subspace is genuinely low-dimensional. The P2 negative cosine was an artifact of random projection, not a property of the models — refuted. Cross-arch path restored modellessly; Recipe C (trained stitching) no longer a blocker. |
| **G6 cross-architecture gain** | baseline Llama + Rust sysprompt | canonical steering transferred Gemma → Llama beats floor on Rust eval | **GO: cross-arch Super-GOAT. NO-GO: demote to intra-arch GOAT (still ships).** **G6a RESULT (2026-07-26, Bench 424): G6a FAILS for the modelless centroid construction.** Centroid agreement after Procrustes = −0.33 (per-model train centroids in shared subspace point in opposite directions even after rotation — Procrustes aligns shape, not location). Difference-of-means construction shows partial signal (+0.46 cross-arch agreement, below 0.5 threshold; per-model Rust-vs-Python margins +0.08 / +0.14 — both positive but asymmetric). JS discrimination negative on both models (−0.32 / −0.03) — the centroid captures a token-count confound, not Rust-idiom signal. **P3b RESULT (2026-07-26, Bench 425): layer 0 discriminates best, Git Re-Basin contradicted.** **P3c RESULT (2026-07-26, Bench 426): three converging failure lines — (1) d_diff agreement ceiling at +0.48 despite 3× more Python data, (2) length detrending REVERSES Python discrimination (+0.19 → −0.15 — the apparent Rust-idiom signal was prompt length), (3) prose margin asymmetric (MiniCPM −0.29, wrong sign). Modelless path declared exhausted.** **Recipe D RESULT (2026-07-27, Bench 427): length-matched corpus controls length at construction time — detrend PASSES for all k (direction is genuinely length-independent). But cross-arch agreement never crosses +0.01 across k ∈ {2,4,8,16} (best +0.009 at k=16, threshold ≥ 0.5). The failure is STRUCTURAL cross-arch disagreement, not length, not noise.** Cross-arch Super-GOAT claim **PERMANENTLY DEMOTED** — moved from "demoted (modelless exhausted)" to "permanently demoted (hidden-state construction exhausted)." Recipe E (gradient descent) NOT opened — failure pattern (cross-arch disagreement, not non-linearity) rules it out. Reopens only on a non-hidden-state construction (AST/clippy/ownership-graph features). |

Per "Report the Floor" rule (Research 322), G1/G6 floor is **good system prompt**, not "no system prompt". Most style gains evaporate against a well-crafted prompt — G6 is the honesty gate.

---

## Phases

### P0 — Skeleton + ProcrustesAdapter (katgpt-rs)
- [x] T1.1 Create `crates/katgpt-canon/src/intent.rs` with `CanonicalIntent` + `ModelAdapter` trait (DONE 2026-07-26)
- [x] T1.2 Implement `ProcrustesAdapter` wrapping `katgpt_spectral::procrustes::orthogonal_procrustes` (DONE 2026-07-26)
- [x] T1.3 G1/G2/G4 tests on synthetic canonical directions (DONE 2026-07-26 — 13 unit tests under `canon` feature, 26 total under `canon_subspace,canon_mask`)
- [x] T1.4 Feature flag `canon`, default-off (DONE 2026-07-26)
- [ ] T1.5 Determinism test across x86_64/aarch64/wasm32 (deferred — would need cross-platform CI; the substrate is BLAKE3 + SVD which are already cross-platform-bit-identical per their own gates)

**P0 layering deviation from original spec (2026-07-26):** the spec at L67 said "New module `crates/katgpt-core/src/canon/`". The substrate split since the proposal was written — `orthogonal_procrustes` lives in `katgpt-spectral` (which depends on katgpt-core), and `thin_svd_into` lives in katgpt-core. Putting canon in katgpt-core would require either (a) a dep cycle (katgpt-core → katgpt-spectral → katgpt-core), (b) moving orthogonal_procrustes to katgpt-core (huge refactor), or (c) putting canon in katgpt-spectral (which is `publish = false`, breaking the open-source claim). Per the proposal's own anticipation ("If the surface grows >1000 LOC, split to `crates/katgpt-canon/`"), we created `crates/katgpt-canon/` from the start — it depends on both katgpt-core (for SVD) and katgpt-spectral (for Procrustes), no cycle, matches the Issue 007 crate-split pattern, and is publishable to crates.io.

**P0 ships 3 adapters (canon, canon_subspace, canon_mask) in one crate** — the original P0/P1/P4 phasing was sequential, but since the substrate is the same and the adapters share the `ModelAdapter` trait + `CanonicalIntent` type, all three ship together. The SubspaceAdapter carries the load-bearing P1 result (Bench 423 G5 GO at k∈{2,4}); the MaskAdapter ships modelless mask APPLICATION only (discovery stays in riir-train per Research 459 §1.3).

### P1 — SubspaceAdapter (katgpt-rs)
- [x] T2.1 Implement joint SVD across N models (extends `spectral_rewire.rs`) — DONE 2026-07-26 as `fit_joint_svd_pair` in `crates/katgpt-canon/src/subspace_adapter.rs`
- [x] T2.2 G1/G2/G4 tests on single-model subspace — DONE 2026-07-26 (5 unit tests + 2 integration tests on planted shared subspace)
- [x] T2.3 Feature flag `canon_subspace` — DONE 2026-07-26

### P2 — Cross-model validation (the make-or-break)
- [ ] T3.1 Load Gemma-2-2B + MiniCPM5-1B (both at `riir-train/data/*.gguf`)
- [ ] T3.2 Collect hidden states on 50-prompt Rust code snippet set
- [ ] T3.3 Fit Procrustes R: gemma_hidden ↔ minicpm_hidden
- [ ] T3.4 **G5: measure cos(R · h_gemma, h_minicpm) on held-out prompts. Decision point.**
- [ ] T3.5 If G5 fails → open issue for stitching-layer fallback in riir-train, demote to GOAT

### P3 — Rust-style canonical direction + G6 (the real test)
- [x] T4.1 Construct `d_Rust_idiom` canonical direction (modelless centroid + difference-of-means, Bench 424)
- [ ] T4.2 G6a: measure cross-arch discrimination of canonical direction on Rust vs non-Rust (Bench 424 — **FAIL for centroid, PARTIAL for d_diff**; Bench 426 — **d_diff also FAILS after length detrending**)
- [ ] T4.2a Intermediate-layer probe (layers 6/12/18 of 24-26) — highest-value next experiment per Git Re-Basin. **P3b RAN 2026-07-26 (Bench 425): Git Re-Basin hypothesis CONTRADICTED — layer 0 discriminates best (+0.19 Python margin), not middle layers (+0.06-0.14); monotonic decrease layer 0→25. The centroid captures surface/lexical features, not semantic Rust-idiom-ness. Cross-arch layer-0 probe still worth running (would need `forward_llama_trace` substrate).**
- [ ] T4.2a.1 Cross-arch layer-0 probe — add `forward_llama_trace` to riir-engine (~150 LOC mirroring Gemma variant) and re-run P3 G6a at layer 0 for both models
- [-] T4.2b Length-normalized projections (address the JS token-count confound). **P3c RAN 2026-07-26 (Bench 426): length detrending REVERSES Python discrimination (Gemma +0.19 → −0.15). The apparent d_diff discrimination was substantially a prompt-length artifact, not a Rust-idiom signal. This is the load-bearing negative for the modelless path.**
- [-] T4.2c Larger contrastive corpus for d_diff (30-50 Python prompts vs current 10). **P3c RAN 2026-07-26 (Bench 426): expanding Python 10 → 30 barely moves d_diff agreement (+0.4645 → +0.4755, +0.011 gain). The +0.46 ceiling is fundamental cross-arch disagreement, not noise.**
- [-] T4.2d Modelless path declared exhausted (P3 + P3b + P3c converge). Cross-arch canonical direction reopens only on riir-train Recipe C or non-hidden-state construction. **Recipe D RAN 2026-07-27 (Bench 427): the riir-train Recipe C/D deferral is CLOSED — Recipe D ran with length-matched corpus + k ∈ {2,4,8,16} sweep, all four k fail the cross-arch gate (best +0.009). Length-matching works (detrend passes) but the failure is structural cross-arch disagreement. Cross-arch Super-GOAT PERMANENTLY DEMOTED — reopens only on non-hidden-state construction (AST/clippy/ownership-graph). Recipe E (gradient descent) NOT opened — failure pattern rules it out.**
- [ ] T4.3 G6b: steer Gemma and MiniCPM via the same canonical direction; measure Rust eval delta vs sysprompt floor (REQUIRES `forward_llama_with_embedding` substrate in riir-engine — deferred until G6a passes)
- [ ] T4.4 If G6 fails → keep as intra-arch GOAT, narrow the selling point, demote verdict

### P4 — MaskAdapter (auxiliary, gated on P3)
- [ ] T5.1 IMP mask discovery in riir-train (lottery ticket) — DEFERRED (training-side)
- [x] T5.2 `MaskAdapter` impl in katgpt-rs (elementwise apply, modelless) — DONE 2026-07-26 in `crates/katgpt-canon/src/mask_adapter.rs` (6 unit tests)
- [ ] T5.3 Test mask transfer across Procrustes-aligned models — DEFERRED (would need composition-with-Procrustes helper, not yet shipped)

### P5 — Promotion / demotion
- [ ] T6.1 If G1–G6 all pass → promote `canon` to default-on; write benchmark note in `.benchmarks/`
- [x] T6.2 If G5/G6 fail → keep opt-in, document scope limit, ship as intra-arch GOAT — **DONE 2026-07-28 (Bench 562).** G1/G2/G4 gates measured and PASS for all three adapters (ProcrustesAdapter + SubspaceAdapter + MaskAdapter). The substrate carries a measured GOAT stamp. Features stay opt-in (default-off) because the cross-arch Super-GOAT headline is permanently demoted (Bench 427); promotion to default-on would require a new proposal re-arguing the substrate's value proposition post-demotion. **Known limitation:** ProcrustesAdapter `project_into` at d=2304 is 3.9ms (O(d²) scaling — not gated against 50µs; the d=256 hot-path gate passes at 29µs). See `Bench 562` for the full gate matrix.

---

## Scope discipline

**What this proposal does NOT do:**

- Does not modify riir-ai/chain/neuron-db (user explicit override — those repos are game/chain/shard-focused).
- Does not change the existing `latent_functor/procrustes_bridge.rs` in riir-ai (that's the game-side consumer; it can adopt the new `canon` module later if beneficial, but that's a separate plan).
- Does not implement STE permutation discovery in katgpt-rs (training → riir-train).
- Does not implement IMP mask discovery in katgpt-rs (training → riir-train).
- Does not train stitching layers unless P2 G5 fails (modelless-first mandate per AGENTS.md §"MANDATORY: exhaust modelless paths before deferring to riir-train").

**What stays private vs open:**

| Open (katgpt-rs, MIT) | Private (riir-train) |
|---|---|
| `CanonicalIntent` type | STE permutation discovery code |
| `ModelAdapter` trait | IMP mask discovery |
| `ProcrustesAdapter` impl | Trained stitching layers (fallback only) |
| `SubspaceAdapter` impl | Activation collection on training data |
| `MaskAdapter` apply (not discovery) | Per-model trained adapter weights |
| Joint SVD algorithm | |

---

## Why this is Super-GOAT candidate (and what would demote it)

**Super-GOAT case (Q1–Q4 all YES mechanically):**
- Novel: no prior art on Git Re-Basin permutation algorithms or canonical intent unification in the 7-repo stack
- New capability: plug-and-play any base model — currently impossible
- Selling point: swap Gemma → Llama without retraining overlays
- Force multiplier: ≥5 systems connect

**What would demote to GOAT:**
- G5 fails: Procrustes loses too much cross-architecture → fall back to trained stitching. Still ships, just needs a training step. Becomes "plug-and-play any same-arch model" instead of "any model".
- G6 fails: canonical steering transferred cross-arch doesn't beat a good system prompt → narrow to "intra-architecture snapshot swap". Still useful (Gemma-A ↔ Gemma-B), narrower selling point.

**What would kill it entirely:**
- G1 fails on the modelless path AND riir-train stitching also fails G1 → no plug-and-play at any tier. Document the negative result and move on. (Unlikely — Research 406 SAR already proved single-model SVD purification works modellessly.)

---

## References

- Research: [katgpt-rs/.research/459_canonical_intent_space_plug_and_play.md](../.research/459_canonical_intent_space_plug_and_play.md)
- riir-train counterpart: [riir-train/.research/406_git_rebasin_universal_subspace.md](../../riir-train/.research/406_git_rebasin_universal_subspace.md)
- Substrate shipped:
  - [Issue 001 / Plan 152 — orthogonal Procrustes](../crates/katgpt-spectral/src/procrustes.rs)
  - [Plan 423 / Research 406 — SAR spectral rewiring](../.research/406_Spectral_Rewiring_Weight_Delta_Purification.md)
- Cousin research:
  - [178 Rosetta cross-model](../.research/178_Rosetta_Neurons_Cross_Model_Alignment.md)
  - [238 LoRA-Muon gauge invariant](../.research/238_LoRA_Muon_Spectral_Low_Rank_Manifold.md)
  - [227 GPart isometric](../.research/227_GPart_Isometric_Partition_Inference.md)
  - [231 SOPTV sparse off-principal](../.research/231_Sparse_Off_Principal_Task_Vector_OPD.md)
- Source papers:
  - [Git Re-Basin (arxiv 2209.04836)](https://arxiv.org/abs/2209.04836)
  - [Universal Weight Subspace (arxiv 2512.05117)](https://arxiv.org/abs/2512.05117)
  - [Lottery Ticket Hypothesis (arxiv 1803.03635)](https://arxiv.org/abs/1803.03635)
