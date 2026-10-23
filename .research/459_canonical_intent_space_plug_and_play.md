# Research 459: Canonical Intent Space — Plug-and-Play Any Base Model

> **Source (three-paper fusion):**
> - [Git Re-Basin: Merging Models Modulo Permutation Symmetries](https://arxiv.org/abs/2209.04836) — Ainsworth, Hayase, Srinivasa (UW), ICLR 2023
> - [The Universal Weight Subspace Hypothesis](https://arxiv.org/abs/2512.05117) — Kaushik, Dec 2025
> - [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) — Frankle, Carbin (MIT), ICLR 2019 Best Paper
> **Date:** 2026-07-25
> **Status:** **CLOSED 2026-07-27 — cross-arch modelless path PERMANENTLY exhausted.** Four converging failure lines: (1) P3 — centroid agreement −0.33 after Procrustes (aligns shape, not location); (2) P3b — layer 0 discriminates best, Git Re-Basin contradicted (vocabulary signal, not semantic); (3) P3c — length detrending reverses Python discrimination (+0.19→−0.15 — the apparent Rust-idiom signal was prompt length); (4) Recipe D — length-matched corpus controls length at construction time (detrend PASSES for all k) but cross-arch agreement never crosses +0.01 across k ∈ {2,4,8,16} (best +0.009 at k=16, threshold ≥ 0.5). The failure is STRUCTURAL cross-arch disagreement, not length, not noise. Recipe E (gradient descent) NOT opened — failure pattern (cross-arch disagreement, not non-linearity) rules it out. P1 G5 still holds (joint-SVD shared subspace preserves pairwise alignment at k∈{2,4} — a real cross-model covariance result, independent of canonical direction existence). Cross-arch Super-GOAT claim **PERMANENTLY DEMOTED** — reopens only on a **non-hidden-state construction** (AST/clippy/ownership-graph features), NOT on any further hidden-state method. Intra-arch claim (ProcrustesAdapter for same-dim pairs) unaffected. See [riir-train/.benchmarks/427_canon_p4_recipe_d_length_matched.md](../../riir-train/.benchmarks/427_canon_p4_recipe_d_length_matched.md) for the decisive Recipe D results.
>
> **⛔ STATUS CORRECTION 2026-10-23 (riir-train Issue 644 T4):** the Issue-571 re-run below fixed the MiniCPM side, but every Gemma-2 hidden state at layer ≥ 3 still came from a serving forward that routed a zero delta-routing query (fixed riir-infer `9414946`). Re-run on the corrected forward, with a reverted-fix control reproducing every earlier number exactly: the four-gate Recipe D FAIL **holds** at every k (riir-train Bench 427). Its mechanism changes again: the Rust−prose margin is now negative on **both** models, so the ceiling is not MiniCPM-specific geometry. Agreement passes only at k=2 (+0.80; k=4 +0.74 → −0.08), and k=2 passes 3 of 4 gates. P3's d_diff agreement falls back to +0.22 (Bench 424), while P3c's rises to +0.82 with Gemma Rust−Python flipping negative (Bench 426). The untested modelless follow-up (prose on the negative side of the contrast) is riir-train Issue 646. The closure stands; it is not yet exhausted.
>
> **⛔ STATUS CORRECTION 2026-09-30 (riir-train Issue 571 T3 — the instrument was broken, the re-run is in):** every MiniCPM5 hidden state behind failure lines (1), (3) and (4) came from a RoPE-broken llama loader + a wrong digit-grouping tokenizer (riir-infer `0b26b9a` / `175f67f` fix both). Re-run same-harness same-args: **the "structural cross-arch disagreement" mechanism is REFUTED** — Recipe D cross-arch agreement reads **+0.9401 (k=2) / +0.7428 (k=4)** against the recorded "never crosses +0.01" (Bench 427 §Re-run); P3c's d_diff agreement **passes** at +0.5229 with Rust−Python discriminating on BOTH models and the detrend reversal one-sided, not both (Bench 426 §Re-run — **the modelless d_diff line REOPENS at candidate strength** per the issue's own flip rule); P3's formal G6a FAIL holds, with d_diff agreement now +0.68, ABOVE threshold (Bench 424 §Re-run). The four-gate outcomes stand as measured (no k passes all four — the surviving blocker at every k is the MiniCPM **prose** same-sign control), so the demotion stands as a gate matter; but the "reopens ONLY on a non-hidden-state construction, NOT on any further hidden-state method" bar was derived from the refuted mechanism and is RETIRED — the hidden-state d_diff construction is back at candidate strength with one named blocker. Any reopen decision re-derives from the bench re-run sections, not from this closure's reasoning.
>
> **⛔ RESOLUTION 2026-09-30 (riir-train Issue 601 T1/T2 — the reopened line's blocker is now MEASURED, and it closes the line again):** the candidate-strength reopen above rested on one unmeasured confound — the prose control was n=5. Issue 601 T1 expanded it 5→30 (mixed registers, shared pool) and added a length-matched paired control (Recipe D's greedy pairing applied to prose) in BOTH harnesses. **Result: the MiniCPM wrong sign is invariant to both controls on both corpora at every k ∈ {2,4,8,16}** (Bench 426/427 §Prose-control expansion: P3c −0.6113 → −0.6065 at n=30, −0.6664 length-matched; Recipe D −0.25 → −0.23 at the gate k=4, −0.15 matched, and −1.07/−0.07/−0.03 at k=2/8/16 — negative everywhere, diluting toward 0⁻ as k grows exactly where the agreement gate collapses). Candidates "small-n" and "length confound" are REFUTED; **genuine geometry is CONFIRMED: MiniCPM5-1B's prose region overlaps its Rust direction in the shared low-k subspace**. The modelless d_diff line therefore CLOSES again — not on the refuted "structural disagreement" story but on a measured ceiling: three of four gates pass (agreement, Python, JS on both models), the prose same-sign control does not clear modellessly at any k. The reopened-then-closed arc is now fully evidenced end to end; a further hidden-state reopen would need a construction that separates prose from Rust ON MiniCPM — nothing in the k-sweep suggests the linear difference-of-means family contains one. Recipe E's premise question ("is the failure non-linearity?") now has its T1 evidence: the failure is a geometric overlap that dilutes with dimension rather than crossing — a learned direction might avoid the region, which is exactly the Recipe E decision the original closure reserved for an owner call.
> **Related Research:** 178 (Rosetta cross-model alignment), 227 (GPart isometric partition), 231 (SOPTV sparse off-principal), 238 (LoRA-Muon gauge invariant), 406 (SAR spectral rewiring), 098/214/444 (lottery ticket lineage)
> **Related Plans:** TBD — gated on Proposal 009 GOAT-gate outcome
> **Cross-ref (riir-train):** Research 406 (Git Re-Basin + Universal Subspace training-side counterpart)
> **Classification:** Public — generic canonical intent space (WHAT, not HOW)

---

## TL;DR

Three foundational weight-geometry papers compose into one architectural claim: **a canonical intent space** — a tokenizer/architecture-neutral latent direction space where intent vectors (Rust idiom, NPC personality, emotion, style) live. Each base model carries a deterministic `ModelAdapter` that projects canonical directions into its specific latent space for steering/decoding. **Plug any frozen base model (Gemma, MiniCPM5, Llama, Qwen) into the system without retraining personality/idiom/style overlays.**

**Distilled for katgpt-rs (modelless, inference-time):** The open primitive is `CanonicalIntent` (a normalized direction vector) + `ModelAdapter` trait with three concrete impls — `ProcrustesAdapter` (substrate ships in `crates/katgpt-spectral/src/procrustes.rs`), `SubspaceAdapter` (extends SAR / Research 406 to a canonical multi-model basis), `MaskAdapter` (applies a precomputed lottery-ticket mask).

**Verdict: Super-GOAT candidate.** Novel mechanism (Q1 ✓ — no prior art on Git Re-Basin permutation algorithms or canonical intent unification in stack), new capability class (Q2 ✓ — plug-and-play any base model is currently impossible), product selling point (Q3 ✓ — "swap Gemma for Llama without retraining overlays"), force multiplier (Q4 ✓ — connects procrustes + freeze/thaw + LoRA + latent_functor + cross_game + KG shard alignment + SAR + SOPTV + GPart + LoRA-Muon — ≥5 systems).

**Honest caveat:** Q1–Q4 are mechanically strong; the empirical claim that Procrustes + canonical projection preserves enough information for downstream steering across architectures is **unproven** and is the G5/G6 GOAT gate (§6). If G5/G6 fail, this downgrades to GOAT (intra-architecture plug-and-play only, no cross-arch).

---

## 1. Paper Core Findings

### 1.1 Git Re-Basin — permutation alignment brings same-arch models into a shared basin

**Claim:** Most SGD solutions belong to a single basin modulo permutation symmetries of hidden units. Two independently-trained same-architecture models can be linearly interpolated barrier-free after permuting one's neurons to align with the other.

**Three algorithms (in order of cost/quality):**
1. **Activation matching** — `argmax_P ⟨P, Z_A Z_B^T⟩_F` over permutation matrices. Linear Assignment Problem (Jonker-Volgenant, polynomial time). Same-input activations, find which neurons correspond.
2. **Weight matching** — coordinate descent on SOBLAP (Sum of Bilinear Assignment Problems; NP-hard, Lemma 1, no PTAS for L>2). **No data needed.** Surprisingly competitive; runs in seconds.
3. **STE matching** — straight-through estimator through the discrete projection. Most expensive, best results.

**Headline results:**
- Zero-barrier linear mode connectivity between ResNet-20 (32× width) on CIFAR-10 — first ever demonstration
- Weight matching runs in seconds; activation matching in minutes; STE in minutes–hours
- Wider models exhibit better LMC; thin models and ConvNeXt are documented counterexamples (App. A.1)
- Models trained on disjoint datasets merge constructively — merged model outperforms either input on test loss with no extra compute/memory
- A counterexample (App. A.6) proves LMC is not universal: it's an emergent property of SGD bias

**Scope limit (critical, often missed):** Git Re-Basin operates **within an architecture family**. Permutation symmetries permute neuron indices in a single network's weight matrices; you cannot permute Llama-7B into Gemma-7B because the dims/tokenizers/attention patterns differ. Cross-architecture transfer needs the **activation-space** story (Stitchable AI), not the weight-space story.

### 1.2 Universal Weight Subspace — models converge to shared spectral subspaces

**Claim:** Across 1100+ models (500 Mistral-7B LoRAs, 500 ViTs, 50 Llama-8B), weight matrices systematically collapse onto **shared low-dimensional spectral subspaces**. A few principal directions capture the majority of variance, regardless of initialization, task, or domain.

**Mechanism:** SVD of each weight matrix W = UΣV^T; the joint subspaces span(U_A ⊗ U_B ⊗ ...) across models capture shared information content.

**What this paper adds over single-model SVD (which Research 406 / SAR already ships):** the **joint** basis across diverse models. SAR projects a weight delta onto one base model's SVD basis. Universal Subspace extends this: there exists a basis that is *simultaneously* good for many models. This is the empirical foundation that makes a canonical intent space plausible.

### 1.3 Lottery Ticket — sparse trainable subnetworks at initialization

**Claim:** Dense randomly-initialized networks contain subnetworks f(x; m·θ₀) ("winning tickets") that — when trained in isolation — match the test accuracy of the original network in the same number of iterations. Their connections won the initialization lottery.

**Algorithm (Iterative Magnitude Pruning, IMP):** train → prune smallest-magnitude p% → reset surviving weights to θ₀ → repeat. Winning tickets at 10–20% of original size match full-net accuracy on Lenet/Conv-2/4/6/VGG-19/Resnet-18. Random reinitialization breaks the win — structure alone is insufficient; the *initialization* is the ticket.

**Two modelless residues (mask discovery itself is training → riir-train):**
- **Mask application at inference** — elementwise multiply m⊙W. Already partly shipped in `riir-neuron-db/src/spectral_flatness.rs` (lottery-ticket init) and `katgpt-rs` pruning paths.
- **Mask transfer across canonical-aligned models** — if model B is Procrustes-aligned to model A, A's winning-ticket mask transfers (under appropriate basis change). This is the novel modelless residue.

---

## 2. What's Modelless vs Training-Only (the §3.5 path-0 decomposition)

| Component | Modelless? | Where |
|---|---|---|
| **Apply discovered permutation π to weights** | ✅ elementwise index permute | katgpt-rs |
| **Activation matching's LAP step** | ✅ Jonker-Volgenant on pre-collected activation matrices | katgpt-rs (algorithm) / riir-train (collect activations on training data) |
| **Weight matching's SOBLAP coord descent** | ✅ pure weight algebra, no data | katgpt-rs |
| **STE matching** | ❌ requires gradient training loop | riir-train |
| **Apply precomputed lottery-ticket mask** | ✅ elementwise multiply | katgpt-rs (partly shipped) |
| **Discover lottery-ticket mask (IMP)** | ❌ iterative training | riir-train |
| **SVD of weight matrix (single-model SAR)** | ✅ already shipped (Research 406 / Plan 423) | katgpt-rs |
| **Joint subspace across N models** | ✅ SVD of stacked bases, deterministic | katgpt-rs (new — extends SAR) |
| **Procrustes rotation R: latent_A → latent_B** | ✅ already shipped (Issue 001 / Plan 152) | katgpt-rs |
| **Trained stitching layer (if modelless loses G5)** | ❌ backprop | riir-train (fallback only) |
| **Canonical intent direction definition** | ✅ user-supplied (Rust idiom, emotion, personality) | katgpt-rs |

The modelless surface is large. Only STE permutation discovery, IMP mask discovery, and the stitching-layer fallback (if needed) need riir-train. See `riir-train/.research/406_git_rebasin_universal_subspace.md`.

---

## 3. Fusion: the Canonical Intent Space architecture

```text
        ┌────────────────────────────────────────────────────────────┐
        │ Canonical Intent Space (architecture-neutral)              │
        │                                                            │
        │   d_Rust_idiom  = normalized direction in canonical space  │
        │   d_curiosity   = ...                                      │
        │   d_valence     = ...                                      │
        │   d_Rosetta_universal = joint subspace basis (§1.2)        │
        └─────────────────────────┬──────────────────────────────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
         ProcrustesAdapter  SubspaceAdapter   MaskAdapter
         R_Gemma            U_Gemma           m_Gemma
         R_Llama            U_Llama           m_Llama
         R_MiniCPM5         U_MiniCPM5        m_MiniCPM5
                  │               │               │
                  └───────────────┼───────────────┘
                                  ▼
                        model_specific_latent
                                  │
                                  ▼
                        frozen_base_model.decode()
                                  │
                                  ▼
                              tokens / actions
```

**The functor framing the user proposed is exact:** `F: CanonicalIntent × ModelAdapter → ModelSpecificLatent → decode()`. Each adapter is a structure-preserving map. Composition of canonical operations (sum, scale, gate via sigmoid) commutes through the adapter because the adapter is linear (Procrustes, subspace projection) or elementwise (mask).

### 3.1 Open primitive spec (katgpt-rs)

```rust
// crates/katgpt-core/src/canon/mod.rs (new module)

/// Architecture-neutral intent direction. Owned, normalized f32 vector.
/// Lives in canonical space — never decoded directly.
#[derive(Clone, Debug)]
pub struct CanonicalIntent {
    pub tag: u64,           // BLAKE3 of label for sync/commit
    pub direction: Vec<f32>, // unit-norm
}

/// Projects a canonical intent into a specific base model's latent space.
/// Modelless: zero training, deterministic given (adapter_state, base_model).
pub trait ModelAdapter: Send + Sync {
    /// Apply the adapter to a canonical direction, writing into `out`.
    /// `out.len() == base_model_latent_dim`.
    fn project_into(&self, canonical: &CanonicalIntent, out: &mut [f32]);

    /// Inverse: extract canonical coordinates from an observed model latent.
    /// Used for "what intent is this activation expressing?" diagnostics.
    fn extract_from(&self, model_latent: &[f32]) -> Vec<f32>;

    /// Latent dim of the target model (adapter output dim).
    fn target_dim(&self) -> usize;

    /// BLAKE3 commitment of the adapter state (for freeze/thaw attestation).
    fn commitment(&self) -> [u8; 32];
}

/// Orthogonal Procrustes rotation (substrate: crates/katgpt-spectral/src/procrustes.rs).
/// Linear, bijective, information-preserving. The default for same-arch swap.
pub struct ProcrustesAdapter { /* rotation R, target_dim */ }

/// Project onto model's SVD basis (extends Research 406 SAR from single-model
/// to canonical joint basis). Lossy but works cross-architecture.
pub struct SubspaceAdapter { /* basis U_k, target_dim */ }

/// Apply a precomputed lottery-ticket mask. Mask discovery → riir-train.
pub struct MaskAdapter { /* mask: BitVec, target_dim */ }
```

The three adapters compose: `(ProcrustesAdapter ∘ SubspaceAdapter)(d)` projects canonical → joint subspace → model basis. Each is independently testable and feature-flagged.

### 3.2 Existing substrate that already does most of this

| Existing | What it does | Reuse for canonical intent |
|---|---|---|
| `crates/katgpt-spectral/src/procrustes.rs` (Issue 001) | Orthogonal Procrustes via Newton-Schulz, bit-identical across x86_64/aarch64/wasm32 | The `ProcrustesAdapter` impl |
| `crates/katgpt-spectral/src/spectral_rewire.rs` (Plan 423, Research 406) | Project weight delta onto base SVD basis | The `SubspaceAdapter` SVD step |
| `crates/katgpt-core/src/closure/functor_edge.rs` | Functor application (`apply_functor_edge_into`) with sigmoid coherence gate | Composition primitive |
| `riir-neuron-db/src/spectral_flatness.rs` | Lottery-ticket Wiener-entropy init | `MaskAdapter` mask storage format |
| `riir-ai/crates/riir-engine/src/latent_functor/procrustes_bridge.rs` | Game-runtime Procrustes wrap | Reference for adapter lifecycle (stays in riir-ai — game-specific) |
| `riir-ai/crates/riir-engine/src/latent_functor/cross_resolution_bridge.rs` | Asymmetric d_src ≠ d_dst functor transfer (Bomber d=8 → Go d=64) | Proof that cross-dim adapter works at runtime |

**The substrate is mostly built.** The new work is: (a) the canonical intent space type + adapter trait (small, ~200 LOC), (b) the joint subspace SVD (extension of SAR — ~150 LOC), (c) cross-architecture validation (the actual GOAT gate).

---

## 4. Novelty gate (§1.5)

### Q1: No prior art? ✅

Grep across all 7 repos for `git_rebasin|rebasin|stitchable|universal_subspace|weight_stitching` → **zero hits**. Closest cousins (all single-mechanism, none unify the three papers):

- **Research 178 (Rosetta)** — cross-model neuron alignment via Pearson + best-buddies. Activation-space, no permutation, no canonical-space abstraction.
- **Research 227 (GPart)** — isometric partition for adapter loading. Storage compression, not cross-model alignment.
- **Research 231 (SOPTV)** — sparse off-principal task vector. Within-model adapter storage, not cross-model.
- **Research 238 (LoRA-Muon)** — gauge-invariant adapter composition. Gauge rebalancing, not cross-model alignment.
- **Research 406 (SAR)** — single-model SVD basis projection. Closest substrate for `SubspaceAdapter`, but single-model only.
- **riir-train/Research 067 (Rosetta cross-game LoRA)** — cross-game LoRA neuron mining (private training counterpart of 178).

**None of them define a canonical intent space + per-model adapter.** The composition is novel.

### Q2: New capability class? ✅

Currently, swapping Gemma → Llama requires retraining every personality/idiom/style LoRA overlay. With canonical intent space, **one canonical direction set works across all aligned models.** No codebase mechanism provides this today.

### Q3: Product selling point? ✅

> "Our system loads any frozen base model — Gemma, MiniCPM5, Llama, Qwen — and the same Rust-style / NPC-personality / emotion-direction overlays work without retraining. The model is a decoder; the intent is canonical."

This is the use case the user described: "we did load gemma in our latent space and play game" — generalized beyond Gemma.

### Q4: Force multiplier (≥2 pillars)? ✅

Connects ≥5 systems: procrustes + freeze/thaw + LoRA hot-swap + latent_functor (cross_game, cross_resolution) + KG shard alignment + SAR + SOPTV + GPart + LoRA-Muon.

### Verdict: **Super-GOAT candidate** — all four Q's YES mechanically.

**Per skill rule §1.5 "No 'candidate' escape hatch"**: writing "Super-GOAT candidate" anywhere triggers the mandatory outputs in the same session. Those outputs land in this same session:

1. **Open primitive spec** → §3.1 above (will land as `crates/katgpt-core/src/canon/` per Proposal 009).
2. **Private guide** → user explicit override: this lives in **riir-train/.research/406_git_rebasin_universal_subspace.md** (NOT riir-ai/chain/neuron-db — user said "this should be in katgpt-rs and riir-train as possible bc other riir-* is focus on game").
3. **Plan(s)** → gated on Proposal 009 GOAT-gate outcome. Do NOT open a plan until G5/G6 settle.

---

## 5. MOAT gate (§1.6) — domain fit

| Domain | In scope? | Why |
|---|---|---|
| **katgpt-rs** (public engine) | ✅ primary | Canonical intent space + adapter trait + 3 concrete impls are generic modelless inference primitives. Public per the "WHAT not HOW" rule. |
| **riir-train** (private training) | ✅ secondary | STE permutation discovery, IMP mask discovery, stitching-layer fallback (if G5 fails). Private per the "training HOW" rule. |
| **riir-ai/chain/neuron-db** | ❌ per user override | User: "other riir-* is focus on game and i dont want it to have this related context there." The existing game-side consumer stays in riir-ai (latent_functor/procrustes_bridge.rs is already there) but **earns no new files** for this work. |

**Verdict:** Strengthens katgpt-rs moat (public engine gains a unique primitive — plug-and-play any base model). Strengthens riir-train moat (training vault gains the permutation/mask discovery IP). Does NOT pollute game/chain/shard repos.

---

## 6. GOAT gate definition (the hard part)

| Gate | Floor | Target |
|---|---|---|
| G1 (correctness) | baseline Gemma + Rust sysprompt | canonical-direction steering produces ≥ baseline on Rust-specific eval (HumanEval-Pack Rust, % clippy-clean, compile-first-try) |
| G2 (perf) | baseline model latency | adapter `project_into` < 50µs (Procrustes is sub-ms; mask is elementwise; subspace is one matvec) |
| G3 (no-regression) | baseline on non-Rust tasks | canonical steering doesn't break general capability (MMLU-lite check) |
| G4 (alloc-free) | inference hot path | `project_into` is zero-alloc after adapter construction (R · h is one SIMD matvec into caller buffer) |
| **G5 (cross-model preservation)** ⭐ | same canonical direction applied via Procrustes to N≥2 same-arch models | cosine sim of steered outputs > 0.7 on held-out prompts. **This is the gate that decides plug-and-play works at all.** |
| **G6 (cross-architecture gain)** ⭐⭐ | baseline Llama + same sysprompt on Rust eval | steering transferred from Gemma to Llama via canonical space produces measurable quality gain over baseline Llama + same sysprompt. **If G6 fails, downgrades to intra-arch GOAT (still ships, narrower scope).** |

The floor for G1/G6 is **"good system prompt"** (per the "Report the Floor" rule, Research 322). Most style gains evaporate against a good prompt — G6 is the gate that decides whether this is Super-GOAT or GOAT.

**If G5 fails (Procrustes loses too much across architectures):** fall back to riir-train for trained stitching layers (the Stitchable AI path). The canonical intent space still ships; the adapter becomes trained instead of deterministic. Demote verdict from Super-GOAT to GOAT.

**If G6 fails (canonical steering transferred cross-arch doesn't beat good sysprompt):** keep as intra-architecture plug-and-play. Demote to GOAT. Still useful (Gemma-A ↔ Gemma-B snapshots) but narrower selling point.

---

## 7. Implementation priority table

| Phase | Scope | Verdict gate | Repo |
|---|---|---|---|
| P0 | `Canon` module skeleton: `CanonicalIntent` + `ModelAdapter` trait + `ProcrustesAdapter` (wraps existing `procrustes.rs`) | G1, G2, G4 on synthetic canonical directions | katgpt-rs |
| P1 | `SubspaceAdapter` (joint SVD across N models — extends `spectral_rewire.rs`) | G1, G2, G4 on single-model | katgpt-rs |
| P2 | Cross-model validation: fit Procrustes Gemma ↔ MiniCPM5 (both at `riir-train/data/*.gguf`), measure G5 cosine preservation | **G5 GO/NO-GO for Super-GOAT** | katgpt-rs (validation harness) + riir-train (data) |
| P3 | Rust-style canonical direction + G6 gate (the real selling point test) | **G6 GO/NO-GO for cross-arch Super-GOAT** | katgpt-rs (eval) + riir-train (fallback if needed) |
| P4 | `MaskAdapter` (lottery ticket mask transfer) — requires riir-train IMP mask discovery | auxiliary gate | katgpt-rs + riir-train |
| P5 | Promote `canon` to default-on if G1–G6 pass; demote to opt-in if G5/G6 fail | — | katgpt-rs |

**P2 is the make-or-break.** Run it before opening a full implementation plan. CARGO_TARGET_DIR=/tmp per AGENTS.md.

---

## 8. Cross-references

- **Proposal:** [katgpt-rs/.proposals/009_canonical_intent_space.md](../.proposals/009_canonical_intent_space.md)
- **Training-side counterpart:** [riir-train/.research/406_git_rebasin_universal_subspace.md](../../riir-train/.research/406_git_rebasin_universal_subspace.md)
- **Substrate (shipped):**
  - [crates/katgpt-spectral/src/procrustes.rs](../crates/katgpt-spectral/src/procrustes.rs) — Issue 001 / Plan 152
  - [crates/katgpt-spectral/src/spectral_rewire.rs](../crates/katgpt-spectral/src/spectral_rewire.rs) — Plan 423 / Research 406
  - [crates/katgpt-core/src/closure/functor_edge.rs](../crates/katgpt-core/src/closure/functor_edge.rs)
- **Closest cousin notes** (single-mechanism, this research unifies them):
  - [178 Rosetta cross-model](178_Rosetta_Neurons_Cross_Model_Alignment.md)
  - [227 GPart isometric](227_GPart_Isometric_Partition_Inference.md)
  - [231 SOPTV sparse off-principal](231_Sparse_Off_Principal_Task_Vector_OPD.md)
  - [238 LoRA-Muon gauge invariant](238_LoRA_Muon_Spectral_Low_Rank_Manifold.md)
  - [406 SAR spectral rewiring](406_Spectral_Rewiring_Weight_Delta_Purification.md)
- **Lottery ticket lineage:** [098 PrudentBanker](098_PrudentBanker_Safe_Delayed_Adversarial_Bandits.md), [214 Spectral Irrep](214_Spectral_Irrep_Compression_Inference.md), [444 IMIR](444_Invariant_Manifold_Inductive_Reasoning_IMIR.md)
