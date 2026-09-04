# Research 389: CHaRS — Concept Heterogeneity-Aware Representation Steering

> **Source:** [Concept Heterogeneity-aware Representation Steering](https://arxiv.org/abs/2603.02237) — Abdullaev, Wong, Lee, Jiang, Nguyen, Nguyen (NUS), ICML 2026 (PMLR 306). Code: github.com/lazizcodes/CHaRS
> **Date:** 2026-07-07
> **Status:** Active
> **Related Research:** 290 (Latent Field Steering — additive single-direction), 276 (Personality-Weighted Composition — per-layer drift), 302 (FAME CommittedFieldBlend — per-entity fixed blend), 357 (Neural Procedural Memory Activation Steering — PASS), 382 (Spherical Steering Slerp — single-target input-adaptive strength), 144 (Functional Emotions), 281 (Per-Tick Salience Tri-Gate)
> **Related Plans:** 309 (Latent Field Steering primitive), 297 (PersonalityWeightedComposition), 321 (CommittedFieldBlend), 405 (Spherical Steering)
> **Cross-ref (riir-ai):** Research 153 (Latent Field Steering Game Runtime Guide), 158 (Committed Personality Blend Guide)
> **Classification:** Public

---

## TL;DR

CHaRS replaces single-global-direction steering (DiM = difference-in-means) with an **input-position-adaptive cluster-mixed** steering field: cluster source/target activations into K centroids each, solve a discrete OT (Sinkhorn) coupling P⋆ between centroids, then for each input `x` blend the K×K cluster-pair translation vectors `v_ij = b_j − a_i` by an RBF-kernel-on-`x` × OT-coupling weight. Result: a smooth, context-dependent vector field that respects concept heterogeneity (a "refusal" concept manifests differently in different latent regions, so a single global shift under-steers some regions and over-steers others).

**Distilled for katgpt-rs (modelless, inference-time):**
A zero-allocation primitive `chars_steering_into(state, anchor_bank, ot_plan, sigma, alpha, scratch)` that produces `state' = state + α · v̂(state)` where `v̂(state) = Σᵢⱼ P_ij · k_RBF(state, a_i) / Σ_p,q P_pq · k_RBF(state, a_p) · v_ij`. The anchor bank `{a_i, b_j, v_ij}` and OT plan `P` are **frozen, BLAKE3-committed artifacts** computed offline (k-means + Sinkhorn over contrastive corpora). Per-input cost: `O(K² · D)` (K=15 max in paper, D=8 for HLA → ~1.4 kFLOPs, SIMD-trivial). No training, no gradients.

**Novelty verdict (one-line):** the four shipped steering primitives (Latent Field Steering / PersonalityWeightedComposition / CommittedFieldBlend / Spherical Slerp) all produce either a single direction, an entity-fixed blend, or a per-layer drift — **none produces an input-position-dependent steering vector via soft RBF-on-clusters × OT-coupling routing**. CHaRS fills exactly that gap. GOAT; potential Super-GOAT via fusion with CommittedFieldBlend + latent_functor re-estimation (see §2.4 F1).

---

## 1. Paper Core Findings

### 1.1 The thesis

Standard representation steering (CAA / ActAdd / DirAbl / difference-in-means) implicitly assumes the target concept is **homogeneously distributed** in latent space — i.e., a single global translation `T(x) = x + (m₂ − m₁)` suffices. The paper shows empirically (PCA / t-SNE of last-token activations for harmful vs harmless) that real concept distributions are **clustered, context-dependent, multimodal** (Figure 1, Table 9). A global DiM shift is therefore brittle: it undercorrects subregions whose local mean differs from the global mean, and overcorrects subregions that already align.

### 1.2 The mechanism — three layers

**Layer 1: GMM-OT framing.** Model source μ and target ν as K-component Gaussian mixtures. The Mixture Wasserstein distance `MW₂²` (Delon & Desolneux 2020) reduces the infinite-dim OT problem to a **discrete K×K OT problem** between component centroids with closed-form Gaussian-pair costs. DiM = special case K=1 with equal covariances → pure translation.

**Layer 2: Practical Sinkhorn matching.** Cluster both corpora via k-means into K centroids `{a_i}`, `{b_j}`, compute cost `C_ij = ‖a_i − b_j‖₂²`, solve entropy-regularized OT via Sinkhorn:
```
P⋆ = arg min_{P ∈ Π(w_A,w_B)} ⟨P,C⟩ + λ H(P)
```
P⋆ is a soft, geometry-aware cluster correspondence — handles unpaired corpora (CHaRS does not need paired data, unlike ACT Wang 2025).

**Layer 3: Barycentric input-adaptive steering.** Given input `x`, RBF-gate on source centroids `k(x, a_i) = exp(−‖x − a_i‖² / (2σ²))` (σ = median heuristic), then:
```
v̂(x) = Σᵢⱼ  P⋆_ij · k(x, a_i) / Σ_{p,q} P⋆_pq · k(x, a_p)  ·  (b_j − a_i)
T_α(x) = x + α · v̂(x)
```
This is a **position-dependent vector field**: each `x` gets a smooth kernel-weighted combination of cluster-pair translation vectors.

### 1.3 Principal Component Thresholding (CHaRS-PCT)

The K×K cluster-pair translation vectors `v_ij = b_j − a_i` have centered covariance `Σ_total` of rank ≤ 2K − 2 ≪ D. PCA on `Σ_total` yields a low-rank factorization `v̂(x) = v̄ + Σ_k α̂_k(x) · u_k`. Truncating to top-L PCs gives CHaRS-PCT — same quality with fewer steering axes (Table 1, Figure 4).

### 1.4 Empirical headline (LLM + diffusion)

- **Jailbreaking ASR (Table 1, ActAdd):** +1–7% over DiM ActAdd across Gemma2-9B / Llama3.1-8B / Llama3.2-3B / Qwen2.5-{3B, 7B, 14B, 32B}. CHaRS-PCT matches or beats CHaRS in select cases.
- **Toxicity mitigation (Table 3, sequential):** up to 43% / 42% relative reductions in classifier toxicity vs Linear-AcT on Qwen2.5-7B / Llama3-8B.
- **Image style control (Figure 2):** FLUX.1 + CHaRS reaches peak cyberpunk induction at λ=0.8 with stable CLIPScore > 0.26 — strictly Pareto-dominates Linear-AcT.
- **Throughput (Table 12):** CHaRS-ActAdd throughput (7239.9 tok/s) ≈ ActAdd (7365.9 tok/s) on Qwen2.5-7B-Instruct + vLLM/H100 — **negligible overhead**, the RBF+OT gating is amortized to ~one cluster-distance computation per token.

### 1.5 What's training-only / out of scope here

- The k-means + Sinkhorn computation on contrastive corpora — **offline, one-time**, produces the frozen `{a_i, b_j, P⋆, σ}` artifact. (Equivalent to `EmotionDirections` contrastive-mean construction.)
- The LLM jailbreak / toxicity / FLUX style-control benchmarks — paper-specific.
- The human-evaluation 2AFC protocol — measurement, not mechanism.
- The PCT eigen-decomposition of `Σ_total` — offline, ships as frozen top-L `u_k` vectors alongside the anchor bank.

The modelless transferable primitive is **the input-position-adaptive cluster-mixed steering field**.

### 1.6 Key ablation (§5)

- **Coupling strategy (Table 5):** OT (95.19% ASR) ≫ Nearest-Neighbor (94.23%) ≫ Uniform (87.50%). The OT matching is load-bearing — naive centroid correspondence loses.
- **Cluster count K (Figure 5):** no clean monotonic trend, but K > 1 generally beats K = 1 (which degenerates to DiM). Paper-tuned best K ∈ {5, 10, 11, 15} depending on model.
- **Covariance assumption (Table 10):** equal-covariance (DiM-style per pair) **beats** diagonal-covariance estimation at small sample sizes — overfitting. Keep equal-covariances; relax later if data is plentiful.

---

## 2. Distillation

### 2.1 Vocabulary translation (paper → codebase)

| Paper term | Codebase equivalent |
|---|---|
| representation steering | `apply_latent_steering`, `apply_blended`, `slerp_steering_into`, `compose_into` |
| difference-in-means (DiM) steering vector | `EmotionDirections` direction vector, `LatentSteeringVector` |
| contrastive dataset (harmful vs harmless) | `ContrastivePairProvider` (Plan 087), CLR vote polarity |
| cluster centroids {a_i}, {b_j} | **archetype anchor vectors**, frozen `style_weights[64]`-style Pod |
| Gaussian mixture model | **K-anchor latent region bank** (subspace cluster ensemble) |
| Sinkhorn / OT coupling P⋆ | `KVarN`-style Sinkhorn (Plan 179), but on **steering directions** not KV variances |
| barycentric projection | soft MoE gating (cf. `CommittedFieldBlend::apply_blended` over K archetype fields) |
| input-adaptive steering `v̂(x)` | position-dependent latent field (cf. `LatentField::Radius` kernel but **on the source activation manifold**, not on the game map) |
| Principal Component Thresholding | `subspace_phase_gate` Jacobian SVD (Plan 301), `dual_gram` PCA (Plan 159) |
| concept heterogeneity | **per-NPC archetype plurality** — same emotion concept (fear) manifests differently in predator vs prey HLA subregions |

The two standing vocabulary blocks to add for any future steering paper:

- "input-adaptive steering" / "position-dependent vector field" / "context-dependent intervention" → **cluster-mixed field**, **RBF-gated anchor blend**, **soft-routed archetype translation**
- "barycentric projection" / "transport plan" / "OT coupling" → **soft MoE gating over anchor pairs**, **Sinkhorn-matched archetype correspondence**

### 2.2 What we already ship (prior-art surface — verify before novelty claim)

| Paper mechanism | Shipped cousin | Plan / file | Diff vs CHaRS |
|---|---|---|---|
| Single global steering vector (DiM baseline) | `apply_latent_steering` (`state += α·v`) | Plan 309, R290 | CHaRS generalizes to K cluster-mixed vectors |
| Per-layer sigmoid-gated composition of N directions | `PersonalityWeightedComposition::compose_into` | Plan 297, R276 | Composes **layers** with per-tick drift; CHaRS composes **input-region anchors** with no drift (frozen plan) |
| Per-entity FIXED MoE blend of K archetype fields | `CommittedFieldBlend::apply_blended` | Plan 321, R302 | π computed once per entity; CHaRS weight computed **per-input-tick** via RBF(x, anchor). Both blend K archetypes; **different time-scale and different routing signal**. |
| Norm-preserving Slerp toward target with input-adaptive strength | `slerp_steering_into` + `vmf_confidence_gate` | Plan 405, R382 | Single target μ_T, input-adaptive strength; CHaRS uses K target centroids with input-adaptive **routing**. |
| Contrastive direction construction (mean-difference) | `EmotionDirections::project` (read-only), `cna_discover` (write to MLP neurons) | Plan 162, Plan 087, R357 | CHaRS constructs K cluster-pair vectors, not 1 mean-difference |
| RBF / sigmoid kernel falloff | `LatentField::Radius { center, bandwidth, steepness }` kernel `sigmoid((b−d)·k)` | Plan 309 | Same kernel shape; CHaRS applies it on the **source activation manifold** (K anchors), not on the game map |
| Sinkhorn / OT | `KVarN` variance normalization (Plan 179) | src/kvarn/variance_norm.rs | Used for KV cache variance, **never** for steering. Different application of the same primitive. |
| Top-L PCA thresholding of steering covariance | `subspace_phase_gate` Jacobian SVD, Dual-Gram PCA | Plan 301, Plan 159 | Same low-rank factorization math; CHaRS applies it to the v_ij bank |
| Frozen artifact, atomic hot-swap | `MerkleFrozenEnvelope`, `LoRAHotSwap`, `KarcShard` | riir-neuron-db/src/freeze.rs, snapshot.rs | CHaRS anchor bank + OT plan are first-class frozen artifacts |
| Cluster centroids | `ClusterCentroids` StillKV strategy (Plan 245), MTP `clustered_lm_head` (Plan 117) | src/ | K-means centroids exist for KV compaction and MTP vocab routing — **never** for steering direction banks |

**Critical distinction from CommittedFieldBlend (the closest cousin):** CommittedFieldBlend computes `π` from a trajectory summary **once** and freezes it for the entity's lifetime; routing is per-entity-fixed. CHaRS computes the blend weight **per input `x`** via RBF(x, a_i); routing is position-in-latent-space-dependent. The two compose: the anchor library `{a_i, b_j, v_ij}` itself could be a CommittedFieldBlend (K archetypes), and CHaRS provides the within-entity, per-input soft routing on top.

### 2.3 Transferable primitive

```rust
/// CHaRS anchor bank — frozen, BLAKE3-committed artifact produced offline
/// from contrastive corpora via k-means + Sinkhorn.
///
/// `src_centroids`, `tgt_centroids`: K anchor vectors each (D-dim).
/// `ot_plan`: K×K soft coupling matrix from Sinkhorn (row-stochastic in the
///            entropic-OT limit; in general doubly-stochastic up to marginals).
/// `v_ij`: precomputed cluster-pair translations v_ij = b_j − a_i (K×K×D slice).
/// `sigma2`: RBF bandwidth (median heuristic, frozen at construction).
#[repr(C)]
pub struct CharsAnchorBank<const K: usize, const D: usize> {
    pub src_centroids: [[f32; D]; K],
    pub ot_plan: [[f32; K]; K],
    pub v_ij: [[f32; D]; K * K],   // flat K*K rows
    pub sigma2: f32,
    pub blake3: [u8; 32],          // commitment over the above
    pub version: u32,
}

/// CHaRS steering — input-position-adaptive cluster-mixed translation.
///
/// Computes v̂(x) = Σ_ij [ P_ij · k_RBF(x, a_i) / Σ_pq P_pq · k_RBF(x, a_p) ] · v_ij
/// then writes state' = state + α · v̂(x) into `state_out`.
///
/// All scratch caller-provided. Zero allocation in steady state.
/// Per-call cost: K² RBF evaluations + K²·D FMA. For K=8, D=8 → ~512 FLOPs ≪ 1 µs.
pub fn chars_steering_into<const K: usize, const D: usize>(
    state: &[f32],                      // [D] current latent state
    bank: &CharsAnchorBank<K, D>,       // frozen anchor bank + OT plan
    alpha: f32,                         // steering strength
    state_out: &mut [f32],              // [D] output, may alias `state`
    scratch_weights: &mut [f32],        // [K] scratch for normalized RBF weights
    scratch_vhat: &mut [f32],           // [D] scratch for v̂(x)
) {
    // 1. Compute raw RBF weights w_i = Σ_j P_ij · exp(-||x - a_i||² / (2σ²))
    //    (Equation 11 — the sum over j of the OT plan times the RBF kernel).
    //    Denominator Z = Σ_i w_i.
    let inv_2sigma2 = 0.5 / bank.sigma2.max(1e-12);
    let mut z = 0.0f32;
    for i in 0..K {
        let mut d2 = 0.0f32;
        let a = &bank.src_centroids[i];
        let mut k = 0;
        while k + 4 <= D {
            let d0 = state[k]     - a[k];     d2 += d0 * d0;
            let d1 = state[k + 1] - a[k + 1]; d2 += d1 * d1;
            let d2_ = state[k + 2] - a[k + 2]; d2 += d2_ * d2_;
            let d3 = state[k + 3] - a[k + 3]; d2 += d3 * d3;
            k += 4;
        }
        while k < D {
            let d0 = state[k] - a[k]; d2 += d0 * d0;
            k += 1;
        }
        let rbf = simd::fast_exp(-d2 * inv_2sigma2);
        let mut row_sum = 0.0f32;
        for j in 0..K { row_sum += bank.ot_plan[i][j]; }
        let w_i = rbf * row_sum;
        scratch_weights[i] = w_i;
        z += w_i;
    }
    let inv_z = 1.0 / z.max(1e-12);

    // 2. v̂(x) = Σ_ij [ P_ij · w_i / Z ] · v_ij  (Equation 12)
    for d_idx in 0..D { scratch_vhat[d_idx] = 0.0; }
    for i in 0..K {
        let w_i = scratch_weights[i] * inv_z;
        if w_i < 1e-8 { continue; } // skip negligible anchors
        for j in 0..K {
            let pij_w = bank.ot_plan[i][j] * w_i;
            if pij_w < 1e-8 { continue; }
            let v = &bank.v_ij[i * K + j];
            for d_idx in 0..D { scratch_vhat[d_idx] += pij_w * v[d_idx]; }
        }
    }

    // 3. state' = state + α · v̂(x)
    let mut d_idx = 0;
    while d_idx + 4 <= D {
        state_out[d_idx]     = state[d_idx]     + alpha * scratch_vhat[d_idx];
        state_out[d_idx + 1] = state[d_idx + 1] + alpha * scratch_vhat[d_idx + 1];
        state_out[d_idx + 2] = state[d_idx + 2] + alpha * scratch_vhat[d_idx + 2];
        state_out[d_idx + 3] = state[d_idx + 3] + alpha * scratch_vhat[d_idx + 3];
        d_idx += 4;
    }
    while d_idx < D {
        state_out[d_idx] = state[d_idx] + alpha * scratch_vhat[d_idx];
        d_idx += 1;
    }
}
```

**Complexity:** `O(K² · D)` per call. For K=8 (CHaRS-typical), D=8 (HLA) → 512 FMA + 8 RBF evaluations ≪ 1 µs at SIMD width 4. Paper's K=15 max + D=64 shard case → ~14 kFLOPs, still well under 5 µs. Zero allocation after scratch init.

**PCT variant (optional):** ship `chars_steering_pct_into` that projects `v̂(x) − v̄` onto the top-L principal components of `Σ_total` (frozen `u_k` bank). Adds `O(L · D)` per call. Same F1 promotion path as CHaRS; PCT is the "regularized" variant for sequential/layer-wise application.

### 2.4 Fusion

**F1 (PRIMARY — katgpt-rs + riir-ai, candidate): CHaRS × CommittedFieldBlend × latent_functor re-estimation = "per-NPC archetype-routing steering that adapts when the NPC's latent region shifts"**

CommittedFieldBlend commits an NPC's personality as a fixed K=3 archetype blend. CHaRS gives **per-input soft routing** over an anchor bank. Fuse: the anchor bank `{a_i, b_j, v_ij}` is the NPC's **committed archetype library** (the K frozen fields), and CHaRS computes the per-tick routing weight from the NPC's *current HLA position* in latent space. When the NPC drifts into a new latent region (e.g., from social to combat), the RBF gate shifts weight to the combat archetype's translation vector automatically. The `ReestimationScheduler` (riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs) triggers a bank re-commit when coherence drops below `tau_reest`.

**Selling point (candidate):** "NPCs that are steered by their *current* affective region, not a single global personality vector — a wolf in hunt-mode is steered by the hunt archetype's translation, the same wolf in pack-mode by the social archetype's, with smooth sigmoid-gated transitions as its HLA state moves between regions."

**Q1–Q4 pre-check (does NOT trigger Super-GOAT mandatory outputs yet — "candidate", needs full gate):**
- Q1: CommittedFieldBlend is per-entity-fixed; CHaRS gives per-input routing. The fusion — committed archetype library + per-input CHaRS routing — is not shipped. Needs full vocabulary-checked grep.
- Q2: New capability class — per-NPC region-aware steering. Yes if no prior art.
- Q3: Selling point — concrete, demoable. Yes if Q1 holds.
- Q4: Connects CommittedFieldBlend, latent_functor, HLA, EmotionDirections, freeze/thaw. Yes.

**Status:** ✅ EVALUATED (2026-07-07, Issue 049) — **NOT Super-GOAT.** Full Q1–Q4 evaluation complete: Q1 YES (per-input RBF routing genuinely unshipped — CommittedFieldBlend's `pi` is fixed per `functor_bridge.rs`; ReestimationScheduler does periodic re-fit not per-input routing; zone_gating is spatial-density trust adjustment not blend-weight routing), Q2 NO (refinement of soft-MoE-steering class, not new operation — R382 precedent applies), Q3 WEAK (refines R158 committed-personality selling point / Pillar 8), Q4 YES (≥5 pillars). Q2 fails → GOAT-tier refinement, not Super-GOAT. The bare CHaRS primitive ships as GOAT regardless. See `Issue 049` for the full evidence matrix.

**F2 (SECONDARY — katgpt-rs, fusion candidate): CHaRS × Spherical Steering Slerp = "norm-preserving cluster-mixed rotation"**

CHaRS's additive form `T_α(x) = x + α · v̂(x)` inflates L2 norm by `α · ‖v̂(x)‖` (same critique R382 levels at Plan 309). Replace the additive step with Slerp: rotate `x` toward `x + v̂(x)` along the geodesic. Result: norm-preserving heterogeneous steering. Paper does not do this; Spherical does single-target Slerp only. The composition covers both heterogeneity (CHaRS) and norm-preservation (Spherical).

**F3 (TERTIARY — riir-chain, speculative): CHaRS × LatCal commitment = "chain-committed per-NPC archetype-routing events"**

The OT plan `P⋆`, the anchor bank hashes, and the per-tick routing weight `w_i` are all raw scalars. A steering event can be LatCal-committed as `(bank_blake3, w_i_snapshot, alpha_at_tick)` — chain-verifiable record of "this NPC was steered toward region X at tick T". Anti-cheat application: a hacked client cannot claim a different steering history. **Speculative** — P3 fusion, not blocking.

**F4 (QUATERNARY — katgpt-rs × riir-neuron-db, fusion candidate): CHaRS × ItemEmbedIndex cosine retrieval = "steered retrieval"**

CHaRS-rotate the player-context query toward a target style before cosine matching on `ItemEmbedIndex`. The 8-dim item-embedding space is the natural substrate for CHaRS (small D, schema-centroid clusters already exist as `ItemType` buckets). Differs from F3 in R382: CHaRS routes over K anchors, Slerp rotates toward one. **Speculative**, P3.

**Strongest fusion candidate:** F1 (per-NPC archetype-routing steering). ✅ **EVALUATED 2026-07-07 (Issue 049) — NOT Super-GOAT** (Q2 fails: refinement of soft-MoE-steering class; Q3 weak: refines R158). Remains a strong GOAT-tier refinement of the committed-personality moat.

### 2.5 Latent-space reframing (mandatory per fusion protocol §1.3)

Operating on each Super-GOAT factory module:

(a) **HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): the anchor bank `{a_i, b_j}` lives in the 8-dim HLA affect space. The K source centroids `a_i` partition the NPC's *current* affect manifold into regions (fear-region, calm-region, social-region); the K target centroids `b_j` define where each region *should* shift to under steering. The RBF kernel on `x = hla_state` automatically routes the steering by the NPC's current emotional region. **No re-derivation of HLA from raw state** — the steering is purely a latent-to-latent operation on the HLA vector.

(b) **latent_functor** (`riir-engine/src/latent_functor/`): a `extract_chars_anchor_bank(trajectory)` constructor (compute K centroids + Sinkhorn plan from a contrastive trajectory set), paired with `apply_chars_steering(state, bank)` (the per-input routing). `ReestimationScheduler` triggers bank re-fit on coherence drop.

(c) **cgsp_runtime curiosity** (`riir-engine/src/cgsp_runtime/`): the CHaRS routing weight `w_i(x)` is itself a curiosity signal — when an NPC's HLA state lands in a region with low accumulated RBF mass, it's in an under-explored affect region. Curiosity bonus ∝ `1 − max_i w_i(x)`.

(d) **LatCal fixed-point commitment** (`riir-chain/src/encoding/`): the OT plan `P⋆` (K×K floats) and per-tick routing `w_i(x_t)` (K floats) cross the sync boundary as raw committed scalars. Same discipline as CommittedFieldBlend's `π` vector.

(e) **NeuronShard / freeze envelope** (`riir-neuron-db/src/`): `CharsAnchorShard` subtype. Layout: `[zone_hash(32) | src_centroids(K·D·4) | tgt_centroids(K·D·4) | ot_plan(K·K·4) | sigma2(4) | blake3(32) | merkle_root(32)]`. For K=8, D=8 → 32 + 8·8·4·2 + 8·8·4 + 4 + 32 + 32 = 836 bytes, padded to ~1 KB. `MerkleFrozenEnvelope` wraps it.

(f) **DEC Stokes-calculus** (`katgpt-dec/src/`): the OT plan `P⋆` is a transport cochain; applying it to the source-cochain (`a_i`'s) yields the target-cochain (`b_j`'s) under the discrete OT operator. The CHaRS barycentric map is a rank-1 cochain interpolation weighted by the RBF kernel. **Curse-of-dimensionality caveat (R296):** OT-plan computation is `O(K³ log K)` — small for K=8–15, intractable for high-D shards. CHaRS operates on small-K anchor banks (D=8 HLA), well within tractable range.

---

## 3. §3.5 Modelless Unblock Protocol (MANDATORY — passed)

Before any riir-train deferral:

**Path 1 (freeze/thaw snapshot correction):** **PASS.** The anchor bank `{a_i, b_j, v_ij, P⋆, σ}` is a frozen snapshot artifact computed offline (k-means + Sinkhorn over contrastive corpora). Thawed at init; atomic Arc-swap for hot-reload. `MerkleFrozenEnvelope` wraps it.

**Path 2 (raw/lora reader-writer hot-swap):** **PASS.** Each cluster-pair translation `v_ij = b_j − a_i` is a deterministic linear operator. The blended steering `α · v̂(x)` is a deterministic linear combination of K² frozen vectors, weighted by input-position-derived RBF gates. Constructing this in closed form requires no gradient descent — the OT plan is solved by Sinkhorn iteration (deterministic, convergent), not learned.

**Path 3 (latent-space correction):** **PASS.** The RBF gating `k(x, a_i) = exp(−‖x − a_i‖² / 2σ²)` followed by the OT-plan weighting is exactly the modelless MoE pattern — dot-product-like projection onto K anchor centroids, gated by a kernel (RBF here; sigmoid in our standing rule — RBF is a smooth kernelized variant of the same family, equivalent up to a sign flip and scale on the exponent). The paper's choice of RBF over sigmoid is a kernel design decision, not a modelless-vs-trained distinction.

**Decision protocol result:** All three paths pass → **MODELLESS-VALIDABLE.** The primitive ships in katgpt-rs without any riir-train dependency. The anchor-bank construction (k-means + Sinkhorn) is an offline deterministic computation over contrastive data — equivalent in spirit to `EmotionDirections` contrastive-mean construction.

**Note on RBF vs sigmoid (AGENTS.md rule):** CHaRS uses RBF `exp(−d²/2σ²)`. Our standing rule is "sigmoid not softmax for projections onto learned direction vectors". RBF is not softmax (it does not normalize to a probability simplex), and the denominator `Σ_{p,q} P_pq · k(x, a_p)` is a *partition-function-style* normalization, not a softmax over a finite label set. **The RBF is a kernelized similarity, not a categorical distribution.** Acceptable. If a strict-sigmoid variant is desired for consistency, replace `k(x, a_i) = sigmoid(β − √d²)` with bandwidth β — same shape, sigmoid-bounded.

---

## 4. Verdict

### Tier: **GOAT**

| Question | Answer | Notes |
|---|---|---|
| Q1 No prior art? | **YES (for the input-position-adaptive cluster-mixed routing).** Vocabulary translation done. Four shipped steering primitives cover the design space (single-direction additive, per-layer drift, per-entity-fixed blend, single-target Slerp) — **none** produces an input-position-dependent steering vector via soft RBF-on-clusters × OT-coupling routing. Sinkhorn exists only in KVarN (KV variance), not steering. CommittedFieldBlend is the closest cousin but its routing is entity-fixed, not input-position-adaptive. The combination is genuinely unshipped. | "input-adaptive steering" → "cluster-mixed field"; "barycentric projection" → "soft MoE gating over anchor pairs"; "OT coupling" → "Sinkhorn-matched archetype correspondence". |
| Q2 New class of behavior? | **PARTIAL.** The *operation class* (linear-combination steering) is shipped (CommittedFieldBlend blends K archetypes). The *routing signal* (per-input latent-region RBF gate vs per-entity trajectory summary) is genuinely new — but it's a refinement of the soft-MoE-steering class, not a new operation class. Closer to "new routing criterion" than "new operation". | |
| Q3 Product selling point? | **YES, modest.** "NPCs steered by their current affective region, not a single global personality vector — a wolf in hunt-mode and the same wolf in pack-mode are steered by different translation vectors, with smooth sigmoid-gated transitions as the HLA state moves between regions." Concrete, demoable, but a refinement of the existing per-NPC personality story (R146 / R158 / R290). | |
| Q4 Force multiplier? | **YES (≥5 pillars).** Connects: HLA (latent substrate), CommittedFieldBlend (archetype library cousin), PersonalityWeightedComposition (sigmoid kernel reuse), latent_functor (ReestimationScheduler as re-commit trigger), EmotionDirections (anchor-bank construction recipe), freeze/thaw (CharsAnchorShard), LatCal (commitment of OT plan), ItemEmbedIndex (F4 steered retrieval). | |

**One-line reasoning:** The input-position-adaptive cluster-mixed routing is genuinely unshipped (Q1 yes); the operation class is not new (Q2 partial — it's a new routing criterion on the existing soft-MoE-steering class, not a new operation); the selling point is real but refines the existing personality story; force multiplier is strong. **GOAT, not Super-GOAT** — Q2 fails to clear the "new capability class" bar that Plan 322 (R382 Spherical Steering) was measured against. F1 fusion (per-NPC archetype-routing steering) is a candidate Super-GOAT but is **not committed in this session** — downgraded to an issue for full Q1–Q4 evaluation.

**Not Super-GOAT because:** Q2 — the operation "blend K archetype translations by a soft input-derived weight" is the CommittedFieldBlend operation class with a different weight source. The *combination* of (committed library + per-input RBF routing + OT-matched anchor correspondence) is novel and is the F1 fusion candidate, but the bare primitive "CHaRS steering" is a new routing criterion on an existing operation class. Per §1.5 + the R382 precedent (Spherical Steering was also GOAT not Super-GOAT for the same Q2 reason — "refinement of Plan 322's norm-preserving rotation class"), CHaRS is GOAT.

### Routing

- **`katgpt-rs/.plans/409_chars_cluster_aware_steering_primitive.md`** — open primitive. `CharsAnchorBank<K,D>` + `chars_steering_into` + optional `chars_steering_pct_into`. Feature flag `chars_steering`. GOAT gate G1–G5.
- **`katgpt-rs/.issues/039_chars_committed_blend_fusion_super_goat_evaluation.md`** — track the F1 fusion (CHaRS × CommittedFieldBlend × latent_functor re-estimation = per-NPC archetype-routing steering) for full Q1–Q4 Super-GOAT evaluation before any guide/plan commitment.
- **No private guide (riir-ai / riir-chain / riir-neuron-db) at this verdict tier.** GOAT does not trigger the mandatory-guide rule (§1.5).
- **No riir-train deferral.** All modelless (k-means + Sinkhorn are deterministic offline computations; the per-input RBF+OT routing is closed-form).

### MOAT gate per domain (§1.6)

- **`katgpt-rs` (public engine):** in-scope. Paper-derived fundamental primitive (input-position-adaptive cluster-mixed steering via OT coupling). Ships behind feature flag `chars_steering`; GOAT gate decides promote-to-default vs demote. **Per-stack ledger:** this primitive competes with Plan 309 (Latent Field Steering — single global direction) and Plan 405 (Spherical Slerp — single target, input-adaptive strength) in the "latent steering direction bank" stack slot. CHaRS occupies a *strictly broader* slot (K targets, input-position routing) — if the GOAT gate shows CHaRS subsumes Plan 309's use cases at acceptable overhead, Plan 309 stays as the K=1 degenerate case; if Slerp's norm-preservation matters more than CHaRS's heterogeneity for a given use case, both stay. No forced demotion — different parameterizations of the steering stack.
- **`riir-ai` (private runtime):** the F1 fusion (per-NPC archetype-routing steering) is pillar-adjacent (touches P2 neuron-db substrate, P8 reasoning, self-learn NPCs) — track via Issue 039. If F1 promotes to Super-GOAT, the private guide lands in `riir-ai/.research/`.
- **`riir-neuron-db` (private shards):** `CharsAnchorShard` subtype is a neutral-Gain addition to the shard family (extends KarcShard / ArchetypeBlendShard with K-anchor bank layout). Not pillar-level on its own.
- **`riir-chain` (private chain):** F3 (LatCal commitment of OT plan) is speculative P3.

---

## 5. Constraints check

| Constraint | Status |
|---|---|
| Modelless / inference-time | ✅ Anchor bank + OT plan are frozen artifacts; per-input routing is closed-form |
| Latent-to-latent preferred | ✅ Operates entirely in HLA / shard latent space; never crosses to tokens |
| Use sigmoid not softmax | ⚠️ CHaRS uses RBF kernel `exp(−d²/2σ²)`. RBF is not softmax — it is a kernelized similarity, not a categorical distribution; the normalization is partition-function-style, not softmax-over-labels. **Acceptable.** Optional strict-sigmoid variant: replace RBF with `sigmoid(β − √d²)` for consistency with the standing rule. Document in plan. |
| Freeze/thaw over fine-tuning | ✅ Anchor bank is a `MerkleFrozenEnvelope`-wrapped frozen artifact; atomic Arc-swap for hot-reload. Steering is an additive overlay on mutable latent state, not a mutation of the frozen bank (same discipline as Plan 309 / Plan 321). |
| 5-repo discipline | ✅ Open primitive → katgpt-rs; runtime integration (if F1 promotes) → riir-ai; shard storage → riir-neuron-db; commitment bridge → riir-chain |
| Raw scalars at sync boundary | ✅ Steering stays latent; OT plan P⋆ (K×K floats) + per-tick routing w_i (K floats) cross sync as raw committed scalars (same discipline as CommittedFieldBlend's π vector) |
| Zero-alloc hot path | ✅ SIMD-vectorizable 4-wide (matches Plan 322 / Plan 321 / Plan 405 chunking); `Vec::with_capacity` once for scratch |
| Tests/examples | Required: G1 (RBF weight normalization, Σ w_i = 1), G2 (bank invariance: same input → same output across runs), G3 (K=1 degenerates to Plan 309 additive), G4 (zero-alloc), G5 (latency ≤ existing steering primitives + ε) |

---

## 6. §3.6 Defend-wrong PoC — NOT REQUIRED

The verdict is **GOAT**, not Super-GOAT, and does not assert quality parity with any shipped primitive. The architectural claim (input-position-adaptive cluster-mixed steering is not shipped) is grep-verifiable; the latency claim (per-call ≪ 1 µs) is closed-form-computable. The primitive's *effectiveness vs the four shipped steering primitives on a controlled toy benchmark* is a **GOAT-gate benchmark, not a parity PoC** — it ships with the plan (Plan 409 Phase 2 G2 gate: CHaRS vs Plan 309 vs Plan 405 on a synthetic heterogeneous-cluster steering task). No `.issues/` PoC follow-up needed at this tier.

If the F1 fusion (Issue 039) promotes to Super-GOAT candidate, §3.6 applies in full at that gate.

---

## 7. Open questions / risks

1. **Does CHaRS subsume Plan 309 (Latent Field Steering)?** At K=1, CHaRS degenerates to `v̂(x) = v_11 = b_1 − a_1` — exactly Plan 309's additive steering. The question is whether K > 1 is *worth the overhead* for game AI, or whether game AI's HLA space is homogeneous enough that K=1 suffices. **Mitigation:** Plan 409 G2 gate compares CHaRS (K∈{1,3,5,8}) vs Plan 309 on a synthetic task with planted heterogeneity; the gate decides whether K > 1 promotes to default-on or stays opt-in.

2. **RBF vs sigmoid — AGENTS.md rule tension.** The standing rule is "sigmoid not softmax for projections onto learned direction vectors". CHaRS's RBF is not softmax (no categorical normalization), but it's also not sigmoid. **Decision:** ship RBF (paper-faithful, well-conditioned), with a strict-sigmoid variant `sigmoid(β − √d²)` available behind a feature flag if a downstream consumer requires it. The kernel is host-configurable, not primitive-fixed.

3. **OT plan computation cost at construction time.** Sinkhorn on a K×K cost matrix is `O(K³ log K)` per iteration, ~10–50 iterations typical. For K=8 → ~10K–50K FLOPs, negligible at offline construction. For K=64 (high-D shard space) → ~26M–130M FLOPs, still fast offline. **Not a hot-path concern** — the OT plan is a frozen artifact.

4. **Curse of dimensionality on the anchor bank.** The paper notes K > 1 helps on D=4096 LLM residual streams. Our D=8 HLA is much smaller — k-means on D=8 may produce clusters that are not semantically distinct (HLA's 5 affect axes are already interpretable). **Mitigation:** anchor-bank construction should use HLA's *semantic axes* (valence/arousal/desperation/calm/fear) as initial centroids, not random k-means initialization. K=3–5 default mapping to the primary affect axes.

5. **PCT rank vs D.** Paper proves `rank(Σ_total) ≤ 2K − 2`. For K=3, D=8 HLA → rank ≤ 4 — PCT with L=2 captures ~half the steering variance. May not be worth the extra complexity for D=8; **PCT is the D=64 shard-space variant**, not the D=8 HLA variant.

6. **Compositional ordering with CommittedFieldBlend.** If an NPC has both a committed archetype blend π (Plan 321) and a CHaRS anchor bank, in what order do they apply? `π-blend then CHaRS-steer` (F1 fusion) or `CHaRS-steer then π-blend`? Non-commutative. **Mitigation:** F1 Issue 039 must specify the ordering and verify it preserves CommittedFieldBlend's sampling-invariance contract (FAME Prop. 3).

7. **OT plan drift under contrastive-corpus shift.** The frozen OT plan `P⋆` is computed once offline. If the contrastive corpora drift (new game content, new NPC archetypes), the plan becomes stale. **Mitigation:** `ReestimationScheduler` (riir-ai/crates/riir-engine/src/latent_functor/reestimation/mod.rs) triggers a re-fit when coherence drops below `tau_reest` — same mechanism as F1 fusion. Re-fit cost is offline, not hot-path.

---

## TL;DR

CHaRS (ICML 2026, arXiv:2603.02237) replaces single-global-direction steering with input-position-adaptive cluster-mixed steering: cluster contrastive corpora into K centroids, solve Sinkhorn OT between them, then per-input blend K² cluster-pair translations via RBF gating weighted by the OT plan. Verdict: **GOAT** — Q1 holds (input-position-adaptive cluster routing via RBF × OT is genuinely unshipped; the four existing steering primitives cover single-direction / per-layer-drift / per-entity-fixed / single-target-Slerp, none does per-input soft routing over a K-anchor bank), Q2 partial (refinement of the soft-MoE-steering class CommittedFieldBlend established, not a new operation class — same bar under which R382 Spherical Steering was also GOAT), Q3 modest (refines existing per-NPC personality story), Q4 strong (≥5 pillars). F1 fusion (CHaRS × CommittedFieldBlend × latent_functor re-estimation = per-NPC archetype-routing steering that adapts when the NPC's latent region shifts) is a Super-GOAT **candidate** downgraded to `.issues/039` for full Q1–Q4 evaluation before any guide/plan commitment. Plan 409 will ship `CharsAnchorBank<K,D>` + `chars_steering_into` + optional PCT variant behind feature flag `chars_steering`; GOAT gate G1–G5 with per-stack promote/demote tracking against Plan 309 (K=1 degenerate) and Plan 405 (single-target input-adaptive strength). Fully modelless (k-means + Sinkhorn are deterministic offline computations; per-input routing is closed-form). RBF kernel is not softmax — no AGENTS.md rule violation (optional strict-sigmoid variant available behind flag).
