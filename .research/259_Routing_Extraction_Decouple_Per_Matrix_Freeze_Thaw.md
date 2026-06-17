# Research 259: Routing-Extraction Decoupling — Per-Matrix Freeze/Thaw Composite (QK-Restore)

> **Source:** [Attention Amnesia in Hybrid LLMs: When CoT Fine-Tuning Breaks Long-Range Recall, and How to Fix It](https://arxiv.org/abs/2606.11052) — Xinyu Zhou, Boyu Zhu, Yi Xu, et al. (HKUST-GZ / UCL / Mistral / Tsinghua / SUTD), 2026-06-09
> **Date:** 2026-06-17
> **Status:** Active — **Super-GOAT verdict** (all 4 novelty gates pass)
> **Classification:** Public (katgpt-rs open primitive — generic per-matrix composite math)
> **Related Research:** 004 (LoRA Architecture), 145 (Wall Attention), 028 (HLA), 070 (GDN2), 100 (EGA), 156 (Weight Isolate Extension), 165 (Q/K/V Projection Sharing), 201 (RAT+ Train Dense Infer Sparse), 227 (GPart)
> **Related Plans:** TBD (Plan 287 — per-matrix freeze/thaw primitive, pending guide finalization)
> **Cross-ref (riir-ai):** Research 138 (QK-Restore Surgical Adapter Composition Guide — the private selling-point doc)
> **Training redirect:** CoT-SFT analysis, gradient locality proofs, QK-Frozen preventive training → riir-train. This note distills only the inference-time per-matrix composite primitive.

---

## TL;DR

The paper proves a **Routing-Extraction Decoupling Theorem**: under any training that exhibits gradient locality (e.g., CoT-SFT where Markov structure causes exponentially decaying gradient magnitude with token distance), the attention weight matrices separate into two functional groups:
- **Routing parameters** (W_Q, W_K): control *where* attention looks. Receive only short-range gradient signal → drift toward local patterns under locality-biased training.
- **Extraction parameters** (W_V, W_O): control *what* content is retrieved. Receive uniform-bounded gradient signal → accumulate task-specific improvements regardless of distance.

The fix — **QK-Restore** — is training-free: transplant W_Q, W_K from a pre-drift snapshot while keeping post-drift W_V, W_O. This recovers long-range routing while preserving task-specific extraction. The paper validates: HypeNet-9B NIAH-S2@256K recovers from 9.4% → 44.0% (+34.6) with zero training cost.

**Distilled for katgpt-rs (modelless, inference-time):** The per-matrix freeze/thaw composite is a **new capability class** for our freeze/thaw runtime. Instead of swapping whole snapshots, we can now surgically compose: routing (QK) from snapshot A + extraction (VO) from snapshot B. This enables "routing-preserving adapter composition" — preserve long-range retrieval geometry from a base checkpoint while adopting domain-specific extraction from a specialist adapter. The Procrustes variant solves for the rotation that preserves the routing gramian R = W_Q · W_K^T while minimizing deviation.

**Commercial selling point (private, in riir-ai guide):** "Our NPC runtime can surgically compose adapters — preserve an NPC's long-term relational routing (who they know, where they've been) while swapping in new skill extraction (combat abilities, dialogue styles) — zero retraining, atomic, BLAKE3-committed."

---

## 1. Paper Core Findings

### 1.1 CoT-Markov Gradient Locality (Theorem 4.6)

Under a Markov model of CoT reasoning (latent states z_t with spectral gap 1−ρ), the expected gradient magnitude on attention logits decays exponentially:

```
g(τ) := E[|∂L/∂e_{t,t-τ}|] ≤ C' · e^{-τ/W},   W = -1/log(ρ)
```

Empirically validated: CoT text has W_corr ≈ 172 tokens (high self-similarity), but attention gradient decay W_grad ≈ 115 tokens. The gap (W_grad < W_corr) means there's a band where data demands long-range attention but training gradient doesn't reinforce it.

### 1.2 Routing-Extraction Gradient Decoupling (Theorem 5.1)

Under CoT-SFT with gradient locality:

| Parameter group | Function | Gradient bound |
|----------------|----------|----------------|
| W_Q, W_K (routing) | Control *where* to attend | E[‖∂L/∂e · k · h^T‖_F] ≤ C_R · ρ^τ (decays) |
| W_V, W_O (extraction) | Control *what* to retrieve | E[‖∂L/∂v‖] ≥ δ_A · c_G > 0 (uniform) |

**Key insight:** CoT-SFT simultaneously corrupts W_Q/W_K (routing drifts local) and improves W_V/W_O (extraction gains task skill). These effects are segregated into disjoint parameter sets.

### 1.3 QK-Restore (Algorithm 1 — Training-Free)

```python
def qk_restore(theta_pre, theta_post, L_attn):
    theta_rep = copy(theta_post)
    for l in L_attn:
        theta_rep[l].W_Q = theta_pre[l].W_Q  # restore routing
        theta_rep[l].W_K = theta_pre[l].W_K  # restore routing
        # W_V, W_O stay from theta_post (keep extraction)
    return theta_rep
```

### 1.4 Procrustes Variant (QK-Pro)

Full QK-Restore discards both harmful routing drift AND beneficial math adaptation entangled in W_Q. QK-Pro solves:

```
min_{W_Q^new, W_K^new} ‖W_Q^new - W_Q^post‖_F + ‖W_K^new - W_K^post‖_F
s.t. W_Q^new · (W_K^new)^T = R^pre   // preserve routing gramian
```

Linearized (fix W_K^new = W_K^pre, solve for W_Q^new via Lagrange multipliers):
```
W_Q^new = W_Q^post + (R^pre − W_Q^post · (W_K^pre)^T) · (W_K^pre · (W_K^pre)^T + λI)^{-1} · W_K^pre
```

### 1.5 Results

| Model | Method | NIAH-S2@256K | NIAH-S3@256K | MATH500 |
|-------|--------|-------------|-------------|---------|
| HypeNet-9B | Pre-train | 67.2 | 52.0 | 15.3 |
| HypeNet-9B | +SFT | 9.4 | 22.8 | 62.3 |
| HypeNet-9B | +QK-Restore | **44.0** (+34.6) | **42.6** (+19.8) | 59.3 (−3.0) |

Key: routing recovery is massive (+34.6), extraction cost is small (−3.0).

### 1.6 Ablation Insights

- **V-Restore fails** — restoring W_V doesn't recover routing (confirms decoupling)
- **Q-only or K-only** — partial recovery, joint QK needed for coherent routing geometry
- **QK-Frozen** (preventive: freeze QK during SFT) — underperforms QK-Restore (post-hoc). Unconstrained SFT allows better W_V adaptation.

---

## 2. Distillation — The Open Primitive

### 2.1 Per-Matrix Snapshot Composite

The generic primitive (no game semantics, no training):

```rust
/// Per-matrix freeze/thaw composite.
///
/// Given two weight snapshots A (routing source) and B (extraction source),
/// produce a composite C where:
/// - C.W_Q = A.W_Q  (routing from A)
/// - C.W_K = A.W_K  (routing from A)
/// - C.W_V = B.W_V  (extraction from B)
/// - C.W_O = B.W_O  (extraction from B)
///
/// Validated by Routing-Extraction Decoupling (arXiv:2606.11052 Theorem 5.1):
/// QK controls routing (where), VO controls extraction (what).
/// These are functionally independent under gradient-locality training.
pub fn per_matrix_composite(
    snapshot_a: &WeightSnapshot,  // routing source (e.g., pre-drift base)
    snapshot_b: &WeightSnapshot,  // extraction source (e.g., post-SFT specialist)
    attn_layers: &[usize],        // which layers to composite
) -> WeightSnapshot {
    let mut composite = snapshot_b.clone();  // start from B (extraction)
    for &layer in attn_layers {
        composite.weights[layer].w_q = snapshot_a.weights[layer].w_q.clone();
        composite.weights[layer].w_k = snapshot_a.weights[layer].w_k.clone();
        // w_v, w_o stay from snapshot_b
    }
    composite.recompute_blake3();
    composite
}
```

### 2.2 Procrustes-Aligned Composite

When full QK transplant is too aggressive (discards beneficial adaptation entangled in W_Q), use Procrustes alignment to preserve the routing gramian while minimizing deviation:

```rust
/// Procrustes-aligned per-matrix composite.
///
/// Preserves routing geometry R = W_Q · W_K^T from snapshot A,
/// while keeping W_Q as close as possible to snapshot B's W_Q.
///
/// Solves: min ‖W_Q^new - W_Q^B‖  s.t.  W_Q^new · (W_K^A)^T = R^A
/// Via: W_Q^new = W_Q^B + (R^A − W_Q^B · (W_K^A)^T) · (W_K^A · (W_K^A)^T + λI)^{-1} · W_K^A
pub fn procrustes_composite(
    snapshot_a: &WeightSnapshot,  // routing geometry source
    snapshot_b: &WeightSnapshot,  // extraction + math adaptation source
    attn_layers: &[usize],
    lambda: f32,                  // regularization (ridge)
) -> WeightSnapshot {
    let mut composite = snapshot_b.clone();
    for &layer in attn_layers {
        let w_q_a = &snapshot_a.weights[layer].w_q;
        let w_k_a = &snapshot_a.weights[layer].w_k;
        let w_q_b = &snapshot_b.weights[layer].w_q;

        let r_a = w_q_a * w_k_a.transpose();  // routing gramian to preserve
        let wk_at_wka = w_k_a * w_k_a.transpose();
        let regularized = wk_at_wka + lambda * Matrix::identity();
        let residual = r_a - w_q_b * w_k_a.transpose();
        let correction = residual * regularized.inverse() * w_k_a;
        composite.weights[layer].w_q = w_q_b + correction;
        composite.weights[layer].w_k = w_k_a.clone();
    }
    composite.recompute_blake3();
    composite
}
```

### 2.3 Routing-Extraction Decoupling Trait

```rust
/// Trait for attention layers that support routing-extraction decoupling.
///
/// Implementors separate their parameters into:
/// - Routing: controls *where* attention looks (Q, K projections)
/// - Extraction: controls *what* content is retrieved (V, O projections)
pub trait RoutingExtractionLayer {
    /// Extract routing parameters (W_Q, W_K) from this layer.
    fn routing_params(&self) -> RoutingParams;

    /// Extract extraction parameters (W_V, W_O) from this layer.
    fn extraction_params(&self) -> ExtractionParams;

    /// Replace routing parameters (for QK-Restore composite).
    fn set_routing(&mut self, routing: &RoutingParams);

    /// Replace extraction parameters (for VO swap).
    fn set_extraction(&mut self, extraction: &ExtractionParams);

    /// Compute routing gramian R = W_Q · W_K^T (for Procrustes).
    fn routing_gramian(&self) -> Matrix {
        let r = self.routing_params();
        r.w_q * r.w_k.transpose()
    }
}
```

### 2.4 Applicability Matrix

| Attention type | Routing params | Extraction params | QK-Restore applicable? |
|---------------|---------------|-------------------|----------------------|
| Softmax attention | W_Q, W_K | W_V, W_O | ✅ Direct (paper's setup) |
| HLA (linear) | W_Q, W_K (state routing), W_K (storage routing) | W_V (stored content) | ✅ Yes — W_K determines what enters state, W_Q determines what's retrieved |
| GDN2 (gated linear) | W_Q, W_K + gate params | W_V | ✅ Yes — gates are routing (erase/write) |
| Wall Attention | W_Q, W_K + gate g_t | W_V, W_O | ✅ Yes — gates are routing (decay) |
| EGA (energy-gated) | W_Q, W_K + w_proj (energy) | W_V, W_O | ✅ Yes — w_proj is routing energy |
| MoE / dMoE | Router weights | Expert weights | ⚠️ Different — router IS pure routing, experts ARE pure extraction |

---

## 3. Novelty Gate — All 4 YES → Super-GOAT

### Q1: No prior art?
**YES.** Vocabulary-translated grep across katgpt-rs + riir-ai (notes + code) for: "selective transplant", "per-matrix freeze", "QK composite", "Procrustes", "routing geometry", "routing extraction decouple". Zero hits. Our `snapshot.rs` swaps whole snapshots only. Research 165 (Q/K/V projection sharing) is about *efficiency* (share projections to halve KV cache), not about *selective transplant* (preserve routing while swapping extraction). The HLA "LoRA Orthogonal Swap" note (R028) is explicitly DEAD because "HLA state is weight-dependent" — that's about KV cache state, not about weight matrix transplant.

### Q2: New class of behavior?
**YES.** Per-matrix composite enables "routing-preserving adapter composition" — a capability that whole-snapshot swap fundamentally cannot provide. Example: compose routing from a long-context base checkpoint (good at finding needles in haystack) with extraction from a domain specialist adapter (good at math reasoning). The result has BOTH long-range retrieval AND domain expertise. This is impossible with whole-snapshot swap (you get either routing OR extraction, not both).

### Q3: Product selling point?
**YES.** "Our NPC runtime surgically composes adapters — preserve an NPC's long-term relational routing (who they know, where they've been, who they fear) while swapping in new skill extraction (combat abilities, dialogue styles, crafting recipes). Zero retraining, atomic, BLAKE3-committed." This is a private moat for riir-ai's game AI product. See riir-ai/.research/138 for the full guide.

### Q4: Force multiplier?
**YES.** Connects to ≥3 existing pillars:
1. **Freeze/thaw runtime** (snapshot.rs) — per-matrix composite is a new operation in the snapshot lifecycle
2. **Adapter routing** (polytope_router.rs, dMoE) — composite enables routing-aware adapter selection (pick the adapter whose extraction best matches, then composite with the base's routing)
3. **EGA / energy-gated attention** — EGA's energy score IS a routing signal; QK-Restore preserves it
4. **Raw-vs-latent boundary** — routing geometry (R = W_Q · W_K^T) is a latent-space object that must be preserved across adapter swaps for anti-cheat consistency

---

## 4. Fusion Ideas

### F1: QK-Restore × Freeze/Thaw — Atomic Per-Matrix Hot-Swap

Current freeze/thaw swaps whole snapshots atomically (Arc swap). Fuse: enable per-matrix atomic swap — readers see either full-A or full-B or composite(QK_A, VO_B), never a torn state. Requires BLAKE3 commitment on the composite, not just the source snapshots.

### F2: QK-Restore × Polytope Router — Routing-Aware Adapter Selection

Polytope router selects between adapters by latent-state similarity. Fuse: instead of selecting one adapter, COMPOSITE — take routing from the closest base checkpoint, extraction from the best-matching specialist. The router now outputs a (routing_source, extraction_source) pair, not a single adapter.

### F3: QK-Restore × EGA — Energy-Gate Preservation

EGA's energy gate `g = σ(α · (ẽ − τ))` is a routing signal computed from W_Q, W_K via the energy projection. When compositing, preserve EGA's energy statistics by keeping the pre-composite energy checkpoint. The energy score distribution should not drift when only extraction is swapped.

### F4: QK-Restore × HLA — State-Compatible LoRA Swap

HLA state S = Σ(W_K · h)(W_V · h)^T is weight-dependent (computed in W_K basis). Naive adapter swap invalidates stored state. Fuse: if new adapter preserves W_K (routing-preserving composite), the stored HLA state remains valid — no state flush needed on adapter hot-swap. **This is huge for MMORPG-scale NPC AI: thousands of NPCs with HLA state can hot-swap extraction adapters without flushing their relational memory.**

### F5: QK-Restore × Procrustes × Manifold Power Iteration (R246)

Manifold Power Iteration (R246) finds the dominant routing direction in MoE routers. Fuse: use the dominant routing direction as the Procrustes constraint — preserve the top-k routing eigenvectors while allowing the rest to adapt. This gives a "spectral routing preservation" that's softer than full gramian preservation.

---

## 5. What Stays Where (3-Repo Discipline)

| Component | Repo | Why |
|-----------|------|-----|
| Per-matrix composite math | katgpt-rs (MIT) | Generic linear algebra, no game semantics |
| Procrustes alignment solver | katgpt-rs (MIT) | Generic optimization |
| RoutingExtractionLayer trait | katgpt-rs (MIT) | Generic trait |
| CoT-SFT gradient analysis, QK-Frozen training | riir-train (private) | Training know-how |
| Game-side composition policies (which NPC gets which routing/extraction) | riir-ai (private) | Game IP |
| HLA state preservation under routing-preserving swap | riir-ai (private) | Runtime IP |
| Polytope router integration (routing-aware composite selection) | riir-ai (private) | Runtime IP |

---

## 6. Validation Protocol (GOAT/Super-GOAT Gate)

### G1: Composite Correctness
- Given two snapshots A, B, the composite C = per_matrix_composite(A, B) must have C.W_Q == A.W_Q, C.W_K == A.W_K, C.W_V == B.W_V, C.W_O == B.W_O for specified layers.
- BLAKE3 of C must be deterministic (same A, B → same C hash).

### G2: Routing Preservation
- On needle-in-haystack retrieval: composite(pre-train QK, post-SFT VO) must recover ≥80% of pre-train NIAH accuracy while retaining ≥90% of post-SFT task accuracy.

### G3: Procrustes Alignment Quality
- QK-Pro composite must preserve routing gramian: ‖R^new − R^pre‖_F < ε.
- QK-Pro must deviate less from post-SFT than full QK-Restore: ‖W_Q^pro − W_Q^post‖ < ‖W_Q^restore − W_Q^post‖.

### G4: HLA State Compatibility
- After routing-preserving adapter swap (W_K unchanged), HLA state S must produce identical readout for the same query. (State is computed in W_K basis; if W_K is preserved, state is valid.)

### G5: Atomicity
- Readers must never see a torn composite (partial QK from A, partial VO from B). Use Arc swap with pre-computed composite.

### G6: Game-Side (riir-ai)
- NPC personality composite: preserve relational routing (grudge memory, friendship) while swapping skill extraction (combat style). NPC behavior must show retained relationships + new skills.

---

## 7. Implementation Priority

| Priority | Task | Repo | Gate |
|----------|------|------|------|
| P0 | `RoutingExtractionLayer` trait + `per_matrix_composite` | katgpt-rs | G1 |
| P0 | `procrustes_composite` solver | katgpt-rs | G3 |
| P1 | Integrate into `snapshot.rs` as new snapshot operation | katgpt-rs | G5 |
| P1 | HLA state compatibility test (W_K preserved → state valid) | katgpt-rs | G4 |
| P2 | riir-ai guide implementation (NPC composition policies) | riir-ai | G6 |
| P2 | Polytope router integration (routing-aware composite selection) | riir-ai | — |
| P3 | Procrustes × Manifold Power Iteration fusion (spectral preservation) | katgpt-rs | — |

---

## TL;DR

**Verdict: Super-GOAT.** Attention Amnesia's Routing-Extraction Decoupling Theorem + QK-Restore is a **new capability class** for our freeze/thaw runtime: per-matrix snapshot composition that preserves routing (W_Q, W_K) from one source while adopting extraction (W_V, W_O) from another. All 4 novelty gates pass: no prior art (grep confirmed), new class (routing-preserving adapter composition is impossible with whole-snapshot swap), clear selling point (NPC relational memory preserved across skill adapter swaps), force multiplier (freeze/thaw + adapter routing + EGA + HLA state compatibility). The open primitive (per-matrix composite + Procrustes alignment) goes to katgpt-rs. The private guide (surgical adapter composition for game NPCs, HLA state preservation, polytope router integration) goes to riir-ai/.research/138. Mandatory outputs created in this session per Super-GOAT protocol. The headline fusion: F4 (HLA state-compatible LoRA swap) — routing-preserving composite means W_K is unchanged, so HLA state S = Σ(W_K·h)(W_V·h)^T remains valid across extraction adapter hot-swaps, enabling thousands of NPCs to gain new skills without flushing relational memory.
