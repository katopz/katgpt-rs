# Research 213: Still — Perceiver-Based KV Cache Compaction for Modelless Inference

**Date:** 2026-06-10 (Updated 2026-06-11 — repo alignment audit)
**Source:** [Still: Amortized KV Cache Compaction in a Single Forward Pass](https://arxiv.org/pdf/2606.07878v1) (O'Neill et al., Baseten, 2026)
**Repo:** [shreyansh26/STILL-Towards-Infinite-Context-Windows](https://github.com/shreyansh26/STILL-Towards-Infinite-Context-Windows)
**Target:** katgpt-rs modelless inference engine
**Status:** SHIPPED (Plan 245, opt-in) — DEMOTED LOSER. AM (Plan 271, default-on, GOAT 9/9) won the KV-compaction stack slot. See §2026-07-02 Post-Ship Verdict.

---

## Paper Summary

Still introduces a per-layer Perceiver-style compactor that synthesizes compact KV caches from full-context caches in a single forward pass. Key innovations:

1. **Amortized synthesis**: Learned latent queries cross-attend to full KV cache, producing compact keys/values — no per-context optimization
2. **Position-free compaction**: Un-rotate RoPE before compaction, re-rotate after — eliminates position-content coupling
3. **Iterative chunked compaction**: Recurrent compression with fixed local ratio + 1-chunk lookahead buffer (described in blog, NOT implemented in repo)
4. **β additive attention bias**: Learned per-latent bias injected into frozen model's attention layers to calibrate attention to synthetic latents (CRITICAL — not optional)
5. **Identity-style initialization**: Zero-init projections with normalized bias vectors → near pass-through at training start
6. **Training**: KL + CE loss; best repo run uses CE-only (kl_weight=0.0); blog uses KL+CE on 8×H200

### Repo vs Blog Results (IMPORTANT — do not conflate)

| Metric | Blog (Baseten, 8×H200, KL+CE) | Repo (single GPU, CE-only) |
|---|---|---|
| MCQ Accuracy (1024 latents, 8x compression) | ~85% | **31.5%** |
| vs Cartridge | N/A | 31.5% vs **88.5%** (cartridge wins) |
| Compression sweep | 128-8192 latents (1-64x) | 1024 only |
| Iterative/chunked | Described | Not implemented |
| Cross-domain | 4 domains | Wikipedia only |
| Training data | Not specified | 920 rows (115 articles × 8 MCQ) |
| Trainable params | ~50M (all layers) | ~7M per layer |

**Takeaway**: The repo reproduction is significantly weaker than blog claims. Any gain projections should be tempered. The synthesis-over-selection claim is blog-theory until repo proves it.

---

## Actual Architecture (from repo code)

### Per-Layer Compactor

At each transformer layer `l`, the compactor takes full KV cache `(K_l, V_l) ∈ R^(H×T×d)` and produces:

```
C_l^K ∈ R^(H×t×d)    — compact keys
C_l^V ∈ R^(H×t×d)    — compact values
β_l   ∈ R^(H×t)      — additive attention bias (CRITICAL)
```

### Latent Queries

- **Learned parameter table** `Z ∈ R^(t × 2d)` — double head dim because input is `[unrotated_key | value]`
- Zero-initialized, shared across KV head groups, expanded per head
- The 2d dimension is NOT optional — output heads split it: `key_head = [I | 0]`, `value_head = [0 | I]`

### Perceiver Block (×2 per layer)

```
Block 1 (active_init=True):
  q = apply_rope(q_proj(Z), linspace(0, T-1, t))    ← queries get RoPE at evenly-spaced positions
  k = apply_rope(k_proj([K_free; V]), orig_pos)      ← keys get RoPE at original positions
  v = v_proj([K_free; V])                             ← no RoPE on values
  Z' = Z + out_proj(softmax(q @ k^T / sqrt(d)) @ v)

Block 2 (active_init=False):
  Same structure but zero-init out_proj (only contributes after training)
  Z'' = Z' + SelfAttn(RMSNorm(Z'))
```

**No final RMSNorm** — preserves natural norm variation in compact K/V that the frozen LLM expects.

### Identity Initialization (IMPLEMENTATION-CRITICAL)

```
q_proj.weight = zeros,  q_proj.bias = normalized_unit_vector(q̂)
k_proj.weight = zeros,  k_proj.bias = q̂ × 10  (large constant → content-independent at init)
v_proj = identity  (straight pass-through)
out_proj = identity (block 1) / zeros (block 2)
key_head = [I | 0]  (extracts first d dims of 2d latent)
value_head = [0 | I]  (extracts last d dims of 2d latent)
bias_head = zero-init Linear(2d → 1)
```

This makes keys content-independent at init — only RoPE differentiates positions. Each latent naturally attends to its positionally-nearest input. Without this, training collapses.

### β (Beta) Additive Attention Bias (SHOWSTOPPER IF MISSING)

- Produced by `bias_head(latents)` → scalar per latent per head
- Injected into frozen model's attention layers via monkey-patching
- Broadcast across all query positions, zero-padded for decode-time tokens
- Shifts attention logits over compact slots → model can upweight/downweight latents
- With identity init, replaces manual `log(T/t)` mass-matching offset
- **Any modelless fusion MUST produce a β equivalent** — without it, the frozen model has no way to calibrate attention to synthetic latents

### RoPE Pipeline (Three-Step, NOT Truly Position-Free)

```
Step 1: Un-rotate    K_free = RoPE⁻¹(K)          — strip position encoding
Step 2: Compress     cross-attention uses OWN internal RoPE:
                      • query positions = linspace(0, T-1, t) (evenly spaced)
                      • key positions = original token positions [0, T)
Step 3: Re-rotate    Ck = RoPE(Z_out[:d], linspace(0, T-1, t))  — restore at compact positions
```

**Correction**: The compactor does NOT operate in a truly position-free frame. The cross-attention applies its own RoPE internally (queries at latent positions, keys at original positions). The "position-free" aspect is only about the KV input — un-rotating prevents blending scrambled position encodings across tokens.

---

## Distillation for katgpt-rs (Modelless)

### Constraint Check
- ✅ No LLM training required — Still's Perceiver compactor is a **separate module** from the base model
- ✅ Inference-time only — compaction is a forward pass through the compactor
- ❌ The compactor IS trained (requires KL/CE training against a base model)
- ❌ The β bias is **learned per-layer** — modelless fusion needs a heuristic equivalent
- ✅ But: the *pattern* (cross-attention synthesis, position-free, iterative) is applicable modellessly

### Fusion Idea 1: StillKV — Heuristic Perceiver Compaction (NO TRAINING)

**Core Insight**: Replace Still's *learned* latent queries with *heuristic* query banks generated at inference time.

**Architecture** (CORRECTED):
```
Full KV cache (T tokens)
    → Un-rotate RoPE (strip position from keys)
    → Concat [K_free; V] per head → 2d-dim input
    → Cross-attention from heuristic latent queries (2d-dim):
        - TF-IDF centroids from token frequencies
        - Attention-sink patterns (first 4 + last K)
        - VortexFlow α-entmax routing scores as query importance
        - BFCF region centroids as spatial anchors
    → Self-attention refinement (2 blocks, zero-init residual)
    → Split 2d output → compact Ck (first d), Cv (last d)
    → Compute heuristic β (see below)
    → Re-rotate Ck at linspace(0, T-1, t) positions
```

**Heuristic Latent Query Generation** (replaces learned Z ∈ R^{t × 2d}):
- **Method A**: Cluster-based — run mini-batch k-means on [K_free; V] concatenation for t clusters, use centroids as queries
- **Method B**: Importance-weighted — use existing DashAttention/VortexFlow attention scores to weight token positions, then subsample weighted average
- **Method C**: Spectral — use existing SpectralQuant eigenbasis to project KV cache to top-t eigenvectors
- **Method D**: MUX-Latent superposition — use existing MUX encoder to produce t superposed latent representations

**Heuristic β (Beta) Generation** (replaces learned bias_head — CRITICAL):
- **β-A**: `log(T/t)` mass-matching offset (Still's pre-identity-init baseline) — simplest
- **β-B**: Attention entropy per latent — `sigmoid(entropy_of_attention_weights)` — latents with diffuse attention get higher bias
- **β-C**: Norm ratio — `||Z_out|| / ||mean(KV)||` per latent — higher norm = higher confidence = higher bias
- **β-D**: VortexFlow routing score — use existing α-entmax sparsity scores as proxy for latent importance

**Integration Points**:
- `QuantizedKVCache` trait extension: add `compact_into(&self, budget: usize) -> CompactKVCache`
- `VortexFlow` provides the cross-attention routing mechanism
- `DashAttention` provides the α-entmax sparsity for attention scoring
- `MUX-Latent` provides the vocabulary superposition encoder
- `BFCF` provides region-based spatial partitioning
- `ThoughtFold` provides the iterative refinement through chain folding
- `KVarN` provides the variance normalization for position-free frame

**Position-Free Compaction** (corrected — internal RoPE exists):
1. Un-rotate: apply inverse RoPE to cached keys → `K_free`
2. Concat: `[K_free; V]` → 2d-dim input per token per head
3. Cross-attention with its OWN RoPE:
   - Query positions: `linspace(0, T-1, t)` (evenly spaced compact positions)
   - Key positions: original token positions `[0, T)`
4. Split output: `Ck = Z_out[:, :d]`, `Cv = Z_out[:, d:]`
5. Re-rotate: apply RoPE to `Ck` at `linspace(0, T-1, t)` positions
6. Offset: continuation tokens get position offset = `original_prefix_len - compact_len`

**Iterative Chunked Compaction** (described in blog, NOT in repo):
- Fixed local compression ratio c (e.g., c=8 means compress every 8t tokens to t)
- 1-chunk lookahead buffer (raw KV) between compressed chunks
- Matches existing SegmentCheckpoint's growing memory pattern
- **Risk**: untested in repo — needs separate validation

### Fusion Idea 2: StillCoT — CoT Trace Compaction via Synthesis

**Core Insight**: Apply Still's synthesis compaction to *thinking traces*, not just prefill context.

**Problem**: ThoughtFold prunes CoT steps by *selection* (keep important, discard rest). Still shows synthesis > selection for information preservation (blog claim, not repo-proven).

**Architecture**:
1. Model generates thinking trace (CoT tokens)
2. After trace complete, compact the thinking KV cache via StillKV synthesis
3. The compact thinking trace becomes "compressed working memory"
4. Generation continues against compact trace + prompt + response prefix

**Gain over ThoughtFold**: ThoughtFold achieves 78% CoT reduction via *selection*. StillKV synthesis could achieve similar or better reduction while preserving more distributed information (not just the "important" tokens but blended summaries). **Caveat**: this is theoretical — repo quality is 31.5% vs blog's 85%.

**Integration**:
- Extends `ChainFolder` trait with `compact_trace()` method
- Uses `FoldCache` for KV rollback + `StillKV` for synthesis compaction
- Gated by feature flag `still_cot`

### Fusion Idea 3: StillRSM — Perceiver-Augmented Routing Slot Memory

**Core Insight**: Raven RSM maintains O(1) routing slots. Still's Perceiver can *synthesize* better slot representations from the full KV cache.

**Architecture**:
- Instead of selecting top-K KV entries for RSM slots
- Use cross-attention from t fixed latent queries (2d-dim) to synthesize slot representations
- Each slot becomes a *blended summary* of related KV entries, not a single entry
- Include heuristic β per slot for attention calibration

**Gain**: Higher information density per slot → better routing decisions in O(1) time.

---

## Verdict: GOAT/Gain Analysis (REVISED)

### Fusion 1: StillKV (Heuristic Perceiver Compaction)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ No training needed — heuristic queries replace learned ones |
| Gain vs existing | 🟡 Moderate — MUX-Latent already gives 14-29x TTFT reduction, StillKV adds synthesis quality |
| Novel fusion | ✅ Perceiver pattern + VortexFlow routing + BFCF regions is novel |
| Complexity | 🔴 High — cross-attention + self-attention + RoPE handling + β heuristic is non-trivial |
| Hot path impact | 🔴 Risky — cross-attention is O(t*T), only worth it if t << T |
| Repo proof | 🔴 Weak — repo STILL is 31.5% accuracy, synthesis-over-selection is blog-theory |
| β heuristic | 🟡 Unknown — no prior art on untrained β, needs benchmarking |

**Verdict: GATE HARDER** — Implement behind `still_kv` feature flag. GOAT gate MUST prove:
1. Quality improvement over MUX-Latent selection at same compression ratio
2. Heuristic β produces non-degenerate attention (no collapse, no dominance)
3. No TTFT regression vs MUX-Latent baseline
4. Cross-attention O(t*T) is amortized by reduced decode cost

The synthesis-vs-selection insight is the key differentiator BUT remains unproven. Do not promote to default until GOAT gate passes.

### Fusion 2: StillCoT (CoT Trace Compaction)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ Inference-time compaction of thinking traces |
| Gain vs existing | 🟡 Moderate — ThoughtFold is selection-based, StillCoT is synthesis-based, but repo proof weak |
| Novel fusion | ✅ Still applied to CoT compression is novel |
| Complexity | 🟡 Moderate — reuses StillKV infra (if StillKV exists) |
| Hot path impact | 🟡 Acceptable — compaction happens after trace complete, not during generation |
| Repo proof | 🔴 Weak — depends on StillKV being correct |

**Verdict: GATE** — Downgraded from GAIN. Depends on StillKV GOAT proof. Implement after StillKV proves the synthesis pattern works. Test against ThoughtFold's 78% reduction benchmark.

### Fusion 3: StillRSM (Perceiver-Augmented RSM)
| Criterion | Assessment |
|-----------|-----------|
| Modelless | ✅ Inference-time only |
| Gain vs existing | 🟡 Moderate — better slot quality vs current selection |
| Novel fusion | ✅ Perceiver for routing slot synthesis is novel |
| Complexity | 🟡 Moderate — smaller scale than full StillKV |
| Hot path impact | 🟡 Acceptable — RSM is already O(1) |

**Verdict: DEFER** — Lower priority. Implement if StillKV proves the synthesis pattern works.

---

## GOAT Gate Matrix (StillKV)

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| G1 | Heuristic queries non-degenerate | Latent attention not collapsed to single position |
| G2 | Heuristic β calibrated | Attention to compact slots not uniform or dominated |
| G3 | Quality ≥ MUX-Latent at same compression | Perplexity or downstream task metric |
| G4 | TTFT ≤ MUX-Latent baseline | Wall-clock measurement |
| G5 | Feature isolation — zero perf hurt when disabled | All existing tests pass with `still_kv` off |
| G6 | Sigmoid bounded, no softmax | All gating uses sigmoid |
| G7 | Files < 2048 lines | Rust files stay focused |
| G8 | No final RMSNorm | Preserve natural norm variation |

### GOAT Decision Flow (REVISED)
```
Feature flag ON → Run benchmark suite
  → If G1-G8 ALL pass AND quality > MUX-Latent: PROMOTE TO DEFAULT (GOAT confirmed)
  → If quality ≤ MUX-Latent: KEEP OPT-IN (selection is already good enough)
  → If β degenerate or attention collapse: REVERT, file issue
  → If TTFT regression > 5%: REVERT, synthesis overhead too high
```

---

## Commercial Strategy Alignment

Per the 003 verdict:
- StillKV is **engine** (MIT, open) — modelless inference-time compaction
- If a trained Perceiver compactor is added later, it becomes **fuel** (riir-ai, private SaaS)
- The position-free compaction pattern is pure engineering → stays in engine
- The heuristic query generation strategies are the open-source moat demonstration
- A trained compactor (CE/KL against base model) would be the SaaS premium
- The β heuristic is engine; trained β is fuel

### Engine/Fuel Split:
| Component | Layer | License |
|-----------|-------|---------|
| Position-free RoPE handling (un-rotate/re-rotate) | Engine | MIT |
| Iterative chunked compaction pipeline | Engine | MIT |
| Heuristic latent query generation (A/B/C/D) | Engine | MIT |
| Heuristic β generation (β-A/β-B/β-C/β-D) | Engine | MIT |
| Cross-attention synthesis module | Engine | MIT |
| QuantizedKVCache trait extension | Engine | MIT |
| 2d latent dim handling / split projection | Engine | MIT |
| Trained Perceiver compactor weights | Fuel | Private (riir-ai) |
| KL/CE training pipeline for compactor | Fuel | Private (riir-ai) |
| Learned β per-layer | Fuel | Private (riir-ai) |

---

## Related Work in Our Stack

| Our Feature | Still Analog | Relationship |
|-------------|-------------|-------------|
| MUX-Latent (Plan 238) | Amortized compression | MUX uses vocabulary superposition; Still uses cross-attention synthesis |
| VortexFlow (Plan 196) | Sparse routing | VortexFlow routes tokens; Still routes entire KV cache to latents |
| KVarN (Plan 179) | Variance normalization | KVarN quantizes; Still compresses via synthesis |
| ThoughtFold (Plan 195) | Iterative compaction | ThoughtFold prunes by selection; Still compresses by synthesis |
| BFCF (Plan 213) | Spatial partitioning | BFCF regions → natural heuristic query clusters |
| SegmentCheckpoint (Plan 226) | Growing memory | SegCheckpoint caches segments; Still compresses them iteratively |
| SP-KV (Research 042) | Token utility prediction | SP-KV predicts utility; Still synthesizes from all tokens |
| ShardKV (Research 109) | Asymmetric K/V | ShardKV separates K/V processing; Still does the same |

---

## Key Reference Equations (CORRECTED)

### Per-Layer Compactor (actual architecture)
```
// Input: K ∈ R^(H×T×d), V ∈ R^(H×T×d), positions ∈ [0, T)
// Output: Ck ∈ R^(H×t×d), Cv ∈ R^(H×t×d), β ∈ R^(H×t)

K_free = un_rotate(K, positions)           // Strip RoPE from keys
X = [K_free; V]                            // 2d-dim input per token per head

// Latent queries Z ∈ R^(t × 2d), zero-initialized
latent_pos = linspace(0, T-1, t)           // Evenly spaced compact positions

// Perceiver Block 1 (active_init=True)
q = apply_rope(q_proj(Z), latent_pos)      // Queries get RoPE at compact positions
k = apply_rope(k_proj(X), positions)       // Keys get RoPE at original positions
v = v_proj(X)                              // No RoPE on values
Z' = Z + out_proj(softmax(q @ k^T / sqrt(d)) @ v)  // Cross-attention

// Perceiver Block 2 (active_init=False, zero-init out_proj)
Z'' = Z' + SelfAttn(RMSNorm(Z'))           // Self-attention among latents

// Output (NO final RMSNorm)
Ck = re_rotate(Z''[:, :d], latent_pos)     // First d dims → compact keys, re-rotate
Cv = Z''[:, d:]                            // Last d dims → compact values
β  = bias_head(Z'')                        // Scalar per latent → additive attention bias
```

### Position-Free Compaction (corrected — internal RoPE)
```
K_free = un_rotate(K, positions)  // Strip RoPE from cached keys
X = [K_free; V]                   // 2d-dim concatenation
// Cross-attention applies its OWN RoPE internally:
//   q_pos = linspace(0, T-1, t)  (compact positions)
//   k_pos = original positions    (full positions)
Z_out = PerceiverBlocks(X, Z, q_pos, k_pos)  // 2 perceiver blocks
Ck = re_rotate(Z_out[:, :d], linspace(0, T-1, t))  // Restore RoPE on compact keys
Cv = Z_out[:, d:]                                    // Values stay position-free
β  = heuristic_beta(Z_out)                           // Must produce attention bias
```

### Iterative Chunked Compaction (blog-only, untested)
```
retained_cache = []
for chunk in chunks:
    prefill(chunk, conditioned_on=retained_cache + lookahead_raw)
    compact_chunk = compact(recent_kv_chunk)   // includes β
    retained_cache.append(compact_chunk)
// Total: T/c + c*t entries (linear at rate 1/c)
```

---

## Audit Trail

### 2026-06-11: Repo Alignment Audit
- **Source**: https://github.com/shreyansh26/STILL-Towards-Infinite-Context-Windows
- **Findings**:
  1. β additive attention bias was entirely missing → **CRITICAL FIX** — added to all fusion ideas
  2. Latent dim is 2d (not d) → **CRITICAL FIX** — updated all architecture descriptions
  3. Internal RoPE in cross-attention → **FIX** — corrected "position-free" to acknowledge internal RoPE
  4. Identity init specifics → **ADDED** — zero-init weights, normalized biases, identity projections
  5. No final RMSNorm → **ADDED** — important for implementation
  6. Repo vs blog conflation → **FIX** — separated metrics, tempered expectations
  7. Training loss → **FIX** — best repo run is CE-only, not KL
  8. StillCoT downgraded GAIN → GATE — depends on StillKV proof
  9. StillKV GOAT gate made explicit with 8 criteria

---

## 2026-07-02 Post-Ship Verdict — DEMOTED LOSER (per-stack outcome)

**Per-stack slot:** KV cache compaction.
**Champion:** Attention Matching (Plan 271, Research 233) — DEFAULT-ON, GOAT 9/9.
**Loser:** StillKV (Plan 245) — opt-in, GOAT passes but with lenient thresholds.

### What shipped since this note was written (2026-06-10)

| Event | Date | Impact |
|------|------|--------|
| Plan 245 StillKV shipped (T1–T27 all done) | 2026-06-10→ | All heuristic strategies (ClusterCentroids, AttentionWeighted, SpectralProjection, BfcfRegionBlend, MuxSuperposition) + iterative chunked compaction + ChainFolder `compact_trace` integration. |
| Plan 271 Attention Matching shipped | 2026-06-14→ | Closed-form OMP/HighestAttn selection + NNLS-optimal β + LS-optimal Cv. GOAT 9/9. **Promoted to DEFAULT-ON.** |
| Plan 271 Phase 6 `adaptive_cot_compaction` shipped | 2026-06-14→ | AM-based adaptive CoT compaction (`src/attn_match_adaptive_cot.rs`) — the AM version of StillCoT. Opt-in. |
| StillKV GOAT metric fix (Issue 011) | 2026-06-29 | G1 lib test had a broken metric (cos_sim across RoPE position ranges — unsatisfiable by design, even perfect roundtrip scored 0.31). Replaced with nearest-neighbor avg-cosine-sim. 81/81 pass. |

### Why StillKV lost the slot

1. **G4 explicitly doesn't require beating selection.** The test comment reads: *"Not required to beat selection — synthesis creates new tokens, not copies."* Threshold is cos_sim > 0.1; H2O selection trivially scores ~1.0 (exact copies). The synthesis-vs-selection question this note posed was **never answered by the GOAT gate** — it was sidestepped.
2. **Quality thresholds are lenient.** G1–G3: cos_sim > 0.10/0.10/0.05 at 8×/16×/32×. ClusterCentroids achieves ~0.59 (nearest-neighbor metric) — passes with 5–12× margin, but nearest-neighbor substitutability is a weak proxy for attention-quality preservation.
3. **T26/T27 prove additivity, not superiority.** StillCoT = ThoughtFold (selection) + StillKV (synthesis). The test asserts `stillcot_total_bytes_saved >= thoughtfold_bytes_saved` — trivially true for any compaction that produces output. Does not prove synthesis quality > selection quality.
4. **AM has provably-optimal β and Cv.** AM's NNLS β is the closed-form minimizer of `||A w − m||²`; its LS Cv is the closed-form minimizer of `||X Cv − Y||²_F`. StillKV's heuristic β (mass-matching offset, entropy, norm ratio, VortexFlow) has no optimality guarantee. The AM note (233) explicitly states: *"replaces heuristic β with NNLS-optimal β — direct upgrade path."*
5. **AM GOAT gates are stricter.** G2 (Cv reconstruction < 5% Frobenius error), G3 (OMP mass > 80%), G5 (perplexity within 5%) vs StillKV's cos_sim > 0.1.

### Per-stack outcome (per AGENTS.md promote/demote rule)

| Primitive | Stack slot | GOAT | Default | Outcome |
|-----------|-----------|------|---------|--------|
| **Attention Matching** (Plan 271) | KV compaction | 9/9 PASS (strict) | **DEFAULT-ON** | **CHAMPION** — promoted. |
| **StillKV** (Plan 245) | KV compaction | 7/7 PASS (lenient) | opt-in | **LOSER** — demoted. Kept opt-in for research; AM is the production path. |
| **adaptive_cot_compaction** (Plan 271 P6) | CoT compaction | GATE | opt-in | **CHAMPION** — the AM-based StillCoT successor. |
| **StillCoT** (Plan 245 T25–T27) | CoT compaction | T26/T27 PASS (additive) | opt-in (via `still_kv` + `chain_fold`) | **LOSER** — superseded by `adaptive_cot_compaction`. |

### What stays (shared infrastructure, NOT demoted)

- `BetaBias` type (`katgpt-kv/src/still_kv/beta_bias.rs`) — **shared** between StillKV and AM. AM uses it with NNLS-optimal values; StillKV uses it with heuristic values. The type itself is infrastructure, not a strategy.
- `CompactKVCache` type — shared output container.
- `CompactableKVCache` trait (`src/types.rs`) — shared interface.
- `PositionFreeCompactor` (RoPE un-rotate/re-rotate) — used by both paths.

These remain because AM depends on them. Only the StillKV *compaction strategies* (ClusterCentroids, AttentionWeighted, SpectralProjection, BfcfRegionBlend, MuxSuperposition) and the heuristic β generators are the demoted loser.

### Follow-up (tracked, not blocking)

- [ ] If StillKV is never used in production, consider deprecating the strategy modules while keeping `BetaBias`/`CompactKVCache`/`PositionFreeCompactor` (AM dependencies). Low priority — opt-in feature flags cost nothing when disabled.
- [ ] The synthesis-vs-selection question remains scientifically open. AM (selection + optimal β) is the modelless champion today, but a future paper might produce a modelless synthesis method with provable optimality (like AM's NNLS). Until then, selection wins.

---

## TL;DR

Still's amortized Perceiver KV compaction was distilled into three modelless fusion ideas. **All three shipped** (Plan 245, opt-in). **All three lost** to Attention Matching (Plan 271, DEFAULT-ON, GOAT 9/9) in the KV-compaction stack slot:

1. **StillKV** — SHIPPED, opt-in, **DEMOTE LOSER**. Heuristic Perceiver synthesis (ClusterCentroids etc.) + heuristic β. GOAT passes but G4 explicitly doesn't require beating selection; thresholds lenient (cos_sim > 0.1). AM's NNLS-optimal β + OMP selection won with strict gates (Cv < 5% error, mass > 80%, perplexity within 5%).
2. **StillCoT** — SHIPPED, opt-in, **SUPERSEDED** by `adaptive_cot_compaction` (Plan 271 Phase 6). T26/T27 prove additivity (trivially true), not superiority.
3. **StillRSM** — DEFER (unchanged, independent of AM).

**Shared infrastructure stays**: `BetaBias`, `CompactKVCache`, `CompactableKVCache`, `PositionFreeCompactor` — AM depends on these. Only the heuristic strategy/β modules are the demoted loser.

**Key corrections from repo audit** (2026-06-11, still valid): (1) β bias is a showstopper — AM resolved it via NNLS, not heuristics; (2) latent dim is 2d not d; (3) cross-attention has internal RoPE; (4) Still repo quality is 31.5% vs blog's 85%. **The synthesis-over-selection insight remains unproven** — AM (selection) won the slot without needing to prove synthesis is better.
