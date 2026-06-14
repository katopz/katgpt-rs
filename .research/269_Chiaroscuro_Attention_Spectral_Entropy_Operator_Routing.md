# Research 269: Chiaroscuro Attention — Spectral-Entropy Operator Routing

**Date:** 2026-06-14
**Paper:** [Chiaroscuro Attention: Spending Compute in the Dark](https://arxiv.org/pdf/2606.08327) — Sikdar (Accenture), Jun 2026
**Status:** Active — Fusion Research (Modelless First)
**Verdict:** CONDITIONAL GOAT — Direct operator-routing idea is training-bound (collapses during training), but **three inference-time primitives** map cleanly to modelless path with strong fusion value. **CHIAR-KV cache fusion is the GOAT piece** (per-token spectral entropy gating applied at storage layer instead of operator layer).

---

## Paper TL;DR

CHIAR-Former is a 4-layer transformer that routes each token to one of three operators — DCT spectral mixing, RBF kernel mixing, or full self-attention — based on **per-token spectral entropy** H(x), computed from the DCT power spectrum of the token embedding.

Three reusable ideas:

1. **Spectral entropy as per-token complexity signal** — `H(x) = -Σ p_k log p_k / log d` where `p_k = |DCT(x)_k|² / ‖DCT(x)‖²`. Bounded [0,1]. Low = smooth/predictable, high = complex/dynamic.

2. **Routing collapse as discovery mechanism** — when learned routing across structurally distinct operators collapses to a subset, the subset is the sufficient operator set. Removing redundant operators (paper removes RBF) improves quality. This is the opposite of MoE collapse (which is failure).

3. **Operating regime characterization** — spectral routing benefits scale with dataset size and naturalistic token diversity. Loses on small datasets (WikiText-2) and synthetic symbolic tasks (ListOps).

### Headline result (paper)
- DCT+Attn variant: Val PPL **36.54** vs full-attention baseline 66.62 → **45% PPL improvement** at **62.5% fewer attention FLOPs** on WikiText-103
- IMDB: matched baseline within 1.2% at 62.5% fewer attention FLOPs
- WikiText-2: loses (83.81 vs 75.19) — small-data regime
- ListOps: loses (63.35% vs 98.85%) — synthetic regime

### Key formulas

Per-token spectral entropy (Eq. 3):
```
H(x) = -Σ_k p_k · log p_k / log d
     where  p_k = |DCT(x)_k|² / ‖DCT(x)‖²
```

Operator regime (Theorem 1):
```
H(x) < τ_lo      → DCT mixing         (O(d log d)   — spectral tail bounded)
τ_lo ≤ H ≤ τ_hi  → RBF mixing         (O(nRd)       — local kernel approx)
H(x) > τ_hi      → full attention     (O(n²d)       — dynamic cross-token)
```

Tau calibration: τ_lo, τ_hi at 33rd/67th percentile of validation H(x). For WikiText-103 (d=256), tokens cluster in [0.817, 0.903] → τ_lo=0.855, τ_hi=0.865.

Operator utilization entropy `U = -Σ q_o log q_o` — U→0 indicates collapse.

---

## What We Already Have (katgpt-rs)

| Component | Status | Location | CHIAR Relevance |
|-----------|--------|----------|-----------------|
| FreqBandit (token-stream DFT) | ✅ Default-on, GOAT | `src/freq_bandit.rs` | Spectral entropy of **token stream** (temporal). Different signal from CHIAR's per-embedding DCT. |
| SpectralQuant KV cache | ✅ GOAT | `src/spectralquant/spectral_kv_cache.rs` | Per-**dimension** rotation+variable-bit quantization. Does NOT use per-token complexity. |
| Spectral Budget Router (Plan 254) | ✅ GOAT | `src/spectral_budget.rs` | Power laws → NS iteration depth. Different application (Newton-Schulz). |
| Breakeven Complexity Router (Plan 250) | ✅ GOAT | `src/breakeven/` | Cost-aware tier promotion. Natural host for "operator cost vs accuracy" routing. |
| DiagonalGate trait (GDN2 + Wall) | ✅ Shipped | `src/diagonal_gate.rs` | Per-dim gate abstraction — same shape as per-token operator gate. |
| VortexFlow trait (Plan 196) | ✅ GOAT 72/72 | `crates/katgpt-core/src/mux/` | Sparse attention router. Operates at KV-block level. CHIAR operates at operator level. |
| EGA spectral salience | ✅ Plan 139 | `src/ega_attn.rs` | Per-key sigmoid gate from energy. Adjacent to CHIAR but does not switch operators. |
| Parallax local linear attn | ✅ Shipped | `crates/katgpt-core/src/parallax_attn.rs` | Alternative operator (linear-time) — candidate "low-entropy" operator. |
| DashAttention α-entmax | ✅ Plan 106 | `src/dash_attn/` | Adaptive sparse — complementary, not operator-switching. |
| InferenceRouter | ✅ Shipped | `src/inference_router.rs` | CPU/GPU/ANE tier router. Already integrates TriggerGate, trust, RV, breakeven. |
| FFT primitives | ✅ Shipped | `crates/katgpt-core/src/flow/fft.rs` | rustfft-based. Easily extended to DCT. |
| MSA Sparse Attention | ⚠️ GOAT FAILED | Plan 256 | Warning: per-block sparsity failed GOAT. CHIAR must avoid this trap. |

### What is **missing** (the gap CHIAR fills)

1. **Per-token embedding DCT spectral entropy** — we have token-stream DFT (FreqBandit) and per-dim spectral quantization (SpectralQuant), but no per-token embedding complexity signal computed from DCT power spectrum.

2. **Operator-level routing trait** — VortexFlow routes KV blocks, DashAttention routes sparsity, but nothing routes between structurally distinct operators per token.

3. **Routing collapse discovery harness** — we have no automated tool that runs an operator router under load, measures utilization entropy, validates collapse findings, and emits "operator subset promotion" recommendations.

4. **Operating regime gate** — we have BreakevenBandit for tier selection, but no signal that says "this prompt is small/synthetic → skip spectral preprocessing entirely."

---

## The Modelless Constraint: What Can We Actually Do?

CHIAR-Former is a **trained** model. The router is a learned gate (STE-trained). The τ thresholds are post-training calibrated. The DCT filter `w ∈ R^d` is learned.

At inference time, we **do not**:
- Train the router (no gradients, no STE)
- Learn the spectral filter `w` (requires backprop)
- Calibrate τ from validation data (we have no validation pass)

At inference time, we **DO**:
- Compute DCT of any embedding (O(d log d), cheap, rustfft available)
- Compute per-token spectral entropy H(x) (single sum)
- Route tokens to existing operators (EGA, Parallax, TiledAttention, KVarN, SpectralQuant)
- Maintain running statistics of operator utilization entropy U
- Auto-promote/demote operators based on observed collapse
- Switch between "spectral mode" and "full-attention mode" based on prompt characteristics

**The mapping:** CHIAR's core insight — *spend compute only where the token is complex* — maps to inference-time **operator selection** and **KV cache compression strategy**. The "complexity signal" is computed analytically (DCT + entropy), not learned.

---

## Creative Fusion Ideas

### Fusion A: Chiaroscuro KV Cache (CHIAR-KV) — **PRIMARY GOAT CANDIDATE**

**Not in the paper.** Apply the chiaroscuro principle at the **KV cache layer**, not the operator layer.

The paper routes *operators* per token. We route *storage fidelity* per token. The CHIAR-Former paper observes: low-entropy tokens carry smooth, predictable information. The natural extension: **low-entropy KV entries can be aggressively compressed without quality loss; high-entropy KV entries must be retained in full precision.**

```text
For each key k_i in KV cache:
  H(k_i) = spectral_entropy(DCT(k_i))
  if H(k_i) < τ_lo:
      store as DCT-truncated representation (top-K low-freq coefficients)
      reconstruction error bounded by spectral tail (Theorem 1)
  elif H(k_i) < τ_hi:
      store as SpectralQuant variable-bit (existing infra)
  else:
      store as full f16 (StillKV)
```

**Why novel:**
- SpectralQuant operates per-dimension (rotation + variable-bit). CHIAR-KV operates per-token.
- KVarN uses variance across positions. CHIAR-KV uses per-token spectral complexity.
- StillKV uses perceptual compaction. CHIAR-KV uses spectral truncation.
- Combines with all three: CHIAR-KV picks the **storage strategy** per token; existing systems handle the storage mechanics.

**Expected gain:**
- 2-4× KV cache compression on naturalistic text (paper's 62.5% FLOP reduction is the upper bound)
- Zero quality loss on smooth tokens (Theorem 1 guarantees reconstruction error bounded by spectral tail)
- Composes with TurboQuant, KVarN, SpectralQuant — they operate on different axes

**Tau calibration without validation data:**
- Online: maintain running distribution of H(k_i) across recent tokens. Set τ_lo, τ_hi at 33rd/67th percentile of the running window (paper's calibration method, but streaming).
- The band [0.817, 0.903] from the paper is for d=256 BPE — our running calibration adapts to any model/tokenizer automatically.

**Modelless:** Pure inference-time computation. No training, no gradients, no learned filter `w`. The DCT is fixed; only τ adapts (and only via streaming percentile, no learning).

---

### Fusion B: ChiaroscuroOp Trait — Operator-Level Routing Framework

**Direct adaptation of paper's idea.** Define an `ChiaroscuroOp` trait that abstracts structurally distinct operators, and a `ChiaroscuroRouter` that dispatches per token.

```rust
pub trait ChiaroscuroOp {
    /// Spectral complexity lower bound: tokens with H(x) < self.threshold_lo
    /// are eligible for this operator.
    fn entropy_lo(&self) -> f32;
    /// Spectral complexity upper bound.
    fn entropy_hi(&self) -> f32;
    /// Relative cost (FLOPs vs full attention = 1.0).
    fn relative_cost(&self) -> f32;
    /// Apply operator to a single token's embedding, writing into `out`.
    fn forward_token(&self, x: &[f32], out: &mut [f32]);
    /// Apply to a batch (SIMD-friendly).
    fn forward_batch(&self, x: &[&[f32]], out: &mut [&mut [f32]]);
}

pub struct ChiaroscuroRouter {
    ops: Vec<Box<dyn ChiaroscuroOp>>,  // sorted by entropy_lo ascending
    /// Per-op utilization counts (for collapse detection).
    utilization: Vec<u64>,
}

impl ChiaroscuroRouter {
    pub fn route(&mut self, x: &[f32]) -> usize {
        let h = spectral_entropy_dct(x);
        // Pick the highest-cost op whose entropy range contains h.
        // Tie-break: prefer lower-cost op (cheaper is better if eligible).
        let mut chosen = 0;
        for (i, op) in self.ops.iter().enumerate() {
            if h >= op.entropy_lo() && h <= op.entropy_hi() {
                chosen = i;
                // continue scanning — later ops are higher-cost
            } else if h > op.entropy_hi() {
                chosen = i; // fallback to highest-seen
            }
        }
        self.utilization[chosen] += 1;
        chosen
    }

    pub fn utilization_entropy(&self) -> f32 {
        let total: u64 = self.utilization.iter().sum();
        if total == 0 { return 0.0; }
        let mut u = 0.0;
        for &c in &self.utilization {
            if c > 0 {
                let p = c as f32 / total as f32;
                u -= p * p.ln();
            }
        }
        u / (self.ops.len() as f32).ln()
    }
}
```

**Concrete operator set (modelless):**
- `DctMixOp` — rustfft DCT + truncation, O(d log d). Low-entropy.
- `LinearAttnOp` — Parallax local linear attention, O(nd). Medium-entropy.
- `FullAttnOp` — TiledAttention, O(n²d). High-entropy.

**STE replaced by:** hard threshold gate. No gradients needed because we don't train the gate — τ is calibrated online from running H(x) percentile.

**Why this is more honest than the paper:** the paper's "hard threshold" variant already achieves PPL 40.55 (only 4 worse than DCT+Attn's 36.54). So the pure-threshold path is **competitive without learned routing**. We ship the threshold variant.

---

### Fusion C: CollapseDiscoveryHarness — Automated Operator Promotion

**The paper's third contribution.** Implement the harness that:
1. Runs the operator router on a calibration prompt stream
2. Measures utilization entropy U over a sliding window
3. If U → 0 (collapse to subset), validates by re-running with the collapsed subset
4. If quality preserved (or improved), emits "promote subset, demote eliminated ops"
5. Hooks into BreakevenBandit so demoted operators are removed from the cost-aware tier matrix

```rust
pub struct CollapseDiscoveryHarness {
    router: ChiaroscuroRouter,
    window: VecDeque<usize>,           // sliding window of op choices
    collapsed_subset: Option<Vec<usize>>,
    quality_baseline: f32,
}

impl CollapseDiscoveryHarness {
    pub fn observe(&mut self, x: &[f32], quality: f32) {
        let op_idx = self.router.route(x);
        self.window.push_back(op_idx);
        if self.window.len() > 1024 { self.window.pop_front(); }

        let u = self.router.utilization_entropy();
        if u < 0.1 && self.collapsed_subset.is_none() {
            // Collapse detected — identify the surviving ops
            let survivors: Vec<usize> = (0..self.router.ops.len())
                .filter(|&i| self.router.utilization[i] > 0)
                .collect();
            self.collapsed_subset = Some(survivors);
        }
    }
}
```

**This is paper's Remark 1 in code form.** No prior implementation in our codebase. The closest analogue is the GEPA reflective bandit (Plan 164) — but GEPA tunes configs, not operator subsets.

---

### Fusion D: Operating Regime Gate (CHIAR-RegimeGate)

**The paper's fourth contribution.** Switch between "CHIAR mode" (spectral preprocessing on) and "vanilla mode" (full attention only) based on prompt characteristics.

```rust
pub struct ChiarRegimeGate {
    /// Tokens seen in current prompt
    prompt_tokens: usize,
    /// Running variance of H(x) across prompt — low variance = synthetic
    h_variance: WelfordVariance,
    /// Switching threshold
    naturalistic_threshold: f32,
}

impl ChiarRegimeGate {
    /// Returns true if CHIAR spectral preprocessing should be applied.
    pub fn should_apply_chiar(&self) -> bool {
        // Paper's regime: large naturalistic → CHIAR wins
        // Small OR low-variance (synthetic) → vanilla wins
        self.prompt_tokens > 4096
            && self.h_variance.variance() > self.naturalistic_threshold
    }
}
```

**Integrates with:** BreakevenBandit (cost-aware), TriggerGate (QPS-aware), Regime-Transition Inference (Plan 215). Adds the **spectral naturalness** signal that none of these have.

---

### Fusion E: CHIAR + EGA Unification (Operator-as-Gate)

**Cross-pollination.** EGA gates value aggregation by per-key energy. CHIAR routes operators by per-token entropy. Both are sigmoid-gated per-token signals.

Unification: a single `ChiaroscuroEnergyGate` that:
- Computes both E (energy) and H (spectral entropy) in one pass (both are O(d))
- Combines via sigmoid: `gate = σ(α_E·Ẽ + α_H·H̃)` 
- Routes to operator AND scales attention weights simultaneously

This eliminates duplicate projections and amortizes the per-token complexity computation across EGA and CHIAR.

---

## GOAT Verdict

### Scorecard

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| Novelty | 8 | Per-token embedding DCT entropy for KV cache routing is not in any prior plan/research. Operator-level routing trait is also new (VortexFlow is block-level). |
| Applicability (modelless) | 9 | All four fusions are pure inference-time. No gradients, no training, no learned filter `w`. Uses existing rustfft, SIMD, DiagonalGate patterns. |
| Performance | 8 | Paper proves 45% PPL gain at 62.5% fewer FLOPs. CHIAR-KV should achieve 2-4× KV compression. Operator routing should reduce decode FLOPs 30-50% on long naturalistic prompts. |
| Commercial (engine) | 7 | Strengthens katgpt-rs as inference framework. KV cache compression is universally needed. Does not expose game IP. |
| Risk | 6 | MSA failed GOAT (Plan 256) — block-sparse sparsity is dangerous. CHIAR must prove per-token routing is more stable than per-block. ListOps result shows CHIAR can hurt on symbolic tasks — regime gate is essential. |
| Composition | 9 | Orthogonal to TurboQuant, KVarN, SpectralQuant (per-token vs per-dim/per-position). Orthogonal to VortexFlow (operator-level vs block-level). Orthogonal to BreakevenBandit (signal vs cost-aware selection). |

**Overall: 7.8/10 — PROCEED**

### Decision: PROCEED with modelless plan 269

- **Fusion A (CHIAR-KV):** PRIMARY — highest commercial value, lowest risk, composes with everything.
- **Fusion B (ChiaroscuroOp trait):** SECONDARY — enables Fusion C, abstract foundation.
- **Fusion C (CollapseDiscoveryHarness):** TERTIARY — research instrument, ships as example/bench.
- **Fusion D (RegimeGate):** QUATERNARY — gates Fusion B/C activation.
- **Fusion E (EGA unification):** DEFERRED — micro-optimization, only if Fusion A/B prove out.

### MSA Failure Mitigation

Plan 256 (MSA blockwise sparse distillation) failed GOAT. Lessons:
1. Block-level sparsity is too coarse — paper's per-token routing succeeds where block routing fails.
2. CHIAR's regime characterization (loses on synthetic) explains why MSA failed on some benchmarks.
3. CHIAR's routing collapse discovery would have caught MSA's failure mode (low utilization entropy → demote).

**Mitigation:** CHIAR-KV starts opt-in. GOAT gate requires (a) ≥2× KV compression, (b) ≤2% quality regression on long naturalistic, (c) zero quality regression on smooth tokens (Theorem 1 guarantee).

---

## Mapping to Existing Infrastructure

| New Component | Reuses | Adds |
|---------------|--------|------|
| `ChiaroscuroKV` (Fusion A) | `rustfft` (existing via flow/fft.rs), `WelfordVariance` (existing in reward_calibrator), SpectralQuant codebooks (per-dim mechanics), StillKV f16 storage | Per-token H(x) computation, per-token storage strategy dispatch, streaming τ calibration |
| `ChiaroscuroOp` trait (Fusion B) | `DiagonalGate` pattern (per-element abstraction), `InferenceBackend` trait (forward_token method shape) | Operator entropy range, relative_cost, utilization counter |
| `CollapseDiscoveryHarness` (Fusion C) | `WelfordVariance`, `VecDeque` window pattern (existing in osc_kv), BreakevenBandit promotion hooks | Utilization entropy U, survivor detection, validation flow |
| `ChiarRegimeGate` (Fusion D) | `TriggerGate`, `RegimeTransitionInference` (Plan 215) | H(x) variance signal, naturalistic threshold |

### Cost Model

| Component | Per-token overhead | Memory | When amortized |
|-----------|-------------------|--------|----------------|
| Per-token DCT + entropy | O(d log d) ≈ 256·8 = 2K FLOPs | 0 (streaming) | Always (negligible vs O(n²d) attention) |
| CHIAR-KV storage strategy | O(d log d) | +1 byte/token (strategy tag) | Pays off at n > 64 (paper's regime) |
| ChiaroscuroOp routing | O(1) threshold check | +N·8 bytes (utilization counters) | Always |
| Collapse harness | O(1) per token, periodic U recompute | +1024 bytes window | N=1024 tokens between checks |
| Regime gate | O(1) variance update | +16 bytes (Welford) | Always |

---

## Why katgpt-rs (Public Engine) — Not riir-ai

Per Commercial Strategy Verdict (003):

| Element | Public or Private | Reasoning |
|---------|------------------|-----------|
| DCT spectral entropy formula | **Public** | Standard signal processing, in every DSP textbook |
| Per-token H(x) routing trait | **Public** | Generic inference framework mechanism (like ConstraintPruner, SpeculativeGenerator) |
| CHIAR-KV cache compression strategy | **Public** | Generic KV cache optimization — universally needed |
| Collapse discovery harness | **Public** | Research instrument — useful for any multi-operator system |
| Operating regime gate | **Public** | Generic signal (prompt length + variance) |
| Specific τ values for our game models | **Private (riir-ai)** | Game-specific calibration — the HOW |
| Specific operator set ordering for our game domains | **Private (riir-ai)** | Game-specific config |
| Trained τ LoRA adapter (riir-ai side, Plan 123) | **Private (riir-ai)** | Trained weights — never shipped |

**Rule of thumb check:** "This system uses DCT entropy to compress KV cache" = public (capability). "For Bomber AI, τ_lo=0.847, τ_hi=0.862, ops=[DCT→FullAttn]" = private (config).

---

## Open Questions

1. **DCT-II vs DCT-IV** — paper uses Type-II (standard). Verify this is optimal for transformer embeddings vs Type-IV (which has better boundary properties).
2. **Embedding vs hidden state** — paper computes H(x) on the input embedding. Should we compute on residual stream output instead? Likely yes for layers > 0.
3. **Streaming τ calibration stability** — paper calibrates once post-training. Streaming 33/67 percentile over window W — what W? 1024? 8192?
4. **CHIAR-KV reconstruction cost** — DCT-truncated storage requires iDCT on read. Is this faster than the FLOPs saved? Roofline check needed.
5. **Interaction with VortexFlow** — should CHIAR-KV decide per-token storage before or after VortexFlow decides per-block attention? Likely before (storage decision is local, attention routing is cross-token).

---

## Related Research (Cross-References)

| Doc | Connection |
|-----|-----------|
| Research 218 (Breakeven Complexity) | Cost-aware routing — CHIAR's operator cost (`relative_cost`) feeds BreakevenBandit's tier matrix |
| Research 222 (Spectral Scaling Laws) | Power laws for NS depth — different application but same spectral toolkit |
| Research 110 (SpectralQuant) | Per-dim quantization — CHIAR-KV is per-token, composes with SpectralQuant's per-dim |
| Research 169 (Oscillatory SSM) | FreqBandit's spectral band → CHIAR's H(x) entropy. Same toolkit, different signal |
| Research 139 (EGA) | Per-key energy gate — Fusion E unifies with CHIAR |
| Research 196 (VortexFlow) | Block-level sparse routing — CHIAR is operator-level, complementary |
| Research 256 (MSA — FAILED) | Block sparse attention failed GOAT. CHIAR must avoid same trap via per-token granularity |
| Research 215 (Regime Transition) | Operating regime inference — CHIAR-RegimeGate extends with spectral signal |
| Research 075 (S2F Collapse-Aware) | Routing collapse detection in thinking — same pattern, different domain |

---

## TL;DR

CHIAR-Former gives us three reusable inference-time primitives: (1) per-token embedding DCT spectral entropy H(x), (2) operator-level routing trait, (3) routing collapse discovery harness, plus an operating regime characterization. The paper's *direct* operator routing requires training (STE, learned filter `w`), but the *threshold variant* (PPL 40.55 vs 36.54) is competitive without learning — so modelless path is viable.

**The GOAT piece is CHIAR-KV cache (Fusion A):** apply per-token spectral entropy gating at the **storage** layer instead of the **operator** layer. Low-entropy tokens → DCT-truncated storage. High-entropy tokens → full precision. Composes with SpectralQuant (per-dim), KVarN (per-position), StillKV (perceptual). Expected 2-4× KV compression with zero quality loss on smooth tokens.

Proceed with modelless Plan 269. Tau-calibrated LoRA adapter (riir-ai side) deferred to Research 123 if CHIAR-KV proves GOAT.
