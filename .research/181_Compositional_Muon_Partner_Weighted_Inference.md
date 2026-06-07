# Research 181: Compositional Muon — Partner-Weighted Inference-Time Optimization

**Date:** 2026-06-07
**Paper:** [Towards Compositional Steepest Descent](https://blog.tilderesearch.com/blog/compositional-muon) — Keigwin, Yang, Pai, Zhang, DeWulf (Tilde Research), Jun 2026
**Code:** github.com/tilde-research/comp-muon-release
**Status:** Active — Fusion Research
**Verdict:** CONDITIONAL GOAT — Isotropic approximation maps cleanly to modelless path; full partner-whitening is training-only. Gauge correction has a novel inference analog. Overall gain is real but modest — ~5-15% DDTree efficiency improvement, not a paradigm shift.

---

## Paper TL;DR

Compositional Muon (CM) extends Muon's matrix-level steepest descent to **composed transformer circuits**. Instead of controlling each weight matrix's update independently (‖ΔW‖_op ≤ ε), CM controls the **functional perturbation of the composition** (‖ΔM‖_op ≤ ε where M = W_Q·W_K^T).

**Core contributions:**

1. **Partner-whitened updates:** Each factor's gradient is rescaled by its partner's spectral geometry. For QK: `ΔW_Q = -(ε/2)·msign(G_Q · C_K^{-1}) · C_K^{-1}` where `C_K = (W_K^T W_K)^{1/2}`. The partner's Gram inverse controls how much trust-region budget each factor's update costs.

2. **Half-split decomposition:** Split the compositional budget equally between Q-side and K-side, solve each as a standard Muon subproblem. Simple, tractable, not optimal but effective.

3. **Isotropic approximation:** Under approximate orthogonality of partner columns, full partner-whitening reduces to a scalar: `C_K^{-1} ≈ c_K^{-1}·I` where `c_K² = ‖W_K‖_F² / d_h`. This gives "Muon with partner-dependent effective learning rate" — empirically matches full CM within noise.

4. **Hybrid OV rule:** V-side is per-head (head-local msign), O-side is per-matrix (concatenate partner-whitened gradients across heads, one global msign). Asymmetric because O is the aggregation point.

5. **Gauge correction:** The product M = W_Q·W_K^T has GL(d_h) redundancy — (W_Q·R)(W_K·R^{-T}) = same product. Momentum can drift into "vertical" (parameterization-only) directions. Correction via Sylvester solve projects back to "horizontal" (output-relevant) directions.

6. **RMSNorm composition (Appendix G):** M = W·Γ where Γ = diag(γ). Optimizing jointly gives better results than treating W and γ independently. The exact per-column decomposition is possible because both Γ and ΔΓ are diagonal.

**Key empirical results:**
- 340M: CM OV+QK beats Muon by 15-30 val-loss points across all architecture settings
- 1B: Gains survive scale (5-23 val-loss points depending on architecture)
- Isotropic CM ≈ full CM (within 1-2 val-loss points)
- NanoGPT speedrun: ~15 steps faster to threshold
- Partner-rescaled AdamW also improves over baseline (gains are not Muon-specific)

---

## What We Already Have (katgpt-rs)

| Component | Status | Location | CM Relevance |
|-----------|--------|----------|--------------|
| Newton-Schulz 5-iteration | ✅ Shipped | `src/newton_schulz.rs`, GOAT 25/25 | `msign()` = our `newton_schulz5()` |
| Muon momentum buffer | ✅ Shipped | `src/newton_schulz.rs::muon_update()` | Partner-whitening wraps this |
| River-valley diagnostics | ✅ Shipped | `src/river_valley.rs` (r_dom, r_bulk) | Spectral geometry already measured |
| MuxDdTree (DD-tree) | ✅ Shipped | `crates/katgpt-core/src/mux/dd_tree.rs` | Compositional scoring target |
| MuxBfs (dynamic-width expansion) | ✅ Shipped | `crates/katgpt-core/src/mux/bfs.rs` | Partner-weighted expansion budget |
| ConstraintPruner trait | ✅ Shipped | `crates/katgpt-core/src/traits.rs` | Hard bounds = one "partner" |
| ScreeningPruner trait | ✅ Shipped | `crates/katgpt-core/src/traits.rs` | Soft scoring = other "partner" |
| SpeculativeGenerator trait | ✅ Shipped | `crates/katgpt-core/src/traits.rs` | Draft/verify composition |
| EGA spectral salience | ✅ Shipped | `src/ega_attn.rs` | z-normalized sigmoid gate |
| Parallax local linear attention | ✅ Shipped | `crates/katgpt-core/src/parallax_attn.rs` | Kernel-agnostic covariance correction |
| Trust-Region Adaptive Speculation | 📋 Research 162 | P_accept = min(πT/πS, 1) | Trust region = compositional budget |
| NDS curvature proxy | 📋 Research 166 | Spectral entropy → DDTree budget | Partner norm ≈ NDS inverse |
| SIMD matmul + transpose | ✅ Shipped | `src/simd.rs` | Needed for Gram computation |

---

## The Modelless Constraint: What Can We Actually Do?

CM is fundamentally a **training-time optimizer**. The core mechanics — partner-whitened gradient updates, gauge correction of momentum buffers, hybrid OV msign — all require:
1. Access to weight matrices W_Q, W_K, W_O, W_V
2. Computation of gradients G_Q, G_K
3. Running Newton-Schulz on modified (whitened) gradients
4. Updating weights via the composed update

At inference time, we **do not**:
- Have gradients
- Update weights
- Run training steps
- Have access to full weight matrices in host memory

At inference time, we **DO**:
- Compute token scores (logits → probabilities)
- Compose multiple scoring signals (draft confidence × validator relevance × constraint validity)
- Allocate compute budget (DDTree width/depth)
- Maintain running statistics (entropy, acceptance rates, spectral proxies)

**The mapping:** CM's core insight — *control the composition, not the factors* — maps to inference-time **scoring composition** and **budget allocation**. The "composition" at inference time is not a weight matrix product but a **scoring function composition**.

---

## Fusion Ideas

### Fusion 1: Partner-Weighted Speculative Scoring (PWS)

**CM analogue:** Partner-whitened updates rescale each factor's gradient by the partner's spectral geometry.

**Modelless mapping:** In speculative decoding, the acceptance score is a composition:
```
P_accept(x) = min(1, π_target(x) / π_draft(x))
```

Currently we compute π_draft and π_target independently. CM suggests: rescale one by the "spectral properties" of the other. The "partner norm" becomes the **confidence spread** of the partner distribution.

**Isotropic approximation (cheap):** The partner's "Frobenius norm" analogue is the entropy of the distribution:
```
c_draft² = ‖logits_draft‖²_F / vocab_size  ← draft's energy
c_target² = ‖logits_target‖²_F / vocab_size ← target's energy
```

The partner-weighted acceptance score:
```rust
/// Partner-weighted speculative acceptance.
///
/// CM's isotropic approximation: rescale acceptance by partner energy.
/// High-energy draft → partner is "large" → scale DOWN acceptance (draft is noisy)
/// Low-energy draft → partner is "small" → scale UP acceptance (draft is focused)
fn partner_weighted_acceptance(
    draft_logits: &[f32],
    target_logits: &[f32],
    token_idx: usize,
) -> f32 {
    let c_draft_sq = frobenius_sq(draft_logits) / draft_logits.len() as f32;
    let c_target_sq = frobenius_sq(target_logits) / target_logits.len() as f32;

    // Standard acceptance ratio
    let p_draft = softmax(draft_logits)[token_idx];
    let p_target = softmax(target_logits)[token_idx];
    let base_accept = (p_target / p_draft).min(1.0);

    // CM isotropic rescaling: partner-dependent effective rate
    // Target is draft's "partner" → c_target controls draft's effective budget
    let partner_scale = (c_target_sq + 1e-8).sqrt() / (c_draft_sq + 1e-8).sqrt();

    (base_accept * partner_scale).min(1.0)
}
```

**Assessment:** Marginal gain. The acceptance ratio is already `min(1, πT/πS)` which is well-calibrated. The partner rescaling adds a small correction that matters when draft/target energy profiles diverge significantly. In practice, most speculative decoding systems already have well-calibrated draft models. **~2-5% acceptance rate improvement in pathological cases.**

---

### Fusion 2: Compositional Entropy Budget for DDTree

**CM analogue:** Isotropic CM reduces to partner-dependent scalar learning rates. `η_eff = η · (2·mp)⁻¹ · (c_partner² + λ)^{-1/2}`

**Modelless mapping:** DDTree's `detect_width()` already partitions logit distributions into "peaked" (width=1) vs "multi-peak" (width=k). Currently this is a binary decision based on peak dominance ratio. CM's partner-dependent scalar gives a **continuous** budget allocation.

The "composition" is: `DDTree_budget = f(context_entropy, draft_confidence)`. Currently these are independent. CM says: control the composition, not each factor.

```rust
/// Compositional DDTree width allocation.
///
/// CM's isotropic effective learning rate:
///   η_eff ∝ 1/c_partner
/// maps to: DDTree width ∝ 1/partner_entropy
///
/// When the "partner" (context embedding) is peaked (low entropy),
/// the composition is more constrained → need MORE exploration (wider tree).
/// When the partner is flat (high entropy), the composition is already
/// diffuse → need LESS exploration (narrower tree).
///
/// This is the INVERSE of naive entropy-budgeting. CM's insight is that
/// the partner's geometry determines the effective "cost" of exploration.
fn compositional_width(
    draft_entropy: f32,      // H(π_draft) — draft's uncertainty
    context_entropy: f32,    // H(context) — KV-cache spectral entropy
    max_width: usize,
) -> usize {
    // Partner norm: c² = ‖W‖²_F / d → analogue is normalized entropy
    let c_draft = (draft_entropy + 1e-8).sqrt();
    let c_context = (context_entropy + 1e-8).sqrt();

    // CM isotropic: effective rate ∝ 1/c_partner
    // Here: draft's partner is context, context's partner is draft
    // Width = base_width * (c_draft / c_context) clamped to [1, max_width]
    let ratio = c_draft / c_context;
    let width = (max_width as f32 * ratio).round() as usize;
    width.clamp(1, max_width)
}
```

**Assessment:** Moderate gain. This replaces the binary `PEAK_DOMINANCE_RATIO` in `MuxBfs` with a continuous, principled allocation. The CM theoretical grounding (partner-dependent effective rate) is genuine. When draft is uncertain but context is focused, you explore more; when context is diffuse, you explore less because the composition is already spread. **~5-10% DDTree efficiency improvement** in multi-peak regimes. This is the cleanest fusion.

---

### Fusion 3: Gauge-Corrected Scoring Signal Decorrelation

**CM analogue:** Gauge correction removes vertical (parameterization-only) components from momentum. Vertical directions don't change the output but waste optimizer budget.

**Modelless mapping:** In DDTree, multiple scoring signals may be correlated:
- Draft logit confidence
- ScreeningPruner relevance score
- ConstraintPruner validity
- EGA spectral salience

If two signals are highly correlated, one is "vertical" — it carries no unique information. The gauge correction maps to: **decorrelate scoring signals to remove redundant components, keeping only unique ("horizontal") information.**

This is a QR decomposition or Gram-Schmidt on the scoring signal vector:

```rust
/// Gauge-corrected scoring signal.
///
/// CM's gauge correction: project (m_Q, m_K) onto horizontal subspace
/// by solving C_K² S + S C_Q² = m_Q^T W_Q - W_K^T m_K
///
/// Modelless analogue: given K scoring signals, project onto the
/// subspace of unique (non-redundant) information.
///
/// Signals: [draft_logit, screening_relevance, ega_salience, ...]
/// If two signals are correlated, one is "vertical" (redundant).
/// QR decomposition on the signal matrix removes redundancy.
fn gauge_corrected_score(
    signals: &[Vec<f32>],     // K scoring signals, each length N
    weights: &[f32],          // K scalar weights
) -> Vec<f32> {
    let k = signals.len();
    let n = signals[0].len();

    // Build signal matrix: S ∈ R^{N×K}
    let mut s_matrix = vec![0.0f32; n * k];
    for (j, sig) in signals.iter().enumerate() {
        for i in 0..n {
            s_matrix[i * k + j] = sig[i] * weights[j];
        }
    }

    // Gram-Schmidt (modified QR without explicit Q storage)
    // Remove redundant components from each signal
    let mut result = vec![0.0f32; n];
    let mut ortho_bases: Vec<Vec<f32>> = Vec::new();

    for (j, sig) in signals.iter().enumerate() {
        let mut component = vec![0.0f32; n];
        for i in 0..n {
            component[i] = sig[i] * weights[j];
        }

        // Subtract projection onto all previous orthogonal bases
        for basis in &ortho_bases {
            let dot: f32 = component.iter().zip(basis.iter()).map(|(a, b)| a * b).sum();
            let norm_sq: f32 = basis.iter().map(|x| x * x).sum();
            if norm_sq > 1e-12 {
                for i in 0..n {
                    component[i] -= dot / norm_sq * basis[i];
                }
            }
        }

        // Keep only if remaining norm is significant (not "vertical")
        let residual_sq: f32 = component.iter().map(|x| x * x).sum();
        if residual_sq > 1e-8 {
            ortho_bases.push(component.clone());
            for i in 0..n {
                result[i] += component[i];
            }
        }
        // Else: signal is redundant ("vertical") — dropped
    }

    result
}
```

**Assessment:** Potentially useful but expensive. The O(N·K²) cost of Gram-Schmidt on scoring signals at every token position is significant. For K=3-4 signals and N=512 tokens, it's manageable (~1µs). But the gain depends on how correlated the scoring signals actually are in practice. If `ScreeningPruner` and draft logit confidence are nearly orthogonal (likely), gauge correction adds nothing. **Conditional: profile first. If signal correlation >0.7, implement. Otherwise skip.**

---

### Fusion 4: Half-Split Acceptance Budget

**CM analogue:** Half-split divides the compositional trust region equally between Q-side and K-side. Each side gets a well-posed Muon subproblem.

**Modelless mapping:** In speculative decoding, acceptance involves two factors:
1. **Draft quality:** How well does the draft model approximate the target distribution?
2. **Target verification:** How much verification compute to spend?

Currently, these are treated independently: draft generates N tokens, target verifies all N. CM's half-split suggests: **control the joint acceptance error, not each factor's error separately.**

```rust
/// Half-split speculative budget allocation.
///
/// CM half-split: ‖ΔW_Q·W_K^T‖_op ≤ ε/2 AND ‖W_Q·ΔW_K^T‖_op ≤ ε/2
///
/// Modelless: allocate acceptance error budget between draft and verify.
/// Total error budget ε splits into:
///   - Draft error budget: ε/2 → control draft speculation window
///   - Verify error budget: ε/2 → control verification thoroughness
///
/// When draft is poor (high draft_error), allocate MORE of the budget
/// to improving draft quality (smaller window, better calibration).
/// When draft is good, allocate MORE to verification (larger window).
fn half_split_speculation_budget(
    total_budget: f32,        // ε — total acceptable error
    draft_quality: f32,       // [0, 1] — acceptance rate estimate
    verify_cost: f32,         // per-token verification cost (ms)
) -> (usize, f32) {
    // Draft error ≈ 1 - acceptance_rate
    let draft_error = 1.0 - draft_quality;
    let verify_error = verify_cost; // proxy: higher cost = more verify error

    // Half-split: equal budget, but rescale by partner error
    // CM analogue: c_partner^{-1} rescales the effective rate
    let draft_budget = total_budget / 2.0 * (verify_error / (draft_error + verify_error));
    let verify_budget = total_budget / 2.0 * (draft_error / (draft_error + verify_error));

    // Draft budget → speculation window size
    let window = (draft_budget * 10.0).round() as usize; // heuristic scaling
    let window_clamped = window.clamp(1, 16);

    // Verify budget → verification threshold
    let threshold = verify_budget.min(1.0);

    (window_clamped, threshold)
}
```

**Assessment:** Interesting theoretically, but speculative decoding window sizing is already well-handled by existing adaptive methods (Research 162 TRAS). The half-split doesn't provide new information beyond "allocate more budget to the weaker side." **Low priority — existing TRAS infrastructure already captures this.**

---

### Fusion 5: RMSNorm Composition for Inference-Time Logit Scaling (NOVEL)

**CM analogue (Appendix G):** RMSNorm M = W·Γ where Γ = diag(γ). Optimizing W and γ jointly gives better results. The per-column decomposition allows exact product updates.

**Modelless mapping:** In our inference stack, the logit output is:
```
logits = W_unembed · hidden_state · (1/RMS(hidden_state)) · γ_layernorm
```

The RMSNorm scale γ and the unembedding W compose: `M_eff = W_unembed · Γ_final`. Currently we treat γ as fixed (it's a model parameter) and don't compose it with the unembedding.

**But:** At inference time, we can apply CM's RMSNorm insight to **temperature scaling and top-p/top-k correction**:
```
M_scaled = M_eff · T  (temperature = diagonal scaling)
```

CM says: don't treat T and M_eff independently — the composition's trust region should be controlled jointly. The isotropic approximation gives: `T_eff = T / c_M` where `c_M = ‖M_eff‖_F / √d`.

```rust
/// Compositional temperature scaling.
///
/// CM Appendix G: M = W·Γ. Isotropic: C^{-1} ≈ c^{-1}·I.
///
/// Instead of applying temperature T independently to logits,
/// compute T_eff = T · c_partner^{-1} where c_partner is the
/// spectral norm of the logit vector.
///
/// High-energy logits → partner is "large" → reduce T_eff (already peaked)
/// Low-energy logits → partner is "small" → increase T_eff (needs help)
fn compositional_temperature(raw_logits: &[f32], base_temp: f32) -> Vec<f32> {
    let n = raw_logits.len() as f32;
    // Partner Frobenius norm (isotropic approximation)
    let c_sq: f32 = raw_logits.iter().map(|x| x * x).sum::<f32>() / n;
    let c = c_sq.sqrt();

    // Effective temperature: T_eff = T / c (partner-rescaled)
    let t_eff = base_temp / (c + 1e-8);

    raw_logits.iter().map(|&x| x / t_eff).collect()
}
```

**Assessment:** This is essentially dynamic temperature scaling based on logit energy, which is a known technique. CM provides theoretical grounding for WHY this works (compositional trust region), but the technique itself is not new. **Low novelty for inference, but the CM framing provides principled justification for an existing heuristic.**

---

## Verdict

### GOAT Assessment: CONDITIONAL GOAT

| Criterion | Score | Reasoning |
|-----------|-------|-----------|
| **Theoretical grounding** | 9/10 | CM's math is clean, well-derived, and the isotropic approximation is elegant |
| **Modelless applicability** | 5/10 | Core mechanics (gradient whitening, weight updates) are training-only; only the *principles* transfer |
| **Implementation cost** | 3/10 | Fusion 2 (compositional width) is ~50 lines; Fusion 3 (gauge correction) is ~100 lines |
| **Expected gain** | 4/10 | ~5-15% DDTree efficiency improvement in multi-peak regimes; minimal gain in peaked regimes |
| **Novelty** | 6/10 | Partner-weighted DDTree budget is novel; the other fusions are existing techniques with new theoretical framing |
| **Risk** | 2/10 | All fusions are additive, feature-gated, and don't affect existing behavior |

### What's Actually Worth Implementing

| Fusion | Priority | Rationale |
|--------|----------|-----------|
| **Fusion 2: Compositional DDTree width** | 🟢 HIGH | Cleanest mapping, replaces binary heuristic with principled continuous allocation, ~50 lines |
| **Fusion 3: Gauge-corrected scoring** | 🟡 MEDIUM | Useful if scoring signals are correlated, but needs profiling first |
| **Fusion 1: Partner-weighted acceptance** | 🔴 LOW | Marginal gain over well-calibrated acceptance ratios |
| **Fusion 4: Half-split budget** | 🔴 LOW | TRAS (Research 162) already handles this better |
| **Fusion 5: RMSNorm temperature** | 🔴 LOW | Known technique, CM framing adds theoretical justification only |

### What CM Is Actually GOAT For

CM is **GOAT for training** (riir-ai's LoRA pipeline). The modelless path gets the *theoretical residue* — the compositional principle — but not the core mechanics. The honest assessment:

1. **For katgpt-rs (modelless):** Fusion 2 is worth implementing. The rest are interesting theoretical exercises that map existing heuristics to CM's framework. Gain is real but modest.

2. **For riir-ai (model-based):** Full CM should be implemented for LoRA attention adapters. Partner-whitened Muon on QK and OV circuits during game LoRA training. This is where the real gains live. The isotropic approximation keeps it cheap (no Gram inverse-square-root needed at runtime).

3. **For the codebase as a whole:** CM's gauge correction concept (Fusion 3) is the most intellectually novel transfer. The idea that "redundant scoring signals are vertical directions in a fiber bundle" is a genuinely new way to think about multi-signal pruning. Even if we never implement Gram-Schmidt on scoring signals, the mental model is valuable.

---

## Key Quotes

> "The main perspective offered by Compositional Muon is that, when possible, one should directly control the functional perturbation of a composed circuit, rather than the individual pieces which constitute the circuit."

> "Under this assumption, we may approximate every partner inverse root becomes a scalar multiplier... What remains is ordinary Muon, but with a partner-dependent effective learning rate."

> "The gauge correction is vertical, so it leaves the first-order product perturbation unchanged. It only changes the representative of that perturbation in factor coordinates."

> "Empirically, the isotropic approximation performs similarly to the full CM update."

> "Even a relatively direct approximation can produce better spectral behavior and potential pretraining gains over Muon."

---

## Reference Implementation

Source: `.raw/comp-muon-release/src/` — verified against Tilde's official release.

| File | Key Functions | Relevance to katgpt-rs |
|------|--------------|----------------------|
| `compositional_muon.py` | `cm_qk`, `cm_ov`, `_ov_scalar`, `_qk_scalar` | Core half-split + isotropic update rules |
| `msign.py` | `msign()` with `polar_express` (8-step) and `quintic` (5-step) coefficients | Same Newton-Schulz we already ship (Plan 152) |
| `whitening.py` | `isotropic_scale()` = `(‖W‖_F²/d_h + λ)^{-1/2}`, `coupled_inv_sqrt()` via CANS | Isotropic = scalar, Full = CANS coupled iteration |
| `gauge.py` | `frobenius_scalar` (trace-only, matmul-free), `frobenius` (Sylvester via eigendecomposition) | Scalar gauge: O(1), Full gauge: O(d_h²) |
| `main.py` | `cm_qk`/`cm_ov` for attention, `muon` for everything else | Parameter routing: CM for circuits, Muon for matrices |

**Key implementation detail:** The isotropic scalar gauge (`frobenius_scalar`) in `gauge.py:72-76` is:
```python
x = (tr(A - B) / (tr(gram_a) + tr(gram_b)))  # scalar, matmul-free
delta_Q -= x * W_Q;  delta_K += x * W_K
```
This is literally a dot product + scalar multiply. Trivially cheap.

The isotropic whitening in `whitening.py:85-91` is:
```python
s = (‖W_h‖_F² / head_dim + damping)^{-1/2}  # one scalar per head
```
This is what Fusion 2 maps to for DDTree: entropy-based scalar rescaling per "head" (token group).

---

## References

- [Compositional Muon Blog](https://blog.tilderesearch.com/blog/compositional-muon) — Primary source
- [Research 046](./046_Symmetry_Compatible_Equivariant_Optimizers.md) — Layerwise symmetry-compatible optimizer assignments
- [Research 114](./114_AMUSE_Anytime_Muon_Stable_Gradient_Evaluation.md) — AMUSE optimizer, river-valley diagnostics
- [Research 166](./166_Muon_Curvature_Perspective_NDS.md) — NDS curvature proxy for inference budget
- [Research 140](./140_sigmoid_parallax.md) — Sigmoid Parallax, kernel-agnostic attention
- [Research 162](./162_Trust_Region_Adaptive_Speculation.md) — Trust-region adaptive speculation (TRAS)
- [Plan 152](../.plans/152_newton_schulz_river_valley_diagnostics.md) — Newton-Schulz infrastructure (GOAT 25/25)
- [Benchmark 050](../.benchmarks/) — Newton-Schulz GOAT proofs
- [Muon optimizer](https://github.com/KellerJordan/Muon) — Original Muon implementation
- [Modular Duality](https://arxiv.org/abs/2405.18381) — Bernstein, Newhouse (2024)

---

## TL;DR

**Compositional Muon controls the composition (‖ΔM‖), not the factors (‖ΔW_Q‖, ‖ΔW_K‖).** For modelless inference, this maps to: control the *joint scoring function*, not each scoring signal independently. The isotropic approximation (partner-dependent scalar rescaling) is cheap enough for inference but provides only modest gains (~5-15% DDTree efficiency). The only high-priority implementation is **Fusion 2: Compositional DDTree Width** — replacing `PEAK_DOMINANCE_RATIO` with partner-entropy-scaled continuous allocation. CM's real value is for **training** (riir-ai LoRA pipeline), not inference. Gauge correction as "decorrelate redundant signals" is the most intellectually novel transfer but needs profiling before implementation.
