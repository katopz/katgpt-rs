# Research 316: DSpark — Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation

> **Source:** [DSpark (DeepSeek-AI, 2026)](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) — Cheng, Yu, Shao, Li, Xiong, et al. (Peking University / DeepSeek-AI). Code: [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT).
> **Date:** 2026-06-27
> **Status:** Done
> **Related Research:** 002 (Speculative Decoding — Leviathan), 156 (Speculative Reconciliation), 162 (Trust-Region Speculation), 177 (Domino — Decoupled Causal Spec Decoding), 243 (Bebop — Entropy-Bounded MTP Acceptance), 295 (AC-Prefix)
> **Related Plans:** 004 (Leviathan distill), 166 (FlashAR anchor), 243 (Bebop Issue 023), 294 (ICT G10 — Bebop H_1→H_2 upgrade)
> **Related Issues:** 003 (this note's optimization — hardware-aware prefix scheduler)
> **Classification:** Public — generic speculative-decode primitives, no game/chain/shard semantics
> **PASS-Redirects (synthesis):** Valluri, Nguyen & Grover [arXiv:2608.20359 "Self-Speculation for Faster Reasoning Models"] — a new drafter axis (CoT budget on the SAME model: partial-CoT answer distribution drafts, full-CoT verifies — training-free, no aux params, vs DSpark's trained semi-AR head) plus suffix decoding (rejected draft remainder seeds a suffix cache to recover spans beyond the first mismatch during continuation; compositional over SuffixDecoding arXiv:2411.04975 / SAM-decoding / Token Recycling). PASS for us: our verify surfaces (`forward_speculative_verify`, tree-masked, p=16 chunk) are all prefix-accept + rollback-discard and suffix recovery is genuinely absent — but the recoverable headroom is bounded (doc-repro lookup acceptance 15.15/16 leaves ≤0.85 tok/verify ≈ 5%; chat is low-overlap, the paper's own worst regime) and "the acceptance axis is CLOSED" (riir-ai Bench 754 — remaining levers are GEMM/attention-side). The hide-the-drafter-behind-CoT trick needs vLLM-style multi-request serving; we are single-stream batch-1 CUDA graphs by design.
> **PASS-Redirects (synthesis):** vLLM blog 2026-08-23 ["Exploring Speculative Decoding in vLLM on AMD GPUs"](https://vllm.ai/blog/2026-08-23-speculative-decoding-amd-gpus) — first-party multi-target replication of this note's semi-AR thesis AND its first refutation datapoint: DSpark beats DFlash on all 4 workloads for Qwen3-8B (1.63×/AR 78.4% vs 1.25×/54.9% at N=7) but LOSES to DFlash on all 4 for gemma-4-31B (2.20×/61.4% vs 2.34×/68.0% MATH500) — family ranking is checkpoint-dependent, not architectural (Research 518 §4). Also re-confirms the §1.2 contract reading the blog states verbatim ("the Markov head... produce[s] a small bias [that] adjusts the base logits produced by the parallel backbone") — the exact wrong-contract riir-ai Issue 717 G2-full first diagnosed by reading the trained head standalone.

> **PASS-Redirects (synthesis):** Feng et al. [arXiv:2610.19792 "JetSpec++: Foresight Parallel Drafting with Self Refinements"] — the successor that dethrones this paper's drafter family on the correction axis: self-refinement (Jacobi passes over token inputs, clean-prefix trained) + foresight pipelining takes τ 8.94 vs this paper's 6.89 and 1.54× wall-clock, same-target same-recipe-budget — distilled as Research 641 (training recipe → riir-train Plan 376 amendments; modelless extraction → katgpt-rs Issue 976). Corroborates the §1.2 contract reading: JetSpec++ Table 12 shows the DSpark-class prefix-blind drafter's τ FALLS under refinement passes (5.86→2.39 control) while a causally-trained one rises.

---

## TL;DR

DSpark is a **production speculative-decoding paper** (DeepSeek-V4 serving, +60–85% per-user speedup at matched throughput vs MTP-1) with two components: (1) a **semi-autoregressive drafter** (parallel backbone + lightweight sequential Markov/RNN head) and (2) a **confidence-scheduled verification** system (confidence head + hardware-aware prefix scheduler). Component (1) is training → riir-train. Component (2) splits: the confidence head's modelless *math* is already shipped in katgpt-rs (`AcceptanceForecast`, `AcceptanceSurrogate`, `LeviathanVerifier`), but the **Hardware-Aware Prefix Scheduler** (Algorithm 1) — a global, multi-request, load-aware greedy verification-budget allocator with a non-anticipating early-stop correctness proof (Appendix A) — is genuinely **NOT shipped** and is the one modelless primitive worth extracting.

**Distilled for katgpt-rs (modelless, inference-time):**

1. **Hardware-Aware Prefix Scheduler** (§3.2.2, Algorithm 1) — given R active requests each with per-position survival probabilities `a_{r,j} = Π_{i≤j} c_{r,i}` and a profiled engine cost curve `SPS(B)` (steps-per-second vs batch size), greedily admit (r,j) candidates in descending `a_{r,j}` order to maximize `Θ = τ · SPS(B)`, with **early-stop when Θ drops** to preserve the non-anticipating property required for lossless speculative decoding. Modelless, O(R·γ log(R·γ)), zero-allocation. **Not in katgpt-rs.**
2. **Sequential Temperature Scaling (STS)** (§3.2.1) — chain-rule-aware post-hoc calibration: because joint survival = Π c_i, calibrate left-to-right via 1D grid search per position, minimizing ECE of the cumulative product. Small refinement to Bebop's `fit_from_warmup`. **Not in katgpt-rs** (Bebop calibrates the marginal `α ≈ a−bH`, not the chain-rule joint).
3. **TV-distance accept-rate label** `c*_k = 1 − ½‖p_d − p_t‖₁` (Eq. 8) — already implicit in `LeviathanVerifier` (RS acceptance = 1−dTV).
4. **Expected accepted length** `τ = Σ_j Π_{i≤j} c_i` — already shipped as `AcceptanceSurrogate::expected_accepted_length` (`caddtree_budget.rs`), with sigmoid gates instead of raw products.

---

## 1. Paper Core Findings

### 1.1 The two bottlenecks DSpark solves

DSpark starts from the per-token latency identity `L = (T_draft + T_verify) / τ` and identifies two bottlenecks that emerge when parallel drafters produce large blocks:

| Bottleneck | Cause | DSpark's fix |
|---|---|---|
| **Suffix decay** (τ drops at later positions) | Parallel drafters predict each position independently → multi-modal collision ("of problem" / "no course") | Semi-autoregressive: parallel backbone + lightweight sequential head (Markov or RNN) |
| **Verification waste** (T_verify burned on rejected suffix tokens, especially under high concurrency) | Static block-length verification ignores both data-side acceptance variance (code vs chat) and system-side load (light vs heavy batch) | Confidence head + hardware-aware prefix scheduler that dynamically prunes per-request verification length by survival probability × engine capacity |

### 1.2 Semi-autoregressive generation (§3.1) — partially modelless (Markov head), partially training (RNN head, confidence head)

Two-stage drafter:
- **Parallel stage**: DFlash-style backbone, single forward pass over the block, produces base logits `U_1..U_γ`.
- **Sequential stage**: a prefix-dependent transition bias `B_k(x_0, x_{<k}, x_k)` added to the base logits before softmax, sampled left-to-right. Two instantiations:
  - **Markov head** (default): `B(x_{k-1}, ·) = W1[x_{k-1}] · W2`, low-rank factorization (r=256) of a V×V transition matrix. W1 = embedding lookup, W2 = logit projection.
  - **RNN head**: gated recurrent state `s_k = σ(W_g z_k) ⊙ s_{k-1} + (1−σ(W_g z_k)) ⊙ tanh(W_c z_k)`, with `z_k = [s_{k-1}; W1[x_{k-1}]; h_k]`. Marginal gain over Markov; not worth the complexity.

Training loss (Eq. 9–12): `α_ce L_ce + α_tv L_tv + α_conf L_conf` with `α_ce=0.1, α_tv=0.9, α_conf=1.0`, position-weighted by `w_k = exp(−(k−1)/γ)`. **All training → riir-train.**

Empirical: 2-layer DSpark beats 5-layer DFlash; capacity advantage at position 1 (deeper parallel backbone possible because T_draft is O(1) not O(γ)) outweighs suffix decay.

**Modelless path check (§3.5):**
- **Markov head** `B = W₁·W₂` — low-rank bigram transition matrix. **CAN be constructed deterministically** from corpus bigram co-occurrence statistics: `W₁[x]` = token embedding, `W₂` = log-bigram-probabilities. This is §3.5 path 2 (deterministic construction, no gradient descent). The quality may be lower than a trained head (corpus bigrams ≠ task-specific transitions), but it's a modelless baseline.
- **RNN head** — recurrent state `s_k` is genuinely learned. **NOT modelless-constructable.** → riir-train.
- **Confidence head** `c_k = σ(w^T [h_k; W₁[x_{k-1}]])` — **circular to compute modellessly**: the analytical `c*_k = 1 − ½‖p_d − p_t‖₁` requires target `p_t`, which is the verification forward pass itself. However, a **modelless proxy** exists: drafter logits entropy `H(p_d)` — low entropy → high confidence → likely accepted. We already ship `AcceptanceForecast` (Bebop, Plan 243) which estimates acceptance from entropy via `α ≈ a − b·H(p)`.

**Verdict: Markov head is modelless-constructable (bigram statistics); confidence head has a modelless entropy proxy (Bebop); RNN head needs training.** This makes a **hybrid modelless DSpark** feasible: DFlash backbone (shipped) + bigram Markov head (deterministic) + Bebop entropy confidence (shipped) + HardwareAwarePrefixScheduler (shipped, Plan 339). No training needed for the modelless variant.

### 1.3 Confidence head (§3.2.1) — math is modelless, weights are trained

`c_k = σ(w⊤ [h_k; W1[x_{k-1}]])` — a single linear projection over `[backbone hidden; prev-token Markov embedding]` + sigmoid. Supervised against the analytical per-step acceptance rate `c*_k = 1 − ½‖p_d − p_t‖₁` (TV complement, Eq. 8).

**Sequential Temperature Scaling (STS)** — post-hoc calibration. Because each `c_i` is a *conditional* probability and the chain rule gives joint = Π c_i, STS calibrates the joint left-to-right: for each position k, 1D grid-search the temperature minimizing ECE of Π_{i≤k} c_i, holding earlier positions fixed. Reduces ECE from 3–8% to ~1%. Order-preserving (doesn't disrupt draft token rankings).

### 1.4 Hardware-Aware Prefix Scheduler (§3.2.2, Algorithm 1) — MODELLESS, SHIPPED (Plan 339)

This is the paper's genuinely novel inference-time primitive. Given:
- R active requests, each with confidence sequence `c_{r,1..γ}`
- Prefix survival probabilities `a_{r,j} = Π_{i≤j} c_{r,i}` (monotone non-increasing in j)
- Profiled engine cost curve `SPS(B)` = steps-per-second for forward-pass batch size B

Maximize expected system-wide token throughput `Θ = τ · SPS(B)` where:
- `B = Σ_r (1 + ℓ_r)` (total verification batch size)
- `τ = Σ_r (1 + Σ_{j≤ℓ_r} a_{r,j})` (expected accepted tokens)

Algorithm: globally sort all (r,j) candidates by `a_{r,j}` descending, greedily admit, O(1) lookup SPS(B) per step, **early-stop when Θ ≤ Θ_best**.

**The non-anticipating early-stop (Appendix A) is a CORRECTNESS result, not just an optimization.** Without the break, retrospective global search leaks future token info into the admission decision for the current token, introducing selection bias that breaks the lossless distribution-preservation guarantee. The paper proves this with a concrete 2-token counterexample (vocab {A,B}, p_t=(0.7,0.3), p_d=(0.5,0.5) → output (0.85,0.15) ≠ target without early-stop).

### 1.5 Production deployment (§5)

DeepSeek-V4-Flash/Pro. Asynchronous scheduler variant: because Zero-Overhead Scheduling (ZOS) requires next-step batch size before current step completes, the scheduler uses 2-step-prior confidence predictions to set the capacity cap K, while still sorting current candidates by up-to-date cumulative confidence. Rank-preserving → causality maintained via the temporal offset. Variable-length verification via flattened token processing + marker-tensor sparse attention.

Results: +51–52% aggregate throughput at moderate SLA, +60–85% (Flash) / +57–78% (Pro) per-user TPS at matched throughput, and crucially extends the feasible interactivity frontier (661%/406% nominal throughput at strict SLAs where MTP-1 collapses).

---

## 2. Distillation

### 2.1 What's already in katgpt-rs (verified by code grep)

| DSpark mechanism | Our shipped equivalent | Status |
|---|---|---|
| RS acceptance = `1 − dTV(p,q)` (Eq. 8) | `LeviathanVerifier` (`crates/katgpt-speculative/src/spechop/verifier.rs`) — real p/q rejection sampling; target probs in `p_distributions_flat` (`crates/katgpt-core/src/speculative/types.rs`) | ✅ shipped |
| Per-step acceptance forecast `α ≈ a − b·H(p)` | `AcceptanceForecast` (`crates/katgpt-speculative/src/acceptance_forecast.rs`, Bebop Plan 243); H_2 upgrade in `crates/katgpt-core/src/ict/bebop_upgrade.rs` (Plan 294 G10) | ✅ shipped |
| Expected accepted length `τ = Σ sigmoid(k·(Π top1 − t))` | `AcceptanceSurrogate::expected_accepted_length[_at_budget[_top1]]` (`crates/katgpt-speculative/src/caddtree_budget.rs`) | ✅ shipped (sigmoid-gated variant) |
| `cumprod(a_i)` atomic primitive | `cumprodsum_scalar/batched[_simd]` (`crates/katgpt-core/src/cumprodsum.rs`) — SIMD-accelerated | ✅ shipped |
| Anchor-block drafting pattern | `flashar_anchor.rs` (Plan 166 — AR anchor + D2F fill), `d2f.rs` (block-parallel D2F with mask tokens) | ✅ shipped (different pattern: FlashAR is AR-stride + D2F-fill; DSpark is anchor-emit + mask-block + sequential head) |
| Confidence head = Linear(d,1)+sigmoid | Architecturally identical to our dot-product + sigmoid direction-vector projection (constraint #2) | ✅ shipped as *shape* (Bebop `AcceptanceForecast`); the *trained weights* are training-only |
| Block-causal attention mask with cross-block isolation | `create_dspark_attention_mask` in the paper's code; our `d2f.rs` has block-causal masks | ✅ shipped (different mask topology) |

### 2.2 What's NOT in katgpt-rs (the gaps)

1. ~~**Hardware-Aware Prefix Scheduler** (Algorithm 1)~~ — **SHIPPED (Plan 339, feature `hardware_aware_scheduler`, GOAT G1-G5 PASS)**. `HardwareAwarePrefixScheduler` + `SpsCurve` + non-anticipating early-stop. Stays opt-in until a real multi-request batch caller exercises it (katgpt-rs is single-request by default).
2. **Sequential Temperature Scaling (STS)**. Our `AcceptanceForecast::fit_from_warmup` fits the marginal `α ≈ a − b·H(p)` via linear regression on `(H, observed_acceptance)` pairs. It does NOT do chain-rule-aware left-to-right calibration of the cumulative product `Π c_i`. STS is a small but principled refinement.
3. **Semi-autoregressive sequential head** (Markov/RNN over parallel backbone logits). **Modelless path exists for the Markov head** (§3.5 path 2: construct `W₁·W₂` from corpus bigram co-occurrence statistics — no gradient descent). The RNN head is genuinely trained → riir-train. A **hybrid modelless DSpark** is feasible: DFlash (shipped) + bigram Markov head (deterministic) + Bebop entropy confidence (shipped) + HardwareAwarePrefixScheduler (shipped). This hybrid is benchmarked against the dual-path consolidation in Plan 485 (Issue 453).

### 2.3 Fusion

The Hardware-Aware Prefix Scheduler fuses with four existing katgpt-rs / riir-ai systems:

1. **× `AcceptanceForecast` (Bebop, Plan 243)**: Bebop produces the per-position `c_k` (via `α ≈ a − b·H_2(p)`). The scheduler consumes `a_{r,j} = Π_{i≤j} c_i` to allocate verification budget across requests. Bebop is the *producer*; the scheduler is the *consumer*. Today Bebop drives per-request adaptive γ (Issue 023); the scheduler would drive *cross-request* adaptive verification budget.
2. **× `AcceptanceSurrogate::expected_accepted_length_at_budget` (caddtree_budget.rs)**: this already does per-request budget-vs-acceptance tradeoff. The scheduler generalizes it to multi-request with a real `SPS(B)` cost curve instead of a synthetic latency estimator.
3. **× riir-ai crowd-scale NPC cognition**: in MMORPG-scale game AI (thousands of concurrent NPCs, 20Hz tick), each NPC's per-tick token generation is a "request". The SPS(B) curve becomes "how many NPC-verifications can the engine fit in one tick". The scheduler allocates the tick's verification budget across NPCs by survival probability × engine capacity. This is the most interesting fusion angle — a multi-NPC verification budget allocator under tick constraints. (Noted but NOT developed here — would be a riir-ai guide if pursued.)
4. **× DDTree + FlashAR consolidation (Proposal 018 §4.7, Issue 453, Plan 485)**: DSpark's semi-AR architecture (parallel backbone + Markov head) is a **simpler alternative** to the dual-path consolidation design. Instead of running two drafters (DFlash+MTP + D2F) with ternary consensus, DSpark uses a single semi-AR drafter with confidence-scheduled verification. The **modelless hybrid** (DFlash + bigram Markov head + Bebop confidence + Plan 339 scheduler) is a benchmark competitor in Plan 485's 4-way PoC. If the modelless hybrid beats the dual-path consolidation, it's the simpler path. DSpark's production results (60-85% over MTP-1) validate the approach.

### 2.4 Why this is NOT Super-GOAT

Novelty gate (Q1–Q4):

1. **No prior art?** Partially. The scheduler is NOT in our codebase. But DSpark §6 cites extensive prior work on system-aware speculation budgets (AngelSlim/D-Cut, Echo, AdaSpec, MagicDec, TETRIS, TurboSpec, Nightjar). So it's not novel in the literature; it's novel *to us*.
2. **New class of behavior?** No. It's a serving-system / batch-scheduling optimization. The system still speculates, verifies, accepts/rejects — just with cross-request budget allocation.
3. **Product selling point?** No for katgpt-rs (single-request public engine). Borderline for riir-ai (multi-NPC tick budget), but "our engine allocates verification compute across NPCs by survival probability" is an optimization, not a moat — cannot finish "our NPCs do X no competitor can" in a way that the scheduler alone enables.
4. **Force multiplier?** Borderline — touches Bebop, caddtree_budget, and (via the riir-ai angle) crowd NPC cognition. But Q1+Q2+Q3 fail kills Super-GOAT.

**Not Super-GOAT. Not GOAT either** — the paper proves the algorithm is correct and reports production gains, but those gains are on DeepSeek-V4 serving infrastructure (CUDA graph replay, ZOS, index-attention kernels). We have no proof the scheduler improves *our* (CPU/SIMD/wgpu, single-request-default) stack without our own benchmark. **Gain tier.** Tracked in Issue 003.

### 2.5 Latent-space reframing check (mandatory per workflow §1 step 3)

Checked. DSpark is fundamentally a **token-level speculative decode + serving-system** paper. The latent-space reframing (HLA / latent_functor / cgsp / neuron-shard / DEC) does NOT strengthen it — forcing one would be the inverse of the R269 failure mode. The SPS(B) cost curve is a deterministic profile (could be LatCal-committed in principle), and the prefix lengths are raw integers (correctly raw, not latent). This is correctly a GOAT/Gain-tier inference-primitive paper, not a Super-GOAT latent-space paper.

---

## 3. Verdict

**Gain.** A production-proven speculative-decoding paper whose modelless inference-time gift is the **Hardware-Aware Prefix Scheduler** (Algorithm 1): a global, multi-request, load-aware greedy verification-budget allocator with a non-anticipating early-stop that preserves the lossless distribution guarantee (Appendix A correctness proof). The scheduler is not in katgpt-rs and is a clean generic primitive (sort + greedy + cost-curve lookup + early-stop). The rest is either training (semi-autoregressive architecture, CE+TV+confidence loss, Markov/RNN head → riir-train) or already shipped (`AcceptanceForecast`, `AcceptanceSurrogate::expected_accepted_length`, `LeviathanVerifier`, `cumprodsum`).

**One-line reasoning:** The scheduler is a genuinely missing modelless primitive with a correctness proof, but it's a serving-system optimization (not a new capability class) whose production gains are on infrastructure we don't have — useful, incremental, needs our own benchmark before promotion.

**Routing:**
- Semi-autoregressive architecture (parallel backbone + Markov/RNN sequential head) + CE/TV/confidence training loss + STS calibration of trained confidence head → **riir-train** (training recipe).
- RS acceptance = 1−dTV, expected accepted length τ, cumprod atomic → **already shipped** (`LeviathanVerifier`, `AcceptanceSurrogate`, `cumprodsum`).
- Per-step acceptance forecast α ≈ a−b·H(p) → **already shipped** (`AcceptanceForecast`, Bebop Plan 243).
- **Hardware-Aware Prefix Scheduler** (Algorithm 1) + non-anticipating early-stop → **katgpt-rs** (this note + Issue 003). Generic open primitive: takes `(per-request survival probabilities, profiled SPS(B) cost curve)` → `(per-request prefix lengths)`.

---

## 4. Implementation Sketch (delegates to Issue 003)

The primitive is small — a greedy allocator over a sorted candidate pool with a cost-curve lookup:

```rust
/// Hardware-aware verification prefix scheduler (DSpark Algorithm 1).
///
/// Given R requests each with per-position survival probabilities and a profiled
/// engine cost curve SPS(B), select per-request prefix lengths ℓ_r that maximize
/// expected system throughput Θ = τ · SPS(B). Early-stops when Θ drops to enforce
/// the non-anticipating property required for lossless speculative decoding
/// (DSpark Appendix A).
///
/// Modelless: sort + greedy + O(1) cost-curve lookup. Zero allocation beyond the
/// sorted candidate index.
pub struct HardwareAwarePrefixScheduler {
    /// Profiled steps-per-second curve, indexed by total verification batch size.
    /// Profile once at engine init; store as a small Vec<f32> or interpolation LUT.
    sps_curve: Box<[f32]>,
}

impl HardwareAwarePrefixScheduler {
    /// Select per-request prefix lengths maximizing Θ = τ · SPS(B).
    ///
    /// `survival_probs[r]` = `[a_{r,1}, a_{r,2}, ..., a_{r,γ_r}]` (monotone non-increasing).
    /// Returns `prefix_lengths[r]` ∈ `[0, γ_r]`.
    pub fn schedule(&self, survival_probs: &[&[f32]]) -> Box<[usize]> {
        // 1. Flatten (r, j) candidates, sort descending by a_{r,j}.
        // 2. Greedily admit; update B, τ; lookup SPS(B); compute Θ.
        // 3. Early-stop when Θ ≤ Θ_best (non-anticipating — see DSpark Appendix A).
        // ...
    }
}
```

**GOAT gate (Issue 003):** benchmark `accepted_tokens/sec` and `μs/step` with vs without the scheduler on a workload with multiple concurrent spec-decode requests and a profiled SPS(B) curve. Promote to default only if ≥5% throughput gain with no quality regression AND the non-anticipating early-stop test passes (output distribution matches `LeviathanVerifier` exactly on a single-request-isolated workload). The early-stop correctness property is the gate's quality arm.

---

## 5. Cross-References

- **`katgpt-rs/.plans/339_hardware_aware_prefix_scheduler.md`** — SHIPPED: `HardwareAwarePrefixScheduler` + `SpsCurve` + non-anticipating early-stop (GOAT G1-G5 PASS, opt-in)
- `katgpt-rs/crates/katgpt-speculative/src/prefix_scheduler.rs` — the shipped scheduler implementation
- `katgpt-rs/crates/katgpt-speculative/src/acceptance_forecast.rs` — `AcceptanceForecast` (Bebop, produces `c_k`; scheduler consumes `Π c_i`)
- `katgpt-rs/crates/katgpt-speculative/src/caddtree_budget.rs` — `AcceptanceSurrogate::expected_accepted_length_at_budget` (per-request analog of the scheduler's multi-request objective)
- `crates/katgpt-forward/src/verifier.rs` — `LeviathanVerifier` (RS = 1−dTV, already shipped)
- `katgpt-rs/crates/katgpt-core/src/cumprodsum.rs` — `cumprodsum_*` (SIMD atomic for `Π c_i`)
- `crates/katgpt-speculative/src/budget.rs` — per-request adaptive budget (generalized by the scheduler)
- `katgpt-rs/.research/243_Bebop_Entropy_Bounded_MTP_Acceptance_Adaptive_Gamma.md` — Bebop `α ≈ a−b·H(p)` (scheduler's per-position input source; modelless confidence proxy for DSpark's trained confidence head)
- `katgpt-rs/.research/177_Domino_Decoupled_Causal_Speculative_Decoding.md` — Domino (concurrent work, CausalEncoder ≈ DSpark RNN head → riir-train)
- `katgpt-rs/.research/295_AC_GPT_Arbitrary_Conditionals_Prefix.md` — AC-Prefix (single-pass conditional evaluation, different angle)
- **`riir-ai/.proposals/018_unique_runtime_training_methodology.md` §4.7** — consolidation design (dual-path consensus + DDTree); DSpark semi-AR is the simpler alternative
- **`riir-ai/.issues/453_consensus_tree_verify_proof.md`** — proof requirement for the consolidation; DSpark is the production-validated baseline to beat
- **`riir-ai/.plans/485_consensus_tree_verify_poc.md`** — PoC includes DSpark modelless hybrid as 4th competitor
- `riir-train/` — semi-autoregressive architecture + CE/TV/confidence loss + Markov/RNN head training (redirect for trained variant; modelless Markov head from bigram statistics stays in katgpt-rs)

---

## TL;DR

DSpark is a production speculative-decoding paper (DeepSeek-V4, +60–85% per-user speedup). Two components: (1) **semi-autoregressive drafter** (parallel backbone + sequential Markov/RNN head) — Markov head is **modelless-constructable** from corpus bigram statistics (§3.5 path 2); RNN head needs training → riir-train; (2) **confidence-scheduled verification** — confidence head math ships as `AcceptanceForecast` (Bebop, modelless entropy proxy), and the **Hardware-Aware Prefix Scheduler SHIPPED** (Plan 339, `hardware_aware_scheduler`, GOAT G1-G5 PASS, opt-in). A **modelless hybrid DSpark** is feasible: DFlash (shipped) + bigram Markov head (deterministic) + Bebop entropy confidence (shipped) + Plan 339 scheduler (shipped) — no training needed. This hybrid is benchmarked against the dual-path consolidation design (Proposal 018 §4.7, Issue 453, Plan 485) as a 4th competitor. Not Super-GOAT (serving-system optimization, not a new capability class). Latent-space reframing deliberately not forced — this is correctly a token-level inference-primitive paper.
