# Research 549: ASEntmax — Length-Adaptive α-Entmax Sparse Attention

> **Source:** Vasylenko, Pitorro, Martins, Treviso. *Long-Context Generalization with Sparse Attention*. [arXiv:2506.16640v4](https://arxiv.org/abs/2506.16640) (ICLR 2026), 2026-03-02. Code: [deep-spin/asentmax](https://github.com/deep-spin/asentmax).
> **Date:** 2026-09-11
> **Status:** Done — verdict 🟢 GOAT (dual-track). Modelless execution → [Issue 747](../.issues/747_asentmax_modelless_mining.md); training arm → riir-train Plan 396 (SECONDARY).
> **Related Research:** 392 (SSMax/GoldShare — the softmax arm of the duality), 068 (DashAttention α-entmax routing), 225 (MSA blockwise sparse), 140 (sigmoid parallax), 258 (attention sinks), 344 (length extrapolation, implicit LM), 362 (HydraHead head importance)
> **Related Plans:** 411 (SSMax + GoldShare — SHIPPED), 106 (DashAttention — SHIPPED default-on), 256 (MSA adaptive k), 287 (sink-aware attention)
> **Cross-ref (riir-train):** Plan 396 (ASEntmax recipe arm — drafter + learned-scaler extraction)
> **Classification:** Public

---

## TL;DR

The paper trains transformers whose attention transformation is **α-entmax** (exact zeros below a threshold τ) with a **per-head learnable length schedule** `(δ + β·(log n)^γ)·z`, achieving 1000× synthetic length extrapolation and 97.4% needle retrieval at 8× training context. Both halves of the base mechanism already ship here — α-entmax as DashAttention routing (Plan 106, default-on) and a length-aware temperature as SSMax (Plan 411, `s_L·log N`). What does NOT ship is the **mirror image of SSMax**: entmax over-sparsifies as the scored set grows (logit range grows `2σ√(2 log n)`, Kamath 2015), and the paper's Eq 10 *derives* the training-free counter-schedule — **γ = −0.5, δ = 0** makes `β·(log n)^{−0.5}·Δ_n = 2σβ√2 = const`. One extreme-value law, two signed arms, same substrate socket (`SsmaxMode`-style mode enum keyed on runtime `log_n`): softmax needs sharpening ∝ log n (shipped), entmax needs damping ∝ (log n)^{−0.5} (this note).

**Distilled for katgpt-rs (modelless, inference-time):**
1. **`asentmax_schedule`** — damping mode for `entmax_1p5`: pre-scale scores by `β·(log n_c)^{−0.5}` (n_c = scored candidate count) before the threshold pass. Derived, zero-training, one `powf` + one mul.
2. **Lemma-2 support controller** — `k̂ = ((α−1)·Δ̂)^{−1/(α−1)}` (α=1.5: `k̂ = 4/Δ̂²`), consuming the shipped `RollingDeltaEstimator` — a derived replacement for the fitted-shaped `sigmoid(w·var+b)` budget in `adaptive_k.rs`.
3. **Prop 6 eviction window** — ALiBi-biased entmax attention has a hard cutoff `d_max = ⌊(z_max − z_min + 1)/m_h⌋ + 1` beyond which mass is *exactly* zero → KV eviction whose correctness gate is **bit-identity** (a theorem, not a tolerance). *(Correction at implementation, Issue 747 P2: the paper's Eq. 110 is `d_max = ⌊(z_max − z_min + 1/(α−1))/m_h + 1⌋` — `+1` inside the floor, numerator `1/(α−1)` = 2 at α=1.5.)*
4. **Lemma 1 incremental decode** — below-threshold additions leave existing entmax probabilities exactly unchanged → O(candidates) incremental recompute instead of O(n log n) resort per decode step.

---

## 1. Paper Core Findings

1. **α-entmax attention**: `p_i = [(α−1)z_i − τ]_+^{1/(α−1)}` — exact zeros below threshold; α=1.5 has a closed-form sort-based algorithm (Peters et al. 2019 — the algorithm our `entmax_1p5` implements); α=2 is sparsemax.
2. **Theory (the paper's real contribution):**
   - **Lemma 1 (non-vanishing attention):** adding tokens with logits below threshold leaves existing probabilities *exactly* unchanged; softmax strictly leaks on every added token.
   - **Lemma 2 (two-level logits):** if gap `Δ ≥ k^{−(α−1)}/(α−1)`, exactly k tokens share attention 1/k each — **independent of n**. Softmax needs `Δ ≥ θ·ln((n−k)c/(k(1−c)))` to hold concentration c — grows with log n.
   - **Prop 1:** softmax normalized entropy → 1 (complete dispersion); entmax with support `|S| = O(n^β)` keeps it ≤ β.
   - **Prop 2:** representational collapse avoided; gradient paths O(sL) not O(nL).
   - **Prop 6:** ALiBi × entmax → hard cutoff distance `d_max` (see TL;DR). **Prop 7/8:** RoPE × entmax → per-frequency cutoffs `d_{k,0} = arccos(τ(α−1)/‖q_k‖‖k_k‖)/g_k` with periodic re-entry windows at `2π/g_k`.
3. **ASEntmax**: per-head `(δ + β(log n)^γ)·z` rescaling; β = softplus(Xw_β), γ = s·tanh(Xw_γ). **Eq 10 (closed form):** for IID Gaussian logits, γ = −0.5, δ = 0 keeps the effective logit range constant as n grows — the derived counter to over-sparsification. Per-head fitted γ varies in sign (Copy task: all-negative = *less* sparse at long n; Max Retrieval: positive).
4. **NAPE**: half heads NoPE, half ALiBi — beats RoPE+ABF for consistent extrapolation.
5. **Empirics:** 95.3% MQMTAR at 65K trained on 64 (1000×); 420M LM: Lambada PPL 41.6 vs 52.4 (softmax), S-NIAH-1 97.4% at 16K vs 0.8%. **Critical failure mode:** plain fixed-α entmax *underperforms* softmax OOD (Copy: 28.5% vs 99.4% at 4×) — adaptive scaling is what fixes it (99.7%). α ≥ 32 collapses to one-hot (gradient death).

## 2. Distillation

### 2.1 Signal-diff vs shipped substrate (§3.6 discipline)

| Shipped cousin | File | Signal it consumes | Paper component's signal | Verdict |
|---|---|---|---|---|
| `entmax_1p5` | `katgpt-attn/src/dash_attn/entmax.rs` | raw chunk scores; **no length, no temperature** | scores + `log n_c` (candidate count) + per-head β,γ | **Uncovered** — no length term exists on any entmax path |
| `apply_ssmax_inplace` | `katgpt-core/src/ssmax.rs` | `log_n` + rolling Δ̂ (`SsmaxMode::Fixed/Adaptive`) | same `log n` key | **Covered for softmax only, γ=1 fixed, sign = sharpen-UP** — the entmax arm needs the *opposite* sign and a power law |
| `AdaptiveKConfig` | `katgpt-attn/src/dash_attn/adaptive_k.rs` | score *variance* via `sigmoid(w·var+b)` | the derived threshold law `k̂ = 4/Δ̂²` | **Partial** — content-adaptive but fitted-shaped; Lemma 2 supplies the closed form from the same Δ̂ the ssmax estimator already produces |

### 2.2 The duality (why this snaps into the substrate)

| Arm | Failure without correction | Law | Correction |
|---|---|---|---|
| **Softmax** (shipped: SSMax γ=1) | dilution — mass leaks to every token | Lemma 2 softmax side: holding concentration needs Δ ∝ log n | sharpen **UP** ∝ log n |
| **α-entmax** (this paper) | over-sparsification — logit range grows, threshold eats the support | `E[Δ_n] = 2σ√(2 log n)` (Kamath 2015) | damp **DOWN** ∝ (log n)^{−0.5} (a log-log schedule — gentler than SSMax) |

### 2.3 Path 0 inventory (component → coverage → extraction → disposition)

| Component | Coverage | Extraction | Disposition |
|---|---|---|---|
| α-entmax transform (α=1.5) | **ships** (Plan 106, default-on routing) | — | — |
| Length temperature, softmax path | **ships** (Plan 411, `s_L·log N`) | — | — |
| Entmax damping schedule γ=−0.5 | **no** | **yes** — derived (Eq 10), β from rolling σ̂ | Issue 747 **P0** |
| γ-power generalization `(δ+β(log n)^γ)` | **no** (γ=1 implicit in SSMax) | yes — closed form + offline grid sweep (frozen table) | Issue 747 P0/P4 |
| Lemma 2 support controller `k̂ = 4/Δ̂²` | partial (sigmoid variance budget) | yes — consumes shipped `RollingDeltaEstimator` | Issue 747 **P1** |
| Lemma 2 softmax side → `SsmaxMode::HoldConcentration{c,k}` analytic θ* | no (only Fixed/Adaptive) | yes — `θ*(n) = Δ̂/ln((n−k)c/(k(1−c)))` | Issue 747 P4 |
| Prop 6 eviction window | no (sink-aware ≠ distance window) | yes — closed form; **bit-identity gate** | Issue 747 **P2** |
| Lemma 1 incremental decode entmax | no (full re-sort per call) | yes — algebraic identity | Issue 747 **P3** |
| Prop 7/8 RoPE cutoff spectrum | no | yes, with periodicity caveat | Issue 747 P4 (stretch) |
| Entropy diagnostic `H(p)/log n ≤ log s/log n` | partial (`effective_rank` ships) | yes — O(s) over support | Issue 747 P4 |
| Per-head learned β,γ | no | offline sweep → frozen table (modelless) **or** extraction run (Plan 396 Ph2) | dual-track bridge |
| NAPE positional encoding | no | **no** — training-side only | Plan 396 (training arm) |
| ASEntmax-native model quality (1000×, S-NIAH 97.4%) | no | **no** — weights-conditioned | Plan 396 Ph1/Ph3 (training arm) |

### 2.4 Fusion

- **ASEntmax schedule × DashAttention routing (the primary fusion):** routing scores are *our* chunk-summary heuristic, not pretrained-weight attention — so length-adaptive rescaling is quality-safe and fully modelless (unlike swapping trained softmax attention on Bonsai/Qwen GGUF weights, which we do NOT claim). The scored candidate set is `n_chunks`, so the √(2 log n) law applies to chunk count — the over-sparsification diagnosis transfers exactly.
- **Prop 6 window × sink-aware KV policy:** sink-aware classifies *what* (NOP vs Broadcast); Prop 6 bounds *where* (hard distance). A theorem-backed eviction policy whose correctness needs no tolerance.
- **Support-masked O(s) compute:** entmax's exact zeros make `Σ_{i∈S} p_i v_i` lossless — the inference dual of Prop 2, feeding `riir-gpu` gather-GEMV.
- **Support-set fingerprint (stretch):** canonical (indices, values) sparse encoding of routing state → BLAKE3-committed shard latent field / KG triples ("head attends block") in riir-neuron-db.

### 2.5 Honest caveats

1. **γ=−0.5 is IID-Gaussian-optimal only.** Real heads are not IID Gaussian (the paper's own per-head γ varies in sign). The derived default must beat the fixed-α baseline first; the offline sweep is the bridge, not decoration.
2. **Quality anchors are the paper's models.** Our gates re-measure on our routing surfaces (Bench 032 harness pattern); the anchors set targets, not pass conditions.
3. **RoPE re-entry periodicity** forbids naive "drop everything past the first cutoff" eviction — retention must respect the window union.
4. **No quality-parity claim is made for pretrained models** — we do not swap softmax→entmax on trained weights (the paper's Copy-table result shows naive fixed-α conversion *hurts*; the retrofit path is Plan 396 Ph2 distillation, gated).

## 3. Verdict

**🟢 GOAT — dual-track.** Modelless: Issue 747 (entmax damping schedule + derived support controller + eviction window, feature-flagged, Bench-032-pattern gates). Model-based: riir-train Plan 396 (SECONDARY — ASEntmax drafter + learned-scaler extraction).

**One-line reasoning:** the base mechanisms ship (entmax routing, SSMax temperature), but the paper *derives* the missing entmax-side arm of a law we already exploit on the softmax side — plus three theorem-backed primitives (support controller, eviction window, incremental decode) that are provable gains over shipped defaults, and a credible training recipe for the one thing modelless cannot produce (content-conditional per-head schedules on trained weights).

**Why NOT Super-GOAT:**

| Q | Answer |
|---|---|
| Q1: No prior art? | **NO** — α-entmax ships (Plan 106), length-aware temperature ships (Plan 411), sigmoid parallax / sink-aware / MSA ship. The γ-power + damping sliver is an extension of shipped families. AdaSplash (ICML 2025) is the kernel, not the schedule; SSMax (Nakanishi 2025) is the softmax arm. |
| Q2: New behavior class? | **NO** — better long-context calibration of existing routing/eviction, not a new capability. |
| Q3: Product selling point? | **WEAK** — "sparse routing stays calibrated at any context length" is incremental over DashAttention + SSMax. |
| Q4: Force multiplier? | **YES** — composes with DashAttention, `block_select_entmax`, SSMax, adaptive_k, sink-aware KV. Not sufficient alone. |

**MOAT gate (katgpt-rs):** attention-stack inference primitive via fusion with two shipped plans — in scope, public, no game/chain/shard semantics. ✓

**Track priority (serving-envelope fit, TTPO rule):** PRIMARY = modelless schedule (lands on the default-on DashAttention routing hot path; zero training); SECONDARY = Plan 396 (the drafter's flat-in-n acceptance needs 40–60 GPU-hrs on the 4090 and lands on the spec-decode lane, not the serving hot path).

**Discard audit (§3.5 — every advocate finding either filed or discarded with mechanism-level reason):**
- *NAPE for inference* — discarded for the modelless track: positional encoding is baked into pretrained GGUF weights; no runtime consumer without retraining. Recorded in Plan 396 (training arm).
- *420M specialist LM* — deferred, not discarded: 140–170 GPU-hrs with no near-term consumer; Plan 396 Ph3 records the recipe for the day an entmax-native owned LM is wanted.
- *Quest-grammar entmax student* — discarded (ranked last by the model-based advocate): quest contexts are short; entmax's advantage is length OOD. Noted in Plan 396 riders.
- *Sort/Reverse-class extrapolation* — discarded: the paper itself shows all methods fail ≥4×; no trainable signal, no consumer.

**PASS-Redirects (synthesis):** none — verdict is GOAT, not PASS. (For the record: the closest shipped cousins are Research 392/Plan 411 [SSMax — the softmax arm of this paper's duality] and Research 068/Plan 106 [DashAttention — the shipped α-entmax].)
