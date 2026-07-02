# Research 362: HydraHead — Causal Head Importance & Heterogeneous Attention Fusion

> **Source:** HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization — Tan, Chen, Shen, Liu, Shen, Wu, Ye (Alibaba Group), arXiv:2606.20097, 18 Jun 2026
> **Date:** 2026-07-02
> **Status:** Done
> **Related Research:** 086 (RTPurbo — sibling, attention-mass head calibration), 244 (FaithfulnessProbe — causal-intervention prior art), 233 (Attention Matching KV compaction), 259 (QK-Restore HypeNet SFT drift), 319 (Olmo Hybrid paired-token gap), 070 (GDN2), 073 (LT2), 353 (HeadSubstitutionGate, in-flight)
> **Related Plans:** 126 (RTPurbo), 278 (FaithfulnessProbe), 287 (Sink-Aware), 182 (Luce Megakernel layer-wise hybrid, riir-ai), 353 (HeadSubstitutionGate), 358 (this note's plan — causal head-importance calibration)
> **Classification:** Public

---

## TL;DR

HydraHead converts a pretrained FA Transformer into a **head-wise FA+LA hybrid** by (a) scoring each attention head's *causal necessity* for a target capability via **activation patching + iterative path patching** (not the observational attention-mass scoring RTPurbo uses), and (b) mixing FA-head and GDN-head outputs through **per-branch RMSNorm + learnable per-head γ scale** (scale-normalized fusion). Trained on 15B tokens via a three-stage parameter-reuse + distillation pipeline, the 1.7B hybrid matches a 3:1 layer-wise hybrid at a 7:1 LA:FA ratio, holds 94.5% Single-NIAH at 256K, and gains +9.66% on hard reasoning.

**Distilled for katgpt-rs (modelless, inference-time):**
1. **Causal head-importance scoring** — replace RTPurbo's observational needle-attention-mass calibration with **activation-patching logit-difference necessity** (Eq 9–10). Strictly stronger: a head may attend strongly to the needle yet be overridden downstream (correlated bystander), whereas causal patching isolates load-bearing heads directly. Forward-pass-only, "a few samples" — modelless.
2. **Path patching (sender/receiver)** — extend the direct-effect probe to **one-step-back indirect effects** (Eq 11). Captures heads that feed a receiver head without writing the signal themselves.
3. **Span-level logit-difference readout with exponential decay** (Eq 9) — multi-token answer support.
4. **Scale-normalized heterogeneous fusion** (Eq 13–14) — independent RMSNorm per branch + learnable per-head γ scalar. Modelless; applies whenever two heterogeneous attention outputs are concatenated (FA+GDN, FA+sparse, dense+Raven, etc.).
5. **Head-wise FA/LA partitioning** and **the three-stage transfer / QKV-decomposition architecture** → **→ riir-train** (training-only; the distillation pipeline, parameter migration, query decomposition all require gradient descent). Noted and stopped for that axis.

**Latent vs raw boundary:** the head-importance *score* is a scalar diagnostic (crosses sync OK); the partition set is a `Vec<usize>` of head indices (config blob, raw); the activation patches operate on local per-head outputs (not synced). No new sync-boundary data introduced.

---

## 1. Paper Core Findings

### 1.1 Head, not layer, is the natural hybridization unit

Figure 2a/b: per-head logit contributions to the correct answer token vary sharply within a layer (only a sparse subset of heads contributes; the rest are nearly inactive), while layer-output cosine similarity (28×28 matrix) varies smoothly across depth with no clean block boundaries. **Conclusion: any per-layer mechanism assignment either wastes FA on unimportant heads or converts away critical ones.** This is the central claim behind head-level hybridization — representational fineness is a symptom, the **causal** argument (next) is the load-bearing one.

### 1.2 Causal-importance head selection (the modelless crown jewel)

Section 4.1 + Appendix C.1. Three-step procedure:

**(a) Counterfactual + span-level readout (Eq 9).** For capability `c`, construct paired inputs `(x, x')` where `x'` is `x` with the answer replaced by a same-type same-length distractor (symmetric token replacement — keeps `x'` on the model's natural distribution; better than additive noise which pushes the residual stream OOD). Readout is the **span-level logit difference with exponential decay**:
```
m(x) = (1/Z) Σ_{j∈A} λ^j · (z_j[a+_j] − z_j[a-_j]),   Z = Σ λ^j,   λ = 0.9
```
Logit difference (not probability) is used because it's approximately linear in the residual stream and monotone in the underlying capability, avoiding softmax-saturation and probability measurement-floor effects.

**(b) Receiver (direct-effect) score via activation patching (Eq 10).** Patch head `(l,h)`'s output with the corrupted-run value, **freeze downstream attention outputs to their clean values** so the patched signal propagates only through the residual stream + MLPs (isolates the head's direct effect). Normalized importance:
```
IE_l,h = (m(x) − m(x; O_l,h ← O_l,h(x'))) / (m(x) − m(x'))   ∈ [0,1]
```
`IE ≈ 0` → dispensable (safely converted to LA); `IE ≈ 1` → alone collapses the capability.

**(c) Sender (indirect-effect) score via path patching.** A head can be causally important without writing directly — by feeding a receiver. For each upstream candidate, run corrupted input, record activations it sends to receivers, run otherwise-clean pass with only those substituted. This is **one-step-back path patching**, iterated until new contributions vanish (long-context retrieval converges in ~2 rounds → a shallow circuit).

**(d) Cross-capability fusion (Eq 11–12).** Per capability `c`: `s_h^(c) = max(IE_recv_h, IE_send_h) · κ_h^(c)` where `κ` is task-consistency (fraction of sub-probes where the head exceeds threshold). Min-max normalize per capability, then weighted-mean fuse across capabilities with equal weights.

**Critical empirical findings (Section 5.7):**
- The score is **stable from ~6 calibration samples** (Spearman ρ ≈ 0.921 at k=6, saturates by k=12). Figure 8 left.
- Robust to **localization context length** — short-context localization transfers to long-context selection (top-K Jaccard ≥ 0.78 from 4K upward, ρ ≥ 0.85 from 4K). Figure 8 mid/right.
- Retrieval is **head-localized, not layer-localized**: per-layer Gini of head importance averages 0.622 (range 0.399–0.915); only ≈6.5% of 448 heads are critical, ≈90.8% are safely convertible; critical heads are scattered across layers (10 layers contain both a critical and a replaceable head — Table 16 counter-examples to layer granularity).
- **Causal knockout faithfulness**: ablating heads by `−drop` collapses retrieval accuracy after only the top few heads (≈1%), while random controls stay near-perfect. Figure 9b. This is the confirmatory check that the score identifies load-bearing heads, not arbitrary subsets.
- **Backup-head caveat**: single-head necessity under-estimates heads with redundant backup pathways. Mitigation: take union across sub-probes, rely on head population not exact ranks.

### 1.3 Scale-normalized head-wise fusion (Eq 13–14)

FA softmax produces sharp low-entropy distributions dominated by query norm; GDN normalization cancels query norm → smoother high-entropy outputs. Naive concatenation destabilizes optimization (Table 5: w/o Norm drops RULER Single -10% at both native and extended lengths). The fix is **independent RMSNorm per branch + index-preserving concatenation + learnable per-head scalar γ**:
```
Ô_h = Norm(O_h),   Õ_{:,h,:} = γ_h · Ô_{:,h,:}
```
Figure 5: at deep layers (18–27), GDN RMS reaches up to 6.2× that of FA — independent RMSNorm per branch is essential. **Scale Modulation (static γ) beats Gated Competition (dynamic softmax gate)** on nearly every metric, +20% on extended Single-NIAH (Table 5). Static is the Default Model.

### 1.4 Branch-specific refinements (training-time architecture)

- **FA branch**: drop RoPE, use log-scale coefficient on query; add auxiliary gate branch (alleviates attention sink in high-precision heads). Gated attention (Qiu et al.).
- **GDN branch**: keep native short conv + gating; add RoPE on Q/K (compensate for limited positional sensitivity of linear recurrence); expand KV heads to match query heads (GQA→MHA).
- **Query decomposition**: split QKV projection matrices for FA vs GDN heads (avoids numerical-precision and gradient-conflict bottlenecks). Table 4: more pronounced gains under interpretability-guided selection than fixed allocation.

These are all **architecture changes that require training** (Stage 1 parameter migration, Stage 2 global logits distillation, Stage 3 long-context fine-tune) — see §1.5.

### 1.5 Three-stage transfer pipeline (training-time, → riir-train)

Stage 1 (Parameter Migration + Layer-wise MSE Alignment): initialize GDN branches by reusing pretrained Q/K/V (channel-wise repeat for GQA→MHA dim mismatch); gate-branch weights near-zero with bias→1 so FA starts ≈ identity; freeze backbone, train only hybrid attention layers with MSE loss against original FA hidden states (Eq 15). 0.3B tokens.

Stage 2 (Global Logits Distillation): unfreeze, KL divergence on final logits + cross-entropy on labels (Eq 16). 1.0B tokens.

Stage 3 (Long-Context Fine-tune): standard NTP at 16K context. 1.0B tokens.

Optimized schedule (Table 9): 0.8B + 4.0B + 1.0B = 5.8B alignment/distill tokens, then 15B total scaling run for the headline result.

### 1.6 Headline results (Sections 5.2, 5.5, 5.9)

- **vs layer-wise 3:1 hybrid at 7:1 ratio** (Table 8): HydraHead at 7:1 matches the layer-wise 3:1 on RULER 16K–256K (within ±3% per length) while gaining +9.66% on Hard reasoning and +2.93% on Easy. Head-wise allocation preserves general-domain capability that layer-wise uniformly disrupts.
- **vs SOTA** (Table 11): at 256K, +54.23% Single / +38.50% Multi-Key over Qwen3-1.7B-YaRN; sustained 94.53% Single / 52.70% Multi-Key where most hybrids (Gemma-3n, Hymba, Jet-Nemotron) collapse to near-zero.
- **Interpretability > naive selection** (Table 6): Global-Interp 98.70 vs Fixed 85.63 vs Global-Rand 59.40 on RULER Single Native. Global random can leave entire layers without FA → structural collapse.
- **Constrained global screening** (≥1 FA head per layer) rescues high sparsity: at 7:1 it gains +23 points on extended Single over unconstrained global screening.

### 1.7 Limitations (Appendix D)

- 1.7B scale only; MoE/multimodal untested.
- Activation patching iterates all heads with separate forward passes — tractable at 1.7B, costly at frontier. Attribution patching (single backward pass, first-order Taylor approx of Eq 10) is the scalable alternative but accuracy for this use case unvalidated.
- Interpretability signal alone insufficient to determine minimal FA budget: only 6.5% of heads are highly critical, but capability already deteriorates at ~10% retained FA. The interpretability ranking is informative but not yet complete.

---

## 2. Distillation

### 2.1 The transferable primitive: causal head-importance scoring

The paper's value to **katgpt-rs** is **not** the hybrid architecture (that requires training — §1.4, §1.5). It is the **causal-intervention head-importance score** that *selects* which heads to keep as FA. This score is:

- **Modelless**: forward-pass-only. The paper itself emphasizes it is "lightweight and one-shot, requiring only a few forward passes over a small calibration set."
- **Strictly stronger than RTPurbo's observational attention-mass scoring**. RTPurbo (R086) asks "does this head attend to the needle?" — observational. HydraHead asks "does corrupting this head's output drop the capability?" — causal. A head may attend strongly to the needle yet be overridden downstream (a *correlated bystander*); causal patching filters these out by construction.
- **Stable and length-robust**: ~6 samples to converge, top-K set transfers from short-context localization to long-context selection.

The paper's knockout study (Fig 9b) is exactly the *confirmatory* check that the score identifies load-bearing heads, not arbitrary subsets. This is the calibration quality argument RTPurbo lacks.

### 2.2 Prior-art surface (what already ships — must not duplicate)

| Mechanism | Where | What ships | What HydraHead adds |
|---|---|---|---|
| **RTPurbo** offline needle-based head calibration (R086, Plan 126) | `katgpt-rs/src/rt_turbo/calibration.rs` | Observational attention-mass scoring → `HeadCalibration { retrieval_set, local_set }` partition at `retrieval_head_ratio` | **Causal necessity** (activation patching IE score) — strictly stronger; **path patching** (indirect effect) — RTPurbo has neither |
| **FaithfulnessProbe** causal intervention (R244, Plan 278) | `katgpt-core/src/faithfulness/probe.rs` | Generic causal intervention on injected memory segments: `probe_intervention(memory, intervention, behavior_metric) -> Delta`. **Direct-effect only.** | (1) **Path patching / sender-receiver indirect effect** — one-step-back attribution. (2) **Span-level logit-difference readout with exponential decay** (Eq 9) — current probe is scalar. (3) Application to **per-attention-head outputs** as the intervention target (currently applied to memory slices, not heads). |
| **HeadSubstitutionGate** (R353, Plan 353, in-flight) | `katgpt-rs/crates/katgpt-core/src/functional_substitution/gate.rs` | Decides when to substitute a real head with a FuncAttn surrogate using IoU (cheap) + FaithfulnessProbe (expensive, cached). | Uses FaithfulnessProbe's direct-effect delta as the cached measurement. HydraHead's **path-patching sender score** would add an *indirect* contribution axis (a head might be a low-direct-effect but high-indirect-effect enabler of substitution). |
| **Sink-Aware** per-head sink classifier (Plan 287) | `katgpt-rs/src/data_probe.rs` | Per-head ternary classification (NOP/Broadcast/None) via stable-rank. Observational. | Orthogonal — classifies sink behavior, not retrieval necessity. Could *combine* with causal score (sink + necessity = 2D head typology). |
| **GDN2** attention variant (R070) | `katgpt-rs/src/gdn2/mod.rs` | Gated DeltaNet-2 as an O(1) recurrent attention decoder; decoupled erase/write gates. | HydraHead uses GDN as the LA branch. GDN2 already ships the LA side; nothing to add to GDN2. |
| **Luce Megakernel** layer-wise hybrid (riir-ai Plan 182) | `riir-ai/.plans/182_luce_megakernel_deltanet_inference.md` | 75% GDN + 25% GQA **layer-wise** hybrid inference (Jet-Nemotron PostNAS pattern). | HydraHead is **head-wise** within a layer, not layer-wise. The fusion: causal head-importance could drive the *layer assignment* in a layer-wise hybrid (currently fixed 3:1 by PostNAS), generalizing RTPurbo's per-head selection to layer-selection. |
| **QK-Restore** (R259) | research note | Per-matrix freeze/thaw: transplant pre-drift W_Q/W_K to recover long-range routing after SFT drift. | HydraHead's Stage-1 parameter migration is the *training-time* analog (initialize GDN from pretrained Q/K/V). QK-Restore is the *runtime* analog for a different failure mode. |
| **Olmo Hybrid paired-token gap** (R319) | research note | Per-token loss gap `Δ_i = ℓ_Tr − ℓ_Hyb` between transformer and hybrid; Proposition 1 `DKL ≤ log|V_τ|`. | HydraHead's *capability* justification for head-wise mixing — open-class content words need FA's high-precision retrieval (R319 finding (i)); closers need transformer's state-closure (R319 finding (ii)). Head-wise lets you keep both per-token. |

**Key gap (the fusion target):** FaithfulnessProbe ships direct-effect causal intervention but has **no path-patching (indirect-effect) mode** and **no span-level readout**. RTPurbo ships head-importance calibration but uses **observational attention-mass, not causal necessity**. **Neither is wired to the other.** The fusion is: *apply FaithfulnessProbe-style causal intervention to per-head outputs to produce RTPurbo-style head calibration, extended with path patching.*

### 2.3 Fusion — `R086 (RTPurbo) × R244 (FaithfulnessProbe) × R353 (HeadSubstitutionGate) × this paper`

**The unification:** make causal intervention the single source of truth for *head importance*, used by:

1. **RTPurbo calibration** (Plan 126): replace `calibrate_from_scores(attention_mass_scores)` with `calibrate_from_causal_scores(IE_scores)` — same `HeadCalibration` partition shape, strictly stronger score.
2. **HeadSubstitutionGate** (Plan 353): the cached `FaithfulnessProfile.behavior_delta_when_replaced` becomes a *causal-necessity* score; add the **sender/indirect** axis so a head that enables substitution indirectly (without writing the substituted feature itself) is also flagged.
3. **Layer-wise hybrid assignment** (riir-ai Plan 182): causal importance generalized from per-head to per-layer aggregation drives the 3:1 layer mask instead of PostNAS fixed ratio.
4. **Sink-Aware typology** (Plan 287): combine `SinkKind` × `IE_score` into a 2D head typology — `Broadcast + high-IE` heads are the true retrieval-critical sink heads; `NOP + high-IE` are state-closure heads (R319 finding (ii)); `Broadcast + low-IE` are correlated bystanders safe to convert.

This produces a **unified head-importance diagnostic** that strengthens three already-shipped primitives (RTPurbo, HeadSubstitutionGate, Sink-Aware) and one riir-ai runtime (Luce Megakernel layer assignment). It is not a new capability class — it is a **measurement-quality upgrade** that makes the existing head-routing decisions more reliable.

### 2.4 The other modelless distillate: scale-normalized heterogeneous fusion

Eq 13–14 is a small, generic primitive: when concatenating outputs from two heterogeneous attention mechanisms (FA+GDN, dense+Raven, FA+sparse), apply **independent RMSNorm per branch** then a **learnable per-head γ scalar** before the output projection. The paper proves static γ > dynamic softmax gate (Table 5). This is modelless and applicable to any future head-wise or branch-wise mixing we ship — currently *not* needed (Plan 182 mixes layer-wise, not head-wise), but ships as a small primitive ready for any future head-mixing runtime.

### 2.5 Latent-to-latent reframing (mandatory per research skill)

How does causal head-importance look when operating on the seven Super-GOAT factory substrates?

- **(a) HLA per-NPC latent state** (`katgpt-core/src/sense/`, `riir-engine/src/hla/`): a "head" in HLA terms is a direction vector in the 8-dim affect space. The causal-importance question becomes: *which HLA direction vectors are causally load-bearing for a given NPC behavior?* This is exactly what `evolve_hla` lacks today — R244's finding that agents "silently ignore condensed memory" applies: an HLA direction may project strongly onto an action yet be overridden downstream. **A per-direction causal probe (analogous to per-head IE score) would close this gap.**
- **(b) `latent_functor/`**: a "head" = a functor application channel. Causal importance = does zeroing this channel break `coherence > tau_reest`? This is a *functor-channel importance* probe — currently `quality_gate.rs` measures aggregate coherence, not per-channel. **The path-patching sender score generalizes to "which upstream functor channels feed the re-estimation trigger."**
- **(c) `cgsp_runtime/`**: curiosity class routing (`curiosity_class_router.rs`) — causal importance of a curiosity class = does suppressing it collapse exploration diversity? **Currently observational (entropy signal); causal patching would be strictly stronger.**
- **(d) LatCal fixed-point**: not applicable — LatCal is raw numeric commitment, not latent.
- **(e) `NeuronShard` dendritic branch** (`riir-neuron-db/src/shard.rs`, `dendritic_lora` feature): **strong match.** A dendritic branch is a sub-circuit of `style_weights[64]`. Causal importance of a branch = does masking it drop retrieval quality for the shard's zone? This is the *per-branch necessity* analog of per-head necessity. Currently `dendritic_lora` ships the branch view but no causal probe on it. **Fusion target: causal dendritic-branch importance → selective branch freeze/thaw.**
- **(f) DEC operators**: not directly applicable.

The HLA (a) and NeuronShard dendritic (e) reframings are the most interesting — both are *latent-state substrates where causal-intervention importance is currently missing*. However, both are riir-* private concerns (HLA tuning is riir-ai; dendritic branch is riir-neuron-db). The katgpt-rs public primitive is the **generic causal head/channel importance scorer** that both can consume.

---

## 3. Verdict

### 3.1 Tier

**GOAT.**

**One-line reasoning:** the causal head-importance score is a provable quality gain over RTPurbo's observational attention-mass calibration (causal > observational by construction; paper's knockout study confirms), and path patching extends our shipped FaithfulnessProbe from direct-effect to indirect-effect — but it is **not a new capability class** (causal intervention already ships as FaithfulnessProbe R244, Super-GOATed; this applies it to a new unit of analysis and adds the indirect axis). The head-wise FA/LA architecture itself requires training → riir-train.

### 3.2 Novelty gate (Q1–Q4)

| Q | Answer | Evidence |
|---|---|---|
| Q1: No prior art? | **NO** — causal intervention ships as `FaithfulnessProbe` (R244/Plan 278); head-importance calibration ships as RTPurbo (R086/Plan 126); head-substitution-via-causal-intervention is in-flight as HeadSubstitutionGate (R353/Plan 353). | grep `causal\|activation.{0,5}patch\|head.{0,5}importance` across notes+code returns all three. |
| Q2: New class of behavior? | **NO** — better measurement of an existing capability (head importance), not a new capability. | RTPurbo already selects retrieval heads; this selects them more reliably. |
| Q3: Product selling point? | **NO** — "our head calibration is causally validated" is a quality claim, not a pillar-level moat. | RTPurbo's selling point ("15% of heads are retrieval-critical, rest are local") is unchanged; this just makes the 15% set more accurate. |
| Q4: Force multiplier? | **YES** — connects RTPurbo (R086) × FaithfulnessProbe (R244) × HeadSubstitutionGate (R353) × Luce Megakernel layer assignment (riir-ai Plan 182) × Sink-Aware typology (Plan 287). | 4-5 systems. |

Q1=NO → **not Super-GOAT.** Proceed to GOAT/Gain.

### 3.3 GOAT gate (per transformer-stack slot)

| Gate | Criterion | This primitive |
|---|---|---|
| G1 correctness | Causal IE score matches paper's knockout ranking on a synthetic head harness | Reproducible — synthetic FA heads with known load-bearing subset; IE score must rank them above bystanders; knockout must collapse capability. |
| G2 perf/quality | Causal calibration produces a *different* (more accurate) head partition than RTPurbo attention-mass on a controlled workload | On a synthetic harness with planted correlated-bystander heads (attend strongly to needle but overridden downstream), causal must exclude them while attention-mass includes them. |
| G3 no-regression | Calibration latency within 2× of RTPurbo's needle-mass calibration | Causal patching is `O(n_heads × n_calibration_samples)` forward passes; RTPurbo is `O(1)` forward pass + per-head mass scan. Expect ~10–100× slower calibration but offline (one-time per model). Acceptable since calibration is amortized. |
| G4 alloc-free / hot-path | Calibration is offline; the *resulting partition* is a `Vec<usize>` consumed at inference with zero overhead | Same as RTPurbo — the partition is config data. |

**Promote/demote tracking (per katgpt-rs MOAT §1.6):** this primitive occupies the **calibration slot** of the RTPurbo stack. Two calibration modes now compete: `attention_mass` (current default, R086) and `causal_necessity` (this plan, R362). GOAT gate decides: if `causal_necessity` produces a measurably more accurate partition (G2), promote to default calibration mode and demote `attention_mass` to opt-in fallback. If they agree on most workloads (no quality gain), keep `attention_mass` default and leave `causal_necessity` opt-in for the long-context-extreme regime where bystander heads matter.

### 3.4 MOAT gate (per domain, §1.6)

- **katgpt-rs MOAT**: "paper-derived fundamental/principle primitive passing GOAT/Gain via fusion, promote/demote tracked per stack (transformer stack + 2D toy games)". **Causal head-importance calibration fits exactly** — it is a transformer-stack-slot primitive (calibration) that passes GOAT via fusion (R086 × R244 × R353). ✓ In scope, ship in katgpt-rs.
- **riir-ai MOAT**: not a pillar-level contribution. The latent reframings (HLA direction importance, NeuronShard dendritic branch importance) are *private follow-ups*, not the open primitive. Note as future riir-ai / riir-neuron-db cross-refs.
- **riir-train MOAT**: the architecture (head-wise FA/LA mixing, three-stage transfer, QKV decomposition) is training-only. Note "→ riir-train" and stop for that axis.

**Strengthens moat: yes (calibration slot upgrade), in-scope (katgpt-rs transformer stack), not pillar-level.** Ship behind feature flag `causal_head_importance`, opt-in, GOAT gate decides promote/demote vs `attention_mass`.

### 3.5 Architecture → riir-train

The following are **training-only** and out of scope for this workflow — noted and stopped:

- Head-wise FA/LA partitioning as a *trained architecture* (requires initializing GDN branches from pretrained Q/K/V + distillation).
- Three-stage transfer pipeline (Stage 1 MSE alignment, Stage 2 KL distillation, Stage 3 long-context NTP).
- Branch-specific refinements (FA NoPE+scale+gate, GDN RoPE+MHA expansion, query decomposition).
- The 15B-token scaling run.

**→ riir-train** for all of the above. The katgpt-rs deliverable is the *modelless head-importance scorer* + *scale-normalized fusion primitive* that any future trained hybrid would consume at inference.

---

## 4. Plan

See [katgpt-rs/.plans/358_causal_head_importance_calibration.md](../.plans/358_causal_head_importance_calibration.md).

**Scope:** ship `CausalHeadImportance` scorer (activation patching + path patching + span-level readout) + `ScaleNormalizedFusion` primitive, both modelless, both feature-gated. Wire as an alternative calibration mode in RTPurbo (`calibrate_from_causal_scores`). GOAT gate G1/G2/G3/G4. Promote/demote vs `attention_mass` per the §3.3 tracking rule.

**What this plan does NOT do:** train a hybrid model, implement the three-stage transfer pipeline, implement branch-specific architecture refinements. Those are riir-train.

---

## 5. Cross-references (future, deferred)

- **riir-ai**: causal importance of HLA direction vectors (§2.5(a)) — private follow-up, applies the open `CausalHeadImportance` primitive to HLA's 8-dim affect space. Cross-ref when scoped.
- **riir-neuron-db**: causal importance of `NeuronShard` dendritic branches (§2.5(e)) — private follow-up, applies the primitive to `dendritic_lora` branch views for selective branch freeze/thaw. Cross-ref when scoped.
- **riir-train**: head-wise FA/LA mixing architecture + three-stage transfer pipeline — note "→ riir-train" and stop.

---

## TL;DR

HydraHead's *architecture* (head-wise FA/LA mixing, three-stage transfer, branch refinements) is **training-only → riir-train**. Its *modelless crown jewel* is the **causal head-importance score** (activation patching IE + path patching sender + span-level logit-diff readout), which is strictly stronger than RTPurbo's observational attention-mass calibration, plus the small **scale-normalized heterogeneous fusion** primitive (independent RMSNorm per branch + learnable per-head γ). **Verdict: GOAT** — causal intervention already ships as `FaithfulnessProbe` (R244, Super-GOATed) and head calibration ships as RTPurbo (R086); this fuses them (R086 × R244 × R353) into a unified causal head-importance diagnostic that upgrades the calibration slot of the RTPurbo stack, extends FaithfulnessProbe with path-patching indirect effects, and enriches HeadSubstitutionGate with a sender axis. Not Super-GOAT (Q1 NO — prior art; Q2 NO — not a new capability class). Plan 358 ships the open primitive behind `causal_head_importance` feature flag; GOAT gate decides promote-to-default vs `attention_mass`. Latent reframings (HLA direction importance, NeuronShard dendritic branch importance) noted as private riir-ai / riir-neuron-db follow-ups.
