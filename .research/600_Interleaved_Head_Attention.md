# Research 600: Interleaved Head Attention (IHA) — Cross-Head Pseudo-Mixing

**Status:** RECORD (verdict landed 2026-10-01; the katgpt-rs op is TRAINING-DEPENDENT and gated on riir-train Issue 606 arms 4+5 — no primitive ships from this note alone)

> **Source:** Duvvuri, Ekbote, Bansal, Tiwari, Khatri, Brandfonbrener, Liang, Dhillon, Zaheer [arXiv:2602.21371 "Interleaved Head Attention"](https://arxiv.org/abs/2602.21371) (Meta FAIR / UT Austin / UC Berkeley / Harvard / MIT, 24 Feb 2026)
> **Date:** 2026-10-01
> **Classification:** Public
> **Related Research:** 086 (RTPurbo — convergent multi-key-retrieval bottleneck signal), 362 (HydraHead causal head-importance — the documented "future head-mixing runtime" gap this paper's mechanism fills)
> **Related Issues:** `katgpt-rs .issues/914` (deterministic mixing runtime, BLOCKED-ON evidence), riir-train `.issues/606` (MHA-vs-IHA micro falsification)

---

## TL;DR

IHA breaks MHA's head isolation: P pseudo-heads per head, each a linear combination of ALL H heads' Q/K/V (mixing tensors `α_Q,α_K,α_V ∈ R^{H×H×P}`), interleaved into a length-N·P sequence with per-virtual-token RoPE phases, then STANDARD per-head attention (FlashAttention-compatible) and a collapse map `R ∈ R^{H×HP}`. Up to P² attention patterns per head for 4H²P params. The paper's OWN THEOREMS are deterministic constructions (one-hot routers, ±1 sign-flips) — only the benchmarks need 240B training tokens. The deterministic distill: **pseudo-head mixing as an opt-in, training-dependent runtime cross-head interaction op** (Feature-Flag step-5 posture), with the implementation gate deferred to a ≤1 GPU-h micro falsification (riir-train Issue 606).

**Distilled for katgpt-rs (deterministic, inference-time):** three constructions from the paper's proofs — (1) the REPLICATION construction (α = 1(m=i) for every pseudo j, R selecting block 1) = exact MHA inclusion in exact arithmetic (the superset-theorem witness; NOTE the paper's one-hot ROUTERS in the filter/CPM-3 proofs are cross-head mixing, NOT identity — equality holds only in the algebraic no-RoPE/no-mask formulation and at f32 tolerance, never bit-for-bit through the interleaved forward), (2) ±1 sign-flip pseudo-pairs = a difference-of-softmax nonlinearity on the repeated-token subspace where MHA is provably LINEAR (Thm 2's separation witness, no λ hyperparameter unlike Diff Transformer), (3) the √k pairwise factorization (k relational patterns from √k query-side × √k key-side generators, `A^i = A^{(h−1)H+(j−1)}`).

---

## 1. Paper core

- **Mechanism** (Alg. 1): project X → per-head Q,K,V `[N,H,d]`; einsum with α (`mhp,nmd→hpnd`) → `[H,P,N,d]` pseudo-tokens; `merge_pseudo` interleaves P into the sequence axis → `[H,NP,d]`; standard scaled-dot-product causal attention per head; un-merge + collapse `R: [H,NP·d]→[H,N·d]`; output projection. FlashAttention-compatible (step 4 IS standard attention); RoPE generated for length NP (each (n,p) virtual token its own phase).
- **Cost**: global IHA O(P²N²d) — factor P² over MHA; mitigated by hybrid schedule 4 sliding-window IHA layers (window `W = N/(2P²)`) : 1 global → 3/5·HN²d ≈ MHA global.
- **Theory**: (T1) `M ⊊ P_P` for P≥2 — strict superset at +4H²P params; separation via P=2 sign-flipped pseudo Q/K producing a tanh-like difference-of-softmax map on repeated-token inputs (every MHA is linear there). (T2) polynomial filter bank `[X,AX,…,A^{k−1}X]`: MHA needs k heads / Θ(kn²) params; IHA ⌈√k⌉ heads / Θ(√k n²) via `A^{(h−1)H+(j−1)}` pairwise factorization (one-hot α). (T3) CPM-3 (order-sensitive modular counting): MHA Nmax heads / Θ(N³); IHA ⌈√Nmax⌉ / Θ(N^2.5).
- **Empirics** (2.4B, 240B tokens, 128 H200, FLOP-matched): RULER Multi-Key Retrieval +27%/+32%/+112% @4k/8k/16k over Global Attention; RULER overall 44.0 vs 35.0 (beats Global+Local 40.6, Diff Transformer 37.2). After OpenThoughts SFT: GSM8K +5.8 Maj@16, MATH-500 +2.8. Appendix D synthetic (1 layer, 8 heads, ~206–240k params, 40k/5k/5k, BCE): binary R∘R (m∈{6..10}, p=.325) + ternary R∘R∘R (m∈{5..8}, p=.264) — IHA beats MHA AND simplicial attention by up to 4.7%.

## 2. §3.5 Path 0 decomposition (inventory)

| Component | Shipped analog? | Extractable without GD? | Disposition |
|---|---|---|---|
| Mixing tensors α_Q/K/V (learned) | **No** — heads are isolated everywhere we ship (HLA per-head state `HlaQHeadState`; katgpt-attn per-head loops; `spectral_pre_rotate` rotates each head's basis into a SHARED eigen-frame, no cross-head content mixing; `ScaleNormalizedFusion` fuses branch OUTPUTS post-attention with scalar γ — different seam) | Yes — the paper's proofs use deterministic α: REPLICATION (α = 1(m=i) ∀ pseudo j + R block-select = the MHA-inclusion witness; the one-hot ROUTERS of the filter/CPM-3 proofs are cross-head mixing, not identity), ±1 sign-flip (nonlinearity witness), dense ±1 Hadamard with 1/√H normalization (`bonsai2_hadamard` is the shipped dense-Hadamard precedent) | **→ katgpt-rs Issue 914** (training-dependent classification), BLOCKED-ON Issue 606 arms 4+5 |
| √k pairwise factorization | No (DEC composes d,δ NILPOTENTLY — d∘d=0 is the opposite regime; no adjacency-power banks ship) | Yes — pure algebra | **Audited discard**: no consumer at our scales (zone graphs N≈dozens → k-hop banks cheap anyway; d≥8 shards hit the DEC curse-of-dimensionality row; decorative risk named). Fusion line §6.1 |
| Interleave + per-virtual-token RoPE + `W=N/(2P²)` | No IHA forward exists to consume them | Yes — closed-form layout/schedule | **Audited discard** → recorded §6.3 (riir-infer watch) |
| ±1 diff-of-softmax contrast | Partial-by-name-only: riir-rag Clifford wedge is an exterior-product admission, not a softmax difference; no documented retrieval-side gap (reverse-grep clean on that axis) | Yes | **Audited discard** — fusion line §6.2 |
| Relation-composition + CPM-3 generators | n/a (fixtures) | Yes — closed-form ground truth | **→ consumed by Issue 606** |
| "Multi-key retrieval is the bottleneck" signal | Convergent: R086 RTPurbo + riir-train Plan 433 (arXiv:2609.39827 PPT verdict — positional-retrieval mechanism) | n/a (signal) | **→ redirect line added to Research 086** |

Paths 1–3 for the learned empirical gains: n/a — the gains install circuits at init; the deterministic mixing op above is exactly what Issue 914 specifies, now classified TRAINING-DEPENDENT (a model trained with, or tolerant of, the mixing is required — paths 1–3 cannot supply that).

## 3. Three-track adversarial panel record

- **No-GD advocate**: ran as subagent (2nd attempt; 1st attempt + the prior-art searcher returned corrupted output and were discarded). Verdict GO: "every theorem in this paper is a construction with zero learned parameters"; table of 7 items with hosts (katgpt-attn, katgpt-dec, katgpt-core) and GOAT gates; flagged the f32 associativity G1 caveat for the √k arm ((AB)C ≠ ABC — pin order or use bool bitsets).
- **Model-based advocate**: subagent failed TWICE with corrupted output; run in-process by the coordinator instead. Findings: (a) Appendix D replication is ≤1 GPU-h and rides the Plan 389 fixture trainer (`riir-bench-algo/src/{train,model}.rs`) — recipe in Issue 606 (arms 4 + 5 are OUR additions); (b) full IHA pretraining (2.4B/240B tok) out of scope — no production from-scratch trainer, we serve fixed checkpoints (Ternary-Bonsai-2-27B-PQ2_0, laya); (c) retrofit onto served MHA checkpoints: audited discard — IHA is a different function class (that IS Thm 2); a one-hot-initialized trained adapter is fine-tuning with no consumer; (d) the micro-rung's downstream consumer is Issue 914's classification itself (training-dependent vs closable). Verdict round 1 sharpened this: arm 4 (fixed-layout-trained) is a TRAINING-side finding, and the modelless-side question needs arm 5 (post-hoc insertion on trained MHA + random-orthogonal control); the HLA-as-fixed-projection-consumer exception was CHECKED AND REFUTED (`riir-ai/crates/riir-engine/src/hla/forward.rs` consumes learned-style `attn_wq/wk/wo` weight matrices).
- **Track priority (REVISED at verdict round 1)**: the original serving-envelope call (modelless PRIMARY) is retired — with the op reclassified training-dependent and no fixed-projection consumer in the stack, **riir-train Issue 606 is the only route to value**; the katgpt-rs op (Issue 914) is opt-in for good absent new evidence, and both tracks are blocked on 606's arms 4+5.

## 4. Coverage / closest cousins (signal-diffed)

1. `katgpt-core/src/causal_head_importance/` (Research 362 / Plan 358) — **the documented gap**: `ScaleNormalizedFusion` doc says "ready for any future head-mixing runtime (currently unused — Plan 182 is layer-wise)". Signal-diff: fuses two attention branches' OUTPUTS post-attention via per-head scalar γ; IHA mixes QKV pre-attention via full H×H×P tensors to create P² interaction PATTERNS. Different seam — the gap stands, and the head-importance scorer is the natural ranking input for WHICH heads to mix.
2. `katgpt-attn/src/funcattn_compose/spectral_pre_rotate.rs` — calibration-time per-head basis rotation into a shared eigen-frame; no cross-head content mixing; forward byte-identical after. Different.
3. `katgpt-hla` / `riir-engine hla` — per-head isolated streaming state (the kernel doc's own update order is per-q-head). No mixing.
4. Published: Talking-Heads (logit mixing after QK product) / Knocking-Heads (weight mixing post-softmax) — the paper's cited cousins, both mix AFTER the attention operator; Diff Transformer = learned λ·difference of two softmaxes (the ±1 deterministic construction reaches a diff-of-softmax with zero params and zero hyperparameters). Nothing published mixes QKV linearly ACROSS heads BEFORE attention while preserving the operator (§4 sweep: only the paper itself + aggregators; a third-party `Frankenstein Transformer` lib implements `iha_attn` — reference impl for any forward port, not prior art).

## 5. Reframes (required before verdict)

- **Game**: per-NPC HLA heads are isolated channels today; cross-head mixing would let one channel's query attend another channel's key/value patterns — composed relational state ("what I saw" × "what I heard" → threat) per NPC at H²P MACs/token (trivial at fixture H=4; a G2 question at production head counts). **Round-1 check: HLA is NOT a fixed-projection consumer** — `riir-ai/crates/riir-engine/src/hla/forward.rs` consumes learned-style `attn_wq/wk/wo` weight matrices, so mixing on HLA is training-dependent, same as any transformer surface.
- **Healer (consumer, #2)**: no MEASURED gap — retrieval is single-fan-out + structural rerank and no documented multi-key-composition deficit exists (reverse-grep clean). Speculative line §6.2 only; no file.

## 6. Verdict (per track, tiers not pooled)

- **Modelless track: Gain-conditional — reclassified TRAINING-DEPENDENT at verdict round 1.** Novel to workspace (Q1 ✓ — no cross-head mixing ships; gap documented in Research 362's own module), plausible new capability class for our per-head kernels (Q2 ~), NO product selling point (Q3 ✗), force-multiplier moderate (Q4). Not Super-GOAT. The reviewer's round-1 correction stands: the strongest available evidence arm (fixed-Hadamard-trained, Issue 606 arm 4) is a TRAINING-side finding — a primitive that needs a model trained with it is the Feature-Flag step-5 case (opt-in, riir-train dependency, no promotion to default), and no fixed-projection consumer exists (HLA checked and refuted). Issue 914 carries that classification; arm 5 (post-hoc insertion + random-orthogonal control) is the honest probe of inference-time composability. §3.6 note: the architectural claim is grep-proven; quality is unproven and no parity is claimed.
- **Training track: Gain (Path 0.5, secondary).** Issue 606 — the Appendix D protocol is fully specified, ≤1 GPU-h, rides the Plan 389 trainer; arms 4+5 are our additions carrying the classification evidence. Param-count honesty: the paper's Appendix D delta (208,513 − 206,849 = 1,664) does NOT equal 4H²P (2,048) at P=H=8 — T2 verifies the actual P/R shape before claiming param-matching. Full pretraining: out of scope. Retrofit: audited discard (§3).

**MOAT gate**: katgpt-rs — fundamental/base primitive via fusion ✓ (deterministic constructions + causal_head_importance fusion). riir-train — active moat, training-method micro-implementation ✓ (the 605 precedent, same trainer). riir-infer — watch only. riir-refine — fusion line only.

**Weakest point (named upfront, revised at round 1):** the katgpt-rs track is now honestly labeled TRAINING-DEPENDENT — its utility requires a model trained with (or tolerant of) the mixing, no fixed-projection consumer exists in the stack (HLA refuted by grep), and the arm-5 insertion probe is expected to show degradation (Thm 2); if arm 5 is catastrophic AND no trained-with-mixing consumer materializes, Issue 914 closes negative and this note's modelless track reduces to the recorded constructions. Secondary: the in-process Model-based panel seat (disclosed §3; its evidence is the paper's Appendix D + Plan 433/389 files, checkable).

**Arm-5 verdict (2026-10-02, [riir-train Bench 621](../../riir-train/.benchmarks/621_issue606_iha_m3_lane_arm1_arm5.md)): the close condition FIRED — Issue 914 CLOSED NEGATIVE** (file removed per noise-reduction; HISTORY row carries the paired-commit hash). Measured: post-hoc insertion of deterministic mixing into 12 trained MHA checkpoints destroys the learned relational function on BOTH tasks — ternary residual == the token-only-majority bound exactly (`150636/217661`, exact rational match; 100% of the learned margin gone), binary probes at/below the majority-class rate; harm basis-change-generic (`|5a − 5b| ≤ 0.4 pts`); `5id` == arm 1 exactly (harness proven). This note's modelless track reduces, exactly as pre-registered above, to the recorded constructions (§5 vocabulary stands as expressivity witnesses only). The training-side question (does training AROUND a fixed layout help — Issue 606 arm 4) stays open in riir-train; a strong positive there plus a materialized consumer is a fresh filing, not a reopen.

## 7. Routing (files created by this verdict)

| File | Repo | Role |
|---|---|---|
| `.research/600_Interleaved_Head_Attention.md` | katgpt-rs | this note |
| `.issues/914_pseudo_head_mixing_runtime.md` | katgpt-rs | CLOSED NEGATIVE 2026-10-02 (arm-5 catastrophic, Bench 621) — file removed per noise-reduction, record in HISTORY.md |
| `.issues/606_iha_relation_composition_micro_rung.md` | riir-train | the evidence rung (Path 0.5; arms 4+5 carry the 914 classification) |
| `.research/086_…md` addendum | katgpt-rs | convergent multi-key-retrieval signal |

## 8. Fusion ideas + watch

1. **√k factorization × DEC cochains**: pairwise operator products as a "polynomial filter" sibling to `hodge_laplacian` (Δ = δd + dδ composes two operators; the factorization composes √k×√k). Discarded for lack of a consumer — revisit IF a zone-KG multi-hop belief consumer lands (Social-domain k-hop reachability).
2. **±1 diff-of-softmax × Clifford wedge**: a zero-param contrast gate for `retrieve_diverse` admission. No measured gap — novelty TBD.
3. **riir-infer watch (no file)**: IF IHA checkpoints ship publicly, the forward-port concerns are: NP-sequence interleave layout, RoPE for length NP (per-virtual-token phase), `W = N/(2P²)` sliding window, 4H²P mixing tensors in the weight format, KV-cache at NP granularity. Reference implementation: `Frankenstein Transformer` `iha_attn` (third-party). G5-parity discipline applies as with any port.

## 9. Prior-art search record (§4)

Web sweep 2026-10-01 via web_search_prime: (a) headline "Interleaved Head Attention" — the paper + aggregators (Semantic Scholar, alphaXiv, ResearchGate, rosinality substack 2026-02-26, LinkedIn) + an Emergent Mind "InterTwining Attention" topic page citing IHA; (b) component "head mixing linear combination QKV projections before attention" — only educational MHA content + the paper's own HTML. No published kill of the mechanism. Internal-first grep (arXiv ID + title across workspace `.research/`): zero prior distillation.
