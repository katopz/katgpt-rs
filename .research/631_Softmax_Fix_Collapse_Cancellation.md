# Research 631: A Fix for Softmax and Cross-Entropy (Softmax Collapse + Cancellation)

> **Source:** "A Fix for Softmax and Cross-Entropy" — Jonathan T. Barron, Dan Gnanapragasam, Michael Tschannen (Google DeepMind), [arXiv:2610.31642](https://arxiv.org/abs/2610.31642), 2026-10-22
> **Date:** 2026-10-23
> **Status:** Active
> **Related Research:** — (closest in-house cousins are code, not notes: `katgpt-core/src/ssmax.rs`, `katgpt-attn/src/dash_attn/asentmax.rs`, `noisy_or_stable` in `katgpt-core/src/lib.rs`)
> **Related Plans:** — (filed as issues: katgpt-rs Issue 958 — modelless numerics primitives; riir-train Issue 645 — training-loss fixes)
> **Cross-ref (riir-train):** Issue 645
> **Classification:** Public

---

## TL;DR

Standard max-subtraction softmax/log-softmax/cross-entropy (every framework's, and every one of OUR hand-written Rust loss sites) carries two floating-point bugs. **Softmax Collapse**: the denominator `1 + e^{-m}` absorbs `e^{-m}` once the logit margin m passes a precision cliff c_d = p·ln2 (≈5.5 bf16 / 16.6 fp32 / 36.7 fp64, p = significand bits), so the target logit's gradient rounds to 0 while non-target gradients stay positive → gradients stop summing to zero → under Adam this produces the "Slingshot" loss spikes and breaks grokking. **Softmax Cancellation**: soft-target CE (distillation) computes softmax(student) and softmax(teacher) separately and subtracts; near-equal logits lose the difference (error ~eps·p vs true difference ~δ·p). The fixes are drop-in log-space forms: log-softmax via softplus with the argmax **excluded from the LSE** (target gradient = −e^{−m}/(1+e^{−m}) ≡ **−σ(−m)** — the fixed CE gradient IS a sigmoid), and soft-target q−p from δ = z_S − z_T via expm1/log1p (exactly 0 at equality). Verdict: **Gain (GOAT-tier adoption)** — modelless numerics primitives in katgpt-core (instrument-grade) + live loss fixes in riir-train. This paper is the workspace's "sigmoid, never softmax" doctrine, cited.

**Distilled for katgpt-rs (modelless, inference-time):**
The fixes are deterministic numerics code — nothing needs training to implement or verify. Closed-form extractions: (1) fixed log-softmax as a forward primitive whose log-prob resolution extends from the cliff c_d out to exponent underflow (~87 nats fp32) instead of saturating at exactly −0.0; (2) the identity fixed-CE-target-gradient ≡ −σ(−m) (doctrine ammunition + a property test for `fast_sigmoid` exactly in the saturation band where approximations deserve their hardest audit); (3) precision-cliff constants c_d = p·ln2 as a lookup table (the paper states 5.5/16.6/36.7 but not the closed form — the derivation is ours); (4) margin-equilibrium law m* ≈ ln(1/ε) as an offline config-audit predictor (default Adam ε=1e-8 → m*≈18.4 sits just ABOVE the fp32 cliff 16.6 → predicts the Slingshot regime from config alone); (5) log-space probability differencing (`logspace_diff`) with a faithful-zero guarantee at equality.

---

## 1. Paper Core Findings

1. **Softmax Collapse (Bug 1).** Max-subtraction sets the target/max logit slot to 0, so the denominator is `1 + e^{-m}`. FP has fine precision near 0 but coarse precision near 1: once m > c_d, `e^{-m}` is lost when added to 1.0. Consequences: forward `logsoftmax(z)_t` evaluates to exactly 0 (loss zeroed); backward `g̃_t = q_t − 1 = 0` while non-target gradients `g̃_j = e^{z_j−z_t}/(1+e^{-m})` stay positive down to exponent underflow → Σg̃ ≠ 0. The fix never adds 1 to e^{−m}: target gradient = −e^{−m}/(1+e^{−m}); logsoftmax via `softplus(−m)` with the argmax excluded from the LSE; hard-target CE = `softplus(−m)` with the t-slot holding the constant 0 (so autodiff derives g_t as the negated sum of non-target gradients — zero-sum by construction).
2. **Softmax Cancellation (Bug 2).** Soft-target CE computes q = softmax(z_S) and p = softmax(z_T) separately, gradient g̃ = q − p. As z_S → z_T the subtraction of nearly-equal probabilities is catastrophic cancellation (error ~eps·p per component), and forward-softmax vs backward-logsoftmax op-sequence mismatch leaves a non-zero residual even at exact equality when the two are evaluated differently. Fix: q−p in log space from δ = z_S − z_T via expm1/log1p → exactly 0 at δ = 0.
3. **Margin equilibrium.** Under AdamW + weight decay, post-memorization logit margins saturate at m* ≈ ln(1/ε). Default ε = 1e-8 → m* ≈ 18.4, just above the fp32 cliff 16.6 → standard CE hits the cliff right before equilibrium → Slingshot. ε ≥ 1e-7 avoids the fp32 cliff (m* ≤ 16.1).
4. **Demonstrated effects.** Eliminates Slingshot spikes, speeds grokking with weight decay (94±5% → 100±0% on a+b mod 31), prevents accuracy collapse without weight decay (23% → 85% on a²+b² mod 97), stops student drift from teacher at distillation init (the bf16 residual ~1e-3 exceeds ε=1e-8 and Adam normalizes it into full-size drift steps). Drop-in, no meaningful slowdown, bf16/fp32/fp64. Beats the alternatives: zero-sum gradient projection (Liu et al. 2605.06152) stalls margins at ~18 and drops accuracy <50% at small ε; StableMax (Prieto et al. 2501.04697) blows margins to 10³–10⁴.

## 2. Distillation

### 2.1 The cliff constants are a closed form (ours — the paper states numbers, not the law)

c_d = p·ln 2, p = significand bits of the format where `1 + e^{-m}` is EVALUATED:

| format | p | c_d (nats) |
|---|---|---|
| fp8 e4m3 | 4 | ≈ 2.8 |
| bf16 | 8 | 5.545 |
| fp16/tf32 | 11 | ≈ 7.6 |
| fp32 | 24 | 16.64 |
| fp64 | 53 | 36.73 |

Matches the paper's 5.5/16.6/36.7. ⚠ The cliff binds in the ACCUMULATOR format, not the storage format: this stack stores f16/bf16 weights but accumulates f32, so the operative cliff stack-wide is 16.6 (verified: no bf16 loss path exists — GPU lanes dequantize to f32 before loss/backward). A bf16-accumulated path would collapse the cliff to 5.5 — scoping the table per-accumulator is part of the primitive's API, not a footnote.

### 2.2 The fixed target gradient IS a sigmoid

`g_t = −e^{−m}/(1+e^{−m}) = −σ(−m)`. The paper's fix is this workspace's founding doctrine ("sigmoid, never softmax") derived independently: a numerically-correct softmax converges to a sigmoid at the margin. Ships as: (a) a doc-citation beside the sigmoid law in katgpt-core; (b) a property test pinning `fast_sigmoid`'s relative error vs the exact form over m ∈ [5.5, 40] — precisely the saturation band where the Cephes polynomial and early-exit envelopes live.

### 2.3 Modelless extraction inventory (Path 0 — all rows EXTRACT, nothing defers)

| Component | Analog ships? | Extraction → home |
|---|---|---|
| Fixed log-softmax (argmax-excluded LSE + softplus) | NO — every own-code site is standard max-subtract f32 (~10 hand-rolled variants; grep agent, 8 repos) | katgpt-core primitive, feature-gated (Issue 958 T5) |
| Fixed hard-target CE (softplus form + expm1 target grad) | NO — `grad = softmax − onehot` verified live at 8+ riir-train sites | riir_train_engine::math primitive + site fixes (Issue 645 T1) |
| Soft-target log-space q−p (expm1/log1p) | NO — probability-space subtraction verified at `vq_codebook/backprop_arms.rs` L68-71; `loss_uopsd/kl.rs` gradient `q_j·S − p̃_j` same family | riir-train F2 (Issue 645 T2) + katgpt-core `logspace_diff` (Issue 958 T4) |
| Cliff constants c_d = p·ln2 | NO | katgpt-core const table (Issue 958 T1) |
| Margin equilibrium m* = ln(1/ε) | NO — all audited Adam configs at ε=1e-8 | offline config-audit instrument (Issue 958 T3); riir-train ε audit (Issue 645 T3, unbundled) |
| `fast_sigmoid` audit band | partial (fast_sigmoid ships; no cliff-band property test) | property test (Issue 958 T2) |

**Strongest modelless consumer:** an f64 golden-NLL lossy-quant gate oracle built on the fixed log-softmax. The workspace's own lossy-surface promotion rule already distrusts flat aggregate perplexity (43–45% exact-match flips under BF16 with flat eval aggregates is the recorded shape); per-token NLL at confident positions is where quant damage hides BELOW the collapse floor — two quant variants can differ 1e-6..1e-3 nats on confident tokens and both read as exact zeros through an incumbent fp32 log-softmax. Offline, inference-only, upgrades an existing gate family (si_zero_shot_ppl golden arm, PQ2_0 / GDN_STATE_HALF lossy arms) rather than inventing a consumer (Issue 958 T6).

### 2.4 Honest scoping (pre-conceded)

The forward absolute-error win is ≤ machine-epsilon per value. Every inference-side benefit is a **differencing** benefit (KL checks, drift verification, quant deltas, comparison lanes, capture fidelity). The fixed log-softmax is **instrument-grade, not product-grade**: it ships as oracle/evaluator/alternate, never over a serving-forward path — the clef lane's expf-fidelity contract and attention's O(1) convex mixtures are correct as-is. Selling it as a serving-forward replacement would burn G5 bit-parity capital for zero measurable return.

### 2.5 Fusion

paper 2610.31642 × in-house `noisy_or_stable` (expm1/log1p "keeps resolution", `katgpt-core/src/lib.rs` L94-98) × `loss_dpo.rs::log_sigmoid` (−softplus(−x), 3-branch stable form, riir-train) = **one numerics doctrine, three shipped instances, zero unification**: the paper's forms are the same resolution law pointed at softmax/CE. The fusion deliverable is consolidation — a single stable-form family in katgpt-core (`logspace_diff`, fixed log-softmax, cliff table) that riir-train's loss sites consume, instead of the current ~10 hand-rolled max-subtract variants. `ssmax`/`asentmax` are orthogonal (they reshape logit geometry pre-softmax; the paper fixes FP evaluation structure) — both can compose.

## 3. Verdict

**Tier: GAIN (GOAT-tier adoption).** Not Super-GOAT: no new behavior class, no product selling point, and we claim NO novelty — the diagnosis is published (Prieto et al. 2501.04697 coined "Softmax Collapse"; Liu et al. 2605.06152 published the mechanism + a projection remedy; TensorFlow publicly documented it with merged f64-only VJP fixes Sep 2026, #126525/#126633/#126748, fp32 #128728 still open at Barron's submission; JAX #13529 documents the zero-sum class since 2022). Barron's two fix IMPLEMENTATIONS have no found prior art, and m* = ln(1/ε) appears novel — but for an adoption verdict the landscape is supporting evidence (Barron's forms are the best available fix per their own Table 1), not a moat. Frameworks will adopt this; our edge is only that our losses are hand-written Rust and must be fixed by hand, now, in the exact lanes (distillation, LoRA SFT) the paper names.

**MOAT gate (per domain):**
- `katgpt-rs` — fits: fundamental base numerics primitives (stable-form family, cliff table), sigmoid doctrine citation. Public-safe.
- `riir-train` — fits: training-loss correctness in the repo's own lanes. (Files there via issue, not a public note.)
- `riir-reflex`/`riir-instinct` — immune by construction (sigmoid-BCE decision lanes; verified structurally cannot hit either bug).
- `riir-infer` — consumer only (f64 golden-NLL oracle arm); forward kernels untouched (§2.4).

**Per-track:** (a) modelless inference — GAIN (Issue 958). (b) self-adaptive — PASS (sigmoid lanes immune; −σ(−m) identity is doctrine ammunition). (c) model-based training — GAIN, PRIMARY (Issue 645).

## 4. Coverage + prior-art record

Grep agent (8 repos, read-only): every softmax/CE/KL site in own code uses standard max-subtract f32; neither fixed form ships anywhere; softplus exists (26 files riir-train, 14 katgpt-rs) but never as log-softmax; expm1/log1p exist (noisy-OR, decay/gating) but zero CE paths; the house sigmoid law is real and load-bearing in decision layers (461 sigmoid-only files in riir-ai alone) but does not reach transformer forwards, attention kernels, or training losses. Canonical in-house specimens: riir-rethink `instinct_laya_head.rs::ce_dlogits` (`p − onehot` — the cleanest Bug-1 form in the workspace, feeding the Plan-425 teacher fine-tune + RIDT teacher-prob dump); riir-train `loss_uopsd/kl.rs` gradient `q_j·S − p̃_j` whose zero-sum test tolerance absorbs the Bug-2-class defect. Reproduction instrument already in-tree: `katgpt-rs numeric_stability/lab.rs` (perturbable tiled online-softmax, f64, mantissa-bit quantization) can replay the paper's cliff figures.

## 5. Routing

| File | Content |
|---|---|
| `katgpt-rs/.issues/958_softmax_fix_numerics_primitives.md` | modelless primitives T1–T6, sequencing d→b→c-audit→e→a→c-gauge, determinism-law seam |
| `riir-train/.issues/645_softmax_collapse_cancellation_loss_fixes.md` | F2/F1/F3 recipes, verified line numbers, probe step 0, margin-histogram gate, G3 waiver convention |

Numbering: `.research/.highwater` 630 → 631; `.issues/.highwater` 957 → 958 (katgpt-rs), 644 → 645 (riir-train); dual_allocation_gate clean on both repos at allocation time.

## 6. Verifier negotiation record (§5)

Round 1 REVISE → round 2 AGREE (harness `request_verdict`, native claude reviewer, session 2aa1d2eb). Material corrections adopted:
- **F2 "provably fires at step 0" WITHDRAWN.** `student_prob` (riir-train `head_train.rs:359-374`) and `softmax_logits_into` (`real_kl_bench.rs:621-636`) are the SAME op sequence (mx over `lg·inv_tau`; `(slot − mx).exp()`; left-to-right sum; `/= s.max(1e-12)`) → bit-identical logits give exactly 0. The paper's non-zero-at-equality requires the two softmaxes to be evaluated DIFFERENTLY — not the case here.
- The advocate's header citation was **misattributed** (not invented): "student==teacher at zero gates" lives at `real_kl_bench.rs:6` and describes the RETIRED Plan-445-T2.2 micro-proxy degeneracy, not the live bench. In the live bench teacher = base on `[context ‖ query]` vs student = base+adapter on `[query]` alone → δ starts LARGE at init and shrinks at convergence, so the cancellation regime is LATE training — exactly the regime the bench's final-KL measurement needs clean. The probe (Issue 645 T0) is the instrument that tests the claim either way.
- F2 restated: near-equality precision loss, per-component error ~eps·p_i vs true difference ~δ·p_i·O(1); corrupted when δ ≈ eps (≈1.2e-7 f32), i.e. immediately after the first updates; whether that matters under the bench's AdamW (lr 1e-3, kl_epsilon 1e-8) is UNMEASURED. Ranked first on COST (CPU-minutes), never on provability.
- Bug-1 stays margin-conditional: the cliff mechanism is code-path-independent, but whether our short LoRA runs ever reach m > 14..16.6 is unknown until the margin histogram runs (<1 GPU-h, Issue 645 T0a).
- Bonus finding adopted into 645: `student_prob` re-loops the full vocab per index i → the L68-71 gradient loop is O(V²) per row; the log-space rewrite computes the LSE once → O(V) (recorded side effect).
