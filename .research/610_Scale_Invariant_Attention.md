# Research 610: Scale-Invariant Attention — Position-Dependent Logit Schedule + the Sigmoid Tilt Transfer

> **Source:** [Scale-invariant Attention](https://arxiv.org/abs/2505.17083) — Ben Anson, Xi Wang, Laurence Aitchison (Bristol / JHU), **arXiv:2505.17083 v2, revision 2025-12-17** (v1 2025-05-20), NeurIPS 2025.
> **Date:** 2026-10-08
> **Status:** Active — GOAT verdict; Plan 622 (katgpt-rs, modelless) + riir-train Plan 449 (CPT, secondary) filed same session. Verdict ping-pong: `request_verdict` claude_code reviewer, 2 rounds, final AGREE with 4 adopted amendments.
> **Related Research:** 392 (SSMax dilution — the shipped cousin's distillation), 135 (Parallax Attention), 140 (sigmoid extension), 549 (HoldTopK lemma)
> **Related Plans:** 411 (SSMax — shipped default-on, `ssmax_temperature`), 622 (this paper's modelless primitive — `scale_invariant_attn`), riir-train 449 (si-ppRoPE LoRA-CPT, SECONDARY)
> **Classification:** Public — generic inference engine mechanics (WHAT, not HOW)

---

## TL;DR

The paper asks what attention must satisfy to generalize from short training contexts to long inference contexts, and answers with two provable desiderata: **scale-invariant total attention** (per-decade unnormalized attention mass `E[Z_t^{tΔ}] = Θ(1)` as t→∞) and **scale-invariant attention sparsity** (per-decade entropy sub-log). Under a Gaussian assumption on logits, a **position-dependent affine logit transform** achieves both in closed form:

```text
L_t → a_t·L_t + m_t
a_t² = 2[log(t/τ+1) − log α + β/α],   m_t = −a_t² + β/α
boundary (a₀², m₀) = (1, 0)  ⟹  α = β = e^0.5      (single knob: τ = 10)
```

Zero learned parameters — the boundary condition pins both constants. Paired with **pp-RoPE** (low-frequency RoPE components dropped; effective angular base 1024 vs θ=10k) it beats LogN/SSMax, ALiBi, and NTK on zero-shot 16× length generalization (train@4k → val@64k, 162M/304M GPT-2-style, FineWeb).

**Commercial value for the stack:** a zero-parameter, zero-alloc, distance-indexed LUT in the QKᵀ epilogue that (a) gives our **normalized-sigmoid attention arm** its first length-generalization mechanism (the tilt-transfer theorem below — the novel part of this fusion), (b) composes with the existing `ssmax` socket under a mutual-exclusion ordering law, and (c) defines a long-context league axis (pp16k/pp64k cells + per-decade mass probe) we currently do not measure.

## Why the shipped cousin does not cover it (signal-diff, §3.6)

`katgpt-core/src/ssmax.rs` (Plan 411, Research 392) multiplies logits by **one scalar** `s_L·log(N)` — `SsmaxMode::{Fixed, Adaptive, HoldTopK}` are all **position-independent**: they consume aggregate sequence length N. The paper's mechanism consumes **per-position distance t = i − j** (Appendix A, Eq. 19: `L^ij = a_{i−j}·S^ij + m_{i−j}`). The paper's Fig. 1 is exactly this signal-diff made visible: uniform LogN scaling keeps total entropy low but forfeits attention to the local 100 tokens as context grows; the position-dependent form preserves local mass (a₀²=1 means distance-0 logits are untouched, with a ~τ-token unscaled ramp). Same object family (logit-level length temperature), different signal, different behavior class. ssmax_dominates_base (Lean) and this paper's harmonic-sum lemma are different theorems about different quantities (gold-retrieval mass vs per-decade mass balance).

## The math (what transfers)

- **Lemma 1/2 + Theorem 1 (paper):** if `E[Ã_t] = α/(t/τ+1)` and `E[Ã_t log Ã_t] = β/(t/τ+1)` then per-decade mass and unnormalized negentropy are Θ(1) — proofs are harmonic-sum integral bounds (App. D/E), elementary and Lean-able in the `ssmax_dominates_base` style.
- **Gaussian construction:** logits `L_t ~ N(m_t, a_t²)` give `E[e^{L_t}] = e^{m_t + a_t²/2}` and `E[L_t e^{L_t}] = (m_t + a_t²)e^{m_t + a_t²/2}` (App. B); solving the two target equations yields Eq. 16; validity needs only `β ≥ α·log α`.
- **Weak sparsity is empirical on the softmax arm** (paper's own admission): H ~ √log t measured (Fig. 2), not proven. Standard unscaled attention gives H ~ log t (App. I.1) — the negative control.
- **Sink carve-out:** App. J's Gaussianity check (Llama-1B/8B, Gemma-2-27B QQ plots) excludes the BOS attention sink. The schedule's premise does not model sinks; our implementation excludes the sink token from the transform (see Plan 622).
- **pp-RoPE pairing law (App. I.3):** scale-invariant RoPE FAILS to generalize; si-NoPE generalizes but underperforms; si-ppRoPE wins. LogN also prefers pp-RoPE. Reading: **low-frequency RoPE components interfere with any position-dependent logit transform.** This is a standing law for every future long-context run (adopted in riir-train 449).

## The novel part of the fusion — sigmoid tilt transfer (derived this session; reviewer re-derived independently)

The paper is entirely softmax/log-normal. Our `parallax_attn` ships a **normalized-sigmoid arm** `p_σ(i,j) = σ(q·k·s)/Σσ(q·k·s)`. Does the schedule transfer?

**Theorem (tilt transfer).** With `L_t ~ N(m_t, a_t²)` under the paper's schedule, `E[σ(L_t)] = (α/2 + o(1))/(t/τ+1)`.

*Proof sketch:* write `σ(L) = e^L·σ(−L)` and apply the exponential tilt `E[e^L f(L)] = e^{m+a²/2}·E_{L′~N(m+a², a²)}[f(L′)]`. Under the schedule `m + a² = β/α = 1` is constant, so `L′ ~ N(1, a_t²)`; as `a_t → ∞`, `σ(−L′) → 1{W<0}` pointwise and dominated convergence gives `E[σ(−L′)] → 1/2`. ∎

Consequences:
- **The same LUT serves both arms** — no new (a, m) derivation. The sigmoid arm's per-decade mass constant is `α·E[σ(−L′_t)]`, which runs from ≈0.29α (t=0, a=1) through ≈0.41α (t=64k, τ=10) to α/2 — Θ(1) with a slow monotone drift, **expected behavior, not a defect**.
- **The probit approximation is a trap.** `E[σ(L)] ≈ Φ(m/√(1+πa²/8))` mispredicts the mass decay as t^−2.26 in the growing-variance regime (true: t^−1). Any schedule built on the probit discount under-attends distant decades. Derive via the tilt, never via probit.
- **Error structure:** the gap from ½ is `Φ(−1/a) − 1/2` at leading order plus a logistic-vs-step correction. The exact quantity is `E[1/(1+e^{L′})]`, `L′ ~ N(1, a²)`, which equals `Φ(−1/a)` plus a correction of order **O(a⁻³)** (logistic minus step is odd about 0; the density's slope there scales as a⁻³). The decay of the *total* correction is slow — `O(1/√log t)` — because `a_t² = 2 log(t/τ+1) + 1` grows only logarithmically. **G1 law (per verdict review): compute `Φ(−1/a_t)` in closed form and assert the Gauss–Hermite value matches within a tolerance scaling as `C·a_t⁻³` (C fixed once in the test). Never assert the one-sided `1/(a√2π)` estimate as a bound (loose at small t), and never pin illustrative values (≈0.29, ≈0.41) — derive them from the formula at test time.**
- **Boundary pin simplifies:** the sigmoid arm at (a₀², m₀) = (1, 0) gives `E[σ(L_0)] = 1/2` exactly by symmetry.
- **What is lost vs the softmax arm:** the Θ(1)-negentropy theorem does not survive the bounded link (`E[Ã log Ã] ≈ −(α·a_t/√2π)/(t/τ+1)`), but the resulting entropy `H ~ a_t/√(2π) ~ √(log t/π)` is **provably** sub-log — the sigmoid arm gets a derivable weak-sparsity rate where the softmax arm only has an empirical one.
- **Generalization:** any bounded sigmoidal link satisfies the same tilt argument — "bounded-link scale-invariance via the tilt identity". Sigmoid is the stack-sanctioned instance (constraint #2), and `parallax_attn` already documents sigmoid's no-sink/no-overflow kernel properties, which compose well with a schedule whose premise excludes sinks.

## Implementation shape (seams verified in-code)

- **Distance-indexed ⇒ per-tile affine, not KV-baked.** The multiplier for cached key j changes as the query advances; the transform must ride the QKᵀ epilogue before online-softmax max tracking (FlexAttention `score_mod` shape — the paper's own implementation vehicle). For a fixed query the distance slice over keys is contiguous (reversed): unit-stride descending LUT reads + 1 FMA per score.
- **LUT:** 2×max_ctx f32 interleaved (a_t, m_t); ~512 KB @64k; O(N) build (<1 ms vectorized); preallocated at session start (`RopeFreqTable` precedent, Issue 024); read-only hot path; G4 trivial.
- **Seams:** `katgpt-core/src/scale_invariant.rs` (new, sibling of `ssmax.rs`) → `attention.rs` SDPA hook → `parallax_attn` via the existing per-key pre-normalization lane machinery (`ParallaxConfig::prior_logits`, gated `prior_logit_lane`, verified lines 187–263 — the additive m_t lane already exists in shape; a_t adds a per-key multiplicative sibling). `riir-infer/src/rope.rs` gains a pp-RoPE `RopeFreqTable` variant (kernel-only posture — see below). Diagnostics in `katgpt-attn/src/chiaroscuro/` (per-decade mass + entropy-in-range probes, τ sweep).
- **Ordering law:** when `scale_invariant_attn` is active, the SSMax length multiplier is BYPASSED — both are length temperatures; stacking double-sharpens. G1 no-double-temperature test.
- **Calibration precondition (verdict amendment 2):** the schedule assumes ~unit-variance base logits (the paper's models all carry QK-norm). Raw-logit checkpoints need a **training-free per-head calibration**: measure each head's logit spread once at load, store as a frozen BLAKE3-checked per-head scale table, normalize before applying the schedule.

## Path 0 inventory

| # | Component | Coverage | Modelless-extractable? | Disposition |
|---|---|---|---|---|
| 1 | Schedule (a_t, m_t) + boundary pin | Partial — ssmax.rs is the position-independent special case (signal-diff above) | YES — zero-param arithmetic | katgpt-core `scale_invariant_attn` (Plan 622) |
| 2 | Desiderata + per-decade mass/entropy probes | No analog (chiaroscuro lacks decade probes) | YES — diagnostics | Plan 622 G1 instruments |
| 3 | Harmonic-sum theorem (decade mass Θ(1)) | ssmax_dominates_base is a different theorem | YES — Lean-able | Plan 622 Lean trio |
| 4 | pp-RoPE pairing law | No (rope.rs standard only) | YES — freq-mask table | Plan 622 (kernel-only) |
| 5 | **Sigmoid tilt transfer** | **No analog anywhere** (internal grep 0 hits; §4 sweep: Apple's sigmoid self-attention theory has no length schedule) | YES — closed form, O(a⁻³) correction | Plan 622 (the novel part of the fusion) |
| 6 | Training recipe (CPT w/ si-ppRoPE) | quest_grammar LoRA + riir-train-gpu `distill_attention.rs` `DistillMode` exist | n/a — track (c) | riir-train Plan 449 (SECONDARY) |
| 7 | HLA transfer | n/a | **NO — audited discard:** `riir-engine/src/hla/` is recurrent-state linear attention; there are no per-key logits to transform, so the schedule's object does not exist in that form. | none |

## Per-track verdicts (never pooled)

- **Track (a) modelless: GOAT** (not Super-GOAT — Q2 game-behavior class is weak because HLA structurally excludes it; Q1/Q3/Q4 firm). Opt-in `scale_invariant_attn` until G1–G4 pass.
- **Track (c) model-based: applicable → riir-train Plan 449, explicitly SECONDARY.** The modelless LUT is the mechanism the serving path consumes (primary); the CPT recipe is how real checkpoints realize the gains. Staged: micro-distill kill-switch (`SdpaToSiPpRoPe`) → 0.4B pilot → 7B-dense LoRA-CPT → optional Bonsai-27B (GDN-hybrid: 48 GDN layers bound the leverage; attention layers are the minority). <100 4090-h total.

## Prior-art sweep (§4)

- Internal-first: arXiv ID + title verbatim across all 15 workspace `.research/` + `.plans/` trees — **0 hits**. Vocab-translated code sweep (ssmax/logn/alibi/entropy/sink/rope/logit-scale) — closest internal cousin `ssmax.rs`, signal-diffed above.
- Published: the paper itself (softmax schedule — published, NeurIPS 2025, cited by 4); Information Entropy Invariance (arXiv:2501.08570, Li et al. — position-independent entropy-invariance temperatures, the paper's own closest-cited family); SSMax/LogN (arXiv:2501.19399 + Nakanishi); Apple "Theory, Analysis, and Best Practices for Sigmoid Self-Attention" (sigmoid analysis, **no** length-generalization schedule); a Nov-2025 blog note (nor-blog) discussing the paper. **No published analog found for sigmoid × scale-invariant or the tilt transfer.**

## Caveats (named first, per verdict review)

1. **The tilt theorem is elementary once stated.** No prior statement found, but an elementary derivation is exactly the kind that surfaces later in a workshop paper; and it is numerically unvalidated until the G1 Gauss–Hermite fixture runs.
2. **Zero-weight-change serving is untested in the paper** — every paper experiment trains with the schedule (from scratch; v2 App. I.5 does Llama-2 7B CPT: degraded at 4k train length, far exceeds all alternatives at 16k/64k). Our G3 zero-shot-on-Bonsai gate may FAIL; until it passes the claim is "schedule ships modelless", not "zero-shot gains on arbitrary checkpoints".
3. **pp-RoPE zero-shot is expected to FAIL** (verdict amendment 3): dropping low-frequency RoPE components on a checkpoint trained with full RoPE changes the position representation the weights were trained for — the paper's own pairing law says this is the part that needs training. Plan 622 ships pp-RoPE KERNEL-ONLY; its quality claim lives in riir-train 449.
4. **Softmax-arm weak sparsity is empirical-only in the paper**; the sigmoid arm trades Θ(1)-negentropy for a provable √log t rate (openly exchanged).
5. **v1/v2 grounding discrepancy (recorded honestly):** the adversarial-panel advocates fetched arXiv v1 and flagged the Llama-2 7B CPT (App. I.5) as absent; it exists in **v2** (revision 2025-12-17), which is the version cited throughout this note. The recipe quoted in the panel brief (AdamW 2e-5, ~50M tokens) matches App. I.5 verbatim.
6. Gaussian premise is derivation-only; App. J suggests it holds on foreign checkpoints modulo sinks; our decade-probe diagnostics close the loop empirically on real Bonsai logits.

## GOAT gate (design; execution = Plan 622)

- **G1:** LUT vs closed form ≤ 1e-6; boundary exactness (a₀²=1, m₀=0); validity assert β ≥ α·log α; per-decade mass flatness on Gaussian logits to t=10⁶ (Δ ∈ {2,10}); **sigmoid transfer: Gauss–Hermite vs Φ(−1/a_t) within C·a_t⁻³ (values derived at test time, never pinned)**; no-double-temperature with SSMax; decade-mass diagnostics on real Bonsai logits (Gaussian-premise loop closed empirically).
- **G2 (`--release`, bench_preflight PROVENANCE, GPU-exclusive):** LUT build <1 ms @64k; pp2048/tg128 deltas within ±3%; Metal/CUDA parity green before any published number.
- **G3 split (amendment 2):** "calibrated zero-shot" (per-head frozen scale table) and "uncalibrated zero-shot" as separate rows; in-dist ppl tolerance pinned per τ ∈ {1,10,100} sweep (App. H protocol); 16× length ppl + retrieval probe on Bonsai-27B with zero weight edits; **pp-RoPE row recorded in advance as expected-FAIL zero-shot** (amendment 3).
- **G4:** preallocated LUT, read-only hot path, counting-allocator canary.
- **Promotion:** opt-in → default only on all gates pass AND modelless gain (it is — nothing touches a gradient). Lean trio lands WITH the flag.

## PASS-Redirects

N/A — Gain verdict (files: this note, Plan 622, riir-train Plan 449).
