# Research 548: TriSpec — Margin-Gated Verification Escalation

> **Source:** "TriSpec: Ternary Speculative Decoding via Lightweight Proxy Verification" — Jiang et al. (Qwen Team, SJTU, HKBU), [arXiv:2601.23180](https://arxiv.org/abs/2601.23180), Feb 2026.
> **Date:** 2026-09-11
> **Status:** Active (issue 745 open)
> **Related Research:** 162 (TRAS trust-region acceptance), 316 (DSpark confidence-scheduled speculation), 068 (RAEv2 intra-model verifier signal), 214 (spectral draft/verifier consistency), 002 (Leviathan/DDTree foundation)
> **Related Plans:** 182 (TRAS), 203 (kurtosis speculation gate), 250 (breakeven routing)
> **Classification:** Public

---

## TL;DR

TriSpec attacks the *verification* axis of speculative decoding (drafting cost `td` and acceptance length `τ` are saturated): a same-family smaller model (Qwen3-1.7B for a 32B target, 82% token-exact alignment vs the EAGLE3 drafter's 68%) pre-verifies drafts in one parallel pass, and a **zero-training margin predicate** `g(p) = 1[top1 − top2 ≥ λ]` (λ=0.5) on the proxy's own distribution decides trust. Trusted → the proxy's verdict stands (target invocation skipped); low-margin → escalate the remaining draft suffix to the target with token pruning. Up to +30–35% speedup over EAGLE3 SD, target invocations halved, ≤1% accuracy degradation.

**Distilled for katgpt-rs (modelless, inference-time):**
The transferable primitive is NOT "use a small model as verifier" — it is the **margin-gated escalation rule**: a rank-gap statistic computed in O(1) from logits already materialized at the accept-decision point, gating whether a cheap evaluator's verdict stands or an expensive one is invoked. The training half (drafter/adapter fitting) is separable and deferred (see §4).

---

## 1. Paper Core Findings

- Latency decomposition `L ∝ (td + tv)/τ`: with single-layer drafters (EAGLE/Medusa) and MTP pushing td→0 and τ→saturation, **per-round verification time `tv` is the dominant remaining axis**.
- Proxy choice: same-family smaller models show strong token-level alignment (82% exact match) AND separability — top1−top2 margin stratifies acceptable vs unacceptable predictions cleanly; the single-layer drafter's confidence is NOT reliable (overconfident, 68% alignment).
- Routing: `τa` (proxy acceptance length) vs `τm` (margin-trusted prefix). If `τa < τm` → accept locally, substitute the proxy's token at the rejection point (Case I). Else → trust the proxy's prefix, prune the draft tree, escalate only the suffix to the target (Case II).
- Lossy lane (≤1% avg degradation) — NOT provably lossless; contrast with the Leviathan residual-accept substrate we ship (`katgpt-core/src/speculative/`).
- Training cost: joint drafter+adapter 115h vs **adapter-only finetune 44h on 8×A100** (0.42B trainable params) — the drafter learns to consume either proxy or target features.

## 2. Distillation — modelless core

**Pinned claim (§1.5 precondition):** margin-gated verification escalation for katgpt-rs's draft-verify + cascade surfaces, consuming per-position logits already materialized at the accept-decision site, distinguished from the published class (TriSpec/MARS/VIA-SD) by (a) substrate fusion — the margin operator as `IndicatorCascade`'s stage-1 firing predicate with d2f residual escalation on distrust, (b) bandit-tuned λ (the paper hand-pins λ=0.5; our UCB1 meta-router already tunes decode-strategy arms on `acceptance_rate × latency_improvement` reward), (c) a healer-consumer mapping (draft→clippy_verify cascade).

**The margin operator is ABSENT workspace-wide** (subagent grep, 5 trees): `SamplerFeatures::from_logits` extracts `top1_share` but no `(top1, top2, gap)` read exists anywhere; the sampler retains no top-2. The primitive is a zero-alloc partial-argmax pass over logits — two running maxima, no sort, no allocation.

**Fusion (paper × shipped substrate):**

| Paper piece | Shipped substrate | Fusion |
|---|---|---|
| Margin predicate `top1−top2 ≥ λ` | `SamplerFeatures::from_logits` (`katgpt-forward/src/d2f/mod.rs:1008`) | add `top2`/`rank_gap` fields; the predicate is then a threshold read |
| Cheap-evaluator-gates-expensive cascade | `IndicatorCascade` (`katgpt-core/src/pruners/indicator_cascade.rs`, Bench 320, ~15× FPR reduction) | margin becomes a stage-1 firing condition alongside the probe bank — a *graded* trigger vs today's OR-fused boolean probes |
| Escalation re-decide + substitute | d2f accept loop + `sample_residual_distribution_into` (Leviathan Eq.3 semantics already ship) | on distrust, the expensive evaluator's residual resampling replaces the token — wiring only, no new machinery |
| Fixed λ=0.5 | UCB1 meta-router, reward `acceptance_rate × latency_improvement` (`katgpt-attn/src/dash_attn/meta_router.rs:162`; `riir-router/src/meta_router/`) | λ as a bandit arm — tuned online per workload instead of hand-pinned; novel-vs-paper, novelty-vs-MARS unverified |
| — | kurtosis speculation gate (Plan 203), TRAS trust-region (R162), DSpark confidence schedule (R316) | margin is the O(1) rank-gap alternative: kurtosis needs the shape; TRAS adapts the window; margin needs two maxima |

**Healer consumer (investment priority #2):** riir-clippy's pipeline is exactly cheap-draft → expensive-verify (`clippy_verify` re-runs cargo clippy, seconds per candidate — hence opt-in). A margin-gated mid-tier (statistical pre-verifier trusted when its margin is high; full cargo re-run only on distrust) would raise Certified-row throughput per wall-clock. NOT filed there this session — riir-clippy is sibling-claimed (Issue 090 lane); file as riir-clippy issue when the claim clears.

## 3. Verdict

**Gain (fusion idea, novelty TBD) — issue 745 filed; no plan, no guide.**

Novelty gate, honestly scored:
- **Q1 no prior art: NO.** The class is published: TriSpec itself, MARS (Margin-Aware Speculative Verification, training-free), VIA-SD (arXiv:2606.12243, intra-model verification routing), Training-Free Per-Step Lossy SD (arXiv:2609.02897, margin-promotion rule), Bi-directional Model Cascading with Proxy Confidence (arXiv:2504.19391). Internal coverage is zero (greps: all narrow patterns + 5 arXiv IDs across 12 note dirs = 0 hits), so this is first *internal* distillation, not novelty.
- **Q2 new behavior class: NO** — trust-gated slow-path routing ships (TRAS R162; IndicatorCascade).
- **Q3 selling point: weak standalone** — "our SD verifies with a margin gate" is table stakes vs MARS.
- **Q4 force multiplier: YES** — sampler + pruner-cascade + meta-router + healer. Insufficient alone.

MOAT gate (katgpt-rs): in-scope (spec-decode slot); would consume `SamplerFeatures` + `IndicatorCascade` + d2f — all katgpt-rs surfaces; no game/chain leakage. The bandit-λ arm and the healer mapping are the two deltas worth proving if the PoC runs; until then the issue carries the sketch.

**Adjacent, not distilled:** Saguaro / "Speculative Speculative Decoding" (Kumar, Dao, May — [arXiv:2603.03251](https://arxiv.org/abs/2603.03251), ICLR 2026): overlaps drafting with verification by pre-drafting predicted verification outcomes; 30% over optimized SD. Engineering pipeline pattern for a serving engine; no modelless primitive extraction beyond TriSpec's here. Recorded to prevent re-fetch.

## 4. Training-track defer (auditable, per §3.5/TTPO)

TriSpec's training half needs a Plan in **riir-train** — deliberately NOT filed this session: riir-train is sibling-claimed (routing restriction: no edits there from this session). Record + unblock condition:

- **What requires GD:** the drafter-side MLP adapter mapping proxy features into the drafter's feature space (adapter-only regime: 0.42B params, 44h on 8×A100). No deterministic construction substitutes for feature-space fitting — Path 0 decomposition: the *routing* math (margin predicate, τa/τm comparison, residual substitution) is fully closed-form and is the modelless track above; the *adapter* is the only GD-bearing component.
- **Second, larger training item:** the recipe's load-bearing premise is a **same-family small proxy**. Bonsai-27B has no small sibling (Kimi-K3 0.4B is cross-family + test-arch-only — alignment premise fails). A 0.5–2B ternary Bonsai sibling would unlock both a proxy-verify lane for the perf league AND the adapter recipe. That is a riir-train-scale training decision (owner-gated GPU budget).
- **Unblock condition:** riir-train sibling claim clears + owner go on GPU arms → file `riir-train/.plans/NNN_trispec_proxy_recipe.md` citing this note; fold into/alongside Plan 394's A-arm queue.

## 5. League note (investment priority #3)

A spec-decode/proxy-verify lane changes the perf-league comparison axis (throughput becomes acceptance-length-dependent) — it is an ENGINE feature lane, not a kernel win, and must not contaminate the kernel-vs-kernel tg128/pp2048 claims. If the Bonsai sibling proxy ever exists, the lane runs as a separately reported matrix row under the fairness manifest, opt-in gated.
