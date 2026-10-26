# Research 635: C2C — Cache-to-Cache Direct Semantic Communication (PASS — founding member of an already-routed class)

**Status:** Done — closed (PASS, 2026-10-26)

> **Source:** [Cache-to-Cache: Direct Semantic Communication Between Large Language Models](https://arxiv.org/abs/2510.03215) — Fu, Min, Zhang, Yan, Dai, Ouyang, Wang (Tsinghua + Infinigence + CUHK + SJTU), arXiv:2510.03215, published ICLR 2026. Code: thu-nics/C2C.
> **Related Research:** 247 (Dense Latent Heterogeneous Communication — **the same class's stronger successor, already Super-GOAT**), 535 (XBridge heterogeneous bridge + prior-art landscape), 289 (RecursiveMAS — the PASS precedent for this family), 251 (Token Economics — LatentMAS/Q-KVComm/TokenDance coverage), 101 (CachePrune — same-model reuse, a different problem), 540 (Vec2Vec NoGD unpaired translation)
> **Related Plans/Issues:** riir-ai Plan 311 (NPC mind-reading runtime), katgpt-rs Plan 280 (`cs_kv_probe`), riir-train Issue 523 T4 (cross-model bridge — defer-with-trigger; **edited this session to add this paper's ID**), riir-train Bench 422 (`riir-train/.benchmarks/422_canon_p2_cross_model_procrustes_validation.md` — cross-arch Procrustes NO-GO)
> **Verdict: PASS.** Founding member (Oct 2025) of the cross-model cache-communication class. The corpus already distilled its stronger successor (arXiv:2606.13594 via Research 247, which cites "prior C2C baselines") and holds the standing defer-with-trigger posture for trained cross-model bridges (riir-train Issue 523 T4, trigger R3 not fired). Every C2C-specific delta is implementation/architecture depth for that deferred lane; none flips the trigger or opens a coverage gap. Classification note only.

## 1. What the paper is (brief — the class is known)

C2C replaces text-to-text (T2T) hand-off between LLMs with direct KV-cache fusion: a per-pair fuser network projects the Sharer's KV-cache into the Receiver's space and adds it residually, layer-aligned terminal-first (final layers paired first), token-aligned by decode→re-encode with maximal-coverage selection for one-to-many tokenizer cases. Components: projection + feature-fusion layers, an input-aware dynamic-weighting head, and a per-layer learnable gate (Gumbel-sigmoid, temperature annealed 1.0→0.001, binary at inference). Both LLMs frozen; only the fuser trains (next-token loss on the Receiver, SFT-style) — ~9 GPU-hours to near-final accuracy (300 steps), 500k OpenHermes-2.5 samples.

Numbers: +6.4–14.2% over individual models, +3.1–5.4% over T2T, ~2.5× faster (parallel cache fusion ~90 ms replaces ~80-token Sharer decode ~1312 ms). Works across families (Qwen/Llama/Gemma) and sizes (0.6B–14B); effective rank rises post-fusion (K 388→395, V 532→560). Oracle experiments establish: (1) cache enrichment at fixed length helps (few-shot prologue then drop the exemplar cache slice: 62.34 vs 58.42 direct), (2) caches are convertible across models (MLP projection + t-SNE containment).

## 2. The deltas vs what we already hold (Path 0 inventory)

| C2C component | Track | Already covered by | Delta verdict |
|---|---|---|---|
| Trained fuser (freeze-both SFT, ~9 GPU-h, Gumbel-sigmoid gate anneal) | (c) model-based | riir-train Issue 523 T4 defer-with-trigger; Research 247 routed the stronger 2606.13594 adapter | recipe rider — prior-art list gains the ID; no new plan while the trigger is unfired |
| Token alignment across heterogeneous tokenizers (max-coverage re-encode) | (a) modelless impl spec | 247's pairs (2606.13594) are same-tokenizer; 535's design law already demands learned bridges + discrete anchors | implementation spec for the deferred lane; recorded |
| Terminal layer alignment (final-layers-first) | (a) | 247 records the monotonic depth map | minor alternative, recorded |
| **Cache enrichment oracle** (prefill E⊕X, decode on the question-aligned slice — few-shot gains at zero extra cache length) | (a) modelless | **new-to-corpus** (not in 247/535) — but no consumer: reflex is modelless classification, laya is an encoder, the league's axis is latency not accuracy; no generative few-shot serving path exists in any hot path | recorded; adoption rides the deferred lane |
| Multi-sharers fuse WITHOUT additional training (Table 13: 60.71 + 46.13 → 64.60) | (a/c) | new datum; sharpens Issue 523 T4's consumer test someday | recorded |
| Hub-and-spoke KV-Latent O(N) unification (M projectors + N fusers to a shared latent; Table 14 works cross-Receivers) | (c) | architecture addendum for Plan 311's many-to-many NPC-mind-reading case | recorded |
| Identical self-fusion beats SFT (C2C same-model-as-both 42.17 vs Single fine-tune 36.81 MMLU) | (a) | evidence datum against same-budget SFT; no consumer | recorded |
| Effective-rank diagnostic as fusion-quality readout | (a) | generic measurement; candidate G1-style quality gate for any future latent-fusion work | recorded |
| Gate-regime law (general-purpose training → broad gates + fine dynamic weights; task-specific → sparse gates + strong weights) | (b) | informational calibration insight for per-layer gates we DO ship (SP-KV `soft_gate_bias`, EGA, escalation gates) | recorded |

## 3. Signal-diff — why the one tempting modelless angle is already dead

The natural "modelless C2C" — a deterministic cross-model cache projector (Procrustes/whitening, no GD) — is **measured NO-GO in this workspace**: riir-train Bench 422 cross-arch Procrustes mean cos ≤ −0.08 (`riir-train/.benchmarks/422_canon_p2_cross_model_procrustes_validation.md`, recorded in riir-train Issue 523), independently confirmed by XBridge's own bridge-only ≈30% F1. C2C itself never claims deterministic cross-arch projection: every heterogeneous pair runs through the trained fuser. Its modelless parts (alignment machinery, enrichment oracle, diagnostics) have no consumer today (§2). No coverage gap opened; no new primitive clears the novelty gate (the class is published, and our stronger-member distillation + fusion are filed).

## 4. Reverse-grep before PASS (mandatory)

Documented gaps this paper could fill: riir-ai Plan 298 T3.3 `share_trajectory` stub (deferred — C2C's multi-sharer/hub results inform its design but unblock nothing; the blocker is a live consumer, unchanged); riir-train Issue 523 T4 (defer-with-trigger — R3 not fired; C2C adds prior art, not a consumer). No shipped config default is contradicted (the stack ships no cross-model cache transport; katgpt-kv `rope.rs` + Plan 280's probe are same-model/importance surfaces). Reverse-grep verdict: no Gain trigger fires.

## 5. Routing (edits made this session)

- **This note (635)** — the classification record; greps on `arXiv:2510.03215`, `Cache-to-Cache`, `C2C` land here.
- **`katgpt-rs/.research/247`** — PASS-Redirects line added at the tail (ID + full title + one-line reason).
- **`riir-train/.issues/523` T4** — the prior-art-to-cite list gains arXiv:2510.03215 (the founding member; its freeze-both recipe + Gumbel-sigmoid gate anneal are the recipe riders if T4's trigger ever fires).

## 6. PASS-Redirects (synthesis)

> **PASS-Redirects (synthesis):** Fu, Min, Zhang, Yan, Dai, Ouyang, Wang [arXiv:2510.03215 "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"] — founding member of the cross-model cache-communication class; the corpus holds its stronger successor (arXiv:2606.13594 / Research 247, Super-GOAT fusion filed with Plans 280/311 + guide riir-ai/.research/133) and the defer-with-trigger posture (riir-train Issue 523 T4); deterministic cross-arch projection is already refuted (riir-train Bench 422); the modelless deltas (enrichment oracle, token-alignment spec, hub topology, multi-sharer composition) have no stack consumer → classification note only.
