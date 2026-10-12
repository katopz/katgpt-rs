# Research 619: MSAR — Multi-Stream Attention Residuals

> **Source:** [Multi-Stream Attention Residuals](https://arxiv.org/abs/2610.13630) — Qin, Zhang, Dai, Zou, Wu, Yang, Xia, Yao, Liu, Yan (SJTU / Xiaohongshu Dots Studio), arXiv:2610.13630 (2026-10-09). Project page: https://github.com/jianke0604/MSAR . **No model weights observed public at distill time** (repo is a project page as of 2026-10-12).
> **Date:** 2026-10-12
> **Update 2026-10-13:** Issue 932 CLOSED NEGATIVE at its T3 viability probe (`56ad37c4c`): the "`h_t = f(h_t) + h_{t-1}`" reading below is wrong for the shipped kernel — `evolve_belief`'s step reads no belief state and cumulative evidence pins ≥ 6/8 dims at −1; the additive kernel is exactly invertible. Single-stream belief stays; record in HISTORY.md (Issue 932), the saturation finding → Issue 934.
> **Status:** GOAT — per-track Gain ×2 + one novelty-TBD fusion idea. Track (c) training: riir-train **Plan 458** filed (MSAR = 4th arm in the existing `xhc_pretraining` harness — zero new infrastructure). Track (a) transformer substrate: katgpt-rs **Issue 933** (attn_res online-softmax merge + fixed-query probe — substrate hygiene, no production consumer today). Track (a/b) modelless fusion: belief-views time-depth transpose → **katgpt-rs Issue 932**, novelty TBD pending five cheap probes — NOT filed as plan, NOT claimed Super-GOAT (three adversarial failure modes gateable pre-build). NOT Super-GOAT overall: the multi-head depth-attention class is published (this paper + DAR + MoDA + the mHC family), and the workspace already ships the single-head parent (AttnRes, `katgpt-transformer::attn_res`) and the recurrent-stream sibling (xHC, `riir-train` Issue 383). **Claude verdict session `e2a09858` (claude_code reviewer): round 1 REVISE — 4 fixes (Issue 932 softmax→sigmoid per house rule + measured sigmoid-vs-softmax readout arm at T4; Plan 458 T5 re-cost 70–140 → 95–140 GPU-h with shown arithmetic incl. the round-2 +25–30% seq-4096 attention correction and the amendment re-priced 150–210 → 90–130 GPU-h; loss sign-convention unified LOWER-is-better; Issue 933 consumer honesty — `src/kimi_k3/` is the only consumer, test-arch-only) — all applied; round 2 AGREE (2 non-blocking pricing corrections also applied).**
> **Related Research:** 061 (Delta Attention Residuals — Plan 097 `delta_routing`, the source-delta cousin), 447 (Kimi-K3 KDA+AttnRes+LatentMoE — the shipped AttnRes consumer), 070 (GDN2 — the trained "retrieval over history" cousin, Plan 105 default-on), 330 (Kimi-K3 modeling divergences, `attn_res` provenance)
> **Related Plans:** 097 (`delta_routing` — speculative architecture support precedent, validated structurally before weights existed), 105 (`gdn2_attention`), riir-train 383 (xHC pretraining POC), riir-train 364 (iso-param verdict discipline — the harness that already killed one paper), riir-train 318 (4B/2B MLA-MoE build — the future pretrain where a residual-variant win cashes out), **riir-train Plan 458** (this paper → MSAR arm)
> **Classification:** Public

---

## TL;DR

MSAR extends Attention Residuals (AttnRes — softmax retrieval over network *depth* with one learned pseudo-query per sublayer) to **multi-head depth attention**. The design problem it solves: multi-head depth retrieval needs diverse heads, but (a) channel-splitting the hidden state breaks the retrieved value's integrity and (b) scalar reweighting of one stream dies under normalization (near-identical keys). MSAR's answer is the paper's transferable core: **derive head diversity from the OTHER axis** — m=4 full-dimensional "views" of each depth source via causal depthwise 1D convolutions over the *sequence* axis (kernels {4,8,12}; stream 0 = identity), each stream running its own depth softmax with its own pseudo-query, fused by a sigmoid gate `H_pre = σ(α·RMSNorm(flatten(Z))·W_pre + b)`. Block mode stores only block summaries and **expands streams on demand** (streams are deterministic functions of summaries — O(Nd) not O(mNd) pipeline payload). Results: 18B/28B MoE pre-training, beats Vanilla/mHC/AttnRes on all 12 benchmarks (avg 52.4 vs 49.4 AttnRes at 28B); scaling-law **iso-loss compute speedup 1.58×** vs Vanilla (AttnRes 1.35×, mHC 1.30×) at +0.3% activated params / +0.5% FLOPs. Ablations: removing convs (4 identical sources, 4 queries) degrades 1.847→1.862 — **source diversity beyond query diversity is the engine**; m=2 captures most of the gain (1.850 vs 1.847 at m=4); N≈8 blocks recovers most of full-depth access.

**Verdict: GOAT — Gain on two tracks, novelty-TBD fusion on the third.** This is a pre-training *architecture* paper: nothing retrofits into shipped checkpoints (Ternary-Bonsai-2-27B, Kimi-K3-0.40B, qwen3.8-27B all keep their trained residuals). The value splits three ways: **(c) training** — the stack already trains BOTH parent mechanisms (AttnRes with full backward gradients in `src/kimi_k3/backward.rs`; mHC-class xHC with forward + analytic backward + iso-param verdict rules in `riir-train` `xhc_pretraining`, Issue 383), so MSAR is a fourth arm in an existing harness, falsifiable at ~70–140 GPU-h on one 4090 (Plan 458); **(a) transformer substrate** — the `msar` forward module belongs in `katgpt-transformer` next to `attn_res.rs` for future checkpoints (Plan 097 precedent: ship architecture support ahead of weights), plus two weight-free decode improvements to the shipped AttnRes (Issue 933); **(a/b) modelless fusion** — the time-depth transpose onto the per-NPC belief chain is a real latent-space candidate with three named failure modes, filed as probes (Issue 932), explicitly NOT claimed until the probes clear.

---

## 1. Paper core

### 1.1 The residual-design axes (paper's Table 7 taxonomy)

| Family | Mechanism | Weights | Sources | Ships here? |
|---|---|---|---|---|
| Single-state recurrence | Vanilla / ReZero / Highway | fixed / static / dynamic | `h_{l-1}` | every shipped model |
| Multi-state recurrence | HC / mHC / xHC / DDL | dynamic | persistent m-stream state | **xHC** (`riir-train xhc_pretraining`, Issue 383) |
| Cross-layer access | DenseNet / DenseFormer / **Block AttnRes** / MoDA / MUDDFormer | static / static / dynamic / dynamic | source list `[b_0..b_n]` | **AttnRes** (`katgpt-transformer::attn_res`, Kimi-K3 native) |
| Multi-stream cross-layer | DAR (2 streams) / **Block MSAR** (m streams) | dynamic | conv-expanded source views | — (this paper → Plan 458 / Issue 933) |

### 1.2 Mechanism (Block MSAR, decode-relevant form)

1. **Sources:** partition L residual sublayers into N≈8 blocks; `b_n = Σ_{j∈B_n} f_j(h_j)` accumulated **single-stream**; embedding is source 0; current block contributes its partial `b^<l_n`.
2. **View expansion (the novel half):** `v^(0) = b`, `v^(r) = E_r[b]` for r=1..m−1, where `E_r` is a causal depthwise conv (kernels {4,8,12}) over the *token* axis, conv weights **zero-init**. Streams are deterministic functions of the stored summary → store 1, expand at consumption, fuse expansion with the depth-attention kernel (never materialize).
3. **Per-stream depth attention:** m pseudo-queries `w^(r)_l` (zero-init), keys = expanded sources, `α^(r) ∝ exp(w^(r)·RMSNorm(v^(r)))`, softmax normalized **within each stream**; values = expanded sources.
4. **Sigmoid-gated fusion:** `h_l = Σ_r g^(r)_l · z^(r)`, `g^(r) = [σ(α_pre·RMSNorm(flatten(Z))·W_pre + b_pre)]_r`, α_pre init 0.01. (Sigmoid, not softmax — the house rule independently.)
5. **Two-phase compute:** phase 1 batches all target-layer queries against completed sources per stream; phase 2 online-softmax-merges the current partial. Last sublayer merges without expansion.

### 1.3 Numbers worth keeping (10B MoE Pile loss ablations)

Vanilla 1.898 · mHC 1.871 · AttnRes 1.867 · AttnRes+mHC 1.861 · **MSAR 1.847** · w/o conv 1.862 · w/o H_pre (mean) 1.851 · m=2: 1.850 · m=8: 1.846 · blocks S=2/4/8: 1.845/1.847/1.852. The AttnRes+mHC row matters: naively *summing* both mechanisms gains only 0.006 over AttnRes, while MSAR gains 0.020 with the same m and S — the advantage is the specific view-retrieval composition, not "add more streams."

---

## 2. Path 0 decomposition (per component)

| Component | Paper role | Modelless analog in this workspace | Signal-diff vs closest shipped cousin | Status |
|---|---|---|---|---|
| Multi-head depth attention (m queries, m sources) | the payload | **not covered**: `apply_attn_res` is single-head (one `score_weight = norm⊙proj`) | AttnRes ships; the multi-stream generalization does not | **new substrate** → Plan 458 T1 (`msar` module in `katgpt-transformer`) |
| Causal-conv view expansion (diversity from the other axis) | source diversity | **not covered** for depth retrieval; xHC uses conv {4} for *stream recurrence*, not source views | xHC = persistent m-stream state mixed by `H_res` recurrence; MSAR = single-stream summaries retrieved by AttnRes-style softmax with conv-expanded keys/values. Confirmed diff (the paper's own AttnRes+mHC ablation shows the naive sum underperforms) | **new substrate** → Plan 458 T1 |
| Expand-on-demand derived views | storage/comm efficiency | **pattern ships** in derived indexes; sync-boundary 5-scalars law corroborates | No-GD advocate: adopt as index/log policy, **DECLINE for the 8-dim belief** — at d=8 a stored multi-scale pyramid is *cheaper* per tick than recompute; expand-on-demand is a storage/wire law, not a hot-path law at bounded d | recorded (design law) |
| Two-phase compute + online-softmax merge | training throughput | **partially covered**: `apply_attn_res` re-scores all entries per call (O(entries) per sublayer — fine at N≈8, wasteful at scale) | online merge keeps running (max,sum,acc); a newly completed block merges O(d). Weight-free, parity tolerance-gated (reduction order differs — ≤1e-6 drift, laya-G5 style) | **extracted** → Issue 933 |
| Cross-stage view caching | prefill/decode memory traffic | not covered: key-views recomputed per call | cache per-summary views, invalidate on write — deterministic | **extracted** → Issue 933 |
| Learned depth-pattern diagnostics (Appendix D) | interpretability | not covered | retrieval-age histogram = machine-readable affective persistence (grudge/grief signal); **1–2 scalars max across the sync boundary** per house law | recorded (instrumentation lane, Issue 932 T5) |
| Derived kernel schedule | implicit ({4,8,12} are training-context constants) | **extractable as a rule**: κ_r from measured autocorrelation half-life τ (κ_r ≈ τ·{0.5, 2, 8}), S from storage budget | literal numbers do NOT transfer (see §3 B2) | **extracted** → Issue 932 T2 |
| 1.58× iso-loss speedup | training result | **not extractable** — GD artifact | learned per-layer retrieval specialization is not reconstructible weight-free | training track → Plan 458 |
| Fixed-query modelless AttnRes (conv-removal ablation suggests most value survives query degradation) | — | candidate: deterministic depth-profile query replacing learned `score_weight` for checkpoints *without* AttnRes weights | weakest extraction (advocate-named): layers may need task-specific lookback fixed queries can't express; E5 probe decides | probe → Issue 933 T3 |

## 3. Modelless fusion: the time-depth transpose (belief views) — novelty TBD

**Claim form (pinned before probes, per §1.5 precondition):** *weight-free multi-view temporal retrieval over a bounded per-agent latent state chain for think-brain belief evolution, consuming the belief trajectory itself, distinguished from `ObservationGate` confidence decay (content-INDEPENDENT — `sigmoid(-λ·Δt)`, only time delta) by content-dependent retrieval, and from GDN2's delta-rule (trained W_q/W_k/W_v projections inside model forward) by zero learned parameters and a state-chain (not token-stream) consumer.*

**The transposition:** `evolve_belief` (katgpt-sense `ReconstructionState`, belief `[f32; 8]`) IS a Vanilla residual chain over ticks — `h_t = f(h_t) + h_{t-1}`-shaped single-stream recurrence. MSAR's recipe transposes: block summaries of the belief trajectory every K ticks (K=8 → 256 B/NPC), m views via causal temporal convs, content query = current belief (or its `temporal_deriv` fast component / drive projection — query diversity without GD), **per-entry sigmoid gates over normalized similarity** (house rule — the paper's softmax belongs to Plan 458's faithful reproduction; sigmoid lets the gate ABSTAIN — all-gates-low = "history carries nothing relevant" — instead of being forced into a convex combination over past summaries, which is exactly the EMA-with-extra-steps failure shape; the sigmoid-vs-softmax readout choice is MEASURED at Issue 932 T4, not assumed), gated fusion back into the update. All weight-free, zero-alloc, ~30 flops/NPC/tick at m=4/K=8/d=8, latent-local (only scalars cross sync). Latent-to-latent, sigmoid-gated, per-NPC — boundary-compliant by construction.

**Why NOT claimed (the three adversarial failure modes — each gateable pre-build, none refuted yet):**

- **B1 Recency degeneracy (sharpest):** `dot(current_belief, past_view)` over a smoothly-evolving 8-dim signal is approximately monotone in age → any normalized-combination readout rebuilds an exponential-decay prior → "EMA with extra steps." Sigmoid gates soften this (abstention instead of a forced convex combination) but do not remove it — the Issue 932 T4 probe measures both readouts.
- **B2 Kernel-scale mismatch:** {4,8,12} ticks at 20 Hz = 0.2–0.6 s vs belief time constants of seconds-to-minutes → three box filters far inside the correlation window produce near-identical views — the paper's own scalar-reweighting failure wearing a conv costume (identical DC component). Fix: derived schedule κ_r ≈ τ·{0.5,2,8} from per-channel autocorrelation half-life + detrended (DC-removed) similarity.
- **B3 The destruction disanalogy (decides viability):** depth-residual retrieval recovers information *destroyed* by intervening non-invertible layers; over time `b_{t-1}` is an *input* to `b_t` — nothing is destroyed by the chain, only by the lossiness of `evolve_belief` (clamps, sigmoid saturation, event bumps). Headroom ∝ that lossiness. Probe: variance of `b_{t-1} | b_t` over logged trajectories. If the update is ~self-restoring on its operating manifold, the honest verdict is "single-stream stays" — a cascade-negative-shaped result, recorded and not retried blind.

**Probe order (Issue 932):** E2 (τ per channel → kernel schedule) → E6 (detrended vs raw similarity) → E1 (lossiness) → E3 (argmax-age entropy on frozen corpus) → E4 (paired LB95 G1 vs single-stream `evolve_belief` on a downstream label). E1/E2/E6 are scripts over existing logs — no code lands until at least E1+E2 clear. Composition law: one multi-scale view constructor feeding both smoothing and retrieval keys — composes with `temporal_deriv`/`SyncBank`, never duplicates them (substrate-first); operator-named (`belief_views`), paper vocabulary in docs only.

**Coverage diffs run (§3.6, each cousin's core signal read):** ObservationGate = content-independent Δt decay → gap stands. `temporal_deriv` = fast−slow *derivative* channel ("how fast"), not retrieval over history ("what was") → gap stands. GDN2 = trained projections, token-stream consumer → different constraint regime → gap stands pending weight-free viability. Episodic sighting / RavenSlot / δ-Mem = event-level memory + consolidation, not self-state-chain retrieval → gap stands. SyncBank<8,3> = multi-channel sync aggregation, not history retrieval → gap stands.

## 4. Prior art (§4 searches — all run 2026-10-12)

1. Headline: "multi-head depth attention residual cross-layer retrieval 2026" → published class confirmed: AttnRes (arXiv:2603.15031, Kimi Team, ~67 citations, validated at trillion-param scale in Kimi K3), mHC (arXiv:2512.24880, DeepSeek — in DeepSeek-V4-Pro), HC (arXiv:2409.19606), DAR (arXiv:2607.18730 — 2-stream cross-attention variant), MoDA (arXiv:2603.15619), xHC (arXiv:2607.14530), DDL, Gated Residual (qwen3.8-next). **The class is published and crowded — no novelty claim available or attempted on the mechanism itself.**
2. Weight-release check: "MSAR github weights jianke0604" → project page only; no checkpoint observed.
3. One unverifiable lead disclosed: a cached search snippet (cctest.ai, dated 2026-07-31) describing "MHAR — multi-head depth reads without adding parameters"; the article 404s at fetch time. Recorded as an unverifiable concurrent-work lead; does not affect our verdicts (we claim implementation + fusion, not class novelty).
4. Rust implementations: an independent Rust AttnRes implementation exists (AbdelStark/attnres) — prior art for "Rust AttnRes," not for anything we file; our shipped `katgpt-transformer::attn_res` predates our awareness of it and is checkpoint-driven (Kimi-K3 weights), not paper-driven.
5. Selling-point framing: "weight-free multi-scale temporal state retrieval for agents" → published agent-memory work (Mem0/Zep/surveys 2026) is episodic/fact retrieval into context, not retrieval over an agent's own latent state chain fused into state updates → fusion claim survives Q1 as *novelty TBD*, killed or confirmed by the probes.

## 5. Fusion — what paper × A × B produces that none alone can

1. **MSAR × xHC harness (Plan 458):** the workspace's residual-arm ladder (Vanilla → xHC/mHC-class → AttnRes → MSAR) under Plan 364's iso-param verdict discipline — the paper's comparison table becomes an arm we can run, including the free falsification tension (xhc_temporal Phase 6' measured conv *parity* on non-temporal tasks where the paper claims conv is load-bearing: 0.015 of the 0.020-nat edge).
2. **MSAR × belief kernel (Issue 932):** the view-diversity law applied to time instead of depth — IF the lossiness probe shows headroom. Either outcome is recorded; the negative is as valuable as the positive (it pins "single-stream belief stays" with evidence).
3. **MSAR × attn_res decode (Issue 933):** online-softmax merge + view caching make long-context AttnRes decode (Kimi-K3-class) cheaper, weight-free, today — independent of whether MSAR weights ever ship.
4. **The design law (workspace-wide):** "retrieval diversity for axis A must live on axis B; channel-splitting breaks value integrity; same-axis scalar reweighting dies under normalization." Applies to riir-rag span views (line/function/file, never channel-split embeddings) and decision-serving example-axis views. Corollary pinned as a negative control: any second retrieval head whose diversity axis is "same axis, reweighted" is the paper's killed ablation.

## 6. Routing (files this verdict creates)

| File | Repo | Track |
|---|---|---|
| `.research/619_MSAR_Multi_Stream_Attention_Residuals.md` (this note) | katgpt-rs (public) | record |
| `.plans/458_msar_residual_arm.md` | riir-train (private) | (c) training — staged harness arm + 150M falsification |
| `.issues/932_belief_views_multiscale_temporal_retrieval_probes.md` | katgpt-rs (public) | (a/b) fusion probes — novelty TBD |
| `.issues/933_attn_res_online_merge_modelless_queries.md` | katgpt-rs (public) | (a) decode substrate — **substrate hygiene, LOWEST priority: the only `apply_attn_res` consumer is `src/kimi_k3/` (test-arch-only per owner); no production consumer today (grepped 2026-10-12)** |

Cross-repo boundary check: the `msar` forward primitive is generic transformer math with no game/chain/shard IP → public katgpt-rs is the correct tier (the `attn_res.rs` precedent); the training harness arm and any ternary recipe are "how" → private riir-train. GOAT-audit skill not triggered pre-plan: no existing katgpt-rs MSAR primitive is consumed (this plan ADDS one); the plan consumes riir-train-internal substrate (xHC harness) — no cross-repo stall/DRY surface to audit.

## 7. Weakest point (named by the coordinator, pre-review)

The training track's value is contingent on a consumer that does not exist yet: Plan 318 (4B/2B MLA-MoE pretrain) is a plan, not a scheduled run — Stage 0 (micro harness, 0 GPU) is cheap and kill-gated (the GRT-micro FAILED precedent proves the gate fires), but Stage 1 spends 70–140 GPU-h to validate a mechanism we may never deploy, at a scale (150M dense) two regimes below where the paper measured (18B/28B MoE). The honest framing recorded in Plan 458's Status: this is a cheap falsification of a strong published claim on an existing harness, not a step toward a scheduled deployment; the adoption decision is owner-gated at Stage 2. Second weakest: the belief-views fusion may die at probe E1 (lossiness) — filed as probes precisely so that death costs one script, not a module.
