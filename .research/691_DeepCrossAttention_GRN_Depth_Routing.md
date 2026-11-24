# Research 691: DeepCrossAttention (DCA) — GRN Depth Routing, Identity-at-Init Retrofit, Rank/Dilution Census

> **Sources:**
> - [DeepCrossAttention: Supercharging Transformer Residual Connections](https://arxiv.org/abs/2502.06785) (Heddes, Javanmard, Axiotis, Fu, Bateni, Mirrokni — Google Research + UC Irvine + USC, ICML 2025; v2 2025-07-23; cited-by ~16 at distill time)
> - [Google Research blog: Supercharging transformer residual connections](https://research.google/blog/supercharging-transformer-residual-connections/) — names the family: Attention residuals (Moonshot, Kimi), Hyper-connections, manifold-constrained hyper-connections; reports Inkling 2 (Thinking Machines) adopted DCA's Q/K/V split + AttnRes
> - [MUDDFormer](https://arxiv.org/abs/2502.12170) (Xiao et al., ICML 2025, 49 cites) — publishes the multiway/QKV-split axis independently; cited by the Google blog's own ecosystem
> **Date:** 2026-11-24
> **Status:** Done — **Gain, per-track**: (c) training → riir-train [Plan 514](../riir-train/.plans/514_dca_grn_residual_arm.md) filed (staged, kill-gated); (a) transformer substrate **PASS-with-redirects** (no forward module until an arm wins); (a/b) modelless folded into Plan 514 Stage 0. **No mechanism-class novelty claimed** — the dynamic-depth-routing family is published and crowded (DenseFormer / HC / mHC / xHC / AttnRes / DAR / MSAR / MUDDFormer) and the workspace already ships the parent (`katgpt-transformer::attn_res`). **Claude verdict session `d2e600b8` (claude_code reviewer): round 1 AGREE with 4 file conditions (three-column Stage-1 reporting law; sigmoid arm merged with Plan 458 T6 as a 3-arm table; LB95-includes-0 → STOP noise gate; Status-line pretrain dependency) — all folded; round 2 final AGREE with 1 correction (Stage 2 + serving retrofit are recorded-Claude-verdict gates, never "owner-gated" — only credential touches are owner gates) — applied.**
> **Related Research:** 447 (Kimi K3 KDA+AttnRes+LatentMoE — the shipped AttnRes parent, `katgpt-transformer::attn_res`), 619 (MSAR — the family taxonomy + the published-class census; its Table 7 already classifies DenseFormer/Block-AttnRes/MUDDFormer on the cross-layer-access axis), 626 (InfiLoop — loop-native attention residuals; cites DenseFormer/HC as static cross-depth mixing), 661 (SimpleHC — the full family landscape: HC/mHC/xHC/AttnRes/oHC/kroHC/go-mHC/spectral-sphere/Frac-Connections/MGR/MuddFormer/DenseFormer), 095 (MGR multi-gate residuals), 417 (Cross-Stage Residual Relocation), 286 (depth invariance)
> **Related Plans:** riir-train [458](../riir-train/.plans/458_msar_residual_arm.md) (MSAR residual arm — the sibling arm family; its T6 sigmoid readout merges with Plan 514's gate-shape table), riir-train 364 (iso-param verdict discipline — the GRT precedent behind the three-column law), riir-train 318 (4B/2B MLA-MoE pretrain — the unscheduled Stage-1/2 consumer), katgpt-rs 097 (support-ahead-of-weights precedent), riir-train 383 (xHC harness — Issue)
> **Classification:** Public
>
> **PASS-Redirects (synthesis):** Pagliardini et al. [arXiv:2402.02622 "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging"] — GRN-v1 is DenseFormer; static cross-depth mixing is classified in Research 619's taxonomy and needs no separate distill. Zhu et al. [arXiv:2409.19606 "Hyper-Connections"] — the fixed-size-stack family, workspace-covered (Research 661 census + riir-train Issue 383 xHC harness). Xiao et al. [arXiv:2502.12170 "MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections"] — publishes the QKV-split axis DCA is known for; kills mechanism novelty on that axis. Menghani et al. [arXiv:2410.04743 "LAuReL: Learned Augmented Residual Layer"] — low-rank aggregation cousin, DCA's Table 5/6 baseline. Kimi Team [arXiv:2603.15031 "Multi-Head Attention Residuals"] — the shipped parent (Research 447/619); DCA's unnormalized additive gate is the one cell it does not cover.

---

## TL;DR

DCA replaces the plain residual sum with a weighted combination over the stack of prior layer outputs: GRN-v1 static scalars (= DenseFormer), GRN-v2 per-dimension Hadamard, GRN-v3 adds an input-dependent unnormalized gate `w̄_t = 1·ReLU(w_t^T G_t)`; at init (b=1, w=0) the model is **byte-identical to a standard residual net**, which is what makes retrofitting possible (identical function at init, then 60k continue-train steps → +0.19 PPL vs +0.02 for continued standard training). DCA proper = three GRN-v3 instances composing Q/K/V from the depth stack (attention itself unchanged). Results: same quality up to 3× faster; 30-layer DCA beats 42-layer transformer on LM1B; no loss spikes; gains shrink with width — exactly what the theory predicts (threshold on collective rank r*/d).

**For this workspace:** the family is saturated — the parent ships (`attn_res.rs`), the taxonomy exists (Research 619), the harness exists (riir-train xhc + kimi_k3 lanes). The three deltas worth filing: **(1)** the gate-SHAPE cell (unnormalized additive ReLU vs softmax-normalized) is uncovered on both sides — the paper's ablation has no softmax-depth-mixing baseline and our lane never ran the unnormalized variant → a 3-arm table (softmax / ReLU-additive / sigmoid-additive) shared with Plan 458 T6; **(2)** identity-at-init as a step-0 bit-identity gate + the retrofit recipe — the one paper result aimed at converged checkpoints; **(3)** the closed-form rank/κ/dilution census (CCA/Bartlett/Ledoit-Wolf over calibration activations) — a 0-GPU GO/NO-GO instrument for whether ANY depth-routing arm is worth Stage-1 GPU-h on a given architecture. Everything else is recorded and redirected.

---

## 1. Paper core

### 1.1 The GRN ladder (Section 3.1)

Stack of prior layer outputs `G_t ∈ R^{d×t}` = `[f_{t-1}(g_{t-1}), …, f_0(g_0)]` (with `f_0(g_0) = x`, the embedding). Standard ResNet = `G_t·1`.

| Variant | Combination | Params/layer | Note |
|---|---|---|---|
| GRN-v1 | `g_t = G_t·b_t`, b ∈ R^t | t | = DenseFormer (static scalars) |
| GRN-v2 | `g_t = (G_t ⊙ b_t)·1`, b ∈ R^{d×t} | d·t | per-dimension Hadamard |
| GRN-v3 | `g_t = (G_t ⊙ (b_t + w̄_t))·1`, `w̄_t = 1·ReLU(w_t^T G_t)`, w_t ∈ R^d | d·t + d | + input-dependent scalar-per-entry gate, broadcast; b init 1, w init 0 |

**Identity at init:** b=1, w=0 → `w̄_t = 0` → `g_t = G_t·1` = standard residual, byte-identical. This is the retrofit enabler and the reason DCA shows no loss spikes (each new gate starts at the incumbent's exact operating point).

**Memory variant (k-DCA):** keep only first + last-k layers explicitly in the stack; intermediates use the plain sum. k=2 → 48% lower inference latency vs full stack; PPL nearly unchanged (Table 1: 2-DCA 14.41 vs 24-DCA 14.35). Weights concentrate on the embedding + the last few layers — the same localization Block AttnRes (N≈8) and MSAR ("N≈8 blocks recovers most") report.

### 1.2 DCA (Section 3.2)

Three independent GRN-v3 instances per decoder block, all fed the same stack `G_t`, producing Q, K, V respectively. Attention mechanism unchanged; skip connections arranged so inputs feed both the attention and FFN inputs. Parameter overhead: the paper's own figure is **<0.2% of model size**.

### 1.3 Theory (Section 4) — the part worth keeping

Low-rank linear model, y = Ax + ε, collective rank `r* = Σ_t rank(V_t) < d`. Class characterizations: `C_GCN-v1 ⊇ {(αI+M)x : rank(M) ≤ r*}` (the scalar α is the extra DOF ResNet lacks), v2 adds a diagonal, v3 adds a rank-1 input-dependent term `σ(w^T x)x`. Theorem 4.3: GRN-v1/v2 achieve a better test-error/param trade-off iff

```
r*/d ≤ T(κ),   T(κ) = (1 + κ(√(κ²+1) − κ))² − 1 = κ(2s + 3κ)/(s + κ)²,  s = √(κ²+1)
```

(the simplified form re-verified by coordinator algebra; endpoints T(0)=0, T(1)=1, monotone in κ = λmin/λmax, the task's condition number). Reading: the gate mostly vetoes — GRN loses its license only on ill-conditioned tasks (κ ≲ 0.2) with high r*/d. Gain bounds GROW with d (at fixed r*) and shrink with r*; in practice r* scales with width, so the operative predictor is the RATIO r*/d and κ. Magnitudes are unpriced — the gate outputs GO/NO-GO, never a predicted PPL delta.

### 1.4 Experiments (Sections 5, LM1B + C4 + ImageNet)

| Cell | Result |
|---|---|
| Depth scaling | 30-layer DCA beats 42-layer transformer (29% fewer params/depth) |
| Training time | 2-DCA reaches transformer PPL in 0.33× the time |
| Width scaling | delta shrinks monotonically 64→1024 width (−2.82 → −0.39 PPL) — matches the rank theory |
| Ablation (6L LM1B) | Transformer 18.98 / GRN-v1 18.80 / GRN-v2 18.43 / GRN-v3 18.41 / DCA 18.06 — **no softmax-depth-mixing baseline anywhere** (verified against Table 7) |
| Retrofit (pretrained 6L, +60k steps) | DCA insert +0.19 PPL vs +0.02 continued training |
| Stability | No loss spikes on any DCA run (transformer shows them) |
| ViT-S/16 ImageNet | +0.7% accuracy (the marginal verdict the theory predicts at large width) |
| Family comparison (Table 5) | DCA 18.06 vs Hyper-Connections 18.65 / DenseFormer 18.80 / LAuReL 18.99, param-matched |

---

## 2. Coverage — the workspace map (§3.6 signal-diffs)

The dynamic-depth-routing family is **published, crowded, and workspace-saturated**. What ships and what each cousin consumes:

| Cousin | Where | Gate/signal it consumes | Diff vs DCA's GRN-v3 |
|---|---|---|---|
| **AttnRes** (Kimi K3, Research 447/330) | SHIPPED: `katgpt-transformer/src/attn_res.rs` (809 LOC; softmax over RMSNorm-scored dots, `score_weight = norm⊙proj`, block_size 4, stream-side at 2 sites/layer; riir-infer twin `transformer/gemma2_attn_res.rs`; full backward in riir-train `src/kimi_k3/`) | Normalized softmax convex combination over block residuals + running prefix | **Gate SHAPE**: softmax (normalized, weights sum to 1, cannot exceed 1) vs GRN-v3 (unnormalized additive ReLU broadcast on top of b≈1 — can exceed 1, preserves scale freedom, identity at init). **Site**: stream mixing vs QKV-input composition. The softmax-vs-GRN cell is uncovered on BOTH sides — the paper's own ablation never ran softmax depth mixing as a baseline. |
| **MSAR** (Research 619) | riir-train Plan 458 (staged) | Multi-head depth attention over the STREAM; causal-conv view expansion; sigmoid-gated fusion | Multi-head stream retrieval vs single-head QKV-input composition. Sibling arm, not duplicate. Plan 458 T6's sigmoid readout merges with Plan 514's gate table (one shared 3-arm table — DRY). |
| **xHC** (riir-train Issue 383) | harness arms | m-stream persistent state recurrence + learned mixing | Persistent streams vs per-site recombination of the full stack. |
| **DenseFormer** | classified (Research 619 taxonomy row "Cross-layer access: static") | static depth weights | = GRN-v1 exactly. No separate distill needed (PASS-Redirect). |
| **MUDDFormer** | classified (same taxonomy row, "dynamic") | per-token dynamic multiway dense | Publishes the QKV-split axis independently — no mechanism novelty available here. |
| **First+last-k / Block-AttnRes localization** | covered as design fact (Block AttnRes N≈8; MSAR "N≈8 recovers most") | — | Recorded, no primitive (full-stack overhead at our scales ≈0.04–0.1% params/MACs — the paper's own <0.2% figure; truncation optimizes a non-problem). |

**Reverse-grep for documented gaps this paper fills:** none found on the serving side — the corpus's residual-lane gaps are all pretraining-side (Plan 458's ladder lacks a DCA arm; the kimi_k3 lane lacks a gate-shape ablation). No serving-consumer gap exists because gains shrink with width and retrofit requires training steps (below).

---

## 3. Path 0 inventory (both advocates merged; coordinator-verified)

| # | Component | Analog ships? | Extraction without GD? | Disposition |
|---|---|---|---|---|
| E1 | Identity-at-init scaffold | Not in the residual lanes (the env-switch law has the same SHAPE for code posture, never checkpoint retrofit) | YES — byte-identity at b=1/w=0, short-circuit predicate, step-0 BLAKE3 gate | Design law for every new arm + the retrofit recipe (Plan 514 T0) |
| E2 | GRN-v1 static b_t, closed-form | Static module unshipped (DenseFormer classified only) | YES at linear readouts — ridge/LS on calibration activations; **intermediate-layer fits OUT OF SCOPE** (shift frozen downstream distributions — the governing trap) | Plan 514 Stage-0 closed-form-fit arm (adjudicates modelless-vs-trained at the same table) |
| E3 | GRN-v2 per-dim ridge | no | YES (same constraint) | folded into the same arm |
| E4 | Global α (theory's extra DOF) | n/a | YES — 1-D CE line search; expected gain ≈0 → the pre-registered FLOOR the b-fit must beat | floor probe inside the fit arm |
| E5 | Rank-ratio r*/d + κ diagnostic | NOT in corpus | YES — CCA + Bartlett χ² (+ seeded permutation null) / Marchenko-Pastur fallback / Ledoit-Wolf shrinkage; **CCA-first** (LayerNorm isotropy false-negatives the raw spectrum) | Plan 514 Stage-0 census instrument |
| E6 | κ-threshold gate | NOT in corpus | YES — closed form (verified); outputs GO/NO-GO only | rides E5 |
| E8 | Dilution census (stack-Gram, ‖Σh‖²/Σ‖h‖², cancellation ratio) | NOT in corpus | YES — same calibration pass | rides E5 — a direct offline measure of the dilution the paper's story rests on |
| E7 | First+last-k ring | design fact covered | YES (base/correction split keeps byte-identity) | RECORDED, no primitive — truncation is a non-problem at our scales |
| E10 | v3 gate w seeded (SVD/CCA direction) | — | HEURISTIC ONLY — no principled optimum, likely loses to trained AND static | optional micro arm, never claimed (discarded as a primitive) |
| — | QKV composition | published (MUDDFormer) + classified (619) | NO — all gains training-lane | Plan 514 Stage-1 arm only |
| — | 3× speedup / 30L-beats-42L / +0.19 PPL / no-spike | — | NO — training observations (spike-freedom is mechanistically explained by identity-at-init, which IS extractable) | G-bars for the arms |

---

## 4. Verdict (per track, never pooled)

**Tier: Gain** (no tier inflation: Q1 fails on mechanism novelty — the class is published and crowded; Q2 fails — no new behavior class, depth-routing behavior is covered by the shipped parent + filed MSAR; the deltas are implementation and instrument cells).

- **Track (a) transformer substrate — PASS-with-redirects.** A GRN/DCA forward module has the Issue-933 consumer-honesty shape: no production consumer, no expected checkpoint (`attn_res.rs` exists because Kimi-K3 weights exist). The forward lands in `katgpt-transformer` only when a Plan-514 arm wins its bake-off (Plan-097 support-ahead-of-weights, executed at arm-win time). No issue filed.
- **Track (c) training — GAIN.** riir-train [Plan 514](../riir-train/.plans/514_dca_grn_residual_arm.md): Stage 0 (0 GPU) = identity-at-init gate + GRN ladder arms (v2 doubles as v3's iso-param control) + closed-form-fit arm + rank/κ/dilution census; the gate-shape table (softmax / ReLU-additive / sigmoid-additive) SHARED with Plan 458 T6 — schedulable now. Stage 1 (+20–55 GPU-h) = iso-FLOP 4-arm cell (12L base / 12L+DCA / 9L-DCA / 9L base, 3 seeds) under the **three-column law** (Plan 364's GRT precedent: an iso-FLOP win with an iso-param loss reads "depth-deletion economics only" = PARITY-AT-BEST, never GAIN) + the **noise kill gate** (paired 3-seed LB95 including 0 → STOP, no escalation). Stage 2 gate = Stage 1 passes both, then a **recorded Claude verdict** promotes (not owner-gated — only credential touches are owner gates). Serving-checkpoint retrofit **DECLINED** for three structural blockers (27B step cost, serving-kernel forward work in every backend, post-quantization re-measurement); the retrofit pilot on an OWNED checkpoint stays an optional lane in the plan.
- **Track (a/b) modelless — folded into Plan 514 Stage 0.** No standalone katgpt-rs serving claim: the governing trap (closed-form fits at intermediate depths shift frozen downstream inputs; headroom smallest at the final readout) + the paper's own retrofit result (gains require training steps) close the frozen-serving angle. No runnable proof path on serving ⇒ no issue (ungoat discipline).

**MOAT gate (§1.6):** katgpt-rs row = transformer-stack primitives — the note is the record, no primitive ships now. riir-train row = active moat, training-method implementations — Plan 514 is the filing. Routing correct.

---

## 5. Fusion (what paper × A × B produces that none alone can)

1. **DCA gate shape × shipped attn_res lane (kimi_k3)** — the softmax-vs-GRN cell is open on BOTH sides; adding the sigmoid arm (house readout, Issue 932 lineage) makes it a 3-arm table that also tests the sigmoid-never-softmax rule on the depth axis for the first time. Shared with Plan 458 T6 so neither plan measures overlapping cells twice.
2. **Rank/κ/dilution census × Plan 458/514 arm ladder** — a 0-GPU GO/NO-GO instrument that prices ANY depth-routing arm before Stage-1 GPU-h: if the census says the architecture sits outside the DCA-beneficial regime, Stage 1 never runs. This is the cheapest falsifier in the file.
3. **Identity-at-init × the harness discipline** — every new residual arm gets a step-0 bit-identity gate + a gate-activity census (the silent-failure mode: an identity-init arm that STAYS at identity reads as "baseline parity win"). The paper supplies the mechanism; our harness discipline supplies the falsifier.

---

## 6. Prior art (§4 searches — all run 2026-11-24)

1. Headline/adoption: "DeepCrossAttention Inkling 2 QKV split depth cross-attention transformer adoption" → DCA cited-by ~16; Google blog confirms Inkling 2 (Thinking Machines) adopted DCA's QKV split + AttnRes. No newer survey displacing the family map.
2. Component: "MUDDFormer multiway dynamic dense connections Q K V separate depth mixing" → arXiv:2502.12170, ICML 2025, 49 cites — the QKV-split axis published independently. Mechanism novelty dead on that axis.
3. Internal-first grep: `2502\.06785|DeepCrossAttention|deep_cross_attention|depth.?wise cross.?attention` across ALL workspace `.research/` + `.plans/` + `.docs/` → **ZERO hits**. DenseFormer/HC/AttnRes coverage lives in Research 447/619/626/661 (family, not this paper).

---

## 7. Routing (files this verdict creates)

| File | Repo | Track |
|---|---|---|
| `.research/691_DeepCrossAttention_GRN_Depth_Routing.md` (this note) | katgpt-rs (public) | record |
| [`.plans/514_dca_grn_residual_arm.md`](../riir-train/.plans/514_dca_grn_residual_arm.md) | riir-train (private) | (c) training — staged, kill-gated |
| `.plans/458_msar_residual_arm.md` T6 cross-link edit | riir-train (private) | shared gate-shape table pointer |

Cross-repo boundary: the mechanism is public (family published); the harness arms and recipes are "how" → private riir-train; no public substrate file until an arm wins (Issue-933 consumer honesty). GOAT-audit not triggered: Plan 514 consumes riir-train-internal substrate (xhc/kimi_k3 lanes); no katgpt-rs primitive is consumed.

---

## 8. Weakest point (named by the coordinator, per protocol)

The training track's value rides an **unscheduled Plan-318-class pretrain** — the identical contingency Research 619 §7 named for MSAR; recorded in Plan 514's Status line (verdict condition 4). The deltas that do NOT share it are the cheap ones: Stage 0 (0 GPU) and the gate-shape table (existing active lane, 1–3 runs). Second: the paper's own theory predicts moderate-to-small Stage-1 effects at 150M/d=768 — a pass may sit inside 3-seed noise; the **noise kill gate** (LB95 including 0 → STOP, no seed/scale escalation) prices that failure at the pre-registered budget instead of an open-ended spend (verdict condition 3).
