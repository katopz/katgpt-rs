# Research 661: SimpleHC — Minimal Gated Residual Stream + Shared Memory Access

> **Status:** GOAT — Gain on track (c) training → **riir-train Plan 493** (SimpleHC arm in the existing `xhc_pretraining` harness, staged + kill-gated). Track (a) substrate: deferred row, no file today (no consumer checkpoint exists — Plan 097 precedent gated on need). Track (a/b) modelless: **two audited discards** (serving group-defer; static gate calibration), both with mechanism-level reasons. **No novelty claim on the class** — the HC family is published and workspace-saturated (095, 519, 571, 619, 629). NOT Super-GOAT. **This is a routing verdict — the GAIN is unproven at our scale** (see §6). Claude verdict session `6e2ff56e` (claude_code reviewer): round 1 REVISE — 3 required fixes (T0.4 init cannot be bit-exact AND trainable — rewritten as α=0 init + tolerance-pinned step-0 equivalence with exact bit-identity only on the 2σ(0)=1.0 write path; G1 promotion gate ADDED to Plan 493 T2.2 — paired LB95>0 vs param-matched control, ties PARK; the "riir-train 383" citation bound to the wrong document — now cites the harness by module path + close hash `bcbc52fb`) + 5 smaller edits (Stage-2 scope honesty — quality transfer, not the efficiency claim; executed-path traffic counter; micro depth/K spec; MDE + UNDECIDABLE pre-registration; Status→line 2) — all applied; **round 2 AGREE** (1 leftover fixed in-commit: bare "Issue 383" in §4 now path+hash form; 3 optional notes taken: non-zero W init pinned at T0.4, an off-init FD probe added at T0.2, T2.2's MDE relabeled a target with the likely-UNDECIDABLE spend disclosure).
> **Source:** [On the Design of Hyper-Connections: Simplification, Stability, and Scalability](https://arxiv.org/abs/2611.05984) — Zhang, Qin, Zou, Dai, Shi, Xia, Yang, Chen, Liu, Cheng, Yan (SJTU / Xiaohongshu Dots Studio), arXiv:2611.05984 (2026-11-04). Code repo https://github.com/aHapBean/SimpleHC — **README + PDF only at distill time, NO training code, NO license file** (tree pinned `67d0f9e587da6e694d8e771008bbb03a0699ed08`; authors say training code + DeepGEMM kernels "next month or two").
> **Date:** 2026-11-09
> **Related Research:** 619 (MSAR — the sibling family distill; its §1.1 table is the family ladder), 629 (depth-memory RLS — covers mHC + the closest systems-trick cousins: state unwinding, windowed writes), 095 (MGR — multi-gate residuals, arXiv:2605.23259), 519 (GRT — gate-bias identity-init recipes, Plan 364 lineage), 571 (Needle3 — mHC 4-lane Sinkhorn in a shipped tiny model)
> **Related Plans:** riir-train **493** (this paper → SimpleHC arm), riir-train 458 (MSAR arm — the sibling carrier; piggyback option), riir-train 364 (CLOSED-negative GRT but the standing verdict machinery: classify3, HP-parity, gate-bias identity-init, Bench-560 sweeps), riir-train xhc_pretraining **harness** (`riir-train-engine` `xhc_forward/xhc_dense/xhc_grt` modules + `tests/xhc_train_phase*.rs`; filed as riir-train issue 383, CLOSED at `bcbc52fb` — cited by path + hash, NOT by the bare number, which now belongs to `.plans/383_posterior_gpu_integration.md`), riir-train 318 (eventual consumer pretrain, unscheduled), katgpt-rs 097 (`delta_routing` — speculative architecture-support precedent)
> **Classification:** Public

---

## TL;DR

SimpleHC is the efficiency-simplified member of the Hyper-Connections family: **learned read/write gates with identity transitions (no residual mixing) + "shared memory access"** — K=4 sublayer groups whose gates are all generated from ONE group-entry state read, with batched base reads, deferred writes, and scalar corrections that exactly preserve preceding sublayer output contributions. Two component-study findings carry the design: (1) learned gating alone ≈ full mHC (55.4 vs 55.0 avg; unconstrained mixing alone 56.0 — two independent interaction groups), and Sinkhorn-constrained mixing kills the mixing benefit; (2) the training overhead is MEMORY TRAFFIC to the expanded residual state, not FLOPs — shared access cuts logical forward traffic 18C → 9C per token·sublayer and end-to-end training overhead from ~7.0% (mHC) to **2.4%** at C=6144, with −31% saved activation storage, equal loss at 30B (1.561), and bounded gradients at a 230B-A6B 10×-LR stress test where mHC spikes to ~1.9×10⁶.

**Distilled for katgpt-rs (modelless, inference-time):** almost nothing survives onto frozen checkpoints — and the honest finding is *why*: the execution scheme is architecture-EXECUTION CO-DESIGN. The deferred-write/scalar-correction algebra (u_j = base read + Σ inner-product corrections; boundary write-back) is the same algebraic family as chunked delta-rule linear attention, which already ships in the GDN lane (chunked-GDN MMA prefill rung: chunk-entry state + intra-chunk causal corrections). SimpleHC's version amortizes GATE GENERATION from a state frozen across K sublayers — a window that per-token recurrent state does not have. The paper's own fixed-gate rows (Table 2: 52.5–53.6 vs 55.4 learned) quantify static gating as inferior, closing the "calibrate gates statically on frozen checkpoints" door with the paper's own data. The value is the **training recipe** (Plan 493) plus recorded design laws (share gate INPUTS never parameters; corrections always-on; never Sinkhorn; identity transitions eliminate the dH_res dynamic gradient path).

---

## 1. Paper core

### 1.1 The architecture (mHC-parameterized, mixing removed)

State X_l ∈ R^{n×C}, n=4 streams. Per sublayer: u_l = X_lᵀ h_pre, o_l = F_l(u_l), **X_{l+1} = X_l + h_post o_lᵀ** (identity transition — mHC's H_res X_l mixing matrix removed). Gates from full state: z_l = RMSNorm(vec(X_l)); ℓ_s = α_s (W_s)ᵀ z_l + b_s; h_pre = σ(ℓ_pre) ∈ (0,1)^n; **h_post = 2σ(ℓ_post) ∈ (0,2)^n**.

### 1.2 Component study (Table 2, 15B-A1B MoE; avg of MMLU/HellaSwag/MATH/MBPP/C3)

| Config | Gates | Mixing | Loss ↓ | Avg |
|---|---|---|---|---|
| Vanilla | — | — | 1.678 | 52.4 |
| HC w/o mixing | C (stream-local) | identity | 1.664 | 55.1 |
| **identityHC** | nC (full-state) | identity | 1.660 | 55.4 |
| mHC (Sinkhorn) | nC | Sinkhorn | 1.659 | 55.0 |
| mHC w/o Sinkhorn | nC | unconstrained | 1.657 | **56.2** |
| fixed read+write | fixed | unconstrained | 1.660 | 56.0 |
| fixed read+write | fixed | Sinkhorn | 1.678 | 52.5 |
| fixed read only | fixed | identity | 1.676 | 53.6 |

Findings: **two effective interaction groups** — learned read/write gating (either gate fixed → drops to 54.7–54.9) and unconstrained mixing (56.0 with both gates fixed); either alone ≈ mHC; combined best (56.2); **Sinkhorn erases mixing's benefit** (52.5 fixed / 55.0 learned). Unconstrained mixing risks cumulative amplification (larger peak residual-path coefficients, Fig 7; gradient spikes at stress). Stream-local gate inputs cost only +0.001–0.003 loss vs full-state.

### 1.3 Shared memory access (the paper's claimed delta over prior efficiency work)

With identity transitions, state at sublayer j decomposes exactly: X_j = X_L + Σ_{k<j} h_post_k o_kᵀ, so the read becomes

**u_j = X_Lᵀ h̄pre_j + Σ_{k<j} (h̄pre_j)ᵀ h̄post_k · o_k** — base read + scalar corrections (Eq 2/4).

Gates for all K=4 sublayers of a group are generated from the group-entry state X_L (Eq 3) — the only approximation vs per-sublayer conditioning. Group-boundary write-back X_{L+K} = X_L + H̄post Oᵀ (Eq 5). Appendix B proves forward AND gradient equivalence to sequential execution with the same shared-input gates — **the gradient must keep every gate's dependence on X_L (detaching gates invalidates it)** — the implementation trap.

Algorithm 1 is fully specified (batched reads V = X_Lᵀ H̄pre; D = tril(H̄preᵀ H̄post, −1); corrections; deferred write-back). Backward: shared activation-gradient access (∇X_L += E + H̄pre Gᵀ; ∇O += Eᵀ H̄post — one scan of E for all K output gradients) + token-packed gate gradients (C→16 reductions per token).

### 1.4 Traffic and cost model (Table 1 + Eq 9/10)

| Per token·sublayer | naive mHC | fused mHC | fused identityHC | SimpleHC |
|---|---|---|---|---|
| total | 34C | 18C | 18C | **9C** |

Q(K) = 4nC/K + 2C + C(K−1)(K+4)/(2K); at n=4: 11.5C / **9C** / 9.25C / 12.375C for K=2/4/8/16 — **K=4 is the measured optimum, not a universal one** (correction cost grows with K). Group-entry-state reuse cuts saved residual storage 1−1/K (75% at K=4); total retained-activation −31% vs mHC (94U+4672T → 65U+1648T, Appendix A.2 ledger).

### 1.5 Results

15B: overhead ladder (efficient impls): mHC 7.0% / identityHC 7.0% / SimpleHC 2.4% at C=6144; PyTorch impls 58.4% vs 22.5% (kernels matter). xHC (16 streams, 4 active) 27.7%. 30B-A2B: loss 1.561 = mHC = identityHC; avg 63.1 vs 62.9/62.8. Stability: max 1000-step rolling grad-norm std after warmup 0.032 (SimpleHC) vs 0.206 (mHC) / 0.248 (identityHC); 230B-A6B @10× LR: mHC spikes ~1.9e6, SimpleHC/identityHC bounded. Mechanistic (E.3): mixing adds a **dynamic gradient path dH_res·X** (measured ~9.4× the fixed-matrix path in one captured spike); identity transitions set dH_res = 0. Gate predictability: layer-specific predictors from the group-entry state reach write-gate MAE 0.0103 (frozen teacher gates 0.0504, random 0.4086); reusing gate PARAMETERS degrades 55.0→53.5 — **share inputs, never parameters**. Removing scalar corrections: loss 1.660→1.705 (+0.045 — larger than the entire mHC-vs-SimpleHC gap).

---

## 2. Path 0 decomposition (per component)

| Component | Paper role | Coverage / extraction in this workspace | Signal-diff vs closest shipped cousin | Status |
|---|---|---|---|---|
| Gates + identity transitions (architecture) | the minimal residual structure | **training arm missing** — xhc harness arms are per-sublayer-conditioned (xHC conv recurrence; GRT gate MLP); no identity-transition-with-shared-conditioning arm | xHC = gates + conv stream-recurrence; SimpleHC = gates + identity + group conditioning. Same harness, new arm | **new arm** → riir-train Plan 493 T0 |
| Shared group-entry gate conditioning (Eq 3) | execution simplification | not shipped anywhere; recipe-level | conditioning choice, not a standalone primitive; predictor heads are trained components — do NOT ship them | **recipe** → Plan 493 (free-gates-vs-heads ablation deferred to Stage 3) |
| Scalar corrections + deferred writes (Eq 2/4/5) | exact execution algebra | **algebraic family ships**: chunked delta rule in the GDN lane (chunk-entry state + intra-chunk causal corrections; the opt-in chunked-GDN MMA prefill rung) | GDN corrections are data-dependent matrices per token (state mutates every token — no frozen-K-step window to amortize); SimpleHC's are gate inner products on a state frozen for K sublayers. Family cousin, different instance; the residual stream it amortizes belongs to an architecture no frozen checkpoint of ours has | **audited discard** (serving) + ships as harness execution (Plan 493 T0/T1) |
| Traffic model Q(K) + activation ledger | cost accounting | our kernel byte-counting discipline exists; no residual-stream lane to apply it to | applies only to HC-trained checkpoints | **extracted as a plan assert** → Plan 493 T0.6 (loud-zero-defended counter) |
| Stability rules (identity kills dH_res path; 10×-LR stress) | training-recipe knowledge | no standing stress-probe row in the harness | new harness telemetry row | **recipe** → Plan 493 (stress probe + saturation kill line) |
| Gate-bias init | "follows the mHC xHC-repo scheme, adapted" | **Plan 364 T5 standing recipe** (σ/bias identity-init; Bench-560 sweep; site-dependence warning) | fused derivation below | **fusion** → Plan 493 T0.4 |
| Two-groups finding + Sinkhorn negative | design law | harness design posture | pre-registered anti-pattern (normalization deleted a measured win — same shape as corpus measurement-honesty rows) | **recorded** → Plan 493 posture (no mixing/sinkhorn arms) |
| Fixed-gate rows (52.5–53.6) | ablation datum | the weight-free deterministic-mixing shape already exists as a measurement lane (`apply_attn_res_depth_profile`, the issue933 probe — gemma-2 carries NO AttnRes weights; the lane inserts score = −λ·age) | the paper quantifies the static-vs-learned gap the probe lane lives in | **recorded** as lineage for the probe lane + discard leg |

## 3. Modelless discards (audited, mechanism-level)

- **Discard A — serving group-defer on deltanet/KV/HLA.** The No-GD advocate's top recommendation (K=4 group-defer + exact corrections in `riir-infer/deltanet`). Refuted on the cousin read: chunked delta-rule linear attention ALREADY implements base-read + intra-chunk corrections (the chunked-GDN MMA prefill rung), and deltanet's per-token state mutates every token, so there is no frozen-K-step window in which to amortize gate generation — the precondition (Eq 3) that makes SimpleHC's traffic law possible. The 9C number is a property of an n=4 expanded residual stream no frozen checkpoint of ours carries (the advocate itself named this weakest point). A single-stream residual's serving traffic is already ~2C/sublayer; nothing to amortize.
- **Discard B — static gate calibration on frozen checkpoints.** Refuted two ways: (1) the paper's own fixed-gate rows measure static gating ~2 pts below learned (Table 2 rows 9–11) — and checkpoints with TRAINED mixing/gates already carry the better version, so replacing them is a lossy transform on league-pinned surfaces with no upside claim; (2) for checkpoints with NO such weights (gemma-2 class), the weight-free deterministic-profile shape already exists as the issue933 measurement lane — the right posture (measurement-scoped, byte-identical disarmed) is taken.

## 4. Fusion subsection

- **paper × Plan 364 T5 (gate-bias identity-init)** = the step-0-vanilla boot: stream-0 read OPEN, **stream-0 write at bias 0 (gate = 2σ(0) = 1.0 exactly)**, all other gates CLOSED → SimpleHC's forward is bit-equivalent to the vanilla arm at init (the coordinator correction of the advocate's draft, whose "writes CLOSED" variant freezes the residual and collapses the model). Bench-560-style sweep explores the deviation direction; the site-dependence warning carries.
- **paper × Plan 458 (MSAR staged ladder)** = piggyback option: SimpleHC as an additional arm sharing Plan 458's Stage-1 vanilla controls → marginal ~50–70 GPU-h instead of ~75–110 standalone. Owner-gated choice.
- **paper × the xhc_pretraining harness** (`riir-train-engine` `xhc_*` modules + `tests/xhc_train_phase*.rs`; riir-train issue 383, CLOSED at `bcbc52fb`) = the family's 5th arm (Vanilla → xHC → GRT → MSAR → SimpleHC), executed under Plan 364's classify3 verdict discipline (param-matched control mandatory; the GRT micro-GAIN→transplant-FAIL precedent is the standing falsification posture).

## 5. Prior-art searches (§4 — run before verdict)

1. Headline ("SimpleHC hyper-connections simplification training overhead identity transitions", recency oneMonth): only the paper itself (2026-11-04) — no competing simplification follow-up in its first week.
2. Family landscape (recency-free + 629/619's earlier sweeps, which found HC ICLR'25 arXiv:2409.19606 · mHC arXiv:2512.24880, in DeepSeek-V4-Pro + GLM-5 · xHC arXiv:2607.14530 NeurIPS'26 · AttnRes arXiv:2603.15031, in Kimi K3 · mHC-lite arXiv:2601.05732 · oHC arXiv:2609.02672 · kroHC arXiv:2601.21579 · go-mHC arXiv:2604.02309 · spectral-sphere arXiv:2603.20896 · Frac-Connections arXiv:2503.14125 · MGR arXiv:2605.23259 = our note 095 · MuddFormer · DenseFormer). **IdentityHC itself is published** (Guo et al. 2609.02672; adopted in Tencent Hy4-preview; "removing mixing preserves quality" prior-reported) — SimpleHC's delta is the two-groups decomposition + the shared-memory-access execution scheme, which its own Related Work positions against block-wise aggregation (AttnRes/Team 2026), kernel fusion (mHC), and sparse updates (xHC).
3. Internal-first grep (`2611.05984|hyper.?connection|mHC|xHC` over all `.research/.plans/.docs`): five family notes + no SimpleHC distill — this note is the first.
4. **Pinned claim:** *recipe transfer of SimpleHC (gates+identity, group-entry conditioning, deferred writes/corrections) into the existing xhc_pretraining harness as a staged, kill-gated arm — consuming Algorithm 1 + Appendix B (no author code), distinguished from the harness's existing arms by shared gate conditioning and identity transitions, with ZERO novelty claimed on the class (published + workspace-saturated).*

## 6. Verdict

**GOAT — Gain on track (c), and the label is a ROUTING verdict: the gain is unproven at our scale.** Two scope statements keep it honest. First, what Stage 2 of Plan 493 can and cannot measure: the CPU-bound harness validates execution (grouped ≡ sequential, FD backward), but the 150M GPU cell tests **quality transfer of identityHC + group-entry conditioning** — the PRIOR-PUBLISHED part of the design (oHC arXiv:2609.02672, Tencent Hy4) — NOT the paper's efficiency claim (2.4% vs ~7% overhead), which would need HC-fused kernels in the GPU training stack, out of this plan's scope. Second, statistical power: at 15B the vanilla→identityHC gap is 0.018 nats; a 3v3-seed 150M cell is powered only for a pre-registered minimum detectable effect — an ambiguous read is UNDECIDABLE, never folded into non-inferiority (details in Plan 493 T2.2). The class is published and internally saturated (five prior family notes); the paper's own delta (the execution scheme) is architecture-execution co-design with no frozen-checkpoint consumer; the modelless extraction surface closes under the paper's own ablations. What remains actionable and cheap is the training recipe against a harness we already run: Stage 0 is 0 GPU-h, the decision read is a ~75–110 GPU-h 150M cell (or ~50–70 marginal piggybacked on Plan 458), kill-gated per the GRT precedent. **Track (a) substrate:** deferred row inside Plan 493 — ship a `simplehc`/`identityhc` forward in `katgpt-transformer` (beside `attn_res.rs`, Plan 097 precedent) only when a consumer checkpoint exists; none does today (league incumbents + Kimi-K3 arch-test carry vanilla/AttnRes residuals). **MOAT gate:** katgpt-rs row satisfied by the generic recipe/lineage record (this note); training how → riir-train (Plan 493, private). No Super-GOAT: no new capability class, no product selling point beyond "our future pretrains get a cheaper residual arm".

**Priorities:** P0 = Plan 493 Stage 0 (0 GPU-h) · P1 = Stage 2 decision cell (owner-gated spend) · P2 = Plan 458 piggyback coordination · P3 = katgpt-transformer forward support (consumer-gated), ternary-STE annex (Stage 3, owner-gated).
