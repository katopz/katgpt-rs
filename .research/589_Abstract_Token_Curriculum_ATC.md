# Research 589: Abstract Token Curriculum (ATC) — Streak-Gated Difficulty Ladders, Corrective Backtracking, Answer-Only Latent Thoughts

> **Source:** [Learn Your Own Thoughts: Abstract Token Curriculum](https://arxiv.org/abs/2609.19717) — Gatmiry, Ghosh, Mirtaheri, J.D. Lee, Haghtalab (UC Berkeley), Abbe (EPFL), Bartlett (DeepMind/Berkeley). v2, 23 Sep 2026.
> **Date:** 2026-09-25
> **Status:** Gain (GOAT-tier, two tracks) — modelless: Issue 887 (`ladder_gate` FSM, katgpt-rs); training: Plan 352 ATC row (riir-train, pre-registered, triggers unfired). NOT Super-GOAT (Q3 selling-point fails: engineering control rule + recipe, not a product differentiator).
> **Related Research:** 442 (LOTUS — closest training cousin, PASS w/ deferred half in 352 row 6), 496 (SPADE hint-regret curriculum — the modelless regulator without a generator), 192 (NextLat/Coconut lineage), 325 (latent-reasoning survey — the family map), 213/523/531 (KV eviction: Still Perceiver, H2O norm-age, random-null), 258 (attention sinks), 097/073 (looped-transformer stack we ship)
> **Related Plans:** riir-train 352 (training backlog — new ATC row, this note's §5), riir-train 330 (latent CoT, CLOSED NEGATIVE Bench 438), riir-ai 288 (dendritic difficulty-gated curriculum), riir-ai 340 (hint-regret guide)
> **Classification:** Public
> **Panel:** three-track adversarial panel run (No-GD + Model-based advocates, §3.5); prior-art web sweep + full-workspace codebase grep executed. Verdict ping-pong: see footer.

---

## TL;DR

ATC trains a transformer to develop **continuous latent thought vectors** (last hidden state fed back as input, bypassing the embedding layer) via a **difficulty curriculum** with **answer-only supervision** — no CoT traces, no intermediate labels. Four mechanisms: **streak-gated advancement** (advance iff held-out accuracy ≥ τ for m=5 consecutive evals), **curriculum backtracking** (re-probe ALL passed stages each gate; retreat to the *shallowest failing* stage on regression — corrective, vs preventive λ=0.1 rehearsal), **truncated backprop** (gradient window over last b=5 latent passes: 27% mem / 63% time, −2.4pp), **candidate dropout** (ρ=0.75 hides copyable candidates; ρ=0 → stuck at stage 0). Theory: **attention simplicity bias** — one GD step on keys concentrates ≥99% of readout attention on the most recent thought token; empirically ~50% mass on last thought, 1.4% on earlier thoughts; the last thought alone is **sufficient** for the answer (earlier ones only regenerate it).

**Verdict: Gain, GOAT-tier, per-track (no pooled verdict):**
- **Modelless track** — the stage-controller is a genuinely unshipped control loop: every shipped streak gate runs the *inverse* direction (halt/suppress/demote on consecutive failures); every shipped fade mechanism (riir-clippy FADED/STALLED/auto_readmit) is detect-only or trigger-only, none actuates a retreat-and-retrain pointer. → **katgpt-rs Issue 887** (`ladder_gate`).
- **Training track** — pre-registered into **riir-train Plan 352** (the G-COCO/LOTUS family backlog). All triggers remain UNFIRED (zero latent-plan consumers); ATC's new evidence recorded for the next adjudication.
- **Not Super-GOAT**: Q1 holds for us (no shipped counterpart; no published instantiation found for streak-gate/backtrack/candidate-dropout), but Q2/Q3 are weak — an engineering control rule and a reproduction recipe, not a new capability class a customer can feel.

---

## 1. Paper core (what is claimed, with the numbers that matter)

1. **Difficulty ladder, answer-only supervision.** Stage s poses difficulty min(s+1, d) with latent budget min(s, d−1) thoughts. Graph reachability depths 3–16 (100k train graphs, 2-layer GPT-2 15M params), parity orders, digit counts. Beats data-curriculum-only on parity across 20 phases (near-perfect vs steady degradation); matches Coconut-with-intermediate-supervision on graphs (99.9%) at **2.6× fewer epochs** (51 vs 132; Coconut fixed-schedule: 468 epochs → 97.6%).
2. **Streak gate + dwell-dominance (Appendix H.2).** (τ=0.9, m=1) → 91.8%; (τ=0.98, m=1) → 98.5%; (τ=0.9, m=5) → 99.9%. **Raising m (dwell) beats raising τ (strictness): +8.1pp vs +6.7pp, and (0.9,5) beats (0.98,1) outright by +1.4pp.** A threshold below the attainable ceiling admits a half-learned difficulty and the run *stops there*.
3. **Curriculum backtracking.** At every gate eval, re-measure accuracy at every passed difficulty *at that difficulty's own latent budget*; if any regressed, return to the shallowest failing one. The load-bearing ablation: under truncated backprop b=5, advance-only collapses to **51.0%** vs **97.4%** with backtracking (−46pp). Under full backprop, backtracking is nearly free (99.9 vs 99.8).
4. **Rehearsal λ=0.1** (preventive) is necessary but not sufficient: with λ=0 the curriculum never passes difficulty 2 (8/8 arms); rehearsal spreads capacity uniformly (λ/(s+1) per stage — 1.1% at s=8) whether or not a stage regressed; backtracking concentrates the entire batch on the earliest failure (corrective). Both are used together.
5. **Truncated backprop** (conceded known for looped transformers — Huginn/Ouro): gradients through last b latent passes only. b=5: 27% peak mem / 63% step time (graphs); 64%/81% (arithmetic); b=2: 27%/57%.
6. **Candidate dropout ρ=0.75**: hides the two visible answer candidates on most rows (copy shortcut explains half the labels for free); ρ=0 → curriculum never leaves stage 0. Evaluation always shows both candidates.
7. **Attention simplicity bias (Theorem 1).** Parity setting, correlation-one assumption, one full-batch GD step on keys only: final-answer attention ≥99% on the most recent thought token, w.p. ≥1/4. **Not shippable as a guarantee** (assumption-bound); the transferable part is the *ordering law*.
8. **Last-thought sufficiency (interventions, §6.4).** Corrupting thoughts 1..k with the last pinned clean → accuracy flat (99%); letting corruption propagate into the last → collapse. One clean thought *regenerates* the whole downstream chain. Readout attention: ~50% last thought / 1.4% earlier thoughts / 2.3% edge tokens (Coconut: 39.6%/17.0%).
9. **The addition result (the quiet bombshell).** 8-digit addition: ATC 98.7–99.6%; **Coconut-with-intermediate-supervision 0%** (stalls: partial sums from operands are unlearnable — the intermediate state the supervisor picked was the wrong thing to compute). Answer-only supervision didn't just match the supervised baseline here; the supervised baseline *failed*.

---

## 2. Prior art (web sweep, 2026-09-25)

- **Headline combination** (difficulty ladder over task distributions + continuous thoughts + zero CoT data): **not found as a published combination**, but the explicit→latent curriculum space is extremely crowded (Coconut 2412.06769 [format curriculum, needs CoT traces], RiM 2605.30343 [stage-2 answer-only refinement, special tokens not continuous chains], LUT 2608.00743, Penelope, SWITCH, LT-Tuning 2602.10229, AdaBack 2506.18110 [per-sample rationale curriculum, shares co-author Abbe]). The defensible residual is the **difficulty axis + the theory**.
- **Attention simplicity bias theorem**: no exact prior art; adjacent (Vashisht & Ramaswamy 2608.06776 QK-sharpening; implicit-bias-of-attention line 2402.05738/2403.08699; distributional simplicity bias 2410.19637).
- **Curriculum backtracking / streak-gated advancement / candidate dropout**: no published prior art found (phrase-based negatives — an honest caveat; conceptually akin to mastery-learning gates).
- **Truncated backprop**: prior art conceded (Geiping/Huginn 2502.05171; Ouro 2510.25741) — ATC's claim is only the *combination* with backtracking for sequential continuous-CoT chains.
- **Last-thought sufficiency**: methodological precedents (interventions 2606.12689; "Weight of Silence" 2607.20952 — even stronger thoughts-not-consulted result on a chess latent model; AGCLR 2606.07720 concept bottleneck; Jacobian-lens 2609.01924 ~2-recurrence readout window). ⚠ **FlashLoop (2609.29812, one week after ATC v1) already exploits cross-loop concentration for KV-cache reduction (6×) on LOOPED models** — any KV-eviction claim derived from ATC must scope to sequential continuous-CoT chains and cite it.

## 3. Codebase coverage (workspace grep, both layers)

| ATC mechanism | Shipped state | Signal-diff vs the closest shipped cousin |
|---|---|---|
| Streak-**advance** gate | **None.** All shipped streaks run the inverse direction: `gain_cost_halt.rs` `inversion_streak` (halt), riir-clippy `pair_fail_streak` (suppress), `EvidenceTier::Withdrawn` 3-consecutive-fails (absorbing demotion), riir-bench-algo `patience` (early-stop) | Direction differs: those demote-on-failure; ATC advances-on-success with reset-on-any-failure. No stage pointer anywhere. |
| Difficulty **retreat** | **None.** riir-clippy `frontier_report` FADED (trailing-tail statistic, report-only, never gates), `gate_calibration` STALLED (zero-advance window, report-only), `active_set::auto_readmit` (re-admits a *demoted* rule on fresh diagnostics — trigger-only, no stage retreat, no retrain loop); riir-ai `rewind_recovery` (collapse-gated rewind, no destination rule) | FADED/auto_readmit consume *resolution counts* and act on *lint admission*; ATC consumes *per-stage held-out accuracy at own budget* and actuates a *training-distribution pointer*. Detect-only vs closed-loop control. |
| Difficulty ladders | PARTIAL: `hint_regret` (per-item content triage: Offer/Retire/Evict — the regulator without a generator), Plan 288 dendritic (HLA-moment difficulty → LoRA branch, schedule-driven), score-bench generations ("a difficulty rung, not a time step") | No advancement predicate + retention in one loop anywhere. |
| Candidate masking | **None** (riir-reflex `slice_leak` is train/eval decontamination — dataset hygiene, different layer; reflex's cal-slice incident is the recorded ρ=0-class failure) | Layer differs: slice-level vs row-level prompt composition. |
| Truncated backprop | PARTIAL: depth-1 xHC (Bench 383), recurrent-depth GRT (Bench 559) — never latent-pass chains + backtracking | Combination unshipped. |
| Keep-last KV | Shipped positional/theorem-driven: `kv_sink_window` (sinks + trailing window), ALiBi×entmax exact `d_max` window, usage-rate eviction, flashmemory recency fallback. **None derive the keep-set from measured readout attention mass at thought boundaries.** Measurement ships: `select_highest_attn_keys` + `mass_coverage`. | Basis differs: positional/theorem vs attention-mass-ordered. |
| Latent-thought training substrate | riir-train Plan 330 (`TrainingInput::Embeddings`, `nf_curriculum.rs`, `AuxLoss`) — **CLOSED NEGATIVE** on Rust codegen (Bench 438); every trained-latent attempt is a recorded null (Plan 276 flip-flops, Bench 679 "untrained latent loop is inert", Issue 568 no-transfer) | 330 had no streak gate, no backtracking, no window, no dropout — and supervised-shaped latents. See §5. |

## 4. Path-0 decomposition (merged panel inventory)

| # | Component | Coverage | Modelless extraction | Route |
|---|---|---|---|---|
| Q1 | Streak-advance FSM (τ, m) | none | YES — counter+threshold+reset | Issue 887 |
| Q2 | Dwell-dominance law (m beats τ; +8.1 vs +6.7pp; (0.9,5)>(0.98,1)) | none | YES — pinned inequality + config defaults | Issue 887 (G1 pin); consumers: reflex cal cadence, gate_calibration |
| Q3 | Backtracking (argmin-j retreat, own-budget re-probe) | detect-only cousins | YES — argmin over monotone stage pointer | Issue 887 |
| Q4 | Rehearsal ratio λ=0.1 (+λ=0 hard negative) | partial (training arms) | YES — batch composition | Issue 887 (negative control) |
| Q5 | Window bound b (27%/63%/−2.4pp; **no-backtrack@b=5 → 51% collapse**) | partial (training) | runtime analog: bounded state window via freeze/thaw | Plan 352 row (training); collapse warning feeds Issue 887's G1 |
| Q6 | Candidate dropout ρ | none at row level | leak predicate (eval hygiene) | noted (reflex slice_leak extension candidate) |
| Q7 | Readout attention-mass ordering (last ≫ earlier, ~36:1) | measurement ships, policy doesn't | YES — offline statistic | gated consumer (see §6) |
| Q8 | Last-thought sufficiency / Markov regeneration | none | YES — checkpoint/replay semantics | gated consumer |
| Q9 | Theorem 1 (99% after one GD step) | none | NO as guarantee (correlation-one assumption) | diagnostic probe only (Plan 352 row T5) |
| Q10 | Difficulty metrics (d/k/digits) | shipped (training tiers) | YES | covered |
| R1–R10 | Full training recipe | Plan 330 substrate (closed negative) | n/a | **Plan 352 ATC row** |

**Advocate-discard audit (§3.5):** No-GD candidates C1/C2/C6 → Issue 887 (adopted); C4 (freeze/thaw bounded window) folded as a noted consumer pattern, not filed — *reason: no latent-chain runtime consumer exists; filing would be a backlog-wearing-an-issue*; C5 (candidate masking) noted in §6 — *reason: reflex's cal-slice fix already removed the live failure; the remaining value is hardening, below the file-bar*. Model-based R1–R6/R8/R10 → Plan 352 row; R7 (ρ mechanism port) discarded — *reason: our corpora carry no multiple-choice candidates; only the lesson transfers (the filter already fights echo/zlib leakage)*; R9→GDN discarded — *reason: delta-rule has no softmax keys; the one-GD-step analysis has no object there*.

## 5. Training track — the Plan 352 ATC row (why a row, not a plan)

The LOTUS precedent (R442 training half → 352 row 6) and 352's batching mandate make the row the DRY home; a new plan would fragment the family. **All triggers remain UNFIRED** (zero latent-plan consumers; Plan 351's mixed verdict stands; no Glacial-tier latent-plan request) — the row is pre-registered exactly like G-COCO. ATC's **new evidence** for the next adjudication:

1. **Answer-only supervision eliminates the CoT-trace teacher requirement** — the largest data cost in the G-COCO recipe (teacher traces) drops to zero; only verifiable answers needed (synthetic generators give them for free).
2. **The addition arm is a mechanism-level reframe of Plan 330's negative**: Coconut's *chosen intermediate state* (partial sums) was unlearnable where answer-only succeeded. 330/Bench 438 supervised-shaped latents — the wrong constraint is a live alternative hypothesis to "latent compression regresses codegen", and 330's own non-goals already carved out non-code scratchpad CoT as untested.
3. **Affordability**: the paper's own scale is 15M params / 100k graphs / ~51-epoch curriculum ≈ 1–4 GPU-h per run on the 4090 (25–50 GPU-h for the decisive ablation matrix) — cheap enough to run *as the gate itself*.
4. **The GDN question is unanswered anywhere**: delta-rule recurrence carries state forward natively (no softmax keys) — does it form latent thoughts *better* than the recency-collapsing softmax the paper's theory describes? No arXiv paper answers this for delta-rule models; `deltanet/` + the Plan 414 Bonsai+GDN lane are the fleet alignment.

Recipe (pre-registered in the row): T1 generators (parity/reachability/digit-count, exact-solver floor compiled in per Report-the-Floor); T2 15M arm with ablation matrix {backtrack on/off} × {b=2,5,full} × {ρ=0,0.75} × {(0.9,5),(0.98,1)} × {λ=0,0.1}; T3 GDN-small arm (2–4M); T4 Kimi-K3-0.4B arch rung (gated on T2/T3); T5 GOAT report (the 51→97 backtracking reproduction is the headline check; epochs-to-95% vs supervised 2.6×; mem/time vs full BPTT; attention-mass probe R9); T6 ternary export via `lota_ternary` (`.bits`-servable day one). Triggers: 352's (a) non-code consumer / (c) Glacial-tier latent-plan consumer, **plus** (d) the GDN lane requesting latent thoughts.

Dead ends recorded (advocate, affirmed): ATC embedding-bypass on the frozen Bonsai GGUF serving artifact (engine surgery, not a training plan); latent thoughts inside GRPO/DPO (no token policy to score); full-BPTT at fleet scale (O(k×) memory, DOA); 8-digit addition as a product (tool call; bench-lane only). Honesty floor: parity/BFS/addition have O(n) modelless exact solvers — gates are method-vs-method within the trained class at matched FLOPs, exact solver compiled in as the acknowledged unbeatable baseline.

## 6. Fusion (paper × shipped, none alone)

1. **hint_regret × ladder FSM (the regulator gets a generator).** `hint_regret` triages content into learnable-hard/mastered/intractable via paired CRN rollouts but never moves a stage pointer; the `ladder_gate` FSM moves stages but needs a difficulty ordinal. Fusion: hint-regret's learnable-share band-gate is the accuracy probe the streak gate consumes; backtracking retreats along the hint-regret ordering. (katgpt-core; the cleanest Super-GOAT-shaped follow-up if it survives its gate.)
2. **Backtracking × the Issue-102 fade clock (riir-clippy).** When the ≥3-generation fade-predictive clock elapses, the corrective layer should be *shallowest-failing retreat + λ-mixed rehearsal*, not demotion — demotion retires a rule; retreat re-trains the family. Pre-registering the shape now (Issue 887 consumer note) costs nothing and honors the "nothing demotes automatically" doctrine.
3. **Dwell-dominance × every gate in the stack.** reflex `cal_min_obs`, `gate_calibration` `MIN_REPEAT_RUNS=2`, promotion gates: prefer longer consecutive confirmation over tighter thresholds — the paper prices that trade at +1.4pp even against the *stricter* threshold.
4. **Last-thought sufficiency × `select_highest_attn_keys` (gated).** Mass-ordered KV eviction at thought/loop boundaries for a *trained sequential-chain* latent model — consumer does not exist (every trained-latent attempt is a recorded null), prior art FlashLoop occupies the looped family. Blocked on Plan 352's ATC row firing + producing an artifact; the measurement half already ships.
5. **ATC × Plan 330's negative**: see §5.2 — the wrong-intermediate hypothesis.

## 7. Non-goals / caveats

- Theorem 1 is parity-specific under correlation-one; only the ordering (Q7) transfers, and magnitudes must be re-measured per model/task.
- All five constants (τ, m, λ, ρ, b) were measured on the paper's GD-trained tasks; they are **priors for defaults and GOAT bar shapes**, not defaults-by-fiat.
- The 51%-collapse number is a theorem-shaped warning: **windowing/eviction without retention re-verification is unsafe** — Issue 887's G1 carries both of the paper's own negative controls so the gate can red.
- Q3 (selling point) fails ⇒ no architectural guide, no pillar claim, no default-on promotion talk.

---

*Verdict path: three-track adversarial panel (No-GD, Model-based) + prior-art sweep + workspace grep, 2026-09-25. Claude verdict: **AGREE, round 1, zero revisions** — the `request_verdict` backend timed out 2× at 180s, so the §5 sub-agent reviewer path ran (same Summary + same `#Verdict:` contract; an equal verdict, never provisional). The reviewer independently verified the cousin signal-diffs in source (`hint_regret/gate.rs`, `frontier_report.rs`, `active_set.rs`), the highwater jumps (587→589 skipping allocated 588; 886→887), and the LOTUS row-6 precedent for the row-not-plan call. Full round record in the landing commit message.*
