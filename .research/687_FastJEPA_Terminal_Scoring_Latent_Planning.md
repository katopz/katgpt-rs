# Research 687: Fast-JEPA — Terminal-Only Predictive Search for Latent World-Model Planning

> **Status:** Done — per-track verdicts locked (**PASS** modelless/runtime + decision family · **PASS** training track with `riir-train` R408 addendum — plan NOT filed on consumer/profit-axis grounds, not on R408's trigger (a), which is circular for the substrate plan). 
> **Claude verdict (2026-11-22, session `683573b4`, claude_code reviewer, 2 rounds):** R1 REVISE — text fixes only, all three per-track verdicts confirmed (reviewer independently re-verified the 572 removal commit, the R408 λ values, the CEM grep, the modelless fold_k reasoning, and the distill NO; decided the weak point: deferral correct but must rest on consumer/profit-axis grounds, not circular trigger (a)). Fixes applied same session; R2 **AGREE** (every fix confirmed in-file; residual R408 Status-line wording ambiguity fixed at commit).
> **Source:** *Fast-JEPA: Advancing the Success–Latency Frontier for Latent World Models* — Yao Zhang, Bin Yu, Shijie Lian, Haishan Liu, Changti Wu, Yu Wu, Kai Chen (Wuhan U + Zhongguancun Academy + HIT + HUST + NJU + ECNU + DeepCybo), [arXiv:2611.25927](https://arxiv.org/abs/2611.25927), 19 Nov 2026, cs.RO. Code repo: `DeepCybo-PhysAI/Fast-JEPA` (user-supplied; the paper README cites `Sofilyzia/Fast-JEPA` — same project, org account), cloned @ `7099e8ca8b6b3b7d95766a0504269c108438628a`, **Apache-2.0**.
> **Date:** 2026-11-22
> **Related Research:** **462 (INTACT — the paper's baseline; Fast-JEPA replicates INTACT's dual-call loss with IDENTICAL λ 0.10/0.05)** · **riir-train 408 (the INTACT training half — this paper completes its missing recipe specification)** · **riir-ai 349 (LeFlow — the complementary verify shape: amortized generation + recursive rollout verify; Fast-JEPA is the terminal-only counterpoint)** · 686 (content-geometry JEPA — same per-track verdict template, 2026-11-20) · 358 (SMWM) / 426 (Temporal Straightening) / 360 (AdaJEPA) — the JEPA-world-model PASS family · riir-ai 324 (GC-IDM — the paper's proposal-ablation arm) · 123 (Latent Functor Runtime — the modelless Direct-mode primitive)
> **Classification:** Public (this note)
> **Domain:** katgpt-rs (this note, public). No new runtime primitive; no plan filed (see §7 for why the training track stays at addendum level).

---

## TL;DR

Fast-JEPA advances the success–latency Pareto frontier of latent world-model planning by **splitting predictive computation by information need**: intermediate anchored multi-horizon predictions guide sequential action-chunk CONSTRUCTION (H generator steps + H−1 anchored queries), while complete candidates are scored by a **single batched terminal query** `J(A) = ‖Q(z_t, A, H) − z_g‖²` under the terminal-goal objective — never re-anchoring on predicted states. The enabling substrate is an **anchor-conditioned multi-horizon world model** `Q_ψ(z_t, A:k)` trained with the Eq-7 multi-horizon anchored loss (all horizons predicted from the FIRST observed representation, conditioned on the action prefix, future targets keeping gradients); the action model is **INTACT's dual-call verbatim** (shared diagonal-Gaussian conditional over next-state displacement `Δ^next = z_{k+1}−z_k` attached + endpoint displacement `Δ^end = sg(z_end)−z_k` stop-gradient, λ = 0.10/0.05 — identical to INTACT's λ_inv/λ_goal). Result (**Table 1**, 3-model mean): 96.50% avg SR at 10.88 ms complete planning latency, 2.41× faster than INTACT (26.19 ms) at higher success. The component study (**Table 4** reference config, seed-0 checkpoint): proposal-only ablation 93.83% @ 5.53 ms; full controller 96.33% @ 10.78 ms; **recursive-scoring variant 96.08% @ 20.54 ms — terminal scoring is +0.25 pp (a tie within noise, not a trade) AND 4.8× faster** (scoring 11.16→2.32 ms). Compute: O(NIHW) → O((H+NI)W). Plus **DynaGoal**, a benchmark for planning under shifted goals/geometry (73.83%), joint planning+execution deadlines (87.67%), and imposed response delays 0–800 ms with buffered-action semantics.

**Verdicts:**
- **Modelless/runtime (katgpt-rs / riir-ai / riir-chain / riir-neuron-db / decision family): PASS.** The H× saving is purchased by the LEARNED anchored multi-horizon predictor — modellessly, terminal prediction of a rule-induced world IS the recursive rollout (`fold_k` of `InducedCwmKernel::advance`), so the mechanism does not transfer to our rule-based stack. The planning-level asymmetry insight (intermediate-for-construction, terminal-for-evaluation) is real and quotable; its consumers here are all in the deferred research line (Motor-Gated DEC rehearsal / R349's unwired scorer), not a prod lane.
- **Training track (riir-train): PASS with R408 addendum — plan NOT filed.** The paper's Eq-7 anchored multi-horizon loss + no-EMA target-gradient retention + SiGReg 0.02 is exactly the JEPA-substrate recipe R408 named as its blocker (B1), and it INDEPENDENTLY VALIDATES R408's core recipe (dual-call λ byte-identical — caveat: R408 §1's "skip SIGReg on a rule-induced substrate" carve-out stands; SiGReg belongs to the trained-substrate path only). The substrate plan is NOT filed, and NOT because of R408's trigger (a) — "a JEPA encoder lands in riir-train" is the substrate plan's own output, so invoking it as the gate is circular. The actual grounds: (i) **profit axis** — new compute must profit seal-*, riir-rethink, riir-refine, or riir-infer; a robotics-benchmark JEPA substrate with no consumer profits none of them (LOW PRIORITY by the standing rule); (ii) **no consumer chain** — Motor-Gated DEC is deferred with no re-arm trigger and Issue 572 was removed (`f8d56cab3`, DEFERRED "unmeasurable on current task set"); (iii) **"implement to unblock" does not apply** — nothing live is blocked. The trigger that matters is **a consumer lane re-arming**; the addendum holds the plan's Phase-0 text for that day.
- **Source-repo distill (DeepCybo-PhysAI/Fast-JEPA): NO** — zero source files (README + figures + Apache-2.0 LICENSE only, repo created 2026-11-22). Nothing to mine in any landed domain; no class in kernel_opt/rust_perf/clippy/sec owns "latent world-model planning algorithms". No lane mapping (no league lane or open perf issue consumes planning-latency-for-robotic-planners). Conversation-only verdict, recorded here for completeness.

**Headline numbers, disambiguated:** 96.50% avg SR @ 10.88 ms is the paper's **Table 1** main comparison (mean over 3 trained models); **96.33% @ 10.78 ms** is the **Table 4** ablation reference configuration (the component study's own baseline row). Both are quoted below with their tables; the ~0.25 pp / 0.10 ms gap between them is model-averaging, not a contradiction.

---

## 1. Paper core findings

| # | Finding | Statement |
|---|---|---|
| 1 | Anchored outcome query | `Q_ψ(z_t, A:k) = W_ψ(z_t, A:k)`, k=1..H — every horizon predicted from the OBSERVED anchor + accumulated action prefix; "the predicted outcome ẑ_k is never substituted for z_t as the anchor of any subsequent world-model query" (§3.2) |
| 2 | Multi-horizon anchored loss (Eq 7) | `L_wm = E[1/(H·d_z) Σ_k ‖Q(z_0, A, k) − z_k‖²]`; future target embeddings RETAIN gradients (no EMA/stop-grad — single-epoch viability, SiGReg carries collapse prevention) |
| 3 | Construction/evaluation asymmetry | Sequential construction consumes intermediate anchored predictions (later actions unspecified → need context); complete candidates need ONLY the terminal under the terminal-goal objective (Eq 4) |
| 4 | Terminal-only CEM | N=64, I=2, K=8 elites; 2 slots reserved (proposal + best-so-far) so refinement cannot worsen predicted cost; final mean also scored; σ bounds [1e-4, 10], σ₀ = 0.25, no smoothing (α=1) |
| 5 | INTACT dual-call replicated | Eq 8–9: shared conditional Gaussian over `{next, end}` displacement conditions, sg on endpoint, λ_next=0.10 / λ_end=0.05 — INTACT's recipe unchanged; the paper's delta is the SUBSTRATE (anchored multi-horizon predictor) + the search reorganization, not the action learning |
| 6 | Compute law | `C_CEM-rollout = (NI+1)·H·W` → `C_Fast-JEPA = H·G + (H+NI)·W`; measured (Table 4 reference config): proposal 93.83% @ 5.53 ms (refinement optional, task-dependent), full 96.33% @ 10.78 ms, recursive scoring 96.08% @ 20.54 ms — **terminal scoring +0.25 pp (tie within noise) and 4.8× faster** (scoring 11.16→2.32 ms) |
| 7 | Training cost | 5.95 GPU-h for the 4-task suite (vs INTACT 6.67, LeWM 18.16, Fast-LeWM 188.05); ViT-Tiny/14 shared encoder, 192-d latents (NOT ℓ2-normalized), one epoch, batch 256, AdamW 5e-4, BF16 |
| 8 | DynaGoal | Shift (drifting/reversing target PushT; target-switch Reacher/Cube; moving-doorway TwoRoom) 73.83%; Deadline (joint planning+execution budget) 87.67%; Delay (0/300/500/800 ms imposed response, buffered-action semantics, env advances between observation and action) — planning latency directly controls control performance in evolving environments |

---

## 2. Path 0 decomposition — component → coverage + signal-diff → verdict

| Component | Coverage (analog ships?) | Signal-diff vs cousin (formula read) | Verdict |
|---|---|---|---|
| Anchored multi-horizon predictor `Q(z_t, A:k)` | NO production analog — `InducedCwmKernel::advance` is one-step rule-induced; the LLM lane's MTP/specdec (riir-infer 1179/1183, R486 Soft-MTP) is the token-domain twin (multi-token from one anchor) | Cwm: `advance(state, action)` recursive, no prefix-conditioned horizon query. MTP: predicts token k from anchor — same SHAPE (anchor + skip) but a trained head on a trained base, no game-side consumer; a modelless `fold_k` over the rule kernel IS the recursive rollout (no H× saving without a learned one-shot predictor) | training-only; modelless residue = none (honest) |
| Terminal-only candidate scoring | Decision serving scores candidates outcome-level with NO rollout (nothing to skip); the game tournament (`induced_cwm/tournament.rs`) MCTS uses rule-stepped rollouts with NO documented latency constraint; `RandomBestOfN` (Plan 359 default, motor_gated_rehearsal) scores N random action sequences — its scorer is the deferred R349 line | The score SHAPE `‖terminal − goal‖` is generic latent distance (dot/cosine family ships everywhere); the LOAD-BEARING half is the predictor that makes the terminal queryable in O(W) — trained | class published (see §3); no modelless primitive |
| Goal-directed proposal construction | **SHIPPED modellessly** — INTACT's deployment path covered by R462 §3.2 (`extract_functor` + `apply_functor_edge_into`, Super-GOAT R123); Fast-JEPA's G_φ differs only in consuming anchored intermediate predictions (a learned conditional) | Same dual-call core, λ identical; the proposal mechanism per se is the R462-covered Direct mode | covered (R462) |
| Proposal-centered CEM with retained slots | No CEM ships (grep: one hit in `riir-poc/adajepa_poc.rs`); `RandomBestOfN` + `Greedy1Step` are the rehearsal strategies; best-so-far/proposal retention is iCEM-class standard | Retention-of-elite trick is small scheduling mechanics; no consumer runs CEM on any prod lane | audited discard — no consumer; re-open if a CEM planner lane ever opens |
| DynaGoal deadline/delay protocol | Deadline-capped retries ship in the decision family (rethink ESC escalation: deadline declines pre-socket, readiness-retry caps, steer_retry pre-generation deadline check); tick tier model (50 ms hot) governs game-side budgets; `bench_preflight.sh` pins box state | DynaGoal's specific shape (success under IMPOSED delay with buffered-action semantics, stale-observation drift) is a benchmark-design parallel — a harness we have no plan for; not a primitive | doc-note only |
| INTACT dual-call training (Eq 8–9) | **R408** — deferred note, recipe table matches byte-for-byte (λ 0.10/0.05, sg on goal, shared operator) | IDENTICAL — the paper is an independent replication + substrate completion | **R408 addendum** (training track) |
| Eq-7 anchored multi-horizon loss + no-EMA targets + SiGReg 0.02 | R408's B1 blocker named this missing ("a JEPA substrate does not exist"); R686's Plan 507 arms SVReg/NextLat — a DIFFERENT latent-pretraining line (belief-state next-latent heads, SG targets) | NextLat is one-step next-latent with SG targets; Fast-JEPA's loss is multi-horizon action-conditioned with RETAINED target gradients — the recipe R408 wanted, not a NextLat refinement (no documented multi-horizon consumer) | **R408 addendum** (recipe completion), plan gated per R408 protocol |

---

## 3. Prior-art search record (§4 — the class-level kills)

1. **PPWM** [arXiv:2610.08627](https://arxiv.org/abs/2610.08627) (Oct 2026) — "predict a finite-horizon trajectory in parallel while retaining causal interaction... in a single forward pass". The anchored multi-horizon SUBSTRATE class is published; Fast-JEPA cites it as a Table-1 competitor. Substrate ≠ novel.
2. **WoTE** (ICCV 2025, BEV world model, `liyingyanUCAS/WoTE`) + **[arXiv:2611.05072](https://arxiv.org/abs/2611.05072)** (3 Nov 2026, "Trajectory Evaluation via **Rollout-Free** World Model for End-to-end Autonomous Driving") — candidate-trajectory scoring without recursive rollout via a world model is PUBLISHED in the driving domain 16 days before Fast-JEPA. The rollout-free-evaluation CLASS is taken; Fast-JEPA's increment is the robotics/latent-control instantiation + proposal-guided refinement + the asymmetry framing + DynaGoal.
3. **VLWM** [arXiv:2606.21775](https://arxiv.org/abs/2606.21775) (Jun 2026) — variable-length latent world models for long-horizon prediction/planning; multi-horizon prediction class again.
4. **INTACT / GC-IDM / LeFlow / Fast-LeWM / POPLIN / LEAP / iCEM** — the proposal-guidance line is the paper's own related-work table; every guidance component is prior art (INTACT Guarded A centers search on the Direct plan; POPLIN uses learned priors to guide planning; LEAP refines goal-conditioned proposals; iCEM retains elites).
5. Internal-first grep (arXiv ID + title verbatim across every workspace `.research/`/`.plans/`): **zero hits**. Closest internal family: R462 (INTACT), R349 (LeFlow), R686 (content-geometry JEPA, same-week), R358/R426/R360 (the JEPA PASS family), R123/Plan 040 (Direct-mode primitives).

**Pinned claim (checked against the searches):** "Asymmetric predictive computation — intermediate anchored multi-horizon predictions for construction, single batched terminal queries for candidate evaluation — advances the success-latency frontier of latent world-model planning." The claim is a COMPOSITION increment on published substrate classes (PPWM multi-horizon + WoTE/2611.05072 rollout-free evaluation + INTACT dual-call/Guarded-A guidance). Solid, useful, not a new behavior class.

---

## 4. Novelty gate (§1.5) — all four NO

| Q | Answer | Evidence |
|---|---|---|
| 1. No prior art? | NO | §3: PPWM (multi-horizon substrate), WoTE + 2611.05072 (rollout-free evaluation class), INTACT/GC-IDM (dual-call + proposal guidance, both already distilled R462/R324), VLWM |
| 2. New behavior class? | NO | Rollout-free candidate scoring is published (driving lane); terminal-prediction-instead-of-rollout is the AlphaGo value-network move at sequence level |
| 3. Product selling point? | NO for our stack | "NPCs plan 2.4× faster" presumes a trained latent world-model planner in prod — none exists; game planning is FSM + rule laws; CGSP/MCTS rollouts are rule-stepped with no documented latency problem |
| 4. Force multiplier (≥2 pillars)? | WEAK | Touches the functor/Induced-CWM/rehearsal cluster — all in the deferred research line, nothing prod |

**Verdict: PASS for modelless/runtime.** Same canonical class as R358/R426/R360/R462 (+R686's modelless half). Not Super-GOAT, not GOAT, not Gain.

---

## 5. Value extraction (§1.55 — what the paper changes for us)

1. **R408's missing recipe is now SPECIFIED** (the one actionable file). R408's B1 blocker said "a JEPA substrate does not exist in riir-train"; Fast-JEPA supplies the complete substrate recipe: Eq-7 anchored multi-horizon loss (all horizons from one anchor, action-prefix conditioning, targets keep gradients), SiGReg 0.02 as the sole anti-collapse (no EMA asymmetry), ViT-Tiny/192-d latents unnormalized, one epoch @ 5.95 GPU-h, plus INDEPENDENT VALIDATION of R408's dual-call table (λ identical). Filed as the R408 addendum this session.
2. **Terminal scoring substitutes LeFlow's verify leg** (fusion observation, recorded in R349's cross-ref): LeFlow = amortized flow generation + recursive frozen-model rollout reranking (its ablation: reranking essential, +10.5 Reacher). Fast-JEPA shows the SAME candidate-comparison job can run on terminal-only anchored queries — if the R349/763 imagine-prune→verify composition ever lands, its verify leg has a cheaper published option (anchored terminal query) — at the cost of needing the multi-horizon predictor instead of the one-step one. Recorded, not filed.
3. **DynaGoal protocol shape** (doc-note): success-under-imposed-delay with buffered-action semantics + joint planning/execution deadlines is a clean evaluation pattern for ANY latency-bounded decision system (the ESC escalation lane's deadline handling and the tick tier model are our analogs); parked as harness-design reference, no consumer.
4. **Independent replication of INTACT's dual-call with identical λ** strengthens R408's recipe table from single-source to two-source — worth its line in the addendum. (Caveat carried: R408 §1's "skip SIGReg on a rule-induced substrate" carve-out stands — SiGReg 0.02 belongs to the trained-substrate path only, never the modelless path.)

---

## 6. Adversarial panel record (§3.5 — two advocates, one round, same batch as the §4 searches)

**No-GD advocate** returned 8 extraction candidates, flagging `W_ψ`/`G_φ` as GD-native. Coordinator dispositions:
- "Terminal-only scoring law" + "anchored multi-horizon composition" + "compute-bound batching law" — the H× saving requires the LEARNED one-shot predictor; modellessly, `fold_k` over the rule-induced kernel IS the recursive rollout. The score/distance SHAPE ships (generic latent distance), the saving does not. → covered/no-primitive rows in §2.
- "CEM-with-protected-slots schedule" — bounded scheduling mechanics, zero consumers (grep: no CEM in prod). → audited discard.
- "Anti-recursion anchor invariant" (bounded drift via sigmoid-gated displacement magnitudes) — a real modelless invariant, but no consumer queries multi-step latent rollouts on our stack (the two-brain model keeps tactical movement RAW; latent rollouts live only in the rehearsal line). → audited discard, re-open with the first multi-step latent-rollout consumer.
- "Deadline-buffered action semantics" + "DynaGoal harness" → doc-note (§5.3).
**Model-based advocate** verified in-repo that Eq-9 = R408's recipe table byte-for-byte (with the §1 caveat that SIGReg is skipped on a rule-induced substrate) and argued for filing the riir-train substrate plan NOW. Coordinator disposition: the plan is NOT filed — and the reviewer-confirmed grounds are the consumer/profit-axis ones, NOT R408's trigger (a) ("a JEPA encoder lands in riir-train" is the substrate plan's own output; gating the plan on it is circular): (i) **profit axis** — new compute must profit seal-*, riir-rethink, riir-refine, or riir-infer; a robotics-benchmark JEPA substrate with no consumer profits none of them (LOW PRIORITY by the standing rule); (ii) **no consumer chain** — Motor-Gated DEC is deferred with no re-arm trigger, Issue 572 removed (`f8d56cab3`, DEFERRED "unmeasurable on current task set"); (iii) **"implement to unblock" does not apply** — nothing live is blocked. The re-arm trigger to name is **a consumer lane re-arming**, not substrate landing. Advocate code facts spot-checked by coordinator (R408 blocker text, 572 removal commit) — verified; the reviewer independently re-verified all three.

---

## 7. MOAT gate + per-track verdicts

| Repo | In-scope? | MOAT contribution | Decision |
|---|---|---|---|
| katgpt-rs (public) | Marginal | None — Direct-mode primitives already shipped (R123/Plan 040); the paper's substrate is trained | This note only |
| riir-ai (private runtime) | Marginal | The consumer cluster (Motor-Gated DEC rehearsal / R349 scorer) is the DEFERRED research line; terminal-scoring refinement recorded in R349's cross-ref, no prod lane | No guide |
| riir-chain / riir-neuron-db | Out of scope | Latent planning never crosses sync | — |
| riir-train (private) | In-scope | Recipe completion for the deferred JEPA-substrate line + independent λ validation | **R408 addendum** |
| Decision family (reflex/instinct/rethink) | Marginal | DynaGoal deadline/delay = benchmark-design parallel; ESC deadline handling already ships | Doc-note in this note |

**Per-track:** modelless/runtime **PASS** · training **PASS** (R408 addendum; substrate plan NOT filed on consumer/profit-axis grounds — trigger (a) is circular for the substrate plan; re-arm = a consumer lane re-arming) · repo distill **NO** (zero code). §3.6 PoC: not required — no quality-parity claim is made (architectural-coverage PASS + substrate mismatch, the R462 §10 posture verbatim).

---

## 8. Source-repo distill verdict (distill skill — recorded for completeness)

`DeepCybo-PhysAI/Fast-JEPA` @ `7099e8ca8b6b3b7d95766a0504269c108438628a` (Apache-2.0, repo created 2026-11-22): **README + `assets/` figures/badges + LICENSE only — ZERO source files.** Per the transcription law, zero `.rs` is never a kill by itself — the kill here is that the source carries no code at all, and the paper's "classes" (latent world-model planning algorithms) have no Rust-shaped bounded fix-space in ANY landed domain (kernel_opt / rust_perf / clippy_lints / sec). No lane mapping (the league lanes are LLM serving; no open issue consumes robotic-planning latency). **Verdict: NO — conversation-only** (this section + the note header are the record; no queue line, no riir-refine artifacts). If the authors push code, re-run the distill verdict on the actual tree.

---

## 9. Files + PASS-Redirects

- **Created:** `katgpt-rs/.research/687_FastJEPA_Terminal_Scoring_Latent_Planning.md` (this note; `.research/.highwater` 686→687).
- **Edited:** `riir-train/.research/408_intact_shared_operator_dual_call_auxiliary.md` (addendum: recipe completion + independent λ validation + the one-trigger-away plan posture) · `katgpt-rs/.research/462_INTACT_...md` (PASS-Redirects line) · `riir-ai/.research/349_LeFlow_...md` (cross-ref line: terminal-scoring verify substitution).
- `.raw/fastjepa` clone removed after pinning (cleanup-at-completion).

> **PASS-Redirects (synthesis):** Zhang et al. [arXiv:2611.25927 "Fast-JEPA: Advancing the Success–Latency Frontier for Latent World Models"] — third-party INTACT-lineage follow-up: anchored multi-horizon substrate + terminal-only candidate scoring (O(NIHW)→O((H+NI)W), 2.41×) + DynaGoal deadline/delay benchmark; PASS modelless (substrate is trained; rollout-free-evaluation class published 2611.05072/WoTE), training half completes deferred R408 (dual-call λ replicated identically), no new primitive beyond the R462-covered Direct mode.
