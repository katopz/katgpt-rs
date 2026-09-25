# Research 590: GRAM — Generative Recursive Reasoning (re-distilled at the belief-host altitude)

**Status:** ACTIVE — **SUPERSEDES [Research 058](058_GRAM_Generative_Recursive_Reasoning.md)** (same paper, arXiv:2605.19376, first distilled 2026-07; this note re-distills at the per-NPC belief-host altitude 058 never examined and overturns two of its standing calls — see §Relation to 058). Modelless track **Gain** (katgpt-rs Issue 895 + riir-ai Issue 1008) · training track **Gain, low priority** (riir-train Plan 419, envelope-fit SECONDARY; absorbs 058 §7.3).

**Source:** arXiv:2605.19376 "Generative Recursive Reasoning" — Baek, Jo, Kim, Ren, Bengio, Ahn (KAIST / Mila / NYU / UdeM, 2026). Website: <https://ahn-ml.github.io/gram-website/>. v2 PDF read (ELBO Eq. 13/14, LPRM A.2, ACT A.1, ablation Tables 1/3, cost Table 7). 058 (written 2026-07, independently from the paper) is a second source: its figures (93.96, 99.7/50.27, 99.05) agree with the v2 extraction — headline reconciliation: 058 leads with the **base** 93.96 Sudoku number, this note leads with the **width-scaled** 97.0 (N=20) number; both appear in both papers' tables, no disagreement.

## TL;DR

GRAM makes recursive latent reasoning **stochastic and wide**: each high-level transition samples `h_t = u_t + ε_t` (deterministic proposal + learnable guidance noise), trains the noise with amortized variational inference, and selects among N parallel latent trajectories with a Latent Process Reward Model reading the latent state pre-decode. Measured: N=20×16 iters beats deterministic 1×320 (97.0 vs 90.5 Sudoku); deterministic recursion **mode-collapses** on multi-solution tasks (≤36.1% coverage vs 90.3%).

**The ablation is DOMAIN-DEPENDENT, not "guided wins"** (058 §7.1 quotes the same pair): zero-mean 94.88 > guided 93.96 on Sudoku (single-solution — guidance *hurts*), but guided 99.7 vs zero-mean 50.27 on N-Queens (multi-solution — guidance *essential*). Any gate we build must run BOTH families.

**Workspace verdict:** the headline (stochastic latent width scaling on DDTree) already shipped via PTRM → Plan 083 and was adjudicated **STRONG VALIDATION, MINIMAL ACTION** by 058 — that verdict STANDS for the logit-domain DDTree lane. What 058 never examined is the **per-NPC belief host** (`evolve_belief` — a deterministic single [f32;8] trajectory; `deliberation_cadence` deepens, `deliberation_trap`/`deliberation_budget` trigger and cap but never WIDEN). On that host the surviving deltas are: (1) **mass-conserving perturbation** — ε in the coexact∪harmonic Hodge subspace so `belief_mass_divergence(ε) ≡ 0` by construction (architecturally stronger than GRAM's own noise, which carries no such invariant); (2) **decode-free latent selection** for a host that has no reward signal (arenas have rewards → BanditPruner; fog-of-war belief deliberation has none → self-consistency); (3) **reallocate-to-width** (kill trapped branches, spend the budget on parallel hypotheses). The μ≠0 *idea* itself is prior internal art — 058 §7.1 proposed it (`SdeConfig.guided`, LOW) for the DDTree lane; Issue 895 supersedes that narrow form (reasons in §Relation). Not Super-GOAT.

## Mechanism (precise — cross-checked 058 ↔ v2 PDF)

- **Transition:** after K low-level refinements `f_L` (K=6 Sudoku, K=4 else), high-level `f_H` emits deterministic proposal `u_t`; guidance `ε_t ~ N(μ_θ(u_t), σ_θ²(u_t)I)` added: `h_t = u_t + ε_t`. Mean = state-dependent direction; variance = exploration budget. Noise at the high level only.
- **Training (amortized VI):** posterior `q_φ(ε|u,y)` target-conditioned (hindsight), prior `p_θ(ε|u)` runs at inference. Per supervision step n: `L^(n) = E_qφ[CE(dec(h), y)] − β·KL(q_φ‖p_θ)`, stop-grad through `h^(n)_{T−1}` (truncated BPTT → constant memory). Nsup=16. KL-balance 0.8; β 0.04–0.5; AdamW 1e-4/wd 1.0/clip 1.0; batch 768; EMA 0.9999. Models 7–27M params. Trained on 8× RTX 4090.
- **LPRM:** Linear(D→1) on h's first token; `Σ_t (v_ψ(z_t) − r)²`, r = final accuracy; joint training; best-of-N selection at inference.
- **ACT halting (A.1):** Linear(D→2) Q-head, TD targets, halt when `σ(q_halt) > 0.5`.
- **Ablations (Table 3):** −stochasticity → 0.0%; −guidance-mean → multi-solution collapse (N-Queens 50.27) while Sudoku survives (94.88); naive stochasticity does not help. SG alone lifts Looped TF 61.25→65.64; +DS → 73.90; +hierarchy → 93.96.
- **Data policy (D.2):** augmentation × sampling complementary — Aug=50 saturates sampling; Aug=0 scales monotonically.
- **Unconditional generation:** empty conditioning → p(x); 99.05% valid Sudoku boards (10.9M/16 steps vs D3PM 55.1M/1000). (058 §6.6 already declined this for games — standing.)
- **Paper's own limitation:** sequential deep supervision is a barrier to foundation-model scaling.

## Published + internal prior art (§4)

**Internal (the controlling constraints — found at round 2 of the verdict ping-pong; the round-1 sweep violated the skill's own pagination rule and missed 058):**
1. **Research 058** — the prior distillation of THIS paper (2026-07): verdict STRONG VALIDATION / MINIMAL ACTION for the DDTree lane; §7.1 learned-mean option (LOW, `SdeConfig.guided`, no new flag); §7.2 → Plan 095; §7.3 KL balance (carried → Plan 419); §8.3 no-new-flags / no-guided-default.
2. **PTRM arXiv:2605.19943** (058's Research 49; Plan 083) — zero-mean stochastic width scaling shipped (`width_rollouts`, `EarlyStopGate`, `TrajectoryCredit`, `inject_sde_noise`, `DDTreeBranchCache.max_branches`).
3. **Research 374 P1/P2/P3** — `deliberation_trap` (FlipDetector over COMPASS_8 heading = the answer-level stuck trigger, swarm deliberation), `deliberation_budget` (think-budget caps + difficulty-scaled cooldown), `cgsp_trap_mode` (FlipDetector over MCTS leading branch); source katgpt-rs Research 546 / Plan 593 / Bench 707 GOAT / arXiv:2609.04963.
4. **Research 250** (arXiv:2511.16886 TRM policy improvement) — the deterministic-RRM theory lane, `self_advantage_gate` default-on.

**Published:** GRAM (cited-by-6); PTRM; Periodic-Table-of-LLM-Reasoning survey; repeated-sampling parallel-TTS work. Class prior art (knowledge-cited): Cobbe 2021 verifiers; Wang 2022 self-consistency; Brown 2024 repeated sampling; Lightman 2023 / Math-Shepherd PRMs; POMCP (Silver & Veness 2010); SVGD. ⚠ Search-limit: the LPRM-specific query returned zero relevant hits; the latent-state-PRM delta is asserted against the class, not a dedicated survey.

## Workspace coverage — signal-diff table (§3.6)

| Shipped cousin | What it consumes | GRAM component | Verdict |
|---|---|---|---|
| Research 058 (the prior GRAM verdict) | DDTree/logit lane adjudication | everything | **SUPERSEDED where scoped** (see §Relation) — its DDTree-lane calls stand |
| PTRM Plan 083 `width_rollouts`/`EarlyStopGate`/`TrajectoryCredit`/`DDTreeBranchCache` | external relevance / scores-as-input, on token trees | width law | **covered** on the DDTree host; **Plan 095 GOAT PENDING 1/3** (G1/G3 await a stochastic game domain — Issues 895/1008 gates complete it, not duplicate it) |
| `BanditPruner` UCB1 (058 §5.3 claimed as LPRM coverage) | **rollout rewards** (domain signal) | LPRM | covered WHERE a reward exists (arenas); **gap on the belief host** — fog-of-war deliberation has no ground-truth reward; self-consistency/latent scoring is the only signal |
| `SdeConfig`/elf_sde | logit magnitudes; zero-mean | guidance direction | **gap** (058 §7.1 named it; Issue 895 supersedes the narrow form) |
| `saddle_escape` (`FlipDetector` + `apply_kick`, quality-gated PoC) + `deliberation_trap`/`deliberation_budget`/`cgsp_trap_mode` | **decoded answer keys** (heading bucket, leading branch id); flip-rate EMA | trap detect + escape | **mostly covered** — trigger + kick + budget caps ship; gaps: belief-state observables (pre-decode), codifferential-divergence escape signal, reallocate-to-WIDTH (budget caps only shorten/cooldown) |
| `katgpt-dec` `hodge_decompose` | flow decomposition; never composed with noise | mass-conserving ε | **gap** (novel composition) |
| `katgpt-canon` `fit_joint_svd_pair` | offline joint SVD | direction table | substrate ships; no outcome-weighted/Beta consumer |
| `katgpt-sense` `evolve_belief` | deterministic single [f32;8] trajectory | N parallel belief trajectories | **gap** (the host) |
| `diversity/temp.rs` | BLAKE3-seeded deterministic zero-mean noise | ε source | covered (reuse) |
| riir-reflex `--horizon-report` | offline ridge over fixseq features | LPRM-analog | covered offline (healer lane); live consumer blocked on Bench-085 reopen trigger — recorded, not re-litigated |
| riir-clippy nonergodic/revive | per-run strategy posterior | multi-trajectory restarts | measured NULL — do not re-litigate without organic ≥2-revert rings |
| `gain_cost_halt::GainCostLoopHalter` (058 §5.6 ACT-claim) | gain/cost loop budget | ACT halting | covered as halter; Plan 419 R8 keeps only the TD Q-head form as a diff |

## Pinned claim (novelty sentence — revised after round 2)

> Mass-conserving guided width rollouts on recursive latent **belief** refinement states — perturbation drawn in the coexact∪harmonic Hodge subspace (`belief_mass_divergence(ε) ≡ 0` by construction) with a success-SVD direction mean updated by Beta counting, **decode-free latent self-consistency selection** (the only signal available where no reward exists), and **reallocate-to-width** on trap detection — for per-NPC multi-hypothesis deliberation. The μ≠0 guidance *idea* is prior internal art (058 §7.1); what is ours is the mass-conserving subspace, the reward-free selection on the belief host, and the width-reallocation of the shipped trap/budget machinery. Distinguished from PTRM/elf_sde (zero-mean, logit/token-tree domain), from POMCP particle belief (world-state particles + generative model vs refinement-trajectory restarts + DEC invariants), and from PRMs (token/step domain, pre-decode).

## Path 0 inventory (§3.5 — dispositions revised after round 2)

| Row | Component | Disposition |
|---|---|---|
| E1 | width-beats-depth law | **Plan 095 PENDING 1/3 — consume, do not duplicate**: Issues 895 T7 / 1008 T5 land G1/G3 on the belief/stochastic-domain host 095 awaited; cite 095 in every gate |
| E2 | structured ε (transversal / Hodge-subspace / σ-sigmoid stagnation gate) | **Issue 895** — consumes `diversity/temp.rs` as source; supersedes 058 §7.1's `SdeConfig.guided` form (belief-host + cochain arms are outside 058's adjudicated logit lane) |
| E3 | decode-free latent selection | **Issue 895** — scoped to the NO-REWARD belief host; where rewards exist, BanditPruner stands (058 §5.3); healer lane covered by horizon_report offline (live consumer = Bench-085 trigger) |
| E4 | deterministic diversity init + farthest-point set | Issue 895 task |
| E5 | trap detector + kill-and-reallocate | **consume** `saddle_escape` FlipDetector/apply_kick + `deliberation_trap`/`deliberation_budget` (374 P1/P2); the new claims are ONLY: belief-state observables, codifferential-divergence escape signal, reallocate-to-width |
| E6 | residual-terminated depth | Issue 1008 task (`k_selector` consumer; EarlyStopGate stands for token trees) |
| E7 | unconditional generation via empty conditioning | **audited discard** (058 §6.6 declined it for games first — standing; frontier-miner argument also holds) |
| E8 | success-SVD direction table + Beta posterior | **Issue 895** (the μ≠0 fill on the belief host; substrate `fit_joint_svd_pair`) |
| E9 | ablation profile as gate design | methodology — now explicitly BOTH-domain (see B4 fix): the winning arm's family AND the losing arm's family (zero-mean wins Sudoku-class) must both run, with the demote condition pre-stated |
| R1–R8 | training recipes | riir-train Plan 419 (+ absorbs 058 §7.3 KL-balance row; R3/R8 carry shipped-coverage diffs per 058 §5.6/§5.7) |

## Relation to Research 058 (the supersede, made explicit)

- **Stands:** the DDTree/logit-lane verdict (STRONG VALIDATION — elf_sde+bandit+bt_rank cover that lane); §6 don't-need list for that lane (hierarchy, ELBO, LPRM, architecture, reparam, unconditional-gen); §8.3's **"do NOT make guided noise the default"** — Issue 895 stays opt-in/default-off.
- **Overturned, with reasons:** (a) §7.1's `SdeConfig.guided`-only LOW form — 058 examined only the logit domain, where GRAM's own ablation shows guidance LOSING (94.88 vs 93.96); the belief-host + cochain arms argue a different surface with a different ablation profile, and a field-extension cannot carry cross-domain construction + selection + allocation; (b) §8.3's "no new feature flags" — same scope argument: 058's call was about the DDTree lane's covered surface; Issue 895's operators live on a host 058 never routed to. The overturn is scoped: no existing flag's posture changes.
- **Carried:** §7.2 → Plan 095 (PENDING — this note's issues complete it); §7.3 KL balance → Plan 419 R9.
- **Closed out in 058's action table by this note:** §7.1 (superseded → 895), §7.2 (consumed → 095 completion via 895/1008), §7.3 (carried → Plan 419).

## Fusion (§1 step 6)

- **GRAM × PTRM-substrate × katgpt-dec**: the mass-conserving width arm — architecturally stronger than GRAM's own noise. DEC-legal regime: cochain belief fields on 2D zone maps (d=2 ≤ 3); the [f32;8] belief takes the transversal arm (no Hodge claim at d=8, curse-of-dim rule).
- **GRAM × deliberation_trap/deliberation_budget × evolve_belief**: the shipped trigger + budget caps gain a WIDEN gear — budget reallocated to parallel hypotheses instead of only cooldown/shortened depth. Behavior: hesitation, second-guessing under fog-of-war, crowd disagreement. mb_personality is static diversity; this is per-decision dynamic diversity.
- **GRAM × horizon_report (healer)**: offline readout already a modelless LPRM; live consumer stays behind the Bench-085 reopen trigger.

## GOAT gates (Issues 895/1008 carry the detail)

- G1 equal-compute width-vs-depth **on BOTH families**: a multi-solution/structured family (where GRAM's guidance wins) AND a single-solution/uniform family (where GRAM's zero-mean wins, 94.88 vs 93.96) — pre-stated demote condition: if the direction table loses or ties on BOTH families, the guidance stays off-by-default forever (closed-negative, recorded). Mass-divergence ≡ 0 arm for the Hodge arm (exact identity). This gate IS Plan 095's pending G1/G3 completion — cite 095.
- G2 O(K) parallel latency, per-NPC budget in the 20 Hz slice; G3 kill-switch bit-identity (`σ=0`/`N=1` ≡ incumbent); G4 zero-alloc, BLAKE3-seeded deterministic noise.
- Promotion only on modelless gain; `guided_width_rollouts` opt-in at every layer.

## Caveats

- Mass-conservation benefit is UNMEASURED — architectural invariant only; no evidence (GRAM's or ours) that failure modes involve mass drift. §3.6: never read as quality parity.
- The belief-host lane presumes `evolve_belief` mode-collapses under ambiguity (GRAM's evidence is THEIR RRMs); Issue 1008 T7's fixture must SHOW the collapse signature in the deterministic arm or the fixture is wrong — and if no live failure exists on our substrate, the lane is inert and should close rather than gold-plate.
- Q1 fails (PTRM/058/374- family) → Gain, not Super-GOAT; claim is composition-level.
- Plan 419's first rung sits behind the Bench-085 reopen trigger — may stay unexecuted (consistent with lowest priority).
- riir-train scale honesty: paper's own limitation; nothing above 27M params claimed.

## PASS-Redirects

n/a (Gain verdict, both tracks). 058 carries the supersede banner; PTRM's Plan 083/095 chain cited throughout; note 250 carries a follow-up line pointing here.
