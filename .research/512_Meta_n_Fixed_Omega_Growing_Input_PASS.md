# Research 512: Meta^n — Fixed Ω, Growing Input, Recursive Self-Improvement (PASS)

> **Source:** [Meta^n: Recursive Self-Improvement through Emergent Depth](https://arxiv.org/pdf/2608.24735) — Kim, Lee, Jwa, Kang (U Minnesota + SNU), arXiv:2608.24735v1, 25 Aug 2026. Code: github.com/minnesotanlp/meta-n
> **Date:** 2026-08-27
> **Status:** Done — closed.
> **Classification:** Public
> **Related Research:** 440 (AIDE² — the canonical code-level-RSI PASS precedent; Meta^n is the strongest empirical descendant in that family and BEATS the Gödel Agent / OpenEvolve / DGM class it diagnoses), 368 (AutoMem — the decision-structure vs LLM-dependent-process test, which Meta^n fails for Ω), 289 (RecursiveMAS — bi-level-already-shipped precedent), 320 (Red Queen Gödel Machine — ε-best-belief + consolidation-guard cousins; Meta^n's consolidation mode is the same zero-regression family), 171 (FrontierCS open-ended evolution), 244 (Self-Evolver faithfulness)
> **Related (riir-clippy):** Issue 049 (the ONE Gain-tier actionable — CLOSED NEGATIVE-no-gain 2026-08-27, [Bench 051](../../riir-clippy/.benchmarks/051_kat_growing_input_poc.md): contemporaneous A/B/G triple all 5/9 strict-keep at 2.8× arm-B tokens; the conditioning gain is regime-bound to Meta^n's cross-task scale — the composer ships as the reproducible artifact, reopen when the store has same-rule coverage), Proposal 005 (KAT heal service network), Issue 040 (fixer re-gate — arm-B pre-pass result corroborated below), Issue 030 (relevance gate — unchanged by this verdict)
> **Verdict: PASS.** Meta^n's core mechanism (one fixed LLM meta-operation Ω recursively writing code layers from its own growing output) is the R440/AIDE² class — **LLM-dependent code generation with no modelless substrate**. Every distillable sub-mechanism already ships in the quintet (verified by grep, §2.2). The one material delta found — the LLM improver's input contract is fixed-shape in our L4/KAT lane while Meta^n's measured ablation says growing input + inter-call conditioning carries ~72% of gain in exactly this loop shape — is filed as riir-clippy Issue 049 (a POC on the CF/general-LLM arm, NOT a primitive). No plan, no guide, no open primitive.

---

## TL;DR

Meta^n holds a single meta-operation Ω (one fixed LLM prompt template) constant and **recurses on its input instead of on the operation**: each application reads the execution traces of the whole solver stack below **plus the code stack that produced them**, then writes the next layer as (a) a strategic pre-process context string and (b) a library of callable helpers. Because Ω never changes it cannot destabilize the system (prior self-referential agents — Gödel Agent, DGM, HyperAgents — cap at ~2.5 meta-levels because they must freeze a driver); because the input strictly grows, each layer reasons from a higher vantage. Depth is set by convergence, an evolutionary archive searches layer chains, and a consolidation mode makes per-task bests monotone by construction. Across 8 benchmark families × 2 backbones it beats Gödel Agent and OpenEvolve everywhere; on ARC-AGI-2 it is the only system above zero (0.331 vs ≤0.054).

**Distilled for katgpt-rs (modelless, inference-time): nothing.** Ω's decision is "given traces + code stack, write the next layer's code" — code generation, the R440 §2.1 test's answer for AIDE² and the same here. No probe/draft/pruner, no freeze/thaw snapshot, no latent projection computes "write a better harness layer". This is the FluxMem/AIDE² class, NOT the AutoMem class. The load-bearing distinction stands: **Meta^n is code-level RSI (LLM); our stack ships latent-state RSI (modelless)** — different ladders, and the modelless-first mandate means we climb ours.

**One honest Gain-tier delta** (the reason this note is not pure classification): our stack DOES contain a live LLM-improver lane — riir-clippy's L4 fallback / KAT fixer — and there the paper's headline finding lands on a measured gap: our improver's prompt is **fixed-shape** (`TRAINED_INSTRUCTION + buggy_code`; the growing TrajStore archive never reaches the LLM), while Meta^n's ablation attributes **~72% of recursion gain to the conditioning channel** (context passed to the improver) vs ~15% to code transfer. Issue 040's arm-B pre-pass (richer caller-built prompt) beating arm A on every partial-credit axis is our own data point in the same direction. Filed as riir-clippy Issue 049 — a POC, gated by the existing scope/relevance gates, on the CF arm only (the trained Bonsai adapter arm keeps its minimal prompt — that is a distribution constraint, and it is correct).

---

## 1. Paper Core Findings

### 1.1 The three paradigms of meta-improvement (§2.1)

| Paradigm | Systems | Realized meta-depth | Why capped |
|---|---|---|---|
| Hand-crafted one-layer meta | FunSearch, AlphaEvolve, OpenEvolve, ADAS | 1 | The meta-process (search loop, mutation, selection) is frozen; the improver never improves |
| Self-referential one-layer | Gödel Agent, DGM, STOP, HyperAgents | ~2.5 | The agent edits its own source but must hold a driver fixed for stability — the 0.5 is partial modifiability above level 2 |
| **Recursive n-layer (Meta^n)** | this paper | 2–6, convergence-set | The improver is frozen **by design**; the recursion is applied to its growing input |

### 1.2 The meta-layer (§2.2)

Build-step: `Ω : {traces^(d-1)} ∪ [C2..C(d-1)] ∪ T ∪ d → C_d` where `C_d = (f_pre^(d), L^(d))` — a pre-process function (injects strategic context before each solver call) plus a library of callable helpers. Run-step: wrapper `M_d` slots `C_d` around `S_(d-1)`; contexts thread inward (`f_pre^(d-1)(t, ctx_d) → ctx_(d-1)`), the union library is prepended with deeper-layers-override-by-name; `S_d = M_d ∘ … ∘ M_2 ∘ S_1`. Ω's prompt template never changes between depths or benchmarks; only the trace summary's formatting adapts (a content adapter, not a second driver).

### 1.3 The decisive ablation (§3.4) — conditioning dominates

Removing (CO-Bench, Gemma): code-library injection −0.020; **inter-layer context (one string) −0.094**; full recursion −0.131. Decomposition: **~72% of recursion's gain is the conditioning channel** (the context string each layer passes down), ~15% the callable code, ~13% the search machinery. Reproduces on GPT-5.2 (+0.080) and AlphaEvolve Math (+0.158), growing as the base solver weakens.

### 1.4 Emergent roles + the depth-3 interference wall (§3.3)

Depth 2: generic cross-task primitives (22/36 CO-Bench winners call an Ω-emitted function; `simulated_annealing` propagates to 15/36). Depth 3: specialization + interference — 41% of (chain, task) pairs strictly regress, 18% substantially. Depth ≥4: rollback/correction — **rollback is exactly 0% at depth 2 under both LLM raters** ("the corrective role does not exist until a layer has something to correct"). Emergence measured over 596 emissions, two independent raters, mean Cohen's κ = 0.59.

### 1.5 Orchestration + the consolidation guard (§2.4, App G)

Evolutionary archive: parent sampling `w(c) ∝ S̄(c) + α/(1+children(c))` (UCB-shaped exploration bonus), temperature cycling, cross-candidate inspiration. Consolidation mode: each candidate targets ONE focus task while inheriting the archive's frozen best traces for all others — **per-task-best monotone by construction** (8/8 tasks zero-regression; +0.10 over a compute-matched best-of-4 control, 95% CI [+0.04, +0.16]).

### 1.6 Baseline diagnosis (§3.2) — why the frozen-driver systems lose

Gödel Agent at 10× budget plateaus at 0.628 vs Meta^n's 0.870: with a held-fixed driver "the model must CHOOSE to modify and usually does not, whereas Ω ALWAYS emits an injection — separating WHAT to change from WHETHER to attempt change." OpenEvolve's per-artifact loop cannot share a cross-task pattern ("avoid scipy") the way Ω's library injections do; Meta^n is also ~13× more sample-efficient (29 vs 378 candidate evaluations) by running one grouped search.

---

## 2. Distillation

### 2.1 The R440/R368 class test (the decisive check)

"What decision does each LLM call compute?" — **"write the next improvement layer (context + helpers) from traces and producer code"** = code generation. No modelless substrate. Same verdict cell as AIDE² (R440 §2.1): FluxMem/AIDE² class, not AutoMem class. §3.5's three paths all fail for the same reason they failed there (a frozen snapshot, a deterministic adapter, and a latent projection cannot synthesize code), and this is **not riir-train either** — Meta^n trains no weights (no GD, no optimizer/loss/schedule; the archive searches code chains). Pure LLM-orchestration layer, outside the quintet's mandate by design. The three-track panel was not spawned: the paper carries no optimizer/loss/RL framing and R440's precedent covers this exact class ("not a training paper… pure LLM-orchestration PASS"). An inline No-GD pass over the components (§2.2 table) found every extracted principle ships; the one partial (B1) is an input-contract refinement of our own LLM lane, filed as an issue.

### 2.2 Sub-mechanism mapping — verified by grep (not name-matched; formulas/sites read)

| Meta^n sub-mechanism | Shipped equivalent | Evidence |
|---|---|---|
| Inter-layer conditioning (context threading between cognitive layers) | `GameQualityGuide` emotion→λ_eff conditioning (`cgsp_runtime/guide.rs` L161-196; `runtime.rs` L306-314 "the very next tick() sees the new λ_eff"); KARC→salience→motivation ordering (`riir-games-civ/civ/map_tick/mod.rs` L949-953); `tick_motivation_attributed` threads the stat vector (`riir-games/motivation/attribution.rs` L324-337) | VERIFIED |
| Rollback tier emerges only with something to correct | `EvidenceTier::Withdrawn` is absorbing — 3 consecutive post-hoc failures enter, only a fresh full-validator pass exits (`riir-clippy/src/memory.rs` L43-98) | VERIFIED |
| Bounded composition of nested improvements | `MAX_FIX_PASSES = 4` fixpoint with overlap-selected splices; nested edits re-fire per pass (`riir-clippy/src/score_bench/mod.rs` L169-179; `bin/heal.rs`) | VERIFIED |
| Convergence-set depth (ϵ·R tolerance + patience) | `GainCostLoopHalter` (gain < cost×τ halt, oscillation patience, l_min floor) + `KnpcSelector` per-NPC planning horizon (`katgpt-core/src/gain_cost_halt.rs` L90+; `cgsp/loop_.rs` L690+). `RecursionLogits` trait ships in katgpt-core; the `AdvantageMarginGate` impl lives in the root crate | VERIFIED (component) |
| Conservative selection under thin evidence | `SelectionMode::BetaPosterior` (DEFAULT) — ε-quantile of Beta(1+S, 1+F), the exact R320 ε-best-belief family (`riir-clippy/src/self_evolve.rs` L126-156, L432-436) | VERIFIED |
| Consolidation guard (zero-regression by construction) | `consolidate_gated`: apply the compressed delta ONLY if the two-sided `can_freeze` gate passes; on failure the shard is **untouched**; `can_freeze_proactive` is a strict downgrade only (`riir-neuron-db/src/consolidation/mod.rs` L1480-1530, L2007+) | VERIFIED |
| Exploration bonus α/(1+children) | **Not shipped — and the nearest cousin was measured NEGATIVE**: Entropic (KL-budgeted Boltzmann tilt) lost to BetaPosterior on all four reward-source×mapping variants (riir-clippy Bench 035, starved-pool regime). No false coverage claimed | NOT SHIPPED (refuted cousin) |
| Archive per-task-best across chains | TrajStore is a growing trajectory log + ELO backfill, not a Pareto/per-task-best archive. Honest gap, but with no consumer demanding it (the 039 T5 gate measured ELO-based selection does NOT replace BetaPosterior at real pool sparsity) | NOT SHIPPED (no consumer) |

### 2.3 The one material delta (B1): the improver's input contract — filed as riir-clippy Issue 049

Meta^n's reframe — "**the gain comes from giving Ω more to read, not from rewriting Ω**" — plus the 72%-conditioning ablation lands directly on a measured gap in our one live LLM-improver lane:

- **Today:** `DaemonL4::generate_fix` sends exactly `TRAINED_INSTRUCTION + buggy_code` (`riir-clippy/src/l4_daemon.rs` L199-202). No trajectory context, no proposer source, no prior outcomes. The growing archive (TrajStore/LatentFixMemory, ~8K+ trajectories) feeds only the modelless selection path — **it never reaches the LLM**.
- **Nuance that keeps this honest:** the minimal prompt is CORRECT for the trained Bonsai adapter — all 1092 v2edit rows share one lint-agnostic instruction line, so enriching that prompt would go off-distribution (documented in-source at `l4_daemon.rs` L23-27). This issue is NOT about that arm.
- **The CF/general-LLM arm has no such constraint:** Issue 040's `request_prompt` already builds caller-rich prompts (instruction + module + modelless candidate + few-shots), and its arm-B **pre-pass** beat raw arm A on every partial-credit axis (v2edit slice 18 vs 12 keeps, parse 20/20 vs 15/20). That is our own single data point consistent with Meta^n's conditioning-dominance finding.
- **Disposition:** a POC — extend the CF-arm prompt with growing input (top-k prior trajectory outcomes for the rule/span, Certified-first; the modelless candidate(s) = "the code that produced them"; a cross-span failure-pattern summary = the paper's depth≥3 structured-summary analog), measure strict-keep vs Issue 040's arm A/B baselines on the same span set, keep the scope/relevance gates as guards. Filed as `riir-clippy/.issues/049_kat_growing_input_conditioning_poc.md`.

### 2.4 Latent vs raw boundary (mandatory check)

Not applicable — the mechanism lives at the LLM-orchestration layer. No latent-state op, no sync crossing, no raw/latent bridge. The 5-scalar sync rule and raw-position anti-cheat discipline are untouched.

### 2.5 Latent-space reframing + game-context check (mandatory per skill)

- **HLA/functor framing:** no per-NPC angle — Ω writes Python scaffolding, not direction vectors. Our latent-state RSI (`evolve_hla`, Raven/δ-Mem consolidation, MAPE-K, freeze/thaw personality divergence) is a different ladder.
- **CGSP framing:** CGSP updates latent state from curiosity, modellessly; Meta^n requires LLM code-gen. Fundamentally different substrates.
- **Game-context reframe (step 4):** "NPCs whose cognitive depth grows with emergent role differentiation (generic→specialist→corrective)" — the adaptive-depth half already ships as the per-NPC gain/cost reasoning-depth halter (R149 + `KnpcSelector`); the emergent-roles half needs an LLM at Ω per NPC, violating the modelless hot-path mandate. Crowd angle (archive-over-chains = per-NPC personality divergence across a population) already ships as freeze/thaw per-entity divergence. **No new capability on either reframe.**

### 2.6 §3.5 modelless-unblock check

Same three-path failure as R440 §2.6, for the same reason: code generation has no modelless substrate. Not a "needs training" case (no GD) — a "needs an LLM" case. riir-train is not the destination (it trains weights, not scaffolds). The one place an LLM improver is sanctioned in our stack is the L4/KAT last-resort lane, and there the paper's contribution is input-contract guidance (§2.3), not a mechanism.

### 2.7 §3.6 PoC requirement — not triggered for the PASS, and the quality-adjacent claim is filed as the POC

This note's PASS is **scope exclusion** (core thesis is LLM-dependent; no parity claim), mirroring R440 §2.7. The one quality-adjacent claim in play — "growing input + conditioning will improve the KAT CF arm's strict-keep" — is **NOT claimed proven here**; it is exactly what riir-clippy Issue 049's POC exists to measure, against Issue 040's recorded baselines. No "already ships at parity" claim is made anywhere in this note.

---

## 3. Verdict

**Tier: PASS** (+ one Gain-tier issue: riir-clippy 049).

| Gate | Criterion | Honest answer |
|---|---|---|
| **Q1** No prior art? | **FAIL.** Every distillable sub-mechanism ships in the quintet (§2.2, grep-verified with formulas/sites). Published prior art covers the components (Voyager skill-library = the callable-helpers half; evolutionary-context work = the conditioning half; DGM/AlphaEvolve = the archive). The composite is Meta^n's own contribution — to the LLM-agent ladder, not to ours. |
| **Q2** New behavior class? | **FAIL.** For the modelless runtime: no. Conditioning, adaptive depth, rollback tiers, gated consolidation all ship. The emergent-depth behavior class requires the LLM at Ω. |
| **Q3** Selling point? | **FAIL.** "Our NPCs recursively self-improve through meta-layers" would require per-NPC LLM calls — outside the modelless-first mandate. Our selling point remains latent-state RSI, which ships. |
| **Q4** Force multiplier? | **NO.** The one cross-lane connection (Meta^n input contract → KAT/L4) refines a single existing lane, already tracked by Proposal 005 + Issue 040. |

### One-line reasoning

Meta^n's Ω computes code generation — the R440/AIDE² class with no modelless substrate — and every extracted principle (inter-layer conditioning, absorbing rollback, bounded fixpoint composition, convergence halting, conservative selection, zero-regression consolidation gating) already ships grep-verified in the quintet; the single material delta is that our LLM improver's input is fixed-shape where Meta^n measured growing-input conditioning at ~72% of gain, which is filed as the riir-clippy 049 POC on the CF arm (the Bonsai adapter's minimal prompt stays — it is a distribution constraint, and correct).

---

## 4. Routing

- **Open primitive** → none.
- **Architectural guide** → none.
- **Plan** → none.
- **Issue** → `riir-clippy/.issues/049_kat_growing_input_conditioning_poc.md` (the KAT/CF-arm input-contract POC; references this note).
- **riir-train** → no. Not a training paper (no weights, no GD). R440's precedent: pure LLM-orchestration PASS is outside the quintet entirely.

---

## 5. Validation signals (confirmatory, not additive)

External evidence validating choices already shipped — recorded so future implementers grep this note before re-litigating:

| Meta^n finding | Validates | Where |
|---|---|---|
| Conditioning (context handed to the improver) carries ~72% of gain; callable code ~15% | Issue 040's arm-B pre-pass beating arm A on every partial-credit axis; the `request_prompt` caller-rich contract for the CF arm | riir-clippy Issue 040/Bench 049 |
| Frozen-driver agents lose because the model "must CHOOSE to modify and usually does not"; Ω ALWAYS emits, separating WHAT from WHETHER | The healer's proposers always emit; the fixpoint always runs (bounded), and the L4 reachability contract means the LLM is only consulted on declared misses — the WHAT/WETHER split is exactly modelless-first, LLM-last-resort | `l4_fallback.rs` reachability contract; `MAX_FIX_PASSES` fixpoint |
| Depth-3 interference: 41% of (chain, task) regressions; rollback role emerges at depth ≥3 | The absorbing `Withdrawn` tier (3-fail entry, re-certification-only exit) and `consolidate_gated`'s untouched-on-failure contract — rollback-as-tier, not rollback-as-prompt | `riir-clippy/src/memory.rs`; `riir-neuron-db/src/consolidation/mod.rs` |
| Convergence-set depth (ϵ·R + patience) rather than fixed depth | `GainCostLoopHalter` shape (gain-vs-cost threshold + oscillation patience + floor) for per-NPC planning depth | `katgpt-core/src/gain_cost_halt.rs` |
| Entropic/Boltzmann-style selection under starved pools | (Inverse-validation) our Bench 035 already measured the entropic tilt NEGATIVE in the starved regime; Meta^n's UCB-shaped bonus is untested here and stays untested until a consumer demands it — consistent with the 039 T5 report-only verdict for ELO | riir-clippy Bench 035, Bench 047 |

---

## Cross-references

- **Canonical class precedent:** `katgpt-rs/.research/440_AIDE2_Recursive_Self_Improvement_PASS.md` — Meta^n is the strongest published descendant evidence for that note's core distinction (code-level RSI vs latent-state RSI): it empirically beats the frozen-driver family (Gödel Agent, OpenEvolve) AIDE² was measured against.
- **Decision-structure test:** `katgpt-rs/.research/368_AutoMem_Metamemory_LLM_Orchestration_PASS.md` — the test Meta^n fails for Ω.
- **Consolidation-guard cousins:** `katgpt-rs/.research/320_Red_Queen_Godel_Machine_Selective_Erasure_Best_Belief.md` — ε-best-belief selection (ships as BetaPosterior) + criterion-versioned erasure family.
- **Bi-level precedent:** `katgpt-rs/.research/289_RecursiveMAS_Pass_Already_Shipped.md`.
- **The actionable half:** `riir-clippy/.issues/049_kat_growing_input_conditioning_poc.md` + Issue 040 (Bench 049) + Proposal 005 (KAT).

## Re-evaluation guard

This note exists to prevent re-running the full pre-flight on this paper or its descendants. Verdict is **PASS**; do not re-distill unless ALL of:

1. A descendant strips the LLM dependency from the improver (deterministic construction of improvement layers from accumulated traces) AND targets latents/shards/direction vectors — then it lands on our ladder.
2. riir-clippy Issue 049's POC measures that growing-input conditioning materially flips the CF arm's strict-keep — that upgrades the *finding's* weight for the KAT lane but still does not make Meta^n a primitive; the issue owns the follow-up.
3. A future consumer demands a per-task-best Pareto archive over fix chains (the one shipped-gap row in §2.2) — file that against the 039 T5 report-only verdict, not against this paper.

The honest one-sentence summary: **Meta^n wins by freezing the improver and feeding it everything — an architecture for LLM code-generation loops; our equivalent lever (feed the improver more) is real but lives in the one LLM lane we have, and it is a POC-sized issue, not a primitive.**

> **Update 2026-09-04 (Research 530 / arXiv:2609.02702):** the conditional in item 2 above RESOLVED **NEGATIVE** — riir-clippy Bench 051 ran the arm-G POC: strict-keep tied 5/9 across arms A/B/G at 2.8× arm-B tokens; verdict `NEGATIVE-no-gain`, Issue 049 closed (file removed per the noise-reduction rule). Trace-as-State recalibrates rather than overturns: the channels were cross-rule (weakly relevant — the paper's Random-Trace arm shows weakly-relevant conditioning content scores BELOW the no-trace first pass) and were placed condition-LAST (after the code — the paper's controlled variable, `[T,x,q]` vs `[x,T,q]`, 26/27 cells favor before). Reopen conditions sharpened: same-rule exemplars + condition-first placement — riir-clippy `.issues/067`.
