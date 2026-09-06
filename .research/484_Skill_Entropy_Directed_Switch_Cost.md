# Research 484: Skill Entropy — Directed Switch-Cost Table for Cross-Mode Sequences

> **Source:** [Toward Skill-Native LLMs: Skill Entropy for Benchmarking and Training Long-Horizon Reasoning](https://arxiv.org/pdf/2608.05139) — He*, Yang* et al. (Princeton / CMU / Toronto / UIUC / Stanford / Oxford), 2026-08-06
> **Code:** [github.com/Gen-Verse/Skill-Entropy-RL](https://github.com/Gen-Verse/Skill-Entropy-RL)
> **Date:** 2026-08-16
> **Status:** CLOSED (primitive shipped) — the Issue 663 modelless primitive landed 2026-08-17 as `katgpt-core` feature `switch_cost` (opt-in) in commit `17909bb8`: `SwitchCostTable` + `FactorizedSwitchCost` (Eq. 7) + `cdf_rank`, GOAT G1/G2/G3/G4 ALL PASS per [Bench 660](../.benchmarks/660_switch_cost_table_goat.md) (3.08 ns lookup, 0 allocs, factorized ranking Spearman ≥ 0.75 + identical argmax). Stays opt-in until a riir-ai consumer A/Bs F1. The healer-consumer row below is measured-dead (Bench 032); training-recipe half (riir-train Plan 319 Gap 6) remains open.
> **Original status:** Active — Gain verdict. Modelless primitive filed as Issue 663 (katgpt-rs); training-recipe half filed as riir-train Plan 319 Gap 6.
> **Related Research:** 241 (SwiR — reactive within-step mode switching; signal-diff below), 211 (Bayesian-Agent skill lifecycle), 172 (MUSE skill memory), 381 (SkillAdaptor step-level fault attribution), 149-equivalent riir-ai `.research/149_Per_NPC_Gain_Cost_Reasoning_Depth_Guide.md` (think-budget consumer), riir-ai `.research/126` (CGSP curiosity), riir-ai `.research/123` (latent functor runtime)
> **Classification:** Public primitive (the math) + private game consumers (riir-ai) + private training recipe (riir-train)
> **Novelty gate (§1.5):** Q1 prior art ✅ (none in-stack — zero grep hits for `switch_cost|transition_cost|switch_difficulty|skill_entropy`; nothing published predates the term), Q2 new behavior class ✅ (proactive transition-hardness-aware cognition — all current mitigations are reactive), Q3 selling point ⚠️ borderline (quest difficulty calibration is a product feature; NPC robustness gain is internal quality), Q4 force multiplier ✅ (latent_functor + cce_runtime + quest_grammar + CGSP/LEO + riir-train GRPO). **3/4 confident → Gain, not Super-GOAT.** No "candidate" phrasing: the primitive ships as Issue 663; the guide-tier doc is deferred until Q3 hardens (a measured quest-difficulty consumer or a demoable no-mode-lock crowd).

---

## TL;DR

The paper defines **Skill Entropy (SkE)** — a *directed pairwise* measure of how hard it is to switch from skill A to skill B inside one reasoning chain:

```
SkE(a, b) = (½·Acc(a) + ½·Acc(b) + α) / (Acc(a, b) + α)     α = 0.1 (Laplace)
```

where `Acc(s)` is a reference model's solo accuracy on skill s and `Acc(a, b)` is per-step accuracy when step-1(a) is immediately followed by step-2(b). SkE ≈ 1 → chaining adds no difficulty; SkE ≫ 1 → hard switch. **Task-level entropy** = mean pairwise SkE along the task's skill sequence. A 558-skill / 9-domain benchmark built on this scale shows frontier models drop monotonically as task entropy rises, and the dominant failure mode is **carrying over the previous step's skill and answer modality instead of switching**. Turning the same signal into a GRPO reward term (CDF-rank match of predicted vs gold skill-sequence entropy, `r = 0.7·r_ans + 0.3·r_ent`) lifts Qwen3-4B from 34.4% → 68.4%.

**Why this matters to us — the failure mode is already shipped in our games.** Issue 054 (riir-mmorpg-examples: NPCs flee toward out-of-bounds targets, get clamped, *stay clamped* — carry-over of flee mode; CLR amplifies it to crowd-scale) and Issue 057 (hero `Idle` short-circuits on `has_quest` before ever reaching the accept branch — reuse of the previous behavioral mode instead of switching) are *exactly* the paper's failure mode in game form. Our mitigations (Issue 054's L0/L1/L2 stack, Issue 057's respawn sweeps) are all **reactive** — they fire after the agent is already stuck. The paper's core insight enables a **proactive** signal: a precomputed, directed, per-pair hardness table that fires *before* the switch fails.

**Distilled modelless primitive (Issue 663):** a generic `SwitchCostTable` — success counters → directed pairwise difficulty matrix → sequence entropy + CDF-rank — with the paper's **factorization trick** `SkE(a,b) ≈ SkE(a, family_b) · SkE(family_a, b)` collapsing O(N²) pairs to O(N·F) for large mode sets.

---

## 1. Paper Core Findings

### 1.1 The measure (§3.1)

- **Directed**: `SkE(a,b) ≠ SkE(b,a)` — measured on different two-step pairs. In their domain matrix, Planning→Info-Extraction is the hardest switch (~4.4); sibling symbolic domains (Math↔Coding) switch cheaply.
- **Reference-model-relative**: computed once under a fixed strong model (Claude-opus-4.7) so it is a stable difficulty scale for all evaluated models. Ablated: partition overlap 82–86% under alternative references; Spearman ρ 0.6–0.77; annotator–model agreement κ 0.444 vs inter-annotator κ 0.625 — the scale substantially reflects human-perceived switching difficulty.
- **Factorized** (Eq. 7, §B.4): full pairwise table needs O(|S|²) ≈ 3.1×10⁵ evals for 558 skills; factorizing through 9 domains (`SkE(a,b) ≈ SkE(a, d_b)·SkE(d_a, b)`, multiplicatively separable leave-cost × land-cost) needs ~5k per direction. Multiplication is the natural composition because no-interaction transitions multiply to ≈1 and a hard factor pulls the product above 1.

### 1.2 The benchmark finding (§3.3)

- Every frontier model loses −4% to −13% when the *same skill* is exercised inside a cross-skill task vs single-skill; Planning degrades most.
- **Per-domain difficulty is decoupled from switch difficulty**: Science is an easy domain (high solo accuracy) yet has the *highest* skill entropies (skills are easy but domain-specific — hard to switch into/out of). Solo competence does not predict switch competence.
- Failure-mode decomposition (Fig. 5): 9–17% of single-skill-correct steps fail inside cross-skill tasks; 31–62% of those new failures come with the model picking a skill from the **wrong domain**; on wrong-domain steps accuracy roughly halves.

### 1.3 The training signal (§4)

- Model emits `<skill> Domain, Skill </skill><answer> … </answer>` per step; reward `r = 0.7·r_ans + 0.3·r_ent`; `r_ent = 1 − |ρ̂ − ρ★|` where ρ̂/ρ★ are **empirical-CDF ranks** of predicted/gold task-level entropy on the training set. CDF-rank normalization makes the reward scale-free across task difficulties.
- Out-of-bank predicted labels resolve via embedding cosine to the nearest canonical skill (<0.5 similarity → no entropy reward — a soft hallucination penalty).
- Qwen3-4B-Instruct: 34.4% → 68.4% (GRPO-only 58.8%; strongest skill-aware baseline STAT 61.4%); transfers to open-ended domains *not in the RL distribution*; plugs into off-the-shelf OpenR1-Math (+1.9% over GRPO on 6 math benchmarks). Reward-weight sweep: (0.7, 0.3) is the peak; entropy-dominant (0.3, 0.7) *hurts* (48.6%) — the structural term must shape, not dominate.

---

## 2. Path 0 Decomposition (§3.5 mandate — inventory, not verdict)

| Paper component | Math or training-loop? | Modelless analog in-stack | Status |
|---|---|---|---|
| Pairwise SkE formula | **Math** (ratio of measured success rates) | None shipped — but derivable from any success-rate telemetry. Trivially modelless: counters in, f32 table out. | ❌ gap → Issue 663 |
| Task/sequence entropy (Eq. 4) | **Math** (mean along sequence) | None | ❌ gap → Issue 663 |
| Domain factorization (Eq. 7) | **Math** (separable product) | None | ❌ gap → Issue 663 |
| CDF-rank normalization | **Math** (empirical CDF) | None shipped; sibling concept = conformal rank normalization in `ConformalIntervalCalibrator` (Plan 340) — same "rank against empirical distribution" move, different purpose | ❌ gap → Issue 663 (generic util) |
| Reference-model protocol | Measurement methodology | Ours inverts productively: measure under the **actual agent** (NPC), not a fixed reference → personalized tables (see §4 caveats) | variant |
| Skill²-Bench (LLM benchmark) | Out of scope | — | redirect |
| GRPO + skill-entropy reward | **Training loop** on top of the math | `loss_grpo.rs` ships GRPO (riir-train); Bench 558 civ action-prediction is the live trainer target; reward math is modelless-computable | → riir-train Plan 319 **Gap 6** |

Path 0 verdict: **the value is the math, not the training loop.** All four math components have no in-stack analog → the primitive is new, modelless-validable, and the RL recipe is a separable riir-train extension consuming it.

---

## 3. Distillation — the modelless primitive

### 3.1 `SwitchCostTable` (open, katgpt-core candidate)

```rust
/// Directed pairwise switch-difficulty table over a bounded mode set.
/// SkE(a,b) = (solo[a]/2 + solo[b]/2 + α) / (paired[a][b] + α)
pub struct SwitchCostTable<const N: usize> {
    solo_success: [u32; N],      // successes of mode run in isolation
    solo_trials:  [u32; N],
    pair_success: [[u32; N]; N], // successes of b given a immediately preceded
    pair_trials:  [[u32; N]; N],
    alpha: f32,                  // Laplace smoothing, paper default 0.1
}

impl<const N: usize> SwitchCostTable<N> {
    pub fn ske(&self, a: usize, b: usize) -> f32;          // hot lookup, no alloc
    pub fn sequence_entropy(&self, seq: &[usize]) -> f32;  // mean pairwise SkE
    pub fn record_solo(&mut self, mode: usize, success: bool);
    pub fn record_switch(&mut self, a: usize, b: usize, success: bool);
}

/// Factorized builder for large mode sets: SkE(a,b) ≈ SkE(a, fam_b)·SkE(fam_a, b)
/// O(N·F) measurements instead of O(N²). Bounded families: [f32; N*F] tables.
pub struct FactorizedSwitchCost<const N: usize, const F: usize> { .. }
```

Properties that fit our constraints exactly: fixed-size arrays for bounded mode enums (behavior-FSM states, quest objective kinds, LEO goal kinds, cognition runtime selections); zero-alloc lookups (SIMD-friendly row ops); deterministic given counter state (BLAKE3-commitable if a snapshot is ever needed); α-smoothing gives defined cold-start behavior.

### 3.2 What counts as a "mode" in each consumer

| Consumer (riir-ai) | Mode set | solo/paired telemetry source |
|---|---|---|
| Behavior FSM / HeroRoutine | states (Idle/Hunt/Flee/Tame/Sleep…) | per-tick FSM logs: state completed without timeout vs entered-after-X |
| Quest chains (quest_grammar) | objective kinds (hunt/tame/deliver/counter…) | quest-completion records per player |
| LEO goals (civ) | goal kinds | goal-progress telemetry (already feeds autocurriculum) |
| Cognition dispatch | CLR/KARC/CWM/ARG selection | per-NPC cognition outcome records |
| Zone experts | zone bundle ids | zone task success rates |

### 3.3 Sequence entropy as a quest-difficulty dial

The paper's rejection-sampling construction (§B.6: sample length L, sample skill sequence, accept iff sequence-entropy lands in target level) is a **modelless quest generator constraint**: sample objective sequences at a target difficulty band. Quest difficulty becomes a measured scalar (calibrated against *actual player completion rates*, not designer guesswork) — this is the strongest Q3 candidate.

---

## 4. Fusion (the Super-GOAT-shaped combination, unplanned but recorded)

### F1 — SkE-gated *preemptive* re-estimation (headline)

Shipped: `ReestimationScheduler` (latent_functor) + `CceReestimationTrigger` (cce_runtime) fire when `coherence < tau_reest` (+cooldown). Paper finding: cross-mode failure concentrates at hard switches and manifests as **carry-over before coherence visibly drops**. Fusion: a second trigger arm — when the entity's incoming transition (goal/mode/zone a→b) has `SkE(a,b) > tau_switch`, fire re-estimation / allocate deeper think budget *at the transition*, ahead of the failure. Reactive coherence + proactive hardness = two independent failure detectors on one shipped substrate.

### F2 — Think-budget allocation at hard switches

Per-NPC reasoning depth (riir-ai research 149) currently scales with gain-cost. Add: scale with *incoming switch hardness* — hard switch → one deep deliberation tick (the L2 deliberation system from Issue 054 is the natural consumer; it already exists but only fires on stuck-detection).

### F3 — Entropy-scheduled curriculum for CGSP / LEO autocurriculum

LEO autocurriculum samples goals by Q-value ("almost there"). Fuse: schedule *goal sequences* by target sequence-entropy (low→med→high bands), so self-play trains the switch, not just the goal — the paper's RL result is evidence the switch signal is trainable/load-bearing.

### F4 — Mode-lock telemetry alarm

The wrong-domain failure decomposition (Fig. 5) maps to: classify stuck-NPC incidents by whether the carried-over mode was wrong-family (structural mode-lock) vs right-family-wrong-execution. A cheap per-tick classifier over the switch table's hot rows gives crowd-level "mode-lock rate" — a measurable quality metric for the Living World demos, and the regression signal for F1/F2.

### Signal-diff vs closest cousins (mandatory, §3.6)

| Cousin | Consumes | SkE consumes | Verdict |
|---|---|---|---|
| SwiR (R241) block-entropy switch | current-step token entropy vs block reference — *within-step, reactive* | precomputed pair-level transition difficulty — *between-step, predictive* | different signal; composable (SwiR decides *how* to think, SkE *when to prepare*) |
| Bayesian-Agent (R211) | per-skill posterior over features → lifecycle actions | pair-level success ratios → difficulty scalar | maintains skills vs measures transitions |
| MUSE ValidatorMemory | per-validator outcome accumulation | per-*pair* accumulation | flat vs directed-pair; the table is the delta |
| ReestimationScheduler | internal coherence (state quality) | incoming transition hardness (structure) | orthogonal triggers, same consumer — F1 |
| LEO autocurriculum | Q-value per goal | entropy across goal *sequences* | point vs sequence difficulty — F3 |
| GEPO (riir-train) | policy-regime entropy → advantage shaping | trace skill-sequence entropy vs gold rank | both GRPO shaping terms, different signal source; composable |

---

## 5. Latent ↔ raw boundary

The table is **latent, local, never synced** — per-NPC or per-archetype derived state (self-adaptive track: updated from runtime telemetry, freeze/thaw-able as a snapshot). Only raw scalars/events cross any boundary ("re-estimation fired at tick T", quest content committed via the existing chain path). Quest difficulty scores are generation-time local; the generated quest crosses the wire as committed raw content. No sync dependency, no replay coupling — replay consumes the *events*, not the table.

## 6. Honest caveats

1. **Self-referential measurement.** The paper fixes a strong reference model; our analog measures the *agent itself*, so the table drifts as the agent learns (that's the point — personalization) but early tables are noisy. α-smoothing + a warm-up floor (min trials per pair before the table arms F1/F2) are required, else the proactive trigger fires on noise.
2. **Bounded mode sets only.** The `[f32; N²]` fixed-size design assumes small enums. Fine-grained skill banks (558-paper-style) need the factorized variant with family indirection — O(N·F) storage, still zero-alloc lookups.
3. **Directionality matters and must be preserved** — the easiest implementation mistake is collapsing to a symmetric cost; the paper's data (and our border-piling: flee→reposition is hard, reposition→flee is not) shows a≠b.
4. **The training half is unproven on our stack** — Gap 6 is gated on the civ Bench-558 baseline (35.68% single-LEO); the paper's +9.6% over GRPO is on Qwen3 cross-skill tasks, not civ traces. No parity claim is made here (§3.6: architectural only).
5. **Q3 honesty.** "NPCs that predict their own hard transitions" is not yet a finishable selling-point sentence. It becomes one if F3/F4 produce a demoable artifact (crowd that visibly recovers from mode-lock, or quest difficulty verified against player completion rates). Until then this stays a Gain with the guide deferred.

## 7. Routing + priorities

| Track | Output | Home | Priority |
|---|---|---|---|
| Modelless primitive | `switch_cost` module (table + entropy + factorization + CDF-rank util) behind `switch_cost` feature | katgpt-core, Issue 663 | P0 — small, self-contained, unblocks all consumers |
| Game consumers | F1 SkE-gated re-estimation arm; F2 think-budget; F4 mode-lock metric | riir-ai (after P0) | P1 — F1 is the falsifiable A/B (stuck-rate with vs without preemptive arm, Issue-054 scenario as the bench) |
| Healer consumer (candidate→**MEASURED, resolves CLOSED**) | Cross-domain span-healing gap measurement (solo vs co-present accuracy per ordered domain pair, on the Issue 017/016/G7 harness) → if a directed gap exists: fix-ordering by switch cost (proactive) ahead of `fix_verify`'s reactive auto-revert | riir-clippy (2026-08-16 addendum: the multi-domain framework is a two-level mixture — inner `pick_domain` MoE gated on exact kernel signatures, outer `DomainRouter` deliberately dense fan-out after Issue 020 killed gating at the ~70% lexical ceiling; the instantiation re-ranker is the shipped modelless analog of Skill-Entropy RL's per-step skill commit) | **Measured 2026-08-16 (riir-clippy Issue 024 / [Bench 032](../../riir-clippy/.benchmarks/032_cross_domain_switch_cost.md), commit `e34ade9`):** 12 directed pairs × 2 orders, paired fresh-baseline. At re-rank pool 8: perf→clippy +8.3pp (one pair, `needless_return`+`vec-with-capacity`, BOTH orders), clippy→perf 0.0pp. Mechanism probed = **retrieval pool crowding, NOT fix interference** — the correct rule crowded to raw similarity rank 9 by loop-shaped rules scoring 3× its solo winner, one position past the pool the re-ranker reads, so the instantiation re-ranker never saw it. Mitigated by `RERANK_POOL` 8→12: **0.0pp both directions, zero drops**, standing eval improved (clippy top-1 87.5→89.6). The paper's P2 branch resolves: **fix-ordering consumer NOT warranted** (it targets a different mechanism than measured); dense fan-out is switching-cost-free at the corrected pool depth |
| Quest difficulty dial | F3-adjacent: sequence-entropy rejection sampling in quest_grammar | riir-ai | P2 — needs quest-completion telemetry volume |
| Training recipe | skill-entropy GRPO reward on civ traces | riir-train Plan 319 Gap 6 | P3 — gated on a triggering consumer + GPU window |
| LLM benchmark (Skill²-Bench) | — | out of scope | redirect |

> **PASS-Redirects (synthesis):** Vallabhaneni, Cagwin & Wild [arXiv:2609.04159 "SENTINEL-RL: Offloading Topological Reasoning from LLM Agents in the Security Operations Center"] — entropy-threshold human escalation ("decision entropy crosses threshold → mandatory human intervention") is this note's entropy-directed-decision family in SOC clothing; covered by Bench 034 entropy-binned UCB1 arm selection + Research 494 conformal dual-threshold risk exit. Pass — no files.
