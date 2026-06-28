# Research 320: Red Queen Gödel Machine — Selective Erasure & Best-Belief Selection

> **Source:** [The Red Queen Gödel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/pdf/2606.26294) — Iacob, Jovanović, Shen, Burkhardt, Kurmanji, Tastan, Sani, Venanzi, Odonnat, Cao, Marino, Qiu, Lane (Cambridge / Flower Labs / NVIDIA / MBZUAI / Inria), arXiv:2606.26294v1, 24 Jun 2026
> **Date:** 2026-06-28
> **Status:** Active
> **Related Research:** 080 (VPD co-evolution), 021 (G-Zero Proposer×Generator), 074-riir-ai (NS-RL three-mode co-evolution taxonomy), 098 (PrudentBanker safe-phased bandit), 300 (SelfCompact closed-unit compaction gate), 301 (Misalignment Indicator Probe Bank)
> **Related Plans:** 137 (PrudentBanker safe-phased bandit ✅), 279 (Manifold Power Iteration MoE Router — Gram cache invalidation on snapshot bump ✅), 315-riir-ai (Policy Cache Cascade — `invalidate_zone_on_collapse` ✅), 253-riir-neuron-db (Merkle-Octree Curator Consensus), 002-riir-neuron-db (`can_freeze` consolidation gate ✅)
> **Cross-ref (riir-ai / riir-neuron-db):** Issue 004 (katgpt-rs/.issues/) for Super-GOAT fusion follow-up; Plan 336 (katgpt-rs/.plans/) for GOAT implementation; potential riir-ai guide if fusion verdict promotes
> **Classification:** Public

---

## TL;DR

RQGM's headline is recursive self-improvement under **non-stationary utilities** via co-evolved learned evaluators — a *training* contribution that routes to riir-train. But the paper ships **two transferable modelless inference primitives** that we do not have as unified abstractions:

1. **Criterion-versioned record store with selective erasure** — every utility record carries a `dep_set` of evaluator slots whose criterion affected it; when a slot's criterion changes (snapshot swap / adapter hot-swap / direction-vector flip), only records in that slot's dep set are erased, all others remain valid. Prop. 2 of the paper proves this preserves criterion consistency. We have *scattered instances* (Plan 279 Gram cache version bump, Plan 315 cascade `invalidate_zone_on_collapse`, Issue 001 HLA eigenbasis BLAKE3-check on reload) but no unified primitive.
2. **ε-best-belief Beta quantile selector** — `BB_ε(a) = I⁻¹_ε(1 + S_a, 1 + F_a)`, the ε-quantile of the Beta posterior, is a conservative lower bound the candidate's true utility exceeds with probability 1 − ε. Used for *selection* (which evaluator/snapshot to promote), complementing our Thompson sampling (used for *exploration*).

The combination yields a **controlled-utility-evolution freeze/thaw pattern**: epoch-local stationarity (criterion frozen within epoch) + boundary replacement (challenger promoted only if it strictly raises ε-best-belief on an anchor) + selective erasure (only dependent records invalidated). This reframes our existing MAPE-K self-healing + `MerkleFrozenEnvelope` + Raven/δ-Mem consolidation under a single epoch-local-stationarity + boundary-replacement contract.

**Distilled for katgpt-rs (modelless, inference-time):**
- The *training loop* (LLM-based evaluator improvement) → riir-train. Not distilled here.
- After reading the riir-ai Super-GOAT corpus (R158/R161/R155) and grepping shipped code (`dec/cache.rs`, Plan 335 `zone_cache.rs`), the distillation **shrinks to one genuinely-new primitive**:
  - `best_belief_score()` / `select_best_belief()` — ε-quantile Beta lower bound over `(successes, failures)` accumulators, used for conservative *selection* of frozen snapshots / archetype blend shards / zone geometry pods. Complements the existing `sample_beta` Thompson sampling (exploration) with a conservative-exploitation counterpart.
- The originally-proposed `CriterionVersionedRecords<D>` is **NOT new** — `DecCache` (`dec/cache.rs`) and `ZoneGeometryCache` (Plan 335) already ship the criterion-versioned erasure pattern with `topology_version` + `invalidate()` + BLAKE3 source-shard validation. Downgraded to a Gain-tier DRY trait extraction (`CriterionVersionedCache<V>`) over those two existing impls.
- The Super-GOAT fusion candidate (per-NPC selective forgetting on personality swap) is **dead** — it's a paraphrase of Research 158 (Committed Personality Blend) §1.3 property #3 (sampling invariance) + §2.4 (sync boundary). See Issue 004 (CLOSED).

---

## 1. Paper Core Findings

### 1.1 Controlled Utility Evolution (the headline mechanism)

The paper's central contribution is making **evaluation part of the self-improvement loop**. Prior Gödel Machine variants (Darwin, Huxley, HyperAgents) assume a stationary external evaluator. RQGM co-evolves the evaluator alongside the task agent, but does so under a disciplined schedule:

- **Epoch-local stationarity (Asm. 2):** within an epoch, the evaluator, artifact-generation protocol, and binary scoring rule are frozen. Outcomes are time-homogeneous Bernoulli with fixed success probability `p_{r,d,j}(a)`.
- **Boundary-only replacement (§3.5):** the evaluator can change only at exponentially-spaced checkpoints. A challenger is promoted only if it strictly raises the ε-best-belief score on an evaluator-independent *anchor* (a fixed ground-truth dataset).
- **Selective erasure (Def. F.2):** when a slot's evaluator is replaced, only utility records whose `dep_set` contains that slot are erased. Records depending on other slots, or on no slot (evaluator-independent), remain. Prop. 2 proves this preserves criterion consistency.

Prop. 6 shows that with exponential checkpoint spacing (ratio ρ > 1), the cumulative reprocessing cost over a budget of B evaluations is **O(B)**, not O(B²). The paper uses ρ = 2.

### 1.2 ε-Best-Belief Selection (the lower-bound criterion)

The paper uses a single rule for both evaluator replacement and final agent selection:

```
BB_ε(a) = I⁻¹_ε(1 + S_a, 1 + F_a)
```

where `I⁻¹_ε` is the inverse regularized incomplete Beta function (the ε-quantile of `Beta(1 + S, 1 + F)`). Prop. 4 proves: under the working posterior, `Pr(U(a*) ≥ BB_ε(a*) | Q_a*) = 1 − ε`. For a sequence of L selections, with probability ≥ 1 − Lε every selected candidate's utility exceeds its reported lower bound (union bound).

This is a **conservative selection rule** — it prefers candidates with high lower-bound evidence, not just high point estimates. Ties favor the incumbent (to avoid unnecessary erasure).

### 1.3 Three-Level Sampling Hierarchy

- **Node level:** Thompson sampling over clade metaproductivity (CMP) — pooled success rate over a node's subtree.
- **Role level:** least-evaluated eligible role.
- **Task level:** least-evaluated eligible task within the chosen role.

Prop. 1 proves the pooled Beta accumulator's posterior mean converges to the role-task-balanced utility almost surely under balanced sampling.

### 1.4 Co-Evolution Curriculum + Adversarial Objectives

- **Curriculum effect (RQ3):** evaluator replacements permanently re-rank the archive (Spearman ρ settles well below 1 and never recovers). A resilient backbone lineage survives; the remainder is re-ranked. Selective erasure is necessary — the no-erasure control stays pinned to the displaced order.
- **Adversarial objectives at boundaries (§5.4):** to correct LLM self-preference bias in paper reviewers, after each replacement the AI-generated papers the displaced reviewer accepted form an adversarial pool; the next epoch additionally rewards rejecting them. The reviewer's raw APReS accuracy drops slightly but its human/AI calibration equalizes — a harder-to-hack signal for the co-evolving writer.

### 1.5 Empirical Headlines

- Polyglot coding: 71.7% held-out pass rate vs prior SOTA 69.9%, at 1.35×–1.72× fewer tokens (co-evolved code reviewer complements test execution).
- Paper writing: co-evolved writers reach 1.78×–1.86× higher reviewer-panel acceptance.
- Proof grading: co-evolved grader 3× more token-efficient than HGM-H at matched accuracy.
- Theoretical guarantees are **epoch-local only** (Prop. 3, 5) — the paper explicitly disclaims global convergence (App. G).

---

## 2. Distillation

### 2.1 What routes to riir-train (NOT distilled here)

- The LLM-based evaluator fine-tuning loop (Gradient descent through evaluator weights).
- The meta-agent's self-modification of its own codebase (requires foundation-model training).
- The DPO/GRPO-style writer improvement from reviewer signal.
- The adversarial-pool curriculum that hardens the writer over epochs.

These are genuine training contributions. Per §3.5 of the research skill, the modelless-unblock protocol was checked: none of these can be realized via freeze/thaw, raw/lora hot-swap, or latent-space correction, because the *evaluator improvement itself* requires gradient-based learning on new data. → riir-train.

### 2.2 What stays modelless (distilled here)

#### 2.2.1 "Selective erasure" is ALREADY SHIPPED — `DecCache` + `ZoneGeometryCache`

**Correction (2026-06-28, after reading Plan 335 + `dec/cache.rs`):** The original draft of this note proposed `CriterionVersionedRecords<D>` as a unification of "scattered instances". That framing understated the prior art. The pattern is **already shipped in two production forms**:

1. **`DecCache`** (`katgpt-core/src/dec/cache.rs`): single-slot criterion-versioned cache. Ships `topology_version: u64`, `is_valid(complex_version)`, `invalidate()`, `mark_face_destroyed(face, version)` (selective dirty-region tracking), AND derived-stat recomputation (`store_hodge(components, version)`, `store_betti(bettis, version)`). This is the full pattern — version tag + validity check + selective invalidation + derived stats.

2. **`ZoneGeometryCache`** (`katgpt-core/src/dec/zone_cache.rs`, Plan 335 Phase 2): multi-entry papaya lock-free HashMap of mmap-backed `ZoneGeometryPod`s. Each entry carries `topology_version: u32` + `pod_header` (BLAKE3-validated). Ships `get_or_regen(zone_hash, shard, raw_state, regen_fn)`, `invalidate(zone_hash, new_topology_version)`, `evict_lru()`, and `validate_against_shard(shard_blake3)` returning `ZoneValidationError::SourceShardHashMismatch`. This is the full pattern at multi-entry scale with BLAKE3-tagged source commitment.

The value of distilling RQGM's Prop. 2 here is therefore NOT a new primitive — it's a **DRY trait extraction** (`CriterionVersionedCache<V>`) over these two existing impls, so that future caches (HLA eigenbasis cache from Issue 001, Gram cache from Plan 279, archetype-blend cache) implement one trait instead of reinventing `is_valid` / `invalidate` / source-hash-check vocabulary. That is a **Gain**, not GOAT.

The modelless analog of RQGM Prop. 2 (selective erasure preserves criterion consistency) IS the invariant these caches already maintain — `DecCache`'s `mark_face_destroyed` only drops the topology-dependent `hodge_cache`, not the structural `betti_cache`; `ZoneGeometryCache::invalidate(zone, new_version)` only drops the one zone whose version bumped, not the whole map. Both are special cases of RQGM Def. F.2's `Erase_m` operator.

#### 2.2.2 Best-Belief ε-Quantile Beta Selector

**The primitive.**

```rust
/// Conservative lower-bound selection over Beta(1 + successes, 1 + failures).
/// BB_ε(a) = I⁻¹_ε(1 + S_a, 1 + F_a) — the ε-quantile of the Beta posterior.
/// The candidate's true utility exceeds BB_ε(a) with probability 1 − ε
/// under the working posterior (Prop. 4 of RQGM).
pub fn best_belief_score(successes: u32, failures: u32, epsilon: f32) -> f32;

/// Select the candidate with the highest best-belief score.
/// Ties favor the incumbent (passed as `incumbent_idx`) to avoid unnecessary
/// slot transitions and the erasure they trigger.
pub fn select_best_belief(
    candidates: &[(u32 /* S */, u32 /* F */)],
    epsilon: f32,
    incumbent_idx: Option<usize>,
) -> usize;
```

**Why this complements our existing bandits.** We have `BanditStrategy::ThompsonSampling` (sample from Beta posterior → exploration), `BanditStrategy::Ucb1` (upper confidence bound → exploration), and `BanditStrategy::SafePhased` (PrudentBanker → safe baseline mixture). We do **not** have the ε-quantile **lower** bound for **conservative selection** of frozen artifacts. The closest is PrudentBanker's geometric α escalation, but that controls the *exploration/exploitation mix*, not a *promotion gate*.

The best-belief selector is the right primitive for: "which frozen snapshot do we promote to active?", "which direction vector do we commit to the shard?", "which adapter do we hot-swap in?". These are **selection** decisions under bounded evidence, where conservatism (lower bound, not point estimate) is the right risk posture. Thompson sampling is for "which arm do we pull next?" (exploration).

**Beta quantile implementation.** `I⁻¹_ε(a, b)` (inverse regularized incomplete Beta) is a standard special function. We already ship `Beta` for Thompson sampling via Jöhnk's algorithm (Plan 030). The quantile needs a different algorithm — either Newton iteration on the incomplete Beta, or a continued-fraction inverse. Both are O(1) and allocation-free. Implementation note: reuse `cfg`-gated math; do not pull a new dep.

#### 2.2.3 Controlled-Utility-Evolution Freeze/Thaw Pattern (architectural)

Combining 2.2.1 + 2.2.2 yields a reframe of our existing freeze/thaw + MAPE-K + consolidation stack:

```
epoch j:  criterion vector κ_j frozen (BLAKE3-tagged per slot)
          ↓
          records accumulate in CriterionVersionedRecords, tagged with κ_j
          ↓
checkpoint c (exponential schedule, ratio ρ = 2):
          for each slot m:
              challengers ← candidate criteria for slot m
              e* ← select_best_belief(challengers ∪ {incumbent}, ε=0.05, incumbent_idx=Some(incumbent))
              if e* ≠ incumbent:
                  criterion[m] ← e*
                  store.erase_slot(m)              // selective erasure
                  recompute_derived_stats(store)   // CMP, Thompson, etc.
```

This is the modelless analog of RQGM Algorithm 1 lines 22–31. The "anchor" in our framing is whatever evaluator-independent ground truth the slot has access to: deterministic-replay verifier (game anti-cheat), arena outcome, formal prover (Lean4), or quorum-committed raw scalars.

**Map to existing modules:**
- `riir-neuron-db/src/freeze.rs` `MerkleFrozenEnvelope` = the epoch freeze (BLAKE3-tagged criterion).
- `riir-neuron-db/src/mape_k.rs` MAPE-K loop = the checkpoint schedule + replan.
- `riir-neuron-db/src/consolidation.rs` Raven/δ-Mem = selective erasure on consolidation.
- `riir-ai/crates/riir-engine/src/latent_functor/reestimation.rs` = the drift-triggered re-estimation scheduler (analog of "evaluator replacement triggered by drift").
- `riir-ai/crates/riir-engine/src/committed_blend/` `CommittedFieldBlend` = the BLAKE3-committed archetype field (analog of "frozen evaluator with anchor lower bound").

None of these today implement the full 4-part contract (epoch freeze + boundary replacement + anchor validation + selective erasure) as a unified abstraction. They each implement a slice.

### 2.3 Fusion — DEAD (covered by R158/R161/R155)

**Correction (2026-06-28):** The original draft proposed a candidate Super-GOAT fusion: "per-NPC co-evolution with selective forgetting on personality swap at MMO scale." Reading the riir-ai Super-GOAT corpus after the fact shows this is **a paraphrase of already-committed work**, not a new capability:

- **Research 158 (Committed Personality Blend)** §1.3 property #3 verbatim: *"Sampling invariance — the personality survives observation gaps. Fog-of-war, network desync, snapshot thaw all preserve the committed blend."* And §2.4: only the K-weight vector `π` (12 bytes) crosses sync as LatCal-committed raw scalars; HLA state (8-dim) stays local per-NPC. **This IS the "position/HP survive bit-identical; affect stays local" claim.**
- **Research 161 (Cognitive Branch)** §1.2: each NPC has a `BranchBank` of ≤ D=8 orthogonal cognitive branches, failures stored branch-local, non-interference by construction. When a branch is swapped, only that branch's episodic store is affected. **This IS selective forgetting at branch granularity.**
- **Research 155 (Sub-Goal Compaction)**: CUCG at MMO scale; `can_freeze` gate (riir-neuron-db Plan 002) and trajectory rubric (Plan 320/333) already recognized as isomorphic (riir-neuron-db Research 007).

Issue 004 tracks the closure with the full Q1–Q4 evidence. **No Super-GOAT guide, no riir-ai plan.** The candidate is dead.

**Lesson (canonical failure mode):** This is the R269/R296 failure pattern in a new guise. The fusion protocol §1 mandates grepping `riir-ai/.research/` + `riir-ai/.plans/` — and the directory listing WAS done (158, 161, 155 appeared), but the guides were not READ because the grep was scoped to paper vocabulary ("selective erasure", "co-evolution"). The guides frame the same mechanism under different vocabulary ("committed personality", "non-interference branches", "sampling invariance"). **Vocabulary translation across repos is insufficient if the translated terms are only grepped, not read.** When a candidate selling point touches per-NPC + memory + swap, mandatorily `read_file` the R136/R146/R149/R152/R155/R158/R161/R163 guide set before claiming novelty.

---

## 3. Verdict

**Tiers (high → low):**

| Tier | Criteria | Routing |
|------|----------|--------|
| **Super-GOAT** | Novel mechanism + new capability class + product selling point + force multiplier (≥2 pillars) | Open primitive → katgpt-rs + private guide → riir-ai/riir-chain/riir-neuron-db |
| **GOAT** | Provable gain over existing approach, but not a new class of capability | Plan + implement → appropriate repo. Feature flag + benchmark. |
| **Gain** | Incremental improvement | Plan only, behind feature flag. |
| **Pass** | Not relevant, OR training-only | One-line note. |

**Verdict: GOAT (for `best_belief` only) + Gain (for DRY trait extraction).**

**One-line reasoning:** The ε-best-belief Beta quantile selector is genuinely new (grep confirms `sample_beta` exists for Thompson sampling, but no inverse-CDF quantile for conservative *selection*) — GOAT. The criterion-versioned-erasure pattern is already shipped in `DecCache` + `ZoneGeometryCache`; the value is a DRY trait extraction — Gain. The Super-GOAT fusion (per-NPC selective forgetting) is dead — covered by riir-ai Research 158/161/155 (Issue 004 CLOSED). The co-evolution training loop routes to riir-train.

**Routing (revised):**
- `katgpt-rs/crates/katgpt-core/src/best_belief.rs` — one new open primitive (`best_belief_score`, `select_best_belief`). Feature gate `best_belief` (opt-in).
- `katgpt-rs/crates/katgpt-core/src/cache_version.rs` — DRY `CriterionVersionedCache<V>` trait + impls for `DecCache` + `ZoneGeometryCache`. Gain-tier, deferred.
- Plan: [`katgpt-rs/.plans/336_controlled_utility_primitives.md`](../.plans/336_controlled_utility_primitives.md) (revised — Phase 1 `best_belief` only is GOAT; Phase 3 trait is Gain).
- Issue: [`katgpt-rs/.issues/004_per_npc_selective_forgetting_super_goat_fusion.md`](../.issues/004_per_npc_selective_forgetting_super_goat_fusion.md) — CLOSED, not Super-GOAT.

**GOAT gate (for `best_belief` only — must pass before promotion to default):**
- **G1 (correctness):** Property test vs reference Beta quantile (e.g. `statrs` test-only dev-dep): max abs error < 1e-4 across grid of `(S ∈ {0..100}, F ∈ {0..100}, ε ∈ {0.01, 0.05, 0.1, 0.25, 0.5})`.
- **G2 (perf):** `best_belief_score` ≤ 100 ns (closed-form, no alloc). `select_best_belief` on 8 candidates ≤ 500 ns.
- **G3 (no regression):** N/A — new module.
- **G4 (alloc-free):** Hot path allocation-free, verified via `dhat`.
- **G5 (sigmoid, not softmax):** N/A — Beta quantile, not a probability distribution selection.

**What stays open after the GOAT plan ships:**
- The DRY `CriterionVersionedCache<V>` trait (Phase 3, Gain) — extract over `DecCache` + `ZoneGeometryCache`. No GOAT gate; just bit-identical `cargo check`.
- The controlled-utility-evolution architectural observation (§2.2.3) — no plan, lives as documentation here.

---

## TL;DR

RQGM (arXiv:2606.26294) is a *training* paper at its headline (co-evolved LLM evaluators under non-stationary utilities) → the training loop routes to riir-train. The original draft of this note proposed two modelless primitives, but reading the riir-ai Super-GOAT corpus (R158 Committed Personality Blend, R161 Cognitive Branch, R155 Sub-Goal Compaction) + grepping shipped code (`dec/cache.rs` `DecCache`, Plan 335 `ZoneGeometryCache`) forced a correction:

- **`best_belief_score()` / `select_best_belief()`** (ε-quantile Beta lower bound `BB_ε = I⁻¹_ε(1+S, 1+F)`) — **genuinely new**, GOAT. Grep confirms `sample_beta` exists (Jöhnk's, for Thompson *exploration*) but no inverse-CDF quantile for conservative *selection*. Complements the existing sampler.
- **`CriterionVersionedRecords<D>`** — **NOT new**. `DecCache` and `ZoneGeometryCache` already ship criterion-versioned erasure (`topology_version` + `invalidate` + BLAKE3 source validation + derived-stat recomputation). Downgraded to a Gain-tier DRY trait extraction.
- **Super-GOAT fusion (per-NPC selective forgetting)** — **DEAD**. Paraphrase of Research 158 §1.3 property #3 (sampling invariance) + §2.4 (sync boundary). Issue 004 CLOSED with evidence.

**Verdict: GOAT for `best_belief` + Gain for DRY trait + Super-GOAT fusion dead.** The intended consumers are freeze/thaw selection (which `ArchetypeBlendShard` to promote) and zone cache promotion (which `ZoneGeometryPod` to keep hot) — using the freeze/thaw + geometry-bin substrate (`MerkleFrozenEnvelope`, `ZoneGeometryCache`), NOT LoRA hot-swap (pre-spinoff vocabulary). Lesson logged: vocabulary translation across repos is insufficient if translated terms are only grepped, not read — the R158/R161/R155 guides frame the same mechanism under "committed personality" / "non-interference branches" / "sampling invariance", invisible to a "selective erasure" grep.
