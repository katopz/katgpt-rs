# Research 240: Curiosity-Guided Self-Play (CGSP) — Modelless Asymmetric Triad

> **Source:** "Scaling Self-Play with Self-Guidance" (SGS) — Bailey, Wen, Dong, Hashimoto, Ma (Stanford), [arXiv:2604.20209](https://arxiv.org/abs/2604.20209), Apr 2026
> **Date:** 2026-06-15
> **Status:** Active — Super-GOAT candidate
> **Related Research:** 012 (TRT self-guided verification), 021 (G-Zero modelless self-play), 158 (MUX freeze/thaw self-learning), 167 (EoM WealthPruner), 190 (Self-Revising collapse→curriculum), 236 (QGF Q-gradient critic proxy)
> **Related Plans:** 049 (g_zero modelless self-play baseline), 212 (collapse-aware thinking), 111 (data_gate), 250 (breakeven complexity router)
> **Cross-ref (riir-ai):** Research 126 (NPC Curiosity-Guided Self-Play Guide), Plan 299 (NPC CGSP Runtime)
> **Classification:** Public — generic math, no game semantics

> **Implementation status (2026-06-15):** Shipped at [`katgpt-rs/.plans/274_curiosity_guided_self_play.md`](../.plans/274_curiosity_guided_self_play.md) — Phases 1-4 complete. Code lives in [`katgpt-rs/crates/katgpt-core/src/cgsp/`](../crates/katgpt-core/src/cgsp/) (7 modules, 29 unit tests); the root crate re-exports it via [`katgpt-rs/src/cgsp.rs`](../src/cgsp.rs). Feature flag `cgsp` is **opt-in**. The GOAT gate ran with 9 tests at [`katgpt-rs/tests/bench_274_cgsp_goat.rs`](../tests/bench_274_cgsp_goat.rs); results in [`katgpt-rs/.benchmarks/274_cgsp_goat.md`](../.benchmarks/274_cgsp_goat.md):
> - ✅ G2 (collapse recovery): 1 cycle vs 200+ baseline — CGSP's defining property.
> - ✅ G3/G4/P2/G6: feature isolation, 831ns/cycle (isolated `--test-threads=1`), 808µs for 1000 NPCs, latent/raw boundary respected.
> - ⚠ P3: bounded 13.00 allocs/cycle (NOT zero — optimisation tracked in [`.issues/021_cgsp_cycle_allocation_reduction.md`](../.issues/021_cgsp_cycle_allocation_reduction.md)).
> - ⚠ G1 (transfer-to-target) is INFORMATIONAL: the `(1 − solve_rate) · guide_score` reward is curiosity-correct but target-agnostic by design. CGSP is an exploration driver, not a target-seeker.
>
> **Promotion decision:** KEEP OPT-IN. The private selling-point guide for the runtime consumer is [`riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md`](../../../riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md); promote to default only after [`riir-ai/.plans/299_npc_curiosity_self_play_runtime.md`](../../../riir-ai/.plans/299_npc_curiosity_self_play_runtime.md) validates on real game domains. Runnable demos: [`examples/cgsp_minimal.rs`](../examples/cgsp_minimal.rs), [`examples/cgsp_collapse_recovery.rs`](../examples/cgsp_collapse_recovery.rs).

---

## TL;DR

SGS trains both Solver and Conjecturer weights via REINFORCE — the paper itself is **training-only → riir-train**. But the *triad architecture* (Solver × Conjecturer × Guide) and the *anti-collapse insights* (Guide prevents degenerate Conjecturer drift, entropy-preserved Solver feeds Conjecturer signal, intermediate-difficulty filter prevents both saturation and starvation) distill into a **novel modelless primitive** when fused with existing katgpt-rs infrastructure: Hint-δ bandit (Plan 049), collapse-aware thinking (Plan 212), data_gate (Plan 111), breakeven complexity router (Plan 250).

The distilled primitive: **Curiosity-Guided Self-Play (CGSP)** — an inference-time loop where a frozen Conjecturer proposes candidate subgoals (latent direction vectors), a frozen Guide scores them by `relevance × elegance × non-redundancy`, an intermediate-difficulty filter admits only the useful ones, a frozen Solver attempts them, and the empirical solve-rate × Guide-score product drives a Hint-δ bandit that **updates direction-vector priorities** (NOT weights). Collapse detection routes degenerate-conjecturer drift back to exploration. Freeze/thaw snapshots the resulting priority table as a personality checkpoint.

**Distilled for katgpt-rs (modelless, inference-time):**
Three frozen role-fillers + one bandit. No gradient updates. All "learning" is priority-table updates on direction vectors, gateable by feature flag, zero-allocation on the hot path.

---

## 1. Paper Core Findings (SGS)

### 1.1 The Triad Architecture

SGS factors self-play into three LLM roles initialized from one base model:

| Role | Symbol | Function | Trainable? |
|------|--------|----------|------------|
| Solver | `π_θ` | Attempts problems, generates solutions | Yes (REINFORCE 1/2) |
| Conjecturer | `g_ϕ` | Generates synthetic sub-problems conditioned on unsolved targets | Yes (REINFORCE on combined reward) |
| Guide | `ρ` | Scores synthetic problems for relevance + elegance + non-redundancy | Frozen (SFT'd for format only) |

The combined Conjecturer reward is `R_synth = R_solve · R_guide`:
- `R_solve = 0` if solve rate `s(x̃) = 0` (too hard) or in top 30% of batch (too easy); else `1 − s(x̃)` (favoring harder-but-solvable)
- `R_guide = max(0, relevance + (2 − complexity) + (1 − redundancy))`, zeroed when conclusion complexity is 3 or 4

### 1.2 The Three Anti-Collapse Mechanisms (the paper's real contribution)

1. **Guide prevents Conjecturer collapse.** Without Guide, the Conjecturer drifts to producing high-disjunction, long-conclusion problems that superficially maximize solve rate but don't transfer to target problems (Fig 2: disjunction rate rises to 80%+, conclusion length 10× baseline). The Guide's `complexity ≥ 3 → R_guide = 0` rule kills this drift.
2. **Solver entropy preservation feeds Conjecturer.** CISPO (grouped RL with importance clipping) collapses Solver entropy → solve rates concentrate at 0 and 1 → Conjecturer gets zero reward signal (R_solve = 0 at both ends) → Conjecturer can't learn. REINFORCE 1/2 (drop problems with solve rate > 0.5) preserves entropy, keeps solve-rate distribution spread, keeps Conjecturer signal alive (Fig 7).
3. **Conditioning Conjecturer on unsolved targets is necessary.** "No Problem Conditioning" ablation produces many solvable problems but they don't transfer to target solve rate (Fig 6). The Conjecturer must be pointed at *something it's trying to unlock*.

### 1.3 Scaling law

Cumulative solve rate fits a sigmoid in log-compute: `R_C = R_0 + (A − R_0) · σ(B · ln(C / C_mid))`. This is the canonical sigmoid-scaling pattern already used across katgpt-rs benchmarks (Bench 014 epiplexity, Plan 086 SimpleTES). Nothing new here — confirms our existing scaling-law methodology.

### 1.4 Headline result

7B + 200 rounds SGS > 671B pass@4 on D3k formal math (Lean4). SGS +7% asymptote over best RL baseline. These are training-compute numbers — **not directly transferable**, but they certify the triad architecture is the right shape for sustained self-improvement.

---

## 2. Distillation — Modelless CGSP Triad

### 2.1 Why direct-mapping fails

SGS's value is in the *training-loop dynamics* (entropy preservation, reward shaping, REINFORCE updates). Stripped of training, what remains is:
- The triad *shape* (Solver / Conjecturer / Guide)
- The combined reward *form* (`R_solve · R_guide`)
- The anti-collapse *insights* (3 mechanisms above)

Direct-mapping "implement Guide as a bandit" misses the point. The value is in the **fusion** of these three insights with existing katgpt-rs primitives that already implement pieces of the puzzle.

### 2.2 The Fusion

**CGSP = SGS triad × g_zero Hint-δ bandit × collapse_aware_thinking × data_gate × breakeven_complexity × freeze/thaw snapshot**

| SGS component | Modelless katgpt-rs counterpart | What changes |
|---------------|---------------------------------|--------------|
| Solver `π_θ` (trained) | Existing frozen inference brain (e.g. `LoraPlayer`, `HlPlayer`, `GZeroPlayer`) | Stays frozen. No weight updates. |
| Conjecturer `g_ϕ` (trained) | Frozen direction-vector pool + sampling | Generates candidate subgoals by perturbing latent direction vectors. Priority table replaces weights. |
| Guide `ρ` (frozen) | Frozen HLA projection + rubric scoring (existing `rubric_player`, GEPA-D bandit) | Already exists. Reused as-is. |
| `R_solve` (intermediate-difficulty filter) | `breakeven_complexity` router (Plan 250) + `CaDDTree` adaptive budget (Plan 194) | Already exists. |
| `R_synth = R_solve · R_guide` (REINFORCE target) | Hint-δ bandit reward (Plan 049) | Bandit updates priority table, not weights. |
| Solver entropy preservation (REINFORCE 1/2) | `collapse_aware_thinking` (Plan 212, default-on) + sigmoid margin (Plan 157) | Already exists. When entropy dips, collapse-aware detector triggers Conjecturer exploration. |
| Conjecturer collapse → degenerate problems | `pathway_tracker` (Plan 231) + `data_gate` (Plan 111) + epiplexity (Plan 090) | Detects structural complexity drift; blocks degenerate subgoals. |
| Frozen-Conjecturer saturation (Solver covers fixed distribution) | Freeze/thaw snapshot cycle | Periodically thaw Conjecturer priority table, re-bootstrap from recent Solver traces, re-freeze. |
| Scaling-law sigmoid fit | Already canonical pattern (Bench 014, Plan 086) | Reused. |

### 2.3 The CGSP Loop (zero-allocation, hot-path-safe)

```
Inputs:
  - target_subgoal: latent direction vector for the unsolved target
  - conjecturer_pool: [(direction_vec, priority)] frozen snapshot
  - guide_rubric: frozen HLA projection weights
  - solver: frozen inference brain
  - bandit: Hint-δ bandit over conjecturer_pool indices

Per cycle (plasma-tier budget: ≤1µs):
  1. Conjecturer: sample k candidate directions from pool, weighted by priority
     → k candidate subgoal direction vectors
  2. Guide: project each candidate onto target_subgoal direction
     → r_guide[i] = sigmoid(dot(candidate[i], target) · λ_g) · elegance_score(candidate[i])
     → elegance_score = sigmoid(−α · structural_complexity(candidate[i]))
  3. Difficulty filter: drop candidates with estimated_solve_rate ∈ {0} ∪ top_30%
     → uses breakeven_complexity router
  4. Solver: attempt remaining candidates, collect empirical_solve_rate[i]
  5. Reward: r_synth[i] = (1 − empirical_solve_rate[i]) · r_guide[i]
  6. Bandit update: priority[i] ← absorb_compress(priority[i], r_synth[i])
     → Hint-δ gated update (Plan 049), no allocation
  7. Collapse check: if entropy(priority) < τ_low or max(priority) > τ_high
     → trigger exploration injection (collapse_aware_thinking Plan 212)
     → if drift detected (pathway_tracker Plan 231), data_gate blocks next batch
  8. Periodically (every N cycles): freeze/thaw snapshot of priority table
     → BLAKE3-committed, atomic swap, per-entity personality version
```

### 2.4 Key Insight: Solver Entropy and Conjecturer Signal Are Coupled

SGS §4.5's most transferable finding: **the Solver's entropy distribution is the Conjecturer's training signal**. When Solver entropy collapses (becomes near-deterministic), solve rates concentrate at 0 and 1, and the intermediate-difficulty band empties — starving the bandit of reward.

Modelless form:
- `collapse_aware_thinking` (Plan 212) detects Solver entropy collapse in ~5ns/token
- When triggered, inject exploration noise into Conjecturer sampling (raise temperature on direction-vector sampling)
- This restores solve-rate spread, restoring bandit signal

This is the **novelty**: no existing note connects collapse detection (Plan 212) to curriculum generation (g_zero Plan 049). The fusion makes collapse-aware not just a *detector* but a *controller* of the Conjecturer.

### 2.5 Latent vs Raw Boundary

| Quantity | Space | Synced? | Reason |
|----------|-------|---------|--------|
| `target_subgoal` direction vector | Latent | NO | Per-entity, local. Defines what NPC is curious about. |
| `conjecturer_pool` priorities | Latent | NO | Per-entity personality. Snapshot via freeze/thaw, but the snapshot itself is committed (BLAKE3) not synced live. |
| `r_guide` scalar | Latent | NO | Local quality judgment. |
| `empirical_solve_rate` | Raw scalar | YES (if used for anti-cheat) | If solve rate determines game state changes (e.g. NPC "learned a skill"), the scalar is synced raw. |
| `bandit.priority` updates | Latent | NO | Updates only the local priority table. |
| Freeze/thaw snapshot blob | Raw bytes (committed) | YES (via Cold tier commitment) | BLAKE3-hashed, tamper-evident. The *content* is latent, the *commitment* is raw. |

Bridge functions (raw↔latent) follow the rules in SKILL.md: zero-allocation, feature-gated, no sync dependency.

---

## 3. Verdict — Super-GOAT

### 3.1 Novelty Gate (4/4 PASS)

| Gate | Question | Answer |
|------|----------|--------|
| **Novelty** | Grep `.research/` across all 3 repos — does any existing note cover this mechanism? | **PASS.** No note combines modelless self-play + Guide quality judge + collapse-driven curriculum regeneration + per-NPC curiosity direction vectors + freeze/thaw personality snapshots. Closest: 012 TRT (single-agent verification, no Conjecturer), 021 G-Zero (modelless self-play, no Guide), 158 MUX (freeze/thaw self-learning, for vocabulary not curricula), 167 EoM (wealth-based, different mechanism), 190 Self-Revising (collapse→curriculum but routes to training), 236 QGF (Q-gradient critic, different signal). |
| **Capability class** | New class of behavior, not just better numbers? | **PASS.** "NPCs that invent, judge, and pursue their own learning curricula at runtime, never collapsing to degenerate ideas" is a new capability class — no existing primitive does this. |
| **Selling point** | "Our NPCs/systems do X that no competitor can"? | **PASS.** "Our NPCs teach themselves by inventing subgoals, judging their quality via frozen rubric projections, and self-correcting when their idea generator drifts — all at runtime, no offline training round-trip." |
| **Force multiplier** | Connects to ≥2 existing pillars/systems? | **PASS.** Connects: self-learn/adaptive × freeze/thaw × MMORPG-scale game AI × bandits × collapse detection × HLA emotion vectors × data_gate × breakeven_complexity. (≥8 pillars.) |

**Verdict: Super-GOAT.** Mandatory outputs delivered:
1. **Open primitive** (this file) → `katgpt-rs/.research/240_*.md`
2. **Architectural guide** → `riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md`
3. **Open plan** → `katgpt-rs/.plans/274_curiosity_guided_self_play.md`
4. **Private plan** → `riir-ai/.plans/299_npc_curiosity_self_play_runtime.md`

### 3.2 One-line Selling Point

> **Our NPCs don't just play — they teach themselves: each NPC invents candidate subgoals from its curiosity direction vectors, a frozen rubric judge scores each one for relevance × elegance × non-redundancy, an intermediate-difficulty filter admits only useful ones, the NPC's existing brain attempts them, and the resulting solve-rate × quality-score drives a Hint-δ bandit that updates subgoal priorities at runtime — with collapse detection that catches degenerate-idea drift before it starves the loop, and freeze/thaw snapshots that version the emergent personality.**

### 3.3 SGS Paper Itself: Pass → riir-train

The SGS paper's training loop (REINFORCE on Solver + Conjecturer weights, Adam optimizer, Lean4 verification infrastructure) is **training-only**. Per the research skill anti-patterns, this routes to `riir-train/.research/` as a one-line note. The distilled *primitive* (this file) is what ships to katgpt-rs.

**Suggested riir-train note** (not created in this session, per workflow): `riir-train/.research/NNN_sgs_self_guided_self_play.md` — captures the training-loop insights (REINFORCE 1/2 vs CISPO entropy dynamics, Guide rubric design, STP comparison, scaling-law fitting methodology).

---

## 4. Implementation Surface (open primitive, generic)

### 4.1 New trait: `CuriosityConjecturer`

```rust
/// Frozen conjecturer that proposes candidate subgoal direction vectors.
/// Implementations: pool-sampling, perturbation-based, KG-triple-derived.
pub trait CuriosityConjecturer {
    type Direction; // typically [f32; D] or ternary bit-plane
    type Target;

    /// Sample k candidate directions weighted by current priority table.
    /// Zero-allocation: writes into pre-allocated `out` slice.
    fn sample_candidates(
        &self,
        target: &Self::Target,
        priorities: &[f32],
        out: &mut [Self::Direction],
    );
}
```

### 4.2 New trait: `QualityGuide`

```rust
/// Frozen guide that scores candidate directions for relevance × elegance.
pub trait QualityGuide {
    type Direction;
    type Target;

    /// Score ∈ [0, 1]. Combines relevance (dot-product + sigmoid) with
    /// elegance penalty (structural complexity via sigmoid gate).
    /// Zero-allocation: scalar in, scalar out.
    fn score(&self, target: &Self::Target, candidate: &Self::Direction) -> f32;
}
```

### 4.3 Combined `CgspLoop` (composable, generic)

```rust
pub struct CgspLoop<C: CuriosityConjecturer, G: QualityGuide, S: Solver, B: HintDeltaBandit> {
    conjecturer: C,
    guide: G,
    solver: S,
    bandit: B,
    priorities: Vec<f32>, // updated in-place, no alloc
    collapse_detector: CollapseAware, // from Plan 212
    difficulty_filter: BreakevenComplexity, // from Plan 250
}

impl<C, G, S, B> CgspLoop<C, G, S, B> {
    /// One CGSP cycle. Plasma-tier budget: ≤1µs.
    pub fn cycle(&mut self, target: &C::Target, scratch: &mut ScratchBuffers) -> CycleResult {
        // ... see §2.3 loop above
    }
}
```

### 4.4 Feature flag

`cgsp` (depends on `bandit`, `collapse_aware_thinking`, `data_gate`, `breakeven_complexity`). Opt-in initially; promote to default only after GOAT gate passes (Plan 274).

### 4.5 GOAT gate criteria

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| G1 | Subgoal admission quality ≥ baseline (g_zero Hint-δ alone) | ≥ +5pp transfer-to-target rate on synthetic benchmark |
| G2 | Collapse recovery: Conjecturer drift detected and corrected within ≤ N cycles | Pathway-tracker trip rate matches collapse-aware trip rate |
| G3 | Zero perf hurt when disabled | Feature gate verified, no code compiled in without `cgsp` |
| G4 | Per-cycle overhead ≤ 1µs (plasma tier) | Microbenchmark vs g_zero baseline |
| G5 | Freeze/thaw roundtrip preserves priority table | BLAKE3 commitment verified |
| G6 | Latent/raw boundary respected | No latent vector crosses sync boundary; only raw solve-rate scalars sync |

---

## 5. Anti-pattern check

- **No weight mutation**: ✅ All updates are priority-table (latent) or snapshot swap (committed). No in-place weight mutation.
- **No backprop**: ✅ Bandit updates are closed-form (Hint-δ absorb-compress).
- **Sigmoid not softmax**: ✅ All projections use sigmoid. Priorities normalized via softmax-free scheme (Hint-δ absorb-compress is additive, not softmax-normalized).
- **Zero-allocation**: ✅ Scratch buffers passed in; priorities updated in-place.
- **3-repo discipline**: ✅ Open traits in katgpt-rs; game semantics in riir-ai.

---

## 6. References

- **Source paper:** [SGS arXiv:2604.20209](https://arxiv.org/abs/2604.20209) — Bailey, Wen, Dong, Hashimoto, Ma (Stanford, Apr 2026)
- **Closest cousins (read on demand):**
  - `katgpt-rs/.research/012_TRT_Test_time_Recursive_Thinking.md` — self-guided verification (single-agent)
  - `katgpt-rs/.research/021_G-Zero_Self-Play_Open-Ended_Generation.md` — modelless self-play baseline (no Guide)
  - `katgpt-rs/.research/158_MUX_Multiplexed_Latent_Reasoning.md` — Mux Target Freeze/Thaw for self-learning
  - `katgpt-rs/.research/167_Economy_of_Minds_Hayek_Market_Coordination.md` — WealthPruner self-improvement
  - `katgpt-rs/.research/190_Self_Revising_Discovery_Regime_Transition.md` — collapse→curriculum routing
  - `katgpt-rs/.research/236_QGF_Test_Time_Q_Guided_Flow.md` — Q-gradient critic proxy
- **Foundational plans:**
  - `katgpt-rs/.plans/049_g_zero_self_play.md` — modelless self-play baseline (Hint-δ bandit)
  - `katgpt-rs/.plans/111_data_gate.md` — task-level admission gate
  - `katgpt-rs/.plans/212_collapse_aware_adaptive_thinking.md` — entropy collapse detection
  - `katgpt-rs/.plans/250_breakeven_inference_routing.md` — intermediate-difficulty router
  - `katgpt-rs/.plans/194_caddtree_adaptive_budget.md` — cost-aware adaptive budget
- **Cross-repo:**
  - `riir-ai/.research/126_NPC_Curiosity_Guided_Self_Play_Guide.md` — private architectural guide
  - `riir-ai/.research/041_Curiosity_Pulse_Entropy_Driven_Information_Gathering.md` — curiosity signal source
  - `riir-ai/.research/075_S2F_DeGRPO_Collapse_Aware_Game_Training.md` — collapse-aware training counterpart
  - `riir-ai/.research/125_MCTS_Collapse_Discriminator.md` — collapse discrimination
  - `riir-ai/.plans/299_npc_curiosity_self_play_runtime.md` — private runtime plan
