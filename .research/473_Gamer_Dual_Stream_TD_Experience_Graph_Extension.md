# Research 473: GAMER — Dual-Stream TD Value Estimation as a Risk-Aware Extension to ExperienceGraph

> **Source:** Zheng, Lin, Chen, Ren, Chen, Cheng, Luo [arXiv:2607.27415 "Bridging Inference-Time Scaling and Episodic Memory with Action-Centric Graphs"] (GAMER), 29 Jul 2026.
> **Date:** distilled 2026-08-10
> **Verdict:** **Gain** — the dual-stream Q+/Q- TD-learning rule is a modelless primitive that extends our shipped `ExperienceGraph` substrate (R301, Super-GOAT) with risk-aware retrieval. The action-graph memory scaffold itself is already covered (ExperienceGraph uses a stronger latent-embedding-based node abstraction); the novel delta is the **decoupling of positive value (suggestion) from negative value (avoidance)** so that high-variance / catastrophic-risk actions are not neutralized by averaging.
> **Related Research:** [riir-neuron-db 301](../../riir-neuron-db/.research/301_Experience_Graph_Super_Goat_Guide.md) (ExperienceGraph Super-GOAT — the closest shipped cousin, single-stream UCB1 fitness), [riir-ai 169](../../riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md) (agent-memory PASS class — GAMER escapes it by decoupling memory from LLM calls), [riir-ai 147](../../riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md) (Engram — different mechanism, O(1) hash-addressed lookup vs reward-bearing lineage).
> **Repo routing:** substrate extension lands in **riir-neuron-db** (`ExperienceNode` Pod gains `q_positive` + `q_negative` fields + dual-stream update API). The TD update rule itself is generic math and could open-ship in katgpt-core, but per R301's private-substrate decision the integration stays private.

---

## TL;DR

GAMER builds an **Action-Centric Graph** over historical agent trajectories (nodes = unique actions, edges = observed temporal transitions), then runs **dual-stream TD learning** to estimate two value functions per node:

- **Q+(v)** — positive potential, updated from raw reward (identifies high-success strategies)
- **Q-(v)** — negative risk, updated from a thresholded binary violation signal `r⁻_t = -1 if r_t < ε else 0` (identifies historically failure-prone actions)

At inference, the agent injects two lists into its prompt: top-K high-Q+ actions (suggestions) and top-K low-Q- actions (avoid-list). The paper proves (Thm 4.2, 4.3) this satisfies **First-Order Stochastic Dominance** over the base policy: for any sample budget N, the expected max reward is ≥ the single-stream baseline.

### What ships vs what's novel

| GAMER component | Ships in our stack? | Where |
|---|---|---|
| Action-graph memory (nodes = actions, edges = temporal transitions) | **PARTIALLY** — `ExperienceGraph` (R301) ships a stronger variant: nodes are `(state, action, reward, parent)` Pods with latent task_embedding; edges are lineage + sibling_hashes. Action-identity is fragile in GAMER ("open door" vs "OpenDoor" are different nodes); ExperienceGraph uses cosine ANN over embeddings → robust. | `riir-neuron-db/src/experience_graph.rs` |
| TD value learning on the graph | **YES, single-stream** — `ExperienceNode` carries `fitness` (running mean reward) + `ucb_score` (UCB1 over visit_count). | `riir-neuron-db/src/experience_graph.rs` |
| **Dual-stream Q+/Q-** (decoupled positive / negative) | **NO — this is the novel delta.** Single-stream `fitness` collapses a 50%-wins-50%-catastrophe action (E[Q]=0) into the same bucket as a step-wasting action (reward 0). GAMER's thresholded violation counter surfaces the risk separately. | **gap** |
| Memory-guided ICL prompt injection (suggestions + avoid-list as natural language) | **NO** — and irrelevant. GAMER's prompt-injection path requires ≥1 LLM call per step, which violates our 20Hz NPC tick budget (R169 PASS class). The modelless analog is: dual-stream retrieval feeds the avoid-list into a `ConstraintPruner` / `ScreeningPruner` rejection filter, not into an LLM prompt. | n/a (LLM-orchestration layer) |
| First-Order Stochastic Dominance theorem | **NOT YET formalized** — could ship as a Lean 4 invariant or property test for any dual-stream retrieval system. The theorem says: monotonic probability reallocation (failure → success regions) guarantees E[max_N] ≥ baseline for any N. | **gap (formal invariant)** |
| Re-verify-against-current-reward defense | **YES** — `ExperienceGraph`'s "graph suggests, current reward decides" (R301 §2.4) is the same defense. | `riir-engine/src/arg_runtime/experience_reuse.rs` |

### Why this is Gain, not Super-GOAT

The novelty gate (§1.5):
- **Q1 (no prior art?):** NO — action-graph memory scaffold already ships as `ExperienceGraph` (R301, Super-GOAT). The dual-stream is a refinement, not a new class.
- **Q2 (new class of behavior?):** PARTIAL — risk-aware retrieval (avoid-list alongside suggestion-list) is a meaningful new capability vs single-stream UCB1, but it is incremental.
- **Q3 (product selling point?):** YES — "Our NPCs avoid historically-catastrophic actions, not just prefer historically-successful ones" is a defensible claim.
- **Q4 (force multiplier?):** YES — extends ExperienceGraph (multiplies R301) + ARG runtime + ConstraintPruner surface.

Q1 fails → not Super-GOAT. Q2 partial + Q3 + Q4 yes → **Gain**.

---

## 1. Paper Core Findings (verified by full read)

### 1.1 The Action-Centric Graph (§3.1)

Given historical trajectories `D = {τ_1, ..., τ_N}`, the graph `G = (V, E)` is constructed as:
- **Nodes:** `V = {a | a ∈ τ, ∀τ ∈ D}` — each unique action representation is a distinct node.
- **Edges:** `e_ij ∈ E` iff action `a_j` was executed immediately following `a_i` in any trajectory.

Branching points = decision states with multiple explored strategies. Cycles = recursive reasoning loops. The graph condenses linear episode traces into a global map of the reasoning space.

**Weakness vs ExperienceGraph:** action identity is string/representation-based. Two episodes with semantically-equivalent but textually-different actions ("go to kitchen" vs "Go to Kitchen" vs "navigate(kitchen)") become different nodes. ExperienceGraph uses cosine ANN over `task_embedding[8]` for similarity — robust to surface variation.

### 1.2 Dual-Stream TD Learning (§3.2) — the load-bearing contribution

Standard TD update minimizes `δ_t = r_t + γ·E[Q(v_{t+1})] - Q(v_t)`, iterated to propagate value signals backward from terminal states.

**The single-stream failure mode GAMER identifies:** an action `a_risky` with 50% reward +1 and 50% reward −1 has E[Q] = 0.5(1) + 0.5(−1) = 0 — indistinguishable from a no-op action `a_idle` with reward 0. The risk signal is neutralized by averaging.

**Dual-stream fix:** maintain two separate value functions.
- `Q+(v)` — updated from raw reward `r_t`. Identifies success-rate.
- `Q-(v)` — updated from thresholded reward:
  ```
  r⁻_t = -1  if r_t < ε
       =  0  otherwise
  ```
  This is a binary violation counter — surfaces failure-rate independently of magnitude.

TD updates are independent per stream:
```
Q+/- (v_t) ← Q+/- (v_t) + α · [ r+/- _t + γ · max/min_{v' ∈ Succ(v_t)} Q+/- (v') − Q+/- (v_t) ]
```
(Q+ uses max successor; Q- uses min successor.)

### 1.3 Memory-Guided Inference (§3.3)

Three prompt components at inference:
1. **Reference Trajectory Guidance** — best historical trajectory `τ*` as few-shot example.
2. **Suggested Actions** — top-K successors by Q+.
3. **Avoid Actions** — top-K successors by |Q-|.

The integration prompt template:
```
This is the best historical trajectory for the current task...
{τ*}

Based on successful trajectories, consider: {A_suggest}

WARNING: Based on unsuccessful trajectories, avoid: {A_avoid}
```

**Caveat:** this path requires LLM calls per inference step — incompatible with our 20Hz modelless runtime. The modelless analog is: dual-stream retrieval drives a `ConstraintPruner` (rejection filter) + a `BanditPruner` (exploration bias), not an LLM prompt.

### 1.4 Theoretical Guarantee (§4)

**Assumption 4.1 (Monotonic Probability Re-allocation):** the memory mechanism shifts probability mass from the failure set `A_avoid` (where `r < ε`) to the success set `A_suggest` (where `r > ε'`), with zero mass moved from `[x, ∞)` to `(−∞, x)` for any threshold `x`.

**Theorem 4.2 (First-Order Stochastic Dominance):** if Q+ and Q- are ε-consistent with the true reward landscape, then `X_mem ⪰_1 X_base` (memory-guided reward distribution FSD-dominates base).

**Theorem 4.3 (Efficiency of Inference Scaling):** for any N ≥ 1, `J_mem(N) ≥ J_base(N)` where `J(N) = E[max_{i=1..N} X_i]`. To hit target reward `x*` with confidence `1-δ`, `N_mem ≤ N_base`.

The proof is one paragraph: FSD ⇒ `[F_mem(x)]^N ≤ [F_base(x)]^N` for all x (since `t ↦ t^N` is monotone on [0,1]); integrating gives `ΔJ = ∫ ([F_base]^N − [F_mem]^N) dx ≥ 0`.

### 1.5 Empirical Results (§5)

- **Benchmarks:** AlfWorld, ScienceWorld, PDDL, Tool-Query (AgentBoard suite).
- **Headline:** +20.81% success rate / +6.17% progress rate vs vanilla best-of-N.
- **Token efficiency:** ~50% reduction vs A-Mem (the second-best method). GAMER doesn't require LLM calls for memory summarization.
- **Compute:** graph construction ~0.0013s/task; TD learning <4s/task on a single Xeon thread.

---

## 2. The Distilled Primitive (modelless)

The transferable primitive is the **dual-stream TD update rule with thresholded negative reward**. The action-graph scaffold is interchangeable (GAMER's action-identity nodes vs ExperienceGraph's latent-embedding nodes — ExperienceGraph is strictly stronger). The prompt-injection inference path is LLM-orchestration and discarded.

### 2.1 The open primitive — `dual_stream_td_update`

Generic math, no game/shard/chain semantics:

```rust
/// Dual-stream TD value update.
///
/// Separates success-rate (Q+) from failure-rate (Q-) so that
/// high-variance catastrophic-risk actions are not neutralized
/// by averaging into the same bucket as no-op actions.
///
/// `q_pos` updated from raw reward; `q_neg` updated from thresholded
/// binary violation (`-1 if r < epsilon else 0`).
///
/// Guarantees (under ε-consistency): First-Order Stochastic Dominance
/// of the dual-stream-guided policy over the single-stream baseline
/// (GAMER Thm 4.2, 4.3).
#[inline]
pub fn dual_stream_td_update(
    q_pos: &mut f32,
    q_neg: &mut f32,
    reward: f32,
    next_q_pos_max: f32,    // max over successors for Q+
    next_q_neg_min: f32,    // min over successors for Q-
    alpha: f32,             // learning rate ∈ (0, 1]
    gamma: f32,             // discount factor ∈ [0, 1]
    epsilon: f32,           // violation threshold
) {
    // Positive stream: standard TD on raw reward.
    let td_pos = reward + gamma * next_q_pos_max - *q_pos;
    *q_pos += alpha * td_pos;

    // Negative stream: thresholded binary violation signal.
    let r_neg = if reward < epsilon { -1.0 } else { 0.0 };
    let td_neg = r_neg + gamma * next_q_neg_min - *q_neg;
    *q_neg += alpha * td_neg;
}

/// Retrieve suggestion list (top-K by Q+) and avoidance list (top-K by |Q-|).
///
/// Both lists are raw / scalar / deterministic — safe to cross the sync boundary.
pub fn dual_stream_retrieve<'a>(
    successors: impl Iterator<Item = (&'a NodeId, f32, f32)>,  // (id, q_pos, q_neg)
    k_suggest: usize,
    k_avoid: usize,
) -> (Vec<NodeId>, Vec<NodeId>) { /* ... */ }
```

### 2.2 The private extension — `ExperienceNode` gains Q+/Q-

The substrate extension lands in `riir-neuron-db/src/experience_graph.rs`:

```rust
// Existing ExperienceNode Pod (R301) currently has:
//   fitness: f32,        // running mean reward (single-stream)
//   ucb_score: f32,      // UCB1(fitness, visit_count, parent_visit_count)
//
// Extension:
//   q_positive: f32,     // success-rate stream (TD on raw reward)
//   q_negative: f32,     // failure-rate stream (TD on thresholded violation)
//   violation_count: u32, // count of r < epsilon observations (raw, syncable)
```

The `latent_seeded_ns_traversal` query gains a `RetrievalMode` parameter:
- `SuggestOnly` — top-K by Q+ (current behavior, back-compat).
- `AvoidOnly` — top-K by |Q-| (the new avoid-list).
- `DualStream` — both lists returned; caller's pruner applies both.

### 2.3 The modelless consumption path (no LLM calls)

The prompt-injection path of GAMER is discarded. The modelless consumption is:

1. **Suggest-list → `BanditPruner` bias** — high-Q+ candidates get exploration bonus.
2. **Avoid-list → `ConstraintPruner` rejection** — high-|Q-| candidates get soft-rejection (sigmoid-gated, not hard filter — preserves escape hatches for non-stationary regimes).
3. **Reference trajectory → `SpeculativeGenerator` drafter** — the historically-best trajectory seeds a speculative decode path (ComposeWith Plan 217 NextLat Belief-State Drafter).

This composition respects the 20Hz tick budget: all operations are µs-scale SIMD/scalar, no LLM calls.

---

## 3. Latent vs Raw Boundary

Per the global sync-boundary rule, the dual-stream values are **raw, deterministic, sync-safe**:

| Field | Type | Crosses sync? | Why |
|---|---|---|---|
| `q_positive` | f32 | YES (raw) | Deterministic given (reward, successor-max, α, γ) — replay-verifiable. |
| `q_negative` | f32 | YES (raw) | Deterministic given (thresholded violation, successor-min, α, γ, ε). |
| `violation_count` | u32 | YES (raw) | Raw integer counter. |
| `task_embedding` (query key) | [f32; 8] | **NO** (latent) | Already local-only in R301. |

The Q+/Q- values join `fitness` and `ucb_score` in the ExperienceNode's raw prefix → BLAKE3-committed, chain-portable, AS-OF-reconstructable. Anti-cheat replay now covers "did this NPC correctly avoid historically-failed actions?" — a new verifiable dimension.

---

## 4. Fusion candidates (the Super-GOAT search — none qualify, but the connections matter)

Per §1 fusion protocol, the closest cousins across the 7 repos:

| Cousin | Repo | Relationship | Fusion product |
|---|---|---|---|
| **ExperienceGraph** (R301, Plan 319/492/493) | riir-neuron-db | The host substrate. Single-stream UCB1 → dual-stream Q+/Q-. | **THIS NOTE'S GAIN** — risk-aware retrieval. |
| **Engram** (R147, Plan 299) | riir-ai + katgpt-rs | Different mechanism (O(1) hash-lookup of static patterns vs reward-bearing lineage). | Orthogonal — Engram handles "what does this NPC know?", ExperienceGraph handles "what has this NPC tried and what happened?" |
| **Raven/δ-Mem consolidation** | riir-neuron-db | Sleep-cycle that consumes ExperienceGraph queries. | Each consolidation cycle's wake-events now carry both success and failure signals → richer consolidation input. |
| **MANCE concept erasure** (R409, Plan 426) | riir-ai | Conflict-aware revision of latent directions. | A high-|Q-| action triggers MANCE-style erasure from the NPC's preferred-direction set. |
| **TILR** (R408, Plan 425) | riir-ai | Trajectory-invariant latent refinement. | TILR + dual-stream = "refine toward Q+, away from Q-" — a directional gradient signal for the latent refinement. |
| **ReMax Q+** (Plan 374) | katgpt-core | Coincidentally same name, different math. ReMax Q+ = expected improvement per action for retry-budget baseline; GAMER Q+ = success-rate TD value. | Do not confuse. Same symbol, different semantics. |
| **SDAR q_negative** | katgpt-pruners | Coincidentally same name, different math. SDAR's q_negative = asymmetric reward surprise absorption; GAMER's Q- = thresholded violation counter. | Do not confuse. The SDAR gate could *consume* the GAMER-style avoid-list, but the underlying math is different. |
| **CLR Claim-Level Reliability** (R255, Plan 284) | katgpt-rs | Reliability-weighted set attention. | CLR weights *which claims to trust*; dual-stream Q+/Q- weights *which actions to attempt vs avoid*. Composition: CLR-weighted Q+ retrieval. |
| **vibe.rs KG triples** | riir-neuron-db | `KgTripleTemplate` emits "NPC explored action X after observing Y". | The dual-stream values become KG triple attributes: `<npc> avoided <action> with Q- = -0.8`. |

No fusion produces a *new class* of behavior (Q1 fails) — each is a refinement / composition with shipped substrate. The strongest fusion is the host (ExperienceGraph) + the dual-stream primitive itself.

---

## 5. The reverse-grep check (§1.55.2)

Before Gain verdict, check for documented limitations the paper could fill:

- **R301 §6 "Honest caveats" #3:** "The reward projection is a modelless guess." → Not addressed by GAMER (GAMER also uses a fixed reward signal).
- **R301 §7 P3 deferred items:** "Per-action granularity lineage (vs per-NPC-per-night)" — GAMER's action-graph IS per-action. **Partially addressed** — the dual-stream extension naturally moves toward per-action granularity because Q+/Q- are per-node (and nodes can be per-action).
- **R169 F5-F9 findings:** all validation signals for shipped choices, not gaps.
- **`riir-neuron-db/.research/303` (Hebbian Fact-Storing):** stores facts, not action-outcome pairs. Orthogonal.
- **No `TODO|FIXME` comments in `experience_graph.rs` mention risk-aware retrieval or dual-stream.** The gap is undocumented but real.

The reverse-grep confirms: the dual-stream gap is real (single-stream `fitness` cannot distinguish high-variance from no-op), undocumented, and the paper fills it. Gain confirmed.

---

## 6. Validation protocol — the GOAT gate

For the dual-stream extension to `ExperienceGraph`:

| Gate | Target | Test |
|---|---|---|
| **G1** correctness | Dual-stream update is bit-identical to single-stream when `epsilon → −∞` (no violations ever recorded) | Property test: `dual_stream_td_update` with `epsilon = f32::NEG_INFINITY` produces Q+ identical to single-stream TD; Q- stays at 0. |
| **G2** perf | Update overhead < 100ns per node per observation | Criterion bench: dual-stream vs single-stream on 1000-node graph. |
| **G3** no-regression | All existing `experience_graph` tests (30/30 from Plan 319 Phase 4) still pass | `cargo test -p riir-neuron-db --lib experience_graph`. |
| **G4** alloc | Q+/Q- are scalar fields on existing Pod — zero new allocation | Inspect Pod layout (no `Vec`, no `String`); AS-OF query returns extended Pod. |
| **G5** quality (risk-aware retrieval) | In a controlled toy domain with a high-variance action (50% +1, 50% −1) and a no-op action (reward 0), dual-stream retrieval correctly avoids the high-variance action while single-stream does not | PoC in `riir-poc/`: 3-action bandit, 1000 episodes, compare single-stream vs dual-stream selection frequency on the high-variance arm. |

G5 is the load-bearing gate: it must show the dual-stream mechanism produces a *different* (and better) selection distribution than single-stream UCB1 on the paper's canonical failure case.

### The First-Order Stochastic Dominance invariant

A secondary formal invariant, suitable for property test or Lean 4 theorem:

> For any retrieval system satisfying Monotonic Probability Re-allocation (Assumption 4.1), `E[max_{i=1..N} X_i]` under dual-stream guidance ≥ under single-stream, for all N ≥ 1.

This is GAMER Thm 4.3. In Rust, a property test can verify it empirically: construct two samplers (single-stream and dual-stream) over the same reward distribution, sample max-of-N for N ∈ {1, 4, 16, 64}, assert dual-stream ≥ single-stream in expectation. Not a Lean theorem today, but the proof is one paragraph and could ship in `riir-neuron-db/.proofs/NeuronDbProof/ExperienceGraph/` as a sibling to the existing ExperienceNode layout spec self-tests.

---

## 7. Plan / Issue routing

This is an extension to existing substrate, not a new primitive. The work is:

1. **riir-neuron-db/.issues/** — file an issue for `ExperienceNode` Q+/Q- field extension + `dual_stream_td_update` API + `RetrievalMode::DualStream` query mode + G1-G5 gate.
2. **katgpt-rs** (optional) — the bare `dual_stream_td_update` function is generic math and could ship in `katgpt-core` as an open primitive sibling to `remax::expected_improvement_per_action`. But per R301's private-substrate decision, the integration stays in riir-neuron-db. **Default: private.** Only open-ship if a future consumer outside the quintet needs it.
3. **riir-ai** (consumer side) — wire the avoid-list into the ARG offline loop's `ConstraintPruner` step. Small change once the substrate extension lands.

No plan is opened yet — the issue defines the scope; the plan opens once the issue is triaged.

---

## 8. Why this is NOT Super-GOAT (the honest demotion)

The four YES questions (§1.5):

| Q | Answer | Reasoning |
|---|---|---|
| Q1 No prior art? | **NO** | `ExperienceGraph` (R301) ships action/experience graphs with reward-bearing lineage + TD-like (UCB1) updates. The action-graph scaffold is covered (with a stronger latent-embedding node abstraction). |
| Q2 New class of behavior? | **PARTIAL** | Risk-aware retrieval (avoid-list) is a meaningful capability, but it's an incremental refinement to single-stream retrieval, not a new behavior class. |
| Q3 Product selling point? | YES | "Our NPCs avoid historically-catastrophic actions, not just prefer historically-successful ones." |
| Q4 Force multiplier? | YES | Extends ExperienceGraph + ARG runtime + ConstraintPruner + Raven consolidation. |

Q1 fails. Not Super-GOAT. The dual-stream is a **Gain** — a real capability gap (single-stream averaging neutralizes risk signal) closed by a modelless primitive (thresholded violation counter alongside success-rate), with a clean formal guarantee (FSD).

---

## 9. Cross-references

- **[riir-neuron-db 301](../../riir-neuron-db/.research/301_Experience_Graph_Super_Goat_Guide.md)** — the host substrate (Super-GOAT, G1-G5 PASS). The single-stream → dual-stream extension is the headline fusion.
- **[riir-neuron-db 300](../../riir-neuron-db/.research/300_Experience_Graph_Database_Foundation_Gain.md)** — the original Trellis distillation + PoC (+11.51% benign reuse).
- **[riir-ai 169](../../riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md)** — the agent-memory PASS class. GAMER escapes it by decoupling memory from LLM calls (the prompt-injection path is still LLM-orchestration, but the *value-learning* path is pure TD and modelless).
- **[riir-ai 147](../../riir-ai/.research/147_Engram_Conditional_Memory_NPC_Guide.md)** — Engram (different mechanism, O(1) pattern lookup vs reward-bearing lineage).
- **[Plan 319](../../riir-neuron-db/.plans/319_experience_graph_query_layer_implementation.md)** — the substrate shipper (Phases 1-4 DONE, DEFAULT-ON).
- **[Plan 492](../../riir-ai/.plans/492_experience_graph_phase5_real_domain_re_gate.md)** — the G5 real-domain re-gate (PASS).
- **GAMER paper:** [arXiv:2607.27415](https://arxiv.org/abs/2607.27415).

---

## PASS-Redirect line for cousin notes

Add to **`riir-neuron-db/.research/301_Experience_Graph_Super_Goat_Guide.md`** near `Related Research:`:

> **PASS-Redirects (synthesis):** Zheng, Lin, Chen, Ren, Chen, Cheng, Luo [arXiv:2607.27415 "Bridging Inference-Time Scaling and Episodic Memory with Action-Centric Graphs"] (GAMER) — Gain, not PASS. The action-graph scaffold + reward-bearing lineage already ships here (single-stream UCB1). The novel delta is **dual-stream Q+/Q- TD learning** (separate success-rate from thresholded-violation-rate so high-variance catastrophic-risk actions are not neutralized by averaging) — filed as extension issue in riir-neuron-db. GAMER's LLM prompt-injection path is R169 PASS class (LLM-orchestration); the modelless consumption is via `ConstraintPruner` avoid-list, not prompts. See [katgpt-rs R473](../../katgpt-rs/.research/473_Gamer_Dual_Stream_TD_Experience_Graph_Extension.md).

Add to **`riir-ai/.research/169_Agent_Native_Memory_Benchmark_PASS.md`** near `PASS-Redirects (synthesis):`:

> **PASS-Redirects (synthesis):** Zheng et al. [arXiv:2607.27415 "GAMER — Graph-based Action-centric Memory with Episodic Reasoning"] — Gain, not PASS (escapes the LLM-orchestration failure class by decoupling memory from LLM calls). The dual-stream Q+/Q- TD-learning value estimation is a modelless primitive that extends the shipped `ExperienceGraph` substrate (R301) with risk-aware retrieval (avoid-list alongside suggestion-list). See [katgpt-rs R473](473_Gamer_Dual_Stream_TD_Experience_Graph_Extension.md).
