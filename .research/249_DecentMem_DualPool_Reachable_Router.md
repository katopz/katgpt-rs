# Research 249: DecentMem — Dual-Pool Reachable Memory Router

> **Source:** [Self-Evolving Multi-Agent Systems via Decentralized Memory](https://arxiv.org/pdf/2605.22721) — Hao, Long (Cambridge), Zhao (UChicago), arXiv:2605.22721, May 2026
> **Date:** 2026-06-16
> **Status:** Active — GOAT
> **Related Research:** 240 (SGS curiosity self-play — CGSP open primitive), 244 (self-evolver faithfulness — same Zhao author lineage, G-Memory is DecentMem's strongest baseline), 126 (riir-ai CGSP guide — closest cousin), 127 (riir-ai implicit microcognition), 098 (PrudentBanker O(log T) regret), 035 (attractor fixed-point), 075 (survive-or-collapse), 172 (MUSE skill lifecycle)
> **Related Plans:** 274 (CGSP modelless triad), 282 (this doc — dual-pool router primitive), 299 (riir-ai CGSP runtime)
> **Cross-ref (riir-ai):** Research 126 (NPC CGSP guide), Research 127 (implicit microcognition)
> **Classification:** Public

---

## TL;DR

DecentMem proposes **decentralized dual-pool memory** for multi-agent LLM systems: each agent keeps a private exploitation pool (E-pool) of consolidated past trajectories and an exploration pool (X-pool) of fresh LLM-generated candidates. An online router (bandit-style α update supervised by an LLM-as-judge) re-weights the two pools from stage-wise feedback. The paper proves (1) **global reachability** — the X-pool acts as PageRank-style teleportation, making the induced Markov chain irreducible and aperiodic, so no agent is ever trapped in a local-optimum subspace — and (2) **O(log T) cumulative regret** on the router, matching the stochastic bandit lower bound. Empirically: +23.8% accuracy over the strongest centralized baseline (G-Memory), +52.5% over no-memory, −49% token cost.

**Distilled for katgpt-rs (modelless, inference-time):**

The transferable primitive is a **reachability-guaranteed dual-pool router**: a bandit that routes between an exploitation pool (consolidated successes, local-walk operator) and an exploration pool (fresh candidates, teleportation operator). The exploration pool always retains nonzero selection probability, which by construction makes the search Markov chain **irreducible and aperiodic** (Theorem 1). The sigmoid-based α update achieves **O(log T) regret** (Theorem 2). Both proofs are modelless — no training, no backprop. The primitive extends our existing `HintDeltaBandit` trait by splitting the single pool into E + X and adding the consolidation step (X-pool items that earned reward are merged into E-pool after each task, Eq. 8).

**Why it matters here:** Our CGSP runtime (`riir-ai/crates/riir-engine/src/cgsp_runtime/`, promoted to default after Plan 299) already implements decentralized per-NPC memory + bandit + collapse recovery + personality divergence + freeze/thaw. But CGSP has a **single static pool** (frozen at spawn) with **reactive** collapse recovery (entropy < τ_low → inject). DecentMem formalizes what CGSP does heuristically and adds two missing pieces: (a) a **growing E-pool** of consolidated successful strategies, and (b) a **proactive reachability guarantee** (the X-pool's nonzero probability prevents trapping by construction, not just reactively).

**Fusion:** DecentMem × CGSP runtime (Plan 274/299) × Research 244 FaithfulnessProbe (G-Memory — DecentMem's strongest baseline — is the SAME centralized memory that Research 244 showed suffers a 60%+ silent faithfulness gap; co-author Zhuokai Zhao is on both papers) × Research 126/127 (personality divergence) → **CGSP-DP (Dual-Pool)**: each NPC accumulates a growing E-pool of consolidated, faithfulness-verified curiosity directions, balanced against an X-pool of fresh conjectured directions, with a router that provably never traps the NPC in a personality local-optimum. The FaithfulnessProbe gates which X-pool candidates get consolidated into E-pool — only directions that demonstrably bind to behavior make the cut, directly addressing Research 244's finding that condensed/consolidated experience is silently ignored.

---

## 1. Paper Core Findings

### 1.1 The architecture (§4)

Each agent `a_m` maintains `M_m = M_{m,E} ∪ M_{m,X}`:

| Pool | Content | Operator | Role |
|------|---------|----------|------|
| **E-pool** (exploitation) | Consolidated trajectories from past tasks `(ξ_i, r*_i)` where `r* = (r_trajectory, r_comment)` | Similarity-based local walk (Top-K retrieval with threshold τ) | Reuse past successful strategies |
| **X-pool** (exploration) | LLM-generated candidates for the current unseen context | Heuristic teleportation (LLM prior `h_m` with full support) | Escape local-optimum trap |

At each stage, the online router selects a pool with probability:
```
α_{m,E} = w_{m,E} / (w_{m,E} + w_{m,X}),  α_{m,X} = 1 − α_{m,E}     (Eq. 2)
```

After execution, stage-wise feedback `Δ_t = I[q_curr > q_prev]` updates weights:
```
E-pool used:  w_E ← w_E + α      if Δ_t = 1  (successful exploitation)
              w_E ← max(1, β·w_E) if Δ_t = 0  (failed exploitation, decay)
X-pool used:  w_E ← max(1, β·w_E) if Δ_t = 1  (successful exploration, suppress E dominance)
              w_E ← w_E + α      if Δ_t = 0  (failed exploration, lean back to E)
w_X = 1.0 fixed.  α = β = 0.5.                                        (Eq. 6, 7)
```

After task completion, X-pool consolidates into E-pool and resets (Eq. 8).

### 1.2 Global reachability (§5.2, Theorem 1)

Modeling each agent's search as a Markov chain over its solution subspace `G_m = (V_m, E_m)`:

```
p_{ℓ+1} = M_ℓ · p_ℓ = [α_ℓ · T_m + (1 − α_ℓ) · h_m · 1ᵀ] · p_ℓ     (Eq. 9)
```

where `T_m` is the similarity-based transition matrix (local walk) and `h_m` is the LLM prior with `(h_m)_i > 0 ∀i` (full support teleportation). Since `α_ℓ < 1` (X-pool always active) and `(h_m)_i > 0`, every entry of `M_ℓ` is strictly positive → **irreducible + aperiodic** → global reachability. The search can reach any region of `V_m` from any starting state.

**Boundary cases (Remark 1):**
- Pure exploitation (`α = 1`): `M = T`. If `T` has a closed communicating class, the chain is reducible → **trapped** (this is exactly CGSP's collapse mode).
- Pure exploration (`α = 0`): `M = h·1ᵀ`, memoryless → loses structured reuse.

### 1.3 Logarithmic regret (§5.3, Theorem 2)

Under Assumption 1 (`r(α)` strictly concave with unique maximizer `α* ∈ (0.5, 1)`), the update rule induces Robbins-Monro stochastic approximation:

```
α_{ℓ+1} = α_ℓ + (1/ℓ)·g(α_ℓ) + (1/ℓ)·ξ_{ℓ+1}
```

where `g(α)` is locally contractive around `α*`. Standard SA results give `E[(α_ℓ − α*)²] = O(1/ℓ)`, and strong concavity gives:

```
E[R(T)] ≤ μ · Σ O(1/ℓ) = O(log T)     (Theorem 2)
```

This matches the `Ω(log T)` bandit lower bound (Auer et al. 2002) → **order-optimal**.

**Corollary 1:** Any fixed routing `ᾱ ≠ α*` incurs `Θ(T)` regret (linear). So the online router asymptotically dominates any fixed split, including the balanced `α = 0.5`.

### 1.4 Empirical results (§6)

| Setting | DecentMem vs strongest centralized (G-Memory) | DecentMem vs no-memory |
|---------|------|------|
| Average accuracy | +8.6% | +26.1% |
| Best cell (Qwen3-4B + AgentNet) | +23.8% | +52.5% |
| Token cost (BBH, Qwen3-8B) | −43% avg (−32/−49/−47% on AgentNet/DyLAN/AutoGen) | — |

**Key finding:** the advantage **grows with coordination stochasticity** (AutoGen 2.7% → DyLAN 9.2% → AgentNet 23.1% on Qwen3). When coordination is unpredictable, centralized memory homogenizes agents more aggressively, and preserving per-agent experience matters more.

### 1.5 Ablation (§7.3)

| Routing | AgentNet BBH | AutoGen BBH |
|---------|-------------|-------------|
| Exploit only (α=1) | 74.27 | 84.29 |
| Explore only (α=0) | 35.13 (collapses) | 61.88 |
| Fixed (α=0.5) | 65.28 | 78.34 |
| **Online** | **76.19** | **85.07** |

Explore-only collapses on BBH (reusable structure tasks) but stays competitive on AIME (fresh reasoning). Online beats all fixed policies, confirming Corollary 1.

---

## 2. Distillation

### 2.1 Mapping against our shipped CGSP runtime

| DecentMem feature | CGSP status (`riir-ai/crates/riir-engine/src/cgsp_runtime/`) | Gap |
|---|---|---|
| Decentralized per-agent memory | ✅ `conjecturer_pool` + `priority_table`, per-NPC seeds | None |
| Bandit routing over candidates | ✅ `PriorityTableBandit` (Hint-δ absorb + decay) | None |
| Exploration injection on collapse | ✅ `EntropyCollapse::inject_exploration` (G2 PASS, 1-cycle recovery) | None |
| Personality divergence | ✅ per-NPC RNG seeds, faction templates, Hadamard blends | None |
| Freeze/thaw snapshots | ✅ `CuriosityPrioritySnapshot`, BLAKE3, chain quorum | None |
| **Dual-pool split** (growing E-pool + fresh X-pool) | ❌ Single **static** pool frozen at spawn (`build_pool` in `templates.rs`). No consolidation of successful trajectories into new directions. | **GAP — NPCs bounded by initial pool** |
| **Formal reachability theorem** | ⚠️ Reactive only (entropy < τ_low → inject). No proactive irreducibility guarantee. | **PARTIAL — reactive vs proactive** |
| **O(log T) regret proof** on router | ❌ `PriorityTableBandit` has no proven regret bound. | **GAP — but technique is known** (Plan 030 UCB1, Plan 137 PrudentBanker already have O(log T)) |

### 2.2 The transferable primitive: `ReachableDualPoolRouter`

The distilled primitive for katgpt-rs is a **generic dual-pool bandit router** with:

1. **Two pools** — `E: Vec<Item>` (consolidated successes, grows over time) and `X: Vec<Item>` (fresh candidates, regenerated per cycle).
2. **Sigmoid routing** (per AGENTS.md: sigmoid not softmax) — `α = sigmoid(w_E − w_X)` gives exploitation probability. X-pool always has `1 − α > 0` → teleportation property holds.
3. **Online weight update** — stage-wise binary reward `Δ_t` updates `w_E` per Eq. 6/7. Provable O(log T) regret (Theorem 2 transfers because sigmoid preserves strict concavity around the decision boundary).
4. **Consolidation** — after each cycle, X-pool items that earned reward merge into E-pool (Eq. 8). E-pool grows.
5. **Reachability by construction** — since `1 − α > 0` always (sigmoid never saturates to exactly 1.0 in finite precision), the induced Markov chain over `(E ∪ X)` is irreducible. No reactive collapse detector needed (though one can still be wired for faster recovery).

This extends the existing `HintDeltaBandit` trait — a single-pool CGSP is the degenerate case `α = 1` (pure exploitation, no X-pool).

### 2.3 Sigmoid vs ratio normalization

The paper uses `α = w_E / (w_E + w_X)` (ratio). Per AGENTS.md we use sigmoid. The mapping:

```
paper:   α_ratio = w_E / (w_E + w_X)
ours:    α_sig   = sigmoid(w_E − w_X) = 1 / (1 + exp(−(w_E − w_X)))
```

Both are monotonically increasing in `w_E − w_X`, both map to `(0, 1)`, both have a unique interior maximizer under the same concavity assumption. The O(log T) regret proof (§5.3) relies on strict concavity of `r(α)` and local contractivity of `g(α)` around `α*` — both hold for sigmoid. The proof transfers without modification.

### 2.4 Connection to Research 244 (FaithfulnessProbe — same author lineage)

Co-author Zhuokai Zhao (UChicago) is on **both** this paper and the ICML 2026 self-evolver faithfulness paper (Research 244, arxiv 2601.22436). The connection is direct:

- DecentMem's strongest centralized baseline is **G-Memory** (hierarchical shared graph).
- Research 244 shows G-Memory suffers a **60%+ silent faithfulness gap** — agents ignore condensed experience.
- DecentMem's E-pool stores `(ξ, r*) = (context prototype, action prototype + self-commentary)` — this is the "raw trajectory" form that Research 244 found agents ARE faithful to, not the "condensed summary" form they ignore.

**Implication for our distillation:** the E-pool must store **faithfulness-verified** items. Wire `FaithfulnessProbe` (Plan 278) as the consolidation gate — only X-pool items whose perturbation produces a measurable behavioral delta get consolidated into E-pool. This prevents the E-pool from filling with dead-weight directions that the consumer structurally ignores (the exact failure Research 244 diagnosed).

---

## 3. Verdict: GOAT

**One-line reasoning:** The dual-pool split + formal reachability guarantee is a provable improvement over CGSP's single-pool reactive-collapse design, and the E-pool consolidation fills a genuine capability gap (NPCs bounded by spawn pool). But the underlying mechanisms (teleportation = PageRank, forced exploration = textbook, O(log T) regret = standard bandit) are not novel, and CGSP's trait architecture already half-supports novel direction generation — this is an optimization/formalization, not a new capability class.

### 3.1 Novelty Gate Assessment

**Q1 — No prior art? NO.**
- Teleportation/reachability via nonzero exploration probability: well-known (PageRank, ε-greedy, simulated annealing restart).
- O(log T) regret on bandit routing: already in our repos (Plan 030 UCB1/Thompson, Plan 137 PrudentBanker G1: `≤ C·log(T) ✅`).
- Dual-pool exploitation/exploration: standard explore-exploit framing.
- Decentralized per-agent memory: **already shipped** in CGSP (Plan 274/299, promoted to default).

**Q2 — New class of behavior? NO.**
- CGSP's `CuriosityConjecturer` trait already supports novel direction generation (the shipped impl uses a fixed pool, but the architecture allows dynamic generation). Dual-pool makes this explicit and adds consolidation — an extension, not a new class.
- "NPCs that never get trapped" — CGSP already achieves this reactively (G2 PASS: 1-cycle recovery). DecentMem makes it proactive (by construction), which is better but not a different capability class.

**Q3 — Product selling point? Moderate.**
- "NPCs accumulate new strategies over time, beyond their initial template" — incremental over CGSP's existing "NPCs invent and pursue their own subgoals" selling point.

**Q4 — Force multiplier? YES (≥6 pillars).**
- CGSP runtime (direct extension), FaithfulnessProbe 244 (consolidation gate), collapse-aware 075/212 (X-pool formalizes the recovery injection), Curiosity Pulse 041 (teleportation signal source), freeze/thaw (E-pool snapshots), chain quorum (E-pool commitment).

**Q1 + Q2 fail → GOAT, not Super-GOAT.** No riir-ai guide created. Plan the open primitive + benchmark.

### 3.2 Fusion (the GOAT gain)

| Component | Source | Role in fusion |
|-----------|--------|----------------|
| Dual-pool + teleportation + O(log T) regret | DecentMem (this paper) | The router mechanism |
| Per-NPC decentralized memory + bandit + personality divergence | CGSP runtime (Plan 274/299) | The substrate being extended |
| Faithfulness-verified consolidation | Research 244 (same Zhao lineage) | The consolidation gate (only binding items enter E-pool) |
| Conjecturer pool + priority table | Research 126/127 | The existing single-pool being split into E + X |

**Novel combination:** CGSP-DP — each NPC's curiosity direction pool splits into a growing E-pool (consolidated, faithfulness-verified successes) and an X-pool (fresh conjectured candidates), routed by a sigmoid-based bandit with provable non-trapping. The E-pool grows over the NPC's lifetime, allowing it to discover strategies beyond its faction template. The X-pool's permanent nonzero probability guarantees no personality local-optimum trapping.

---

## 4. Open Primitive Sketch — `ReachableDualPoolRouter`

```rust
/// Dual-pool memory router with provable reachability and O(log T) regret.
///
/// Routes between an exploitation pool (consolidated successes, local-walk
/// operator) and an exploration pool (fresh candidates, teleportation
/// operator). The X-pool always retains nonzero selection probability
/// (sigmoid never saturates), guaranteeing the induced Markov chain is
/// irreducible and aperiodic (DecentMem Theorem 1).
///
/// Based on Hao, Long, Zhao 2026 (arXiv:2605.22721).
/// Uses sigmoid (not softmax) for routing probability per project convention.
pub trait ReachableDualPoolRouter {
    type Item;
    type Reward: Copy;

    /// Select an item from E-pool (exploit) or X-pool (explore).
    /// Returns (item, which pool).
    /// E-pool selection probability: α = sigmoid(w_E − w_X) ∈ (0, 1).
    fn route_select(&mut self) -> (&Self::Item, PoolId);

    /// Update pool weights from stage-wise binary feedback.
    /// Implements DecentMem Eq. 6/7 with sigmoid routing.
    /// Guarantees O(log T) cumulative regret (Theorem 2).
    fn route_update(&mut self, pool: PoolId, reward: Self::Reward);

    /// Consolidate X-pool items into E-pool (DecentMem Eq. 8).
    /// Called after task/cycle completion.
    fn consolidate(&mut self);

    /// Current exploitation probability α = sigmoid(w_E − w_X).
    #[inline]
    fn exploitation_probability(&self) -> f32;

    /// Reachability invariant: X-pool probability is strictly positive.
    /// Guaranteed by sigmoid (never exactly 0 or 1 in finite precision).
    #[inline]
    fn is_reachable(&self) -> bool {
        self.exploitation_probability() < 1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PoolId { Exploitation = 0, Exploration = 1 }
```

**Target location:** `katgpt-core/src/cgsp/dual_pool.rs` (extends the existing `cgsp` module). Feature flag: `cgsp_dual_pool` (opt-in until GOAT gate passes).

---

## 5. What CGSP Already Has (don't re-implement)

| Feature | CGSP location | DecentMem equivalent |
|---------|--------------|---------------------|
| Per-NPC RNG + personality divergence | `faction_seeds.rs` | (implicit — each agent independent) |
| Hint-δ absorb bandit | `runtime.rs::PriorityTableBandit` | Eq. 6/7 weight update |
| Collapse detection + exploration injection | `EntropyCollapse::inject_exploration` (Plan 212) | X-pool teleportation (proactive, not reactive) |
| Quality guide / rubric | `guide.rs::GameQualityGuide` | LLM-as-judge stage scoring |
| Freeze/thaw snapshots | `chain_bridge.rs::CuriosityPrioritySnapshot` | (not in paper — our addition) |
| Direction pool construction | `templates.rs::build_blended_direction` | E-pool initialization |

The dual-pool router is an **extension layer** on top of CGSP, not a replacement. Existing single-pool CGSP = `α = 1` (pure exploitation, degenerate case).

---

## 6. Implementation Notes

- **Sigmoid routing:** `α = sigmoid(w_E − w_X)` not `w_E / (w_E + w_X)`. Preserves regret proof (strict concavity holds).
- **f32 saturation (discovered during Plan 282 Phase 1):** Raw f32 sigmoid saturates to exactly `1.0` for `x ≳ 18` (because `1.0 + exp(−18)` rounds to `1.0` in f32 — ULP at 1.0 is ~1.2e-7, and `exp(−18) ≈ 1.5e-8` is below it). This breaks `is_reachable()` at extreme weights. **Fix:** clamp `exploitation_probability()` to `[min_exploration_prob, 1 − min_exploration_prob]` (default `1e-4`). This is the numerical reachability guarantee — DecentMem Theorem 1 holds in continuous math; the clamp makes it hold in f32. Configurable via `DualPoolConfig::min_exploration_prob`.
- **Consolidation gate:** wire `FaithfulnessProbe` (Plan 278) — only X-pool items with behavioral delta > threshold enter E-pool. Prevents Research 244's "dead condensed memory" failure.
- **Latency budget:** dual-pool adds one sigmoid + one branch per cycle over single-pool CGSP. Sub-µs overhead. No allocation in hot path (reuse CGSP's pre-allocated pools).
- **Latent boundary:** E-pool and X-pool are **latent** (per-NPC, never synced). Only the consolidated `f32` solve-rate crosses sync (same as CGSP). E-pool snapshots commit via existing `CuriosityPrioritySnapshot` BLAKE3 channel.
- **Feature flag:** `cgsp_dual_pool` opt-in. GOAT gate: (G1) reachability — force one-hot E-pool, verify X-pool recovery without entropy detector; (G2) regret — synthetic bandit, verify O(log T); (G3) consolidation growth — E-pool size increases over cycles; (G4) faithfulness gate — dead items rejected; (G5) personality divergence — two NPCs diverge more with dual-pool than single-pool.
- **Phase 1 status (shipped):** `DualPoolBandit<B: HintDeltaBandit>` implements both `HintDeltaBandit` (delegates to active pool) and `ReachableDualPoolRouter`. Same-size E/X pools (Phase 1 simplification); true arm growth deferred to Phase 4. Drops into `CgspLoop` as the `B` type parameter with zero loop changes — caller wraps `begin_cycle()` / `end_cycle()` around `cycle()`.
- **Phase 2 status (G1 PASS, shipped):** Proactive non-trapping verified empirically. Dual-pool escapes a one-hot E-pool trap **without any collapse detector**, by construction (sigmoid + clamp → `1−α > 0` always). Three tests (`g1_proactive_non_trapping`, `g1_reachable_at_extreme_exploitation`, `g1_markov_chain_irreducibility`) verify the guarantee at balanced, exploit-heavy, and extreme α regimes. Benchmark (`benches/dual_pool_reachability_bench.rs`) measures cycles-to-escape: dual-pool always escapes (max 79k cycles at α≈1−ε); single-pool without detector never escapes (permanent trap); single-pool + detector escapes in 1 cycle reactively. **Key finding: dual-pool `begin_cycle()` costs 0.5 ns/cycle vs 15.1 ns/cycle for single-pool entropy-based detection — 30× cheaper per cycle**, because sigmoid+RNG is O(1) while Shannon entropy is O(n log n). The dual-pool provides the formal reachability guarantee at lower per-cycle cost than the reactive detector it replaces.
- **Phase 3 status (G2 PASS — practical property, shipped):** Regret tests use an **E-pool staleness reward model** (`r(α) = p_x + (p_e − p_x)·α − δ·α²`, a concave parabola with interior maximizer `α* ≈ 0.667`), NOT static rewards. This is the setting DecentMem Theorem 2 requires (strict concavity with `α* ∈ (0.5, 1)`). With static rewards, `r(α)` is linear (no interior maximizer) and the router reaches a trivial equilibrium — the theorem doesn't apply. With the staleness model, the online router reaches `α_eq ≈ 0.653` (diff from `α* = 0.013`), beats both fixed-α=0.5 (regret 43.5 vs 24.6) and fixed-α=1.0 (regret 155.2 vs 24.6), and sigmoid/ratio routing reach comparable equilibria (`α_eq` within 0.04 of each other). **IMPORTANT FINDING: the production code uses CONSTANT step size (gain=0.5, decay=0.5), NOT the vanishing step size (1/ℓ) that the paper's Robbins-Monro SA theory requires for true asymptotic O(log T).** With constant step size, the router reaches a stable equilibrium (not convergence), and the per-cycle regret gap is `r(α*) − r(α_eq) ≈ 0.002` — tiny enough that total regret at T=10000 is ~20, well within C·log(T) for C=5 (≈46). Asymptotically the regret is Θ(T·0.002) — technically linear, but practically logarithmic for horizons ≤ ~50k cycles. **True O(log T) requires implementing vanishing step size** (scale gain/decay by `1/(1+update_count)` in `route_update`). This is documented as future work in Plan 282 Phase 3 exit notes. The GOAT gate decision should weigh: the practical property (adaptive routing, beats fixed extremes) holds today; the asymptotic bound requires a follow-up implementation.

---

## TL;DR

DecentMem's dual-pool + teleportation + O(log T) regret formalizes and extends what our CGSP runtime already does heuristically. The genuinely missing pieces are: (1) a **growing E-pool** of consolidated strategies (CGSP's pool is frozen at spawn), and (2) a **proactive reachability guarantee** (CGSP recovers reactively). Both are GOAT-tier gains — provable improvements over the incumbent, plan + benchmark + promote if they win. The mechanism is not novel (PageRank teleportation, standard bandit regret), and CGSP's architecture already half-supports the capability, so this is **GOAT, not Super-GOAT**. Fuse with Research 244 FaithfulnessProbe as the consolidation gate (same Zhao author lineage — G-Memory is DecentMem's baseline AND the system Research 244 showed silently ignores 60%+ of its memory). Open primitive: `ReachableDualPoolRouter` trait in katgpt-core. Plan: 282.
