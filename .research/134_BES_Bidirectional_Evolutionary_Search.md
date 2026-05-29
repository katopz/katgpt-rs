# Research 134: BES — Bidirectional Evolutionary Search

> **Source:** [Self-Improving Language Models with Bidirectional Evolutionary Search](https://arxiv.org/pdf/2605.28814) — Xu, Qi, Su, Ye, Lakkaraju, Kakade, Du (Harvard/MIT), 2026-05
> **Date:** 2026-05, distilled 2026-05-29
> **Related Research:** 091 (SpecHop), 052 (GFlowNet), 076 (SR²AM), 075 (Data Gate), 128 (Proof Sketch Evolution), 129 (OPUS Boltzmann)
> **Related Plans:** 131 (SpecHop), 112 (SR²AM), 111 (Data Gate)
> **GOAT Pillars:** Validates all 4 pillars indirectly (search diversity + goal decomposition)

---

## TL;DR

BES couples **forward trajectory evolution** (expansion + 4 recombination operators) with **backward goal decomposition** (recursive sub-goal tree). Theoretical proofs show: (1) expansion-only search is confined to a narrow entropy shell, evolution operators escape it; (2) backward sub-goals give exponential sample reduction. Experiments on logical reasoning, multi-hop QA, and open problem solving show BES outperforms GRPO, Tree-GRPO, and evolutionary baselines.

## Verdict: ⚠️ NO GAIN — Validates Existing Design

**Why no gain:**

| BES Component | Our Existing Equivalent | Gap |
|---------------|----------------------|-----|
| Forward evolution operators (crossover, combination, translocation, deletion) | `DiversityStrategy::Decompose` + `Combine` in proof sketch (Plan 128), GFlowNet shortest-path diversity (R023) | BES operates on LLM reasoning trajectories via prompt engineering — not a Rust algorithm |
| Backward goal decomposition → sub-goal tree | `ProofGoalCache` + `DTreeGoalCache` (Plan 128), ROPD rubric multi-criteria scoring (R036) | We already do hierarchical goal caching with blake3 dedup |
| Boltzmann parent selection (Eq. 3) | `OpusBanditPruner` Boltzmann selection (Plan 129), BanditPruner UCB1 | Same math, already implemented |
| Pair scoring for complementary parents (Eq. 6) | SR²AM uncertainty-aware selection (Plan 112), SpecHop hop-level DDTree (Plan 131) | Pair coverage = our multi-arm bandit over diverse candidates |
| Entropy shell theorem (Thm 4.4) | Validates GFlowNet exploration (R023) + SpecHop multi-hop (R091) | Theoretical validation only — no code change |
| Exponential sample reduction (Thm 4.5) | Validates our Data Gate self-play stability (R075) + SR²AM budget allocation (R076) | Confirms decomposition approach we already use |

**Critical mismatch:** BES requires an LLM to perform both evolution operators (prompt the LLM to combine/recombine trajectories) and goal decomposition (prompt the LLM to break goals into sub-goals). This is LLM orchestration at training/inference time — not a Rust inference algorithm. Our codebase does Rust-side constraint checking, verification, and scoring — the LLM is external.

## Key Ideas Distilled

### 1. Entropy Shell Confinement (Theorem 4.4)

**What:** Autoregressive expansion (tree search, best-of-N) generates candidates with log-probability close to trajectory entropy H_T ± εT. All reachable candidates live in a set of size ≈ exp(H_T + εT).

**Proof intuition:**
- Per-step surprisal Z_t = -log P(y_t | y_{<t}) has bounded deviation (Azuma-Hoeffding)
- Total information S_T = ΣZ_t concentrates around H_T exponentially
- Evolution operators (crossover at splice point s) break inter-block dependence
- Expected surprise of crossover: E[-log P(V, U')] = H_T + I(V;U) > H_T + εT

**Why it matters for us:** Validates that our GFlowNet (R023) and SpecHop (R091) multi-path exploration is the right approach — single-path search is provably confined.

### 2. Evolution Operators

| Operator | What | Analogy |
|----------|------|---------|
| Combination | Concat suffixes of 2 trajectories beyond shared prefix | Sexual recombination |
| Deletion | Remove interior step from trajectory | Point mutation (loss) |
| Translocation | Replace one step with step from another trajectory | Gene translocation |
| Crossover | Splice prefix of one onto tail of another | Chromosomal crossover |

**Selection:** Boltzmann distribution over backward scores with temperature annealing (τ_0 → τ_end). Unexplored nodes get λ=0.1 bonus. Pair selection maximizes joint sub-goal coverage.

**Why it matters for us:** Validates our `DiversityStrategy::Combine` + `Decompose` in proof sketches. The temperature annealing matches our OPUS Boltzmann τ scheduling.

### 3. Backward Goal Decomposition (Theorem 4.5)

**What:** Recursively decompose problem into sub-goals with per-goal verifiers. Score each candidate as: s(n,g) = α·V_g(x,n) + (1-α)·avg(children scores).

**Key result:** Terminal-only search needs N_term = Ω(1/Πp_i) candidates. Bidirectional needs N_bidir = O(p_min^{-1} · log(m/δ)). Ratio is exponential in sub-goal count m.

**Why it matters for us:** Validates our hierarchical proof goal cache design (Plan 128). Our blake3 dedup + per-branch scoring already captures this.

### 4. Cost Analysis

| Method | Accuracy | Walltime |
|--------|----------|----------|
| GRPO (MuSiQue 3B) | 2.1% (-1.9) | 64s |
| Tree-GRPO | 3.9% (-0.1) | 240s |
| BES | 7.0% (+3.0) | 309s |

BES is ~30% slower than Tree-GRPO but 2× accuracy on hard tasks. **Aligned with optimization.md: the overhead is in LLM calls, not Rust hot-path.**

## What We Already Have That Covers This

1. **OPUS Boltzmann Selection** (Plan 129) — Same Boltzmann parent selection with temperature annealing
2. **Proof Sketch Evolution** (Plan 128) — `DiversityStrategy::Decompose` + `Combine` with Elo-rated population
3. **SpecHop Multi-Hop** (Plan 131) — Continuous multi-hop speculation with hop-level DDTree
4. **SR²AM Configurator** (Plan 112) — Adaptive planning decisions (PlanNew/PlanExtend/PlanSkip/SpecHop)
5. **GFlowNet** (R023) — Trajectory diversity via shortest-path flow, escapes mode-collapse
6. **Parallel-Probe 2D** (Plan 133) — Consensus-based parallel branch control
7. **Data Gate** (Plan 111) — Self-play stability through data gating

## What's Missing (But Not Actionable)

| Gap | Why Not Actionable |
|-----|--------------------|
| LLM-prompted crossover operators | Requires LLM at runtime — we do Rust-side verification only |
| LLM-driven goal decomposition | Same — requires LLM to decompose problems |
| Trajectory-level recombination | Our trajectories are token-level DDTree branches, not reasoning steps |
| Sub-goal verifier trees | Our `ProofGoalCache` + `DTreeGoalCache` already does hierarchical dedup |

## Feature Gate Decision

**No feature gate needed.** BES is an LLM orchestration pattern (how to prompt/call LLMs during training and inference), not a Rust inference algorithm. Our existing Rust components already implement the mathematically equivalent operations at the token/tree level:

- Boltzmann selection → `OpusBanditPruner` (Plan 129)
- Goal decomposition → `ProofGoalCache` hierarchy (Plan 128)
- Trajectory diversity → GFlowNet + `DiversityStrategy` (R023, Plan 128)
- Multi-path exploration → SpecHop + Parallel-Probe (Plan 131, Plan 133)

## References

- Paper: https://arxiv.org/abs/2605.28814
- Code: https://github.com/Embodied-Minds-Lab/BES
- Our validation: R023 (GFlowNet), R076 (SR²AM), R091 (SpecHop), Plan 128 (Proof Sketch), Plan 129 (OPUS)
