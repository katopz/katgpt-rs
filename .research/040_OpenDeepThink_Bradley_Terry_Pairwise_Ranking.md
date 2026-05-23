# OpenDeepThink Distillation — Model-Based & Modelless Paths

**Source:** [OpenDeepThink: Parallel Reasoning via Bradley–Terry Aggregation](https://arxiv.org/pdf/2605.15177) — Zhou et al., 2026 (UCSD, Princeton, UW, UCB)
**Date:** 2026-05-19
**Related Plans:** 049 (G-Zero), 052 (GFlowNet), 071 (ROPD Modelless), 072 (SDAR Modelless), 073 (SDAR Model-Based)
**Raw Code:** `.raw/open-deep-think/`

---

## Paper Summary

OpenDeepThink is a **population-based test-time compute** framework that:

1. Samples `n` candidate solutions in parallel
2. Evolves them over `T` generations via **pairwise Bradley–Terry (BT) comparison**
3. Each generation: compare → BT rank → preserve elite (top 25%) → mutate top 75% with feedback → discard bottom 25%
4. Final dense BT round selects the best candidate

**Key results:** +405 Elo on Codeforces (Gemini 3.1 Pro), transfers across models without retuning.

---

## Core Ideas to Distill

| Idea | Paper Mechanism | Our Analog | Novelty to Us |
|------|----------------|------------|---------------|
| **Pairwise > Pointwise** | BT over pairwise LLM judge (86% vs 59%) | ScreeningPruner::relevance() is pointwise | **HIGH** — we don't do pairwise |
| **BT Global Ranking** | L-BFGS fit of `P(i≻j) = σ(sᵢ - sⱼ)` | BanditPruner Q-value ranking | **MEDIUM** — BT is more principled than UCB |
| **Feedback-Driven Mutation** | Loser critiques → rewrite prompt | AbsorbCompress heuristic promotion | **MEDIUM** — we use reward, not critique |
| **Elite Preservation** | Top 25% carried unchanged | DDTree top-k selection | **LOW** — already implicit |
| **Population Evolution** | n=20 candidates, T=3 generations | Multi-candidate DDTree | **LOW** — already parallel |
| **Negative Feedback Signal** | Negative critique carries ~all mutation signal | DeltaGatedAbsorbCompress uses δ | **MEDIUM** — aligns with our findings |
| **Dense Final Selection** | M=10 comparisons for final BT | SpeculativeVerifier top-1 pick | **LOW** — already pick top-1 |

---

## Modelless Path (microgpt-rs)

### What Maps

```
OpenDeepThink                    Our Stack
─────────────                    ─────────
Population of n candidates  →    DDTree speculative candidates
Pairwise comparison         →    NEW: PairwisePruner trait
BT ranking                  →    NEW: BT aggregation over DDTree leaves
Elite preservation          →    DDTree top-k pruning (already exists)
Feedback-driven mutation    →    AbsorbCompress with critique input
Bottom discard              →    ConstraintPruner hard rejection
Negative feedback signal    →    DeltaGatedAbsorbCompress (δ = negative gap proxy)
```

### Distillation: PairwisePruner

The **highest-value distillation** is replacing pointwise `ScreeningPruner::relevance()` with pairwise comparison. OpenDeepThink proves pairwise judgment (86%) dramatically outperforms pointwise (59%) because:

- Pointwise scoring has **positive bias** — high recall on correct, poor recall on wrong
- Pairwise reduces judgment to a **relative contrast** — no calibrated threshold needed
- BT aggregation **internalizes opponent strength** — raw win rate can't do this

```rust
/// Pairwise relevance: compare candidate A vs B, return relative preference.
///
/// Unlike ScreeningPruner::relevance() which scores each candidate in isolation,
/// PairwisePruner asks "which of these two is better?" — sidestepping positive bias.
pub trait PairwisePruner {
    /// Compare two candidates at the same DDTree depth.
    /// Returns value > 0.5 if A preferred, < 0.5 if B preferred, 0.5 if tie.
    fn compare(
        &self,
        depth: usize,
        token_a: usize,
        token_b: usize,
        context: &[usize],
    ) -> f32;
}

/// BT-aggregated ranking over pairwise comparisons.
///
/// From OpenDeepThink: P(i ≻ j) = σ(sᵢ - sⱼ), fit via L-BFGS.
/// Internalizes opponent strength — unlike raw win rate, a candidate that
/// beats strong opponents ranks higher than one that beats weak ones.
pub struct BradleyTerryRanker {
    /// Regularization λ (paper: 0.01)
    lambda: f32,
}
```

### Distillation: CritiqueGatedAbsorbCompress

OpenDeepThink's mutation uses **pairwise critique** (not just win/loss). The key finding: **negative feedback carries nearly all the signal**. Telling the mutator "what went wrong" is actionable; "what went right" adds nothing.

This aligns with our SDAR gated result (Plan 072): asymmetric trust (endorse positive gaps, attenuate negative) but the **modelless sigmoid gate showed no arena improvement** over rubric gating.

```rust
/// AbsorbCompress variant using pairwise critique for promotion.
///
/// OpenDeepThink insight: negative critique (losses) drives improvement.
/// Positive critique (wins) is statistically indistinguishable from no feedback.
///
/// Our adaptation: aggregate pairwise losses → gap vector → promote only
/// heuristics that address identified weaknesses.
pub struct CritiqueGatedAbsorbCompress {
    inner: Box<dyn AbsorbCompress>,
    /// Only promote when negative critique gap exceeds threshold
    negative_gap_threshold: f32,
}
```

### Existing Alignment

Our stack already implements several OpenDeepThink patterns implicitly:

| OpenDeepThink | Our Existing | Plan |
|---------------|-------------|------|
| Population sampling | DDTree speculative candidates | Core |
| Selection pressure | BanditPruner UCB1 ranking | 030 |
| Evolution loop | G-Zero self-play rounds | 049 |
| Feedback signal | HintDelta (log-prob shift) | 049 |
| Sigmoid gating | SDAR sigmoid gate on δ | 072 |
| Multi-criterion scoring | ROPD rubric vector gaps | 071 |
| Asymmetric trust | SDAR `σ(β·gap)` | 072 |

### Benchmark Prediction

Based on our existing negative results:
- **SDAR modelless sigmoid gating** (Plan 072): ELO 954 ≈ Rubric 955, no arena improvement
- **ROPD rubric modelless** (Plan 071): measurable improvement over baseline
- **GFlowNet modelless** (Plan 052): no DDTree node change, no quality gain

**Prediction:** PairwisePruner + BT ranking is the most likely to show gain because it addresses a **different bottleneck** (selection quality, not reward signal quality). Our negative results were all on reward signal modulation — the selection mechanism itself was unchanged.

---

## Model-Based Path (riir-ai)

### What Maps

```
OpenDeepThink                    Our Stack
─────────────                    ─────────
LLM as judge                →    LeviathanVerifier (LoRA target model, p/q rejection sampling)
LLM as rubric judge         →    RubricReward (LLM rubric + verifier scores for GRPO)
LLM as generator            →    LoRA-adapted Generator (frozen base)
WASM constraint check       →    WasmPruner (domain ground truth — keep as-is)
Pairwise comparison         →    NEW: PairwiseValidator trait
BT ranking                  →    NEW: BT over GRPO group advantage
Feedback-driven mutation    →    NEW: critique-conditioned DPO pairs
Evolution generations       →    GZeroLoop rounds (already exists)
Same model as judge/gen     →    Self-distillation (already our design)
```

### Distillation: BT-Augmented GRPO

Our `GZeroLoop` (Plan 059) uses GRPO with group advantage: generate K rollouts, compute reward, advantage = (r - μ) / σ. OpenDeepThink's BT ranking could replace the reward aggregation:

```rust
/// BT-augmented GRPO: replace scalar reward with BT score.
///
/// Instead of: advantage_i = (r_i - μ) / σ
/// Use:        advantage_i = bt_score_i (from pairwise comparisons)
///
/// OpenDeepThink proves BT internalizes opponent strength at K=4,
/// where sampling noise is non-negligible. Our GRPO uses K=8,
/// but the principle holds: BT > raw win rate for ranking.
pub struct BtGrpoConfig {
    /// Use BT scores instead of raw rewards for advantage computation
    use_bt_ranking: bool,
    /// Per-candidate comparison count (paper: K=4, we use K=8 rollouts)
    comparisons_per_candidate: usize,
    /// BT regularization (paper: λ=0.01)
    bt_lambda: f32,
}
```

### Distillation: PairwiseValidator

The WASM Validator SDK (`riir-validator-sdk`) currently exposes:

```rust
pub trait Validator {
    fn is_valid(&self, input: &str) -> bool;       // binary accept/reject
    fn relevance(&self, input: &str) -> f32;        // pointwise score
}
```

OpenDeepThink suggests adding a **pairwise comparison** method:

```rust
pub trait Validator {
    fn is_valid(&self, input: &str) -> bool;
    fn relevance(&self, input: &str) -> f32;
    
    /// NEW: Pairwise comparison — "which input is better?"
    /// Returns: >0.5 if a preferred, <0.5 if b preferred, 0.5 if tie.
    /// Optional: None means fall back to relevance() comparison.
    fn compare(&self, a: &str, b: &str) -> Option<f32> { None }
}
```

This is a **WASM ABI extension** — requires versioning. Low priority given our negative modelless results, but the interface is cheap to add.

### Distillation: Critique-Conditioned DPO

OpenDeepThink mutates losers using the **natural-language critique** from pairwise comparison. In model-based terms, this is:

```
Standard DPO:  (chosen, rejected) → L_DPO
Critique DPO:  (chosen, rejected, critique) → L_DPO + L_critique-conditioned
```

Our `delta_filter.rs` already produces filtered (chosen, rejected) pairs. Adding critique would mean:

1. During GRPO rollout, pairwise compare candidates
2. Collect loser critiques ("why this lost")
3. Condition DPO rejected samples on critique → model learns to avoid identified failure modes

**Cost:** ~2× comparison budget (K pairs per generation × T generations). OpenDeepThink uses ~285 API calls per problem.

### Integration with Existing Training

```text
Current GZeroLoop (Plan 059):
  Proposer → Generator → HintDelta → GRPO → DPO → LoRA update

With OpenDeepThink distillation:
  Proposer → Generator → HintDelta → PairwiseCompare → BT rank
                                                    ↓
                                          GRPO (BT advantage) → DPO → LoRA update
                                                    ↓
                                          Critique → condition DPO rejected samples
```

---

## Cross-Domain Applicability

OpenDeepThink's HLE experiment (Table 2b) is highly relevant to our multi-domain stack:

| Domain Type | OpenDeepThink Result | Our Analog | Prediction |
|-------------|---------------------|------------|------------|
| Objectively verifiable (math, physics, bio) | +5 to +17 points | Code transpilation, game moves | **Positive** |
| Subjective (humanities, social science) | -25 to -30 points | Advice, creative writing | **Negative** |

This aligns with our architecture: game domains have **ground truth** (win/loss), code has **compiler verdict** (accept/reject). These are exactly where OpenDeepThink excels. For subjective domains, our existing modelless stack (rubric, δ-gating) is more appropriate.

---

## Verdict

### What to Adopt

| Priority | Idea | Target | Effort | Expected Gain |
|----------|------|--------|--------|---------------|
| **P1** | PairwisePruner + BT ranking | microgpt-rs DDTree | 3 days | **High** — addresses selection bottleneck |
| **P2** | BT-augmented GRPO advantage | riir-ai GZeroLoop | 2 days | Medium — replaces scalar reward |
| **P3** | PairwiseValidator compare() | riir-validator-sdk | 1 day | Low — ABI extension, future-proofing |
| **P4** | Critique-conditioned DPO | riir-ai loss pipeline | 3 days | Medium — if P1/P2 show gain |

### ✅ GOAT Proof Passed (Plan 079)

`bt_rank` feature gate — `tests/bench_bt_rank_goat.rs` — 4/4 proofs:

| Proof | Result | Verdict |
|-------|--------|---------|
| BT > Pointwise (true best) | 33.6% vs 23.0%, Δ=+10.6pp | ✅ BT wins |
| BT > Win Rate (Kendall τ) | 0.6354 vs 0.6196 | ✅ BT wins |
| Sparse K=2 top-3 hit | 55.0% ≥ 50% | ✅ Graceful degradation |
| Perfect oracle K=10 | 83.8% > 70% | ✅ Monotonic scaling |

Run: `cargo test --features bt_rank --test bench_bt_rank_goat -- --nocapture`

### What NOT to Adopt

| Idea | Reason |
|------|--------|
| Population size n=20 | Our DDTree already explores many candidates via beam search; fixed population isn't the bottleneck |
| Self-refinement loops | Paper confirms Huang et al.: self-correction without external feedback degrades. Our δ-gating already addresses this |
| K=4 comparisons per candidate | Our GRPO K=8 already exceeds this; BT marginal gain at higher K is small |

### What We Already Do (BT Enhances)

| Existing | How | OpenDeepThink Enhancement |
|----------|-----|--------------------------|
| **LoRA as judge** (`LeviathanVerifier`) | Target model (LoRA-adapted) verifies draft tokens via p/q rejection sampling | Pairwise compare DDTree candidates via LoRA log-probs → BT rank instead of single pass accept/reject |
| **LoRA as reward** (`RubricReward`) | LLM rubric + verifier scores rollouts for GRPO | Pairwise rollout comparison → BT advantage replaces scalar `(student_score - teacher_score) / max` |
| **WASM validators** (`WasmPruner`) | Domain-specific constraint checking (syntax, game rules) | Already ground-truth — BT not applicable here, keep as-is |
| **HintDelta** (`DeltaGatedAbsorbCompress`) | Log-prob shift with/without hint as intrinsic reward | δ is already pairwise-adjacent (comparing two contexts) — BT formalizes the ranking |

**Key correction:** We already have LoRA-as-judge at multiple points in the pipeline. OpenDeepThink's pairwise BT doesn't replace judging — it **aggregates** multiple judgments more principledly than pointwise scoring. Our `LeviathanVerifier` judges one draft at a time; BT would let us judge *pairs* of drafts and rank them globally.

### Key Takeaway

OpenDeepThink's **core insight** — pairwise BT ranking outperforms pointwise scoring for selection — is the highest-value distillation. Our stack has been optimizing **reward signal quality** (δ, rubric gaps, sigmoid gates) but hasn't touched the **selection mechanism** itself. All our negative results (SDAR modelless, GFlowNet) were reward modulation, not selection.

The BT ranking idea should be tested as a new pruner variant that compares DDTree candidates pairwise (using `LeviathanVerifier` log-probs for both candidates) rather than scoring them independently. If it works, it ports naturally to the model-based path (BT advantage for GRPO via `RubricReward`).

**Honest assessment:** Given our string of negative/near-zero results on modelless reward modulation (SDAR ELO 954 ≈ Rubric 955), the selection mechanism is the untested variable. OpenDeepThink provides both the theoretical justification (86% vs 59% accuracy) and the implementation pattern (L-BFGS BT fit) to justify a focused experiment. We already have LoRA-as-judge (`LeviathanVerifier`, `RubricReward`) — the gap is in *how* we aggregate those judgments, not whether we have them.

---

## Relationship to Existing Plans

| Plan | Status | OpenDeepThink Connection |
|------|--------|-------------------------|
| 049 G-Zero | ✅ Phase 1+2 | δ signal is pairwise-adjacent (comparing with/without hint) |
| 052 GFlowNet | ✅ Done | Flow regularization ≈ BT regularization (both penalize extreme scores) |
| 071 ROPD Modelless | ✅ Done | Rubric scoring is pointwise — BT could replace it |
| 072 SDAR Modelless | ✅ Done (negative) | Sigmoid gating is reward modulation, not selection — explains null result |
| 073 SDAR Model-Based | ✅ Done | SDAR gate at gradient level — orthogonal to BT selection |
| **079** BT Selection | ✅ GOAT proof passed | `bt_rank` feature — BT > pointwise (+10.6pp), > win rate (τ +0.016) |

---

## References

- Zhou et al., "OpenDeepThink: Parallel Reasoning via Bradley–Terry Aggregation," arXiv:2605.15177, 2026
- Bradley & Terry, "Rank Analysis of Incomplete Block Designs," Biometrika, 1952
- Singh et al., "V1: Unifying Generation and Self-Verification," arXiv:2603.04304, 2026 (concurrent: pairwise > pointwise)
- Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet," arXiv:2310.01798, 2023