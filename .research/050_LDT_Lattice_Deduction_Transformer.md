# Research 50: LDT — Lattice Deduction Transformers

> **Paper:** [Lattice Deduction Transformers](https://arxiv.org/pdf/2605.08605) — Liam Davis, Leopold Haller, Alberto Alfarano, Mark Santolucito (Amherst · Axiom · Barnard/Columbia), May 2026
> **Date:** 2026-05, distilled 2026-05
> **Verdict: STRONG CONCEPTUAL ALIGNMENT — LDT's lattice-state search is our existing `ConstraintPruner` + `DDTree` + `MCTS` stack, viewed through abstract interpretation. Three actionable distillations: (1) asymmetric BCE loss for DDTree candidate elimination, (2) conflict detection head for early backtracking in arenas, (3) α-operator for multi-solution supervision in Go/Maze. No new architecture needed — feature-gated enhancements to existing traits.**

---

## 1. TL;DR

LDT trains a tiny recurrent transformer (800K params) to perform **sound deduction** on a lattice:
- Each forward pass eliminates impossible candidates (never eliminates a correct one)
- When deduction stalls, **stochastic branching** guesses and backtracks on conflict
- **α-operator** aggregates all valid solutions still consistent with current state → domain-agnostic supervision target
- **On-policy training**: same Solve loop at train and inference time
- 100% on Sudoku-Extreme, Snowflake Sudoku; 99.9% on Maze-Hard (frontier LLMs = 0%)

The key insight for us: **LDT's lattice is our `ConstraintPruner` generalized**. We already have sound pruning (binary valid/invalid), search trees (DDTree, MCTS), backtracking (Sudoku solver, Go), and multi-solution settings (maze shortest paths, Go openings). LDT adds three things we don't have:

1. **Asymmetric loss** — false elimination penalized 8× more than false retention (w+/w− = 8)
2. **Explicit conflict head** — separate sigmoid for "this state is unsatisfiable"
3. **α-operator** — `x ⊓ α({y ∈ Y | y consistent with x})` as on-policy training target

---

## 2. What LDT Actually Does

### 2.1 Architecture

```
Input lattice state (multi-hot: |V| sigmoids per cell)
    │
    ▼
┌─────────────────────────────────┐
│  Recurrent Transformer          │
│  4 attention layers × 16 loops  │
│  d=128, 4 heads, FFN=4×        │
│  ~800K params                   │
│                                 │
│  Per-iteration outputs:         │
│    b(ℓ) = candidate logits      │
│    c(ℓ) = conflict logit (CLS)  │
└──────────┬──────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  Step Operator (Algorithm 2)     │
│                                 │
│  1. Eliminate candidates < θ     │
│  2. Check conflict (CLS or ∅)   │
│  3. Check solved (singleton)     │
│  4. If neither: branch           │
│     - Pick random multi-candidate│
│     - Pin to softmax sample      │
└──────────┬──────────────────────┘
           │
           ▼
   Updated lattice state (fewer candidates)
```

### 2.2 The Lattice Structure

The abstract domain is a **grid powerset lattice** `A = {1,...,k} → P(V)`:
- Each position tracks a set of still-viable candidates
- `⊤` = all candidates alive (no information)
- `⊥` = some cell has empty candidate set (inconsistency)
- Order: pointwise inclusion (`a ⊑ b iff a(i) ⊆ b(i) for all i`)
- Meet: pointwise intersection
- Join: pointwise union

**Sound deduction**: `ded_p(a) = α(γ(a) ∩ ||p||)` — keep only candidates that survive in at least one valid solution. Never removes a correct candidate, but may fail to remove incorrect ones (incomplete).

### 2.3 The α-Operator (Key Innovation)

For training target, given solution set Y and current state x:

```
ŷ = x ⊓ α({y ∈ Y | y consistent with x})
```

This is: **intersect current state with the abstraction of all solutions still consistent with it**. As the state commits, the consistent subset shrinks and α sharpens. Multi-solution problems (K>1) get progressively tighter supervision without changing architecture.

For Sudoku (K=1): α reduces to single ground truth. For Maze (K=512): α aggregates 512 shortest paths, tightening as branching decisions eliminate alternatives.

### 2.4 The Three Loss Terms

Per internal iteration ℓ (all 16 supervised equally):

| Loss | Target | Purpose |
|------|--------|---------|
| L_BCE(ℓ) | Asymmetric BCE on σ(b(ℓ)) vs ŷ | Sound candidate elimination (w+/w− = 8) |
| L_CLS(ℓ) | Symmetric BCE on σ(c(ℓ)) vs 1[ŷ=⊥] | Conflict detection |
| L_CE(ℓ) | Per-cell softmax CE on b(ℓ) at singleton cells | Faster convergence |

**Total**: `L = (1/L) Σ [L_BCE + 0.1·L_CLS + 0.2·L_CE]`

The asymmetric BCE is critical: false elimination (removing a correct candidate) is penalized 8× more than false retention (keeping a wrong candidate). This makes the model **conservative** — it only eliminates when confident.

### 2.5 Key Results

| Model | Params | Sudoku-Extreme | Snowflake | Maze-Hard | Training |
|-------|--------|----------------|-----------|-----------|----------|
| Claude Opus 4.6 | ? | 0% | 0% | 0% | — |
| GPT-5.4 | ? | 0% | 0% | 0% | — |
| HRM | 27M | 55% | — | 74.5% | — |
| TRM | 5M | 87.4% | — | 85.3% | 36h (4×L40S) |
| Sotaku | 800K | 98.9% | — | — | 2h40m (1×H100) |
| **LDT** | **800K** | **100%** | **100%** | **99.9%** | **15m (1×B200)** |

---

## 3. Mapping to Our Architecture

### 3.1 Structural Equivalence

| LDT Concept | Our Equivalent | Status |
|-------------|----------------|--------|
| Grid powerset lattice `A` | `ConstraintPruner::is_valid()` + `ScreeningPruner::relevance()` | ✅ Already have |
| Sound deduction operator | DDTree with pruning | ✅ Already have |
| Stochastic branching | DDTree expansion + sampling | ✅ Already have |
| Backtracking on conflict | MCTS + Go backtracking | ✅ Already have |
| Recurrent transformer | HLA recurrent attention (Plan 057) | ✅ Already have |
| α-operator (multi-solution) | K-sample shortest paths in Maze, Go openings | ⚠️ Partial — need α loss target |
| Asymmetric BCE loss | Standard CE loss only | ❌ Not implemented |
| Conflict detection head | No separate conflict signal | ❌ Not implemented |
| On-policy Solve loop | G-Zero self-play loop (Plan 049) | ✅ Already have |
| Parallel solve (M slots × K chains) | DDTree batch + MCTS parallel | ✅ Already have |

### 3.2 The Core Mapping: Lattice = Pruner Stack

LDT's lattice deduction is literally what our pruner stack does:

```rust
// LDT: "eliminate every candidate whose confidence falls below θ_elim"
// Us:
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
}

// LDT: "graded confidence per candidate"
// Us:
pub trait ScreeningPruner: Send + Sync {
    fn relevance(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> f32;
}
```

LDT uses continuous sigmoids per candidate (like `ScreeningPruner::relevance()` returns `f32`), then thresholds at θ_elim. Our DDTree already does this — `ScreeningPruner` scores are blended into log-probs, and low-relevance tokens are pruned.

**The mapping is direct**: LDT's per-cell candidate sigmoids = our per-token `ScreeningPruner::relevance()`. LDT's θ_elim = our pruning threshold. LDT's conflict CLS = we don't have this (new).

### 3.3 What's Genuinely New From LDT

#### N1: Asymmetric BCE Loss

LDT's w+/w− = 8 means: "punish removing a correct candidate 8× harder than keeping a wrong one." This makes the model **sound but incomplete** — it prefers to leave candidates alive rather than risk eliminating a correct one.

**Our gap**: DDTree uses standard log-prob sampling. No asymmetric penalty for pruning correct tokens vs keeping wrong ones.

**Where it applies**: Any DDTree + `ConstraintPruner` combination. The `ScreeningPruner::relevance()` score could be trained with asymmetric loss to be more conservative.

**Modelless path**: Pre-compute the asymmetry as a threshold adjustment. If w+/w− = 8, set θ_elim = 1/(1+8) ≈ 0.11. No training needed — just tune the existing pruning threshold.

**Model-based path**: Train the `ScreeningPruner` with asymmetric BCE in riir-gpu (WGSL kernel change).

#### N2: Conflict Detection Head

LDT has a separate CLS sigmoid that fires when the state is unsatisfiable (⊥). This enables **early backtracking** — instead of exploring a dead branch until it runs out of candidates, the conflict head says "this branch is hopeless, backtrack now."

**Our gap**: DDTree and MCTS only detect dead ends by exhausting candidates or hitting terminal states. No explicit "this subtree is doomed" signal.

**Where it applies**: MCTS in Go (early cutoff of losing branches), DDTree in speculative decoding (early rejection of hopeless drafts), Bomber arenas (early detection of unwinnable states).

**Modelless path**: `EntropyAnomalyDetector` (Plan 061) already flags high-entropy states. Repurpose as conflict signal — if entropy exceeds threshold, treat as conflict and prune.

**Model-based path**: Add a conflict head to the transformer output in riir-engine. Single sigmoid, trained with BCE against "is this state terminal/losing."

#### N3: α-Operator for Multi-Solution Supervision

LDT's α-operator aggregates valid solutions consistent with current state:

```
ŷ = x ⊓ α({y ∈ Y | y consistent with x})
```

This gives **progressively tighter supervision** as search commits to particular branches. At the start (⊤), all solutions are alive and α is broad. As branching eliminates alternatives, α narrows to the surviving solutions.

**Our gap**: Our modelless distillation (Plan 052 GFlowNet, Plan 053 δ-Mem, Plan 071 ROPD, Plan 072 SDAR) uses single-solution or fixed targets. No progressive tightening based on search state.

**Where it applies**: 
- Go MCTS: multiple joseki variations are valid from a position; α would narrow as the game commits
- Maze shortest paths: K=512 paths narrow as branching commits
- DDTree speculative decoding: multiple valid continuations narrow as tokens commit

**Modelless path**: Pre-compute K solutions per puzzle, filter by consistency at each step, use intersection as target. This is what the Sudoku solver already does implicitly.

**Model-based path**: Add α-operator to LoRA training targets in riir-gpu. For Go: sample K professional games from same position, compute α at each move.

---

## 4. Honest Gap Analysis

### 4.1 What We Already Cover

| LDT Feature | Our Coverage | Evidence |
|-------------|-------------|----------|
| Sound pruning | `ConstraintPruner` trait | Plan 007, 021 |
| Graded relevance | `ScreeningPruner` trait | Plan 021 |
| Search + backtracking | DDTree + MCTS | Plan 005, 056 |
| Multi-solution settings | Maze shortest paths, Go openings | Plan 056, 065 |
| On-policy training | G-Zero self-play loop | Plan 049 |
| Recurrent transformer | HLA (Plan 057), D2F recurrent denoising | Plan 057, 066 |
| Parallel inference | DDTree batch, MCTS parallel | Plan 013, 056 |
| Train/test compute tradeoff | Bandit explore/exploit (Plan 030) | Plan 030 |

### 4.2 What's Genuinely Missing

| Gap | Priority | Complexity | Expected Gain |
|-----|----------|------------|---------------|
| Asymmetric pruning threshold | P1 | Low — tune θ_elim | ~10-20% fewer false prunes in DDTree |
| Conflict detection via entropy | P1 | Low — reuse Plan 061 | Early backtracking in MCTS/arenas |
| α-operator for Go training | P2 | Medium — K-sample + intersection | Tighter LoRA supervision targets |
| Asymmetric BCE in WGSL | P2 | Medium — kernel change | Better ScreeningPruner training |
| Conflict head in transformer | P3 | High — architecture change | Requires riir-engine modification |

### 4.3 What We Don't Need

| LDT Feature | Why Skip |
|-------------|----------|
| Lattice encoding layer | Our `ConstraintPruner`/`ScreeningPruner` already encode domain constraints as Rust traits, not neural tensors |
| Sigmoid per candidate | We use log-prob space (more numerically stable) |
| 16-iteration recurrent loop | Our HLA already has recurrent state; D2F already has multi-step denoising |
| Dataset-level symmetry augmentation | We already have this for Go (rotations/reflections) and Sudoku (digit permutations) |
| Per-step symmetry wrapping at inference | Already explored in Plan 083 PTRM — found not worth the overhead for our scale |
| Variable-topology covering grid | Not applicable to our fixed-vocabulary transformer |

---

## 5. Distillation Strategy

### 5.1 Modelless Path (katgpt-rs, feature-gated)

All behind `lattice_deduction` feature gate:

#### T1: Asymmetric Pruning Threshold

No training needed. Adjust existing DDTree pruning threshold based on LDT's insight:

```rust
/// Asymmetric pruning threshold derived from LDT's w+/w− = 8.
/// θ_elim = 1/(1 + w+/w−) ≈ 0.111
/// This makes pruning conservative: only eliminate when very confident.
pub const LDT_THETA_ELIM: f32 = 1.0 / (1.0 + 8.0);
```

Apply in DDTree expansion: instead of pruning at some fixed relevance threshold, use θ_elim. This is a one-line config change.

**Proof**: Run Sudoku speculative solve with default threshold vs LDT threshold. Measure: (a) solve rate, (b) false prune rate (correct tokens eliminated).

#### T2: Conflict Detection via Entropy

Reuse `EntropyAnomalyDetector` (Plan 061) as conflict signal:

```rust
/// LDT-style conflict detection using entropy anomaly.
/// When per-position entropy drops below threshold, the state is "conflicted"
/// (too many candidates eliminated → likely wrong path).
pub trait ConflictDetector: Send + Sync {
    /// Returns true if state is likely unsatisfiable (should backtrack).
    fn is_conflicted(&self, logits: &[f32], depth: usize) -> bool;
}
```

Wire into DDTree step: after pruning, check conflict. If conflicted, don't expand further — backtrack to parent.

**Proof**: Run MCTS Go games with and without early conflict cutoff. Measure: (a) average search depth before backtracking, (b) win rate vs random.

#### T3: α-Operator for Multi-Target Distillation

For domains with multiple valid solutions (maze, Go):

```rust
/// LDT α-operator: intersect current state with abstraction of consistent solutions.
/// ŷ = x ⊓ α({y ∈ Y | y consistent with x})
///
/// Returns progressively tighter target as search commits.
pub fn alpha_intersect(
    current_state: &[u16],      // current candidate sets (bitfield per position)
    solutions: &[[u16]],        // K pre-computed valid solutions
) -> Vec<u16> {
    let consistent: Vec<_> = solutions.iter()
        .filter(|sol| is_consistent(current_state, sol))
        .collect();
    
    current_state.iter().zip(/* per-position union of consistent solutions */)
        .map(|(current, alpha)| current & alpha)  // ⊓ = intersection
        .collect()
}
```

**Proof**: Generate K=16 shortest paths for 15×15 maze. Run DDTree with α-target vs single-target. Measure convergence speed.

### 5.2 Model-Based Path (riir-ai, feature-gated)

Behind `lattice_deduction` feature gate in riir-gpu and riir-engine:

#### T4: Asymmetric BCE WGSL Kernel

Add asymmetric weights to existing `cross_entropy.wgsl`:

```wgsl
// LDT asymmetric BCE: w+ * log(σ) + w- * log(1-σ)
// w+/w− = 8 → penalize false elimination 8× harder
fn asymmetric_bce(pred: f32, target: f32, w_pos: f32, w_neg: f32) -> f32 {
    let eps = 1e-7;
    let p = clamp(pred, eps, 1.0 - eps);
    if target > 0.5 {
        -w_pos * log(p)
    } else {
        -w_neg * log(1.0 - p)
    }
}
```

Use for LoRA training of ScreeningPruner relevance scores. Train on Go positions: positive = moves from professional games, negative = illegal/losing moves.

**Proof**: Train LoRA with symmetric vs asymmetric BCE on Go positions. Measure: (a) false prune rate on validation set, (b) move accuracy in arena.

#### T5: Conflict Head for Game State Evaluation

Add a conflict/loss head to the transformer in riir-engine:

```rust
/// LDT-style conflict detection head.
/// Single sigmoid output: 1.0 = state is unsatisfiable (losing/unwinnable).
#[cfg(feature = "lattice_deduction")]
pub struct ConflictHead {
    /// Linear projection from hidden dim → 1
    weight: Tensor, // [1, hidden_dim]
    bias: Tensor,   // [1]
}
```

Train with BCE against game outcomes. At inference, use for early MCTS pruning.

**Proof**: Add conflict head to Go LoRA training. Measure: (a) conflict accuracy on held-out games, (b) MCTS speedup from early cutoff.

---

## 6. Feature Gate Design

```toml
# katgpt-rs/Cargo.toml
[features]
# LDT-inspired lattice deduction enhancements
lattice_deduction = []  # Enables:
                        # - LDT_THETA_ELIM asymmetric threshold
                        # - ConflictDetector trait + entropy impl
                        # - alpha_intersect multi-solution operator
                        # - Sudoku/Maze/Go proof benchmarks
```

```toml
# riir-ai/crates/riir-gpu/Cargo.toml
[features]
lattice_deduction = []  # Enables:
                        # - asymmetric_bce WGSL kernel
                        # - alpha_operator training target
                        # - conflict_head BCE loss
```

```toml
# riir-ai/crates/riir-engine/Cargo.toml
[features]
lattice_deduction = []  # Enables:
                        # - ConflictHead struct
                        # - forward_with_conflict() method
```

**Zero impact on existing code**: all new code behind feature gates. Default build unchanged.

---

## 7. Cross-Reference with Existing Research

| Existing Research | LDT Connection |
|-------------------|----------------|
| R21 G-Zero | LDT's on-policy Solve loop = G-Zero's self-play loop; δ signal is modelless α-operator analog |
| R34 D2F | LDT's iterative refinement = D2F's block denoising; both project through discrete states |
| R35 Attractor | LDT's fixed-point on lattice = Attractor's fixed-point in latent space |
| R37 REAP | LDT's model-based/modelless = our ConstraintPruner (modelless) → ScreeningPruner (model-based) spectrum |
| R36 ROPD | LDT's α-operator = ROPD's rubric scoring; both are state-dependent supervision targets |
| R49 PTRM | LDT's recurrent transformer = PTRM's looped model; both use width > depth |
| R48 HRM-Text | LDT's recurrent loops = HRM's H-cycles; both unroll fixed depth |

---

## 8. Key Insight: Why LDT Works (And What It Means For Us)

LDT's success comes from **one design choice**: the model operates on an interpretable intermediate representation (the lattice) rather than a latent space. This means:

1. **Soundness is structural**: the lattice guarantees you never eliminate a correct candidate, regardless of model quality
2. **Search is cheap**: backtracking is just "restore previous lattice state" — no gradient, no KV cache
3. **Training target is domain-agnostic**: α-operator works for any lattice, not just Sudoku
4. **Train/test compute tradeoff**: more training → better deduction → less search → faster inference

For us, this validates our **trait-based architecture**: `ConstraintPruner` and `ScreeningPruner` are our lattice. They guarantee soundness structurally (Rust type system), not statistically (model confidence). LDT's contribution is showing that **training the pruner itself** (not just the transformer) with asymmetric loss + α-targets yields better search.

---

## 9. Verdict and Priority

### 9.1 Verdict: STRONG VALIDATION + 3 ACTIONABLE DISTILLATIONS

LDT validates our existing architecture:
- ✅ Trait-based sound deduction (ConstraintPruner = lattice)
- ✅ Search + backtracking (DDTree + MCTS = Solve loop)
- ✅ On-policy training (G-Zero = on-policy Solve)
- ✅ Train/test compute tradeoff (Bandit = explore/exploit)

LDT adds 3 new techniques we should adopt:
1. **P1 — Asymmetric threshold** (modelless, zero training, feature-gated)
2. **P1 — Conflict detection via entropy** (modelless, reuse Plan 061, feature-gated)
3. **P2 — α-operator for multi-solution** (model-based, LoRA training target, feature-gated)

### 9.2 Action Items

| Task | Project | Feature Gate | Priority | Effort |
|------|---------|-------------|----------|--------|
| T1: LDT θ_elim threshold | katgpt-rs | `lattice_deduction` | P1 | 1 day |
| T2: ConflictDetector trait | katgpt-rs | `lattice_deduction` | P1 | 2 days |
| T3: alpha_intersect | katgpt-rs | `lattice_deduction` | P2 | 2 days |
| T4: Asymmetric BCE WGSL | riir-gpu | `lattice_deduction` | P2 | 1 day |
| T5: ConflictHead | riir-engine | `lattice_deduction` | P3 | 3 days |

### 9.3 What NOT To Do

1. ❌ Don't implement lattice encoding as neural tensor — our trait-based approach is better (type-safe, zero-cost)
2. ❌ Don't build a recurrent transformer for Sudoku — we already have a backtracking solver that's faster
3. ❌ Don't replace DDTree with LDT-style Solve loop — DDTree already does the same thing with different terminology
4. ❌ Don't add sigmoid-per-candidate encoding — our log-prob space is numerically superior
5. ❌ Don't train an 800K param model from scratch — our modelless path is cheaper

---

## 10. Paper Metadata

- **arXiv**: 2605.08605v1
- **Date**: May 2026
- **Code**: Not released
- **Key benchmarks**: Sudoku-Extreme, Snowflake Sudoku, Maze-Hard
- **Related work**: TRM (R10), HRM (R9), Sotaku (R11) — all in our research corpus
- **Cross-references**: R21 (G-Zero δ ≈ α), R34 (D2F ≈ iterative refinement), R35 (Attractor ≈ fixed-point), R37 (REAP ≈ model-based/modelless), R49 (PTRM ≈ recurrent)

---

## Appendix A: LDT Hyperparameters (Reference)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| d (embed dim) | 128 | Sudoku/Snowflake |
| d (embed dim) | 192 | Maze-Hard |
| Layers | 4 | Attention layers |
| Heads | 4 | Per layer |
| Loops | 16 | Internal iterations per forward pass |
| FFN mult | 4.0 | Feed-forward expansion |
| w+/w− | 8.0 | Asymmetric BCE weights |
| λ_cls | 0.1 | CLS loss weight |
| λ_ce | 0.2 | CE loss weight |
| θ_elim | ~0.11 | Candidate elimination threshold |
| θ_cls (eval) | 0.6 | Conflict threshold at inference |
| τ_decide | 1.5 | Branching temperature |
| K (maze) | 512 | Multi-solution samples |
| Batch size | 512 | Parallel solve |
| M slots × K chains | 8 × 64 | Inference parallelism |