# Plan 121: RMSD — Relevance-Masked Self-Distillation

> **Status:** ❌ NO GOAT — Complete (T1–T13 all done, 46 structural proofs pass), but **negative arena result**: RMSD does not improve over SDAR in tournament play. Demoted to 🪦.
> **Branch:** `develop/feature/121_rmsd_distill`
> **Depends on:** Plan 072 (SDAR gate ✅), Plan 073 (SDAR loss ✅), Plan 074 (Interventional SFT ✅), Plan 080 (BT rank ✅)
> **Research:** `.research/081_RMSD_Relevance_Masked_Self_Distillation.md`
> **Source:** [Relevance-Masked Self-Distillation](https://www.appliedcompute.com/research/relevance-masked-self-distillation) — Applied Compute, 2026
> **Feature gate:** `rmsd_distill` (opt-in, depends on `sdar_gate` + `bandit`)
> **Goal:** Extend SDAR's uniform token gating with RMSD's two-step relevance mask: pre-filter T positions by logprob magnitude, then select S most relevant positions via judge. This concentrates gradient on the ~5-10 positions that actually carry learning signal, yielding 2× data efficiency and higher performance ceiling.

## Tasks

- [x] T1: Implement `LogprobMagnitudeFilter` — top-T positions by |Δlogprob| (Step 1 of RMSD)
- [x] T2: Implement `TopKlApproximator` — top-K=500 vocabulary KL approximation for efficiency
- [x] T3: Implement `rmsd_loss()` in `rmsd_relevance.rs` — combining SDAR gate + RMSD mask
- [x] T4: Implement `RelevanceMask` trait and `MagnitudeJudge` — S-position selection (Step 2 of RMSD)
- [x] T5: Implement `TeacherContinuation` — snapshot student Q → new teacher on plateau
- [x] T6: Implement modelless `RmsdRelevanceFilter` — action-level magnitude pre-filter for katgpt-rs
- [x] T7: Add feature gate `rmsd_distill` to `Cargo.toml` + module registration + bomber player
- [x] T8: Add `bomber_16_rmsd_tournament` example — RMSD vs SDAR vs OPSD vs SFT vs Random
- [x] T9: GOAT proof — RMSD ≥ SDAR on OOD elicitation + capability preservation (1000 rounds, bomber)
- [x] T10: GOAT proof — RMSD with continuation ≥ RMSD without (ablation)
- [x] T11: Benchmark — `bench_rmsd_modelless.rs` throughput test
- [x] T12: Update README.md with RMSD section
- [x] T13: Update research 081 with benchmark results

## Summary

RMSD identifies that ~80% of token positions in self-distillation carry noise, not signal. Where SDAR applies sigmoid gating to ALL tokens, RMSD introduces a two-step precision filter:

1. **Step 1 (Heuristic):** Select T=20 positions with highest |teacher_logprob - student_logprob| magnitude
2. **Step 2 (Judge):** From those T, select S=5 most relevant positions via verifier/judge

Only the S selected positions receive gradient. This composes naturally with SDAR:

```
SDAR: HOW MUCH to trust each token  → σ(β·Δt) modulation
RMSD: WHETHER to train on each token → {0, 1} mask from two-step filter
Combined: loss[t] = sdar_gate(Δt) * is_relevant(t) * reverse_kl[t]
```

**Key results from paper (Qwen3-4B):**
- 2× data efficiency (90 steps vs 150 for OPSD)
- Higher ceiling (PinappleOnly 0.740 vs 0.480)
- Perfect specificity (1.000 — zero off-topic degradation)
- ~5% less wall clock time despite extra judge calls

### What We Already Have (DO NOT reimplement)

| Component | Location | Role |
|-----------|----------|------|
| `sdar_gate()` / `sdar_modulate()` | `src/pruners/sdar_gate.rs` | σ(β·x) sigmoid gate — reuse as modulation layer |
| `sdar_loss()` | `riir-gpu/src/loss_sdar.rs` | Token-level SDAR loss — **extend with RMSD mask** |
| `kl_divergence()` | `riir-gpu/src/distill.rs` | Reverse KL — **extend with top-K approximation** |
| `LossMask` | `riir-gpu/src/training_loop.rs` | Binary token mask — **extend with relevance scoring** |
| `SdarBanditPruner<P>` | `src/pruners/sdar/mod.rs` | Modelless SDAR bandit — **extend with magnitude pre-filter** |
| `RubricReward` | `riir-gpu/src/ropd/` | Rubric scoring — reuse judge infrastructure |
| `VerifierClient` | `riir-gpu/src/ropd/client.rs` | Judge client — **repurpose for token selection** |
| `LeviathanVerifier` | Referenced in research 040 | LoRA-as-Judge — use as token relevance judge |
| `ScreeningPruner` trait | `katgpt-rs-core/src/traits.rs` | Relevance scoring — analogue for modelless path |
| `freeze()`/`thaw()` | `src/pruners/bomber/players.rs` | Bandit knowledge persistence — reuse pattern |
| `loss_masked.wgsl` | `riir-gpu/src/kernels/` | GPU masked loss kernel — **extend with RMSD positions** |
| `GZeroLoop` | `riir-gpu/src/gzero_loop.rs` | Self-play loop — integration point |

### What's New (Implement)

| Component | Description | Location |
|-----------|-------------|----------|
| `LogprobMagnitudeFilter` | Top-T positions by \|Δlogprob\| — Step 1 | `riir-gpu/src/loss_rmsd.rs` |
| `TopKlApproximator` | Top-K=500 vocab KL — efficiency | `riir-gpu/src/loss_rmsd.rs` |
| `RelevanceMask` | Trait for position selection strategies | `riir-gpu/src/loss_rmsd.rs` |
| `JudgeSelectFilter` | Verifier-based S selection — Step 2 | `riir-gpu/src/loss_rmsd.rs` |
| `RmsdConfig` | T, S, K, continuation trigger config | `riir-gpu/src/loss_rmsd.rs` |
| `RmsdMetrics` | Logprob magnitudes, mask density, judge selections | `riir-gpu/src/loss_rmsd.rs` |
| `TeacherContinuation` | Snapshot student → teacher on plateau | `riir-gpu/src/loss_rmsd.rs` |
| `RmsdRelevanceFilter` | Modelless action-level magnitude pre-filter | `src/pruners/rmsd_relevance.rs` |
| `RmsdPlayer` | Bomber arena player with RMSD filtering | `src/pruners/bomber/rmsd_player.rs` |

## Architecture

### Module Structure

```
# riir-ai (model-based)
crates/riir-gpu/src/
├── loss_rmsd.rs              # LogprobMagnitudeFilter, TopKlApproximator, RmsdConfig, rmsd_loss()
├── rmsd_judge.rs             # RelevanceMask trait, JudgeSelectFilter (verifier-based)
├── rmsd_continuation.rs      # TeacherContinuation (LoRA snapshot + plateau detection)
└── kernels/
    └── loss_rmsd.wgsl        # GPU kernel: masked KL on selected positions

# katgpt-rs (modelless)
src/pruners/
├── rmsd_relevance.rs         # RmsdRelevanceFilter — action-level magnitude pre-filter
├── rmsd_gate.rs              # Feature gate re-exports
└── bomber/
    └── rmsd_player.rs        # RmsdPlayer for bomber arena

tests/
├── bench_rmsd_modelless.rs   # Modelless throughput benchmark
└── test_121_rmsd_goat.rs     # GOAT proofs
```

### Key Types

```rust
// riir-gpu/src/loss_rmsd.rs

/// RMSD configuration.
/// Defaults from paper: T=20, S=5, K=500.
pub struct RmsdConfig {
    /// Heuristic pre-filter: select top T positions by |Δlogprob| (paper: T=20).
    pub top_t: usize,
    /// Judge selection: final S positions from T (paper: S=5).
    pub top_s: usize,
    /// Top-K vocabulary approximation for KL (paper: K=500).
    pub top_k: usize,
    /// SDAR sigmoid sharpness β (paper: 5.0, reuse SDAR_BETA).
    pub beta: f32,
    /// SDAR distillation coefficient λ (paper: 0.01, reuse SDAR default).
    pub lambda: f32,
    /// Continuation phase trigger: patience steps without improvement.
    pub plateau_patience: usize,
    /// Whether to use continuation phase (teacher update on plateau).
    pub use_continuation: bool,
}

impl Default for RmsdConfig {
    fn default() -> Self {
        Self {
            top_t: 20,
            top_s: 5,
            top_k: 500,
            beta: 5.0,
            lambda: 0.01,
            plateau_patience: 30,
            use_continuation: true,
        }
    }
}

/// Metrics from RMSD training step.
pub struct RmsdMetrics {
    /// Total positions in rollout.
    pub total_positions: usize,
    /// Positions passing heuristic pre-filter (T).
    pub heuristic_filtered: usize,
    /// Positions selected by judge (S).
    pub judge_selected: usize,
    /// Mean logprob magnitude of selected positions.
    pub mean_selected_magnitude: f32,
    /// Mean logprob magnitude of rejected positions.
    pub mean_rejected_magnitude: f32,
    /// Mask density: S / total_positions.
    pub mask_density: f32,
    /// Raw SDAR loss (all positions).
    pub raw_sdar_loss: f32,
    /// RMSD loss (selected positions only).
    pub rmsd_loss: f32,
}

/// Step 1: Heuristic pre-filter.
/// Selects top-T positions by |teacher_logp - student_logp| magnitude.
pub struct LogprobMagnitudeFilter {
    pub top_t: usize,
}

impl LogprobMagnitudeFilter {
    /// Select positions with highest magnitude logprob difference.
    /// Returns sorted by magnitude descending.
    pub fn filter(&self, teacher_logp: &[f32], student_logp: &[f32]) -> Vec<(usize, f32)> {
        let deltas: Vec<(usize, f32)> = teacher_logp.iter()
            .zip(student_logp.iter())
            .enumerate()
            .map(|(i, (t, s))| (i, (t - s).abs()))
            .filter(|(_, d)| *d > 0.0)
            .collect();

        let mut sorted = deltas;
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.top_t);
        sorted
    }
}

/// Top-K vocabulary KL approximation.
/// Avoids full-vocabulary computation by summing only over top-K student tokens.
pub struct TopKlApproximator {
    pub top_k: usize,
}

impl TopKlApproximator {
    /// Compute KL_topK(p || q) = Σ_{k∈topK(p)} p(k) · log(p(k)/q(k))
    pub fn kl_topk(&self, student_probs: &[f32], teacher_probs: &[f32]) -> f32 {
        // Find top-K student indices
        let mut indexed: Vec<(usize, f32)> = student_probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(self.top_k);

        // Sum KL over top-K only
        let eps = 1e-10;
        indexed.iter()
            .map(|&(i, p)| p * (p / (teacher_probs[i] + eps)).ln())
            .filter(|v| v.is_finite())
            .sum()
    }
}
```

```rust
// riir-gpu/src/rmsd_judge.rs

/// Trait for Step 2: relevance-based position selection.
/// Different judge strategies for selecting S from T positions.
pub trait RelevanceMask: Send + Sync {
    /// Select up to `top_s` positions from `candidates` as most relevant.
    /// `candidates`: (position_idx, magnitude) from Step 1 heuristic filter.
    /// `context`: student prompt, teacher prompt, student rollout for judge.
    fn select(
        &self,
        candidates: &[(usize, f32)],
        context: &JudgeContext,
        top_s: usize,
    ) -> Vec<usize>;
}

/// Context provided to judge for token selection.
pub struct JudgeContext {
    pub student_prompt_tokens: Vec<u32>,
    pub teacher_prompt_tokens: Vec<u32>,
    pub student_rollout_tokens: Vec<u32>,
    pub task_description: String,
}

/// Magnitude-only judge: selects top-S by magnitude from heuristic pre-filter.
/// Simplest strategy — no LLM judge, just takes highest magnitudes.
/// Useful as baseline and for domains where magnitude correlates with relevance.
pub struct MagnitudeJudge;

impl RelevanceMask for MagnitudeJudge {
    fn select(
        &self,
        candidates: &[(usize, f32)],
        _context: &JudgeContext,
        top_s: usize,
    ) -> Vec<usize> {
        candidates.iter()
            .take(top_s)
            .map(|(i, _)| *i)
            .collect()
    }
}

/// Entropy-weighted judge: uses per-position entropy to weight relevance.
/// Based on EGAD (Zhang et al., 2026) — high-entropy positions carry more information.
/// Combines magnitude with entropy: score = magnitude * entropy_weight.
pub struct EntropyWeightedJudge {
    pub entropy_weight: f32, // λ for magnitude vs entropy balance
}

impl RelevanceMask for EntropyWeightedJudge {
    fn select(
        &self,
        candidates: &[(usize, f32)],
        context: &JudgeContext,
        top_s: usize,
    ) -> Vec<usize> {
        // In practice, entropy would be computed from student rollout logits
        // For now, approximate: positions with moderate magnitude + high entropy
        // Full implementation requires logits from forward pass
        candidates.iter()
            .take(top_s)
            .map(|(i, _)| *i)
            .collect()
    }
}

/// Verifier-based judge: uses our LoRA-as-Judge / RubricReward infrastructure.
/// Passes candidate positions to verifier with rubric asking "which positions
//  are most relevant to improving student behavior?"
pub struct VerifierJudge {
    // Uses existing VerifierClient infrastructure from ropd
    pub rubric: String,
}
```

```rust
// riir-gpu/src/rmsd_continuation.rs

/// Teacher continuation: snapshot student → teacher when plateau detected.
/// Simpler than VPD's full EM cycle (no E-step training, just weight copy).
pub struct TeacherContinuation {
    pub plateau_patience: usize,
    pub best_metric: f32,
    pub steps_without_improvement: usize,
    pub teacher_updated: bool,
}

impl TeacherContinuation {
    pub fn new(plateau_patience: usize) -> Self {
        Self {
            plateau_patience,
            best_metric: f32::NEG_INF,
            steps_without_improvement: 0,
            teacher_updated: false,
        }
    }

    /// Check if plateau reached and teacher should be updated.
    /// Returns true when continuation phase should trigger.
    pub fn check_plateau(&mut self, current_metric: f32) -> bool {
        if current_metric > self.best_metric + 1e-6 {
            self.best_metric = current_metric;
            self.steps_without_improvement = 0;
            false
        } else {
            self.steps_without_improvement += 1;
            self.steps_without_improvement >= self.plateau_patience
        }
    }

    /// Reset for next continuation cycle.
    pub fn reset(&mut self) {
        self.best_metric = f32::NEG_INF;
        self.steps_without_improvement = 0;
        self.teacher_updated = false;
    }
}
```

### Core Loss Function

```rust
// riir-gpu/src/loss_rmsd.rs

/// RMSD loss: two-step filtered reverse-KL with SDAR sigmoid gating.
///
/// Pipeline:
///   1. Compute Δlogprob = teacher_logp - student_logp at each position
///   2. SDAR gate: g[t] = σ(β · Δt) — asymmetric trust modulation
///   3. Step 1: Select top-T positions by |Δlogprob| magnitude
///   4. Step 2: Judge selects top-S from T as most relevant
///   5. Loss = (1/|S|) · Σ_{t∈S} g[t] · KL_topK(student[t] || teacher[t])
pub fn rmsd_loss(
    student_logprobs: &[Vec<f32>],  // [seq_len][vocab_size]
    teacher_logprobs: &[Vec<f32>],  // [seq_len][vocab_size]
    config: &RmsdConfig,
    judge: &dyn RelevanceMask,
    context: &JudgeContext,
) -> (f32, RmsdMetrics) {
    let seq_len = student_logprobs.len();
    let approximator = TopKlApproximator { top_k: config.top_k };
    let filter = LogprobMagnitudeFilter { top_t: config.top_t };

    // Compute per-position Δlogprob
    let delta_logp: Vec<f32> = (0..seq_len)
        .map(|t| {
            let student_p = student_logprobs[t].iter().fold(f32::NEG_INFINITY, f32::max);
            let teacher_p = teacher_logprobs[t].iter().fold(f32::NEG_INFINITY, f32::max);
            teacher_p - student_p
        })
        .collect();

    // SDAR sigmoid gate: g[t] = σ(β · Δt)
    let gates: Vec<f32> = delta_logp.iter()
        .map(|&d| sdar_gate(d, config.beta))
        .collect();

    // Step 1: Heuristic pre-filter (top-T by magnitude)
    let candidates = filter.filter(
        &delta_logp.iter().map(|d| d.max(0.0)).collect::<Vec<_>>(),
        &delta_logp.iter().map(|d| d.min(0.0).abs()).collect::<Vec<_>>(),
    );

    // Step 2: Judge selection (top-S from T)
    let selected = judge.select(&candidates, context, config.top_s);

    // Compute RMSD loss on selected positions only
    let mut total_loss = 0.0f32;
    for &t in &selected {
        let kl = approximator.kl_topk(&student_logprobs[t], &teacher_logprobs[t]);
        total_loss += gates[t] * kl;
    }
    let rmsd_loss = if !selected.is_empty() {
        config.lambda * total_loss / selected.len() as f32
    } else {
        0.0
    };

    // Compute metrics
    let selected_magnitudes: Vec<f32> = selected.iter()
        .map(|&t| delta_logp[t].abs())
        .collect();
    let rejected_magnitudes: Vec<f32> = (0..seq_len)
        .filter(|t| !selected.contains(t))
        .map(|t| delta_logp[t].abs())
        .collect();

    let metrics = RmsdMetrics {
        total_positions: seq_len,
        heuristic_filtered: candidates.len(),
        judge_selected: selected.len(),
        mean_selected_magnitude: mean(&selected_magnitudes),
        mean_rejected_magnitude: mean(&rejected_magnitudes),
        mask_density: selected.len() as f32 / seq_len.max(1) as f32,
        raw_sdar_loss: 0.0, // Computed separately for comparison
        rmsd_loss,
    };

    (rmsd_loss, metrics)
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() { 0.0 } else { values.iter().sum::<f32>() / values.len() as f32 }
}
```

### Modelless Architecture (katgpt-rs)

```rust
// src/pruners/rmsd_relevance.rs

/// Modelless relevance filter: action-level analogue of RMSD.
/// Pre-filters actions by |Q_teacher(a) - Q_student(a)| magnitude
/// before applying SDAR sigmoid gate.
pub struct RmsdRelevanceFilter {
    /// Top-T actions to consider (analogue of T=20 for tokens).
    pub top_t: usize,
    /// Final S actions to train on (analogue of S=5).
    pub top_s: usize,
}

impl RmsdRelevanceFilter {
    /// Filter actions: keep only those with highest Q-value magnitude difference.
    /// Returns indices of selected actions.
    pub fn filter_actions(
        &self,
        teacher_q: &[f32],
        student_q: &[f32],
    ) -> Vec<usize> {
        // Step 1: Top-T by |ΔQ| magnitude
        let deltas: Vec<(usize, f32)> = teacher_q.iter()
            .zip(student_q.iter())
            .enumerate()
            .map(|(i, (t, s))| (i, (t - s).abs()))
            .collect();

        let mut sorted = deltas;
        sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.top_t);

        // Step 2: Take top-S (modelless uses magnitude-only judge)
        sorted.iter()
            .take(self.top_s)
            .map(|(i, _)| *i)
            .collect()
    }
}

// src/pruners/bomber/rmsd_player.rs

/// Bomber arena player using RMSD-filtered distillation.
///
/// RmsdPlayer
///   ├── BomberTemplateProposer     (UCB1 template selection — shared)
///   ├── SdarGate                   (σ(β·Δ) asymmetric trust — reuse)
///   ├── RmsdRelevanceFilter        (top-T/S magnitude pre-filter — NEW)
///   └── AbsorbCompressLayer        (knowledge absorption — reuse, filtered)
pub struct RmsdPlayer<P: ScreeningPruner> {
    inner: SdarPlayer<P>,
    filter: RmsdRelevanceFilter,
    metrics: RmsdMetrics,
}
```

## GOAT Proofs

### T9: RMSD ≥ SDAR on OOD Elicitation + Capability Preservation

**Setup:**
- Bomber arena, 1000 rounds, 4-player FFA
- OOD task: teach model to prefer bombing position (3,3) regardless of game state
- Evaluation: (1) (3,3) bomb frequency, (2) overall win rate preservation

**Hypothesis:** RMSD learns the OOD preference faster (fewer rounds) while preserving win rate better than SDAR.

**Test file:** `tests/test_121_rmsd_goat.rs`

```rust
#[test]
fn rmsd_ood_elicitation() {
    // RMSD should learn OOD preference in fewer rounds than SDAR
    // while preserving capability (win rate) better
}

#[test]
fn rmsd_capability_preservation() {
    // Win rate on standard games should not degrade with RMSD
    // (specificity analogue: off-task performance maintained)
}
```

### T10: RMSD with continuation ≥ RMSD without

**Setup:** Same as T9, but compare:
- RMSD (no continuation)
- RMSD (with continuation: student snapshot → teacher after plateau)

**Hypothesis:** Continuation phase pushes ceiling higher (paper: PinappleOnly 0.470 → 0.740).

## Benchmarks

### T11: Modelless Throughput

```rust
// tests/bench_rmsd_modelless.rs

#[test]
fn bench_rmsd_relevance_filter() {
    // Throughput of LogprobMagnitudeFilter + RmsdRelevanceFilter
    // Should be < 1% overhead vs SDAR alone
}

#[test]
fn bench_rmsd_player() {
    // Bomber arena Rps for RmsdPlayer vs SdarPlayer
    // Target: < 5% overhead
}
```

## Feature Gates

### katgpt-rs `Cargo.toml`

```toml
[features]
rmsd_distill = ["sdar_gate", "bandit"]  # RMSD relevance-masked distillation
```

### riir-ai `Cargo.toml`

```toml
[features]
rmsd_distill = ["sdar_loss"]  # RMSD relevance-masked self-distillation loss
```

## Integration with Existing Systems

### Composing RMSD + SDAR

```
Training step:
  1. Student rollout (on-policy)
  2. Teacher forward pass (hint-conditioned)
  3. Compute per-position Δlogprob
  4. SDAR sigmoid gate: σ(β·Δt) for each position
  5. RMSD Step 1: top-T by |Δlogprob| magnitude
  6. RMSD Step 2: judge selects S from T
  7. Loss = λ · Σ_{t∈S} σ(β·Δt) · KL_topK(student[t] || teacher[t])
  8. [Optional] Plateau check → student snapshot → new teacher
```

### Composing RMSD + Interventional SFT

Both operate on token masking but on different axes:
- Interventional SFT: mask agent tokens (who wrote this?)
- RMSD: mask irrelevant positions (is this position informative?)

They compose: `loss[t] = is_world(t) * is_relevant(t) * sdar_gate(Δt) * kl[t]`

### Composing RMSD + VPD

RMSD's continuation phase is a lightweight alternative to VPD's full EM:
- RMSD: snapshot student → teacher (no training)
- VPD: actively train teacher via BCO E-step

If both feature gates enabled, prefer VPD's E-step over RMSD's simple snapshot.

## Hyperparameter Guide

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| T (heuristic filter) | 20 | 10-50 | Higher = more recall, lower precision |
| S (judge selection) | 5 | 3-10 | Higher = more gradient, more noise |
| K (top-K vocab) | 500 | 100-1000 | Higher = more accurate KL, more compute |
| β (SDAR sharpness) | 5.0 | 1.0-10.0 | Reuse SDAR_BETA |
| λ (distill weight) | 0.01 | 0.001-0.1 | Reuse SDAR default |
| plateau_patience | 30 | 10-100 | Steps without improvement before teacher update |

## Expected Outcomes

Based on paper results adapted to our game domains:

| Metric | SDAR (baseline) | RMSD (expected) | Delta |
|--------|----------------|-----------------|-------|
| OOD behavior acquisition | Moderate | Fast | ~2× fewer rounds |
| Capability preservation | Good | Better | Near-zero degradation |
| Training stability | Good | Better | No collapse after teacher update |
| Throughput overhead | Baseline | +1-5% | Judge + filter compute |
| Ceiling (continuation) | Plateau | Higher | Teacher update pushes past plateau |

## File Map

```
# riir-ai (model-based)
crates/riir-gpu/src/
├── loss_rmsd.rs              # NEW: Core RMSD loss, LogprobMagnitudeFilter, TopKlApproximator, RmsdConfig
├── rmsd_judge.rs             # NEW: RelevanceMask trait, MagnitudeJudge, EntropyWeightedJudge, VerifierJudge
├── rmsd_continuation.rs      # NEW: TeacherContinuation plateau detection
├── kernels/
│   └── loss_rmsd.wgsl        # NEW: GPU kernel for masked KL on selected positions
├── lib.rs                    # MOD: Add mod rmsd_judge, rmsd_continuation, loss_rmsd behind feature gate
└── Cargo.toml                # MOD: Add rmsd_distill feature

# katgpt-rs (modelless)
src/pruners/
├── rmsd_relevance.rs         # NEW: RmsdRelevanceFilter (action-level magnitude pre-filter)
├── rmsd_gate.rs              # NEW: Feature gate re-exports
├── mod.rs                    # MOD: Add rmsd_relevance module behind feature gate
├── bomber/
│   └── rmsd_player.rs        # NEW: RmsdPlayer for bomber arena
├── bomber/mod.rs             # MOD: Add rmsd_player module behind feature gate
└── Cargo.toml                # MOD: Add rmsd_distill feature

tests/
├── test_121_rmsd_goat.rs     # NEW: GOAT proofs (T9, T10)
└── bench_rmsd_modelless.rs   # NEW: Throughput benchmarks (T11)

katgpt-rs/examples/
└── bomber_16_rmsd_tournament.rs  # NEW: RMSD vs SDAR vs Random tournament (T8)

# Docs
katgpt-rs/.research/081_RMSD_Relevance_Masked_Self_Distillation.md  # UPDATED: T13
katgpt-rs/README.md                                                  # UPDATED: T12
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Judge quality varies across domains | Start with MagnitudeJudge (no LLM needed), upgrade to VerifierJudge |
| T, S hyperparameters domain-specific | Paper defaults (T=20, S=5) are reasonable starting point |
| LoRA continuation may not work as well as full-weight | Test with bandit Q-value continuation first (modelless) |
| Top-K KL approximation accuracy | K=500 is conservative; can increase if quality drops |
| Game domains may have different noise profile than text | Start with bomber (discrete positions), validate on Go/FFT |
| Feature gate interactions (RMSD + VPD) | Document priority: VPD E-step > RMSD snapshot when both enabled |

## References

- RMSD paper: https://www.appliedcompute.com/research/relevance-masked-self-distillation
- SDAR (our existing): `.research/038_SDAR_Self_Distilled_Agentic_RL.md`
- Interventional SFT (our existing): `.research/043_Interventional_SFT_Causal_Token_Masking.md`
- VPD (our existing): `.research/080_VPD_Variational_Policy_Distillation.md`
- EGAD (entropy-guided distillation): Zhang et al. (2026) arXiv:2605.01732