//! RMSD — Relevance-Masked Self-Distillation for modelless stack.
//!
//! Extends SDAR's uniform token gating with a two-step relevance mask:
//! 1. Pre-filter T actions by |Q_teacher - Q_student| magnitude
//! 2. Select S most relevant actions (magnitude-only judge in modelless path)
//!
//! Only S selected actions receive SDAR sigmoid gating.
//! This concentrates learning signal on the actions that carry actual information.
//!
//! # Architecture
//!
//! ```text
//! RmsdRelevanceFilter
//!   ├── Step 1: Top-T by |ΔQ| magnitude  (LogprobMagnitudeFilter)
//!   ├── Step 2: Top-S from T             (MagnitudeJudge)
//!   └── Metrics                          (RmsdMetrics)
//!
//! TeacherContinuation
//!   └── Snapshot student → teacher on plateau
//! ```
//!
//! # Key Insight
//!
//! SDAR applies uniform sigmoid gating to all positions. RMSD adds a relevance
//! pre-filter so that only the ~5-10 positions carrying actual signal receive
//! gated updates. This prevents noise positions from diluting the learning signal.
//!
//! **Source:** Applied Compute (2026). Relevance-Masked Self-Distillation.
//!
//! Plan 125: RMSD relevance-masked self-distillation.
//! Feature gate: `rmsd_distill`

use std::cmp::Ordering;

// ── Config ──────────────────────────────────────────────────────

/// RMSD configuration for modelless path.
/// Defaults from paper: T=20 (top actions), S=5 (final selection).
#[derive(Clone, Debug)]
pub struct RmsdConfig {
    /// Heuristic pre-filter: select top T actions by |ΔQ| magnitude (paper: T=20).
    pub top_t: usize,
    /// Judge selection: final S actions from T (paper: S=5).
    pub top_s: usize,
}

impl Default for RmsdConfig {
    fn default() -> Self {
        Self {
            top_t: 20,
            top_s: 5,
        }
    }
}

// ── Metrics ─────────────────────────────────────────────────────

/// Metrics from RMSD filtering step.
#[derive(Clone, Debug, Default)]
pub struct RmsdMetrics {
    /// Total actions considered.
    pub total_actions: usize,
    /// Actions passing heuristic pre-filter (T).
    pub heuristic_filtered: usize,
    /// Actions selected by judge (S).
    pub judge_selected: usize,
    /// Mean |ΔQ| magnitude of selected actions.
    pub mean_selected_magnitude: f32,
    /// Mean |ΔQ| magnitude of rejected actions.
    pub mean_rejected_magnitude: f32,
    /// Mask density: S / total_actions.
    pub mask_density: f32,
}

// ── LogprobMagnitudeFilter ──────────────────────────────────────

/// Step 1: Heuristic pre-filter by magnitude.
/// Selects top-T positions by |teacher_logprob - student_logprob| magnitude.
#[derive(Clone, Debug)]
pub struct LogprobMagnitudeFilter {
    /// Number of top positions to keep.
    pub top_t: usize,
}

impl LogprobMagnitudeFilter {
    /// Create a new filter with given T.
    pub fn new(top_t: usize) -> Self {
        Self { top_t }
    }

    /// Select positions with highest magnitude difference.
    /// Returns sorted by magnitude descending: (index, magnitude).
    pub fn filter(&self, teacher_logp: &[f32], student_logp: &[f32]) -> Vec<(usize, f32)> {
        let mut deltas: Vec<(usize, f32)> = teacher_logp
            .iter()
            .zip(student_logp.iter())
            .enumerate()
            .map(|(i, (t, s))| (i, (t - s).abs()))
            .filter(|(_, d)| *d > 0.0)
            .collect();

        deltas.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        deltas.truncate(self.top_t);
        deltas
    }
}

// ── TopKlApproximator ───────────────────────────────────────────

/// Top-K vocabulary KL approximation for efficiency.
/// Avoids full-vocabulary computation by summing only over top-K tokens.
#[derive(Clone, Debug)]
pub struct TopKlApproximator {
    /// Number of top vocabulary items to consider.
    pub top_k: usize,
}

impl TopKlApproximator {
    /// Create a new approximator with given K.
    pub fn new(top_k: usize) -> Self {
        Self { top_k }
    }

    /// Compute KL_topK(p || q) = Σ_{k∈topK(p)} p(k) · log(p(k)/q(k))
    pub fn kl_topk(&self, student_probs: &[f32], teacher_probs: &[f32]) -> f32 {
        let len = student_probs.len().min(teacher_probs.len());
        if len == 0 {
            return 0.0;
        }

        // Find top-K student indices
        let mut indexed: Vec<(usize, f32)> = student_probs[..len]
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        indexed.truncate(self.top_k);

        let mut kl = 0.0f32;
        for (idx, p) in &indexed {
            if *p <= 0.0 {
                continue;
            }
            let q = teacher_probs.get(*idx).copied().unwrap_or(f32::EPSILON);
            let q_safe = q.max(f32::EPSILON);
            kl += p * (p / q_safe).ln();
        }
        kl
    }
}

// ── RelevanceMask Trait ─────────────────────────────────────────

/// Trait for position/action selection strategies.
pub trait RelevanceMask {
    /// Select S positions from T candidates.
    /// Returns indices of selected positions.
    fn select(&self, candidates: &[(usize, f32)], total: usize) -> Vec<usize>;
}

/// Magnitude-only judge: select top-S by |ΔQ| magnitude (modelless path).
#[derive(Clone, Debug)]
pub struct MagnitudeJudge {
    /// Number of final selections.
    pub top_s: usize,
}

impl MagnitudeJudge {
    /// Create a new magnitude judge with given S.
    pub fn new(top_s: usize) -> Self {
        Self { top_s }
    }
}

impl RelevanceMask for MagnitudeJudge {
    fn select(&self, candidates: &[(usize, f32)], _total: usize) -> Vec<usize> {
        candidates
            .iter()
            .take(self.top_s)
            .map(|(i, _)| *i)
            .collect()
    }
}

// ── RmsdRelevanceFilter ─────────────────────────────────────────

/// Modelless relevance filter: action-level analogue of RMSD.
/// Pre-filters actions by |Q_teacher(a) - Q_student(a)| magnitude
/// before applying SDAR sigmoid gate.
///
/// Two-step process:
/// 1. Top-T by |ΔQ| magnitude (heuristic pre-filter)
/// 2. Top-S from T (magnitude-only judge for modelless path)
#[derive(Clone, Debug)]
pub struct RmsdRelevanceFilter {
    /// Top-T actions to consider (Step 1).
    pub top_t: usize,
    /// Final S actions to train on (Step 2).
    pub top_s: usize,
}

impl RmsdRelevanceFilter {
    /// Create a new filter with given T and S.
    pub fn new(top_t: usize, top_s: usize) -> Self {
        Self { top_t, top_s }
    }

    /// Filter actions: keep only those with highest Q-value magnitude difference.
    /// Returns indices of selected actions and metrics.
    pub fn filter_actions(
        &self,
        teacher_q: &[f32],
        student_q: &[f32],
    ) -> (Vec<usize>, RmsdMetrics) {
        let total = teacher_q.len().min(student_q.len());

        // Step 1: Top-T by |ΔQ| magnitude (filter out zero deltas)
        let mut deltas: Vec<(usize, f32)> = teacher_q[..total]
            .iter()
            .zip(student_q[..total].iter())
            .enumerate()
            .map(|(i, (t, s))| (i, (t - s).abs()))
            .filter(|(_, d)| *d > 0.0)
            .collect();

        deltas.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let heuristic_filtered = deltas.len().min(self.top_t);
        let candidates = &deltas[..heuristic_filtered];

        // Step 2: Top-S (magnitude-only judge)
        let selected: Vec<usize> = candidates
            .iter()
            .take(self.top_s)
            .map(|(i, _)| *i)
            .collect();

        // Compute metrics
        let selected_magnitudes: Vec<f32> = selected
            .iter()
            .filter_map(|&i| deltas.iter().find(|(idx, _)| *idx == i).map(|(_, m)| *m))
            .collect();

        let rejected_magnitudes: Vec<f32> = deltas
            .iter()
            .filter(|(idx, _)| !selected.contains(idx))
            .map(|(_, m)| *m)
            .collect();

        let mean_selected = if selected_magnitudes.is_empty() {
            0.0
        } else {
            selected_magnitudes.iter().sum::<f32>() / selected_magnitudes.len() as f32
        };

        let mean_rejected = if rejected_magnitudes.is_empty() {
            0.0
        } else {
            rejected_magnitudes.iter().sum::<f32>() / rejected_magnitudes.len() as f32
        };

        let metrics = RmsdMetrics {
            total_actions: total,
            heuristic_filtered,
            judge_selected: selected.len(),
            mean_selected_magnitude: mean_selected,
            mean_rejected_magnitude: mean_rejected,
            mask_density: if total > 0 {
                selected.len() as f32 / total as f32
            } else {
                0.0
            },
        };

        (selected, metrics)
    }
}

// ── TeacherContinuation ─────────────────────────────────────────

/// Snapshot student → teacher on plateau detection.
/// In modelless path: snapshot student Q-values as new teacher reference
/// when metric plateaus for `patience` steps.
#[derive(Clone, Debug)]
pub struct TeacherContinuation {
    /// Steps without improvement before snapshot.
    pub plateau_patience: usize,
    /// Best metric seen so far.
    best_metric: f32,
    /// Steps since last improvement.
    steps_without_improvement: usize,
    /// Whether teacher has been updated.
    teacher_updated: bool,
}

impl TeacherContinuation {
    /// Create a new continuation tracker.
    pub fn new(plateau_patience: usize) -> Self {
        Self {
            plateau_patience,
            best_metric: f32::NEG_INFINITY,
            steps_without_improvement: 0,
            teacher_updated: false,
        }
    }

    /// Check if metric has plateaued. Returns true if teacher should be updated.
    pub fn check_plateau(&mut self, metric: f32) -> bool {
        if metric > self.best_metric {
            self.best_metric = metric;
            self.steps_without_improvement = 0;
            self.teacher_updated = false;
            return false;
        }

        self.steps_without_improvement += 1;
        if self.steps_without_improvement >= self.plateau_patience && !self.teacher_updated {
            self.teacher_updated = true;
            return true;
        }
        false
    }

    /// Reset continuation state.
    pub fn reset(&mut self) {
        self.best_metric = f32::NEG_INFINITY;
        self.steps_without_improvement = 0;
        self.teacher_updated = false;
    }

    /// Whether teacher was updated.
    pub fn was_updated(&self) -> bool {
        self.teacher_updated
    }

    /// Best metric seen.
    pub fn best_metric(&self) -> f32 {
        self.best_metric
    }
}

// ── Core Loss Function ──────────────────────────────────────────

/// Compute RMSD loss combining SDAR gate + relevance mask.
///
/// `rmsd_loss = Σ_{i∈selected} sdar_gate(Δ_i) * reverse_kl_i`
///
/// In modelless path, this operates on action-level Q-values:
/// - Δ_i = teacher_q[i] - student_q[i]
/// - reverse_kl_i approximated by |Δ_i|
/// - sdar_gate from existing `sdar_gate` module
pub fn rmsd_loss(selected: &[usize], teacher_q: &[f32], student_q: &[f32], beta: f32) -> f32 {
    if selected.is_empty() {
        return 0.0;
    }

    let mut total = 0.0f32;
    for &i in selected {
        let teacher_val = teacher_q.get(i).copied().unwrap_or(0.0);
        let student_val = student_q.get(i).copied().unwrap_or(0.0);
        let gap = teacher_val - student_val;

        // SDAR sigmoid gate
        let gate = {
            let z = beta * gap;
            if z >= 0.0 {
                1.0 / (1.0 + (-z).exp())
            } else {
                let ez = z.exp();
                ez / (1.0 + ez)
            }
        };

        // Reverse KL proxy: |Δ|
        let kl_proxy = gap.abs();

        total += gate * kl_proxy;
    }

    total / selected.len() as f32
}
