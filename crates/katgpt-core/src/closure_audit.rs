//! Issue 1244 — `closure_audit`: matched-capacity self-prediction gap as a
//! macrostate gate.
//!
//! Distilled from "Representational Effective Theory" (arXiv:2605.09294;
//! the note is `.research/692_RET_Macrostate_Closure_Audit.md`, the filed
//! task `.issues/1244_ret_closure_audit_primitive.md`).
//!
//! # The question
//!
//! Is a recorded latent summary `z` a **closed macrostate** of its own
//! dynamics — does `z_t` alone predict `z_{t+1}` as well as `z_t` plus the
//! un-summarized remainder `x_t` — or is the remainder load-bearing?
//! Nothing else in the workspace answers this: the contraction certificates
//! (Issue 667) are analytical over designed kernels;
//! `latent_trajectory_geometry` (Plan 342) is explicitly non-predictive;
//! the bisimulation quotients (Plan 324) are exact over DISCRETE graphs.
//! This primitive measures the gap empirically over any recorded
//! trajectory.
//!
//! # The measurement (matched capacity, closed form)
//!
//! Arms — an identical capacity menu per side:
//!
//! - **ridge** — a `λ` sweep over the closed-form normal equations
//!   `(XᵀX + λI) W = XᵀY` (Cholesky, f64); `λ` is the capacity proxy.
//! - **kNN** — regression over the `k ∈ {1, 8, 64}` nearest training
//!   windows (Euclidean on squared distances, total-order tie rule).
//!
//! Sides: **summary** (`z_t → z_{t+1}`) vs **augmented**
//! (`(z_t, x_t) → z_{t+1}`). `gap = R²(best augmented) − R²(best
//! summary)`. Capacity is MATCHED — the same arm menu on both sides — so
//! the gap isolates the microstate's marginal predictability, never
//! predictor capacity.
//!
//! Split: **blocked FORWARD** — the trajectory-major global prefix trains,
//! the suffix tests. Never an iid shuffle: autocorrelated trajectories
//! leak under one.
//!
//! # The typed verdict (two conditions — the pre-registered defense)
//!
//! Linear + kNN arms can underfit BOTH sides; a gap ≈ 0 with weak absolute
//! fit must never read green. Therefore:
//!
//! - `Closed` iff `gap ≤ epsilon` AND `R²(summary) ≥ r2_floor`;
//! - `Lossy` iff `gap > epsilon` AND `R²(summary) ≥ r2_floor` — the
//!   remainder is load-bearing;
//! - `Undecidable` iff the test support is under `support_floor`, OR
//!   `R²(summary) < r2_floor`, OR no microstate was supplied — never a
//!   green zero.
//!
//! `epsilon` and the floor are OUR bars pinned at filing; the paper's
//! held-out R² (0.908–0.979 vs matched ceilings 0.716/0.574/0.786/0.705)
//! is PROVENANCE, never our bar.
//!
//! # Determinism + allocation laws
//!
//! - Every reduction is a sequential f64 fold in a pinned order (the
//!   Issue-940 law) — the same input yields a bit-identical report
//!   (asserted by a fixture test).
//! - kNN neighbor ties break on `(distance², reference index)` — a total
//!   order, deterministic under `select_nth_unstable_by`.
//! - [`WindowRing`] is the online collection half: per-tick
//!   [`WindowRing::push`] never allocates once sized (the G4 gate).
//! - [`run`] is an OFFLINE instrument with a bounded, disclosed budget —
//!   [`ClosureReport::budget`] carries every phase wall plus the kNN
//!   reference posture (full pass or capped).
//!
//! # NO GOAT claim (ungoat discipline)
//!
//! Consumers are report-only censuses first (riir-ai `.issues/1259` sync
//! surface; riir-refine `.issues/256` fix trajectories). The G2 budget bar
//! (ridge sweep ≤ 50 ms at T=1e4, D=64, release) is a FLOOR bar on the
//! offline instrument, never a perf claim. The paper's trained-predictor
//! capacity sweep is the escalation track (Research 692 §5 — plan NOT
//! filed); the closed-form arms here are the disclosed modelless floor.
//!
//! # What this is NOT
//!
//! - NOT a probability / confidence interval — the "Report the Floor"
//!   conformal-naive rule does not apply (the `latent_confounder_audit`
//!   posture).
//! - NOT a router or a fix: the audit measures; the caller decides.
//! - NOT a halt/serve-path signal: nothing here runs per tick; an online
//!   consumer only COLLECTS windows ([`WindowRing`]) and runs the audit
//!   offline.

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The pre-registered default gap bar: an augmented arm must beat the
/// summary arm by more than this R² margin for the remainder to count as
/// load-bearing.
pub const DEFAULT_EPSILON: f64 = 0.05;

/// The pre-registered default summary-fit floor: below this absolute R²
/// the verdict is [`UndecidableReason::WeakSummaryFit`], never `Closed`.
pub const DEFAULT_R2_FLOOR: f64 = 0.5;

/// The default minimum held-out window count; below it the verdict is
/// [`UndecidableReason::WeakSupport`].
pub const DEFAULT_SUPPORT_FLOOR: usize = 128;

/// The default blocked-FORWARD train fraction (prefix trains, suffix
/// tests).
pub const DEFAULT_TRAIN_FRACTION: f64 = 0.6;

/// The default ridge `λ` grid — ascending, the capacity sweep.
pub const DEFAULT_LAMBDA_GRID: [f64; 5] = [1e-3, 1e-2, 1e-1, 1.0, 10.0];

/// The default kNN neighbor counts.
pub const DEFAULT_KNN_KS: [usize; 3] = [1, 8, 64];

/// Audit configuration. All fields are validated by [`validate_config`]
/// before any measurement runs.
#[derive(Clone, Debug)]
pub struct ClosureAuditConfig {
    /// Prefix fraction used for training (`[0.05, 0.95]`).
    pub train_fraction: f64,
    /// Ridge `λ` grid (ascending, capacity proxy; every `λ > 0`).
    pub lambda_grid: Vec<f64>,
    /// kNN neighbor counts (every `k ≥ 1`; an EMPTY grid runs the ridge
    /// arms only — the G2 bench's ridge-only posture).
    pub knn_ks: Vec<usize>,
    /// The gap bar: `gap ≤ epsilon` counts as closed.
    pub epsilon: f64,
    /// The absolute summary-fit floor.
    pub r2_floor: f64,
    /// Minimum held-out windows for any verdict but
    /// [`UndecidableReason::WeakSupport`].
    pub support_floor: usize,
    /// Optional cap on the kNN training-reference set (strided subsample,
    /// disclosed in [`ClosureBudget::knn_ref_train`]). `None` = the full
    /// pass (the honest, slower posture).
    pub knn_ref_cap: Option<usize>,
}

impl Default for ClosureAuditConfig {
    fn default() -> Self {
        Self {
            train_fraction: DEFAULT_TRAIN_FRACTION,
            lambda_grid: DEFAULT_LAMBDA_GRID.to_vec(),
            knn_ks: DEFAULT_KNN_KS.to_vec(),
            epsilon: DEFAULT_EPSILON,
            r2_floor: DEFAULT_R2_FLOOR,
            support_floor: DEFAULT_SUPPORT_FLOOR,
            knn_ref_cap: None,
        }
    }
}

fn finite_positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

/// Typed configuration refusal — never a silent default.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClosureAuditError {
    /// The trajectory slice was empty.
    NoTrajectories,
    /// The window set was empty (every trajectory shorter than two states).
    NoWindows,
    /// A trajectory (or microstate row) disagreed with the established dim.
    DimMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    /// A zero-width latent or microstate.
    ZeroDimension,
    /// `micro` was supplied with a different trajectory count.
    MicroLengthMismatch { trajectories: usize, micro: usize },
    /// A non-finite value in the input windows (typed, naming the window).
    NonFiniteInput { window: usize },
    /// The held-out targets are constant — R² has no denominator.
    DegenerateTargets,
    /// A configuration field refused.
    BadConfig { reason: &'static str },
}

impl fmt::Display for ClosureAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTrajectories => write!(f, "closure audit: no trajectories"),
            Self::NoWindows => write!(f, "closure audit: no transition windows"),
            Self::DimMismatch {
                context,
                expected,
                got,
            } => write!(
                f,
                "closure audit: {context} dim {got} disagrees with established {expected}"
            ),
            Self::ZeroDimension => write!(f, "closure audit: zero-width input"),
            Self::MicroLengthMismatch {
                trajectories,
                micro,
            } => write!(
                f,
                "closure audit: microstate count {micro} disagrees with trajectory count {trajectories}"
            ),
            Self::NonFiniteInput { window } => {
                write!(f, "closure audit: non-finite value in window {window}")
            }
            Self::DegenerateTargets => write!(
                f,
                "closure audit: held-out targets are constant (zero variance) — R² undefined"
            ),
            Self::BadConfig { reason } => write!(f, "closure audit: bad config: {reason}"),
        }
    }
}

impl std::error::Error for ClosureAuditError {}

fn validate_config(config: &ClosureAuditConfig) -> Result<(), ClosureAuditError> {
    if !config.train_fraction.is_finite() || !(0.05..=0.95).contains(&config.train_fraction) {
        return Err(ClosureAuditError::BadConfig {
            reason: "train_fraction must be finite within [0.05, 0.95]",
        });
    }
    if config.lambda_grid.is_empty() || config.lambda_grid.iter().any(|l| !finite_positive(*l)) {
        return Err(ClosureAuditError::BadConfig {
            reason: "lambda_grid must be non-empty with every lambda finite > 0",
        });
    }
    if config.knn_ks.contains(&0) {
        return Err(ClosureAuditError::BadConfig {
            reason: "knn_ks entries must each be >= 1 (an empty grid runs the ridge arms only)",
        });
    }
    if !finite_positive(config.epsilon) && config.epsilon != 0.0 {
        return Err(ClosureAuditError::BadConfig {
            reason: "epsilon must be finite >= 0",
        });
    }
    if !config.r2_floor.is_finite() || !(0.0..=1.0).contains(&config.r2_floor) {
        return Err(ClosureAuditError::BadConfig {
            reason: "r2_floor must be finite within [0, 1]",
        });
    }
    if config.support_floor == 0 {
        return Err(ClosureAuditError::BadConfig {
            reason: "support_floor must be >= 1",
        });
    }
    if config.knn_ref_cap == Some(0) {
        return Err(ClosureAuditError::BadConfig {
            reason: "knn_ref_cap must be None or Some(>= 1)",
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Verdict + report types
// ---------------------------------------------------------------------------

/// Why the audit refused to adjudicate. Typed, never folded into a pass.
#[derive(Clone, Debug, PartialEq)]
pub enum UndecidableReason {
    /// No microstate was supplied — there is nothing to compare against.
    NoMicrostate,
    /// Held-out support under the configured floor.
    WeakSupport {
        test_windows: usize,
        support_floor: usize,
    },
    /// The summary side itself predicts too weakly — both arms are likely
    /// underfit; the gap is uninformative.
    WeakSummaryFit { summary_r2: f64, r2_floor: f64 },
}

/// The audit's verdict. `Closed`/`Lossy` carry the numbers; `Undecidable`
/// carries the reason. A gap ≈ 0 with a weak absolute fit NEVER reads
/// `Closed`.
#[derive(Clone, Debug, PartialEq)]
pub enum ClosureVerdict {
    /// The summary is a closed macrostate at this ε and floor.
    Closed { gap: f64, summary_r2: f64 },
    /// The microstate remainder is load-bearing.
    Lossy { gap: f64, summary_r2: f64 },
    /// The audit refuses to adjudicate.
    Undecidable { reason: UndecidableReason },
}

/// Which side an arm measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditSide {
    Summary,
    Augmented,
}

/// One arm of the matched-capacity menu.
#[derive(Clone, Debug, PartialEq)]
pub enum ArmFamily {
    /// Ridge with this `λ` (the capacity proxy).
    Ridge { lambda: f64 },
    /// kNN regression with this neighbor count.
    Knn { k: usize },
}

/// One measured arm. `PartialEq` ignores `wall_ms` (walls are timing, not
/// measurement — the determinism gate compares the numbers).
#[derive(Clone, Debug)]
pub struct ArmRow {
    pub side: AuditSide,
    pub family: ArmFamily,
    pub r2: f64,
    pub wall_ms: f64,
}

impl PartialEq for ArmRow {
    fn eq(&self, other: &Self) -> bool {
        self.side == other.side && self.family == other.family && self.r2 == other.r2
    }
}

/// Window support for the split.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClosureSupport {
    /// Total transition windows built.
    pub windows: usize,
    /// Prefix windows used for training.
    pub train_windows: usize,
    /// Suffix windows held out.
    pub test_windows: usize,
}

/// The disclosed offline budget — every phase wall plus the kNN reference
/// posture. Walls are DISCLOSED, never pinned (box-state governs timing).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClosureBudget {
    pub window_build_ms: f64,
    pub ridge_summary_ms: f64,
    pub ridge_augmented_ms: f64,
    pub knn_summary_ms: f64,
    pub knn_augmented_ms: f64,
    pub total_ms: f64,
    /// kNN training references actually used (post-cap).
    pub knn_ref_train: usize,
    /// The configured cap, if any.
    pub knn_ref_cap: Option<usize>,
}

/// The full audit report.
#[derive(Clone, Debug)]
pub struct ClosureReport {
    pub verdict: ClosureVerdict,
    /// Best summary-side R² across the arm menu.
    pub summary_best_r2: f64,
    /// Best augmented-side R² (`None` when no microstate was supplied).
    pub augmented_best_r2: Option<f64>,
    /// `R²(augmented) − R²(summary)` (`None` when no microstate).
    pub gap: Option<f64>,
    /// Every arm, in menu order (ridge grid, then kNN ks; summary side
    /// first, augmented second).
    pub arms: Vec<ArmRow>,
    pub support: ClosureSupport,
    pub budget: ClosureBudget,
}

impl PartialEq for ClosureReport {
    fn eq(&self, other: &Self) -> bool {
        self.verdict == other.verdict
            && self.summary_best_r2 == other.summary_best_r2
            && self.augmented_best_r2 == other.augmented_best_r2
            && self.gap == other.gap
            && self.arms == other.arms
            && self.support == other.support
    }
}

/// The typed verdict law, exposed so consumers can replay/what-if a
/// boundary without re-running arms. Order: support, then absolute fit,
/// then the gap.
pub fn closure_verdict(
    gap: f64,
    summary_r2: f64,
    test_windows: usize,
    config: &ClosureAuditConfig,
) -> ClosureVerdict {
    if test_windows < config.support_floor {
        return ClosureVerdict::Undecidable {
            reason: UndecidableReason::WeakSupport {
                test_windows,
                support_floor: config.support_floor,
            },
        };
    }
    if summary_r2 < config.r2_floor {
        return ClosureVerdict::Undecidable {
            reason: UndecidableReason::WeakSummaryFit {
                summary_r2,
                r2_floor: config.r2_floor,
            },
        };
    }
    if gap <= config.epsilon {
        ClosureVerdict::Closed { gap, summary_r2 }
    } else {
        ClosureVerdict::Lossy { gap, summary_r2 }
    }
}

// ---------------------------------------------------------------------------
// WindowRing — the alloc-free online collection half (G4)
// ---------------------------------------------------------------------------

/// One borrowed transition window.
#[derive(Clone, Copy, Debug)]
pub struct WindowRef<'a> {
    pub z_t: &'a [f32],
    pub z_next: &'a [f32],
    pub micro: Option<&'a [f32]>,
}

/// Fixed-capacity ring of transition windows. [`WindowRing::push`] copies
/// into pre-allocated slots — zero allocation once sized (the G4 gate).
/// When full, the oldest window is overwritten (counted in
/// [`WindowRing::overwritten`], never silent).
pub struct WindowRing {
    dim: usize,
    micro_dim: Option<usize>,
    cur: Vec<Vec<f32>>,
    next: Vec<Vec<f32>>,
    micro: Vec<Option<Vec<f32>>>,
    /// Index of the OLDEST slot.
    head: usize,
    len: usize,
    overwritten: usize,
}

impl WindowRing {
    /// Pre-allocates `capacity` slots. Every later push is alloc-free.
    pub fn with_capacity(capacity: usize, dim: usize, micro_dim: Option<usize>) -> Self {
        assert!(capacity >= 1, "window ring capacity must be >= 1");
        assert!(dim >= 1, "window ring dim must be >= 1");
        if let Some(md) = micro_dim {
            assert!(md >= 1, "window ring micro dim must be >= 1");
        }
        let mut cur = Vec::with_capacity(capacity);
        let mut next = Vec::with_capacity(capacity);
        let mut micro = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            cur.push(vec![0.0; dim]);
            next.push(vec![0.0; dim]);
            micro.push(micro_dim.map(|md| vec![0.0; md]));
        }
        Self {
            dim,
            micro_dim,
            cur,
            next,
            micro,
            head: 0,
            len: 0,
            overwritten: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.cur.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn micro_dim(&self) -> Option<usize> {
        self.micro_dim
    }

    /// Windows dropped by full-capacity overwrites.
    pub fn overwritten(&self) -> usize {
        self.overwritten
    }

    /// Copies one window into the ring. Zero allocation once sized.
    /// Panics on a shape mismatch — the shape is fixed at construction.
    pub fn push(&mut self, z_t: &[f32], z_next: &[f32], micro: Option<&[f32]>) {
        assert_eq!(z_t.len(), self.dim, "window ring: z_t dim mismatch");
        assert_eq!(z_next.len(), self.dim, "window ring: z_next dim mismatch");
        match (self.micro_dim, micro) {
            (None, Some(_)) => panic!("window ring: unexpected microstate"),
            (Some(md), Some(m)) => assert_eq!(m.len(), md, "window ring: micro dim mismatch"),
            (Some(_), None) => panic!("window ring: missing microstate"),
            (None, None) => {}
        }
        let slot = if self.len < self.capacity() {
            self.len
        } else {
            self.overwritten += 1;
            self.head
        };
        self.cur[slot].copy_from_slice(z_t);
        self.next[slot].copy_from_slice(z_next);
        if let (Some(slot_micro), Some(m)) = (self.micro[slot].as_mut(), micro) {
            slot_micro.copy_from_slice(m);
        }
        if self.len < self.capacity() {
            self.len += 1;
        } else {
            self.head = (self.head + 1) % self.capacity();
        }
    }

    /// Borrowed window `i` in ring order (oldest first). None out of range.
    pub fn window(&self, i: usize) -> Option<WindowRef<'_>> {
        if i >= self.len {
            return None;
        }
        let slot = (self.head + i) % self.capacity();
        Some(WindowRef {
            z_t: &self.cur[slot],
            z_next: &self.next[slot],
            micro: self.micro[slot].as_deref(),
        })
    }

    /// Every window in ring order (oldest first).
    pub fn windows(&self) -> impl Iterator<Item = WindowRef<'_>> + '_ {
        (0..self.len).filter_map(move |i| self.window(i))
    }
}

// ---------------------------------------------------------------------------
// The audit
// ---------------------------------------------------------------------------

/// Runs the audit over recorded trajectories (each a sequence of summary
/// vectors) with an optional parallel microstate recording. Offline
/// instrument; the report discloses every phase wall.
///
/// Panics-free on validated input: every refusal is a typed
/// [`ClosureAuditError`].
///
/// Each trajectory is a FLAT row-major recording whose length must be a
/// nonzero multiple of `dim` — the width is EXPLICIT (a flat recording's
/// length cannot encode it). The same law binds the microstate rows via
/// `micro_dim`. Use [`run_from_ring`] to audit windows collected per
/// tick (the ring fixes the shape at construction).
pub fn run(
    trajectories: &[&[f32]],
    dim: usize,
    micro: Option<(&[&[f32]], usize)>,
    config: &ClosureAuditConfig,
) -> Result<ClosureReport, ClosureAuditError> {
    if trajectories.is_empty() {
        return Err(ClosureAuditError::NoTrajectories);
    }
    if dim == 0 {
        return Err(ClosureAuditError::ZeroDimension);
    }
    for traj in trajectories.iter() {
        if traj.is_empty() || !traj.len().is_multiple_of(dim) {
            return Err(ClosureAuditError::DimMismatch {
                context: "trajectory length (flat row-major; must be a nonzero multiple of dim)",
                expected: dim,
                got: traj.len(),
            });
        }
    }
    let micro_dim = match micro {
        None => None,
        Some((rows, md)) => {
            if md == 0 {
                return Err(ClosureAuditError::ZeroDimension);
            }
            if rows.len() != trajectories.len() {
                return Err(ClosureAuditError::MicroLengthMismatch {
                    trajectories: trajectories.len(),
                    micro: rows.len(),
                });
            }
            for row in rows.iter() {
                if row.is_empty() || !row.len().is_multiple_of(md) {
                    return Err(ClosureAuditError::DimMismatch {
                        context: "microstate row length (flat row-major; must be a nonzero multiple of micro dim)",
                        expected: md,
                        got: row.len(),
                    });
                }
            }
            Some(md)
        }
    };
    // Build the window list, trajectory-major (the blocked-FORWARD split
    // orders on exactly this sequence). Each flat recording yields
    // (steps − 1) transition windows.
    let steps_of = |flat: &[f32], width: usize| flat.len() / width;
    let total: usize = trajectories
        .iter()
        .map(|t| steps_of(t, dim).saturating_sub(1))
        .sum();
    let mut windows = Vec::with_capacity(total);
    let (micro_rows, md) = match micro {
        Some((rows, md)) => (Some(rows), md),
        None => (None, 0),
    };
    for (t_idx, traj) in trajectories.iter().enumerate() {
        let steps = steps_of(traj, dim);
        for i in 0..steps.saturating_sub(1) {
            windows.push(WindowRef {
                z_t: &traj[i * dim..(i + 1) * dim],
                z_next: &traj[(i + 1) * dim..(i + 2) * dim],
                micro: micro_rows
                    .and_then(|rows| rows.get(t_idx))
                    .and_then(|row| row.get(i * md..(i + 1) * md)),
            });
        }
    }
    run_core(&windows, dim, micro_dim, config)
}

/// Runs the audit over a [`WindowRing`]'s contents (the online collection
/// path — collect per tick, audit offline).
pub fn run_from_ring(
    ring: &WindowRing,
    config: &ClosureAuditConfig,
) -> Result<ClosureReport, ClosureAuditError> {
    if ring.is_empty() {
        return Err(ClosureAuditError::NoWindows);
    }
    let windows: Vec<WindowRef<'_>> = ring.windows().collect();
    run_core(&windows, ring.dim(), ring.micro_dim(), config)
}

/// The core audit over borrowed windows.
fn run_core(
    windows: &[WindowRef<'_>],
    dim: usize,
    micro_dim: Option<usize>,
    config: &ClosureAuditConfig,
) -> Result<ClosureReport, ClosureAuditError> {
    let t_start = Instant::now();
    validate_config(config)?;
    let n = windows.len();
    if n < 2 {
        return Ok(early_undecidable(
            n.saturating_sub(1),
            config,
            t_start.elapsed(),
        ));
    }

    // ---- window build (one pass, three flat f64 buffers) ----
    let t_build = Instant::now();
    let mut x_all = vec![0.0f64; n * dim];
    let mut y_all = vec![0.0f64; n * dim];
    let mut m_all = micro_dim.map(|md| vec![0.0f64; n * md]);
    for (i, w) in windows.iter().enumerate() {
        if w.z_t.len() != dim || w.z_next.len() != dim {
            return Err(ClosureAuditError::DimMismatch {
                context: "window",
                expected: dim,
                got: w.z_t.len().min(w.z_next.len()),
            });
        }
        for (j, v) in w.z_t.iter().enumerate() {
            let v = f64::from(*v);
            if !v.is_finite() {
                return Err(ClosureAuditError::NonFiniteInput { window: i });
            }
            x_all[i * dim + j] = v;
        }
        for (j, v) in w.z_next.iter().enumerate() {
            let v = f64::from(*v);
            if !v.is_finite() {
                return Err(ClosureAuditError::NonFiniteInput { window: i });
            }
            y_all[i * dim + j] = v;
        }
        if let (Some(buf), Some(m)) = (m_all.as_mut(), w.micro) {
            if m.len() != micro_dim.unwrap_or(0) {
                return Err(ClosureAuditError::DimMismatch {
                    context: "window microstate",
                    expected: micro_dim.unwrap_or(0),
                    got: m.len(),
                });
            }
            for (j, v) in m.iter().enumerate() {
                let v = f64::from(*v);
                if !v.is_finite() {
                    return Err(ClosureAuditError::NonFiniteInput { window: i });
                }
                buf[i * micro_dim.unwrap_or(0) + j] = v;
            }
        }
    }
    let window_build_ms = t_build.elapsed().as_secs_f64() * 1e3;

    // ---- blocked-FORWARD split (trajectory-major prefix/suffix) ----
    let train_windows = (((n as f64) * config.train_fraction).floor() as usize).clamp(1, n - 1);
    let test_windows = n - train_windows;
    let support = ClosureSupport {
        windows: n,
        train_windows,
        test_windows,
    };
    if test_windows < config.support_floor {
        let mut report = early_undecidable(test_windows, config, t_start.elapsed());
        report.support = support;
        report.budget.window_build_ms = window_build_ms;
        return Ok(report);
    }

    // Shared held-out denominator (SST against per-column held-out means —
    // the same denominator for every arm on both sides).
    let y_sst = {
        let y_te = &y_all[train_windows * dim..];
        let mean = column_means(y_te, test_windows, dim);
        let sst = sst_of(y_te, test_windows, dim, &mean);
        if sst <= 0.0 {
            return Err(ClosureAuditError::DegenerateTargets);
        }
        sst
    };

    let mut arms: Vec<ArmRow> = Vec::with_capacity(
        (config.lambda_grid.len() + config.knn_ks.len()) * if micro_dim.is_some() { 2 } else { 1 },
    );
    let mut ridge_summary_ms = 0.0;
    let mut ridge_augmented_ms = 0.0;
    let mut knn_summary_ms = 0.0;
    let mut knn_augmented_ms = 0.0;

    let mut best_summary = f64::NEG_INFINITY;
    let mut best_augmented = f64::NEG_INFINITY;

    // The augmented design matrix is built ONCE and shared by the ridge and
    // kNN arms (width = dim + micro_dim, interleaved row-major).
    let x_aug = micro_dim.map(|md| {
        let width = dim + md;
        let mut buf = vec![0.0f64; n * width];
        for i in 0..n {
            buf[i * width..i * width + dim].copy_from_slice(&x_all[i * dim..(i + 1) * dim]);
            buf[i * width + dim..(i + 1) * width].copy_from_slice(
                m_all
                    .as_ref()
                    .map(|b| &b[i * md..(i + 1) * md])
                    .unwrap_or(&[]),
            );
        }
        buf
    });

    // ---- ridge sweep, summary side ----
    {
        let x_tr = &x_all[..train_windows * dim];
        let x_te = &x_all[train_windows * dim..];
        for &lambda in &config.lambda_grid {
            let t = Instant::now();
            let r2 = ridge_r2(
                x_tr,
                x_te,
                &y_all,
                train_windows,
                test_windows,
                dim,
                dim,
                lambda,
                y_sst,
            );
            let arm_ms = t.elapsed().as_secs_f64() * 1e3;
            ridge_summary_ms += arm_ms;
            arms.push(ArmRow {
                side: AuditSide::Summary,
                family: ArmFamily::Ridge { lambda },
                r2,
                wall_ms: arm_ms,
            });
            if r2 > best_summary {
                best_summary = r2;
            }
        }
    }

    // ---- ridge sweep, augmented side ----
    if micro_dim.is_some() {
        let md = micro_dim.unwrap_or(0);
        let width = dim + md;
        let x_aug = x_aug.as_deref().unwrap_or(&[]);
        for &lambda in &config.lambda_grid {
            let t = Instant::now();
            let r2 = ridge_r2(
                &x_aug[..train_windows * width],
                &x_aug[train_windows * width..],
                &y_all,
                train_windows,
                test_windows,
                width,
                dim,
                lambda,
                y_sst,
            );
            let arm_ms = t.elapsed().as_secs_f64() * 1e3;
            ridge_augmented_ms += arm_ms;
            arms.push(ArmRow {
                side: AuditSide::Augmented,
                family: ArmFamily::Ridge { lambda },
                r2,
                wall_ms: arm_ms,
            });
            if r2 > best_augmented {
                best_augmented = r2;
            }
        }
    }

    // ---- kNN arms (reference set shared per side across the ks) ----
    {
        let refs = capped_ref_indices(train_windows, config.knn_ref_cap);
        for &k in &config.knn_ks {
            let t = Instant::now();
            let r2 = knn_r2(
                &x_all,
                &y_all,
                &refs,
                train_windows,
                test_windows,
                dim,
                dim,
                k.min(refs.len()),
                y_sst,
            );
            let arm_ms = t.elapsed().as_secs_f64() * 1e3;
            knn_summary_ms += arm_ms;
            arms.push(ArmRow {
                side: AuditSide::Summary,
                family: ArmFamily::Knn { k },
                r2,
                wall_ms: arm_ms,
            });
            if r2 > best_summary {
                best_summary = r2;
            }
        }
    }
    if micro_dim.is_some() {
        let md = micro_dim.unwrap_or(0);
        let width = dim + md;
        let x_aug = x_aug.as_deref().unwrap_or(&[]);
        let refs = capped_ref_indices(train_windows, config.knn_ref_cap);
        for &k in &config.knn_ks {
            let t = Instant::now();
            let r2 = knn_r2(
                x_aug,
                &y_all,
                &refs,
                train_windows,
                test_windows,
                width,
                dim,
                k.min(refs.len()),
                y_sst,
            );
            let arm_ms = t.elapsed().as_secs_f64() * 1e3;
            knn_augmented_ms += arm_ms;
            arms.push(ArmRow {
                side: AuditSide::Augmented,
                family: ArmFamily::Knn { k },
                r2,
                wall_ms: arm_ms,
            });
            if r2 > best_augmented {
                best_augmented = r2;
            }
        }
    }

    // ---- verdict ----
    let budget = ClosureBudget {
        window_build_ms,
        ridge_summary_ms,
        ridge_augmented_ms,
        knn_summary_ms,
        knn_augmented_ms,
        total_ms: t_start.elapsed().as_secs_f64() * 1e3,
        knn_ref_train: capped_ref_indices(train_windows, config.knn_ref_cap).len(),
        knn_ref_cap: config.knn_ref_cap,
    };
    let verdict = match micro_dim {
        None => ClosureVerdict::Undecidable {
            reason: UndecidableReason::NoMicrostate,
        },
        Some(_) => {
            let gap = best_augmented - best_summary;
            closure_verdict(gap, best_summary, test_windows, config)
        }
    };
    Ok(ClosureReport {
        verdict,
        summary_best_r2: best_summary,
        augmented_best_r2: if micro_dim.is_some() {
            Some(best_augmented)
        } else {
            None
        },
        gap: if micro_dim.is_some() {
            Some(best_augmented - best_summary)
        } else {
            None
        },
        arms,
        support,
        budget,
    })
}

/// The shared "nothing measurable" report — typed Undecidable, empty arms.
fn early_undecidable(
    test_windows: usize,
    config: &ClosureAuditConfig,
    total: std::time::Duration,
) -> ClosureReport {
    ClosureReport {
        verdict: ClosureVerdict::Undecidable {
            reason: UndecidableReason::WeakSupport {
                test_windows,
                support_floor: config.support_floor,
            },
        },
        summary_best_r2: 0.0,
        augmented_best_r2: None,
        gap: None,
        arms: Vec::new(),
        support: ClosureSupport {
            windows: test_windows,
            train_windows: 0,
            test_windows,
        },
        budget: ClosureBudget {
            window_build_ms: 0.0,
            ridge_summary_ms: 0.0,
            ridge_augmented_ms: 0.0,
            knn_summary_ms: 0.0,
            knn_augmented_ms: 0.0,
            total_ms: total.as_secs_f64() * 1e3,
            knn_ref_train: 0,
            knn_ref_cap: config.knn_ref_cap,
        },
    }
}

// ---------------------------------------------------------------------------
// Numeric cores — sequential f64 folds in pinned order (Issue-940)
// ---------------------------------------------------------------------------

const CHOLESKY_PIVOT_FLOOR: f64 = 1e-10;

/// Column means over a row-major `n × d` block — sequential f64.
fn column_means(block: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut means = vec![0.0f64; d];
    for row in 0..n {
        let r = &block[row * d..(row + 1) * d];
        for j in 0..d {
            means[j] += r[j];
        }
    }
    if n > 0 {
        for m in &mut means {
            *m /= n as f64;
        }
    }
    means
}

/// Pooled SST against per-column means — sequential f64.
fn sst_of(block: &[f64], n: usize, d: usize, means: &[f64]) -> f64 {
    let mut sst = 0.0f64;
    for row in 0..n {
        let r = &block[row * d..(row + 1) * d];
        for j in 0..d {
            let e = r[j] - means[j];
            sst += e * e;
        }
    }
    sst
}

/// Fits ridge on the train block and returns the held-out pooled R².
/// `(XᵀX + λI) W = XᵀY` via Cholesky; folds sequential f64.
#[allow(clippy::too_many_arguments)]
fn ridge_r2(
    x_tr: &[f64],
    x_te: &[f64],
    y_all: &[f64],
    n_tr: usize,
    n_te: usize,
    d: usize,
    dout: usize,
    lambda: f64,
    y_sst: f64,
) -> f64 {
    // Gram + rhs (row-major d×d / d×dout), built in one train pass.
    let mut gram = vec![0.0f64; d * d];
    let mut rhs = vec![0.0f64; d * dout];
    for row in 0..n_tr {
        let x = &x_tr[row * d..(row + 1) * d];
        let y = &y_all[row * dout..(row + 1) * dout];
        for i in 0..d {
            let xi = x[i];
            for (g, xj) in gram[i * d + i..i * d + d].iter_mut().zip(x[i..].iter()) {
                *g += xi * xj;
            }
            for (b, yk) in rhs[i * dout..(i + 1) * dout].iter_mut().zip(y.iter()) {
                *b += xi * yk;
            }
        }
    }
    for i in 0..d {
        for j in 0..i {
            gram[i * d + j] = gram[j * d + i];
        }
        gram[i * d + i] += lambda;
    }
    // Cholesky (lower, in place) with the deterministic pivot floor.
    for i in 0..d {
        for j in 0..=i {
            let mut s = gram[i * d + j];
            for k in 0..j {
                s -= gram[i * d + k] * gram[j * d + k];
            }
            let v = if i == j {
                (s.max(CHOLESKY_PIVOT_FLOOR)).sqrt()
            } else {
                s / gram[j * d + j]
            };
            gram[i * d + j] = v;
        }
    }
    // Solve L Z = rhs (forward), then Lᵀ W = Z (back). rhs doubles as Z/W.
    for i in 0..d {
        let lii = gram[i * d + i];
        for k in 0..dout {
            rhs[i * dout + k] /= lii;
        }
        let zi = &mut rhs[i * dout..(i + 1) * dout];
        let zik = zi.to_vec();
        for j in (i + 1)..d {
            let lij = gram[j * d + i];
            if lij != 0.0 {
                let zj = &mut rhs[j * dout..(j + 1) * dout];
                for k in 0..dout {
                    zj[k] -= lij * zik[k];
                }
            }
        }
    }
    for i in (0..d).rev() {
        let wii = gram[i * d + i];
        for k in 0..dout {
            rhs[i * dout + k] /= wii;
        }
        let wik = rhs[i * dout..(i + 1) * dout].to_vec();
        // Push w[i] into the rows BELOW it: coefficient L[i][j] =
        // gram[i*d + j] (j < i — the LOWER triangle; L[j][i] with j < i
        // would read the stale pre-Cholesky upper triangle).
        for j in 0..i {
            let lij = gram[i * d + j];
            if lij != 0.0 {
                let wj = &mut rhs[j * dout..(j + 1) * dout];
                for k in 0..dout {
                    wj[k] -= lij * wik[k];
                }
            }
        }
    }
    // Held-out predictions + SSE.
    let mut sse = 0.0f64;
    let mut pred = vec![0.0f64; dout];
    for row in 0..n_te {
        let x = &x_te[row * d..(row + 1) * d];
        let y = &y_all[(n_tr + row) * dout..(n_tr + row + 1) * dout];
        pred.fill(0.0);
        for j in 0..d {
            let xj = x[j];
            if xj != 0.0 {
                for k in 0..dout {
                    pred[k] += xj * rhs[j * dout + k];
                }
            }
        }
        for k in 0..dout {
            let e = y[k] - pred[k];
            sse += e * e;
        }
    }
    1.0 - sse / y_sst
}

/// kNN regression R² over the (capped) training references. Distances are
/// squared Euclidean; ties break on `(distance², reference index)` — a
/// total order, deterministic under `select_nth_unstable_by`.
#[allow(clippy::too_many_arguments)]
fn knn_r2(
    x_all: &[f64],
    y_all: &[f64],
    refs: &[usize],
    train_windows: usize,
    test_windows: usize,
    d: usize,
    dout: usize,
    k: usize,
    y_sst: f64,
) -> f64 {
    let k_eff = k.max(1).min(refs.len());
    let mut scratch: Vec<(f64, u32)> = Vec::with_capacity(refs.len());
    let mut pred = vec![0.0f64; dout];
    let mut sse = 0.0f64;
    for row in 0..test_windows {
        let x = &x_all[(train_windows + row) * d..(train_windows + row + 1) * d];
        let y = &y_all[(train_windows + row) * dout..(train_windows + row + 1) * dout];
        scratch.clear();
        for (ri, &ref_row) in refs.iter().enumerate() {
            let r = &x_all[ref_row * d..(ref_row + 1) * d];
            let mut d2 = 0.0f64;
            for j in 0..d {
                let e = x[j] - r[j];
                d2 += e * e;
            }
            scratch.push((d2, ri as u32));
        }
        if k_eff < scratch.len() {
            scratch.select_nth_unstable_by(k_eff - 1, |a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
        }
        let head = &mut scratch[..k_eff];
        head.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.cmp(&b.1))
        });
        pred.fill(0.0);
        for (_d2, ri) in head.iter() {
            let y_ref = &y_all[refs[*ri as usize] * dout..(refs[*ri as usize] + 1) * dout];
            let w = 1.0 / k_eff as f64;
            for kk in 0..dout {
                pred[kk] += w * y_ref[kk];
            }
        }
        for kk in 0..dout {
            let e = y[kk] - pred[kk];
            sse += e * e;
        }
    }
    1.0 - sse / y_sst
}

/// Evenly-spaced reference subsample — lands EXACTLY on
/// `min(cap, n_train)` points (deterministic: first and last rows always
/// kept, interior evenly spaced).
fn capped_ref_indices(n_train: usize, cap: Option<usize>) -> Vec<usize> {
    match cap {
        None => (0..n_train).collect(),
        Some(c) if c >= n_train => (0..n_train).collect(),
        Some(1) => vec![0],
        Some(c) => (0..c).map(|i| i * (n_train - 1) / (c - 1)).collect(),
    }
}

// ---------------------------------------------------------------------------
// Fixture generator (tests + the G2 bench share it; NOT a data source)
// ---------------------------------------------------------------------------

/// Fixture regimes for the G1 gates and the G2 budget bench.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureKind {
    /// `z_{t+1} = A·z_t` — exact ORTHOGONAL linear dynamics (a product of
    /// disjoint Givens rotations: unit spectral radius, so the state norm
    /// survives the whole recording); the microstate is pure noise. Both
    /// arms ≈ 1, gap ≈ 0 → `Closed`.
    Identity,
    /// `z_{t+1} = 0.9·z_t + 0.4359·ε` — AR(1) with stationary variance 1;
    /// the microstate is pure noise → `Closed`.
    Noise,
    /// `z_{t+1} = 0.8·z_t + 0.6·x_t + 0.01·ε` — the microstate DRIVE is
    /// load-bearing. Summary R² ≈ 0.64, gap ≈ 0.36 → `Lossy`.
    Injection,
    /// The degenerate-abstraction plant: two INDEPENDENT AR chains with
    /// DIFFERENT persistence — u fast (ρ 0.3), h slow (ρ 0.95, the
    /// high-variance chain) — so a mixture of the two is NOT an AR chain
    /// of its own. The planted summary is `x = u + 0.5·h` recorded as
    /// `[x, 0.5·x]` — one dim a pure function of the other (the degenerate
    /// abstraction): the pad that SHOULD have carried the decomposition
    /// carries a copy of the mixture instead, and `h` is exactly what the
    /// summary cannot disentangle. The microstate `[u, h]` carries it.
    /// microstate `[u, h]` carries it. Theory (Var(u) ≈ 0.21, Var(h) ≈
    /// 1.95; the h-chain's noise enters the mixture scaled by w², so a
    /// LARGE weight shrinks the gap): summary alone predicts at
    /// R² ≈ 0.62, the decomposed state at R² ≈ 0.72 — gap ≈ 0.09 →
    /// `Lossy` — NEVER `Closed` (the remainder is load-bearing, and the
    /// fit clears the floor).
    Degenerate,
}

/// Fixture specification. `Degenerate` fixes dim = 2 / micro_dim = 2
/// regardless of the fields (the plant is a two-chain coupled system).
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct FixtureSpec {
    pub kind: FixtureKind,
    pub steps: usize,
    pub dim: usize,
    pub micro_dim: usize,
    pub seed: u64,
}

/// SplitMix64 — the house deterministic test RNG (Box–Muller normals).
struct SplitMix64 {
    state: u64,
    pending: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            pending: None,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    fn next_normal(&mut self) -> f64 {
        if let Some(v) = self.pending.take() {
            return v;
        }
        let u1 = (1.0 - self.next_uniform()).max(f64::MIN_POSITIVE);
        let u2 = self.next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.pending = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Builds `(trajectory, microstate)` fixture recordings — deterministic
/// per seed. TEST-AND-BENCH ONLY: never a data source.
#[doc(hidden)]
pub fn build_fixture(spec: &FixtureSpec) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = SplitMix64::new(spec.seed);
    match spec.kind {
        FixtureKind::Identity | FixtureKind::Noise => {
            let dim = spec.dim.max(1);
            let micro_dim = spec.micro_dim.max(1);
            let mut traj = Vec::with_capacity(spec.steps);
            let mut micro = Vec::with_capacity(spec.steps);
            let a: Vec<f64> = match spec.kind {
                // A = the product of disjoint Givens rotations — ORTHOGONAL,
                // spectral radius exactly 1: the state norm is preserved
                // forever, so the gram stays at O(n) scale against the
                // absolute ridge λ (a decaying A would underflow the late
                // windows under λ and zero the fit).
                FixtureKind::Identity => {
                    let mut a = vec![0.0f64; dim * dim];
                    for i in 0..dim {
                        a[i * dim + i] = 1.0;
                    }
                    let mut pair = 0;
                    while pair + 1 < dim {
                        let theta = 0.3 + 0.1 * (pair as f64);
                        let (c, s) = (theta.cos(), theta.sin());
                        for j in 0..dim {
                            let a0j = a[j * dim + pair];
                            let a1j = a[j * dim + pair + 1];
                            a[j * dim + pair] = c * a0j - s * a1j;
                            a[j * dim + pair + 1] = s * a0j + c * a1j;
                        }
                        pair += 2;
                    }
                    a
                }
                _ => Vec::new(),
            };
            let mut z: Vec<f64> = (0..dim).map(|_| rng.next_normal()).collect();
            for _ in 0..spec.steps {
                traj.push(z.iter().map(|v| *v as f32).collect());
                micro.push((0..micro_dim).map(|_| rng.next_normal() as f32).collect());
                match spec.kind {
                    FixtureKind::Identity => {
                        let mut nz = vec![0.0f64; dim];
                        for i in 0..dim {
                            let mut acc = 0.0;
                            for j in 0..dim {
                                acc += a[i * dim + j] * z[j];
                            }
                            nz[i] = acc;
                        }
                        z = nz;
                    }
                    FixtureKind::Noise => {
                        for v in &mut z {
                            *v = 0.9 * *v + 0.435889894354067 * rng.next_normal();
                        }
                    }
                    _ => unreachable!(),
                }
            }
            (traj, micro)
        }
        FixtureKind::Injection => {
            let dim = spec.dim.max(1);
            let micro_dim = spec.micro_dim.max(1);
            let drive = dim.min(micro_dim);
            let mut traj = Vec::with_capacity(spec.steps);
            let mut micro = Vec::with_capacity(spec.steps);
            let mut z: Vec<f64> = (0..dim).map(|_| rng.next_normal()).collect();
            for _ in 0..spec.steps {
                traj.push(z.iter().map(|v| *v as f32).collect());
                let x: Vec<f64> = (0..micro_dim).map(|_| rng.next_normal()).collect();
                micro.push(x.iter().map(|v| *v as f32).collect());
                let mut nz = vec![0.0f64; dim];
                for i in 0..dim {
                    nz[i] = if i < drive {
                        0.8 * z[i] + 0.6 * x[i] + 0.01 * rng.next_normal()
                    } else {
                        0.8 * z[i]
                    };
                }
                z = nz;
            }
            (traj, micro)
        }
        FixtureKind::Degenerate => {
            let mut traj = Vec::with_capacity(spec.steps);
            let mut micro = Vec::with_capacity(spec.steps);
            let mut u = rng.next_normal();
            let mut h = rng.next_normal();
            for _ in 0..spec.steps {
                let x = u + 0.5 * h;
                traj.push(vec![x as f32, (0.5 * x) as f32]);
                micro.push(vec![u as f32, h as f32]);
                // Independent chains with DIFFERENT persistence: a
                // mixture of the two is not an AR chain of its own, so
                // the mixture summary genuinely loses the decomposition.
                let nu = 0.3 * u + 0.19f64.sqrt() * rng.next_normal();
                let nh = 0.95 * h + 0.19f64.sqrt() * rng.next_normal();
                u = nu;
                h = nh;
            }
            (traj, micro)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — the G1 fixture battery + determinism + boundary laws
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "closure_audit"))]
mod tests {
    use super::*;

    fn fixture(
        kind: FixtureKind,
        steps: usize,
        dim: usize,
        micro_dim: usize,
        seed: u64,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        build_fixture(&FixtureSpec {
            kind,
            steps,
            dim,
            micro_dim,
            seed,
        })
    }

    /// Flattens a per-step fixture into the flat row-major recording that
    /// [`run`] consumes.
    fn flatten(states: &[Vec<f32>]) -> Vec<f32> {
        states.iter().flatten().copied().collect()
    }

    fn audit(traj: &[Vec<f32>], micro: &[Vec<f32>]) -> ClosureReport {
        let flat = flatten(traj);
        let mflat = flatten(micro);
        let dim = traj[0].len();
        let md = micro[0].len();
        run(
            &[&flat],
            dim,
            Some((&[&mflat], md)),
            &ClosureAuditConfig::default(),
        )
        .expect("audit runs")
    }

    #[test]
    fn identity_fixture_reads_closed() {
        let (traj, micro) = fixture(FixtureKind::Identity, 2048, 8, 8, 11);
        let report = audit(&traj, &micro);
        assert!(
            report.summary_best_r2 > 0.95,
            "identity summary r2 {} must exceed 0.95",
            report.summary_best_r2
        );
        let aug = report.augmented_best_r2.expect("augmented present");
        assert!(aug > 0.95, "identity augmented r2 {aug} must exceed 0.95");
        let gap = report.gap.expect("gap present");
        assert!(gap.abs() < 0.02, "identity gap {gap} must be within ±0.02");
        assert!(
            matches!(report.verdict, ClosureVerdict::Closed { .. }),
            "identity must read Closed, got {:?}",
            report.verdict
        );
    }

    #[test]
    fn noise_fixture_reads_closed() {
        let (traj, micro) = fixture(FixtureKind::Noise, 2048, 8, 8, 22);
        let report = audit(&traj, &micro);
        assert!(
            report.summary_best_r2 >= 0.5,
            "noise summary r2 {} must clear the 0.5 floor",
            report.summary_best_r2
        );
        let gap = report.gap.expect("gap present");
        assert!(
            gap.abs() < 0.03,
            "noise gap {gap} must be within ±0.03 (pure-noise microstate adds nothing)"
        );
        assert!(matches!(report.verdict, ClosureVerdict::Closed { .. }));
    }

    #[test]
    fn injection_fixture_reads_lossy() {
        let (traj, micro) = fixture(FixtureKind::Injection, 2048, 8, 8, 33);
        let report = audit(&traj, &micro);
        assert!(
            report.summary_best_r2 >= 0.5,
            "injection summary r2 {} must clear the floor (expect ≈0.64)",
            report.summary_best_r2
        );
        let gap = report.gap.expect("gap present");
        assert!(
            gap > 0.15,
            "injection gap {gap} must exceed 0.15 (expect ≈0.36)"
        );
        assert!(
            matches!(report.verdict, ClosureVerdict::Lossy { .. }),
            "injection must read Lossy, got {:?}",
            report.verdict
        );
    }

    #[test]
    fn degenerate_plant_never_reads_closed() {
        let (traj, micro) = fixture(FixtureKind::Degenerate, 2048, 2, 2, 44);
        let report = audit(&traj, &micro);
        // The headline: the degenerate abstraction (dim1 = 0.5·dim0) must
        // COLLAPSE closure — the mixture summary cannot see the
        // decomposition, so the plant is Lossy, never Closed.
        assert!(
            !matches!(report.verdict, ClosureVerdict::Closed { .. }),
            "degenerate plant must NOT read Closed, got {:?}",
            report.verdict
        );
        assert!(matches!(report.verdict, ClosureVerdict::Lossy { .. }));
        let gap = report.gap.expect("gap present");
        assert!(
            gap > 0.06,
            "degenerate gap {gap} must exceed 0.06 (expect ≈0.09)"
        );
        assert!(
            report.summary_best_r2 > 0.58 && report.summary_best_r2 < 0.66,
            "degenerate summary r2 {} must sit near the 0.62 pin",
            report.summary_best_r2
        );
        let aug = report.augmented_best_r2.expect("augmented present");
        assert!(
            aug > 0.68,
            "degenerate augmented r2 {aug} must exceed 0.68 (expect ≈0.72)"
        );
    }

    #[test]
    fn determinism_bit_identical() {
        let (traj, micro) = fixture(FixtureKind::Injection, 512, 8, 8, 55);
        let a = audit(&traj, &micro);
        let b = audit(&traj, &micro);
        assert_eq!(a, b, "same input must yield a bit-identical report");
    }

    #[test]
    fn no_microstate_is_undecidable() {
        let (traj, _micro) = fixture(FixtureKind::Noise, 512, 8, 8, 66);
        let flat = flatten(&traj);
        let report = run(&[&flat], 8, None, &ClosureAuditConfig::default()).expect("runs");
        assert_eq!(
            report.verdict,
            ClosureVerdict::Undecidable {
                reason: UndecidableReason::NoMicrostate
            }
        );
        assert!(report.gap.is_none());
        assert!(report.augmented_best_r2.is_none());
        // The summary side was still measured.
        assert!(report.summary_best_r2 > 0.5);
        assert!(!report.arms.is_empty());
    }

    #[test]
    fn weak_support_is_undecidable() {
        let (traj, micro) = fixture(FixtureKind::Identity, 40, 4, 4, 77);
        let flat = flatten(&traj);
        let mflat = flatten(&micro);
        let report = run(
            &[&flat],
            4,
            Some((&[&mflat], 4)),
            &ClosureAuditConfig::default(),
        )
        .expect("runs");
        assert!(matches!(
            report.verdict,
            ClosureVerdict::Undecidable {
                reason: UndecidableReason::WeakSupport { .. }
            }
        ));
    }

    #[test]
    fn tiny_support_reads_undecidable_not_closed() {
        // A near-perfect identity world with 4 held-out windows must never
        // read Closed — the two-condition defense is the point.
        let (traj, micro) = fixture(FixtureKind::Identity, 12, 3, 3, 88);
        let flat = flatten(&traj);
        let mflat = flatten(&micro);
        let report = run(
            &[&flat],
            3,
            Some((&[&mflat], 3)),
            &ClosureAuditConfig::default(),
        )
        .expect("runs");
        assert!(!matches!(report.verdict, ClosureVerdict::Closed { .. }));
    }

    #[test]
    fn verdict_boundary_law() {
        let config = ClosureAuditConfig {
            epsilon: 0.05,
            r2_floor: 0.5,
            support_floor: 128,
            ..ClosureAuditConfig::default()
        };
        // gap exactly at ε reads Closed (the ≤ boundary).
        assert!(matches!(
            closure_verdict(0.05, 0.6, 128, &config),
            ClosureVerdict::Closed { .. }
        ));
        // One ulp over ε reads Lossy.
        assert!(matches!(
            closure_verdict(0.05 + f64::EPSILON, 0.6, 128, &config),
            ClosureVerdict::Lossy { .. }
        ));
        // Weak absolute fit reads Undecidable even with a tiny gap.
        assert_eq!(
            closure_verdict(0.01, 0.4999, 10_000, &config),
            ClosureVerdict::Undecidable {
                reason: UndecidableReason::WeakSummaryFit {
                    summary_r2: 0.4999,
                    r2_floor: 0.5
                }
            }
        );
        // Weak support dominates every other branch.
        assert_eq!(
            closure_verdict(0.5, 0.1, 127, &config),
            ClosureVerdict::Undecidable {
                reason: UndecidableReason::WeakSupport {
                    test_windows: 127,
                    support_floor: 128
                }
            }
        );
    }

    #[test]
    fn knn_cap_caps_references_and_discloses() {
        let (traj, micro) = fixture(FixtureKind::Noise, 512, 4, 4, 99);
        let flat = flatten(&traj);
        let mflat = flatten(&micro);
        let config = ClosureAuditConfig {
            knn_ref_cap: Some(64),
            ..ClosureAuditConfig::default()
        };
        let report = run(&[&flat], 4, Some((&[&mflat], 4)), &config).expect("runs");
        assert_eq!(report.budget.knn_ref_cap, Some(64));
        assert_eq!(report.budget.knn_ref_train, 64);
        // The uncapped posture discloses the full train count.
        let full = audit(&traj, &micro);
        assert_eq!(full.budget.knn_ref_cap, None);
        assert_eq!(full.budget.knn_ref_train, full.support.train_windows);
    }

    #[test]
    fn dim_mismatch_refuses_typed() {
        let (traj, _micro) = fixture(FixtureKind::Noise, 64, 4, 4, 101);
        let flat = flatten(&traj);
        // Count mismatch: two micro rows for one trajectory.
        let two_rows = [vec![0.0f32; 256], vec![0.0f32; 256]];
        let row_a = two_rows[0].as_slice();
        let row_b = two_rows[1].as_slice();
        assert_eq!(
            run(
                &[&flat],
                4,
                Some((&[row_a, row_b], 4)),
                &ClosureAuditConfig::default()
            ),
            Err(ClosureAuditError::MicroLengthMismatch {
                trajectories: 1,
                micro: 2
            })
        );
        // Flat-shape mismatch: a recording whose length is not a multiple
        // of dim.
        let mut wrong_dim = flat.clone();
        wrong_dim.pop();
        assert!(matches!(
            run(&[&wrong_dim], 4, None, &ClosureAuditConfig::default()),
            Err(ClosureAuditError::DimMismatch { .. })
        ));
    }

    #[test]
    fn ring_round_trips_through_run() {
        let (traj, micro) = fixture(FixtureKind::Noise, 400, 4, 4, 202);
        let mut ring = WindowRing::with_capacity(64, 4, Some(4));
        for t in 0..traj.len() - 1 {
            ring.push(&traj[t], &traj[t + 1], Some(&micro[t]));
        }
        assert_eq!(ring.len(), 64);
        assert_eq!(ring.overwritten(), traj.len() - 1 - 64);
        let via_ring = run_from_ring(&ring, &ClosureAuditConfig::default()).expect("runs");
        // The ring holds the NEWEST 64 windows — its audit runs on that
        // suffix, which is a different slice than the full trajectory's.
        // Compare against the equivalent suffix slice instead.
        let suffix = flatten(&traj[traj.len() - 65..]);
        let suffix_micro = flatten(&micro[traj.len() - 65..]);
        let suffix_report = run(
            &[&suffix],
            4,
            Some((&[&suffix_micro], 4)),
            &ClosureAuditConfig::default(),
        )
        .expect("runs");
        assert_eq!(via_ring, suffix_report);
        let full_flat = flatten(&traj);
        let full_micro = flatten(&micro);
        let full = run(
            &[&full_flat],
            4,
            Some((&[&full_micro], 4)),
            &ClosureAuditConfig::default(),
        )
        .expect("runs");
        assert_ne!(via_ring, full);
    }

    #[test]
    fn ring_rejects_shape_drift() {
        let mut ring = WindowRing::with_capacity(4, 3, Some(2));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ring.push(&[0.0; 4], &[0.0; 4], Some(&[0.0; 2]));
        }));
        assert!(
            result.is_err(),
            "dim drift must panic (shape fixed at construction)"
        );
    }

    #[test]
    fn budget_fields_disclosed() {
        let (traj, micro) = fixture(FixtureKind::Identity, 512, 6, 6, 303);
        let report = audit(&traj, &micro);
        let b = &report.budget;
        assert!(b.window_build_ms >= 0.0 && b.total_ms >= b.window_build_ms);
        assert!(b.ridge_summary_ms >= 0.0 && b.ridge_augmented_ms >= 0.0);
        assert!(b.knn_summary_ms >= 0.0 && b.knn_augmented_ms >= 0.0);
        // Arms carry their own walls in menu order: summary ridge (5,
        // λ ascending), augmented ridge (5), summary kNN (3), augmented
        // kNN (3).
        assert_eq!(report.arms.len(), 2 * (5 + 3));
        assert_eq!(report.arms[0].family, ArmFamily::Ridge { lambda: 1e-3 });
        assert_eq!(report.arms[0].side, AuditSide::Summary);
        assert_eq!(report.arms[4].family, ArmFamily::Ridge { lambda: 10.0 });
        assert_eq!(report.arms[5].side, AuditSide::Augmented);
        assert_eq!(report.arms[10].family, ArmFamily::Knn { k: 1 });
        assert_eq!(report.arms[10].side, AuditSide::Summary);
        assert_eq!(report.arms[13].side, AuditSide::Augmented);
        assert_eq!(report.arms[15].family, ArmFamily::Knn { k: 64 });
    }
}
