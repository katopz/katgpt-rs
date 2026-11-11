//! Convergence instrumentation for the Family-B latent-thought loop (Issue 991).
//!
//! [`LatentThoughtKernel::step`](crate::latent_thought::LatentThoughtKernel::step)
//! runs K inner attractor iterations per tick with the SAME input. Each inner
//! iteration is a fixed-point iteration `s ← 2·σ(W_s·s + W_x·x + b) − 1`; once
//! successive states stop moving in f32, further iterations are exact no-ops —
//! the kernel keeps paying for iterations whose result cannot change. This
//! module answers the premise question behind an early-exit lever: **do real
//! inputs converge before K?**
//!
//! # The census arm (report-only, Issue 991 T1)
//!
//! Armed by the env var `KATGPT_BELIEF_CONVERGENCE_CENSUS=1` (the exact
//! literal arms; unset or any other value disarms — the opt-in env posture).
//! When armed, the K>1 path of the latent-thought `step()` additionally
//! observes the residual ‖s_{t+1} − s_t‖ between successive inner iterations
//! and records, per step() call, the first inner-iteration index where the
//! residual crosses [`CONVERGENCE_EPS`]. The iteration MATH is untouched: the
//! armed path is bit-identical to the disarmed loop (the residual reads a
//! COPY of the previous state; it never participates in the update). When
//! disarmed the only delta on the hot path is one atomic load per step()
//! call.
//!
//! Read + clear the accumulated counts with
//! [`convergence_census_snapshot`] + [`convergence_census_reset`] — the
//! `convergence_census` example at the repo root drives the bench-family
//! input sequences through them, one (family, K) cell at a time. Counts are
//! process-lifetime LOCAL instrumentation, never synced (the latent-vs-raw
//! law); the residual still folds sequentially so the T2 analytic fixture is
//! exact.
//!
//! # The detector (Issue 991 T2 fixture gate)
//!
//! [`ConvergenceDetector`] is the residual statistic the early-exit arm (T4)
//! would consume: residual below `eps` for `window` consecutive inner
//! iterations ⇒ converged at that observation. [`residual_l2`] is the
//! residual: a sequential `+=` fold + one sqrt — no platform-lane SIMD
//! kernel, no allocation.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Pre-registered census/detector threshold (Issue 991 T1/T2): the residual
/// crossing that declares the inner fixed point reached. 1e-6 sits at f32
/// resolution for unit-scale sigmoid states — crossing it usually coincides
/// with the iteration reaching its EXACT f32 fixed point (bit-identical
/// successive states), which is why the census counts exact-zero residuals
/// separately: a bit-identical successor is the STRONGEST early-exit
/// guarantee (further iterations are provable no-ops for a deterministic
/// map), strictly better than the ε-tolerance the issue pre-registered.
pub const CONVERGENCE_EPS: f32 = 1e-6;

/// Census detection window (consecutive crossings required): 1 — the
/// first-crossing index is the informative statistic for "does the input
/// converge before K". The T4 early-exit arm may pick a wider window; the
/// census measures the earliest honest crossing.
pub(crate) const CENSUS_WINDOW: u32 = 1;

/// Histogram slot cap: converged-at indices are clamped to 63 (the sweep
/// never runs K > 64; a crossing beyond that is a `never` row anyway).
const HIST_SLOTS: usize = 64;

/// L2 residual ‖prev − cur‖₂ over successive inner-iteration states.
/// Sequential `+=` fold (determinism posture; the quantity is local-only, so
/// the sync law does not bind — the fold order is pinned anyway to keep the
/// T2 analytic fixture exact).
#[inline]
pub fn residual_l2(prev: &[f32], cur: &[f32]) -> f32 {
    debug_assert_eq!(prev.len(), cur.len(), "residual slice length mismatch");
    let mut acc = 0.0f32;
    for (p, c) in prev.iter().zip(cur.iter()) {
        let d = p - c;
        acc += d * d;
    }
    acc.sqrt()
}

/// The fixed-point convergence detector: fires when the residual has been
/// below `eps` for `window` consecutive observations.
#[derive(Clone, Debug)]
pub struct ConvergenceDetector {
    eps: f32,
    window: u32,
    streak: u32,
    observed: u32,
    fired_at: Option<u32>,
}

impl ConvergenceDetector {
    /// A detector over `eps` requiring `window` consecutive sub-`eps`
    /// observations. `window == 0` is clamped to 1 (a window of zero would
    /// fire without evidence).
    pub fn new(eps: f32, window: u32) -> Self {
        Self {
            eps,
            window: window.max(1),
            streak: 0,
            observed: 0,
            fired_at: None,
        }
    }

    /// Observe one residual. Returns true exactly once, on the observation
    /// where the window is first satisfied.
    pub fn observe(&mut self, residual: f32) -> bool {
        if self.fired_at.is_some() {
            return false;
        }
        let idx = self.observed;
        self.observed += 1;
        if residual < self.eps {
            self.streak += 1;
            if self.streak >= self.window {
                self.fired_at = Some(idx);
                return true;
            }
        } else {
            self.streak = 0;
        }
        false
    }

    /// The 0-based observation index where the detector fired, if it fired.
    pub fn fired_at(&self) -> Option<u32> {
        self.fired_at
    }
}

// ─── The census accumulator (env-armed, process-local) ──────────────────────

struct CensusState {
    /// converged-at histogram: slot = min(fired_at, 63), crossed inputs only.
    hist: [AtomicU64; HIST_SLOTS],
    /// step() calls with NO crossing within K.
    never: AtomicU64,
    /// step() calls observed (armed path only).
    inputs: AtomicU64,
    /// Σ (fired_at + 1) over crossed inputs — the exit points the early-exit
    /// arm would take (1-based inner-iteration index).
    exits_sum: AtomicU64,
    /// Σ (K − (fired_at + 1)) over crossed inputs — inner iterations the
    /// early-exit arm would have skipped at the observed posture.
    saved_iters: AtomicU64,
    /// step() calls whose residual hit EXACTLY 0 within K (bit-identical
    /// successor — the strongest early-exit guarantee class).
    exact_zero: AtomicU64,
}

impl CensusState {
    const fn new() -> Self {
        Self {
            hist: [const { AtomicU64::new(0) }; HIST_SLOTS],
            never: AtomicU64::new(0),
            inputs: AtomicU64::new(0),
            exits_sum: AtomicU64::new(0),
            saved_iters: AtomicU64::new(0),
            exact_zero: AtomicU64::new(0),
        }
    }
}

static CENSUS: OnceLock<Option<CensusState>> = OnceLock::new();

fn census() -> Option<&'static CensusState> {
    CENSUS
        .get_or_init(|| {
            let armed = std::env::var("KATGPT_BELIEF_CONVERGENCE_CENSUS").is_ok_and(|v| v == "1");
            if armed {
                Some(CensusState::new())
            } else {
                None
            }
        })
        .as_ref()
}

/// Whether the census instrumentation is armed in this process (the exact
/// env literal `KATGPT_BELIEF_CONVERGENCE_CENSUS=1`).
pub fn convergence_census_armed() -> bool {
    census().is_some()
}

/// Whether the census has ever been armed in this process — true after a
/// single armed observation even if the counters were since reset. Distinct
/// from [`convergence_census_armed`]: the example uses it to refuse loud on
/// a run that forgot the env var, independent of resets.
pub fn convergence_census_ever_armed() -> bool {
    CENSUS.get().is_some_and(|c| c.is_some())
}

/// Record one step() call's convergence outcome (armed path only; a no-op
/// when disarmed).
pub(crate) fn census_record(k_iters: u8, fired_at: Option<u32>, exact_zero: bool) {
    let Some(state) = census() else { return };
    state.inputs.fetch_add(1, Ordering::Relaxed);
    if exact_zero {
        state.exact_zero.fetch_add(1, Ordering::Relaxed);
    }
    match fired_at {
        Some(t) => {
            let exit = u64::from(t) + 1;
            state.exits_sum.fetch_add(exit, Ordering::Relaxed);
            state
                .saved_iters
                .fetch_add(u64::from(k_iters).saturating_sub(exit), Ordering::Relaxed);
            let slot = (t as usize).min(HIST_SLOTS - 1);
            state.hist[slot].fetch_add(1, Ordering::Relaxed);
        }
        None => {
            state.never.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// One census cell's accumulated counts (a plain value copy of the atomics).
#[derive(Clone, Debug)]
pub struct CensusSnapshot {
    /// step() calls observed.
    pub inputs: u64,
    /// step() calls with NO crossing within K.
    pub never: u64,
    /// Σ (fired_at + 1) over crossed inputs (1-based exit points).
    pub exits_sum: u64,
    /// Σ (K − (fired_at + 1)) over crossed inputs — inner iterations an
    /// early-exit arm at the observed posture would have skipped.
    pub saved_iters: u64,
    /// step() calls whose residual hit exactly 0 within K.
    pub exact_zero: u64,
    /// converged-at histogram over crossed inputs (slot = min(fired_at, 63)).
    pub hist: [u64; HIST_SLOTS],
}

impl Default for CensusSnapshot {
    fn default() -> Self {
        Self {
            inputs: 0,
            never: 0,
            exits_sum: 0,
            saved_iters: 0,
            exact_zero: 0,
            hist: [0; HIST_SLOTS],
        }
    }
}

impl CensusSnapshot {
    /// Crossed inputs (`inputs − never`).
    pub fn crossed(&self) -> u64 {
        self.inputs - self.never
    }

    /// Mean exit point (1-based inner-iteration index) over crossed inputs.
    /// `NaN` when nothing crossed.
    pub fn mean_exit(&self) -> f64 {
        let crossed = self.crossed();
        if crossed == 0 {
            return f64::NAN;
        }
        self.exits_sum as f64 / crossed as f64
    }

    /// The crossed fraction of observed inputs in [0, 1] (0 when empty).
    pub fn crossed_share(&self) -> f64 {
        if self.inputs == 0 {
            return 0.0;
        }
        self.crossed() as f64 / self.inputs as f64
    }
}

/// Copy the accumulated census counts. `None` when the census is disarmed in
/// this process (the env var is not the arming literal).
pub fn convergence_census_snapshot() -> Option<CensusSnapshot> {
    let state = census()?;
    let hist = std::array::from_fn(|i| state.hist[i].load(Ordering::Relaxed));
    Some(CensusSnapshot {
        inputs: state.inputs.load(Ordering::Relaxed),
        never: state.never.load(Ordering::Relaxed),
        exits_sum: state.exits_sum.load(Ordering::Relaxed),
        saved_iters: state.saved_iters.load(Ordering::Relaxed),
        exact_zero: state.exact_zero.load(Ordering::Relaxed),
        hist,
    })
}

/// Zero the accumulated census counts (a census runner calls this between
/// (family, K) cells). No-op when the census was never armed.
pub fn convergence_census_reset() {
    if let Some(Some(state)) = CENSUS.get() {
        for slot in &state.hist {
            slot.store(0, Ordering::Relaxed);
        }
        state.never.store(0, Ordering::Relaxed);
        state.inputs.store(0, Ordering::Relaxed);
        state.exits_sum.store(0, Ordering::Relaxed);
        state.saved_iters.store(0, Ordering::Relaxed);
        state.exact_zero.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Issue 991 T2 fixture gate** — a synthetic recurrence at a known
    /// contraction rate must converge at the analytically-predicted step
    /// (detector fires within ±1 of it).
    ///
    /// Fixture: x_{t+1} = 0.5·x_t per coordinate, x_0 = 1.0, dim = 32. The
    /// residual between observation t−1 and t is 0.5^t · 0.5 · √32 (powers of
    /// 0.5 are EXACT in f32, so the fixture is rounding-free up to the final
    /// sqrt). Crossing ε = 1e-6: 0.5^t < 2·1e-6/√32 = 3.54e-7 ⇒ t > 21.43 ⇒
    /// the detector must fire at observation index 22 ± 1.
    #[test]
    fn contraction_fixture_fires_at_predicted_step() {
        let dim = 32usize;
        let mut x = vec![1.0f32; dim];
        let mut det = ConvergenceDetector::new(CONVERGENCE_EPS, 1);
        let mut fired_at = None;
        for t in 0..32u32 {
            let prev = x.clone();
            for v in x.iter_mut() {
                *v *= 0.5;
            }
            if det.observe(residual_l2(&prev, &x)) {
                fired_at = Some(t);
                break;
            }
        }
        let fired = fired_at.expect("the contraction fixture must fire within 32 steps");
        let predicted = 22;
        assert!(
            (i64::from(fired) - predicted).abs() <= 1,
            "detector fired at {fired}, predicted {predicted}±1 (residual_t = 0.5^t·0.5·√32)"
        );
    }

    /// **Issue 991 T2 fixture gate** — a non-contracting sequence MUST NOT
    /// trip within K.
    #[test]
    fn non_contracting_never_fires() {
        let dim = 32usize;
        let mut x = vec![0.0f32; dim];
        let mut det = ConvergenceDetector::new(CONVERGENCE_EPS, 1);
        for _ in 0..64 {
            let prev = x.clone();
            for v in x.iter_mut() {
                *v += 0.01;
            }
            let r = residual_l2(&prev, &x);
            assert!(r > CONVERGENCE_EPS, "fixture residual must stay above eps");
            assert!(
                !det.observe(r),
                "a non-contracting sequence must never fire"
            );
        }
        assert_eq!(det.fired_at(), None);
    }

    /// Window semantics: an isolated crossing does not fire a window-2
    /// detector; two CONSECUTIVE crossings do (firing names the SECOND).
    #[test]
    fn window_of_two_needs_consecutive_crossings() {
        let mut det = ConvergenceDetector::new(1e-3, 2);
        // crossing, above, crossing, above, then two consecutive crossings.
        let pattern = [1e-4f32, 1e-1, 1e-4, 1e-1, 1e-4, 1e-4];
        let mut fired_at = None;
        for (i, r) in pattern.iter().enumerate() {
            if det.observe(*r) {
                fired_at = Some(i);
                break;
            }
        }
        assert_eq!(
            fired_at,
            Some(5),
            "window 2 fires on the second consecutive crossing"
        );
        assert_eq!(det.fired_at(), Some(5));
        // observe() after firing stays false.
        assert!(!det.observe(1e-5));
    }

    /// Bit-identical successive states are a zero residual — the strongest
    /// early-exit guarantee class the census counts separately.
    #[test]
    fn bit_identical_states_yield_zero_residual() {
        let a = vec![0.37f32; 32];
        let b = a.clone();
        assert_eq!(residual_l2(&a, &b), 0.0);
    }

    /// Sanity on the snapshot/reset lifecycle (armed only — this test does
    /// not require the env var; it checks the DISARMED posture refuses).
    #[test]
    fn disarmed_posture_snapshot_is_none_and_record_is_noop() {
        // This test binary is not launched with the arming env var (the
        // gate suite never sets it), so the census must be disarmed here.
        assert!(
            !convergence_census_armed(),
            "the test gate must not launch with KATGPT_BELIEF_CONVERGENCE_CENSUS=1"
        );
        census_record(3, Some(1), false);
        census_record(3, None, false);
        assert!(convergence_census_snapshot().is_none());
        convergence_census_reset(); // no-op when never armed
        assert!(!convergence_census_ever_armed());
    }

    /// **Issue 991 T2 cost bar**: detector residual cost ≤ 0.5 µs/call at
    /// dim 32 (one norm + compare). Release-only — a dev-profile number
    /// measures the debug build, not the detector:
    /// `cargo test -p katgpt-micro-belief --release --lib detector_cost -- --ignored --nocapture`
    /// Quote the box state (load, power) beside any number from a shared box.
    #[test]
    #[ignore]
    fn detector_cost_bar_dim32_under_500ns() {
        let dim = 32usize;
        let a = vec![0.5f32; dim];
        let b = vec![0.49f32; dim];
        let warmup = 10_000u32;
        let n = 200_000u32;
        let mut sink = 0.0f32;
        for _ in 0..warmup {
            sink += residual_l2(&a, &b);
        }
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            sink += residual_l2(&a, &b);
        }
        let dt = t0.elapsed();
        let per_call_ns = dt.as_nanos() as f64 / f64::from(n);
        assert!(sink.is_finite(), "sink must stay live (dead-loop elision)");
        println!("residual_l2 dim={dim}: {per_call_ns:.1} ns/call over {n} calls ({dt:?})");
        println!("NOTE: box load inflates this — quote box state beside the number");
        assert!(
            per_call_ns <= 500.0,
            "pre-registered bar ≤ 500 ns/call, measured {per_call_ns:.1}"
        );
    }
}
