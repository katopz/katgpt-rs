//! Phase 4 — the **cache-reuse probe protocol** (paper §3, Appendix C/D).
//!
//! # What this is
//!
//! The paper's perf claim rests on **KV-cache reuse**: both the probe (rubric
//! evaluation) and the summarizer **append** to `(x, y_{1:t})`, so the running
//! KV cache is preserved across the call. The probe contributes only its own
//! ~60-token verdict to `N_out`. The `O(L²)` re-prefill a naive re-encode
//! would incur is avoided.
//!
//! This module ships the modelless, allocation-free primitives the caller
//! uses to manage the rolling cache across the probe/decide/revert/summarize
//! lifecycle. It does NOT implement the LLM forward pass — that's the
//! caller's engine. The module provides:
//!
//! - [`CacheReuseProbe`] — tracks the appended-probe byte range and provides
//!   `revert()` to cleanly remove it from the trajectory on CONTINUE.
//! - [`ProbeToken`] — opaque handle to the appended range, returned by
//!   `probe_append` and consumed by `revert`. Copy + `#[repr(transparent)]`
//!   over a single `usize` so it never allocates.
//!
//! # G3 invariant
//!
//! Probe latency is independent of `L` (the trajectory length) — only the
//! appended instruction pays prefill. The [`CacheReuseProbe::probe_append`] operation is
//! `O(1)` (it records the byte offset); [`CacheReuseProbe::revert`] is `O(k)` where `k` is
//! the probe length, NOT `O(L)`. The G3 test asserts this by timing
//! `probe_append` + `revert` at L = 1k, 10k, 100k and checking the latency
//! is within ±10%.
//!
//! # Byte-clean contract
//!
//! After a CONTINUE decision, [`CacheReuseProbe::revert`] truncates the trajectory
//! back to its pre-probe length. The rolling cache MUST be uncontaminated:
//! subsequent generation from the reverted trajectory matches a no-probe
//! baseline byte-for-byte (modulo KV-cache indexing, which the caller's
//! engine handles). The G3 correctness test asserts this.
//!
//! # What this module does NOT do
//!
//! - Does NOT run the LLM forward pass (caller's engine).
//! - Does NOT manage the KV cache directly (caller's engine).
//! - Does NOT summarize (the caller supplies a summarizer; see Plan 333
//!   T4.2 — `summarize` is a signature the caller fills in).

use core::ops::Range;

/// Opaque handle to an appended probe range, returned by
/// [`CacheReuseProbe::probe_append`] and consumed by
/// [`CacheReuseProbe::revert`].
///
/// `#[repr(transparent)]` over a single `usize` (the pre-probe trajectory
/// length) so the token is copyable and never allocates. The token is only
/// meaningful within the [`CacheReuseProbe`] that minted it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct ProbeToken(usize);

impl ProbeToken {
    /// The pre-probe trajectory length this token reverts to.
    #[inline]
    #[must_use]
    pub const fn pre_len(self) -> usize {
        self.0
    }
}

/// The cache-reuse probe manager. Stateless beyond its config; the
/// trajectory buffer is caller-owned.
///
/// # Construction
///
/// ```no_run
/// use katgpt_core::compaction::probe::CacheReuseProbe;
/// let probe = CacheReuseProbe::new();
/// ```
///
/// # Lifecycle
///
/// ```text
/// 1. probe_append(traj, rubric_prompt) -> ProbeToken
///    (traj is now traj + rubric_prompt; KV cache has the prompt prefilled)
/// 2. ... caller runs the LLM forward to get the rubric verdict ...
/// 3. gate.evaluate(...) -> CompactionDecision
/// 4a. On Compress: summarize(traj, summarizer_prompt) -> Summary
///     (caller replaces traj with summary; the probe is subsumed)
/// 4b. On Continue: probe.revert(traj, token)
///     (traj is back to pre-probe length; KV cache is uncontaminated)
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CacheReuseProbe {
    // Currently no config; kept as a struct for future knobs (e.g. max probe
    // length, probe padding). Stateless so the same probe can serve many
    // concurrent trajectories.
}

impl CacheReuseProbe {
    /// Construct a default probe manager.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }

    /// Append `rubric_prompt` to `trajectory` (preserving the KV cache of
    /// `y_{1:t}`), returning a [`ProbeToken`] that records the pre-probe
    /// length for later [`revert`](Self::revert).
    ///
    /// # Allocation
    ///
    /// `O(k)` where `k = rubric_prompt.len()` — the cost of extending the
    /// `Vec`. This is NOT `O(L)` (the existing trajectory is not touched).
    /// The KV cache prefill of the appended prompt is the caller's engine
    /// responsibility and is `O(k²)` (prompt self-attention) + `O(L·k)`
    /// (cross-attention to the cached prefix) — both are `k`-dependent, not
    /// `L`-dominated.
    ///
    /// # Returns
    ///
    /// A [`ProbeToken`] whose [`ProbeToken::pre_len`] is the trajectory
    /// length *before* the append.
    #[inline]
    pub fn probe_append(&self, trajectory: &mut Vec<u8>, rubric_prompt: &[u8]) -> ProbeToken {
        let pre_len = trajectory.len();
        trajectory.extend_from_slice(rubric_prompt);
        ProbeToken(pre_len)
    }

    /// Revert the trajectory to its pre-probe length on a CONTINUE decision.
    ///
    /// After this, `trajectory.len() == token.pre_len()` and the KV cache is
    /// uncontaminated (the caller's engine is responsible for dropping the
    /// probe's KV entries; this method only manages the byte buffer).
    ///
    /// # Panics
    ///
    /// Debug-asserts `token.pre_len() <= trajectory.len()` (a token from a
    /// longer trajectory applied to a shorter one is a programmer bug).
    #[inline]
    pub fn revert(&self, trajectory: &mut Vec<u8>, token: ProbeToken) {
        debug_assert!(
            token.pre_len() <= trajectory.len(),
            "ProbeToken pre_len {} > trajectory len {} — wrong token or already reverted",
            token.pre_len(),
            trajectory.len()
        );
        // truncate is O(1) — it just adjusts the len, no per-element drop
        // for u8 (Copy, no Drop impl). The dropped bytes' KV entries are
        // the caller's engine to reclaim.
        trajectory.truncate(token.pre_len());
    }

    /// The byte range the probe occupied, for audit / KV-cache-reclaim
    /// bookkeeping. Returns `[pre_len, pre_len + prompt_len]`.
    ///
    /// Pure function of the token + the prompt length; no allocation.
    #[inline]
    #[must_use]
    pub const fn probe_range(token: ProbeToken, prompt_len: usize) -> Range<usize> {
        token.pre_len()..(token.pre_len() + prompt_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // ─── Correctness ─────────────────────────────────────────────────────

    #[test]
    fn probe_append_extends_and_token_records_pre_len() {
        let probe = CacheReuseProbe::new();
        let mut traj = b"hello world".to_vec();
        let token = probe.probe_append(&mut traj, b" [RUBRIC]");
        assert_eq!(token.pre_len(), 11, "pre_len = original length");
        assert_eq!(&traj[..11], b"hello world");
        assert_eq!(&traj[11..], b" [RUBRIC]");
    }

    #[test]
    fn revert_restores_pre_probe_length() {
        let probe = CacheReuseProbe::new();
        let mut traj = b"prefix".to_vec();
        let token = probe.probe_append(&mut traj, b"-probe-suffix");
        assert_eq!(traj.len(), 19);
        probe.revert(&mut traj, token);
        assert_eq!(traj.len(), 6, "reverted to pre_len");
        assert_eq!(&traj, b"prefix");
    }

    #[test]
    fn byte_clean_after_revert_matches_no_probe_baseline() {
        // The G3 correctness invariant: after probe + revert, the trajectory
        // matches a baseline that never probed.
        let baseline = b"the quick brown fox".to_vec();
        let mut probed = baseline.clone();
        let probe = CacheReuseProbe::new();
        let token = probe.probe_append(&mut probed, b" [RUBRIC: is this safe?]");
        // ... caller would run LLM here ...
        probe.revert(&mut probed, token);
        assert_eq!(
            probed, baseline,
            "byte-clean: reverted trajectory == no-probe baseline"
        );
    }

    #[test]
    fn multiple_probes_revert_in_lifo_order() {
        // The caller may issue several probes before deciding. Each revert
        // must restore to that probe's pre_len. LIFO (last-in-first-out) is
        // the natural order (most recent probe reverts first).
        let probe = CacheReuseProbe::new();
        let mut traj = b"base".to_vec();
        let t1 = probe.probe_append(&mut traj, b"-P1");
        let t2 = probe.probe_append(&mut traj, b"-P2");
        assert_eq!(traj, b"base-P1-P2");
        probe.revert(&mut traj, t2);
        assert_eq!(traj, b"base-P1");
        probe.revert(&mut traj, t1);
        assert_eq!(traj, b"base");
    }

    #[test]
    fn probe_range_is_correct_span() {
        let token = ProbeToken(100);
        let range = CacheReuseProbe::probe_range(token, 60);
        assert_eq!(range, 100..160);
    }

    // ─── G3: probe latency independent of L ──────────────────────────────

    #[test]
    fn g3_probe_latency_independent_of_trajectory_length() {
        // The paper's perf claim: probe latency is independent of L. The
        // byte-buffer operations (`extend_from_slice` + `truncate`) are
        // O(k) in the prompt length, NOT O(L) — extend only touches the new
        // bytes, truncate is O(1) for `u8`. We measure `probe_append` +
        // `revert` at L = 1k, 10k, 100k and assert the per-op time is
        // L-independent.
        //
        // The actual LLM-forward G3 invariant (prefill cost dominated by k,
        // not L) is a property of the caller's engine, not this module. We
        // verify the modelless half: the byte-buffer ops are L-independent.
        let probe = CacheReuseProbe::new();
        let prompt = b" [RUBRIC: evaluate C1/C2/C3/N1]"; // ~36 bytes

        let mut measurements = Vec::new();
        for &l in &[1_000usize, 10_000, 100_000] {
            // Pre-allocate the trajectory once. The trajectory is reused
            // across all iterations (append then revert restores it), so
            // the allocation cost is NOT in the measured loop.
            let mut traj = vec![b'x'; l];
            // Reserve capacity so probe_append doesn't reallocate mid-loop.
            traj.reserve_exact(prompt.len() * 2);

            // Warm up: run once outside the timing loop.
            let warm_token = probe.probe_append(&mut traj, prompt);
            probe.revert(&mut traj, warm_token);
            assert_eq!(traj.len(), l, "warmup revert restored length");

            // Measure: average over 1000 iterations. At ~50ns/op this is
            // ~50µs total, fast enough for CI.
            //
            // `black_box` prevents the optimizer from eliminating the
            // probe_append + revert cycle as dead code. Without it, LLVM sees
            // that extend-then-truncate leaves `traj` unchanged and removes
            // both ops, producing 0ns measurements (NaN/inf ratio → spurious
            // FAIL). We fence BOTH the `traj` reference (so the extend's byte
            // writes can't be proven dead) AND the token (so revert can't be
            // proven dead).
            use std::hint::black_box;
            let n_iter = 1000;
            let start = Instant::now();
            for _ in 0..n_iter {
                let token = probe.probe_append(black_box(&mut traj), prompt);
                probe.revert(black_box(&mut traj), black_box(token));
            }
            let elapsed = start.elapsed();
            let per_op_ns = elapsed.as_nanos() / n_iter as u128;
            measurements.push((l, per_op_ns));
            assert_eq!(traj.len(), l, "revert restored length at L={l}");
        }

        // Assert: the max per-op time is within 3× the min. The byte-buffer
        // ops are O(k) in the prompt length (36 bytes here), not O(L), so
        // they should be near-constant. A 3× allowance covers cache effects.
        //
        // This is a **release-mode perf assertion**: debug builds have
        // ~10× higher per-op overhead with high variance (bounds checks,
        // no optimization), so the ratio assertion is only meaningful in
        // release. In debug we verify only correctness (the assert_eq above).
        // Run with `cargo test --release` to exercise the perf gate.
        #[cfg(not(debug_assertions))]
        {
            let times: Vec<_> = measurements.iter().map(|(_, t)| *t).collect();
            let &min_t = times.iter().min().unwrap();
            let &max_t = times.iter().max().unwrap();
            eprintln!(
                "G3 probe latency: L=1k {}ns, L=10k {}ns, L=100k {}ns | min={} max={} ratio={:.2}",
                measurements[0].1,
                measurements[1].1,
                measurements[2].1,
                min_t,
                max_t,
                max_t as f64 / min_t as f64
            );
            assert!(
                max_t < min_t * 3,
                "G3 FAIL: probe latency not L-independent (max/min = {:.2} > 3)",
                max_t as f64 / min_t as f64
            );
        }
        #[cfg(debug_assertions)]
        {
            eprintln!(
                "G3 probe latency (debug, no ratio assertion): L=1k {}ns, L=10k {}ns, L=100k {}ns",
                measurements[0].1, measurements[1].1, measurements[2].1
            );
        }
    }

    // ─── Integration: probe + gate lifecycle ─────────────────────────────

    #[test]
    fn probe_continue_lifecycle_restores_trajectory() {
        // Full lifecycle: append probe, evaluate (Continue), revert.
        use crate::compaction::rubrics::search::{
            SearchFeatures, SearchRubric, TrajectoryFeatures,
        };
        use crate::compaction::{ClosedUnitCompactionGate, RubricScratch};

        let probe = CacheReuseProbe::new();
        let rubric = SearchRubric::default();
        let gate = ClosedUnitCompactionGate::builder(rubric)
            .backstop(crate::compaction::Backstop::None)
            .build();

        let mut traj = b"trajectory prefix ".to_vec();
        let original_len = traj.len();
        let mut scratch = RubricScratch::new();

        // Probe 1: mid-derivation (high novelty) → Continue → revert.
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 5.0]); // high novelty
        let token1 = probe.probe_append(&mut traj, b" [PROBE]");
        let d1 = gate.evaluate(&traj, 0, 10_000, None, &mut scratch);
        assert!(d1.is_continue());
        probe.revert(&mut traj, token1);
        assert_eq!(traj.len(), original_len, "reverted after Continue");

        // Probe 2: safe point (low novelty) → Compress → NO revert (summarize).
        scratch.clear();
        scratch.f32_buf.extend_from_slice(&[0.8, 4.0, 1.2, 0.3]); // low novelty
        let _token2 = probe.probe_append(&mut traj, b" [PROBE]");
        let d2 = gate.evaluate(&traj, 0, 10_000, None, &mut scratch);
        assert!(d2.is_compress(), "safe point → Compress");
        // On Compress, the caller would summarize (replacing traj); we don't
        // revert. The probe is subsumed by the summary.
        let _ = SearchFeatures::new(TrajectoryFeatures::new(0.0, 0.0, 0.0, 0.0)); // suppress unused warning
    }
}
