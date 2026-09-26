#![cfg(all(feature = "lt2_looped", feature = "kl_depth_probe"))]
//! Issue 898 — KL effective depth: measured exit calibration for the looped
//! runtime (arXiv:2609.19107, `.research/592`). Headline numbers land in
//! `.benchmarks/899_kl_effective_depth_goat.md`.
//!
//! Instrument: `LoopDeepRun { snapshot_every: 1, capture_logits: true }` —
//! the logit lens at every loop, sharing the Issue 717 tripwire's `lm_head`
//! matmul — fed to `katgpt_core::loop_depth_probe`.
//!
//! Gates (pre-stated in the issue, thresholds fixed BEFORE the first run):
//!
//! - **Kill-switch** — `capture_logits` on vs `run: None`: final logits
//!   bit-identical (the lens observes, it does not perturb).
//! - **Lens premise** — the snapshot at loop τ of a T-loop run is
//!   bit-identical to the FINAL readout of a (τ+1)-loop run. Without this,
//!   "effective depth" would not be the natural exit point at all.
//! - **G1b** — determinism (byte-identical KL vectors across two runs),
//!   last-loop KL == 0 exactly, the planted identity-loop canary (write
//!   fraction must read exactly 0 and the stall detector must fire), and the
//!   monotone-decay signature TESTED per checkpoint and reported (never
//!   assumed — a non-monotone checkpoint is a finding, not a failure).
//! - **G1 / G1-flat** — branch decided by the data, rule stated up front:
//!   FLAT iff every sample's oracle exit is loop 1 (the readout argmax never
//!   moves across loops — nothing to calibrate; the hand-tuned defaults
//!   stand, passing negative). Otherwise STRUCTURED: fit the KL threshold on
//!   half A, require holdout hit rate ≥ 0.80 (±1 loop) on half B.
//! - **R9** — retuned threshold (fit on the target checkpoint's half A) vs
//!   the transferred one (fit on a different checkpoint), both scored on
//!   the target's half B. Reported; the transferred arm is the negative
//!   control.
//! - **G2** — probe cost (KL profile over T snapshots) with a box-state
//!   PROVENANCE line. Ceiling asserted only under `--release`.
//! - **G4** — the probe path (capture + KL profile + write fractions) is
//!   allocation-free once warm (tracking allocator, deterministic counters).
//!
//! Run: `cargo test -p katgpt-rs --release --features kl_depth_probe --test issue_898_kl_depth_goat -- --nocapture --test-threads=1`

#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[path = "common/alloc_tracking.rs"]
mod alloc_tracking;

use katgpt_core::loop_depth_probe::{
    DepthHistogram, DepthSample, agreement_exit, effective_depth, fit_threshold, holdout_hit_rate,
    is_monotone_nonincreasing, kl_profile, spread, stall_onset, write_fractions,
};
use katgpt_rs::hla::MultiLayerAhlaCache;
use katgpt_rs::transformer::loop_deep::LoopDeepRun;
use katgpt_rs::transformer::{
    ForwardContext, MultiLayerKVCache, TransformerWeights, forward_looped,
};
use katgpt_rs::types::{Config, HybridPattern, LoopMode, ResidualGate, Rng, SdpaOutputGate};

/// Loop count under calibration (the paper's k ∈ {2..8} flatness range).
const T: usize = 8;
/// Sequences × positions per checkpoint (micro `block_size` = 16).
const N_SEQ: usize = 4;
const N_POS: usize = 16;
/// KL threshold candidates (nats), log-spaced. The first run's grid ended at
/// 0.3 and every checkpoint fit AT that edge — an edge optimum is not an
/// optimum — so the grid was widened to 10 nats (Bench 899 records both).
const CANDIDATES: [f32; 12] = [
    1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0,
];
/// Pre-stated G1 holdout bar (±1 loop).
const G1_HIT_BAR: f32 = 0.80;
const G1_TOL: usize = 1;

/// A served looped fixture: seeded weights + a deterministically
/// constructed residual-gate schedule (no training anywhere).
struct Checkpoint {
    name: &'static str,
    config: Config,
    weights: TransformerWeights,
    gate: ResidualGate,
    sdpa: SdpaOutputGate,
}

fn checkpoint(name: &'static str, seed: u64, gate_decay: Option<f32>, t: usize) -> Checkpoint {
    let mut config = Config::micro();
    config.loop_mode = LoopMode::WeightShared { loop_count: t };
    config.hybrid_pattern = HybridPattern::Uniform;
    config.hla_mode = katgpt_rs::types::HlaMode::Ahla;
    let mut rng = Rng::new(seed);
    let weights = TransformerWeights::new(&config, &mut rng);
    let gate = match gate_decay {
        None => ResidualGate::new(t, config.n_embd),
        Some(d) => ResidualGate::new_loop_stable(t, config.n_embd, d),
    };
    let sdpa = SdpaOutputGate::new(config.n_head, config.head_dim, config.n_embd);
    Checkpoint {
        name,
        config,
        weights,
        gate,
        sdpa,
    }
}

fn checkpoints(t: usize) -> [Checkpoint; 3] {
    [
        checkpoint("seed42-zero-gate", 42, None, t),
        checkpoint("seed7-stable-gate0.2", 7, Some(0.2), t),
        checkpoint("seed1234-stable-gate0.2", 1234, Some(0.2), t),
    ]
}

/// Deterministic token stream for sequence `s`.
fn token(s: usize, pos: usize, vocab: usize) -> usize {
    (s * 7 + pos * 3 + 1) % vocab
}

/// Per-sample measurement: every loop's readout, the final readout, and
/// every loop's state.
struct Sample {
    loop_logits: Vec<Vec<f32>>,
    final_logits: Vec<f32>,
    states: Vec<Vec<f32>>,
}

struct Runner<'a> {
    ck: &'a Checkpoint,
    ctx: ForwardContext,
    cache: MultiLayerKVCache,
    ahla: MultiLayerAhlaCache,
}

impl<'a> Runner<'a> {
    fn new(ck: &'a Checkpoint) -> Self {
        Self {
            ck,
            ctx: ForwardContext::new(&ck.config),
            cache: MultiLayerKVCache::new(&ck.config),
            ahla: MultiLayerAhlaCache::new(&ck.config),
        }
    }

    fn step(&mut self, tok: usize, pos: usize, run: Option<&mut LoopDeepRun>) -> Vec<f32> {
        forward_looped(
            &mut self.ctx,
            &self.ck.weights,
            &mut self.cache,
            &mut self.ahla,
            tok,
            pos,
            &self.ck.config,
            &self.ck.gate,
            &self.ck.sdpa,
            None,
            None,
            #[cfg(feature = "weight_shared_advantage_gate")]
            None,
            None,
            #[cfg(feature = "gain_cost_halt")]
            None,
            run,
            #[cfg(feature = "cadence_gate")]
            None,
        )
        .to_vec()
    }
}

fn lens_run() -> LoopDeepRun {
    let mut run = LoopDeepRun::new(1);
    run.check_logits = false;
    run.capture_logits = true;
    run.capture_states = true;
    run
}

/// Measure every (sequence, position) sample of one checkpoint.
fn measure(ck: &Checkpoint) -> Vec<Sample> {
    let mut out = Vec::with_capacity(N_SEQ * N_POS);
    let mut run = lens_run();
    for s in 0..N_SEQ {
        let mut r = Runner::new(ck);
        for pos in 0..N_POS {
            run.stats.clear();
            let final_logits = r.step(token(s, pos, ck.config.vocab_size), pos, Some(&mut run));
            out.push(Sample {
                loop_logits: run.stats.logit_snapshots().map(<[f32]>::to_vec).collect(),
                final_logits,
                states: run.stats.state_snapshots.clone(),
            });
        }
    }
    out
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map_or(0, |(i, _)| i)
}

fn profile(s: &Sample) -> Vec<f32> {
    let mut kl = Vec::new();
    kl_profile(
        &s.final_logits,
        s.loop_logits.iter().map(Vec::as_slice),
        &mut kl,
    );
    kl
}

fn oracle(s: &Sample) -> usize {
    let am: Vec<usize> = s.loop_logits.iter().map(|l| argmax(l)).collect();
    agreement_exit(&am).expect("non-empty")
}

// ── Kill-switch + lens premise ──────────────────────────────────────────

#[test]
fn kill_switch_capture_is_bit_identical() {
    for ck in checkpoints(T) {
        for s in 0..N_SEQ {
            let (mut a, mut b) = (Runner::new(&ck), Runner::new(&ck));
            let mut run = lens_run();
            for pos in 0..N_POS {
                let tok = token(s, pos, ck.config.vocab_size);
                run.stats.clear();
                let off = a.step(tok, pos, None);
                let on = b.step(tok, pos, Some(&mut run));
                assert!(
                    off.iter().zip(&on).all(|(x, y)| x.to_bits() == y.to_bits()),
                    "{}: capture_logits perturbed the readout at seq {s} pos {pos}",
                    ck.name
                );
                // The last lens snapshot IS the readout.
                let last = run.stats.logit_snapshot(T - 1).expect("T snapshots");
                assert!(
                    last.iter()
                        .zip(&on)
                        .all(|(x, y)| x.to_bits() == y.to_bits())
                );
                assert_eq!(run.stats.logit_snapshot_count(), T);
            }
        }
    }
    println!(
        "[kill-switch] ✅ capture_logits on == run None, bit-identical (3 ckpt × {N_SEQ}×{N_POS})"
    );
}

#[test]
fn lens_premise_snapshot_tau_equals_exit_at_tau() {
    // The loop-τ snapshot of a T-loop run must equal the final readout of a
    // (τ+1)-loop run: the lens reads exactly what exiting there would emit.
    // Each k-loop checkpoint shares the T-loop checkpoint's weights and the
    // first k rows of its gate schedule (same seed / same construction).
    for (i, ck_t) in checkpoints(T).iter().enumerate() {
        let samples = measure(ck_t);
        for k in 1..=T {
            let ck_k = &checkpoints(k)[i];
            let mut idx = 0;
            for s in 0..N_SEQ {
                let mut r = Runner::new(ck_k);
                for pos in 0..N_POS {
                    let out = r.step(token(s, pos, ck_k.config.vocab_size), pos, None);
                    // Positions > 0 read a KV cache written by k-loop passes,
                    // while the T-loop run's cache holds its own passes — so
                    // the premise is exact only at pos 0.
                    if pos == 0 {
                        let lens = &samples[idx].loop_logits[k - 1];
                        assert!(
                            lens.iter()
                                .zip(&out)
                                .all(|(x, y)| x.to_bits() == y.to_bits()),
                            "{}: lens at loop {k} != exit-at-{k} readout (seq {s})",
                            ck_t.name
                        );
                    }
                    idx += 1;
                }
            }
        }
    }
    println!("[lens] ✅ snapshot τ == exit-at-(τ+1) readout, bit-identical at pos 0, k=1..{T}");
}

// ── G1b — determinism, last-loop zero, canary, monotone signature ───────

#[test]
fn g1b_determinism_canary_and_monotone_signature() {
    for ck in checkpoints(T) {
        let (a, b) = (measure(&ck), measure(&ck));
        let mut monotone = 0usize;
        for (sa, sb) in a.iter().zip(&b) {
            let (pa, pb) = (profile(sa), profile(sb));
            assert!(
                pa.iter().zip(&pb).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{}: KL non-deterministic",
                ck.name
            );
            assert_eq!(
                pa[T - 1],
                0.0,
                "{}: last-loop KL must be exactly 0",
                ck.name
            );
            monotone += usize::from(is_monotone_nonincreasing(&pa, 1e-3));
        }
        println!(
            "[G1b] {:<26} monotone-decay signature: {monotone}/{} samples (tested, not assumed)",
            ck.name,
            a.len()
        );
    }
    // Planted identity loop: a state that repeats must write exactly 0 and
    // the stall detector must fire at the first repeat.
    let h = [0.3f32, -1.2, 0.7, 2.0];
    let mut wf = Vec::new();
    write_fractions([&h[..], &h[..], &h[..]], &mut wf);
    assert_eq!(wf, [0.0, 0.0], "identity-loop canary must read exactly 0");
    assert_eq!(
        stall_onset(&wf, 0.0),
        Some(1),
        "stall detector must fire on the identity loop"
    );
    println!("[G1b] ✅ determinism + last-loop KL=0 + identity-loop canary fires");
}

// ── G1 / G1-flat + R9 + reported spectra ────────────────────────────────

struct Calib {
    name: &'static str,
    kl: Vec<Vec<f32>>,
    oracle: Vec<usize>,
    wf: Vec<Vec<f32>>,
    /// Per-sample, per-loop readout argmax.
    am: Vec<Vec<usize>>,
}

impl Calib {
    fn half(&self, parity: usize) -> Vec<DepthSample<'_>> {
        (0..self.kl.len())
            .filter(|i| i % 2 == parity)
            .map(|i| DepthSample {
                kl: &self.kl[i],
                oracle_exit: self.oracle[i],
            })
            .collect()
    }
}

fn calibrate(ck: &Checkpoint) -> Calib {
    let samples = measure(ck);
    let mut wf_buf = Vec::new();
    Calib {
        name: ck.name,
        kl: samples.iter().map(profile).collect(),
        oracle: samples.iter().map(oracle).collect(),
        am: samples
            .iter()
            .map(|s| s.loop_logits.iter().map(|l| argmax(l)).collect())
            .collect(),
        wf: samples
            .iter()
            .map(|s| {
                write_fractions(s.states.iter().map(Vec::as_slice), &mut wf_buf);
                wf_buf.clone()
            })
            .collect(),
    }
}

#[test]
fn g1_holdout_calibration_or_flat_passing_negative() {
    let calibs: Vec<Calib> = checkpoints(T).iter().map(calibrate).collect();
    println!();
    println!(
        "Issue 898 G1 — KL effective depth calibration (T={T}, {} samples/ckpt)",
        N_SEQ * N_POS
    );
    let mut fits = Vec::new();
    for c in &calibs {
        let flat = c.oracle.iter().all(|&o| o == 1);
        let (a, b) = (c.half(0), c.half(1));
        let fit = fit_threshold(&a, &CANDIDATES).expect("non-empty");
        let hit = holdout_hit_rate(&b, fit.threshold, G1_TOL).expect("non-empty");

        // Reported spectra: oracle + predicted depth histograms, loss-vs-k
        // flatness (mean KL at each loop count), write-fraction stall.
        let mut oracle_h = DepthHistogram::<T>::default();
        let mut pred_h = DepthHistogram::<T>::default();
        for (kl, &o) in c.kl.iter().zip(&c.oracle) {
            oracle_h.record(o);
            if let Some(k) = effective_depth(kl, fit.threshold) {
                pred_h.record(k);
            }
        }
        let loss_k: Vec<f32> = (1..T)
            .map(|k| c.kl.iter().map(|p| p[k]).sum::<f32>() / c.kl.len() as f32)
            .collect();
        let max_kl = c.kl.iter().flatten().fold(0.0f32, |m, &v| m.max(v));
        let mean_wf_last = c.wf.iter().map(|w| w[w.len() - 1]).sum::<f32>() / c.wf.len() as f32;
        let mean_wf_first = c.wf.iter().map(|w| w[0]).sum::<f32>() / c.wf.len() as f32;
        let stalls =
            c.wf.iter()
                .filter(|w| stall_onset(w, 1e-3).is_some())
                .count();

        println!(
            "  {:<26} branch={} fit thr={:e} MAE(A)={:.3} unpred={} holdout(B,±{G1_TOL})={:.3}",
            c.name,
            if flat { "FLAT" } else { "STRUCTURED" },
            fit.threshold,
            fit.mean_abs_error,
            fit.unpredicted,
            hit
        );
        println!(
            "    oracle exit hist {:?} (mean {:.2}) | predicted {:?} (mean {:.2})",
            oracle_h.counts,
            oracle_h.mean_in_range().unwrap_or(f64::NAN),
            pred_h.counts,
            pred_h.mean_in_range().unwrap_or(f64::NAN)
        );
        println!(
            "    max KL {max_kl:.3e} nats | loss-vs-k (k=2..{T}) spread {:.3e} | write-frac first {mean_wf_first:.3e} last {mean_wf_last:.3e} | stalled(ε=1e-3) {stalls}/{}",
            spread(&loss_k).unwrap_or(f32::NAN),
            c.wf.len()
        );

        // Calibrated default: a FIXED exit at the p95 effective depth on
        // half A (the KL rule needs the final readout, so it is an offline
        // calibration, not a runtime exit). Scored on half B against running
        // all T loops (the hand-tuned default), at the compute it saves.
        let mut depths_a: Vec<usize> = a
            .iter()
            .filter_map(|s| effective_depth(s.kl, fit.threshold))
            .collect();
        depths_a.sort_unstable();
        let p95 =
            depths_a[((depths_a.len() as f64 * 0.95).ceil() as usize).clamp(1, depths_a.len()) - 1];
        let agree_at = |k: usize| {
            let idx: Vec<usize> = (0..c.kl.len()).filter(|i| i % 2 == 1).collect();
            idx.iter()
                .filter(|&&i| c.am[i][k - 1] == c.am[i][T - 1])
                .count() as f32
                / idx.len() as f32
        };
        let edge = fit.threshold == CANDIDATES[CANDIDATES.len() - 1];
        println!(
            "    calibrated fixed exit (p95 of A) = {p95}/{T} loops ({:.0}% compute saved): argmax agreement on B {:.3} vs 1.000 at T | per-k agreement {:?}{}",
            100.0 * (1.0 - p95 as f64 / T as f64),
            agree_at(p95),
            (1..=T)
                .map(|k| (agree_at(k) * 1000.0).round() / 1000.0)
                .collect::<Vec<_>>(),
            if edge { " | ⚠ fit at grid EDGE" } else { "" }
        );
        let mut wf_at_exit: Vec<f32> =
            c.wf.iter()
                .zip(&c.oracle)
                .filter(|&(_, &o)| o >= 2)
                .map(|(w, &o)| w[o - 2])
                .collect();
        wf_at_exit.sort_by(f32::total_cmp);
        if !wf_at_exit.is_empty() {
            println!(
                "    write fraction of the step INTO the oracle exit: median {:.3e} (n={}) — the ε a write-fraction halt would need",
                wf_at_exit[wf_at_exit.len() / 2],
                wf_at_exit.len()
            );
        }

        if flat {
            println!(
                "    → G1-flat PASSING NEGATIVE: argmax never moves across loops; hand-tuned defaults stand"
            );
        } else {
            assert!(
                hit >= G1_HIT_BAR,
                "{}: G1 holdout hit rate {hit:.3} < {G1_HIT_BAR} (±{G1_TOL} loop)",
                c.name
            );
        }
        fits.push(fit.threshold);
    }

    // R9: retuned (own half A) vs transferred (next checkpoint's fit), both
    // scored on the target's half B.
    println!("  R9 retuned vs transferred (scored on target half B, ±{G1_TOL}):");
    for (i, c) in calibs.iter().enumerate() {
        let src = (i + 1) % calibs.len();
        let b = c.half(1);
        let retuned = holdout_hit_rate(&b, fits[i], G1_TOL).expect("non-empty");
        let transferred = holdout_hit_rate(&b, fits[src], G1_TOL).expect("non-empty");
        println!(
            "    {:<26} retuned {retuned:.3} (thr {:e}) | transferred from {} {transferred:.3} (thr {:e})",
            c.name, fits[i], calibs[src].name, fits[src]
        );
    }
}

// ── G2 — probe cost with provenance ─────────────────────────────────────

fn provenance() -> String {
    let load = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unavailable".into());
    format!(
        "loadavg {load}, profile {}",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}

#[test]
fn g2_probe_cost_with_provenance() {
    let ck = &checkpoints(T)[1];
    let samples = measure(ck);
    let mut kl = Vec::with_capacity(T);
    let iters = 20_000usize;
    let mut sink = 0.0f32;
    let t0 = std::time::Instant::now();
    for i in 0..iters {
        let s = &samples[i % samples.len()];
        kl_profile(
            std::hint::black_box(&s.final_logits),
            s.loop_logits.iter().map(Vec::as_slice),
            &mut kl,
        );
        sink += std::hint::black_box(kl[0]);
    }
    let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box(sink);
    println!(
        "[G2] KL profile T={T} × vocab {}: {ns:.0} ns/profile — PROVENANCE: {}",
        ck.config.vocab_size,
        provenance()
    );
    assert!(ns > 0.0, "timed region vanished (loud zero)");
    if !cfg!(debug_assertions) {
        // One profile must cost well under one looped forward pass.
        assert!(ns < 50_000.0, "G2: {ns:.0} ns/profile ≥ 50 µs ceiling");
    }
}

// ── G4 — zero-alloc probe path once warm ────────────────────────────────

#[test]
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
fn g4_probe_path_is_alloc_free_when_warm() {
    use katgpt_core::alloc::{get_alloc_stats, reset_alloc_stats};
    reset_alloc_stats();
    let _v: Vec<u8> = Vec::with_capacity(8);
    let (sentinel, _) = get_alloc_stats();
    assert!(
        sentinel > 0,
        "TrackingAllocator not installed — alloc gate vacuous"
    );

    let ck = &checkpoints(T)[1];
    let mut r = Runner::new(ck);
    let mut run = lens_run();
    let mut kl = Vec::with_capacity(T);
    let mut wf = Vec::with_capacity(T);
    // Warm twice: the first call grows the snapshot buffers, the first
    // `clear()` after it grows the spare pool that recycles them (8 Vec
    // headers = 192 B, measured). Steady state is the claim.
    for _ in 0..2 {
        run.stats.clear();
        let _ = r.step(1, 0, Some(&mut run));
    }
    let final_logits = run
        .stats
        .logit_snapshot(T - 1)
        .expect("T snapshots")
        .to_vec();

    reset_alloc_stats();
    for _ in 0..8 {
        run.stats.clear();
        // Same position each time: pos 0 re-writes the same cache slot.
        let _ = forward_looped(
            &mut r.ctx,
            &ck.weights,
            &mut r.cache,
            &mut r.ahla,
            1,
            0,
            &ck.config,
            &ck.gate,
            &ck.sdpa,
            None,
            None,
            #[cfg(feature = "weight_shared_advantage_gate")]
            None,
            None,
            #[cfg(feature = "gain_cost_halt")]
            None,
            Some(&mut run),
            #[cfg(feature = "cadence_gate")]
            None,
        );
        kl_profile(&final_logits, run.stats.logit_snapshots(), &mut kl);
        write_fractions(run.stats.state_snapshots.iter().map(Vec::as_slice), &mut wf);
        std::hint::black_box((kl.len(), wf.len()));
    }
    let (count, bytes) = get_alloc_stats();
    assert_eq!(
        count, 0,
        "probe path allocated {count} times ({bytes} B) over 8 warm calls"
    );
    assert_eq!(
        run.stats.state_snapshots.len(),
        T,
        "clear() must not leave stale snapshots"
    );
    println!("[G4] ✅ 8 warm lens runs (capture + KL profile + write fractions): 0 allocations");
}
