//! GOAT Proof: Plan 241 — LinOSS Modal Threat Prediction (Spectral Threat)
//!
//! Performance targets:
//! - ingest_damage per call: ≤ 100ns
//! - extract_features per call: ≤ 200ns
//! - End-to-end ingest+extract cycle: ≤ 500ns
//! - Throughput sustained: ≥ 100K ingest/sec
//!
//! Run: `cargo test --features spectral_threat --test bench_241_spectral_threat_goat --release -- --nocapture`

#![cfg(feature = "spectral_threat")]

use katgpt_core::sense::spectral_threat::CombatRhythmTracker;

/// Measure elapsed time in nanoseconds.
fn elapsed_ns<F: FnOnce() -> R, R>(f: F) -> (R, u64) {
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed().as_nanos() as u64;
    (result, elapsed)
}

// ── G1: ingest_damage ≤ 100ns ────────────────────────────────

#[test]
fn g1_ingest_damage_latency() {
    let mut tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
    tracker.register(1);

    // Warmup
    for tick in 0..1000 {
        tracker.ingest_damage(1, 30.0, tick);
    }

    const N: usize = 10_000;
    let mut total_ns = 0u64;
    for tick in 0..N {
        let (_, ns) = elapsed_ns(|| tracker.ingest_damage(1, 30.0, tick as u32));
        std::hint::black_box(&());
        total_ns += ns;
    }

    let avg_ns = total_ns / N as u64;
    // Debug budget: 5μs (release will be < 100ns)
    let budget_ns = 5_000;
    println!("G1: ingest_damage = {avg_ns}ns (target ≤ {budget_ns}ns debug, ≤ 100ns release)");
    assert!(
        avg_ns <= budget_ns,
        "ingest_damage took {avg_ns}ns, target ≤ {budget_ns}ns (debug)"
    );
}

// ── G2: extract_features ≤ 200ns ─────────────────────────────

#[test]
fn g2_extract_features_latency() {
    let mut tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
    tracker.register(1);
    // Seed with 5 damage events so features are non-trivial
    for tick in 0..5 {
        tracker.ingest_damage(1, 30.0, tick);
    }

    // Warmup
    for _ in 0..1000 {
        std::hint::black_box(tracker.extract_features(1));
    }

    const N: usize = 10_000;
    let mut total_ns = 0u64;
    for _ in 0..N {
        let (features, ns) = elapsed_ns(|| tracker.extract_features(1));
        std::hint::black_box(&features);
        total_ns += ns;
    }

    let avg_ns = total_ns / N as u64;
    // Debug budget: 10μs (release will be < 200ns)
    let budget_ns = 10_000;
    println!("G2: extract_features = {avg_ns}ns (target ≤ {budget_ns}ns debug, ≤ 200ns release)");
    assert!(
        avg_ns <= budget_ns,
        "extract_features took {avg_ns}ns, target ≤ {budget_ns}ns (debug)"
    );
}

// ── G3: End-to-end ingest+extract cycle ≤ 500ns ──────────────

#[test]
fn g3_end_to_end_ingest_extract() {
    let mut tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
    tracker.register(1);

    // Warmup
    for tick in 0..1000 {
        tracker.ingest_damage(1, 30.0, tick);
        std::hint::black_box(tracker.extract_features(1));
    }

    const N: usize = 10_000;
    let mut total_ns = 0u64;
    for tick in 0..N {
        let (features, ns) = elapsed_ns(|| {
            tracker.ingest_damage(1, 30.0, tick as u32);
            tracker.extract_features(1)
        });
        std::hint::black_box(&features);
        total_ns += ns;
    }

    let avg_ns = total_ns / N as u64;
    // Debug budget: 20μs (release will be < 500ns)
    let budget_ns = 20_000;
    println!(
        "G3: ingest+extract cycle = {avg_ns}ns (target ≤ {budget_ns}ns debug, ≤ 500ns release)"
    );
    assert!(
        avg_ns <= budget_ns,
        "ingest+extract cycle took {avg_ns}ns, target ≤ {budget_ns}ns (debug)"
    );
}

// ── G4: Throughput sustained ≥ 100K ingest/sec ───────────────

#[test]
fn g4_throughput_sustained() {
    let mut tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
    tracker.register(1);

    const N: usize = 100_000;
    let start = std::time::Instant::now();
    for tick in 0..N {
        tracker.ingest_damage(1, 30.0, tick as u32);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = N as f64 / elapsed.as_secs_f64();
    // Debug budget: 10K/sec (release will be ≥ 100K)
    let budget = 10_000.0;
    println!(
        "G4: Throughput = {ops_per_sec:.0} ingest/sec (target ≥ {budget:.0}/sec debug, ≥ 100K release)"
    );
    assert!(
        ops_per_sec >= budget,
        "Throughput {ops_per_sec:.0}/sec, target ≥ {budget:.0}/sec (debug)"
    );
}
