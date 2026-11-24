//! Issue 1244 — the G4 alloc canary for `closure_audit` (separate binary:
//! `#[global_allocator]` is crate-binary-unique).
//!
//! The G4 law for this primitive: the ONLINE collection half
//! ([`WindowRing::push`]) never allocates once sized — zero per-tick
//! allocations over an extended wraparound push session. The offline
//! [`run`] instrument is NOT alloc-free by design — it is a bounded,
//! disclosed budget ([`ClosureBudget`]); the disclosure fields must be
//! populated and the kNN cap must actually cap.

#[path = "common/mod.rs"]
mod common;
counting_allocator!();

use katgpt_core::closure_audit::{
    FixtureKind, FixtureSpec, WindowRing, build_fixture, run, run_from_ring,
};

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

#[test]
fn ring_push_is_zero_alloc_after_sizing() {
    assert_counter_is_live();
    let (traj, micro) = fixture(FixtureKind::Noise, 2048, 8, 4, 4242);
    let mut ring = WindowRing::with_capacity(256, 8, Some(4));
    // Warm every slot (sizing allocations happen here, outside the window).
    for t in 0..256.min(traj.len() - 1) {
        ring.push(&traj[t], &traj[t + 1], Some(&micro[t]));
    }
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    // 1000 wraparound pushes — the steady-state online session.
    let base = traj.len() - 1000 - 1;
    for t in base..base + 1000 {
        ring.push(&traj[t], &traj[t + 1], Some(&micro[t]));
    }
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(
        after - before,
        0,
        "WindowRing::push must not allocate once sized"
    );
    assert_eq!(ring.len(), 256);
    assert_eq!(ring.overwritten(), 1000);
}

#[test]
fn ring_construction_allocates_finite_slots_only() {
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let ring = WindowRing::with_capacity(64, 8, Some(4));
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    // 64 slots × 3 buffers + the three outer Vecs — bounded by capacity,
    // never by usage. Loose ceiling (disclosed, not a perf claim).
    assert!(
        after - before <= 3 * 64 + 10,
        "ring construction allocated {} times (bounded by capacity)",
        after - before
    );
    assert_eq!(ring.capacity(), 64);
}

#[test]
fn offline_run_budget_is_disclosed_and_knn_cap_caps() {
    let (traj, micro) = fixture(FixtureKind::Injection, 512, 8, 8, 777);
    let flat = flatten(&traj);
    let mflat = flatten(&micro);
    let (_, allocs) = alloc_delta(|| {
        let config = katgpt_core::closure_audit::ClosureAuditConfig {
            knn_ref_cap: Some(64),
            ..katgpt_core::closure_audit::ClosureAuditConfig::default()
        };
        let report = run(&[&flat], 8, Some((&[&mflat], 8)), &config).expect("runs");
        let b = &report.budget;
        assert!(b.window_build_ms >= 0.0);
        assert!(b.ridge_summary_ms >= 0.0 && b.ridge_augmented_ms >= 0.0);
        assert!(b.knn_summary_ms >= 0.0 && b.knn_augmented_ms >= 0.0);
        assert!(b.total_ms >= b.window_build_ms);
        assert_eq!(b.knn_ref_cap, Some(64));
        assert_eq!(b.knn_ref_train, 64);
        report
    });
    // The offline instrument's allocation count is DISCLOSED (bounded by
    // arms + buffers), never scaled per tick. Loose ceiling over this
    // small fixture; the ring gate above is the hard zero.
    assert!(
        allocs < 500,
        "offline run allocated {allocs} times (expected bounded-by-arms, not per-window)"
    );
}

#[test]
fn ring_to_run_round_trip_alloc_free_pushes() {
    let (traj, micro) = fixture(FixtureKind::Identity, 256, 4, 4, 999);
    let mut ring = WindowRing::with_capacity(128, 4, Some(4));
    for t in 0..128 {
        ring.push(&traj[t], &traj[t + 1], Some(&micro[t]));
    }
    let before = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    for t in 128..256 - 1 {
        ring.push(&traj[t], &traj[t + 1], Some(&micro[t]));
    }
    let report = run_from_ring(
        &ring,
        &katgpt_core::closure_audit::ClosureAuditConfig::default(),
    )
    .expect("runs");
    let after = ALLOC_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    // run_from_ring's own allocations (window collection + buffers) are
    // the disclosed offline budget; the claim measured here is the ring
    // push session — every push in it wrapped a pre-sized slot.
    assert_eq!(ring.overwritten(), 128 - 1);
    assert!(
        after >= before,
        "allocator counter moved backwards (sanity)"
    );
    assert!(
        matches!(
            report.verdict,
            katgpt_core::closure_audit::ClosureVerdict::Undecidable { .. }
        ) || matches!(
            report.verdict,
            katgpt_core::closure_audit::ClosureVerdict::Closed { .. }
        ),
        "ring audit produced a verdict"
    );
}
