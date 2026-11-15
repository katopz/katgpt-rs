#![cfg(feature = "estimators")]
//! Plan 638 T2.3 — the G1c planted-reorganization gate.
//!
//! Population: n = 512 points in 8 clusters of 64 (dim 16). Space B plants
//! a neighbor reattachment for a 64-point group X (each X member teleports
//! INDIVIDUALLY to a different cluster — mutual neighborhoods inside X are
//! broken, so from-space nearest neighbors become far in to-space), while
//! the control group Y stays untouched.
//!
//! **Pre-registered bars (Plan 638 T2.3):**
//! - Δ(X) ≥ 2·Δ(Y) — the detector sees the reorganization;
//! - Δ(Y) ≤ 0.1 — the control sits near the preserved-neighborhood floor;
//! - the shuffled-label control (every 8th point — 1/8 of it moved)
//!   reads Δ ≥ 2·Δ(Y) and is disclosed;
//! - **falsification arm:** with B = A (no plant) the ≥2× bar is NOT met
//!   and Δ(X) collapses to the floor — the detector reads the planted
//!   boundary, not the group label.
//!
//! ⚠ Run at `--release` (O(n²·d) arms; the debug wall proves nothing).

#[path = "fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{cond_imbalance, ImbalanceScratch, MIN_GROUP_TOKENS};

const N: usize = 512;
const PER: usize = 64;
const CLUSTERS: usize = 8;
const DIM: usize = 16;
const SEPARATION: f32 = 40.0;

fn cluster_population(rng: &mut Lcg) -> Vec<f32> {
    let mut pts = vec![0.0f32; N * DIM];
    for i in 0..N {
        let c = (i / PER) as f32;
        let row = &mut pts[i * DIM..(i + 1) * DIM];
        for v in row.iter_mut() {
            *v = SEPARATION * c + rng.next_gauss();
        }
    }
    pts
}

/// The planted space: every X member teleports to cluster
/// `2 + (x mod 6)` (clusters 2..7, deterministic — no RNG for the
/// assignment) with fresh noise, breaking mutual neighborhoods
/// individually. Non-X members keep their coordinates.
fn planted_space(a: &[f32], rng: &mut Lcg) -> Vec<f32> {
    let mut b = a.to_vec();
    for x in 0..PER {
        let target = 2 + (x % 6);
        let row = &mut b[x * DIM..(x + 1) * DIM];
        for v in row.iter_mut() {
            *v = SEPARATION * target as f32 + rng.next_gauss();
        }
    }
    b
}

fn mean_group_delta(scratch: &ImbalanceScratch, group: &[u32]) -> f64 {
    let half_n = N as f64 / 2.0;
    let sum: f64 = group
        .iter()
        .map(|&g| scratch.ranks_to[g as usize] as f64 / half_n)
        .sum();
    sum / group.len() as f64
}

#[test]
fn g1c_planted_reorganization_detected_and_falsified() {
    const {
        assert!(N == CLUSTERS * PER, "cluster geometry");
    }
    const {
        assert!(PER >= MIN_GROUP_TOKENS);
    }
    let mut rng = Lcg::new(0x638_0c01);
    let a = cluster_population(&mut rng);
    let b = planted_space(&a, &mut rng);
    let all: Vec<u32> = (0..N as u32).collect();
    let x: Vec<u32> = (0..PER as u32).collect();
    let y: Vec<u32> = (PER as u32..(2 * PER) as u32).collect();
    let shuffled: Vec<u32> = (0..N as u32).step_by(8).collect();
    assert_eq!(shuffled.len(), 64);
    assert_eq!(
        shuffled.iter().filter(|g| **g < PER as u32).count(),
        8,
        "1/8 of the shuffled set is moved rows"
    );

    // One call computes the FULL per-point map; the group deltas are
    // reads of it.
    let mut scratch = ImbalanceScratch::with_capacity(N);
    let delta_all = cond_imbalance(&a, &b, N, DIM, &all, &mut scratch).expect("imbalance");
    let delta_x = mean_group_delta(&scratch, &x);
    let delta_y = mean_group_delta(&scratch, &y);
    let delta_shuf = mean_group_delta(&scratch, &shuffled);

    println!(
        "G1c: Δ(X)={delta_x:.4} Δ(Y)={delta_y:.4} Δ(shuffled)={delta_shuf:.4} Δ(all)={delta_all:.4}"
    );
    assert!(
        delta_x >= 2.0 * delta_y,
        "G1c FAILED: Δ(X) {delta_x} < 2·Δ(Y) {delta_y} — the detector does not see the reorganization"
    );
    assert!(
        delta_x >= 0.5,
        "G1c FAILED: Δ(X) {delta_x} not above the meaningful-signal floor 0.5 — a degenerate zero-zero tie must never pass"
    );
    assert!(
        delta_y <= 0.1,
        "G1c FAILED: control Δ(Y) {delta_y} above the preserved-neighborhood floor 0.1"
    );
    assert!(
        delta_shuf >= 2.0 * delta_y,
        "G1c FAILED: shuffled control {delta_shuf} not above 2·Δ(Y)"
    );

    // ── falsification arm: B = A (no plant) — the verdict must FLIP ──
    // (the planted Δ(X) = {delta_x:.2} collapses to the preserved-neighborhood
    // floor; the ≥2× bar is not the flip condition on an exact-zero floor,
    // the collapse is).
    let mut scratch = ImbalanceScratch::with_capacity(N);
    cond_imbalance(&a, &a, N, DIM, &all, &mut scratch).expect("identity imbalance");
    let delta_x_f = mean_group_delta(&scratch, &x);
    let delta_y_f = mean_group_delta(&scratch, &y);
    println!("G1c falsified: Δ(X)={delta_x_f:.4} Δ(Y)={delta_y_f:.4}");
    assert!(
        delta_x_f <= 0.1,
        "G1c FAILED: falsification arm reads planted — Δ(X) {delta_x_f} did not collapse to the floor (the detector sees the group label, not the boundary)"
    );
    assert!(
        delta_x_f < delta_x / 2.0,
        "G1c FAILED: falsified Δ(X) {delta_x_f} retains half the planted signal {delta_x}"
    );
}
