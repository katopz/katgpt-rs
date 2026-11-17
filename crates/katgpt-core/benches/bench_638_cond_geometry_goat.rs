//! Plan 638 T2.4 — the G2 latency gate: the offline DOCUMENT posture.
//!
//! The paper's own audit shape: n = 1218 (≈ the tokens of one document),
//! D = 64. One ball pass + one `cond_id` + one `cond_imbalance` must
//! complete ≤ 50 ms in release, single-shot measured.
//!
//! **Pre-registered bar (Plan 638 T2.4):** total wall ≤ 50 ms, release,
//! single shot. The box state for any QUOTED number comes from
//! `scripts/bench_preflight.sh` (the plan record quotes its PROVENANCE
//! line; this binary measures the bar itself).

#![cfg(feature = "estimators")]

#[path = "../tests/fixture_cond_geometry/mod.rs"]
mod fixture;

use fixture::Lcg;
use katgpt_core::data_probe::cond_geometry::{
    BallCountScratch, ImbalanceScratch, ball_counts, cond_id, cond_imbalance,
};
use std::time::Instant;

const N: usize = 1218;
const DIM: usize = 64;
const BAR_MS: u128 = 50;

fn main() {
    let mut rng = Lcg::new(0x638_2001);
    let pts: Vec<f32> = (0..N * DIM).map(|_| rng.next_gauss()).collect();
    let to: Vec<f32> = (0..N * DIM).map(|_| rng.next_gauss()).collect();
    let all: Vec<u32> = (0..N as u32).collect();
    let group: Vec<u32> = (0..600u32).collect();

    let mut counts = BallCountScratch::with_capacity(N);
    let mut imb = ImbalanceScratch::with_capacity(N);

    let t0 = Instant::now();
    ball_counts(&pts, N, DIM, &all, 0.95, &mut counts).expect("counts");
    let t_ball = t0.elapsed();
    let d = cond_id(&counts, &group).expect("estimate");
    let t_id = t0.elapsed();
    let delta = cond_imbalance(&pts, &to, N, DIM, &group, &mut imb).expect("imbalance");
    let total = t0.elapsed();

    let ball_ms = t_ball.as_millis();
    let id_ms = t_id.as_millis();
    let total_ms = total.as_millis();
    println!(
        "G2 (n={N}, D={DIM}, release, single-shot): ball {ball_ms} ms | +cond_id {id_ms} ms | +cond_imbalance {total_ms} ms — d̂={d:.3} Δ={delta:.4}"
    );
    if total_ms <= BAR_MS {
        println!("G2 PASS: {total_ms} ms ≤ {BAR_MS} ms");
    } else {
        println!("G2 FAIL: {total_ms} ms > {BAR_MS} ms — the pre-registered bar is missed");
        std::process::exit(1);
    }
}
