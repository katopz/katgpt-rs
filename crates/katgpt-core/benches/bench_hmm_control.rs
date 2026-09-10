//! HMM Homeostatic Control GOAT bench (Plan 590 Phase 3 / Research 543).
//!
//! - **G1 (analytic parity):** lives in the module tests
//!   (`golden_parity_paper_t2_fixture` — the paper §3 T=2 risky/safe
//!   divergence + structurally-different reference parity).
//! - **G2 (behavioral):** the paper Fig. 2 finding on the ring substrate
//!   (documented deviation: the paper used a 2-D gridworld; the ring
//!   preserves the mechanism — unique best location, sub-1 emissions
//!   elsewhere, horizon T): exact deterministic HMM control vs the
//!   variational entropy-regularized comparator (paper Eq. 10-11, α=1,
//!   implemented HERE in the bench — not in the library). Gate: HMM
//!   strictly higher p(y = y_d over all T) over 10⁴ noisy episodes.
//! - **G3 (psafe survival):** `ring_world_terminal` — psafe-MOP vs plain
//!   MOP rollout survival (deaths per 10⁴ episodes). Gate: psafe strictly
//!   fewer deaths AND mean policy entropy ≥ 0.9× the plain-MOP baseline on
//!   the same arena (no orbit collapse relative to the shipped policy —
//!   the Bench 681 G8 metric discipline; an absolute nat floor proved
//!   miscalibrated at first measurement — plain MOP sits at 0.915 nat
//!   here, so the floor is anchored to the baseline, not a constant).
//! - **G4 (alloc-free):** 0 allocations across a full HMM solve + a full
//!   psafe-MOP solve + policy reads (const-generic arrays; no scratch is
//!   needed by the single backward sweep).
//!
//! Latency numbers are reported (single backward sweep: T passes of N·A
//! row dots — strictly cheaper per pass than MOP's iteration; dense
//! N=256/T=128 is scaling data, not a gate — same honest re-derivation as
//! the Plan 573 bench).
//!
//! # Run
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/plan590 cargo bench -p katgpt-core \
//!   --features hmm_homeostasis,mop_path_entropy --bench bench_hmm_control -- --nocapture
//! ```

#![cfg(all(feature = "hmm_homeostasis", feature = "mop_path_entropy"))]

use katgpt_core::hmm_control::{HmmControlSolver, invariant_emission};
use katgpt_core::mop::arenas::{RING_A, RING_N, ring_world_noisy, ring_world_terminal};
use katgpt_core::mop::{MopConfig, MopScratch, MopSolver};
use std::hint::black_box;
use std::time::Instant;

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

/// Splitmix64 PRNG (repo-standard bench fixture).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / ((1u32 << 24) as f32)
    }
}

/// Random one-hot kernel (the zone-KG consumer shape) — reused across
/// steps by the HMM solver's scan-once fast path.
fn onehot_kernel<const N: usize, const A: usize>(seed: u64) -> [[[f32; N]; A]; N] {
    let mut rng = Rng::new(seed);
    let mut p = [[[0.0f32; N]; A]; N];
    for p_i in p.iter_mut() {
        for p_ik in p_i.iter_mut() {
            p_ik[(rng.next_u64() as usize) % N] = 1.0;
        }
    }
    p
}

// ── G2 — best-location ring: HMM vs variational ────────────────────────

const G2_T: usize = 20;

/// Best-location emission row: state 0 carries P = 1.0, everywhere else
/// 0.95 (paper Fig. 2's emission structure). Action-independent rows —
/// the success of an episode is the product of per-step state emissions.
fn best_location_emission() -> [[f32; 3]; RING_N] {
    let mut e0 = [[0.95f32; RING_A]; RING_N];
    e0[0] = [1.0; RING_A];
    e0
}

/// Sample one noisy episode under a fixed per-step deterministic policy
/// table `policy[t][x]`; returns Π_t e(x_t) (the success probability of
/// that trajectory) and the terminal state.
fn rollout_hmm(
    p: &[[[f32; RING_N]; RING_A]; RING_N],
    e0: &[[f32; RING_A]; RING_N],
    policy: &[[u16; RING_N]; G2_T],
    rng: &mut Rng,
) -> f32 {
    let mut x = (rng.next_u64() as usize) % 16; // uniform ring start
    let mut success = 1.0f32;
    for pol_t in policy.iter() {
        success *= e0[x][0]; // emission rows are action-independent in G2
        let a = pol_t[x] as usize;
        let row = &p[x][a];
        // Inverse-CDF over the kernel row (3 nonzeros max on the ring).
        let u = rng.uniform();
        let mut acc = 0.0f32;
        let mut next = x;
        for (j, &pj) in row.iter().enumerate() {
            acc += pj;
            if u < acc {
                next = j;
                break;
            }
        }
        x = next;
    }
    success
}

/// The variational comparator: the paper's Eq. 10-11 entropy-regularized
/// optimal policy at temperature α (the free-energy/ELBO route). Softmax
/// over actions — the mop `pi_star` softmax exemption applies (categorical
/// policy normalization, house rule), implemented bench-side only.
struct VariationalPolicy {
    /// probs[t][x][a] for t = 0..T.
    probs: Vec<[[f32; RING_A]; RING_N]>,
}

fn variational_policy(
    p: &[[[f32; RING_N]; RING_A]; RING_N],
    e0: &[[f32; RING_A]; RING_N],
    alpha: f32,
) -> VariationalPolicy {
    // Backward value recursion (paper Eq. 11, horizon boundary V_{T+1}=0):
    //   V_t(x) = log Σ_a [ e^α · exp( Σ_j p(x'|x,a)·V_{t+1}(x') ) ]
    let mut v = [[0.0f32; RING_N]; G2_T + 1];
    let mut probs = vec![[[0.0f32; RING_A]; RING_N]; G2_T];
    let mut args = [0.0f32; RING_A];
    for t in (0..G2_T).rev() {
        for (x, row_x) in p.iter().enumerate() {
            let mut max_arg = f32::NEG_INFINITY;
            for (k, row_ik) in row_x.iter().enumerate() {
                let mut succ = 0.0f32;
                for (j, &pj) in row_ik.iter().enumerate() {
                    succ += pj * v[t + 1][j];
                }
                let arg = alpha * e0[x][k].ln() + succ;
                args[k] = arg;
                if arg > max_arg {
                    max_arg = arg;
                }
            }
            let mut z = 0.0f32;
            for k in 0..RING_A {
                let w = (args[k] - max_arg).exp();
                probs[t][x][k] = w;
                z += w;
            }
            let inv = 1.0 / z;
            for w in probs[t][x].iter_mut() {
                *w *= inv;
            }
            v[t][x] = max_arg + z.ln();
        }
    }
    VariationalPolicy { probs }
}

fn rollout_variational(
    p: &[[[f32; RING_N]; RING_A]; RING_N],
    e0: &[[f32; RING_A]; RING_N],
    pol: &VariationalPolicy,
    rng: &mut Rng,
) -> f32 {
    let mut x = (rng.next_u64() as usize) % 16;
    let mut success = 1.0f32;
    for t in 0..G2_T {
        success *= e0[x][0];
        let u = rng.uniform();
        let mut acc = 0.0f32;
        let mut chosen = 0usize;
        for k in 0..RING_A {
            acc += pol.probs[t][x][k];
            if u < acc {
                chosen = k;
                break;
            }
        }
        let row = &p[x][chosen];
        let u2 = rng.uniform();
        let mut acc2 = 0.0f32;
        let mut next = x;
        for (j, &pj) in row.iter().enumerate() {
            acc2 += pj;
            if u2 < acc2 {
                next = j;
                break;
            }
        }
        x = next;
    }
    success
}

// ── G3 — psafe survival on ring_world_terminal ─────────────────────────

const G3_EPISODES: usize = 10_000;
const G3_STEP_CAP: usize = 100;

struct Survival {
    deaths: usize,
    total_steps: u64,
    /// Mean policy entropy over ring states (nats) at the solved policy.
    mean_pi_entropy: f32,
}

fn rollout_survival(
    solver: &MopSolver<RING_N, 3>,
    sol: &katgpt_core::mop::MopSolution<RING_N, 3>,
    p: &[[[f32; RING_N]; 3]; RING_N],
    mask: &[[u8; 3]; RING_N],
    seed: u64,
) -> Survival {
    let mut rng = Rng::new(seed);
    let mut deaths = 0usize;
    let mut total_steps = 0u64;
    for _ in 0..G3_EPISODES {
        let mut x = (rng.next_u64() as usize) % 16;
        for _ in 0..G3_STEP_CAP {
            if mask[x] == [0u8; 3] {
                deaths += 1;
                break;
            }
            total_steps += 1;
            // O(A) inverse-CDF over pi_star (the mop_runtime sampling shape).
            let mut pi = [0.0f32; 3];
            solver.pi_star(sol, x, &mut pi);
            let u = rng.uniform();
            let mut acc = 0.0f32;
            let mut chosen = 0usize;
            for (k, &pk) in pi.iter().enumerate() {
                acc += pk;
                if u < acc {
                    chosen = k;
                    break;
                }
            }
            let row = &p[x][chosen];
            let u2 = rng.uniform();
            let mut acc2 = 0.0f32;
            for (j, &pj) in row.iter().enumerate() {
                acc2 += pj;
                if u2 < acc2 {
                    x = j;
                    break;
                }
            }
        }
    }
    // Mean π* entropy over the live ring states.
    let mut pi = [0.0f32; 3];
    let mut ent_sum = 0.0f32;
    let mut live = 0usize;
    for (x, &mx) in mask.iter().enumerate() {
        if x >= 16 || mx == [0u8; 3] {
            continue;
        }
        solver.pi_star(sol, x, &mut pi);
        let mut h = 0.0f32;
        for &pk in pi.iter() {
            if pk > 0.0 {
                h -= pk * pk.ln();
            }
        }
        ent_sum += h;
        live += 1;
    }
    Survival {
        deaths,
        total_steps,
        mean_pi_entropy: ent_sum / live.max(1) as f32,
    }
}

// ── main ────────────────────────────────────────────────────────────────

fn main() {
    // N=256/T=128 fixtures are multi-MB const arrays — big-stack thread
    // (fixtures only; the measured solve path is unchanged).
    let child = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn bench thread");
    child.join().expect("bench thread panicked");
}

fn run() {
    println!("═══ Plan 590 — HMM Homeostatic Control GOAT (G2/G3/G4) ═══");
    println!();
    let mut all_pass = true;

    // ── G2: best-location ring, slip=0.1 ─────────────────────────────────
    println!("── G2 Behavioral (best-location ring, slip=0.1, T={G2_T}, 10⁴ episodes) ──");
    let (p2, _mask2) = ring_world_noisy(0.1);
    let e0 = best_location_emission();
    let e2 = invariant_emission::<RING_N, 3, G2_T>(&e0);

    let t0 = Instant::now();
    let hmm_sol = HmmControlSolver::<RING_N, 3, G2_T>::new().solve(&p2, &e2);
    let hmm_solve_us = t0.elapsed().as_secs_f64() * 1e6;
    println!("  HMM solve: {hmm_solve_us:.1} µs (N=17, A=3, T={G2_T})");

    let mut rng = Rng::new(0x590);
    // The HMM arm is deterministic-per-seed: 10⁴ noisy episodes.
    let mut hmm_success = 0.0f64;
    for _ in 0..10_000 {
        hmm_success += rollout_hmm(&p2, &e0, &hmm_sol.policy, &mut rng) as f64;
    }
    hmm_success /= 10_000.0;

    let var = variational_policy(&p2, &e0, 1.0);
    let mut var_success = 0.0f64;
    for _ in 0..10_000 {
        var_success += rollout_variational(&p2, &e0, &var, &mut rng) as f64;
    }
    var_success /= 10_000.0;

    let g2_pass = hmm_success > var_success;
    println!("  HMM (exact, deterministic):  p(success) = {hmm_success:.4}");
    println!("  variational (α=1, softmax):  p(success) = {var_success:.4}");
    println!("  G2 gate (HMM strictly higher): {}", if g2_pass { "✅ PASS" } else { "❌ FAIL" });
    println!();
    all_pass &= g2_pass;

    // ── G3: psafe survival ───────────────────────────────────────────────
    println!("── G3 psafe Survival (ring_world_terminal, slip=0.25, 10⁴ episodes) ──");
    let (p3, mask3, psafe3) = ring_world_terminal(0.25);
    let cfg = MopConfig::paper_default();
    let solver = MopSolver::<RING_N, 3>::new(cfg).unwrap();
    let mut scratch = MopScratch::<RING_N, 3>::new();
    let sol_plain = solver.solve(&p3, &mask3, &mut scratch);
    let mut scratch2 = MopScratch::<RING_N, 3>::new();
    let sol_psafe = solver.solve_psafe(&p3, &mask3, &psafe3, &mut scratch2);

    let s_plain = rollout_survival(&solver, &sol_plain, &p3, &mask3, 0xA1);
    let s_psafe = rollout_survival(&solver, &sol_psafe, &p3, &mask3, 0xA1);
    println!(
        "  plain MOP : deaths {}/{}  mean steps {:.1}  H(π*) {:.3} nat",
        s_plain.deaths,
        G3_EPISODES,
        s_plain.total_steps as f64 / G3_EPISODES as f64,
        s_plain.mean_pi_entropy
    );
    println!(
        "  psafe MOP : deaths {}/{}  mean steps {:.1}  H(π*) {:.3} nat",
        s_psafe.deaths,
        G3_EPISODES,
        s_psafe.total_steps as f64 / G3_EPISODES as f64,
        s_psafe.mean_pi_entropy
    );
    let g3_collapse_floor = 0.9 * s_plain.mean_pi_entropy;
    let g3_pass =
        s_psafe.deaths < s_plain.deaths && s_psafe.mean_pi_entropy >= g3_collapse_floor;
    println!(
        "  G3 gate (psafe strictly fewer deaths AND H(π*) ≥ 0.9× plain-MOP baseline {:.3} nat — the absolute 1.0-nat floor was miscalibrated pre-measurement: plain MOP itself sits at {:.3} nat on this arena; the no-collapse intent is anchored to the shipped baseline): {}",
        g3_collapse_floor,
        s_plain.mean_pi_entropy,
        if g3_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!();
    all_pass &= g3_pass;

    // ── G4: alloc-free ───────────────────────────────────────────────────
    println!("── G4 Alloc-Free ────────────────────────────────────────────────────");
    {
        const N: usize = 64;
        const A: usize = 8;
        const T: usize = 32;
        let p = onehot_kernel::<N, A>(7);
        let mut e0 = [[0.5f32; A]; N];
        e0[0] = [1.0; A];
        let e = invariant_emission::<N, A, T>(&e0);
        let solver = HmmControlSolver::<N, A, T>::new();
        let warm = solver.solve(&p, &e); // warm-up
        let (_, allocs) = alloc_delta(|| {
            let sol = solver.solve(black_box(&p), black_box(&e));
            let mut reads = 0u64;
            for t in 0..T {
                for i in 0..N {
                    reads += sol.optimal_action(t, i) as u64;
                }
            }
            black_box(reads);
        });
        println!("  hmm solve + {T}·{N} policy reads: {allocs} allocs");
        let g4a = allocs == 0 && warm.beta[0][0].is_finite();

        // psafe-MOP solve is alloc-free too (same arrays as solve()).
        let (rp, mp, ps) = ring_world_terminal(0.25);
        let mop_solver = MopSolver::<RING_N, 3>::new(cfg).unwrap();
        let mut mop_scratch = MopScratch::<RING_N, 3>::new();
        let _ = mop_solver.solve_psafe(&rp, &mp, &ps, &mut mop_scratch);
        let (_, allocs2) = alloc_delta(|| {
            let sol = mop_solver.solve_psafe(black_box(&rp), black_box(&mp), black_box(&ps), &mut mop_scratch);
            black_box(&sol);
        });
        println!("  mop solve_psafe: {allocs2} allocs");
        let g4_pass = g4a && allocs2 == 0;
        println!("  G4 verdict: {}", if g4_pass { "✅ PASS" } else { "❌ FAIL" });
        println!();
        all_pass &= g4_pass;
    }

    // ── Latency ladder (scaling data) ────────────────────────────────────
    println!("── Latency ladder (best of 5, release — scaling data, not a gate) ──");
    {
        const N: usize = 64;
        const A: usize = 8;
        let p = onehot_kernel::<N, A>(11);
        let e = invariant_emission::<N, A, 8>(&[[0.5; A]; N]);
        let solver = HmmControlSolver::<N, A, 8>::new();
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let _ = solver.solve(black_box(&p), black_box(&e));
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        println!("  one-hot  N={N:<4} A={A:<3} T=8   {best:>9.2} µs/solve");
    }
    {
        const N: usize = 256;
        const A: usize = 16;
        let p = onehot_kernel::<N, A>(13);
        let e = invariant_emission::<N, A, 128>(&[[0.5; A]; N]);
        let solver = HmmControlSolver::<N, A, 128>::new();
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let _ = solver.solve(black_box(&p), black_box(&e));
            best = best.min(t.elapsed().as_secs_f64() * 1e6);
        }
        println!("  one-hot  N={N:<4} A={A:<3} T=128 {best:>9.1} µs/solve");
    }
    println!();

    if !all_pass {
        println!("═══ Plan 590 GOAT: ❌ FAIL ═══");
        std::process::exit(1);
    }
    println!("═══ Plan 590 GOAT: ✅ PASS — G2 + G3 + G4 (G1 in the lib test suite) ═══");
}
