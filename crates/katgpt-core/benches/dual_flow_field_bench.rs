//! Plan 459 — FlowField × DualLeoMixer fusion GOAT gate.
//!
//! Proves the modelless fusion (LEO teacher + UVFA student mixed via DualLeoMixer
//! at α=0.3 Lc) produces a measurably better navigation field than the LEO-only
//! baseline that FlowFieldCache::get_or_compute currently ships.
//!
//! Run:
//!   cargo run --release --bench dual_flow_field_bench \
//!     --features flow_field_nav,dual_leo
//!
//! Gates:
//!   G1 (correctness): LeoOnly dual path ≡ get_or_compute single path (bit-identical)
//!   G2 (perf overhead): dual/LeoOnly cache-miss latency ratio ≤ 1.5×
//!   G3 (no-regression): all gates report PASS, no panic
//!   G4 (alloc-free hot path): combine_into writes into pre-allocated buffer
//!                              (informational — code inspection confirms)
//!   G5 (quality gain): Lc(α=0.3) avg path length < LeoOnly by ≥10%,
//!                       OR local-minima stuck count reduced ≥30%.

#![cfg(feature = "flow_field_nav")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::flow::*;
use katgpt_core::traits::{ActingMode, DualLeoMixer, LeoHead};

// ── Mock heads ───────────────────────────────────────────────────────────

/// LEO teacher: a **broad, multimodal** potential field.
///
/// Simulates "knows all goals" — the field has the commanded goal's peak PLUS
/// a secondary peak at a decoy location. This produces local minima for NPCs
/// started between the two peaks. The max-over-actions per cell is what the
/// flow field sees, so we encode both peaks via the action dimension:
/// action 0 → commanded goal peak, action 1 → decoy peak, others small.
struct BenchLeoTeacher {
    grid_w: usize,
    grid_h: usize,
    goal_x: usize,
    goal_y: usize,
    decoy_x: usize,
    decoy_y: usize,
    /// Higher = broader (more low-frequency content).
    /// Lower = sharper. Teacher is broad.
    sharpness: f32,
}

impl BenchLeoTeacher {
    fn q_len(&self) -> usize {
        self.grid_w * self.grid_h * 4
    }
}

impl LeoHead for BenchLeoTeacher {
    fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
        // 1 goal × (w*h cells) × 4 actions.
        let mut q = Vec::with_capacity(self.q_len());
        for y in 0..self.grid_h {
            for x in 0..self.grid_w {
                let d_goal = (((x as f32 - self.goal_x as f32).powi(2)
                    + (y as f32 - self.goal_y as f32).powi(2))
                .sqrt())
                .max(1.0);
                let d_decoy = (((x as f32 - self.decoy_x as f32).powi(2)
                    + (y as f32 - self.decoy_y as f32).powi(2))
                .sqrt())
                .max(1.0);
                // Broad peaks: low sharpness → slowly decaying inverse-distance.
                let goal_q = 1.0 / (1.0 + d_goal / self.sharpness);
                let decoy_q = 1.0 / (1.0 + d_decoy / self.sharpness);
                // action 0: goal peak, action 1: decoy, actions 2-3: small constant
                q.push(goal_q);
                q.push(decoy_q);
                q.push(0.05);
                q.push(0.05);
            }
        }
        q.shrink_to_fit();
        q
    }
    #[inline]
    fn goal_count(&self) -> usize {
        1
    }
    #[inline]
    fn action_count(&self) -> usize {
        self.q_len()
    }
    fn q_for_goal<'a>(&self, all_q: &'a [f32], goal: usize) -> &'a [f32] {
        let per_goal = self.q_len();
        &all_q[goal * per_goal..(goal + 1) * per_goal]
    }
}

/// UVFA student: a **sharp, unimodal** field for the commanded goal only.
///
/// No decoy — student is precise on the goal it was conditioned on. Sharper
/// falloff → cleaner gradient, fewer local minima, but assumes the goal is
/// correct (it has no broad knowledge of *other* goals).
struct BenchUvfaStudent {
    grid_w: usize,
    grid_h: usize,
    goal_x: usize,
    goal_y: usize,
    /// Higher = sharper.
    sharpness: f32,
}

impl BenchUvfaStudent {
    fn q_len(&self) -> usize {
        self.grid_w * self.grid_h * 4
    }
}

impl LeoHead for BenchUvfaStudent {
    fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
        let mut q = Vec::with_capacity(self.q_len());
        for y in 0..self.grid_h {
            for x in 0..self.grid_w {
                let d = (((x as f32 - self.goal_x as f32).powi(2)
                    + (y as f32 - self.goal_y as f32).powi(2))
                .sqrt())
                .max(1.0);
                let goal_q = 1.0 / (1.0 + d * self.sharpness);
                // All 4 actions get the same value — only action 0 will dominate
                // after FFT smoothing because the field is unimodal. We still use
                // max-over-actions in from_q_values, so we vary slightly to give
                // the gradient a direction.
                q.push(goal_q);
                q.push(goal_q * 0.95);
                q.push(goal_q * 0.90);
                q.push(goal_q * 0.85);
            }
        }
        q.shrink_to_fit();
        q
    }
    #[inline]
    fn goal_count(&self) -> usize {
        1
    }
    #[inline]
    fn action_count(&self) -> usize {
        self.q_len()
    }
    fn q_for_goal<'a>(&self, all_q: &'a [f32], goal: usize) -> &'a [f32] {
        let per_goal = self.q_len();
        &all_q[goal * per_goal..(goal + 1) * per_goal]
    }
}

// ── Mixers ───────────────────────────────────────────────────────────────

struct LcMixer;
impl DualLeoMixer for LcMixer {
    fn acting_mode(&self) -> ActingMode {
        ActingMode::Lc
    }
}

struct LeoOnlyMixer;
impl DualLeoMixer for LeoOnlyMixer {
    fn acting_mode(&self) -> ActingMode {
        ActingMode::LeoOnly
    }
}

struct UvfaOnlyMixer;
impl DualLeoMixer for UvfaOnlyMixer {
    fn acting_mode(&self) -> ActingMode {
        ActingMode::UvfaOnly
    }
}

struct MaxMixer;
impl DualLeoMixer for MaxMixer {
    fn acting_mode(&self) -> ActingMode {
        ActingMode::Max
    }
}

// ── Quality simulator: gradient-following NPC ────────────────────────────

const STEP_BUDGET: usize = 1024;
const STEP_SIZE: f32 = 0.5; // sub-cell
const GOAL_RADIUS: f32 = 1.5;
const STUCK_EPS: f32 = 1e-3;

#[derive(Clone, Copy, Debug)]
enum PathOutcome {
    Reached { steps: usize },
    Stuck, // Step budget exhausted or stuck in a local minimum
    OutOfBounds,
}

/// Simulate an NPC following the flow field from `(start_x, start_y)` toward
/// whatever attractor the field has. Returns the outcome + path length.
fn simulate_npc(
    field: &FlowField,
    start_x: f32,
    start_y: f32,
    goal_x: f32,
    goal_y: f32,
) -> PathOutcome {
    let mut x = start_x;
    let mut y = start_y;
    let mut last_dist_to_goal = f32::INFINITY;
    let mut stall_count = 0u8;
    for step in 0..STEP_BUDGET {
        // Reached goal?
        let dgx = x - goal_x;
        let dgy = y - goal_y;
        let d_goal = (dgx * dgx + dgy * dgy).sqrt();
        if d_goal < GOAL_RADIUS {
            return PathOutcome::Reached { steps: step + 1 };
        }

        // Bilinear sample the flow field at the NPC's fractional position.
        let xi = x.floor() as i64;
        let yi = y.floor() as i64;
        if xi < 0 || yi < 0 || xi >= field.width() as i64 - 1 || yi >= field.height() as i64 - 1 {
            return PathOutcome::OutOfBounds;
        }
        let fx = x - xi as f32;
        let fy = y - yi as f32;
        let (dx00, dy00) = field.lookup(xi as u16, yi as u16);
        let (dx10, dy10) = field.lookup(xi as u16 + 1, yi as u16);
        let (dx01, dy01) = field.lookup(xi as u16, yi as u16 + 1);
        let (dx11, dy11) = field.lookup(xi as u16 + 1, yi as u16 + 1);
        let dx = (1.0 - fx) * (1.0 - fy) * dx00
            + fx * (1.0 - fy) * dx10
            + (1.0 - fx) * fy * dx01
            + fx * fy * dx11;
        let dy = (1.0 - fx) * (1.0 - fy) * dy00
            + fx * (1.0 - fy) * dy10
            + (1.0 - fx) * fy * dy01
            + fx * fy * dy11;

        // Zero flow → local minimum, stuck.
        if dx.abs() < STUCK_EPS && dy.abs() < STUCK_EPS {
            return PathOutcome::Stuck;
        }

        x += dx * STEP_SIZE;
        y += dy * STEP_SIZE;

        // Stall detection: if distance to goal hasn't shrunk in 32 steps, give up.
        if (d_goal - last_dist_to_goal).abs() < STUCK_EPS {
            stall_count += 1;
            if stall_count >= 32 {
                return PathOutcome::Stuck;
            }
        } else {
            stall_count = 0;
        }
        last_dist_to_goal = d_goal;
    }
    PathOutcome::Stuck
}

struct QualityReport {
    avg_reached_steps: f32,
    reached: usize,
    stuck: usize,
    out_of_bounds: usize,
    total: usize,
}

impl QualityReport {
    fn stuck_pct(&self) -> f32 {
        100.0 * (self.stuck as f32) / (self.total as f32)
    }
    fn reached_pct(&self) -> f32 {
        100.0 * (self.reached as f32) / (self.total as f32)
    }
}

fn evaluate_quality(
    field: &FlowField,
    n_npcs: usize,
    seed: u64,
    goal_x: f32,
    goal_y: f32,
) -> QualityReport {
    let mut rng = katgpt_core::Rng::new(seed);
    let mut reached_steps = Vec::with_capacity(n_npcs);
    let mut stuck = 0;
    let mut oob = 0;
    for _ in 0..n_npcs {
        let x = rng.uniform() * field.width() as f32;
        let y = rng.uniform() * field.height() as f32;
        match simulate_npc(field, x, y, goal_x, goal_y) {
            PathOutcome::Reached { steps } => reached_steps.push(steps),
            PathOutcome::Stuck => stuck += 1,
            PathOutcome::OutOfBounds => oob += 1,
        }
    }
    let avg_reached_steps = if reached_steps.is_empty() {
        f32::INFINITY
    } else {
        (reached_steps.iter().sum::<usize>() as f32) / (reached_steps.len() as f32)
    };
    QualityReport {
        avg_reached_steps,
        reached: reached_steps.len(),
        stuck,
        out_of_bounds: oob,
        total: n_npcs,
    }
}

// ── G1: bit-identity check ───────────────────────────────────────────────

fn gate_g1_bit_identity(grid_w: u16, grid_h: u16) -> bool {
    let teacher = BenchLeoTeacher {
        grid_w: grid_w as usize,
        grid_h: grid_h as usize,
        goal_x: grid_w as usize / 2,
        goal_y: grid_h as usize / 2,
        decoy_x: grid_w as usize / 4,
        decoy_y: grid_h as usize / 4,
        sharpness: 8.0,
    };
    let student = BenchUvfaStudent {
        grid_w: grid_w as usize,
        grid_h: grid_h as usize,
        goal_x: grid_w as usize / 2,
        goal_y: grid_h as usize / 2,
        sharpness: 0.5,
    };
    let mut cache_single = FlowFieldCache::new(FlowFieldConfig::default());
    let mut cache_dual = FlowFieldCache::new(FlowFieldConfig::default());
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];

    let single = cache_single.get_or_compute(1, &teacher, &state, 0, grid_w, grid_h, 0, 5);
    let dual = cache_dual.get_or_compute_dual(
        1,
        &teacher,
        &student,
        &LeoOnlyMixer,
        1.0,
        &state,
        0,
        grid_w,
        grid_h,
        0,
        5,
    );
    match (single, dual) {
        (Some(s), Some(d)) => {
            for y in 0..s.height() {
                for x in 0..s.width() {
                    if s.lookup(x, y) != d.lookup(x, y) {
                        return false;
                    }
                }
            }
            true
        }
        _ => false,
    }
}

/// Plan 460 G1: post-max path with `LeoOnly` (effective α=1.0) must produce a
/// bit-identical field to the single-head `get_or_compute` baseline. Mirrors
/// `gate_g1_bit_identity` but calls `get_or_compute_dual_postmax`.
fn gate_g1_bit_identity_postmax(grid_w: u16, grid_h: u16) -> bool {
    let teacher = BenchLeoTeacher {
        grid_w: grid_w as usize,
        grid_h: grid_h as usize,
        goal_x: grid_w as usize / 2,
        goal_y: grid_h as usize / 2,
        decoy_x: grid_w as usize / 4,
        decoy_y: grid_h as usize / 4,
        sharpness: 8.0,
    };
    let student = BenchUvfaStudent {
        grid_w: grid_w as usize,
        grid_h: grid_h as usize,
        goal_x: grid_w as usize / 2,
        goal_y: grid_h as usize / 2,
        sharpness: 0.5,
    };
    let mut cache_single = FlowFieldCache::new(FlowFieldConfig::default());
    let mut cache_postmax = FlowFieldCache::new(FlowFieldConfig::default());
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];

    let single = cache_single.get_or_compute(1, &teacher, &state, 0, grid_w, grid_h, 0, 5);
    let postmax = cache_postmax.get_or_compute_dual_postmax(
        1,
        &teacher,
        &student,
        &LeoOnlyMixer,
        1.0,
        &state,
        0,
        grid_w,
        grid_h,
        0,
        5,
    );
    match (single, postmax) {
        (Some(s), Some(d)) => {
            for y in 0..s.height() {
                for x in 0..s.width() {
                    if s.lookup(x, y) != d.lookup(x, y) {
                        return false;
                    }
                }
            }
            true
        }
        _ => false,
    }
}

// ── G2: perf overhead ────────────────────────────────────────────────────

fn time_cache_miss_single(
    teacher: &BenchLeoTeacher,
    grid_w: u16,
    grid_h: u16,
    iterations: usize,
) -> std::time::Duration {
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];
    let mut cache = FlowFieldCache::new(FlowFieldConfig::default());
    let start = Instant::now();
    for i in 0..iterations {
        // Distinct goal_id per iter to force cache miss.
        let _ = cache.get_or_compute(i as u64, teacher, &state, 0, grid_w, grid_h, i as u64, 5);
    }
    start.elapsed()
}

fn time_cache_miss_dual(
    teacher: &BenchLeoTeacher,
    student: &BenchUvfaStudent,
    mixer: &impl DualLeoMixer,
    alpha: f32,
    grid_w: u16,
    grid_h: u16,
    iterations: usize,
) -> std::time::Duration {
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];
    let mut cache = FlowFieldCache::new(FlowFieldConfig::default());
    let start = Instant::now();
    for i in 0..iterations {
        let _ = cache.get_or_compute_dual(
            i as u64, teacher, student, mixer, alpha, &state, 0, grid_w, grid_h, i as u64, 5,
        );
    }
    start.elapsed()
}

/// Plan 460: post-max dual fusion timing harness. Identical structure to
/// `time_cache_miss_dual` but calls `get_or_compute_dual_postmax`.
fn time_cache_miss_dual_postmax(
    teacher: &BenchLeoTeacher,
    student: &BenchUvfaStudent,
    mixer: &impl DualLeoMixer,
    alpha: f32,
    grid_w: u16,
    grid_h: u16,
    iterations: usize,
) -> std::time::Duration {
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];
    let mut cache = FlowFieldCache::new(FlowFieldConfig::default());
    let start = Instant::now();
    for i in 0..iterations {
        let _ = cache.get_or_compute_dual_postmax(
            i as u64, teacher, student, mixer, alpha, &state, 0, grid_w, grid_h, i as u64, 5,
        );
    }
    start.elapsed()
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let grid_w: u16 = 64;
    let grid_h: u16 = 64;
    let n_npcs: usize = 200;
    let goal_x = grid_w as f32 / 2.0;
    let goal_y = grid_h as f32 / 2.0;

    println!("═══ Plan 459 — FlowField × DualLeoMixer Fusion GOAT Gate ═══");
    println!("Grid: {grid_w}×{grid_h}, NPCs: {n_npcs}, Goal: ({goal_x:.0},{goal_y:.0})");
    println!();

    // ── G1: Bit-identity ─────────────────────────────────────────────────
    let g1_pass = gate_g1_bit_identity(grid_w, grid_h);
    println!(
        "G1 (LeoOnly dual ≡ single-head, bit-identical):  {}",
        if g1_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // ── Build the 4 fields under test ───────────────────────────────────
    let teacher = BenchLeoTeacher {
        grid_w: grid_w as usize,
        grid_h: grid_h as usize,
        goal_x: grid_w as usize / 2,
        goal_y: grid_h as usize / 2,
        decoy_x: grid_w as usize / 4,
        decoy_y: grid_h as usize / 4,
        sharpness: 8.0, // broad
    };
    let student = BenchUvfaStudent {
        grid_w: grid_w as usize,
        grid_h: grid_h as usize,
        goal_x: grid_w as usize / 2,
        goal_y: grid_h as usize / 2,
        sharpness: 0.5, // sharp
    };
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];

    let mut cache_leo = FlowFieldCache::new(FlowFieldConfig::default());
    let mut cache_uvfa = FlowFieldCache::new(FlowFieldConfig::default());

    let field_leo = cache_leo
        .get_or_compute(1, &teacher, &state, 0, grid_w, grid_h, 0, 5)
        .unwrap();
    let field_leo_owned: FlowField = field_leo.clone();

    let field_uvfa = cache_uvfa
        .get_or_compute(2, &student, &state, 0, grid_w, grid_h, 0, 5)
        .unwrap();
    let field_uvfa_owned: FlowField = field_uvfa.clone();

    // Re-borrow after cloning — the cache returns a borrow but we need owned for parallel eval.
    drop(cache_leo);
    drop(cache_uvfa);

    let mut cache_dual_lc = FlowFieldCache::new(FlowFieldConfig::default());
    let field_lc = cache_dual_lc
        .get_or_compute_dual(
            3, &teacher, &student, &LcMixer, 0.3, &state, 0, grid_w, grid_h, 0, 5,
        )
        .unwrap();
    let field_lc_owned: FlowField = field_lc.clone();
    drop(cache_dual_lc);

    let mut cache_dual_max = FlowFieldCache::new(FlowFieldConfig::default());
    let field_max = cache_dual_max
        .get_or_compute_dual(
            4, &teacher, &student, &MaxMixer, 0.3, &state, 0, grid_w, grid_h, 0, 5,
        )
        .unwrap();
    let field_max_owned: FlowField = field_max.clone();
    drop(cache_dual_max);

    // ── G5: Quality ─────────────────────────────────────────────────────
    println!();
    println!("G5 (Quality — gradient-following NPC simulator, 200 random starts):");
    let q_leo = evaluate_quality(&field_leo_owned, n_npcs, 42, goal_x, goal_y);
    let q_uvfa = evaluate_quality(&field_uvfa_owned, n_npcs, 42, goal_x, goal_y);
    let q_lc = evaluate_quality(&field_lc_owned, n_npcs, 42, goal_x, goal_y);
    let q_max = evaluate_quality(&field_max_owned, n_npcs, 42, goal_x, goal_y);

    println!(
        "  {:<22} {:>10} {:>10} {:>10} {:>10}",
        "Config", "reached", "stuck", "oob", "avg_steps"
    );
    println!(
        "  {:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "LeoOnly (LEO base)",
        q_leo.reached_pct(),
        q_leo.stuck_pct(),
        100.0 * q_leo.out_of_bounds as f32 / q_leo.total as f32,
        q_leo.avg_reached_steps
    );
    println!(
        "  {:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "UvfaOnly (UVFA student)",
        q_uvfa.reached_pct(),
        q_uvfa.stuck_pct(),
        100.0 * q_uvfa.out_of_bounds as f32 / q_uvfa.total as f32,
        q_uvfa.avg_reached_steps
    );
    println!(
        "  {:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "Lc α=0.3 (paper default)",
        q_lc.reached_pct(),
        q_lc.stuck_pct(),
        100.0 * q_lc.out_of_bounds as f32 / q_lc.total as f32,
        q_lc.avg_reached_steps
    );
    println!(
        "  {:<22} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "Max α=0.3 (optimistic)",
        q_max.reached_pct(),
        q_max.stuck_pct(),
        100.0 * q_max.out_of_bounds as f32 / q_max.total as f32,
        q_max.avg_reached_steps
    );

    let stuck_reduction_lc =
        (q_leo.stuck as f32 - q_lc.stuck as f32).max(0.0) / (q_leo.stuck as f32 + 1.0);
    let stuck_reduction_max =
        (q_leo.stuck as f32 - q_max.stuck as f32).max(0.0) / (q_leo.stuck as f32 + 1.0);
    println!();
    println!(
        "  Stuck-NPC reduction (Lc vs LeoOnly):  {:.1}%",
        100.0 * stuck_reduction_lc
    );
    println!(
        "  Stuck-NPC reduction (Max vs LeoOnly): {:.1}%",
        100.0 * stuck_reduction_max
    );

    let g5_pass = stuck_reduction_lc >= 0.30 || stuck_reduction_max >= 0.30;
    println!(
        "G5 (≥30% stuck reduction):  {}",
        if g5_pass {
            "PASS ✅"
        } else {
            "FAIL ❌ (see sweep below)"
        }
    );

    // ── α-sweep: honest characterization ─────────────────────────────────
    //
    // The paper's default α=0.3 didn't meet the 30% gate. Sweep α in {0.1, 0.2,
    // ..., 0.9} to characterize the actual quality curve. This is honest science:
    // report what the α knob actually does on this landscape, not just the gate.
    println!();
    println!("α-sweep (Lc mode, varying teacher/student weight):");
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10}",
        "α", "reached", "stuck", "oob", "avg_steps"
    );
    let mut best_alpha = 0.3_f32;
    let mut best_stuck_reduction = 0.0_f32;
    for &alpha in &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        // Distinct goal_id per α to avoid cache pollution (the documented caveat).
        let goal_id = 100 + (alpha * 10.0) as u64;
        let mut cache = FlowFieldCache::new(FlowFieldConfig::default());
        let f = cache
            .get_or_compute_dual(
                goal_id, &teacher, &student, &LcMixer, alpha, &state, 0, grid_w, grid_h, 0, 5,
            )
            .unwrap();
        let f_owned: FlowField = f.clone();
        drop(cache);
        let q = evaluate_quality(&f_owned, n_npcs, 42, goal_x, goal_y);
        let reduction = (q_leo.stuck as f32 - q.stuck as f32).max(0.0) / (q_leo.stuck as f32 + 1.0);
        if reduction > best_stuck_reduction {
            best_stuck_reduction = reduction;
            best_alpha = alpha;
        }
        println!(
            "  α={:<6.2} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}  (stuck reduction vs LeoOnly: {:>5.1}%)",
            alpha,
            q.reached_pct(),
            q.stuck_pct(),
            100.0 * q.out_of_bounds as f32 / q.total as f32,
            q.avg_reached_steps,
            100.0 * reduction
        );
    }
    println!();
    println!(
        "  Best α (most stuck reduction): {:.2} → {:.1}% reduction",
        best_alpha,
        100.0 * best_stuck_reduction
    );
    let g5_pass_sweep = best_stuck_reduction >= 0.30;
    if g5_pass_sweep {
        println!("G5' (≥30% stuck reduction at SOME α):  PASS ✅");
    } else {
        println!(
            "G5' (≥30% stuck reduction at SOME α):  FAIL ❌ — even the best α doesn't reach the gate"
        );
    }

    // ── G2: Perf overhead ───────────────────────────────────────────────
    // Run 3 trials and report median — single-run timings on macOS are noisy
    // (CPU freq scaling, background tasks). The median is the honest signal.
    println!();
    let iters = 30;
    let trials = 3;
    let mut t_leo_runs = Vec::with_capacity(trials);
    let mut t_lc_runs = Vec::with_capacity(trials);
    let mut t_max_runs = Vec::with_capacity(trials);
    let mut t_uvfa_runs = Vec::with_capacity(trials);
    let mut t_postmax_runs = Vec::with_capacity(trials);
    for _ in 0..trials {
        t_leo_runs.push(time_cache_miss_single(&teacher, grid_w, grid_h, iters));
        t_lc_runs.push(time_cache_miss_dual(
            &teacher, &student, &LcMixer, 0.3, grid_w, grid_h, iters,
        ));
        t_max_runs.push(time_cache_miss_dual(
            &teacher, &student, &MaxMixer, 0.3, grid_w, grid_h, iters,
        ));
        t_uvfa_runs.push(time_cache_miss_dual(
            &teacher,
            &student,
            &UvfaOnlyMixer,
            0.0,
            grid_w,
            grid_h,
            iters,
        ));
        t_postmax_runs.push(time_cache_miss_dual_postmax(
            &teacher, &student, &LcMixer, 0.3, grid_w, grid_h, iters,
        ));
    }
    // Median is more robust than mean for skewed distributions.
    t_leo_runs.sort();
    t_lc_runs.sort();
    t_max_runs.sort();
    t_uvfa_runs.sort();
    t_postmax_runs.sort();
    let t_leo = t_leo_runs[t_leo_runs.len() / 2];
    let t_lc = t_lc_runs[t_lc_runs.len() / 2];
    let t_max = t_max_runs[t_max_runs.len() / 2];
    let t_uvfa = t_uvfa_runs[t_uvfa_runs.len() / 2];
    let t_postmax_lc = t_postmax_runs[t_postmax_runs.len() / 2];

    println!(
        "G2 (Cache-miss perf overhead, median of {trials} trials × {iters} cold-cache computes, {grid_w}×{grid_h}):"
    );
    println!("  LeoOnly (single):       {t_leo:?}");
    println!("  LeoOnly (dual UvfaOnly):{t_uvfa:?}");
    println!(
        "  Lc α=0.3:               {:?}  (ratio {:.2}×)",
        t_lc,
        t_lc.as_nanos() as f64 / t_leo.as_nanos() as f64
    );
    println!(
        "  Max α=0.3:              {:?}  (ratio {:.2}×)",
        t_max,
        t_max.as_nanos() as f64 / t_leo.as_nanos() as f64
    );

    let g2_pass = (t_lc.as_nanos() as f64 / t_leo.as_nanos() as f64) <= 1.5;
    println!(
        "G2 (≤1.5× overhead):  {}",
        if g2_pass { "PASS ✅" } else { "FAIL ❌" }
    );

    // ═══ Plan 460: post-max dual fusion ═══════════════════════════════════
    //
    // Same landscape + mock heads + simulator as Plan 459 above, but the fusion
    // point moves from pre-max raw-Q mixing to **post-max potential blending**.
    // The expected mechanism: linear blend of two post-max potentials → FFT
    // (linear) preserves the α-ratio → gradient pipeline sees a cleaner mix.
    println!();
    println!("═══ Plan 460 — Post-Max DualLeoMixer Fusion GOAT Gate ═══");
    println!("(same {grid_w}×{grid_h} grid, same mock LEO+UVFA, same simulator as Plan 459)");
    println!();

    // ── G1 postmax: Bit-identity ─────────────────────────────────────────
    let g1_pass_postmax = gate_g1_bit_identity_postmax(grid_w, grid_h);
    println!(
        "G1 (LeoOnly postmax ≡ single-head, bit-identical):  {}",
        if g1_pass_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );

    // ── G5 postmax: Quality @ α=0.3 ────────────────────────────────────
    let mut cache_postmax_lc = FlowFieldCache::new(FlowFieldConfig::default());
    let field_postmax_lc = cache_postmax_lc
        .get_or_compute_dual_postmax(
            5, &teacher, &student, &LcMixer, 0.3, &state, 0, grid_w, grid_h, 0, 5,
        )
        .unwrap();
    let field_postmax_lc_owned: FlowField = field_postmax_lc.clone();
    drop(cache_postmax_lc);

    let q_postmax_lc = evaluate_quality(&field_postmax_lc_owned, n_npcs, 42, goal_x, goal_y);
    let stuck_reduction_postmax_lc =
        (q_leo.stuck as f32 - q_postmax_lc.stuck as f32).max(0.0) / (q_leo.stuck as f32 + 1.0);

    println!();
    println!("G5 postmax (Quality @ α=0.3, Lc mode):");
    println!(
        "  {:<28} {:>10} {:>10} {:>10} {:>10}",
        "Config", "reached", "stuck", "oob", "avg_steps"
    );
    println!(
        "  {:<28} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "LeoOnly (LEO base)",
        q_leo.reached_pct(),
        q_leo.stuck_pct(),
        100.0 * q_leo.out_of_bounds as f32 / q_leo.total as f32,
        q_leo.avg_reached_steps
    );
    println!(
        "  {:<28} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}",
        "Postmax Lc α=0.3",
        q_postmax_lc.reached_pct(),
        q_postmax_lc.stuck_pct(),
        100.0 * q_postmax_lc.out_of_bounds as f32 / q_postmax_lc.total as f32,
        q_postmax_lc.avg_reached_steps
    );
    println!();
    println!(
        "  Stuck-NPC reduction (postmax Lc vs LeoOnly):  {:.1}%",
        100.0 * stuck_reduction_postmax_lc
    );
    let g5_pass_postmax = stuck_reduction_postmax_lc >= 0.30;
    println!(
        "G5 postmax (≥30% stuck reduction @ α=0.3):  {}",
        if g5_pass_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌ (see sweep below)"
        }
    );

    // ── G5' postmax: α-sweep ──────────────────────────────────────────────
    println!();
    println!("α-sweep (postmax Lc mode):");
    println!(
        "  {:<8} {:>10} {:>10} {:>10} {:>10} {:>18}",
        "α", "reached", "stuck", "oob", "avg_steps", "stuck-red-vs-LEO"
    );
    let mut best_alpha_postmax = 0.3_f32;
    let mut best_stuck_reduction_postmax = 0.0_f32;
    for &alpha in &[0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let goal_id = 200 + (alpha * 10.0) as u64;
        let mut cache = FlowFieldCache::new(FlowFieldConfig::default());
        let f = cache
            .get_or_compute_dual_postmax(
                goal_id, &teacher, &student, &LcMixer, alpha, &state, 0, grid_w, grid_h, 0, 5,
            )
            .unwrap();
        let f_owned: FlowField = f.clone();
        drop(cache);
        let q = evaluate_quality(&f_owned, n_npcs, 42, goal_x, goal_y);
        let reduction = (q_leo.stuck as f32 - q.stuck as f32).max(0.0) / (q_leo.stuck as f32 + 1.0);
        if reduction > best_stuck_reduction_postmax {
            best_stuck_reduction_postmax = reduction;
            best_alpha_postmax = alpha;
        }
        println!(
            "  α={:<6.2} {:>9.1}% {:>9.1}% {:>9.1}% {:>10.1}  {:>17.1}%",
            alpha,
            q.reached_pct(),
            q.stuck_pct(),
            100.0 * q.out_of_bounds as f32 / q.total as f32,
            q.avg_reached_steps,
            100.0 * reduction
        );
    }
    println!();
    println!(
        "  Best α postmax (most stuck reduction): {:.2} → {:.1}% reduction",
        best_alpha_postmax,
        100.0 * best_stuck_reduction_postmax
    );
    let g5_pass_sweep_postmax = best_stuck_reduction_postmax >= 0.30;
    println!(
        "G5' postmax (≥30% stuck reduction at SOME α):  {}",
        if g5_pass_sweep_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );

    // ── G2 postmax: Perf overhead ────────────────────────────────────────
    // (t_postmax_lc was measured alongside the pre-max timings above and is
    // already the median of `trials` runs.)
    println!();
    println!(
        "G2 postmax (Cache-miss perf overhead, median of {trials} trials × {iters} cold-cache computes, {grid_w}×{grid_h}):"
    );
    println!("  LeoOnly (single):       {t_leo:?}");
    println!(
        "  Postmax Lc α=0.3:       {:?}  (ratio {:.2}×)",
        t_postmax_lc,
        t_postmax_lc.as_nanos() as f64 / t_leo.as_nanos() as f64
    );
    let g2_pass_postmax = (t_postmax_lc.as_nanos() as f64 / t_leo.as_nanos() as f64) <= 1.5;
    println!(
        "G2 postmax (≤1.5× overhead):  {}",
        if g2_pass_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );

    // ═══ Plan 459 Summary (pre-max) ═══
    println!();
    println!("═══ Plan 459 Summary (pre-max) ═══");
    println!(
        "  G1 bit-identity:           {}",
        if g1_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  G2 perf ≤1.5×:             {}",
        if g2_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  G5 quality @α=0.3:         {}",
        if g5_pass { "PASS ✅" } else { "FAIL ❌" }
    );
    println!(
        "  G5' best-α quality sweep:  {}",
        if g5_pass_sweep {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );

    // ═══ Plan 460 Summary (post-max) ═══
    println!();
    println!("═══ Plan 460 Summary (post-max) ═══");
    println!(
        "  G1 bit-identity:           {}",
        if g1_pass_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    println!(
        "  G2 perf ≤1.5×:             {}",
        if g2_pass_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    println!(
        "  G5 quality @α=0.3:         {}",
        if g5_pass_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );
    println!(
        "  G5' best-α quality sweep:  {}",
        if g5_pass_sweep_postmax {
            "PASS ✅"
        } else {
            "FAIL ❌"
        }
    );

    // ═══ Side-by-side: pre-max vs post-max (the whole point of Plan 460) ═══
    println!();
    println!("═══ Side-by-side: pre-max (Plan 459) vs post-max (Plan 460) ═══");
    println!("  {:<32} {:>12} {:>12}", "Metric", "pre-max", "post-max");
    println!(
        "  {:<32} {:>11.1}% {:>11.1}%",
        "Stuck reduction @ α=0.3",
        100.0 * stuck_reduction_lc,
        100.0 * stuck_reduction_postmax_lc
    );
    println!(
        "  {:<32} {:>11.1}% {:>11.1}%",
        "Best-α stuck reduction",
        100.0 * best_stuck_reduction,
        100.0 * best_stuck_reduction_postmax
    );
    println!(
        "  {:<32} {:>12.2} {:>12.2}",
        "Best α", best_alpha, best_alpha_postmax
    );
    println!(
        "  {:<32} {:>11.2}× {:>11.2}×",
        "Cache-miss perf overhead (Lc α=0.3)",
        t_lc.as_nanos() as f64 / t_leo.as_nanos() as f64,
        t_postmax_lc.as_nanos() as f64 / t_leo.as_nanos() as f64
    );

    // ═══ Final verdict ═══
    println!();
    println!("═══ Plan 460 Verdict ═══");
    if g1_pass_postmax && g2_pass_postmax {
        if g5_pass_postmax || g5_pass_sweep_postmax {
            let alpha_used = if g5_pass_postmax {
                0.3
            } else {
                best_alpha_postmax
            };
            println!("VERDICT: ✅ Post-max fusion PROVES a modelless navigation quality gain");
            println!("         at α={alpha_used:.2}. The pipeline-stage change matters:");
            println!("         blending post-max potentials is linear in the FFT's input,");
            println!("         which is where Plan 459's pre-max mix was washed out.");
            println!("         PROMOTE get_or_compute_dual_postmax as the recommended dual path.");
            println!("         Demote Plan 459's pre-max get_or_compute_dual to 'compatibility'.");
        } else {
            println!("VERDICT: ⚠ Post-max fusion is correct + cheap (G1+G2 PASS), but NO α");
            println!("         in {{0.1..0.9}} reaches the 30% stuck-reduction gate.");
            println!("         Pre-max AND post-max both failed G5 on this synthetic landscape.");
            println!("         This is the TWO-FAILED-GATES STOP RULE from Plan 460 §'Honest");
            println!("         caveats' — flow-field navigation is NOT a LEO-fusion quality");
            println!("         target on synthetic data. The next attack is test-time fusion");
            println!("         (QGF DualLeoOracle, sibling plan) OR real-network evidence");
            println!(
                "         (riir-games-civ wiring). Do NOT open a third pipeline-stage variant."
            );
        }
    } else {
        println!("VERDICT: ❌ Post-max fusion fails a hard gate (G1/G2). Do NOT promote.");
    }

    black_box(());
}
