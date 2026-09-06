//! GOAT Benchmark for Fourier-Smoothed Flow Fields (Plan 242 T7).
//!
//! Compares individual LEO Q-lookup vs shared FlowField for 100 NPCs.
//! Criterion: >20% CPU improvement required to promote to default.
//!
//! Run: cargo bench --bench flow_field_bench --features flow_field_nav

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use katgpt_core::Rng;
use katgpt_core::flow::*;
use katgpt_core::traits::LeoHead;

// ── Mock LeoHead for benchmarking ──────────────────────────

/// Simulates a LeoHead that returns synthetic Q-values for a grid.
/// Q-values are higher near the goal cell, creating a natural gradient.
struct BenchLeoHead {
    goals: usize,
    actions: usize,
    grid_w: usize,
    grid_h: usize,
    goal_x: usize,
    goal_y: usize,
}

impl BenchLeoHead {
    fn new(
        goals: usize,
        actions: usize,
        grid_w: usize,
        grid_h: usize,
        goal_x: usize,
        goal_y: usize,
    ) -> Self {
        Self {
            goals,
            actions,
            grid_w,
            grid_h,
            goal_x,
            goal_y,
        }
    }
}

impl LeoHead for BenchLeoHead {
    fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
        // For each cell and action, produce Q-values where cells closer to goal
        // have higher values. This creates a smooth potential field.
        let total_cells = self.grid_w * self.grid_h;
        let mut q = Vec::with_capacity(self.goals * total_cells * self.actions);
        for _g in 0..self.goals {
            for y in 0..self.grid_h {
                for x in 0..self.grid_w {
                    // Distance to goal (inverse = higher Q closer to goal)
                    let dist = (((x as f32 - self.goal_x as f32).powi(2)
                        + (y as f32 - self.goal_y as f32).powi(2))
                    .sqrt())
                    .max(1.0);
                    let q_val = 1.0 / dist;
                    for _a in 0..self.actions {
                        q.push(q_val);
                    }
                }
            }
        }
        q.shrink_to_fit();
        q
    }

    fn goal_count(&self) -> usize {
        self.goals
    }

    fn action_count(&self) -> usize {
        self.actions
    }
}

// ── Helpers ──────────────────────────────────────────────────

/// Generate random NPC positions in the grid using our XorShift64 Rng.
fn random_positions(rng: &mut Rng, n: usize, max_x: u16, max_y: u16) -> Vec<(u16, u16)> {
    (0..n)
        .map(|_| {
            let x = (rng.uniform() * max_x as f32) as u16;
            let y = (rng.uniform() * max_y as f32) as u16;
            (x, y)
        })
        .collect()
}

/// Generate random sub-cell NPC positions as (f32, f32).
fn random_positions_f32(rng: &mut Rng, n: usize, max_x: f32, max_y: f32) -> Vec<(f32, f32)> {
    (0..n)
        .map(|_| (rng.uniform() * max_x, rng.uniform() * max_y))
        .collect()
}

// ── Benchmark A: Individual LEO Q-lookup (baseline) ──────────

fn bench_individual_leo(c: &mut Criterion) {
    let grid_w: usize = 64;
    let grid_h: usize = 64;
    let n_npcs: usize = 100;
    let n_goals: usize = 1;
    let actions: usize = 4;

    let head = BenchLeoHead::new(n_goals, actions, grid_w, grid_h, grid_w / 2, grid_h / 2);

    // Pre-generate NPC positions
    let mut rng = Rng::new(42);
    let npc_positions = random_positions(&mut rng, n_npcs, grid_w as u16, grid_h as u16);

    let state = vec![0.0f32; grid_w * grid_h];

    c.bench_function("individual_leo_100_npcs", |b| {
        b.iter(|| {
            // Each NPC does its own Q-lookup per tick
            let all_q = head.all_goals_q(black_box(&state));
            let mut results = Vec::with_capacity(n_npcs);
            for &(x, y) in &npc_positions {
                let cell_idx = (y as usize * grid_w + x as usize) * actions;
                let best_action = all_q[cell_idx..cell_idx + actions]
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by(|a, b| katgpt_core::float_order::cmp_for_max(a.1, b.1))
                    .map_or(0, |(i, _)| i);
                results.push(best_action);
            }
            black_box(&results);
        })
    });
}

// ── Benchmark B: Shared FlowField lookup ──────────

fn bench_flow_field_lookup(c: &mut Criterion) {
    let grid_w: u16 = 64;
    let grid_h: u16 = 64;
    let n_npcs: usize = 100;
    let n_goals: usize = 1;
    let actions: usize = 4;

    let head = BenchLeoHead::new(
        n_goals,
        actions,
        grid_w as usize,
        grid_h as usize,
        grid_w as usize / 2,
        grid_h as usize / 2,
    );

    // Pre-generate NPC positions (sub-cell for bilinear interpolation)
    let mut rng = Rng::new(42);
    let npc_positions = random_positions_f32(&mut rng, n_npcs, grid_w as f32, grid_h as f32);

    // Build flow field once (the "shared" part)
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];
    let all_q = head.all_goals_q(&state);
    let mut grid = LeoPotentialGrid::from_q_values(grid_w, grid_h, &all_q, actions);
    fft_smooth(grid.potential_mut(), grid_w as usize, grid_h as usize, 0.25);
    let flow_field = grid.gradient();

    c.bench_function("flow_field_100_npcs", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(n_npcs);
            for &pos in &npc_positions {
                let (dx, dy) = flow_steering(black_box(&flow_field), pos);
                results.push((dx, dy));
            }
            black_box(&results);
        })
    });
}

// ── Benchmark C: FFT smoothing cost ──────────

fn bench_fft_smooth_cost(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_smooth_grid");
    for size in [32usize, 64, 128] {
        let mut grid_data = vec![0.0f32; size * size];
        // Add a spike in the center
        grid_data[size / 2 * size + size / 2] = 10.0;

        group.throughput(Throughput::Elements((size * size) as u64));
        group.bench_with_input(BenchmarkId::new("grid", size), &size, |b, &sz| {
            b.iter(|| {
                fft_smooth(black_box(&mut grid_data), sz, sz, 0.25);
            });
        });
    }
    group.finish();
}

// ── Benchmark D: Dynamic obstacle response ──────────

fn bench_dynamic_obstacle(c: &mut Criterion) {
    let grid_w: u16 = 64;
    let grid_h: u16 = 64;
    let actions: usize = 4;

    let head = BenchLeoHead::new(
        1,
        actions,
        grid_w as usize,
        grid_h as usize,
        grid_w as usize / 2,
        grid_h as usize / 2,
    );
    let state = vec![0.0f32; grid_w as usize * grid_h as usize];

    c.bench_function("obstacle_recompute", |b| {
        b.iter(|| {
            let all_q = head.all_goals_q(black_box(&state));
            let mut grid = LeoPotentialGrid::from_q_values(grid_w, grid_h, &all_q, actions);
            // Insert obstacle wall
            for y in 20..40 {
                grid.mark_blocked(32, y);
            }
            inflate_obstacles(grid.blocked_mut(), grid_w, grid_h, 1);
            fft_smooth(grid.potential_mut(), grid_w as usize, grid_h as usize, 0.25);
            let _field = grid.gradient();
        })
    });
}

criterion_group!(
    benches,
    bench_individual_leo,
    bench_flow_field_lookup,
    bench_fft_smooth_cost,
    bench_dynamic_obstacle,
);
criterion_main!(benches);
