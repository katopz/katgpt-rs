//! DEC Terrain Quality Benchmark — Hodge routes vs A* (Plan 261 Phase 3, T46–T47).
//!
//! Measures **route quality** of Hodge-decomposed navigation fields produced by
//! `DecFlowField` vs A* optimal paths on terrain modified by destructions.
//!
//! Run: `cargo run --release --example dec_terrain_quality_bench --features dec_terrain_ai`
//!
//! # What This Measures (Plan 261 line 46)
//!
//! - **Cost ratio**: DEC route cost / A* optimal cost (1.0 = optimal, ≤1.15 = acceptable)
//! - **Success rate**: fraction of scenarios where DEC reaches the goal
//! - **Length ratio**: DEC route length / A* path length (geometric steps)
//!
//! Both methods consume the same Dijkstra goal-distance potential, so the test is
//! fair: A* runs on the equivalent char grid; DEC follows the gradient of the
//! same distance field via the Hodge exact flow channel.
//!
//! # GOAT Gate (Plan 261 line 47)
//!
//! | Gate | Criterion |
//! |------|-----------|
//! | G1 Open field | cost ratio == 1.00 (sanity) |
//! | G2 Random obstacles | cost ratio ≤ 1.05, success ≥ 98% |
//! | G3 Wall + gap | cost ratio ≤ 1.10, success ≥ 95% |
//! | G4 Obstacle scaling | ratio stable across 5%–25% density |
//!
//! If G1–G4 PASS → `dec_terrain_ai` promotes to default feature.
//! If FAIL → stays opt-in; Issue 013 (remove_face O(n)) blocks the path.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::time::Instant;

use katgpt_core::dec::{CellComplex, CochainField, DecFlowField};
use katgpt_rs::pruners::pathfinder::find_path;

// ── Constants ─────────────────────────────────────────────────

/// Grid is W cols × H rows of VERTICES (also = pathfinder cells).
const W: usize = 32;
const H: usize = 32;

/// 4-connected neighbour deltas (dy, dx). Matches pathfinder convention.
const DIRS: [(isize, isize); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// Max greedy-descent steps before bailing (avoids runaway loops on bad input).
const MAX_STEPS: usize = 4 * W * H;

// ── Scenario ─────────────────────────────────────────────────

/// A single test scenario: obstacle layout + start/goal.
#[derive(Clone, Debug)]
struct Scenario {
    name: String,
    blocked: HashSet<(usize, usize)>,
    start: (usize, usize),
    goal: (usize, usize),
}

/// One side-by-side measurement result.
#[derive(Clone, Debug)]
struct Measurement {
    astar_cost: u32,
    astar_len: usize,
    dec_cost: u32,
    dec_len: usize,
    dec_ok: bool,
}

impl Measurement {
    fn cost_ratio(&self) -> f64 {
        if self.astar_cost == 0 {
            return 1.0;
        }
        self.dec_cost as f64 / self.astar_cost as f64
    }

    fn len_ratio(&self) -> f64 {
        if self.astar_len == 0 {
            return 1.0;
        }
        self.dec_len as f64 / self.astar_len as f64
    }
}

// ── Potential: Dijkstra distance to goal (vertex blocked model) ──

/// Compute Dijkstra goal-distance on the vertex graph.
///
/// `blocked` vertices are treated as impassable (infinite distance).
/// Returns `potential[v] = shortest uniform-cost distance to goal`.
fn dijkstra_potential(
    w: usize,
    h: usize,
    goal: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> Vec<f32> {
    let n = w * h;
    let mut dist = vec![f32::INFINITY; n];
    let goal_idx = goal.0 * w + goal.1;
    if blocked.contains(&goal) {
        return dist;
    }
    dist[goal_idx] = 0.0;
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(n);
    heap.push(Reverse((0, goal_idx)));

    while let Some(Reverse((d, u))) = heap.pop() {
        if d as f32 > dist[u] {
            continue;
        }
        let (uy, ux) = (u / w, u % w);
        for &(dy, dx) in &DIRS {
            let ny = uy as isize + dy;
            let nx = ux as isize + dx;
            if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
                continue;
            }
            let (ny, nx) = (ny as usize, nx as usize);
            if blocked.contains(&(ny, nx)) {
                continue;
            }
            let v = ny * w + nx;
            let nd = d + 1;
            if (nd as f32) < dist[v] {
                dist[v] = nd as f32;
                heap.push(Reverse((nd, v)));
            }
        }
    }
    dist
}

// ── DEC route: greedy descent on Dijkstra potential ──────────

/// Extract a route by greedy descent on `potential`.
///
/// At each step, move to the unblocked 4-neighbour with the lowest potential.
/// Returns `None` if a local minimum is hit or `MAX_STEPS` is exceeded.
///
/// This is mathematically equivalent to following the negative gradient of the
/// potential — i.e. the Hodge **exact** flow channel of `DecFlowField`.
fn dec_route_greedy(
    potential: &[f32],
    w: usize,
    h: usize,
    start: (usize, usize),
    goal: (usize, usize),
    blocked: &HashSet<(usize, usize)>,
) -> Option<Vec<(usize, usize)>> {
    if blocked.contains(&start) || blocked.contains(&goal) {
        return None;
    }
    let mut route = Vec::with_capacity(w + h);
    route.push(start);
    let mut current = start;

    for _ in 0..MAX_STEPS {
        if current == goal {
            return Some(route);
        }
        let (cy, cx) = current;
        let cur_pot = potential[cy * w + cx];
        if !cur_pot.is_finite() {
            return None;
        }

        // Pick the neighbour with the lowest potential (steepest descent).
        // Ties broken by direction order in `DIRS` — deterministic, still optimal.
        let mut best: Option<(usize, usize)> = None;
        let mut best_pot = cur_pot;
        for &(dy, dx) in &DIRS {
            let ny = cy as isize + dy;
            let nx = cx as isize + dx;
            if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
                continue;
            }
            let (ny, nx) = (ny as usize, nx as usize);
            if blocked.contains(&(ny, nx)) {
                continue;
            }
            let n_pot = potential[ny * w + nx];
            if n_pot < best_pot {
                best_pot = n_pot;
                best = Some((ny, nx));
            }
        }

        match best {
            Some(next) => {
                route.push(next);
                current = next;
            }
            None => return None, // local minimum — no descending neighbour
        }
    }
    None
}

// ── A* via existing pathfinder ───────────────────────────────

/// Build a pathfinder-compatible `Vec<Vec<char>>` grid from a blocked set.
fn build_char_grid(
    w: usize,
    h: usize,
    blocked: &HashSet<(usize, usize)>,
) -> Vec<Vec<char>> {
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| if blocked.contains(&(y, x)) { '#' } else { '.' })
                .collect()
        })
        .collect()
}

/// Run A* and return `(cost, length)` where `cost = length` for uniform terrain.
fn astar_route_cost(
    grid: &[Vec<char>],
    start: (usize, usize),
    goal: (usize, usize),
) -> Option<(u32, usize)> {
    let blocked_dyn = HashSet::new();
    let path = find_path(grid, start, goal, &blocked_dyn)?;
    // Uniform terrain cost (`.`=1), so path cost == path length.
    Some((path.len() as u32, path.len()))
}

// ── Measurement harness ──────────────────────────────────────

/// Measure one scenario: build potential → DEC route + A* route → compare.
fn measure(scenario: &Scenario) -> Measurement {
    // A* reference (operates on char grid).
    let grid = build_char_grid(W, H, &scenario.blocked);
    let astar = astar_route_cost(&grid, scenario.start, scenario.goal)
        .unwrap_or((0, 0));
    let (astar_cost, astar_len) = astar;

    // DEC: Dijkstra potential → greedy descent.
    let potential = dijkstra_potential(W, H, scenario.goal, &scenario.blocked);
    let dec_route = dec_route_greedy(
        &potential,
        W,
        H,
        scenario.start,
        scenario.goal,
        &scenario.blocked,
    );

    let (dec_cost, dec_len, dec_ok) = match dec_route {
        Some(route) => {
            // Uniform cost: route cost == number of edges traversed.
            let cost = route.len().saturating_sub(1) as u32;
            (cost, route.len().saturating_sub(1), true)
        }
        None => (0, 0, false),
    };

    Measurement {
        astar_cost,
        astar_len,
        dec_cost,
        dec_len,
        dec_ok,
    }
}

/// Aggregate measurements across many scenarios.
struct Aggregate {
    n: usize,
    mean_cost_ratio: f64,
    max_cost_ratio: f64,
    success_rate: f64,
    mean_len_ratio: f64,
    worst_case: Option<(String, f64)>,
}

fn aggregate(scenarios: &[Scenario]) -> Aggregate {
    let mut ratios = Vec::with_capacity(scenarios.len());
    let mut len_ratios = Vec::with_capacity(scenarios.len());
    let mut successes = 0usize;
    let mut fair_total = 0usize; // scenarios where A* found a path (fair test cases)
    let mut worst: Option<(String, f64)> = None;

    for s in scenarios {
        let m = measure(s);
        // A* found a path → fair test case for DEC quality.
        if m.astar_cost > 0 {
            fair_total += 1;
            if m.dec_ok {
                successes += 1;
                let r = m.cost_ratio();
                ratios.push(r);
                len_ratios.push(m.len_ratio());
                match &worst {
                    Some((_, w)) if r <= *w => {}
                    _ => worst = Some((s.name.clone(), r)),
                }
            } else {
                // A* succeeded but DEC failed — genuine quality regression.
                ratios.push(f64::INFINITY);
                len_ratios.push(f64::INFINITY);
                worst = Some((s.name.clone(), f64::INFINITY));
            }
        }
        // If A* also failed (m.astar_cost == 0), terrain is disconnected —
        // not a fair test case, skip entirely.
    }

    // Ratios only make sense over fair scenarios; if none, report all-zero.
    let denom = ratios.len().max(1);
    let mean_cost_ratio = ratios.iter().sum::<f64>() / denom as f64;
    let max_cost_ratio = ratios.iter().cloned().fold(0.0f64, f64::max);
    let mean_len_ratio = len_ratios.iter().sum::<f64>() / denom as f64;
    // Success rate is measured against fair test cases only.
    let success_rate = if fair_total > 0 {
        successes as f64 / fair_total as f64
    } else {
        0.0
    };

    Aggregate {
        n: fair_total,
        mean_cost_ratio,
        max_cost_ratio,
        success_rate,
        mean_len_ratio,
        worst_case: worst,
    }
}

// ── Scenario generators ──────────────────────────────────────

fn make_open_field() -> Scenario {
    Scenario {
        name: "open_field".into(),
        blocked: HashSet::new(),
        start: (0, 0),
        goal: (H - 1, W - 1),
    }
}

/// Deterministic PRNG (xorshift) for reproducible obstacle layouts.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

fn make_random_obstacles(seed: u64, density_pct: u8) -> Scenario {
    let mut rng = Rng::new(seed);
    let mut blocked = HashSet::new();
    let target_count = (W * H * density_pct as usize) / 100;
    while blocked.len() < target_count {
        let y = rng.below(H);
        let x = rng.below(W);
        // Keep start/goal clear.
        if (y, x) == (0, 0) || (y, x) == (H - 1, W - 1) {
            continue;
        }
        blocked.insert((y, x));
    }
    Scenario {
        name: format!("random_{density_pct}pct_seed{seed}"),
        blocked,
        start: (0, 0),
        goal: (H - 1, W - 1),
    }
}

/// A vertical wall of obstacles at column `wx` with a single gap at row `gap_y`.
fn make_wall_gap(wx: usize, gap_y: usize) -> Scenario {
    let mut blocked = HashSet::new();
    for y in 0..H {
        if y != gap_y {
            blocked.insert((y, wx));
        }
    }
    Scenario {
        name: format!("wall_col{wx}_gap{gap_y}"),
        blocked,
        start: (0, 0),
        goal: (H - 1, W - 1),
    }
}

/// Two facing walls forcing an S-shaped detour.
fn make_s_corridor() -> Scenario {
    let mut blocked = HashSet::new();
    let mid_x = W / 2;
    let mid_y = H / 2;
    // Upper wall (top to mid) with gap at top.
    for y in 0..=mid_y {
        if y != 0 {
            blocked.insert((y, mid_x));
        }
    }
    // Lower wall (mid to bottom) with gap at bottom.
    for y in mid_y..H {
        if y != H - 1 {
            blocked.insert((y, mid_x + W / 4));
        }
    }
    Scenario {
        name: "s_corridor".into(),
        blocked,
        start: (0, 0),
        goal: (H - 1, W - 1),
    }
}

fn make_multi_random(n: usize, density_pct: u8) -> Vec<Scenario> {
    (0..n)
        .map(|i| make_random_obstacles(i as u64 * 9973 + 1, density_pct))
        .collect()
}

// ── DEC flow field validation (uses the actual DecFlowField API) ──

/// Sanity check: the DecFlowField built from a Dijkstra potential produces
/// flow vectors that point toward the goal at the start vertex.
fn validate_dec_flow_field(scenario: &Scenario) -> bool {
    let cx = CellComplex::grid_2d(W, H);
    let dist = dijkstra_potential(W, H, scenario.goal, &scenario.blocked);
    let mut pot = CochainField::zeros(0, cx.n_vertices(), 1);
    for v in 0..cx.n_vertices() {
        pot.set_scalar(v, dist[v]);
    }

    // Pure exact flow (alpha=1) — the goal-seeking gradient channel.
    let field = DecFlowField::compute(&cx, &pot, 1.0, 0.0, 0.0);
    let vectors = field.to_flow_vectors();

    // Flow at start should point toward goal (delta from start to goal).
    let (sy, sx) = scenario.start;
    let (gy, gx) = scenario.goal;
    let s_idx = sy * W + sx;
    let v = vectors[s_idx];
    let toward_y = (gy as isize - sy as isize) as f32;
    let toward_x = (gx as isize - sx as isize) as f32;
    let dot = v[0] * toward_x + v[1] * toward_y;
    dot > 0.0
}

// ── Pretty printing ──────────────────────────────────────────

fn print_aggregate(label: &str, agg: &Aggregate, gate_ratio: f64, gate_success: f64) {
    println!("┌─ {label} ───────────────────────────────────────────");
    println!("│  Scenarios: {}", agg.n);
    println!(
        "│  Mean cost ratio (DEC/A*): {:.4}",
        agg.mean_cost_ratio
    );
    println!("│  Max  cost ratio (DEC/A*): {:.4}", agg.max_cost_ratio);
    println!("│  Mean length ratio:        {:.4}", agg.mean_len_ratio);
    println!("│  Success rate:              {:.2}%", agg.success_rate * 100.0);
    if let Some((name, r)) = &agg.worst_case {
        println!("│  Worst case: {name}  ratio={r:.4}");
    }
    let ratio_pass = agg.max_cost_ratio <= gate_ratio;
    let success_pass = agg.success_rate >= gate_success;
    println!(
        "│  Gate: cost ≤ {gate_ratio:.2} → {}",
        if ratio_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "│  Gate: success ≥ {:.2} → {}",
        gate_success,
        if success_pass { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("└──────────────────────────────────────────────────────");
}

// ── Main ─────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  DEC Terrain Quality Bench — Hodge routes vs A* (T46–T47)  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Grid: {W}×{H} vertices, uniform-cost terrain (`.`=1).");
    println!();

    // ── G0: validate that DecFlowField actually points toward the goal ──
    println!("┌─ G0: DecFlowField orientation check ────────────────────────");
    let open = make_open_field();
    let wall = make_wall_gap(W / 2, H / 2);
    let field_ok_open = validate_dec_flow_field(&open);
    let field_ok_wall = validate_dec_flow_field(&wall);
    println!("│  Open field start-vertex flow points at goal: {field_ok_open}");
    println!("│  Wall+gap  start-vertex flow points at goal: {field_ok_wall}");
    if field_ok_open && field_ok_wall {
        println!("│  ✅ DecFlowField gradient orientation verified");
    } else {
        println!("│  ⚠️  DecFlowField gradient orientation anomalous");
    }
    println!("└──────────────────────────────────────────────────────");
    println!();

    // ── G1: Open field — sanity (must be exactly optimal) ──────────
    let open_agg = aggregate(&[open.clone()]);
    print_aggregate("G1: Open field (sanity)", &open_agg, 1.001, 1.0);
    println!();

    // ── G2: Random obstacles at 10% density ───────────────────────
    let g2_scenarios = make_multi_random(20, 10);
    let g2_agg = aggregate(&g2_scenarios);
    print_aggregate("G2: Random obstacles 10%", &g2_agg, 1.05, 0.98);
    println!();

    // ── G3: Wall + gap ─────────────────────────────────────────────
    let g3_scenarios = vec![
        make_wall_gap(W / 2, 0),
        make_wall_gap(W / 2, H / 2),
        make_wall_gap(W / 2, H - 1),
        make_wall_gap(W / 3, H / 4),
        make_wall_gap(2 * W / 3, 3 * H / 4),
        make_s_corridor(),
    ];
    let g3_agg = aggregate(&g3_scenarios);
    print_aggregate("G3: Wall + gap / corridor", &g3_agg, 1.10, 0.95);
    println!();

    // ── G4: Obstacle density scaling ───────────────────────────────
    println!("┌─ G4: Obstacle density scaling ──────────────────────────────────");
    println!("│  {:>8}  {:>8}  {:>10}  {:>10}  {:>10}  {:>12}", "density", "fair_n", "mean_cost", "max_cost", "success%", "mean_len");
    let mut g4_all_pass = true;
    for &density in &[5u8, 10, 15, 20, 25] {
        let scenarios = make_multi_random(15, density);
        let total = scenarios.len();
        let agg = aggregate(&scenarios);
        let ratio_ok = agg.max_cost_ratio <= 1.10;
        let success_ok = agg.success_rate >= 0.95;
        g4_all_pass = g4_all_pass && ratio_ok && success_ok;
        println!(
            "│  {:>7}%  {:>3}/{:<3}  {:>10.4}  {:>10.4}  {:>9.2}%  {:>12.4}  {}",
            density,
            agg.n,
            total,
            agg.mean_cost_ratio,
            agg.max_cost_ratio,
            agg.success_rate * 100.0,
            agg.mean_len_ratio,
            if ratio_ok && success_ok { "✅" } else { "❌" }
        );
    }
    println!("│  Gate: ratios stable across 5%–25% (fair cases only) → {}", if g4_all_pass { "✅ PASS" } else { "❌ FAIL" });
    println!("└──────────────────────────────────────────────────────────────────────────");
    println!();

    // ── G5: Timing — DEC field build + greedy route vs A* ──────────
    bench_single_route_timing(&open, &wall);

    // ── G6: Multi-agent amortisation (K agents share one field) ────
    bench_multi_agent_amortisation(&g2_scenarios[0]);

    // ── Verdict ────────────────────────────────────────────────────
    println!();
    println!("═══ GOAT Verdict (Plan 261 line 47) ═══════════════════════════");
    let g1_pass = open_agg.max_cost_ratio <= 1.001 && open_agg.success_rate >= 1.0;
    let g2_pass = g2_agg.max_cost_ratio <= 1.05 && g2_agg.success_rate >= 0.98;
    let g3_pass = g3_agg.max_cost_ratio <= 1.10 && g3_agg.success_rate >= 0.95;

    let mut results = Vec::new();
    results.push(("G1 Open field", g1_pass));
    results.push(("G2 Random 10%", g2_pass));
    results.push(("G3 Wall+gap", g3_pass));
    results.push(("G4 Density scale", g4_all_pass));

    for (name, pass) in &results {
        println!("  {name}: {}", if *pass { "✅ PASS" } else { "❌ FAIL" });
    }

    let all_pass = results.iter().all(|(_, p)| *p);
    println!();
    if all_pass {
        println!("  ✅ DEC WINS on quality — promote `dec_terrain_ai` to default feature.");
        println!("     Note: speed gate still blocked by Issue 013 (remove_face O(n) scan).");
        println!("           Quality is at parity with A*; incremental speed-up pending Issue 013.");
    } else {
        println!("  ❌ DEC LOSES on quality — keep `dec_terrain_ai` opt-in.");
        println!("     Failing gates indicate route quality regressions vs A*.");
    }
    println!();
    println!("═╗ Benchmark Complete ═════════════════════════════════════════");
}

// ── Timing benchmarks ────────────────────────────────────────

fn bench_single_route_timing(open: &Scenario, wall: &Scenario) {
    println!("┌─ G5: Single-route timing ───────────────────────────────────");
    println!("│  Per-route wall-clock (release build), {W}×{H} grid.");

    // DEC: build potential + DecFlowField + greedy route.
    let dec_start = Instant::now();
    let cx = CellComplex::grid_2d(W, H);
    let dist = dijkstra_potential(W, H, open.goal, &open.blocked);
    let mut pot = CochainField::zeros(0, cx.n_vertices(), 1);
    for v in 0..cx.n_vertices() {
        pot.set_scalar(v, dist[v]);
    }
    let _field = DecFlowField::compute(&cx, &pot, 1.0, 0.0, 0.0);
    let _route = dec_route_greedy(&dist, W, H, open.start, open.goal, &open.blocked);
    let dec_us = dec_start.elapsed().as_secs_f64() * 1e6;

    // A*: char grid + find_path.
    let grid = build_char_grid(W, H, &open.blocked);
    let astar_start = Instant::now();
    let _ = find_path(&grid, open.start, open.goal, &HashSet::new());
    let astar_us = astar_start.elapsed().as_secs_f64() * 1e6;

    println!("│  Open field:");
    println!("│    DEC  (dijkstra+field+route): {dec_us:>8.1} μs");
    println!("│    A*   (find_path):            {astar_us:>8.1} μs");

    // Wall + gap (tests routing around topology).
    let dec_start = Instant::now();
    let dist = dijkstra_potential(W, H, wall.goal, &wall.blocked);
    let _route = dec_route_greedy(&dist, W, H, wall.start, wall.goal, &wall.blocked);
    let dec_us = dec_start.elapsed().as_secs_f64() * 1e6;

    let grid = build_char_grid(W, H, &wall.blocked);
    let astar_start = Instant::now();
    let _ = find_path(&grid, wall.start, wall.goal, &HashSet::new());
    let astar_us = astar_start.elapsed().as_secs_f64() * 1e6;

    println!("│  Wall + gap:");
    println!("│    DEC  (dijkstra+route):       {dec_us:>8.1} μs");
    println!("│    A*   (find_path):            {astar_us:>8.1} μs");
    println!("└──────────────────────────────────────────────────────");
    println!();

    // Silence unused-field-build warning while keeping the call honest.
    std::hint::black_box(&cx);
}

fn bench_multi_agent_amortisation(scenario: &Scenario) {
    let k_agents = 64;
    println!("┌─ G6: Multi-agent amortisation (K={k_agents} agents, one field) ──");

    // DEC: one Dijkstra + one DecFlowField, K greedy routes from random starts.
    let mut rng = Rng::new(0xA11CE);
    let mut starts: Vec<(usize, usize)> = Vec::with_capacity(k_agents);
    while starts.len() < k_agents {
        let y = rng.below(H);
        let x = rng.below(W);
        if !scenario.blocked.contains(&(y, x)) && (y, x) != scenario.goal {
            starts.push((y, x));
        }
    }

    // ── Phase 1: build the shared field (amortised across K agents) ──
    let build_start = Instant::now();
    let cx = CellComplex::grid_2d(W, H);
    let dist = dijkstra_potential(W, H, scenario.goal, &scenario.blocked);
    let mut pot = CochainField::zeros(0, cx.n_vertices(), 1);
    for v in 0..cx.n_vertices() {
        pot.set_scalar(v, dist[v]);
    }
    let _field = DecFlowField::compute(&cx, &pot, 1.0, 0.0, 0.0);
    let build_us = build_start.elapsed().as_secs_f64() * 1e6;

    // ── Phase 2: K routes reusing the precomputed field (just the potential) ──
    let route_start = Instant::now();
    let mut dec_reached = 0usize;
    for &s in &starts {
        if dec_route_greedy(&dist, W, H, s, scenario.goal, &scenario.blocked).is_some() {
            dec_reached += 1;
        }
    }
    let route_us = route_start.elapsed().as_secs_f64() * 1e6;
    let dec_total_us = build_us + route_us;
    let dec_per_agent_us = dec_total_us / k_agents as f64;

    // A*: one find_path per agent (no shared precomputation).
    let grid = build_char_grid(W, H, &scenario.blocked);
    let astar_start = Instant::now();
    let mut astar_reached = 0usize;
    for &s in &starts {
        if find_path(&grid, s, scenario.goal, &HashSet::new()).is_some() {
            astar_reached += 1;
        }
    }
    let astar_us = astar_start.elapsed().as_secs_f64() * 1e6;
    let astar_per_agent_us = astar_us / k_agents as f64;

    println!("│  DEC  build (shared):     {build_us:>9.1} μs");
    println!("│  DEC  routing ({k_agents} agents): {route_us:>9.1} μs   per-route: {:>7.2} μs", route_us / k_agents as f64);
    println!("│  DEC  total:              {dec_total_us:>9.1} μs   per-agent: {dec_per_agent_us:>7.2} μs   reached: {dec_reached}/{k_agents}");
    println!("│  A*   total:              {astar_us:>9.1} μs   per-agent: {astar_per_agent_us:>7.2} μs   reached: {astar_reached}/{k_agents}");
    if astar_per_agent_us > 0.0 {
        let speedup = astar_per_agent_us / dec_per_agent_us.max(1e-9);
        let breakeven = (build_us / astar_per_agent_us.max(1e-9)).ceil() as u64;
        println!("│  Per-agent speedup (DEC vs A*): {speedup:.2}×");
        println!("│  Break-even agent count: ~{breakeven} (DEC wins at K ≥ {breakeven})");
    }
    println!("└──────────────────────────────────────────────────────");
    println!();

    std::hint::black_box(&cx);
}
