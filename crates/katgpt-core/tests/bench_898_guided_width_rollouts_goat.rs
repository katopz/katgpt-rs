//! Issue 895 T7 — guided width rollouts GOAT gate (Bench 898). Completes
//! Plan 095's pending width-vs-depth G1/G3 on the stochastic host 095 was
//! waiting for; no third lane is opened.
//!
//! Families (modelless, synthetic, fixed seeds). A graph 3-colouring CSP on
//! n = 10 vertices over 4 planted-3-colourable graphs, solved by a
//! deterministic projected-gradient refinement step on the soft assignment
//! `p ∈ [0,1]^{n×3}` (edge-conflict + one-hot + clue penalties; the "proposal"
//! u_t of GRAM, no weights, no training). Decode = argmax per vertex.
//!
//! - MULTI (N-Queens / graph-colouring class, GRAM's guidance-wins family):
//!   2 clues with ≥ 4 valid completions.
//! - SINGLE (Sudoku class, GRAM's zero-mean-wins family): 2–5 clues with
//!   EXACTLY one completion (verified by enumeration).
//!
//! Every instance starts from the uninformed uniform assignment p = 1/3.
//! Train and test clue sets are disjoint. Equal compute: every arm spends
//! N·K = 128 step calls per instance.
//!
//! Arms: D = deterministic 1×128 (the incumbent); Z = zero-mean width 8×16
//! (transversal ε, table-absent); I = isotropic ablation; G = guided width
//! (Z + the T5 success-SVD table fit on the TRAIN split, posterior updated on
//! the train outcomes); R = Z + T6 trap-kill-reallocate. Selection is ALWAYS
//! the decode-free `latent_value` (no reward, no decode, no oracle); the
//! oracle columns (any-valid, coverage) are reported beside it.
//!
//! PRE-STATED (written before the run, not re-tuned after):
//! - G1 (width vs depth, Issue 895 T7): PASS iff on BOTH families the
//!   selected solve rate of Z is not below D by more than 2 SE (paired) AND
//!   it is above D by more than 2 SE on at least one family.
//! - Demote condition (Research 590, never weakened): the table G vs Z on
//!   the branch-valid rate, paired over test instances — WIN > +2 SE,
//!   LOSS < −2 SE, else TIE. G loses-or-ties on BOTH ⇒ guided stays
//!   off-by-default forever (closed-negative).
//! - E9: |Z − D| selected-solve delta > 2 SE on at least one family, else
//!   the feature is inert on these fixtures.
//! - Plan 095 G1: width (best N at NK = 128) − N = 1 ≥ 10 pp on any family.
//!   Plan 095 G3: width ≥ depth on ≥ 2 of 3 domains — only 2 domains are
//!   measured here, so it needs BOTH.
//! - Mass arm: `belief_mass_divergence(ε) == 0.0` exactly over every draw,
//!   and under a linear cochain step every mass-arm branch keeps the
//!   deterministic branch's divergence trajectory.
//! - G2 (belief host, d = 8, N = 8, K = 16): best-of ≤ 50 µs per decision;
//!   t(K=32)/t(K=16) ∈ [1.5, 2.5] (O(K)); paired width/1×NK ratio reported.
//! - G3: N = 1 and σ = 0 ≡ the incumbent `evolve_belief` loop, `to_bits`,
//!   over 256 random states; table-absent ≡ empty table.
//! - G4: 0 allocations across steady-state decisions.
//!
//! Run:
//!   cargo test -p katgpt-core --release \
//!     --features guided_width_hodge,sense_composition \
//!     --test bench_898_guided_width_rollouts_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::{ab_median_ratio, best_of_us};
use katgpt_core::dec::{
    CellComplex, CochainField, belief_mass_divergence, codifferential_into, hodge_laplacian_into,
};
use katgpt_core::guided_width::belief_host::guided_evolve_belief;
use katgpt_core::guided_width::hodge_arm::{DivergenceProbe, MassConserving};
use katgpt_core::guided_width::{
    DirectionFitScratch, DirectionPosterior, DirectionTable, Guidance, GuidedWidthConfig,
    GuidedWidthScratch, Hooks, StagnationGate, Transversal, TrapReallocConfig,
    diverse_set_into, guided_width_rollouts,
};
use katgpt_core::sense::reconstruction::ReconstructionState;

// ── G4: counting allocator ────────────────────────────────────────────────

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

// ── deterministic RNG (fixture generation only) ───────────────────────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unif(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ── the colouring CSP ─────────────────────────────────────────────────────

const NV: usize = 10;
const NC: usize = 3;
const D: usize = NV * NC;
const N_GRAPHS: usize = 4;
const PER_SPLIT: usize = 32;
const NK: usize = 128;
const N_W: usize = 8;
const K_W: usize = NK / N_W;
const ETA: f32 = 0.2;
const LAMBDA: f32 = 1.0;
const MU: f32 = 1.0;

struct Graph {
    edges: Vec<(usize, usize)>,
    solutions: Vec<[u8; NV]>,
}

#[derive(Clone)]
struct Instance {
    graph: usize,
    clues: Vec<(usize, u8)>,
    completions: usize,
}

fn make_graph(seed: u64) -> Graph {
    let mut r = Rng::new(seed);
    let planted: Vec<u8> = (0..NV).map(|_| r.below(NC) as u8).collect();
    let mut edges = Vec::new();
    for u in 0..NV {
        for v in (u + 1)..NV {
            if planted[u] != planted[v] && r.unif() < 0.55 {
                edges.push((u, v));
            }
        }
    }
    let mut solutions = Vec::new();
    let mut col = [0u8; NV];
    let total = NC.pow(NV as u32);
    for code in 0..total {
        let mut c = code;
        for slot in col.iter_mut() {
            *slot = (c % NC) as u8;
            c /= NC;
        }
        if edges.iter().all(|&(u, v)| col[u] != col[v]) {
            solutions.push(col);
        }
    }
    Graph { edges, solutions }
}

fn completions(g: &Graph, clues: &[(usize, u8)]) -> usize {
    g.solutions
        .iter()
        .filter(|s| clues.iter().all(|&(v, c)| s[v] == c))
        .count()
}

/// Disjoint train/test instance pools for one family on one graph.
fn make_instances(g: &Graph, gi: usize, multi: bool, seed: u64) -> (Vec<Instance>, Vec<Instance>) {
    let mut r = Rng::new(seed);
    let mut pool: Vec<Instance> = Vec::new();
    let mut tries = 0;
    while pool.len() < 2 * PER_SPLIT && tries < 200_000 {
        tries += 1;
        let k = if multi { 2 } else { 2 + r.below(4) };
        let sol = g.solutions[r.below(g.solutions.len())];
        let mut vs: Vec<usize> = Vec::new();
        while vs.len() < k {
            let v = r.below(NV);
            if !vs.contains(&v) {
                vs.push(v);
            }
        }
        vs.sort_unstable();
        let clues: Vec<(usize, u8)> = vs.iter().map(|&v| (v, sol[v])).collect();
        let n = completions(g, &clues);
        let ok = if multi { n >= 4 } else { n == 1 };
        if ok && !pool.iter().any(|p| p.clues == clues) {
            pool.push(Instance {
                graph: gi,
                clues,
                completions: n,
            });
        }
    }
    assert_eq!(pool.len(), 2 * PER_SPLIT, "instance pool too small (graph {gi})");
    let test = pool.split_off(PER_SPLIT);
    (pool, test)
}

/// The deterministic refinement step (the proposal): one projected-gradient
/// step on the colouring energy. Zero-allocation.
fn color_step(g: &Graph, clues: &[(usize, u8)], h: &mut [f32]) {
    let mut grad = [0.0f32; D];
    for &(u, v) in &g.edges {
        for c in 0..NC {
            grad[u * NC + c] += h[v * NC + c];
            grad[v * NC + c] += h[u * NC + c];
        }
    }
    for v in 0..NV {
        let s: f32 = (0..NC).map(|c| h[v * NC + c]).sum::<f32>() - 1.0;
        for c in 0..NC {
            grad[v * NC + c] += 2.0 * LAMBDA * s;
        }
    }
    for &(v, cc) in clues {
        for c in 0..NC {
            let t = if c == cc as usize { 1.0 } else { 0.0 };
            grad[v * NC + c] += 2.0 * MU * (h[v * NC + c] - t);
        }
    }
    for i in 0..D {
        h[i] = (h[i] - ETA * grad[i]).clamp(0.0, 1.0);
    }
}

fn decode(h: &[f32]) -> [u8; NV] {
    let mut out = [0u8; NV];
    for v in 0..NV {
        let mut best = 0usize;
        for c in 1..NC {
            if h[v * NC + c] > h[v * NC + best] {
                best = c;
            }
        }
        out[v] = best as u8;
    }
    out
}

fn valid(g: &Graph, clues: &[(usize, u8)], col: &[u8; NV]) -> bool {
    g.edges.iter().all(|&(u, v)| col[u] != col[v]) && clues.iter().all(|&(v, c)| col[v] == c)
}

fn seed_of(tag: &str, family: usize, inst: usize) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"bench898");
    h.update(tag.as_bytes());
    h.update(&(family as u64).to_le_bytes());
    h.update(&(inst as u64).to_le_bytes());
    *h.finalize().as_bytes()
}

// ── arms ──────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Det,
    Zero,
    Iso,
    Guided,
    Realloc,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Det => "D 1x128 deterministic",
            Arm::Zero => "Z 8x16 zero-mean transversal",
            Arm::Iso => "I 8x16 zero-mean isotropic",
            Arm::Guided => "G 8x16 guided (T5 table)",
            Arm::Realloc => "R 8x16 Z + trap-realloc (T6)",
        }
    }
}

/// Per-instance outcome of one arm.
#[derive(Clone, Copy, Default)]
struct Outcome {
    sel: f32,
    any: f32,
    cov: f32,
    bv: f32,
}

struct Ctx {
    scratch: GuidedWidthScratch,
}

fn base_cfg(n: usize, k: usize, seed: [u8; 32]) -> GuidedWidthConfig {
    GuidedWidthConfig {
        n_branches: n,
        k_steps: k,
        gate: StagnationGate::DEFAULT,
        seed,
        ..GuidedWidthConfig::DEFAULT
    }
}

#[allow(clippy::too_many_arguments)]
fn run_arm(
    ctx: &mut Ctx,
    graphs: &[Graph],
    inst: &Instance,
    arm: Arm,
    n: usize,
    k: usize,
    seed: [u8; 32],
    guidance: Option<(&DirectionTable, &DirectionPosterior)>,
) -> Outcome {
    let g = &graphs[inst.graph];
    let h0 = [1.0f32 / NC as f32; D];
    let mut out = [0.0f32; D];
    let mut step = |h: &mut [f32]| color_step(g, &inst.clues, h);
    let mut cfg = base_cfg(n, k, seed);
    if arm == Arm::Det {
        cfg.n_branches = 1;
    }
    if arm == Arm::Realloc {
        cfg.trap = Some(TrapReallocConfig::DEFAULT);
    }
    let hooks = match (arm, guidance) {
        (Arm::Guided, Some((t, p))) => Hooks {
            guidance: Some(Guidance {
                table: t,
                posterior: p,
                epsilon: 0.05,
                bias_tau: 0.5,
            }),
            ..Hooks::default()
        },
        _ => Hooks::default(),
    };
    let mut arm_p = if arm == Arm::Iso {
        Transversal::ISOTROPIC
    } else {
        Transversal::TRANSVERSAL
    };
    let rep = guided_width_rollouts(
        &h0,
        &cfg,
        hooks,
        &mut arm_p,
        &mut step,
        &mut ctx.scratch,
        &mut out,
    );
    let sel = valid(g, &inst.clues, &decode(&out)) as u8 as f32;
    if rep.incumbent {
        return Outcome {
            sel,
            any: sel,
            cov: sel,
            bv: sel,
        };
    }
    let nb = ctx.scratch.last_branches();
    let mut seen: Vec<[u8; NV]> = Vec::new();
    let mut n_valid = 0usize;
    for b in 0..nb {
        let col = decode(ctx.scratch.branch(b));
        if valid(g, &inst.clues, &col) {
            n_valid += 1;
            if !seen.contains(&col) {
                seen.push(col);
            }
        }
    }
    Outcome {
        sel,
        any: (n_valid > 0) as u8 as f32,
        cov: seen.len() as f32,
        bv: n_valid as f32 / nb as f32,
    }
}

fn mean(xs: &[f32]) -> f32 {
    xs.iter().sum::<f32>() / xs.len() as f32
}

/// Paired mean difference and its standard error.
fn paired(a: &[f32], b: &[f32]) -> (f32, f32) {
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let m = mean(&d);
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / (d.len() as f32 - 1.0).max(1.0);
    (m, (var / d.len() as f32).sqrt())
}

fn verdict(diff: f32, se: f32) -> &'static str {
    if diff > 2.0 * se && diff > 0.0 {
        "WIN"
    } else if diff < -2.0 * se && diff < 0.0 {
        "LOSS"
    } else {
        "TIE"
    }
}

fn gate(name: &str, pass: bool, detail: String, fails: &mut Vec<String>) {
    println!("{} {name}: {detail}", if pass { "PASS" } else { "FAIL" });
    if !pass {
        fails.push(name.to_owned());
    }
}

fn box_state() {
    let run = |cmd: &str, args: &[&str]| {
        std::process::Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
            .unwrap_or_else(|| "n/a".into())
    };
    println!("box: loadavg {}", run("sysctl", &["-n", "vm.loadavg"]));
    println!("box: swap {}", run("sysctl", &["-n", "vm.swapusage"]));
    let vm = run("vm_stat", &[]);
    let free = vm
        .lines()
        .filter(|l| l.starts_with("Pages free") || l.starts_with("Pages speculative"))
        .filter_map(|l| {
            l.split(':')
                .nth(1)
                .map(|v| v.trim().trim_end_matches('.').to_owned())
        })
        .filter_map(|v| v.parse::<u64>().ok())
        .sum::<u64>();
    println!(
        "box: free+speculative {:.2} GB (16 KiB pages)",
        free as f64 * 16384.0 / 1e9
    );
    println!(
        "box: power {}",
        run("pmset", &["-g", "batt"]).replace('\n', " | ")
    );
    let pm = run("pmset", &["-g"]);
    let mode = pm
        .lines()
        .find(|l| l.trim_start().starts_with("powermode"))
        .unwrap_or("powermode n/a")
        .trim()
        .to_owned();
    println!("box: {mode}");
}

// ── G1: both families ─────────────────────────────────────────────────────

struct FamilyResult {
    name: &'static str,
    per_arm: Vec<(Arm, Vec<Outcome>)>,
    sweep: Vec<(usize, f32, f32, f32)>,
}

fn fit_table(
    ctx: &mut Ctx,
    graphs: &[Graph],
    train: &[Instance],
    fam: usize,
) -> Option<(Vec<DirectionTable>, Vec<DirectionPosterior>)> {
    let mut tables = Vec::new();
    let mut posts = Vec::new();
    let h0 = [1.0f32 / NC as f32; D];
    for (gi, g) in graphs.iter().enumerate().take(N_GRAPHS) {
        // Pass 1: zero-mean width on the TRAIN split, log successful Δh.
        let mut deltas: Vec<f32> = Vec::new();
        let mut w: Vec<f32> = Vec::new();
        for (ii, inst) in train.iter().enumerate().filter(|(_, x)| x.graph == gi) {
            let mut out = [0.0f32; D];
            let mut step = |h: &mut [f32]| color_step(g, &inst.clues, h);
            let cfg = base_cfg(N_W, K_W, seed_of("train", fam, ii));
            guided_width_rollouts(
                &h0,
                &cfg,
                Hooks::default(),
                &mut Transversal::default(),
                &mut step,
                &mut ctx.scratch,
                &mut out,
            );
            for b in 0..ctx.scratch.last_branches() {
                let hb = ctx.scratch.branch(b);
                if valid(g, &inst.clues, &decode(hb)) {
                    deltas.extend(hb.iter().zip(&h0).map(|(x, y)| x - y));
                    w.push(1.0);
                }
            }
        }
        let mut fs = DirectionFitScratch::with_capacity(w.len(), D);
        let table = DirectionTable::fit(&deltas, &w, D, 6, &mut fs)?;
        // Pass 2: Beta posterior over the train outcomes of the guided arm.
        let mut post = DirectionPosterior::new(table.len());
        for (ii, inst) in train.iter().enumerate().filter(|(_, x)| x.graph == gi) {
            let mut out = [0.0f32; D];
            let mut step = |h: &mut [f32]| color_step(g, &inst.clues, h);
            let cfg = base_cfg(N_W, K_W, seed_of("train2", fam, ii));
            let snapshot = post;
            guided_width_rollouts(
                &h0,
                &cfg,
                Hooks {
                    guidance: Some(Guidance {
                        table: &table,
                        posterior: &snapshot,
                        epsilon: 0.05,
                        bias_tau: 0.5,
                    }),
                    ..Hooks::default()
                },
                &mut Transversal::default(),
                &mut step,
                &mut ctx.scratch,
                &mut out,
            );
            for b in 1..ctx.scratch.last_branches() {
                if let Some((j, _)) = ctx.scratch.branch_direction(b) {
                    let ok = valid(g, &inst.clues, &decode(ctx.scratch.branch(b)));
                    post.record(j as usize, ok);
                }
            }
        }
        println!(
            "  table graph {gi}: {} rows → r={} bias {:?} commit {}",
            w.len(),
            table.len(),
            (0..table.len())
                .map(|j| (table.bias(j) * 100.0).round() / 100.0)
                .collect::<Vec<_>>(),
            &blake3::Hash::from(table.commitment()).to_hex()[..16]
        );
        tables.push(table);
        posts.push(post);
    }
    Some((tables, posts))
}

fn run_family(
    ctx: &mut Ctx,
    graphs: &[Graph],
    name: &'static str,
    fam: usize,
    multi: bool,
) -> FamilyResult {
    let mut train = Vec::new();
    let mut test = Vec::new();
    for (gi, g) in graphs.iter().enumerate() {
        let (tr, te) = make_instances(g, gi, multi, 1000 * (fam as u64 + 1) + gi as u64);
        train.extend(tr);
        test.extend(te);
    }
    let mean_comp = test.iter().map(|i| i.completions as f32).sum::<f32>() / test.len() as f32;
    println!(
        "\n== family {name}: {} test / {} train instances, mean completions {mean_comp:.1}",
        test.len(),
        train.len()
    );
    let (tables, posts) = fit_table(ctx, graphs, &train, fam).expect("table fit");
    let arms = [Arm::Det, Arm::Zero, Arm::Iso, Arm::Guided, Arm::Realloc];
    let mut per_arm = Vec::new();
    for &arm in &arms {
        let mut outs = Vec::with_capacity(test.len());
        for (ii, inst) in test.iter().enumerate() {
            let (n, k) = if arm == Arm::Det { (1, NK) } else { (N_W, K_W) };
            let gd = (arm == Arm::Guided).then(|| (&tables[inst.graph], &posts[inst.graph]));
            outs.push(run_arm(ctx, graphs, inst, arm, n, k, seed_of("test", fam, ii), gd));
        }
        let col = |f: fn(&Outcome) -> f32| mean(&outs.iter().map(f).collect::<Vec<_>>());
        println!(
            "  {:<32} selected {:.3}  any-valid {:.3}  coverage {:.2}  branch-valid {:.3}",
            arm.name(),
            col(|o| o.sel),
            col(|o| o.any),
            col(|o| o.cov),
            col(|o| o.bv)
        );
        per_arm.push((arm, outs));
    }
    // Plan 095 width sweep at fixed NK = 128 (arm Z).
    let mut sweep = Vec::new();
    for n in [1usize, 2, 4, 8, 16, 32] {
        let k = NK / n;
        let mut sel = Vec::new();
        let mut any = Vec::new();
        let mut cov = Vec::new();
        for (ii, inst) in test.iter().enumerate() {
            let o = run_arm(ctx, graphs, inst, Arm::Zero, n, k, seed_of("sweep", fam, ii), None);
            sel.push(o.sel);
            any.push(o.any);
            cov.push(o.cov);
        }
        println!(
            "  sweep N={n:<2} K={k:<3}  selected {:.3}  any-valid {:.3}  coverage {:.2}",
            mean(&sel),
            mean(&any),
            mean(&cov)
        );
        sweep.push((n, mean(&sel), mean(&any), mean(&cov)));
    }
    FamilyResult {
        name,
        per_arm,
        sweep,
    }
}

fn arm_col(fr: &FamilyResult, arm: Arm, f: fn(&Outcome) -> f32) -> Vec<f32> {
    fr.per_arm
        .iter()
        .find(|(a, _)| *a == arm)
        .map(|(_, o)| o.iter().map(f).collect())
        .unwrap()
}

// ── mass-conservation arm (T1 b) ──────────────────────────────────────────

fn mass_arm(fails: &mut Vec<String>) {
    println!("\n== mass-conservation arm (T1 b)");
    let cx = CellComplex::grid_2d(12, 12);
    let mut arm = MassConserving::new(&cx, None).expect("admission");
    let ne = cx.n_edges();
    let mut eps = vec![0.0f32; ne];
    let mut nonzero = 0usize;
    let mut draws = 0usize;
    for seed in 0..500u64 {
        for sigma in [1e-4f32, 0.01, 0.25, 0.7, 2.0] {
            arm.draw_into(seed, sigma, &mut eps);
            let f = CochainField::from_vec(1, 1, eps.clone());
            if belief_mass_divergence(&cx, &f) != 0.0 {
                nonzero += 1;
            }
            draws += 1;
        }
    }
    gate(
        "MASS belief_mass_divergence(ε) ≡ 0",
        nonzero == 0,
        format!("{nonzero} non-zero of {draws} draws (grid 12×12, {ne} edges)"),
        fails,
    );
    // Cochain host: h ← h − η Δ₁ h (linear heat step on an edge flow). The
    // divergence evolves autonomously (δ₁Δ₁ = Δ₀δ₁), so a divergence-free ε
    // must leave every branch on branch 0's divergence trajectory.
    let run = |mass: bool| -> f32 {
        let mut inp = CochainField::zeros(1, ne, 1);
        let mut outp = CochainField::zeros(1, ne, 1);
        let mut su = CochainField::zeros(2, cx.n_faces(), 1);
        let mut sl = CochainField::zeros(0, cx.n_vertices(), 1);
        let mut sr = CochainField::zeros(1, ne, 1);
        let mut step = |h: &mut [f32]| {
            inp.data.copy_from_slice(h);
            hodge_laplacian_into(&cx, &inp, &mut outp, &mut su, &mut sl, &mut sr);
            for (x, &l) in h.iter_mut().zip(&outp.data) {
                *x -= 0.05 * l;
            }
        };
        let mut r = Rng::new(77);
        let h0: Vec<f32> = (0..ne).map(|_| r.unif() - 0.5).collect();
        let cfg = GuidedWidthConfig {
            n_branches: 6,
            k_steps: 12,
            gate: StagnationGate {
                sigma_max: 0.2,
                stuck_tol: 1e9, // always "stuck": full σ every step
                ..StagnationGate::DEFAULT
            },
            seed: [5u8; 32],
            ..GuidedWidthConfig::DEFAULT
        };
        let mut scratch = GuidedWidthScratch::with_capacity(6, ne);
        let mut out = vec![0.0f32; ne];
        if mass {
            let mut p = MassConserving::new(&cx, None).unwrap();
            let mut probe_s = DivergenceProbe::new(&cx);
            let mut probe = |d: &[f32]| probe_s.circulation(d);
            let hooks = Hooks {
                probe: Some(&mut probe),
                ..Hooks::default()
            };
            guided_width_rollouts(&h0, &cfg, hooks, &mut p, &mut step, &mut scratch, &mut out);
        } else {
            guided_width_rollouts(
                &h0,
                &cfg,
                Hooks::default(),
                &mut Transversal::default(),
                &mut step,
                &mut scratch,
                &mut out,
            );
        }
        let mut div0 = CochainField::zeros(0, cx.n_vertices(), 1);
        let mut divb = CochainField::zeros(0, cx.n_vertices(), 1);
        let f0 = CochainField::from_vec(1, 1, scratch.branch(0).to_vec());
        codifferential_into(&cx, &f0, &mut div0);
        let mut worst = 0.0f32;
        for b in 1..scratch.last_branches() {
            let fb = CochainField::from_vec(1, 1, scratch.branch(b).to_vec());
            codifferential_into(&cx, &fb, &mut divb);
            for (a, c) in div0.data.iter().zip(&divb.data) {
                worst = worst.max((a - c).abs());
            }
        }
        worst
    };
    let with_mass = run(true);
    let with_transversal = run(false);
    gate(
        "MASS cochain host keeps branch-0 divergence",
        with_mass < 1e-4 && with_transversal > 100.0 * with_mass.max(1e-7),
        format!(
            "max |δ₁h_b − δ₁h_0|: mass arm {with_mass:.2e} vs transversal arm {with_transversal:.2e}"
        ),
        fails,
    );
}

// ── belief host helpers (G2/G3/G4) ────────────────────────────────────────

fn belief_state(r: &mut Rng) -> ReconstructionState {
    let belief: [f32; 8] = std::array::from_fn(|_| r.unif() * 2.0 - 1.0);
    let mut s = ReconstructionState::new(belief);
    let acts: [f32; 6] = std::array::from_fn(|_| r.unif());
    let sel: [bool; 6] = std::array::from_fn(|_| r.unif() < 0.8);
    s.accumulate(&sel, &acts);
    s
}

fn belief_cfg(n: usize, k: usize, sigma: f32, seed: u64) -> GuidedWidthConfig {
    let mut sd = [0u8; 32];
    sd[..8].copy_from_slice(&seed.to_le_bytes());
    GuidedWidthConfig {
        n_branches: n,
        k_steps: k,
        gate: StagnationGate {
            sigma_max: sigma,
            ..StagnationGate::DEFAULT
        },
        seed: sd,
        ..GuidedWidthConfig::DEFAULT
    }
}

fn g3(fails: &mut Vec<String>) {
    println!("\n== G3 kill switch (belief host, real evolve_belief)");
    let mut r = Rng::new(898);
    let mut mismatches = 0usize;
    let mut scratch = GuidedWidthScratch::with_capacity(8, 8);
    for case in 0..256usize {
        let base = belief_state(&mut r);
        let k = 1 + case % 24;
        let mut reference = ReconstructionState::new(*base.belief());
        reference.accumulate(&[true; 6], &base.evidence().kind_activations);
        for _ in 0..k {
            reference.evolve_belief();
        }
        for (n, sigma) in [(1usize, 0.25f32), (8, 0.0)] {
            let mut s = ReconstructionState::new(*base.belief());
            s.accumulate(&[true; 6], &base.evidence().kind_activations);
            let rep = guided_evolve_belief(
                &mut s,
                &belief_cfg(n, k, sigma, case as u64),
                Hooks::default(),
                &mut Transversal::default(),
                &mut scratch,
            );
            if !rep.incumbent
                || s.belief().map(f32::to_bits) != reference.belief().map(f32::to_bits)
            {
                mismatches += 1;
            }
        }
    }
    gate(
        "G3 N=1 / σ=0 ≡ incumbent evolve_belief",
        mismatches == 0,
        format!("{mismatches} bit mismatches over 256 states × 2 kill-switch configs"),
        fails,
    );
    // Table-absent ≡ empty table, and determinism.
    let empty = DirectionTable::from_parts(8, vec![], vec![]).unwrap();
    let post = DirectionPosterior::new(0);
    let mut diff = 0usize;
    for case in 0..64u64 {
        let base = belief_state(&mut r);
        let cfg = belief_cfg(8, 16, 0.25, case);
        let mut a = ReconstructionState::new(*base.belief());
        a.accumulate(&[true; 6], &base.evidence().kind_activations);
        let mut b = ReconstructionState::new(*base.belief());
        b.accumulate(&[true; 6], &base.evidence().kind_activations);
        let ra = guided_evolve_belief(&mut a, &cfg, Hooks::default(), &mut Transversal::default(), &mut scratch);
        let rb = guided_evolve_belief(
            &mut b,
            &cfg,
            Hooks {
                guidance: Some(Guidance {
                    table: &empty,
                    posterior: &post,
                    epsilon: 0.05,
                    bias_tau: 0.5,
                }),
                ..Hooks::default()
            },
            &mut Transversal::default(),
            &mut scratch,
        );
        if ra != rb || a.belief().map(f32::to_bits) != b.belief().map(f32::to_bits) {
            diff += 1;
        }
    }
    gate(
        "G3 table-absent ≡ isotropic+transversal (empty table)",
        diff == 0,
        format!("{diff} differences over 64 decisions"),
        fails,
    );
}

fn g2(fails: &mut Vec<String>) {
    println!("\n== G2 latency (belief host d=8, release)");
    box_state();
    let mut r = Rng::new(4242);
    let states: Vec<[f32; 8]> = (0..64).map(|_| *belief_state(&mut r).belief()).collect();
    let evid = belief_state(&mut r);
    let acts = evid.evidence().kind_activations;
    let mut scratch = GuidedWidthScratch::with_capacity(8, 8);
    let mut per_k = Vec::new();
    for k in [16usize, 32] {
        let mut i = 0usize;
        let mut st = ReconstructionState::new(states[0]);
        st.accumulate(&[true; 6], &acts);
        let us = best_of_us(50, 400, || {
            *st.belief_mut() = black_box(states[i % states.len()]);
            let cfg = belief_cfg(8, k, 0.25, i as u64);
            i += 1;
            let t = Instant::now();
            let rep = guided_evolve_belief(
                &mut st,
                black_box(&cfg),
                Hooks::default(),
                &mut Transversal::default(),
                &mut scratch,
            );
            let e = t.elapsed();
            black_box((rep, *st.belief()));
            e
        });
        println!("  best-of decision N=8 K={k}: {us:.2} µs");
        per_k.push(us);
    }
    gate(
        "G2 per-decision latency N=8 K=16",
        per_k[0] <= 50.0,
        format!("{:.2} µs (bar ≤ 50 µs)", per_k[0]),
        fails,
    );
    let ratio = per_k[1] / per_k[0];
    gate(
        "G2 O(K) scaling t(K=32)/t(K=16)",
        (1.5..=2.5).contains(&ratio),
        format!("{ratio:.2} (bar [1.5, 2.5])"),
        fails,
    );
    // Paired equal-compute ratio: width 8×16 vs depth 1×128 (report).
    let mut sa = ReconstructionState::new(states[0]);
    sa.accumulate(&[true; 6], &acts);
    let mut sb = ReconstructionState::new(states[0]);
    sb.accumulate(&[true; 6], &acts);
    let mut scratch_b = GuidedWidthScratch::with_capacity(8, 8);
    let (mut ka, mut kb) = (0.0f32, 0.0f32);
    let ab = ab_median_ratio(
        15,
        200,
        20,
        |i| {
            *sa.belief_mut() = black_box(states[i % states.len()]);
            let cfg = belief_cfg(8, 16, 0.25, i as u64);
            guided_evolve_belief(&mut sa, black_box(&cfg), Hooks::default(), &mut Transversal::default(), &mut scratch);
            ka += black_box(sa.belief()[i % 8]);
        },
        |i| {
            *sb.belief_mut() = black_box(states[i % states.len()]);
            let cfg = belief_cfg(1, 128, 0.25, i as u64);
            guided_evolve_belief(&mut sb, black_box(&cfg), Hooks::default(), &mut Transversal::default(), &mut scratch_b);
            kb += black_box(sb.belief()[i % 8]);
        },
    );
    ab.report("G2 width 8x16 / depth 1x128 (equal compute; report)");
    black_box(ka + kb);
    // Structural O(K)-parallel property: branches share nothing before
    // selection — t(N=16)/t(N=8) at fixed K measures the serial N·K work.
    let mut i = 0usize;
    let mut st = ReconstructionState::new(states[0]);
    st.accumulate(&[true; 6], &acts);
    let mut scratch16 = GuidedWidthScratch::with_capacity(16, 8);
    let us16 = best_of_us(50, 400, || {
        *st.belief_mut() = black_box(states[i % states.len()]);
        let cfg = belief_cfg(16, 16, 0.25, i as u64);
        i += 1;
        let t = Instant::now();
        let rep = guided_evolve_belief(&mut st, black_box(&cfg), Hooks::default(), &mut Transversal::default(), &mut scratch16);
        let e = t.elapsed();
        black_box((rep, *st.belief()));
        e
    });
    println!(
        "  best-of decision N=16 K=16: {us16:.2} µs → t(N=16)/t(N=8) = {:.2} (serial N·K + O(N²d) scorer; report)",
        us16 / per_k[0]
    );
}

fn g4(fails: &mut Vec<String>) {
    println!("\n== G4 allocations (steady state)");
    let mut r = Rng::new(31337);
    let mut st = belief_state(&mut r);
    let mut scratch = GuidedWidthScratch::with_capacity(8, 8);
    let dirs: Vec<f32> = (0..16).map(|i| ((i * 7 % 5) as f32) - 2.0).collect();
    let table = DirectionTable::from_parts(8, dirs, vec![0.9, 0.1]).unwrap();
    let mut post = DirectionPosterior::new(2);
    let cx = CellComplex::grid_2d(8, 8);
    let ne = cx.n_edges();
    let mut mass = MassConserving::new(&cx, None).unwrap();
    let mut probe_s = DivergenceProbe::new(&cx);
    let mut flow_scratch = GuidedWidthScratch::with_capacity(6, ne);
    let flow0: Vec<f32> = (0..ne).map(|_| r.unif() - 0.5).collect();
    let mut flow_out = vec![0.0f32; ne];
    let mut set = [0usize; 4];
    let mut cycle = |i: usize, st: &mut ReconstructionState, post: &mut DirectionPosterior| {
        let mut cfg = belief_cfg(8, 16, 0.25, i as u64);
        cfg.trap = Some(TrapReallocConfig::DEFAULT);
        let snap = *post;
        let rep = guided_evolve_belief(
            st,
            &cfg,
            Hooks {
                guidance: Some(Guidance {
                    table: &table,
                    posterior: &snap,
                    epsilon: 0.05,
                    bias_tau: 0.5,
                }),
                frozen_direction: Some(table.direction(0)),
                ..Hooks::default()
            },
            &mut Transversal::default(),
            &mut scratch,
        );
        if let Some((j, _)) = rep.direction {
            post.record(j as usize, i.is_multiple_of(3));
        }
        black_box(diverse_set_into(&mut scratch, 4, &mut set));
        let mut step = |h: &mut [f32]| {
            for x in h.iter_mut() {
                *x *= 0.97;
            }
        };
        let mut probe = |d: &[f32]| probe_s.circulation(d);
        let fcfg = GuidedWidthConfig {
            n_branches: 6,
            k_steps: 8,
            trap: Some(TrapReallocConfig::DEFAULT),
            ..belief_cfg(6, 8, 0.2, i as u64)
        };
        guided_width_rollouts(
            &flow0,
            &fcfg,
            Hooks {
                probe: Some(&mut probe),
                ..Hooks::default()
            },
            &mut mass,
            &mut step,
            &mut flow_scratch,
            &mut flow_out,
        );
        black_box(&flow_out);
    };
    for i in 0..50 {
        cycle(i, &mut st, &mut post);
    }
    let before = allocs();
    for i in 50..1050 {
        cycle(i, &mut st, &mut post);
    }
    let n = allocs() - before;
    gate(
        "G4 zero allocations",
        n == 0,
        format!("{n} allocations over 1000 decisions (belief host + table + trap + returned set + mass arm + probe)"),
        fails,
    );
}

fn main() {
    let mut fails: Vec<String> = Vec::new();
    println!("Bench 898 — guided width rollouts GOAT (Issue 895 T7; completes Plan 095 G1/G3)");
    let graphs: Vec<Graph> = (0..N_GRAPHS as u64).map(|s| make_graph(0x898 + s)).collect();
    for (i, g) in graphs.iter().enumerate() {
        println!("graph {i}: {} edges, {} proper 3-colourings", g.edges.len(), g.solutions.len());
    }
    let mut ctx = Ctx {
        scratch: GuidedWidthScratch::with_capacity(32, D),
    };
    let t0 = Instant::now();
    let multi = run_family(&mut ctx, &graphs, "MULTI (N-Queens class)", 0, true);
    let single = run_family(&mut ctx, &graphs, "SINGLE (Sudoku class)", 1, false);
    println!("\n(G1 fixtures ran in {:.1}s)", t0.elapsed().as_secs_f32());

    println!("\n== G1 verdicts (paired over test instances, ±2 SE)");
    let mut g1_ok_both = true;
    let mut g1_win_any = false;
    let mut e9_any = false;
    let mut table_results = Vec::new();
    let mut p095_g1 = false;
    let mut p095_width_ge_depth = 0;
    for fr in [&multi, &single] {
        let d = arm_col(fr, Arm::Det, |o| o.sel);
        let z = arm_col(fr, Arm::Zero, |o| o.sel);
        let (dz, se) = paired(&z, &d);
        let v = verdict(dz, se);
        println!("  {}: Z − D selected = {dz:+.3} ± {se:.3} → {v}", fr.name);
        if v == "LOSS" {
            g1_ok_both = false;
        }
        if v == "WIN" {
            g1_win_any = true;
        }
        if v != "TIE" {
            e9_any = true;
        }
        let dcov = mean(&arm_col(fr, Arm::Zero, |o| o.cov)) - mean(&arm_col(fr, Arm::Det, |o| o.cov));
        println!("    E9 signature: coverage delta Z − D = {dcov:+.2} distinct valid solutions");
        let gb = arm_col(fr, Arm::Guided, |o| o.bv);
        let zb = arm_col(fr, Arm::Zero, |o| o.bv);
        let (dg, seg) = paired(&gb, &zb);
        let tv = verdict(dg, seg);
        let (dgs, segs) = paired(&arm_col(fr, Arm::Guided, |o| o.sel), &z);
        let (dgc, segc) = paired(&arm_col(fr, Arm::Guided, |o| o.cov), &arm_col(fr, Arm::Zero, |o| o.cov));
        println!(
            "    table G − Z: branch-valid {dg:+.3} ± {seg:.3} → {tv} (decision metric); selected {dgs:+.3} ± {segs:.3}; coverage {dgc:+.2} ± {segc:.2} (reports)"
        );
        table_results.push(tv);
        let (dr, ser) = paired(&arm_col(fr, Arm::Realloc, |o| o.sel), &z);
        let (dri, seri) = paired(&arm_col(fr, Arm::Iso, |o| o.bv), &zb);
        println!(
            "    T6 R − Z selected {dr:+.3} ± {ser:.3}; isotropic I − Z branch-valid {dri:+.3} ± {seri:.3} (reports)"
        );
        let n1 = fr.sweep[0].1;
        let best_w = fr.sweep[1..].iter().map(|s| s.1).fold(f32::MIN, f32::max);
        let best_any = fr.sweep[1..].iter().map(|s| s.2).fold(f32::MIN, f32::max);
        println!(
            "    Plan 095: best width selected {best_w:.3} vs N=1 {n1:.3} (Δ {:+.1} pp); oracle any-valid best {best_any:.3}",
            100.0 * (best_w - n1)
        );
        if best_w - n1 >= 0.10 {
            p095_g1 = true;
        }
        if best_w >= n1 {
            p095_width_ge_depth += 1;
        }
    }
    gate(
        "G1 width N×K vs depth 1×NK on BOTH families",
        g1_ok_both && g1_win_any,
        format!("no family LOSS: {g1_ok_both}; ≥1 family WIN: {g1_win_any}"),
        &mut fails,
    );
    gate("E9 measured delta over the deterministic arm", e9_any, format!("non-TIE on ≥1 family: {e9_any}"), &mut fails);
    let table_wins_any = table_results.contains(&"WIN");
    println!(
        "DEMOTE-CONDITION (pre-stated): table G vs Z = MULTI {} / SINGLE {} → {}",
        table_results[0],
        table_results[1],
        if table_wins_any {
            "NOT triggered (table wins on ≥1 family)"
        } else {
            "TRIGGERED — guided stays off-by-default forever (closed-negative)"
        }
    );
    println!(
        "Plan 095 G1 (width ≥ +10 pp on any domain): {}; Plan 095 G3 (width ≥ depth on ≥2 of 3 domains; 2 measured): {}/2 → {}",
        if p095_g1 { "PASS" } else { "FAIL" },
        p095_width_ge_depth,
        if p095_width_ge_depth >= 2 { "PASS" } else { "FAIL" }
    );

    mass_arm(&mut fails);
    g3(&mut fails);
    g4(&mut fails);
    g2(&mut fails);

    println!();
    if fails.is_empty() {
        println!("Bench 898: ALL GATES PASS");
    } else {
        println!("Bench 898: {} gate(s) FAILED: {fails:?}", fails.len());
        std::process::exit(1);
    }
}
