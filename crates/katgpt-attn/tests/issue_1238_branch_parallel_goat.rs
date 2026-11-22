//! Issue 1238 — fused multi-branch FUNCATTN GOAT cell (select-vs-fuse-vs-wide).
//!
//! Verdict: `.benchmarks/992_issue1238_branch_parallel_funcattn_cell.md`. Pre-registration was
//! the Issue-1238 PINNED block (committed `f8aa01e67` before this harness ran). Task and
//! FD-SGD recipe reuse the Plan-286 T3.2 harness
//! (`tests/funcattn_g2_funcattn_vs_parallax_vs_sdpa.rs`): sinusoidal regression
//! with a cross-feature tanh term, n=64, d_model=8, k=8, α=0.01, τ=0.1.
//!
//! Arms (params): A1 d_b=8 K=1 (384) · A2 d_b=5 K=2 Σ/√2 (390) · A3 = A2 +
//! learned concat-projection (518) · A4 = A2 + shared slice basis (390) ·
//! A5 d_b=12 K=1 (720) · A6 two FROZEN independently-trained d_b=5 snapshots,
//! mean-fused (390) · A6b = A6 fused Σ/√2 (negative control).
//!
//! The test ASSERTS only G1 (bit-identical re-run); every other gate is a
//! research verdict, printed as `VERDICT …` lines and recorded in
//! `.benchmarks/` (the Plan-286 G2 precedent).
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/i1238 cargo test -p katgpt-attn --release \
//!   --features funcattn_branch_parallel,funcattn_freeze_thaw \
//!   --test issue_1238_branch_parallel_goat -- --nocapture
//! ```

#![cfg(all(feature = "funcattn_branch_parallel", feature = "funcattn_freeze_thaw"))]

use std::hint::black_box;
use std::time::Instant;

use katgpt_attn::funcattn_compose::branch_parallel::{
    BranchFusion, BranchParallelConfig, BranchParallelScratch, BranchWeights,
    branch_parallel_forward,
};
use katgpt_attn::funcattn_compose::freeze_thaw::{FuncAttnSnapshotStore, FuncAttnWeightsSnapshot};
use katgpt_core::funcattn::FuncAttnBasis;

const N: usize = 64;
const M: usize = 8;
const KB: usize = 8;
const ALPHA: f32 = 0.01;
const TEMP: f32 = 0.1;
const FD_EPS: f32 = 1e-2;
const STEPS_SHORT: usize = 150;
const STEPS_LONG: usize = 500;
const GATE_SEEDS: [u64; 5] = [
    0x1238_0001,
    0x1238_0002,
    0x1238_0003,
    0x1238_0004,
    0x1238_0005,
];
const PILOT_SEED: u64 = 0x1238_F11E;
const LR_LADDER: [f32; 3] = [1.0, 0.5, 0.25];

// ── PRNG (xorshift64*, mirrors the Plan-286 harness) ─────────────────────

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / ((1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn eye(d: usize) -> Vec<f32> {
    let mut w = vec![0.0; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    w
}

/// Row-normalised random init; the first `min(rows, cols)` rows are
/// Gram-Schmidt orthonormalised (rows beyond `cols` cannot be).
fn orth_rows(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
    let mut w: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32()).collect();
    for i in 0..rows {
        if i < cols {
            for j in 0..i {
                let dot: f32 = (0..cols).map(|c| w[i * cols + c] * w[j * cols + c]).sum();
                for c in 0..cols {
                    w[i * cols + c] -= dot * w[j * cols + c];
                }
            }
        }
        let norm = (0..cols)
            .map(|c| w[i * cols + c] * w[i * cols + c])
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for c in 0..cols {
            w[i * cols + c] /= norm;
        }
    }
    w
}

fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

fn target(x: &[f32], y: &mut [f32]) {
    for i in 0..x.len() / M {
        let r = &x[i * M..(i + 1) * M];
        let cross = 0.5 * (r[2] + r[3]).tanh();
        for j in 0..M {
            y[i * M + j] = (3.0 * r[0]).sin() * (r[1] + 0.2 * j as f32).cos() + cross;
        }
    }
}

fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f32>() / a.len() as f32
}

struct Data {
    x: Vec<f32>,
    y: Vec<f32>,
    xt: Vec<f32>,
    yt: Vec<f32>,
}

fn data(seed: u64) -> Data {
    let mut rng = Rng::new(seed);
    let mut draw = || -> Vec<f32> { (0..N * M).map(|_| 0.5 * rng.next_f32()).collect() };
    let (x, xt) = (draw(), draw());
    let (mut y, mut yt) = (vec![0.0; N * M], vec![0.0; N * M]);
    target(&x, &mut y);
    target(&xt, &mut yt);
    Data { x, y, xt, yt }
}

// ── Arms ─────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Arm {
    name: &'static str,
    k: usize,
    d: usize,
    fusion: BranchFusion,
    shared_basis: bool,
    concat: bool,
}

const A1: Arm = Arm {
    name: "A1",
    k: 1,
    d: 8,
    fusion: BranchFusion::SumInvSqrtK,
    shared_basis: false,
    concat: false,
};
const A2: Arm = Arm {
    name: "A2",
    k: 2,
    d: 5,
    fusion: BranchFusion::SumInvSqrtK,
    shared_basis: false,
    concat: false,
};
const A3: Arm = Arm {
    name: "A3",
    k: 2,
    d: 5,
    fusion: BranchFusion::SumInvSqrtK,
    shared_basis: false,
    concat: true,
};
const A4: Arm = Arm {
    name: "A4",
    k: 2,
    d: 5,
    fusion: BranchFusion::SumInvSqrtK,
    shared_basis: true,
    concat: false,
};
const A5: Arm = Arm {
    name: "A5",
    k: 1,
    d: 12,
    fusion: BranchFusion::SumInvSqrtK,
    shared_basis: false,
    concat: false,
};
/// One A6 snapshot: an independently trained half-width complete predictor.
const HALF: Arm = Arm {
    name: "half",
    k: 1,
    d: 5,
    fusion: BranchFusion::Mean,
    shared_basis: false,
    concat: false,
};

impl Arm {
    fn cfg(&self) -> BranchParallelConfig {
        BranchParallelConfig {
            d_model: M,
            d_branch: self.d,
            k: KB,
            basis: FuncAttnBasis::Sigmoid,
            alpha: ALPHA,
            temperature: TEMP,
            cholesky_jitter: 1e-6,
            fusion: self.fusion,
        }
    }
    fn shared_len(&self) -> usize {
        if self.shared_basis {
            self.d * M + KB * self.d
        } else {
            0
        }
    }
    fn branch_len(&self) -> usize {
        let fa = 3 * self.d * self.d + if self.shared_basis { 0 } else { KB * self.d };
        2 * M * self.d + fa
    }
    fn params(&self) -> usize {
        self.shared_len()
            + self.k * self.branch_len()
            + if self.concat { M * self.k * M } else { 0 }
    }

    fn init(&self, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let d = self.d;
        let mut p = Vec::with_capacity(self.params());
        if self.shared_basis {
            p.extend(orth_rows(d, M, &mut rng));
            p.extend(orth_rows(KB, d, &mut rng));
        }
        for _ in 0..self.k {
            let pin = if d == M {
                eye(M)
            } else {
                orth_rows(d, M, &mut rng)
            };
            let pout = transpose(&pin, d, M);
            p.extend(&pin);
            if !self.shared_basis {
                p.extend(orth_rows(KB, d, &mut rng));
            }
            for _ in 0..3 {
                p.extend(eye(d));
            }
            p.extend(pout);
        }
        if self.concat {
            let s = 1.0 / (self.k as f32).sqrt();
            for r in 0..M {
                for _b in 0..self.k {
                    for c in 0..M {
                        p.push(if r == c { s } else { 0.0 });
                    }
                }
            }
        }
        assert_eq!(p.len(), self.params());
        p
    }

    /// Slice `p` into per-branch weights (writes into `out`, max 2 branches).
    fn views<'a>(&self, p: &'a [f32]) -> ([BranchWeights<'a>; 2], &'a [f32]) {
        let d = self.d;
        let (shared, rest) = p.split_at(self.shared_len());
        let (pb, wb_shared) = if self.shared_basis {
            let (a, b) = shared.split_at(d * M);
            (Some(a), Some(b))
        } else {
            (None, None)
        };
        let mut off = 0;
        let mut take = |len: usize| {
            let s = &rest[off..off + len];
            off += len;
            s
        };
        let mut mk = || {
            let in_proj = take(d * M);
            let w_basis = match wb_shared {
                Some(w) => w,
                None => take(KB * d),
            };
            let (w_q, w_k, w_v) = (take(d * d), take(d * d), take(d * d));
            let out_proj = take(M * d);
            BranchWeights {
                in_proj,
                basis_in_proj: pb,
                w_basis,
                w_q,
                w_k,
                w_v,
                out_proj,
            }
        };
        let b0 = mk();
        let b1 = if self.k > 1 { mk() } else { b0 };
        let tail = &rest[off..];
        ([b0, b1], tail)
    }
}

struct Work {
    s: BranchParallelScratch,
    u: Vec<f32>,
    out: Vec<f32>,
}

impl Work {
    fn new(arm: &Arm) -> Self {
        Self {
            s: BranchParallelScratch::new(N, &arm.cfg()),
            u: vec![0.0; 2 * N * M],
            out: vec![0.0; N * M],
        }
    }
}

fn forward(arm: &Arm, p: &[f32], x: &[f32], w: &mut Work) {
    let cfg = arm.cfg();
    let (br, tail) = arm.views(p);
    let ok = if arm.concat {
        // Each branch at scale 1, then the learned W_f over the concat.
        let one = BranchParallelConfig {
            fusion: BranchFusion::Mean,
            ..cfg
        };
        for b in 0..arm.k {
            let u = &mut w.u[b * N * M..(b + 1) * N * M];
            if branch_parallel_forward(x, &br[b..b + 1], &one, &mut w.s, u).is_err() {
                w.out.fill(f32::NAN);
                return;
            }
        }
        for i in 0..N {
            for r in 0..M {
                let mut acc = 0.0f32;
                for b in 0..arm.k {
                    let wf = &tail[r * arm.k * M + b * M..r * arm.k * M + (b + 1) * M];
                    let u = &w.u[b * N * M + i * M..b * N * M + (i + 1) * M];
                    for c in 0..M {
                        acc += wf[c] * u[c];
                    }
                }
                w.out[i * M + r] = acc;
            }
        }
        Ok(())
    } else {
        branch_parallel_forward(x, &br[..arm.k], &cfg, &mut w.s, &mut w.out)
    };
    if ok.is_err() {
        w.out.fill(f32::NAN);
    }
}

fn loss(arm: &Arm, p: &[f32], x: &[f32], y: &[f32], w: &mut Work) -> f32 {
    forward(arm, p, x, w);
    mse(&w.out, y)
}

/// FD-SGD (central differences, sequential coordinate sweep — the Plan-286
/// recipe). Returns the parameter vector at each requested checkpoint.
fn train(arm: &Arm, init: &[f32], d: &Data, lr: f32, checkpoints: &[usize]) -> Vec<Vec<f32>> {
    let mut p = init.to_vec();
    let mut w = Work::new(arm);
    let inv = 1.0 / (2.0 * FD_EPS);
    let last = *checkpoints.last().unwrap();
    let mut snaps = Vec::new();
    for step in 1..=last {
        for i in 0..p.len() {
            let o = p[i];
            p[i] = o + FD_EPS;
            let lp = loss(arm, &p, &d.x, &d.y, &mut w);
            p[i] = o - FD_EPS;
            let lm = loss(arm, &p, &d.x, &d.y, &mut w);
            let g = (lp - lm) * inv;
            p[i] = if g.is_finite() { o - lr * g } else { f32::NAN };
        }
        if checkpoints.contains(&step) {
            snaps.push(p.clone());
        }
    }
    snaps
}

#[derive(Clone, Copy, Debug)]
struct Score {
    train: f32,
    test: f32,
}

fn score(arm: &Arm, p: &[f32], d: &Data) -> Score {
    let mut w = Work::new(arm);
    Score {
        train: loss(arm, p, &d.x, &d.y, &mut w),
        test: loss(arm, p, &d.xt, &d.yt, &mut w),
    }
}

/// A6 / A6b: route both halves' FUNCATTN weights through BLAKE3-committed
/// snapshots + the store, then fuse the two complete predictors.
fn frozen_fused(halves: [&[f32]; 2], fusion: BranchFusion, d: &Data) -> Score {
    let mut snaps = Vec::new();
    for (v, h) in halves.iter().enumerate() {
        let (br, _) = HALF.views(h);
        let b = br[0];
        let snap = FuncAttnWeightsSnapshot::from_weights(
            HALF.d,
            KB,
            FuncAttnBasis::Sigmoid,
            ALPHA,
            TEMP,
            b.w_basis.to_vec(),
            b.w_q.to_vec(),
            b.w_k.to_vec(),
            b.w_v.to_vec(),
            v as u64,
        );
        let store = FuncAttnSnapshotStore::new(snap);
        let cur = store.current().expect("store readable");
        assert!(cur.verify(), "snapshot commitment must verify");
        snaps.push((cur, b.in_proj, b.out_proj));
    }
    let br: Vec<BranchWeights<'_>> = snaps
        .iter()
        .map(|(s, pin, pout)| BranchWeights {
            in_proj: pin,
            basis_in_proj: None,
            w_basis: &s.w_basis,
            w_q: &s.w_q,
            w_k: &s.w_k,
            w_v: &s.w_v,
            out_proj: pout,
        })
        .collect();
    let cfg = BranchParallelConfig { fusion, ..A2.cfg() };
    let mut s = BranchParallelScratch::new(N, &cfg);
    let mut out = vec![0.0; N * M];
    let mut eval =
        |x: &[f32], y: &[f32]| match branch_parallel_forward(x, &br, &cfg, &mut s, &mut out) {
            Ok(()) => mse(&out, y),
            Err(_) => f32::NAN,
        };
    Score {
        train: eval(&d.x, &d.y),
        test: eval(&d.xt, &d.yt),
    }
}

// ── Statistics (pre-registered rule) ─────────────────────────────────────

fn verdict(arm: &[f32], reference: &[f32]) -> (f32, f32, f32, &'static str) {
    let r: Vec<f32> = arm
        .iter()
        .zip(reference)
        .map(|(a, b)| {
            if a.is_finite() && b.is_finite() {
                100.0 * (a - b) / b
            } else {
                f32::INFINITY
            }
        })
        .collect();
    let n = r.len() as f32;
    let mean = r.iter().sum::<f32>() / n;
    let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n - 1.0);
    let se = (var / n).sqrt();
    let delta = if se.is_finite() {
        (2.5 * se).max(1.0)
    } else {
        f32::INFINITY
    };
    let class = if !mean.is_finite() {
        "LOSS"
    } else if mean < -delta {
        "WIN"
    } else if mean > delta {
        "LOSS"
    } else {
        "TIE"
    };
    (mean, se, delta, class)
}

fn par_map<T: Send, R: Send>(jobs: Vec<T>, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(jobs.len().max(1));
    let jobs: Vec<(usize, T)> = jobs.into_iter().enumerate().collect();
    let queue = std::sync::Mutex::new(jobs);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let job = queue.lock().unwrap().pop();
                    let Some((i, t)) = job else { break };
                    let r = f(t);
                    results.lock().unwrap().push((i, r));
                }
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|(i, _)| *i);
    r.into_iter().map(|(_, v)| v).collect()
}

fn init_seed(seed: u64, arm: &Arm, salt: u64) -> u64 {
    seed ^ (arm.d as u64) << 32 ^ (arm.k as u64) << 40 ^ salt.wrapping_mul(0x9E37_79B9)
}

/// Pilot: the largest LR on the ladder for which every arm stays finite and
/// lowers train MSE over STEPS_LONG on the PILOT seed.
fn pick_lr() -> f32 {
    let d = data(PILOT_SEED);
    let arms = [A1, A2, A3, A4, A5, HALF];
    for lr in LR_LADDER {
        let ok = par_map(arms.to_vec(), |a| {
            let init = a.init(init_seed(PILOT_SEED, &a, 1));
            let s0 = score(&a, &init, &d).train;
            let p = train(&a, &init, &d, lr, &[STEPS_LONG]);
            let s = score(&a, &p[0], &d).train;
            eprintln!("[pilot] lr={lr} {:>4} init {s0:.5} -> {s:.5}", a.name);
            s.is_finite() && s < s0
        });
        if ok.iter().all(|&b| b) {
            return lr;
        }
    }
    panic!("no LR on the ladder keeps every arm finite and learning");
}

#[test]
fn issue_1238_branch_parallel_goat() {
    for a in [A1, A2, A3, A4, A5, HALF] {
        eprintln!(
            "arm {:>4}: K={} d_b={} params={}",
            a.name,
            a.k,
            a.d,
            a.params()
        );
    }
    let lr = pick_lr();
    eprintln!("PILOT lr = {lr}");

    // G1 — bit-identical re-run (A2, first gate seed, short budget).
    {
        let d = data(GATE_SEEDS[0]);
        let init = A2.init(init_seed(GATE_SEEDS[0], &A2, 1));
        let a = train(&A2, &init, &d, lr, &[20]);
        let b = train(&A2, &init, &d, lr, &[20]);
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&a[0]),
            bits(&b[0]),
            "G1: FD-SGD re-run must be bit-identical"
        );
        eprintln!("VERDICT G1 PASS (bit-identical re-run)");
    }

    // Train every (arm, seed) once to STEPS_LONG with a STEPS_SHORT checkpoint.
    let trained = [A1, A2, A3, A4, A5];
    let mut jobs = Vec::new();
    for &seed in &GATE_SEEDS {
        for a in trained {
            jobs.push((a, seed, 1u64));
        }
        jobs.push((HALF, seed, 0xA));
        jobs.push((HALF, seed, 0xB));
    }
    let t0 = Instant::now();
    let runs = par_map(jobs.clone(), |(a, seed, salt)| {
        let d = data(seed);
        let init = a.init(init_seed(seed, &a, salt));
        train(&a, &init, &d, lr, &[STEPS_SHORT, STEPS_LONG])
    });
    eprintln!(
        "trained {} runs in {:.1}s",
        runs.len(),
        t0.elapsed().as_secs_f64()
    );

    for (ci, steps) in [STEPS_SHORT, STEPS_LONG].into_iter().enumerate() {
        let mut test = std::collections::BTreeMap::<&str, Vec<f32>>::new();
        let mut train_m = std::collections::BTreeMap::<&str, Vec<f32>>::new();
        let per_seed = trained.len() + 2;
        for (si, &seed) in GATE_SEEDS.iter().enumerate() {
            let d = data(seed);
            let base = si * per_seed;
            for (ai, a) in trained.iter().enumerate() {
                let s = score(a, &runs[base + ai][ci], &d);
                test.entry(a.name).or_default().push(s.test);
                train_m.entry(a.name).or_default().push(s.train);
            }
            let ha = &runs[base + trained.len()][ci];
            let hb = &runs[base + trained.len() + 1][ci];
            for (name, h) in [("halfA", ha), ("halfB", hb)] {
                let s = score(&HALF, h, &d);
                test.entry(name).or_default().push(s.test);
                train_m.entry(name).or_default().push(s.train);
            }
            for (name, f) in [
                ("A6", BranchFusion::Mean),
                ("A6b", BranchFusion::SumInvSqrtK),
            ] {
                let s = frozen_fused([ha, hb], f, &d);
                test.entry(name).or_default().push(s.test);
                train_m.entry(name).or_default().push(s.train);
            }
        }
        eprintln!("\n=== budget {steps} steps (lr {lr}) — per-seed held-out MSE ===");
        for (k, v) in &test {
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let tr = train_m[k].iter().sum::<f32>() / v.len() as f32;
            eprintln!("{k:>6}: test {v:.5?} mean {mean:.5} | train mean {tr:.5}");
        }
        let rows = [
            ("G2", "A2", "A1"),
            ("G3", "A2", "A5"),
            ("T6", "A3", "A2"),
            ("T5", "A4", "A2"),
            ("A6gate", "A6", "A1"),
            ("A6b", "A6b", "A6"),
            ("A6vsA2", "A6", "A2"),
            ("A6vsHalf", "A6", "halfA"),
        ];
        for (g, a, r) in rows {
            let (mean, se, delta, class) = verdict(&test[a], &test[r]);
            eprintln!(
                "VERDICT {g} budget={steps} {a} vs {r}: r̄={mean:+.2}pp SE={se:.2} δ={delta:.2} → {class}"
            );
        }
    }

    // G5 — forward latency, A2 vs A1, interleaved chunks (median ratio).
    let d = data(GATE_SEEDS[0]);
    let p1 = A1.init(1);
    let p2 = A2.init(2);
    let (mut w1, mut w2) = (Work::new(&A1), Work::new(&A2));
    const CHUNK: usize = 200;
    let mut ratios = Vec::new();
    for _ in 0..101 {
        let t = Instant::now();
        for _ in 0..CHUNK {
            forward(&A1, black_box(&p1), black_box(&d.xt), &mut w1);
            black_box(&w1.out);
        }
        let a = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..CHUNK {
            forward(&A2, black_box(&p2), black_box(&d.xt), &mut w2);
            black_box(&w2.out);
        }
        let b = t.elapsed().as_secs_f64();
        assert!(a > 0.0, "A1 timed region vanished (loud zero)");
        ratios.push(b / a);
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ratios[ratios.len() / 2];
    eprintln!(
        "VERDICT G5 A2/A1 forward wall median ratio {med:.3} (bar ≤ 2.2) → {}",
        if med <= 2.2 { "PASS" } else { "FAIL" }
    );
}
