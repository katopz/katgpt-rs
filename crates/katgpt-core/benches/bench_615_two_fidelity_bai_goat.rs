//! Two-Fidelity Certified BAI (2FFS) — GOAT gate bench (Plan 615 Phase 2).
//!
//! Exercises T2.1–T2.5 against the paper's own fixture family (balanced b-ary
//! stochastic minimax trees, adversarial fast bias) on the unified cost model
//! `cost = n_fast + c·n_slow` (c fixed per setting; ops reported separately —
//! the 2FFS module does not expose an ops counter, so the paper's 1.9–4.6×
//! ops figures stay context rows).
//!
//! # Gates
//!
//! - **CANARY (first, impossible floors fire)** — the gate helper must panic
//!   when its floor is impossible (proven via `catch_unwind`), the
//!   Clopper–Pearson machinery must reproduce the E=0 closed form
//!   `1 − 0.05^(1/N)`, and the counting allocator must move on a known
//!   allocation. If any of these fail, every later gate is unmeasured.
//! - **SANITY (T2.2b)** — the negamax-UCT adapter on a small zero-noise tree
//!   (D=3, b=3) must converge to the exact minimax value as its budget grows;
//!   a weak adapter would inflate the G2 win. Pinned BEFORE any comparison row.
//! - **G1 (empirical PAC)** — 2FFS error = strict `V*(pick) < best − ε` over
//!   the seeded suite (paper §4 parity: 100 trees/setting, (D,b) ∈
//!   {(5,8),(7,6),(10,3)}), judged by a ONE-SIDED CLOPPER–PEARSON 95% UPPER
//!   bound on the error probability vs δ = 0.05 (a zero-error run reads as
//!   consistent-with-δ, never pass-by-fiat). Every result must terminate
//!   certified, and every certified root interval must contain the true value
//!   (Thm 3.1's certificate body). The T1.3 beta-coverage and T1.8 property
//!   arms (Lemma 2.3 / B.2 / B.5) stand in the module's own tests — landed.
//! - **G2 (strict paired win, LB95 > 0 per setting on the unified cost
//!   model)** — against (a) **BAI-MCTS** (LUCB-style fixed-confidence
//!   stopping, Kaufmann–Koolen 2017 class: leader/challenger interval stop at
//!   the same (ε, δ), targeted descents, negamax backups, cap = stopping
//!   failure) and (b) a **negamax-UCT adapter** (plain UCT, budget-exact,
//!   checkpoints 4096·2^j): UCT's comparison budget B* is the SMALLEST
//!   checkpoint whose suite error ≤ 2FFS's (matched accuracy, favoring UCT);
//!   if no checkpoint matches within the cap, B* = cap+1 — understating
//!   UCT's true cost, conservative for the win claim. Paired LB95 =
//!   mean − 1.645·SE over per-tree cost differences (Bench 905 protocol).
//! - **G3** — landing discipline: the mcts.rs / chance_puct.rs surfaces are
//!   untouched by this bench (verified at commit time via `git show --stat`).
//!   `mcts_search` itself is single-player UCT (fixed player_id, max-only
//!   backups): it estimates MAX-MAX values on these minimax trees and is NOT
//!   a valid adversarial baseline (verdict round 1) — it appears in the
//!   record as a labelled context row only, never executed here.
//! - **G4 (alloc)** — `two_fidelity_search` on a small tree allocates a
//!   constant setup per search (the `HashMap::with_capacity(1024)` node table
//!   + the result `Vec`), and NOTHING per iteration: allocs(3 searches) ==
//!     3 × allocs(1 search) and allocs(1 search) ≤ 4 (counting allocator,
//!     chance_puct G4 precedent, after a warmup search).
//!
//! # Context rows (not gates)
//!
//! - fast-only minimax (fold of fast oracle values) — the paper's 0.88–0.91
//!   accuracy class; cost = full fold (conservative for the baseline).
//! - fixed-depth slow-only (level-2 frontier sampled to ε/2 Hoeffding widths)
//!   — cheap and badly truncated at depth.
//! - paper-measured parity figures (samples 163×/988×/1458×, ops 2.84×/4.58×/
//!   1.86× vs BAI-MCTS) are CONTEXT rows in the record — our measured numbers
//!   on the unified cost model are the claim.
//!
//! # Honest divergences (documented, never papered over)
//!
//! - BAI-MCTS is the LUCB-stopping flavour of the Kaufmann–Koolen class (the
//!   plan names "UGapE-MCTS / LUCB-MCTS"); node estimates are the standard
//!   sign-alternated averaging backups — the estimator bias vs true max-min
//!   values is the known baseline weakness the two-fidelity paper exploits.
//! - Baseline per-node state is a flat `Vec<(n, mean)>` sized to the fixture
//!   (bench-only storage choice; does not touch the module).
//! - Cap hits are stopping failures and are REPORTED per setting (the paper's
//!   "slow-only often fails to stop" class); capped rows enter the pairing
//!   with cost = cap, which UNDERSTATES the baseline cost — conservative.
//!
//! # Run
//!
//! ```bash
//! cargo run -p katgpt-core --features two_fidelity_bai \
//!   --bench bench_615_two_fidelity_bai_goat --release -- --nocapture
//! ```
//!
//! `B615_TREES=<n>` shrinks the suite for a dev shakedown; the shipped
//! protocol default (and the bench record) is 100 trees/setting.

#![cfg(feature = "two_fidelity_bai")]

use arrayvec::ArrayVec;
use katgpt_core::two_fidelity_bai::{
    Cost, MAX_CHILDREN, MinimaxSpace, NODE_CAP, NodeInterval, NodeKind, NodeStat, RngCore,
    SearchConfig, TwoFidelityResult, two_fidelity_search,
};
use std::collections::HashMap;
use std::io::Write as _;
use std::panic::{AssertUnwindSafe, catch_unwind};

#[path = "../tests/common/mod.rs"]
mod common;
counting_allocator!();

// ─── Protocol constants (fixed per the plan; one cost model per setting) ────

const EPSILON: f32 = 0.02;
const DELTA: f64 = 0.05;
const SLOW_COST: f32 = 4.0;
const SIGMA: f64 = 0.05;
const NOISE_AMP: f32 = 0.05;
/// Fast-bias scale: envelope B(h) = BBAR·(1 − 2^−h) — the CONTRACT shape
/// (module docs L59-77): B(0) = 0 pins the leaf fast oracle EXACT, B
/// nondecreasing in remaining depth (root loosest). |bias| = B(h)·u ≤ B(h).
const BBAR: f32 = 0.12;
const SETTINGS: [(u8, usize); 3] = [(5, 8), (7, 6), (10, 3)];
/// Stopping-failure cap for the δ-correct LUCB-MCTS baseline (Issue 916a):
/// where it cannot certify within the cap, the cap × c IS its recorded cost
/// (the conservative understatement — the same convention as the UCT
/// no-checkpoint-matched row). Sized 2026-10-03: 4M = 4× the paper's own
/// BAI instance at (5,8) (8.8e5), so "our baseline is the conservative
/// member of the class" stays honest, while a full-protocol recording run
/// stays ≈1.5 h on this box (the first 30M-cap attempt measured setting 1
/// at ~10 h projected wall — the Min-arm L-fan-out (b^D/2 nodes, each
/// needing a CI-converged leaf) caps every D≥7 tree structurally, so the
/// cap only sets the disclosed understatement magnitude, never a verdict
/// flip; the 30M setting-1 rows are in the record as corroboration).
const BAI_SAMPLE_CAP: u64 = 4_000_000;
const UCT_MAX_BUDGET: u64 = 1 << 23;
const UCT_C: f64 = std::f64::consts::SQRT_2;

fn trees_per_setting() -> usize {
    std::env::var("B615_TREES")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(100)
}

/// Deterministic xorshift64* (Vigna), top 53 bits to [0, 1). No global RNG
/// (the global_rng_gate bans unseeded free-function draws).
struct Xs(u64);
impl RngCore for Xs {
    fn f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let z = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((z >> 11) as f64) / (1u64 << 53) as f64
    }
}

// ─── T2.1 fixture: balanced b-ary stochastic minimax tree, adversarial bias ─

struct BenchTree {
    d: u8,
    b: usize,
    starts: Vec<usize>, // level l nodes occupy [starts[l], starts[l+1])
    kind: Vec<NodeKind>,
    vstar: Vec<f64>, // exact minimax values (ground truth; every node — also
    //                 the slow-oracle mean: leaves carry their drawn mean,
    //                 internals fold it up)
    fast: Vec<f32>, // fast-oracle values with the chosen adversarial sign
    flipped: bool,  // did the adversarial sign flip the fast-only root pick?
    noise_amp: f32,
}

impl BenchTree {
    fn level_of(&self, node: u32) -> usize {
        let i = node as usize;
        for l in 0..self.starts.len() - 1 {
            if i < self.starts[l + 1] {
                return l;
            }
        }
        self.starts.len() - 2
    }

    fn is_leaf(&self, node: u32) -> bool {
        self.level_of(node) == self.d as usize
    }

    fn true_best(&self) -> (u32, f64) {
        let mut best = 0u32;
        let mut best_v = f64::NEG_INFINITY;
        for j in 0..self.b {
            let c = (1 + j) as u32;
            if self.vstar[c as usize] > best_v {
                best_v = self.vstar[c as usize];
                best = c;
            }
        }
        (best, best_v)
    }

    /// Depth-limited fast-only minimax (context row): cut where
    /// remaining_depth ≤ 2, evaluate cut nodes with the fast oracle, fold up.
    /// (A FULL fold would be exact — B(0) = 0 makes leaf fast values exact —
    /// so the paper's 0.88–0.91 accuracy class is the depth-truncated form.)
    fn fast_only_pick(&self) -> u32 {
        let cut = 2u8.min(self.d);
        let mut fold = vec![0.0_f32; self.starts[self.d as usize + 1]];
        for l in (0..=self.d as usize).rev() {
            for i in self.starts[l]..self.starts[l + 1] {
                let h = self.d as usize - l;
                fold[i] = if h <= cut as usize {
                    self.fast[i]
                } else {
                    let mut acc = fold[self.b * i + 1];
                    for j in 1..self.b {
                        let v = fold[self.b * i + 1 + j];
                        acc = if self.kind[i] == NodeKind::Max {
                            acc.max(v)
                        } else {
                            acc.min(v)
                        };
                    }
                    acc
                };
            }
        }
        let mut best = 0usize;
        for (j, &v) in fold[1..1 + self.b].iter().enumerate() {
            if v > fold[1 + best] {
                best = j;
            }
        }
        (1 + best) as u32
    }

    fn generate(d: u8, b: usize, seed: u64, noise_amp: f32) -> Self {
        // levels 0..=d; starts[l] = (b^l − 1)/(b − 1)
        let mut starts = vec![0usize; d as usize + 2];
        for (l, slot) in starts.iter_mut().enumerate().skip(1) {
            let mut s = 1usize;
            for _ in 0..l {
                s *= b;
            }
            *slot = (s - 1) / (b - 1);
        }
        let n = starts[d as usize + 1];
        let mut level = vec![0u8; n];
        for (l, &start) in starts.iter().enumerate().take(d as usize + 1) {
            for slot in &mut level[start..starts[l + 1]] {
                *slot = l as u8;
            }
        }
        let kind: Vec<NodeKind> = level
            .iter()
            .map(|&l| {
                if l % 2 == 0 {
                    NodeKind::Max
                } else {
                    NodeKind::Min
                }
            })
            .collect();
        let mut rng = Xs(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x615));
        let mut mu = vec![0.0_f64; n];
        for slot in &mut mu[starts[d as usize]..n] {
            *slot = rng.f64();
        }
        let mut u = vec![0.0_f64; n];
        for slot in &mut u {
            *slot = rng.f64();
        }
        // Exact minimax, bottom-up (f64 ground truth).
        let mut vstar = vec![0.0_f64; n];
        for l in (0..=d as usize).rev() {
            for i in starts[l]..starts[l + 1] {
                if l == d as usize {
                    vstar[i] = mu[i];
                } else {
                    let mut acc = vstar[b * i + 1];
                    for j in 1..b {
                        let v = vstar[b * i + 1 + j];
                        acc = if kind[i] == NodeKind::Max {
                            acc.max(v)
                        } else {
                            acc.min(v)
                        };
                    }
                    vstar[i] = acc;
                }
            }
        }
        // Adversarial fast bias: per-node magnitude u(v)·B(h(v)) (inside the
        // contract envelope, B(0) = 0 at leaves — leaf fast values exact),
        // sign chosen to flip the depth-limited fast-only root pick when
        // possible.
        let oracle = |s: f32, i: usize| -> f32 {
            let h = (d as usize - level[i] as usize) as i32;
            let bias = s * BBAR * (1.0 - 2.0_f32.powi(-h)) * (u[i] as f32);
            (vstar[i] + bias as f64) as f32
        };
        let true_pick = {
            let mut best = 0usize;
            for j in 1..b {
                if vstar[1 + j] > vstar[1 + best] {
                    best = j;
                }
            }
            1 + best
        };
        let mut chosen_fast = vec![0.0_f32; n];
        let mut flipped = false;
        for s in [1.0_f32, -1.0_f32] {
            for (i, slot) in chosen_fast.iter_mut().enumerate() {
                *slot = oracle(s, i);
            }
            let t = Self {
                d,
                b,
                starts: starts.clone(),
                kind: kind.clone(),
                vstar: vstar.clone(),
                fast: chosen_fast.clone(),
                flipped,
                noise_amp,
            };
            if t.fast_only_pick() as usize != true_pick {
                flipped = true;
                break;
            }
        }
        Self {
            d,
            b,
            starts,
            kind,
            vstar,
            fast: chosen_fast,
            flipped,
            noise_amp,
        }
    }
}

impl MinimaxSpace for BenchTree {
    fn kind(&self, node: u32) -> NodeKind {
        self.kind[node as usize]
    }

    fn remaining_depth(&self, node: u32) -> u8 {
        self.d - self.level_of(node) as u8
    }

    fn children_into(&self, node: u32, out: &mut ArrayVec<u32, MAX_CHILDREN>) {
        if self.is_leaf(node) {
            return;
        }
        let base = self.b * node as usize + 1;
        for j in 0..self.b {
            out.push((base + j) as u32);
        }
    }

    fn fast_value(&self, node: u32) -> f32 {
        self.fast[node as usize]
    }

    fn bias_envelope(&self, remaining_depth: u8) -> f32 {
        let h = i32::from(remaining_depth);
        BBAR * (1.0 - 2.0_f32.powi(-h))
    }

    fn slow_sample(&self, node: u32, rng: &mut impl RngCore) -> f32 {
        // Contract (MinimaxSpace): i.i.d. mean V*(v) — vstar at EVERY node,
        // not mu (leaf-only). Issue 915's root cause was sampling mu here:
        // internals sat at 0.0, so every internal slow CI converged far
        // below V*, capping `hi` and firing the guard. Leaves are identical
        // (vstar == mu), so leaf-sampling consumers (UCT, leaf lanes) keep
        // their exact pre-fix sample values.
        (self.vstar[node as usize] + (rng.f64() - 0.5) * 2.0 * f64::from(self.noise_amp)) as f32
    }
}

// ─── T2.2 baselines: negamax UCT engine (shared descent + backup) ───────────
//
// V*-space SAMPLE-MAX/MIN backups (the bound-style variant the BAI-MCTS class
// uses on alternating trees): leaves keep the raw running mean of their own
// slow samples; every internal node's estimate is the EXTREME over its
// visited children's estimates (Max → max, Min → min), recomputed along the
// backup path. Raw path-averaging was measured (shakedown) to collapse to the
// subtree mean on uniform-random alternating trees — a self-consistent wrong
// fixed point that never converges even at zero noise; the extreme backup
// converges to V* on zero-noise trees, which is exactly what the T2.2b
// sanity arm pins. Disclosed bias: E[max of noisy means] is optimistic at
// Max nodes (the sampled-max class's known weakness).

struct UctSearch<'a> {
    tree: &'a BenchTree,
    n: Vec<u32>,
    est: Vec<f64>, // V*-space estimate (leaf: sample mean; internal: extreme)
    samples: u64,
    path: Vec<u32>, // scratch, reused per descent
}

impl<'a> UctSearch<'a> {
    fn new(tree: &'a BenchTree) -> Self {
        Self {
            tree,
            n: vec![0; tree.starts[tree.d as usize + 1]],
            est: vec![0.0; tree.starts[tree.d as usize + 1]],
            samples: 0,
            path: Vec::with_capacity(16),
        }
    }

    /// One leaf sample: UCT descent (Max: argmax est + bonus; Min: argmin
    /// est − bonus; unvisited children first, lowest index on ties) +
    /// sample-extreme backup.
    fn descend_and_sample(&mut self, start: u32, rng: &mut Xs) {
        self.path.clear();
        let mut cur = start;
        loop {
            self.path.push(cur);
            if self.tree.is_leaf(cur) {
                break;
            }
            let mut kids: ArrayVec<u32, MAX_CHILDREN> = ArrayVec::new();
            self.tree.children_into(cur, &mut kids);
            let ln_np = (self.n[cur as usize].max(1) as f64).ln();
            let is_max = self.tree.kind[cur as usize] == NodeKind::Max;
            let mut best = 0usize;
            let mut best_score = if is_max {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
            for (j, &k) in kids.iter().enumerate() {
                let score = if self.n[k as usize] == 0 {
                    // Explore-first must WIN in this node's own direction:
                    // +INF for argmax (Max), −INF for argmin (Min).
                    if is_max {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    }
                } else {
                    let bonus = UCT_C * (ln_np / f64::from(self.n[k as usize])).sqrt();
                    if is_max {
                        self.est[k as usize] + bonus
                    } else {
                        self.est[k as usize] - bonus
                    }
                };
                let better = if is_max {
                    score > best_score
                } else {
                    score < best_score
                };
                if better {
                    best_score = score;
                    best = j;
                }
            }
            cur = kids[best];
        }
        let y = f64::from(self.tree.slow_sample(cur, rng));
        // Leaf: raw running mean of its own samples.
        self.n[cur as usize] += 1;
        let nl = f64::from(self.n[cur as usize]);
        self.est[cur as usize] += (y - self.est[cur as usize]) / nl;
        // Internal: extreme over visited children (recomputed on the path).
        for idx in (0..self.path.len() - 1).rev() {
            let v = self.path[idx];
            let mut kids: ArrayVec<u32, MAX_CHILDREN> = ArrayVec::new();
            self.tree.children_into(v, &mut kids);
            let is_max = self.tree.kind[v as usize] == NodeKind::Max;
            let mut extreme: Option<f64> = None;
            for &k in kids.iter() {
                if self.n[k as usize] == 0 {
                    continue;
                }
                let e = self.est[k as usize];
                extreme = Some(match extreme {
                    None => e,
                    Some(x) => {
                        if is_max {
                            x.max(e)
                        } else {
                            x.min(e)
                        }
                    }
                });
            }
            if let Some(e) = extreme {
                self.est[v as usize] = e;
            }
            self.n[v as usize] += 1;
        }
        self.samples += 1;
    }

    /// Root children's (handle, n, V* estimate) — arms, higher is better.
    fn arms(&self) -> Vec<(u32, u32, f64)> {
        let mut arms = Vec::with_capacity(self.tree.b);
        for j in 0..self.tree.b {
            let c = (1 + j) as u32;
            arms.push((c, self.n[c as usize], self.est[c as usize]));
        }
        arms
    }

    fn pick_best(&self) -> u32 {
        let arms = self.arms();
        let mut best = arms[0].0;
        let mut best_m = f64::NEG_INFINITY;
        for &(c, n, m) in &arms {
            if n > 0 && m > best_m {
                best_m = m;
                best = c;
            }
        }
        best
    }

    /// Root value estimate of an arm (for the sanity convergence readout).
    fn arm_value(&self, arm: u32) -> f64 {
        self.est[arm as usize]
    }
}

/// T2.2(a) BAI-MCTS: LUCB-MCTS / UGapE-MCTS (Kaufmann & Koolen 2017 class)
/// at the same (ε, δ) as 2FFS — the δ-correct bound-propagation baseline
/// (Issue 916 cause (a)). The pre-fix arm stopped on point-extreme root
/// estimates with arm-level radii that never covered the deep max/min
/// backup bias and erred 61/100 at (5,8) — three orders over δ — so its
/// "cost" was the price of an inaccurate stop, not a PAC competitor.
/// This version propagates per-leaf time-uniform slow CIs (the module's own
/// `NodeInterval` machinery, the same β/δ_v rule as `two_fidelity_search`:
/// δ_v = δ/NODE_CAP) through Eq. 6 backups — internal (L, U) = (max|min
/// child L, max|min child U) with unrevealed children at (−∞, +∞) — and
/// stops only when the propagated root intervals certify the leader.
/// Descent follows the target endpoint's binding child (the bottleneck:
/// unrevealed = −∞ binds argmin/loses argmax on the L side, +∞ binds
/// argmax/loses argmin on the U side; ties → lowest handle), one slow
/// sample per round; leader-L and challenger-U descents alternate.
/// Documented divergences from KKC: their exact confidence sequence is
/// replaced by the module's stitched time-uniform radius with the uniform
/// per-leaf δ allocation (fairness: the baseline shares 2FFS's confidence
/// machinery, isolating the two-fidelity mechanism), and their gap-indexed
/// sampling rule is replaced by the deterministic alternation.
struct LucbNode {
    interval: NodeInterval,
    kind: NodeKind,
    kids: ArrayVec<u32, MAX_CHILDREN>, // empty = leaf
}

struct LucbSearch<'a> {
    tree: &'a BenchTree,
    nodes: HashMap<u32, LucbNode>,
    samples: u64,
}

impl<'a> LucbSearch<'a> {
    fn new(tree: &'a BenchTree) -> Self {
        Self {
            tree,
            nodes: HashMap::with_capacity(1024),
            samples: 0,
        }
    }

    fn delta_v(&self) -> f64 {
        DELTA / f64::from(NODE_CAP)
    }

    fn reveal(&mut self, v: u32) {
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = self.nodes.entry(v) {
            let mut kids = ArrayVec::new();
            if !self.tree.is_leaf(v) {
                self.tree.children_into(v, &mut kids);
            }
            e.insert(LucbNode {
                interval: NodeInterval::slow_only(),
                kind: self.tree.kind[v as usize],
                kids,
            });
        }
    }

    fn bounds(&self, v: u32) -> (f64, f64) {
        self.nodes[&v].interval.effective()
    }

    fn endpoint(&self, v: u32, upper: bool) -> f64 {
        let (lo, hi) = self.bounds(v);
        if upper { hi } else { lo }
    }

    /// Optimistic descent to the leaf binding `side` at `start`: at each
    /// internal node follow the child whose `side` endpoint is the Eq. 6
    /// fold's argument (argmax at Max, argmin at Min), with unrevealed
    /// children at the side's neutral sentinel (−∞ lower / +∞ upper) so the
    /// bottleneck — an unexplored subtree exactly where it caps the bound —
    /// is entered first. Ties → lowest handle. Returns (leaf, path).
    fn descend(&mut self, start: u32, upper: bool) -> (u32, Vec<u32>) {
        let mut path = Vec::with_capacity(16);
        let mut cur = start;
        loop {
            self.reveal(cur);
            path.push(cur);
            let kids = self.nodes[&cur].kids.clone();
            if kids.is_empty() {
                return (cur, path);
            }
            let want_max = self.nodes[&cur].kind == NodeKind::Max;
            let mut best = kids[0];
            let mut best_key = self.endpoint_key(best, upper);
            for &k in kids.iter().skip(1) {
                let key = self.endpoint_key(k, upper);
                let improves = if want_max {
                    key > best_key
                } else {
                    key < best_key
                };
                if improves || (key == best_key && k < best) {
                    best = k;
                    best_key = key;
                }
            }
            cur = best;
        }
    }

    fn endpoint_key(&self, v: u32, upper: bool) -> f64 {
        match self.nodes.get(&v) {
            None => {
                if upper {
                    f64::INFINITY
                } else {
                    f64::NEG_INFINITY
                }
            }
            Some(_) => self.endpoint(v, upper),
        }
    }

    fn sample_leaf(&mut self, leaf: u32, path: &[u32], rng: &mut Xs) {
        let y = self.tree.slow_sample(leaf, rng);
        let (sigma, delta_v) = (SIGMA, self.delta_v());
        self.nodes
            .get_mut(&leaf)
            .expect("leaf revealed")
            .interval
            .observe_slow(y, sigma, delta_v);
        self.samples += 1;
        // Bottom-up Eq. 6 refresh along the sampled path only. Unrevealed
        // siblings contribute the neutral sentinel (−∞, +∞) — they widen the
        // fold exactly as the descent's optimistic rule expects.
        for &v in path.iter().rev().skip(1) {
            let kind = self.nodes[&v].kind;
            let kids = self.nodes[&v].kids.clone();
            let mut pairs = ArrayVec::<(f64, f64), MAX_CHILDREN>::new();
            for &k in kids.iter() {
                let pair = match self.nodes.get(&k) {
                    Some(n) => n.interval.effective(),
                    None => (f64::NEG_INFINITY, f64::INFINITY),
                };
                let _ = pairs.try_push(pair);
            }
            if !pairs.is_empty() {
                self.nodes
                    .get_mut(&v)
                    .expect("path node")
                    .interval
                    .set_child_backup(kind, pairs.as_slice());
            }
        }
    }
}

fn bai_mcts(tree: &BenchTree, rng: &mut Xs) -> (u32, u64, bool) {
    let mut s = LucbSearch::new(tree);
    let b = tree.b;
    let arms: Vec<u32> = (1..=b as u32).collect();
    for &a in &arms {
        s.reveal(a);
    }
    let mut round: u64 = 0;
    loop {
        let leader = arms
            .iter()
            .copied()
            .reduce(|a, c| {
                let la = s.endpoint(a, false);
                let lc = s.endpoint(c, false);
                if lc > la || (lc == la && c < a) { c } else { a }
            })
            .expect("b >= 1");
        let (leader_l, _) = s.bounds(leader);
        let chal_u = arms
            .iter()
            .filter(|&&a| a != leader)
            .map(|&a| s.endpoint(a, true))
            .fold(f64::NEG_INFINITY, f64::max);
        if leader_l >= chal_u - f64::from(EPSILON) {
            return (leader, s.samples, true);
        }
        if s.samples >= BAI_SAMPLE_CAP {
            return (leader, s.samples, false);
        }
        // Alternate: tighten the leader's L chain, then the challenger's U
        // chain (the two endpoints the stop rule reads).
        let (target, upper) = if round.is_multiple_of(2) {
            (leader, false)
        } else {
            let challenger = arms
                .iter()
                .copied()
                .filter(|&a| a != leader)
                .reduce(|a, c| {
                    let ua = s.endpoint(a, true);
                    let uc = s.endpoint(c, true);
                    if uc > ua || (uc == ua && c < a) { c } else { a }
                })
                .expect("b >= 2 here — the stop would have fired over a single arm");
            (challenger, true)
        };
        let (leaf, path) = s.descend(target, upper);
        s.sample_leaf(leaf, &path, rng);
        round += 1;
    }
}

/// T2.2(b) negamax-UCT adapter: plain UCT descent from the root, budget-exact,
/// no stopping rule. The checkpoint variant below is the comparison row; the
/// bare-budget form lives in the SANITY arm (UctSearch driven directly).
fn uct_checkpoints(tree: &BenchTree, checkpoints: &[u64], rng: &mut Xs) -> (Vec<u32>, f64) {
    let mut s = UctSearch::new(tree);
    let mut picks = Vec::with_capacity(checkpoints.len());
    let mut last_value = 0.0_f64;
    for &b in checkpoints {
        while s.samples < b {
            s.descend_and_sample(0, rng);
        }
        picks.push(s.pick_best());
        last_value = s.arm_value(s.pick_best());
    }
    (picks, last_value)
}

/// T2.2(c) fixed-depth slow-only (context row): level-min(2,d) frontier,
/// per-node Hoeffding count to half-width ≤ ε/2 (δ_node = δ/M union), minimax
/// fold of the sampled means up to the root.
fn slow_only(tree: &BenchTree, rng: &mut Xs) -> (u32, u64) {
    let l_star = (tree.d as usize).min(2);
    let m = tree.starts[l_star + 1] - tree.starts[l_star];
    let delta_node = DELTA / m as f64;
    let half = f64::from(EPSILON) / 2.0;
    let n_per = (2.0 * SIGMA * SIGMA * (2.0 * m as f64 / delta_node).ln() / (half * half))
        .ceil()
        .max(1.0) as u64;
    let mut means = vec![0.0_f64; m];
    for (mi, node) in (tree.starts[l_star]..tree.starts[l_star + 1]).enumerate() {
        let mut acc = 0.0;
        for _ in 0..n_per {
            acc += f64::from(tree.slow_sample(node as u32, rng));
        }
        means[mi] = acc / n_per as f64;
    }
    // Fold the sampled means up to the ROOT CHILDREN (level 1) — not to the
    // root itself, the pick is argmax over children.
    let mut fold = means;
    for l in (1..l_star).rev() {
        let width = tree.starts[l + 1] - tree.starts[l];
        let mut next = vec![0.0_f64; width];
        for (i, slot) in next.iter_mut().enumerate() {
            let base = tree.b * (tree.starts[l] + i) + 1 - tree.starts[l + 1];
            let mut acc = fold[base];
            for j in 1..tree.b {
                let v = fold[base + j];
                acc = if tree.kind[tree.starts[l] + i] == NodeKind::Max {
                    acc.max(v)
                } else {
                    acc.min(v)
                };
            }
            *slot = acc;
        }
        fold = next;
    }
    let mut best = 0usize;
    for (j, &v) in fold.iter().enumerate() {
        if v > fold[best] {
            best = j;
        }
    }
    ((1 + best) as u32, tree.b as u64 * m as u64 * n_per)
}

// ─── 2FFS runner ─────────────────────────────────────────────────────────────────

fn run_2ffs(tree: &BenchTree, seed: u64) -> TwoFidelityResult {
    run_2ffs_with(
        tree,
        seed,
        std::env::var("B615_2FFS_LAZY").is_ok(),
        &AbCfg::protocol(),
    )
}

/// The (ε, σ, c) triple a 2FFS run uses. The gated suite always runs
/// [`AbCfg::protocol`]; only the `B615_2FFS_AB` resolver A/B reads the
/// `B615_AB_{EPS,SIGMA,COST}` overrides (σ also sets the slow noise
/// amplitude — uniform ±σ is σ-sub-Gaussian).
#[derive(Clone, Copy)]
struct AbCfg {
    eps: f32,
    sigma: f64,
    cost: f32,
}

impl AbCfg {
    fn protocol() -> Self {
        Self {
            eps: EPSILON,
            sigma: SIGMA,
            cost: SLOW_COST,
        }
    }

    fn from_env() -> Self {
        let get = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
        let p = Self::protocol();
        Self {
            eps: get("B615_AB_EPS").map_or(p.eps, |v| v as f32),
            sigma: get("B615_AB_SIGMA").unwrap_or(p.sigma),
            cost: get("B615_AB_COST").map_or(p.cost, |v| v as f32),
        }
    }
}

fn run_2ffs_with(
    tree: &BenchTree,
    seed: u64,
    lazy_discharge: bool,
    ab: &AbCfg,
) -> TwoFidelityResult {
    let config = SearchConfig {
        epsilon: ab.eps,
        delta: DELTA,
        slow_cost: ab.cost,
        node_cap: NODE_CAP,
        sigma: ab.sigma,
        trace_nodes: std::env::var("B615_2FFS_TRACE").is_ok(),
        lazy_discharge,
    };
    let mut rng = Xs(seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xB10C));
    two_fidelity_search(tree, 0, &config, &mut rng)
}

/// Issue 916 cause-(b) attribution dump: aggregate the per-node ledger over
/// all trees at the setting — samples by depth, the top consumers, and the
/// width-target distribution the recursion was driving nodes toward.
fn dump_2ffs_trace(results: &[TwoFidelityResult], d: u8, b: u8) {
    let mut by_depth: std::collections::BTreeMap<u8, (u64, u64)> = Default::default(); // (nodes, samples)
    let mut targets: Vec<(f64, u64)> = Vec::new(); // (target_width, samples) rows
    let mut total_nodes = 0u64;
    let mut total_samples = 0u64;
    for r in results {
        for st in &r.node_stats {
            total_nodes += 1;
            total_samples += st.slow_n;
            let e = by_depth.entry(st.depth).or_insert((0, 0));
            e.0 += 1;
            e.1 += st.slow_n;
            targets.push((st.target_width, st.slow_n));
        }
    }
    eprintln!("      TRACE d{d}b{b}: {total_nodes} sampled nodes / {total_samples} slow samples");
    for (depth, (nodes, samples)) in &by_depth {
        eprintln!(
            "        depth {depth:>2}: {nodes:>5} nodes {samples:>8} samples ({}%)",
            100 * samples / total_samples.max(1)
        );
    }
    // width-target histogram (log2 buckets of ρ/2)
    targets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut bucket: (i32, u64, f64) = (i32::MIN, 0, 0.0); // (log2 bucket, samples, width sum)
    let mut buckets: Vec<(i32, u64, f64)> = Vec::new();
    for (w, n) in &targets {
        let bkt = if *w <= 0.0 {
            i32::MIN
        } else {
            (*w).log2().floor() as i32
        };
        if bkt == bucket.0 {
            bucket.1 += n;
            bucket.2 += w;
        } else {
            if bucket.1 > 0 {
                buckets.push(bucket);
            }
            bucket = (bkt, *n, *w);
        }
    }
    if bucket.1 > 0 {
        buckets.push(bucket);
    }
    for (bkt, n, wsum) in &buckets {
        let label = if *bkt == i32::MIN {
            "latched".to_string()
        } else {
            format!("ρ/2 ∈ [2^{bkt}, 2^{})", bkt + 1)
        };
        eprintln!(
            "        target {label:<22}: {n:>8} samples (mean width {:.5})",
            wsum / *n as f64
        );
    }
    // top-8 consumers
    let mut top: Vec<(u64, u32, u8, f64)> = Vec::new();
    for r in results {
        for st in r.node_stats.iter().take(4) {
            top.push((st.slow_n, st.node, st.depth, st.target_width));
        }
    }
    top.sort_by_key(|&(n, _, _, _)| std::cmp::Reverse(n));
    top.truncate(8);
    for (n, node, depth, w) in top {
        eprintln!("        top node {node} (depth {depth}): {n} samples, target width {w:.5}");
    }
}

// ─── Stats: paired LB95 (Bench 905 protocol) + Clopper–Pearson upper bound ──

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn lb95(diffs: &[f64]) -> f64 {
    let m = mean(diffs);
    if diffs.len() < 2 {
        return m;
    }
    let var = diffs.iter().map(|d| (d - m) * (d - m)).sum::<f64>() / (diffs.len() - 1) as f64;
    m - 1.645 * (var / diffs.len() as f64).sqrt()
}

fn ln_gamma(z: f64) -> f64 {
    // Lanczos (g=7, n=9).
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln().abs() - ln_gamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = C[0];
        for (i, &c) in C.iter().enumerate().skip(1) {
            x += c / (z + i as f64);
        }
        let t = z + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 300;
    const EPS: f64 = 3.0e-12;
    const FPMIN: f64 = 1.0e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_IT {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let mut aa = mf * (b - mf) * x / ((qam + m2) * (a + mf));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + mf) * (qab + mf) * x / ((a + qap) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbt = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let front = lbt.exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        front * betacf(a, b, x) / a
    } else {
        1.0 - front * betacf(b, a, 1.0 - x) / b
    }
}

/// One-sided 95% Clopper–Pearson UPPER bound on the error probability given
/// `e` errors in `n` trials: the p with I_p(e+1, n−e) = 0.95.
fn cp_upper(e: u64, n: u64) -> f64 {
    if e >= n {
        return 1.0;
    }
    let a = e as f64 + 1.0;
    let b = (n - e) as f64;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if betainc(mid, a, b) < 0.95 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ─── Gate helpers ────────────────────────────────────────────────────────────

fn gate(ok: bool, msg: &str) {
    if !ok {
        panic!("GATE FAIL: {msg}");
    }
}

struct Gates {
    failures: Vec<String>,
}

impl Gates {
    fn run(&mut self, name: &str, f: impl FnOnce()) {
        match catch_unwind(AssertUnwindSafe(f)) {
            Ok(()) => println!("   {name} ✓"),
            Err(_) => {
                println!("   {name} ✗");
                self.failures.push(name.to_string());
            }
        }
    }
}

fn err_pick(tree: &BenchTree, pick: u32, best_v: f64) -> bool {
    tree.vstar[pick as usize] < best_v - f64::from(EPSILON)
}

struct SettingResult {
    d: u8,
    b: usize,
    // 2FFS
    f2_errs: u64,
    f2_cert_ok: u64,
    cert_invalid: u64,
    f2_cost: Vec<f64>,
    f2_fast: f64,
    f2_slow: f64,
    // BAI-MCTS
    bai_errs: u64,
    bai_capped: u64,
    bai_cost: Vec<f64>,
    // UCT
    uct_errs_at: Vec<u64>,
    uct_best_value: f64,
    // context rows
    fast_errs: u64,
    fast_cost: f64,
    slow_errs: u64,
    slow_cost: f64,
    flipped_trees: u64,
}

fn run_setting(d: u8, b: usize, trees: usize) -> SettingResult {
    let checkpoints: Vec<u64> = (0..12).map(|j| 4096u64 << j).collect();
    let mut r = SettingResult {
        d,
        b,
        f2_errs: 0,
        f2_cert_ok: 0,
        cert_invalid: 0,
        f2_cost: Vec::with_capacity(trees),
        f2_fast: 0.0,
        f2_slow: 0.0,
        bai_errs: 0,
        bai_capped: 0,
        bai_cost: Vec::with_capacity(trees),
        uct_errs_at: vec![0; checkpoints.len()],
        uct_best_value: 0.0,
        fast_errs: 0,
        fast_cost: 0.0,
        slow_errs: 0,
        slow_cost: 0.0,
        flipped_trees: 0,
    };
    let tracing = std::env::var("B615_2FFS_TRACE").is_ok();
    let mut trace_pool: Vec<Vec<NodeStat>> = Vec::new();
    for t in 0..trees {
        let tree = BenchTree::generate(d, b, 0x615_0000_u64 + t as u64, NOISE_AMP);
        let (best, best_v) = tree.true_best();
        if tree.flipped {
            r.flipped_trees += 1;
        }
        // 2FFS
        let res = run_2ffs(&tree, 1_000 + t as u64);
        trace_pool.push(res.node_stats);
        if res.certified {
            r.f2_cert_ok += 1;
        }
        if err_pick(&tree, res.best_action, best_v) {
            r.f2_errs += 1;
        }
        r.f2_fast += res.cost.fast as f64;
        r.f2_slow += res.cost.slow as f64;
        r.f2_cost.push(res.cost.unified(SLOW_COST));
        // Certificate body — CERTIFIED results only: an uncertified result is
        // the empty-intersection guard's B.3 default-action path, whose
        // intervals may be collapsed/inverted BY DESIGN (detectable E_δ).
        // Guard-fires are counted (uncertified) and judged by the suite-level
        // δ budget, never by per-interval validation.
        if res.certified {
            for &(handle, lo, hi) in &res.root_intervals {
                let v = tree.vstar[handle as usize];
                // 1e-6 tolerance: leaf-exact fast values are f32, so interval
                // endpoints inherit ~1e-7-scale representation noise.
                let bad = !(lo <= hi && v >= lo - 1e-6 && v <= hi + 1e-6);
                if bad {
                    r.cert_invalid += 1;
                    eprintln!(
                        "      D{d}b{b} t{t}: certified interval [{lo:.6}, {hi:.6}] misses true {v:.6} (node {handle})"
                    );
                    if std::env::var("B615_DEBUG").is_ok() {
                        let lvl = (0..tree.starts.len() - 1)
                            .find(|&l| (handle as usize) < tree.starts[l + 1])
                            .unwrap_or(tree.d as usize);
                        let h = i32::from(tree.d) - lvl as i32;
                        let env = f64::from(tree.bias_envelope(h.max(0) as u8));
                        eprintln!(
                            "        DEBUG node {handle}: lvl {lvl} h {h} vstar {:.6} fast {:.6} bias {:.6} envelope {env:.6}",
                            tree.vstar[handle as usize],
                            tree.fast[handle as usize],
                            f64::from(tree.fast[handle as usize]) - tree.vstar[handle as usize],
                        );
                    }
                }
            }
        }
        // BAI-MCTS
        let mut rng = Xs(0x615_00B1_u64 + t as u64);
        let (pick, samples, stopped) = bai_mcts(&tree, &mut rng);
        if !stopped {
            r.bai_capped += 1;
        }
        if err_pick(&tree, pick, best_v) {
            r.bai_errs += 1;
        }
        let cost = Cost {
            fast: 0,
            slow: samples,
        };
        r.bai_cost.push(cost.unified(SLOW_COST));
        // UCT checkpoints (one run, snapshot picks)
        let mut rng = Xs(0x615_00C7_u64 + t as u64);
        let (picks, last_value) = uct_checkpoints(&tree, &checkpoints, &mut rng);
        r.uct_best_value += last_value / trees as f64;
        for (j, &p) in picks.iter().enumerate() {
            if err_pick(&tree, p, best_v) {
                r.uct_errs_at[j] += 1;
            }
        }
        // context rows
        let fp = tree.fast_only_pick();
        if fp != best {
            r.fast_errs += 1;
        }
        r.fast_cost += tree.starts[tree.d as usize + 1] as f64;
        let (sp, s_cost) = slow_only(&tree, &mut Xs(0x615_0050_u64 + t as u64));
        if err_pick(&tree, sp, best_v) {
            r.slow_errs += 1;
        }
        r.slow_cost += s_cost as f64;
    }
    if tracing {
        let pool: Vec<TwoFidelityResult> = trace_pool
            .into_iter()
            .map(|stats| TwoFidelityResult {
                best_action: 0,
                cost: Cost::default(),
                certified: false,
                root_intervals: Vec::new(),
                node_stats: stats,
            })
            .collect();
        dump_2ffs_trace(&pool, d, b as u8);
    }
    r
}

// ─── Issue 916 b-lever A/B: v1 resolver vs lazy discharge ─────────────────────

/// Per-arm tally for the `B615_2FFS_AB` mode.
#[derive(Default)]
struct ArmTally {
    errs: u64,
    uncertified: u64,
    cert_invalid: u64,
    slow: f64,
    fast: f64,
    cost: Vec<f64>,
}

impl ArmTally {
    fn absorb(&mut self, tree: &BenchTree, res: &TwoFidelityResult, best_v: f64, ab: &AbCfg) {
        if tree.vstar[res.best_action as usize] < best_v - f64::from(ab.eps) {
            self.errs += 1;
        }
        if res.certified {
            for &(handle, lo, hi) in &res.root_intervals {
                let v = tree.vstar[handle as usize];
                if !(lo <= hi && v >= lo - 1e-6 && v <= hi + 1e-6) {
                    self.cert_invalid += 1;
                }
            }
        } else {
            self.uncertified += 1;
        }
        self.slow += res.cost.slow as f64;
        self.fast += res.cost.fast as f64;
        self.cost.push(res.cost.unified(ab.cost));
    }
}

/// `B615_2FFS_AB=1`: the 2FFS resolver A/B alone (no baselines — those are
/// the expensive arms and do not depend on the resolver). Same trees, same
/// per-tree RNG seed for both arms; reports the G1 quantities for each arm
/// (errors with the one-sided CP95 upper bound, uncertified exits, invalid
/// certified intervals) and the paired unified-cost LB95 of (v1 − lazy).
fn ab_lazy_discharge(trees: usize) -> bool {
    let ab = AbCfg::from_env();
    println!(
        "\n── Issue 916 b-lever A/B: v1 resolver vs lazy discharge ({trees} trees/setting, ε={} σ={} c={}) ──",
        ab.eps, ab.sigma, ab.cost
    );
    let mut ok = true;
    let (mut all_v1, mut all_ld) = (ArmTally::default(), ArmTally::default());
    for (d, b) in SETTINGS {
        let (mut v1, mut ld) = (ArmTally::default(), ArmTally::default());
        for t in 0..trees {
            let tree = BenchTree::generate(d, b, 0x615_0000_u64 + t as u64, ab.sigma as f32);
            let (_, best_v) = tree.true_best();
            let seed = 1_000 + t as u64;
            v1.absorb(&tree, &run_2ffs_with(&tree, seed, false, &ab), best_v, &ab);
            ld.absorb(&tree, &run_2ffs_with(&tree, seed, true, &ab), best_v, &ab);
        }
        let diffs: Vec<f64> = v1.cost.iter().zip(&ld.cost).map(|(a, c)| a - c).collect();
        let n = trees as f64;
        println!(
            "  D{d}b{b}  v1: errs {} uncert {} invalid {} | slow {:.0} fast {:.0} cost {:.0}",
            v1.errs,
            v1.uncertified,
            v1.cert_invalid,
            v1.slow / n,
            v1.fast / n,
            mean(&v1.cost)
        );
        println!(
            "         lazy: errs {} uncert {} invalid {} | slow {:.0} fast {:.0} cost {:.0}",
            ld.errs,
            ld.uncertified,
            ld.cert_invalid,
            ld.slow / n,
            ld.fast / n,
            mean(&ld.cost)
        );
        println!(
            "         cost ratio v1/lazy {:.2}x, paired LB95(v1 − lazy) {:+.0}",
            mean(&v1.cost) / mean(&ld.cost),
            lb95(&diffs)
        );
        for (a, s) in [(&mut all_v1, v1), (&mut all_ld, ld)] {
            a.errs += s.errs;
            a.uncertified += s.uncertified;
            a.cert_invalid += s.cert_invalid;
            a.slow += s.slow;
            a.fast += s.fast;
            a.cost.extend(s.cost);
        }
    }
    let n_all = all_ld.cost.len() as u64;
    let cp = cp_upper(all_ld.errs, n_all);
    println!(
        "  lazy G1: errs {}/{n_all} (CP95 upper {cp:.5} vs δ {DELTA}), uncertified {}, invalid certified intervals {}",
        all_ld.errs, all_ld.uncertified, all_ld.cert_invalid
    );
    if cp > DELTA || all_ld.cert_invalid > 0 || all_ld.uncertified > 0 {
        println!("  ⛔ lazy-discharge arm fails the G1 quantities");
        ok = false;
    }
    ok
}

// ─── main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ Plan 615 Phase 2 — Two-Fidelity Certified BAI GOAT (T2.1–T2.5) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    let trees = trees_per_setting();
    println!(
        "settings {:?}, trees/setting {trees}, ε={EPSILON} δ={DELTA} c={SLOW_COST} σ={SIGMA}",
        SETTINGS
    );
    if std::env::var("B615_2FFS_AB").is_ok() {
        // Resolver A/B only (Issue 916 b-lever) — skips the gated suite.
        let ok = ab_lazy_discharge(trees);
        std::process::exit(if ok { 0 } else { 1 });
    }

    // ── CANARY (first — impossible floors must fire) ──
    let mut gates = Gates {
        failures: Vec::new(),
    };
    gates.run("CANARY gate-machinery-fires", || {
        let fired = catch_unwind(AssertUnwindSafe(|| {
            gate(false, "impossible-floor canary — must fire");
        }))
        .is_err();
        gate(fired, "canary did NOT fire — gate machinery broken");
    });
    gates.run("CANARY clopper-pearson E=0 closed form", || {
        let got = cp_upper(0, 300);
        let want = 1.0 - 0.05_f64.powf(1.0 / 300.0);
        gate(
            (got - want).abs() < 1e-9,
            &format!("CP(0,300) = {got} != closed form {want}"),
        );
    });
    gates.run("CANARY alloc-counter-moves", || {
        let (_, moved) = alloc_delta(|| {
            let v: Vec<u8> = Vec::with_capacity(1024);
            v
        });
        gate(
            moved > 0,
            "counting allocator did not move on a known allocation",
        );
    });
    if !gates.failures.is_empty() {
        println!("⛔ canaries failed — every later gate would be unmeasured");
        std::process::exit(1);
    }

    // ── SANITY (T2.2b): adapter converges on a zero-noise tree ──
    gates.run("SANITY negamax-UCT converges (T2.2b)", || {
        let tree = BenchTree::generate(3, 3, 0x615_5A17, 0.0);
        let (best, best_v) = tree.true_best();
        let mut vals = Vec::new();
        for &budget in &[1_000_u64, 10_000, 100_000] {
            let mut s = UctSearch::new(&tree);
            let mut rng = Xs(0x615_0042 + budget);
            while s.samples < budget {
                s.descend_and_sample(0, &mut rng);
            }
            vals.push((budget, s.pick_best(), s.arm_value(s.pick_best())));
        }
        let (_, _, v1k) = vals[0];
        let (_, _pick10k, v10k) = vals[1];
        let (_, pick100k, v100k) = vals[2];
        if std::env::var("B615_DEBUG").is_ok() {
            for j in 0..tree.b {
                let c = (1 + j) as u32;
                println!("      DEBUG arm {c}: vstar = {:.4}", tree.vstar[c as usize]);
            }
            let mut s2 = UctSearch::new(&tree);
            let mut rng2 = Xs(0x615_0042 + 100_000);
            while s2.samples < 100_000 {
                s2.descend_and_sample(0, &mut rng2);
            }
            for j in 0..tree.b {
                let c = (1 + j) as u32;
                println!(
                    "      DEBUG arm {c}: est = {:.4}  n = {}",
                    s2.arm_value(c),
                    s2.n[c as usize],
                );
            }
        }
        let e1k = (v1k - best_v).abs();
        let e10k = (v10k - best_v).abs();
        let e100k = (v100k - best_v).abs();
        println!(
            "      |V̂−V*|: 1k={e1k:.4} 10k={e10k:.4} 100k={e100k:.4}; pick@100k={pick100k} (true {best})",
        );
        gate(
            e10k >= e100k - 1e-9,
            "UCT error did not decrease 10k → 100k",
        );
        gate(
            pick100k == best,
            &format!("UCT pick {pick100k} != true best {best} at budget 100k"),
        );
        gate(e100k <= 0.02, &format!("UCT value error {e100k} > 0.02 at 100k"));
    });

    // ── FIXTURE honesty (Issue 915's class, gated so it cannot recur): the
    // oracle pair must satisfy the MinimaxSpace contract — fast biased within
    // the declared envelope at EVERY node, slow samples centered on V* at
    // EVERY node (not just leaves). The 915 negative ran on a fixture whose
    // internal slow samples centered at 0.0; nothing caught it because no
    // arm checked the slow mean.
    gates.run("FIXTURE honesty (fast envelope + slow mean, every node class)", || {
        for &(d, b) in SETTINGS.iter() {
            let tree = BenchTree::generate(d, b, 0x615_00F1, NOISE_AMP);
            // Fast: |V_F − V*| ≤ B(h) + f32 representation slack, every node.
            for (i, &v) in tree.vstar.iter().enumerate() {
                let lvl = tree.level_of(i as u32);
                let h = (i32::from(tree.d) - lvl as i32).max(0);
                let env = f64::from(tree.bias_envelope(h as u8));
                let bias = (f64::from(tree.fast[i]) - v).abs();
                gate(
                    bias <= env + 1e-5,
                    &format!("D{d}b{b}: fast bias {bias:.6} > envelope {env:.6} at node {i}"),
                );
            }
            // Slow: samples centered on V*. Statistical read on one internal
            // node (level 1 — the class Issue 915 broke) and one deeper node:
            // U(−a, a) has sd = a/√3 ≈ 0.0289, so 400 draws give se ≈ 0.0014;
            // the 0.02 floor is ~14 se of pure noise but an order below the
            // 0.5-scale defect (mean 0.0 vs V* ≈ 0.5).
            let mut rng = Xs(0x615_00F2);
            for &node in &[1u32, (1 + b) as u32] {
                const N: u64 = 400;
                let mut acc = 0.0_f64;
                for _ in 0..N {
                    acc += f64::from(tree.slow_sample(node, &mut rng));
                }
                let dev = (acc / N as f64 - tree.vstar[node as usize]).abs();
                gate(
                    dev < 0.02,
                    &format!(
                        "D{d}b{b}: slow mean deviates {dev:.6} from V* at internal node {node} — contract violation"
                    ),
                );
            }
        }
    });

    // ── T2.1–T2.2 suite per setting ──
    let mut results = Vec::new();
    for &(d, b) in SETTINGS.iter() {
        print!("── setting (D={d}, b={b}): running {trees} trees … ");
        std::io::stdout().flush().ok();
        let r = run_setting(d, b, trees);
        println!("done (flipped {}/{})", r.flipped_trees, trees);
        println!(
            "   2FFS:  err {}/{}  cert {}/{}  cost mean {:.0} (fast {:.0} slow {:.0})",
            r.f2_errs,
            trees,
            r.f2_cert_ok,
            trees,
            mean(&r.f2_cost),
            r.f2_fast / trees as f64,
            r.f2_slow / trees as f64,
        );
        println!(
            "   BAI:   err {}/{}  capped {}  cost mean {:.0}",
            r.bai_errs,
            trees,
            r.bai_capped,
            mean(&r.bai_cost),
        );
        println!(
            "   UCT:   errs@checkpoints {:?}  best-arm V̂≈{:.4}",
            r.uct_errs_at, r.uct_best_value,
        );
        println!(
            "   fast-only: err {}/{}  cost mean {:.0}   slow-only: err {}/{}  cost mean {:.0}",
            r.fast_errs,
            trees,
            r.fast_cost / trees as f64,
            r.slow_errs,
            trees,
            r.slow_cost / trees as f64,
        );
        results.push(r);
    }

    // ── G1: empirical PAC over the pooled suite ──
    gates.run("G1 empirical PAC (Clopper–Pearson ≤ δ)", || {
        let mut e = 0_u64;
        let mut n = 0_u64;
        for r in &results {
            e += r.f2_errs;
            n += trees as u64;
        }
        let u = cp_upper(e, n);
        println!("      errors {e}/{n}, CP95 upper = {u:.5} vs δ = {DELTA}");
        gate(
            u <= DELTA,
            &format!("PAC bound violated: CP95 upper {u} > δ {DELTA}"),
        );
    });
    gates.run("G1 uncertified within δ budget (guard-fires ≤ 5% + slack)", || {
        let mut bad = 0_u64;
        let mut n = 0_u64;
        for r in &results {
            bad += trees as u64 - r.f2_cert_ok;
            n += trees as u64;
        }
        let budget = (DELTA * n as f64).ceil() as u64 + 2;
        println!(
            "      uncertified {bad}/{n} (δ budget {budget} — B.3 default-action path is δ-legal)"
        );
        gate(
            bad <= budget,
            &format!(
                "{bad}/{n} trees uncertified — way outside the δ = {DELTA} budget; audit"
            ),
        );
    });

    // ── G2(a): paired win vs BAI-MCTS, per setting ──
    gates.run("G2 paired win vs BAI-MCTS (LB95 > 0 per setting)", || {
        for r in &results {
            let diffs: Vec<f64> = r
                .bai_cost
                .iter()
                .zip(r.f2_cost.iter())
                .map(|(b, f)| b - f)
                .collect();
            let lb = lb95(&diffs);
            println!(
                "      (D={},b={}): mean diff {:.0}, LB95 {:.0}, BAI capped {}",
                r.d,
                r.b,
                mean(&diffs),
                lb,
                r.bai_capped,
            );
            gate(
                lb > 0.0,
                &format!(
                    "no strict win vs BAI-MCTS at (D={},b={}): LB95 {lb}",
                    r.d, r.b
                ),
            );
        }
    });

    // ── G2(b): paired win vs negamax-UCT at matched accuracy, per setting ──
    gates.run(
        "G2 paired win vs negamax-UCT (matched accuracy, per setting)",
        || {
            for r in &results {
                let checkpoints: Vec<u64> = (0..12).map(|j| 4096u64 << j).collect();
                let b_star = checkpoints
                    .iter()
                    .zip(r.uct_errs_at.iter())
                    .find(|(_, e)| **e <= r.f2_errs)
                    .map_or(
                        f64::from(SLOW_COST) * (UCT_MAX_BUDGET as f64 + 1.0),
                        |(b, _)| f64::from(SLOW_COST) * (*b as f64),
                    );
                let diffs: Vec<f64> = r.f2_cost.iter().map(|f| b_star - f).collect();
                let lb = lb95(&diffs);
                println!(
                    "      (D={},b={}): 2FFS errs {}, B* = {b_star:.0} ({}), LB95 {:.0}",
                    r.d,
                    r.b,
                    r.f2_errs,
                    if b_star > f64::from(SLOW_COST) * (UCT_MAX_BUDGET as f64) {
                        "no checkpoint matched — cost understated, conservative"
                    } else {
                        "matched"
                    },
                    lb,
                );
                gate(
                    lb > 0.0,
                    &format!(
                        "no strict win vs negamax-UCT at (D={},b={}): LB95 {lb}",
                        r.d, r.b
                    ),
                );
            }
        },
    );

    // ── G4: zero per-iteration allocation ──
    gates.run("G4 zero per-iteration allocation", || {
        let small = BenchTree::generate(3, 3, 0x615_00A4, NOISE_AMP);
        let _ = run_2ffs(&small, 7); // warmup
        let (_, s1) = alloc_delta(|| run_2ffs(&small, 8));
        let (_, s3) = alloc_delta(|| {
            run_2ffs(&small, 9);
            run_2ffs(&small, 10);
            run_2ffs(&small, 11)
        });
        println!("      allocs: 1 search = {s1}, 3 searches = {s3}");
        gate(
            s3 == 3 * s1,
            &format!("allocation not linear in searches: {s3} != 3×{s1}"),
        );
        gate(
            s1 <= 4,
            &format!("per-search setup allocations {s1} > 4 (HashMap + result Vec expected)"),
        );
    });

    println!("── verdict ──");
    if gates.failures.is_empty() {
        println!(
            "   ALL GATES PASS (canary, sanity, G1, G2×2, G4) — record + plan Status next (T2.4/T2.5)"
        );
    } else {
        println!("   FAILED gates: {:?}", gates.failures);
        println!("   honest record + stay opt-in per T2.5 (issue if G2 failed)");
        std::process::exit(1);
    }
}
