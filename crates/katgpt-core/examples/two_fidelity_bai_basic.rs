//! Plan 615 Phase 3 T3.1 — `two_fidelity_bai` basic example.
//!
//! Builds a small synthetic minimax tree with TWO oracles per node — a fast
//! one (deterministic, biased within a declared envelope) and a slow one
//! (stochastic, unbiased) — runs the certified two-fidelity search, and
//! prints the cost ledger and the first-class certificate: the certified
//! ε-optimal root action plus every root child's certified interval.
//!
//! Run with:
//! ```sh
//! cargo run --example two_fidelity_bai_basic --features two_fidelity_bai --release
//! ```

use arrayvec::ArrayVec;
use katgpt_core::two_fidelity_bai::{
    MAX_CHILDREN, MinimaxSpace, NodeKind, RngCore, SearchConfig, two_fidelity_search,
};

// ── Synthetic minimax tree ────────────────────────────────────────────────
//
// Root (Max, handle 0) with 3 Min children; each Min child has 3 Max
// grandchildren; leaves carry true values. Node ids are breadth-first:
// root = 0, its children 1..=3, grandchildren 4..=12 (child c owns
// 3*(c-1)+4 .. 3*(c-1)+6). True leaf values are chosen so child 2's
// subtree is worth ~0.8 while children 1 and 3 sit near 0.3/0.4 — a root
// gap the ε-stop can certify cheaply.
struct DemoTree {
    /// Slow-oracle mean = the true minimax value at EVERY node (the
    /// `MinimaxSpace` contract; leaves fold up by exact minimax).
    vstar: Vec<f64>,
    /// Fast-oracle values: vstar + deterministic per-node bias, |bias| ≤
    /// B(remaining_depth) = BBAR·(1 − 2^−h) (B(0) = 0 ⇒ leaves exact).
    fast: Vec<f32>,
}

const BBAR: f32 = 0.12;
const NOISE: f32 = 0.05;

impl DemoTree {
    fn build() -> Self {
        // True leaf values (node ids 4..=12); each root child is the MIN
        // over its three leaves: child 1 → 0.30, child 2 → 0.62 (the
        // winner), child 3 → 0.40 — a gap the ε-stop can certify cheaply.
        let leaf_true: [f64; 9] = [
            0.30, 0.95, 0.55, // child 1
            0.80, 0.99, 0.62, // child 2 (the winner)
            0.40, 0.90, 0.70, // child 3
        ];
        let mut vstar = vec![0.0f64; 13];
        vstar[4..13].copy_from_slice(&leaf_true);
        // Root children (Min): min over their three leaves.
        for c in 0..3usize {
            let base = 4 + 3 * c;
            vstar[1 + c] = vstar[base].min(vstar[base + 1]).min(vstar[base + 2]);
        }
        vstar[0] = vstar[1].max(vstar[2]).max(vstar[3]);

        // Deterministic fast bias inside the envelope (a fixed xorshift so
        // the example prints the same tree every run).
        let mut xs = Xs(0x615_BA51);
        let mut fast = vec![0.0f32; 13];
        for (i, &v) in vstar.iter().enumerate() {
            let level = match i {
                0 => 0,
                1..=3 => 1,
                4..=12 => 2,
                _ => unreachable!(),
            };
            // remaining_depth: leaves (level 2) → h = 0 ⇒ envelope 0 ⇒ exact.
            let h = 2 - level;
            let envelope = BBAR * (1.0 - 2.0f32.powi(-h));
            let u = xs.f64() as f32 - 0.5; // ±0.5 scaled
            fast[i] = (v + f64::from(envelope) * f64::from(u)) as f32;
        }
        Self { vstar, fast }
    }

    fn kind_of(&self, node: u32) -> NodeKind {
        match node {
            0 => NodeKind::Max,
            1..=3 => NodeKind::Min,
            _ => NodeKind::Max,
        }
    }

    fn level_of(&self, node: u32) -> u8 {
        match node {
            0 => 0,
            1..=3 => 1,
            _ => 2,
        }
    }

    fn is_leaf(&self, node: u32) -> bool {
        self.level_of(node) == 2
    }
}

impl MinimaxSpace for DemoTree {
    fn kind(&self, node: u32) -> NodeKind {
        self.kind_of(node)
    }

    fn remaining_depth(&self, node: u32) -> u8 {
        2 - self.level_of(node)
    }

    fn children_into(&self, node: u32, out: &mut ArrayVec<u32, MAX_CHILDREN>) {
        if self.is_leaf(node) {
            return;
        }
        match node {
            0 => {
                for c in 1..=3 {
                    out.push(c);
                }
            }
            p @ 1..=3 => {
                let base = 4 + 3 * (p - 1);
                for g in base..base + 3 {
                    out.push(g);
                }
            }
            _ => {}
        }
    }

    fn fast_value(&self, node: u32) -> f32 {
        self.fast[node as usize]
    }

    fn bias_envelope(&self, remaining_depth: u8) -> f32 {
        BBAR * (1.0 - 2.0f32.powi(-i32::from(remaining_depth)))
    }

    fn slow_sample(&self, node: u32, rng: &mut impl RngCore) -> f32 {
        // Contract: i.i.d. mean V*(v) — vstar at EVERY node — uniform noise
        // ±NOISE (sub-Gaussian parameter NOISE/√3 ≤ declared σ).
        (self.vstar[node as usize] + f64::from((rng.f64() - 0.5) as f32 * 2.0 * NOISE)) as f32
    }
}

/// Deterministic xorshift64* — no global RNG (the global_rng_gate bans
/// unseeded free-function draws; examples seed their own).
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

fn main() {
    let tree = DemoTree::build();
    let config = SearchConfig {
        epsilon: 0.02,
        delta: 0.05,
        slow_cost: 4.0,
        node_cap: 65_536,
        sigma: 0.05,
        trace_nodes: false,
        // Recommended (Issue 916(b), Bench 615 addendum): same PAC guarantee,
        // 196–1117× lower cost than the v1 resolver on the bench fixture.
        lazy_discharge: true,
    };

    println!("true root value: {:.4}", tree.vstar[0]);
    for c in 1..=3u32 {
        println!("  child {c} true value: {:.4}", tree.vstar[c as usize]);
    }

    let result = two_fidelity_search(&tree, 0, &config, &mut Xs(0x615_00B1));

    println!("\ncertified: {}", result.certified);
    println!("best action: child {}", result.best_action);
    println!(
        "cost: {} fast + {} slow (unified {:.0} at c = {})",
        result.cost.fast,
        result.cost.slow,
        result.cost.unified(config.slow_cost),
        config.slow_cost,
    );
    println!("certificate (root children's certified intervals):");
    for &(handle, lo, hi) in &result.root_intervals {
        println!(
            "  child {handle}: [{lo:.4}, {hi:.4}]  (true {:.4})",
            tree.vstar[handle as usize]
        );
    }
    assert!(result.certified, "honest oracles must certify here");
    assert_eq!(result.best_action, 2, "child 2 owns the 0.62 subtree");
}
