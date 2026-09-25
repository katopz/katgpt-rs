//! Issue 892 T3 — chance_puct GOAT gate (G2 latency + G4 alloc-free).
//!
//! G1 (Q-sign / chance weighting / top_k / terminal / determinism) lives in
//! the module's unit tests (`cargo test -p katgpt-core --features
//! chance_puct --lib chance_puct`); G3 (quality vs depth-2 exhaustive) is
//! the Tetris consumer `examples/tetris_08_puct_goat.rs`.
//!
//! G2  latency: µs per decision at budgets 100 / 400 / 1600 on a heap-free
//!     synthetic game (16 actions, 4 chance outcomes, depth ≤ 12, ~1 in 29
//!     afterstates a loss) — the SEARCH overhead, with the game's own cost
//!     near zero. Absolute budget ⇒ `best_of_us` (the minimum is the
//!     load-invariant quantity); `black_box` on arguments AND result (fat
//!     LTO deletes a dead-result timed loop — Issue 855). Scaling bar: the
//!     per-simulation cost at b1600 stays within 3× of b100 (a tree walk is
//!     O(depth), not O(budget)).
//! G4  alloc-free: a warm search (same root, same seed, same budget —
//!     identical tree, so arena/scratch capacity already fits) allocates 0.
//!
//! Harness = false, exit 1 on failure. Run:
//!   cargo test -p katgpt-core --features chance_puct --release \
//!     --test bench_892_chance_puct_goat -- --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[path = "../../../tests/common/ab_timing.rs"]
mod ab_timing;

use ab_timing::best_of_us;
use katgpt_core::chance_puct::{ChanceGame, ChancePuct, ChancePuctConfig};

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

// ── The synthetic heap-free game ──────────────────────────────────────────

#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

#[inline]
fn unit(x: u64) -> f32 {
    (x >> 40) as f32 / (1u64 << 24) as f32
}

#[derive(Clone, Copy)]
struct Synth {
    h: u64,
    depth: u8,
    after: bool,
}

const N_ACTIONS: u64 = 16;
const N_OUTCOMES: u64 = 4;
const MAX_DEPTH: u8 = 12;

impl ChanceGame for Synth {
    type Action = u8;
    type Outcome = u8;
    fn is_terminal(&self) -> bool {
        self.after && self.h.is_multiple_of(29)
    }
    fn value(&self) -> f32 {
        unit(mix(self.h ^ 0xabcd)) * 4.0 - 2.0 + self.depth as f32 * 0.1
    }
    fn actions(&self, out: &mut Vec<(u8, f32)>) {
        if self.depth >= MAX_DEPTH {
            return;
        }
        for a in 0..N_ACTIONS {
            out.push((a as u8, unit(mix(self.h ^ (a + 1))) * 4.0 - 2.0));
        }
    }
    fn apply(&self, a: u8) -> Self {
        Self {
            h: mix(self.h.wrapping_add(a as u64 + 1)),
            depth: self.depth,
            after: true,
        }
    }
    fn outcomes(&self, out: &mut Vec<(u8, f32)>) {
        for o in 0..N_OUTCOMES {
            out.push((o as u8, 0.25 + unit(mix(self.h ^ (o + 100)))));
        }
    }
    fn resolve(&self, o: u8) -> Self {
        Self {
            h: mix(self.h ^ (o as u64 + 7)),
            depth: self.depth + 1,
            after: false,
        }
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
        .find(|l| l.starts_with("Pages free"))
        .unwrap_or("n/a")
        .to_owned();
    println!("box: {free} (16 KiB pages)");
    println!(
        "box: power {}",
        run("pmset", &["-g", "batt"]).replace('\n', " | ")
    );
}

fn g2_latency() -> bool {
    let root = Synth {
        h: 0x1234_5678,
        depth: 0,
        after: false,
    };
    let mut per_sim = Vec::new();
    let mut ok = true;
    for budget in [100u32, 400, 1600] {
        let cfg = ChancePuctConfig {
            budget,
            ..Default::default()
        };
        let mut s = ChancePuct::new(cfg);
        let iters = if budget >= 1600 { 20 } else { 50 };
        let us = best_of_us(3, iters, || {
            let mut rng = fastrand::Rng::with_seed(7);
            let t = Instant::now();
            let p = s.search(black_box(&root), black_box(&mut rng));
            black_box(p);
            t.elapsed()
        });
        let per = us * 1000.0 / budget as f64;
        println!(
            "G2 b{budget:<5} {us:>9.1} µs/decision  {per:>7.1} ns/sim  tree {} nodes",
            s.tree_size()
        );
        per_sim.push(per);
    }
    let ratio = per_sim[2] / per_sim[0];
    let pass = ratio <= 3.0;
    println!(
        "G2 scaling: ns/sim b1600 / b100 = {ratio:.2} (bar ≤ 3.0) {}",
        if pass { "PASS" } else { "FAIL" }
    );
    ok &= pass;
    ok
}

fn g4_alloc_free() -> bool {
    let root = Synth {
        h: 0xdead_beef,
        depth: 0,
        after: false,
    };
    let mut s = ChancePuct::new(ChancePuctConfig::default());
    let warm = s.search(&root, &mut fastrand::Rng::with_seed(11));
    let before = ALLOCS.load(Ordering::Relaxed);
    let again = s.search(black_box(&root), &mut fastrand::Rng::with_seed(11));
    let n = ALLOCS.load(Ordering::Relaxed) - before;
    let same = warm == again;
    println!(
        "G4 warm search allocs = {n} (bar 0), repeat pick identical = {same} {}",
        if n == 0 && same { "PASS" } else { "FAIL" }
    );
    n == 0 && same
}

fn main() {
    println!("bench_892: chance_puct GOAT gate — G2 latency + G4 alloc-free");
    box_state();
    let mut ok = g2_latency();
    ok &= g4_alloc_free();
    if !ok {
        println!("bench_892: GATE FAILED");
        std::process::exit(1);
    }
    println!("bench_892: ALL GATES PASSED");
}
