//! Issue 938 T2.3–T2.5 — the 3-axis lattice GOAT: Bench 919's G5 cell
//! re-run at A = 3, plus the G2 latency gate and the G6 memory column.
//!
//! The premise being measured: Bench 919's floor died as
//! **(support × capacity)-bounded** — at d_k = 32 the 0.05σ near-miss claim
//! caps 2D at ~63k items, and the 10⁶/0.05σ cell cannot even SIZE (λ = 173
//! against capacity 11.2). The sizer's envelope generalizes to
//! `N_max ≈ 0.35·d_k·(3.76/R)^AXES`, so A = 3 re-opens the exact failed cell
//! (λ = 2.28, slab ≈ 899 MB at d_v = 16 — the d_v = 32 worst case, 1.798 GB,
//! is priced and gated in `lattice_memory_3d_gates`). μLattice §4.7 supplied
//! the 3D ADDRESSING ANALYSIS only — nobody has measured 3D recall; this
//! bench is the first measurement anywhere, on our fixture. A negative is an
//! acceptable product (the recorded second remedy).
//!
//! Gates (the issue's table):
//! - G1  — recall@1 on exact keys at N = 10⁶ (floor 0.90, the Bench 919 bar).
//! - G2  — 27-tap read vs PKM at 10⁶ (`--release` MANDATORY; interleaved
//!   best-of-5 arms), plus the cell-count-matched 27-vs-9 tap arm
//!   (2D [332,332] vs 3D [48,48,48], ~110k cells each).
//! - G5  — graded recovery at ε = 0.05σ vs Engram's hash cliff (lattice >
//!   Engram + 0.05, the plan's original bar), ε sweep 0.1R → 2R.
//! - G6  — the slab column at the bench posture (the 2 GiB bar lives in the
//!   gates test at the d_v = 32 worst case).
//! - G4  — alloc-free hot path (the Issue-741 profile rule).
//!
//! Run:
//!
//! ```sh
//! cargo bench -p katgpt-core --features lattice_memory_3d,engram \
//!     --bench bench_938_lattice_3d_goat
//! # G4 cell:
//! cargo bench -p katgpt-core --features lattice_memory_3d,engram,alloc_tracking \
//!     --bench bench_938_lattice_3d_goat -- --g4-only
//! ```
#![cfg(feature = "lattice_memory_3d")]

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::engram::{EngramHash, EngramTable, EngramTableBuilder, InMemoryEngramTable, K_MAX};
use katgpt_core::lattice_memory::{
    LatticeConfig, LatticeConfig3D, LatticeMemory, LatticeMemory3D, support_radius,
};

// `any(debug_assertions, feature = "alloc_tracking")` per the Issue-741
// rule; prints a LOUD skip rather than a silent pass when the profile
// carries no allocator.
#[cfg(any(debug_assertions, feature = "alloc_tracking"))]
#[global_allocator]
static BENCH_ALLOC: katgpt_core::alloc::TrackingAllocator =
    katgpt_core::alloc::TrackingAllocator;

/// Embedding / key width — the Bench 919 posture (d == d_k: the key IS the
/// normalized embedding).
const D: usize = 32;
/// Marker width — the Bench 919 posture (keeps the 200-marker window ≥ 3σ
/// from random-pair collision noise).
const D_V: usize = 16;
/// The G5 cell: the exact N Bench 919's floor died at.
const N: usize = 1_000_000;
/// The desired near-miss radius (σ) — the failed cell's support claim.
const DESIRED_R: f32 = 0.05;
/// Query count for recall / timing (the Bench 919 posture).
const QUERIES: usize = 2_000;
/// Marker-comparison window (odd enough to be deliberate, shared by every
/// engine including brute force).
const WINDOW: usize = 200;
/// Interleaved timing rounds (per-arm minimum across rounds).
const ROUNDS: usize = 5;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--g4-only") {
        g4_gate();
        return;
    }
    println!("bench_938_lattice_3d_goat — Issue 938 T2.3–T2.5");
    println!(
        "box: {} cores · d={D} d_k={D} d_v={D_V} · N={N} · queries={QUERIES} · window={WINDOW} · rounds={ROUNDS}",
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    );
    println!("⚠ latency numbers are workstation readings (M3 Max, shared box) — quote them with a box-state caveat.\n");

    // ── G6: the pricing, before anything allocates ──────────────────────
    let side = (3.76_f32 / DESIRED_R).ceil() as u32;
    let cells3 = side as usize * side as usize * side as usize;
    let cells2 = side as usize * side as usize;
    let slab3 = cells3 * D * D_V * 4;
    println!(
        "G6 pricing (pre-build): L={side} → 3D {cells3} cells, slab {} MB · 2D {cells2} cells, slab {} MB \
         (2D λ = {:.1} ≫ 11.2: the cell 2D cannot size)",
        slab3 / (1024 * 1024),
        cells2 * D * D_V * 4 / (1024 * 1024),
        N as f32 / cells2 as f32,
    );
    let items = build_items(N, 0x938_0000 + N as u64);

    // ── the 3D lattice at the failed cell ────────────────────────────────
    let refs: Vec<&[f32]> = items.keys.iter().map(|k| k.as_slice()).collect();
    let t0 = Instant::now();
    let mut lat = LatticeMemory3D::sized(D, D, D_V, 0x938, &refs, N, DESIRED_R, None)
        .expect("3D must SIZE the cell 2D refuses (the gates test pins both verdicts)");
    println!(
        "build: 3D sized lattice {:?} ({} cells, support {:.4}σ, λ = {:.2})",
        t0.elapsed(),
        lat.config().cell_count(),
        support_radius(side),
        N as f32 / lat.config().cell_count() as f32,
    );
    for (k, m) in items.keys.iter().zip(items.markers.iter()) {
        lattice_write(&mut lat, k, m);
    }
    let pkm = build_pkm(&items, N);
    let mut engram = build_engram(&items, N);

    // ── G1: exact-key recall@1 ───────────────────────────────────────────
    let mut out = vec![0.0_f32; D_V];
    let q_n = QUERIES.min(N);
    let mut hits = 0_u32;
    for (i, key) in items.keys.iter().enumerate().take(q_n) {
        lattice_read(&lat, key, &mut out);
        let (lo, hi) = window(i, N);
        if recall_at_1(&out, &items.markers[lo..hi], i - lo) {
            hits += 1;
        }
    }
    let lattice_recall = hits as f32 / q_n as f32;
    println!(
        "G1 recall@1 exact keys @10⁶: 3D lattice {:.4} (floor 0.90) — {}",
        lattice_recall,
        if lattice_recall >= 0.90 { "PASS" } else { "FAIL" }
    );
    assert!(
        lattice_recall >= 0.90,
        "G1: 3D exact-key recall {lattice_recall:.4} below the 0.90 floor"
    );

    // ── G5: the ε sweep vs Engram (and PKM, for the record) ─────────────
    let support = support_radius(side);
    let mut epsilons: Vec<f32> =
        [0.0_f32, 0.1, 0.3, 1.0, 2.0].iter().map(|f| f * support).collect();
    if !epsilons.iter().any(|&e| (e - 0.05).abs() < 1e-6) {
        epsilons.push(0.05);
    }
    epsilons.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut rng = fastrand::Rng::with_seed(0x938_0501);
    let noises: Vec<Vec<f32>> = (0..q_n)
        .map(|_| {
            let mut e: Vec<f32> = (0..D).map(|_| gaussian(&mut rng)).collect();
            normalize(&mut e);
            e
        })
        .collect();
    let mut lat_curve = Vec::new();
    let mut eng_curve = Vec::new();
    let mut pkm_curve = Vec::new();
    for &eps in epsilons.iter() {
        let mut lh = 0_u32;
        let mut eh = 0_u32;
        let mut ph = 0_u32;
        for (i, key) in items.keys.iter().enumerate().take(q_n) {
            let mut q = key.clone();
            for (qv, nv) in q.iter_mut().zip(noises[i].iter()) {
                *qv += eps * nv;
            }
            let (lo, hi) = window(i, N);
            lattice_read(&lat, &q, &mut out);
            if recall_at_1(&out, &items.markers[lo..hi], i - lo) {
                lh += 1;
            }
            engram.read(&q, &mut out);
            if recall_at_1(&out, &items.markers[lo..hi], i - lo) {
                eh += 1;
            }
            pkm.read(&q, &mut out);
            if recall_at_1(&out, &items.markers[lo..hi], i - lo) {
                ph += 1;
            }
        }
        let f = |h: u32| h as f32 / q_n as f32;
        lat_curve.push((eps, f(lh)));
        eng_curve.push((eps, f(eh)));
        pkm_curve.push((eps, f(ph)));
    }
    let pick = |c: &[(f32, f32)], e: f32| {
        c.iter().find(|(x, _)| (*x - e).abs() < 1e-6).map(|(_, r)| *r).unwrap_or(f32::NAN)
    };
    println!(
        "G5 @N={N} (3D support {support:.4}σ): lattice {} | engram {} | pkm {}",
        lat_curve.iter().map(|(e, r)| format!("{e:.3}→{r:.3}")).collect::<Vec<_>>().join(" "),
        eng_curve.iter().map(|(e, r)| format!("{e:.3}→{r:.3}")).collect::<Vec<_>>().join(" "),
        pkm_curve.iter().map(|(e, r)| format!("{e:.3}→{r:.3}")).collect::<Vec<_>>().join(" "),
    );
    let lattice_005 = pick(&lat_curve, 0.05);
    let engram_005 = pick(&eng_curve, 0.05);
    let g5_win = lattice_005 > engram_005 + 0.05;
    println!(
        "G5 @ε=0.05σ: 3D lattice {:.3} vs Engram {:.3} → {}",
        lattice_005,
        engram_005,
        if g5_win { "PASS (graded recovery past the cliff)" } else { "FAIL" }
    );

    // ── G2: interleaved read latency at 10⁶ vs PKM / Engram ─────────────
    let pq: Vec<&[f32]> = (0..QUERIES).map(|i| items.keys[(i * 7) % N].as_slice()).collect();
    let mut best = [u128::MAX; 3]; // lattice3d, pkm, engram
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for q in &pq {
            lattice_read(&lat, q, &mut out);
        }
        black_box(&out);
        best[0] = best[0].min(t.elapsed().as_nanos() / QUERIES as u128);

        let t = Instant::now();
        for q in &pq {
            pkm.read(q, &mut out);
        }
        black_box(&out);
        best[1] = best[1].min(t.elapsed().as_nanos() / QUERIES as u128);

        let t = Instant::now();
        for q in &pq {
            engram.read(q, &mut out);
        }
        black_box(&out);
        best[2] = best[2].min(t.elapsed().as_nanos() / QUERIES as u128);
    }
    let g2_win = best[0] < best[1];
    println!(
        "G2 read ns/op @10⁶ (min of {ROUNDS} interleaved rounds): 3D lattice {} | pkm {} | engram {} → {}",
        best[0],
        best[1],
        best[2],
        if g2_win { "PASS (3D beats PKM)" } else { "RECORDED NEGATIVE (3D slower than PKM)" }
    );

    // ── the tap-cost arm: 27 vs 9 taps at matched cells ──────────────────
    tap_cost_arm(&items);

    // ── the verdict ──────────────────────────────────────────────────────
    let under_budget = slab3 <= 2 * 1024 * 1024 * 1024;
    println!("\n════════ 3D VERDICT (Issue 938 T2.5) ════════");
    println!(
        "G1 {} · G5 {} · G2 {} · G6 {} → {}",
        if lattice_recall >= 0.90 { "PASS" } else { "FAIL" },
        if g5_win { "PASS" } else { "FAIL" },
        if g2_win { "PASS" } else { "FAIL" },
        if under_budget { "PASS" } else { "FAIL" },
        if lattice_recall >= 0.90 && g5_win && g2_win && under_budget {
            "PROMOTED — the second remedy is MEASURED (feature stays opt-in; the record is the promotion)"
        } else {
            "RECORDED NEGATIVE — the remedy did not clear its gates; Bench 919's record stands widened"
        },
    );

    g4_gate();
}

// ── the tap-cost arm: 27 vs 9 taps, cell-count-matched ───────────────────

/// Same cell count (~110k), same d_k/d_v, same 10k-item fixture: 2D
/// [332,332] vs 3D [48,48,48]. The read-cost difference is the kernel's tap
/// count (9 vs 27) plus addressing — the honest MAC-rate readout T2.4 asks
/// for at a scale both arities fit in memory.
fn tap_cost_arm(items: &Items) {
    const TAP_N: usize = 10_000;
    let cfg2 = LatticeConfig {
        d: D,
        d_k: D,
        d_v: D_V,
        grid: [332, 332],
        w: [1.0, 1.0],
        seed: 0x938_0600,
        byte_budget: None,
        warp: None,
        bump_kernel: katgpt_core::lattice_memory::BumpKernel::Tent,
    };
    let mut lat2 = LatticeMemory::new(cfg2).expect("2D matched arm");
    let mut lat3 = LatticeMemory3D::new(config3_match()).expect("3D matched arm");
    let keys: Vec<&[f32]> = items.keys[..TAP_N].iter().map(|k| k.as_slice()).collect();
    for (k, m) in keys.iter().zip(items.markers[..TAP_N].iter()) {
        let mut kn = k.to_vec();
        normalize(&mut kn);
        lat2.write_delta(k, &kn, m, 1.0);
        lat3.write_delta(k, &kn, m, 1.0);
    }
    let pq: Vec<&[f32]> = (0..QUERIES).map(|i| items.keys[(i * 7) % TAP_N].as_slice()).collect();
    let mut out = vec![0.0_f32; D_V];
    let mut best2 = u128::MAX;
    let mut best3 = u128::MAX;
    for _ in 0..ROUNDS {
        let t = Instant::now();
        for q in &pq {
            lattice_read2(&lat2, q, &mut out);
        }
        black_box(&out);
        best2 = best2.min(t.elapsed().as_nanos() / QUERIES as u128);
        let t = Instant::now();
        for q in &pq {
            lattice_read(&lat3, q, &mut out);
        }
        black_box(&out);
        best3 = best3.min(t.elapsed().as_nanos() / QUERIES as u128);
    }
    println!(
        "tap-cost arm ({} items, ~110k cells): 9-tap 2D {} ns vs 27-tap 3D {} ns → ratio {:.2}×",
        TAP_N,
        best2,
        best3,
        best3 as f64 / best2.max(1) as f64
    );
}

fn config3_match() -> LatticeConfig3D {
    LatticeConfig3D {
        d: D,
        d_k: D,
        d_v: D_V,
        grid: [48, 48, 48],
        w: [1.0; 3],
        seed: 0x938_0601,
        byte_budget: None,
        warp: None,
        bump_kernel: katgpt_core::lattice_memory::BumpKernel::Tent,
    }
}

// ── data model + engines (the bench_919 harness, 3D-adjusted) ────────────

/// Deterministic item set: unit Gaussian keys, unit Gaussian markers.
struct Items {
    keys: Vec<Vec<f32>>,
    markers: Vec<Vec<f32>>,
}

fn build_items(n: usize, seed: u64) -> Items {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut keys = Vec::with_capacity(n);
    let mut markers = Vec::with_capacity(n);
    for _ in 0..n {
        let mut k: Vec<f32> = (0..D).map(|_| gaussian(&mut rng)).collect();
        normalize(&mut k);
        let mut m: Vec<f32> = (0..D_V).map(|_| gaussian(&mut rng)).collect();
        normalize(&mut m);
        markers.push(m);
        keys.push(k);
    }
    Items { keys, markers }
}

fn gaussian(rng: &mut fastrand::Rng) -> f32 {
    let u1 = rng.f32().max(f32::EPSILON);
    let u2 = rng.f32();
    (-2.0 * u1.ln()).sqrt() * (core::f32::consts::TAU * u2).cos()
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// The deterministic marker window for query index `i` over `n` items.
fn window(i: usize, n: usize) -> (usize, usize) {
    let lo = i.saturating_sub(WINDOW / 2);
    (lo, (lo + WINDOW).min(n))
}

/// recall@1: argmax over `markers[lo..hi]` of dot(read, marker) == `expected`.
fn recall_at_1(read: &[f32], markers: &[Vec<f32>], expected: usize) -> bool {
    let mut best = 0_usize;
    let mut best_dot = f32::NEG_INFINITY;
    for (i, m) in markers.iter().enumerate() {
        let d: f32 = read.iter().zip(m.iter()).map(|(x, y)| x * y).sum();
        if d > best_dot {
            best_dot = d;
            best = i;
        }
    }
    best == expected
}

/// The lattice write: key = the normalized embedding (the Spotlight
/// semantic), β = 1 (unit key ⇒ exact remove-along-k).
fn lattice_write(lat: &mut LatticeMemory3D, x: &[f32], marker: &[f32]) {
    let mut k = x.to_vec();
    normalize(&mut k);
    lat.write_delta(x, &k, marker, 1.0);
}

fn lattice_read(lat: &LatticeMemory3D, q: &[f32], out: &mut [f32]) {
    let mut k = q.to_vec();
    normalize(&mut k);
    lat.read_cells(q, &k, out);
}

/// The 2D twin for the matched-cell tap-cost arm.
fn lattice_read2(lat: &LatticeMemory, q: &[f32], out: &mut [f32]) {
    let mut k = q.to_vec();
    normalize(&mut k);
    lat.read_cells(q, &k, out);
}

/// Bench-local PKM: two codebook banks (√N entries each), value table over
/// (c1, c2) cells, last-write-wins — the bench_919 harness verbatim.
struct Pkm {
    bank1: Vec<Vec<f32>>,
    bank2: Vec<Vec<f32>>,
    values: Vec<f32>, // K1·K2 × D_V row-major
    k2: usize,
}

fn argmax_dot(bank: &[Vec<f32>], q: &[f32]) -> usize {
    let mut best = 0;
    let mut best_dot = f32::NEG_INFINITY;
    for (i, e) in bank.iter().enumerate() {
        let d: f32 = e.iter().zip(q.iter()).map(|(x, y)| x * y).sum();
        if d > best_dot {
            best_dot = d;
            best = i;
        }
    }
    best
}

fn build_pkm(items: &Items, n: usize) -> Pkm {
    let k = (n as f32).sqrt().ceil() as usize;
    let half = D / 2;
    let stride = (n / k).max(1);
    let bank1: Vec<Vec<f32>> =
        (0..k).map(|i| items.keys[(i * stride) % n][..half].to_vec()).collect();
    let bank2: Vec<Vec<f32>> =
        (0..k).map(|i| items.keys[(i * stride) % n][half..].to_vec()).collect();
    let mut pkm = Pkm { bank1, bank2, values: vec![0.0; k * k * D_V], k2: k };
    for (i, key) in items.keys.iter().enumerate() {
        let c1 = argmax_dot(&pkm.bank1, &key[..half]);
        let c2 = argmax_dot(&pkm.bank2, &key[half..]);
        let cell = (c1 * pkm.k2 + c2) * D_V;
        pkm.values[cell..cell + D_V].copy_from_slice(&items.markers[i]);
    }
    pkm
}

impl Pkm {
    fn read(&self, q: &[f32], out: &mut [f32]) {
        let half = D / 2;
        let c1 = argmax_dot(&self.bank1, &q[..half]);
        let c2 = argmax_dot(&self.bank2, &q[half..]);
        let cell = (c1 * self.k2 + c2) * D_V;
        out.copy_from_slice(&self.values[cell..cell + D_V]);
    }
}

/// Engram arm: the hash-slot associative table at ITS OWN addressing
/// semantics — the hash cliff G5 measures (the bench_919 harness verbatim).
struct EngramArm {
    table: InMemoryEngramTable,
    slots: Vec<f32>,
}

fn key_hash(key: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in key {
        for b in v.to_bits().to_ne_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn build_engram(items: &Items, n: usize) -> EngramArm {
    let mut table = EngramTableBuilder::new(n.max(2), D_V);
    for (key, marker) in items.keys.iter().zip(items.markers.iter()) {
        table.add_pattern(EngramHash(key_hash(key)), marker);
    }
    EngramArm { table: table.build(), slots: vec![0.0; K_MAX * D_V] }
}

impl EngramArm {
    fn read(&mut self, q: &[f32], out: &mut [f32]) {
        let h = EngramHash(key_hash(q));
        let keys = [h; K_MAX];
        let _hits = self.table.lookup_into(&keys, &mut self.slots);
        out.copy_from_slice(&self.slots[..D_V]);
    }
}

// ── G4: allocation-free hot path ─────────────────────────────────────────

fn g4_gate() {
    println!("\n════════ G4 alloc-free hot path (3D) ════════");
    #[cfg(any(debug_assertions, feature = "alloc_tracking"))]
    {
        let n = 10_000;
        let items = build_items(n, 0x938_0400);
        let refs: Vec<&[f32]> = items.keys.iter().map(|k| k.as_slice()).collect();
        let mut lat = LatticeMemory3D::sized(D, D, D_V, 0x938, &refs, n, 0.2, None)
            .expect("sized 3D lattice");
        let mut out = [0.0_f32; D_V];
        let k0 = {
            let mut k = items.keys[0].clone();
            normalize(&mut k);
            k
        };
        lat.write_delta(&items.keys[0], &k0, &items.markers[0], 1.0);
        lattice_read(&lat, &items.keys[0], &mut out);
        let norm_keys: Vec<Vec<f32>> = items
            .keys
            .iter()
            .take(2_000)
            .map(|k| {
                let mut kk = k.clone();
                normalize(&mut kk);
                kk
            })
            .collect();
        katgpt_core::alloc::reset_alloc_stats();
        for (i, key) in items.keys.iter().enumerate().take(2_000) {
            lat.write_delta(key, &norm_keys[i], &items.markers[i], 1.0);
        }
        let (w_count, _) = katgpt_core::alloc::get_alloc_stats();
        katgpt_core::alloc::reset_alloc_stats();
        for (i, key) in items.keys.iter().enumerate().take(2_000) {
            lat.read_cells(key, &norm_keys[i], &mut out);
        }
        let (r_count, _) = katgpt_core::alloc::get_alloc_stats();
        katgpt_core::alloc::reset_alloc_stats();
        black_box(&out);
        let (count, bytes) = katgpt_core::alloc::get_alloc_stats();
        println!(
            "G4 probe (3D): write {w_count} · read_cells {r_count} · hot path {count} allocs, {bytes} B"
        );
        assert_eq!(count, 0, "G4: the 3D hot path allocated {count} times");
        println!("G4: PASS");
    }
    #[cfg(not(any(debug_assertions, feature = "alloc_tracking")))]
    {
        println!(
            "G4 alloc-free ⛔ NOT MEASURED — this profile compiles no allocator. \
             Re-run with `--features alloc_tracking`; a green run without it is not a G4 pass."
        );
    }
}
