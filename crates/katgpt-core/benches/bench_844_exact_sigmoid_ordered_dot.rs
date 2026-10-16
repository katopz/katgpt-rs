//! GOAT gate — `exact_sigmoid` / `exact_sigmoid_f64` / `dot_f32_ordered`
//! substrate promotion (riir-chain Issue 156 T1; Bench 844).
//!
//! ```bash
//! cargo bench -p katgpt-core --bench bench_844_exact_sigmoid_ordered_dot
//! ```
//!
//! **G1 correctness** — the exact variants are characterized against the
//! highest-precision oracle std can express, with the two-branch form's
//! value proposition asserted rather than assumed:
//! - G1a (f32): ULP distance vs the f64-computed-and-narrowed reference
//!   over a dense grid + edges, pinned to the MEASURED maximum (never an
//!   invented constant); max abs error reported. Contrast arm: the same
//!   scan over `fast_sigmoid`, which must measure strictly worse (its own
//!   doc concedes 90.6% bit-exact / 1.19e-7 max err, and it clamps to
//!   exactly 0.0/1.0 past ±40 where the true value is a representable
//!   tiny/one-minus-tiny).
//! - G1b (f64): property gates, because the f64 variant has no
//!   higher-precision std oracle (a ULP gate against libm f64 would be
//!   circular — the implementation IS f64 libm): reflection
//!   `σ(x) + σ(-x) = 1` within 1 ULP at 1.0, monotone non-decreasing, and
//!   exact bounds at 0 / ±∞ / NaN / ±800 / the ±40 saturation boundary.
//! - G1c (`dot_f32_ordered`): the frozen sequential-fold value on a
//!   cancellation-heavy crafted input, and the anti-dedup pin — the same
//!   input through `simd_dot_f32` must differ (every backend reassociates:
//!   NEON/AVX2/wasm-simd128 lanes, or the 4-accumulator + `mul_add` scalar
//!   fallback). If a future backend converges with the ordered fold, this
//!   gate reds and the two kernels' reason to coexist is re-adjudicated.
//!
//! **G2 perf** — REPORTED, not barred: the claim this bench exists for is
//! exactness, not speed, and a latency bar without its box state is not a
//! measurement. Numbers are best-of-N minimum ns/op (the load-robust
//! statistic) so the doc can cite them with the box state recorded beside.
//!
//! **G3 no-regression** — additive by construction: no existing call site
//! is rerouted (the delegation happens downstream in riir-chain, behind its
//! own bit-identity pin), `fast_sigmoid` and `simd_dot_f32` are untouched,
//! and the gates above re-prove the untouched kernels' behavior.
//!
//! **G4 alloc-free** — the promoted fns are pure stack math by inspection
//! (no `Vec`/`String`/closure-capture allocation anywhere in the bodies);
//! no allocator arm is wired because there is no allocation path to count.

use std::hint::black_box;
use std::time::Instant;

use katgpt_core::simd::{dot_f32_ordered, fast_sigmoid, simd_dot_f32};
use katgpt_core::{exact_sigmoid, exact_sigmoid_f64};
use katgpt_types::approx::ApproxContract;

/// Deterministic xorshift — seeded, local, never the global RNG.
struct Xorshift(u64);

impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

fn verdict(ok: bool) -> &'static str {
    match ok {
        true => "✓",
        false => "✗",
    }
}

/// The f64-computed sigmoid, narrowed — the highest-precision reference
/// std can express.
fn f64_ref(x: f32) -> f32 {
    (1.0f64 / (1.0 + (-(x as f64)).exp())) as f32
}

/// Best-of-`rounds` minimum nanoseconds per call for an f32-accumulating
/// closure (the load-robust statistic — a loaded box can only make a round
/// slower). The argument varies the query across iterations so LLVM cannot
/// fold the loop.
fn best_of_ns_f32(rounds: usize, iters: usize, mut f: impl FnMut(u32) -> f32) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        let mut acc = 0.0_f32;
        for i in 0..iters as u32 {
            acc += f(i);
        }
        black_box(acc);
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

/// Same statistic for an f64-accumulating closure (the dot kernels).
fn best_of_ns_f64(rounds: usize, iters: usize, mut f: impl FnMut(u32) -> f64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        let mut acc = 0.0_f64;
        for i in 0..iters as u32 {
            acc += f(i);
        }
        black_box(acc);
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        if ns < best {
            best = ns;
        }
    }
    best
}

fn main() {
    println!("═══ Bench 844 — exact_sigmoid / dot_f32_ordered substrate GOAT ═══");

    // ── G1a: f32 exactness vs the f64 reference ────────────────────────────────
    // The ULP halves are two named approximation contracts (Issue 944); the
    // abs-error maxima stay plain folds beside them.
    let mut exact = ApproxContract::new("exact_sigmoid vs f64 reference");
    let mut fast = ApproxContract::new("fast_sigmoid vs f64 reference");
    let mut exact_max_abs = 0.0f32;
    let mut fast_max_abs = 0.0f32;
    let mut x = -80.0f32;
    while x <= 80.0 {
        let r = f64_ref(x);
        exact.measure(exact_sigmoid(x), r);
        fast.measure(fast_sigmoid(x), r);
        exact_max_abs = exact_max_abs.max((exact_sigmoid(x) - r).abs());
        fast_max_abs = fast_max_abs.max((fast_sigmoid(x) - r).abs());
        x += 0.0137;
    }
    // Edges inside the normal-output regime.
    for edge in [-1e-30f32, 1e-30, 1e-20, -1e-20, 1.0, -1.0, 27.5, -27.5] {
        let r = f64_ref(edge);
        exact.measure(exact_sigmoid(edge), r);
        fast.measure(fast_sigmoid(edge), r);
    }
    let exact_max_ulp = exact.max_ulps().expect("sigmoid outputs are finite");
    let fast_max_ulp = fast.max_ulps().expect("sigmoid outputs are finite");
    let g1a = exact_max_ulp <= 4 && exact_max_abs <= 1e-6 && exact_max_ulp < fast_max_ulp;
    println!(
        "G1a f32 exactness: exact max {exact_max_ulp} ULP / {exact_max_abs:.3e} abs · fast max \
         {fast_max_ulp} ULP / {fast_max_abs:.3e} abs (grid [-80, 80] step 0.0137 + edges) {}",
        verdict(g1a)
    );
    assert!(
        exact_max_ulp <= 4,
        "exact_sigmoid drifts {exact_max_ulp} ULP from the f64 reference"
    );
    assert!(exact_max_abs <= 1e-6);
    assert!(
        exact_max_ulp < fast_max_ulp,
        "the exact variant must measure strictly closer than the approximation"
    );

    // ── G1b: f64 property gates ──────────────────────────────────────────
    let mut reflection_ok = true;
    let mut monotone_ok = true;
    let mut prev = -1.0f64;
    let mut x = -50.0f64;
    while x <= 50.0 {
        let s = exact_sigmoid_f64(x) + exact_sigmoid_f64(-x);
        if (s - 1.0).abs() > f64::EPSILON {
            reflection_ok = false;
        }
        let v = exact_sigmoid_f64(x);
        if v < prev {
            monotone_ok = false;
        }
        prev = v;
        x += 0.011;
    }
    let bounds_ok = exact_sigmoid_f64(0.0) == 0.5
        && exact_sigmoid_f64(f64::INFINITY) == 1.0
        && exact_sigmoid_f64(f64::NEG_INFINITY) == 0.0
        && exact_sigmoid_f64(f64::NAN).is_nan()
        && exact_sigmoid_f64(800.0) == 1.0
        && exact_sigmoid_f64(-800.0) == 0.0
        && exact_sigmoid_f64(-40.0) > 0.0;
    let g1b = reflection_ok && monotone_ok && bounds_ok;
    println!(
        "G1b f64 properties: reflection {} · monotone {} · bounds {} {}",
        verdict(reflection_ok),
        verdict(monotone_ok),
        verdict(bounds_ok),
        verdict(g1b)
    );
    assert!(g1b);

    // f32 bounds (same edge law as the f64 variant).
    let f32_bounds_ok = exact_sigmoid(0.0) == 0.5
        && exact_sigmoid(f32::INFINITY) == 1.0
        && exact_sigmoid(f32::NEG_INFINITY) == 0.0
        && exact_sigmoid(f32::NAN).is_nan()
        && exact_sigmoid(-50.0) > 0.0
        && exact_sigmoid(-50.0) < 1e-17
        && exact_sigmoid(800.0) == 1.0
        && exact_sigmoid(-800.0) == 0.0;
    println!(
        "G1b f32 bounds (incl. the non-clamped far tail) {}",
        verdict(f32_bounds_ok)
    );
    assert!(f32_bounds_ok);

    // ── G1c: ordered-dot pins ────────────────────────────────────────────
    let a4 = [1e8f32, 1.0, -1e8, 1.0];
    let b4 = [1.0f32; 4];
    let ordered_crafted = dot_f32_ordered(&a4, &b4);
    let simd_crafted = simd_dot_f32(&a4, &b4, 4);
    let frozen_ok = ordered_crafted == 1.0;
    let differs_ok = ordered_crafted != simd_crafted;
    println!(
        "G1c ordered-dot pins: frozen sequential value 1.0 {} (got {ordered_crafted}) · \
         differs from simd_dot_f32 {} (simd got {simd_crafted}) {}",
        verdict(frozen_ok),
        verdict(differs_ok),
        verdict(frozen_ok && differs_ok)
    );
    assert!(
        frozen_ok,
        "the sequential fold stopped being sequential: got {ordered_crafted}"
    );
    assert!(
        differs_ok,
        "dot_f32_ordered == simd_dot_f32 on the crafted pin input — the kernels have \
         converged; their reason to coexist must be re-adjudicated"
    );

    // ── G2: timings (REPORTED, not barred — box state lives in the doc) ──
    let t_fast = best_of_ns_f32(50, 10_000, |i| fast_sigmoid((i % 1000) as f32 * 0.1 - 50.0));
    let t_exact = best_of_ns_f32(50, 10_000, |i| {
        exact_sigmoid((i % 1000) as f32 * 0.1 - 50.0)
    });
    let t_exact_f64 = best_of_ns_f32(50, 10_000, |i| {
        exact_sigmoid_f64((i % 1000) as f64 * 0.1 - 50.0) as f32
    });

    let mut seed = Xorshift(0x9E3779B97F4A7C15);
    let mut pseudo = || (seed.next_u64() >> 40) as f32 / (1 << 24) as f32 - 8.0;
    let mut pools: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(4);
    for _ in 0..4 {
        let a: Vec<f32> = (0..1024).map(|_| pseudo()).collect();
        let b: Vec<f32> = (0..1024).map(|_| pseudo()).collect();
        pools.push((a, b));
    }
    let pools_ref = &pools;
    let t_simd = best_of_ns_f64(50, 200, |i| {
        let (a, b) = &pools_ref[(i % 4) as usize];
        simd_dot_f32(a, b, 1024) as f64
    });
    let t_ordered = best_of_ns_f64(50, 200, |i| {
        let (a, b) = &pools_ref[(i % 4) as usize];
        dot_f32_ordered(a, b) as f64
    });

    println!(
        "G2 perf (best-of-50 min, REPORTED not barred): fast_sigmoid {t_fast:.1} ns · \
         exact_sigmoid {t_exact:.1} ns · exact_sigmoid_f64 {t_exact_f64:.1} ns · \
         simd_dot 1024 {t_simd:.1} ns · dot_ordered 1024 {t_ordered:.1} ns"
    );

    println!(
        "═══ Verdict: G1a {} G1b {} G1c {} — exactness gates asserted, perf reported ═══",
        verdict(g1a),
        verdict(g1b),
        verdict(frozen_ok && differs_ok)
    );
}
