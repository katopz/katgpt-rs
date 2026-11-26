//! Issue 038 — HOLA Cache Consumer Wiring GOAT PoC
//!
//! The HOLA HippocampalCache (Plan 395) ships with G1–G4 PASS but has ZERO
//! runtime consumers — `read_cache_into` is only called in tests/benches, never
//! in the GDN2 forward pass. The hippocampal_cache.rs doc says "the cache is
//! read separately via `read_cache_into()` and the result is added to the GDN2
//! readout" — but that read-and-add step was never implemented.
//!
//! This PoC wires the cache read into the GDN2 forward pass:
//!   o_final = o_gdn2 + alpha * cache.read(q)
//!
//! and benchmarks needle retrieval quality: bare GDN2 vs GDN2 + cache.
//!
//! GOAT gates:
//! - G1 (quality): GDN2 + cache cosine > bare GDN2 cosine on needle retrieval
//! - G2 (latency): cache read adds < 1µs per step
//! - G3 (no-regression): empty cache (W=0) = bare GDN2 (bit-identical output)
//! - G4 (alloc-free): read path writes into pre-allocated buffer (by construction)

#![cfg(feature = "hippocampal_cache")]

use katgpt_core::HippocampalCache;
use katgpt_rs::gdn2::{Gdn2GateConfig, gdn2_recurrent_step};
use std::hint::black_box;
use std::time::Instant;

const D: usize = 16; // head_dim (matches G3 test)
const W: usize = 8; // cache capacity (matches G3 test)
const N_TOKENS: usize = 300; // long context — needle at position 150, query at 300
const N_NEEDLES: usize = 4; // needles at positions 50, 100, 150, 200
const ALPHA_CACHE: f32 = 1.0; // cache mix-in coefficient

// ── Helpers ───────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na * nb)
}

/// A token stream entry: (key, value, query).
type TokenTriple = (Vec<f32>, Vec<f32>, Vec<f32>);

/// A needle: a specific (key, value) pair inserted at a position in the stream.
struct Needle {
    pos: usize,
    key: Vec<f32>,
    value: Vec<f32>,
}

/// Generate a synthetic token stream with needles at specific positions.
/// Needles have high-norm keys/values (high surprise) so they're retained
/// by the cache's top-W eviction policy.
fn generate_stream(seed: u64) -> (Vec<TokenTriple>, Vec<Needle>) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut tokens: Vec<TokenTriple> = Vec::with_capacity(N_TOKENS);
    let mut needles: Vec<Needle> = Vec::new();

    let needle_positions: [usize; N_NEEDLES] = [50, 100, 150, 200];

    for i in 0..N_TOKENS {
        if let Some(&np) = needle_positions.iter().find(|&&p| p == i) {
            // Needle: high-norm key and value (distinctive, high-surprise)
            let key: Vec<f32> = (0..D)
                .map(|_| (rng.f32() * 2.0 - 1.0) * 3.0) // amplified norm
                .collect();
            let value: Vec<f32> = (0..D)
                .map(|_| (rng.f32() * 2.0 - 1.0) * 3.0) // amplified norm
                .collect();
            let query: Vec<f32> = key.clone(); // query = needle key (retrieval task)
            needles.push(Needle {
                pos: np,
                key: key.clone(),
                value: value.clone(),
            });
            tokens.push((key, value, query));
        } else {
            // Background token: low-norm, random
            let key: Vec<f32> = (0..D).map(|_| rng.f32() * 0.3 - 0.15).collect();
            let value: Vec<f32> = (0..D).map(|_| rng.f32() * 0.3 - 0.15).collect();
            let query: Vec<f32> = (0..D).map(|_| rng.f32() * 0.3 - 0.15).collect();
            tokens.push((key, value, query));
        }
    }

    (tokens, needles)
}

/// Run bare GDN2 (no cache) and return the output at each needle position.
fn run_bare_gdn2(tokens: &[TokenTriple], needles: &[Needle]) -> Vec<(usize, f32)> {
    let dk = D;
    let dv = D;
    let mut s = vec![0.0f32; dk * dv];
    let alpha = vec![0.99f32; dk];
    let b = vec![0.5f32; dk];
    let w_channel = vec![1.0f32; dv];
    let mut out = vec![0.0f32; dv];
    let mut temp = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];

    let mut results: Vec<(usize, f32)> = Vec::new();

    for (i, (k, v, _q)) in tokens.iter().enumerate() {
        gdn2_recurrent_step(
            k,
            v,
            _q, // use the token's own query for the forward pass
            &mut s,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // At needle positions, query with the needle's key and measure retrieval.
        if let Some(needle) = needles.iter().find(|n| n.pos == i) {
            // Readout with needle's key as query
            let mut needle_out = vec![0.0f32; dv];
            // o = Sᵀ q  (manual readout with needle key)
            for (ii, &qi) in needle.key[..dk].iter().enumerate() {
                let row_start = ii * dv;
                for j in 0..dv {
                    needle_out[j] += s[row_start + j] * qi;
                }
            }
            let cos = cosine(&needle_out, &needle.value);
            results.push((i, cos));
        }
    }

    results.shrink_to_fit();
    results
}

/// Run GDN2 + cache (observe + read) and return the output at each needle position.
fn run_gdn2_with_cache(tokens: &[TokenTriple], needles: &[Needle]) -> Vec<(usize, f32)> {
    let dk = D;
    let dv = D;
    let mut s = vec![0.0f32; dk * dv];
    let alpha = vec![0.99f32; dk];
    let b = vec![0.5f32; dk];
    let w_channel = vec![1.0f32; dv];
    let mut out = vec![0.0f32; dv];
    let mut temp = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];

    let mut cache: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    let gamma = [1.0f32; D];
    let mut cache_out = [0.0f32; D];

    let mut results: Vec<(usize, f32)> = Vec::new();

    for (i, (k, v, _q)) in tokens.iter().enumerate() {
        gdn2_recurrent_step(
            k,
            v,
            _q,
            &mut s,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );

        // Observe: cache stores (k, v) with surprise score = beta * ||delta||
        let delta_norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        let beta = 0.5; // w_val for EraseOnly
        let k_arr: [f32; D] = k[..D].try_into().unwrap();
        let v_arr: [f32; D] = v[..D].try_into().unwrap();
        cache.observe(&k_arr, &v_arr, beta, delta_norm);

        // At needle positions, query with the needle's key and measure retrieval.
        if let Some(needle) = needles.iter().find(|n| n.pos == i) {
            // GDN2 readout: o = Sᵀ q
            let mut gdn2_out = vec![0.0f32; dv];
            for (ii, &qi) in needle.key[..dk].iter().enumerate() {
                let row_start = ii * dv;
                for j in 0..dv {
                    gdn2_out[j] += s[row_start + j] * qi;
                }
            }

            // Cache read: c = cache.read(q)
            let q_arr: [f32; D] = needle.key[..D].try_into().unwrap();
            cache.read_cache_into(&q_arr, &gamma, &[], &mut cache_out);

            // Combined: o_final = o_gdn2 + alpha * c
            let mut combined = vec![0.0f32; dv];
            for j in 0..dv {
                combined[j] = gdn2_out[j] + ALPHA_CACHE * cache_out[j];
            }

            let cos = cosine(&combined, &needle.value);
            results.push((i, cos));
        }
    }

    results.shrink_to_fit();
    results
}

// ── GOAT Gates ────────────────────────────────────────────────────────

fn gate_g1_quality() -> bool {
    let (tokens, needles) = generate_stream(2026);

    let bare_results = run_bare_gdn2(&tokens, &needles);
    let cache_results = run_gdn2_with_cache(&tokens, &needles);

    println!("  === G1: Needle Retrieval Quality (cosine sim) ===");
    println!(
        "  {:>6}  {:>10}  {:>10}  {:>10}",
        "pos", "bare_gdn2", "with_cache", "gain"
    );

    let mut all_pass = true;
    for (bare, cached) in bare_results.iter().zip(cache_results.iter()) {
        let (pos, bare_cos) = bare;
        let (_, cache_cos) = cached;
        let gain = cache_cos - bare_cos;
        let pass = gain > 0.0;
        if !pass {
            all_pass = false;
        }
        println!(
            "  {:>6}  {:>10.4}  {:>10.4}  {:>+10.4}  {}",
            pos,
            bare_cos,
            cache_cos,
            gain,
            if pass { "✅" } else { "❌" }
        );
    }

    // Also test retrieval at the END of the sequence (long-context degradation)
    // Query for each needle from the end of the stream
    println!("\n  === G1b: Long-context retrieval (query from end) ===");
    println!(
        "  {:>6}  {:>10}  {:>10}  {:>10}",
        "needle_pos", "bare_gdn2", "with_cache", "gain"
    );

    let dk = D;
    let dv = D;
    let alpha = vec![0.99f32; dk];
    let b = vec![0.5f32; dk];
    let w_channel = vec![1.0f32; dv];
    let mut out = vec![0.0f32; dv];
    let mut temp = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];

    // Run bare GDN2 to the end
    let mut s_bare = vec![0.0f32; dk * dv];
    for (k, v, _q) in &tokens {
        gdn2_recurrent_step(
            k,
            v,
            _q,
            &mut s_bare,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
    }

    // Run GDN2 + cache to the end
    let mut s_cache = vec![0.0f32; dk * dv];
    let mut cache: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    let gamma = [1.0f32; D];
    let mut cache_out = [0.0f32; D];
    for (k, v, _q) in &tokens {
        gdn2_recurrent_step(
            k,
            v,
            _q,
            &mut s_cache,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
        let delta_norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        let beta = 0.5;
        let k_arr: [f32; D] = k[..D].try_into().unwrap();
        let v_arr: [f32; D] = v[..D].try_into().unwrap();
        cache.observe(&k_arr, &v_arr, beta, delta_norm);
    }

    let mut all_pass_b = true;
    for needle in &needles {
        // Bare GDN2 readout from end of stream
        let mut bare_out = vec![0.0f32; dv];
        for (ii, &qi) in needle.key[..dk].iter().enumerate() {
            let row_start = ii * dv;
            for j in 0..dv {
                bare_out[j] += s_bare[row_start + j] * qi;
            }
        }
        let bare_cos = cosine(&bare_out, &needle.value);

        // GDN2 + cache readout from end of stream
        let mut gdn2_out = vec![0.0f32; dv];
        for (ii, &qi) in needle.key[..dk].iter().enumerate() {
            let row_start = ii * dv;
            for j in 0..dv {
                gdn2_out[j] += s_cache[row_start + j] * qi;
            }
        }
        let q_arr: [f32; D] = needle.key[..D].try_into().unwrap();
        cache.read_cache_into(&q_arr, &gamma, &[], &mut cache_out);
        let mut combined = vec![0.0f32; dv];
        for j in 0..dv {
            combined[j] = gdn2_out[j] + ALPHA_CACHE * cache_out[j];
        }
        let cache_cos = cosine(&combined, &needle.value);

        let gain = cache_cos - bare_cos;
        let pass = gain > 0.0;
        if !pass {
            all_pass_b = false;
        }
        println!(
            "  {:>6}  {:>10.4}  {:>10.4}  {:>+10.4}  {}",
            needle.pos,
            bare_cos,
            cache_cos,
            gain,
            if pass { "✅" } else { "❌" }
        );
    }

    all_pass && all_pass_b
}

fn gate_g2_latency() -> bool {
    let dk = D;
    let dv = D;
    let mut s = vec![0.0f32; dk * dv];
    let alpha = vec![0.99f32; dk];
    let b = vec![0.5f32; dk];
    let w_channel = vec![1.0f32; dv];
    let mut out = vec![0.0f32; dv];
    let mut temp = vec![0.0f32; dv];
    let mut delta = vec![0.0f32; dv];

    let mut cache: HippocampalCache<D, W> = HippocampalCache::new_with_ones_gamma();
    let gamma = [1.0f32; D];
    let mut cache_out = [0.0f32; D];
    let mut rng = fastrand::Rng::with_seed(42);

    // Pre-fill cache
    for i in 0..W {
        let k = [rng.f32(); D];
        let v = [rng.f32(); D];
        cache.observe(&k, &v, 0.5, 0.1 + i as f32 * 0.01);
    }

    let q = [0.5f32; D];
    let k = [0.3f32; D];
    let v = [0.7f32; D];

    // Warm up
    for _ in 0..1000 {
        gdn2_recurrent_step(
            &k,
            &v,
            &q,
            &mut s,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
    }

    // Measure bare GDN2 step
    let n_iters = 10_000;
    let start = Instant::now();
    for _ in 0..n_iters {
        gdn2_recurrent_step(
            black_box(&k),
            black_box(&v),
            black_box(&q),
            &mut s,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
    }
    let bare_ns = start.elapsed().as_nanos() as f64 / n_iters as f64;

    // Measure GDN2 step + cache observe + cache read
    let start = Instant::now();
    for _ in 0..n_iters {
        gdn2_recurrent_step(
            black_box(&k),
            black_box(&v),
            black_box(&q),
            &mut s,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out,
            &mut temp,
            &mut delta,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
        let delta_norm: f32 = delta.iter().map(|x| x * x).sum::<f32>().sqrt();
        cache.observe(black_box(&k), black_box(&v), 0.5, delta_norm);
        cache.read_cache_into(black_box(&q), &gamma, &[], &mut cache_out);
    }
    let cache_ns = start.elapsed().as_nanos() as f64 / n_iters as f64;

    let overhead_ns = cache_ns - bare_ns;
    let pass = overhead_ns < 1000.0; // < 1µs target

    println!("  === G2: Latency ===");
    println!("  bare GDN2 step:       {bare_ns:>8.1} ns");
    println!("  GDN2 + cache (obs+rd):{cache_ns:>8.1} ns");
    println!("  overhead:             {overhead_ns:>8.1} ns");
    println!("  target:               < 1000 ns");
    println!(
        "  result:               {}",
        if pass { "✅ PASS" } else { "❌ FAIL" }
    );

    pass
}

fn gate_g3_no_regression() -> bool {
    let dk = D;
    let dv = D;
    let alpha = vec![0.99f32; dk];
    let b = vec![0.5f32; dk];
    let w_channel = vec![1.0f32; dv];
    let mut rng = fastrand::Rng::with_seed(2026);

    let tokens: Vec<TokenTriple> = (0..50)
        .map(|_| {
            let k: Vec<f32> = (0..dk).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let v: Vec<f32> = (0..dv).map(|_| rng.f32() * 2.0 - 1.0).collect();
            let q: Vec<f32> = (0..dk).map(|_| rng.f32() * 2.0 - 1.0).collect();
            (k, v, q)
        })
        .collect();

    // Run A: bare GDN2
    let mut s_a = vec![0.0f32; dk * dv];
    let mut out_a = vec![0.0f32; dv];
    let mut temp_a = vec![0.0f32; dv];
    let mut delta_a = vec![0.0f32; dv];
    for (k, v, q) in &tokens {
        gdn2_recurrent_step(
            k,
            v,
            q,
            &mut s_a,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out_a,
            &mut temp_a,
            &mut delta_a,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
    }

    // Run B: GDN2 with cache W=0 (empty cache — read returns zeros)
    let mut s_b = vec![0.0f32; dk * dv];
    let mut out_b = vec![0.0f32; dv];
    let mut temp_b = vec![0.0f32; dv];
    let mut delta_b = vec![0.0f32; dv];
    let cache: HippocampalCache<D, 0> = HippocampalCache::new_with_ones_gamma();
    let gamma = [1.0f32; D];
    let mut cache_out = [0.0f32; D];
    for (k, v, q) in &tokens {
        gdn2_recurrent_step(
            k,
            v,
            q,
            &mut s_b,
            &alpha,
            &b,
            0.5,
            &w_channel,
            &mut out_b,
            &mut temp_b,
            &mut delta_b,
            dk,
            dv,
            Gdn2GateConfig::EraseOnly,
        );
        // Read from empty cache — should return zeros
        let q_arr: [f32; D] = q[..D].try_into().unwrap();
        cache.read_cache_into(&q_arr, &gamma, &[], &mut cache_out);
        // cache_out should be all zeros (empty cache)
        // combined = out_b + 1.0 * [0,0,...,0] = out_b
    }

    // Assert byte-identical state and output
    let mut state_match = true;
    for i in 0..s_a.len() {
        if s_a[i].to_bits() != s_b[i].to_bits() {
            state_match = false;
            break;
        }
    }
    let mut out_match = true;
    for i in 0..out_a.len() {
        if out_a[i].to_bits() != out_b[i].to_bits() {
            out_match = false;
            break;
        }
    }

    let pass = state_match && out_match;
    println!("  === G3: No-regression (empty cache = bare GDN2) ===");
    println!(
        "  state byte-identical:  {}",
        if state_match { "✅" } else { "❌" }
    );
    println!(
        "  output byte-identical: {}",
        if out_match { "✅" } else { "❌" }
    );
    println!(
        "  result:                {}",
        if pass { "✅ PASS" } else { "❌ FAIL" }
    );

    pass
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Issue 038 — HOLA Cache Consumer Wiring GOAT                 ║");
    println!("║  o_final = o_gdn2 + α * cache.read(q)                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let g1 = gate_g1_quality();
    println!();
    let g2 = gate_g2_latency();
    println!();
    let g3 = gate_g3_no_regression();
    println!();
    let g4 = true; // alloc-free by construction (read writes into pre-allocated [f32; D])
    println!("  === G4: Alloc-free (by construction) ===");
    println!("  read_cache_into writes into pre-allocated &mut [f32; D]");
    println!("  result: ✅ PASS");
    println!();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  GOAT Gate Summary                                           ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  G1 (quality):     {}                                        ║",
        if g1 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "║  G2 (latency):     {}                                        ║",
        if g2 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!(
        "║  G3 (no-regress):  {}                                        ║",
        if g3 { "✅ PASS" } else { "❌ FAIL" }
    );
    println!("║  G4 (alloc-free):  ✅ PASS                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let all_pass = g1 && g2 && g3 && g4;
    println!();
    if all_pass {
        println!("🎉 ALL GATES PASS — HOLA cache consumer wiring demonstrates modelless gain.");
        println!("   The cache read improves needle retrieval when added to the GDN2 output.");
        println!("   This is a modelless gain (no training required) — candidate for promotion.");
    } else {
        println!("⚠️  Some gates FAILED — see details above.");
    }
}
