//! Issue 926 — absorbed-vs-reconstruct MLA decode GOAT gate (G2).
//!
//! Times the two decode paths at the kimi_k3_0_40b geometry (d_h=64, d_c=128,
//! v_h=64, n_h=16) over seq {1K, 4K, 16K} with random weights. The FLOP
//! arithmetic says ~60× (the issue's table); G2 asks what the WALL CLOCK says
//! — if the step is bound elsewhere (cache bandwidth over the latent matrix,
//! rope dots), the win shrinks, and a flat/negative result is a valid outcome
//! (the feature stays opt-in, the negative is recorded).
//!
//! Method:
//! - The cache is pre-filled DIRECTLY (random latents + rope keys) so the
//!   timed step is pure decode attention, not prefill.
//! - Every timed step uses `rewind_to(seq)` (the Ouro pattern) so the cache
//!   length is CONSTANT across steps — each step overwrites the same slot.
//! - INTERLEAVED A/B pairs (recon, absorbed) back-to-back, median over pairs —
//!   two sequentially-timed arms 30s apart measure the BOX, not the code
//!   (the +5.2%/+21.7% lesson). Pair-relative ratios are printed beside the
//!   absolute medians.
//! - `black_box` on both arms' outputs (the loud-zero defence: an optimiser
//!   that deletes an arm must not print a number).
//!
//! # argv: [--pairs N] [--seqs 1024,4096,16384]
//!
//! Box state is part of every published number: quote `scripts/bench_preflight`
//! provenance (or free RAM + load + power source) beside the medians.

use std::time::Instant;

use katgpt_attn::mla::{
    MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights, mla_forward_token,
    mla_forward_token_absorbed,
};
use katgpt_kv::shard_kv::rope::RopeFreqs;

#[inline]
fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// Deterministic LCG (the MlaWeights::random spirit) for cache fill + inputs.
struct Lcg(u64);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (u64::MAX >> 33) as f32) * 2.0 - 1.0
    }
}

fn parse_arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let pairs: usize = parse_arg(&args, "--pairs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0); // 0 = adaptive per seq
    let seqs: Vec<usize> = parse_arg(&args, "--seqs")
        .map(|v| {
            v.split(',')
                .filter_map(|s| s.parse().ok())
                .collect::<Vec<usize>>()
        })
        .unwrap_or_else(|| vec![1024, 4096, 16384]);

    let config = MlaConfig::kimi_k3_0_40b();
    let weights = MlaWeights::random(&config, 926);
    let d = config.hidden_size;
    let d_c = config.kv_lora_rank;
    let d_r = config.d_r();

    println!("bench_926_mla_absorbed_decode — absorbed vs reconstruct MLA decode");
    println!("config: kimi_k3_0_40b (d_h={}, d_c={}, d_r={}, v_h={}, n_h={})",
        config.d_h(), d_c, d_r, config.v_head_dim, config.n_heads);
    println!("feature mla_absorbed: ON (both arms compiled; reconstruct = reference)\n");

    let mut rng = Lcg(0x926_926_926_926);
    let mut h: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.1).collect();

    let mut row_summary: Vec<String> = Vec::new();

    for &seq in &seqs {
        let max_seq = seq + 8; // headroom: the timed step appends one token
        let mut cache = MlaKVCache::new(&config, max_seq);
        let mut scratch = MlaForwardScratch::new(&config, max_seq);
        let mut rope = RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta);

        // Pre-fill the cache DIRECTLY: random latents + rope keys. The timed
        // step only reads them (pure decode attention).
        for _ in 0..seq {
            let lat: Vec<f32> = (0..d_c).map(|_| rng.next_f32() * 0.05).collect();
            let kr: Vec<f32> = (0..d_r).map(|_| rng.next_f32() * 0.05).collect();
            cache.append(&lat, &kr);
        }

        // Warm both arms (JIT-free but simd/power-state warm + rewind pattern).
        for arm in 0..2 {
            cache.rewind_to(seq);
            for v in h.iter_mut() {
                *v = (*v + 0.001).clamp(-1.0, 1.0);
            }
            let out = if arm == 0 {
                mla_forward_token(&config, &weights, &mut cache, &mut scratch, &mut rope, &h)
            } else {
                mla_forward_token_absorbed(
                    &config, &weights, &mut cache, &mut scratch, &mut rope, &h,
                )
            };
            black_box(&out[..8.min(d)]);
        }

        // Adaptive pair count: keep each arm's total under ~1.5 s per seq.
        let n_pairs = if pairs > 0 {
            pairs
        } else if seq >= 16384 {
            8
        } else if seq >= 4096 {
            16
        } else {
            40
        };

        let mut recon_us: Vec<f64> = Vec::with_capacity(n_pairs);
        let mut abs_us: Vec<f64> = Vec::with_capacity(n_pairs);
        for _ in 0..n_pairs {
            // ── arm A: reconstruct ──
            cache.rewind_to(seq);
            for v in h.iter_mut() {
                *v = (*v + 0.0013).clamp(-1.0, 1.0);
            }
            let t0 = Instant::now();
            let out = mla_forward_token(&config, &weights, &mut cache, &mut scratch, &mut rope, &h);
            black_box(&out[..8.min(d)]);
            recon_us.push(t0.elapsed().as_secs_f64() * 1e6);

            // ── arm B: absorbed ──
            cache.rewind_to(seq);
            for v in h.iter_mut() {
                *v = (*v + 0.0017).clamp(-1.0, 1.0);
            }
            let t0 = Instant::now();
            let out = mla_forward_token_absorbed(
                &config, &weights, &mut cache, &mut scratch, &mut rope, &h,
            );
            black_box(&out[..8.min(d)]);
            abs_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        let median = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.total_cmp(b));
            v[v.len() / 2]
        };
        let min = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = |v: &[f64]| v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let (m_r, m_a) = (median(recon_us.clone()), median(abs_us.clone()));
        let ratio = m_r / m_a;
        // Pair-wise relative spread on the ABSORBED arm = the noise floor tell.
        let spread = (max(&abs_us) - min(&abs_us)) / m_a * 100.0;

        println!("seq={seq:>6}  pairs={n_pairs}");
        println!(
            "  reconstruct: median {m_r:>10.1} µs/step  ({:>8.0} µs spread min-max {:+.0}%..{:+.0}%)",
            m_r,
            (min(&recon_us) / m_r - 1.0) * 100.0,
            (max(&recon_us) / m_r - 1.0) * 100.0,
        );
        println!(
            "  absorbed:    median {m_a:>10.1} µs/step  (spread {spread:.1}%)",
        );
        println!("  speedup:     {ratio:.2}×  (absorbed/recon wall-clock)");
        println!();
        row_summary.push(format!("seq={seq}: {ratio:.2}× (recon {m_r:.1} µs, absorbed {m_a:.1} µs)"));
    }

    println!("── summary (G2 rows) ──");
    for r in &row_summary {
        println!("  {r}");
    }
}
