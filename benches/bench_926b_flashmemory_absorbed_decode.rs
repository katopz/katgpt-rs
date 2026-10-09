//! Issue 926 SECONDARY scope — flashmemory absorbed-vs-reconstruct decode (G2).
//!
//! The dense lane's re-pin is `bench_926_mla_absorbed_decode`; this is the
//! sparse lane's paired A/B (`mla_forward_token_flashmemory_reconstruct` vs
//! `mla_forward_token_flashmemory_absorbed`) under the SAME discipline:
//! kimi_k3_0_40b geometry, random weights, seq {1K, 4K, 16K}, INTERLEAVED
//! pairs (median over pairs — two sequentially-timed arms measure the BOX,
//! not the code), `black_box` on both arms' outputs (the loud-zero defence).
//!
//! Sparse-lane state: each arm owns a FULL state set (KV cache, scratch,
//! rope, block cache, selector) fed IDENTICAL inputs, so block selection is
//! identical by construction (the G1 contract — `Selection` is derived from
//! the same centroids both sides). The cache is filled by running each arm
//! over the same token sequence (untimed), then each timed step
//! `rewind_to(seq)`s the KV cache so the attended length is CONSTANT across
//! steps (the Ouro pattern from the dense bench). Both arms see the same
//! `step` parameter per pair, so the cached-selection refresh posture is
//! identical — the measurement prices the steady-state SELECTED-BLOCK READ,
//! which is the step-6-class cost absorption removes.
//!
//! A flat/negative ratio is a valid outcome (the feature stays opt-in; the
//! negative is recorded). The FLOP model says ~60× fewer up-projection
//! FLOPs at large seq; G2 asks what the wall clock says.
//!
//! # argv: [--pairs N] [--seqs 1024,4096,16384]
//!
//! Box state is part of every published number: quote `scripts/bench_preflight`
//! provenance beside the medians. Runbook precondition: load < 2, no sibling
//! rustc/census (the dense re-pin's rule — the sparse read is bandwidth-bound
//! over the same cache).
//!
//! ```text
//! CARGO_TARGET_DIR=/tmp/katgpt-926-root cargo bench -p katgpt-rs \
//!   --features mla_absorbed,flashmemory_sparse \
//!   --bench bench_926b_flashmemory_absorbed_decode
//! ```

use std::time::Instant;

use katgpt_attn::dash_attn::flashmemory_sparse::{
    FlashMemoryBlockCache, FlashMemoryConfig, FlashMemorySelector,
    mla_forward_token_flashmemory_absorbed, mla_forward_token_flashmemory_reconstruct,
};
use katgpt_attn::mla::{MlaConfig, MlaForwardScratch, MlaKVCache, MlaWeights};
use katgpt_kv::shard_kv::rope::RopeFreqs;

#[inline]
fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// Deterministic LCG (same spirit as the dense bench's fill).
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

/// One arm's full state set (the spec-match test's per-side instance shape).
struct Arm {
    cache: MlaKVCache,
    scratch: MlaForwardScratch,
    rope: RopeFreqs,
    block_cache: FlashMemoryBlockCache,
    selector: FlashMemorySelector,
}

impl Arm {
    fn new(config: &MlaConfig, fm: &FlashMemoryConfig, max_seq: usize) -> Self {
        let max_blocks = max_seq.div_ceil(fm.block_size).max(1);
        Self {
            cache: MlaKVCache::new(config, max_seq),
            scratch: MlaForwardScratch::new(config, max_seq),
            rope: RopeFreqs::new_with_theta(config.qk_rope_head_dim, config.rope_theta),
            block_cache: FlashMemoryBlockCache::new(config, fm, max_seq),
            selector: FlashMemorySelector::new(fm.clone(), config.n_heads, max_blocks),
        }
    }
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
    // Paper defaults: block 64, refresh 64, threshold 0.5.
    let fm = FlashMemoryConfig::default();
    let d = config.hidden_size;

    println!("bench_926b_flashmemory_absorbed_decode — absorbed vs reconstruct FLASHMEMORY decode");
    println!(
        "config: kimi_k3_0_40b (d_c={}, n_h={}) | fm: block={} refresh={} thr={}",
        config.kv_lora_rank, config.n_heads, fm.block_size, fm.refresh_period, fm.threshold
    );
    println!("feature mla_absorbed,flashmemory_sparse: ON (both arms compiled)\n");

    let mut rng = Lcg(0x926_000_926_926);
    let mut h: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.1).collect();

    let mut row_summary: Vec<String> = Vec::new();

    for &seq in &seqs {
        let max_seq = seq + 8;
        let mut a = Arm::new(&config, &fm, max_seq);
        let mut b = Arm::new(&config, &fm, max_seq);

        // FILL (untimed): run both arms over the IDENTICAL h sequence so both
        // end at `seq` cached tokens with identical block/selector state.
        for step in 0..seq {
            for v in h.iter_mut() {
                *v = (*v + 0.0011).clamp(-1.0, 1.0);
            }
            let out_a = mla_forward_token_flashmemory_reconstruct(
                &config, &weights, &mut a.cache, &mut a.scratch, &mut a.rope, &h,
                &mut a.block_cache, &mut a.selector, step,
            );
            black_box(&out_a[..8.min(d)]);
            let out_b = mla_forward_token_flashmemory_absorbed(
                &config, &weights, &mut b.cache, &mut b.scratch, &mut b.rope, &h,
                &mut b.block_cache, &mut b.selector, step,
            );
            black_box(&out_b[..8.min(d)]);
        }

        // Adaptive pair count: keep each arm's timed total ~1.5 s per seq.
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
        for i in 0..n_pairs {
            // Constant attended length per timed step (the Ouro rewind).
            a.cache.rewind_to(seq);
            b.cache.rewind_to(seq);
            let step = seq + i; // both arms see the SAME step → same refresh posture

            // ── arm A: reconstruct ──
            for v in h.iter_mut() {
                *v = (*v + 0.0013).clamp(-1.0, 1.0);
            }
            let t0 = Instant::now();
            let out = mla_forward_token_flashmemory_reconstruct(
                &config, &weights, &mut a.cache, &mut a.scratch, &mut a.rope, &h,
                &mut a.block_cache, &mut a.selector, step,
            );
            black_box(&out[..8.min(d)]);
            recon_us.push(t0.elapsed().as_secs_f64() * 1e6);

            // ── arm B: absorbed ──
            b.cache.rewind_to(seq);
            for v in h.iter_mut() {
                *v = (*v + 0.0017).clamp(-1.0, 1.0);
            }
            let t0 = Instant::now();
            let out = mla_forward_token_flashmemory_absorbed(
                &config, &weights, &mut b.cache, &mut b.scratch, &mut b.rope, &h,
                &mut b.block_cache, &mut b.selector, step,
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
        let spread = (max(&abs_us) - min(&abs_us)) / m_a * 100.0;

        println!("seq={seq:>6}  pairs={n_pairs}");
        println!(
            "  fm-reconstruct: median {m_r:>10.1} µs/step  ({:>8.0} µs spread min-max {:+.0}%..{:+.0}%)",
            m_r,
            (min(&recon_us) / m_r - 1.0) * 100.0,
            (max(&recon_us) / m_r - 1.0) * 100.0,
        );
        println!("  fm-absorbed:    median {m_a:>10.1} µs/step  (spread {spread:.1}%)");
        println!("  speedup:     {ratio:.2}×  (absorbed/recon wall-clock)");
        println!();
        row_summary.push(format!(
            "seq={seq}: {ratio:.2}× (recon {m_r:.1} µs, absorbed {m_a:.1} µs)"
        ));
    }

    println!("── summary (G2 rows, flashmemory lane) ──");
    for r in &row_summary {
        println!("  {r}");
    }
}
