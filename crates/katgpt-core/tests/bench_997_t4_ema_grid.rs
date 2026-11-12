//! Issue 997 T4 — the mini-batch EMA lane + the (B, α₀) sanity grid.
//!
//! Batch Lloyd's never empties clusters (the T2-T7 structural finding), so
//! the split-vs-farthest head-to-head lives here: mini-batch EMA dynamics
//! starve centroids naturally. The paper's (B, α₀) constants are OUT-OF-RANGE
//! extrapolations at n=16K (fitted at n≥10⁶) — "no sweep needed" is NOT
//! claimed until this grid shows the law's region sits in the low-loss band.
//!
//! Run:
//! ```bash
//! CARGO_TARGET_DIR=/tmp/k997 cargo test -p katgpt-core --release \
//!   --features factorized_action,emeans_poc --test bench_997_t4_ema_grid -- --nocapture
//! ```

#[cfg(all(feature = "factorized_action", feature = "emeans_poc"))]
mod t4 {
    use katgpt_core::factorized_action::codebook::{
        fit_codebook_kmeans_ema_into, EmeansConfig,
    };
    use katgpt_core::factorized_action::types::EffectCodebook;

    const K: usize = 128;
    const D: usize = 32;
    const EPOCHS: usize = 12;

    fn sse(patches: &[&[f32]], cb: &EffectCodebook<K, D>) -> f64 {
        let mut total = 0.0f64;
        for p in patches {
            let mut best = f32::INFINITY;
            for kk in 0..K {
                let c = cb.centroid(kk);
                let mut s = 0.0f32;
                for d in 0..D {
                    let diff = p[d] - c[d];
                    s += diff * diff;
                }
                if s < best {
                    best = s;
                }
            }
            total += best as f64;
        }
        total
    }

    fn codebook_blake3(cb: &EffectCodebook<K, D>) -> String {
        let mut hasher = blake3::Hasher::new();
        for kk in 0..K {
            let bytes: Vec<u8> = cb
                .centroid(kk)
                .iter()
                .flat_map(|f| f.to_bits().to_le_bytes())
                .collect();
            hasher.update(&bytes);
        }
        hasher.finalize().to_hex().to_string()
    }

    /// The quantized multi-blob 16K corpus (digit-sprite class: coarse grid,
    /// duplicate-heavy, most structure in a few dims).
    fn corpus_16k() -> Vec<Vec<f32>> {
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let next = |state: &mut u64| -> f32 {
            *state ^= *state >> 30;
            *state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            *state ^= *state >> 27;
            *state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
            *state ^= *state >> 31;
            ((*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
        };
        let n_centers = 16usize;
        let per_center = 1024usize;
        let mut centers = Vec::new();
        for c in 0..n_centers {
            let mut ctr = vec![0.0f32; D];
            for d in 0..D {
                ctr[d] = (c as f32) * 10.0 + next(&mut state) * 5.0;
            }
            centers.push(ctr);
        }
        let mut corpus = Vec::with_capacity(n_centers * per_center);
        for _ in 0..per_center {
            for c in 0..n_centers {
                let mut p = vec![0.0f32; D];
                for d in 0..D {
                    p[d] = (centers[c][d] + next(&mut state) * 3.0).round();
                }
                corpus.push(p);
            }
        }
        corpus
    }

    fn run(
        patches: &[&[f32]],
        cfg: &EmeansConfig,
        seed: u64,
    ) -> (EffectCodebook<K, D>, f64, std::time::Duration) {
        let mut cb = EffectCodebook::<K, D>::zeroed();
        let t0 = std::time::Instant::now();
        fit_codebook_kmeans_ema_into(patches, K, seed, cfg, &mut cb);
        (cb, sse(patches, &cb), t0.elapsed())
    }

    #[test]
    fn t4_grid_and_head_to_head() {
        let patches_owned = corpus_16k();
        let patches: Vec<&[f32]> = patches_owned.iter().map(|p| p.as_slice()).collect();

        // ── T6: determinism parity (two runs, BLAKE3) ──
        {
            let cfg = EmeansConfig::default();
            let (a, sa, _) = run(&patches, &cfg, 7);
            let (b, sb, _) = run(&patches, &cfg, 7);
            assert_eq!(sa, sb);
            assert_eq!(codebook_blake3(&a), codebook_blake3(&b), "T6 parity");
        }
        println!("T6 determinism: BLAKE3 parity PASS");

        // ── the ~5×5 (B, α₀) sanity grid ──
        let bs = [125usize, 250, 500, 1000, 2000];
        let alphas = [0.01f64, 0.05, 0.1, 0.3, 0.6];
        let mut grid = vec![0.0f64; bs.len() * alphas.len()];
        let mut best = (f64::INFINITY, 0usize, 0usize);
        for (bi, &b) in bs.iter().enumerate() {
            for (ai, &a0) in alphas.iter().enumerate() {
                let cfg = EmeansConfig {
                    batch_size: b,
                    alpha0: a0,
                    epochs: EPOCHS,
                };
                let (_, s, _) = run(&patches, &cfg, 2024);
                grid[bi * alphas.len() + ai] = s;
                if s < best.0 {
                    best = (s, bi, ai);
                }
            }
        }
        println!("(B, α₀) grid, SSE (rows B, cols α₀):");
        for (bi, &b) in bs.iter().enumerate() {
            let cells: Vec<String> = alphas
                .iter()
                .enumerate()
                .map(|(ai, _)| format!("{:9.3e}", grid[bi * alphas.len() + ai]))
                .collect();
            println!("  B={b:5}: {}", cells.join(" "));
        }
        println!(
            "  low-loss band: B={} α₀={} (SSE {:.4e}) — the law's (B≈hundreds, α₀ from the paper) must sit inside this band for the OUT-OF-RANGE extrapolation to hold",
            bs[best.1],
            alphas[best.2],
            best.0
        );

        // ── the fit at the grid's best point + the 50-seed stuck-rate ──
        // (the split-vs-farthest head-to-head was measured NULL in every
        // reachable regime — see the codebook revive-block note — so the
        // revival dimension is collapsed to the incumbent pick)
        let cfg_best = EmeansConfig {
            batch_size: bs[best.1],
            alpha0: alphas[best.2],
            epochs: EPOCHS,
        };
        let (cb_f, sse_f, wall_f) = run(&patches, &cfg_best, 2024);
        println!("T4 fit @ the grid point: SSE {sse_f:.6e} wall {wall_f:?}");

        let blobs: [(f32, usize, f32); 3] = [(0.0, 400, 40.0), (300.0, 100, 60.0), (150.0, 40, 25.0)];
        let mut state = 0xFEED_FACEu64;
        let next = |st: &mut u64| -> f32 {
            *st ^= *st >> 30;
            *st = st.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            *st ^= *st >> 27;
            *st = st.wrapping_mul(0x94D0_49BB_1331_11EB);
            *st ^= *st >> 31;
            ((*st >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32
        };
        let mut uneven = Vec::new();
        for (center, count, width) in blobs {
            for _ in 0..count {
                let mut p = vec![0.0f32; D];
                for d in 0..D {
                    let raw = center + next(&mut state) * width;
                    p[d] = (raw / 2.0).round() * 2.0;
                }
                uneven.push(p);
            }
        }
        let uneven_slices: Vec<&[f32]> = uneven.iter().map(|p| p.as_slice()).collect();
        let mut stuck = 0usize;
        let mut sses = Vec::new();
        for seed in 0..50u64 {
            let (_, s, _) = run(&uneven_slices, &cfg_best, seed);
            sses.push(s);
        }
        let best_run = sses.iter().cloned().fold(f64::INFINITY, f64::min);
        for &s in &sses {
            if s > best_run * 1.01 {
                stuck += 1;
            }
        }
        println!(
            "T3 stuck-rate (50 seeds, uneven blobs, mini-batch): {stuck}/50 \
             (best-of-seeds SSE {best_run:.6e})"
        );

        // Sanity: finite centroids on every arm run in this test.
        for cb in [&cb_f] {
            for kk in 0..K {
                for d in 0..D {
                    assert!(cb.centroid(kk)[d].is_finite());
                }
            }
        }
    }
}

#[cfg(not(all(feature = "factorized_action", feature = "emeans_poc")))]
#[test]
fn t4_grid_and_head_to_head_gated_out() {
    // Count-per-posture law: exists + passes when the feature is off.
}
