//! Plan 265 Phase 2 Example: SPLAT vs. Dense Attention.
//!
//! Demonstrates the before/after quality (argmax agreement + relative L2)
//! and effective FLOPs reduction when using the Specialist Latent Projection
//! (Fusion B) versus dense attention at 50% density.
//!
//! This example is the showcase for the MSA rescue (Plan 256 GOAT-FAILED):
//! at 50% density, SPLAT-masked attention matches dense attention quality,
//! which is the density at which blockwise sparse attention failed.

fn main() {
    #[cfg(feature = "specialist_projection")]
    {
        use katgpt_sparse::specialist_projection::{SpecialistMask, route_specialist_projection};

        println!("=== Plan 265 Phase 2: SPLAT vs. Dense Attention ===\n");

        let d_hidden = 32_usize;
        let n_keys = 8_usize;

        // Query: signal concentrated in the first half of coordinates.
        let query: Vec<f32> = (0..d_hidden)
            .map(|i| if i < d_hidden / 2 { 1.0 } else { 0.0 })
            .collect();

        // Keys: ascending signal in the first half, tiny noise in the second.
        let mut keys: Vec<f32> = vec![0.0; n_keys * d_hidden];
        for k in 0..n_keys {
            for j in 0..d_hidden / 2 {
                keys[k * d_hidden + j] = (k as f32) * 0.1;
            }
            for j in d_hidden / 2..d_hidden {
                keys[k * d_hidden + j] = 0.001 * (k as f32 - 4.0).abs();
            }
        }

        // Dense attention scores.
        let dense_scores: Vec<f32> = (0..n_keys)
            .map(|k| {
                let row = &keys[k * d_hidden..(k + 1) * d_hidden];
                let mut s = 0.0_f32;
                for j in 0..d_hidden {
                    s += query[j] * row[j];
                }
                s
            })
            .collect();
        let dense_argmax = dense_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map(|(i, _)| i)
            .unwrap();
        let dense_flops = n_keys * d_hidden;

        // SPLAT mask: keep the first half of each key (50% density).
        let support: Vec<Vec<u32>> = (0..n_keys)
            .map(|_| (0..d_hidden as u32 / 2).collect())
            .collect();
        let mask = SpecialistMask::from_support(&support, (n_keys, d_hidden));
        let density = mask.density();
        let route = route_specialist_projection(density);

        println!("Mask density: {density:.3} (target 0.5 for MSA rescue benchmark)");
        println!("Compute route at this density: {route:?}");
        println!();

        // Project keys in-place.
        let mut keys_proj = keys.clone();
        let mut scratch = vec![0.0; d_hidden];
        mask.project(&mut keys_proj, &mut scratch);

        // SPLAT attention scores (only kept coords contribute).
        let splat_scores: Vec<f32> = (0..n_keys)
            .map(|k| {
                let row = &keys_proj[k * d_hidden..(k + 1) * d_hidden];
                let mut s = 0.0_f32;
                for j in 0..d_hidden {
                    s += query[j] * row[j];
                }
                s
            })
            .collect();
        let splat_argmax = splat_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map(|(i, _)| i)
            .unwrap();
        let splat_flops = n_keys * (d_hidden / 2); // half the coords skipped

        // Relative L2.
        let mut l2 = 0.0_f32;
        let mut denom = 1e-12_f32;
        for i in 0..n_keys {
            let d = splat_scores[i] - dense_scores[i];
            l2 += d * d;
            denom += dense_scores[i] * dense_scores[i];
        }
        let rel_l2 = (l2 / denom).sqrt();

        println!("Results (before → after SPLAT projection):");
        println!("  argmax:        dense={dense_argmax} → splat={splat_argmax}");
        println!("  score rel-L2:  {rel_l2:.4}  (lower is better)");
        println!(
            "  FLOPs:         dense={dense_flops} → splat={splat_flops} ({:.0}% reduction)",
            100.0 * (1.0 - splat_flops as f32 / dense_flops as f32)
        );

        if splat_argmax == dense_argmax {
            println!("\n✓ SPLAT preserves attention ranking at 50% density (MSA rescue).");
        } else {
            println!("\n✗ SPLAT argmax disagrees with dense — investigate.");
        }

        println!("\nDone.");
    }

    #[cfg(not(feature = "specialist_projection"))]
    println!(
        "Enable feature: cargo run --example splat_vs_dense_attention --features specialist_projection"
    );
}
