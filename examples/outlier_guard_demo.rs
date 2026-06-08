//! Plan 224 Example: Outlier-Aware Quantization Guard Demo

fn main() {
    #[cfg(feature = "outlier_guard")]
    {
        use katgpt_rs::spectralquant::outlier_guard::OutlierGuard;
        use katgpt_rs::types::{OutlierAction, OutlierGuardConfig};

        println!("=== Plan 224: Outlier-Aware Quantization Guard Demo ===\n");

        // Normal model — no outliers
        let config = OutlierGuardConfig::default();
        let mut guard = OutlierGuard::new(config);

        for layer in 0..4 {
            let weights: Vec<f32> = (0..512)
                .map(|i| {
                    let x = i as f32 / 512.0;
                    (x * 6.28 * (layer as f32 + 1.0)).sin() * 0.3
                })
                .collect();
            let d = guard.scan_layer(&weights, layer, &format!("layer{}.ffn.up", layer));
            println!("  Layer {}: D={:.4}", layer, d);
        }
        let report = guard.report();
        println!(
            "\nNormal model: {} layers, {} flagged",
            report.total_scanned, report.total_flagged
        );

        // Attacked model — outlier injection
        let attack_config = OutlierGuardConfig {
            on_detection: OutlierAction::Warn,
            ..Default::default()
        };
        let mut attack_guard = OutlierGuard::new(attack_config);

        for layer in 0..4 {
            let mut weights: Vec<f32> = (0..512)
                .map(|i| {
                    let x = i as f32 / 512.0;
                    (x * 6.28).sin() * 0.3
                })
                .collect();
            // Inject outliers in layer 2
            if layer == 2 {
                for i in (0..512).step_by(32) {
                    weights[i] = 512.0;
                }
            }
            let d = attack_guard.scan_layer(&weights, layer, &format!("layer{}.ffn.up", layer));
            let status = if d > 0.15 { "FLAGGED" } else { "OK" };
            println!("  Layer {}: D={:.4} [{}]", layer, d, status);
        }
        let report = attack_guard.report();
        println!(
            "\nAttacked model: {} layers, {} flagged, max D={:.4}",
            report.total_scanned, report.total_flagged, report.max_ks_d
        );

        println!("\nDone.");
    }

    #[cfg(not(feature = "outlier_guard"))]
    println!("Enable feature: cargo run --example outlier_guard_demo --features outlier_guard");
}
