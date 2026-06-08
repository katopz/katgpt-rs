//! Plan 222 Example: Critical Interval Solver Switching Demo

fn main() {
    #[cfg(feature = "critical_interval_gate")]
    {
        use katgpt_rs::dllm_solver::{
            CriticalIntervalConfig, is_critical_interval, select_solver, shannon_entropy,
        };

        println!("=== Plan 222: Critical Interval Solver Switching Demo ===\n");

        let config = CriticalIntervalConfig::new(1000);
        println!(
            "Config: vocab_size={}, H_critical={:.4}",
            config.vocab_size, config.h_critical
        );

        // Low entropy (peaked distribution)
        let mut peaked = vec![0.001f32; 1000];
        peaked[0] = 0.5;
        peaked[1] = 0.3;
        let h_low = shannon_entropy(&peaked);
        println!(
            "\nPeaked distribution: H={:.4}, critical={}",
            h_low,
            is_critical_interval(h_low, &config)
        );
        println!("  Solver: {:?}", select_solver(h_low, &config));

        // High entropy (uniform distribution)
        let uniform = vec![0.001f32; 1000];
        let h_high = shannon_entropy(&uniform);
        println!(
            "\nUniform distribution: H={:.4}, critical={}",
            h_high,
            is_critical_interval(h_high, &config)
        );
        println!("  Solver: {:?}", select_solver(h_high, &config));

        println!("\nDone.");
    }

    #[cfg(not(feature = "critical_interval_gate"))]
    println!(
        "Enable feature: cargo run --example critical_interval_demo --features critical_interval_gate"
    );
}
