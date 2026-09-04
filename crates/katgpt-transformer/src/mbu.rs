//! Memory Bandwidth Utilization (MBU) diagnostics for transformer forward passes.
//!
//! Measures how efficiently the forward pass uses memory bandwidth by tracking
//! bytes read from weight buffers and comparing against theoretical peak.
//!
//! Plan 160 (Kog CPU Monokernel Fusion) — task T9.

use katgpt_core::types::{Config, kv_dim};

// ---------------------------------------------------------------------------
// MbuCounter — accumulates bytes read during a forward pass
// ---------------------------------------------------------------------------

/// Tracks cumulative bytes read from weight buffers per forward pass.
pub struct MbuCounter {
    bytes_read: u64,
}

impl MbuCounter {
    pub fn new() -> Self {
        Self { bytes_read: 0 }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.bytes_read = 0;
    }

    #[inline]
    pub fn add_bytes(&mut self, n: u64) {
        self.bytes_read += n;
    }

    /// Adds `weights.len() * 4` bytes (each f32 is 4 bytes).
    #[inline]
    pub fn add_weight_matrix(&mut self, weights: &[f32]) {
        self.bytes_read += weights.len() as u64 * 4;
    }
}

impl Default for MbuCounter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MbuReport — formatted MBU report for a benchmark run
// ---------------------------------------------------------------------------

/// Formatted MBU report for a benchmark run.
///
/// Field order: u64 pairs grouped before Duration (12 bytes) to eliminate padding.
pub struct MbuReport {
    pub bytes_read: u64,
    pub tokens_generated: u64,
    pub elapsed: std::time::Duration,
}

impl MbuReport {
    pub fn bytes_per_token(&self) -> u64 {
        self.bytes_read
            .checked_div(self.tokens_generated)
            .unwrap_or(0)
    }

    /// Achieved bandwidth in GB/s.
    pub fn bandwidth_gbps(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 {
            return 0.0;
        }
        let gb = self.bytes_read as f64 / 1e9;
        gb / secs
    }

    /// Percentage of peak memory bandwidth utilized.
    pub fn mbu_percent(&self, peak_bandwidth_gbps: f64) -> f64 {
        if peak_bandwidth_gbps == 0.0 {
            return 0.0;
        }
        (self.bandwidth_gbps() / peak_bandwidth_gbps) * 100.0
    }

    /// Formatted report string for benchmark output.
    pub fn fmt_report(&self, peak_gbps: f64) -> String {
        let bpt = self.bytes_per_token();
        let bw = self.bandwidth_gbps();
        let mbu = self.mbu_percent(peak_gbps);
        format!(
            "MBU Report: {:.2} GB/s achieved / {:.0} GB/s peak ({:.1}% utilization), {} bytes/token, {} tokens in {:?}",
            bw, peak_gbps, mbu, bpt, self.tokens_generated, self.elapsed
        )
    }
}

// ---------------------------------------------------------------------------
// Weight byte calculations
// ---------------------------------------------------------------------------

/// Bytes read per transformer layer from weight matrices.
pub fn per_layer_weight_bytes(config: &Config) -> u64 {
    let n = config.n_embd as u64;
    let kv = kv_dim(config) as u64;
    let mlp = config.mlp_hidden as u64;
    let f4: u64 = 4; // sizeof(f32)

    // QKV weights
    let qkv = (n * n + 2 * kv * n) * f4;
    // Output projection
    let wo = n * n * f4;
    // MLP: up-projection + down-projection
    let mlp_weights = (mlp * n + n * mlp) * f4;

    qkv + wo + mlp_weights
}

/// Total weight bytes read per generated token (all layers + embeddings + lm_head).
pub fn per_token_weight_bytes(config: &Config) -> u64 {
    let n = config.n_embd as u64;
    let v = config.vocab_size as u64;
    let f4: u64 = 4;

    let layers = per_layer_weight_bytes(config) * config.n_layer as u64;
    let token_embed = n * f4;
    let pos_embed = n * f4;
    let lm_head = v * n * f4;

    layers + token_embed + pos_embed + lm_head
}

/// Theoretical peak memory bandwidth for the platform (GB/s).
///
/// Conservative default: 100 GB/s (Apple Silicon M-series baseline).
pub fn peak_bandwidth_gbps() -> f64 {
    100.0
}

/// Formats a single benchmark result line.
pub fn fmt_benchmark_line(label: &str, tok_s: f64, ms_tok: f64, mbu_pct: f64) -> String {
    format!("{label:<20} {tok_s:>8.1} tok/s  {ms_tok:>7.2} ms/tok  MBU {mbu_pct:>5.1}%")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_per_token_weight_bytes_micro() {
        let config = Config::micro();
        // micro: n_embd=16, n_head=4, head_dim=4, n_kv_head=4, n_layer=1,
        //         vocab_size=27, mlp_hidden=64
        // kv_dim = 4 * 4 = 16
        //
        // Per-layer:
        //   wq: 16*16*4 = 1024
        //   wk: 16*16*4 = 1024
        //   wv: 16*16*4 = 1024
        //   wo: 16*16*4 = 1024
        //   w1: 64*16*4 = 4096
        //   w2: 16*64*4 = 4096
        //   total: 12288
        //
        // Per-token (1 layer):
        //   layers: 12288
        //   token_embed: 16*4 = 64
        //   pos_embed:   16*4 = 64
        //   lm_head: 27*16*4 = 1728
        //   total: 14144

        assert_eq!(kv_dim(&config), 16);
        assert_eq!(per_layer_weight_bytes(&config), 12_288);
        assert_eq!(per_token_weight_bytes(&config), 14_144);
    }

    #[test]
    fn test_mbu_counter() {
        let mut counter = MbuCounter::new();
        assert_eq!(counter.bytes_read, 0);

        counter.add_bytes(100);
        assert_eq!(counter.bytes_read, 100);

        let weights = [1.0f32, 2.0, 3.0, 4.0];
        counter.add_weight_matrix(&weights);
        assert_eq!(counter.bytes_read, 116); // 100 + 4*4

        counter.reset();
        assert_eq!(counter.bytes_read, 0);
    }

    #[test]
    fn test_mbu_report_bandwidth() {
        let report = MbuReport {
            bytes_read: 1_000_000_000,                        // 1 GB
            elapsed: std::time::Duration::from_secs_f64(0.1), // 100ms
            tokens_generated: 10,
        };
        // 1 GB in 0.1 s = 10 GB/s
        assert!((report.bandwidth_gbps() - 10.0).abs() < 0.01);
        assert_eq!(report.bytes_per_token(), 100_000_000);
        // 10 GB/s / 100 GB/s peak = 10%
        assert!((report.mbu_percent(100.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fmt_benchmark_line() {
        let line = fmt_benchmark_line("prompt", 50.0, 20.0, 45.3);
        assert!(line.contains("prompt"));
        assert!(line.contains("50.0 tok/s"));
        assert!(line.contains("20.00 ms/tok"));
        assert!(line.contains("45.3%"));
    }
}
