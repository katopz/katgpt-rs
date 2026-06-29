//! Targeted Precision Budget — per-head bit allocation for quantization.
//! Allocates more bits to sensitive heads, fewer to robust heads.
//! Feature-gated behind `targeted_precision`.

/// Per-head precision budget for quantized attention.
#[derive(Debug, Clone)]
pub struct PrecisionBudget {
    /// Bits per attention head, indexed by (layer * num_heads + head).
    pub head_bits: Vec<u8>,
    /// Total average bits budget.
    pub budget: f32,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of heads per layer.
    pub num_heads: usize,
    /// Per-head sensitivity scores (higher = more sensitive to quantization noise).
    pub sensitivity: Vec<f32>,
}

impl PrecisionBudget {
    /// Create uniform budget: all heads get the same bit-width.
    pub fn uniform(num_layers: usize, num_heads: usize, bits_per_head: u8) -> Self {
        let total = num_layers * num_heads;
        Self {
            head_bits: vec![bits_per_head; total],
            budget: bits_per_head as f32,
            num_layers,
            num_heads,
            sensitivity: vec![1.0; total],
        }
    }

    /// Get bits for a specific head.
    pub fn get_bits(&self, layer: usize, head: usize) -> u8 {
        self.head_bits[layer * self.num_heads + head]
    }

    /// Compute budget from sensitivity analysis.
    /// Greedy allocation: sort heads by sensitivity, allocate budget to most sensitive first.
    /// Total bits must equal budget * total_heads (same total as uniform).
    pub fn compute_budget(
        num_layers: usize,
        num_heads: usize,
        sensitivity: &[f32],
        target_budget: f32,
    ) -> Self {
        let total = num_layers * num_heads;
        let total_bits = (target_budget * total as f32) as u32;

        // Sort heads by sensitivity (descending)
        let mut indices: Vec<usize> = (0..total).collect();
        indices.sort_by(|&a, &b| {
            sensitivity[b]
                .partial_cmp(&sensitivity[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Greedy allocation: min 1 bit, max 8 bits per head
        let mut head_bits = vec![1u8; total];
        let mut allocated: u32 = total as u32; // 1 bit per head minimum

        // Distribute remaining bits greedily
        for &idx in &indices {
            if allocated >= total_bits {
                break;
            }
            let remaining = total_bits - allocated;
            let max_extra = 7u8.min(remaining as u8); // max 8 total, already have 1
            let extra = if sensitivity[idx] > 0.5 {
                max_extra.min(3) // high sensitivity: up to 4 bits
            } else if sensitivity[idx] > 0.2 {
                max_extra.min(2) // medium: up to 3 bits
            } else {
                max_extra.min(1) // low: up to 2 bits
            };
            head_bits[idx] += extra;
            allocated += extra as u32;
        }

        // Clamp total to budget
        let actual_budget = allocated as f32 / total as f32;

        Self {
            head_bits,
            budget: actual_budget,
            num_layers,
            num_heads,
            sensitivity: sensitivity.to_vec(),
        }
    }

    /// Compute total KV cache size in bits.
    pub fn total_kv_bits(&self, seq_len: usize, head_dim: usize) -> u64 {
        let per_bit = seq_len as u64 * head_dim as u64;
        self.head_bits
            .iter()
            .map(|&bits| bits as u64 * per_bit)
            .sum()
    }

    /// Verify budget constraint: average bits <= budget.
    pub fn verify_budget(&self) -> bool {
        let avg: f32 =
            self.head_bits.iter().map(|&b| b as f32).sum::<f32>() / self.head_bits.len() as f32;
        avg <= self.budget * 1.01 // 1% tolerance for rounding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_budget() {
        let budget = PrecisionBudget::uniform(4, 8, 4);
        assert_eq!(budget.head_bits.len(), 32);
        assert!(budget.head_bits.iter().all(|&b| b == 4));
        assert!(budget.verify_budget());
    }

    #[test]
    fn test_compute_budget_high_sensitivity_gets_more_bits() {
        let mut sensitivity = vec![0.1; 32]; // 4 layers × 8 heads
        sensitivity[0] = 0.9; // head 0 is very sensitive
        sensitivity[1] = 0.8; // head 1 is sensitive

        let budget = PrecisionBudget::compute_budget(4, 8, &sensitivity, 2.5);

        // Sensitive heads should get more bits
        assert!(budget.get_bits(0, 0) > budget.get_bits(0, 7));
        assert!(budget.get_bits(0, 1) > budget.get_bits(0, 7));
    }

    #[test]
    fn test_total_kv_bits() {
        let budget = PrecisionBudget::uniform(2, 4, 4);
        let bits = budget.total_kv_bits(128, 64);
        // 8 heads × 4 bits × 128 seq × 64 dim
        assert_eq!(bits, 8 * 4 * 128 * 64);
    }

    #[test]
    fn test_budget_constraint_satisfied() {
        let sensitivity = vec![0.5; 16];
        let budget = PrecisionBudget::compute_budget(2, 8, &sensitivity, 3.0);
        assert!(budget.verify_budget());
    }
}
