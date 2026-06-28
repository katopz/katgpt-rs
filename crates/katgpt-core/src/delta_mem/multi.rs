//! Parallel memory states per domain (δ-mem MSW adaptation).
//!
//! Verified from `delta_impl.py` L795-803 (_reshape_state_heads):
//!   state shape: [batch, num_state_heads, rank, rank]
//!   scan: reshape to [batch*heads, rank, rank], scan independently
//!   reads: per-head einsum, then concat to [batch, seq, state_read_dim]
//!
//! Our adaptation: one DeltaMemoryState per domain from PromptRouter.
//! No learned routing — domain is determined by the request context.

use std::collections::HashMap;

use super::state::{DeltaMemoryConfig, DeltaMemorySnapshot, DeltaMemoryState};

/// Aggregation strategy for cross-domain readouts.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Use only the routed domain's readout (no cross-domain).
    RoutedOnly,
    /// Weight by domain bandit Q-values.
    BanditWeighted,
}

/// Multi-domain associative memory (δ-mem Multi-State Write adaptation).
///
/// Each domain gets its own independent `DeltaMemoryState`.
/// Domains are determined by the PromptRouter, not learned routing.
pub struct MultiDomainMemory {
    /// Per-domain memory states.
    states: HashMap<String, DeltaMemoryState>,
    /// Default config for new states.
    config: DeltaMemoryConfig,
    /// Pre-allocated readout buffer for `read_aggregated` (rank elements).
    readout_buf: Vec<f32>,
    /// Pre-allocated weighted accumulation buffer for `read_aggregated` (rank elements).
    weighted_buf: Vec<f32>,
}

impl MultiDomainMemory {
    /// Create a new multi-domain memory.
    pub fn new(config: DeltaMemoryConfig) -> Self {
        let rank = config.rank;
        Self {
            states: HashMap::new(),
            readout_buf: vec![0.0f32; rank],
            weighted_buf: vec![0.0f32; rank],
            config,
        }
    }

    /// Read from the specified domain's memory state.
    ///
    /// Returns `None` if the domain doesn't exist yet.
    pub fn read_domain(&self, domain: &str, query: &[f32]) -> Option<Vec<f32>> {
        self.states.get(domain).map(|s| s.read(query))
    }

    /// Read from the specified domain's memory state into a pre-allocated buffer.
    ///
    /// Returns `false` if the domain doesn't exist yet.
    pub fn read_domain_into(&self, domain: &str, query: &[f32], out: &mut [f32]) -> bool {
        match self.states.get(domain) {
            Some(s) => {
                s.read_into(query, out);
                true
            }
            None => false,
        }
    }

    /// Write to a domain's memory state.
    ///
    /// Creates the domain if it doesn't exist.
    pub fn write_domain(&mut self, domain: &str, key: &[f32], value: &[f32]) {
        self.ensure_domain(domain);
        if let Some(state) = self.states.get_mut(domain) {
            state.write(key, value);
        }
    }

    /// Get or create a domain's state.
    pub fn ensure_domain(&mut self, domain: &str) {
        if !self.states.contains_key(domain) {
            self.states.insert(
                domain.to_string(),
                DeltaMemoryState::new(self.config.clone()),
            );
        }
    }

    /// Snapshot all domain states.
    pub fn snapshot_all(&self) -> HashMap<String, DeltaMemorySnapshot> {
        self.states
            .iter()
            .map(|(k, v)| (k.clone(), v.snapshot()))
            .collect()
    }

    /// Reset all domain states.
    pub fn reset_all(&mut self) {
        for state in self.states.values_mut() {
            state.reset();
        }
    }

    /// Reset a specific domain's state.
    pub fn reset_domain(&mut self, domain: &str) {
        if let Some(state) = self.states.get_mut(domain) {
            state.reset();
        }
    }

    /// Number of domains with memory states.
    pub fn domain_count(&self) -> usize {
        self.states.len()
    }

    /// Get domain names.
    pub fn domains(&self) -> Vec<&str> {
        self.states.keys().map(|s| s.as_str()).collect()
    }

    /// Get config reference.
    pub fn config(&self) -> &DeltaMemoryConfig {
        &self.config
    }

    /// Read with aggregation strategy across domains.
    ///
    /// `RoutedOnly`: returns the routed domain's readout.
    /// `BanditWeighted`: weighted average of all domain readouts by their update counts.
    ///
    /// Takes `&mut self` because `BanditWeighted` reuses the internal
    /// `readout_buf` / `weighted_buf` scratch buffers (zero-alloc hot path).
    pub fn read_aggregated(
        &mut self,
        domain: &str,
        query: &[f32],
        strategy: AggregationStrategy,
    ) -> Option<Vec<f32>> {
        let rank = self.config.rank;
        match strategy {
            AggregationStrategy::RoutedOnly => {
                let mut out = vec![0.0f32; rank];
                if self.read_domain_into(domain, query, &mut out) {
                    Some(out)
                } else {
                    None
                }
            }
            AggregationStrategy::BanditWeighted => {
                self.weighted_buf[..rank].fill(0.0f32);
                let mut total_weight = 0.0f32;
                let mut routed_exists = false;

                for state in self.states.values() {
                    state.read_into(query, &mut self.readout_buf[..rank]);
                    let weight = state.update_count() as f32 + 1.0;
                    for (i, r) in self.readout_buf[..rank].iter().enumerate() {
                        self.weighted_buf[i] += r * weight;
                    }
                    total_weight += weight;
                    routed_exists = true;
                }

                if !routed_exists {
                    return None;
                }

                if total_weight > 0.0 {
                    for w in self.weighted_buf[..rank].iter_mut() {
                        *w /= total_weight;
                    }
                }
                Some(self.weighted_buf[..rank].to_vec())
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_memory() {
        let mem = MultiDomainMemory::new(DeltaMemoryConfig::default());
        assert_eq!(mem.domain_count(), 0);
        assert!(mem.read_domain("unknown", &[0.0; 8]).is_none());
    }

    #[test]
    fn test_write_creates_domain() {
        let mut mem = MultiDomainMemory::new(DeltaMemoryConfig::default());
        mem.write_domain("coding", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mem.domain_count(), 1);
        assert!(mem.domains().contains(&"coding"));
    }

    #[test]
    fn test_read_write_roundtrip() {
        let mut mem = MultiDomainMemory::new(DeltaMemoryConfig::default());
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let val = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        mem.write_domain("coding", &key, &val);

        let readout = mem.read_domain("coding", &key);
        assert!(readout.is_some());
        let readout = readout.unwrap();
        // After write, readout should be non-zero
        assert!(readout.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_cross_domain_isolation() {
        let mut mem = MultiDomainMemory::new(DeltaMemoryConfig { rank: 4, ..Default::default() });
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let val = vec![0.0, 1.0, 0.0, 0.0];

        // Write to "coding" domain
        mem.write_domain("coding", &key, &val);

        // "math" domain should still be empty
        assert!(mem.read_domain("math", &key).is_none());

        // Ensure "math" and check it's independent
        mem.ensure_domain("math");
        let math_readout = mem.read_domain("math", &key).unwrap();
        assert!(math_readout.iter().all(|&x| x.abs() < 1e-6), "Math domain should start at zero");
    }

    #[test]
    fn test_snapshot_all() {
        let mut mem = MultiDomainMemory::new(DeltaMemoryConfig { rank: 4, ..Default::default() });
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let val = vec![0.0, 1.0, 0.0, 0.0];
        mem.write_domain("a", &key, &val);
        mem.write_domain("b", &key, &val);

        let snapshots = mem.snapshot_all();
        assert_eq!(snapshots.len(), 2);
        assert!(snapshots.contains_key("a"));
        assert!(snapshots.contains_key("b"));
    }

    #[test]
    fn test_reset_all() {
        let mut mem = MultiDomainMemory::new(DeltaMemoryConfig { rank: 4, ..Default::default() });
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let val = vec![0.0, 1.0, 0.0, 0.0];
        mem.write_domain("a", &key, &val);
        mem.write_domain("b", &key, &val);

        mem.reset_all();
        // States should be zeroed
        let readout = mem.read_domain("a", &key).unwrap();
        assert!(readout.iter().all(|&x| x.abs() < 1e-6));
    }

    #[test]
    fn test_read_aggregated_routed_only() {
        let mut mem = MultiDomainMemory::new(DeltaMemoryConfig { rank: 4, ..Default::default() });
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let val = vec![0.0, 1.0, 0.0, 0.0];
        mem.write_domain("coding", &key, &val);
        mem.ensure_domain("math");

        let readout = mem.read_aggregated("coding", &key, AggregationStrategy::RoutedOnly);
        assert!(readout.is_some());
    }

    /// Bit-identity guard: `read_aggregated(_, _, BanditWeighted)` must agree
    /// with the per-domain `read_domain` weighted by `update_count + 1`.
    /// Catches regressions in the scratch-buffer reuse path.
    #[test]
    fn test_read_aggregated_bandit_weighted_matches_naive() {
        let rank = 4;
        let mut mem =
            MultiDomainMemory::new(DeltaMemoryConfig { rank, ..Default::default() });
        let key = vec![1.0, 0.0, 0.0, 0.0];
        let val = vec![0.0, 1.0, 0.0, 0.0];
        mem.write_domain("a", &key, &val);
        mem.write_domain("b", &key, &val);

        // Naive reference implementation using `read_domain` (allocating).
        let mut weighted = vec![0.0f32; rank];
        let mut total_weight = 0.0f32;
        for name in mem.domains() {
            let readout = mem.read_domain(name, &key).unwrap();
            let w = (mem.snapshot_all()[name].update_count as f32) + 1.0;
            for (i, r) in readout.iter().enumerate() {
                weighted[i] += r * w;
            }
            total_weight += w;
        }
        for w in weighted.iter_mut() {
            *w /= total_weight;
        }

        let aggregated = mem
            .read_aggregated("a", &key, AggregationStrategy::BanditWeighted)
            .expect("domains exist");
        assert_eq!(aggregated.len(), rank);
        for (a, b) in aggregated.iter().zip(weighted.iter()) {
            assert!((a - b).abs() < 1e-6, "bandit_weighted drift: {} vs {}", a, b);
        }
    }
}
