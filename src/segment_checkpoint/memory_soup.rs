//! Memory-Soup DDTree Branch Merging.
//! At DDTree leaf evaluation, computes γ-weighted average of cached branch KV states.
//! Feature-gated behind `memory_soup_dtree`.

/// γ-weighted branch state for DDTree leaf evaluation.
#[derive(Debug, Clone)]
pub struct BranchState {
    /// Segment IDs contributing to this branch.
    pub segment_ids: Vec<u32>,
    /// γ gate values for each segment.
    pub gates: Vec<f32>,
    /// Weighted average of cached branch KV states.
    pub weighted_state: Vec<f32>,
}

/// Compute γ-weighted average of cached branch states.
/// Uses sigmoid gates (NOT softmax) per project convention.
pub fn compute_branch_state(
    branch_kv: &[Vec<f32>],               // KV states from current branch
    cached_states: &[(u32, &[f32], f32)], // (segment_id, cached_kv, gate)
) -> BranchState {
    if branch_kv.is_empty() && cached_states.is_empty() {
        return BranchState {
            segment_ids: Vec::new(),
            gates: Vec::new(),
            weighted_state: Vec::new(),
        };
    }

    // Need at least one branch KV vector to determine dimension.
    if branch_kv.is_empty() {
        // Only cached states — use first cached_kv for dimension.
        let dim = cached_states[0].1.len();
        let mut weighted_state = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;
        let mut segment_ids = Vec::new();
        let mut gates = Vec::new();

        for &(id, cached_kv, gate) in cached_states {
            segment_ids.push(id);
            gates.push(gate);
            total_weight += gate;
            for (i, &v) in cached_kv.iter().enumerate().take(dim) {
                weighted_state[i] += gate * v;
            }
        }

        if total_weight > 0.0 {
            for v in weighted_state.iter_mut() {
                *v /= total_weight;
            }
        }

        return BranchState {
            segment_ids,
            gates,
            weighted_state,
        };
    }

    let dim = branch_kv[0].len();
    let mut weighted_state = vec![0.0f32; dim];
    let mut total_weight = 0.0f32;

    let mut segment_ids = Vec::new();
    let mut gates = Vec::new();

    // Current branch contribution (weight = 1.0)
    for kv in branch_kv {
        for (i, &v) in kv.iter().enumerate().take(dim) {
            weighted_state[i] += v;
        }
    }
    let branch_count = branch_kv.len() as f32;
    total_weight += branch_count;

    // Cached segment contributions
    for &(id, cached_kv, gate) in cached_states {
        segment_ids.push(id);
        gates.push(gate);
        total_weight += gate;
        for (i, &v) in cached_kv.iter().enumerate().take(dim) {
            weighted_state[i] += gate * v;
        }
    }

    // Normalize
    if total_weight > 0.0 {
        for v in weighted_state.iter_mut() {
            *v /= total_weight;
        }
    }

    BranchState {
        segment_ids,
        gates,
        weighted_state,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_branch_state_empty() {
        let state = compute_branch_state(&[], &[]);
        assert!(state.weighted_state.is_empty());
    }

    #[test]
    fn test_branch_state_single_branch() {
        let branch = vec![vec![1.0, 2.0, 3.0]];
        let state = compute_branch_state(&branch, &[]);
        assert!((state.weighted_state[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_branch_state_with_cached() {
        let branch = vec![vec![2.0, 4.0]];
        let cached = vec![1.0, 2.0];
        let gate = 0.5f32;
        let state = compute_branch_state(&branch, &[(0u32, cached.as_slice(), gate)]);
        // (2.0 + 0.5*1.0) / 1.5 = 1.667, (4.0 + 0.5*2.0) / 1.5 = 3.333
        assert!((state.weighted_state[0] - (2.0 + 0.5) / 1.5).abs() < 1e-5);
        assert!((state.weighted_state[1] - (4.0 + 1.0) / 1.5).abs() < 1e-5);
    }
}
