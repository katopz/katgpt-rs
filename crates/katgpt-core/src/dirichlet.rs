//! Dirichlet Energy structural alignment diagnostic — core computation.
//!
//! E(E) = Σ_{i,j} A_{ij} ‖h_{e_i} - h_{e_j}‖² over embeddings w.r.t. a
//! sparse adjacency graph. Lower energy means structurally aligned entities.

// ── Core computation ──────────────────────────────────────────

/// Compute Dirichlet Energy over embeddings w.r.t. adjacency graph.
///
/// E(E) = Σ_{i,j} A_{ij} ‖h_{e_i} - h_{e_j}‖²
///
/// Lower energy = more structurally aligned (entities connected by edges
/// have similar embeddings).
///
/// # Arguments
/// * `embeddings` — flat slice of embeddings, shape [n_entities × dim]
/// * `dim` — embedding dimension
/// * `adjacency` — sparse adjacency pairs [(i, j), ...] where A_{ij} = 1
///
/// # Returns
/// Total Dirichlet Energy (f32).
///
/// # Panics
/// Panics if any adjacency index is out of bounds for the embedding matrix.
pub fn dirichlet_energy(embeddings: &[f32], dim: usize, adjacency: &[(usize, usize)]) -> f32 {
    let n_entities = embeddings.len() / dim;
    let mut energy = 0.0f32;

    for &(i, j) in adjacency {
        debug_assert!(
            i < n_entities,
            "adjacency index {i} out of bounds ({n_entities} entities)"
        );
        debug_assert!(
            j < n_entities,
            "adjacency index {j} out of bounds ({n_entities} entities)"
        );

        let row_i = i * dim;
        let row_j = j * dim;
        let mut dist_sq = 0.0f32;
        for d in 0..dim {
            let diff = embeddings[row_i + d] - embeddings[row_j + d];
            dist_sq += diff * diff;
        }
        energy += dist_sq;
    }

    energy
}

// ── Adjacency construction helpers ────────────────────────────

/// Build functor adjacency from paired entity indices.
///
/// For N pairs (a_i, b_i), creates edges: (a_0, b_0), (a_1, b_1), ...
/// This is the paper's A_{ij} = 1 iff entities i,j are related by functor.
pub fn functor_adjacency(pairs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    pairs.to_vec()
}

/// Build position-neighbor adjacency from consecutive positions.
///
/// For positions 0..n, creates edges (p, p+1) for each consecutive pair.
/// This is the default structural graph when no domain-specific adjacency
/// is available.
pub fn consecutive_adjacency(n_positions: usize) -> Vec<(usize, usize)> {
    (0..n_positions.saturating_sub(1))
        .map(|p| (p, p + 1))
        .collect()
}

// ── KV cache probe ────────────────────────────────────────────

/// Probe KV cache key embeddings for structural alignment.
///
/// Computes Dirichlet Energy over KV cache keys at a given layer,
/// using position-adjacency (user-specified pairs).
///
/// # Arguments
/// * `keys` — flat slice of key embeddings, shape [n_positions × kv_dim]
/// * `kv_dim` — dimension of each key vector
/// * `adjacency` — sparse adjacency pairs
///
/// # Returns
/// (energy, normalized_energy) where normalized = energy / max(n_edges, 1).
pub fn kv_cache_dirichlet_energy(
    keys: &[f32],
    kv_dim: usize,
    adjacency: &[(usize, usize)],
) -> (f32, f32) {
    let energy = dirichlet_energy(keys, kv_dim, adjacency);
    let n_edges = adjacency.len().max(1) as f32;
    (energy, energy / n_edges)
}
