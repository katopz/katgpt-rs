//! IdentityEdge — no-op edge, baseline for GOAT gate 1.

use super::traits::DenseEdge;
use super::types::{DenseHidden, MeshScratch};

/// An edge that passes the predecessor output through unchanged.
///
/// With topology `[1, 1, 1]` + `IdentityEdge`, the DenseMesh must produce
/// identical output to a vanilla single-pass `forward()` (gate 1 correctness).
pub struct IdentityEdge;

impl IdentityEdge {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for IdentityEdge {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseEdge for IdentityEdge {
    fn route_into(&self, from: &DenseHidden, scratch: &mut MeshScratch) {
        // Copy `from` into scratch.edge_output.
        debug_assert_eq!(from.len(), scratch.edge_output.len());
        scratch.edge_output.data.copy_from_slice(&from.data);
    }

    fn cost_hint(&self) -> f32 {
        0.0
    }

    fn name(&self) -> &str {
        "identity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_edge_copies_input() {
        let mut from = DenseHidden::zeros(2, 4);
        for (i, v) in from.rows_mut().iter_mut().enumerate() {
            *v = i as f32;
        }
        let mut scratch = MeshScratch::new(2, 4);
        let edge = IdentityEdge::new();
        edge.route_into(&from, &mut scratch);
        // Output matches input.
        for (i, v) in scratch.edge_output.rows().iter().enumerate() {
            assert_eq!(*v, i as f32);
        }
    }
}
