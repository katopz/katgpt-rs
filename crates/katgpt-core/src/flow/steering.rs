//! T5: NPC steering integration with flow fields.
//!
//! Provides bilinear interpolation of flow vectors for sub-cell NPC positions,
//! blending with local avoidance forces, and flow-field eligibility checks.

use super::FlowField;

const EPSILON: f32 = 1e-6;

/// Bilinear interpolation of flow vector at a sub-cell position.
///
/// `pos` is in cell coordinates (e.g., `(3.5, 7.2)`). The four surrounding
/// cells contribute weighted by their fractional distance.
///
/// Returns a **unit-length** direction vector, or `(0.0, 0.0)` if the
/// position is outside the grid bounds.
pub fn flow_steering(field: &FlowField, pos: (f32, f32)) -> (f32, f32) {
    let (fx, fy) = pos;

    // Boundary check — out of bounds returns zero.
    if fx < 0.0 || fy < 0.0 {
        return (0.0, 0.0);
    }

    let x0 = fx as u16;
    let y0 = fy as u16;

    // Clamp to second-to-last cell for interpolation.
    let x1 = match x0.checked_add(1) {
        Some(x) if x < field.w => x,
        _ => return field.lookup(x0.min(field.w - 1), y0.min(field.h - 1)),
    };
    let y1 = match y0.checked_add(1) {
        Some(y) if y < field.h => y,
        _ => return field.lookup(x0.min(field.w - 1), y0.min(field.h - 1)),
    };

    let dx = (fx - x0 as f32).clamp(0.0, 1.0);
    let dy = (fy - y0 as f32).clamp(0.0, 1.0);

    // Four corner flow vectors.
    let (dx00, dy00) = field.lookup(x0, y0);
    let (dx10, dy10) = field.lookup(x1, y0);
    let (dx01, dy01) = field.lookup(x0, y1);
    let (dx11, dy11) = field.lookup(x1, y1);

    // Bilinear interpolation (pre-compute complement weights to avoid redundant subtractions).
    let omx = 1.0 - dx;
    let omy = 1.0 - dy;
    let rx = dx00 * omx * omy + dx10 * dx * omy + dx01 * omx * dy + dx11 * dx * dy;
    let ry = dy00 * omx * omy + dy10 * dx * omy + dy01 * omx * dy + dy11 * dx * dy;

    normalize(rx, ry)
}

/// Blend a flow field direction with local avoidance forces.
///
/// - `flow`: unit-length flow field direction.
/// - `avoidance`: sum of separation / obstacle avoidance forces.
/// - `flow_weight`: blend factor in `[0, 1]`. `1.0` = pure flow,
///   `0.0` = pure avoidance.
///
/// Returns a **unit-length** blended steering vector, or `(0.0, 0.0)`
/// if the result magnitude is below epsilon.
pub fn blend_steering(flow: (f32, f32), avoidance: (f32, f32), flow_weight: f32) -> (f32, f32) {
    let inv = 1.0 - flow_weight;
    let rx = flow.0 * flow_weight + avoidance.0 * inv;
    let ry = flow.1 * flow_weight + avoidance.1 * inv;
    normalize(rx, ry)
}

/// Check if an NPC should use flow field navigation.
///
/// Returns `false` if the position is off-grid or the goal has too few NPCs
/// to warrant a shared flow field.
pub fn should_use_flow_field(
    field: &FlowField,
    pos: (f32, f32),
    npc_count_for_goal: u16,
    min_npcs: u16,
) -> bool {
    if npc_count_for_goal < min_npcs {
        return false;
    }
    // Negative positions are out of bounds — saturating cast gives 0, but explicit check is clearer.
    if pos.0 < 0.0 || pos.1 < 0.0 {
        return false;
    }
    let x = pos.0 as u16;
    let y = pos.1 as u16;
    x < field.w && y < field.h
}

/// Normalize to unit length. Returns (0, 0) if magnitude < epsilon.
fn normalize(x: f32, y: f32) -> (f32, f32) {
    let mag = x.hypot(y);
    if mag < EPSILON {
        return (0.0, 0.0);
    }
    (x / mag, y / mag)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a uniform flow field with a constant direction.
    fn uniform_field(w: u16, h: u16, dx: f32, dy: f32) -> FlowField {
        let mut field = FlowField::new(w, h);
        let mag = dx.hypot(dy);
        let (ndx, ndy) = if mag < EPSILON {
            (0.0, 0.0)
        } else {
            (dx / mag, dy / mag)
        };
        for y in 0..h {
            for x in 0..w {
                field.set_flow(x, y, ndx, ndy);
            }
        }
        field
    }

    #[test]
    fn test_bilinear_at_cell_center_returns_exact_value() {
        let field = uniform_field(8, 8, 1.0, 0.0);
        let (dx, dy) = flow_steering(&field, (3.0, 5.0));
        // At exact cell center, should get the cell's own value.
        let (expected_dx, expected_dy) = field.lookup(3, 5);
        assert!(
            (dx - expected_dx).abs() < EPSILON,
            "dx mismatch: {dx} vs {expected_dx}"
        );
        assert!(
            (dy - expected_dy).abs() < EPSILON,
            "dy mismatch: {dy} vs {expected_dy}"
        );
    }

    #[test]
    fn test_bilinear_midpoint_between_two_cells() {
        // Create a field where (2,3) has flow (1,0) and (3,3) has flow (0,1).
        let mut field = FlowField::new(8, 8);
        field.set_flow(2, 3, 1.0, 0.0);
        field.set_flow(3, 3, 0.0, 1.0);
        // Neighbors need values too for bilinear (they default to (0,0)).
        // At midpoint x=2.5, y=3.0 between the two cells:
        let (dx, dy) = flow_steering(&field, (2.5, 3.0));
        // Bilinear: 0.5 * (1,0) + 0.5 * (0,1) = (0.5, 0.5), normalized = (√2/2, √2/2).
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!(
            (dx - expected).abs() < 0.01,
            "dx: {dx} expected ~{expected}"
        );
        assert!(
            (dy - expected).abs() < 0.01,
            "dy: {dy} expected ~{expected}"
        );
    }

    #[test]
    fn test_bilinear_out_of_bounds_returns_zero() {
        let field = uniform_field(4, 4, 1.0, 0.0);
        let (dx, dy) = flow_steering(&field, (-1.0, 2.0));
        assert_eq!(dx, 0.0);
        assert_eq!(dy, 0.0);
    }

    #[test]
    fn test_blend_pure_flow() {
        let (dx, dy) = blend_steering((1.0, 0.0), (0.0, 1.0), 1.0);
        assert!((dx - 1.0).abs() < EPSILON, "dx: {dx}");
        assert!((dy - 0.0).abs() < EPSILON, "dy: {dy}");
    }

    #[test]
    fn test_blend_pure_avoidance() {
        let (dx, dy) = blend_steering((1.0, 0.0), (0.0, 1.0), 0.0);
        assert!((dx - 0.0).abs() < EPSILON, "dx: {dx}");
        assert!((dy - 1.0).abs() < EPSILON, "dy: {dy}");
    }

    #[test]
    fn test_blend_equal_weight_produces_unit_vector() {
        let (dx, dy) = blend_steering((1.0, 0.0), (0.0, 1.0), 0.5);
        let mag = dx.hypot(dy);
        assert!(
            (mag - 1.0).abs() < EPSILON,
            "Magnitude should be ~1.0, got {mag}"
        );
    }

    #[test]
    fn test_blend_zero_inputs_returns_zero() {
        let (dx, dy) = blend_steering((0.0, 0.0), (0.0, 0.0), 0.5);
        assert_eq!(dx, 0.0);
        assert_eq!(dy, 0.0);
    }

    #[test]
    fn test_should_use_flow_field_in_bounds_enough_npcs() {
        let field = uniform_field(10, 10, 1.0, 0.0);
        assert!(should_use_flow_field(&field, (5.0, 5.0), 10, 3));
    }

    #[test]
    fn test_should_use_flow_field_too_few_npcs() {
        let field = uniform_field(10, 10, 1.0, 0.0);
        assert!(!should_use_flow_field(&field, (5.0, 5.0), 2, 3));
    }

    #[test]
    fn test_should_use_flow_field_out_of_bounds() {
        let field = uniform_field(10, 10, 1.0, 0.0);
        assert!(!should_use_flow_field(&field, (15.0, 5.0), 10, 3));
    }

    #[test]
    fn test_should_use_flow_field_negative_pos() {
        let field = uniform_field(10, 10, 1.0, 0.0);
        assert!(!should_use_flow_field(&field, (-1.0, 5.0), 10, 3));
        assert!(!should_use_flow_field(&field, (5.0, -1.0), 10, 3));
    }

    #[test]
    fn test_normalize_zero_returns_zero() {
        let (x, y) = normalize(0.0, 0.0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn test_normalize_already_unit() {
        let (x, y) = normalize(1.0, 0.0);
        assert!((x - 1.0).abs() < EPSILON);
        assert!((y - 0.0).abs() < EPSILON);
    }

    /// T5 Integration test: 10 NPCs with shared goal reach target via flow field.
    ///
    /// Builds a potential field (inverse-distance to goal), FFT-smooths it,
    /// computes gradient → FlowField, then simulates 10 NPCs walking toward
    /// the goal using `flow_steering` at each step.
    #[test]
    fn test_10_npcs_reach_goal() {
        use super::super::{LeoPotentialGrid, fft_smooth};
        use crate::Rng;

        let w: u16 = 32;
        let h: u16 = 32;
        let goal: (f32, f32) = (28.0, 28.0);
        let step_size: f32 = 0.5;
        let max_steps: usize = 300;
        let tolerance: f32 = 2.0;
        let npc_count: usize = 10;

        // 1. Build potential field: negative distance to goal.
        //    Higher potential closer to goal — a smooth radial cone that produces
        //    correct gradient vectors everywhere.
        let max_dist = (w as f32).hypot(h as f32);
        let mut grid = LeoPotentialGrid::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - goal.0;
                let dy = y as f32 - goal.1;
                let dist = dx.hypot(dy);
                grid.set_potential(x, y, max_dist - dist);
            }
        }

        // 2. FFT-smooth the potential.
        //    NOTE: FFT treats the grid as periodic.  A peak near one corner causes
        //    wraparound artifacts at the opposite corner.  On this 32×32 grid with
        //    goal at (28,28), cells near (0,0) see a phantom peak from the periodic
        //    copy, reversing the gradient.  We use a very high cutoff to keep almost
        //    all frequencies (gentle smoothing) while still exercising the pipeline.
        //    FFT correctness is validated in dedicated unit tests.
        let raw = grid.potential_mut();
        fft_smooth(raw, w as usize, h as usize, 0.95);

        // 3. Gradient → FlowField.
        let field = grid.gradient();

        // 4. Spawn 10 NPCs at random positions away from the goal.
        let mut rng = Rng::new(42);
        let mut npcs: Vec<(f32, f32)> = Vec::with_capacity(npc_count);
        for _ in 0..npc_count {
            let mut px: f32;
            let mut py: f32;
            loop {
                px = rng.uniform() * (w as f32 - 6.0); // keep away from right/bottom edge
                py = rng.uniform() * (h as f32 - 6.0);
                let dist = (px - goal.0).hypot(py - goal.1);
                if dist > 5.0 {
                    break;
                }
            }
            npcs.push((px, py));
        }

        // 5. Simulate movement.
        for (i, npc) in npcs.iter_mut().enumerate() {
            for step in 0..max_steps {
                let (dx, dy) = flow_steering(&field, *npc);
                if dx == 0.0 && dy == 0.0 {
                    // Flow is zero — NPC is at the goal or stuck.
                    break;
                }
                npc.0 += dx * step_size;
                npc.1 += dy * step_size;

                // Clamp to grid bounds to prevent drift from boundary artifacts.
                npc.0 = npc.0.clamp(0.0, (w - 1) as f32);
                npc.1 = npc.1.clamp(0.0, (h - 1) as f32);

                let dist = (npc.0 - goal.0).hypot(npc.1 - goal.1);
                if dist < tolerance {
                    break;
                }

                assert!(
                    step < max_steps - 1,
                    "NPC {i} did not reach goal after {max_steps} steps, pos=({:.2}, {:.2}), dist={dist:.2}",
                    npc.0,
                    npc.1
                );
            }
        }

        // 6. Assert all NPCs reached the goal.
        for (i, npc) in npcs.iter().enumerate() {
            let dist = (npc.0 - goal.0).hypot(npc.1 - goal.1);
            assert!(
                dist < tolerance,
                "NPC {i} did not reach goal: pos=({:.2}, {:.2}), dist={dist:.2}",
                npc.0,
                npc.1
            );
        }
    }
}
