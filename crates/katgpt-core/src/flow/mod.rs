//! Fourier-Smoothed Potential Fields for LEO crowd navigation (Plan 242).
//!
//! Maps LEO Q-values onto a 2D spatial grid, applies FFT low-pass
//! smoothing to remove local minima, then extracts a gradient flow
//! field for O(1) per-NPC steering lookups.

mod cache;
mod fft;
pub mod steering;

pub use cache::FlowFieldCache;
pub use fft::{fft_smooth, fft_smooth_into, inflate_obstacles, inflate_obstacles_with_snapshot};
pub use steering::{blend_steering, flow_steering, should_use_flow_field};

// ---------------------------------------------------------------------------
// FlowField — 2D grid of unit flow vectors
// ---------------------------------------------------------------------------

/// 2D grid of flow vectors — preferred movement direction per cell.
///
/// One per goal (or per goal-group for shared goals).
/// Vectors are stored row-major as `[w * h * 2]` with `(dx, dy)` pairs.
pub struct FlowField {
    /// Width in cells.
    pub w: u16,
    /// Height in cells.
    pub h: u16,
    /// Pre-computed row stride in flow elements: `w as usize * 2`.
    /// Avoids recomputing `w * 2` on every lookup/set/is_blocked call.
    stride: usize,
    /// Flow vectors: `[w * h * 2]` — `(dx, dy)` per cell, row-major.
    /// Normalised to unit length or zero for blocked cells.
    flow: Vec<f32>,
}

impl FlowField {
    /// Create a zero-initialised flow field.
    pub fn new(w: u16, h: u16) -> Self {
        let stride = (w as usize) * 2;
        let len = stride * (h as usize);
        Self {
            w,
            h,
            stride,
            flow: vec![0.0f32; len],
        }
    }

    /// O(1) flow-vector lookup. Returns `(0.0, 0.0)` for out-of-bounds.
    #[inline]
    pub fn lookup(&self, x: u16, y: u16) -> (f32, f32) {
        match (x < self.w, y < self.h) {
            (true, true) => {
                let idx = (y as usize) * self.stride + (x as usize) * 2;
                (self.flow[idx], self.flow[idx + 1])
            }
            _ => (0.0, 0.0),
        }
    }

    /// O(1) flow-vector lookup without bounds checks.
    ///
    /// **Safety**: Caller must guarantee `x < self.w` and `y < self.h`.
    #[inline]
    pub(crate) unsafe fn lookup_unchecked(&self, x: u16, y: u16) -> (f32, f32) {
        let idx = (y as usize) * self.stride + (x as usize) * 2;
        unsafe {
            (
                *self.flow.get_unchecked(idx),
                *self.flow.get_unchecked(idx + 1),
            )
        }
    }

    /// O(1) flow-vector write. No-op for out-of-bounds.
    #[inline]
    pub fn set_flow(&mut self, x: u16, y: u16, dx: f32, dy: f32) {
        match (x < self.w, y < self.h) {
            (true, true) => {
                let idx = (y as usize) * self.stride + (x as usize) * 2;
                self.flow[idx] = dx;
                self.flow[idx + 1] = dy;
            }
            _ => {}
        }
    }

    /// A cell is blocked when its flow vector is zero.
    #[inline]
    pub fn is_blocked(&self, x: u16, y: u16) -> bool {
        if x >= self.w || y >= self.h {
            return true;
        }
        let idx = (y as usize) * self.stride + (x as usize) * 2;
        self.flow[idx] == 0.0 && self.flow[idx + 1] == 0.0
    }

    #[inline]
    pub fn width(&self) -> u16 {
        self.w
    }

    #[inline]
    pub fn height(&self) -> u16 {
        self.h
    }
}

// ---------------------------------------------------------------------------
// LeoPotentialGrid — Q-value potential on a 2D grid
// ---------------------------------------------------------------------------

/// Maps LEO Q-values onto a 2D spatial grid for a specific goal.
///
/// Intermediate structure — consumed by FFT smoothing then turned into a
/// [`FlowField`] via [`Self::gradient`].
pub struct LeoPotentialGrid {
    /// Width in cells.
    pub w: u16,
    /// Height in cells.
    pub h: u16,
    /// Q-values: `[w * h]` — max-Q or expected value per cell for one goal.
    potential: Vec<f32>,
    /// Blocked cells (obstacles, walls). Bitfield, 1 = blocked.
    blocked: Vec<u64>,
}

impl LeoPotentialGrid {
    /// Create a zero-potential, all-unblocked grid.
    pub fn new(w: u16, h: u16) -> Self {
        let cells = (w as usize) * (h as usize);
        let words = (cells + 63) / 64;
        Self {
            w,
            h,
            potential: vec![0.0f32; cells],
            blocked: vec![0u64; words],
        }
    }

    /// Build from flat LEO Q-values.
    ///
    /// `q_values` layout: `[w * h * actions_per_cell]`, row-major.
    /// For each cell the *max* over actions becomes the potential.
    pub fn from_q_values(w: u16, h: u16, q_values: &[f32], actions_per_cell: usize) -> Self {
        let cells = (w as usize) * (h as usize);
        assert_eq!(
            q_values.len(),
            cells * actions_per_cell,
            "q_values length mismatch"
        );
        let words = (cells + 63) / 64;

        let mut potential = vec![f32::NEG_INFINITY; cells];
        for cell_idx in 0..cells {
            let start = cell_idx * actions_per_cell;
            let mut max_q = f32::NEG_INFINITY;
            for a in 0..actions_per_cell {
                // Safety: start + a < cells * actions_per_cell == q_values.len()
                let val = unsafe { *q_values.get_unchecked(start + a) };
                if val > max_q {
                    max_q = val;
                }
            }
            potential[cell_idx] = max_q;
        }

        Self {
            w,
            h,
            potential,
            blocked: vec![0u64; words],
        }
    }

    /// Mark a cell as blocked (obstacle).
    #[inline]
    pub fn mark_blocked(&mut self, x: u16, y: u16) {
        match (x < self.w, y < self.h) {
            (true, true) => {
                let cell = (y as usize) * (self.w as usize) + (x as usize);
                let word = cell >> 6;
                let bit = cell & 63;
                self.blocked[word] |= 1u64 << bit;
            }
            _ => {}
        }
    }

    /// Check whether a cell is blocked.
    #[inline]
    pub fn is_blocked(&self, x: u16, y: u16) -> bool {
        match (x < self.w, y < self.h) {
            (true, true) => {
                let cell = (y as usize) * (self.w as usize) + (x as usize);
                let word = cell >> 6;
                let bit = cell & 63;
                self.blocked[word] & (1u64 << bit) != 0
            }
            _ => false,
        }
    }

    /// Read the potential value at `(x, y)`. Returns `0.0` for out-of-bounds.
    #[inline]
    pub fn potential(&self, x: u16, y: u16) -> f32 {
        match (x < self.w, y < self.h) {
            (true, true) => self.potential[(y as usize) * (self.w as usize) + (x as usize)],
            _ => 0.0,
        }
    }

    /// Write a potential value at `(x, y)`. No-op for out-of-bounds.
    #[inline]
    pub fn set_potential(&mut self, x: u16, y: u16, val: f32) {
        match (x < self.w, y < self.h) {
            (true, true) => {
                self.potential[(y as usize) * (self.w as usize) + (x as usize)] = val;
            }
            _ => {}
        }
    }

    /// Compute the negative-gradient flow field via finite differences.
    ///
    /// `dx = V(x+1,y) - V(x-1,y)`, `dy = V(x,y+1) - V(x,y-1)`.
    /// Boundary cells use forward/backward differences.
    /// Blocked cells receive a zero vector. Non-zero vectors are normalised to unit length.
    pub fn gradient(&self) -> FlowField {
        let w = self.w as usize;
        let h = self.h as usize;
        let pot = &self.potential;
        let mut field = FlowField::new(self.w, self.h);

        // Running word/bit counters for blocked bitfield — avoids division per cell.
        let mut word = 0usize;
        let mut bit = 0usize;
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                // Blocked → zero flow (using running counters)
                let is_set = self.blocked[word] & (1u64 << bit) != 0;

                // Advance running counters
                bit += 1;
                if bit == 64 {
                    bit = 0;
                    word += 1;
                }

                if is_set {
                    continue;
                }

                let cell = row + x;
                let v_center = pot[cell];

                // Central differences with boundary fallback (direct index, no bounds check)
                let vx_left = if x > 0 { pot[row + x - 1] } else { v_center };
                let vx_right = if x + 1 < w {
                    pot[row + x + 1]
                } else {
                    v_center
                };
                let vy_up = if y > 0 { pot[row - w + x] } else { v_center };
                let vy_down = if y + 1 < h {
                    pot[row + w + x]
                } else {
                    v_center
                };

                // Gradient (flow toward higher potential)
                let dx = vx_right - vx_left;
                let dy = vy_down - vy_up;
                let len_sq = dx * dx + dy * dy;
                if len_sq < 1e-8f32 {
                    continue; // flat gradient — leave as (0, 0)
                }
                let len = len_sq.sqrt();
                let ndx = dx / len;
                let ndy = dy / len;

                let idx = cell * 2;
                field.flow[idx] = ndx;
                field.flow[idx + 1] = ndy;
            }
        }

        field
    }

    /// Mutable access to the raw potential slice.
    #[inline]
    pub fn potential_mut(&mut self) -> &mut [f32] {
        &mut self.potential
    }

    /// Read-only access to the raw potential slice.
    #[inline]
    pub fn potential_slice(&self) -> &[f32] {
        &self.potential
    }

    /// Read-only access to the blocked bitfield.
    #[inline]
    pub fn blocked(&self) -> &[u64] {
        &self.blocked
    }

    /// Mutable access to the blocked bitfield.
    #[inline]
    pub fn blocked_mut(&mut self) -> &mut [u64] {
        &mut self.blocked
    }
}

// ---------------------------------------------------------------------------
// FlowFieldConfig — tunable parameters
// ---------------------------------------------------------------------------

/// FFT smoothing parameters.
#[derive(Clone, Copy)]
pub struct FlowFieldConfig {
    /// Low-pass cutoff frequency (fraction of Nyquist). Default: 0.25.
    pub cutoff: f32,
    /// Minimum gradient magnitude to produce a flow vector. Default: 1e-4.
    pub min_gradient: f32,
    /// Recompute threshold: how many cells must change to trigger FFT. Default: 5.
    pub dirty_threshold: u16,
    /// Obstacle inflation radius (cells). Default: 1.
    pub obstacle_radius: u8,
}

impl Default for FlowFieldConfig {
    fn default() -> Self {
        Self {
            cutoff: 0.25,
            min_gradient: 1e-4,
            dirty_threshold: 5,
            obstacle_radius: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- FlowField ---

    #[test]
    fn flow_field_lookup_after_set() {
        let mut ff = FlowField::new(4, 4);
        assert_eq!(ff.lookup(0, 0), (0.0, 0.0));

        ff.set_flow(2, 3, 0.6, -0.8);
        assert_eq!(ff.lookup(2, 3), (0.6, -0.8));
    }

    #[test]
    fn flow_field_oob_returns_zero() {
        let ff = FlowField::new(4, 4);
        assert_eq!(ff.lookup(4, 0), (0.0, 0.0));
        assert_eq!(ff.lookup(0, 4), (0.0, 0.0));
        assert_eq!(ff.lookup(255, 255), (0.0, 0.0));
    }

    #[test]
    fn flow_field_is_blocked_zero_vector() {
        let mut ff = FlowField::new(4, 4);
        assert!(ff.is_blocked(0, 0)); // zero = blocked

        ff.set_flow(1, 1, 1.0, 0.0);
        assert!(!ff.is_blocked(1, 1));
    }

    #[test]
    fn flow_field_set_oob_is_noop() {
        let mut ff = FlowField::new(2, 2);
        ff.set_flow(5, 5, 1.0, 1.0); // should not panic
        assert_eq!(ff.lookup(0, 0), (0.0, 0.0));
    }

    // --- LeoPotentialGrid ---

    #[test]
    fn potential_grid_new_is_zero() {
        let g = LeoPotentialGrid::new(3, 3);
        for y in 0..3u16 {
            for x in 0..3u16 {
                assert_eq!(g.potential(x, y), 0.0);
            }
        }
    }

    #[test]
    fn potential_grid_from_q_values_max() {
        // 2×2 grid, 3 actions per cell
        let q = vec![
            // cell (0,0): max = 0.5
            0.1, 0.5, 0.3, // cell (1,0): max = 0.9
            0.9, 0.2, 0.1, // cell (0,1): max = 0.7
            0.7, 0.4, 0.2, // cell (1,1): max = 0.6
            0.0, 0.6, 0.3,
        ];
        let g = LeoPotentialGrid::from_q_values(2, 2, &q, 3);
        assert!((g.potential(0, 0) - 0.5).abs() < 1e-6);
        assert!((g.potential(1, 0) - 0.9).abs() < 1e-6);
        assert!((g.potential(0, 1) - 0.7).abs() < 1e-6);
        assert!((g.potential(1, 1) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn potential_grid_blocked_roundtrip() {
        let mut g = LeoPotentialGrid::new(8, 8);
        assert!(!g.is_blocked(3, 5));
        g.mark_blocked(3, 5);
        assert!(g.is_blocked(3, 5));
        assert!(!g.is_blocked(3, 4)); // neighbor not blocked
    }

    #[test]
    fn potential_grid_blocked_oob() {
        let g = LeoPotentialGrid::new(4, 4);
        assert!(!g.is_blocked(4, 0));
        assert!(!g.is_blocked(0, 4));
    }

    #[test]
    fn gradient_points_toward_higher_potential() {
        // 5×5 grid with peak at center (2,2).
        // Set a gradient slope: potential increases toward center.
        let mut g = LeoPotentialGrid::new(5, 5);
        for y in 0..5u16 {
            for x in 0..5u16 {
                let dx = (x as f32 - 2.0).abs();
                let dy = (y as f32 - 2.0).abs();
                g.set_potential(x, y, 10.0 - dx - dy);
            }
        }

        let ff = g.gradient();
        let (dx, dy) = ff.lookup(1, 1);

        // At (1,1): V(0,1)=8, V(2,1)=10 → dx = 10-8 = 2 > 0
        // At (1,1): V(1,0)=8, V(1,2)=10 → dy = 10-8 = 2 > 0
        assert!(dx > 0.0, "dx at (1,1) should be positive, got {dx}");
        assert!(dy > 0.0, "dy at (1,1) should be positive, got {dy}");

        // Should be unit length
        let len = (dx * dx + dy * dy).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "gradient should be unit, len={len}"
        );
    }

    #[test]
    fn gradient_blocked_cell_gets_zero() {
        let mut g = LeoPotentialGrid::new(3, 3);
        g.set_potential(1, 1, 10.0);
        g.mark_blocked(0, 0);

        let ff = g.gradient();
        assert!(ff.is_blocked(0, 0));
    }

    #[test]
    fn gradient_flat_field_is_zero() {
        let g = LeoPotentialGrid::new(4, 4);
        let ff = g.gradient();
        for y in 0..4u16 {
            for x in 0..4u16 {
                assert!(ff.is_blocked(x, y), "flat field should be all blocked/zero");
            }
        }
    }

    // --- FlowFieldConfig ---

    #[test]
    fn config_default_values() {
        let c = FlowFieldConfig::default();
        assert!((c.cutoff - 0.25).abs() < 1e-6);
        assert_eq!(c.obstacle_radius, 1);
        assert!((c.min_gradient - 1e-4).abs() < 1e-8);
        assert_eq!(c.dirty_threshold, 5);
    }
}
