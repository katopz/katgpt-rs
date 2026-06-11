//! T4: Per-goal shared flow field cache.
//!
//! Maps `goal_id → FlowField`. Recomputes when dirty cells exceed threshold.
//! Main entry point: [`FlowFieldCache::get_or_compute`].

use std::collections::HashMap;

use rustfft::num_complex::Complex;

use super::{FlowField, FlowFieldConfig, LeoPotentialGrid, fft_smooth_into, inflate_obstacles};
use crate::traits::LeoHead;

/// Per-goal cached flow field with dirty-tracking metadata.
struct CachedField {
    field: FlowField,
    /// Number of cells changed since last compute.
    dirty_count: u16,
    /// Tick at which this field was last (re)computed.
    last_tick: u64,
}

/// Per-goal shared flow field cache.
///
/// Maps `goal_id → FlowField`. Recomputes when dirty cells exceed threshold.
/// Use [`FlowFieldCache::get_or_compute`] as the main entry point — it
/// orchestrates the full pipeline: Q-value extraction → grid → FFT → gradient.
pub struct FlowFieldCache {
    /// `goal_id → (FlowField, dirty_count, last_computed_tick)`.
    fields: HashMap<u64, CachedField>,
    config: FlowFieldConfig,
    /// Minimum NPCs sharing a goal to warrant a flow field.
    min_npcs: u16,
    /// Pre-allocated FFT scratch buffers — reused across calls to avoid per-compute allocation.
    fft_buf: Vec<Complex<f32>>,
    fft_col_buf: Vec<Complex<f32>>,
    /// Pre-allocated potential buffer — reused across calls.
    potential_buf: Vec<f32>,
    /// Pre-allocated blocked bitfield buffer.
    blocked_buf: Vec<u64>,
}

impl FlowFieldCache {
    /// Create a new cache with the given FFT smoothing config.
    pub fn new(config: FlowFieldConfig) -> Self {
        Self {
            fields: HashMap::new(),
            config,
            min_npcs: 1,
            fft_buf: Vec::new(),
            fft_col_buf: Vec::new(),
            potential_buf: Vec::new(),
            blocked_buf: Vec::new(),
        }
    }

    /// Set the minimum number of NPCs sharing a goal to warrant a shared flow field.
    pub fn with_min_npcs(mut self, min: u16) -> Self {
        self.min_npcs = min;
        self
    }

    /// Get a cached flow field for a goal, if present and not dirty.
    ///
    /// Returns `None` if not cached or if the field needs recompute (dirty).
    /// Use [`get_or_compute`](Self::get_or_compute) for automatic recompute.
    pub fn get(&self, goal_id: u64) -> Option<&FlowField> {
        match self.fields.get(&goal_id) {
            Some(cached) if cached.dirty_count == 0 => Some(&cached.field),
            _ => None,
        }
    }

    /// Mark cells dirty for a goal (obstacle changed).
    ///
    /// Increments the dirty count. If it reaches or exceeds the configured
    /// threshold, the entry is invalidated — the next `get_or_compute` will
    /// rebuild it from scratch.
    pub fn mark_dirty(&mut self, goal_id: u64, cells_changed: u16) {
        match self.fields.get_mut(&goal_id) {
            Some(cached) => {
                cached.dirty_count = cached.dirty_count.saturating_add(cells_changed);
                if cached.dirty_count >= self.config.dirty_threshold {
                    self.fields.remove(&goal_id);
                }
            }
            None => {
                // Not cached — nothing to dirty.
            }
        }
    }

    /// Get or compute a flow field for a goal.
    ///
    /// - `head` provides Q-values via the LEO framework.
    /// - `state` is the current observation state.
    /// - `goal_idx` selects which goal's Q-slice to use.
    /// - `grid_w`, `grid_h` define the spatial grid dimensions.
    /// - `tick` is the current simulation tick (used to avoid redundant recomputes).
    /// - `npc_count` is how many NPCs share this goal — returns `None` if < `min_npcs`.
    ///
    /// Returns `None` if:
    /// - `npc_count < min_npcs` (individual LEO should be used instead).
    /// - The goal index is out of range for the head.
    pub fn get_or_compute<H: LeoHead>(
        &mut self,
        goal_id: u64,
        head: &H,
        state: &[f32],
        goal_idx: usize,
        grid_w: u16,
        grid_h: u16,
        tick: u64,
        npc_count: u16,
    ) -> Option<&FlowField> {
        // Skip if too few NPCs share this goal.
        if npc_count < self.min_npcs {
            return None;
        }

        // Check for a valid cached entry.
        let needs_recompute = match self.fields.get(&goal_id) {
            Some(cached) => cached.dirty_count > 0 || cached.last_tick != tick,
            None => true,
        };
        if !needs_recompute {
            return self.fields.get(&goal_id).map(|c| &c.field);
        }

        // Validate goal index.
        if goal_idx >= head.goal_count() {
            return None;
        }

        // Extract Q-values for this goal.
        let all_q = head.all_goals_q(state);
        let goal_q = head.q_for_goal(&all_q, goal_idx);

        // Derive actions_per_cell from total action count and grid dimensions.
        let total_cells = (grid_w as usize) * (grid_h as usize);
        let actions_per_cell = match total_cells {
            0 => return None,
            cells => head.action_count() / cells,
        };
        if actions_per_cell == 0 {
            return None;
        }

        // Build potential grid from Q-values.
        let mut grid = LeoPotentialGrid::from_q_values(grid_w, grid_h, goal_q, actions_per_cell);

        // Inflate obstacles before FFT to prevent flow into walls.
        let words_per_row = ((grid_w as usize) + 63) / 64;
        let blocked_words = words_per_row * (grid_h as usize);

        // Resize scratch buffers if grid dimensions changed.
        self.blocked_buf.resize(blocked_words, 0u64);
        self.blocked_buf[..blocked_words].fill(0);

        // Copy blocked state into bitfield for inflation.
        for y in 0..grid_h {
            for x in 0..grid_w {
                if grid.is_blocked(x, y) {
                    let word = (y as usize) * words_per_row + (x as usize) / 64;
                    let bit = (x as usize) % 64;
                    self.blocked_buf[word] |= 1u64 << bit;
                }
            }
        }

        inflate_obstacles(
            &mut self.blocked_buf,
            grid_w,
            grid_h,
            self.config.obstacle_radius,
        );

        // Apply inflated obstacles back to the grid.
        for y in 0..grid_h {
            for x in 0..grid_w {
                let word = (y as usize) * words_per_row + (x as usize) / 64;
                let bit = (x as usize) % 64;
                if self.blocked_buf[word] & (1u64 << bit) != 0 {
                    grid.mark_blocked(x, y);
                }
            }
        }

        // FFT smooth the potential field to remove local minima — reuse scratch buffers.
        self.potential_buf.resize(total_cells, 0.0);
        for y in 0..grid_h {
            for x in 0..grid_w {
                self.potential_buf[(y as usize) * (grid_w as usize) + (x as usize)] = grid.potential(x, y);
            }
        }

        fft_smooth_into(
            &mut self.potential_buf,
            grid_w as usize,
            grid_h as usize,
            self.config.cutoff,
            &mut self.fft_buf,
            &mut self.fft_col_buf,
        );

        // Write smoothed values back.
        for y in 0..grid_h {
            for x in 0..grid_w {
                let idx = (y as usize) * (grid_w as usize) + (x as usize);
                grid.set_potential(x, y, self.potential_buf[idx]);
            }
        }

        // Compute gradient → FlowField.
        let field = grid.gradient();

        // Store in cache.
        self.fields.insert(
            goal_id,
            CachedField {
                field,
                dirty_count: 0,
                last_tick: tick,
            },
        );

        self.fields.get(&goal_id).map(|c| &c.field)
    }

    /// Invalidate a specific goal's cached flow field.
    pub fn invalidate(&mut self, goal_id: u64) {
        self.fields.remove(&goal_id);
    }

    /// Clear all cached fields.
    pub fn clear(&mut self) {
        self.fields.clear();
    }

    /// Number of cached fields.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.fields.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dummy LeoHead for testing cache without a real neural network.
    ///
    /// Produces Q-values for a grid of `grid_w × grid_h` cells with
    /// `actions` per cell. `all_goals_q()` returns `[goals × cells × actions]`.
    struct DummyHead {
        goals: usize,
        actions: usize,
        grid_w: u16,
        grid_h: u16,
    }

    impl DummyHead {
        fn cells(&self) -> usize {
            (self.grid_w as usize) * (self.grid_h as usize)
        }
    }

    impl LeoHead for DummyHead {
        fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
            // Layout: [goals * cells * actions].
            // The cache assumes q_for_goal returns [cells * actions] per goal.
            let cells = self.cells();
            let mut q = Vec::with_capacity(self.goals * cells * self.actions);
            for _g in 0..self.goals {
                for c in 0..cells {
                    for a in 0..self.actions {
                        let val = if a == 0 {
                            1.0 / (1.0 + (c as f32 - (cells as f32 / 2.0)).abs())
                        } else {
                            0.1
                        };
                        q.push(val);
                    }
                }
            }
            q
        }

        fn goal_count(&self) -> usize {
            self.goals
        }

        fn action_count(&self) -> usize {
            // Total elements per goal = cells × actions_per_cell.
            self.cells() * self.actions
        }

        /// Override to return the full spatial Q-slice for a goal
        /// (cells × actions) instead of the default per-observation slice.
        fn q_for_goal<'a>(&self, all_q: &'a [f32], goal: usize) -> &'a [f32] {
            let cells = self.cells();
            let per_goal = cells * self.actions;
            let start = goal * per_goal;
            &all_q[start..start + per_goal]
        }
    }

    fn test_config() -> FlowFieldConfig {
        FlowFieldConfig {
            cutoff: 0.25,
            obstacle_radius: 1,
            min_gradient: 1e-4,
            dirty_threshold: 5,
        }
    }

    fn make_head() -> DummyHead {
        DummyHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
        }
    }

    #[test]
    fn test_cache_returns_none_for_below_min_npcs() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(3);
        let head = make_head();

        let result = cache.get_or_compute(42, &head, &[1.0, 2.0], 0, 4, 4, 0, 2);
        assert!(
            result.is_none(),
            "Should return None when npc_count < min_npcs"
        );
    }

    #[test]
    fn test_cache_computes_and_caches() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        // First call computes.
        let result = cache.get_or_compute(42, &head, &[1.0, 2.0], 0, 4, 4, 0, 5);
        assert!(result.is_some(), "Should compute and return flow field");

        // Second call returns cached.
        let result2 = cache.get(42);
        assert!(result2.is_some(), "Should return cached field on get()");
    }

    #[test]
    fn test_cache_hit_on_same_tick() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        let r1 = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 10, 5);
        assert!(r1.is_some());

        // Same tick — should return cached without recompute.
        let r2 = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 10, 5);
        assert!(r2.is_some());
    }

    #[test]
    fn test_dirty_tracking_invalidates() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        // Compute initial field.
        let _ = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 0, 5);

        // get() should succeed (dirty_count = 0).
        assert!(cache.get(1).is_some());

        // Mark dirty below threshold.
        cache.mark_dirty(1, 3);
        // dirty_count = 3 < 5 (threshold), should still be cached but get() returns None
        // because dirty_count > 0.
        assert!(
            cache.get(1).is_none(),
            "Dirty field should not be returned by get()"
        );

        // Mark dirty past threshold — should invalidate entirely.
        cache.mark_dirty(1, 3);
        assert!(
            cache.fields.is_empty(),
            "Cache should be invalidated when dirty_count >= threshold"
        );
    }

    #[test]
    fn test_invalidate_removes_entry() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        let _ = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 0, 5);
        assert_eq!(cache.len(), 1);

        cache.invalidate(1);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_clear_removes_all() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        let _ = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 0, 5);
        let _ = cache.get_or_compute(2, &head, &[0.0], 0, 4, 4, 0, 5);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_invalid_goal_index_returns_none() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        // goal_idx = 5 but only 1 goal.
        let result = cache.get_or_compute(1, &head, &[0.0], 5, 4, 4, 0, 5);
        assert!(result.is_none(), "Out-of-range goal_idx should return None");
    }

    #[test]
    fn test_mark_dirty_nonexistent_goal_is_noop() {
        let mut cache = FlowFieldCache::new(test_config());
        // Should not panic.
        cache.mark_dirty(999, 10);
        assert_eq!(cache.len(), 0);
    }
}
