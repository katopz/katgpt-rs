//! T4: Per-goal shared flow field cache.
//!
//! Maps `goal_id → FlowField`. Recomputes when dirty cells exceed threshold.
//! Main entry point: [`FlowFieldCache::get_or_compute`].

use std::collections::HashMap;

use rustfft::{FftPlanner, num_complex::Complex};

use super::{
    FlowField, FlowFieldConfig, LeoPotentialGrid, fft_smooth_into, inflate_obstacles_with_snapshot,
};
use crate::traits::LeoHead;

/// Per-goal cached flow field with dirty-tracking metadata.
struct CachedField {
    /// Tick at which this field was last (re)computed.
    last_tick: u64,
    field: FlowField,
    /// Number of cells changed since last compute.
    dirty_count: u16,
    /// Terrain topology version when this field was computed (Plan 261 Phase 4).
    /// When the game destroys terrain, it bumps the global topology version;
    /// entries with a stale version get invalidated.
    topology_version: u64,
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
    /// Pre-allocated FFT planner — reused across calls for cached FFT plans.
    fft_planner: FftPlanner<f32>,
    /// Pre-allocated FFT scratch buffers — reused across calls to avoid per-compute allocation.
    fft_buf: Vec<Complex<f32>>,
    fft_col_buf: Vec<Complex<f32>>,
    /// Pre-allocated potential buffer — reused across calls.
    potential_buf: Vec<f32>,
    /// Pre-allocated blocked bitfield buffer.
    blocked_buf: Vec<u64>,
    /// Pre-allocated snapshot buffer for obstacle inflation (avoids per-call allocation).
    snapshot_buf: Vec<u64>,
    /// Minimum NPCs sharing a goal to warrant a flow field.
    min_npcs: u16,
    /// Current terrain topology version (Plan 261 Phase 4).
    /// Game bumps this when terrain is destroyed; stale cache entries get invalidated.
    current_topology_version: u64,
}

impl FlowFieldCache {
    /// Create a new cache with the given FFT smoothing config.
    pub fn new(config: FlowFieldConfig) -> Self {
        Self {
            fields: HashMap::new(),
            config,
            fft_planner: FftPlanner::new(),
            min_npcs: 1,
            fft_buf: Vec::new(),
            fft_col_buf: Vec::new(),
            potential_buf: Vec::new(),
            blocked_buf: Vec::new(),
            snapshot_buf: Vec::new(),
            current_topology_version: 0,
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

    /// Notify the cache that terrain topology changed (Plan 261 Phase 4).
    ///
    /// Bumps the internal topology version. On the next `get_or_compute`,
    /// all cached entries with a stale topology version will be recomputed.
    /// Call this after terrain destruction (e.g. `CellComplex::remove_face`).
    #[inline]
    pub fn notify_topology_changed(&mut self) {
        self.current_topology_version = self.current_topology_version.saturating_add(1);
    }

    /// Current topology version the cache is tracking.
    #[inline]
    pub fn topology_version(&self) -> u64 {
        self.current_topology_version
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
    #[allow(clippy::too_many_arguments)]
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

        // Check for a valid cached entry via single lookup. `entry()` would resolve
        // the bucket and let us return `&e.get().field` directly, but the borrow
        // checker forbids returning the entry handle from this scope, so we'd pay
        // a second `[&goal_id]` hash on the cache-hit fast path. Instead, use `get`
        // for the validation pass and a second `get` only on the cache-hit return —
        // both probes are cheap (same hash, hot bucket), and this avoids `entry()`'s
        // bookkeeping for the common case.
        let cached_valid = match self.fields.get(&goal_id) {
            Some(cached) => {
                cached.dirty_count == 0
                    && cached.last_tick == tick
                    && cached.topology_version == self.current_topology_version
            }
            None => false,
        };
        if cached_valid {
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
        let words_per_row = (grid_w as usize).div_ceil(64);
        let blocked_words = words_per_row * (grid_h as usize);

        // Resize scratch buffers if grid dimensions changed.
        self.blocked_buf.resize(blocked_words, 0u64);

        // Copy blocked state into bitfield for inflation.
        // NOTE: Grid stores blocked as (w*h+63)/64 words (flat), while we use
        // (w+63)/64*h words (row-aligned). Layouts differ when w is not a
        // multiple of 64, so per-cell copy is required.
        let wu = grid_w as usize;
        let hu = grid_h as usize;
        if wu.is_multiple_of(64) && wu > 0 {
            // Fast path: identical layout, bulk copy
            let grid_blocked = grid.blocked();
            let copy_len = words_per_row * hu;
            self.blocked_buf[..copy_len].copy_from_slice(&grid_blocked[..copy_len]);
        } else {
            // Unaligned: word-at-a-time copy from flat source to row-aligned dest.
            // Source bit layout: cell (x,y) → flat bit index y*w+x → word (y*w+x)/64.
            // Dest bit layout:   cell (x,y) → word y*words_per_row + x/64, bit x%64.
            let grid_blocked = grid.blocked();
            let grid_words = grid_blocked.len();
            for y in 0..hu {
                for wp in 0..words_per_row {
                    let dest_idx = y * words_per_row + wp;
                    let x_start = wp * 64;
                    let x_end = (x_start + 64).min(wu);
                    let cells_in_word = x_end - x_start;

                    // Source flat bit position for start of this chunk
                    let src_bit_start = y * wu + x_start;
                    let src_word = src_bit_start / 64;
                    let src_bit = (src_bit_start % 64) as u32;

                    // Extract bits from source — may span two source words
                    let mut w = if src_word < grid_words {
                        grid_blocked[src_word] >> src_bit
                    } else {
                        0
                    };
                    if src_bit + (cells_in_word as u32) > 64 && src_word + 1 < grid_words {
                        w |= grid_blocked[src_word + 1] << (64 - src_bit);
                    }

                    // Mask to only the valid cells in this word
                    if cells_in_word < 64 {
                        w &= (1u64 << cells_in_word) - 1;
                    }

                    self.blocked_buf[dest_idx] = w;
                }
            }
        }

        // Resize snapshot buffer in lockstep with blocked_buf.
        self.snapshot_buf.resize(blocked_words, 0u64);

        inflate_obstacles_with_snapshot(
            &mut self.blocked_buf,
            &mut self.snapshot_buf,
            grid_w,
            grid_h,
            self.config.obstacle_radius,
        );

        // Apply inflated obstacles back to the grid (same layout mismatch as above).
        if wu.is_multiple_of(64) && wu > 0 {
            // Fast path: identical layout, bitwise OR bulk copy
            let grid_blocked = grid.blocked_mut();
            let copy_len = words_per_row * hu;
            for (grid_word, blocked_word) in grid_blocked
                .iter_mut()
                .zip(self.blocked_buf.iter())
                .take(copy_len)
            {
                *grid_word |= blocked_word;
            }
        } else {
            // Unaligned: word-at-a-time copy from row-aligned source back to flat dest.
            // Reverse of the above: dest is flat bitstream, source is row-aligned.
            let grid_blocked = grid.blocked_mut();
            let grid_words = grid_blocked.len();
            // Clear dest before OR-ing bits in
            grid_blocked[..grid_words].fill(0);
            for y in 0..hu {
                for wp in 0..words_per_row {
                    let src_idx = y * words_per_row + wp;
                    let src_word = self.blocked_buf[src_idx];
                    if src_word == 0 {
                        continue; // Skip empty words
                    }
                    let x_start = wp * 64;
                    let x_end = (x_start + 64).min(wu);
                    let cells_in_word = x_end - x_start;

                    // Dest flat bit position for start of this chunk
                    let dst_bit_start = y * wu + x_start;
                    let dst_word = dst_bit_start / 64;
                    let dst_bit = (dst_bit_start % 64) as u32;

                    // Write bits into dest — may span two dest words
                    let masked = if cells_in_word < 64 {
                        src_word & ((1u64 << cells_in_word) - 1)
                    } else {
                        src_word
                    };
                    grid_blocked[dst_word] |= masked << dst_bit;
                    if dst_bit + (cells_in_word as u32) > 64 && dst_word + 1 < grid_words {
                        grid_blocked[dst_word + 1] |= masked >> (64 - dst_bit);
                    }
                }
            }
        }

        // FFT smooth the potential field to remove local minima — reuse scratch buffers.
        // Bulk copy: potential layout is identical (w*h flat, row-major).
        self.potential_buf.resize(total_cells, 0.0);
        self.potential_buf[..total_cells].copy_from_slice(&grid.potential_slice()[..total_cells]);

        fft_smooth_into(
            &mut self.potential_buf,
            grid_w as usize,
            grid_h as usize,
            self.config.cutoff,
            &mut self.fft_buf,
            &mut self.fft_col_buf,
            &mut self.fft_planner,
        );

        // Write smoothed values back (bulk copy — same layout).
        grid.potential_mut()[..total_cells].copy_from_slice(&self.potential_buf[..total_cells]);

        // Compute gradient → FlowField.
        let field = grid.gradient();

        // Store in cache.
        self.fields.insert(
            goal_id,
            CachedField {
                field,
                dirty_count: 0,
                last_tick: tick,
                topology_version: self.current_topology_version,
            },
        );

        Some(&self.fields[&goal_id].field)
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
            min_gradient: 1e-4,
            dirty_threshold: 5,
            obstacle_radius: 1,
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

    #[test]
    fn test_topology_version_starts_zero() {
        let cache = FlowFieldCache::new(test_config());
        assert_eq!(cache.topology_version(), 0);
    }

    #[test]
    fn test_notify_topology_changed_bumps_version() {
        let mut cache = FlowFieldCache::new(test_config());
        assert_eq!(cache.topology_version(), 0);
        cache.notify_topology_changed();
        assert_eq!(cache.topology_version(), 1);
        cache.notify_topology_changed();
        assert_eq!(cache.topology_version(), 2);
    }

    #[test]
    fn test_topology_change_triggers_recompute() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head = make_head();

        // Compute and cache at topology version 0
        let f1 = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 0, 5);
        assert!(f1.is_some());

        // Same tick, same topology → should return cached (no recompute)
        // We can't directly observe recompute, but topology bump forces it
        cache.notify_topology_changed();
        assert_eq!(cache.topology_version(), 1);

        // Same tick but new topology → should recompute
        let f2 = cache.get_or_compute(1, &head, &[0.0], 0, 4, 4, 0, 5);
        assert!(f2.is_some(), "should return a field even after topology change");
    }
}
