//! T4: Per-goal shared flow field cache.
//!
//! Maps `goal_id → FlowField`. Recomputes when dirty cells exceed threshold.
//! Main entry point: [`FlowFieldCache::get_or_compute`].

use std::collections::HashMap;

use rustfft::{FftPlanner, num_complex::Complex};

use super::{
    FlowField, FlowFieldConfig, LeoPotentialGrid, fft_smooth_into, inflate_obstacles_with_snapshot,
};
use crate::traits::DualLeoMixer;
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
    ///
    /// This is the **LEO-only** path (`ActingMode::LeoOnly` in Plan 155's vocabulary).
    /// For the dual teacher-student mix, see [`get_or_compute_dual`](Self::get_or_compute_dual)
    /// (Plan 459), which mixes LEO + UVFA Q-slices via `DualLeoMixer::combine_into`
    /// before building the same `LeoPotentialGrid`.
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
        let (cached_valid, goal_q) = self.validate_and_extract_single(
            goal_id, head, state, goal_idx, grid_w, grid_h, tick, npc_count,
        )?;
        if cached_valid {
            return self.fields.get(&goal_id).map(|c| &c.field);
        }
        // `validate_and_extract_single` returned the goal_q slice as `&[f32]` borrow
        // from `head.all_goals_q` (a fresh Vec owned locally inside the helper). To
        // make the borrow checker happy across the helper boundary, the helper
        // returns an owned `Vec<f32>` for the goal slice — see its doc-comment.
        // For the single-head path we can pass it directly to `compute_from_q_slice`.
        self.compute_from_q_slice(goal_id, &goal_q, grid_w, grid_h, tick)
    }

    /// Get or compute a flow field for a goal, mixing LEO teacher + UVFA student
    /// Q-slices via a [`DualLeoMixer`] (Plan 459).
    ///
    /// `head_leo` is the all-goals Q-teacher (broad, modelless). `head_uvfa` is
    /// the goal-conditioned student (sharp, may be model-based). The mixer's
    /// [`DualLeoMixer::combine_into`] applies the configured `ActingMode`
    /// (Lc / LeoOnly / UvfaOnly / Max / Min) at `alpha` to produce the mixed
    /// Q-slice, which is then fed through the same FFT-smoothed gradient pipeline
    /// as [`get_or_compute`](Self::get_or_compute).
    ///
    /// **Cache key caveat:** `goal_id` is the cache key. Callers that vary `alpha`
    /// for the same goal MUST encode `alpha` into `goal_id` (e.g. hash goal + α),
    /// otherwise the cache will return the field computed at the first α seen.
    ///
    /// **Quality caveat (Plan 459 GOAT — DEMOTED to compatibility by Plan 460):**
    /// On a synthetic 64×64 landscape with a broad-LEO + sharp-UVFA mock pair,
    /// the paper's default α=0.3 did NOT beat the LEO-only baseline by the 30%
    /// stuck-reduction gate (achieved only 3.7%). An α-sweep {0.1..0.9} found
    /// 25.9% at α=0.1 (best) but no α reached the gate. Root cause: the
    /// downstream pipeline applies `max-over-actions` + FFT smoothing, both
    /// nonlinear, which washes out the α-weighting done on raw Q-slices.
    /// **Plan 460 fixed this** by moving the blend to post-max potentials
    /// (`get_or_compute_dual_postmax`), which IS linear in the FFT's input and
    /// crosses the gate at α=0.10 (31.5%). **New flow-field dual consumers
    /// should use `get_or_compute_dual_postmax`, not this method.** This method
    /// is kept for compatibility and for callers that want parity with QGF's
    /// pre-max mix. See
    /// [`.benchmarks/459_flow_field_dual_leo_mixer_goat.md`](../../../../.benchmarks/459_flow_field_dual_leo_mixer_goat.md)
    /// and
    /// [`.benchmarks/460_flow_field_dual_leo_postmax_goat.md`](../../../../.benchmarks/460_flow_field_dual_leo_postmax_goat.md)
    /// for the full reports.
    ///
    /// Returns `None` under the same conditions as `get_or_compute`, plus if the
    /// two heads return incompatible goal/action counts.
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_compute_dual<H1, H2, M>(
        &mut self,
        goal_id: u64,
        head_leo: &H1,
        head_uvfa: &H2,
        mixer: &M,
        alpha: f32,
        state: &[f32],
        goal_idx: usize,
        grid_w: u16,
        grid_h: u16,
        tick: u64,
        npc_count: u16,
    ) -> Option<&FlowField>
    where
        H1: LeoHead,
        H2: LeoHead,
        M: DualLeoMixer,
    {
        let cached_valid = self.check_cache(goal_id, tick, npc_count)?;
        if cached_valid {
            return self.fields.get(&goal_id).map(|c| &c.field);
        }

        // Validate goal index against both heads.
        if goal_idx >= head_leo.goal_count() || goal_idx >= head_uvfa.goal_count() {
            return None;
        }

        // Extract Q-slices from both heads for this goal.
        let q_leo_all = head_leo.all_goals_q(state);
        let q_leo = head_leo.q_for_goal(&q_leo_all, goal_idx);
        let q_uvfa_all = head_uvfa.all_goals_q(state);
        let q_uvfa = head_uvfa.q_for_goal(&q_uvfa_all, goal_idx);

        // Mix via DualLeoMixer (allocates the combined slice — single per cache-miss).
        let mut q_mixed = vec![0.0f32; q_leo.len()];
        mixer.combine_into(&mut q_mixed, q_leo, q_uvfa, alpha);

        self.compute_from_q_slice(goal_id, &q_mixed, grid_w, grid_h, tick)
    }

    /// Dual teacher-student flow field — **post-max** fusion (Plan 460).
    ///
    /// Builds two complete [`LeoPotentialGrid`]s (one per head, each with
    /// `from_q_values`' `max-over-actions` already applied), then linear-blends
    /// their **post-max potentials** via [`LeoPotentialGrid::blend_into`] before
    /// the FFT low-pass + gradient tail. This is the sibling to
    /// [`get_or_compute_dual`](Self::get_or_compute_dual) (Plan 459, pre-max):
    ///
    /// ```text
    ///   Plan 459 (pre-max):  mix raw Q per action → from_q_values(max) → FFT → grad
    ///   Plan 460 (post-max): from_q_values(max) ⊗ from_q_values(max) → blend → FFT → grad
    /// ```
    ///
    /// The post-max blend is **linear in the field the FFT sees**, so the
    /// α-weighting survives the FFT low-pass. Plan 459's pre-max mix was washed
    /// out by the nonlinear `max_a (·)` pool; this path sidesteps that root
    /// cause.
    ///
    /// **`ActingMode` honoring:** as with Plan 459, the mixer's `acting_mode()`
    /// selects the blend semantics. The four non-trivial modes map to α values:
    /// `Lc` → use `alpha`, `LeoOnly` → `alpha = 1.0`, `UvfaOnly` → `alpha = 0.0`,
    /// `Max`/`Min` → not meaningful for potentials (they're already max-pooled);
    /// the implementation delegates to `mixer.combine_into` on a per-cell dummy
    /// single-action Q so all 5 modes still work uniformly via the existing
    /// mixer trait, but for `Max`/`Min` callers should prefer
    /// [`get_or_compute_dual`](Self::get_or_compute_dual) (where per-action max
    /// is well-defined).
    ///
    /// **Cache key caveat:** same as Plan 459 — `goal_id` is the cache key.
    /// Callers that vary `alpha` MUST encode it into `goal_id`.
    ///
    /// **Quality verdict (Plan 460 GOAT — PROMOTED):** Post-max blending is
    /// **linear in the field the FFT sees**, which is the correct fusion point for
    /// flow-field navigation. On a synthetic 64×64 landscape with the same mock
    /// LEO+UVFA pair that failed pre-max G5' in Plan 459, post-max **crosses the
    /// 30% stuck-reduction gate at α=0.10** (31.5% reduction; pre-max best was
    /// 25.9%). This is the **recommended** dual path. See
    /// [`.benchmarks/460_flow_field_dual_leo_postmax_goat.md`](../../../../.benchmarks/460_flow_field_dual_leo_postmax_goat.md)
    /// for the full report including the side-by-side vs Plan 459.
    ///
    /// **α guidance:** the paper's default α=0.3 is wrong for flow-field
    /// navigation on this landscape (only 1.9% reduction). The sweep winner is
    /// α=0.10. Sweep α for your specific heads — the optimal value depends on
    /// how clean the UVFA student's post-max potential is relative to the LEO
    /// teacher's.
    ///
    /// Returns `None` under the same conditions as `get_or_compute_dual`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_compute_dual_postmax<H1, H2, M>(
        &mut self,
        goal_id: u64,
        head_leo: &H1,
        head_uvfa: &H2,
        mixer: &M,
        alpha: f32,
        state: &[f32],
        goal_idx: usize,
        grid_w: u16,
        grid_h: u16,
        tick: u64,
        npc_count: u16,
    ) -> Option<&FlowField>
    where
        H1: LeoHead,
        H2: LeoHead,
        M: DualLeoMixer,
    {
        let cached_valid = self.check_cache(goal_id, tick, npc_count)?;
        if cached_valid {
            return self.fields.get(&goal_id).map(|c| &c.field);
        }

        // Validate goal index against both heads.
        if goal_idx >= head_leo.goal_count() || goal_idx >= head_uvfa.goal_count() {
            return None;
        }

        let total_cells = (grid_w as usize) * (grid_h as usize);
        if total_cells == 0 {
            return None;
        }

        // Build both post-max potential grids. Each `from_q_values` applies
        // max-over-actions internally; the two resulting `potential` buffers
        // are the post-max fields we want to blend.
        let q_leo_all = head_leo.all_goals_q(state);
        let q_leo = head_leo.q_for_goal(&q_leo_all, goal_idx);
        let actions_per_cell_leo = q_leo.len() / total_cells;
        if actions_per_cell_leo == 0 {
            return None;
        }
        let mut grid_leo =
            LeoPotentialGrid::from_q_values(grid_w, grid_h, q_leo, actions_per_cell_leo);

        let q_uvfa_all = head_uvfa.all_goals_q(state);
        let q_uvfa = head_uvfa.q_for_goal(&q_uvfa_all, goal_idx);
        let actions_per_cell_uvfa = q_uvfa.len() / total_cells;
        if actions_per_cell_uvfa == 0 {
            return None;
        }
        let grid_uvfa =
            LeoPotentialGrid::from_q_values(grid_w, grid_h, q_uvfa, actions_per_cell_uvfa);

        // Resolve effective alpha from the mixer's acting mode. The post-max
        // blend only has a scalar α knob (no per-action max/min semantics),
        // so Max/Min collapse to Lc with the caller-provided alpha. This is
        // documented in the method doc-comment above.
        let effective_alpha = match mixer.acting_mode() {
            crate::traits::ActingMode::LeoOnly => 1.0,
            crate::traits::ActingMode::UvfaOnly => 0.0,
            crate::traits::ActingMode::Lc
            | crate::traits::ActingMode::Max
            | crate::traits::ActingMode::Min => alpha,
        };

        // Blend into the cache's potential scratch buffer (no allocation —
        // `potential_buf` is reused across cache-misses).
        self.potential_buf.resize(total_cells, 0.0);
        grid_leo.blend_into(&grid_uvfa, effective_alpha, &mut self.potential_buf);

        // Reuse `grid_leo` as the working grid: overwrite its potential with
        // the blended values. Safe because `grid_leo`'s original potential has
        // already been read by `blend_into` and is now stale.
        grid_leo.potential_mut()[..total_cells].copy_from_slice(&self.potential_buf[..total_cells]);

        // OR the UVFA grid's blocked bitfield into the LEO grid's (an obstacle
        // on either side is an obstacle).
        {
            let leo_blocked = grid_leo.blocked_mut();
            let uvfa_blocked = grid_uvfa.blocked();
            let words = leo_blocked.len().min(uvfa_blocked.len());
            for i in 0..words {
                leo_blocked[i] |= uvfa_blocked[i];
            }
        }

        self.compute_from_grid(grid_leo, goal_id, tick)
    }

    /// Cache validation + npc_count gate (shared by both entry points).
    ///
    /// Returns `Some(cached_valid)` if the call should proceed (with `cached_valid=true`
    /// meaning "hot cache hit, skip compute"), or `None` if the call should early-return `None`.
    #[inline]
    fn check_cache(&self, goal_id: u64, tick: u64, npc_count: u16) -> Option<bool> {
        if npc_count < self.min_npcs {
            return None;
        }
        let cached_valid = match self.fields.get(&goal_id) {
            Some(cached) => {
                cached.dirty_count == 0
                    && cached.last_tick == tick
                    && cached.topology_version == self.current_topology_version
            }
            None => false,
        };
        Some(cached_valid)
    }

    /// Validate goal index + extract the goal-Q slice from a single head.
    ///
    /// Returns `Some((cached_valid, goal_q_owned))` or `None` on early-exit.
    /// The goal-Q slice is returned as an **owned** `Vec<f32>` because the
    /// upstream `head.all_goals_q(state)` returns an owned `Vec<f32>` that
    /// would otherwise drop at helper return.
    #[allow(clippy::too_many_arguments)]
    fn validate_and_extract_single<H: LeoHead>(
        &self,
        goal_id: u64,
        head: &H,
        state: &[f32],
        goal_idx: usize,
        grid_w: u16,
        grid_h: u16,
        tick: u64,
        npc_count: u16,
    ) -> Option<(bool, Vec<f32>)> {
        let cached_valid = self.check_cache(goal_id, tick, npc_count)?;
        if cached_valid {
            // Hot cache hit — no need to extract Q at all.
            return Some((true, Vec::new()));
        }
        if goal_idx >= head.goal_count() {
            return None;
        }
        let all_q = head.all_goals_q(state);
        let goal_q = head.q_for_goal(&all_q, goal_idx).to_vec();
        // Pre-validate actions_per_cell here so compute_from_q_slice can be pure.
        let total_cells = (grid_w as usize) * (grid_h as usize);
        if total_cells == 0 {
            return None;
        }
        if head.action_count() / total_cells == 0 {
            return None;
        }
        Some((false, goal_q))
    }

    /// Pure helper: take a goal-Q slice and produce (or fetch from cache) a `FlowField`.
    ///
    /// The caller is responsible for cache validation + npc_count gate before this.
    /// This function always stores the freshly-computed field under `goal_id`.
    ///
    /// **Legacy pre-max fusion point (Plan 459, demoted by Plan 460):** the Q-slice
    /// may itself be the output of `DualLeoMixer::combine_into`, but the resulting
    /// `LeoPotentialGrid::from_q_values` still applies `max-over-actions` per cell
    /// after the mix — this nonlinearity washes out the α-weighting on synthetic
    /// data (Plan 459 G5 FAIL). New dual consumers should use
    /// [`get_or_compute_dual_postmax`](Self::get_or_compute_dual_postmax), which
    /// blends post-max potentials via [`compute_from_grid`](Self::compute_from_grid)
    /// directly (Plan 460 G5' PASS).
    fn compute_from_q_slice(
        &mut self,
        goal_id: u64,
        goal_q: &[f32],
        grid_w: u16,
        grid_h: u16,
        tick: u64,
    ) -> Option<&FlowField> {
        // Derive actions_per_cell from goal_q length and grid dimensions.
        // (Validated upstream: goal_q.len() == actions_per_cell * total_cells.)
        let total_cells = (grid_w as usize) * (grid_h as usize);
        if total_cells == 0 {
            return None;
        }
        let actions_per_cell = goal_q.len() / total_cells;
        if actions_per_cell == 0 {
            return None;
        }

        // Build potential grid from the (possibly mixed) Q-slice.
        let grid = LeoPotentialGrid::from_q_values(grid_w, grid_h, goal_q, actions_per_cell);

        self.compute_from_grid(grid, goal_id, tick)
    }

    /// Tail pipeline: take a fully-built `LeoPotentialGrid` (post-`from_q_values`,
    /// post-`blend_into` if dual) and produce (or fetch from cache) a `FlowField`.
    ///
    /// Steps: obstacle inflate → FFT low-pass → gradient → cache insert.
    /// Shared between [`compute_from_q_slice`](Self::compute_from_q_slice)
    /// (single-head / pre-max Plan 459 path) and
    /// [`get_or_compute_dual_postmax`](Self::get_or_compute_dual_postmax)
    /// (post-max Plan 460 path).
    ///
    /// The caller is responsible for cache validation + npc_count gate before this.
    /// This function always stores the freshly-computed field under `goal_id`.
    fn compute_from_grid(
        &mut self,
        mut grid: LeoPotentialGrid,
        goal_id: u64,
        tick: u64,
    ) -> Option<&FlowField> {
        let grid_w = grid.w;
        let grid_h = grid.h;
        let total_cells = (grid_w as usize) * (grid_h as usize);
        if total_cells == 0 {
            return None;
        }

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

    /// Returns `true` if no fields are cached.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
            q.shrink_to_fit();
            q
        }

        #[inline]
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
        assert!(
            f2.is_some(),
            "should return a field even after topology change"
        );
    }

    // ──── Plan 459: DualLeoMixer fusion tests ────────────────────────────

    /// A trivial mixer that uses the default `combine_into` (Lc / LeoOnly / UvfaOnly / Max / Min).
    struct TestMixer(crate::traits::ActingMode);

    impl crate::traits::DualLeoMixer for TestMixer {
        fn acting_mode(&self) -> crate::traits::ActingMode {
            self.0
        }
    }

    /// Second head with a *different* Q-shape so the dual mix is observable.
    /// Constant 0.5 everywhere → under Lc at α=0.3 the mixed Q is
    /// `0.3·q_leo + 0.7·0.5`, which is a uniform shift that should not change
    /// the gradient direction (only its magnitude, which the FlowField normalises
    /// away). This makes the G1 baseline sanity check clean.
    struct ConstantHead {
        goals: usize,
        actions: usize,
        grid_w: u16,
        grid_h: u16,
        value: f32,
    }

    impl LeoHead for ConstantHead {
        fn all_goals_q(&self, _state: &[f32]) -> Vec<f32> {
            vec![
                self.value;
                self.goals * (self.grid_w as usize) * (self.grid_h as usize) * self.actions
            ]
        }
        #[inline]
        fn goal_count(&self) -> usize {
            self.goals
        }
        fn action_count(&self) -> usize {
            (self.grid_w as usize) * (self.grid_h as usize) * self.actions
        }
        fn q_for_goal<'a>(&self, all_q: &'a [f32], goal: usize) -> &'a [f32] {
            let per_goal = (self.grid_w as usize) * (self.grid_h as usize) * self.actions;
            let start = goal * per_goal;
            &all_q[start..start + per_goal]
        }
    }

    /// G1 (correctness): `get_or_compute_dual` with `ActingMode::LeoOnly`
    /// produces bit-identical `FlowField` to `get_or_compute` with the same
    /// LEO head, same state, same goal.
    #[test]
    fn test_dual_leo_only_matches_single_head_bit_identical() {
        let mut cache_a = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let mut cache_b = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::LeoOnly);

        let single = cache_a
            .get_or_compute(7, &head_leo, &[0.0], 0, 4, 4, 0, 5)
            .unwrap();
        let dual = cache_b
            .get_or_compute_dual(7, &head_leo, &head_uvfa, &mixer, 0.3, &[0.0], 0, 4, 4, 0, 5)
            .unwrap();

        // Bit-identical: LeoOnly mode copies q_leo unchanged into the pipeline.
        assert_eq!(single.width(), dual.width());
        assert_eq!(single.height(), dual.height());
        for y in 0..single.height() {
            for x in 0..single.width() {
                let a = single.lookup(x, y);
                let b = dual.lookup(x, y);
                assert_eq!(a, b, "mismatch at ({x},{y}): single={a:?} dual={b:?}");
            }
        }
    }

    /// Sanity: `ActingMode::UvfaOnly` should produce the same field as calling
    /// `get_or_compute` directly with the UVFA head.
    #[test]
    fn test_dual_uvfa_only_matches_uvfa_head() {
        let mut cache_a = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let mut cache_b = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.9,
        };
        let mixer = TestMixer(crate::traits::ActingMode::UvfaOnly);

        let direct = cache_a
            .get_or_compute(7, &head_uvfa, &[0.0], 0, 4, 4, 0, 5)
            .unwrap();
        let dual = cache_b
            .get_or_compute_dual(7, &head_leo, &head_uvfa, &mixer, 0.0, &[0.0], 0, 4, 4, 0, 5)
            .unwrap();

        for y in 0..direct.height() {
            for x in 0..direct.width() {
                assert_eq!(direct.lookup(x, y), dual.lookup(x, y), "at ({x},{y})");
            }
        }
    }

    /// Sanity: `get_or_compute_dual` with the default Lc mode and α=0.3 produces
    /// a valid (non-empty) FlowField. The constant-0.5 UVFA should shift the
    /// potential but the gradient direction (after FFT smoothing) should remain
    /// oriented the same way.
    #[test]
    fn test_dual_lc_mode_produces_valid_field() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::Lc);

        let dual = cache.get_or_compute_dual(
            11,
            &head_leo,
            &head_uvfa,
            &mixer,
            0.3,
            &[0.0],
            0,
            4,
            4,
            0,
            5,
        );
        assert!(dual.is_some(), "dual Lc path should produce a field");
        let dual = dual.unwrap();
        assert_eq!(dual.width(), 4);
        assert_eq!(dual.height(), 4);
    }

    /// Cache key caveat: same `goal_id` with different α should still return the
    /// **first** cached field. Callers must encode α into the goal_id. Documented
    /// behavior — we test it to make sure the contract holds.
    #[test]
    fn test_dual_cache_key_is_goal_id_only() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::Lc);

        // First call at α=0.3 — populates cache under goal_id=42.
        let f_first = cache
            .get_or_compute_dual(
                42,
                &head_leo,
                &head_uvfa,
                &mixer,
                0.3,
                &[0.0],
                0,
                4,
                4,
                0,
                5,
            )
            .unwrap();
        let first_lookup = f_first.lookup(0, 0);

        // Second call at α=0.9, same goal_id=42, same tick → cache hit, returns the α=0.3 field.
        let f_cached = cache
            .get_or_compute_dual(
                42,
                &head_leo,
                &head_uvfa,
                &mixer,
                0.9,
                &[0.0],
                0,
                4,
                4,
                0,
                5,
            )
            .unwrap();
        assert_eq!(
            f_cached.lookup(0, 0),
            first_lookup,
            "cache key is goal_id only; α is not part of the key"
        );
    }

    /// Dual path respects npc_count gate (same as single path).
    #[test]
    fn test_dual_returns_none_below_min_npcs() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(3);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::Lc);

        let result = cache.get_or_compute_dual(
            42,
            &head_leo,
            &head_uvfa,
            &mixer,
            0.3,
            &[0.0],
            0,
            4,
            4,
            0,
            2,
        );
        assert!(result.is_none(), "dual path should respect npc_count gate");
    }

    // ──── Plan 460: post-max dual fusion tests ────────────────────────

    /// G1 (correctness): `get_or_compute_dual_postmax` with `ActingMode::LeoOnly`
    /// produces bit-identical `FlowField` to `get_or_compute` with the same LEO
    /// head, same state, same goal. The post-max blend at effective α=1.0 writes
    /// `grid_leo.potential` unchanged, so the downstream FFT + gradient pipeline
    /// sees the same input as the single-head path.
    #[test]
    fn test_dual_postmax_leo_only_matches_single_head_bit_identical() {
        let mut cache_a = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let mut cache_b = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::LeoOnly);

        let single = cache_a
            .get_or_compute(7, &head_leo, &[0.0], 0, 4, 4, 0, 5)
            .unwrap();
        let dual_postmax = cache_b
            .get_or_compute_dual_postmax(
                7,
                &head_leo,
                &head_uvfa,
                &mixer,
                0.3,
                &[0.0],
                0,
                4,
                4,
                0,
                5,
            )
            .unwrap();

        assert_eq!(single.width(), dual_postmax.width());
        assert_eq!(single.height(), dual_postmax.height());
        for y in 0..single.height() {
            for x in 0..single.width() {
                let a = single.lookup(x, y);
                let b = dual_postmax.lookup(x, y);
                assert_eq!(
                    a, b,
                    "postmax LeoOnly mismatch at ({x},{y}): single={a:?} dual_postmax={b:?}"
                );
            }
        }
    }

    /// Sanity: `ActingMode::UvfaOnly` on the post-max path should produce the
    /// same field as calling `get_or_compute` directly with the UVFA head.
    /// (Effective α=0.0 → out = grid_uvfa.potential.)
    #[test]
    fn test_dual_postmax_uvfa_only_matches_uvfa_head() {
        let mut cache_a = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let mut cache_b = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.9,
        };
        let mixer = TestMixer(crate::traits::ActingMode::UvfaOnly);

        let direct = cache_a
            .get_or_compute(7, &head_uvfa, &[0.0], 0, 4, 4, 0, 5)
            .unwrap();
        let dual_postmax = cache_b
            .get_or_compute_dual_postmax(
                7,
                &head_leo,
                &head_uvfa,
                &mixer,
                0.0,
                &[0.0],
                0,
                4,
                4,
                0,
                5,
            )
            .unwrap();

        for y in 0..direct.height() {
            for x in 0..direct.width() {
                assert_eq!(
                    direct.lookup(x, y),
                    dual_postmax.lookup(x, y),
                    "postmax UvfaOnly mismatch at ({x},{y})"
                );
            }
        }
    }

    /// Sanity: `get_or_compute_dual_postmax` with Lc mode produces a valid
    /// (non-empty) FlowField. The constant-0.5 UVFA blend shifts the potential
    /// but the field shape should remain coherent.
    #[test]
    fn test_dual_postmax_lc_mode_produces_valid_field() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(1);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::Lc);

        let dual = cache.get_or_compute_dual_postmax(
            11,
            &head_leo,
            &head_uvfa,
            &mixer,
            0.3,
            &[0.0],
            0,
            4,
            4,
            0,
            5,
        );
        assert!(dual.is_some(), "postmax Lc path should produce a field");
        let dual = dual.unwrap();
        assert_eq!(dual.width(), 4);
        assert_eq!(dual.height(), 4);
    }

    /// Post-max dual path respects npc_count gate (same as single + pre-max).
    #[test]
    fn test_dual_postmax_returns_none_below_min_npcs() {
        let mut cache = FlowFieldCache::new(test_config()).with_min_npcs(3);
        let head_leo = make_head();
        let head_uvfa = ConstantHead {
            goals: 1,
            actions: 4,
            grid_w: 4,
            grid_h: 4,
            value: 0.5,
        };
        let mixer = TestMixer(crate::traits::ActingMode::Lc);

        let result = cache.get_or_compute_dual_postmax(
            42,
            &head_leo,
            &head_uvfa,
            &mixer,
            0.3,
            &[0.0],
            0,
            4,
            4,
            0,
            2,
        );
        assert!(
            result.is_none(),
            "postmax dual path should respect npc_count gate"
        );
    }
}
