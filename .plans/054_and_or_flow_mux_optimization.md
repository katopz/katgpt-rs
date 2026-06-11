# Plan 054: and_or / flow / mux Optimization

## Modules
- `and_or/` — AND-OR tree for hierarchical goal decomposition
- `flow/` — FFT flow fields for LEO crowd navigation
- `mux/` — vocabulary-simplex superposition tree search

## Optimization Targets

### 1. flow/fft.rs — `fft_smooth` allocations
- [x] `inflate_obstacles`: Replace `Vec<(usize, usize)>` with double-buffer snapshot pattern — eliminates per-cell pair allocation, uses clamped bounds instead of inner-loop boundary checks
- [x] Early return for `radius == 0`
- [x] ~~`fft_smooth` planner caching per grid size~~ (deferred — `rustfft::FftPlanner` already caches internally, no action needed)

### 2. flow/cache.rs — `get_or_compute` hot path
- [x] Use `fft_smooth_into` instead of `fft_smooth` to reuse scratch buffers across calls (eliminates per-call `Vec<Complex<f32>>` + `Vec<Complex<f32>>` col_buf allocations)
- [x] Pre-allocate `fft_buf`, `fft_col_buf`, `potential_buf`, `blocked_buf` as reusable fields on `FlowFieldCache` struct
- [x] Reuse `blocked_buf` with `resize` + `fill(0)` instead of `vec![0u64; n]` per compute

### 3. mux/dd_tree.rs — `collect_leaf_paths` allocations
- [x] Added `LeafPaths` flat storage struct with `buf: Vec<usize>` + `offsets: Vec<usize>`
- [x] Added `collect_leaf_paths_flat()` using index-based DFS — one contiguous Vec instead of one Vec per leaf
- [x] Kept `collect_leaf_paths()` as backward-compatible wrapper

### 4. mux/demux.rs — `demux` allocation
- [x] Kept O(k²) scan (correct for arbitrary u32 token IDs, k ≤ 32)
- [x] Output `sorted_tokens` unchanged (needs to be Vec since caller owns it)

### 5. mux/freeze_thaw.rs — `MuxPatternStore`
- [x] `pattern_count()` — O(1) via cached `total_patterns` field, incremented on freeze

### 6. flow/mod.rs — `from_q_values` SIMD friendliness
- [x] Chunked loop (4 cells at a time) to help LLVM auto-vectorize the max reduction

### 7. and_or/types.rs — Tree metrics
- [x] `solved_count` / `unsolved_count` — branch-free `solution.is_some() as usize` instead of if-else
- [x] Added doc comments pointing to fused `leaf_stats` for callers needing both counts

## Validation
- [x] `cargo test -p katgpt-core --features ...` — 256/256 pass
