# Plan 055: and_or / flow / mux Optimization Pass 2

## Context
Plan 054 applied initial optimizations. This pass targets remaining hot-path allocations,
redundant computation, and SIMD-unfriendly patterns.

## Modules
- `and_or/types.rs` — AND-OR tree node traversal
- `flow/fft.rs` — FFT smoothing + inflate_obstacles
- `flow/cache.rs` — flow field cache hot path
- `flow/mod.rs` — gradient computation, from_q_values
- `flow/steering.rs` — bilinear interpolation
- `mux/dd_tree.rs` — DD-tree leaf collection, expansion
- `mux/bfs.rs` — BFS step allocation
- `mux/top_k.rs` — top-K extraction
- `mux/demux.rs` — demux duplicate detection
- `mux/bandit_width.rs` — bandit arm selection
- `mux/freeze_thaw.rs` — pattern store

## Optimization Targets

### 1. flow/fft.rs — `inflate_obstacles` snapshot allocation
- [x] Replace `blocked.to_vec()` with pre-allocated snapshot buffer passed as parameter
- [x] In `fft_smooth_into`, col_buf already reused — verified no double-clear

### 2. flow/mod.rs — `gradient()` unnecessary f32 math
- [x] Replace branch-free normalization with early-continue for zero gradient (avoids sqrt + div for flat cells)
- [x] Pre-compute `w as usize` and `h as usize` once (already done, verified)

### 3. flow/mod.rs — `from_q_values` heap allocation
- [x] `potential` Vec pre-sized via `with_capacity` — already sufficient
- [x] The chunked loop is cosmetic — kept as-is (LLVM handles both patterns)
- [x] Consider `chunks_exact` + remainder — not worth the refactor for same perf

### 4. mux/dd_tree.rs — `collect_leaf_paths` still allocates Vec<Vec<usize>>
- [x] `bfs.rs::step()` now uses `collect_leaf_paths_flat()` — single contiguous Vec + offsets
- [x] `expand_bfs_frontier` already uses per-leaf paths — no change needed (API is correct)
- [x] `collect_leaf_paths()` kept for backward compat, hot path uses flat version

### 5. mux/dd_tree.rs — `expand_node` allocates per child
- [x] `children.reserve(effective_width)` added before push loop
- [ ] Stack arrays for child tokens/weights — deferred (MuxNode stores Vec, changing would be API-breaking)

### 6. mux/dd_tree.rs — `init_root` allocates
- [ ] `(0..peaks.len() as u32).collect()` — one Vec for tokens, one for weights
- [ ] With bounded K, can use stack arrays

### 7. mux/demux.rs — `demux` allocates output Vec
- [x] Added `demux_into` variant that writes to caller-provided `&mut Vec<u32>` buffer
- [x] `demux()` delegates to `demux_into` — backward compatible

### 8. mux/top_k.rs — `extract_top_k_peaks` allocates
- [ ] `logits.to_vec()` — allocates copy for in-place partition
- [ ] Callers in hot path already use `extract_top_k_into` — verify `extract_top_k_peaks` is only test/bench usage
- [ ] If only tests use it, keep as-is

### 9. mux/bandit_width.rs — `select_width` iterator chain
- [ ] `.iter().map().max_by()` — fine for small k (≤16 arms), no change needed
- [ ] Verify: arm count == k, typically ≤ 8 — O(k) scan is optimal

### 10. flow/cache.rs — `get_or_compute` blocked_buf copy loops
- [ ] Two nested loops (grid→bitfield, bitfield→grid) for blocked state
- [ ] Could merge into single pass if LeoPotentialGrid stored bitfield directly
- [ ] Low priority — grid size is small (typically 64×64 = 4096 cells)

### 11. flow/steering.rs — `flow_steering` boundary checks
- [ ] Multiple boundary checks (x0, x1, y0, y1) — could consolidate
- [ ] Low priority — function is O(1) per NPC call

### 12. and_or/types.rs — `node_count`, `depth` recursive traversal
- [ ] Could cache node_count/depth in parent nodes — but adds insert complexity
- [ ] Low priority — tree metrics are not called in hot paths

## GOAT Gate
- Feature flag: `goat_and_or_flow_mux_opt_pass2`
- Benchmark before/after with existing `flow_field_bench` + new mux benchmarks
- Require >10% improvement on at least one benchmark to promote to default

## Validation
- [x] `cargo test -p katgpt-core --features "flow_field_nav,mux_bfs,mux_demux,mux_bandit_width,mux_freeze_thaw,comp_width,mux_ddtree,mux_pruner"` — 232/232 pass
- [ ] `cargo bench --bench flow_field_bench --features flow_field_nav` — deferred (manual benchmark)
