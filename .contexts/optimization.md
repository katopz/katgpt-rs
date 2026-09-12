# Optimization Skill

Hot-path Rust optimization patterns. Apply to any microsecond-sensitive code.

## Checklist (Quick Reference)

This is a LOOP. Keep optimizing until nothing left to improve.

1. **Complexity**: O(n) scan → O(1) lookup, merge HashMap lookups, merge loops, **boundary-vs-volume** (Stokes/divergence theorem: integrate boundary `O(n^{(d-1)/d})` instead of interior `O(n)` for low-dim d≤3 region mass queries — curse-of-dim caps it at d≤3)
2. **Allocations**: `String` → `&'static str`, pre-allocate, eliminate `.to_string()`
3. **Layout**: field reorder (u64→u32→u8), `#[repr(u8)]` enums, remove `#[repr(C)]`
4. **Arithmetic**: `f32` counters → `u32`/`u64`
5. **Iterators**: fix double `chunks_exact()`, use index arithmetic
6. **Concurrency**: `Arc<RwLock<HashMap>>` → `papaya`, `Mutex<u32>` → `AtomicU32`
7. **SIMD**: chunked loops, branch-free inner loops
8. **Caching**: pre-compute lookup tables, compute once not N×M
9. **Errors**: `unwrap()` → `?`, `let _ =` → `.log_err()`

## Termination

This is a LOOP. You keep optimizing until you cannot find anything to optimize.

After each turn, end your response with exactly one of:

- `Continue optimizing remaining files.` — you made code changes this turn
- `No optimizations this pass.` — you read files but changed nothing

If you only summarized what you read without changing code, say "no optimizations this pass".

## Rules

- Do NOT create plans or issues.
- Do NOT create new files unless necessary.
- Commit when done (`perf:` or `refactor:` prefix).
- Use sigmoid, not softmax.

## When to Optimize

1. Profile first — never optimize without numbers
2. Identify the top 3 bottlenecks (80% of time is in 20% of code)
3. Measure after each change — some "optimizations" make things worse
4. Run both debug (reveals algorithmic cost) and release (reveals compiler optimizations)

## Do

### Profiling

- Break down complex functions into per-component micro-benchmarks
- Use `std::hint::black_box()` to prevent dead-code elimination
- Warm up before measuring (100+ iterations) to prime CPU caches
- Run 10,000+ iterations for stable results
- Print component-level breakdowns with `--nocapture` test harness
- Compare same-commit, back-to-back runs to isolate feature impact from system noise

### Data Structures

- Use fixed-size arrays `[T; N]` when domain is bounded
- Pre-compute lookup tables once, store in config/context — O(1) reads beat O(n) scans
- Track per-slot aggregates during insert/evict instead of scanning on read
- Cache allocations: `Vec::with_capacity()` once, `clear()` + reuse across calls
- Pass pre-allocated scratch buffers as `&mut [T]` parameters instead of allocating inside hot loops

### Manifold / Cell-Complex Geometry

For code operating on cell complexes / meshes / grid manifolds (DEC, FEM, game maps):

- **Boundary-vs-volume (Stokes / divergence theorem)**: to compute a region's total mass / energy / activation magnitude, integrate over the boundary `∂M` (surface area, `O(n^{(d-1)/d})` cells) instead of the interior `M` (volume, `O(n)` cells). Valid when the field is curl-free / exact; reconstruction error is bounded by the harmonic component (compute via Hodge decomposition). **Win shrinks fast as dimension d grows — practical only for d ≤ 3 (2D game maps, 3D belief regions, KG embeddings). For d ≥ 8 (HLA state) or d ≥ 64 (style weights) the boundary is larger than the interior, so boundary-only is a loss.**
- **Conservation-by-construction**: identity `curl(grad)=0` / `div(curl)=0` enforced by DEC operator construction (not a soft penalty) gives mass-conservation invariants for free. Use as a modelless validator: if `div(flow) > τ`, mass leaked/created = anomaly.
- **Pre-compute incidence / Hodge on topology change only**: DEC operators (`exterior_derivative`, `codifferential`, `hodge_decompose`) depend only on the cell complex topology, not on the field values. Compute once on map/complex load, cache, invalidate only when topology changes — zero per-tick DEC op cost on a stable map.
- **Cache the Hodge spectrum / Betti numbers** alongside the operators — they are topology invariants reused across every field query.

### SIMD / Auto-vectorization

**Auto-vectorization (let LLVM do the work):**

- Write chunked loops (4 or 8 elements at a time) to help LLVM auto-vectorize
- Cast `usize` → `u64` slices (same layout on 64-bit) for wider SIMD lanes
- Use `u64` equality comparison — compiler maps to `_mm256_cmpeq_epi64` on AVX2
- Keep inner loops branch-free (use `bool as usize` instead of `if`)
- Verify with release build — SIMD benefits only appear with optimizations enabled

**Portable `std::simd` recipes (when auto-vec isn't enough):**

Reach for `std::simd` only after profiling shows auto-vectorization failing. Patterns below are distilled from mcyoung's `vb64` writeup (https://mcyoung.xyz/2023/11/27/simd-base64/). Full skeleton in `recipes/swizzle_lookup.rs`.

- **Branchless range dispatch**: replace `match` on byte ranges with `simd_ge`/`simd_le` masks + `mask.select(splat_a, splat_b)`. 1 select beats N branches.
- **Perfect-hash lookup via `swizzle_dyn`**: if `(byte >> 4) - (byte == c)` distinguishes all ranges, build an 8-entry offset table and do 1 shuffle instead of N compares. Index vector must be same width as lookup table.
- **Widening cast for sub-byte packing**: `sextets.cast::<u16>() << Simd::from([2,4,6,8])`, then split into `lo = v.cast::<u8>()` and `hi = (v >> 8).cast::<u8>()`, OR them after rotating hi by 1 lane. Lets bits cross byte boundaries without per-bit ops.
- **Lane-deletion swizzle**: when every k-th lane is garbage, use a const swizzle `|i| i + i/(k-1)` to skip those lanes. Compile-time indices = single `vpshufb`.
- **Slop-buffer commit**: `out.reserve(final_len + N/4)`, write full SIMD vectors via `ptr.cast::<Simd<u8,N>>().write_unaligned()`, only call `set_len()` after success. On error, never commit — garbage writes vanish.
- **Delayed failure**: accumulate `error |= !ok` from each iteration, return `Err` once after the loop. Errors are rare; don't pay branch cost per chunk.
- **Unroll-and-jam with overlapping loads**: use `chunks_exact(N)` for the hot path + `Simd::from_slice()`. For the remainder, load `u64` from `p` and `p+len-8` (overlap by 1 byte), OR them — 2 loads cover any 8–15 byte tail.

**Decision rules:**

- SIMD wins: lane count ≥ 8, branch-heavy parse/codec, no allocator in loop, input ≥ 16 bytes
- Scalar wins: input < 16 bytes, branch predicts well (profile says so), cold path, or auto-vec already covers it
- `swizzle_dyn` requires index vector length == lookup table length — pad the table if needed
- Bound all generic SIMD fns with `LaneCount<N>: SupportedLaneCount`
- Tune `N` by benchmark; on x86-64 with AVX2, `N = 32` (one YMM) is usually optimal

### Parallelism / Rayon

- Only parallelize when per-task work exceeds thread-pool overhead (~5μs for rayon)
- Benchmark serial vs parallel at actual workload size before committing
- Rule of thumb: parallelism wins only when per-iteration work > 10μs or count > 1000
- Use `rayon::join(|| left(), || right())` for recursive divide-and-conquer — the primitive that powers Rayon; work-stealing ensures threads don't sit idle
- Use `.par_sort()` and `.par_extend()` instead of manual `.par_iter().collect()` — Rayon provides optimized parallel versions of stdlib algorithms
- Use custom `ThreadPool` to isolate core usage (e.g., reserve CPU for a web server):

```text
let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
pool.install(|| { /* parallel code here */ });
```

- Prefer contiguous data (`Vec`, slices) — Rayon splits chunks efficiently; `LinkedList` forces traversal before splitting
- Profile before and after with `criterion` — warm-up cost and orchestration overhead may negate gains

#### Parallelism: When to Use What

| Feature | `std::iter` | `rayon::par_iter` | `tokio::spawn` |
|---|---|---|---|
| **Best For** | Small data / Simple logic | Big data / CPU-heavy | I/O / Networking |
| **Overhead** | Zero | Medium (task splitting) | High (runtime / context switch) |
| **Execution** | Sequential | Multi-threaded (parallel) | Concurrent (event loop) |

### Allocation

- Pre-build lookup tables and cached data in config structs via builder pattern
- Reuse scratch buffers across loop iterations instead of allocating per-iteration
- Pre-allocate output arrays upfront, write in-place instead of collecting per-iteration
- Reorder struct fields to eliminate padding (group by alignment: u64 → u32 → u8)
- Use `#[repr(u8)]` on field-less enums to guarantee 1-byte size

### WASM FFI / Sandboxed Execution

- **Batch API**: Serialize shared state once, validate all N×M combinations in one FFI call (amortize ~250ns FFI floor across 24 pairs → 5–6× speedup)
- **Zero-copy serialization**: Use fixed-size stack buffers (`[u8; 1024]`) instead of `Vec::with_capacity()` — eliminates allocator overhead in tight loops (~3.6× faster)
- **Fuel budgeting**: Set WASM fuel proportional to worst-case algorithmic complexity, not average case. BFS on bounded grids with N entities can spike 4–5× above typical. Fuzz-test with max inputs to find the ceiling.
- **Lock-free instance pools**: Use `papaya::HashMap<ThreadId, T>` for per-thread WASM stores — lock-free reads on existing entries, uncontended `Mutex` per thread. Better than a single global `Mutex` for multi-threaded servers.
- **Batch state layout**: Omit per-item data from shared state; pass player/entity arrays separately. Grid(169) + bombs(N×4) shared once, then per-entity `(id, x, y)` array alongside action indices and output results buffer.
- **TypedFunc clone**: `wasmtime::TypedFunc` is cheap to clone (handle index). Clone it to release `&self` borrow before calling mutable `Store` methods — avoids borrow-checker conflicts with zero cost.

### Caching

- Compute once per position/context, not per-sample-per-position (N×M → M calls)
- Pre-compute values that don't change across samples (entropy, base path, threshold)

## Don't

### Don't: Rayon for tiny workloads

```text
// BAD: rayon overhead (~5μs) >> computation (~0.1μs per row)
let counts: Vec<_> = (0..10).into_par_iter()
    .map(|i| compute_row(variants, i))
    .collect();

// GOOD: serial for small m
let counts: Vec<_> = (0..10)
    .map(|i| compute_row(variants, i))
    .collect();
```

Threshold: rayon wins only at m ≥ 64 with μs/row work, or m ≥ 1000 with ns/row work.

### Don't: GPU for microsecond workloads

GPU kernel launch overhead is ~50μs. If your computation is 2-5μs, GPU is a net negative.
GPU wins only for: batched matmul, large tensor ops, or when you can amortize launch across many ops.

### Don't: Allocate inside hot loops

```text
// BAD: allocates every call, every sample, every position
for sample in 0..10 {
    let support = rule.support(vocab_size); // Vec allocation!
}

// GOOD: pre-compute once, reuse
let config = Config::default().with_cached_data(size);
for sample in 0..10 {
    let support = config.data_for(rule); // &[T] — zero alloc
}
```

### Don't: Linear scan for hot-path queries

```text
// BAD: O(n) scan per query
fn query(&self, key: usize) -> f32 {
    for item in &self.items {
        if item.key == key { ... }
    }
}

// GOOD: O(1) precomputed index
struct Store {
    stats: [SlotStats; MAX_SLOTS], // updated on insert/evict
}
fn query(&self, key: usize) -> f32 {
    self.stats[key].rate() // O(1)
}
```

### Don't: Recompute unchanged values

```text
// BAD: same value recomputed N×M times
for sample in 0..N {
    for &pos in &positions {
        let h = expensive_calc(data[pos]); // SAME value every sample!
    }
}

// GOOD: compute once per position
let cache: Vec<f32> = positions.iter()
    .map(|&pos| expensive_calc(data[pos]))
    .collect(); // M calls instead of N×M
```

### Don't: Parallelize without measuring

Always benchmark before AND after adding parallelism. If the serial version is faster, keep serial.
Parallel overhead: thread wake (~2μs) + work stealing (~3μs) + synchronization.
If your total work is < 10μs, parallelism will make it slower.

### Don't: Use Mutex in Rayon closures

`Mutex` introduces contention — 16 threads fighting for one lock effectively run sequentially (or slower).
Prefer atomic types or reduce/fold patterns:

```text
// BAD: shared Mutex — threads serialize on lock
let results = Mutex::new(Vec::new());
(0..1000).into_par_iter().for_each(|i| {
    results.lock().unwrap().push(compute(i));  // contention!
});

// GOOD: map + collect — threads work independently, merge at end
let results: Vec<_> = (0..1000).into_par_iter()
    .map(|i| compute(i))
    .collect();
```

### Don't: Ignore Rayon panic propagation

If a closure inside a Rayon thread panics, Rayon propagates that panic to the calling thread.
This can crash your entire application if not handled at the top level. Wrap parallel closures
in `catch_unwind` or ensure invariants are validated before entering Rayon.

### Don't: Ignore cache locality in parallel splits

Splitting work too finely loses CPU cache benefits. Processing contiguous chunks is faster than
jumping across memory addresses in parallel. Prefer chunk-based splitting over per-element parallelism
when data is large but per-element work is small.

### Don't: Ignore binary bloat from feature flags

Adding code behind a feature flag still affects the entire binary when enabled:
- Larger binary → more icache misses → slower hot loops in unrelated code
- Feature-gated code in the same crate affects code layout and branch prediction

Mitigation:
- Isolate feature-gated benchmarks into separate binaries (`[[bin]]`) or test files
- Compare no-feature vs with-feature on the same commit, back-to-back
- If regressions appear only with feature enabled and code is properly gated, it's binary bloat, not a bug

### Don't: Under-budget WASM fuel for complex algorithms

WASM fuel limits prevent infinite loops but can silently trap legitimate computation.
Complex BFS/graph algorithms with N entities on bounded domains can spike well above average:

```text
// BAD: fuel based on average case, traps on worst case
const FUEL_PER_CALL: u64 = 10_000;  // sufficient for 1–2 bombs
// BFS with 4+ bombs × 4 directions × range × 169 cells = ~40K ops → SILENT TRAP

// GOOD: fuel based on worst-case analysis + headroom
const FUEL_PER_CALL: u64 = 50_000;  // 16 bombs × 4 dirs × range 3 × 169 cells ≈ 40K + margin
```

Symptom: WASM returns `false` for valid inputs that should return `true`. Only manifests with complex inputs. Batch APIs may mask this if they use higher fuel multipliers. Fuzz-test with maximum entity counts to catch fuel traps.

### Don't: Serialize per-item when state is shared across a batch

When validating N items against the same state (e.g., N players on one game grid), serializing the state N times wastes both allocation and FFI overhead:

```text
// BAD: 24 × (serialize + FFI + compute) = ~12µs/tick
for player in 0..4 {
    for action in 0..6 {
        let state = serialize(grid, player, action);  // 24 serializations!
        wasm.is_valid(state);                          // 24 FFI calls!
    }
}

// GOOD: 1 × (serialize + FFI + batch compute) = ~1.7µs/tick
let state = serialize_grid(grid, bombs);               // 1 serialization
wasm.batch_validate(state, players, actions, results);  // 1 FFI call
```

The batch API turns N×M individual calls into 1 call. The WASM module internally loops over all combinations, reusing the parsed state. For 4 players × 6 actions, this gives ~5.8× speedup.

### Don't: Compare benchmarks across different CPU thermal states

Laptop CPUs throttle aggressively. A 30% "regression" may just be heat.
Always compare same-commit, back-to-back runs to isolate feature impact from system noise.

## Profiling Template

```text
// tests/prof_bench.rs — run with: cargo test --features X prof_bench -- --nocapture
#[cfg(feature = "X")]
#[test]
fn prof_components() {
    let warmup = 100;
    let iters = 10000;
    
    for _ in 0..warmup { black_box(component_a()); }
    let start = Instant::now();
    for _ in 0..iters { black_box(component_a()); }
    let t_a = start.elapsed();
    
    // ... same pattern for component_b, component_c ...
    
    println!("  Component A: {:.2} μs", t_a.as_micros() as f64 / iters as f64);
    println!("  Total Δ:     {:.2} μs", total.as_micros() as f64 / iters as f64);
}
```

## WASM FFI Batch Template

```text
// Pattern: batch validate N items × M actions in one FFI call
//
// Memory layout written to WASM:
//   [0..state_end)         shared state (grid + bombs, no per-entity data)
//   [players_off..+N×12)   entity array: N × (id, x, y) as u32 LE
//   [actions_off..+M×4)    action indices as u32 LE
//   [results_off..+N×M×4)  output: u32 LE results (0/1 or Q16.16)
//
// WASM export signature:
//   batch_is_valid(state_ptr, state_len, players_ptr, player_count,
//                  actions_ptr, action_count, results_ptr) -> u32

const MAX_ENTITIES: usize = 4;
const ACTION_COUNT: usize = 6;
const ACTIONS_BYTES: [u8; ACTION_COUNT * 4] = [0,0,0,0, 1,0,0,0, 2,0,0,0, 3,0,0,0, 4,0,0,0, 5,0,0,0];

fn batch_validate(&self, grid: &Grid, players: &[(u8,i32,i32)], bombs: &[Bomb]) -> BatchResult {
    self.with_inner(|inner| {
        // 1. Serialize shared state once (zero-copy stack buffer)
        let (state_bytes, state_tokens) = inner.state_buf.serialize_grid(grid, bombs);
        let mut tmp = [0u8; 1024];
        tmp[..state_bytes].copy_from_slice(inner.state_buf.as_bytes(state_bytes));

        // 2. Compute aligned offsets
        let players_off = (state_bytes + 7) & !7;  // align8
        let actions_off = players_off + players.len() * 12;
        let results_off = actions_off + ACTION_COUNT * 4;

        // 3. Write to WASM memory
        inner.write_memory(0, &tmp[..state_bytes])?;
        inner.write_memory(players_off, &players_to_bytes(players))?;
        inner.write_memory(actions_off, &ACTIONS_BYTES)?;

        // 4. Call batch export
        let batch_fn = inner.batch_fn.as_ref()?.clone();
        batch_fn.call(&mut inner.store, (0, state_tokens, players_off as u32,
            players.len() as u32, actions_off as u32, ACTION_COUNT as u32,
            results_off as u32))?;

        // 5. Read results
        Some(BatchResult::from_memory(inner, results_off, players.len(), ACTION_COUNT))
    })
}
```
