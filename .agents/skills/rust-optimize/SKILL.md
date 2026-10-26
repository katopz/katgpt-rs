---
name: rust-optimize
description: >-
  Optimize Rust code until nothing left to improve. Loops automatically.
  Use when the user says "optimize" or "/optimize".
disable-model-invocation: true
---

# Rust Optimization

Optimize `*.rs` files in the target scope using the checklist below.
This is a LOOP. You keep optimizing until you cannot find anything to optimize.

## Termination

After each turn, end your response with exactly one of:

- `Continue optimizing remaining files.` — you made code changes this turn
- `No optimizations this pass.` — you read files but changed nothing

If you only summarized what you read without changing code, say "no optimizations this pass".

## Checklist

1. **Complexity**: O(n) scan → O(1) lookup, merge HashMap lookups, merge loops, **boundary-vs-volume** (Stokes/divergence theorem: integrate boundary `O(n^{(d-1)/d})` instead of interior `O(n)` for low-dim d≤3 region mass queries — curse-of-dim caps it at d≤3)
2. **Allocations**: `String` → `&'static str`, pre-allocate, eliminate `.to_string()`
3. **Layout**: field reorder (u64→u32→u8), `#[repr(u8)]` enums, remove `#[repr(C)]`
4. **Arithmetic**: `f32` counters → `u32`/`u64`
5. **Iterators**: fix double `chunks_exact()`, use index arithmetic
6. **Concurrency**: `Arc<RwLock<HashMap>>` → `papaya`, `Mutex<u32>` → `AtomicU32`
7. **SIMD**: chunked loops, branch-free inner loops
8. **Caching**: pre-compute lookup tables, compute once not N×M
9. **Errors**: `unwrap()` → `?`, `let _ =` → `.log_err()`
10. **Bevy / Real-time loop** (see § Bevy / Real-time Loop for details):
    - Event loop spin: `WinitSettings::default()` (Continuous) pegs CPU at 100% even idle — switch to `UpdateMode::reactive_low_power(max_wait)` matching tick rate
    - Vsync ≠ event loop: `PresentMode::Fifo` controls GPU only, independent of CPU loop spin
    - Static geometry: bake to `LineList` / `TriangleList` mesh once, NEVER redraw via `gizmos.line()` per-frame (each call is CPU + per-frame buffer grow)
    - Asset lifecycle: despawning a Bevy `Entity` does NOT release its `Assets<Mesh>` / `Assets<StandardMaterial>` handles — call `meshes.remove(handle.0.id())` + `materials.remove(...)` on despawn or leak indefinitely (regrow-on-new-EntityId patterns leak ~N assets per cycle)
    - GUI repaint storms: `bevy_egui` sets `set_request_repaint_callback` that fires `bevy_winit::WakeUp` on `delay.is_zero()` — bypasses reactive `max_wait` and spins the loop. Diagnose via in-process perf overlay before assuming the engine itself is the hog.
    - Frame-rate cap fallback: a `Last`-schedule `std::thread::sleep` system capped at e.g. 60 FPS is a belt-and-suspenders defense when vsync + reactive mode somehow don't throttle
11. **Perf diagnosis**: ship an in-process perf overlay (`sysinfo` for CPU/RAM + Bevy's `Time::delta()` for FPS) inside the GUI itself — external `ps` sampling misses focused-window swings and requires keeping a terminal attached
12. **Sorted indexes & scans** (see § Sorted Indexes & Scans for details): seek to a prefix (never skip-walk), visitor/borrowing scans (0 alloc), bulk insert via sorted merge, order maintained at mutation not per-read, sort keys computed once, early-exit bounded scans

## Rules

- Do NOT create plans or issues.
- Do NOT create new files unless necessary.
- Commit when done (`perf:` or `refactor:` prefix).
- Use sigmoid, not softmax.


---

# Optimization Skill

Hot-path Rust optimization patterns. Apply to any microsecond-sensitive code.

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

### Sorted Indexes & Scans

Patterns measured on the riir-neuron-db LocalKv / ShardIndex / Bm25 wave (2026-10-24..26, ndb issues 633/634/635/636) and their consumers:

- **Seek, don't skip-walk.** A scan starting at a prefix must move the cursor straight to the first key >= prefix. An `O(skipped)` `skip_while` walk over 50k keys was **342.6x** slower than a direct seek (`LocalKv::scan_after_prefix`, exhaustive 1600-pair G1); the same seek at consumers made verdict-agreement scans read **1270-1430x** less data (riir-dapps issue 636 T4).
- **Visitor scans, not materialized Vecs.** `scan_prefix_visit(&mut impl FnMut(&K, &V))` beat the collecting scan **4.9-6.3x** with **0 allocs/row** — the closure consumes each row in place; nothing builds intermediate key/value Vecs.
- **Bulk insert = sorted merge.** Building an index by merging N sorted batches (one two-finger walk) beat N individual inserts **35-327x** at build and **137x** per flush — and is byte-identical to sequential insertion, so determinism gates stay green (`ShardIndex::insert_shards_batch`).
- **Read values from the index you are already iterating.** A second random probe per row vs reading the value adjacent in the sorted index: **2.6x** per scan. Same family — denormalize hot metadata into the entry: inlining doc length into the Bm25 posting removed a second `doc_lengths` probe per posting, **3.3-3.5x** at 9.6k scored postings, bit-identical.
- **Maintain order at mutation, never at read.** Keep the side structure ascending at insert/evict and every read is sort-free: the AOI fan-out deleted its per-fan-out roster sort (riir-game-sdk issue 051); KarmaLedger histories kept sorted once -> binary-search lookups + a sortless broadcast, **18.1x** on the SOCL path (riir-ai issue 1125 Finding 3).
- **Compute sort keys once.** A `sort_by` whose key is computed per comparison is `O(n log n)` key builds; cache keys before sorting (or `sort_by_cached_key`) — the `pull_unproven` cached-key fix (riir-dapps issue 132).
- **Early-exit bounded scans.** A bounded sweep stops one decodable row past its bound — reads `max_rows + 1` rows, never the remainder of the set (tuna expiry sweep).
- **Cheap derived key columns.** A hull-key column beside the expensive key lets queries pre-filter on a cheap scalar (**1.08-1.25x**, promoted default); column-major layout for argmax/top-k over many rows sharing a type-run (**1.6x** `query_top_k`).
- **Resolve names to indices at build time.** Name -> id lookups resolved once when the book/library is built, never per log/render line (seal-std issue 003 F1+F3: log-time name resolution in seal-anim, once-per-book index in seal-quest).
- **Pack compound keys.** Two u32 ids packed into one u64 -> one hash probe, zero-alloc: BPE merge ranks re-keyed to packed token-id pairs, **2.03x** encode, bit-identical output, constant alloc (riir-infer issue 1118).
- **Fast-path the n=1 batch.** A batch API's single-entry arm skips the batch machinery the common call doesn't need (`put_batch_durable`).
- **Cache derived aggregates incrementally.** `chain_commitment_cache`: cache the root, check per-TX divergence against it in O(1), rebuild in parallel on miss — **22x** at N=1e5 (riir-chain issue 166, promoted default).
- **Batch I/O under the receiver's queue cap.** Emitting DTLS outputs every 16 datagrams (under the peer's 30-deep rx queue) batched SACKs into burst steps and killed 150-250 ms stalls from per-pair dribbles + silently-dropped `Err`s (seal-remake issue 119). Bounded batches also make the emission count auditable via census counters.

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

### Bevy / Real-time Loop

Real-time render loops fail in ways the standard CPU profile doesn't catch. The hot CPU path can be 0% algorithmic work — it can be the **event loop itself spinning between frames**. Patterns below are distilled from the `bevy_egui_orchard_demo` 100%-CPU saga (commits `d6fb415` → `565524f` → `01b4687` in `riir-game-sdk`).

#### Event loop spin (bevyengine/bevy#10261)

Bevy's default `WinitSettings` uses `UpdateMode::Continuous`, which spins the winit event loop as fast as possible. **Even an empty Bevy app sits at ~50% CPU on macOS** (Apple Silicon). This is a known Bevy issue — not user code.

Fix: switch to reactive mode so the loop parks between events:

```rust
use bevy::winit::{UpdateMode, WinitSettings};
use std::time::Duration;

app.insert_resource(WinitSettings {
    // max_wait = sim tick rate. Scene can't visibly change faster than the
    // sim produces state, so waiting match-tick-rate between idle redraws is free.
    focused_mode: UpdateMode::reactive_low_power(Duration::from_millis(50)), // 20 Hz sim
    unfocused_mode: UpdateMode::reactive_low_power(Duration::from_secs(1)),
});
```

**Vsync is independent.** `PresentMode::Fifo` controls GPU-side blocking on the swapchain; it does NOT prevent the CPU event loop from spinning. You need BOTH `PresentMode::Fifo` (GPU vsync) AND `WinitSettings::reactive_low_power` (CPU loop).

#### Static geometry → bake once, draw never

NEVER redraw static line/mesh data via per-frame APIs:

```rust
// BAD: ~4k gizmos.line() calls per frame, each pushing to a per-frame CPU buffer
fn draw_overlays_system(mut gizmos: Gizmos, overlays: Res<Overlays>) {
    for &[a, b] in &overlays.grid_lines { gizmos.line(a, b, COLOR); }    // 728 calls
    for &[a, b] in &overlays.isolines   { gizmos.line(a, b, COLOR); }    // 3450 calls
}

// GOOD: bake into a LineList mesh once at Startup, render cost is GPU-only
fn setup_scene(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, overlays: Res<Overlays>) {
    let mesh = build_line_list_mesh(&overlays.grid_lines, &overlays.isolines);
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { unlit: true, ..default() })),
    ));
}
```

#### Asset lifecycle (despawn ≠ release)

Despawning a Bevy `Entity` does NOT release its asset handles. The `Assets<T>` cache is keyed by `AssetId` and only grows.

```rust
// BAD: leaks mesh + material forever. Regrow-on-new-EntityId patterns
// (apples harvested + respawned) compound: each cycle adds N leaked assets.
for (be, _mirror) in &mirrors {
    if !present.contains(&mirror.snap_id) {
        commands.entity(be).despawn();  // entity gone, mesh + material still in cache
    }
}

// GOOD: release assets alongside despawn
for (be, _mirror, mesh_h, mat_h) in &mirrors {
    if !present.contains(&mirror.snap_id) {
        meshes.remove(mesh_h.0.id());
        materials.remove(mat_h.0.id());
        commands.entity(be).despawn();
    }
}
```

Symptom: RSS grows linearly over a long session (e.g. 200MB → 900MB over an hour). The `bevy_egui_orchard_demo` saga hit exactly this.

#### GUI repaint-request storms

`bevy_egui` (and similar immediate-mode GUI integrations) install a `set_request_repaint_callback` that fires `bevy_winit::WakeUp` events on `delay.is_zero()`. **These bypass the reactive `max_wait`** — the loop wakes immediately, processes the egui frame, egui repaints again, fires another WakeUp, infinite 60+ Hz loop.

**Important nuance (verified in egui 0.33 source):** plain `ui.label(format!(...))` does NOT call `Context::request_repaint()`. Repaints originate only from: hover/click animations (only while in-progress), tooltips, scroll areas, `CollapsingHeader`, `SidePanel`/`TopBottomPanel` resize handles, explicit `ctx.request_repaint()` calls. A `format!("frames {}", n)` label whose value changes every frame does NOT itself drive repaints — but it IS wasteful (the displayed number flickers). Throttle per-frame-changing displayed values to 1 Hz for readability, not for perf.

Diagnose via in-process perf overlay before assuming the engine itself is the hog. If `ORCHARD_NO_EGUI=1` drops CPU dramatically, the culprit is egui-side. Use finer-grained bisection (`ORCHARD_NO_PERF_OVERLAY`, `ORCHARD_NO_MINIMAP`, `ORCHARD_NO_DEBUG_PANEL`) to identify WHICH panel — typically it's a resize handle the cursor is parked over, or a hover-state animation that hasn't settled.

Mitigations:
- Override `ctx.set_request_repaint_callback` to enforce a min delay
- Use egui's `RunningMode::Reactive` if exposed
- Accept the cost — UI responsiveness requires repaints on hover/click
- Gate individual panels behind env vars so the user can bisect

#### Frame-rate cap fallback

A `Last`-schedule sleep is a brute-force defense when vsync + reactive mode somehow don't throttle (driver quirks, repaint storms, etc.):

```rust
fn frame_rate_cap_system(mut last_frame_end: Local<Option<Instant>>) {
    const TARGET_FPS: u64 = 60;
    let target = Duration::from_secs_f32(1.0 / TARGET_FPS as f32);
    let now = Instant::now();
    if let Some(last) = *last_frame_end {
        let elapsed = now.duration_since(last);
        if elapsed < target { std::thread::sleep(target - elapsed); }
    }
    *last_frame_end = Some(Instant::now());
}
app.add_systems(Last, frame_rate_cap_system);
```

#### In-process perf overlay

External `ps -p $PID -o %cpu,rss` sampling misses per-frame swings and requires a terminal attached. Ship a perf overlay inside the GUI itself so the user can see what's burning CPU while they interact:

```rust
use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

#[derive(Resource)]
struct PerfMonitor {
    sys: System,
    pid: Pid,
    last_refresh: Instant,
    cpu_pct: f32,
    rss_bytes: u64,
    fps_history: Vec<f32>,
    cpu_history: Vec<f32>,
}

impl PerfMonitor {
    fn refresh(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_refresh) < Duration::from_millis(250) { return; }
        self.sys.refresh_processes_specifics(
            ProcessesToUpdate::Some(&[self.pid]), false, ProcessRefreshKind::everything(),
        );
        if let Some(p) = self.sys.process(self.pid) {
            self.cpu_pct = p.cpu_usage();      // 100.0 = 1 core fully used
            self.rss_bytes = p.memory();        // bytes on macOS, bytes on Linux
        }
        self.last_refresh = now;
    }
}
```

Note `sysinfo` MSRV: 0.32 supports Rust 1.93; 0.39+ requires 1.95. The `sysinfo` API changed between versions (`refresh_process_specifics` in 0.39 → `refresh_processes_specifics(ProcessesToUpdate::Some(&[pid]), ...)` in 0.32).

#### Bisection via env vars

When a perf issue only manifests under specific conditions (focused window, mouse hover, etc.), ship env-var-driven kill switches so the user can isolate the culprit without code changes:

```rust
struct RunFlags {
    no_egui: bool,           // ORCHARD_NO_EGUI=1 — kills entire EguiPlugin
    no_perf_overlay: bool,   // ORCHARD_NO_PERF_OVERLAY=1 — hides only the 📊 perf window
    no_minimap: bool,        // ORCHARD_NO_MINIMAP=1 — hides only the left-panel minimap
    no_debug_panel: bool,    // ORCHARD_NO_DEBUG_PANEL=1 — hides only the bottom debug panel
    no_shadows: bool,        // ORCHARD_NO_SHADOWS=1 — disables shadow-map rendering
    no_sim: bool,            // ORCHARD_NO_SIM=1 — skips the per-tick sim step
    no_overlays: bool,       // ORCHARD_NO_OVERLAYS=1 — skips the static LineList overlay mesh
    bench_secs: Option<f32>, // ORCHARD_BENCH=<n> auto-exits after n seconds
}
```

The user runs each variant for ~10s, notes CPU%, and the env var that drops CPU is the culprit subsystem. **Start coarse (NO_EGUI) then go fine (NO_PERF_OVERLAY / NO_MINIMAP / NO_DEBUG_PANEL)** — this distinguishes "egui as a whole" from "a specific panel" and avoids mis-attributing the cause to the wrong surface.

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

### Don't: Reach for branchless / micro-idioms without measuring

The ShardIndex branchless-search probe measured **REJECTED** — the cheap derived key column won instead, in the same issue on the same benchmark (ndb issue 633 T5). Branchless is a hypothesis, not a win; the boring index-shaped fix frequently beats it.

### Don't: Keep sorted order at every write site

Order-maintenance is a cost paid per mutation, and adding it at a second site just doubles the maintenance. Sorted-insert at the flush path measured **NEGATIVE** (riir-ai issue 1125 Finding 2); bucket-partitioning before the per-tick global sort measured **NEGATIVE** too (Finding 1). Keep order at the ONE site where mutation is rarest/cheapest and let every other path consume it sorted.

### Don't: Clone keys inside scan predicates

Key clones were **60-70%** of a scan's cost; the seek+walk share was only 13-18% (ndb issue 633 T3, declined on measurement). Borrow `&K` through the visitor; clone only at the API boundary.

### Don't: Trust a concurrency test that doesn't rendezvous

An ordering assertion can pass green on scheduler luck — the `put_batch_durable` T2 order race held a "vacuity floor" that was pure scheduling fortune until the test was made to rendezvous participants explicitly. Concurrent-order tests must synchronize their participants (barrier/channel); otherwise the floor is vacuous and the fix ships unproven.

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
