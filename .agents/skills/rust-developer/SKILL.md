---
name: rust-developer
description: "Rust engineering conventions — style, cryptography, concurrency, hot-loop optimization, latent-vs-raw domain discipline, build hygiene, debugging (print loops vs the one-attach lldb lane, dev-full locals, Zed lldb-dap), and known footguns. Use for any Rust code change: writing/editing `.rs` files, choosing crates (hashing, concurrency, storage), designing types or hot paths, running cargo/clippy, debugging a failing test or native bug, or reviewing Rust for SOLID/DRY/perf. Not project-specific; project AGENTS.md may override."
---

# Rust Developer

Apply these to every Rust change. Project `AGENTS.md` rules win when they
conflict.

## 1. Style & Idioms

- **File size:** keep `.rs` files under **3200 lines**; prefer **< 2048**. Split
  by responsibility when approaching the limit.
- **IDs:** `Uuid::now_v7()` (time-sortable) — not `Uuid::new_v4()`.
- **Decoupling:** put shared structs/impls in a `types.rs` module so they have
  no logic deps.
- **Control flow:** prefer `match` over chained `if`; use early returns.
- **Enums:** add `#[repr(u8)]` on field-less enums to guarantee 1-byte size.
- **Architecture target:** perf/sec first. Use hexagonal/ports-and-adapters
  only for cold APIs (config, I/O, external services); use ECS for game hot
  paths. SOLID, DRY, modular, generic, decoupled, prod-grade.

## 2. Cryptography, Hashing & Storage

- **Hashing:** `blake3` instead of SHA-1 or SHA-256 (faster + simpler API).
- **Passwords:** `argon2` (not bcrypt/PBKDF2/SHA-*).
- **Encrypted token storage:** turso/libsql with encryption.
- **Identity / fingerprinting:** prefer BLAKE3 of canonical bytes over ad-hoc
  hashing.

## 3. Concurrency

- **Lock-free maps:** prefer `papaya` over `Arc<RwLock<HashMap>>` for hot
  read-mostly maps.
- **Atomic ordering:** model what you actually need; don't default to `SeqCst`
  everywhere, but don't claim a weaker ordering unless you can justify it.

## 4. Performance & Optimization

### Data structures
- Use fixed-size arrays `[T; N]` when the domain is bounded.
- Pre-compute lookup tables once; store in config/context — O(1) reads beat
  O(n) scans.
- Cache allocations: `Vec::with_capacity()` once, then `clear()` + reuse
  across calls.
- Pass pre-allocated scratch buffers as `&mut [T]` parameters instead of
  allocating inside hot loops.

### Allocation discipline
- Pre-build lookup tables and cached data in config structs via a builder.
- Reuse scratch buffers across loop iterations.
- Pre-allocate output arrays upfront; write in-place instead of `collect`-ing
  per-iteration.
- Reorder struct fields to eliminate padding (group by alignment:
  `u64` → `u32` → `u8`).

### SIMD / auto-vectorization
- Write chunked loops (4 or 8 elements) to help LLVM vectorize.
- Keep inner loops **branch-free**.

### Caching
- Compute once per position/context, not per-sample-per-position.
- Pre-compute values that don't change across samples.

### Hot-loop rules (non-negotiable)
- Don't allocate inside hot loops.
- Don't linear-scan for hot-path queries — use a map/index.
- Don't recompute unchanged values.

## 5. Latent vs Raw Space Discipline

When designing state that crosses a sync, replay, or persistence boundary,
classify the domain first:

| Domain | Examples | Representation |
|---|---|---|
| **Physical** | position, velocity, HP, wallet balance | **Raw exact values** — deterministic replay, quorum sync, and anti-cheat need bit-identical reconstruction across nodes. |
| **Semantic** | emotion, mood, curiosity, style, habit | **Latent space** — dot-product + sigmoid projections onto learned direction vectors. |
| **Social** | encounters, relationships, factions | **KG triples** derived from proximity in latent/embedding space, not raw coordinate distance. |

### Anti-patterns (do not do)
- Never encode position as an embedding then decode back for sync — lossy
  round-trip breaks quorum.
- Never use latent similarity to validate movement claims — deterministic
  replay needs exact `(x, y)` at exact tick.
- Never send a full embedding over the network when a scalar projection
  suffices — sync the 5 scalars (valence, arousal, desperation, calm, fear),
  not the 64-dim vector.
- Never gate raw sync correctness behind a feature flag — latent features are
  opt-in; raw sync is always-on.

## 6. Build & Tooling Hygiene

- **Locked cargo?** Use an isolated target dir per task and remove it when
  done:
  ```sh
  CARGO_TARGET_DIR=/tmp/<plan-name> cargo check
  rm -rf /tmp/<plan-name>
  ```
  This avoids stomping on sibling agents' build caches and avoids lock
  contention.
- **Pre-completion gate:** `cargo clippy --all-targets -- -D warnings`
  (zero warnings), then `cargo clean` if you want a pristine state.
- **Combo-regression check:** when adding a field to a `#[derive]`-heavy type
  or a constructor, run `cargo test --all-features` (or
  `cargo hack --each-feature` if available) — single-feature checks miss the
  class of bugs where no single feature turns on all field writers.
- **Feature flag discipline:** every new primitive ships behind an opt-in
  feature; promote to default-on only after a GOAT-style gate
  (correctness + perf + no-regression + alloc/quality) passes.

## 7. Known Footguns

- **`sqlx::raw_sql` + `fetch_optional`:** despite the name, `fetch_optional`
  on a `raw_sql` query actually returns `Result<Row>`, **not**
  `Result<Option<Row>>`. Use `fetch_one` semantics or wrap manually.
- **Adding a field to a `Pod`/`#[repr(C)]` struct:** audit **every**
  constructor (`new`, `new_unchecked`, `from_bytes`, any builder) — it's easy
  to forget one and leave the field uninitialized. Single-feature CI won't
  catch it; `--all-features` will.
- **`papaya` HashMaps with multiple maps in one type:** each map has its own
  epoch collector, so a single external guard cannot span both. Pin
  internally per method (cheap) and document it.
- **AI borrow-checker appeasement (audit AI-ported Rust for this class):**
  code can compile clean and pass deterministic replay hashes while being
  architecturally hollow. The tells and their fixes:
  - `std::mem::take(&mut self.field)` check-out/check-in — the field is
    EMPTY between the take and the reassign; any nested call that reads it
    sees nothing. Fix: disjoint borrows (`split_at_mut`), split the struct
    into sub-structs, or pass `&mut` through the helpers.
  - `.clone()`/`.to_vec()`/`.collect()` immediately before a loop that
    mutates `self` — the clone exists only to drop a borrow, and it pays
    per tick. Fix: index-based iteration, disjoint sub-structs, or
    restructure the borrows — never clone per-tick to appease rustc.
  - `thread::spawn`/`thread::scope` inside a per-frame/per-item function —
    spawn cost is paid every call. Fix: build the pool (rayon
    `ThreadPool`, worker ring) once and reuse it.
  - Cloning a large buffer (pixel frame, audio block) into an `Arc` per
    send — a full memcpy per frame. Fix: move the buffer, or recycle
    through a bounded pool with the receiver returning buffers. Clone only
    small payloads.
  - Dense id domains (`0..n` unit ids) behind `HashMap`/`HashSet` — a flat
    slot-indexed array answers by index: no hashing, no rehash on growth.

## 8. Debugging — the lldb escape hatch

Print lanes first — unit-test `println!`/`dbg!`, tracing, the app's own debug
pods. Reach for a real debugger only for the residual class: segfault/UB, a
hang, nondeterminism caught in the act, or state unreachable without a
rebuild per question.

**The one-attach law — compile once, ask many questions.** A print lane pays
one recompile per question; an attached debugger pays once and then every
inspection is ~free. Never recompile to change what you look at:

```sh
cargo test --profile dev-full --no-run -p <pkg>   # dev-full = real locals
ls -t target/debug/deps/<pkg>-* | grep -v -e '\.d$' -e '\.rlib$'   # the test binary
rust-lldb --batch \
  -o 'breakpoint set -f <file>.rs -l <line>' \
  -o run -o 'frame variable' -o quit \
  -- target/debug/deps/<pkg>-<hash> <test_name> --exact --test-threads=1
# <test_name> = FULL path (tests::<name>) — a bare name silently runs 0 tests.
# at the stop: frame variable / p <expr> / bt / c — `rust-gdb` is the Linux twin
```

- **Profile coupling (fleet law):** the build-hygiene discipline sets
  `[profile.dev] debug = "line-tables-only"` with deps at `debug = false` —
  backtraces keep file/line, LOCALS DO NOT EXIST. Real inspection needs the
  `[profile.dev-full]` escape hatch (`inherits = "dev"`, `debug = true`;
  present in every repo carrying the discipline). Pay the dev-full tax once
  per attach, never per question.
- **Determinism first:** pin a nondeterministic failure before inspecting it —
  `--test-threads=1`, seeded RNG, one repro. The debugger sees one
  interleaving; make it the pinned one.
- **Toolchain:** macOS — rustup shims `rust-lldb`/`rust-gdb` on PATH, Apple
  lldb at `/usr/bin/lldb`; Zed's built-in DAP debugger wants `lldb-dap` on
  PATH (Xcode ships it at
  `/Applications/Xcode.app/Contents/Developer/usr/bin/lldb-dap` — symlink it
  into a PATH dir when absent) and a per-project `.zed/debug.json` is the
  owner's interactive lane. Zed agents get NO debugger tools — the CLI lane
  above IS the agent lane. Linux — `rust-gdb`/`lldb-dap` from the distro
  toolchain.
- **Honest limits:** async stepping and macro-heavy frames still step badly
  (JetBrains Rust debugging survey, 2026-11-10) — prefer repro + prints
  there; the debugger earns its keep on native-core bugs. `-f` takes the
  BASENAME (`-f lib.rs`); an absolute path can stay pending. lldb may warn
  "no plugin for the language rust" — the rustup shim's formatters still
  render the common shapes (Option, slices, POD structs).
- **Never quote a loop number without box state** (bench_preflight
  PROVENANCE / rig status — the house law). Fleet record:
  `.benchmarks/986_rust_lldb_debug_loop.md` — attach-vs-print A/B, dev-full
  tax, break-even.

## 9. Response Format

- Always end a Rust-task response with a **TL;DR**.
- Define technical terms inline on first use (e.g., "Pod (plain-old-data,
  `unsafe` to byte-cast)"); don't assume prior context.
