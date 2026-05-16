# Plan 064: Percepta Full RIIR — transformer-vm in Rust

Complete Rust port of Percepta's `transformer-vm` (Apache-2.0 © Percepta). Distill ~9K lines of Python+C++ into idiomatic Rust under MIT. Prove Rust is better. Show them what's possible.

**Master plan for all Percepta distillation.** Plan 063 (CHT) is Task Group A within this plan.

## Why

1. **Prove Rust beats Python+C++** — One language, one binary, zero GC, deterministic perf
2. **Show Percepta** — Clean Rust port might inspire collaboration, hiring, or mutual respect
3. **We're secure** — Our production pipeline (DFlash, TurboQuant, DDTree) is untouched. This is research code.
4. **Research playground** — Watching a transformer execute WASM bytecodes is cool. Do it because we can.
5. **micro-gpt IS for research** — Slower than native is fine. Proving it works in Rust is the point.

## Reference Source Map

| Source File | Lines | What | Target Rust File |
|-------------|-------|------|-----------------|
| `attention/hull2d_cht.h` | 419 | CHT data structure (upper+lower hull, HullMeta) | `src/percepta/cht.rs` |
| `attention/hull_cache.py` | 44 | Python wrapper for CHT | (merged into cht.rs) |
| `attention/standard_cache.py` | 32 | O(n) softmax reference | `src/percepta/standard_cache.rs` |
| `graph/core.py` | 449 | Expression/Dimension DSL, fetch, reglu, stepglu, persist | `src/percepta/graph.rs` |
| `wasm/interpreter.py` | 637 | 35-opcode WASM machine as computation graph | `src/percepta/wasm/interpreter.rs` |
| `wasm/reference.py` | 667 | Reference trace generator | `src/percepta/wasm/reference.rs` |
| `scheduler/milp.py` | 814 | MILP: 4-phase layer assignment, slot reuse, minimize d_model | `src/percepta/scheduler.rs` |
| `model/weights.py` | 776 | Analytical weight construction: graph → weight matrices | `src/percepta/weights.rs` |
| `model/transformer.py` | ~40 | VanillaTransformer with ReGLU FFN | `src/percepta/transformer.rs` |
| `model/transformer.cpp` | 473 | Standalone C++ inference engine | (Rust native — no separate engine needed) |
| `compilation/compile_wasm.py` | 703 | C→WASM compilation pipeline | `src/percepta/compile.rs` |
| `compilation/decoder.py` | 664 | WASM MVP binary decoder | `src/percepta/wasm/decoder.rs` |
| `compilation/lower.py` | 1808 | Lower unsupported ops (MUL, DIV, AND, etc.) | `src/percepta/wasm/lower.rs` |
| `compilation/runtime.h` | 155 | C runtime for WASM programs | `src/percepta/runtime.h` (keep as-is) |
| `specialize.py` | 148 | First Futamura projection | `src/percepta/specialize.rs` |
| `evaluator.py` | 404 | Graph evaluator (exact arithmetic) | `src/percepta/evaluator.rs` |
| `runner.py` | 301 | CLI runner (C++ or Python inference) | `src/percepta/runner.rs` |
| `build.py` | ~50 | Build universal transformer weights | (merged into weights.rs) |
| **Total** | **~9096** | | |

## Target Module Structure

```
src/percepta/
├── mod.rs              — Module index + re-exports
├── types.rs            — Shared types: Vec2, TieBreak, HullMeta, constants (BIG, HARD_K, etc.)
├── cht.rs              — Dynamic CHT: Line, CHT (BTreeSet-based LineContainer)
├── hull.rs             — HullHalf + HardAttentionHead (upper + lower + edge metadata)
├── standard_cache.rs   — O(n) softmax KV cache reference implementation
├── encoding.rs         — Parabolic key encoding: encode_key, encode_query, clear_key
├── cumsum.rs           — Cumulative sum via uniform attention (fetch_sum)
├── gates.rs            — ReGLU, stepglu, multiply, persist primitives
├── graph.rs            — Expression + Dimension DSL (5 primitive types + ProgramGraph)
├── scheduler.rs        — MILP scheduling (4-phase layer assignment)
├── weights.rs          — Analytical weight construction (graph + schedule → tensors)
├── transformer.rs      — VanillaTransformer with ReGLU FFN, d_model=36, n_heads=18
├── specialize.rs       — First Futamura projection (program → specialized weights)
├── evaluator.rs        — Graph evaluator with exact arithmetic
├── compile.rs          — C→WASM→token prefix pipeline (orchestrates compiler + decoder + lower)
├── runner.rs           — CLI: compile, build, evaluate, specialize, run
├── wasm/
│   ├── mod.rs          — WASM module index
│   ├── decoder.rs      — WASM MVP binary decoder (opcode + immediate parsing)
│   ├── lower.rs        — Lower unsupported ops to supported sequences
│   ├── interpreter.rs  — 35-opcode WASM machine as computation graph
│   └── reference.rs    — Reference trace generator for correctness testing
└── legacy.rs           — KVCache2D (Graham Scan) — kept for regression testing
```

## Task Groups (Dependency Order)

### TG-A: CHT Hull KV Cache (Plan 063)

**Depends on:** Nothing. **Source:** `hull2d_cht.h` (419 lines) + `hull_cache.py` (44 lines)

- [ ] **A1:** Create `src/percepta/` module directory, move `src/percepta.rs` → `src/percepta/mod.rs`
- [ ] **A2:** Implement `TieBreak` enum, `HullMeta` value aggregation in `types.rs`
- [ ] **A3:** Implement `CHT` data structure (BTreeSet-based LineContainer) in `cht.rs`
- [ ] **A4:** Implement `HullHalf` + `HardAttentionHead` in `hull.rs`
- [ ] **A5:** Implement parabolic key encoding in `encoding.rs`
- [ ] **A6:** Implement cumulative sum in `cumsum.rs`
- [ ] **A7:** Implement O(n) softmax reference in `standard_cache.rs`
- [ ] **A8:** Keep legacy `KVCache2D` as `KVCache2DLegacy` in `legacy.rs`
- [ ] **A9:** Port all 30+ existing tests to `HardAttentionHead`
- [ ] **A10:** Integration with `StreamingSolver` + `Sudoku9x9`
- [ ] **A11:** Benchmark: Graham Scan vs CHT throughput

### TG-B: ReGLU Gate Primitives

**Depends on:** TG-A. **Source:** `graph/core.py` (gates portion, ~80 lines)

- [ ] **B1:** Implement `reglu(a, b) = relu(b) * a` — 1 FFN neuron
- [ ] **B2:** Implement `stepglu(a, b) = a * step(b >= 0)` — 2 neurons + persist
- [ ] **B3:** Implement `multiply(a, b) = a * b` — 2 neurons + persist (full multiplication)
- [ ] **B4:** Implement `persist(expr)` — materialize expression into residual slot
- [ ] **B5:** Unit tests: each gate produces correct output for known inputs
- [ ] **B6:** Integration test: gates compose into conditional logic (if/else pattern)

### TG-C: Expression/Dimension DSL

**Depends on:** TG-B. **Source:** `graph/core.py` (449 lines)

- [ ] **C1:** Implement `Expression` — sparse linear combination of dimensions
- [ ] **C2:** Implement `Dimension` enum with 6 variants:
  - `InputDimension` — token embedding values
  - `ReGLUDimension` — relu(b)*a gated FFN unit
  - `PersistDimension` — materialize expression into residual slot
  - `LookUpDimension` — attention-based retrieval from token history
  - `CumSumDimension` — cumulative sum via attention averaging
  - `GenericDimension` — named intermediate value
- [ ] **C3:** Implement `ProgramGraph` — DAG of expressions + dimensions
- [ ] **C4:** Implement `fetch()`, `fetch_sum()`, `reglu()`, `stepglu()`, `persist()` builder methods
- [ ] **C5:** Implement graph validation (cycle detection, dimension consistency)
- [ ] **C6:** Unit tests: build simple computation graphs, verify dimension counts

### TG-D: MILP Scheduling

**Depends on:** TG-C. **Source:** `scheduler/milp.py` (814 lines)

- [ ] **D1:** Add `good_lp` or `highs` crate dependency for MILP solving
- [ ] **D2:** Implement 4-phase layer assignment (Attention → Persist1 → FFN → Persist2)
- [ ] **D3:** Implement `interval_coloring` for slot reuse across layers
- [ ] **D4:** Implement d_model minimization objective
- [ ] **D5:** Implement schedule output: slot assignments, layer map, head allocation
- [ ] **D6:** Unit tests: schedule small programs, verify slot reuse, compare with Python reference output

### TG-E: WASM Decoder + Lowering

**Depends on:** Nothing (parallel with TG-A). **Source:** `decoder.py` (664 lines) + `lower.py` (1808 lines)

- [ ] **E1:** Implement WASM MVP binary decoder (parse header, sections, code sections)
- [ ] **E2:** Parse opcodes + immediates for 35 supported opcodes
- [ ] **E3:** Implement lowering passes for unsupported ops:
  - MUL → ADD-based expansion
  - DIV → SUB-based expansion
  - MOD, AND, OR, XOR, SHL, SHR → supported op sequences
- [ ] **E4:** Handle both constant and variable operands in lowering
- [ ] **E5:** Implement C runtime injection (runtime.h functions)
- [ ] **E6:** Unit tests: decode real WASM binaries, verify opcode sequences match Python reference
- [ ] **E7:** Test: lower MUL program, verify output matches Python-lowered version

### TG-F: WASM Interpreter as Computation Graph

**Depends on:** TG-C + TG-E. **Source:** `interpreter.py` (637 lines)

- [ ] **F1:** Implement circle-point opcode dispatch (r²=32045 geometric hashing)
- [ ] **F2:** Implement 35 opcodes as computation graph nodes:
  - Stack ops: DROP, SELECT, CONST
  - Local/global: LOCAL_GET, LOCAL_SET, LOCAL_TEE, GLOBAL_GET, GLOBAL_SET
  - Memory: LOAD, LOAD8_S/U, LOAD16_S/U, STORE, STORE8, STORE16
  - Control: HALT, RETURN, CALL, BR, BR_IF
  - Comparison: EQZ, EQ, NE, LT_S/U, GT_S/U, LE_S/U, GE_S/U
  - Arithmetic: ADD, SUB, OUTPUT
- [ ] **F3:** Implement byte-serial arithmetic with carry propagation
- [ ] **F4:** Implement stack, memory, locals, cursor, call depth tracking via attention + cumsum
- [ ] **F5:** Unit tests: each opcode produces correct graph node
- [ ] **F6:** Integration: compile + interpret simple C programs (hello, addition, fibonacci)

### TG-G: Analytical Weight Construction

**Depends on:** TG-C + TG-D + TG-F. **Source:** `weights.py` (776 lines)

- [ ] **G1:** Implement `expr_to_tensor` — map graph + schedule → weight matrices
- [ ] **G2:** Implement attention head weight construction (parabolic encoding, HARD_K scaling)
- [ ] **G3:** Implement FFN weight construction (ReGLU gates, slot assignments)
- [ ] **G4:** Implement embedding + unembedding layers
- [ ] **G5:** Verify weight matrices match Python reference for known programs
- [ ] **G6:** Unit tests: construct weights for simple programs, verify dimensional correctness

### TG-H: Transformer Execution

**Depends on:** TG-G. **Source:** `transformer.py` (~40 lines) + `transformer.cpp` (473 lines)

- [ ] **H1:** Implement `VanillaTransformer` with ReGLU FFN (d_model=36, n_heads=18, n_layers=7)
- [ ] **H2:** Integrate CHT hull cache from TG-A as attention backend
- [ ] **H3:** Implement autoregressive generation loop
- [ ] **H4:** Implement token encoding/decoding (byte-level execution trace)
- [ ] **H5:** Verify: run hello.c through full pipeline, output matches Python reference
- [ ] **H6:** Verify: run sudoku.c through full pipeline, solves correctly

### TG-I: Futamura Specialization

**Depends on:** TG-H. **Source:** `specialize.py` (148 lines)

- [ ] **I1:** Implement `_cursor_lookup` — bake instruction table into FFN weights
- [ ] **I2:** Implement piecewise-constant step function encoding
- [ ] **I3:** Implement specialized model generation (smaller, no instruction-fetch attention)
- [ ] **I4:** Verify: specialized collatz matches universal model output but runs faster

### TG-J: CLI + Evaluator + Runner

**Depends on:** TG-H + TG-I. **Source:** `evaluator.py` (404 lines) + `runner.py` (301 lines) + `compile_wasm.py` (703 lines)

- [ ] **J1:** Implement graph evaluator (exact arithmetic, no transformer weights needed)
- [ ] **J2:** Implement reference trace generator (execute WASM directly, produce expected output)
- [ ] **J3:** Implement compile CLI: C source → WASM → lowered → token prefix
- [ ] **J4:** Implement build CLI: token prefix + schedule → transformer weights
- [ ] **J5:** Implement run CLI: weights + token prefix → autoregressive execution
- [ ] **J6:** Implement specialize CLI: universal model → specialized model
- [ ] **J7:** Implement eval CLI: graph evaluator for correctness verification
- [ ] **J8:** End-to-end test: compile → build → run for all example programs (hello, addition, collatz, fibonacci, min_cost_matching, sudoku)
- [ ] **J9:** Benchmark: Rust transformer vs Python transformer vs C++ transformer throughput

### TG-K: Examples + Documentation

**Depends on:** TG-J. **Source:** `examples/` directory

- [ ] **K1:** Port C examples (keep as-is — they're C source, language-agnostic)
- [ ] **K2:** Add Rust-specific examples and benchmarks
- [ ] **K3:** Write module documentation for each `src/percepta/` file
- [ ] **K4:** Update README with full Percepta section (remove "known limitations" as they're fixed)
- [ ] **K5:** Add `percepta_full` feature flag that enables the entire stack
- [ ] **K6:** Write a blog post: "transformer-vm in Rust — 9K lines of Python+C++ → idiomatic Rust"

## Design Decisions

1. **No separate C++ engine** — Rust IS the native engine. The reference has a C++ inference engine because Python is slow. We don't have that problem.

2. **`good_lp` crate for MILP** — Rust LP/MILP solver interface. Can use HiGHS backend (same as reference) or pure-Rust solver. Evaluate after TG-C.

3. **Feature flags** — `percepta_cht` for TG-A only, `percepta_full` for everything. Default off.

4. **File size limit** — `lower.rs` may exceed 2048 lines (source is 1808). Split into `lower/arithmetic.rs`, `lower/logic.rs`, `lower/shift.rs` if needed.

5. **Keep runtime.h as-is** — The C runtime is injected into WASM programs at compile time. It stays as a C header file.

6. **Test against Python reference** — Use `.raw/transformer-vm/` as oracle. Every Rust output must match Python output for the same inputs.

## Dependencies (New Crates)

| Crate | Purpose | Needed By |
|-------|---------|-----------|
| `ordered_float` | `Ord` wrapper for `f64` (breakpoints in CHT) | TG-A |
| `good_lp` or `highs` | MILP solver (4-phase scheduling) | TG-D |

No other new dependencies. We already have: `rayon` (parallelism), `serde` (serialization), `blake3` (hashing).

## Constraints

- Each file < 2048 lines (split if needed)
- All existing tests must continue to pass
- Feature-gated: `percepta_cht` and `percepta_full`, default off
- No Python or C++ dependency at runtime (Rust only)
- Match Python reference output exactly for all example programs
- Apache-2.0 attribution in every file derived from transformer-vm

## Success Criteria

- [ ] All 6 example programs (hello, addition, collatz, fibonacci, min_cost_matching, sudoku) compile and execute correctly through the Rust transformer
- [ ] Output matches Python reference exactly
- [ ] Futamura specialization works (specialized model produces same output as universal)
- [ ] Rust transformer is faster than Python transformer (obvious) and competitive with C++ transformer
- [ ] Graph evaluator matches transformer output (exact arithmetic verification)
- [ ] All TG-A tests pass (CHT fixes V-shape, arbitrary 2D points, cumulative sum)
- [ ] Zero Python dependency at runtime
- [ ] Full module documentation

## References

- `.raw/transformer-vm/` — Full reference implementation (Apache-2.0 © Percepta)
- `.raw/transformer-vm/attention/hull2d_cht.h` — CHT data structure (419 lines)
- `.raw/transformer-vm/graph/core.py` — Expression/Dimension DSL (449 lines)
- `.raw/transformer-vm/wasm/interpreter.py` — WASM interpreter (637 lines)
- `.raw/transformer-vm/scheduler/milp.py` — MILP scheduling (814 lines)
- `.raw/transformer-vm/model/weights.py` — Weight construction (776 lines)
- `.raw/transformer-vm/compilation/lower.py` — WASM lowering (1808 lines)
- `.research/31_percepta_deep_dive.md` — Full gap analysis (9 gaps, P0–P6)
- `.research/32_percepta_distillation_strategy.md` — Full RIIR strategy verdict
- `.research/03_Commercial_Open_Source_Strategy_Verdict.md` — Engine/Fuel split strategy
- `.plans/063_percepta_cht_hull_kv_cache.md` — TG-A detailed plan (CHT upgrade)