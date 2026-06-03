# Percepta: Transformer-VM in Rust — Full Detail

> TLDR: Rust port of Percepta's transformer-vm — O(log N) 2D convex hull attention, WASM interpreter in transformer weights.
> Distilled ~9K lines Python+C++ into idiomatic Rust. Apache-2.0 → MIT.
> See main README for position in the production stack.

A Rust port of [Percepta's transformer-vm](https://github.com/Percepta-Core/transformer-vm) — a transformer that executes arbitrary C programs by compiling a WebAssembly interpreter into weights, with O(log N) decoding via 2D geometric attention. **The reference is Apache-2.0** — we distilled ~9K lines of Python+C++ into idiomatic Rust: one language, one binary, zero GC. See [Plan 064](.plans/064_percepta_full_riir.md) for the master plan.

### Core Mechanism: Parabolic Key Encoding

The geometric trick that enables exact discrete retrieval in 2D attention heads:

- **Key encoding:** k ↦ (2k, −k²) — points lie on a downward-opening parabola
- **Query direction:** q ↦ (q, 1)
- **Attention score:** 2qk − k² = −(k − q)² + q² — **uniquely maximized when k = q**
- **Hull decoding:** restricting heads to d=2 turns argmax into a supporting-point query on the convex hull → **O(log N)** via ternary search over unimodal dot-product sequence

### Feature Flags

| Flag | Depends On | What It Enables |
|------|-----------|-----------------|
| `percepta` | `ordered-float` | CHT hull cache (upper+lower), `HullMeta`, `TieBreak`, parabolic encoding, `CumSum`, `StandardCache` |
| `percepta_gates` | `percepta` | + ReGLU, stepglu, multiply, persist gate primitives |
| `percepta_graph` | `percepta_gates` | + Expression/Dimension DSL, `ProgramGraph`, `GraphBuilder` |
| `percepta_wasm` | `percepta_graph` | + WASM decoder + lowering + interpreter (pure Rust, not wasmtime) |
| `percepta_compile` | `percepta_wasm` + `good_lp` | + MILP scheduler + weight construction + transformer execution + Futamura specialization + evaluator + runner |

### Implementation Status (Plan 064)

| TG | What | Source | Target | Status |
|----|------|--------|--------|:------:|
| **A** | CHT Hull KV Cache | `hull2d_cht.h` (419 lines) | `cht.rs` + `hull.rs` + `encoding.rs` + `cumsum.rs` + `standard_cache.rs` | ✅ |
| **B** | ReGLU/stepglu gates | `core.py` (gates portion) | `gates.rs` | ✅ |
| **C** | Expression/Dimension DSL | `core.py` (449 lines) | `graph/types.rs` + `graph/mod.rs` | ✅ |
| **D** | MILP scheduling | `milp.py` (814 lines) | `scheduler.rs` | ✅ |
| **E** | WASM decoder + lowering | `decoder.py` + `lower.py` (2472 lines) | `wasm/decoder.rs` + `wasm/lower.rs` | ✅ |
| **F** | WASM interpreter | `interpreter.py` (637 lines) | `wasm/interpreter/` (dispatch, arithmetic, tokens) | ✅ |
| **G** | Weight construction | `weights.py` (776 lines) | `weights.rs` | ✅ |
| **H** | Transformer execution | `transformer.py` + `.cpp` (513 lines) | `transformer.rs` (Rust native, no C++ needed) | ✅ |
| **I** | Futamura specialization | `specialize.py` (148 lines) | `specialize.rs` | ✅ |
| **J** | Evaluator + runner | `evaluator.py` + `runner.py` (705 lines) | `evaluator.rs` + `runner.rs` | ✅ |
| **K** | Examples + docs + benchmarks | `examples/` | Port + benchmark | 🔄 |

**Key result:** ~9K lines Python+C++ → idiomatic Rust. One language, one binary, zero GC.

### Module Structure

```
src/percepta/
├── mod.rs              — Module index + re-exports
├── types.rs            — HullMeta, TieBreak, Vec2, HARD_K constant
├── cht.rs              — Dynamic CHT: Line, CHT (Vec-based LineContainer)
├── hull.rs             — HullHalf + HardAttentionHead + BruteAttentionHead
├── encoding.rs         — Parabolic key encoding: encode_key, encode_query, clear_key
├── cumsum.rs           — Cumulative sum via uniform attention (fetch_sum)
├── standard_cache.rs   — O(n) softmax KV cache reference implementation
├── gates.rs            — ReGLU, stepglu, multiply, persist primitives
├── scheduler.rs        — MILP scheduling (4-phase layer assignment, interval_coloring)
├── weights.rs          — Analytical weight construction: graph + schedule → tensors
├── transformer.rs      — VanillaTransformer with ReGLU FFN + CHT hull cache
├── specialize.rs       — First Futamura projection (program → specialized weights)
├── evaluator.rs        — Graph evaluator with exact arithmetic (no weights needed)
├── runner.rs           — Pipeline runner: compile → build → run → evaluate
├── compile.rs          — C source → WASM → lowered bytecode → token prefix (percepta_compile)
├── legacy.rs           — KVCache2D (Graham Scan) — kept for regression testing
├── graph/
│   ├── mod.rs          — Graph module index + re-exports
│   └── types.rs        — Expression, Dimension, DimensionKind, LookUp, ProgramGraph, GraphBuilder
└── wasm/
    ├── mod.rs          — WASM module index + re-exports
    ├── decoder.rs      — WASM MVP binary decoder (opcode + immediate parsing)
    ├── lower.rs        — Lower unsupported ops (MUL, DIV, etc.) to basic sequences
    └── interpreter/
        ├── mod.rs      — Interpreter builder (universal + specialized modes)
        ├── dispatch.rs — Circle-point opcode dispatch (r²=32045 geometric hashing)
        ├── arithmetic.rs — Byte-serial ALU (add, sub, carry propagation)
        └── tokens.rs   — Input/output token vocabulary construction
```

### Compiler Stack — Component Status

| Component | Description | Status |
|-----------|-------------|:------:|
| **CHT hull cache** | Dynamic CHT: upper+lower hull, `HullMeta` aggregation, `TieBreak` (LATEST/AVERAGE) | ✅ |
| **Parabolic keys** | k → (2k, −k²) with `inv_log_pos * 0.3` tie-break, `clear_key * 1e30` erase | ✅ |
| **Cumulative sum** | `fetch_sum`: uniform attention (AVERAGE tie-break) × position = exact running sum | ✅ |
| **LookUp gates** | Exact key-value retrieval via 2D parabolic attention (`HARD_K=1e10` → hardmax) | ✅ |
| **ReGLU gates** | `relu(b)*a` (1 FFN neuron), `step(b≥0)` (2 neurons), `a*b` (2 neurons + persist) | ✅ |
| **Computation graph** | `Expression` (sparse linear combo) / `Dimension` DAG → intermediate representation | ✅ |
| **MILP scheduling** | `good_lp`/microlp: 4-phase layer assignment, `interval_coloring` slot reuse, minimizes `d_model` | ✅ |
| **WASM decoder** | WASM MVP binary parser: sections, opcodes, immediates, data segments | ✅ |
| **WASM lowering** | MUL, DIV, AND, OR, XOR, SHL, SHR, ROTL, ROTR, CLZ, CTZ, POPCNT → basic op sequences | ✅ |
| **WASM interpreter** | 36 opcodes as circle-point dispatch (r²=32045), byte-serial carry propagation | ✅ |
| **Weight construction** | `expr_to_vector`: graph + schedule → analytical weight matrices, no training needed | ✅ |
| **Transformer execution** | `VanillaTransformer`: autoregressive generation with CHT hull cache, ReGLU FFN | ✅ |
| **Futamura specialization** | `_cursor_lookup`: bake instruction table into FFN weights (smaller, faster model) | ✅ |
| **Universal model** | WASM bytecode as input tokens, instruction fetch via attention at `5*cursor+1` | ✅ |
| **Graph evaluator** | Exact arithmetic evaluation of computation graph (no weights needed) | ✅ |
| **Pipeline runner** | compile → build → run → evaluate orchestration | ✅ |

### What We Implement (Legacy — always available, no feature flags)

- **`KVCache2D`**: Upper convex hull maintenance via Graham Scan (amortized O(1) append)
- **`fast_attention`**: Ternary search over hull vertices → O(log H) where H = hull size
- **`linear_attention`**: O(N) baseline for correctness verification
- **Arithmetic computation**: add, sub, mul, div, mod, power via incremental attention trace
- **DFA execution**: divisible-by-3 state machine verified on 0..=1000
- **Backtracking search**: 4×4 Sudoku, 8-Queens, 9×9 Arto Inkala with hull compression
- **`StreamingSolver`**: Step-by-step solve events matching Percepta's demo output
- **`SymbolicValidator`**: Constraint pruning bridge to speculative decoding (DDTree)

### Verified Properties

- **960 arithmetic ops**: all a+b, a×b, a−b, a÷b for a,b ∈ 0..=10
- **Unimodality**: dot products over hull vertices proven bitonic across 360° query sweep
- **Supporting point**: `linear_attention` ≡ `fast_attention` for convex distributions
- **Hull compression**: backtracking traces compress valleys (dead ends), retain peaks (explorations)
- **V-shape now PASSES**: CHT dual hull handles concave-up (V-shaped) key distributions correctly
- **100K trace stress**: fast attention agrees with linear at scale
- **19 CHT tests**: upper hull, lower hull, V-shape, edge metadata, tie-breaking
- **50 graph tests**: Expression arithmetic, Dimension kinds, ProgramGraph validation
- **23 scheduler tests**: slot reuse, layer assignment, interval coloring
- **22 decoder tests**: WASM binary parsing, opcode sequences, lowering output

**From blog**: k-sparse softmax (nested hulls, O(k + log n)), 3D heads (3D convex hulls), programs into weights (gradient descent no longer the only way to modify a model).

📁 `src/percepta/` — Full module: CHT, hull, encoding, cumsum, gates, graph, scheduler, weights, transformer, specialize, evaluator, runner, wasm/
📁 `.plans/064_percepta_full_riir.md` — **Master plan**: all 11 task groups with tasks, module map, success criteria
📁 `.research/032_percepta_distillation_strategy.md` — **Full RIIR verdict** (why take everything, Apache-2.0 → MIT)
📁 `.research/031_percepta_deep_dive.md` — Gap analysis + **comparison table** (what each Python/C++ does better)
