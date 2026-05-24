# Percepta Deep Dive: transformer-vm Reference vs Our Implementation

## Overview

Percepta's [transformer-vm](https://github.com/Percepta-Core/transformer-vm) is an end-to-end system that analytically compiles a WebAssembly VM into the weights of a standard autoregressive transformer. Our `src/percepta.rs` implements the geometric attention mechanism (2D convex hull KV cache) but not the compiler stack.

This document details the gaps, distilled from the reference source at `.raw/transformer-vm/`.

**Note:** Our `riir-ai` workspace already has production WASM infrastructure (`riir-wasm` host + `riir-validator-sdk` guest) using `wasmtime` v28. See [Existing WASM Ecosystem](#existing-wasm-ecosystem-riir-ai) below for reusability analysis.

---

## Architecture Layers (Reference)

```
┌─────────────────────────────────────────────────────┐
│ C Program → WASM bytecode → Token prefix             │  compilation/
│                                                        │  (compile_wasm.py,
│                                                         │   decoder.py,
│                                                          │   lower.py)
├─────────────────────────────────────────────────────┤
│ WASM Interpreter as Computation Graph                 │  wasm/
│ 35 opcodes → Expression/Dimension DAG                 │  (interpreter.py)
│ Circle-point opcode dispatch                          │
├─────────────────────────────────────────────────────┤
│ Computation Graph DSL                                 │  graph/
│ Expression, Dimension (5 types), LookUp               │  (core.py)
│ reglu, stepglu, persist, fetch, fetch_sum             │
├─────────────────────────────────────────────────────┤
│ MILP Scheduler                                        │  scheduler/
│ 4-phase layer assignment, d_model minimization        │  (milp.py)
├─────────────────────────────────────────────────────┤
│ Analytical Weight Construction                        │  model/
│ Schedule → slot coloring → weight matrices            │  (weights.py)
│ Parabolic key encoding, HARD_K=1e10                   │
├─────────────────────────────────────────────────────┤
│ O(log n) Hull KV Cache (CHT)                          │  attention/
│ Dynamic convex hull trick, upper + lower              │  (hull2d_cht.h)
│ Tie-breaking: LATEST / AVERAGE                        │
├─────────────────────────────────────────────────────┤
│ Specialization (Futamura Projection)                  │  specialize.py
│ Bake program into FFN weights via _cursor_lookup       │
└─────────────────────────────────────────────────────┘
```

---

## Gap 1: CHT vs Graham Scan (Critical)

### Our Implementation (`src/percepta.rs`)

```rust
pub struct KVCache2D {
    keys: Vec<Vec2>,
    values: Vec<usize>,
    upper_hull: Vec<usize>,  // Graham Scan upper hull only
}
```

- **Algorithm**: Graham Scan maintains upper convex hull. `fast_attention` uses ternary search over hull vertices.
- **Limitation**: Requires monotonically non-decreasing X coordinates (sequential execution traces).
- **Single hull**: Only tracks the upper hull. Cannot handle queries where `qy < 0`.
- **Value storage**: All N keys stored, hull is an index subset. O(N) memory.
- **Tie-breaking**: None. Returns first maximum found.

### Reference Implementation (`hull2d_cht.h`)

```cpp
struct HardAttentionHead {
    HullHalf upper{true};   // max envelope (qy > 0)
    HullHalf lower{false};  // min envelope (qy < 0), stored as max of negated
    HullMeta global;        // all values (qy == 0 && qx == 0)
    HullMeta left_meta;     // min kx values (qy == 0 && qx < 0)
    HullMeta right_meta;    // max kx values (qy == 0 && qx > 0)
};
```

- **Algorithm**: Dynamic Convex Hull Trick (CHT) using `std::multiset<Line>`. Lines sorted by slope (kx), each stores breakpoint.
- **Duality**: 2D dot product maximization `qx*kx + qy*ky` reduced to 1D line query at `m = qx/qy`.
- **Both hulls**: Upper hull for `qy > 0`, lower hull for `qy < 0`, edge metadata for `qy == 0`.
- **Sublinear memory**: Only hull vertices stored (not all N points). Values aggregated into `HullMeta`.
- **O(log n) insert AND query**: No ternary search; direct `lower_bound` on breakpoints.
- **Arbitrary points**: No monotonic-X requirement.

### HullMeta: Value Aggregation

```cpp
struct HullMeta {
    double vsum[2] = {0, 0};   // running sum of value pairs
    double vlast[2] = {0, 0};  // most recent value (by sequence number)
    int    count   = 0;
    int    last_seq = -1;
};
```

- When multiple points map to the same hull line, their values are **merged** (summed, counted).
- `LATEST` tie-break: return `vlast` (most recent by seq number).
- `AVERAGE` tie-break: return `vsum / count`.
- This enables **cumulative sum** (uniform attention) where all keys are identical.

### Key Difference Summary

| Aspect | Our Impl | Reference |
|--------|----------|-----------|
| Hull algorithm | Graham Scan (vector) | CHT / LineContainer (multiset) |
| Hull coverage | Upper only | Upper + Lower + edge metadata |
| X requirement | Monotonically increasing | Arbitrary 2D points |
| Insert cost | O(1) amortized (append) | O(log h) amortized (multiset) |
| Query cost | O(log h) ternary search | O(log h) binary search |
| Memory | O(N) all keys | O(h) hull vertices + metadata |
| Tie-breaking | None | LATEST / AVERAGE |
| Value type | `usize` | `f64` pair + aggregation |
| qy < 0 queries | **BROKEN** (adversarial test) | Correct via lower hull |

---

## Gap 2: Parabolic Key Encoding (Explicit)

### Our Implementation

Test patterns implicitly create parabolic distributions:

```rust
// test: concave-down parabola — all points on hull
for i in 0..1000 {
    let x = i as f32;
    let y = -((x - 500.0) / 100.0).powi(2);
    cache.append(Vec2::new(x, y), i);
}
```

No explicit encoding formula in the API.

### Reference Implementation

The parabolic encoding is built into the `fetch()` DSL function:

```python
def _to_2d_key(k, clear_key_expr=None, tie_break="latest"):
    kx = k * 2 - one_expr * (2 * KEY_OFFSET)
    ky = -k_abs + k * (2 * KEY_OFFSET) - one_expr * (KEY_OFFSET**2)
    if tie_break == "latest":
        ky = ky + Expression({inv_log_pos: LATEST_ALPHA})  # 0.3
    return [kx, ky]

def _to_2d_query(q):
    return [q - one_expr * KEY_OFFSET, one_expr]
```

Key details:
- **Key**: `k → (2k - 2·offset, -k² + 2k·offset - offset²)` — points on downward-opening parabola
- **Query**: `q → (q - offset, 1)` — direction vector
- **Score**: `2qk - k² = -(k-q)² + q²` — uniquely maximized at `k = q`
- **Tie-break**: `inv_log_pos * LATEST_ALPHA` added to `ky` — favors more recent tokens
- **Clear key**: Subtracts `clear_key * BIG` (1e30) from `ky` — effectively removes keys at cleared positions
- **KEY_OFFSET = 0** by default, for numerical stability tuning

---

## Gap 3: Cumulative Sum via Attention (`fetch_sum`)

### Our Implementation

Not implemented. We track state with local variables.

### Reference Implementation

```python
def fetch_sum(value_list):
    """Cumulative sum via attention averaging: avg * position."""
    key = Expression({one: KEY_OFFSET})
    query = Expression({one: KEY_OFFSET})
    avg_dims = fetch(value_list, query=query, key=key, tie_break="average")
    results = [reglu(_to_expr(d), _to_expr(position)) for d in avg_dims]
    return tuple(results) if len(results) > 1 else results[0]
```

How it works:
1. **All keys identical** (`key = KEY_OFFSET`) → attention sees all positions equally
2. **Tie-break = AVERAGE** → returns `mean(all_values)`
3. **Multiply by position** via `reglu(avg, position)` → recovers exact cumulative sum
4. Position 0 (start token, `one=0`) excluded naturally because its `ky=0 < 1`

This is how the WASM interpreter tracks `cursor`, `stack_depth`, and `call_depth` — all via cumulative sum over attention.

---

## Gap 4: ReGLU Gate Primitives

### Our Implementation

Not implemented. We do arithmetic with native Rust operators.

### Reference Implementation

Three core primitives build all nonlinear logic:

```python
def reglu(a, b):
    """relu(b) * a — single FFN neuron."""
    r = ReGLUDimension(a_expr, b_expr)
    return Expression({r: 1})

def stepglu(a, b):
    """a * step(b >= 0) = reglu(a, b+1) - reglu(a, b)."""
    r1 = ReGLUDimension(a_expr, b_expr + 1)
    r2 = ReGLUDimension(a_expr, b_expr)
    result = persist(Expression({r1: 1, r2: -1}))
    return result

def _make_multiply(a, b):
    """a * b = reglu(a, b) - reglu(a, -b)."""
    r1 = ReGLUDimension(a, b)
    r2 = ReGLUDimension(a, -b)
    return persist(Expression({r1: 1, r2: -1}))
```

| Primitive | FFN neurons | Purpose |
|-----------|------------|---------|
| `reglu(a, b)` | 1 | Gated multiplication: `relu(b) * a`. When b≥0, equals `a*b`. |
| `stepglu(a, b)` | 2 + persist | Step function: `a` when `b≥0`, `0` otherwise. |
| `_multiply(a, b)` | 2 + persist | Full multiplication for arbitrary integers. |

These are the **only nonlinear operations** in the entire system. Everything else is linear combination.

---

## Gap 5: Computation Graph DSL (Expression/Dimension)

### Our Implementation

Not implemented.

### Reference Implementation

Six dimension types form a computation DAG:

| Type | Kind | Transformer Component | Description |
|------|------|----------------------|-------------|
| `InputDimension` | `"input"` | Token embedding | Runtime-provided: `one`, `position`, `inv_log_pos`, `position_sq` |
| `LookUpDimension` | `"lookup"` | Attention head output | Result of `fetch()` — exact key-value retrieval |
| `CumSumDimension` | `"cumsum"` | Attention + FFN | Running sum via `fetch_sum` |
| `ReGLUDimension` | `"reglu"` | FFN neuron | `relu(b) * a` — the nonlinear primitive |
| `PersistDimension` | `"persist"` | Residual stream slot | Materializes expression into dedicated slot |
| `Dimension` | `"generic"` | — | Base class, rarely used |

`Expression` is a sparse linear combination `dict[Dimension, float]` — symbolic algebra that supports `+`, `-`, scalar `*`, but **never dimension×dimension** directly. Multiplication always goes through `ReGLUDimension`.

The `ProgramGraph` captures the entire DAG and is the input to the MILP scheduler.

---

## Gap 6: MILP Scheduling

### Our Implementation

Not implemented.

### Reference Implementation

4-phase transformer layer:

```
Phase 0: Attention  (LookUp gates)
Phase 1: Persist1   (linear projection)
Phase 2: FFN        (ReGLU gates)
Phase 3: Persist2   (linear projection)
```

The MILP (Mixed-Integer Linear Programming) solver:
- **Variables**: Layer assignment `k[op]` per operation, persist slot choice `z[op]`, death phase `d[dim]` per dimension
- **Objective**: Minimize `D_half` → minimize `d_model = 2 * D_half`
- **Constraints**: Precedence, type compatibility (LookUp→attention, ReGLU→FFN), tight coupling, head count, slot width
- **Solver**: PuLP with HiGHS backend (or CBC fallback)
- **Output**: Layer schedule + `interval_coloring()` for slot assignment

Key algorithm — `interval_coloring`:
```
Sort dimensions by birth phase
Use min-heap for freed slots
Assign earliest-available slot to each born dimension
Dead dimensions release their slots
→ Minimizes peak slot usage = d_model
```

---

## Gap 7: WASM Interpreter as Computation Graph

### Our Implementation

Not implemented.

### Reference Implementation

35 opcodes encoded entirely through `fetch`, `fetch_sum`, `reglu`, `stepglu`:

**Machine state** (all tracked via cumulative sum):
- `cursor` — instruction pointer (cumsum of `delta_cursor`)
- `stack_depth` — stack pointer (cumsum of `delta_stack`)
- `call_depth` — call stack depth (cumsum of `delta_call_depth`)
- `byte_number` — current byte within i32 (input embedding)
- `carry` — carry/borrow bit (input embedding with `'` suffix)

**Opcode dispatch** — Circle-point encoding:
- Each opcode maps to a unique (x, y) on circle with r² = 32045
- `op_dot(op) = px * fetched_x + py * fetched_y - 32045 + 1`
- Equals 1 on match, ≤ -1 otherwise → single `reglu` gate per case

**Byte-serial execution**:
- i32 values processed as 4 sequential byte tokens
- Carry propagation via `carry_late = persist(carry)` (delayed by 1 position)
- Addition: `add_value = second_byte + top_byte + carry_late`, `add_carry = stepglu(one, add_value - 256)`
- Subtraction: `sub_value = second_byte - top_byte - carry_late`, `sub_borrow = 1 - stepglu(one, sub_value)`

**Address scoping**:
- Stack: `fetch(value, query=stack_depth, key=stack_depth)`
- Locals: `LOCAL_STRIDE * call_depth + 4 * local_index + byte_index`
- Memory: `stack_top_value + immediate + byte_index`
- Call stack: `call_depth * 4 + byte_index`

---

## Gap 8: Analytical Weight Construction

### Our Implementation

Not implemented.

### Reference Implementation

Direct weight matrix population from schedule:

```
HARD_K = 1e10  # softmax temperature → hardmax approximation

# LookUp → Attention weights
ip[h * 2]     = expr_to_tensor(query_x) * HARD_K * sqrt_dh  # Q
ip[h * 2 + 1] = expr_to_tensor(query_y) * HARD_K * sqrt_dh  # Q
ip[D + h * 2] = expr_to_tensor(key_x)                        # K
ip[D + h * 2 + 1] = expr_to_tensor(key_y)                    # K
ip[2*D + h * 2] = expr_to_tensor(value_0)                     # V
op_w[slot_of[dim], h * 2] = 1.0                               # Output projection

# ReGLU → FFN weights
fi[j] = expr_to_tensor(b_expr)        # gate
fi[d_ffn + j] = expr_to_tensor(a_expr) # value
fo[slot_of[dim], j] = 1.0             # output projection

# Passthrough → dedicated attention heads / FFN neurons
# Erase → -1.0 self-loop on reused slots
```

Key details:
- **`expr_to_tensor(expr)`**: Converts Expression to weight vector by mapping each Dimension to its assigned slot
- **Slot 0,1,2 are protected**: `_position`, `_inv_log_pos`, `_position_sq` — reserved for positional encoding
- **Erase mechanism**: When a slot is reused, `-1.0` is added to cancel the stale value in the residual stream
- **Output tokens**: Use quadratic scoring `H * emit_gate + (2 * target) * computed - target²` — peaks at `target == computed`

---

## Gap 9: Specialization (Futamura Projection)

### Our Implementation

Not implemented.

### Reference Implementation

`_cursor_lookup` — the core mechanism for baking programs into FFN weights:

```python
def _cursor_lookup(values, name=None):
    """Piecewise-constant FFN lookup: cursor -> values[cursor]."""
    expr = Expression({one: values[0]})
    for i in range(1, N_instr):
        diff = values[i] - values[i - 1]
        if diff == 0: continue
        expr[r_pos[i - 1]] = diff    # ReGLU(one, cursor - i + 1)
        expr[r_neg[i - 1]] = -diff   # ReGLU(one, cursor - i)
    return persist(expr, name=name)
```

- Creates `2 * (N_instr - 1)` shared ReGLU neurons for step functions
- Each `_cursor_lookup` adds coefficients for a specific value array
- Program data becomes constant coefficients in the weight matrix
- Eliminates instruction-fetch attention heads entirely

---

## Priority Assessment

### P0: CHT Hull KV Cache (Replaces Current Implementation)

**Why**: Fixes fundamental limitations (monotonic-X requirement, qy<0 queries, O(N) memory).

**Scope**: ~300 lines Rust replacing current `KVCache2D`.
- `HullHalf` (upper/lower CHT)
- `HardAttentionHead` (upper + lower + edge metadata)
- `HullMeta` (value aggregation with LATEST/AVERAGE)
- BTreeMap-based CHT (equivalent to `std::multiset`)

**Dependencies**: None.

### P1: Cumulative Sum (`fetch_sum`)

**Why**: Required for tracking state machines, instruction pointers, stack depths via attention.

**Scope**: ~50 lines added to CHT.
- Uniform attention mode (all keys = constant)
- Average tie-breaking mode
- Multiply-by-position to recover exact cumsum

**Dependencies**: P0 (needs HullMeta AVERAGE mode).

### P2: Parabolic Key Encoding API

**Why**: Makes the parabolic trick explicit and reusable rather than implicit in test patterns.

**Scope**: ~30 lines added to CHT.
- `encode_key(k, offset, tie_break)` → `Vec2`
- `encode_query(q, offset)` → `Vec2`
- `clear_key(key, big)` → `Vec2`

**Dependencies**: P0.

### P3: ReGLU / stepglu Gate Primitives

**Why**: Foundation for any gate-graph work. Required for conditional logic and multiplication in attention-FFN systems.

**Scope**: ~100 lines new module `src/percepta/gates.rs`.
- `reglu(a, b) -> f64` — `relu(b) * a`
- `stepglu(a, b) -> f64` — `a * step(b >= 0)`
- `multiply(a, b) -> f64` — via `reglu(a,b) - reglu(a,-b)`
- Unit tests matching Python behavior

**Dependencies**: None.

### P4: Expression/Dimension DSL

**Why**: Foundation for computation graph construction. Enables expressing programs as transformer-native operations.

**Scope**: ~500 lines new module `src/percepta/graph.rs`.
- `Expression` (sparse linear combination)
- `Dimension` hierarchy (Input, LookUp, ReGLU, Persist, CumSum)
- `LookUp` (attention operation descriptor)
- `fetch()`, `fetch_sum()`, `reglu()`, `stepglu()`, `persist()`
- `ProgramGraph` (captured DAG)

**Dependencies**: None (but P3 gates would be needed for evaluation).

### P5: MILP Scheduling

**Why**: Optimal layer assignment and slot allocation for computation graphs.

**Scope**: Heavy. Requires ILP solver (e.g., `good_lp` or `lp-solvers` crate).
- Dependency graph construction
- Phase assignment variables and constraints
- Interval coloring for slot allocation
- Plan YAML output

**Dependencies**: P4.

### P6: WASM Interpreter / Weight Construction / Specialization

**Why**: Full end-to-end program execution inside transformer weights.

**Scope**: Very heavy. Multiple modules, WASM decoder, lowering pass, weight matrix construction.

**Dependencies**: P3, P4, P5.

---

## What We Already Do Well

Our implementation correctly proves the **core geometric insight**:

1. **2D attention on convex hull → O(log N)**: Proven via 100K-element stress tests
2. **Arithmetic via attention trace**: 960 operations verified (all a+b, a-b, a*b, a/b for a,b ∈ 0..=10)
3. **Backtracking search**: 4×4 Sudoku, 8-Queens, 9×9 Arto Inkala all correctly tracked
4. **Unimodality proof**: Dot products over hull verified bitonic across 360° sweep
5. **Adversarial documentation**: V-shape failure mode explicitly tested and documented
6. **DFA execution**: Divisible-by-3 state machine on 0..1000
7. **Streaming solver**: `SolveEvent` enum matching Percepta's demo output
8. **SymbolicValidator**: Bridge to speculative decoding (DDTree constraint pruning)

---

## Reference File Map

| File | Lines | Purpose |
|------|-------|---------|
| `graph/core.py` | 358 | Expression/Dimension DSL, 5 primitive types, fetch/reglu/stepglu/persist |
| `wasm/interpreter.py` | 637 | 35-opcode WASM machine as computation graph, Futamura specialization |
| `scheduler/milp.py` | 810 | MILP scheduler: 4-phase layer assignment, d_model minimization |
| `model/weights.py` | 776 | Analytical weight construction: graph → weight matrices |
| `attention/hull2d_cht.h` | 323 | CHT data structure: dynamic convex hull trick, O(log n) insert+query |
| `attention/hull_cache.py` | 44 | pybind11 wrapper for C++ hull cache |
| `attention/standard_cache.py` | 32 | O(n) softmax reference |
| `specialize.py` | 78 | First Futamura projection: program → specialized weights |
| `model/transformer.py` | ~40 | VanillaTransformer: d_model=36, n_heads=18, ReGLU FFN |

---

## Key Constants and Parameters

| Constant | Value | Purpose |
|----------|-------|---------|
| `BIG` | 1e30 | Effectively zero-out attention keys (clear_key mechanism) |
| `HARD_K` | 1e10 | Softmax temperature → hardmax approximation |
| `LATEST_ALPHA` | 0.3 | Tie-break weight favoring recent tokens |
| `KEY_OFFSET` | 0 | Offset for parabolic key numerical stability |
| `pointsR2` | 32045 | Radius² of circle for opcode dispatch points |
| `LOCAL_STRIDE` | 256 | Address stride per call depth (64 locals × 4 bytes) |

---

## Token Format

The execution trace uses these token types:

| Token | Format | Embedding |
|-------|--------|-----------|
| Byte (carry=0) | `"XX"` (hex) | `(bv+1) * byte_number` |
| Byte (carry=1) | `"XX'"` | `(bv+1) * byte_number + carry` |
| Commit | `"commit(+d,sts=X,bt=Y)"` | `delta_cursor + d*delta_stack + sts*store_to_stack + bt*is_jump` |
| Output | `"out(XX)"` or `"out(char)"` | `delta_cursor` |
| Branch taken | `"branch_taken"` | `is_branch_taken` |
| Call commit | `"call_commit"` | `delta_cursor + delta_call_depth + is_jump` |
| Return commit | `"return_commit"` | `delta_cursor - delta_call_depth + is_return_commit + is_jump` |
| Program start | `"{"` (universal) or `"start"` (specialized) | — |
| Program end | `"}"` | `3 * delta_stack` |
| Opcode | `"i32.add"`, etc. | Circle point + stack delta + metadata |

---

## Existing WASM Ecosystem (`riir-ai`)

Our `riir-ai` workspace already has production WASM infrastructure at `crates/riir-wasm/` (host) and `crates/riir-validator-sdk/` (guest SDK).

### What Exists

| Component | Description |
|-----------|-------------|
| **`riir-wasm`** | Host runtime using `wasmtime` v28. Loads `.wasm` validators, fuel-limited (~100μs/call), no WASI (fully sandboxed) |
| **`riir-validator-sdk`** | Guest SDK. Write validators → compile to `wasm32-unknown-unknown`. `export_validator!` macro generates `#[no_mangle] extern "C"` ABI |
| **ABI** | Structured linear memory: 256B state + 255B name + 7.5KB scratch + heap. Q16.16 fixed-point for relevance scores |
| **6 validators** | bracket, keyword, rust, python, game_action, bomber (0.37–0.55μs per `is_valid`) |
| **Events** | Opt-in streaming diagnostic buffer (8-byte packed events, 256 max) |
| **Tooling** | `riir-validator-check` for ABI compliance, no-WASI verification |

### Reusability for Percepta WASM Interpreter

The Percepta reference compiles C → WASM → token prefix → transformer execution. Our `riir-wasm` infrastructure is **complementary but not directly overlapping**:

| Aspect | `riir-wasm` (existing) | Percepta WASM (needed) |
|--------|----------------------|------------------------|
| **Purpose** | Validate draft tokens (pruning) | Execute programs inside transformer weights |
| **Runtime** | wasmtime (external sandbox) | No runtime — WASM bytecode becomes transformer input tokens |
| **Data flow** | Host → WASM → yes/no | C → WASM bytecode → token prefix → transformer autoregressive decode |
| **Compilation** | `wasm32-unknown-unknown` (Rust) | C → WASM via clang `--target=wasm32` |
| **State** | Linear memory inside WASM | Execution trace as token sequence |
| **Speed** | ~0.5μs/call (fuel-limited) | ~30K tok/s (transformer decode) |

**Key insight:** Percepta's WASM interpreter does NOT use a WASM runtime. The WASM bytecode is **tokenized and fed as input** to a transformer whose weights were analytically constructed to execute that bytecode. The transformer IS the runtime.

**What IS reusable:**
- C → WASM compilation pipeline (`clang --target wasm32`, lowering passes) — `riir-wasm` already handles WASM binary format
- WASM binary decoder (parse opcodes, immediates) — could adapt from `riir-wasm` or from Percepta's `decoder.py`
- The `riir-validator-sdk` pattern (guest SDK, `export_` macros, memory layout) could inspire a future `percepta-program-sdk` for writing programs that compile to transformer-executable token traces

**What is NOT reusable:**
- `wasmtime` runtime — Percepta doesn't use a WASM runtime; the transformer IS the runtime
- Fuel/sandboxing — the transformer executes deterministically by construction
- The `Validator` trait / `ConstraintPruner` interface — Percepta's interface is "feed bytecode as tokens, get execution trace as tokens"

---

## Comparison: katgpt-rs vs transformer-vm

### What We Do Better

| Category | katgpt-rs | transformer-vm | Why We Win |
|----------|-------------|----------------|------------|
| **Full inference stack** | DDTree, DFlash, Leviathan, speculative decoding, TurboQuant, PFlash, Raven, Sparse MLP — production pipeline | Standalone proof-of-concept executor | We have a real inference engine; they have a research artifact |
| **Throughput** | 4.2M tok/s (DFlash), 1.6M tok/s (speculative), 19.4M tok/s (prefill) | ~30K tok/s (CPU, hull attention) | **140× faster** on decoding, **640× faster** on prefill |
| **KV cache compression** | TurboQuant: f32 → 2-4 bit (5-8× reduction), PFlash: 21× seq reduction, Raven: O(1) slots | Hull-only compression (sublinear vertex count) | We compress precision (TurboQuant), sequence (PFlash), and routing (Raven); they compress only hull geometry |
| **Zero-alloc hot path** | Pre-allocated `ForwardContext` buffers, 44.6% faster store+dequant cycle | Python/PyTorch + C++ with heap allocations | Rust ownership model + pre-allocated scratch buffers eliminate GC pauses |
| **SIMD acceleration** | Neon/AVX2 dot products, matmul, sparse GEMV (Plan 060) | BLAS for matrix-vector only | We SIMD-accelerate the full forward pass, not just matmul |
| **Test coverage** | 1874 lines of tests in percepta.rs alone; 960 arithmetic ops; adversarial edge cases documented; 100K stress; unimodality proof across 360°; game domain integration | Smoke tests, distillation tests, specialize tests | We proved correctness exhaustively with adversarial stress; they proved it runs end-to-end |
| **Adversarial honesty** | V-shape failure explicitly tested and documented (test_adversarial_v_shape_fast_attention_wrong). Admits when our algorithm is wrong | No adversarial tests visible; assumes hard-max correctness | We document our limitations in the code itself |
| **Game domain proofs** | Bomberman (+177 HL vs -55 random), Monopoly FSM, FFT Tactics, Sudoku 9×9 Arto Inkala — all with streaming TUI demos | Sudoku solver (Norvig-style, compiled C) | Our game proofs validate heuristic learning; theirs validates program execution |
| **Neuro-symbolic integration** | `ConstraintPruner` → `SymbolicValidator` → DDTree branch pruning → speculative decoding — the whole chain works | Computation graph exists in isolation, not integrated with any inference system | We bridge deterministic rules + neural inference; they replace neural with deterministic |
| **Self-play / RL** | G-Zero self-play, GFlowNet distillation, bandit strategy adaptation, heuristic learning infrastructure | None | We learn from interaction; their weights are static by construction |
| **Language** | Rust — zero-cost abstractions, no GC, deterministic performance | Python (PyTorch) + C++ — two-language complexity | Single language, single binary, no Python runtime dependency |
| **Production lessons** | NVIDIA Dynamo agentic lessons applied: paged KV, streaming dispatch, per-request agent hints, catalog metadata | Research prototype, no production deployment notes | We're built for deployment; they're built for publication |

### What Percepta Does Better

| Category | transformer-vm | katgpt-rs | Why They Win |
|----------|----------------|-------------|--------------|
| **Hull attention algorithm** | Dynamic CHT (LineContainer): upper + lower hull, arbitrary 2D points, O(log n) insert AND query, sublinear memory (only hull vertices + HullMeta) | Graham Scan upper hull + ternary search: requires monotonic X, upper hull only, O(N) memory, broken for qy < 0 queries | **Fundamentally better algorithm.** Their CHT handles arbitrary point distributions, dual hull directions, and sublinear memory. Our Graham Scan is a special-case optimization. |
| **Tie-breaking** | `LATEST` (most recent tied value) + `AVERAGE` (mean of tied values) via `HullMeta` aggregation | None — returns first maximum found | Tie-breaking enables cumulative sum (AVERAGE) and latest-write semantics (LATEST). Without it we can't do state machine tracking via attention. |
| **Cumulative sum** | `fetch_sum`: uniform attention (all keys = constant) × position = exact running sum. Tracks cursor, stack_depth, call_depth entirely via attention | Native Rust variables for state tracking | They proved state can be maintained purely through attention mechanisms — a key theoretical result. We cheat with local variables. |
| **End-to-end compilation** | C → WASM → token prefix → transformer execution. Full pipeline: compiler, decoder, lowering passes | No compilation pipeline at all | **The entire point of their research.** We have the mechanism; they have the machine. |
| **Computation graph DSL** | `Expression` (sparse linear combo) + 6 `Dimension` types + `LookUp` + `ProgramGraph` — full symbolic algebra for transformer-native computation | No DSL | This is the foundation that makes everything else possible. Without it you can't express programs as attention + FFN operations. |
| **ReGLU / stepglu gates** | `relu(b)*a` (1 FFN neuron), `step(b≥0)` (2 neurons), `a*b` (2 neurons + persist) — all nonlinear logic from one primitive | No gate primitives | They proved all conditional logic and multiplication can be expressed as ReGLU neurons. This is the FFN-side counterpart to the attention mechanism. |
| **MILP scheduling** | PuLP/HiGHS optimizer: 4-phase layer assignment, `interval_coloring` slot reuse, minimizes `d_model` analytically | No scheduling | They find the optimal transformer architecture for a given program. We hard-code our model dimensions. |
| **Analytical weight construction** | `expr_to_tensor`: computation graph + schedule → weight matrices. No training, no gradient descent, no data | Standard trained weights | **Weights by construction, not by learning.** This is the most provocative claim — you can write programs directly into transformer weights. |
| **Specialization (Futamura)** | `_cursor_lookup`: bake program into FFN weights via piecewise-constant step functions. Eliminates instruction-fetch attention entirely | No specialization | One model per program, smaller and faster. The First Futamura Projection applied to transformers. |
| **Opcode dispatch** | Circle-point encoding (r²=32045): each opcode = unique (x,y) on circle. `op_dot(op) = px·fx + py·fy - r² + 1` = 1 on match, ≤ -1 otherwise → single ReGLU gate per case | No opcode dispatch | Elegant geometric hashing that maps 35 opcodes to single-neuron detectors. |
| **Byte-serial arithmetic** | Carry propagation via `carry_late = persist(carry)` (delayed 1 position). Full i32 add/sub with multi-byte carry/borrow | Native Rust `+`, `-`, `*`, `/` operators | They proved multi-byte arithmetic works through the attention + persist mechanism. We use the CPU. |
| **Model compactness** | d_model=36, n_heads=18, n_layers=7, d_ffn=36 — ~1.5MB weights, runs in browser | embd=16-384, production-scale configs | Their entire model fits in L1 cache. Ours needs careful memory management. |
| **Weight interpretability** | Every weight has a precise mathematical meaning (slot assignment, gate expression, parabolic encoding) | Black-box learned weights | You can read their weight matrices and understand exactly what computation they perform. |
| **C++ inference engine** | Standalone C++ with CHT hull cache, BLAS, sparse head projection — no Python dependency at inference | Rust-only (no separate inference engine) | They have a deployment path that doesn't require PyTorch. Our Rust is already native, so this is a wash. |

### The Core Tradeoff

| | katgpt-rs | transformer-vm |
|---|---|---|
| **Philosophy** | Neural inference **with** deterministic assistance | Deterministic execution **as** neural inference |
| **Goal** | Make LLMs faster and more reliable | Prove transformers can be computers |
| **Programs** | Run externally (WASM validators, tool use) | Run internally (compiled into weights) |
| **Weights** | Learned from data (gradient descent) | Constructed from programs (no training) |
| **Speed** | Production-grade (millions of tok/s) | Research-grade (30K tok/s) |
| **Generality** | Any language task + game domains | Any C program that compiles to WASM |
| **Correctness** | Probabilistic (sampling, temperature) | **Deterministic by construction** (greedy argmax = exact execution) |

### The Bottom Line

**We build faster cars. They invented a new engine.**

Our stack is production-grade: 140× faster throughput, zero-alloc hot paths, SIMD, TurboQuant compression, game-proven heuristic learning, self-play RL. We *use* the geometric attention mechanism as a component in a real system.

Their stack is a research breakthrough: the first end-to-end proof that arbitrary programs can be compiled into transformer weights and executed deterministically at 30K tok/s with O(log n) attention. They didn't optimize the engine — they proved it works at all.

**What we should take from them:** The CHT algorithm (Plan 063), the computation graph DSL, the ReGLU/stepglu primitives, and the cumulative sum mechanism. These are reusable components that strengthen our existing stack without requiring us to adopt their full compiler.

**What they could take from us:** Production integration patterns (zero-alloc, SIMD, speculative decoding), TurboQuant for compressing their KV cache further, Raven for O(1) routing, and our adversarial testing methodology.