# Percepta Distillation Strategy: What to Take, What to Keep, What to Build

**Date:** 2025-06
**Status:** Verdict — Execute Phase A+B Now, Evaluate Phase C Later
**Context:** Percepta's transformer-vm is Apache-2.0 (confirmed from LICENSE + pyproject.toml). Per our strategy in `03_Commercial_Open_Source_Strategy_Verdict.md`, we distill open-source components to Rust and open them under MIT.

---

## TL;DR

**Take all their goodies.** The code is Apache-2.0. We're legally and ethically clear. Distill to Rust, open source under MIT, strengthen our engine. But do it in phases — each layer depends on the previous.

---

## The WASM Confusion (Resolved)

Percepta's "WASM" and our WASM are **two completely different things**:

| Aspect | Percepta's "WASM" | Our WASM (riir-wasm) |
|--------|-------------------|---------------------|
| **What** | C programs compiled to WASM bytecode, tokenized, fed as input to transformer | Rust validators compiled to .wasm, run in wasmtime sandbox |
| **Runtime** | The transformer IS the runtime — attention mechanisms execute the bytecodes | wasmtime v28 — standard WASM runtime |
| **Speed** | ~30K tok/s ≈ ~30KB/s program execution | ~0.5μs/call, near-native speed |
| **Purpose** | Prove transformers can deterministically execute arbitrary programs | Validate draft tokens for constraint pruning |
| **Conflict** | None — they're orthogonal systems that happen to share an acronym |

**No either/or choice needed. Support both. They complement each other.**

---

## Phase A: CHT + Cumulative Sum + Parabolic Encoding (P0–P2)

**Status:** Plan 063 (in progress)

Take the attention-side improvements. These directly strengthen our core product:

| Component | What | Why We Need It |
|-----------|------|---------------|
| **Dynamic CHT** | Replace Graham Scan with LineContainer (`BTreeSet<Line>`) | Fixes broken qy<0, arbitrary 2D points, O(log n) insert+query, sublinear memory |
| **Dual hull** | Upper hull (qy>0) + lower hull (qy<0) + edge metadata | Correct attention in all query directions |
| **TieBreak enum** | `LATEST` (most recent) + `AVERAGE` (mean) | Enables state tracking and cumulative sum |
| **HullMeta** | Aggregated values on hull vertices (count, sum, last) | Sublinear memory — only store hull, not all points |
| **Cumulative sum** | `fetch_sum`: uniform attention × position = exact running sum | Track cursor, stack depth, call depth via attention |
| **Parabolic key encoding** | k → (2k, −k²), q → (q, 1), score = −(k−q)² + q² | Exact key-value match, already partially implemented |

**Target:** `src/percepta/cht.rs`, `src/percepta/hull.rs`

**License:** MIT (our derivative work from Apache-2.0 source)

---

## Phase B: ReGLU/stepglu Gate Primitives (P3)

**Status:** New plan needed

Take the FFN-side improvements. These enable programmatic weight construction:

| Component | What | Why We Need It |
|-----------|------|---------------|
| **reglu(a, b)** | `relu(b) * a` — 1 FFN neuron for gated output | Basic nonlinear primitive |
| **stepglu(a, b)** | `a * step(b ≥ 0)` — 2 neurons for conditional | Conditional logic as FFN |
| **multiply(a, b)** | `a * b` — 2 neurons + persist for full multiplication | Arithmetic as FFN |
| **persist(expr)** | Materialize expression into residual slot | State propagation across layers |

**Target:** `src/percepta/gates.rs`

**Why this matters for RIIR:** The constraint pruner architecture (`ConstraintPruner` trait → DDTree) could potentially be compiled into transformer FFN weights using these primitives. Instead of running validators in wasmtime at inference time, the validation logic becomes part of the model weights. This is speculative but theoretically possible.

**License:** MIT (our derivative work from Apache-2.0 source)

---

## Phase C: Full Compiler Stack (P4–P6) — PIVOT DECISION

**Status:** Evaluate after Phase B completes

This is a **different product** from RIIR. Only pursue if we want to offer "compile your program into transformer weights" as a product.

| Layer | Component | What It Does |
|-------|-----------|-------------|
| **P4** | Expression/Dimension DSL | Symbolic algebra for transformer-native computation |
| **P5** | MILP Scheduling | Optimal layer/slot assignment (PuLP/HiGHS) |
| **P6** | WASM Interpreter | 35 opcodes as computation graph |
| **P6** | Weight Construction | Graph + schedule → weight matrices, no training |
| **P6** | Futamura Specialization | Bake program into FFN weights |

**Why this is a pivot:** Our current product is "Python → Rust translation that compiles." This would be "C → transformer weights that execute deterministically." Different customers, different value prop, different everything.

**Honest assessment:** The full compiler stack is academically brilliant but commercially unproven. No one is asking to run C programs inside transformer weights. The market for RIIR (Python → Rust) is real and growing. Don't pivot unless Phase A+B reveals unexpected demand.

---

## What Stays Secret

Per our strategy (`03_Commercial_Open_Source_Strategy_Verdict.md`), the open engine needs closed fuel:

| Secret | What | Why Defensible |
|--------|------|---------------|
| `lora.bin` | Trained Python→Rust adapter weights | Needs millions of verified pairs to be useful |
| `validator.wasm` | Domain-specific constraint pruners | Accumulated edge case knowledge from Episode DB |
| Episode DB | Compiler errors, corrections, patterns | Data flywheel — grows with every job |
| Semantic validator | `cargo check` → DDTree feedback loop | Orchestration speed, not a magical algorithm |
| Orchestration | Repo chunking, GPU pool, parallel translation | Engineering complexity |

**Wasmtime is NOT a secret.** It's Apache-2.0 infrastructure by bytecodealliance. Our secrets are WHAT we run in wasmtime (validators) and HOW we generate those validators (orchestration + episode DB), not wasmtime itself.

---

## Speed Comparison (Honest)

| System | What | Speed | Verdict |
|--------|------|-------|---------|
| Our wasmtime validators | Token validation | ~0.5μs/call | Production-grade |
| Their C++ transformer engine | Program execution | ~30K tok/s (30KB/s) | Research-grade |
| Their Python transformer | Program execution | Much slower than C++ | Development only |

For raw execution, wasmtime is ~1000× faster. But this comparison misses the point — Percepta's contribution is proving transformers CAN execute programs deterministically, not doing it fast.

---

## Execution Order

```
Phase A (now):     CHT + CumSum + Parabolic → src/percepta/cht.rs, hull.rs
                   Plan 063 in progress
                   
Phase B (next):    ReGLU/stepglu/persist → src/percepta/gates.rs
                   New plan after 063 completes
                   
Phase C (maybe):   DSL → MILP → WASM interpreter → weights → Futamura
                   DECISION GATE: only if Phase B reveals product-market fit
                   for "programs as weights"
```

---

## Legal Basis

- **Source:** transformer-vm by Percepta-Core, Apache-2.0
- **Our license:** MIT for all distilled Rust code
- **Obligation:** Include Apache-2.0 NOTICE attribution in our derivative files
- **Permitted:** Derivative works, commercial use, modification, distribution
- **Not permitted:** Use Percepta trademark without permission (standard Apache clause)

---

## References

- `.raw/transformer-vm/LICENSE` — Apache-2.0, Copyright 2026 Percepta
- `.raw/transformer-vm/pyproject.toml` — `license = {text = "Apache-2.0"}`
- `.research/31_percepta_deep_dive.md` — Full gap analysis (9 gaps, P0–P6)
- `.plans/063_percepta_cht_hull_kv_cache.md` — CHT upgrade plan
- `.research/03_Commercial_Open_Source_Strategy_Verdict.md` — Engine/Fuel split strategy