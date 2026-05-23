# Percepta Distillation Strategy: Full RIIR of transformer-vm

**Date:** 2025-06
**Status:** Verdict — Take Everything. Full RIIR.
**Context:** Percepta's transformer-vm is Apache-2.0 (confirmed from LICENSE + pyproject.toml). Per our strategy in `03_Commercial_Open_Source_Strategy_Verdict.md`, we distill open-source components to Rust and open them under MIT.

---

## TL;DR

**Take all their goodies.** The code is Apache-2.0. We're legally and ethically clear. Distill to Rust, open source under MIT, strengthen our engine. Take everything, not just the attention mechanism.

**Why take everything (not just Phase A):**

1. **Prove Rust is better than Python+C++** — One language, one binary, zero GC, deterministic perf. The full transformer-vm in Rust is a definitive proof point.
2. **Show Percepta what's possible** — If they see a clean Rust port that's faster and more maintainable, they might change their stack, or at minimum we inspire each other.
3. **We're already secure** — Our production inference pipeline (4.2M tok/s DFlash, TurboQuant, DDTree) is untouched. This is research code in `src/percepta/`. No risk.
4. **micro-gpt is for research** — Their WASM interpreter running inside a transformer is fascinating even if it's slower than native. It proves a concept, not a benchmark. We're the right home for it.
5. **It's fun** — Watching a transformer execute WASM bytecodes deterministically is objectively cool. Do it because we can.

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

## Phase C: Full Compiler Stack (P4–P6) — DO IT

**Status:** New plan after Phase B

The full compiler stack, ported from Python to Rust. This is the "transformer as computer" product — compile arbitrary C programs into transformer weights.

| Layer | Component | What It Does | Source File |
|-------|-----------|-------------|-------------|
| **P4** | Expression/Dimension DSL | Symbolic algebra for transformer-native computation | `graph/core.py` (449 lines) |
| **P5** | MILP Scheduling | Optimal layer/slot assignment | `scheduler/milp.py` (814 lines) |
| **P6a** | WASM Decoder | Parse WASM binary into opcode stream | `compilation/decoder.py` (664 lines) |
| **P6b** | WASM Lowering | Lower unsupported ops (MUL, DIV, etc.) to supported ones | `compilation/lower.py` (1808 lines) |
| **P6c** | WASM Interpreter | 35-opcode machine as computation graph | `wasm/interpreter.py` (637 lines) |
| **P6d** | Weight Construction | Graph + schedule → weight matrices, no training | `model/weights.py` (776 lines) |
| **P6e** | Futamura Specialization | Bake program into FFN weights | `specialize.py` (148 lines) |
| **P6f** | Transformer (ReGLU) | VanillaTransformer with ReGLU FFN | `model/transformer.py` (~40 lines) |
| **P6g** | C→WASM Compiler | Compile C source to WASM token prefix | `compilation/compile_wasm.py` (703 lines) |
| **P6h** | Graph Evaluator | Run computation graph with exact arithmetic | `evaluator.py` (404 lines) |

**This is NOT a pivot.** Our core product remains RIIR. This is a research-grade proof that Rust can do everything Python+C++ can do, but better. The transformer-as-computer code lives in `src/percepta/` alongside our existing attention mechanism. It doesn't replace or compete with our inference pipeline.

---

## Capability Comparison: What Percepta CAN'T Do

Percepta compiles C programs into transformer weights for deterministic execution. Impressive, but limited to static programs. Our `riir-ai` pipeline does things they fundamentally cannot:

| Capability | Percepta | Us (riir-ai) |
|-----------|----------|--------------|
| **Sudoku (deterministic)** | ✅ C→WASM→weights, ~30K tok/s | ✅ Hull attention, ~13M steps/s |
| **Bomberman (game AI)** | ❌ Could compile C AI, but static — no learning | ✅ LoRA + validator.wasm + bandit adaptation |
| **Self-play improvement** | ❌ Weights constructed, never updated | ✅ G-Zero, HL heuristic learning, episode DB flywheel |
| **Dynamic rule hotswap** | ❌ Recompile needed to change behavior | ✅ Hotswap validators at runtime (bomber_dynamic_rules_demo) |
| **Trained model inference** | ❌ No training paradigm — analytical weights only | ✅ lora.bin loaded for real game policy (bandit_with_real_model_demo) |
| **Adaptive strategies** | ❌ Deterministic by construction | ✅ Multi-armed bandit, Thompson sampling, decaying epsilon |
| **Cross-domain transfer** | ❌ One model per program (Futamura) | ✅ Same engine for code translation, game AI, constraint satisfaction |

**The bomber demos in `riir-ai/crates/riir-examples/` are the proof:**

- `bomber_demo.rs` — loads `bomber_validator.wasm` + `game_lora.bin`, runs mini-arena, A/B validates WASM vs native
- `bomber_tech_ab_demo.rs` — tech tree A/B testing,ValidatorPlayer vs GreedyPlayer vs HLPlayer
- `bomber_dynamic_rules_demo.rs` — hotswap validation rules mid-game at runtime
- `bandit_with_real_model_demo.rs` — loads real lora.bin, runs Leviathan verification, compares Rust vs Python validators

Percepta could theoretically compile a Bomberman AI into transformer weights, but:
1. It would play the SAME way every time (deterministic)
2. It can't learn from outcomes or improve
3. The model would need to be much larger for game state
4. Execution at ~30K tok/s is too slow for real-time game decisions
5. No mechanism for loading trained weights or adapting strategies

**Our moat isn't speed — it's learning.** They proved transformers can execute programs. We proved transformers can *learn to play games*.

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

### Current: UNFAIR — Different Algorithms, Different Machines

Our `sudoku_04_percepta_vs` example shows 92,000× faster. That's misleading:

| | Ours | Percepta |
|---|---|---|
| **Algorithm** | Rust backtracking + hull attention | Transformer executes WASM bytecodes |
| **Machine** | Our Mac | Their machine (unknown specs) |
| **Comparison** | Steps/s (backtracking) | tok/s (WASM interpretation) |

Speed difference is mostly algorithm (backtracking vs WASM-in-transformer), NOT language (Rust vs C++).

### Fair Benchmark Plan (After Plan 064)

After we RIIR their transformer-vm, we can run a **fair** comparison:

1. **Same algorithm**: Our Rust transformer-vm executing the same WASM bytecodes
2. **Same inputs**: Same Sudoku puzzle, same token prefix
3. **Same machine**: Our Mac, both binaries compiled with `-O2`/`--release`
4. **Metric**: tok/s (identical computation per token)

Then we compare:
- Python transformer (their code, our machine)
- C++ transformer (their engine, our machine)
- Rust transformer (our port, our machine)

This isolates language/runtime performance from algorithm differences.

### Raw Execution Speed (Oracles)

| System | What | Speed | Context |
|--------|------|-------|---------|
| Our wasmtime validators | Token validation | ~0.5μs/call | Production pipeline |
| Our Rust backtracking | Sudoku solve | ~13M steps/s | Different algorithm |
| Their C++ transformer | Program execution | ~30K tok/s | Research artifact |
| Their Python transformer | Program execution | Much slower than C++ | Development only |

For raw program execution, wasmtime is ~1000× faster than their transformer. But their contribution is proving transformers CAN execute programs deterministically — the speed will improve with better implementations.

---

## Execution Order

```
Phase A (now):     CHT + CumSum + Parabolic → src/percepta/cht.rs, hull.rs
                   Plan 063 in progress
                   
Phase B (next):    ReGLU/stepglu/persist → src/percepta/gates.rs
                   New plan after 063 completes
                   
Phase C (DO IT):   DSL → MILP → WASM interpreter → weights → Futamura
                   Full RIIR of transformer-vm (~9K lines Python+C++ → Rust)
                   Then: fair benchmark, same algorithm, same machine
                   Plan 064: 11 task groups, all committed
```

### Head-to-Head Benchmarks

| Benchmark | Status | Fair? | Where |
|-----------|--------|-------|-------|
| Sudoku (hull attention vs reported numbers) | ✅ Done | ❌ Different algo/machine | `examples/sudoku_04_percepta_vs.rs` |
| Sudoku regression tests | ✅ Done | ❌ Same | `tests/integration.rs` |
| Sudoku (Rust transformer-vm vs C++ transformer-vm) | ❌ After Plan 064 | ✅ Same algo/machine | `src/percepta/runner.rs` |
| Bomberman (learning vs static) | ❌ N/A — they can't do it | N/A — capability gap | `riir-ai/crates/riir-examples/` |
| lora.bin + validator.wasm vs percepta_wasm | ❌ After Plan 064 | Different paradigm | `riir-ai/crates/riir-examples/` |

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
- `.research/031_percepta_deep_dive.md` — Full gap analysis (9 gaps, P0–P6)
- `.plans/063_percepta_cht_hull_kv_cache.md` — CHT upgrade plan
- `.research/003_Commercial_Open_Source_Strategy_Verdict.md` — Engine/Fuel split strategy
