# Plan 259: Spec Compile — Modelless Spec-to-Constraint Compilation

> **Research:** 229 (ProgramAsWeights Spec Compile Verdict — GAIN for symbolic path)
> **Related Plans:** 131 (SpecHop), 110 (Subterranean Procedure Compilation), 228 (Vocab Channel Pruner)
> **Feature Gates:** `spec_pruner`, `spec_compile`
> **Status:** In Progress

## Summary

Research 229 identified that ProgramAsWeights compiles NL specs into neural LoRA adapters (~22MB), but our katgpt-rs architecture can do fundamentally better for many spec types by compiling specs into **symbolic constraint rules** at inference time — zero training, zero neural weights, O(1) bitmap execution.

This plan implements a modelless spec compilation pipeline: NL spec → `SpecCompiler` → `Vec<SpecRule>` → `CompiledSpec` pruner. For classification specs (sentiment, urgency), this gives 100% valid outputs with <1μs per token using RoaringBitmap token-set lookups. For structured specs (JSON, email), DFA-based format constraints and DDTree marginals provide format-safe generation. For fuzzy specs, GPU/ANE ternary adapter fallback kicks in.

**Target: Modelless spec compilation with O(1) bitmap execution, zero neural weights, GOAT-proven per phase.**

---

## Feature Gates

```toml
# Cargo.toml
spec_pruner = []                   # SpecAsPruner: NL spec → ConstraintPruner rules (F1)
spec_compile = ["spec_pruner"]     # Full spec compilation suite (F1+F2+F3+F5+F6)
```

Add to `full` feature:
```toml
full = [..., "spec_compile"]
```

---

## Tasks

### Phase 1: SpecAsPruner Core (F1) — GOAT proof

- [x] **T1**: Create `src/pruners/spec_compile/types.rs` — `SpecRule` type with token pattern + `CompactBitmap` allowed/blocked sets, `CompiledSpec` wrapper, `SpecType` enum
- [x] **T2**: Create `src/pruners/spec_compile/compiler.rs` — `SpecCompiler` struct that compiles spec string into `Vec<SpecRule>`, handles classification/extraction/format specs, pattern-based label extraction
- [x] **T3**: Create `src/pruners/spec_compile/pruner.rs` — `impl ConstraintPruner for CompiledSpec` with O(1) bitmap lookup per token, batch validation, zero-alloc hot path
- [x] **T4**: Create `src/pruners/spec_compile/screening.rs` — `impl ScreeningPruner for CompiledSpec` with relevance scoring for ambiguous specs
- [x] **T5**: GOAT proof — classification spec (sentiment, urgency) → pruner → verify 100% valid outputs, 17 unit tests pass

### Phase 2: SpecAsMarginals (F2) — structured output

- [x] **T6**: Create `src/pruners/spec_compile/marginals.rs` — `SpecMarginals` type with token probability bias from spec, integration with DDTree marginal interface
- [x] **T7**: Implement `spec_to_marginals()` — compile structured specs into DDTree-compatible marginals, JSON schema → structural token bias, regex-like constraints → token-set marginals
- [x] **T8**: GOAT proof — JSON repair spec → marginals → DDTree → verify >95% valid JSON output, <10μs per token

### Phase 3: SpecDFA (F3) — format specs

- [x] **T9**: Create `src/pruners/spec_compile/dfa.rs` — `SpecDFA` type extending PartialParser for arbitrary formats, DFA state transitions from format descriptions, token-to-character-set mapping
- [x] **T10**: Implement `compile_format_spec()` — NL format description → DFA transitions, email, phone, date, URL format compilation, state-aware token filtering
- [x] **T11**: GOAT proof — email extraction spec → DFA → verify >90% extraction accuracy on messy input

### Phase 4: SpecProof (F6) — verification

- [x] **T12**: Create `src/pruners/spec_compile/proof.rs` — `SpecProof` type with BLAKE3 commitment of compiled spec rules, `SpecCommitment` struct for tamper detection
- [x] **T13**: Implement `verify_spec_compilation()` — prove compiled pruner enforces spec constraints, deterministic verification: spec string → compile → BLAKE3 hash → compare against commitment
- [x] **T14**: GOAT proof — spec → compile → verify → tamper test, verify catches >99% of spec violations

### Phase 5: SpecChain (F5) — composition

- [x] **T15**: Create `src/pruners/spec_compile/chain.rs` — `SpecChain` with AND/OR composition of compiled specs, bitmap intersection (AND) and union (OR), chained pruner execution
- [x] **T16**: Integration with MUX-Latent wire patch for spec composition over network (Plan 243), serialized spec chain transport, remote spec composition protocol
- [x] **T17**: GOAT proof — chained specs (extract email → classify domain) → verify <5μs per token total

### Phase 6: CPU/SIMD/GPU/ANE adaptive routing

- [x] **T18**: Create `src/pruners/spec_compile/router.rs` — threshold-based routing: simple spec → CPU (bitmap), complex spec → SIMD (batch), fuzzy spec → GPU/ANE (ternary adapter fallback)
- [x] **T19**: Implement `SpecRouter` — auto-detect spec complexity, route to appropriate compute tier, complexity heuristic: rule count × token-set size × DFA states, fallback chain: CPU → SIMD → GPU → ANE

### Phase 7: Module index & integration

- [x] **T20**: Create `src/pruners/spec_compile/mod.rs` — module index, re-exports, `#[cfg(feature = "spec_pruner")]` and `#[cfg(feature = "spec_compile")]` gates
- [x] **T21**: Add `spec_pruner` and `spec_compile` feature gates to `Cargo.toml`, add `pub mod spec_compile` to `src/pruners/mod.rs`
- [x] **T22**: Benchmarks & documentation — `.benchmarks/259_spec_compile_goat.md`, update README tech table

---

## Module Structure

```
src/pruners/spec_compile/
├── mod.rs                    # Module index, re-exports, feature gates
├── types.rs                  # SpecRule, CompiledSpec, SpecCompileError
├── compiler.rs               # SpecCompiler: spec string → Vec<SpecRule>
├── pruner.rs                 # impl ConstraintPruner for CompiledSpec
├── screening.rs              # impl ScreeningPruner for CompiledSpec
├── marginals.rs              # SpecMarginals: spec → DDTree marginals
├── dfa.rs                    # SpecDFA: format spec → DFA transitions
├── proof.rs                  # SpecProof: BLAKE3 commitment + verification
├── chain.rs                  # SpecChain: AND/OR composition
└── router.rs                 # SpecRouter: adaptive compute tier routing
```

---

## Architecture

### Core Compilation Flow

```
NL Spec String
    ↓ SpecCompiler::compile()
    ↓ keyword extraction + token-set construction
Vec<SpecRule> { pattern, allowed: RoaringBitmap, blocked: RoaringBitmap }
    ↓ CompiledSpec::new(rules)
    ↓
┌───────────────────────────────────────────────┐
│ CompiledSpec (implements ConstraintPruner)     │
│                                                │
│ Hot path: token_id → bitmap.contains() → O(1) │
│   allowed.contains(token) && !blocked(token)   │
└───────────────────────────────────────────────┘
    ↓ SpecRouter routes based on complexity:
    ├── CPU bitmap: simple classification specs
    ├── SIMD batch: medium-complexity specs
    ├── GPU ternary: fuzzy/semantic specs (fallback)
    └── ANE: latency-critical fuzzy specs (fallback)
```

### Spec Type → Compilation Strategy

| Spec Type | Example | Compilation Strategy | Phase |
|-----------|---------|---------------------|-------|
| Classification | "only positive sentiment" | Token-set bitmap (allowed/blocked) | F1 |
| Structured output | "output valid JSON" | DDTree marginals + structural bias | F2 |
| Format extraction | "extract emails" | DFA state transitions | F3 |
| Composed | "extract emails, classify domain" | AND/OR chain | F5 |
| Fuzzy/semantic | "make it sound professional" | GPU/ANE ternary fallback | F6 routing |

### Key Types

```rust
/// A single compiled spec rule.
/// Hot path: O(1) RoaringBitmap lookup per token.
#[derive(Debug, Clone)]
pub struct SpecRule {
    /// Human-readable pattern description.
    pub pattern: String,
    /// Token IDs allowed by this rule (whitelist).
    pub allowed: RoaringBitmap,
    /// Token IDs explicitly blocked by this rule (blacklist).
    pub blocked: RoaringBitmap,
    /// Rule priority for conflict resolution.
    pub priority: u8,
}

/// Compiled spec ready for use as ConstraintPruner.
#[derive(Debug, Clone)]
pub struct CompiledSpec {
    rules: Vec<SpecRule>,
    proof: SpecProof,
    complexity: SpecComplexity,
}

/// Complexity classification for SpecRouter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpecComplexity {
    /// Simple bitmap lookup — CPU path.
    Simple,
    /// Medium complexity — SIMD batch path.
    Medium,
    /// Fuzzy/semantic — GPU ternary adapter fallback.
    Fuzzy,
}

/// BLAKE3 commitment of compiled spec rules.
#[derive(Debug, Clone)]
pub struct SpecProof {
    /// BLAKE3 hash of the compiled rules.
    pub commitment: [u8; 32],
    /// Original spec string for verification.
    pub spec_source: String,
}
```

### Key Trait Implementations

```rust
impl ConstraintPruner for CompiledSpec {
    /// O(1) per token: bitmap intersection check.
    /// Zero-alloc hot path — no Vec, no HashMap lookup.
    fn prune(&self, token_ids: &[u32]) -> Vec<u32> {
        token_ids.iter()
            .filter(|&&id| self.is_allowed(id))
            .copied()
            .collect()
    }
}

impl ScreeningPruner for CompiledSpec {
    /// Relevance scoring for ambiguous specs.
    /// Uses rule match count as relevance signal.
    fn relevance(&self, token_ids: &[u32]) -> f64 {
        // sigmoid over matching rule count
    }
}
```

---

## GOAT Proof Targets

| Phase | Spec | Target | Metric |
|-------|------|--------|--------|
| F1 | Classification (sentiment, urgency) | 100% valid outputs | <1μs/token |
| F2 | JSON repair spec | >95% valid JSON | <10μs/token |
| F3 | Email extraction | >90% accuracy | <5μs/token |
| F6 | Verification | >99% violation detection | BLAKE3 commitment |
| F5 | Chained spec (extract → classify) | Correct end-to-end | <5μs/token |

### GOAT Proof Protocol

1. **Compile** spec string into `CompiledSpec`
2. **Run** spec pruner on test corpus (1000+ tokens)
3. **Verify** all outputs satisfy spec constraints
4. **Benchmark** per-token latency against threshold
5. **Commit** BLAKE3 proof of compilation correctness

---

## Testing Strategy

1. **Unit tests per task** — each T has `#[cfg(test)]` module with basic correctness tests
2. **GOAT proofs** — T5, T8, T11, T14, T17 with benchmark thresholds
3. **Before/after comparison** — with spec pruner vs without, measure valid-output rate delta
4. **Property tests** — proptest: random spec strings compile without panic, compiled pruner never allows blocked tokens
5. **Integration tests** — spec → compile → pruner → DDTree → output → verify format
6. **Compatibility tests** — no panics with default feature combinations

### Expected Gains Table

| Spec Type | Without Spec Pruner | With Spec Pruner | Gain |
|-----------|--------------------|--------------------|------|
| Classification | ~85% valid (model-only) | 100% valid | +15% |
| JSON output | ~80% valid (model-only) | >95% valid | +15% |
| Email extraction | ~60% precision (model-only) | >90% precision | +30% |
| Chained spec | ~70% end-to-end (model-only) | >90% end-to-end | +20% |
| Latency overhead | N/A | <1μs/token (CPU) | Negligible |

---

## Constraints

- **Modelless first** — no LLM training required, pure symbolic compilation
- **Files <2048 lines** — each module file stays under limit
- **Zero-alloc hot path** — `ConstraintPruner::prune()` does not allocate in the inner loop
- **CPU/SIMD/GPU/ANE auto-route** — threshold-based routing with fallback chain
- **BLAKE3 for commitments** — `SpecProof` uses blake3 for spec verification
- **papaya for lock-free concurrent spec cache** — concurrent `CompiledSpec` cache without `Arc<RwLock<HashMap>>`
- **`#[repr(u8)]` on field-less enums** — 1-byte `SpecComplexity`, `SpecCompileError`
- **`Uuid::now_v7()`** for any spec chain identifiers

---

## Dependencies

| Dependency | Purpose | Feature Gate |
|-----------|---------|-------------|
| `roaring` | RoaringBitmap for O(1) token-set lookups | `spec_pruner` |
| `blake3` | Spec proof commitments | `spec_pruner` |
| `papaya` | Lock-free concurrent spec cache | `spec_compile` |
| DDTree (existing) | Marginals integration for F2 | `spec_compile` |
| PartialParser (existing) | DFA extension for F3 | `spec_compile` |
| MUX-Latent (existing) | Wire patch for F5 network composition | `spec_compile` |

---

## Implementation Order

```
Phase 1: SpecAsPruner Core (F1)            [~4h]  ← GOAT proof first
Phase 2: SpecAsMarginals (F2)              [~3h]
Phase 3: SpecDFA (F3)                      [~3h]
Phase 4: SpecProof (F6)                    [~2h]
Phase 5: SpecChain (F5)                    [~3h]
Phase 6: CPU/SIMD/GPU/ANE routing          [~2h]
Phase 7: Module index & integration        [~2h]
─────────────────────────────────────────────────
Total estimate:                            ~19h
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Bitmap too coarse for fuzzy specs | GPU/ANE ternary adapter fallback for fuzzy cases |
| DFA state explosion for complex formats | State limit with graceful degradation to marginals |
| SpecCompiler requires NLP for NL specs | Start with keyword-based extraction, defer to LLM-assisted compilation later |
| Chained spec latency compounding | AND uses bitmap intersection (O(1)), OR uses bitmap union (O(1)) |
| Feature gate confusion (`spec_pruner` vs `spec_compile`) | `spec_compile` depends on `spec_pruner`, clear docs |

---

## Success Criteria

1. `CompiledSpec` compiles classification specs into `Vec<SpecRule>` with RoaringBitmap token sets
2. `ConstraintPruner` implementation achieves <1μs per token for simple specs
3. GOAT proofs pass for all 5 phases (F1, F2, F3, F5, F6)
4. `SpecProof` BLAKE3 commitment detects >99% of spec violations
5. `SpecChain` AND/OR composition maintains <5μs per token
6. `SpecRouter` correctly routes based on spec complexity
7. Zero overhead when `spec_pruner` feature disabled

---

## TL;DR

Compile NL specs into symbolic RoaringBitmap constraint rules — zero neural weights, zero training, O(1) per-token execution. Phase 1 (SpecAsPruner) proves GOAT with classification specs → 100% valid outputs. Subsequent phases add marginals (JSON), DFA (email), proof (BLAKE3), chain (AND/OR), and adaptive CPU/SIMD/GPU/ANE routing.
