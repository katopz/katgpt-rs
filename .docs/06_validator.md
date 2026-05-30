# katgpt-rs: Constraint Validator

## What is Validator?
Deterministic validation — a neuro-symbolic inference system where `rustc`/`syn` acts as the deterministic referee inside the speculative decoding loop. The LLM drafts token sequences; a rules engine validates them against the Rust AST before target model verification.

## The Grand Vision (from Research)

```
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE (Production)                        │
│                                                                  │
│  User Prompt ──► BPE Encode ──► Draft Model (katgpt-rs)       │
│                                      │                           │
│                               DDTree Branches                   │
│                                      │                           │
│                          ┌───────────▼───────────┐              │
│                          │   SynPruner (Validator)│              │
│                          │   ┌─────────────────┐  │              │
│                          │   │ bracket balance  │  │  Tier 0 DFA │
│                          │   │ keyword accept   │  │  ~100ns/tok │
│                          │   └─────────────────┘  │              │
│                          │   ┌─────────────────┐  │              │
│                          │   │ syn full parse   │  │  Tier 1 AST │
│                          │   │ (completed paths)│  │  ~1-10μs    │
│                          │   └─────────────────┘  │              │
│                          └───────────┬───────────┘              │
│                                      │                           │
│                          Validated DDTree Branches               │
│                                      │                           │
│                          ┌───────────▼───────────┐              │
│                          │  Target Model Verify   │              │
│                          │  (semantic quality)    │              │
│                          └───────────┬───────────┘              │
│                                      │                           │
│                          ┌───────────▼───────────┐              │
│                          │  cargo check (final)   │              │
│                          │  OK → anyrag + Turso   │              │
│                          │  ERR → feedback loop   │              │
│                          └───────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (Offline Batch)                      │
│                                                                  │
│  Rust Docs ──┐                                                   │
│  GitHub ────┼──► ingester ──► BPE tokenize ──► syn filter       │
│  Crates.io ─┘       │                │                │         │
│                     │                │                ▼         │
│              concept sharding    vocab build    only valid AST   │
│              via anyrag          merge + train   0 errors/warns  │
│                     │                │                │         │
│                     ▼                ▼                ▼         │
│              Turso episodic    tokenizer.json    clean.jsonl     │
│              (hidden states)   (BPE merges)     (training data) │
│                     │                │                │         │
│                     └────────────────┼────────────────┘         │
│                                      ▼                           │
│                              LoRA fine-tune                     │
│                              (lora.bin)                         │
│                                      │                           │
│                              Draft model upgrade                │
│                              (higher acceptance rate)           │
└─────────────────────────────────────────────────────────────────┘
```

## Status
| Component | Status | Notes |
|-----------|--------|-------|
| Path encoding (u128) | ✅ Working | 16-bit slots, max token 65535, max depth 8 |
| BPE Tokenizer | ✅ Working | encode/decode/train, `Config::bpe()` |
| PartialParser | ✅ Working | Bracket balance DFA |
| SynPruner | ✅ Working | Two-tier validation (DFA + syn) |
| CompilerFeedback | ✅ Working | Extracts suggestions from "expected" patterns in syn errors |
| Training data pipeline | ✅ Working | Via riir-burner + riir-gpu |

## Path Encoding

```rust
// crates/katgpt-core/src/traits.rs (re-exported via speculative/types.rs)
parent_path: u128  // 16 bits per depth → max token 65535, max depth 8
```
- Extract: `(path >> (depth * 16)) & 0xFFFF`
- Push: `(path << 16) | token_idx as u128`
- Zero-alloc variant: `extract_parent_tokens_into(path, n, buf)`

## BPE Tokenizer (`tokenizer/`)

### Core Types
```rust
// tokenizer/types.rs
#[derive(Clone, Serialize, Deserialize)]
pub struct BpeTokenizer {
    #[serde(with = "map_serde")]
    pub vocab_to_id: HashMap<String, usize>,
    pub id_to_vocab: Vec<String>,
    pub merges: Vec<MergeRule>,
    #[serde(skip)]
    pub merge_ranks: HashMap<(String, String), usize>,
    pub bos_id: usize,
    pub eos_id: usize,
    pub pad_id: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MergeRule {
    pub left: String,
    pub right: String,
    pub merged: String,
}
```

### Training Algorithm
1. Read .rs files → byte sequences
2. Initialize vocab = 256 byte-level tokens
3. Count adjacent byte-pair frequencies
4. Merge most frequent pair → new token
5. Repeat until target vocab_size (4096) reached

### Corpus Sources
- `rust-lang/rust` (~30K files, ~15M lines)
- Top 100 crates.io crates (~20K files, ~10M lines)
- Rust docs/examples (~5K files, ~2M lines)

### Expected Common Merges
`fn `, `pub `, `let `, `mut `, `impl `, ` ->`, `::`, `use `, `Result<`, `Option<`, `Vec<`, `String`, `async `, `#[`, `match `, `struct `

### Config
```rust
// crates/katgpt-core/src/types.rs
impl Config {
    pub fn bpe() -> Self {
        Self { vocab_size: 4096, block_size: 256, n_embd: 32, n_head: 4,
               head_dim: 8, n_layer: 1, n_kv_head: 4, mlp_hidden: 128,
               bos_token: 1, temperature: 0.8, draft_lookahead: 8,
               tree_budget: 32, ... }
    }
    pub fn bpe_draft() -> Self {
        Self { vocab_size: 4096, block_size: 256, n_embd: 16, n_head: 2,
               head_dim: 8, n_layer: 1, n_kv_head: 2, mlp_hidden: 64,
               bos_token: 1, temperature: 0.8, draft_lookahead: 8,
               tree_budget: 32, ... }
    }
}
```
- Total weights: ~1.1 MB (target), ~130 KB (draft)

## PartialParser (`validator/partial_parser.rs`)

NOT a full Rust parser. A pragmatic DFA:

```rust
pub struct PartialParser {
    paren_depth: i32,
    brace_depth: i32,
    bracket_depth: i32,
    angle_depth: i32,
    in_string: bool,
    in_char: bool,
    in_block_comment: bool,
    in_line_comment: bool,
}
```

- `is_valid(code: &str) -> bool` — incremental state update; resets state then scans
- `is_balanced() -> bool` — check if all bracket depths are zero
- `reset()` — reset parser state to initial
- Rejects: negative bracket depth (too many closing brackets)
- Accepts: everything else (permissive by design, unclosed brackets ok for partial code)
- Latency: ~50ns per token call

### Honest Assessment
This is "Phase 0 Validator":
- Catches ~10-20% of invalid branches (unbalanced delimiters)
- Near-zero false negatives (rarely prunes valid code)
- The remaining ~80-90% pass to Tier 1 (syn) and Tier 2 (cargo check)

## SynPruner (`validator/syn_pruner.rs`)

### Two-Tier Validation

| Tier | When | Method | Latency |
|------|------|--------|---------|
| Tier 0: DFA | Per-token in DDTree | PartialParser bracket balance | ~50ns |
| Tier 1: syn | Per-path after DDTree | `syn::parse_str` as `syn::Stmt` | ~1-10μs |
| Tier 2: cargo check | Post-generation, offline | `cargo check` subprocess | ~100ms-1s |

```rust
pub struct SynPruner {
    tokenizer: Arc<BpeTokenizer>,
    parser: PartialParser,
}
```

### Key Methods
- `new(tokenizer: Arc<BpeTokenizer>) -> Self`
- `validate(&mut self, code: &str) -> PruneResult` — full two-tier validation
- `is_valid_quick(&mut self, code: &str) -> bool` — Tier 0 only (DDTree hot path)

### ConstraintPruner Implementation (DDTree hot path)
```rust
impl ConstraintPruner for SynPruner {
    fn is_valid(&self, _depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool {
        let mut all_tokens = parent_tokens.to_vec();
        all_tokens.push(token_idx);
        let code = BpeTokenizerImpl::decode(&self.tokenizer, &all_tokens);
        let mut parser = PartialParser::new();
        parser.is_valid(&code)
    }
}
```
Only Tier 0 (bracket balance) runs in the DDTree hot path. Tier 1 (syn) is too expensive per-node.

### Tier 1 Validation (on complete paths)
```rust
pub fn validate(&mut self, code: &str) -> PruneResult {
    // Tier 0: bracket balance
    if !self.parser.is_valid(code) {
        return PruneResult { is_valid: false, error_kind: ErrorKind::UnbalancedBrackets };
    }
    // Tier 1: syn parse
    match syn::parse_str::<syn::Stmt>(code) {
        Ok(_) => PruneResult { is_valid: true, error_kind: ErrorKind::None },
        Err(e) => PruneResult { is_valid: false, error_kind: ErrorKind::SynError(e.to_string()) },
    }
}
```

## PruneResult & ErrorKind (`validator/types.rs`)

```rust
#[derive(Debug, Clone)]
pub struct PruneResult {
    pub is_valid: bool,
    pub error_kind: ErrorKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    None,
    UnbalancedBrackets,
    SynError(String),
}
```

## CompilerFeedback (`validator/types.rs`)

```rust
#[derive(Debug, Clone)]
pub struct CompilerFeedback {
    pub error_message: String,
    pub failing_code: String,
    pub suggestion: Option<String>,
}
```
- `extract_suggestion(error_msg: &str) -> Option<String>` — looks for "expected" patterns in syn error messages
- `to_context() -> String` — formats as `Error: ... \n Suggestion: ...` for next LLM prompt

Integrated into the speculative loop: after cargo check fails, `CompilerFeedback` extracts the error and injects it as context for the next generation attempt.

## Feature Flag
```toml
validator = ["syn", "proc-macro2"]
```
- `syn` is an optional dependency, NOT required for core functionality
- BPE tokenizer works without syn
- Only SynPruner requires syn
- All validator modules gated: `#[cfg(feature = "validator")]`

## ConstraintPruner Trait

Defined in `crates/katgpt-core/src/traits.rs`, re-exported via `speculative/types.rs`:

```rust
pub trait ConstraintPruner: Send + Sync {
    fn is_valid(&self, depth: usize, token_idx: usize, parent_tokens: &[usize]) -> bool;
    fn batch_is_valid(&self, depth: usize, candidates: &[usize], parent_tokens: &[usize], results: &mut [bool]);
}
```

## Dependency Chain
All phases complete:
- ✅ Phase 0: Path encoding fix (u128) — prerequisite for BPE vocab > 31
- ✅ Phase 1: BPE Tokenizer — encode/decode/train
- ✅ Phase 2: SynPruner — PartialParser + ConstraintPruner impl
- ✅ Phase 3: Integration — examples, benchmarks

## Key References
- `.research/000_Neuro-Symbolic LLM Architecture.md` — Original Validator concept
- `.research/001_Advanced Neuro-Symbolic Rust Translation.md` — Grand Unification architecture
- `syn` crate — Rust AST parser
- `riir-burner` + `riir-gpu` — Training data pipeline (BPE training corpus)
