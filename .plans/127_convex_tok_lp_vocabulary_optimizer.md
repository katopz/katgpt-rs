# Plan 127: ConvexTok — LP Vocabulary Optimizer for ToaST (Modelless Path)

**Branch:** `develop/feature/127_convex_tok_lp_vocab`
**Depends on:** Plan 122 (ToaST split-tree tokenizer — completed)
**Research:** 087 (ConvexTok — Tokenisation via Convex Relaxations)
**Paper:** [Tokenisation via Convex Relaxations](https://arxiv.org/pdf/2605.22821) (Tempus et al., 2026)
**Status:** ✅ Complete (T1–T9, 12 GOAT proofs)

---

## Tasks

- [x] T1: Tokenisation graph types (`tokenizer/convex_types.rs`) ✅
- [x] T2: Tokenisation graph construction from pretokenized corpus (`tokenizer/convex_graph.rs`) ✅
- [x] T3: LP formulation via `good_lp`/HiGHS (`tokenizer/convex_solver.rs`) ✅
- [x] T4: Rounding schemes — Det / Bias / Int (`tokenizer/convex_rounding.rs`) ✅
- [x] T5: Optimality certification — LP bound + gap computation (`tokenizer/convex_certify.rs`) ✅
- [x] T6: ConvexTok → ToaST vocabulary import (`tokenizer/convex_toast_bridge.rs`) ✅
- [x] T7: Feature gate `convex_tok` + module glue ✅
- [x] T8: GOAT proof — 12/12 tests (types, construction, LP solve, rounding, certification, ToaST interop) ✅
- [x] T9: Benchmark — compression vs BPE vs manual ToaST on synthetic corpus ✅

---

## Context

ConvexTok reformulates tokenizer vocabulary construction as a **Linear Program (LP)**, solving it via convex relaxation. Key properties:

1. **Globally near-optimal** — within 0.2% of LP-proven lower bound at vocab ≥ 32k
2. **Certifiable** — LP dual provides proven lower bound; any tokenizer's optimality gap is measurable
3. **Orthogonal to ToaST** — ConvexTok selects vocabulary (which tokens), ToaST segments text (how to tokenize). Together they form a complete globally-optimal pipeline.
4. **Det rounding consistently beats BPE** on BpB (~0.1-0.15% improvement)

This plan covers the **modelless inference path** in katgpt-rs:
- LP graph construction from pretokenized text
- LP solving via existing `good_lp`/HiGHS dependency
- Three rounding schemes (Det/Bias/Int)
- Optimality certification
- Bridge to ToaST (ConvexTok vocab → ToastTokenizer)

The **model-based training path** (full corpus pipeline, n-gram counting, LM training) is deferred to riir-ai Plan 109.

---

## T1: Tokenisation Graph Types

File: `katgpt-rs/src/tokenizer/convex_types.rs`

### Core Types

```rust
/// A vertex in the tokenisation graph.
/// Represents a position between two bytes in the concatenated dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VertexId(u32);

/// A free edge (byte-edge) in the tokenisation graph.
/// Always available, doesn't consume vocabulary budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FreeEdgeId(u32);

/// A priced edge (token-edge) in the tokenisation graph.
/// Requires its colour to be selected in the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PricedEdgeId(u32);

/// A colour representing a potential token.
/// All priced edges with the same byte-substring share a colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColourId(u32);

/// The tokenisation graph.
/// Encodes the full tokenisation problem as a DAG with coloured edges.
#[derive(Clone, Debug)]
pub struct TokenisationGraph {
    /// Number of vertices.
    pub n_vertices: usize,
    /// Source vertex (v₀₀).
    pub source: VertexId,
    /// Sink vertex (vₙₗ).
    pub sink: VertexId,
    /// Free edges: (from, to) pairs — always byte-edges.
    pub free_edges: Vec<(VertexId, VertexId)>,
    /// Priced edges: (from, to, colour) — token-edges coloured by substring.
    pub priced_edges: Vec<(VertexId, VertexId, ColourId)>,
    /// Colour → byte-substring mapping.
    pub colour_bytes: Vec<Vec<u8>>,
    /// Flow difference vector: -1 at source, +1 at sink, 0 elsewhere.
    /// Encoded as (vertex, value) pairs for sparsity.
    pub flow_diff: Vec<(VertexId, i32)>,
}

/// LP solution before rounding.
/// Contains fractional values for all variables.
#[derive(Clone, Debug)]
pub struct LpSolution {
    /// Fractional free-edge usage: f_e ∈ [0, 1].
    pub f: Vec<f64>,
    /// Fractional priced-edge usage: p_e ∈ [0, 1].
    pub p: Vec<f64>,
    /// Fractional colour selection: c_c ∈ [0, 1].
    pub c: Vec<f64>,
    /// LP objective value (proven lower bound on compression).
    pub lp_value: f64,
    /// Vocabulary budget K used.
    pub budget_k: usize,
}

/// A rounded (discrete) vocabulary selection.
#[derive(Clone, Debug)]
pub struct RoundedVocabulary {
    /// Selected colour IDs (c_c = 1).
    pub selected_colours: Vec<ColourId>,
    /// Corresponding byte-substrings.
    pub selected_bytes: Vec<Vec<u8>>,
    /// Number of selected colours (≤ K).
    pub n_selected: usize,
    /// Tokenised compression value after shortest-path recovery.
    pub compression_value: f64,
    /// Rounding scheme used.
    pub rounding_scheme: RoundingScheme,
}

/// Rounding scheme variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RoundingScheme {
    /// Deterministic: top-K colours by LP value c.
    Det,
    /// Biased: top-K by c / token_length (favours shorter tokens).
    Bias,
    /// Integral-only: keep only c ≥ 0.999.
    Int,
}

/// Optimality certification result.
#[derive(Clone, Debug)]
pub struct OptimalityCert {
    /// LP lower bound on compression.
    pub lp_lower_bound: f64,
    /// Actual compression achieved.
    pub actual_compression: f64,
    /// Optimality gap: (actual - lp) / lp × 100%.
    pub gap_percent: f64,
    /// Whether the tokeniser is within 1% of optimal.
    pub within_one_percent: bool,
    /// Fraction of LP solution that was already integral.
    pub integrality_fraction: f64,
}
```

### Key Design Decisions

- `VertexId(u32)`, `PricedEdgeId(u32)`, `ColourId(u32)` — newtypes for type safety, u32 sufficient (paper's largest graph: 26M vertices)
- `TokenisationGraph` owns all topology — no references, easy serialization
- `LpSolution` stores raw f64 vectors — no sparse optimization needed (solver returns dense)
- `RoundedVocabulary` is self-contained — can be serialized and loaded without the graph
- `RoundingScheme` is an enum (not trait) — only 3 schemes from paper, no extensibility needed

---

## T2: Tokenisation Graph Construction

File: `katgpt-rs/src/tokenizer/convex_graph.rs`

### Algorithm

Given a pretokenized corpus (list of byte-strings):

1. **Build vertices:** For each byte-string of length n, create n+1 vertices. Merge last vertex of string i with first vertex of string i+1.
2. **Build free edges:** Connect adjacent vertices within each string.
3. **Build priced edges:** Connect non-adjacent vertices (span ≥ 2 bytes) within each string.
4. **Assign colours:** Group priced edges by their byte-substring. Each unique substring = one colour.
5. **Build flow vector:** d[source] = -1, d[sink] = +1, d[others] = 0.

```rust
pub struct GraphBuilder;

impl GraphBuilder {
    /// Build a tokenisation graph from pretokenized byte-strings.
    ///
    /// # Arguments
    /// * `pretokens` — List of byte-strings (already pre-tokenized by regex)
    /// * `max_token_len` — Maximum token length to consider (default: 64)
    ///
    /// # Returns
    /// The tokenisation graph ready for LP formulation.
    pub fn build(pretokens: &[Vec<u8>], max_token_len: usize) -> TokenisationGraph { ... }
}
```

### Complexity

For N pretokens of average length L:
- Vertices: O(N·L)
- Free edges: O(N·L)
- Priced edges: O(N·L²) — but bounded by `max_token_len`
- Colours: O(N·L²) worst case, but typically much fewer (many duplicate substrings)

For micro benchmarks (N=1000, L=10): ~10K vertices, ~100K edges — trivially fast.
For production scale (N=600K, L=100): ~60M vertices, ~6B edges — need chunked construction.

**Strategy:** Build graph incrementally, process one pretoken at a time, merge colours across pretokens via `HashMap<Vec<u8>, ColourId>`.

---

## T3: LP Formulation via good_lp/HiGHS

File: `katgpt-rs/src/tokenizer/convex_solver.rs`

### LP Formulation

```
min  Σ p_e + Σ f_e                    (total path length = compression)
s.t. P·p + F·f = d                    (flow conservation)
     p_e ≤ c_{colour(e)}  ∀e ∈ P     (edge usable only if colour selected)
     Σ c_c ≤ K                        (vocabulary budget)
     0 ≤ f, p, c ≤ 1                  (LP relaxation)
```

Where:
- P, F are incidence matrices (sparse)
- d is the flow difference vector
- K is the vocabulary budget

### Implementation

```rust
pub struct ConvexSolver;

impl ConvexSolver {
    /// Solve the LP relaxation for the tokenisation graph.
    ///
    /// # Arguments
    /// * `graph` — The tokenisation graph
    /// * `budget_k` — Maximum number of colours (vocabulary budget)
    ///
    /// # Returns
    /// The LP solution with fractional variables and objective value.
    pub fn solve(graph: &TokenisationGraph, budget_k: usize) -> Result<LpSolution, String> {
        // 1. Create good_lp problem
        // 2. Add variables: f (free), p (priced), c (colours), all in [0, 1]
        // 3. Add flow constraints: P·p + F·f = d
        // 4. Add colour constraints: p_e ≤ c_{colour(e)} for each priced edge
        // 5. Add budget constraint: Σ c_c ≤ K
        // 6. Set objective: min Σ p_e + Σ f_e
        // 7. Solve via HiGHS (default solver for good_lp)
        // 8. Extract solution vectors
    }
}
```

### Dependency Note

`good_lp` with `highs` feature is already in `Cargo.toml` (used by Percepta Plan 064 TG-D). No new dependencies needed. Just add `good_lp` to the `convex_tok` feature's required dependencies.

For the `convex_tok` feature gate, we need `good_lp` as an optional dep (it already is):

```toml
# Already exists:
good_lp = { version = "1", optional = true, default-features = false, features = ["highs", "microlp"] }
```

### Performance Expectations

| Scale | Pretokens | Vocab Budget | Variables | Constraints | Solve Time |
|-------|-----------|-------------|-----------|-------------|------------|
| Micro | 100 | 256 | ~10K | ~10K | <1s |
| Small | 10K | 4K | ~1M | ~1M | ~10s |
| Medium | 100K | 32K | ~10M | ~10M | ~10min |
| Large | 600K | 128K | ~100M | ~100M | ~4hr |

Paper reports 4hr on GH200 for their full-scale experiment. Our micro/small benchmarks will be seconds.

---

## T4: Rounding Schemes

File: `katgpt-rs/src/tokenizer/convex_rounding.rs`

### Three Schemes from Paper

```rust
pub struct Rounder;

impl Rounder {
    /// Deterministic rounding: top-K colours by LP value.
    /// Best for BpB (bits-per-byte) metric.
    pub fn det(graph: &TokenisationGraph, solution: &LpSolution) -> RoundedVocabulary {
        let mut indexed: Vec<(usize, f64)> = solution.c.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<ColourId> = indexed.iter()
            .take(solution.budget_k)
            .map(|(i, _)| ColourId(*i as u32))
            .collect();
        Self::build_vocabulary(graph, selected, RoundingScheme::Det)
    }

    /// Biased rounding: top-K by c / token_length.
    /// Favours shorter tokens for OOD generalization.
    /// Best for intrinsic metrics (compression, vocab utilisation).
    pub fn bias(graph: &TokenisationGraph, solution: &LpSolution) -> RoundedVocabulary {
        let mut scored: Vec<(usize, f64)> = solution.c.iter().enumerate().map(|(i, &c)| {
            let len = graph.colour_bytes[i].len().max(1) as f64;
            (i, c / len)
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<ColourId> = scored.iter()
            .take(solution.budget_k)
            .map(|(i, _)| ColourId(*i as u32))
            .collect();
        Self::build_vocabulary(graph, selected, RoundingScheme::Bias)
    }

    /// Integral-only rounding: keep only c ≥ 0.999.
    /// Reveals which tokens the LP considers "forced".
    /// Typically selects fewer than K tokens.
    pub fn int(graph: &TokenisationGraph, solution: &LpSolution) -> RoundedVocabulary {
        const THRESHOLD: f64 = 0.999;
        let selected: Vec<ColourId> = solution.c.iter().enumerate()
            .filter(|(_, &c)| c >= THRESHOLD)
            .map(|(i, _)| ColourId(i as u32))
            .collect();
        Self::build_vocabulary(graph, selected, RoundingScheme::Int)
    }

    /// Build a RoundedVocabulary from selected colours.
    /// Computes compression via shortest path over the graph with selected colours.
    fn build_vocabulary(
        graph: &TokenisationGraph,
        selected: Vec<ColourId>,
        scheme: RoundingScheme,
    ) -> RoundedVocabulary { ... }
}
```

### Shortest Path Recovery

After rounding `c`, recover optimal `p` and `f` via shortest path:
1. Build edge weights: free edges cost 1, priced edges cost 1 only if their colour is selected
2. Run DAG shortest path from source to sink (topological order = position order)
3. Compression = path length

This is O(V + E) — linear in graph size, negligible compared to LP solve time.

---

## T5: Optimality Certification

File: `katgpt-rs/src/tokenizer/convex_certify.rs`

```rust
pub struct Certifier;

impl Certifier {
    /// Certify how close a tokenizer is to LP-proven optimal.
    ///
    /// # Arguments
    /// * `lp_solution` — The LP relaxation solution (provides lower bound)
    /// * `rounded` — The rounded vocabulary (provides actual compression)
    ///
    /// # Returns
    /// Certification with optimality gap and integrality fraction.
    pub fn certify(lp_solution: &LpSolution, rounded: &RoundedVocabulary) -> OptimalityCert {
        let lp_lower_bound = lp_solution.lp_value;
        let actual_compression = rounded.compression_value;
        let gap_percent = (actual_compression - lp_lower_bound) / lp_lower_bound * 100.0;

        let integrality_fraction = {
            let integral_count = lp_solution.c.iter().filter(|&&c| c >= 0.999 || c <= 0.001).count();
            integral_count as f64 / lp_solution.c.len().max(1) as f64
        };

        OptimalityCert {
            lp_lower_bound,
            actual_compression,
            gap_percent,
            within_one_percent: gap_percent <= 1.0,
            integrality_fraction,
        }
    }

    /// Certify an arbitrary tokenizer (e.g., BPE) against the LP bound.
    ///
    /// # Arguments
    /// * `lp_solution` — LP solution providing the lower bound
    /// * `tokenizer_compression` — Compression achieved by the tokenizer on the same data
    pub fn certify_external(
        lp_solution: &LpSolution,
        tokenizer_compression: f64,
    ) -> OptimalityCert {
        let gap_percent = (tokenizer_compression - lp_solution.lp_value)
            / lp_solution.lp_value * 100.0;

        OptimalityCert {
            lp_lower_bound: lp_solution.lp_value,
            actual_compression: tokenizer_compression,
            gap_percent,
            within_one_percent: gap_percent <= 1.0,
            integrality_fraction: 0.0, // unknown for external tokenizer
        }
    }
}
```

---

## T6: ConvexTok → ToaST Bridge

File: `katgpt-rs/src/tokenizer/convex_toast_bridge.rs`

Converts a `RoundedVocabulary` into a `ToastTokenizer` for inference.

```rust
pub struct ConvexToToastBridge;

impl ConvexToToastBridge {
    /// Convert a ConvexTok rounded vocabulary to a ToaST tokenizer.
    ///
    /// # Arguments
    /// * `rounded` — The ConvexTok rounded vocabulary
    /// * `ngram_counts` — Byte n-gram counts for split tree construction
    /// * `special_tokens` — Special token bytes (BOS, EOS, PAD, UNK)
    ///
    /// # Returns
    /// A ToastTokenizer ready for inference.
    pub fn to_toast_tokenizer(
        rounded: &RoundedVocabulary,
        ngram_counts: &HashMap<Vec<u8>, u64>,
        special_tokens: &SpecialTokens,
    ) -> ToastTokenizer {
        // 1. Build vocab_to_id: all selected colours + all single bytes + special tokens
        // 2. Build id_to_vocab: reverse mapping
        // 3. Build split trees for all pretokens using SplitTreeBuilder
        // 4. Return ToastTokenizer
    }
}

pub struct SpecialTokens {
    pub bos: Vec<u8>,
    pub eos: Vec<u8>,
    pub pad: Vec<u8>,
    pub unk: Vec<u8>,
}
```

### Key Insight

ConvexTok optimizes **which** tokens to include. ToaST optimizes **how** to segment with those tokens. The bridge is trivial: just pass the vocabulary bytes to ToaST's split tree builder.

---

## T7: Feature Gate + Module Glue

### `katgpt-rs/Cargo.toml` update

```toml
[features]
# Add:
convex_tok = ["dep:good_lp", "toast_tokenizer"]  # ConvexTok LP vocabulary optimizer (Plan 127, Research 087)
```

`convex_tok` depends on `toast_tokenizer` because:
- `convex_toast_bridge.rs` imports `ToastTokenizer` and `SplitTreeBuilder`
- Rounding output naturally flows into ToaST for inference
- No point having ConvexTok without ToaST to use the result

`good_lp` is already an optional dep. We just add it to the feature.

### `katgpt-rs/src/tokenizer/mod.rs` update

```rust
mod bpe;
mod types;

pub use bpe::{BpeTokenizerImpl, BpeTrainer};
pub use types::{BpeTokenizer, MergeRule};

#[cfg(feature = "toast_tokenizer")]
mod toast_types;
#[cfg(feature = "toast_tokenizer")]
mod toast_builder;
#[cfg(feature = "toast_tokenizer")]
mod toast_inference;

#[cfg(feature = "toast_tokenizer")]
pub use toast_types::{SplitNode, SplitTree, ToastTokenizer};
#[cfg(feature = "toast_tokenizer")]
pub use toast_builder::SplitTreeBuilder;
#[cfg(feature = "toast_tokenizer")]
pub use toast_inference::ToastTokenizerImpl;

#[cfg(feature = "convex_tok")]
mod convex_types;
#[cfg(feature = "convex_tok")]
mod convex_graph;
#[cfg(feature = "convex_tok")]
mod convex_solver;
#[cfg(feature = "convex_tok")]
mod convex_rounding;
#[cfg(feature = "convex_tok")]
mod convex_certify;
#[cfg(feature = "convex_tok")]
mod convex_toast_bridge;

#[cfg(feature = "convex_tok")]
pub use convex_types::{
    VertexId, FreeEdgeId, PricedEdgeId, ColourId,
    TokenisationGraph, LpSolution, RoundedVocabulary,
    RoundingScheme, OptimalityCert,
};
#[cfg(feature = "convex_tok")]
pub use convex_graph::GraphBuilder;
#[cfg(feature = "convex_tok")]
pub use convex_solver::ConvexSolver;
#[cfg(feature = "convex_tok")]
pub use convex_rounding::Rounder;
#[cfg(feature = "convex_tok")]
pub use convex_certify::Certifier;
#[cfg(feature = "convex_tok")]
pub use convex_toast_bridge::{ConvexToToastBridge, SpecialTokens};
```

### File Structure

```
katgpt-rs/src/tokenizer/
├── mod.rs                  # Module index + re-exports
├── bpe.rs                  # BPE tokenizer (existing)
├── types.rs                # BPE types (existing)
├── toast_types.rs          # ToaST types (Plan 122)
├── toast_builder.rs        # ToaST split tree builder (Plan 122)
├── toast_inference.rs      # ToaST recursive descent (Plan 122)
├── convex_types.rs         # NEW: ConvexTok graph/LP types
├── convex_graph.rs         # NEW: Graph construction
├── convex_solver.rs        # NEW: LP formulation + solving
├── convex_rounding.rs      # NEW: Det/Bias/Int rounding
├── convex_certify.rs       # NEW: Optimality certification
└── convex_toast_bridge.rs  # NEW: ConvexTok → ToaST bridge
```

---

## T8: GOAT Proof

Test file: `katgpt-rs/tests/bench_127_convex_tok_goat.rs`

### GOAT Criteria (12 tests)

```rust
#[cfg(test)]
mod tests {
    // ── T1: Types ──

    #[test]
    fn g01_graph_construction_from_pretokens() {
        // Build graph from ["hello", "world"]
        // Verify vertex count, edge count, colour count
    }

    #[test]
    fn g02_graph_vertex_merge() {
        // Verify last vertex of string 0 merged with first vertex of string 1
    }

    #[test]
    fn g03_colour_partition_disjoint() {
        // Verify colour groups are disjoint and cover all priced edges
    }

    // ── T3: LP Solver ──

    #[test]
    fn g04_lp_solves_within_tolerance() {
        // Solve LP on micro corpus (10 pretokens, K=32)
        // Verify objective is finite and all variables in [0, 1]
    }

    #[test]
    fn g05_lp_lower_bound_property() {
        // Verify LP value ≤ any feasible integer solution
        // (compare with greedy tokenization of same corpus)
    }

    // ── T4: Rounding ──

    #[test]
    fn g06_det_rounding_selects_exactly_k() {
        // Det rounding should select exactly K colours
    }

    #[test]
    fn g07_bias_rounding_penalizes_long_tokens() {
        // Bias scoring should rank short tokens higher than long tokens with same c
    }

    #[test]
    fn g08_int_rounding_selects_only_integral() {
        // Int rounding should only select colours with c ≥ 0.999
        // May select fewer than K
    }

    #[test]
    fn g09_rounded_vocabulary_has_valid_bytes() {
        // All selected colours should map to non-empty byte sequences
    }

    // ── T5: Certification ──

    #[test]
    fn g10_optimality_gap_non_negative() {
        // Gap should always be ≥ 0 (LP is a lower bound)
    }

    #[test]
    fn g11_det_within_five_percent_on_micro() {
        // On micro corpus, Det should be within 5% of LP optimal
        // (paper shows <1% at 32k+, we use smaller scale)
    }

    // ── T6: ToaST Bridge ──

    #[test]
    fn g12_toast_bridge_encode_decode_roundtrip() {
        // Build ConvexTok vocab → ToaST tokenizer → encode → decode = identity
    }
}
```

---

## T9: Benchmark — Compression vs BPE vs Manual ToaST

Test file: `katgpt-rs/tests/bench_127_convex_tok_compression.rs`

### Methodology

1. Build synthetic corpus (100 pretokens, English-like with common substrings)
2. Build tokenisation graph
3. Solve LP at K=256 and K=1024
4. Round with all three schemes
5. Build ToaST tokenizer from each rounded vocab
6. Compare compression (tokens/byte) against:
   - BPE trained on same corpus
   - ToaST with hand-picked vocabulary
   - Theoretical LP lower bound

### Expected Results (from paper trends)

| Metric | BPE | Det | Bias | LP Bound |
|--------|-----|-----|------|----------|
| Compression (tokens/byte) | baseline | ~0.1% better | ~0.05% better | ~0.5% better |
| Vocab utilisation | lower | higher | highest | N/A |
| Optimality gap | ~1-3% | ~0.1-0.5% | ~0.05-0.2% | 0% (lower bound) |

---

## Implementation Order

```
T1 (types)
  ↓
T2 (graph construction) → T3 (LP solver) → T4 (rounding) → T5 (certification)
  ↓                                                        ↓
T7 (feature gate)                                    T6 (ToaST bridge)
  ↓
T8 (GOAT proof — 12 tests)
  ↓
T9 (benchmark)
```

T1–T5 are sequential (each builds on the previous).
T6 depends on T4 (needs RoundedVocabulary) and Plan 122 (needs ToastTokenizer).
T7 is a config change (Cargo.toml + mod.rs).
T8 depends on all T1–T6.
T9 is standalone benchmark.

---

## Future (riir-ai Plan 109 — Model-Based Path)

The following belong in riir-ai (requires corpus pipeline + LM training):

1. **Full corpus n-gram counting** — stream ClimbMix/custom corpus, count byte n-grams
2. **Pre-tokenization regex pipeline** — GPT-4o style length-limited regex splitting
3. **LP solving at production scale** — 100M+ variables, may need chunked/incremental solving
4. **LM training with ConvexTok tokenizer** — train GPT-2 style model, measure BpB and CORE
5. **Rényi efficiency benchmark** — deferred from Plan 122 T6, use ConvexTok vocab
6. **Multilingual extension** — per-language cost reweighting in LP objective
7. **SuperBPE integration** — relax pre-tokenization constraints (paper's future work)

---

## References

- Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821
- Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705 (our ToaST, Plan 122)
- Kudo (2018). Subword regularization. ACL 2018. (Unigram LM — similar shortest-path inference)
- HiGHS LP solver: https://highs.dev/ (already in Cargo.toml via good_lp)
- Reference implementation: `.raw/tokenisation_lp/` (paper authors' Python code)