//! ConvexTok rounding schemes — convert fractional LP solutions to discrete vocabulary selections.
//!
//! After the LP relaxation produces fractional colour variables `c`, we must round to a
//! discrete vocabulary of at most K colours. Three rounding strategies are provided:
//!
//! - **Det** (deterministic): top-K by LP value `c`. Best for BpB metric.
//! - **Bias** (biased): top-K by `c / token_length`. Favours shorter tokens for OOD generalization.
//! - **Int** (integral-only): keep only `c >= 0.999`. Reveals "forced" tokens.
//!
//! After rounding, optimal path variables are recovered via DAG shortest path.
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

use std::cmp::Ordering;

use super::convex_types::{
    ColourId, LpSolution, RoundedVocabulary, RoundingScheme, TokenisationGraph,
};

/// Rounding engine for converting fractional LP solutions to discrete vocabulary selections.
///
/// All methods take an immutable snapshot of the graph + LP solution and return
/// a `RoundedVocabulary` with the selected colours and compression value computed
/// via shortest-path recovery over the DAG.
pub struct Rounder;

impl Rounder {
    /// Deterministic rounding: select top-K colours by LP value `c`.
    ///
    /// Picks the K colours with the highest fractional LP value. This is the
    /// simplest rounding and tends to optimise for the BpB (bits-per-byte) metric
    /// since it directly prioritises colours the LP found most valuable.
    pub fn det(graph: &TokenisationGraph, solution: &LpSolution) -> RoundedVocabulary {
        let mut indexed: Vec<(usize, f64)> = solution.c.iter().copied().enumerate().collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let selected: Vec<ColourId> = indexed
            .iter()
            .take(solution.budget_k)
            .map(|(i, _)| ColourId(*i as u32))
            .collect();

        Self::build_vocabulary(graph, selected, RoundingScheme::Det)
    }

    /// Biased rounding: select top-K by `c / token_length`.
    ///
    /// Divides each colour's LP value by its byte-length, favouring shorter tokens.
    /// This produces vocabularies that generalise better to out-of-distribution text
    /// because shorter tokens have higher coverage. Best for intrinsic metrics
    /// (compression ratio, vocabulary utilisation).
    pub fn bias(graph: &TokenisationGraph, solution: &LpSolution) -> RoundedVocabulary {
        let mut scored: Vec<(usize, f64)> = solution
            .c
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let len = graph.colour_bytes[i].len().max(1) as f64;
                (i, c / len)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let selected: Vec<ColourId> = scored
            .iter()
            .take(solution.budget_k)
            .map(|(i, _)| ColourId(*i as u32))
            .collect();

        Self::build_vocabulary(graph, selected, RoundingScheme::Bias)
    }

    /// Integral-only rounding: keep only colours with `c >= 0.999`.
    ///
    /// Reveals which tokens the LP considers "forced" — i.e., colours whose LP value
    /// is essentially 1.0 regardless of the budget constraint. Typically selects fewer
    /// than K tokens, providing insight into the LP's structural preferences.
    pub fn int(graph: &TokenisationGraph, solution: &LpSolution) -> RoundedVocabulary {
        const THRESHOLD: f64 = 0.999;

        let selected: Vec<ColourId> = solution
            .c
            .iter()
            .enumerate()
            .filter(|(_, c)| **c >= THRESHOLD)
            .map(|(i, _)| ColourId(i as u32))
            .collect();

        Self::build_vocabulary(graph, selected, RoundingScheme::Int)
    }

    /// Build a `RoundedVocabulary` from selected colour IDs.
    ///
    /// Computes the compression value via DAG shortest path over the tokenisation
    /// graph, using only selected colours for priced edges.
    fn build_vocabulary(
        graph: &TokenisationGraph,
        selected: Vec<ColourId>,
        scheme: RoundingScheme,
    ) -> RoundedVocabulary {
        // Build O(1) lookup for selected colour IDs
        let selected_set: Vec<bool> = {
            let mut set = vec![false; graph.n_colours()];
            for cid in &selected {
                set[cid.0 as usize] = true;
            }
            set
        };

        // Compute compression via DAG shortest path
        let compression = dag_shortest_path(graph, &selected_set);

        // Extract byte-substrings for selected colours
        let selected_bytes: Vec<Vec<u8>> = selected
            .iter()
            .map(|cid| graph.colour_bytes[cid.0 as usize].clone())
            .collect();

        let n_selected = selected.len();

        RoundedVocabulary {
            selected_colours: selected,
            selected_bytes,
            n_selected,
            compression_value: compression,
            rounding_scheme: scheme,
        }
    }
}

/// Compute shortest path from source to sink over the DAG.
///
/// Free edges always have cost 1. Priced edges have cost 1 only if their
/// colour is in `selected_set`; otherwise they are treated as impassable (infinity).
///
/// Vertices are numbered 0..n_vertices-1 in topological order (edges only go
/// forward), so a single forward pass suffices — no explicit topological sort needed.
///
/// Returns `f64::INFINITY` if no path exists (should not happen for well-formed graphs
/// since free edges always form a valid path).
fn dag_shortest_path(graph: &TokenisationGraph, selected_set: &[bool]) -> f64 {
    if graph.n_vertices == 0 {
        return f64::INFINITY;
    }

    let source = graph.source.0 as usize;
    let sink = graph.sink.0 as usize;

    // Distance array initialised to infinity
    let mut dist = vec![f64::INFINITY; graph.n_vertices];
    dist[source] = 0.0;

    // Build adjacency list: (to_vertex, cost)
    // We process in topological order (vertex index order), so we need
    // outgoing edges grouped by source vertex.
    // Count out-degree per vertex for pre-allocation
    let mut out_deg = vec![0usize; graph.n_vertices];
    for &(from, _) in &graph.free_edges {
        out_deg[from.0 as usize] += 1;
    }
    for &(from, to, colour) in &graph.priced_edges {
        let colour_idx = colour.0 as usize;
        if selected_set.get(colour_idx).copied() == Some(true) {
            out_deg[from.0 as usize] += 1;
        }
    }
    let mut adj: Vec<Vec<(usize, f64)>> = out_deg.into_iter().map(|d| Vec::with_capacity(d)).collect();

    // Free edges: always cost 1
    for &(from, to) in &graph.free_edges {
        adj[from.0 as usize].push((to.0 as usize, 1.0));
    }

    // Priced edges: cost 1 only if colour selected, otherwise impassable
    for &(from, to, colour) in &graph.priced_edges {
        let colour_idx = colour.0 as usize;
        let cost = match selected_set.get(colour_idx).copied() {
            Some(true) => 1.0,
            _ => f64::INFINITY,
        };
        // Only add edges that are actually usable to avoid polluting the adjacency list
        if cost.is_finite() {
            adj[from.0 as usize].push((to.0 as usize, cost));
        }
    }

    // DAG shortest path: forward pass in topological order
    for v in 0..graph.n_vertices {
        if dist[v].is_infinite() {
            continue;
        }
        for &(to, cost) in &adj[v] {
            let candidate = dist[v] + cost;
            if candidate < dist[to] {
                dist[to] = candidate;
            }
        }
    }

    dist[sink]
}

#[cfg(test)]
mod tests {
    use super::super::convex_types::VertexId;
    use super::*;
    use crate::tokenizer::convex_graph::GraphBuilder;
    use crate::tokenizer::convex_solver::ConvexSolver;

    /// Helper: build graph + solve LP for given pretokens and budget.
    fn solve(pretokens: &[Vec<u8>], budget_k: usize) -> (TokenisationGraph, LpSolution) {
        let graph = GraphBuilder::build(pretokens, 64);
        let solution =
            ConvexSolver::solve(&graph, budget_k).expect("LP should solve for test inputs");
        (graph, solution)
    }

    // ── Det rounding ──────────────────────────────────────────

    #[test]
    fn det_selects_exactly_k_colours_when_available() {
        // "abcdef" has many colours; budget 2 should select exactly 2
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e', b'f']];
        let (graph, sol) = solve(&pretokens, 2);

        let vocab = Rounder::det(&graph, &sol);

        assert_eq!(vocab.n_selected, 2, "det should select exactly K colours");
        assert_eq!(vocab.rounding_scheme, RoundingScheme::Det);
        assert_eq!(vocab.selected_colours.len(), 2);
        assert_eq!(vocab.selected_bytes.len(), 2);
    }

    #[test]
    fn det_compression_is_finite() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e']];
        let (graph, sol) = solve(&pretokens, 3);

        let vocab = Rounder::det(&graph, &sol);

        assert!(
            vocab.compression_value.is_finite(),
            "compression should be finite, got {}",
            vocab.compression_value
        );
        assert!(
            vocab.compression_value >= 1.0,
            "compression should be >= 1.0, got {}",
            vocab.compression_value
        );
    }

    #[test]
    fn det_budget_exceeds_colours_selects_all() {
        // "ab" → 1 colour, budget 100 → select 1
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b']];
        let (graph, sol) = solve(&pretokens, 100);

        let vocab = Rounder::det(&graph, &sol);

        assert_eq!(vocab.n_selected, 1, "should select all available colours");
        // With "ab" selected, compression = 1
        assert!(
            vocab.compression_value < 1.01,
            "compression should be ~1.0, got {}",
            vocab.compression_value
        );
    }

    #[test]
    fn det_budget_zero_selects_none() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c']];
        let (graph, sol) = solve(&pretokens, 0);

        let vocab = Rounder::det(&graph, &sol);

        assert_eq!(vocab.n_selected, 0, "budget 0 should select no colours");
        // Without priced edges, must use 3 free edges
        assert!(
            vocab.compression_value > 2.99,
            "compression should be ~3.0 (all free), got {}",
            vocab.compression_value
        );
    }

    // ── Bias rounding ─────────────────────────────────────────

    #[test]
    fn bias_selects_exactly_k_colours_when_available() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e', b'f']];
        let (graph, sol) = solve(&pretokens, 2);

        let vocab = Rounder::bias(&graph, &sol);

        assert_eq!(vocab.n_selected, 2, "bias should select exactly K colours");
        assert_eq!(vocab.rounding_scheme, RoundingScheme::Bias);
    }

    #[test]
    fn bias_compression_is_finite() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'h', b'e', b'l', b'l', b'o']];
        let (graph, sol) = solve(&pretokens, 3);

        let vocab = Rounder::bias(&graph, &sol);

        assert!(
            vocab.compression_value.is_finite(),
            "compression should be finite, got {}",
            vocab.compression_value
        );
    }

    #[test]
    fn bias_penalizes_long_tokens() {
        // Construct a scenario where det and bias select different colours.
        // With "abcdef" and budget 1:
        //   det picks the colour with highest c (likely "abcdef" — length 6)
        //   bias divides by length, so shorter tokens score higher
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e', b'f']];
        let (graph, sol) = solve(&pretokens, 1);

        let det_vocab = Rounder::det(&graph, &sol);
        let bias_vocab = Rounder::bias(&graph, &sol);

        // Both select 1 colour, but they may differ
        assert_eq!(det_vocab.n_selected, 1);
        assert_eq!(bias_vocab.n_selected, 1);

        // Bias should select a token no longer than det's selection
        let bias_len = bias_vocab.selected_bytes[0].len();
        let det_len = det_vocab.selected_bytes[0].len();
        assert!(
            bias_len <= det_len,
            "bias should favour shorter tokens: bias={bias_len} vs det={det_len}"
        );
    }

    #[test]
    fn bias_budget_zero_selects_none() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c']];
        let (graph, sol) = solve(&pretokens, 0);

        let vocab = Rounder::bias(&graph, &sol);

        assert_eq!(vocab.n_selected, 0);
    }

    // ── Int rounding ──────────────────────────────────────────

    #[test]
    fn int_selects_only_integral_colours() {
        // "ab" with budget 1 → the single colour "ab" should be nearly integral (~1.0)
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b']];
        let (graph, sol) = solve(&pretokens, 1);

        let vocab = Rounder::int(&graph, &sol);

        assert_eq!(vocab.rounding_scheme, RoundingScheme::Int);
        assert!(
            vocab.n_selected <= 1,
            "int should select at most the integral colours"
        );
        // With budget 1 and a single useful colour, it should be integral
        assert_eq!(vocab.n_selected, 1, "colour should be integral");
    }

    #[test]
    fn int_typically_selects_fewer_than_k() {
        // "abcdef" with budget 10 → many colours available but few are integral
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e', b'f']];
        let (graph, sol) = solve(&pretokens, 10);

        let vocab = Rounder::int(&graph, &sol);

        assert!(
            vocab.n_selected <= sol.budget_k,
            "int should select ≤ K colours, got {} vs K={}",
            vocab.n_selected,
            sol.budget_k
        );
    }

    #[test]
    fn int_budget_zero_selects_none() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c']];
        let (graph, sol) = solve(&pretokens, 0);

        let vocab = Rounder::int(&graph, &sol);

        assert_eq!(vocab.n_selected, 0);
    }

    #[test]
    fn int_compression_is_finite() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e']];
        let (graph, sol) = solve(&pretokens, 3);

        let vocab = Rounder::int(&graph, &sol);

        assert!(
            vocab.compression_value.is_finite(),
            "compression should be finite, got {}",
            vocab.compression_value
        );
    }

    // ── Cross-scheme consistency ──────────────────────────────

    #[test]
    fn all_schemes_produce_valid_bytes() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e']];
        let (graph, sol) = solve(&pretokens, 3);

        for (name, vocab) in [
            ("det", Rounder::det(&graph, &sol)),
            ("bias", Rounder::bias(&graph, &sol)),
            ("int", Rounder::int(&graph, &sol)),
        ] {
            assert_eq!(
                vocab.selected_colours.len(),
                vocab.selected_bytes.len(),
                "{name}: colours and bytes should have same length"
            );
            assert_eq!(
                vocab.n_selected,
                vocab.selected_colours.len(),
                "{name}: n_selected should match actual count"
            );
        }
    }

    #[test]
    fn all_schemes_compression_at_least_lp_value() {
        // LP value is a lower bound; rounded compression should be >= LP value
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd']];
        let (graph, sol) = solve(&pretokens, 2);

        let lp_val = sol.lp_value;

        for (name, vocab) in [
            ("det", Rounder::det(&graph, &sol)),
            ("bias", Rounder::bias(&graph, &sol)),
            ("int", Rounder::int(&graph, &sol)),
        ] {
            assert!(
                vocab.compression_value >= lp_val - 1e-6,
                "{name}: compression ({}) should be >= LP value ({})",
                vocab.compression_value,
                lp_val
            );
        }
    }

    #[test]
    fn all_schemes_compression_within_graph_size() {
        // Compression cannot exceed the number of free edges (worst case: all single bytes)
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e']];
        let (graph, sol) = solve(&pretokens, 2);

        let max_compression = graph.free_edges.len() as f64;

        for (name, vocab) in [
            ("det", Rounder::det(&graph, &sol)),
            ("bias", Rounder::bias(&graph, &sol)),
            ("int", Rounder::int(&graph, &sol)),
        ] {
            assert!(
                vocab.compression_value <= max_compression + 1e-6,
                "{name}: compression ({}) should be <= max ({})",
                vocab.compression_value,
                max_compression
            );
        }
    }

    // ── Shortest path edge cases ──────────────────────────────

    #[test]
    fn empty_graph_yields_infinite_compression() {
        // Build vocabulary with no graph — selected_set is empty, no vertices
        let graph = TokenisationGraph {
            n_vertices: 0,
            source: VertexId(0),
            sink: VertexId(0),
            free_edges: Vec::new(),
            priced_edges: Vec::new(),
            colour_bytes: Vec::new(),
            flow_diff: Vec::new(),
        };
        let selected_set: Vec<bool> = vec![];
        let result = dag_shortest_path(&graph, &selected_set);
        assert!(result.is_infinite(), "empty graph should yield infinity");
    }

    #[test]
    fn single_byte_graph_compression_is_one() {
        // Single byte "a" → source=0, sink=1, one free edge, no priced edges
        let graph = TokenisationGraph {
            n_vertices: 2,
            source: VertexId(0),
            sink: VertexId(1),
            free_edges: vec![(VertexId(0), VertexId(1))],
            priced_edges: Vec::new(),
            colour_bytes: Vec::new(),
            flow_diff: vec![(VertexId(0), -1), (VertexId(1), 1)],
        };
        let selected_set: Vec<bool> = vec![];
        let result = dag_shortest_path(&graph, &selected_set);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "single free edge should give compression=1, got {result}"
        );
    }

    #[test]
    fn unselected_colour_edges_are_skipped() {
        // "ab" with "ab" colour NOT selected → must use 2 free edges → compression = 2
        let graph = TokenisationGraph {
            n_vertices: 3,
            source: VertexId(0),
            sink: VertexId(2),
            free_edges: vec![(VertexId(0), VertexId(1)), (VertexId(1), VertexId(2))],
            priced_edges: vec![(VertexId(0), VertexId(2), ColourId(0))],
            colour_bytes: vec![vec![b'a', b'b']],
            flow_diff: vec![(VertexId(0), -1), (VertexId(2), 1)],
        };
        // Colour NOT selected
        let selected_set = vec![false];
        let result = dag_shortest_path(&graph, &selected_set);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "unselected colour should force free-edge path, got {result}"
        );
    }

    #[test]
    fn selected_colour_shortens_path() {
        // Same graph but colour IS selected → compression = 1
        let graph = TokenisationGraph {
            n_vertices: 3,
            source: VertexId(0),
            sink: VertexId(2),
            free_edges: vec![(VertexId(0), VertexId(1)), (VertexId(1), VertexId(2))],
            priced_edges: vec![(VertexId(0), VertexId(2), ColourId(0))],
            colour_bytes: vec![vec![b'a', b'b']],
            flow_diff: vec![(VertexId(0), -1), (VertexId(2), 1)],
        };
        let selected_set = vec![true];
        let result = dag_shortest_path(&graph, &selected_set);
        assert!(
            (result - 1.0).abs() < 1e-10,
            "selected colour should shorten path to 1, got {result}"
        );
    }

    // ── Multi-pretoken merging ────────────────────────────────

    #[test]
    fn det_multiple_pretokens() {
        // "ab" + "cd" — two pretokens merged at boundary
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b'], vec![b'c', b'd']];
        let (graph, sol) = solve(&pretokens, 2);

        let vocab = Rounder::det(&graph, &sol);

        assert_eq!(vocab.n_selected, 2);
        assert!(
            vocab.compression_value.is_finite(),
            "compression should be finite, got {}",
            vocab.compression_value
        );
    }

    #[test]
    fn colour_deduplication_across_pretokens_rounding() {
        // "ab" + "ab" → same colour deduplicated, budget 1
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b'], vec![b'a', b'b']];
        let (graph, sol) = solve(&pretokens, 1);

        assert_eq!(
            graph.colour_bytes.len(),
            1,
            "should have 1 deduplicated colour"
        );

        let vocab = Rounder::det(&graph, &sol);

        assert_eq!(vocab.n_selected, 1);
        assert_eq!(vocab.selected_bytes[0], vec![b'a', b'b']);
    }
}
