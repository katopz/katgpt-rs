//! ConvexTok tokenisation graph construction from pretokenized byte-strings.
//!
//! Builds a directed acyclic graph (DAG) where vertices represent byte positions
//! and edges represent potential token spans. Free edges cover single bytes;
//! priced edges cover multi-byte substrings coloured by their byte content.
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

use std::collections::HashMap;

use super::convex_types::{ColourId, TokenisationGraph, VertexId};

/// Default maximum token length to consider for priced edges.
pub const DEFAULT_MAX_TOKEN_LEN: usize = 64;

/// Builder for tokenisation graphs from pretokenized byte-strings.
///
/// Constructs the DAG incrementally, processing one pretoken at a time and
/// merging colours across pretokens via a `HashMap<Vec<u8>, ColourId>`.
pub struct GraphBuilder;

impl GraphBuilder {
    /// Build a tokenisation graph from pretokenized byte-strings.
    ///
    /// # Algorithm
    /// 1. Build vertices: for each byte-string of length n, create n+1 vertices.
    ///    Merge last vertex of string i with first vertex of string i+1.
    /// 2. Build free edges: connect adjacent vertices within each string.
    /// 3. Build priced edges: connect non-adjacent vertices (span ≥ 2 bytes).
    /// 4. Assign colours: group priced edges by byte-substring.
    /// 5. Build flow vector: d[source] = -1, d[sink] = +1.
    ///
    /// # Arguments
    /// * `pretokens` — List of byte-strings (already pre-tokenized by regex)
    /// * `max_token_len` — Maximum token length to consider for priced edges
    ///
    /// # Returns
    /// The tokenisation graph ready for LP formulation.
    pub fn build(pretokens: &[Vec<u8>], max_token_len: usize) -> TokenisationGraph {
        if pretokens.is_empty() {
            return TokenisationGraph {
                n_vertices: 0,
                source: VertexId(0),
                sink: VertexId(0),
                free_edges: Vec::new(),
                priced_edges: Vec::new(),
                colour_bytes: Vec::new(),
                flow_diff: Vec::new(),
            };
        }

        let mut next_vertex: u32 = 0;
        let total_bytes: usize = pretokens.iter().map(|p| p.len()).sum();
        let mut free_edges: Vec<(VertexId, VertexId)> = Vec::with_capacity(total_bytes);
        let mut priced_edges: Vec<(VertexId, VertexId, ColourId)> = Vec::with_capacity(
            total_bytes.saturating_sub(pretokens.len()) * max_token_len.max(2).saturating_sub(1),
        );

        // Colour deduplication: byte-substring → ColourId
        // Use borrowed slices during construction to avoid allocating Vec<u8>
        // for every substring. We build colour_bytes only for new colours.
        let mut colour_map: HashMap<&[u8], ColourId> = HashMap::new();
        let mut colour_bytes: Vec<Vec<u8>> = Vec::new();

        // Track source and sink
        let source = VertexId(0);

        for (pretoken_idx, pretoken) in pretokens.iter().enumerate() {
            let len = pretoken.len();
            if len == 0 {
                continue;
            }

            // Create vertices for this pretoken: len + 1 positions
            let base_vertex = next_vertex;
            next_vertex += (len + 1) as u32;

            // Free edges: adjacent vertices (single-byte spans)
            for i in 0..len {
                let from = VertexId(base_vertex + i as u32);
                let to = VertexId(base_vertex + i as u32 + 1);
                free_edges.push((from, to));
            }

            // Priced edges: spans of 2..=max_token_len bytes
            for start in 0..len {
                let max_end = (start + max_token_len).min(len);
                for end in (start + 2)..=max_end {
                    let from = VertexId(base_vertex + start as u32);
                    let to = VertexId(base_vertex + end as u32);
                    let substring = &pretoken[start..end];

                    let colour_id = match colour_map.get(substring) {
                        Some(&cid) => cid,
                        None => {
                            let cid = ColourId(colour_bytes.len() as u32);
                            colour_map.insert(substring, cid);
                            colour_bytes.push(substring.to_vec());
                            cid
                        }
                    };

                    priced_edges.push((from, to, colour_id));
                }
            }

            // Merge: if not the last pretoken, the last vertex of this pretoken
            // becomes the first vertex of the next pretoken by adjusting next_vertex.
            // We don't create a new vertex for the start of the next pretoken —
            // instead, the next iteration's base_vertex = current next_vertex - 1
            // to share the boundary vertex.
            if pretoken_idx < pretokens.len() - 1 {
                // Merge: the next pretoken's first vertex is this pretoken's last vertex.
                // We already incremented next_vertex for all positions including the last one.
                // To share: decrement next_vertex by 1 so the next pretoken reuses
                // the boundary vertex as its starting position.
                next_vertex -= 1;
            }
        }

        let sink = VertexId(next_vertex - 1);
        let n_vertices = next_vertex as usize;

        // Flow difference vector: -1 at source, +1 at sink
        let mut flow_diff = Vec::with_capacity(2);
        if source != sink {
            flow_diff.push((source, -1_i32));
            flow_diff.push((sink, 1_i32));
        }

        TokenisationGraph {
            n_vertices,
            source,
            sink,
            free_edges,
            priced_edges,
            colour_bytes,
            flow_diff,
        }
    }

    /// Build with default maximum token length (64 bytes).
    pub fn build_default(pretokens: &[Vec<u8>]) -> TokenisationGraph {
        Self::build(pretokens, DEFAULT_MAX_TOKEN_LEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_corpus_returns_empty_graph() {
        let graph = GraphBuilder::build(&[], DEFAULT_MAX_TOKEN_LEN);
        assert_eq!(graph.n_vertices, 0);
        assert_eq!(graph.free_edges.len(), 0);
        assert_eq!(graph.priced_edges.len(), 0);
        assert_eq!(graph.colour_bytes.len(), 0);
    }

    #[test]
    fn single_byte_produces_only_free_edge() {
        // Single byte "a" → 2 vertices, 1 free edge, 0 priced edges
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a']];
        let graph = GraphBuilder::build(&pretokens, DEFAULT_MAX_TOKEN_LEN);

        assert_eq!(graph.n_vertices, 2);
        assert_eq!(graph.source, VertexId(0));
        assert_eq!(graph.sink, VertexId(1));
        assert_eq!(graph.free_edges.len(), 1);
        assert_eq!(graph.priced_edges.len(), 0);
        assert_eq!(graph.flow_diff.len(), 2);
    }

    #[test]
    fn two_bytes_produces_free_and_priced() {
        // "ab" → 3 vertices, 2 free edges, 1 priced edge (colour "ab")
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b']];
        let graph = GraphBuilder::build(&pretokens, DEFAULT_MAX_TOKEN_LEN);

        assert_eq!(graph.n_vertices, 3);
        assert_eq!(graph.free_edges.len(), 2);
        assert_eq!(graph.priced_edges.len(), 1);
        assert_eq!(graph.colour_bytes.len(), 1);
        assert_eq!(graph.colour_bytes[0], vec![b'a', b'b']);
    }

    #[test]
    fn two_pretokens_merge_boundary_vertex() {
        // "ab" + "cd" → merged at boundary
        // Without merge: 3 + 3 = 6 vertices
        // With merge: 3 + 3 - 1 = 5 vertices
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b'], vec![b'c', b'd']];
        let graph = GraphBuilder::build(&pretokens, DEFAULT_MAX_TOKEN_LEN);

        assert_eq!(graph.n_vertices, 5);
        assert_eq!(graph.source, VertexId(0));
        assert_eq!(graph.sink, VertexId(4));
        // 2 free edges per pretoken
        assert_eq!(graph.free_edges.len(), 4);
        // 1 priced edge per pretoken (each 2-byte substring)
        assert_eq!(graph.priced_edges.len(), 2);
        assert_eq!(graph.colour_bytes.len(), 2);
    }

    #[test]
    fn colour_deduplication_across_pretokens() {
        // "ab" + "ab" → same colour "ab" shared
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b'], vec![b'a', b'b']];
        let graph = GraphBuilder::build(&pretokens, DEFAULT_MAX_TOKEN_LEN);

        // Both priced edges should reference the same colour
        assert_eq!(graph.colour_bytes.len(), 1);
        assert_eq!(graph.colour_bytes[0], vec![b'a', b'b']);
        assert_eq!(graph.priced_edges.len(), 2);
        assert_eq!(graph.priced_edges[0].2, graph.priced_edges[1].2);
    }

    #[test]
    fn max_token_len_limits_priced_edges() {
        // "abcde" with max_token_len=2 → only 2-byte priced edges
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd', b'e']];
        let graph = GraphBuilder::build(&pretokens, 2);

        // Priced edges: ab, bc, cd, de (4 edges, 4 colours)
        assert_eq!(graph.priced_edges.len(), 4);
        // Free edges: a, b, c, d, e (5 edges)
        assert_eq!(graph.free_edges.len(), 5);
    }

    #[test]
    fn flow_diff_source_minus_one_sink_plus_one() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c']];
        let graph = GraphBuilder::build(&pretokens, DEFAULT_MAX_TOKEN_LEN);

        assert_eq!(graph.flow_diff.len(), 2);
        let source_diff = graph
            .flow_diff
            .iter()
            .find(|(v, _)| *v == graph.source)
            .map(|(_, d)| *d);
        let sink_diff = graph
            .flow_diff
            .iter()
            .find(|(v, _)| *v == graph.sink)
            .map(|(_, d)| *d);
        assert_eq!(source_diff, Some(-1));
        assert_eq!(sink_diff, Some(1));
    }

    #[test]
    fn three_byte_string_all_substrings() {
        // "abc" → priced edges for: ab, bc, abc
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c']];
        let graph = GraphBuilder::build(&pretokens, DEFAULT_MAX_TOKEN_LEN);

        assert_eq!(graph.n_vertices, 4);
        assert_eq!(graph.free_edges.len(), 3);
        // 2-byte: ab, bc = 2 priced edges
        // 3-byte: abc = 1 priced edge
        assert_eq!(graph.priced_edges.len(), 3);
        // 3 unique colours
        assert_eq!(graph.colour_bytes.len(), 3);
    }
}
