//! ConvexTok tokenisation graph types for LP-based vocabulary optimization.
//!
//! Models the tokenisation problem as a directed acyclic graph (DAG) with coloured edges,
//! where finding an optimal vocabulary reduces to an LP relaxation + rounding problem.
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

use serde::{Deserialize, Serialize};

// ── Newtype IDs ────────────────────────────────────────────────

/// A vertex in the tokenisation graph.
/// Represents a position between two bytes in the concatenated dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexId(pub u32);

/// A free edge (byte-edge) in the tokenisation graph.
/// Always available; doesn't consume vocabulary budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FreeEdgeId(pub u32);

/// A priced edge (token-edge) in the tokenisation graph.
/// Requires its colour to be selected in the vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PricedEdgeId(pub u32);

/// A colour representing a potential token.
/// All priced edges with the same byte-substring share a colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColourId(pub u32);

// ── Core Graph ─────────────────────────────────────────────────

/// The tokenisation graph.
/// Encodes the full tokenisation problem as a DAG with coloured edges.
///
/// Vertices represent positions between bytes. Free edges span single bytes
/// (always usable). Priced edges span multi-byte substrings and require their
/// colour to be selected in the vocabulary budget.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl TokenisationGraph {
    /// Number of free edges in the graph.
    pub fn n_free_edges(&self) -> usize {
        self.free_edges.len()
    }

    /// Number of priced edges in the graph.
    pub fn n_priced_edges(&self) -> usize {
        self.priced_edges.len()
    }

    /// Number of distinct colours (unique byte-substrings).
    pub fn n_colours(&self) -> usize {
        self.colour_bytes.len()
    }
}

// ── LP Solution ────────────────────────────────────────────────

/// LP solution before rounding.
/// Contains fractional values for all variables.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl LpSolution {
    /// Count how many colour variables are nearly integral (c ≥ 0.999 or c ≤ 0.001).
    pub fn integral_colour_count(&self) -> usize {
        self.c.iter().filter(|&&c| c >= 0.999 || c <= 0.001).count()
    }

    /// Fraction of colour variables that are nearly integral.
    pub fn integrality_fraction(&self) -> f64 {
        if self.c.is_empty() {
            return 1.0;
        }
        self.integral_colour_count() as f64 / self.c.len() as f64
    }
}

// ── Rounded Vocabulary ─────────────────────────────────────────

/// A rounded (discrete) vocabulary selection.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

// ── Rounding Scheme ────────────────────────────────────────────

/// Rounding scheme variants (Section 4 of the paper).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum RoundingScheme {
    /// Deterministic: top-K colours by LP value c.
    /// Best for BpB (bits-per-byte) metric.
    Det,
    /// Biased: top-K by c / token_length (favours shorter tokens).
    /// Best for intrinsic metrics (compression, vocab utilisation).
    Bias,
    /// Integral-only: keep only c ≥ 0.999.
    /// Reveals which tokens the LP considers "forced".
    Int,
}

impl std::fmt::Display for RoundingScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoundingScheme::Det => write!(f, "det"),
            RoundingScheme::Bias => write!(f, "bias"),
            RoundingScheme::Int => write!(f, "int"),
        }
    }
}

// ── Optimality Certification ───────────────────────────────────

/// Optimality certification result.
/// Compares achieved compression against the LP-proven lower bound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimalityCert {
    /// LP lower bound on compression.
    pub lp_lower_bound: f64,
    /// Actual compression achieved.
    pub actual_compression: f64,
    /// Optimality gap: (actual - lp) / lp × 100%.
    pub gap_percent: f64,
    /// Fraction of LP solution that was already integral.
    pub integrality_fraction: f64,
    /// Whether the tokeniser is within 1% of optimal.
    pub within_one_percent: bool,
}

impl std::fmt::Display for OptimalityCert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.within_one_percent {
            "✅ ≤1%"
        } else {
            "❌ >1%"
        };
        write!(
            f,
            "gap={:.2}% {} (lp={:.4}, actual={:.4}, integral={:.0}%)",
            self.gap_percent,
            status,
            self.lp_lower_bound,
            self.actual_compression,
            self.integrality_fraction * 100.0,
        )
    }
}
