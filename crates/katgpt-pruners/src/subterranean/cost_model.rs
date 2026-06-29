//! ProcedureCostModel — complexity-proportional budget allocation for procedure compilation.
//!
//! Paper: arXiv:2605.22502 — cost advantage scales with procedure complexity.
//! Reported ratios: 128× for 14 nodes, 462× for 55 nodes.

use crate::subterranean::ProcedureGraph;
use crate::subterranean::path_enumerator::PathEnumerator;

// ── ProcedureCostModel ─────────────────────────────────────────

/// Cost model based on paper's finding: compiled procedures achieve 87–98%
/// of frontier quality at 128–462× lower cost than in-context baselines.
///
/// The cost advantage scales with procedure complexity (node count and path count).
/// Paper: "cost scales roughly as O(ln(path_count)) for the compiled approach,
/// while in-context scales linearly with path_count."
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProcedureCostModel {
    /// Number of nodes in the procedure graph.
    pub node_count: usize,
    /// Total number of unique acyclic paths.
    pub path_count: usize,
    /// Average number of steps (edges) across all paths.
    pub avg_path_length: f64,
}

impl ProcedureCostModel {
    /// Build a cost model by enumerating paths on the given graph.
    ///
    /// Uses `max_depth` as safety limit for the path enumerator.
    /// Returns `None` if path enumeration finds no paths.
    pub fn from_graph<G: ProcedureGraph>(graph: &G, max_depth: usize) -> Option<Self> {
        let enumerator = PathEnumerator::new(graph, max_depth);
        let paths = enumerator.enumerate();

        match paths.is_empty() {
            true => None,
            false => {
                let total_steps: usize = paths.iter().map(|p| p.step_count()).sum();
                let avg_path_length = total_steps as f64 / paths.len() as f64;

                Some(Self {
                    node_count: graph.node_count(),
                    path_count: paths.len(),
                    avg_path_length,
                })
            }
        }
    }

    /// Build a cost model from pre-computed statistics.
    pub fn from_stats(node_count: usize, path_count: usize, avg_path_length: f64) -> Self {
        Self {
            node_count,
            path_count,
            avg_path_length,
        }
    }

    /// Estimate cost ratio: compiled (fine-tuned) vs. in-context prompting.
    ///
    /// Paper reports ~128× for 14-node graphs (86 paths) and ~462× for 55-node graphs (2381 paths).
    /// Base per-token cost ratio is ~65×; volume factor scales with ln(path_count).
    pub fn cost_ratio_vs_in_context(&self) -> f64 {
        let base = 65.0; // Per-token cost ratio (paper: ~65×)
        let volume_factor = 1.0 + (self.path_count as f64).ln().max(1.0);
        base * volume_factor
    }

    /// Recommended training budget proportional to complexity.
    ///
    /// Scales the base budget by ln(path_count), reflecting the paper's finding
    /// that training cost grows logarithmically while inference savings grow linearly.
    pub fn recommended_budget(&self, base_budget: usize) -> usize {
        let multiplier = (self.path_count as f64).ln().ceil().max(1.0) as usize;
        base_budget * multiplier
    }

    /// Training time estimate in hours on a single A100.
    ///
    /// Paper reports ~3-4 hours for 55 nodes (2381 paths).
    /// Scales linearly with path count, capped at 4 hours for practical limits.
    pub fn estimated_training_hours(&self) -> f64 {
        1.0 + (self.path_count as f64 / 500.0).min(3.0)
    }

    /// Quality estimate: expected fraction of frontier model quality.
    ///
    /// Paper: 87–98% depending on procedure complexity.
    /// Higher complexity → slightly lower quality but still >85%.
    pub fn estimated_quality_fraction(&self) -> f64 {
        match self.path_count {
            0..=50 => 0.98,
            51..=200 => 0.95,
            201..=1000 => 0.92,
            1001..=5000 => 0.89,
            _ => 0.87,
        }
    }

    /// Whether LoRA is likely sufficient for this procedure.
    ///
    /// Paper: "LoRA fails to approach full fine-tuning on procedural tasks."
    /// Only recommends LoRA for trivial procedures (< 10 paths).
    pub fn lora_feasible(&self) -> bool {
        self.path_count < 10
    }

    /// Complexity tier classification.
    pub fn complexity_tier(&self) -> ComplexityTier {
        match self.path_count {
            0..=10 => ComplexityTier::Trivial,
            11..=100 => ComplexityTier::Simple,
            101..=1000 => ComplexityTier::Moderate,
            1001..=5000 => ComplexityTier::Complex,
            _ => ComplexityTier::HighlyComplex,
        }
    }

    /// Total estimated tokens saved per inference vs. in-context approach.
    ///
    /// Assumes each path step corresponds to ~100 tokens of procedure context.
    pub fn tokens_saved_per_inference(&self) -> usize {
        // In-context needs avg_path_length * ~100 tokens per inference
        // Compiled approach needs 0 additional tokens
        (self.avg_path_length * 100.0) as usize
    }

    /// Break-even point: how many inferences before training cost is amortized.
    ///
    /// Compares training cost (hours × GPU cost) against per-inference savings.
    /// Uses a simplified model: training_cost = hours × $2/hr (A100),
    /// savings = tokens_saved × $0.00001/token per inference.
    pub fn break_even_inferences(&self) -> usize {
        let training_cost_usd = self.estimated_training_hours() * 2.0;
        let savings_per_inference_usd = self.tokens_saved_per_inference() as f64 * 0.00001;
        match savings_per_inference_usd > 0.0 {
            true => (training_cost_usd / savings_per_inference_usd).ceil() as usize,
            false => usize::MAX,
        }
    }
}

// ── ComplexityTier ─────────────────────────────────────────────

/// Classification of procedure graph complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ComplexityTier {
    /// 0–10 paths: LoRA may suffice.
    Trivial,
    /// 11–100 paths: paper's "travel" category.
    Simple,
    /// 101–1000 paths: paper's "zoom" category.
    Moderate,
    /// 1001–5000 paths: paper's "insurance" category.
    Complex,
    /// 5000+ paths: requires careful budget planning.
    HighlyComplex,
}

impl std::fmt::Display for ComplexityTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trivial => write!(f, "Trivial (0-10 paths)"),
            Self::Simple => write!(f, "Simple (11-100 paths)"),
            Self::Moderate => write!(f, "Moderate (101-1000 paths)"),
            Self::Complex => write!(f, "Complex (1001-5000 paths)"),
            Self::HighlyComplex => write!(f, "HighlyComplex (5000+ paths)"),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_ratio_paper_ballpark() {
        // Paper: 128× for ~86 paths (14 nodes)
        let model_14 = ProcedureCostModel::from_stats(14, 86, 10.0);
        let ratio_14 = model_14.cost_ratio_vs_in_context();
        assert!(
            ratio_14 > 100.0,
            "14-node ratio should be >100×, got {ratio_14}"
        );
        assert!(
            ratio_14 < 500.0,
            "14-node ratio should be <500×, got {ratio_14}"
        );

        // Paper: 462× for ~2381 paths (55 nodes)
        let model_55 = ProcedureCostModel::from_stats(55, 2381, 20.0);
        let ratio_55 = model_55.cost_ratio_vs_in_context();
        assert!(
            ratio_55 > 300.0,
            "55-node ratio should be >300×, got {ratio_55}"
        );
        assert!(
            ratio_55 < 800.0,
            "55-node ratio should be <800×, got {ratio_55}"
        );
    }

    #[test]
    fn test_cost_ratio_monotonic_with_path_count() {
        let model_low = ProcedureCostModel::from_stats(10, 50, 5.0);
        let model_mid = ProcedureCostModel::from_stats(20, 500, 10.0);
        let model_high = ProcedureCostModel::from_stats(50, 5000, 15.0);

        let ratio_low = model_low.cost_ratio_vs_in_context();
        let ratio_mid = model_mid.cost_ratio_vs_in_context();
        let ratio_high = model_high.cost_ratio_vs_in_context();

        assert!(
            ratio_low < ratio_mid,
            "More paths should yield higher cost ratio: {ratio_low} vs {ratio_mid}"
        );
        assert!(
            ratio_mid < ratio_high,
            "More paths should yield higher cost ratio: {ratio_mid} vs {ratio_high}"
        );
    }

    #[test]
    fn test_budget_scaling_monotonic() {
        let base = 1000;
        let model_low = ProcedureCostModel::from_stats(10, 50, 5.0);
        let model_high = ProcedureCostModel::from_stats(50, 5000, 15.0);

        let budget_low = model_low.recommended_budget(base);
        let budget_high = model_high.recommended_budget(base);

        assert!(
            budget_low < budget_high,
            "More paths should yield larger budget: {budget_low} vs {budget_high}"
        );
    }

    #[test]
    fn test_training_hours_reasonable() {
        let model = ProcedureCostModel::from_stats(55, 2381, 20.0);
        let hours = model.estimated_training_hours();

        // Paper: ~3-4 hours for 55 nodes
        assert!(hours >= 2.0, "Training hours should be >= 2, got {hours}");
        assert!(hours <= 4.0, "Training hours should be <= 4, got {hours}");
    }

    #[test]
    fn test_quality_fraction_bounds() {
        let model = ProcedureCostModel::from_stats(10, 5, 3.0);
        let quality = model.estimated_quality_fraction();

        assert!(quality >= 0.85, "Quality should be >= 85%, got {quality}");
        assert!(quality <= 1.0, "Quality should be <= 100%, got {quality}");
    }

    #[test]
    fn test_lora_feasibility() {
        let trivial = ProcedureCostModel::from_stats(5, 5, 2.0);
        assert!(trivial.lora_feasible(), "Trivial graph should allow LoRA");

        let complex = ProcedureCostModel::from_stats(55, 2381, 20.0);
        assert!(
            !complex.lora_feasible(),
            "Complex graph should not allow LoRA"
        );
    }

    #[test]
    fn test_complexity_tier() {
        assert_eq!(
            ProcedureCostModel::from_stats(5, 5, 2.0).complexity_tier(),
            ComplexityTier::Trivial
        );
        assert_eq!(
            ProcedureCostModel::from_stats(14, 86, 10.0).complexity_tier(),
            ComplexityTier::Simple
        );
        assert_eq!(
            ProcedureCostModel::from_stats(30, 500, 12.0).complexity_tier(),
            ComplexityTier::Moderate
        );
        assert_eq!(
            ProcedureCostModel::from_stats(55, 2381, 20.0).complexity_tier(),
            ComplexityTier::Complex
        );
        assert_eq!(
            ProcedureCostModel::from_stats(100, 10000, 25.0).complexity_tier(),
            ComplexityTier::HighlyComplex
        );
    }

    #[test]
    fn test_tokens_saved_per_inference() {
        let model = ProcedureCostModel::from_stats(14, 86, 10.0);
        let saved = model.tokens_saved_per_inference();
        // avg_path_length=10 × 100 tokens/step = 1000 tokens
        assert_eq!(saved, 1000);
    }

    #[test]
    fn test_break_even_finite() {
        let model = ProcedureCostModel::from_stats(14, 86, 10.0);
        let break_even = model.break_even_inferences();
        assert!(break_even > 0, "Break-even should be positive");
        assert!(break_even < usize::MAX, "Break-even should be finite");
    }

    #[test]
    fn test_from_graph_integration() {
        /// Minimal graph: 0 -> 1 -> 2(terminal).
        struct LinearGraph;

        impl ProcedureGraph for LinearGraph {
            type NodeId = u32;
            type Condition = String;

            fn start_node(&self) -> Self::NodeId {
                0
            }
            fn terminal_nodes(&self) -> &[Self::NodeId] {
                &[2]
            }
            fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
                static EMPTY: [(u32, Option<String>); 0] = [];
                static A: [(u32, Option<String>); 1] = [(1, None)];
                static B: [(u32, Option<String>); 1] = [(2, None)];
                match node {
                    0 => &A,
                    1 => &B,
                    _ => &EMPTY,
                }
            }
            fn node_count(&self) -> usize {
                3
            }
            fn edge_count(&self) -> usize {
                2
            }
            fn node_label(&self, node: Self::NodeId) -> &str {
                match node {
                    0 => "A",
                    1 => "B",
                    2 => "C",
                    _ => "?",
                }
            }
        }

        let graph = LinearGraph;
        let model = ProcedureCostModel::from_graph(&graph, 100);

        assert!(model.is_some(), "Should build cost model from linear graph");
        let m = model.unwrap();
        assert_eq!(m.node_count, 3);
        assert_eq!(m.path_count, 1);
        assert_eq!(m.avg_path_length, 2.0);
    }

    #[test]
    fn test_complexity_tier_display() {
        assert_eq!(
            format!("{}", ComplexityTier::Trivial),
            "Trivial (0-10 paths)"
        );
        assert_eq!(
            format!("{}", ComplexityTier::HighlyComplex),
            "HighlyComplex (5000+ paths)"
        );
    }
}
