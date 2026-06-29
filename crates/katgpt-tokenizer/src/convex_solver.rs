//! ConvexTok LP solver using good_lp/HiGHS.
//!
//! Formulates the tokenisation vocabulary optimization as a linear program:
//! - Variables: free-edge usage (f), priced-edge usage (p), colour selection (c)
//! - Objective: minimize total path length (Σf + Σp)
//! - Constraints: flow conservation, colour activation, vocabulary budget
//!
//! **Source:** Tempus et al. (2026). Tokenisation via Convex Relaxations. arXiv:2605.22821

use good_lp::{
    Expression, ProblemVariables, Solution, SolverModel, Variable, default_solver, variable,
};

use super::convex_types::{LpSolution, TokenisationGraph};

/// LP solver for tokenisation graph vocabulary optimization.
///
/// Formulates and solves:
/// ```text
/// min  Σ p_e + Σ f_e                    (total path length = compression)
/// s.t. P·p + F·f = d                    (flow conservation at each vertex)
///      p_e ≤ c_{colour(e)}  ∀e ∈ P     (edge usable only if colour selected)
///      Σ c_c ≤ K                        (vocabulary budget)
///      0 ≤ f, p, c ≤ 1                  (LP relaxation)
/// ```
pub struct ConvexSolver;

impl ConvexSolver {
    /// Solve the LP relaxation for the tokenisation graph.
    ///
    /// # Arguments
    /// * `graph` — The tokenisation graph (from `GraphBuilder`)
    /// * `budget_k` — Maximum number of colours (vocabulary budget)
    ///
    /// # Returns
    /// The LP solution with fractional variables and objective value,
    /// or an error message if the LP is infeasible or the graph is empty.
    pub fn solve(graph: &TokenisationGraph, budget_k: usize) -> Result<LpSolution, String> {
        if graph.n_vertices == 0 {
            return Err("Cannot solve LP for empty graph".into());
        }

        // Phase 1: Create decision variables — all in [0, 1]
        let mut vars = ProblemVariables::new();

        let f_vars: Vec<Variable> = (0..graph.n_free_edges())
            .map(|_| vars.add(variable().min(0.0).max(1.0)))
            .collect();

        let p_vars: Vec<Variable> = (0..graph.n_priced_edges())
            .map(|_| vars.add(variable().min(0.0).max(1.0)))
            .collect();

        let c_vars: Vec<Variable> = (0..graph.n_colours())
            .map(|_| vars.add(variable().min(0.0).max(1.0)))
            .collect();

        // Phase 2: Objective — minimize Σ f_e + Σ p_e
        let objective: Expression = f_vars
            .iter()
            .chain(p_vars.iter())
            .map(|&v| Expression::from(v))
            .sum();

        let mut model = vars.minimise(objective).using(default_solver);

        // Phase 3: Flow conservation at each vertex
        add_flow_constraints(&mut model, graph, &f_vars, &p_vars);

        // Phase 4: Colour activation — p_e ≤ c_{colour(e)}
        for (e, &(_, _, colour)) in graph.priced_edges.iter().enumerate() {
            let p_expr = Expression::from(p_vars[e]);
            let c_expr = Expression::from(c_vars[colour.0 as usize]);
            model.add_constraint((p_expr - c_expr).leq(0.0));
        }

        // Phase 5: Budget — Σ c_c ≤ K
        if !c_vars.is_empty() {
            let budget_expr: Expression = c_vars.iter().map(|&v| Expression::from(v)).sum();
            model.add_constraint(budget_expr.leq(budget_k as f64));
        }

        // Phase 6: Solve via HiGHS
        let solution = model.solve().map_err(|e| format!("LP solve failed: {e}"))?;

        // Phase 7: Extract solution vectors. Compute `lp_value` in a single
        // fused pass over f + p instead of two separate sums.
        let f: Vec<f64> = f_vars.iter().map(|&v| solution.value(v)).collect();
        let p: Vec<f64> = p_vars.iter().map(|&v| solution.value(v)).collect();
        let c: Vec<f64> = c_vars.iter().map(|&v| solution.value(v)).collect();
        let lp_value: f64 = f.iter().chain(p.iter()).copied().sum();

        Ok(LpSolution {
            f,
            p,
            c,
            lp_value,
            budget_k,
        })
    }
}

/// Add flow conservation constraints for all vertices with incident edges.
///
/// For each vertex v:
///   Σ(entering flows) − Σ(leaving flows) = demand[v]
///
/// Where demand comes from `graph.flow_diff`:
///   source: −1 (one unit of net outflow)
///   sink:   +1 (one unit of net inflow)
///   others:  0 (inflow = outflow)
///
/// Hot-path optimization: replace `HashMap<u32, Vec<_>>` incidence map with a
/// `Vec<Vec<_>>` indexed by `vertex_id` — direct O(1) indexing beats hashing
/// for the dense vertex-id space. Same for the demand lookup.
fn add_flow_constraints<M: SolverModel>(
    model: &mut M,
    graph: &TokenisationGraph,
    f_vars: &[Variable],
    p_vars: &[Variable],
) {
    let n = graph.n_vertices;

    // Build per-vertex incidence lists: (edge_index, coefficient, is_free)
    // coefficient: +1 for entering, −1 for leaving
    // Vec<Vec<_>> indexed by vertex_id — O(1) lookup, no hashing.
    let mut incidence: Vec<Vec<(usize, f64, bool)>> = (0..n).map(|_| Vec::new()).collect();

    // Pre-count outgoing + incoming edges per vertex to avoid Vec reallocations.
    // One pass through each edge list suffices.
    let mut deg = vec![0usize; n];
    for &(from, to) in &graph.free_edges {
        deg[from.0 as usize] += 1;
        deg[to.0 as usize] += 1;
    }
    for &(from, to, _) in &graph.priced_edges {
        deg[from.0 as usize] += 1;
        deg[to.0 as usize] += 1;
    }
    for (v, d) in incidence.iter_mut().zip(deg.iter()) {
        Vec::reserve(v, *d);
    }

    for (e, &(from, to)) in graph.free_edges.iter().enumerate() {
        incidence[from.0 as usize].push((e, -1.0, true)); // leaves from
        incidence[to.0 as usize].push((e, 1.0, true)); // enters to
    }

    for (e, &(from, to, _)) in graph.priced_edges.iter().enumerate() {
        incidence[from.0 as usize].push((e, -1.0, false)); // leaves from
        incidence[to.0 as usize].push((e, 1.0, false)); // enters to
    }

    // Demand lookup: dense Vec indexed by vertex_id. Default 0.0.
    let mut demand = vec![0.0f64; n];
    for &(v, d) in &graph.flow_diff {
        demand[v.0 as usize] = d as f64;
    }

    // Add flow conservation constraint for each vertex with incident edges.
    for (vertex_id, incidents) in incidence.iter().enumerate() {
        if incidents.is_empty() {
            continue;
        }
        let expr: Expression = incidents
            .iter()
            .map(|&(e, coeff, is_free)| {
                let var = if is_free { f_vars[e] } else { p_vars[e] };
                Expression::from(var) * coeff
            })
            .sum();

        let d = demand[vertex_id];
        model.add_constraint(expr.eq(d));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convex_graph::GraphBuilder;

    #[test]
    fn empty_graph_returns_error() {
        let graph = GraphBuilder::build(&[], 64);
        let result = ConvexSolver::solve(&graph, 10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty graph"));
    }

    #[test]
    fn single_byte_no_priced_edges() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a']];
        let graph = GraphBuilder::build(&pretokens, 64);

        let sol = ConvexSolver::solve(&graph, 10).unwrap();

        // Only one free edge, must be fully used
        assert!(
            sol.f[0] > 0.99,
            "free edge should be ~1.0, got {}",
            sol.f[0]
        );
        assert!(sol.p.is_empty(), "no priced edges expected");
        assert!(sol.c.is_empty(), "no colours expected");
        assert!(
            sol.lp_value > 0.99,
            "lp_value should be ~1.0, got {}",
            sol.lp_value
        );
    }

    #[test]
    fn two_bytes_budget_zero_forces_free_edges() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b']];
        let graph = GraphBuilder::build(&pretokens, 64);

        let sol = ConvexSolver::solve(&graph, 0).unwrap();

        // Budget 0: cannot use priced edge → must use 2 free edges
        assert!(
            sol.lp_value > 1.99,
            "lp_value should be ~2.0, got {}",
            sol.lp_value
        );
        assert!(
            sol.c[0] < 0.01,
            "colour should not be selected, got {}",
            sol.c[0]
        );
    }

    #[test]
    fn two_bytes_budget_one_enables_compression() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b']];
        let graph = GraphBuilder::build(&pretokens, 64);

        let sol = ConvexSolver::solve(&graph, 1).unwrap();

        // Budget 1: can use priced edge "ab" → compression = 1
        assert!(
            sol.lp_value < 1.01,
            "lp_value should be ~1.0, got {}",
            sol.lp_value
        );
        assert!(
            sol.c[0] > 0.5,
            "colour should be selected, got {}",
            sol.c[0]
        );
    }

    #[test]
    fn four_bytes_multiple_colours() {
        // "abcd" → priced edges: ab, bc, cd, abc, bcd, abcd
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b', b'c', b'd']];
        let graph = GraphBuilder::build(&pretokens, 4);

        let sol = ConvexSolver::solve(&graph, 2).unwrap();

        // With budget 2, can select up to 2 colours.
        // Best compression: pick "abcd" (spans 4 bytes in 1 token)
        // LP value should be ≤ 2 (1 priced edge + at most 1 free edge)
        assert!(
            sol.lp_value <= 2.01,
            "lp_value should be ≤ 2.0 with budget 2, got {}",
            sol.lp_value
        );
        assert_eq!(sol.c.len(), 6, "expected 6 colours");
        // At least one colour should be meaningfully selected
        let any_selected = sol.c.iter().any(|&c| c > 0.5);
        assert!(any_selected, "at least one colour should be selected");
    }

    #[test]
    fn budget_exceeds_colours_selects_all() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'a', b'b']];
        let graph = GraphBuilder::build(&pretokens, 64);

        let sol = ConvexSolver::solve(&graph, 100).unwrap();

        // Budget >> n_colours → all colours can be selected
        assert!(
            sol.c[0] > 0.5,
            "colour should be selected with excess budget"
        );
        assert!(
            sol.lp_value < 1.01,
            "lp_value should be ~1.0, got {}",
            sol.lp_value
        );
    }

    #[test]
    fn solution_respects_bounds() {
        let pretokens: Vec<Vec<u8>> = vec![vec![b'h', b'e', b'l', b'l', b'o']];
        let graph = GraphBuilder::build(&pretokens, 4);

        let sol = ConvexSolver::solve(&graph, 3).unwrap();

        // All variables should be in [0, 1]
        for (i, &f) in sol.f.iter().enumerate() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&f),
                "f[{i}] = {f} out of bounds"
            );
        }
        for (i, &p) in sol.p.iter().enumerate() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&p),
                "p[{i}] = {p} out of bounds"
            );
        }
        for (i, &c) in sol.c.iter().enumerate() {
            assert!(
                (-1e-6..=1.0 + 1e-6).contains(&c),
                "c[{i}] = {c} out of bounds"
            );
        }
    }
}
