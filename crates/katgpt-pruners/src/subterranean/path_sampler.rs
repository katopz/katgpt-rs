//! PathSampler — structured training data generation from procedure graph paths.
//!
//! Paper §3: "sample a path → scenario variables → generate turn-by-turn"
//! Produces training samples by decomposing trajectories into individual
//! decision points with valid action context.

use crate::subterranean::ProcedureGraph;
use crate::subterranean::path_enumerator::PathEnumerator;
use crate::subterranean::types::Trajectory;

// ── Sample ─────────────────────────────────────────────────────

/// A single training sample extracted from a trajectory at a specific turn.
///
/// Represents one decision point: given the path so far, which node was chosen
/// from the set of valid next actions.
#[derive(Debug, Clone)]
pub struct Sample<NodeId> {
    /// The full trajectory this sample was drawn from.
    pub trajectory: Trajectory<NodeId>,
    /// Index within the trajectory where this decision was made.
    pub turn_index: usize,
    /// Label of the current node at this turn.
    pub node_label: String,
    /// Valid next nodes available at this decision point.
    pub valid_next_actions: Vec<NodeId>,
    /// The action actually chosen in this trajectory.
    pub chosen_action: NodeId,
}

impl<NodeId: Copy + Eq + std::fmt::Debug> Sample<NodeId> {
    /// Whether the chosen action matches one of the valid next actions.
    pub fn is_valid_choice(&self) -> bool {
        self.valid_next_actions.contains(&self.chosen_action)
    }

    /// Number of alternative actions that were available but not chosen.
    pub fn alternative_count(&self) -> usize {
        match self.valid_next_actions.contains(&self.chosen_action) {
            true => self.valid_next_actions.len().saturating_sub(1),
            false => self.valid_next_actions.len(),
        }
    }

    /// Branching factor at this decision point.
    pub fn branching_factor(&self) -> usize {
        self.valid_next_actions.len()
    }
}

// ── SampleFilter ───────────────────────────────────────────────

/// Filter criteria for selecting which turns become training samples.
#[derive(Debug, Clone, Default)]
pub enum SampleFilter {
    /// Include all decision points in the trajectory.
    #[default]
    All,
    /// Include only the first N turns of each trajectory.
    FirstN(usize),
    /// Include only decision points with branching factor >= threshold.
    MinBranching(usize),
    /// Include only decision points where a condition was met.
    OnlyConditional,
}

// ── PathSampler ────────────────────────────────────────────────

/// Generates structured training data from procedure graph paths.
///
/// Paper: "sample a path → scenario variables → generate turn-by-turn"
/// Decomposes sampled trajectories into individual decision-point samples
/// suitable for bandit training or supervised fine-tuning.
pub struct PathSampler<'a, G: ProcedureGraph> {
    enumerator: PathEnumerator<'a, G>,
    rng: fastrand::Rng,
}

impl<'a, G: ProcedureGraph> PathSampler<'a, G> {
    /// Create a new path sampler with the given graph and RNG seed.
    ///
    /// `max_depth` is passed through to the internal `PathEnumerator`.
    pub fn new(graph: &'a G, max_depth: usize, seed: u64) -> Self {
        Self {
            enumerator: PathEnumerator::new(graph, max_depth),
            rng: fastrand::Rng::with_seed(seed),
        }
    }

    /// Access the underlying path enumerator.
    pub fn enumerator(&self) -> &PathEnumerator<'a, G> {
        &self.enumerator
    }

    /// Sample `n` random trajectories from the graph.
    ///
    /// Shorter paths receive higher weight (inverse-length weighting).
    pub fn sample_trajectories(&mut self, n: usize) -> Vec<Trajectory<G::NodeId>> {
        self.enumerator.sample(n, &mut self.rng)
    }

    /// Generate training samples from sampled trajectories.
    ///
    /// Each trajectory is decomposed into individual decision-point samples.
    /// The `filter` controls which turns are included.
    pub fn generate_samples(
        &mut self,
        n_trajectories: usize,
        filter: &SampleFilter,
    ) -> Vec<Sample<G::NodeId>> {
        let trajectories = self.sample_trajectories(n_trajectories);
        let graph = self.enumerator.graph();

        let mut samples = Vec::new();

        for trajectory in trajectories {
            let trajectory_samples = self.extract_samples(&trajectory, graph, filter);
            samples.extend(trajectory_samples);
        }

        samples
    }

    /// Generate samples from a single pre-existing trajectory.
    pub fn samples_from_trajectory(
        &self,
        trajectory: &Trajectory<G::NodeId>,
        filter: &SampleFilter,
    ) -> Vec<Sample<G::NodeId>> {
        self.extract_samples(trajectory, self.enumerator.graph(), filter)
    }

    /// Generate samples from all enumerated paths (exhaustive).
    ///
    /// Warning: can produce a very large number of samples for complex graphs.
    pub fn generate_all_samples(&self, filter: &SampleFilter) -> Vec<Sample<G::NodeId>> {
        let all_paths = self.enumerator.enumerate();
        let graph = self.enumerator.graph();

        let mut samples = Vec::new();
        for trajectory in &all_paths {
            samples.extend(self.extract_samples(trajectory, graph, filter));
        }
        samples
    }

    /// Count total samples that would be generated without materializing them.
    pub fn count_samples(&self, filter: &SampleFilter) -> usize {
        let all_paths = self.enumerator.enumerate();
        let graph = self.enumerator.graph();

        let mut count = 0;
        for trajectory in &all_paths {
            count += self.count_samples_for_trajectory(trajectory, graph, filter);
        }
        count
    }

    /// Reseed the RNG for reproducibility.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = fastrand::Rng::with_seed(seed);
    }

    // ── Internal helpers ───────────────────────────────────────

    /// Extract samples from a single trajectory based on filter criteria.
    fn extract_samples(
        &self,
        trajectory: &Trajectory<G::NodeId>,
        graph: &G,
        filter: &SampleFilter,
    ) -> Vec<Sample<G::NodeId>> {
        let mut samples = Vec::new();

        // Each turn is a decision point: at node i, we chose to go to node i+1
        for turn_index in 0..trajectory.path.len().saturating_sub(1) {
            let current_node = trajectory.path[turn_index];

            // Get valid next actions from the graph
            let edges = graph.edges_from(current_node);
            let valid_next: Vec<G::NodeId> = edges.iter().map(|(next, _)| *next).collect();

            // No decision point if no alternatives
            if valid_next.is_empty() {
                continue;
            }

            let chosen_action = trajectory.path[turn_index + 1];

            // Apply filter
            match filter {
                SampleFilter::All => {}
                SampleFilter::FirstN(max) => {
                    if turn_index >= *max {
                        continue;
                    }
                }
                SampleFilter::MinBranching(min) => {
                    if valid_next.len() < *min {
                        continue;
                    }
                }
                SampleFilter::OnlyConditional => {
                    let condition = trajectory.conditions_met.get(turn_index);
                    match condition {
                        Some(Some(_)) => {} // Has condition, include
                        _ => continue,      // No condition, skip
                    }
                }
            }

            let node_label = graph.node_label(current_node).to_string();

            samples.push(Sample {
                trajectory: trajectory.clone(),
                turn_index,
                node_label,
                valid_next_actions: valid_next,
                chosen_action,
            });
        }

        samples
    }

    /// Count samples for a trajectory without building Sample structs.
    fn count_samples_for_trajectory(
        &self,
        trajectory: &Trajectory<G::NodeId>,
        graph: &G,
        filter: &SampleFilter,
    ) -> usize {
        let mut count = 0;

        for turn_index in 0..trajectory.path.len().saturating_sub(1) {
            let current_node = trajectory.path[turn_index];
            let edges = graph.edges_from(current_node);
            let valid_count = edges.len();

            if valid_count == 0 {
                continue;
            }

            match filter {
                SampleFilter::All => {}
                SampleFilter::FirstN(max) => {
                    if turn_index >= *max {
                        continue;
                    }
                }
                SampleFilter::MinBranching(min) => {
                    if valid_count < *min {
                        continue;
                    }
                }
                SampleFilter::OnlyConditional => {
                    let condition = trajectory.conditions_met.get(turn_index);
                    match condition {
                        Some(Some(_)) => {}
                        _ => continue,
                    }
                }
            }

            count += 1;
        }

        count
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Reusable diamond graph: A(0) -> B(1), A(0) -> C(2), both -> D(3, terminal).
    struct DiamondGraph;

    impl ProcedureGraph for DiamondGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[3]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            static A_EDGES: [(u32, Option<String>); 2] = [(1, None), (2, None)];
            static B_EDGES: [(u32, Option<String>); 1] = [(3, None)];
            static C_EDGES: [(u32, Option<String>); 1] = [(3, None)];
            match node {
                0 => &A_EDGES,
                1 => &B_EDGES,
                2 => &C_EDGES,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            4
        }
        fn edge_count(&self) -> usize {
            4
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "A",
                1 => "B",
                2 => "C",
                3 => "D",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_sample_trajectories_count() {
        let graph = DiamondGraph;
        let mut sampler = PathSampler::new(&graph, 100, 42);

        let trajectories = sampler.sample_trajectories(2);
        assert_eq!(trajectories.len(), 2);
    }

    #[test]
    fn test_generate_samples_all_filter() {
        let graph = DiamondGraph;
        let mut sampler = PathSampler::new(&graph, 100, 42);

        let samples = sampler.generate_samples(2, &SampleFilter::All);

        // Diamond graph has 2 paths, each with 2 decision points:
        // Path 0->1->3: turn 0 (A->B vs A->C), turn 1 (B->D)
        // Path 0->2->3: turn 0 (A->B vs A->C), turn 1 (C->D)
        assert!(!samples.is_empty());

        for sample in &samples {
            assert!(sample.is_valid_choice());
            assert_eq!(
                sample.trajectory.path[sample.turn_index + 1],
                sample.chosen_action
            );
        }
    }

    #[test]
    fn test_generate_samples_first_n_filter() {
        let graph = DiamondGraph;
        let mut sampler = PathSampler::new(&graph, 100, 42);

        // Only include first turn (the branching decision)
        let samples = sampler.generate_samples(10, &SampleFilter::FirstN(1));

        for sample in &samples {
            assert_eq!(sample.turn_index, 0);
            // At node A, branching factor should be 2
            assert_eq!(sample.branching_factor(), 2);
        }
    }

    #[test]
    fn test_generate_samples_min_branching_filter() {
        let graph = DiamondGraph;
        let mut sampler = PathSampler::new(&graph, 100, 42);

        // Only include turns with branching factor >= 2
        let samples = sampler.generate_samples(10, &SampleFilter::MinBranching(2));

        for sample in &samples {
            assert!(sample.branching_factor() >= 2);
        }
    }

    #[test]
    fn test_count_samples_matches_generate() {
        let graph = DiamondGraph;
        let sampler = PathSampler::new(&graph, 100, 42);

        let filter = SampleFilter::All;
        let count = sampler.count_samples(&filter);
        let generated = sampler.generate_all_samples(&filter);

        assert_eq!(count, generated.len());
    }

    #[test]
    fn test_samples_from_single_trajectory() {
        let graph = DiamondGraph;
        let sampler = PathSampler::new(&graph, 100, 42);

        // Manually create a trajectory
        let mut trajectory = Trajectory::from_start(0u32);
        trajectory.push(1, None);
        trajectory.push(3, None);

        let samples = sampler.samples_from_trajectory(&trajectory, &SampleFilter::All);

        // Turn 0: A -> B (branching factor 2)
        // Turn 1: B -> D (branching factor 1)
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].turn_index, 0);
        assert_eq!(samples[0].branching_factor(), 2);
        assert_eq!(samples[0].chosen_action, 1);
        assert_eq!(samples[1].turn_index, 1);
        assert_eq!(samples[1].chosen_action, 3);
    }

    #[test]
    fn test_sample_alternative_count() {
        let graph = DiamondGraph;
        let sampler = PathSampler::new(&graph, 100, 42);

        let mut trajectory = Trajectory::from_start(0u32);
        trajectory.push(1, None);

        let samples = sampler.samples_from_trajectory(&trajectory, &SampleFilter::All);

        // At A with 2 valid actions, choosing B leaves 1 alternative
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].alternative_count(), 1);
    }

    /// Graph with conditional edges for testing OnlyConditional filter.
    struct ConditionalGraph;

    impl ProcedureGraph for ConditionalGraph {
        type NodeId = u32;
        type Condition = String;

        fn start_node(&self) -> Self::NodeId {
            0
        }
        fn terminal_nodes(&self) -> &[Self::NodeId] {
            &[2, 3]
        }
        fn edges_from(&self, node: Self::NodeId) -> &[(Self::NodeId, Option<Self::Condition>)] {
            static EMPTY: [(u32, Option<String>); 0] = [];
            // A -> B (conditional: "enemy_nearby"), A -> C (unconditional terminal)
            static A_EDGES: [(u32, Option<String>); 2] = [
                (1, Some(String::new())), // conditional placeholder
                (3, None),
            ];
            static B_EDGES: [(u32, Option<String>); 1] = [(2, None)];
            match node {
                0 => &A_EDGES,
                1 => &B_EDGES,
                _ => &EMPTY,
            }
        }
        fn node_count(&self) -> usize {
            4
        }
        fn edge_count(&self) -> usize {
            3
        }
        fn node_label(&self, node: Self::NodeId) -> &str {
            match node {
                0 => "A",
                1 => "B",
                2 => "C",
                3 => "D",
                _ => "?",
            }
        }
    }

    #[test]
    fn test_only_conditional_filter() {
        let graph = ConditionalGraph;
        let sampler = PathSampler::new(&graph, 100, 42);

        let all_paths = sampler.enumerator().enumerate();

        // Find a path that has a conditional step
        for path in &all_paths {
            let samples = sampler.samples_from_trajectory(path, &SampleFilter::OnlyConditional);
            for sample in &samples {
                // Every sample must have come from a conditional edge
                let condition = sample.trajectory.conditions_met.get(sample.turn_index);
                assert!(condition.is_some());
            }
        }
    }
}
