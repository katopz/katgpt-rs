//! Contrastive Neuron Attribution (CNA) — Sparse circuit discovery and runtime modulation.
//!
//! Discovers sparse MLP neuron circuits from contrastive prompt pairs and enables
//! runtime modulation of those neurons during inference. Inspired by activation
//! engineering and mechanistic interpretability research.
//!
//! # Architecture
//!
//! - [`CnaNeuron`] — a single discovered neuron from contrastive pair analysis
//! - [`CnaCircuit`] — a discovered circuit of neurons with metadata
//! - [`CnaDiscoveryConfig`] — hyperparameters for circuit discovery
//! - [`CnaModulator`] — runtime modulation state for steering discovered circuits
//! - [`CnaScreeningPruner`] — model-based pruner using discovered circuits
//!
//! # Usage
//!
//! ```rust,ignore
//! // 1. Discover circuit from contrastive activations
//! let circuit = cna_discover(&pos_acts, &neg_acts, n_layers, mlp_hidden, &config);
//!
//! // 2. Create modulator for runtime steering
//! let modulator = CnaModulator { circuit, multiplier: 0.0 }; // ablate
//!
//! // 3. Call in forward pass between matmul_relu and matmul(w2)
//! cna_modulate(&mut hidden, layer_idx, &modulator);
//! ```

use std::collections::{HashMap, HashSet};

use crate::speculative::types::ScreeningPruner;

// ── Types ───────────────────────────────────────────────────────

/// A single discovered neuron from contrastive pair analysis.
#[derive(Debug, Clone)]
pub struct CnaNeuron {
    /// Transformer layer index.
    pub layer: usize,
    /// Index into post-ReLU MLP activations (`ctx.hidden`).
    pub index: usize,
    /// Mean activation difference: `|mean_pos - mean_neg|`.
    pub delta: f32,
}

/// A discovered circuit of neurons identified by contrastive pair analysis.
#[derive(Debug, Clone)]
pub struct CnaCircuit {
    /// Sparse set of neurons, sorted by `|delta|` descending.
    pub neurons: Vec<CnaNeuron>,
    /// Secondary index for O(1) membership checks on `(layer, index)`.
    pub neuron_set: HashSet<(usize, usize)>,
    /// Pre-computed layer → neuron indices within `neurons` vec for O(k_layer) modulation.
    pub layer_index: HashMap<usize, Vec<usize>>,
    /// Universal neurons filtered out `(layer, index)` — fired in ≥80% of diverse prompts.
    pub universal_excluded: Vec<(usize, usize)>,
    /// Secondary index for O(1) universal exclusion checks.
    pub universal_excluded_set: HashSet<(usize, usize)>,
    /// Number of positive prompts used in discovery.
    pub n_positive: usize,
    /// Number of negative prompts used in discovery.
    pub n_negative: usize,
    /// Total MLP activation slots: `n_layer * mlp_hidden`.
    pub total_mlp_activations: usize,
}

/// Configuration for CNA discovery.
#[derive(Debug, Clone)]
pub struct CnaDiscoveryConfig {
    /// Fraction of total MLP activations to select (default: 0.001 = 0.1%).
    pub top_pct: f32,
    /// Threshold for universal neuron filtering: neuron is universal if it appears
    /// in top-0.1% for this fraction of diverse prompts (default: 0.8).
    pub universal_threshold: f32,
    /// Only capture activations from the last fraction of layers (default: 0.15 = last 15%).
    pub late_layer_fraction: f32,
}

impl Default for CnaDiscoveryConfig {
    fn default() -> Self {
        Self {
            top_pct: 0.001,
            universal_threshold: 0.8,
            late_layer_fraction: 0.15,
        }
    }
}

/// Runtime modulation state for steering discovered circuits.
#[derive(Debug, Clone)]
pub struct CnaModulator {
    /// The discovered circuit to modulate.
    pub circuit: CnaCircuit,
    /// Per-neuron multiplier:
    /// - `0.0` = ablate (zero out)
    /// - `1.0` = baseline (no change)
    /// - `>1.0` = amplify
    pub multiplier: f32,
}

// ── Discovery ───────────────────────────────────────────────────

/// Discover contrastive neurons from activation captures.
///
/// For each neuron `(layer, index)`, computes:
/// `δ = mean_positive_activation - mean_negative_activation`
///
/// Then selects the top-k neurons by `|δ|`, where
/// `k = ceil(top_pct * n_layers * mlp_hidden)`.
///
/// # Arguments
///
/// * `positive_activations` — `(layer_idx, activations_slice)` for positive prompts
/// * `negative_activations` — `(layer_idx, activations_slice)` for negative prompts
/// * `n_layers` — total number of transformer layers
/// * `mlp_hidden` — size of MLP hidden dimension
/// * `config` — discovery hyperparameters
///
/// # Returns
///
/// A [`CnaCircuit`] with the top neurons sorted by `|delta|` descending.
pub fn cna_discover(
    positive_activations: &[(usize, &[f32])],
    negative_activations: &[(usize, &[f32])],
    n_layers: usize,
    mlp_hidden: usize,
    config: &CnaDiscoveryConfig,
) -> CnaCircuit {
    let total_slots = n_layers * mlp_hidden;
    let k = match config.top_pct {
        pct if pct <= 0.0 => 1,
        pct => ((pct * total_slots as f32).ceil() as usize).max(1),
    };

    // Build per-neuron sums and counts for positive activations.
    let mut pos_sum = vec![0.0f64; total_slots];
    let mut pos_count = vec![0usize; total_slots];
    for &(layer, activations) in positive_activations {
        let base = layer * mlp_hidden;
        for (i, &val) in activations.iter().enumerate() {
            let idx = base + i;
            if idx < total_slots {
                pos_sum[idx] += val as f64;
                pos_count[idx] += 1;
            }
        }
    }

    // Build per-neuron sums and counts for negative activations.
    let mut neg_sum = vec![0.0f64; total_slots];
    let mut neg_count = vec![0usize; total_slots];
    for &(layer, activations) in negative_activations {
        let base = layer * mlp_hidden;
        for (i, &val) in activations.iter().enumerate() {
            let idx = base + i;
            if idx < total_slots {
                neg_sum[idx] += val as f64;
                neg_count[idx] += 1;
            }
        }
    }

    // Compute deltas for every neuron that has at least one observation.
    let mut candidates: Vec<CnaNeuron> = Vec::with_capacity(total_slots);
    for slot in 0..total_slots {
        let layer = slot / mlp_hidden;
        let index = slot % mlp_hidden;

        let mean_pos = match pos_count[slot] {
            0 => 0.0f64,
            c => pos_sum[slot] / c as f64,
        };
        let mean_neg = match neg_count[slot] {
            0 => 0.0f64,
            c => neg_sum[slot] / c as f64,
        };

        let delta = (mean_pos - mean_neg).abs() as f32;
        // Only include neurons with non-zero delta to avoid noise.
        if delta > 0.0 {
            candidates.push(CnaNeuron {
                layer,
                index,
                delta,
            });
        }
    }

    // Partial sort: O(n) select top-k by |delta| descending instead of O(n log n) full sort.
    if k < candidates.len() {
        candidates.select_nth_unstable_by(k, |a, b| {
            b.delta
                .partial_cmp(&a.delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(k);
    } else {
        candidates.sort_unstable_by(|a, b| {
            b.delta
                .partial_cmp(&a.delta)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Build layer → neuron-indices lookup for O(k_layer) modulation.
    let mut layer_index: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, neuron) in candidates.iter().enumerate() {
        layer_index.entry(neuron.layer).or_default().push(i);
    }

    CnaCircuit {
        neuron_set: candidates.iter().map(|n| (n.layer, n.index)).collect(),
        layer_index,
        universal_excluded_set: HashSet::new(),
        neurons: candidates,
        universal_excluded: Vec::new(),
        n_positive: positive_activations.len(),
        n_negative: negative_activations.len(),
        total_mlp_activations: total_slots,
    }
}

/// Detect universal neurons from diverse prompt activations.
///
/// A neuron is universal if it's in the top-0.1% for ≥`threshold` fraction of prompts.
/// Universal neurons are generally not content-specific and should be excluded
/// from contrastive circuits.
///
/// # Arguments
///
/// * `diverse_activations` — per-prompt: list of `(layer, activations)`
/// * `n_layers` — total number of transformer layers
/// * `mlp_hidden` — size of MLP hidden dimension
/// * `threshold` — fraction of prompts a neuron must appear in to be universal
///
/// # Returns
///
/// Set of `(layer, index)` pairs to exclude from circuits.
pub fn detect_universal_neurons(
    diverse_activations: &[Vec<(usize, Vec<f32>)>],
    n_layers: usize,
    mlp_hidden: usize,
    threshold: f32,
) -> Vec<(usize, usize)> {
    if diverse_activations.is_empty() {
        return Vec::new();
    }

    let total_slots = n_layers * mlp_hidden;
    let n_prompts = diverse_activations.len();
    let top_k = match (total_slots as f32 * 0.001).ceil() as usize {
        0 => 1,
        k => k,
    };

    // Count how many prompts each neuron appears in the top-0.1%.
    let mut appearance_count = vec![0usize; total_slots];

    for prompt_acts in diverse_activations {
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(total_slots);
        for &(layer, ref activations) in prompt_acts {
            let base = layer * mlp_hidden;
            for (i, &val) in activations.iter().enumerate() {
                let idx = base + i;
                if idx < total_slots {
                    scored.push((idx, val));
                }
            }
        }

        // Partial-sort for top-0.1% by activation magnitude descending.
        // select_nth_unstable_by partitions around the k-th element in O(n),
        // then we truncate to keep only the top-k.
        if top_k < scored.len() {
            scored.select_nth_unstable_by(top_k, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(top_k);
        } else {
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        for &(slot, _) in &scored {
            appearance_count[slot] += 1;
        }
    }

    // Neurons appearing in >= threshold fraction of prompts are universal.
    let min_appearances = (threshold * n_prompts as f32).ceil() as usize;
    let mut universal = Vec::new();
    for (slot, &count) in appearance_count.iter().enumerate().take(total_slots) {
        if count >= min_appearances {
            let layer = slot / mlp_hidden;
            let index = slot % mlp_hidden;
            universal.push((layer, index));
        }
    }

    universal
}

// ── Forward Hook ────────────────────────────────────────────────

/// Modulate post-ReLU MLP activations according to discovered circuit.
///
/// Call this between `matmul_relu` and `matmul(w2)` in `forward_base()`.
///
/// If `multiplier == 1.0`, returns immediately (baseline, no-op).
/// O(k_layer) where k_layer = number of circuit neurons for this layer only.
#[inline]
pub fn cna_modulate(hidden: &mut [f32], layer_idx: usize, modulator: &CnaModulator) {
    if modulator.multiplier == 1.0 {
        return;
    }
    if let Some(indices) = modulator.circuit.layer_index.get(&layer_idx) {
        for &ni in indices {
            let neuron = &modulator.circuit.neurons[ni];
            if neuron.index < hidden.len() {
                hidden[neuron.index] *= modulator.multiplier;
            }
        }
    }
}

// ── ScreeningPruner ─────────────────────────────────────────────

/// Model-based [`ScreeningPruner`] that uses discovered CNA circuits for relevance scoring.
///
/// CNA is primarily a runtime modulation tool (forward hook). As a `ScreeningPruner`,
/// it provides a baseline relevance of 1.0, allowing composition with
/// `BanditPruner<CnaScreeningPruner>` for online refinement.
pub struct CnaScreeningPruner {
    circuit: CnaCircuit,
}

impl CnaScreeningPruner {
    /// Create a new CNA screening pruner from a discovered circuit.
    pub fn new(circuit: CnaCircuit) -> Self {
        Self { circuit }
    }

    /// Access the underlying circuit.
    pub fn circuit(&self) -> &CnaCircuit {
        &self.circuit
    }

    /// Check if a given neuron `(layer, index)` is part of the discovered circuit.
    pub fn is_circuit_neuron(&self, layer: usize, index: usize) -> bool {
        self.circuit.neuron_set.contains(&(layer, index))
    }

    /// Check if a given neuron `(layer, index)` was excluded as universal.
    pub fn is_universal_excluded(&self, layer: usize, index: usize) -> bool {
        self.circuit
            .universal_excluded_set
            .contains(&(layer, index))
    }
}

impl ScreeningPruner for CnaScreeningPruner {
    fn relevance(&self, _depth: usize, _token_idx: usize, _parent_tokens: &[usize]) -> f32 {
        // CNA is primarily a runtime modulation tool (forward hook).
        // As a ScreeningPruner, provides baseline relevance of 1.0,
        // allowing composition with BanditPruner for online refinement.
        1.0
    }
}

// ── Game Domain Contrastive Pairs ───────────────────────────────

/// A contrastive pair provider for CNA discovery.
///
/// Game domains implement this to produce positive/negative observations
/// from episode data, using their `StateHeuristic` for labeling.
pub trait ContrastivePairProvider: Send + Sync {
    /// Domain name (e.g., "bomber", "go", "fft").
    fn domain(&self) -> &str;

    /// Positive observations: (layer-like_index, feature_vector).
    /// These represent "good" states/actions from episodes.
    fn positive_observations(&self) -> Vec<(usize, Vec<f32>)>;

    /// Negative observations: (layer-like_index, feature_vector).
    /// These represent "bad" states/actions from episodes.
    fn negative_observations(&self) -> Vec<(usize, Vec<f32>)>;

    /// Number of observations in each set: (positive, negative).
    fn observation_count(&self) -> (usize, usize);
}

/// Go domain contrastive pair provider.
///
/// Uses `GoHeuristic` scores to classify moves as positive (high heuristic)
/// or negative (low heuristic). Maps board states to pseudo-activation vectors
/// for CNA discovery.
///
/// The "layer" dimension maps to board regions:
/// - Layer 0: corners (3×3 regions at each corner)
/// - Layer 1: edges (border rows/cols minus corners)
/// - Layer 2: center (inner board area)
/// - Layer 3: global (aggregate features)
#[cfg(feature = "go")]
pub mod go_pairs {
    use super::ContrastivePairProvider;
    use crate::pruners::game_state::StateHeuristic;
    use crate::pruners::go::{GoAction, GoCell, GoHeuristic, GoState};

    /// Threshold for positive moves: heuristic score above this fraction of max.
    const POSITIVE_THRESHOLD: f32 = 0.7;
    /// Threshold for negative moves: heuristic score below this fraction of max.
    const NEGATIVE_THRESHOLD: f32 = 0.3;
    /// Number of feature layers for Go board regions.
    const GO_FEATURE_LAYERS: usize = 4;

    /// Go contrastive pair provider.
    pub struct GoContrastivePairs {
        /// Collected positive observations.
        positive: Vec<(usize, Vec<f32>)>,
        /// Collected negative observations.
        negative: Vec<(usize, Vec<f32>)>,
        /// Board size.
        board_size: usize,
    }

    impl GoContrastivePairs {
        /// Create a new empty provider for the given board size.
        pub fn new(board_size: usize) -> Self {
            Self {
                positive: Vec::new(),
                negative: Vec::new(),
                board_size,
            }
        }

        /// Collect contrastive pairs from a completed game replay.
        ///
        /// Replays all moves, evaluates each position with `GoHeuristic`,
        /// and classifies as positive or negative based on threshold.
        /// Feature vector = flattened board state per region.
        pub fn collect_from_states(&mut self, states: &[(GoState, f32)]) {
            let heuristic = GoHeuristic;
            let _feature_dim = self.board_size * self.board_size;

            for (state, _move_quality) in states {
                let score = heuristic.evaluate(state, 0); // Black perspective

                // Extract board features per region layer
                let features = self.extract_features(state);

                // Classify based on heuristic score
                if score > POSITIVE_THRESHOLD {
                    for (layer, vec) in features.iter().enumerate() {
                        self.positive.push((layer, vec.clone()));
                    }
                } else if score < NEGATIVE_THRESHOLD {
                    for (layer, vec) in features.iter().enumerate() {
                        self.negative.push((layer, vec.clone()));
                    }
                }
            }
        }

        /// Extract board features as 4 pseudo-activation layers.
        fn extract_features(&self, state: &GoState) -> [Vec<f32>; GO_FEATURE_LAYERS] {
            let n = self.board_size;
            let dim = n * n;

            // Layer 0: corners (stone presence in 3x3 corner regions)
            let mut corners = vec![0.0f32; dim];
            // Layer 1: edges (border cells)
            let mut edges = vec![0.0f32; dim];
            // Layer 2: center (non-border cells)
            let mut center = vec![0.0f32; dim];
            // Layer 3: global influence (liberty count per cell)
            let mut global = vec![0.0f32; dim];

            for row in 0..n {
                for col in 0..n {
                    let idx = row * n + col;
                    let cell = state.board[idx];
                    let cell_val = match cell {
                        GoCell::Black => 1.0,
                        GoCell::White => -1.0,
                        GoCell::Empty => 0.0,
                    };

                    let is_corner = (row < 3 || row >= n - 3) && (col < 3 || col >= n - 3);
                    let is_edge = row == 0 || row == n - 1 || col == 0 || col == n - 1;

                    if is_corner {
                        corners[idx] = cell_val;
                    }
                    if is_edge {
                        edges[idx] = cell_val;
                    }
                    if !is_edge {
                        center[idx] = cell_val;
                    }
                    global[idx] = cell_val; // simplified
                }
            }

            [corners, edges, center, global]
        }

        /// Collect from a single game: list of (state, player_id, action).
        /// Classifies each move by post-move heuristic evaluation.
        pub fn collect_from_game(&mut self, moves: &[(GoState, u8, GoAction)]) {
            let heuristic = GoHeuristic;

            for (state, player_id, _action) in moves {
                let score = heuristic.evaluate(state, *player_id);
                let features = self.extract_features(state);

                if score > POSITIVE_THRESHOLD {
                    for (layer, vec) in features.iter().enumerate() {
                        self.positive.push((layer, vec.clone()));
                    }
                } else if score < NEGATIVE_THRESHOLD {
                    for (layer, vec) in features.iter().enumerate() {
                        self.negative.push((layer, vec.clone()));
                    }
                }
            }
        }
    }

    impl ContrastivePairProvider for GoContrastivePairs {
        fn domain(&self) -> &str {
            "go"
        }

        fn positive_observations(&self) -> Vec<(usize, Vec<f32>)> {
            self.positive.clone()
        }

        fn negative_observations(&self) -> Vec<(usize, Vec<f32>)> {
            self.negative.clone()
        }

        fn observation_count(&self) -> (usize, usize) {
            (self.positive.len(), self.negative.len())
        }
    }
}

/// Bomber domain contrastive pair provider.
///
/// Uses `BomberHeuristic` to classify positions as positive (safe, good moves)
/// or negative (blast zone, trapped). Maps grid states to pseudo-activation vectors.
#[cfg(feature = "bomber")]
pub mod bomber_pairs {
    use super::ContrastivePairProvider;
    use crate::pruners::bomber::{ARENA_H, ARENA_W};
    use crate::pruners::game_state::{BomberHeuristic, BomberState, StateHeuristic};

    const POSITIVE_THRESHOLD: f32 = 0.5;
    const NEGATIVE_THRESHOLD: f32 = -0.3;
    const BOMBER_FEATURE_LAYERS: usize = 3;

    /// Bomber contrastive pair provider.
    pub struct BomberContrastivePairs {
        positive: Vec<(usize, Vec<f32>)>,
        negative: Vec<(usize, Vec<f32>)>,
    }

    impl Default for BomberContrastivePairs {
        fn default() -> Self {
            Self::new()
        }
    }

    impl BomberContrastivePairs {
        /// Create a new empty provider.
        pub fn new() -> Self {
            Self {
                positive: Vec::new(),
                negative: Vec::new(),
            }
        }

        /// Collect from game states with player ID.
        pub fn collect(&mut self, states: &[(BomberState, u8)]) {
            let heuristic = BomberHeuristic;

            for (state, player_id) in states {
                let score = heuristic.evaluate(state, *player_id);
                let features = self.extract_features(state);

                if score > POSITIVE_THRESHOLD {
                    for (layer, vec) in features.iter().enumerate() {
                        self.positive.push((layer, vec.clone()));
                    }
                } else if score < NEGATIVE_THRESHOLD {
                    for (layer, vec) in features.iter().enumerate() {
                        self.negative.push((layer, vec.clone()));
                    }
                }
            }
        }

        /// Extract features as 3 pseudo-activation layers.
        fn extract_features(&self, state: &BomberState) -> [Vec<f32>; BOMBER_FEATURE_LAYERS] {
            let dim = ARENA_W * ARENA_H;

            // Layer 0: cell types
            let mut cells = vec![0.0f32; dim];
            // Layer 1: player positions
            let mut players = vec![0.0f32; dim];
            // Layer 2: blast zones
            let mut blast = vec![0.0f32; dim];

            for r in 0..ARENA_H {
                for c in 0..ARENA_W {
                    let idx = r * ARENA_W + c;
                    // Cell type encoding
                    cells[idx] = match state.cells[idx] {
                        crate::pruners::bomber::Cell::Floor => 0.0,
                        crate::pruners::bomber::Cell::FixedWall => -1.0,
                        crate::pruners::bomber::Cell::DestructibleWall => -0.5,
                        crate::pruners::bomber::Cell::PowerUpHidden(_) => 0.5,
                    };
                    // Blast zone
                    if state.is_in_blast_zone((c as i32, r as i32)) {
                        blast[idx] = -1.0;
                    }
                }
            }

            // Player positions
            for (i, player) in state.players.iter().enumerate() {
                if player.alive {
                    let (px, py) = (player.pos.0 as usize, player.pos.1 as usize);
                    if py < ARENA_H && px < ARENA_W {
                        let idx = py * ARENA_W + px;
                        players[idx] = if i == 0 { 1.0 } else { -0.5 };
                    }
                }
            }

            [cells, players, blast]
        }
    }

    impl ContrastivePairProvider for BomberContrastivePairs {
        fn domain(&self) -> &str {
            "bomber"
        }

        fn positive_observations(&self) -> Vec<(usize, Vec<f32>)> {
            self.positive.clone()
        }

        fn negative_observations(&self) -> Vec<(usize, Vec<f32>)> {
            self.negative.clone()
        }

        fn observation_count(&self) -> (usize, usize) {
            (self.positive.len(), self.negative.len())
        }
    }
}

/// FFT Tactics domain contrastive pair provider.
///
/// Classifies battle actions as positive (high heuristic: kills, heals low-HP)
/// or negative (low heuristic: waste, wait).
#[cfg(feature = "fft")]
pub mod fft_pairs {
    use super::ContrastivePairProvider;
    use crate::pruners::fft::battle::BattleState;
    use crate::pruners::fft::types::{ActionType, Unit};

    const FFT_FEATURE_DIM: usize = 64; // Fixed feature vector size
    const FFT_FEATURE_LAYERS: usize = 2;

    /// FFT contrastive pair provider.
    pub struct FftContrastivePairs {
        positive: Vec<(usize, Vec<f32>)>,
        negative: Vec<(usize, Vec<f32>)>,
    }

    impl Default for FftContrastivePairs {
        fn default() -> Self {
            Self::new()
        }
    }

    impl FftContrastivePairs {
        /// Create a new empty provider.
        pub fn new() -> Self {
            Self {
                positive: Vec::new(),
                negative: Vec::new(),
            }
        }

        /// Score an action heuristically (mirrors g_zero_player heuristic).
        fn heuristic_score(action: ActionType, unit: &Unit, state: &BattleState) -> f32 {
            let hp_pct = unit.hp_pct();
            let can_cast = crate::pruners::fft::status::can_cast(unit, &state.effects);
            let enemies: Vec<u8> = state
                .units
                .iter()
                .enumerate()
                .filter(|(_, u)| u.team != unit.team && u.alive)
                .map(|(i, _)| i as u8)
                .collect();
            let allies: Vec<u8> = state
                .units
                .iter()
                .enumerate()
                .filter(|(_, u)| u.team == unit.team && u.alive)
                .map(|(i, _)| i as u8)
                .collect();

            match action {
                ActionType::Attack if !enemies.is_empty() => 2.0,
                ActionType::Defend => 1.0,
                ActionType::BlackMagic
                    if !enemies.is_empty() && can_cast && unit.can_afford(action) =>
                {
                    2.5
                }
                ActionType::WhiteMagic
                    if !allies.is_empty() && can_cast && unit.can_afford(action) =>
                {
                    let wounded = allies
                        .iter()
                        .any(|&a| state.units[a as usize].hp_pct() < 0.7);
                    if wounded { 3.0 } else { 0.5 }
                }
                ActionType::Potion if hp_pct < 0.5 && unit.can_afford(action) => 3.0,
                ActionType::Wait => 0.0,
                _ => f32::NEG_INFINITY,
            }
        }

        /// Collect from battle state observations.
        pub fn collect(&mut self, observations: &[(BattleState, u8, ActionType)]) {
            for (state, unit_id, action) in observations {
                let unit = &state.units[*unit_id as usize];
                let score = Self::heuristic_score(*action, unit, state);
                let features = Self::extract_features(state, *unit_id);

                if score >= 2.0 {
                    for (layer, vec) in features.iter().enumerate() {
                        self.positive.push((layer, vec.clone()));
                    }
                } else if score <= 0.5 {
                    for (layer, vec) in features.iter().enumerate() {
                        self.negative.push((layer, vec.clone()));
                    }
                }
            }
        }

        /// Extract features as 2 pseudo-activation layers.
        fn extract_features(state: &BattleState, unit_id: u8) -> [Vec<f32>; FFT_FEATURE_LAYERS] {
            let mut unit_features = vec![0.0f32; FFT_FEATURE_DIM];
            let mut battle_features = vec![0.0f32; FFT_FEATURE_DIM];

            // Layer 0: unit features (hp, mp, stats)
            let unit = &state.units[unit_id as usize];
            if FFT_FEATURE_DIM >= 8 {
                unit_features[0] = unit.hp as f32 / unit.stats.max_hp as f32;
                unit_features[1] = unit.mp as f32 / unit.stats.max_mp as f32;
                unit_features[2] = unit.stats.atk as f32 / 20.0;
                unit_features[3] = unit.stats.def as f32 / 20.0;
                unit_features[4] = unit.stats.mag as f32 / 20.0;
                unit_features[5] = unit.stats.speed as f32 / 10.0;
                unit_features[6] = if unit.defending { 1.0 } else { 0.0 };
                unit_features[7] = unit.pos.x as f32 / 8.0 + unit.pos.y as f32 / 8.0;
            }

            // Layer 1: battle context (team HP, enemy HP, tick)
            let alive_allies = state
                .units
                .iter()
                .filter(|u| u.team == unit.team && u.alive)
                .count();
            let alive_enemies = state
                .units
                .iter()
                .filter(|u| u.team != unit.team && u.alive)
                .count();
            let total_ally_hp: f32 = state
                .units
                .iter()
                .filter(|u| u.team == unit.team && u.alive)
                .map(|u| u.hp as f32 / u.stats.max_hp as f32)
                .sum();
            let total_enemy_hp: f32 = state
                .units
                .iter()
                .filter(|u| u.team != unit.team && u.alive)
                .map(|u| u.hp as f32 / u.stats.max_hp as f32)
                .sum();

            if FFT_FEATURE_DIM >= 8 {
                battle_features[0] = alive_allies as f32;
                battle_features[1] = alive_enemies as f32;
                battle_features[2] = total_ally_hp;
                battle_features[3] = total_enemy_hp;
                battle_features[4] = state.tick as f32 / 200.0;
                battle_features[5] = state.effects.len() as f32 / 20.0;
                battle_features[6] = state.events.len() as f32 / 100.0;
                battle_features[7] =
                    (alive_allies as f32) / (alive_allies + alive_enemies).max(1) as f32;
            }

            [unit_features, battle_features]
        }
    }

    impl ContrastivePairProvider for FftContrastivePairs {
        fn domain(&self) -> &str {
            "fft"
        }

        fn positive_observations(&self) -> Vec<(usize, Vec<f32>)> {
            self.positive.clone()
        }

        fn negative_observations(&self) -> Vec<(usize, Vec<f32>)> {
            self.negative.clone()
        }

        fn observation_count(&self) -> (usize, usize) {
            (self.positive.len(), self.negative.len())
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_layer_index(neurons: &[CnaNeuron]) -> HashMap<usize, Vec<usize>> {
        let mut idx: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, n) in neurons.iter().enumerate() {
            idx.entry(n.layer).or_default().push(i);
        }
        idx
    }

    fn make_neuron(layer: usize, index: usize, delta: f32) -> CnaNeuron {
        CnaNeuron {
            layer,
            index,
            delta,
        }
    }

    #[test]
    fn test_circuit_construction() {
        let neurons = vec![
            make_neuron(2, 5, 0.9),
            make_neuron(1, 3, 0.7),
            make_neuron(0, 10, 0.3),
        ];
        let neuron_set: HashSet<(usize, usize)> =
            neurons.iter().map(|n| (n.layer, n.index)).collect();
        let circuit = CnaCircuit {
            neurons,
            neuron_set,
            layer_index: build_layer_index(&[
                make_neuron(2, 5, 0.9),
                make_neuron(1, 3, 0.7),
                make_neuron(0, 10, 0.3),
            ]),
            universal_excluded: vec![(0, 1)],
            universal_excluded_set: HashSet::from_iter([(0, 1)]),
            n_positive: 5,
            n_negative: 5,
            total_mlp_activations: 6 * 64,
        };

        // Neurons should be sorted by |delta| descending (done manually here).
        assert_eq!(circuit.neurons[0].delta, 0.9);
        assert_eq!(circuit.neurons[1].delta, 0.7);
        assert_eq!(circuit.neurons[2].delta, 0.3);
        assert_eq!(circuit.n_positive, 5);
        assert_eq!(circuit.n_negative, 5);
    }

    #[test]
    fn test_discovery_basic() {
        // 2 layers, mlp_hidden = 4
        let n_layers = 2;
        let mlp_hidden = 4;
        let config = CnaDiscoveryConfig {
            top_pct: 0.25, // top 25% = 2 neurons out of 8
            ..Default::default()
        };

        // Positive: neuron (1, 2) has high activation, (0, 0) moderate
        let pos_acts: Vec<(usize, &[f32])> =
            vec![(0, &[1.0, 0.0, 0.0, 0.0]), (1, &[0.0, 0.0, 5.0, 0.0])];

        // Negative: neuron (1, 2) has low activation
        let neg_acts: Vec<(usize, &[f32])> =
            vec![(0, &[0.0, 0.0, 0.0, 0.0]), (1, &[0.0, 0.0, 0.5, 0.0])];

        let circuit = cna_discover(&pos_acts, &neg_acts, n_layers, mlp_hidden, &config);

        // Top neuron should be (1, 2) with delta ≈ 4.5
        assert!(!circuit.neurons.is_empty());
        assert_eq!(circuit.neurons[0].layer, 1);
        assert_eq!(circuit.neurons[0].index, 2);
        assert!((circuit.neurons[0].delta - 4.5).abs() < 0.01);
    }

    #[test]
    fn test_modulate_baseline() {
        let neurons = vec![make_neuron(0, 0, 1.0)];
        let circuit = CnaCircuit {
            neuron_set: neurons.iter().map(|n| (n.layer, n.index)).collect(),
            layer_index: build_layer_index(&neurons),
            neurons,
            universal_excluded: vec![],
            universal_excluded_set: HashSet::new(),
            n_positive: 1,
            n_negative: 1,
            total_mlp_activations: 4,
        };
        let modulator = CnaModulator {
            circuit,
            multiplier: 1.0,
        };

        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        cna_modulate(&mut hidden, 0, &modulator);

        // Baseline multiplier = 1.0 → no change.
        assert_eq!(hidden, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_modulate_ablate() {
        let neurons = vec![make_neuron(0, 0, 1.0), make_neuron(0, 2, 0.5)];
        let circuit = CnaCircuit {
            neuron_set: neurons.iter().map(|n| (n.layer, n.index)).collect(),
            layer_index: build_layer_index(&neurons),
            neurons,
            universal_excluded: vec![],
            universal_excluded_set: HashSet::new(),
            n_positive: 1,
            n_negative: 1,
            total_mlp_activations: 4,
        };
        let modulator = CnaModulator {
            circuit,
            multiplier: 0.0,
        };

        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        cna_modulate(&mut hidden, 0, &modulator);

        // Indices 0 and 2 should be zeroed; 1 and 3 untouched.
        assert_eq!(hidden, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_modulate_amplify() {
        let neurons = vec![make_neuron(0, 1, 0.8)];
        let circuit = CnaCircuit {
            neuron_set: neurons.iter().map(|n| (n.layer, n.index)).collect(),
            layer_index: build_layer_index(&neurons),
            neurons,
            universal_excluded: vec![],
            universal_excluded_set: HashSet::new(),
            n_positive: 1,
            n_negative: 1,
            total_mlp_activations: 4,
        };
        let modulator = CnaModulator {
            circuit,
            multiplier: 2.0,
        };

        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        cna_modulate(&mut hidden, 0, &modulator);

        // Only index 1 should be doubled.
        assert_eq!(hidden, vec![1.0, 4.0, 3.0, 4.0]);
    }

    #[test]
    fn test_discovery_config_default() {
        let config = CnaDiscoveryConfig::default();
        assert!((config.top_pct - 0.001).abs() < f32::EPSILON);
        assert!((config.universal_threshold - 0.8).abs() < f32::EPSILON);
        assert!((config.late_layer_fraction - 0.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_universal_detection() {
        let n_layers = 1;
        let mlp_hidden = 10;

        // 5 diverse prompts. Neuron (0, 0) is always in top activations → universal.
        let diverse: Vec<Vec<(usize, Vec<f32>)>> = (0..5)
            .map(|_| vec![(0, vec![10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])])
            .collect();

        let universal = detect_universal_neurons(&diverse, n_layers, mlp_hidden, 0.8);

        // Neuron (0, 0) should be detected as universal.
        assert!(universal.contains(&(0, 0)));
    }

    #[test]
    fn test_universal_detection_empty() {
        let universal = detect_universal_neurons(&[], 1, 10, 0.8);
        assert!(universal.is_empty());
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = CnaCircuit {
            neurons: vec![],
            neuron_set: HashSet::new(),
            layer_index: HashMap::new(),
            universal_excluded: vec![],
            universal_excluded_set: HashSet::new(),
            n_positive: 0,
            n_negative: 0,
            total_mlp_activations: 4,
        };
        let modulator = CnaModulator {
            circuit,
            multiplier: 0.0,
        };

        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        cna_modulate(&mut hidden, 0, &modulator);

        // Empty circuit → nothing to modulate → no change.
        assert_eq!(hidden, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sparse_selection() {
        let n_layers = 1;
        let mlp_hidden = 100;
        let config = CnaDiscoveryConfig {
            top_pct: 0.01, // top 1% = 1 neuron
            ..Default::default()
        };

        // Positive: index 50 has the highest activation.
        let pos_hidden: Vec<f32> = (0..100).map(|i| if i == 50 { 10.0 } else { 0.1 }).collect();
        let neg_hidden: Vec<f32> = vec![0.1; 100];

        let pos_acts: Vec<(usize, &[f32])> = vec![(0, &pos_hidden)];
        let neg_acts: Vec<(usize, &[f32])> = vec![(0, &neg_hidden)];

        let circuit = cna_discover(&pos_acts, &neg_acts, n_layers, mlp_hidden, &config);

        // Only 1 neuron selected, and it should be (0, 50).
        assert_eq!(circuit.neurons.len(), 1);
        assert_eq!(circuit.neurons[0].layer, 0);
        assert_eq!(circuit.neurons[0].index, 50);
    }

    #[test]
    fn test_screening_pruner_relevance() {
        let neurons = vec![make_neuron(0, 0, 1.0)];
        let circuit = CnaCircuit {
            neuron_set: neurons.iter().map(|n| (n.layer, n.index)).collect(),
            layer_index: build_layer_index(&neurons),
            neurons,
            universal_excluded: vec![],
            universal_excluded_set: HashSet::new(),
            n_positive: 1,
            n_negative: 1,
            total_mlp_activations: 4,
        };
        let pruner = CnaScreeningPruner::new(circuit);

        // Baseline relevance is always 1.0.
        assert!((pruner.relevance(0, 0, &[]) - 1.0).abs() < f32::EPSILON);
        assert!((pruner.relevance(5, 100, &[1, 2, 3]) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_modulate_different_layer() {
        let neurons = vec![make_neuron(1, 0, 1.0)]; // layer 1
        let circuit = CnaCircuit {
            neuron_set: neurons.iter().map(|n| (n.layer, n.index)).collect(),
            layer_index: build_layer_index(&neurons),
            neurons,
            universal_excluded: vec![],
            universal_excluded_set: HashSet::new(),
            n_positive: 1,
            n_negative: 1,
            total_mlp_activations: 8,
        };
        let modulator = CnaModulator {
            circuit,
            multiplier: 0.0,
        };

        // Modulating layer 0 should not affect anything (circuit targets layer 1).
        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        cna_modulate(&mut hidden, 0, &modulator);
        assert_eq!(hidden, vec![1.0, 2.0, 3.0, 4.0]);

        // Modulating layer 1 should ablate index 0.
        let mut hidden = vec![1.0, 2.0, 3.0, 4.0];
        cna_modulate(&mut hidden, 1, &modulator);
        assert_eq!(hidden, vec![0.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_discovery_multiple_observations() {
        // Multiple positive and negative observations should average correctly.
        let pos_acts: Vec<(usize, &[f32])> = vec![(0, &[2.0, 0.0]), (0, &[4.0, 0.0])];
        let neg_acts: Vec<(usize, &[f32])> = vec![(0, &[0.0, 0.0]), (0, &[0.0, 0.0])];

        let config = CnaDiscoveryConfig {
            top_pct: 0.5,
            ..Default::default()
        };

        let circuit = cna_discover(&pos_acts, &neg_acts, 1, 2, &config);

        // Mean positive for index 0 = 3.0, mean negative = 0.0, delta = 3.0.
        assert_eq!(circuit.neurons[0].layer, 0);
        assert_eq!(circuit.neurons[0].index, 0);
        assert!((circuit.neurons[0].delta - 3.0).abs() < 0.01);
    }

    // ── Contrastive Pair Tests ─────────────────────────────────

    #[test]
    fn test_contrastive_pair_trait_bounds() {
        // Verify ContrastivePairProvider is Send + Sync
        #[allow(dead_code)]
        fn assert_send_sync<T: Send + Sync>() {}
        // The trait requires Send + Sync, so this is compile-time verified
        // by the trait definition itself
    }
}
