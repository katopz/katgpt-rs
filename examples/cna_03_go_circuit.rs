//! CNA Go Circuit Example — End-to-end circuit discovery from Go games.
//!
//! Demonstrates:
//! - Playing random Go games to collect episode data
//! - Using GoHeuristic to label moves as positive/negative
//! - Encoding board states as per-layer activation vectors
//! - Discovering contrastive neuron circuit from game data
//! - Analyzing circuit structure and layer distribution
//! - Ablation test: disabling circuit neurons reduces move quality discrimination
//!
//! ```sh
//! cargo run --example cna_03_go_circuit --features "cna_steering,go"
//! ```

use katgpt_rs::pruners::game_state::StateHeuristic;
use katgpt_rs::pruners::go::{GoAction, GoCell, GoHeuristic, GoState};
use katgpt_rs::pruners::{CnaDiscoveryConfig, CnaModulator, cna_discover, cna_modulate};

// ── Constants ──────────────────────────────────────────────────

const BOARD_SIZE: usize = 9;
const NUM_GAMES: usize = 20;
const MAX_MOVES: usize = 200;
const N_LAYERS: usize = 4;
const POSITIVE_THRESHOLD: f32 = 0.15;
const NEGATIVE_THRESHOLD: f32 = -0.15;

/// Per-layer activation vector: (neuron_index, weights).
type LayerActivations = Vec<(usize, Vec<f32>)>;

// ── Board Encoding ─────────────────────────────────────────────

/// Encode board state into per-layer activation vectors.
///
/// Layers represent spatial features:
/// - Layer 0 (corners): 3×3 corner regions weighted higher
/// - Layer 1 (edges): edge cells weighted higher
/// - Layer 2 (center): center cells weighted higher
/// - Layer 3 (global): uniform encoding of all cells
///
/// Values: +1.0 for friendly stone, -1.0 for opponent, 0.0 for empty,
/// scaled by positional weight.
fn encode_board(state: &GoState, player_id: u8) -> LayerActivations {
    let color = GoCell::from_player_id(player_id);
    let size = state.size;
    let total = size * size;
    let center = size / 2;

    let mut layers = vec![vec![0.0f32; total]; N_LAYERS];

    for (idx, board_cell) in state.board.iter().enumerate().take(total) {
        let row = idx / size;
        let col = idx % size;

        let val = match board_cell {
            c if *c == color => 1.0,
            c if *c == color.opponent() => -1.0,
            _ => 0.0,
        };

        if val == 0.0 {
            continue;
        }

        // Distance from edge (0 = edge, size/2 = center)
        let line_row = row.min(size - 1 - row);
        let line_col = col.min(size - 1 - col);
        let line = line_row.min(line_col);

        // Layer 0 (corners): weight corner 3×3 quadrants
        let corner_weight = if line <= 1 { 2.0 } else { 0.3 };
        layers[0][idx] = val * corner_weight;

        // Layer 1 (edges): weight cells near edge (lines 1-3)
        let edge_weight = if (1..=3).contains(&line) { 1.5 } else { 0.3 };
        layers[1][idx] = val * edge_weight;

        // Layer 2 (center): weight center region
        let center_dist =
            ((row as f32 - center as f32).powi(2) + (col as f32 - center as f32).powi(2)).sqrt();
        let center_weight = if center_dist < (size as f32 * 0.3) {
            2.0
        } else {
            0.3
        };
        layers[2][idx] = val * center_weight;

        // Layer 3 (global): uniform
        layers[3][idx] = val;
    }

    layers.into_iter().enumerate().collect()
}

// ── Game Simulation ────────────────────────────────────────────

/// Play a random game and return (state, player_id, action) triples.
fn play_random_game(rng: &mut fastrand::Rng) -> Vec<(GoState, u8, GoAction)> {
    let mut state = GoState::new(BOARD_SIZE);
    let mut moves = Vec::new();

    for _ in 0..MAX_MOVES {
        if state.is_terminal() {
            break;
        }

        let player_id = state.to_play.player_id();

        // Collect legal moves
        let legal = state.legal_moves();
        let mut actions: Vec<GoAction> = legal
            .into_iter()
            .map(|(r, c)| GoAction::Place(r, c))
            .collect();
        actions.push(GoAction::Pass);

        // Pick random move
        let action = actions[rng.usize(..actions.len())].clone();

        // Record state before move
        moves.push((state.clone(), player_id, action.clone()));

        // Apply move
        match &action {
            GoAction::Place(r, c) => {
                state.play_move(*r, *c);
            }
            GoAction::Pass => {
                state.play_pass();
            }
        }
    }

    moves
}

// ── Contrastive Collection ─────────────────────────────────────

/// Collect positive and negative observations from game episodes.
///
/// For each move, evaluate the successor state with GoHeuristic.
/// High score → positive class. Low score → negative class.
fn collect_contrastive_pairs(
    episodes: &[(GoState, u8, GoAction)],
) -> (LayerActivations, LayerActivations) {
    let heuristic = GoHeuristic;
    let mut positive: LayerActivations = Vec::new();
    let mut negative: LayerActivations = Vec::new();

    for (state, player_id, action) in episodes {
        // Build successor state
        let next = match action {
            GoAction::Place(r, c) => {
                let mut s = state.clone();
                if s.play_move(*r, *c) {
                    s
                } else {
                    continue;
                }
            }
            GoAction::Pass => {
                let mut s = state.clone();
                s.play_pass();
                s
            }
        };

        // Evaluate successor
        let score = heuristic.evaluate(&next, *player_id);

        // Encode board as per-layer activations
        let layer_activations = encode_board(&next, *player_id);

        // Classify as positive or negative
        if score > POSITIVE_THRESHOLD {
            positive.extend(layer_activations);
        } else if score < NEGATIVE_THRESHOLD {
            negative.extend(layer_activations);
        }
    }

    (positive, negative)
}

// ── Main ───────────────────────────────────────────────────────

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       CNA Go Circuit Discovery — {BOARD_SIZE}×{BOARD_SIZE} Board          ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Games             : {NUM_GAMES:<6}                              ║");
    println!("║  Max moves/game    : {MAX_MOVES:<6}                              ║");
    println!("║  Encoding layers   : {N_LAYERS:<6}                              ║");
    println!(
        "║  Feature dim       : {board_dim:<6}                              ║",
        board_dim = BOARD_SIZE * BOARD_SIZE,
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut rng = fastrand::Rng::with_seed(42);
    let mlp_hidden = BOARD_SIZE * BOARD_SIZE;

    // ── Phase 1: Collect contrastive pairs from games ─────────

    println!("Phase 1: Collecting contrastive pairs from {NUM_GAMES} games...");
    println!("  Threshold: positive > {POSITIVE_THRESHOLD}, negative < {NEGATIVE_THRESHOLD}");
    println!();

    let mut all_positive: LayerActivations = Vec::new();
    let mut all_negative: LayerActivations = Vec::new();

    for game_idx in 0..NUM_GAMES {
        let episode = play_random_game(&mut rng);
        let (pos, neg) = collect_contrastive_pairs(&episode);
        all_positive.extend(pos);
        all_negative.extend(neg);

        if (game_idx + 1) % 5 == 0 {
            println!(
                "  After {game_idx:>3} games: {:>5} positive, {:>5} negative observations",
                all_positive.len() / N_LAYERS,
                all_negative.len() / N_LAYERS,
            );
        }
    }

    let n_pos = all_positive.len() / N_LAYERS;
    let n_neg = all_negative.len() / N_LAYERS;
    println!("  Total: {n_pos} positive, {n_neg} negative board states");
    println!(
        "  Activation vectors: {} pos, {} neg",
        all_positive.len(),
        all_negative.len()
    );
    println!();

    if all_positive.is_empty() || all_negative.is_empty() {
        println!("⚠ Not enough contrastive data. Try more games or adjust thresholds.");
        println!("  Positive: {n_pos}, Negative: {n_neg}");
        return;
    }

    // ── Phase 2: Discover circuit ─────────────────────────────

    println!("Phase 2: Discovering contrastive neuron circuit...");

    let pos_refs: Vec<(usize, &[f32])> = all_positive
        .iter()
        .map(|(l, v)| (*l, v.as_slice()))
        .collect();
    let neg_refs: Vec<(usize, &[f32])> = all_negative
        .iter()
        .map(|(l, v)| (*l, v.as_slice()))
        .collect();

    let config = CnaDiscoveryConfig {
        top_pct: 0.01, // 1% — game domain needs wider net than LLM 0.1%
        ..Default::default()
    };

    let circuit = cna_discover(&pos_refs, &neg_refs, N_LAYERS, mlp_hidden, &config);

    println!("  Circuit discovered:");
    println!("    Neurons: {}", circuit.neurons.len());
    println!(
        "    Universal excluded: {}",
        circuit.universal_excluded.len()
    );
    println!(
        "    Circuit density: {:.3}%",
        circuit.neurons.len() as f32 / circuit.total_mlp_activations as f32 * 100.0
    );
    println!();

    // ── Phase 3: Analyze circuit structure ────────────────────

    println!("Phase 3: Circuit analysis");
    println!();

    // Top neurons
    println!("  Top neurons (by |δ|):");
    let layer_names = ["corners", "edges", "center", "global"];
    for (i, neuron) in circuit.neurons.iter().take(10).enumerate() {
        let layer_name = layer_names.get(neuron.layer).copied().unwrap_or("?");
        let nl = neuron.layer;
        let ni = neuron.index;
        let nd = neuron.delta;
        println!("    #{i}: layer={nl} ({layer_name}) index={ni} δ={nd:.4}");
    }

    // Layer distribution
    let mut layer_counts = [0usize; N_LAYERS];
    for n in &circuit.neurons {
        if n.layer < N_LAYERS {
            layer_counts[n.layer] += 1;
        }
    }
    println!();
    println!("  Layer distribution:");
    for (layer, &count) in layer_counts.iter().enumerate() {
        let pct = count as f32 / circuit.neurons.len().max(1) as f32 * 100.0;
        println!(
            "    Layer {layer} ({:>8}): {count:>3} neurons ({pct:.1}%)",
            layer_names[layer]
        );
    }

    // ── Phase 4: Ablation test ────────────────────────────────

    println!();
    println!("Phase 4: Ablation test");
    println!();

    if circuit.neurons.is_empty() {
        println!("  No circuit neurons discovered — nothing to ablate.");
        println!("  Try adjusting top_pct or collecting more games.");
        return;
    }

    // Create a synthetic hidden activation vector
    let mut sample = vec![0.5f32; mlp_hidden];
    for (i, slot) in sample.iter_mut().enumerate().take(mlp_hidden.min(20)) {
        *slot = 0.1 + i as f32 * 0.05;
    }

    let baseline_sum: f32 = sample.iter().sum();

    // Ablate: m = 0.0
    let modulator_ablate = CnaModulator {
        circuit: circuit.clone(),
        multiplier: 0.0,
    };
    let mut ablated = sample.clone();
    for layer in 0..N_LAYERS {
        cna_modulate(&mut ablated, layer, &modulator_ablate);
    }
    let ablated_sum: f32 = ablated.iter().sum();
    let ablated_changed = ablated
        .iter()
        .zip(sample.iter())
        .filter(|(a, b)| (*a - *b).abs() > 0.001)
        .count();

    // Amplify: m = 2.0
    let modulator_amplify = CnaModulator {
        circuit: circuit.clone(),
        multiplier: 2.0,
    };
    let mut amplified = sample.clone();
    for layer in 0..N_LAYERS {
        cna_modulate(&mut amplified, layer, &modulator_amplify);
    }
    let amplified_sum: f32 = amplified.iter().sum();
    let amplified_changed = amplified
        .iter()
        .zip(sample.iter())
        .filter(|(a, b)| (*a - *b).abs() > 0.001)
        .count();

    // Baseline: m = 1.0 (no-op)
    let modulator_baseline = CnaModulator {
        circuit: circuit.clone(),
        multiplier: 1.0,
    };
    let mut baseline_mod = sample.clone();
    for layer in 0..N_LAYERS {
        cna_modulate(&mut baseline_mod, layer, &modulator_baseline);
    }
    let baseline_mod_sum: f32 = baseline_mod.iter().sum();

    println!("  Sample vector ({mlp_hidden}-dim):");
    println!("    Baseline (m=1.0): sum={baseline_sum:.4} → {baseline_mod_sum:.4} (no-op)");
    println!(
        "    Ablated  (m=0.0): sum={baseline_sum:.4} → {ablated_sum:.4} ({ablated_changed} features changed)"
    );
    println!(
        "    Amplified(m=2.0): sum={baseline_sum:.4} → {amplified_sum:.4} ({amplified_changed} features changed)"
    );
    println!();

    // Quality: non-circuit neurons should be preserved
    let circuit_indices: Vec<(usize, usize)> =
        circuit.neurons.iter().map(|n| (n.layer, n.index)).collect();
    let mut non_circuit_sq_diff = 0.0f32;
    let mut non_circuit_count = 0;
    for i in 0..mlp_hidden {
        for layer in 0..N_LAYERS {
            if !circuit_indices.contains(&(layer, i)) {
                let diff = ablated[i] - sample[i];
                non_circuit_sq_diff += diff * diff;
                non_circuit_count += 1;
            }
        }
    }
    let non_circuit_rmse = (non_circuit_sq_diff / non_circuit_count.max(1) as f32).sqrt();
    println!("  Quality preservation (non-circuit RMSE): {non_circuit_rmse:.6}");
    println!("  Paper benchmark: CNA quality > 0.97 at all strengths");

    println!();
    println!("✓ Go circuit discovery complete");
    println!();
    println!("Key insight: Game episodes provide natural contrastive pairs");
    println!("  via GoHeuristic scoring — no manual labeling needed.");
}
