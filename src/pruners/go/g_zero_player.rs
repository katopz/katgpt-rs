//! G-Zero self-play for Go — template-based self-play with HintDelta and absorb-compress.
//!
//! Plan 065 Phase 4 (Tasks T32-T36):
//! - T32: `GoSelfPlayResult` — per-episode result tracking
//! - T33: `GoTemplateProposer` — UCB1 template selection with 4 Go templates
//! - T34: HintDelta computation — modelless δ = score(template_move) - score(best_non_template)
//! - T35: `GoDeltaGatedAbsorbCompress` — promote high-δ templates to hard constraints
//! - T36: `run_gzero_selfplay` — main self-play loop
//
//!   Adapted from the general G-Zero TemplateProposer pattern but Go-specific:
//! - Query = board state as flat token array (0=empty, 1=black, 2=white)
//! - Hint = template-suggested move coordinates
//! - δ = score difference between template-proposed and best non-template move

use std::io::Write;
use std::time::Instant;

use super::players::GoTemplate;
use super::replay::GoReplay;
use super::state::GoState;
use super::types::{GoAction, GoCell};
use crate::pruners::game_state::GameState;

// ── Constants ──────────────────────────────────────────────────

/// Number of Go templates (CornerStar, Capture, Defend, Tenuki).
const NUM_TEMPLATES: usize = 4;

/// Max moves = board_size² × this factor (prevents infinite games).
const MAX_MOVES_FACTOR: usize = 3;

/// UCB1 exploration constant.
const UCB1_C: f32 = 2.0;

// ── MoveDelta (T34) ────────────────────────────────────────────

/// Delta recorded for a single move within an episode.
///
/// Modelless approximation: δ = score(template_move) - max(score(non_template_moves)).
/// Positive δ means the template identified a move better than any non-matching move.
/// Negative δ means the template's suggestion was worse than what was available elsewhere.
/// Zero δ means the template's move was the best available (tie with non-template best).
#[derive(Clone, Debug)]
pub struct MoveDelta {
    /// Move number (1-based).
    pub move_num: usize,
    /// Template used for this move.
    pub template: GoTemplate,
    /// Board-state token representation (flat array of i8: 0=empty, 1=black, 2=white).
    pub board_tokens: Vec<i8>,
    /// Suggested move from the template.
    pub suggested_move: (usize, usize),
    /// Greedy score of suggested move.
    pub hinted_score: f32,
    /// Best greedy score among legal moves NOT matching this template.
    pub best_unhinted_score: f32,
    /// δ = hinted_score - best_unhinted_score.
    pub delta: f32,
}

// ── GoSelfPlayResult (T32) ─────────────────────────────────────

/// Result of a single G-Zero self-play episode.
#[derive(Clone, Debug)]
pub struct GoSelfPlayResult {
    /// Episode number (1-based).
    pub episode: usize,
    /// Winner of the game (None for draw).
    pub winner: Option<GoCell>,
    /// Total moves played.
    pub total_moves: usize,
    /// Per-move delta values for the template that was active.
    pub move_deltas: Vec<MoveDelta>,
    /// Game duration.
    pub duration: std::time::Duration,
}

// ── Local Helpers ──────────────────────────────────────────────
// Issue 001 H-20: `board_neighbors` and `flood_group` were copy-pasted across
// players.rs, g_zero_player.rs, and autoresearch.rs. Now imported from
// `go::utils` so all three call sites share one implementation.
use super::utils::{board_neighbors, flood_group};

/// Stones captured by `me` between two states (before → after).
fn captures_for(me: GoCell, before: &GoState, after: &GoState) -> u32 {
    match me {
        GoCell::Black => after.captured_black.saturating_sub(before.captured_black),
        GoCell::White => after.captured_white.saturating_sub(before.captured_white),
        GoCell::Empty => 0,
    }
}

/// True if (row, col) is a corner star point for the given board size.
fn is_star_point(row: usize, col: usize, size: usize) -> bool {
    match size {
        9 => matches!((row, col), (2, 2) | (2, 6) | (4, 4) | (6, 2) | (6, 6)),
        13 => matches!(
            (row, col),
            (3, 3) | (3, 6) | (3, 9) | (6, 3) | (6, 6) | (6, 9) | (9, 3) | (9, 6) | (9, 9)
        ),
        19 => matches!(
            (row, col),
            (3, 3) | (3, 9) | (3, 15) | (9, 3) | (9, 9) | (9, 15) | (15, 3) | (15, 9) | (15, 15)
        ),
        _ => false,
    }
}

/// True if on first line (edge) of the board.
fn is_first_line(row: usize, col: usize, size: usize) -> bool {
    row == 0 || row == size - 1 || col == 0 || col == size - 1
}

/// Center proximity bonus: 1.0 at center, 0.0 at corners.
fn center_bonus(row: usize, col: usize, size: usize) -> f32 {
    let center = (size - 1) as f32 / 2.0;
    let max_dist = center;
    if max_dist == 0.0 {
        return 1.0;
    }
    let dist = ((row as f32 - center).abs() + (col as f32 - center).abs()) / 2.0;
    1.0 - dist / max_dist
}

/// Check if move at `idx` is adjacent to an own group with ≤ 2 liberties (defend).
fn is_defend_move(state: &GoState, idx: usize) -> bool {
    let me = state.to_play;
    for n in board_neighbors(idx, state.size) {
        if state.board[n] == me {
            let (_, libs) = flood_group(&state.board, n, state.size);
            if libs.len() <= 2 {
                return true;
            }
        }
    }
    false
}

/// Check if placing at (row, col) captures opponent stones.
fn is_capture_move(state: &GoState, row: usize, col: usize) -> bool {
    let me = state.to_play;
    let action = GoAction::Place(row, col);
    let new_state = state.advance(&action, me.player_id());
    captures_for(me, state, &new_state) > 0
}

/// Check if a move matches the given template.
fn matches_template(
    template: GoTemplate,
    state: &GoState,
    row: usize,
    col: usize,
    last_move: Option<(usize, usize)>,
) -> bool {
    let size = state.size;
    let idx = state.flat_index(row, col);
    match template {
        GoTemplate::CornerStar => is_star_point(row, col, size),
        GoTemplate::Capture => is_capture_move(state, row, col),
        GoTemplate::Defend => is_defend_move(state, idx),
        GoTemplate::Tenuki => match last_move {
            Some((lr, lc)) => {
                let dist =
                    ((row as i32 - lr as i32).abs() + (col as i32 - lc as i32).abs()) as usize;
                dist > size / 3
            }
            None => true,
        },
    }
}

/// Greedy move score: captures, liberties, atari threats, center, edge penalty, self-atari.
///
/// Re-implemented locally because `players::greedy_score` is private.
fn compute_move_score(state: &GoState, row: usize, col: usize) -> f32 {
    let me = state.to_play;
    let opp = me.opponent();
    let size = state.size;
    let idx = state.flat_index(row, col);

    let action = GoAction::Place(row, col);
    let new_state = state.advance(&action, me.player_id());

    // 1. Capture priority
    let caps = captures_for(me, state, &new_state);
    let mut score = caps as f32 * 10.0;

    // 2. Liberty gain of resulting group
    let (_, libs) = flood_group(&new_state.board, idx, size);
    score += libs.len() as f32 * 0.5;

    // 3. Atari threat: opponent groups with 1 liberty after placement
    for n in board_neighbors(idx, size) {
        if new_state.board[n] == opp {
            let (_, opp_libs) = flood_group(&new_state.board, n, size);
            if opp_libs.len() == 1 {
                score += 5.0;
            }
        }
    }

    // 4. Center bonus
    score += center_bonus(row, col, size) * 2.0;

    // 5. Edge penalty (unless capturing)
    if caps == 0 && is_first_line(row, col, size) {
        score -= 3.0;
    }

    // 6. Self-atari penalty
    if libs.len() == 1 && caps == 0 {
        score -= 20.0;
    }

    score
}

// ── T34: HintDelta Computation ─────────────────────────────────

/// Compute Go-specific hint delta for a move.
///
/// δ = score(template_move) - max(score(non_template_legal_moves))
///
/// The baseline is the best legal move that does NOT match the template.
/// This ensures:
/// - δ > 0 when the template finds a move better than anything non-template play would find
/// - δ ≈ 0 when the template's move is comparable to the best non-template move
/// - δ < 0 when the template leads to a suboptimal move
///
/// Falls back to 0.0 baseline if all legal moves match the template (rare but possible).
pub fn compute_go_delta(
    state: &GoState,
    template: GoTemplate,
    template_move: (usize, usize),
    legal_moves: &[(usize, usize)],
    last_move: Option<(usize, usize)>,
) -> MoveDelta {
    let board_tokens: Vec<i8> = state.board.iter().map(|c| *c as i8).collect();
    compute_go_delta_from_tokens(
        state,
        template,
        template_move,
        legal_moves,
        last_move,
        board_tokens,
    )
}

/// Pre-allocated variant: reuses `buf` for board token encoding.
/// Clears and refills `buf` with the current board state as i8 tokens.
pub fn compute_go_delta_into(
    state: &GoState,
    template: GoTemplate,
    template_move: (usize, usize),
    legal_moves: &[(usize, usize)],
    last_move: Option<(usize, usize)>,
    buf: &mut Vec<i8>,
) -> MoveDelta {
    buf.clear();
    buf.extend(state.board.iter().map(|c| *c as i8));
    compute_go_delta_from_tokens(
        state,
        template,
        template_move,
        legal_moves,
        last_move,
        std::mem::take(buf),
    )
}

/// Core logic shared by both variants.
fn compute_go_delta_from_tokens(
    state: &GoState,
    template: GoTemplate,
    template_move: (usize, usize),
    legal_moves: &[(usize, usize)],
    last_move: Option<(usize, usize)>,
    board_tokens: Vec<i8>,
) -> MoveDelta {
    let hinted_score = compute_move_score(state, template_move.0, template_move.1);

    // Baseline: best score among legal moves NOT matching this template.
    let best_unhinted_score = legal_moves
        .iter()
        .filter(|&&(r, c)| !matches_template(template, state, r, c, last_move))
        .map(|&(r, c)| compute_move_score(state, r, c))
        .fold(f32::NEG_INFINITY, f32::max);

    // If all moves match the template (or no non-matching moves), baseline = 0.
    let baseline = if best_unhinted_score.is_infinite() {
        0.0
    } else {
        best_unhinted_score
    };

    let delta = hinted_score - baseline;

    MoveDelta {
        move_num: state.move_count as usize + 1,
        template,
        board_tokens,
        suggested_move: template_move,
        hinted_score,
        best_unhinted_score: baseline,
        delta,
    }
}

// ── Template Delta Accumulator ─────────────────────────────────

/// Accumulates δ observations for a single template.
#[derive(Clone, Debug)]
struct TemplateDeltaAccumulator {
    total: f32,
    count: usize,
}

impl TemplateDeltaAccumulator {
    fn new() -> Self {
        Self {
            total: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, delta: f32) {
        self.total += delta;
        self.count += 1;
    }

    fn mean(&self) -> f32 {
        match self.count {
            0 => 0.0,
            _ => self.total / self.count as f32,
        }
    }
}

// ── T33: GoTemplateProposer ────────────────────────────────────

/// Per-template bandit stats for UCB1 selection.
#[derive(Clone, Debug)]
struct TemplateBanditStats {
    /// Total accumulated δ.
    total_delta: f32,
    /// Number of δ observations.
    delta_count: usize,
    /// Number of times selected (pulls).
    pulls: usize,
}

impl TemplateBanditStats {
    fn new() -> Self {
        Self {
            total_delta: 0.0,
            delta_count: 0,
            pulls: 0,
        }
    }

    fn mean_delta(&self) -> f32 {
        match self.delta_count {
            0 => 0.0,
            _ => self.total_delta / self.delta_count as f32,
        }
    }

    fn ucb1(&self, total_pulls: usize) -> f32 {
        if self.pulls == 0 || total_pulls == 0 {
            return f32::MAX;
        }
        let q = self.mean_delta();
        let n = self.pulls as f32;
        let total = total_pulls as f32;
        q + (UCB1_C * total.ln() / n).sqrt()
    }

    fn observe(&mut self, delta: f32) {
        self.total_delta += delta;
        self.delta_count += 1;
    }

    fn pull(&mut self) {
        self.pulls += 1;
    }
}

/// Go-specific template proposer with UCB1 selection.
///
/// Adapted from the general G-Zero TemplateProposer pattern but for Go:
/// - Query = board state as flat token array (0=empty, 1=black, 2=white)
/// - Hint = template-suggested move coordinates
/// - δ = score difference between template-proposed and best-unhinted
pub struct GoTemplateProposer {
    /// Per-template UCB1 stats.
    stats: Vec<TemplateBanditStats>,
    /// Last move by this player (for Tenuki template matching).
    last_move: Option<(usize, usize)>,
    /// Total pulls across all templates.
    total_pulls: usize,
}

impl GoTemplateProposer {
    /// Create a new proposer with 4 templates matching `GoTemplate` variants.
    pub fn new() -> Self {
        let stats = (0..NUM_TEMPLATES)
            .map(|_| TemplateBanditStats::new())
            .collect();
        Self {
            stats,
            last_move: None,
            total_pulls: 0,
        }
    }

    /// Select a template via UCB1.
    pub fn select_template(&mut self) -> GoTemplate {
        let idx = (0..NUM_TEMPLATES)
            .max_by(|&a, &b| {
                self.stats[a]
                    .ucb1(self.total_pulls)
                    .partial_cmp(&self.stats[b].ucb1(self.total_pulls))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        self.stats[idx].pull();
        self.total_pulls += 1;
        template_from_idx(idx)
    }

    /// Propose moves matching the given template.
    ///
    /// Returns all legal moves matching the template. Falls back to all legal moves
    /// if none match (ensures a move can always be played).
    pub fn propose_moves(
        &self,
        template: GoTemplate,
        state: &GoState,
        legal_moves: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let matching: Vec<_> = legal_moves
            .iter()
            .filter(|&&(r, c)| matches_template(template, state, r, c, self.last_move))
            .copied()
            .collect();

        match matching.is_empty() {
            true => legal_moves.to_vec(),
            false => matching,
        }
    }

    /// Observe a delta value for a template (updates bandit stats).
    pub fn observe_delta(&mut self, template: GoTemplate, delta: f32) {
        let idx = template as usize;
        if idx < self.stats.len() {
            self.stats[idx].observe(delta);
        }
    }

    /// Get average δ for a template.
    pub fn mean_delta(&self, template: GoTemplate) -> f32 {
        let idx = template as usize;
        match idx < self.stats.len() {
            true => self.stats[idx].mean_delta(),
            false => 0.0,
        }
    }

    /// Get pull count for a template.
    pub fn pull_count(&self, template: GoTemplate) -> usize {
        let idx = template as usize;
        match idx < self.stats.len() {
            true => self.stats[idx].pulls,
            false => 0,
        }
    }

    /// Update the last move played by this player (for Tenuki template matching).
    pub fn update_last_move(&mut self, row: usize, col: usize) {
        self.last_move = Some((row, col));
    }
}

impl Default for GoTemplateProposer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert template index to `GoTemplate`.
fn template_from_idx(idx: usize) -> GoTemplate {
    match idx % NUM_TEMPLATES {
        0 => GoTemplate::CornerStar,
        1 => GoTemplate::Capture,
        2 => GoTemplate::Defend,
        _ => GoTemplate::Tenuki,
    }
}

// ── T35: DeltaGatedAbsorbCompress ──────────────────────────────

/// Configuration for delta-gated absorb-compress in Go.
#[derive(Clone, Debug)]
pub struct GoDeltaGatedConfig {
    /// Minimum δ threshold for promotion.
    pub delta_threshold: f32,
    /// Minimum observations before promotion.
    pub min_observations: usize,
    /// Maximum templates to promote per compress cycle.
    pub max_promotions: usize,
}

impl Default for GoDeltaGatedConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 0.1,
            min_observations: 50,
            max_promotions: 1,
        }
    }
}

/// Per-template accumulated stats for absorb-compress.
#[derive(Clone, Debug)]
struct TemplateDeltaStats {
    /// Total δ accumulated.
    total_delta: f32,
    /// Number of observations.
    count: usize,
    /// Whether this template has been promoted.
    is_promoted: bool,
}

impl TemplateDeltaStats {
    fn new() -> Self {
        Self {
            total_delta: 0.0,
            count: 0,
            is_promoted: false,
        }
    }

    fn mean(&self) -> f32 {
        match self.count {
            0 => 0.0,
            _ => self.total_delta / self.count as f32,
        }
    }
}

/// Promotes high-δ templates to hard constraints (absorb-compress for Go).
///
/// Templates that consistently produce positive δ are "absorbed" —
/// their matching moves get priority in the self-play loop.
pub struct GoDeltaGatedAbsorbCompress {
    /// Per-template accumulated stats.
    template_deltas: Vec<TemplateDeltaStats>,
    /// Config.
    config: GoDeltaGatedConfig,
    /// Templates that have been promoted (absorbed).
    promoted: Vec<GoTemplate>,
}

impl GoDeltaGatedAbsorbCompress {
    /// Create a new absorb-compress with 4 templates.
    pub fn new(config: GoDeltaGatedConfig) -> Self {
        let template_deltas = (0..NUM_TEMPLATES)
            .map(|_| TemplateDeltaStats::new())
            .collect();
        Self {
            template_deltas,
            config,
            promoted: Vec::new(),
        }
    }

    /// Record a δ observation for a template.
    pub fn observe(&mut self, template: GoTemplate, delta: f32) {
        let idx = template as usize;
        if idx < self.template_deltas.len() {
            self.template_deltas[idx].total_delta += delta;
            self.template_deltas[idx].count += 1;
        }
    }

    /// Check if any template has enough observations to consider compression.
    pub fn should_compress(&self) -> bool {
        self.template_deltas
            .iter()
            .any(|s| s.count >= self.config.min_observations && !s.is_promoted)
    }

    /// Promote high-δ templates, return newly promoted templates.
    ///
    /// A template is promoted if:
    /// - It has ≥ `min_observations` observations
    /// - Its mean δ ≥ `delta_threshold`
    /// - It hasn't been promoted yet
    pub fn compress(&mut self) -> Vec<GoTemplate> {
        let mut newly_promoted = Vec::new();
        let mut promotions = 0;

        for (idx, stats) in self.template_deltas.iter_mut().enumerate() {
            if promotions >= self.config.max_promotions {
                break;
            }
            if stats.is_promoted {
                continue;
            }
            if stats.count < self.config.min_observations {
                continue;
            }
            if stats.mean() < self.config.delta_threshold {
                continue;
            }
            stats.is_promoted = true;
            let template = template_from_idx(idx);
            self.promoted.push(template);
            newly_promoted.push(template);
            promotions += 1;
        }

        newly_promoted
    }

    /// Check if a template has been promoted.
    pub fn is_promoted(&self, template: GoTemplate) -> bool {
        let idx = template as usize;
        match idx < self.template_deltas.len() {
            true => self.template_deltas[idx].is_promoted,
            false => false,
        }
    }

    /// Get all promoted templates.
    pub fn promoted_templates(&self) -> &[GoTemplate] {
        &self.promoted
    }

    /// Get average δ for a template.
    pub fn mean_delta(&self, template: GoTemplate) -> f32 {
        let idx = template as usize;
        match idx < self.template_deltas.len() {
            true => self.template_deltas[idx].mean(),
            false => 0.0,
        }
    }
}

// ── T36: GoGZeroSelfPlay ──────────────────────────────────────

/// Configuration for G-Zero self-play.
#[derive(Clone, Debug)]
pub struct GoGZeroSelfPlayConfig {
    /// Board size (default: 9).
    pub board_size: usize,
    /// Number of episodes (default: 500).
    pub num_episodes: usize,
    /// Whether to use delta-gated absorb-compress.
    pub use_delta_gating: bool,
    /// Delta-gated config.
    pub delta_config: GoDeltaGatedConfig,
    /// Print progress every N episodes.
    pub progress_interval: usize,
    /// Initial komi (default: 7.5).
    pub initial_komi: f32,
    /// Enable adaptive komi adjustment (default: true).
    pub adaptive_komi: bool,
    /// Base komi adjustment step — scaled proportionally by imbalance severity (default: 10.0).
    ///
    /// Proportional: `step = base × (|win_rate − 0.5| / 0.2)` so 70%→1×, 90%→2×, 100%→2.5×.
    pub komi_adjustment_step: f32,
    /// Minimum allowed komi (default: 0.0).
    pub komi_min: f32,
    /// Maximum allowed komi (default: 50.0).
    pub komi_max: f32,
    /// Number of episodes between komi adjustments (default: 50).
    pub komi_window: usize,
    /// Use score-based rewards instead of binary win/loss (default: true).
    pub score_based_rewards: bool,
    /// Swap colors each episode so each proposer plays both sides (default: true).
    ///
    /// Agent A plays Black on even episodes, White on odd episodes.
    /// Agent B plays White on even episodes, Black on odd episodes.
    /// This naturally balances win rates toward ~50/50.
    pub swap_colors: bool,
}

impl Default for GoGZeroSelfPlayConfig {
    fn default() -> Self {
        Self {
            board_size: 9,
            num_episodes: 500,
            use_delta_gating: true,
            delta_config: GoDeltaGatedConfig::default(),
            progress_interval: 50,
            initial_komi: 7.5,
            adaptive_komi: true,
            komi_adjustment_step: 10.0,
            komi_min: 0.0,
            komi_max: 50.0,
            komi_window: 50,
            score_based_rewards: true,
            swap_colors: true,
        }
    }
}

/// Aggregate results from G-Zero self-play.
#[derive(Clone, Debug)]
pub struct GoGZeroSelfPlayResults {
    /// Per-episode results.
    pub episodes: Vec<GoSelfPlayResult>,
    /// Black wins.
    pub black_wins: usize,
    /// White wins.
    pub white_wins: usize,
    /// Draws.
    pub draws: usize,
    /// Total delta across all moves.
    pub total_delta: f32,
    /// Average delta per move.
    pub avg_delta_per_move: f32,
    /// Per-template δ evolution (template_idx → vec of (episode, mean_delta)).
    pub template_delta_history: Vec<Vec<(usize, f32)>>,
    /// Promoted templates from absorb-compress.
    pub promoted_templates: Vec<GoTemplate>,
    /// Total duration.
    pub duration: std::time::Duration,
    /// Komi adjustment history: (episode, komi).
    pub komi_history: Vec<(usize, f32)>,
    /// Final komi value at end of self-play.
    pub final_komi: f32,
    /// Average score margin across all episodes.
    pub avg_score_margin: f32,
    /// Number of episodes where colors were swapped.
    pub swapped_episodes: usize,
}

/// Run G-Zero self-play.
///
/// Two GZero proposers play against each other. Each turn:
/// 1. Proposer selects template via UCB1
/// 2. Proposer proposes moves matching the template
/// 3. Compute δ for the best proposed move
/// 4. Feed δ back to proposer and delta-gated absorb-compress
/// 5. Play the move with highest greedy score among candidates
///
/// After each game:
/// - Record episode result
/// - Every 100 episodes, apply absorb-compress
/// - Every `progress_interval` episodes, print stats
pub fn run_gzero_selfplay(
    config: &GoGZeroSelfPlayConfig,
    _rng: &mut fastrand::Rng,
) -> GoGZeroSelfPlayResults {
    let start = Instant::now();
    let max_moves = config.board_size * config.board_size * MAX_MOVES_FACTOR;

    let mut proposer_a = GoTemplateProposer::new();
    let mut proposer_b = GoTemplateProposer::new();
    let mut absorb_compress = GoDeltaGatedAbsorbCompress::new(config.delta_config.clone());

    let mut legal_buf: Vec<(usize, usize)> =
        Vec::with_capacity(config.board_size * config.board_size);
    let mut episodes: Vec<GoSelfPlayResult> = Vec::with_capacity(config.num_episodes);
    let mut black_wins = 0usize;
    let mut white_wins = 0usize;
    let mut draws = 0usize;
    let mut total_delta = 0.0_f32;
    let mut total_moves = 0usize;
    let mut current_komi = config.initial_komi;
    let mut komi_history: Vec<(usize, f32)> = Vec::new();
    let mut total_score_margin = 0.0_f32;
    let mut swapped_episodes = 0usize;
    // Raw (unnormalized) scores per episode for score-margin-guided komi.
    let mut episode_raw_scores: Vec<f32> = Vec::with_capacity(config.num_episodes);

    // Per-template delta evolution: (episode, mean_delta) per template
    let mut template_delta_history: Vec<Vec<(usize, f32)>> =
        (0..NUM_TEMPLATES).map(|_| Vec::new()).collect();

    for episode_idx in 0..config.num_episodes {
        let episode_num = episode_idx + 1;
        let episode_start = Instant::now();

        let mut state = GoState::with_komi(config.board_size, current_komi);
        let mut replay = GoReplay::new(config.board_size, current_komi);
        let mut move_deltas: Vec<MoveDelta> = Vec::new();
        let mut board_tokens_buf: Vec<i8> =
            Vec::with_capacity(config.board_size * config.board_size);

        // Per-episode per-template delta accumulators
        let mut episode_template_deltas: Vec<TemplateDeltaAccumulator> = (0..NUM_TEMPLATES)
            .map(|_| TemplateDeltaAccumulator::new())
            .collect();

        let mut moves_played = 0usize;

        // Swap colors on odd episodes so each agent plays both sides equally.
        let swapped = config.swap_colors && episode_idx % 2 == 1;
        if swapped {
            swapped_episodes += 1;
        }

        while !state.is_terminal() && moves_played < max_moves {
            state.legal_moves_into(&mut legal_buf);
            let legal_moves = &legal_buf;

            // No legal moves → pass
            if legal_moves.is_empty() {
                let player = state.to_play;
                state.play_pass();
                replay.record(&GoAction::Pass, player, 0);
                moves_played += 1;
                continue;
            }

            // Select proposer based on current player and swap state.
            // When swapped: Agent A plays White, Agent B plays Black.
            let proposer = match (swapped, state.to_play) {
                (false, GoCell::Black) | (true, GoCell::White) => &mut proposer_a,
                (false, GoCell::White) | (true, GoCell::Black) => &mut proposer_b,
                (_, GoCell::Empty) => continue,
            };

            // Select template via UCB1
            let template = proposer.select_template();

            // Propose matching moves
            let candidates = proposer.propose_moves(template, &state, &legal_moves);

            // Pick best candidate by greedy score
            let best_move = candidates
                .iter()
                .max_by(|&&a, &&b| {
                    compute_move_score(&state, a.0, a.1)
                        .partial_cmp(&compute_move_score(&state, b.0, b.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(legal_moves[0]);

            // Compute δ for this move (reusing pre-allocated board buffer)
            let move_delta = compute_go_delta_into(
                &state,
                template,
                best_move,
                &legal_moves,
                proposer.last_move,
                &mut board_tokens_buf,
            );

            // Feed δ back to proposer
            proposer.observe_delta(template, move_delta.delta);

            // Feed δ to shared absorb-compress
            absorb_compress.observe(template, move_delta.delta);

            // Track per-episode per-template delta
            episode_template_deltas[template as usize].observe(move_delta.delta);

            move_deltas.push(move_delta);

            // Play the move
            let player = state.to_play;
            let legal_count = legal_moves.len();
            state.play_move(best_move.0, best_move.1);
            replay.record(
                &GoAction::Place(best_move.0, best_move.1),
                player,
                legal_count,
            );

            // Update last move for Tenuki template matching
            proposer.update_last_move(best_move.0, best_move.1);

            moves_played += 1;
        }

        // Force game end if max_moves reached
        if !state.is_terminal() {
            state.play_pass();
            state.play_pass();
        }

        // Determine winner
        let winner = state.get_winner();
        match winner {
            Some(GoCell::Black) => black_wins += 1,
            Some(GoCell::White) => white_wins += 1,
            _ => draws += 1,
        }

        // Score-based reward tracking
        let score = state.score();
        let score_margin = if config.score_based_rewards {
            score / score.abs().max(1.0) // normalized [-1, 1]
        } else {
            0.0
        };
        total_score_margin += score_margin;
        episode_raw_scores.push(score);

        // Finalize replay
        replay.finalize(winner, state.score());

        // Aggregate deltas
        let episode_delta: f32 = move_deltas.iter().map(|d| d.delta).sum();
        total_delta += episode_delta;
        total_moves += move_deltas.len();

        // Record per-template delta for this episode
        for (t_idx, acc) in episode_template_deltas.iter().enumerate() {
            if acc.count > 0 {
                template_delta_history[t_idx].push((episode_num, acc.mean()));
            }
        }

        let episode_duration = episode_start.elapsed();
        episodes.push(GoSelfPlayResult {
            episode: episode_num,
            winner,
            total_moves: moves_played,
            move_deltas,
            duration: episode_duration,
        });

        // Absorb-compress every 100 episodes
        if config.use_delta_gating && episode_num % 100 == 0 {
            let promoted = absorb_compress.compress();
            for t in &promoted {
                let mean = absorb_compress.mean_delta(*t);
                eprintln!("  [absorb] Episode {episode_num}: promoted {t:?} (mean δ={mean:.3})");
            }
        }

        // Adaptive komi — score-margin-guided with damping.
        //
        // Instead of win-rate thresholds (which oscillate wildly in small windows),
        // use the average raw score margin to compute a precise komi delta.
        // Positive avg_score = Black advantage → increase komi.
        // Damping factor (0.5) prevents overshoot; clamp to max_step prevents wild swings.
        if config.adaptive_komi && episode_num % config.komi_window == 0 && episode_num > 0 {
            let window_start = episode_num.saturating_sub(config.komi_window);
            let window_scores = &episode_raw_scores[window_start..episode_num];
            let window_total = window_scores.len().max(1);
            let avg_score: f32 = window_scores.iter().sum::<f32>() / window_total as f32;

            // Score-guided delta: avg_score is positive when Black is ahead.
            // To compensate, increase komi (give White more points).
            // Damping = 0.5 to converge without overshooting.
            const DAMPING: f32 = 0.5;
            let raw_delta = avg_score * DAMPING;
            // Clamp to base step to prevent oscillation at extreme scores.
            let clamped_delta =
                raw_delta.clamp(-config.komi_adjustment_step, config.komi_adjustment_step);

            let old_komi = current_komi;
            if clamped_delta.abs() > 0.1 {
                current_komi =
                    (current_komi + clamped_delta).clamp(config.komi_min, config.komi_max);
            }

            if old_komi != current_komi {
                let black_wr = window_scores.iter().filter(|&&s| s > 0.0).count() as f32
                    / window_total as f32
                    * 100.0;
                eprintln!(
                    "  [komi] Episode {episode_num}: {old_komi:.1} → {current_komi:.1} (avg margin: {avg_score:+.1}, B win: {black_wr:.0}%)"
                );
            }

            komi_history.push((episode_num, current_komi));
        }

        // Progress print
        if episode_num % config.progress_interval == 0 {
            let total_games = black_wins + white_wins + draws;
            let black_wr = match total_games {
                0 => 0.0,
                _ => black_wins as f32 / total_games as f32 * 100.0,
            };
            let white_wr = match total_games {
                0 => 0.0,
                _ => white_wins as f32 / total_games as f32 * 100.0,
            };
            let avg_delta = match total_moves {
                0 => 0.0,
                _ => total_delta / total_moves as f32,
            };
            let num_promoted = absorb_compress.promoted_templates().len();
            eprintln!(
                "  [{episode_num}/{total}] B:{black_wr:.0}% W:{white_wr:.0}% D:{draws} komi={current_komi:.1} avg_δ={avg_delta:.3} promoted={num_promoted}",
                total = config.num_episodes
            );
            let _ = std::io::stderr().flush();
        }
    }

    let avg_delta_per_move = match total_moves {
        0 => 0.0,
        _ => total_delta / total_moves as f32,
    };

    let promoted = absorb_compress.promoted_templates().to_vec();
    let avg_score_margin = if episodes.is_empty() {
        0.0
    } else {
        total_score_margin / episodes.len() as f32
    };

    GoGZeroSelfPlayResults {
        episodes,
        black_wins,
        white_wins,
        draws,
        total_delta,
        avg_delta_per_move,
        template_delta_history,
        promoted_templates: promoted,
        duration: start.elapsed(),
        komi_history,
        final_komi: current_komi,
        avg_score_margin,
        swapped_episodes,
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_template_proposer_selects_all_templates() {
        let mut proposer = GoTemplateProposer::new();
        let mut selected = [false; NUM_TEMPLATES];

        // Run enough selections to explore all templates via UCB1
        for _ in 0..200 {
            let template = proposer.select_template();
            selected[template as usize] = true;
            // Observe a small positive delta to encourage exploration
            proposer.observe_delta(template, 0.1);
        }

        for (idx, was_selected) in selected.iter().enumerate() {
            assert!(
                *was_selected,
                "Template {} ({:?}) was never selected",
                idx,
                template_from_idx(idx)
            );
        }
    }

    #[test]
    fn compute_go_delta_positive_for_good_move() {
        let mut state = GoState::new(9);

        // Place some stones to create a non-trivial board
        state.play_move(0, 1); // B
        state.play_move(2, 0); // W
        state.play_move(1, 2); // B
        state.play_move(5, 5); // W

        let legal_moves = state.legal_moves();
        assert!(!legal_moves.is_empty());

        // Pick the best move as the template move (δ should be >= 0)
        let best = legal_moves
            .iter()
            .max_by(|&&a, &&b| {
                compute_move_score(&state, a.0, a.1)
                    .partial_cmp(&compute_move_score(&state, b.0, b.1))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap();

        let delta = compute_go_delta(&state, GoTemplate::CornerStar, best, &legal_moves, None);

        // When the template picks the actual best move, δ should be >= 0
        // (because baseline is best non-matching move, which may be lower)
        assert!(
            delta.delta >= -0.001,
            "Expected δ >= 0 for best move, got {}",
            delta.delta
        );
    }

    #[test]
    fn compute_go_delta_negative_for_bad_move() {
        // On an empty 9x9 board, a corner move scores less than center
        let state = GoState::new(9);
        let legal_moves = state.legal_moves();

        // Template picks corner (0,0) which is a poor opening
        let template_move = (0, 0);
        let delta = compute_go_delta(
            &state,
            GoTemplate::CornerStar,
            template_move,
            &legal_moves,
            None,
        );

        // Corner should have negative δ (worse than best non-matching move)
        assert!(
            delta.delta < 0.0,
            "Expected δ < 0 for corner move on empty board, got {}",
            delta.delta
        );
    }

    #[test]
    fn delta_gated_absorb_compress_promotes_high_delta() {
        let config = GoDeltaGatedConfig {
            delta_threshold: 0.05,
            min_observations: 10,
            max_promotions: 1,
        };
        let mut ac = GoDeltaGatedAbsorbCompress::new(config);

        // Feed high positive deltas to Capture template
        for _ in 0..20 {
            ac.observe(GoTemplate::Capture, 0.5);
        }

        assert!(ac.should_compress());
        let promoted = ac.compress();
        assert_eq!(promoted.len(), 1);
        assert_eq!(promoted[0], GoTemplate::Capture);
        assert!(ac.is_promoted(GoTemplate::Capture));
    }

    #[test]
    fn delta_gated_absorb_compress_requires_min_observations() {
        let config = GoDeltaGatedConfig {
            delta_threshold: 0.05,
            min_observations: 50,
            max_promotions: 1,
        };
        let mut ac = GoDeltaGatedAbsorbCompress::new(config);

        // Feed high positive deltas but not enough observations
        for _ in 0..10 {
            ac.observe(GoTemplate::Capture, 0.5);
        }

        assert!(!ac.should_compress());
        let promoted = ac.compress();
        assert!(promoted.is_empty());
        assert!(!ac.is_promoted(GoTemplate::Capture));
    }

    #[test]
    fn gzero_selfplay_completes_episodes() {
        let mut rng = fastrand::Rng::with_seed(42);
        let config = GoGZeroSelfPlayConfig {
            board_size: 9,
            num_episodes: 10,
            use_delta_gating: false,
            delta_config: GoDeltaGatedConfig::default(),
            progress_interval: 100,
            initial_komi: 7.5,
            adaptive_komi: false,
            komi_adjustment_step: 2.0,
            komi_min: 0.0,
            komi_max: 20.0,
            komi_window: 100,
            score_based_rewards: false,
            swap_colors: false,
        };

        let results = run_gzero_selfplay(&config, &mut rng);

        assert_eq!(results.episodes.len(), 10);
        assert_eq!(results.black_wins + results.white_wins + results.draws, 10);

        // Every episode should have at least 1 move
        for ep in &results.episodes {
            assert!(ep.total_moves > 0, "Episode {} had 0 moves", ep.episode);
        }

        // Should have at least one decisive game (with komi 7.5, draws are extremely unlikely)
        assert!(
            results.black_wins + results.white_wins > 0,
            "Expected at least one decisive game"
        );
    }

    #[test]
    fn gzero_selfplay_tracks_delta_evolution() {
        let mut rng = fastrand::Rng::with_seed(123);
        let config = GoGZeroSelfPlayConfig {
            board_size: 9,
            num_episodes: 20,
            use_delta_gating: true,
            delta_config: GoDeltaGatedConfig {
                delta_threshold: 0.05,
                min_observations: 5,
                max_promotions: 1,
            },
            progress_interval: 100,
            initial_komi: 7.5,
            adaptive_komi: false,
            komi_adjustment_step: 2.0,
            komi_min: 0.0,
            komi_max: 20.0,
            komi_window: 100,
            score_based_rewards: false,
            swap_colors: false,
        };

        let results = run_gzero_selfplay(&config, &mut rng);

        // Template delta history should be populated for at least some templates
        let non_empty = results
            .template_delta_history
            .iter()
            .filter(|h| !h.is_empty())
            .count();
        assert!(
            non_empty > 0,
            "Expected at least one template with delta history"
        );

        // Total delta should be a real finite number
        assert!(
            results.total_delta.is_finite(),
            "Total delta should be finite"
        );
    }

    #[test]
    fn board_tokens_representation() {
        let mut state = GoState::new(9);

        // Empty board → all zeros
        let tokens: Vec<i8> = state.board.iter().map(|c| *c as i8).collect();
        assert!(
            tokens.iter().all(|&t| t == 0),
            "Empty board should be all 0"
        );

        // Place Black at (4,4), White at (0,0)
        state.play_move(4, 4); // B
        state.play_move(0, 0); // W

        let tokens: Vec<i8> = state.board.iter().map(|c| *c as i8).collect();
        let idx_44 = 4 * 9 + 4;
        let idx_00 = 0;

        assert_eq!(tokens[idx_44], 1, "Black at (4,4) should be 1");
        assert_eq!(tokens[idx_00], 2, "White at (0,0) should be 2");

        // Most cells should still be empty (0)
        let empty_count = tokens.iter().filter(|&&t| t == 0).count();
        assert!(
            empty_count >= 79,
            "Expected at least 79 empty cells, got {empty_count}"
        );
    }
}
