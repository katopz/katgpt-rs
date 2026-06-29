//! Go head-to-head tournament infrastructure — plays our players against AutoGo agents.
//!
//! Plan 065 Phase 3 (T25–T27):
//! - **T25**: `GoTournamentConfig`, `GoTournamentResult`, `GoPlayerType`
//! - **T26**: `run_tournament()` — plays N games via API with dual-move semantics (G2)
//! - **T27**: `AutoGoProxyPlayer` — adapter wrapping AutoGo REST API as a `GoPlayer`
//!
//! ## Dual-Move Semantics (G2)
//!
//! The AutoGo API applies both our move AND the AI's response in a single HTTP call.
//! This means the game loop is:
//! 1. `new_game()` → get initial `legal_moves`
//! 2. Our player picks from `legal_moves` → `make_move()`
//! 3. Response has AI move baked in → read new `legal_moves`
//! 4. Repeat until `is_over`

use std::io::Write;
use std::time::Instant;

use fastrand::Rng;

use super::autogo_client::{AutoGoClient, AutoGoGameState};
use super::players::{
    GoGZeroPlayer, GoGreedyPlayer, GoHLPlayer, GoMctsPlayer, GoPlayer, GoRandomPlayer,
    GoValidatorPlayer,
};
#[cfg(all(feature = "sdpg_bandit", feature = "go"))]
use super::sdpg_player::GoSdpgPlayer;
use super::state::GoState;
use super::types::{GoAction, GoCell};

// ── GoPlayerType (T25) ─────────────────────────────────────────

/// Player type selector for tournament configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GoPlayerType {
    /// Random legal moves with 2% pass.
    Random,
    /// Greedy: captures + liberties + positional scoring.
    Greedy,
    /// Safety-first rules layered on greedy.
    Validator,
    /// Bandit Q-learning over 8 move categories.
    HL,
    /// Template proposer with local UCB1.
    GZero,
    /// MCTS with GoHeuristic rollout.
    MCTS,
    /// MCTS with custom budget and rollout depth.
    MctsCustom {
        /// Number of MCTS iterations.
        budget: usize,
        /// Maximum rollout depth.
        rollout_depth: usize,
    },
    /// SDPG oracle-informed self-distilled policy gradient (Plan 194).
    #[cfg(all(feature = "sdpg_bandit", feature = "go"))]
    Sdpg,
}

impl GoPlayerType {
    /// Create a boxed player instance of this type.
    pub fn create_player(&self) -> Box<dyn GoPlayer> {
        match self {
            Self::Random => Box::new(GoRandomPlayer),
            Self::Greedy => Box::new(GoGreedyPlayer),
            Self::Validator => Box::new(GoValidatorPlayer),
            Self::HL => Box::new(GoHLPlayer::new()),
            Self::GZero => Box::new(GoGZeroPlayer::new()),
            Self::MCTS => Box::new(GoMctsPlayer::default_player()),
            Self::MctsCustom {
                budget,
                rollout_depth,
            } => Box::new(GoMctsPlayer::new(*budget, *rollout_depth)),
            #[cfg(all(feature = "sdpg_bandit", feature = "go"))]
            Self::Sdpg => Box::new(GoSdpgPlayer::new()),
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Random => "Random",
            Self::Greedy => "Greedy",
            Self::Validator => "Validator",
            Self::HL => "HL",
            Self::GZero => "GZero",
            Self::MCTS => "MCTS",
            Self::MctsCustom { .. } => "MCTS-Custom",
            #[cfg(all(feature = "sdpg_bandit", feature = "go"))]
            Self::Sdpg => "SDPG",
        }
    }
}

// ── GoTournamentConfig (T25) ───────────────────────────────────

/// Configuration for a head-to-head tournament against AutoGo agents.
///
/// ## Example
///
/// ```ignore
/// use katgpt_rs::pruners::go::tournament::{GoTournamentConfig, GoPlayerType};
///
/// let config = GoTournamentConfig {
///     board_size: 9,
///     num_games: 100,
///     our_player: GoPlayerType::Greedy,
///     their_agent: "gnugo1".to_string(),
///     autogo_url: "http://localhost:8000".to_string(),
///     play_both_colors: true,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct GoTournamentConfig {
    /// Board dimension (9, 13, or 19).
    pub board_size: usize,
    /// Number of games to play.
    pub num_games: usize,
    /// Our player strategy.
    pub our_player: GoPlayerType,
    /// AutoGo agent name (e.g. "random", "gnugo1").
    pub their_agent: String,
    /// AutoGo server URL.
    pub autogo_url: String,
    /// If true, our player plays both Black and White (alternating each game).
    pub play_both_colors: bool,
}

impl Default for GoTournamentConfig {
    fn default() -> Self {
        Self {
            board_size: 9,
            num_games: 10,
            our_player: GoPlayerType::Random,
            their_agent: "random".to_string(),
            autogo_url: "http://localhost:8000".to_string(),
            play_both_colors: true,
        }
    }
}

// ── GameOutcome ────────────────────────────────────────────────

/// Outcome of a single game from our perspective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameOutcome {
    /// We won.
    Win,
    /// They won.
    Loss,
    /// Draw (jigo — extremely unlikely with fractional komi).
    Draw,
}

// ── GoTournamentResult (T25) ───────────────────────────────────

/// Aggregate results from a head-to-head tournament.
#[derive(Clone, Debug)]
pub struct GoTournamentResult {
    /// Number of games we won.
    pub our_wins: usize,
    /// Number of games they won.
    pub their_wins: usize,
    /// Number of draws.
    pub draws: usize,
    /// Average score delta (positive = our advantage).
    pub avg_score_delta: f32,
    /// Games per second throughput.
    pub games_per_sec: f32,
    /// Total moves across all games.
    pub total_moves: usize,
    /// Per-game results.
    pub games: Vec<GameResult>,
}

/// Result of a single head-to-head game.
#[derive(Clone, Debug)]
pub struct GameResult {
    /// Game outcome from our perspective.
    pub outcome: GameOutcome,
    /// Score delta from our perspective (positive = we're ahead).
    pub score_delta: f32,
    /// Our color in this game.
    pub our_color: GoCell,
    /// Total moves played (including AI moves).
    pub total_moves: usize,
    /// Game duration.
    pub duration: std::time::Duration,
    /// AutoGo game result string (e.g. "W+2.5").
    pub result_string: String,
}

impl GoTournamentResult {
    /// Win rate as a fraction (0.0 to 1.0).
    pub fn win_rate(&self) -> f32 {
        if self.games.is_empty() {
            return 0.0;
        }
        self.our_wins as f32 / self.games.len() as f32
    }

    /// Total games completed successfully.
    pub fn total_games(&self) -> usize {
        self.games.len()
    }

    /// Average moves per game.
    pub fn avg_moves(&self) -> f32 {
        if self.games.is_empty() {
            return 0.0;
        }
        self.total_moves as f32 / self.games.len() as f32
    }

    /// Print a formatted summary to stdout.
    pub fn print_summary(&self, player_label: &str, agent: &str) {
        let wr = self.win_rate() * 100.0;
        println!();
        println!("  Result: {player_label} vs {agent}");
        println!(
            "  {}W / {}L / {}D ({:.1}%) avg {:.1} moves, {:.1} games/s",
            self.our_wins,
            self.their_wins,
            self.draws,
            wr,
            self.avg_moves(),
            self.games_per_sec,
        );
    }
}

// ── Result Parsing ─────────────────────────────────────────────

/// Parsed Go result string (e.g. "W+2.5", "B+1").
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedResult {
    /// Winning color.
    pub winner: GoCell,
    /// Margin of victory.
    pub margin: f32,
}

/// Parse a Go result string like "W+2.5" or "B+1".
///
/// Returns `None` for unrecognized formats.
pub fn parse_go_result(result: &str) -> Option<ParsedResult> {
    let result = result.trim();
    if result.is_empty() {
        return None;
    }

    let (color_char, margin_str) = result.split_once('+')?;
    let winner = match color_char.trim() {
        "B" | "b" => GoCell::Black,
        "W" | "w" => GoCell::White,
        _ => return None,
    };

    let margin: f32 = margin_str.trim().parse().ok()?;
    Some(ParsedResult { winner, margin })
}

// ── API State Bridge ───────────────────────────────────────────

/// Convert an AutoGo API game state to our `GoState` for player consumption.
///
/// Note: `ko_point` is not available from the API, so it's set to `None`.
/// This is acceptable because our players don't rely on `ko_point` for move selection
/// (they only use `legal_moves` which the API provides).
fn api_state_to_go_state(api_state: &AutoGoGameState) -> GoState {
    let mut state = GoState::new(api_state.size);

    // Convert 2D board to flat array
    for r in 0..api_state.size {
        for c in 0..api_state.size {
            let idx = state.flat_index(r, c);
            if let Some(&v) = api_state.board.get(r).and_then(|row| row.get(c)) {
                state.board[idx] = GoCell::from_i8(v).unwrap_or(GoCell::Empty);
            }
        }
    }

    // Set current player
    state.to_play = match api_state.to_play {
        1 => GoCell::Black,
        2 => GoCell::White,
        _ => GoCell::Black,
    };

    state
}

// ── Outcome Update Helper ──────────────────────────────────────

/// Call `update_outcome(won)` on bandit-based players (HL, GZero).
fn update_player_outcome(player: &mut dyn GoPlayer, won: bool) {
    if let Some(hl) = player.as_any_mut().downcast_mut::<GoHLPlayer>() {
        hl.update_outcome(won);
    }
    if let Some(gz) = player.as_any_mut().downcast_mut::<GoGZeroPlayer>() {
        gz.update_outcome(won);
    }
}

// ── Single Game Runner (T26) ───────────────────────────────────

/// Play a single game against an AutoGo agent.
///
/// Uses the API's dual-move semantics (G2): each `make_move()` call plays our move
/// AND triggers the AI's response. The returned state has both moves applied.
fn play_api_game(
    player: &mut dyn GoPlayer,
    client: &AutoGoClient,
    our_color: GoCell,
    board_size: usize,
    their_agent: &str,
    rng: &mut Rng,
) -> Result<GameResult, String> {
    let start = Instant::now();
    let color_str = match our_color {
        GoCell::Black => "black",
        GoCell::White => "white",
        _ => unreachable!(),
    };

    // Start new game on AutoGo server
    let mut api_state = client
        .new_game(board_size, color_str, their_agent)
        .map_err(|e| format!("Failed to start game: {e}"))?;

    let mut total_moves = 0usize;
    let max_moves = board_size * board_size * 3; // Safety limit

    while !api_state.is_over && total_moves < max_moves {
        // Convert API state to our GoState for player consumption
        let go_state = api_state_to_go_state(&api_state);
        let legal_moves: Vec<(usize, usize)> = api_state.legal_moves.clone();

        // Our player selects a move
        let action = player.select_move(&go_state, &legal_moves, rng);

        // Send move to API (response includes both our move AND AI's response)
        let new_api_state = match &action {
            GoAction::Place(row, col) => client.make_move(&api_state.game_id, *row, *col),
            GoAction::Pass => client.pass_move(&api_state.game_id),
        }
        .map_err(|e| format!("API error: {e}"))?;

        api_state = new_api_state;

        // Dual-move semantics: each API call counts as 2 moves (ours + theirs)
        total_moves += 2;
    }

    // Parse result
    let result_string = api_state
        .result
        .clone()
        .unwrap_or_else(|| "Unknown".to_string());

    let (outcome, score_delta) = match parse_go_result(&result_string) {
        Some(parsed) => {
            let won = parsed.winner == our_color;
            let score_delta = if won { parsed.margin } else { -parsed.margin };
            let outcome = if won {
                GameOutcome::Win
            } else {
                GameOutcome::Loss
            };
            (outcome, score_delta)
        }
        None => (GameOutcome::Draw, 0.0),
    };

    let duration = start.elapsed();

    Ok(GameResult {
        outcome,
        score_delta,
        our_color,
        total_moves,
        duration,
        result_string,
    })
}

// ── Tournament Runner (T26) ────────────────────────────────────

/// Run a head-to-head tournament against an AutoGo agent.
///
/// Plays `config.num_games` games via the AutoGo API, alternating colors if
/// `play_both_colors` is true. Returns aggregate results.
///
/// ## Errors
///
/// Returns an error if the AutoGo server is unreachable or any API call fails.
pub fn run_tournament(
    config: &GoTournamentConfig,
    rng: &mut Rng,
) -> Result<GoTournamentResult, String> {
    let client = AutoGoClient::new(&config.autogo_url);
    let mut player = config.our_player.create_player();

    // Verify server is reachable
    let agents = client
        .list_agents()
        .map_err(|e| format!("Cannot reach AutoGo server at {}: {e}", config.autogo_url))?;

    if !agents.contains(&config.their_agent) {
        return Err(format!(
            "Agent '{}' not found. Available: {:?}",
            config.their_agent, agents
        ));
    }

    log::info!(
        "Starting tournament: {} vs {} ({} games, {}x{}, both_colors={})",
        config.our_player.label(),
        config.their_agent,
        config.num_games,
        config.board_size,
        config.board_size,
        config.play_both_colors,
    );

    let mut games = Vec::with_capacity(config.num_games);
    let tournament_start = Instant::now();

    for i in 0..config.num_games {
        // Determine our color for this game
        let our_color = if config.play_both_colors {
            match i % 2 {
                0 => GoCell::Black,
                _ => GoCell::White,
            }
        } else {
            GoCell::Black
        };

        let color_label = match our_color {
            GoCell::Black => "B",
            GoCell::White => "W",
            _ => unreachable!(),
        };

        print!(
            "  [{:>3}/{}] {}({}) ",
            i + 1,
            config.num_games,
            player.name(),
            color_label
        );
        let _ = std::io::stdout().flush();

        match play_api_game(
            player.as_mut(),
            &client,
            our_color,
            config.board_size,
            &config.their_agent,
            rng,
        ) {
            Ok(result) => {
                let outcome_str = match result.outcome {
                    GameOutcome::Win => "WIN",
                    GameOutcome::Loss => "LOSS",
                    GameOutcome::Draw => "DRAW",
                };
                println!(
                    "{:<4} {:>8} {:>3} moves ({:.1}s)",
                    outcome_str,
                    result.result_string,
                    result.total_moves,
                    result.duration.as_secs_f64()
                );

                update_player_outcome(player.as_mut(), result.outcome == GameOutcome::Win);
                games.push(result);
            }
            Err(e) => {
                println!("ERROR: {e}");
                log::error!("Game {}/{} failed: {e}", i + 1, config.num_games);
            }
        }
    }

    player.reset();

    let total_duration = tournament_start.elapsed();
    let mut our_wins = 0usize;
    let mut their_wins = 0usize;
    let mut draws = 0usize;
    let mut total_moves: usize = 0;
    let mut score_delta_sum = 0.0f32;
    for g in &games {
        total_moves += g.total_moves;
        score_delta_sum += g.score_delta;
        match g.outcome {
            GameOutcome::Win => our_wins += 1,
            GameOutcome::Loss => their_wins += 1,
            GameOutcome::Draw => draws += 1,
        }
    }
    let avg_score_delta = if games.is_empty() {
        0.0
    } else {
        score_delta_sum / games.len() as f32
    };
    let games_per_sec = if total_duration.as_secs_f32() > 0.0 {
        games.len() as f32 / total_duration.as_secs_f32()
    } else {
        0.0
    };

    let result = GoTournamentResult {
        our_wins,
        their_wins,
        draws,
        avg_score_delta,
        games_per_sec,
        total_moves,
        games,
    };

    log::info!(
        "Tournament complete: {}W {}D {}L ({:.1}%, {:.1} games/s)",
        result.our_wins,
        result.draws,
        result.their_wins,
        result.win_rate() * 100.0,
        result.games_per_sec,
    );

    Ok(result)
}

// ── Tournament Batch Runner ────────────────────────────────────

/// A tournament definition: our player vs their agent.
#[derive(Clone)]
pub struct TournamentDef {
    /// Our player type.
    pub our_player: GoPlayerType,
    /// AutoGo agent name.
    pub their_agent: &'static str,
}

impl TournamentDef {
    /// Create a new tournament definition.
    pub const fn new(our_player: GoPlayerType, their_agent: &'static str) -> Self {
        Self {
            our_player,
            their_agent,
        }
    }

    /// Label for display.
    pub fn label(&self) -> String {
        format!("{} vs {}", self.our_player.label(), self.their_agent)
    }
}

/// Run multiple tournaments sequentially, collecting results.
///
/// Returns a vec of `(TournamentDef, GoTournamentResult)` pairs.
/// Failed tournaments are skipped with an error logged.
pub fn run_tournament_batch(
    tournaments: &[TournamentDef],
    config_template: &GoTournamentConfig,
    rng: &mut Rng,
) -> Vec<(TournamentDef, GoTournamentResult)> {
    let mut results = Vec::with_capacity(tournaments.len());

    for (idx, tournament) in tournaments.iter().enumerate() {
        println!(
            "\nTournament {}/{}: {}",
            idx + 1,
            tournaments.len(),
            tournament.label()
        );
        println!("{}", "─".repeat(50));

        let mut config = config_template.clone();
        config.our_player = tournament.our_player.clone();
        config.their_agent = tournament.their_agent.to_string();

        match run_tournament(&config, rng) {
            Ok(result) => {
                result.print_summary(tournament.our_player.label(), tournament.their_agent);
                results.push((tournament.clone(), result));
            }
            Err(e) => {
                log::error!("Tournament '{}' failed: {e}", tournament.label());
                println!("  SKIPPED: {e}");
            }
        }
    }

    results
}

/// Print a final results table from a batch of tournament results.
pub fn print_batch_table(results: &[(TournamentDef, GoTournamentResult)]) {
    println!();
    println!("══════════════════════════════════════════════════════════════");
    println!("  HEAD-TO-HEAD RESULTS");
    println!("══════════════════════════════════════════════════════════════");
    println!(
        "  {:<14} {:<12} {:>6} {:>8} {:>10}",
        "Our Player", "vs Agent", "Win%", "W/L/D", "Avg Moves"
    );
    println!("  ──────────────  ───────────  ─────  ────────  ─────────");

    for (def, result) in results {
        let wr = result.win_rate() * 100.0;
        let wld = format!("{}/{}/{}", result.our_wins, result.their_wins, result.draws);
        println!(
            "  {:<14} {:<12} {:>5.1}% {:>8} {:>10.1}",
            def.our_player.label(),
            def.their_agent,
            wr,
            wld,
            result.avg_moves(),
        );
    }

    println!("══════════════════════════════════════════════════════════════");
}

// ── T27: AutoGoProxyPlayer ─────────────────────────────────────

/// Proxy player that delegates move selection to an AutoGo agent via REST API.
///
/// This adapter wraps an AutoGo agent as a `GoPlayer`, enabling:
/// - Control experiments (AutoGo agent vs AutoGo agent)
/// - Baseline measurements within the internal tournament framework
///
/// ## Limitations
///
/// Because the AutoGo API uses dual-move semantics (our move triggers AI response),
/// this proxy is designed for use within the `play_api_game` flow where the API
/// manages game state. For internal-only tournaments, use the local player implementations.
pub struct AutoGoProxyPlayer<'a> {
    /// The agent name on the AutoGo server.
    agent: String,
    /// Reference to the shared API client.
    client: &'a AutoGoClient,
    /// Current game ID (set by `start_game`).
    game_id: Option<String>,
}

impl<'a> AutoGoProxyPlayer<'a> {
    /// Create a new proxy player for the given AutoGo agent.
    pub fn new(agent: &str, client: &'a AutoGoClient) -> Self {
        Self {
            agent: agent.to_lowercase(),
            client,
            game_id: None,
        }
    }

    /// Start a new game on the AutoGo server.
    ///
    /// Call before using this player. Sets up the game ID for API calls.
    pub fn start_game(
        &mut self,
        board_size: usize,
        our_color: GoCell,
    ) -> Result<AutoGoGameState, String> {
        let color_str = match our_color {
            GoCell::Black => "black",
            GoCell::White => "white",
            _ => unreachable!(),
        };

        let state = self
            .client
            .new_game(board_size, color_str, &self.agent)
            .map_err(|e| format!("Failed to start game: {e}"))?;

        self.game_id = Some(state.game_id.clone());
        Ok(state)
    }

    /// Get the current game ID, if a game is in progress.
    pub fn game_id(&self) -> Option<&str> {
        self.game_id.as_deref()
    }

    /// Agent name.
    pub fn agent(&self) -> &str {
        &self.agent
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_result_black_wins() {
        let parsed = parse_go_result("B+2.5").unwrap();
        assert_eq!(parsed.winner, GoCell::Black);
        assert!((parsed.margin - 2.5).abs() < 0.01);
    }

    #[test]
    fn parse_result_white_wins() {
        let parsed = parse_go_result("W+7.5").unwrap();
        assert_eq!(parsed.winner, GoCell::White);
        assert!((parsed.margin - 7.5).abs() < 0.01);
    }

    #[test]
    fn parse_result_integer_margin() {
        let parsed = parse_go_result("B+1").unwrap();
        assert_eq!(parsed.winner, GoCell::Black);
        assert!((parsed.margin - 1.0).abs() < 0.01);
    }

    #[test]
    fn parse_result_lowercase() {
        let parsed = parse_go_result("w+3.5").unwrap();
        assert_eq!(parsed.winner, GoCell::White);
        assert!((parsed.margin - 3.5).abs() < 0.01);
    }

    #[test]
    fn parse_result_whitespace() {
        let parsed = parse_go_result("  W + 3.5  ").unwrap();
        assert_eq!(parsed.winner, GoCell::White);
        assert!((parsed.margin - 3.5).abs() < 0.01);
    }

    #[test]
    fn parse_result_invalid() {
        assert!(parse_go_result("").is_none());
        assert!(parse_go_result("X+1").is_none());
        assert!(parse_go_result("B-1").is_none());
        assert!(parse_go_result("B").is_none());
        assert!(parse_go_result("+1").is_none());
    }

    #[test]
    fn player_type_labels() {
        assert_eq!(GoPlayerType::Random.label(), "Random");
        assert_eq!(GoPlayerType::Greedy.label(), "Greedy");
        assert_eq!(GoPlayerType::Validator.label(), "Validator");
        assert_eq!(GoPlayerType::HL.label(), "HL");
        assert_eq!(GoPlayerType::GZero.label(), "GZero");
        assert_eq!(GoPlayerType::MCTS.label(), "MCTS");
        assert_eq!(
            GoPlayerType::MctsCustom {
                budget: 100,
                rollout_depth: 30
            }
            .label(),
            "MCTS-Custom"
        );
    }

    #[test]
    fn player_type_creates_instances() {
        let mut rng = fastrand::Rng::new();
        let state = GoState::new(9);
        let legal = state.legal_moves();

        for pt in [
            GoPlayerType::Random,
            GoPlayerType::Greedy,
            GoPlayerType::Validator,
            GoPlayerType::HL,
            GoPlayerType::GZero,
            GoPlayerType::MCTS,
        ] {
            let mut p = pt.create_player();
            let action = p.select_move(&state, &legal, &mut rng);
            assert!(
                matches!(action, GoAction::Place(_, _)),
                "{} returned Pass on empty board",
                p.name()
            );
        }
    }

    #[test]
    fn mcts_custom_creates_with_params() {
        let pt = GoPlayerType::MctsCustom {
            budget: 500,
            rollout_depth: 100,
        };
        let player = pt.create_player();
        assert_eq!(player.name(), "MCTS");
    }

    #[test]
    fn config_default() {
        let config = GoTournamentConfig::default();
        assert_eq!(config.board_size, 9);
        assert_eq!(config.num_games, 10);
        assert_eq!(config.our_player, GoPlayerType::Random);
        assert_eq!(config.their_agent, "random");
        assert!(config.play_both_colors);
    }

    #[test]
    fn result_win_rate() {
        let result = GoTournamentResult {
            our_wins: 7,
            their_wins: 2,
            draws: 1,
            avg_score_delta: 3.5,
            games_per_sec: 10.0,
            total_moves: 700,
            games: vec![
                GameResult {
                    outcome: GameOutcome::Win,
                    score_delta: 5.0,
                    our_color: GoCell::Black,
                    total_moves: 100,
                    duration: std::time::Duration::from_secs(1),
                    result_string: "B+5.0".to_string(),
                };
                7
            ],
        };
        assert!((result.win_rate() - 1.0).abs() < 0.01);
        assert_eq!(result.total_games(), 7);
        assert!((result.avg_moves() - 100.0).abs() < 0.01);
    }

    #[test]
    fn result_empty_games() {
        let result = GoTournamentResult {
            our_wins: 0,
            their_wins: 0,
            draws: 0,
            avg_score_delta: 0.0,
            games_per_sec: 0.0,
            total_moves: 0,
            games: vec![],
        };
        assert_eq!(result.win_rate(), 0.0);
        assert_eq!(result.total_games(), 0);
        assert_eq!(result.avg_moves(), 0.0);
    }

    #[test]
    fn outcome_equality() {
        assert_eq!(GameOutcome::Win, GameOutcome::Win);
        assert_ne!(GameOutcome::Win, GameOutcome::Loss);
        assert_ne!(GameOutcome::Win, GameOutcome::Draw);
    }

    #[test]
    fn api_state_to_go_state_conversion() {
        let api_state = AutoGoGameState {
            game_id: "test".to_string(),
            board: vec![vec![0, 0, 0], vec![0, 1, 0], vec![0, 0, 2]],
            size: 3,
            to_play: 2,
            last_move: Some((2, 2)),
            is_over: false,
            result: None,
            legal_moves: vec![(0, 0), (0, 1), (0, 2)],
            human_color: 1,
            message: String::new(),
        };

        let go_state = api_state_to_go_state(&api_state);

        assert_eq!(go_state.size, 3);
        assert_eq!(go_state.to_play, GoCell::White);
        assert_eq!(go_state.at(1, 1), GoCell::Black);
        assert_eq!(go_state.at(2, 2), GoCell::White);
        assert_eq!(go_state.at(0, 0), GoCell::Empty);
    }

    #[test]
    fn api_state_handles_empty_board() {
        let api_state = AutoGoGameState {
            game_id: "test".to_string(),
            board: vec![vec![0, 0], vec![0, 0]],
            size: 2,
            to_play: 1,
            last_move: None,
            is_over: false,
            result: None,
            legal_moves: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
            human_color: 1,
            message: String::new(),
        };

        let go_state = api_state_to_go_state(&api_state);
        assert_eq!(go_state.to_play, GoCell::Black);
        assert_eq!(go_state.at(0, 0), GoCell::Empty);
        assert_eq!(go_state.at(1, 1), GoCell::Empty);
    }

    #[test]
    fn tournament_def_label() {
        let def = TournamentDef::new(GoPlayerType::Greedy, "gnugo1");
        assert_eq!(def.label(), "Greedy vs gnugo1");
    }

    #[test]
    fn parsed_result_debug() {
        let pr = ParsedResult {
            winner: GoCell::Black,
            margin: 2.5,
        };
        assert!(format!("{pr:?}").contains("Black"));
    }

    #[test]
    fn game_result_clone_debug() {
        let gr = GameResult {
            outcome: GameOutcome::Win,
            score_delta: 3.5,
            our_color: GoCell::Black,
            total_moves: 150,
            duration: std::time::Duration::from_secs(2),
            result_string: "B+3.5".to_string(),
        };
        let cloned = gr.clone();
        assert_eq!(cloned.outcome, GameOutcome::Win);
        assert!(format!("{gr:?}").contains("GameResult"));
    }
}
