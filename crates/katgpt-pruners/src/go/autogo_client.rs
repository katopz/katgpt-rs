//! AutoGo REST API client — calls `play.py` FastAPI server for head-to-head Go.
//!
//! ## API Dual-Move Semantics (G2)
//!
//! `make_move()` plays our move AND immediately triggers AutoGo's AI response.
//! The returned [`AutoGoGameState`] has BOTH moves applied. This means:
//! - One HTTP call = two Go moves (ours + theirs)
//! - We only need to call `make_move()` on our turn
//! - If we play White, `new_game(color=white)` triggers AI's first move automatically
//!
//! ## Blocking HTTP (G8)
//!
//! Uses [`reqwest::blocking::Client`] because tournament play is sequential
//! (one game at a time with per-game state). Parallelism, if needed later,
//! comes from multiple game_ids each with their own blocking client in a thread.

use serde::{Deserialize, Serialize};

/// Errors from AutoGo API interactions.
#[derive(Debug)]
pub enum AutoGoError {
    /// HTTP request failed (network, timeout, etc.).
    Http(reqwest::Error),
    /// API returned a non-success status code.
    Status(u16, String),
    /// Response body could not be parsed.
    Parse(reqwest::Error),
}

impl std::fmt::Display for AutoGoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(e) => write!(f, "HTTP request failed: {e}"),
            Self::Status(code, body) => write!(f, "API error {code}: {body}"),
            Self::Parse(e) => write!(f, "Failed to parse response: {e}"),
        }
    }
}

impl std::error::Error for AutoGoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Http(e) | Self::Parse(e) => Some(e),
            Self::Status(_, _) => None,
        }
    }
}

type Result<T> = std::result::Result<T, AutoGoError>;

/// AutoGo REST API client (calls `play.py` FastAPI server).
///
/// ## Example
///
/// ```no_run
/// use katgpt_rs::pruners::go::AutoGoClient;
///
/// let client = AutoGoClient::new("http://localhost:8000");
/// let agents = client.list_agents().unwrap();
/// let state = client.new_game(9, "black", "random").unwrap();
/// println!("Game {} started, legal moves: {}", state.game_id, state.legal_moves.len());
/// ```
pub struct AutoGoClient {
    base_url: String,
    client: reqwest::blocking::Client,
}

/// Response from AutoGo's `GameState` model.
///
/// Field names match the actual API response from `play.py:GameState`.
/// Note: `human_color` (not `color`) — the API uses "human" to mean "the REST client".
#[derive(Clone, Debug, Deserialize)]
pub struct AutoGoGameState {
    /// Unique game identifier.
    pub game_id: String,
    /// Board state as 2D array: 0=empty, 1=black, 2=white.
    pub board: Vec<Vec<i8>>,
    /// Board dimension (e.g. 9 for 9×9).
    pub size: usize,
    /// Current player to move: 1=BLACK, 2=WHITE.
    pub to_play: i8,
    /// Last move played as (row, col), or None if game start / pass.
    pub last_move: Option<(usize, usize)>,
    /// Whether the game has ended.
    pub is_over: bool,
    /// Game result in standard notation, e.g. `"W+2.5"`, `"B+1"`.
    pub result: Option<String>,
    /// Legal moves available as (row, col) pairs.
    pub legal_moves: Vec<(usize, usize)>,
    /// Our color (the REST client): 1=BLACK, 2=WHITE.
    pub human_color: i8,
    /// Status message from the server.
    pub message: String,
}

/// Move request matching AutoGo's `MoveRequest` model.
#[derive(Serialize)]
struct MoveRequest {
    row: Option<usize>,
    col: Option<usize>,
    pass_move: bool,
}

impl AutoGoClient {
    /// Create a new client pointing at the AutoGo server.
    ///
    /// `base_url` should include scheme and port, e.g. `"http://localhost:8000"`.
    pub fn new(base_url: &str) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// List available AI agents on the AutoGo server.
    ///
    /// Typical agents: `"random"`, `"gnugo"`, `"gtp"`.
    pub fn list_agents(&self) -> Result<Vec<String>> {
        let url = format!("{}/api/agents", self.base_url);
        log::debug!("GET {url}");
        let response = self.client.get(&url).send().map_err(AutoGoError::Http)?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(AutoGoError::Status(status.as_u16(), body));
        }
        response.json::<Vec<String>>().map_err(AutoGoError::Parse)
    }

    /// Start a new game against an AutoGo agent.
    ///
    /// ## Arguments
    /// * `size` — Board dimension (9, 13, or 19).
    /// * `color` — Our color: `"black"` or `"white"`.
    /// * `agent` — Agent name from [`AutoGoClient::list_agents`].
    ///
    /// If we play White, AutoGo plays Black's first move immediately.
    pub fn new_game(&self, size: usize, color: &str, agent: &str) -> Result<AutoGoGameState> {
        let color = color.to_lowercase();
        let agent = agent.to_lowercase();
        let url = format!(
            "{}/api/new_game?size={size}&color={color}&agent={agent}",
            self.base_url
        );
        log::debug!("POST {url}");
        let response = self.client.post(&url).send().map_err(AutoGoError::Http)?;
        Self::parse_response(response)
    }

    /// Get the current state of an existing game.
    pub fn get_game(&self, game_id: &str) -> Result<AutoGoGameState> {
        let url = format!("{}/api/game/{game_id}", self.base_url);
        log::debug!("GET {url}");
        let response = self.client.get(&url).send().map_err(AutoGoError::Http)?;
        Self::parse_response(response)
    }

    /// Play a stone move. Response includes BOTH our move AND AI's response (G2).
    ///
    /// One HTTP call = two Go moves. The returned state has both applied.
    pub fn make_move(&self, game_id: &str, row: usize, col: usize) -> Result<AutoGoGameState> {
        let url = format!("{}/api/game/{game_id}/move", self.base_url);
        let body = MoveRequest {
            row: Some(row),
            col: Some(col),
            pass_move: false,
        };
        log::debug!("POST {url} — move ({row},{col})");
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(AutoGoError::Http)?;
        Self::parse_response(response)
    }

    /// Pass turn. Uses same endpoint with `pass_move=true` (matches API).
    ///
    /// Two consecutive passes end the game.
    pub fn pass_move(&self, game_id: &str) -> Result<AutoGoGameState> {
        let url = format!("{}/api/game/{game_id}/move", self.base_url);
        let body = MoveRequest {
            row: None,
            col: None,
            pass_move: true,
        };
        log::debug!("POST {url} — pass");
        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(AutoGoError::Http)?;
        Self::parse_response(response)
    }

    /// Parse an API response into [`AutoGoGameState`], handling errors.
    fn parse_response(response: reqwest::blocking::Response) -> Result<AutoGoGameState> {
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            return Err(AutoGoError::Status(status.as_u16(), body));
        }
        response
            .json::<AutoGoGameState>()
            .map_err(AutoGoError::Parse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autogo_error_display_formats() {
        let err = AutoGoError::Status(404, "not found".to_string());
        assert_eq!(format!("{err}"), "API error 404: not found");
    }

    #[test]
    fn client_new_trims_trailing_slash() {
        let client = AutoGoClient::new("http://localhost:8000/");
        assert_eq!(client.base_url, "http://localhost:8000");
    }
}
