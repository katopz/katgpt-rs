//! SonltPlayer — SON-LT (Sparse OPD-Native LoRA Training) arena player.
//!
//! Plan 296 Phase 7 T7.1: loads a SON-LT trained multi-adapter LoRA file
//! (6 adapters: q, k, v, o, mlp1, mlp2) and runs a full `Config::game()`
//! Transformer forward pass over the 169-cell board encoding to predict
//! the next Bomberman action.
//!
//! Training format (matches `riir_gpu::game`):
//! - Input sequence: 169 board cell tokens (values 0-3) + 1 action token (4-9).
//! - Training layout: `input = tokens[0..169]`, `target = tokens[1..170]`, so
//!   the model at position 168 (last board cell) predicts the action token.
//!   The action-prediction logits live at position 168, NOT position 169
//!   (see Issue 306 root-cause fix in `predict_action`).
//! - Board token map: Floor=0, FixedWall=1, DestructibleWall=2, PowerUpHidden=3.
//! - Action token map: GameAction value (0-5) + BOARD_VOCAB(=4) → logits[4..10].
//!
//! Forward pass mirrors `forward_drafter_with_lora` (drafter_lora.rs L286-422):
//! 1. Embed token + position.
//! 2. RMSNorm (pre-attention, twice — matches forward_base).
//! 3. QKV projections with `lora_apply`.
//! 4. Multi-head attention with GQA over KV cache.
//! 5. Output projection with `lora_apply` + residual.
//! 6. MLP w1 (ReLU) + w2 with `lora_apply` + residual.
//! 7. LM head → logits (10-dim vocab).
//!
//! Safety: the model's argmax action is filtered through `is_safe_action`,
//! falling back to heuristic `score_action` when the model's pick is unsafe
//! or when LoRA failed to load. 10% epsilon-greedy exploration matches
//! `LoraPlayer`.

use std::any::Any;

use fastrand::Rng;

use crate::transformer::TransformerWeights;
use crate::types::{
    Config, LoraAdapter, Rng as CrateRng, kv_dim, lora_apply, matmul, matmul_relu, rmsnorm,
};

use super::arena::ArenaGrid;
use super::players::{
    ACTION_COUNT, ALL_ACTIONS, KnownBomb, BomberPlayer, in_blast_zone, is_safe_action,
    move_target, score_action, update_bombs, update_powerups,
};
use super::{BomberAction, Cell, GameEvent, GridPos};

// ── Constants ──────────────────────────────────────────────────

/// Board vocab size: 4 cell kinds (Floor, FixedWall, DestructibleWall, PowerUpHidden).
const BOARD_VOCAB: usize = 4;

/// Number of board cells in a 13×13 arena.
const BOARD_CELLS: usize = 169;

/// Action vocab size: Up/Down/Left/Right/Bomb/Wait (no Detonate — model only sees 6).
const ACTION_VOCAB: usize = 6;

/// Epsilon-greedy exploration rate (matches LoraPlayer).
const EPSILON: f32 = 0.10;

// ── SonltPlayer ────────────────────────────────────────────────

/// SON-LT LoRA Bomber player.
///
/// Holds 6 LoRA adapters + a base `Config::game()` Transformer. The base
/// weights are random — the LoRA delta is what carries learned behavior
/// (SON-LT trains only the adapters, freezing the base). The forward pass
/// applies all 6 adapters inline during a single-layer Transformer pass.
///
/// Falls back to heuristic `score_action` when:
/// - Adapter file fails to load (all `lora_*` fields `None`).
/// - Adapter count ≠ 6.
/// - Adapter dimensions don't match `Config::game()`.
/// - The model's argmax action is unsafe (filtered by `is_safe_action`).
pub struct SonltPlayer {
    _id: u8,
    // LoRA adapters (None when load failed or dims mismatch).
    lora_q: Option<LoraAdapter>,
    lora_k: Option<LoraAdapter>,
    lora_v: Option<LoraAdapter>,
    lora_o: Option<LoraAdapter>,
    lora_mlp1: Option<LoraAdapter>,
    lora_mlp2: Option<LoraAdapter>,
    // Base Transformer (random init — LoRA carries the learned delta).
    weights: TransformerWeights,
    config: Config,
    // Forward pass scratch buffers (pre-allocated, reused across calls).
    lora_buf: Vec<f32>,
    x: Vec<f32>,
    xr: Vec<f32>,
    xr2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    hidden: Vec<f32>,
    logits: Vec<f32>,
    key_cache: Vec<f32>,
    value_cache: Vec<f32>,
    // Game state tracking.
    known_bombs: Vec<KnownBomb>,
    known_powerups: Vec<(i32, i32)>,
    last_dir: Option<BomberAction>,
}

impl SonltPlayer {
    /// Create a SonltPlayer with LoRA loaded from a SON-LT adapter file.
    ///
    /// The file must contain exactly 6 adapters in order: q, k, v, o, mlp1, mlp2.
    /// On any failure (missing file, wrong count, dim mismatch), falls back to
    /// heuristic-only mode — the player still works, just without the LoRA delta.
    pub fn new_with_lora(id: u8, lora_path: &str) -> Self {
        let config = Config::game();
        let mut rng = CrateRng::new(0xC0FFEE);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Attempt to load the multi-adapter file.
        let loaded = LoraAdapter::load(std::path::Path::new(lora_path))
            .ok()
            .filter(|v| v.len() == 6);

        let (lq, lk, lv, lo, lm1, lm2) = match loaded {
            Some(v) => {
                // Validate each adapter's in/out dims match Config::game() projections.
                let n = config.n_embd;
                let kvd = kv_dim(&config);
                let mlp = config.mlp_hidden;
                // (adapter, expected_in, expected_out)
                let checks: [(Option<&LoraAdapter>, usize, usize); 6] = [
                    (Some(&v[0]), n, n),    // q
                    (Some(&v[1]), n, kvd),  // k
                    (Some(&v[2]), n, kvd),  // v
                    (Some(&v[3]), n, n),    // o
                    (Some(&v[4]), n, mlp),  // mlp1
                    (Some(&v[5]), mlp, n),  // mlp2
                ];
                let dims_ok = checks
                    .iter()
                    .all(|(a, ein, eout)| {
                        a.map(|ad| ad.in_dim == *ein && ad.out_dim == *eout)
                            .unwrap_or(false)
                    });
                if dims_ok {
                    let mut it = v.into_iter();
                    let q = it.next().unwrap();
                    let k = it.next().unwrap();
                    let vv = it.next().unwrap();
                    let o = it.next().unwrap();
                    let m1 = it.next().unwrap();
                    let m2 = it.next().unwrap();
                    (Some(q), Some(k), Some(vv), Some(o), Some(m1), Some(m2))
                } else {
                    eprintln!(
                        "SonltPlayer: adapter dims mismatch — falling back to heuristic"
                    );
                    (None, None, None, None, None, None)
                }
            }
            None => {
                eprintln!(
                    "SonltPlayer: LoRA load failed or wrong adapter count — heuristic mode"
                );
                (None, None, None, None, None, None)
            }
        };

        let rank = lq.as_ref().map_or(0, |a| a.rank);
        let n = config.n_embd;
        let kvd = kv_dim(&config);
        let block_size = config.block_size;
        let mlp_hidden = config.mlp_hidden;
        let vocab_size = config.vocab_size;

        Self {
            _id: id,
            lora_q: lq,
            lora_k: lk,
            lora_v: lv,
            lora_o: lo,
            lora_mlp1: lm1,
            lora_mlp2: lm2,
            weights,
            config,
            // Scratch buffers sized for Config::game().
            lora_buf: vec![0.0; rank],
            x: vec![0.0; n],
            xr: vec![0.0; n],
            xr2: vec![0.0; n],
            q: vec![0.0; n],
            k: vec![0.0; kvd],
            v: vec![0.0; kvd],
            attn_out: vec![0.0; n],
            scores: vec![0.0; block_size],
            hidden: vec![0.0; mlp_hidden],
            logits: vec![0.0; vocab_size],
            key_cache: vec![0.0; block_size * kvd],
            value_cache: vec![0.0; block_size * kvd],
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            last_dir: None,
        }
    }

    /// Returns true if all 6 LoRA adapters loaded successfully.
    fn lora_active(&self) -> bool {
        self.lora_q.is_some()
    }

    /// Encode the arena board into 169 tokens (values 0-3).
    ///
    /// Iterates cells in row-major order (y outer, x inner) to match the
    /// training-time serialization in `serialize_board`.
    fn encode_board(&self, grid: &ArenaGrid) -> Vec<usize> {
        let mut tokens = Vec::with_capacity(BOARD_CELLS);
        for y in 0..grid.height {
            for x in 0..grid.width {
                let t = match grid.cells[y][x] {
                    Cell::Floor => 0,
                    Cell::FixedWall => 1,
                    Cell::DestructibleWall => 2,
                    Cell::PowerUpHidden(_) => 3,
                };
                tokens.push(t);
            }
        }
        // Pad/truncate to exactly BOARD_CELLS in case arena isn't 13×13.
        tokens.resize(BOARD_CELLS, 0);
        tokens
    }

    /// Run the LoRA-augmented forward pass over the board and predict an action.
    ///
    /// Feeds all 169 board tokens through the Transformer (building KV cache).
    /// The logits produced at the **last board position** (position 168) are
    /// the action predictions — during training, `target[168] = action_token`
    /// (the shifted-target layout: `input = tokens[0..169]`, `target =
    /// tokens[1..170]`). Reads `logits[BOARD_VOCAB..BOARD_VOCAB+ACTION_VOCAB]`
    /// and returns the argmax mapped back to a `BomberAction`.
    ///
    /// # Issue 306 root-cause fix (2026-06-28)
    ///
    /// Previously this function did an EXTRA forward at position 169 (with
    /// BOS token 0) and read those logits. That was a train/inference
    /// mismatch: training only runs positions 0..=168 (seq_len=169), so
    /// position 169 was never trained — its logits were essentially random,
    /// which is why ANY trained LoRA made the player worse (0% survival).
    /// The fix reads logits from position 168 (the last board cell), which
    /// IS the action-prediction slot the model was trained on.
    ///
    /// Returns `None` if LoRA is not active.
    fn predict_action(&mut self, grid: &ArenaGrid) -> Option<BomberAction> {
        if !self.lora_active() {
            return None;
        }

        let tokens = self.encode_board(grid);
        // Reset KV cache for a fresh sequence.
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);

        // Forward all 169 board tokens (positions 0..=168). After the loop,
        // `self.logits` holds the position-168 logits, which predict the
        // action token (target[168] in the shifted-target training layout).
        // No extra forward at position 169 — see Issue 306 root-cause note above.
        for (pos, &token) in tokens.iter().enumerate() {
            forward_game_with_lora(
                &self.config,
                &self.weights,
                self.lora_q.as_ref()?,
                self.lora_k.as_ref()?,
                self.lora_v.as_ref()?,
                self.lora_o.as_ref()?,
                self.lora_mlp1.as_ref()?,
                self.lora_mlp2.as_ref()?,
                token,
                pos,
                &mut self.lora_buf,
                &mut self.x,
                &mut self.xr,
                &mut self.xr2,
                &mut self.q,
                &mut self.k,
                &mut self.v,
                &mut self.attn_out,
                &mut self.scores,
                &mut self.hidden,
                &mut self.logits,
                &mut self.key_cache,
                &mut self.value_cache,
            );
        }

        // Argmax over action logits [4..10). No softmax (project rule).
        let action_logits = &self.logits[BOARD_VOCAB..BOARD_VOCAB + ACTION_VOCAB];
        let best_idx = action_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Map GameAction (0-5) → BomberAction. Detonate (6) not in model vocab.
        Some(game_action_to_bomber(best_idx))
    }
}

impl BomberPlayer for SonltPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);

        // Step 1: try the LoRA model's prediction.
        let model_action = self.predict_action(grid);

        // Step 2: safety filter. If the model's pick is unsafe, fall back to
        // heuristic scoring over all actions.
        let mut best = match model_action {
            Some(a) if is_safe_action(&a, grid, pos, &self.known_bombs) => a,
            _ => {
                // Heuristic fallback: score all actions, pick the best safe one.
                let mut heur_best = BomberAction::Wait;
                let mut heur_score = f32::NEG_INFINITY;
                for action in ALL_ACTIONS.iter() {
                    let s = score_action(
                        action,
                        grid,
                        pos,
                        &self.known_bombs,
                        &self.known_powerups,
                        self.last_dir,
                    );
                    if s > heur_score {
                        heur_score = s;
                        heur_best = *action;
                    }
                }
                heur_best
            }
        };

        // Step 3: epsilon-greedy exploration (matches LoraPlayer).
        if rng.f32() < EPSILON {
            let safe_moves: Vec<BomberAction> = ALL_ACTIONS
                .iter()
                .copied()
                .filter(|a| {
                    let is_move = matches!(
                        a,
                        BomberAction::Up
                            | BomberAction::Down
                            | BomberAction::Left
                            | BomberAction::Right
                    );
                    if !is_move {
                        return false;
                    }
                    let target = move_target(a, pos);
                    grid.is_walkable(target.x, target.y)
                        && !in_blast_zone(target, grid, &self.known_bombs)
                })
                .collect();
            if !safe_moves.is_empty() {
                best = safe_moves[rng.usize(0..safe_moves.len())];
            }
        }

        // Track last direction + own bomb placement.
        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        if best == BomberAction::Bomb {
            self.known_bombs.push((
                (pos.x, pos.y),
                super::DEFAULT_BLAST_RANGE,
                super::BOMB_FUSE_TICKS,
            ));
        }

        // Touch ACTION_COUNT to avoid unused-import warning in some builds.
        let _ = ACTION_COUNT;
        best
    }

    fn name(&self) -> &str {
        if self.lora_active() {
            "SON-LT"
        } else {
            "SON-LT-Fallback"
        }
    }

    fn emoji(&self) -> &str {
        "🧠"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.last_dir = None;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Map a GameAction index (0-5) to a BomberAction.
///
/// GameAction: Up=0, Down=1, Left=2, Right=3, Bomb=4, Wait=5.
/// BomberAction adds Detonate=6, which the model cannot emit.
fn game_action_to_bomber(idx: usize) -> BomberAction {
    match idx {
        0 => BomberAction::Up,
        1 => BomberAction::Down,
        2 => BomberAction::Left,
        3 => BomberAction::Right,
        4 => BomberAction::Bomb,
        _ => BomberAction::Wait, // 5 or out-of-range → Wait
    }
}

// ── Forward pass with LoRA ─────────────────────────────────────

/// Single-layer Transformer forward pass with 6 LoRA adapters applied inline.
///
/// Mirrors `forward_drafter_with_lora` (drafter_lora.rs L286-422) exactly,
/// but takes the 6 adapters as individual `Option<&LoraAdapter>` refs so the
/// caller can selectively disable them. Here all 6 are always `Some` when
/// `lora_active()` is true.
///
/// No softmax — attention uses the standard scaled-dot-product softmax
/// internally (that's part of attention, not action selection). Action
/// selection uses argmax over logits (project rule: sigmoid not softmax for
/// independent scores, but argmax is equivalent for picking the top action).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn forward_game_with_lora(
    config: &Config,
    weights: &TransformerWeights,
    lora_q: &LoraAdapter,
    lora_k: &LoraAdapter,
    lora_v: &LoraAdapter,
    lora_o: &LoraAdapter,
    lora_mlp1: &LoraAdapter,
    lora_mlp2: &LoraAdapter,
    token: usize,
    pos: usize,
    lora_buf: &mut [f32],
    x: &mut [f32],
    xr: &mut [f32],
    xr2: &mut [f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    attn_out: &mut [f32],
    scores: &mut [f32],
    hidden: &mut [f32],
    logits: &mut [f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
) {
    let n = config.n_embd;
    let hd = config.head_dim;
    let kvd = kv_dim(config);
    let n_kv = config.n_kv_head;
    let layer_weights = &weights.layers[0];

    // 1. Embedding: x = wte[token] + wpe[pos]
    let tok_off = token * n;
    let pos_off = pos * n;
    for i in 0..n {
        x[i] = weights.wte[tok_off + i] + weights.wpe[pos_off + i];
    }

    // 2. Pre-attention: RMSNorm → save residual → RMSNorm (matches forward_base)
    rmsnorm(&mut x[..n]);
    xr[..n].copy_from_slice(&x[..n]);
    rmsnorm(&mut x[..n]);

    // 3. QKV projections with LoRA
    matmul(q, &layer_weights.attn_wq, &x[..n], n, n);
    lora_apply(q, lora_q, &x[..n], lora_buf);

    matmul(k, &layer_weights.attn_wk, &x[..n], kvd, n);
    lora_apply(k, lora_k, &x[..n], lora_buf);

    matmul(v, &layer_weights.attn_wv, &x[..n], kvd, n);
    lora_apply(v, lora_v, &x[..n], lora_buf);

    // 4. Store K,V in per-position cache
    let pos_off_cache = pos * kvd;
    key_cache[pos_off_cache..pos_off_cache + kvd].copy_from_slice(&k[..kvd]);
    value_cache[pos_off_cache..pos_off_cache + kvd].copy_from_slice(&v[..kvd]);

    // 5. Multi-head attention with GQA
    let scale = 1.0 / (hd as f32).sqrt();
    attn_out[..n].fill(0.0);
    let t_n = pos + 1;

    for h in 0..config.n_head {
        let kv_group = h * n_kv / config.n_head;
        let q_off = h * hd;
        let kv_off = kv_group * hd;

        // Pass 1: compute scores, find max
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..t_n {
            let k_off = t * kvd + kv_off;
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += q[q_off + d] * key_cache[k_off + d];
            }
            let score = dot * scale;
            scores[t] = score;
            max_score = max_score.max(score);
        }

        // Pass 2: exp and accumulate sum
        let mut sum = 0.0f32;
        for t in 0..t_n {
            scores[t] = (scores[t] - max_score).exp();
            sum += scores[t];
        }
        let inv_sum = 1.0 / sum;

        // Pass 3: weighted value accumulation
        for d in 0..hd {
            let mut val = 0.0f32;
            for t in 0..t_n {
                val += scores[t] * inv_sum * value_cache[t * kvd + kv_off + d];
            }
            attn_out[q_off + d] = val;
        }
    }

    // 6. Output projection with LoRA + residual
    matmul(&mut x[..n], &layer_weights.attn_wo, &attn_out[..n], n, n);
    lora_apply(&mut x[..n], lora_o, &attn_out[..n], lora_buf);

    for i in 0..n {
        x[i] += xr[i];
    }

    // 7. MLP: save residual → RMSNorm → MLP with LoRA → residual
    xr2[..n].copy_from_slice(&x[..n]);
    rmsnorm(&mut x[..n]);

    // MLP w1 with ReLU + LoRA
    matmul_relu(hidden, &layer_weights.mlp_w1, &x[..n], config.mlp_hidden, n);
    lora_apply(hidden, lora_mlp1, &x[..n], lora_buf);

    // MLP w2 + LoRA
    matmul(
        &mut x[..n],
        &layer_weights.mlp_w2,
        &hidden[..config.mlp_hidden],
        n,
        config.mlp_hidden,
    );
    lora_apply(
        &mut x[..n],
        lora_mlp2,
        &hidden[..config.mlp_hidden],
        lora_buf,
    );

    // Residual
    for i in 0..n {
        x[i] += xr2[i];
    }

    // 8. LM Head
    matmul(logits, &weights.lm_head, &x[..n], config.vocab_size, n);
}
