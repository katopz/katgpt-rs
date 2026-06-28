//! FftLoRAPlayer — SON-LT (Sparse OPD-Native LoRA Training) arena player for FFT.
//!
//! Plan 296 Phase 7 T7.3: loads a SON-LT trained multi-adapter LoRA file
//! (6 adapters: q, k, v, o, mlp1, mlp2) and runs a full `Config::game_fft()`
//! Transformer forward pass over the 57-token battle state to predict the
//! next action type.
//!
//! # Action Decoding
//!
//! The model only learns `ActionType` prediction (9 tokens). Target_id and
//! move_to are filled by heuristic decoders (mirror the Bomber `SonltPlayer`
//! pattern — model predicts WHAT to do, heuristics decide WHO/WHERE):
//! - `target_id` — weakest enemy for offensive actions, lowest-HP ally for
//!   support actions.
//! - `move_to` — toward nearest enemy for melee, away for retreat.
//!
//! Falls back to a safety-filtered greedy heuristic when:
//! - The LoRA adapter file fails to load (all `lora_*` fields `None`).
//! - Adapter count ≠ 6 or dimensions mismatch `Config::game_fft()`.
//! - The model's predicted action is not affordable (e.g., not enough MP).
//!
//! # Architecture
//!
//! Mirrors `bomber::sonlt_player::SonltPlayer` exactly:
//! - `Config::game_fft()` (vocab=19, seq=58, n_embd=32, n_layer=1, rank=4).
//! - Single Transformer layer with all 6 LoRA adapters applied inline.
//! - Argmax over `logits[FFT_ACTION_OFFSET..FFT_ACTION_OFFSET+9]`.

use std::any::Any;
use std::path::Path;

use fastrand::Rng;

use crate::transformer::TransformerWeights;
use crate::types::{
    Config, LoraAdapter, Rng as CrateRng, kv_dim, lora_apply, matmul, matmul_relu, rmsnorm,
};

use super::battle::BattleState;
use super::players::{FftPlayer, move_away, move_toward, nearest_enemy_pos, weakest_target};
use super::replay_encode::{FFT_STATE_LEN, encode_battle_state};
use super::types::{Action, ActionType};

// ── Constants ────────────────────────────────────────────────────

/// State vocab (matches `riir_gpu::game::fft_replay::FFT_STATE_VOCAB`).
const STATE_VOCAB: usize = 10;

/// Action offset (10) — first 9 action tokens live at logits[10..19].
const ACTION_OFFSET: usize = STATE_VOCAB;

/// Number of action types (9).
const ACTION_COUNT: usize = 9;

/// Epsilon-greedy exploration rate (matches Bomber SonltPlayer).
const EPSILON: f32 = 0.10;

// ── FftLoRAPlayer ────────────────────────────────────────────────

/// SON-LT LoRA FFT player.
///
/// Holds 6 LoRA adapters + a base `Config::game_fft()` Transformer. The base
/// weights are random — the LoRA delta carries learned behavior (SON-LT trains
/// only the adapters, freezing the base). The forward pass applies all 6
/// adapters inline during a single-layer Transformer pass.
pub struct FftLoRAPlayer {
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
    // Reusable state encoding buffer.
    state_tokens: [u8; FFT_STATE_LEN],
    lora_loaded: bool,
}

impl FftLoRAPlayer {
    /// Create a FftLoRAPlayer with LoRA loaded from a SON-LT adapter file.
    ///
    /// The file must contain exactly 6 adapters in order: q, k, v, o, mlp1, mlp2.
    /// On any failure (missing file, wrong count, dim mismatch), falls back to
    /// heuristic-only mode — the player still works, just without the LoRA delta.
    pub fn new_with_lora(id: u8, lora_path: &Path) -> Self {
        let config = Config::game_fft();
        let mut rng = CrateRng::new(0xC0FFEE);
        let weights = TransformerWeights::new(&config, &mut rng);

        // Attempt to load the multi-adapter file.
        let loaded = LoraAdapter::load(lora_path)
            .ok()
            .filter(|v| v.len() == 6);

        let (lq, lk, lv, lo, lm1, lm2, lora_loaded) = match loaded {
            Some(v) => {
                // Validate each adapter's in/out dims match Config::game_fft() projections.
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
                let dims_ok = checks.iter().all(|(a, ein, eout)| {
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
                    (Some(q), Some(k), Some(vv), Some(o), Some(m1), Some(m2), true)
                } else {
                    eprintln!(
                        "FftLoRAPlayer: adapter dims mismatch — falling back to heuristic"
                    );
                    (None, None, None, None, None, None, false)
                }
            }
            None => {
                eprintln!(
                    "FftLoRAPlayer: LoRA load failed or wrong adapter count — heuristic mode ({})",
                    lora_path.display()
                );
                (None, None, None, None, None, None, false)
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
            state_tokens: [0u8; FFT_STATE_LEN],
            lora_loaded,
        }
    }

    /// Create a FftLoRAPlayer in heuristic-only mode (no LoRA file).
    /// Useful for baseline comparisons.
    pub fn new_heuristic(id: u8) -> Self {
        let dummy = Path::new("/nonexistent/fft_lora_heuristic_only.bin");
        Self::new_with_lora(id, dummy)
    }

    /// Returns true if all 6 LoRA adapters loaded successfully.
    pub fn lora_active(&self) -> bool {
        self.lora_loaded
    }

    /// Run the LoRA-augmented forward pass over the battle state and predict
    /// the next action type.
    ///
    /// # Issue 306 root-cause fix (2026-06-28, mirrors Bomber `SonltPlayer`)
    ///
    /// Previously this function fed all 57 state tokens (positions 0..=56),
    /// then did an EXTRA forward at position 57 (with BOS token 0) and read
    /// those logits. That was a train/inference mismatch: training only runs
    /// positions 0..=56 (seq_len = FFT_GAME_SEQ_LEN - 1 = 57, shifted-target
    /// layout: input=tokens[0..57], target=tokens[1..58]). Position 57 was
    /// never trained — its logits were essentially random, which is why any
    /// trained LoRA underperformed (T7.3 smoke: 16% Party win rate).
    ///
    /// The fix reads logits from position 56 (the last state cell), which IS
    /// the action-prediction slot the model was trained on (target[56] =
    /// action token at sequence position 57).
    ///
    /// Returns `None` if LoRA is not active.
    fn predict_action_type(&mut self, state: &BattleState) -> Option<ActionType> {
        if !self.lora_active() {
            return None;
        }

        // Encode battle state into the 57-token buffer.
        encode_battle_state(state, &mut self.state_tokens);

        // Reset KV cache for a fresh sequence.
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);

        // Forward all 57 state tokens (positions 0..=56). After the loop,
        // `self.logits` holds the position-56 logits, which predict the
        // action token (target[56] in the shifted-target training layout).
        // No extra forward at position 57 — see Issue 306 root-cause note above.
        for (pos, &token) in self.state_tokens.iter().enumerate() {
            forward_fft_with_lora(
                &self.config,
                &self.weights,
                self.lora_q.as_ref()?,
                self.lora_k.as_ref()?,
                self.lora_v.as_ref()?,
                self.lora_o.as_ref()?,
                self.lora_mlp1.as_ref()?,
                self.lora_mlp2.as_ref()?,
                token as usize,
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

        // Argmax over action logits [10..19). No softmax (project rule).
        let best_idx = self.logits[ACTION_OFFSET..ACTION_OFFSET + ACTION_COUNT]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Decode action token → ActionType.
        let action = ActionType::from(best_idx);
        Some(action)
    }

    /// Heuristic fallback when LoRA is inactive or the predicted action is
    /// invalid/unaffordable. Implements a compact greedy policy:
    /// - Critical HP → Potion if available.
    /// - Low-HP ally in range and can heal → WhiteMagic.
    /// - Enemy in range and can attack → Attack (or BlackMagic if affordable).
    /// - Else → Defend (or move toward nearest enemy).
    fn heuristic_action(&self, unit_id: u8, state: &BattleState) -> Action {
        let unit = &state.units[unit_id as usize];
        let hp_pct = unit.hp_pct();
        let reachable = state.reachable_positions(unit_id);
        let enemy_team = BattleState::enemy_team(unit.team);

        let move_to = nearest_enemy_pos(state, unit.pos, unit.team)
            .and_then(|ep| move_toward(&reachable, ep));

        // Critical HP → potion.
        if hp_pct < 0.3 && unit.can_afford(ActionType::Potion) {
            return Action {
                action_type: ActionType::Potion,
                target_id: Some(unit_id),
                move_to,
            };
        }

        // Heal low-HP ally.
        let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
        if unit.can_afford(ActionType::WhiteMagic) {
            for &ally in &allies {
                if state.units[ally as usize].hp_pct() < 0.4 {
                    return Action {
                        action_type: ActionType::WhiteMagic,
                        target_id: Some(ally),
                        move_to,
                    };
                }
            }
        }

        // Attack weakest enemy in range.
        let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
        if !enemies.is_empty() {
            let target = weakest_target(state, &enemies);
            if unit.can_afford(ActionType::BlackMagic) {
                return Action {
                    action_type: ActionType::BlackMagic,
                    target_id: target,
                    move_to,
                };
            }
            return Action {
                action_type: ActionType::Attack,
                target_id: target,
                move_to,
            };
        }

        // Retreat if low HP; else move toward enemy.
        let move_to = if hp_pct < 0.5 {
            nearest_enemy_pos(state, unit.pos, unit.team).and_then(|ep| move_away(&reachable, ep))
        } else {
            move_to
        };

        Action {
            action_type: ActionType::Defend,
            target_id: None,
            move_to,
        }
    }

    /// Given a predicted ActionType, fill in target_id and move_to via the
    /// heuristic decoders. If the predicted action is not affordable, returns
    /// None (caller falls back to `heuristic_action`).
    fn decode_action(
        &self,
        action_type: ActionType,
        unit_id: u8,
        state: &BattleState,
    ) -> Option<Action> {
        let unit = &state.units[unit_id as usize];
        if !unit.can_afford(action_type) {
            return None;
        }

        let enemy_team = BattleState::enemy_team(unit.team);
        let reachable = state.reachable_positions(unit_id);
        let move_to = nearest_enemy_pos(state, unit.pos, unit.team)
            .and_then(|ep| move_toward(&reachable, ep));

        let target_id = match action_type {
            ActionType::Attack | ActionType::BlackMagic => {
                // Target the weakest enemy in range.
                let enemies = state.targets_in_range(unit.pos, unit.stats.range, enemy_team);
                if enemies.is_empty() {
                    // No target in range — fall back to heuristic (which may move).
                    return None;
                }
                weakest_target(state, &enemies)
            }
            ActionType::WhiteMagic | ActionType::Esuna | ActionType::CurePoison => {
                // Target the lowest-HP ally in range (or self).
                let allies = state.targets_in_range(unit.pos, unit.stats.range, unit.team);
                allies
                    .iter()
                    .min_by_key(|&&id| state.units[id as usize].hp)
                    .copied()
                    .or(Some(unit_id))
            }
            ActionType::Potion => Some(unit_id),
            ActionType::Dispel => {
                // Target nearest enemy (for debuff removal on enemy).
                state
                    .targets_in_range(unit.pos, unit.stats.range, enemy_team)
                    .first()
                    .copied()
            }
            ActionType::Defend | ActionType::Wait => None,
        };

        Some(Action {
            action_type,
            target_id,
            move_to,
        })
    }
}

impl FftPlayer for FftLoRAPlayer {
    fn select_action(&mut self, unit_id: u8, state: &BattleState, rng: &mut Rng) -> Action {
        // Step 1: try the LoRA model's prediction.
        let predicted = if self.lora_active() {
            self.predict_action_type(state)
                .and_then(|action_type| self.decode_action(action_type, unit_id, state))
        } else {
            None
        };

        // Step 2: fall back to heuristic when the model's pick is invalid.
        let action = predicted.unwrap_or_else(|| self.heuristic_action(unit_id, state));

        // Step 3: epsilon-greedy exploration. Occasionally override with Wait
        // (a safe no-op) to avoid determinism exploitation by opponents.
        if rng.f32() < EPSILON {
            // Pick a random affordable action.
            let candidates: [ActionType; 5] = [
                ActionType::Attack,
                ActionType::Defend,
                ActionType::Wait,
                ActionType::BlackMagic,
                ActionType::WhiteMagic,
            ];
            let unit = &state.units[unit_id as usize];
            for &candidate in candidates.iter() {
                if unit.can_afford(candidate) {
                    return Action {
                        action_type: candidate,
                        target_id: None,
                        move_to: None,
                    };
                }
            }
        }

        action
    }

    fn name(&self) -> &'static str {
        if self.lora_active() {
            "FFT-SON-LT"
        } else {
            "FFT-SON-LT-Fallback"
        }
    }

    fn reset(&mut self) {
        // No per-game state to reset (KV cache is cleared on each forward pass).
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ── Forward pass with LoRA ───────────────────────────────────────

/// Single-layer Transformer forward pass with 6 LoRA adapters applied inline.
///
/// Mirrors `bomber::sonlt_player::forward_game_with_lora` exactly — the
/// Transformer is generic over input domains. The only FFT-specific aspect is
/// the Config passed in (`Config::game_fft()`).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn forward_fft_with_lora(
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

    // 1. Embedding: x = wte[token] + wpe[pos].
    let tok_off = token * n;
    let pos_off = pos * n;
    for i in 0..n {
        x[i] = weights.wte[tok_off + i] + weights.wpe[pos_off + i];
    }

    // 2. Pre-attention: RMSNorm → save residual → RMSNorm.
    rmsnorm(&mut x[..n]);
    xr[..n].copy_from_slice(&x[..n]);
    rmsnorm(&mut x[..n]);

    // 3. QKV projections with LoRA.
    matmul(q, &layer_weights.attn_wq, &x[..n], n, n);
    lora_apply(q, lora_q, &x[..n], lora_buf);

    matmul(k, &layer_weights.attn_wk, &x[..n], kvd, n);
    lora_apply(k, lora_k, &x[..n], lora_buf);

    matmul(v, &layer_weights.attn_wv, &x[..n], kvd, n);
    lora_apply(v, lora_v, &x[..n], lora_buf);

    // 4. Store K,V in per-position cache.
    let pos_off_cache = pos * kvd;
    key_cache[pos_off_cache..pos_off_cache + kvd].copy_from_slice(&k[..kvd]);
    value_cache[pos_off_cache..pos_off_cache + kvd].copy_from_slice(&v[..kvd]);

    // 5. Multi-head attention with GQA.
    let scale = 1.0 / (hd as f32).sqrt();
    attn_out[..n].fill(0.0);
    let t_n = pos + 1;

    for h in 0..config.n_head {
        let kv_group = h * n_kv / config.n_head;
        let q_off = h * hd;
        let kv_off = kv_group * hd;

        // Pass 1: compute scores, find max.
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

        // Pass 2: exp and accumulate sum.
        let mut sum = 0.0f32;
        for t in 0..t_n {
            scores[t] = (scores[t] - max_score).exp();
            sum += scores[t];
        }
        let inv_sum = 1.0 / sum;

        // Pass 3: weighted value accumulation.
        for d in 0..hd {
            let mut val = 0.0f32;
            for t in 0..t_n {
                val += scores[t] * inv_sum * value_cache[t * kvd + kv_off + d];
            }
            attn_out[q_off + d] = val;
        }
    }

    // 6. Output projection with LoRA + residual.
    matmul(&mut x[..n], &layer_weights.attn_wo, &attn_out[..n], n, n);
    lora_apply(&mut x[..n], lora_o, &attn_out[..n], lora_buf);

    for i in 0..n {
        x[i] += xr[i];
    }

    // 7. MLP: save residual → RMSNorm → MLP with LoRA → residual.
    xr2[..n].copy_from_slice(&x[..n]);
    rmsnorm(&mut x[..n]);

    // MLP w1 with ReLU + LoRA.
    matmul_relu(hidden, &layer_weights.mlp_w1, &x[..n], config.mlp_hidden, n);
    lora_apply(hidden, lora_mlp1, &x[..n], lora_buf);

    // MLP w2 + LoRA.
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

    // Residual.
    for i in 0..n {
        x[i] += xr2[i];
    }

    // 8. LM Head.
    matmul(logits, &weights.lm_head, &x[..n], config.vocab_size, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_player_works_without_lora() {
        // Smoke test: create a heuristic-only player and run a single action
        // selection. Should not panic.
        let mut player = FftLoRAPlayer::new_heuristic(0);
        assert!(!player.lora_active());

        let state = BattleState::new();
        let mut rng = Rng::with_seed(0);
        let action = player.select_action(0, &state, &mut rng);
        // Heuristic should pick something reasonable for a full-HP Knight at
        // the start (likely Defend + move toward enemy, since no one is in range).
        assert!(
            matches!(
                action.action_type,
                ActionType::Defend | ActionType::Wait | ActionType::Attack
            ),
            "heuristic should pick a valid action, got {:?}",
            action.action_type
        );
    }

    #[test]
    fn heuristic_player_decodes_target_and_move() {
        let player = FftLoRAPlayer::new_heuristic(0);
        let state = BattleState::new();
        // Move the Knight (unit 0) adjacent to an enemy so attack is possible.
        // The default battle has party at (1,1)/(1,6)/(0,3)/(0,5) and enemy at
        // (6,1)/(6,6)/(7,3)/(7,5) — too far for the Knight (range=1) at start.
        // The heuristic should fall through to Defend.
        let action = player.heuristic_action(0, &state);
        // Without enemies in range, should Defend.
        assert!(
            matches!(action.action_type, ActionType::Defend | ActionType::Wait),
            "knight with no enemies in range should Defend/Wait, got {:?}",
            action.action_type
        );
    }

    #[test]
    fn lora_player_name_reflects_loaded_state() {
        let heuristic = FftLoRAPlayer::new_heuristic(0);
        assert_eq!(heuristic.name(), "FFT-SON-LT-Fallback");
    }
}
