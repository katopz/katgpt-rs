//! Battle state encoder for FFT SON-LT (Plan 296 T7.3).
//!
//! Shared encoding logic used by:
//! - `examples/fft_05_replay_gen.rs` (writer side — encodes `BattleState` → 57 tokens)
//! - `src/pruners/fft/lora_player.rs` (reader side — same encoding for inference)
//!
//! Layout MUST match `riir-gpu/src/game/fft_replay.rs`:
//!   `[tick, u0_team, u0_class, u0_hp, u0_mp, u0_x, u0_y, u0_alive, u1_..., ..., u7_...]`
//! (57 tokens total = 1 tick + 8 units × 7 fields).
//!
//! Token values stay in range `0..FFT_STATE_VOCAB` (=10).

use super::battle::BattleState;
use super::types::{Class, Team, Unit, GRID_H, GRID_W};

/// State vocab size — must match `riir_gpu::game::fft_replay::FFT_STATE_VOCAB`.
pub const FFT_STATE_VOCAB: usize = 10;

/// Number of units in a battle (4 party + 4 enemy).
pub const FFT_UNIT_COUNT: usize = 8;

/// Tokens per unit (team, class, hp, mp, x, y, alive).
pub const FFT_TOKENS_PER_UNIT: usize = 7;

/// State length (excludes action token): 1 tick + 8 × 7 = 57.
pub const FFT_STATE_LEN: usize = 1 + FFT_UNIT_COUNT * FFT_TOKENS_PER_UNIT;

/// Encode a `BattleState` into 57 state tokens.
///
/// Output buffer must be exactly `FFT_STATE_LEN` (=57) bytes long.
/// Panics if `out.len() != FFT_STATE_LEN`.
pub fn encode_battle_state(state: &BattleState, out: &mut [u8]) {
    assert_eq!(
        out.len(),
        FFT_STATE_LEN,
        "output buffer must be exactly FFT_STATE_LEN ({}) bytes, got {}",
        FFT_STATE_LEN,
        out.len(),
    );

    // Position 0: tick bucket (0..9). Quantize tick into 10 buckets (max tick
    // we care about is ~200; bucket = min(tick / 22, 9)).
    let tick_bucket = ((state.tick / 22) as usize).min(FFT_STATE_VOCAB - 1);
    out[0] = tick_bucket as u8;

    // Per-unit fields (7 tokens × 8 units = 56 tokens).
    // If the battle has fewer than 8 units, pad with zeros (dead, no class).
    for i in 0..FFT_UNIT_COUNT {
        let base = 1 + i * FFT_TOKENS_PER_UNIT;
        if let Some(unit) = state.units.get(i) {
            out[base] = encode_team(unit.team);
            out[base + 1] = encode_class(unit.class);
            out[base + 2] = encode_hp_bucket(unit);
            out[base + 3] = encode_mp_bucket(unit);
            out[base + 4] = encode_pos_axis(unit.pos.x);
            out[base + 5] = encode_pos_axis(unit.pos.y);
            out[base + 6] = encode_alive(unit);
        } else {
            // Pad with a "dead placeholder" token pattern.
            out[base..base + FFT_TOKENS_PER_UNIT].fill(0);
        }
    }
}

/// Encode team: Party=0, Enemy=1.
#[inline]
fn encode_team(team: Team) -> u8 {
    match team {
        Team::Party => 0,
        Team::Enemy => 1,
    }
}

/// Encode class into 0..5 range. Matches `Class` declaration order.
#[inline]
fn encode_class(class: Class) -> u8 {
    match class {
        Class::Knight => 0,
        Class::Archer => 1,
        Class::BlackMage => 2,
        Class::WhiteMage => 3,
        Class::Monk => 4,
        Class::TimeMage => 5,
    }
}

/// Encode HP bucket: 0..7 via `(hp / max_hp * 8).clamp(0, 7)`.
/// Dead units → bucket 0.
#[inline]
fn encode_hp_bucket(unit: &Unit) -> u8 {
    if !unit.alive || unit.stats.max_hp <= 0 {
        return 0;
    }
    let pct = unit.hp.max(0) as f32 / unit.stats.max_hp as f32;
    ((pct * 8.0).floor() as usize).clamp(0, 7) as u8
}

/// Encode MP bucket: 0..3 via `(mp / max_mp * 4).clamp(0, 3)`.
#[inline]
fn encode_mp_bucket(unit: &Unit) -> u8 {
    if unit.stats.max_mp <= 0 {
        return 0;
    }
    let pct = unit.mp.max(0) as f32 / unit.stats.max_mp as f32;
    ((pct * 4.0).floor() as usize).clamp(0, 3) as u8
}

/// Encode a position axis value (0..GRID_W or GRID_H). Clamps to vocab range.
#[inline]
fn encode_pos_axis(v: i32) -> u8 {
    if v < 0 {
        0
    } else if v >= GRID_W.max(GRID_H) {
        // 8 for an 8×8 grid — fits within FFT_STATE_VOCAB=10.
        ((GRID_W.max(GRID_H) - 1) as u8).min((FFT_STATE_VOCAB - 1) as u8)
    } else {
        v as u8
    }
}

/// Encode alive flag: 0=dead, 1=alive.
#[inline]
fn encode_alive(unit: &Unit) -> u8 {
    if unit.alive {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_len_is_57() {
        assert_eq!(FFT_STATE_LEN, 57);
    }

    #[test]
    fn encode_default_battle_state_produces_57_tokens() {
        let state = BattleState::new();
        let mut buf = [0u8; FFT_STATE_LEN];
        encode_battle_state(&state, &mut buf);
        // First token = tick bucket of tick=0 → 0.
        assert_eq!(buf[0], 0);
        // First unit (Party Knight at (1,1)) alive with full HP.
        assert_eq!(buf[1], 0); // team Party
        assert_eq!(buf[2], 0); // class Knight
        assert_eq!(buf[3], 7); // full hp → bucket 7
        assert_eq!(buf[4], 2); // mp = max_mp/2 = 10 → 10/20=0.5 → bucket 2 (of 4)
        assert_eq!(buf[5], 1); // pos_x
        assert_eq!(buf[6], 1); // pos_y
        assert_eq!(buf[7], 1); // alive
        // Tokens stay within state vocab range.
        for &v in &buf {
            assert!(v < FFT_STATE_VOCAB as u8, "token {} out of range: {}", v, v);
        }
    }

    #[test]
    fn encoding_is_deterministic() {
        let state = BattleState::new();
        let mut a = [0u8; FFT_STATE_LEN];
        let mut b = [0u8; FFT_STATE_LEN];
        encode_battle_state(&state, &mut a);
        encode_battle_state(&state, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn tick_bucket_advances_with_tick() {
        let mut state = BattleState::new();
        let mut buf = [0u8; FFT_STATE_LEN];

        state.tick = 0;
        encode_battle_state(&state, &mut buf);
        assert_eq!(buf[0], 0);

        state.tick = 22;
        encode_battle_state(&state, &mut buf);
        assert_eq!(buf[0], 1);

        state.tick = 200;
        encode_battle_state(&state, &mut buf);
        assert_eq!(buf[0], 9);
    }
}
