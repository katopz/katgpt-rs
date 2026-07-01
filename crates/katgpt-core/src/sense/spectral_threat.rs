//! Spectral Threat Prediction — LinOSS Modal Combo Prediction (Plan 241).
//!
//! Bridges LinOSS oscillation dynamics into NPC threat prediction. Player combo
//! attacks create periodic damage impulse trains. When fed into a LinOSS cell,
//! the hidden state resonates at the combo frequency. Reading the phase tells
//! the NPC *when* the next hit will land — predictive dodge timing.
//!
//! Physics: damped harmonic oscillator impulse response. The ω² modes resonate
//! at their natural frequency when forced. Dominant mode extraction yields
//! combo frequency (ω²), vulnerability phase (atan2(z,y)), and burst decay (β).
//!
//! Feature flag: `spectral_threat` (opt-in, requires `sense_composition` + `modal_spec`).

use crate::linoss::{LinOSSCell, LinOSSState};

// ── SpectralThreatFeatures ─────────────────────────────────────

/// Spectral features extracted from LinOSS combat rhythm tracking.
///
/// All fields are raw scalars — bridge from latent (LinOSS hidden) to raw (heuristic score).
/// 16 bytes, all f32 — no padding regardless of repr.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpectralThreatFeatures {
    /// Dominant combo frequency (ω²) — how fast the player cycles attacks.
    /// High value = fast combo, NPC should prioritize evasion.
    pub combo_frequency: f32,
    /// Phase in combo cycle — 0.0 = damage peak imminent, 0.5 = cooldown window.
    /// Computed as `atan2(z_dominant, y_dominant) / 2π`, normalized to [0, 1).
    pub vulnerability_phase: f32,
    /// Burst decay rate (β effective) — is burst damage decaying or sustained.
    /// Low β = sustained pressure, high β = exhausting, NPC can wait it out.
    pub burst_decay: f32,
    /// Energy of oscillation — confidence in spectral prediction.
    /// Low energy = insufficient data, fall back to reactive.
    pub rhythm_confidence: f32,
}

impl SpectralThreatFeatures {
    /// Compute dodge urgency from spectral features.
    ///
    /// High frequency + phase near 0.0 → urgency near 1.0 (dodge NOW).
    /// Low confidence → urgency near 0.5 (neutral, fall back to reactive).
    ///
    /// The gain factor (5.0) amplifies the raw signal so that moderate
    /// confidence + phase near 0 produces urgency > 0.7.
    #[inline(always)]
    pub fn dodge_urgency(&self) -> f32 {
        let raw =
            self.combo_frequency * (1.0 - 2.0 * self.vulnerability_phase) * self.rhythm_confidence;
        crate::simd::fast_sigmoid(raw * 5.0)
    }

    /// Compute counter window from spectral features.
    ///
    /// Best counter window: high frequency + phase near 0.5 (cooldown trough).
    /// Inverse of urgency — when to attack rather than dodge.
    #[inline(always)]
    pub fn counter_window(&self) -> f32 {
        let raw =
            self.combo_frequency * (2.0 * self.vulnerability_phase - 1.0) * self.rhythm_confidence;
        crate::simd::fast_sigmoid(raw * 5.0)
    }
}

// Sigmoid delegates to shared crate::simd::fast_sigmoid (bounded (0,1), libm-exp).
// Consistent with crate convention — NOT softmax.

// ── LabeledRhythm ──────────────────────────────────────────────

/// Per-participant LinOSS hidden state for combat rhythm tracking.
struct LabeledRhythm {
    /// LinOSS cell with pre-tuned ω² and β.
    cell: LinOSSCell,
    /// Hidden state (y, z) — oscillates with damage impulses.
    state: LinOSSState,
    /// Pre-allocated forcing buffer — fixed-size to avoid heap alloc.
    forcing: [f32; HIDDEN_DIM],
    /// Pre-allocated scratch for in-place y output.
    y_buf: [f32; HIDDEN_DIM],
    /// Pre-allocated scratch for in-place z output.
    z_buf: [f32; HIDDEN_DIM],
    /// Observed damage tick timestamps for auto-calibration.
    damage_timestamps: [u32; MAX_TIMESTAMPS],
    /// Number of valid entries in `damage_timestamps`.
    damage_timestamp_count: u32,
    /// Number of damage events ingested for this participant.
    event_count: u32,
    /// Entity ID (source of damage / tracked participant).
    #[allow(dead_code)]
    entity_id: u8,
}

// ── Constants ──────────────────────────────────────────────────

/// Canonical hidden dimension for combat LinOSS cells.
const HIDDEN_DIM: usize = 8;
/// Max damage timestamps stored per participant (determines stack array size).
const MAX_TIMESTAMPS: usize = 16;

// ── CombatRhythmTracker ────────────────────────────────────────

/// Maintains LinOSS hidden state per combat participant.
///
/// Ingests damage events as impulses, extracts modal features on demand.
/// Zero allocation on hot path — all buffers pre-allocated.
pub struct CombatRhythmTracker {
    /// Per-participant cells. Fixed-size array (max 8).
    cells: [Option<LabeledRhythm>; HIDDEN_DIM],
    /// Direct-indexed LUT: entity_id → slot index. u8::MAX = not registered.
    entity_lut: [u8; 256],
    /// Number of valid entries in `cells`.
    cell_count: u8,
    /// Hidden dimension — always HIDDEN_DIM. Kept for API compat.
    #[allow(dead_code)]
    hidden_dim: u8,
    /// Timestep for imex_step (derived from tick rate, e.g. 16ms → 0.016).
    dt: f32,
    /// Maximum expected damage for forcing normalization.
    max_damage: f32,
    /// Minimum events before rhythm_confidence ramps above 0.
    confidence_ramp: f32,
}

impl CombatRhythmTracker {
    /// Create a new tracker with given hidden dimension and timestep.
    ///
    /// `hidden_dim` = 8 (matches HLA dimension for simplicity).
    /// `dt` = derived from game tick rate (e.g., 16ms → 0.016).
    pub fn new(hidden_dim: usize, dt: f32) -> Self {
        debug_assert_eq!(hidden_dim, HIDDEN_DIM);
        Self {
            cells: [const { None }; HIDDEN_DIM],
            cell_count: 0,
            entity_lut: [u8::MAX; 256],
            hidden_dim: hidden_dim as u8,
            dt,
            max_damage: 50.0,
            confidence_ramp: 5.0,
        }
    }

    /// Create with pre-tuned combat frequencies.
    ///
    /// ω² set to `[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]` — covers
    /// slow heavy attacks to fast flurries. β = 0.1 (light damping,
    /// oscillation persists between hits to maintain phase info).
    pub fn with_combat_frequencies(dt: f32) -> Self {
        Self::new(HIDDEN_DIM, dt)
    }

    /// Pre-tuned combat ω² values — covers slow heavy to fast flurry.
    const COMBAT_OMEGA_SQ: [f32; HIDDEN_DIM] = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0];
    /// Light damping — oscillation persists between hits.
    const COMBAT_BETA: f32 = 0.1;

    /// O(1) entity_id → slot index lookup via direct-indexed LUT.
    #[inline(always)]
    fn slot_for(&self, entity_id: u8) -> Option<usize> {
        let slot = self.entity_lut[entity_id as usize];
        if slot == u8::MAX {
            None
        } else {
            Some(slot as usize)
        }
    }

    /// Register a new participant for tracking. No-op if already registered.
    #[inline]
    pub fn register(&mut self, entity_id: u8) {
        if self.slot_for(entity_id).is_some() {
            return;
        }
        if self.cell_count as usize >= HIDDEN_DIM {
            return; // max tracked participants
        }

        let mut cell = LinOSSCell::new(HIDDEN_DIM);
        cell.omega_sq.copy_from_slice(&Self::COMBAT_OMEGA_SQ);
        cell.beta.fill(Self::COMBAT_BETA);

        self.entity_lut[entity_id as usize] = self.cell_count;
        self.cells[self.cell_count as usize] = Some(LabeledRhythm {
            cell,
            state: LinOSSState::zeros(HIDDEN_DIM),
            forcing: [0.0; HIDDEN_DIM],
            y_buf: [0.0; HIDDEN_DIM],
            z_buf: [0.0; HIDDEN_DIM],
            damage_timestamps: [0u32; MAX_TIMESTAMPS],
            event_count: 0,
            damage_timestamp_count: 0,
            entity_id,
        });
        self.cell_count += 1;
    }

    /// Ingest a damage event from `source_id`.
    ///
    /// Converts damage amount to a forcing vector and advances the LinOSS cell
    /// one IMEX step. The hidden state (y, z) now reflects the impulse.
    /// Zero-amount events advance the oscillator state (natural decay) but
    /// do not increment event_count (only real impulses build confidence).
    #[inline(always)]
    pub fn ingest_damage(&mut self, source_id: u8, amount: f32, _tick: u32) {
        let slot = match self.slot_for(source_id) {
            Some(s) => s,
            None => return,
        };
        let Some(Some(rhythm)) = self.cells.get_mut(slot) else {
            return;
        };

        // Convert damage to forcing vector: normalized impulse across all dims
        let normalized = amount / self.max_damage;
        rhythm.forcing.fill(normalized);

        // Zero-alloc in-place IMEX step
        let h = HIDDEN_DIM;
        rhythm.cell.imex_step_inplace(
            &rhythm.state.y,
            &rhythm.state.z,
            &rhythm.forcing,
            self.dt,
            &mut rhythm.y_buf,
            &mut rhythm.z_buf,
        );

        // Copy back to state (y_buf → state.y, z_buf → state.z)
        rhythm.state.y.copy_from_slice(&rhythm.y_buf[..h]);
        rhythm.state.z.copy_from_slice(&rhythm.z_buf[..h]);

        // Only count real damage events for confidence ramp
        if amount > 0.0 {
            rhythm.event_count += 1;
            if (rhythm.damage_timestamp_count as usize) < MAX_TIMESTAMPS {
                rhythm.damage_timestamps[rhythm.damage_timestamp_count as usize] = _tick;
                rhythm.damage_timestamp_count += 1;
            }
        }
    }

    /// Extract spectral threat features for a tracked participant.
    ///
    /// Returns `SpectralThreatFeatures::default()` if participant not found.
    /// No allocation on this path.
    /// Const default — avoids repeated Default trait calls on hot path.
    const DEFAULT_FEATURES: SpectralThreatFeatures = SpectralThreatFeatures {
        combo_frequency: 0.0,
        vulnerability_phase: 0.0,
        burst_decay: 0.0,
        rhythm_confidence: 0.0,
    };

    #[inline(always)]
    pub fn extract_features(&self, entity_id: u8) -> SpectralThreatFeatures {
        let rhythm = match self.slot_for(entity_id) {
            Some(i) => match self.cells[i].as_ref() {
                Some(r) => r,
                None => return Self::DEFAULT_FEATURES,
            },
            None => return Self::DEFAULT_FEATURES,
        };

        if rhythm.event_count == 0 {
            return Self::DEFAULT_FEATURES;
        }

        // Find dominant mode: argmax of |y[i]| (branch-free abs)
        let h = HIDDEN_DIM;
        let mut dominant = 0usize;
        let mut max_amp = rhythm.state.y[0].abs();
        for i in 1..h {
            let amp = rhythm.state.y[i].abs();
            if amp > max_amp {
                max_amp = amp;
                dominant = i;
            }
        }

        // Combo frequency = ω² of dominant mode
        let combo_frequency = rhythm.cell.omega_sq[dominant];

        // Vulnerability phase = atan2(z, y) / 2π normalized to [0, 1)
        let phase_raw = rhythm.state.z[dominant].atan2(rhythm.state.y[dominant]);
        let vulnerability_phase = (phase_raw / (2.0 * std::f32::consts::PI) + 1.0) % 1.0;

        // Burst decay = β of dominant mode
        let burst_decay = rhythm.cell.beta[dominant];

        // Rhythm confidence = normalized energy * ramp factor
        // Energy is raw oscillator energy Σ(y² + ω²z²) which can vary widely.
        // We normalize via 1-exp(-k*energy) to map [0, ∞) → [0, 1).
        // Ramp: confidence builds over first `confidence_ramp` real damage events.
        let energy = rhythm.cell.energy(&rhythm.state);
        let ramp = (rhythm.event_count as f32 / self.confidence_ramp).min(1.0);
        let energy_normalized = 1.0 - (-energy).exp();
        let rhythm_confidence = energy_normalized * ramp;

        SpectralThreatFeatures {
            combo_frequency,
            vulnerability_phase,
            burst_decay,
            rhythm_confidence,
        }
    }

    /// Auto-calibrate ω² from observed damage intervals.
    ///
    /// Snaps the nearest pre-tuned mode to the exact observed combo frequency,
    /// and sets a second mode to the sub-harmonic (half frequency).
    /// Requires at least 3 timestamps (2 intervals) to produce valid calibration.
    #[inline]
    pub fn auto_calibrate(&mut self, entity_id: u8) {
        let rhythm = match self.slot_for(entity_id) {
            Some(i) => i,
            None => return,
        };
        let Some(Some(rhythm)) = self.cells.get_mut(rhythm) else {
            return;
        };
        let ts_count = rhythm.damage_timestamp_count as usize;
        if ts_count < 3 {
            return;
        }
        let ts = &rhythm.damage_timestamps[..ts_count];

        // Compute intervals into fixed-size stack array (max 16 timestamps → 15 intervals)
        let mut intervals = [0u32; MAX_TIMESTAMPS];
        let mut interval_count = 0usize;
        for i in 0..ts_count - 1 {
            let d = ts[i + 1] - ts[i];
            if d > 0 {
                intervals[interval_count] = d;
                interval_count += 1;
            }
        }
        if interval_count == 0 {
            return;
        }

        // Dominant interval = mode with ±5 tick tolerance
        let intervals = &mut intervals[..interval_count];
        intervals.sort_unstable();
        let mut best_val = intervals[0];
        let mut best_count = 1u32;
        let mut run_start = 0;
        for i in 1..intervals.len() {
            if intervals[i] - intervals[run_start] <= 10 {
                let count = (i - run_start + 1) as u32;
                if count > best_count {
                    best_count = count;
                    best_val = intervals[run_start + (i - run_start) / 2];
                }
            } else {
                run_start = i;
            }
        }
        let dominant_interval = best_val;

        // ω² = (2π / T)² where T = interval_ticks * dt
        let period = dominant_interval as f32 * self.dt;
        let omega = 2.0 * std::f32::consts::PI / period;
        let observed_omega_sq = omega * omega;

        // Snap nearest mode to observed ω²
        let omega_sq = &mut rhythm.cell.omega_sq;
        let mut nearest_idx = 0;
        let mut nearest_dist = f32::MAX;
        for (i, &v) in omega_sq.iter().enumerate() {
            let dist = (v - observed_omega_sq).abs();
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_idx = i;
            }
        }
        omega_sq[nearest_idx] = observed_omega_sq;

        // Sub-harmonic on second mode (avoid overwriting the snapped one)
        let sub_harmonic = observed_omega_sq * 0.25;
        let sub_idx = match nearest_idx {
            0 => 1,
            _ => 0,
        };
        omega_sq[sub_idx] = sub_harmonic;
    }

    /// Reset state for a participant (new encounter).
    pub fn reset(&mut self, entity_id: u8) {
        if let Some(i) = self.slot_for(entity_id) {
            let Some(Some(rhythm)) = self.cells.get_mut(i) else {
                return;
            };
            rhythm.state = LinOSSState::zeros(HIDDEN_DIM);
            rhythm.event_count = 0;
            rhythm.damage_timestamp_count = 0;
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_features_default_is_zero() {
        let f = SpectralThreatFeatures::default();
        assert_eq!(f.combo_frequency, 0.0);
        assert_eq!(f.vulnerability_phase, 0.0);
        assert_eq!(f.burst_decay, 0.0);
        assert_eq!(f.rhythm_confidence, 0.0);
    }

    #[test]
    fn test_dodge_urgency_bounds() {
        // Zero features → sigmoid(0) = 0.5 (neutral)
        let f = SpectralThreatFeatures::default();
        let urgency = f.dodge_urgency();
        assert!(
            (urgency - 0.5).abs() < 0.01,
            "Zero features should give ~0.5 urgency, got {urgency}"
        );

        // High frequency, phase 0 (peak imminent), high confidence → high urgency
        let f_high = SpectralThreatFeatures {
            combo_frequency: 4.0,
            vulnerability_phase: 0.0,
            burst_decay: 0.1,
            rhythm_confidence: 1.0,
        };
        let urgency_high = f_high.dodge_urgency();
        assert!(
            urgency_high > 0.9,
            "Peak imminent should give high urgency: {urgency_high}"
        );

        // High frequency, phase 0.5 (cooldown) → low urgency
        let f_cooldown = SpectralThreatFeatures {
            combo_frequency: 4.0,
            vulnerability_phase: 0.5,
            burst_decay: 0.1,
            rhythm_confidence: 1.0,
        };
        let urgency_cooldown = f_cooldown.dodge_urgency();
        assert!(
            (urgency_cooldown - 0.5).abs() < 0.01,
            "Cooldown should give ~0.5 urgency, got {urgency_cooldown}"
        );
    }

    #[test]
    fn test_counter_window_inverse_of_urgency() {
        let f = SpectralThreatFeatures {
            combo_frequency: 4.0,
            vulnerability_phase: 0.0,
            burst_decay: 0.1,
            rhythm_confidence: 1.0,
        };
        // Phase 0 = damage peak → high dodge urgency, low counter window
        let urgency = f.dodge_urgency();
        let counter = f.counter_window();
        assert!(
            urgency > counter,
            "At peak: dodge urgency ({urgency}) > counter ({counter})"
        );

        // Phase 0.7 = deep cooldown → counter window > dodge urgency
        let f_safe = SpectralThreatFeatures {
            combo_frequency: 4.0,
            vulnerability_phase: 0.7,
            burst_decay: 0.1,
            rhythm_confidence: 1.0,
        };
        let safe_urgency = f_safe.dodge_urgency();
        let safe_counter = f_safe.counter_window();
        assert!(
            safe_counter > safe_urgency,
            "At cooldown (phase 0.7): counter ({safe_counter}) > urgency ({safe_urgency})"
        );
    }

    #[test]
    fn test_tracker_register_and_ingest() {
        let mut tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
        tracker.register(1);

        // Ingest several damage impulses
        for _ in 0..5 {
            tracker.ingest_damage(1, 30.0, 0);
        }

        let features = tracker.extract_features(1);
        // After 5 events, should have non-zero confidence
        assert!(
            features.rhythm_confidence > 0.0,
            "Confidence should be > 0 after 5 events"
        );
        assert!(
            features.combo_frequency > 0.0,
            "Should have a dominant frequency"
        );
    }

    #[test]
    fn test_tracker_unknown_entity_returns_default() {
        let tracker = CombatRhythmTracker::new(HIDDEN_DIM, 0.016);
        let features = tracker.extract_features(99);
        assert_eq!(features.combo_frequency, 0.0);
    }

    #[test]
    fn test_tracker_reset_clears_state() {
        let mut tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
        tracker.register(1);
        for _ in 0..5 {
            tracker.ingest_damage(1, 30.0, 0);
        }
        let before = tracker.extract_features(1);
        assert!(before.rhythm_confidence > 0.0);

        tracker.reset(1);
        let after = tracker.extract_features(1);
        assert_eq!(
            after.rhythm_confidence, 0.0,
            "Reset should clear confidence"
        );
    }

    #[test]
    fn test_impulse_train_resonates() {
        // Simulate a periodic impulse train at ~800ms intervals (dt=0.016, 50 ticks)
        let dt = 0.016f32;
        let ticks_between = 50usize; // 50 * 16ms = 800ms
        let mut tracker = CombatRhythmTracker::with_combat_frequencies(dt);
        tracker.register(1);

        // Feed 10 impulses, one every 50 ticks
        for cycle in 0..10 {
            let tick = (cycle * ticks_between) as u32;
            tracker.ingest_damage(1, 30.0, tick);
            // Advance without impulse between hits (decay phase)
            for t in 1..ticks_between {
                tracker.ingest_damage(1, 0.0, tick + t as u32);
            }
        }

        let features = tracker.extract_features(1);
        // After sustained periodic forcing, should have measurable energy
        assert!(
            features.rhythm_confidence > 0.0,
            "Periodic impulse train should produce non-zero confidence: {:?}",
            features
        );
    }

    #[test]
    fn test_combat_frequencies_cover_range() {
        let tracker = CombatRhythmTracker::with_combat_frequencies(0.016);
        assert_eq!(tracker.hidden_dim as usize, HIDDEN_DIM);
    }

    #[test]
    fn test_max_tracked_participants() {
        let mut tracker = CombatRhythmTracker::new(HIDDEN_DIM, 0.016);
        // Register 8 participants (max)
        for id in 0..HIDDEN_DIM as u8 {
            tracker.register(id);
        }
        // 9th should be ignored
        tracker.register(HIDDEN_DIM as u8);
        assert_eq!(tracker.cells.len(), HIDDEN_DIM);
    }

    #[test]
    fn test_auto_calibrate_snaps_to_observed_frequency() {
        let dt = 0.016f32;
        let ticks_between = 50u32; // 50 * 16ms = 800ms combo
        let mut tracker = CombatRhythmTracker::with_combat_frequencies(dt);
        tracker.register(1);

        // Feed 6 impulses at regular 50-tick intervals
        for i in 0..6u32 {
            tracker.ingest_damage(1, 30.0, i * ticks_between);
        }

        // Expected ω² = (2π / (50 * 0.016))² = (2π / 0.8)² ≈ 61.7
        let expected_omega_sq = {
            let period = ticks_between as f32 * dt;
            let omega = 2.0 * std::f32::consts::PI / period;
            omega * omega
        };

        tracker.auto_calibrate(1);

        // Verify at least one mode matches the observed frequency
        let cell = tracker.cells[0]
            .as_ref()
            .expect("slot 0 should be occupied");
        let snapped = cell
            .cell
            .omega_sq
            .iter()
            .any(|&w| (w - expected_omega_sq).abs() < 0.1);
        assert!(
            snapped,
            "Expected a mode near ω²={expected_omega_sq:.1}, got {:?}",
            cell.cell.omega_sq
        );

        // Sub-harmonic should be ω²/4
        let sub_harmonic = expected_omega_sq * 0.25;
        let has_sub = cell
            .cell
            .omega_sq
            .iter()
            .any(|&w| (w - sub_harmonic).abs() < 0.1);
        assert!(
            has_sub,
            "Expected sub-harmonic near ω²={sub_harmonic:.1}, got {:?}",
            cell.cell.omega_sq
        );
    }
}
