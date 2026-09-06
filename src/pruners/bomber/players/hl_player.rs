//! P5: HLPlayer — bandit-adapted policy with absorb-compress.
//!
//! Extracted from `players.rs` (Issue 175).

use std::any::Any;

use fastrand::Rng;

#[cfg(feature = "bandit")]
use std::sync::Arc;

#[cfg(feature = "bandit")]
use crate::pruners::{SharedBanditStats, TrialRecord};

#[cfg(feature = "bandit")]
use crate::pruners::trial_log::SharedTrialLog;

use crate::pruners::bomber::{
    ArenaGrid, BomberAction, BomberFrozenBandit, Cell, GameEvent, GridPos,
};

#[cfg(feature = "bomber-wasm")]
use crate::types::LoraAdapter;

#[cfg(feature = "bomber-wasm")]
use super::helpers::lora_score_actions;
use super::helpers::{
    action_index, count_escape_routes, has_escape_route, in_blast_zone, index_to_action,
    intercept_score, is_safe_action, move_target, predict_direction, score_action, trap_score,
    update_bombs, update_opponents, update_powerups,
};
use super::{
    ACTION_COUNT, ALL_ACTIONS, BOMB_FUSE_TICKS, BomberPlayer, DEFAULT_BLAST_RANGE, KnownBomb,
    KnownOpponent,
};

#[cfg(any(
    feature = "contextual_bandit",
    feature = "binned_blend",
    feature = "kernel_blend"
))]
use crate::pruners::bomber::blend_context;

#[cfg(feature = "contextual_bandit")]
use crate::pruners::bomber::contextual_bandit::ContextualBandit;

#[cfg(feature = "binned_blend")]
use crate::pruners::bomber::blend_estimators::BinnedBlendEstimator;

#[cfg(feature = "kernel_blend")]
use crate::pruners::bomber::blend_estimators::KernelBlendEstimator;

#[cfg(feature = "kernel_blend")]
use crate::pruners::bomber::blend_estimators::KernelState;

#[cfg(feature = "bomber-wasm")]
use crate::pruners::bomber::wasm_pruner::BomberWasmPruner;

/// P4: Full HL — bandit-adapted policy with absorb-compress.
///
/// Blends policy scoring (60%) with bandit Q-values (40%) and adds:
/// - ε-greedy exploration (10%)
/// - Safety validation layer
/// - Absorb-compress: prunes consistently bad actions
/// - Trial logging for outcome attribution
pub struct HLPlayer {
    pub(crate) _id: u8,
    pub(crate) known_bombs: Vec<KnownBomb>,
    pub(crate) known_powerups: Vec<(i32, i32)>,
    pub(crate) known_opponents: Vec<KnownOpponent>,
    pub(crate) q_values: [f32; ACTION_COUNT],
    pub(crate) visits: [u32; ACTION_COUNT],
    pub(crate) total_pulls: u32,
    pub(crate) compressed: [bool; ACTION_COUNT],
    pub(crate) round_actions: Vec<BomberAction>,
    /// Per-tick shaped reward for each action in `round_actions` (Issue 371
    /// Option 2). Parallel Vec — `round_rewards[i]` is the shaped reward for
    /// `round_actions[i]`. Computed at decision time (when board state is
    /// available) and consumed by `update_outcome` to fix the
    /// credit-assignment dilution (T3 evidence #3): instead of distributing
    /// one `base_reward` uniformly, each tick carries a blast-zone-shaped
    /// signal so the n-armed bandit can learn context-dependent safety.
    pub(crate) round_rewards: Vec<f32>,
    /// Per-tick board-state context vector for each action in `round_actions`
    /// (Issue 371 Option 1 — T6). Parallel Vec — `round_contexts[i]` is the
    /// 7-dim context `φ(s)` at the moment `round_actions[i]` was chosen.
    /// Consumed by `update_outcome` so the contextual bandit can update
    /// `θ_a` per (arm, context) instead of per-arm — the principled fix for
    /// T3 evidence #3 (the n-armed bandit cannot learn context-dependent
    /// safety because the same action gets the same Q regardless of board
    /// state).
    #[cfg(any(
        feature = "contextual_bandit",
        feature = "binned_blend",
        feature = "kernel_blend"
    ))]
    pub(crate) round_contexts: Vec<[f32; blend_context::CONTEXT_DIM]>,
    /// Linear contextual bandit (Issue 371 Option 1 — T6). When present,
    /// replaces the n-armed `arm_q` lookup in `select_action`'s centered blend
    /// and the `update_arm_q` call in `update_outcome`. The n-armed fields
    /// (`q_values`, `visits`, ...) stay for `compress_cycle` / `compress_report`
    /// diagnostics and `SharedBanditStats` compat — they are not updated when
    /// the contextual bandit is active.
    #[cfg(feature = "contextual_bandit")]
    pub(crate) contextual_bandit: ContextualBandit,
    /// Binned nonlinear blend estimator (Plan 436 / Issue 428). When present,
    /// replaces the n-armed `arm_q` lookup with a per-(bin, arm) Q table
    /// conditioned on blast_proximity. Mutually exclusive with `kernel_blend`
    /// and `contextual_bandit` in practice.
    #[cfg(feature = "binned_blend")]
    pub(crate) binned_blend: BinnedBlendEstimator,
    /// Kernel nonlinear blend estimator (Plan 436 / Issue 428). When present,
    /// replaces the n-armed `arm_q` lookup with a Nadaraya-Watson weighted
    /// average over observed (context, arm, reward) triples. Mutually exclusive
    /// with `binned_blend` and `contextual_bandit` in practice.
    #[cfg(feature = "kernel_blend")]
    pub(crate) kernel_blend: KernelBlendEstimator,
    pub(crate) last_dir: Option<BomberAction>,
    /// Shared bandit stats for multi-agent cooperative learning.
    /// When `Some`, Q-values/visits/compressed are delegated here.
    #[cfg(feature = "bandit")]
    pub(crate) shared_stats: Option<Arc<SharedBanditStats>>,
    /// Shared trial log for multi-agent episode recording (Issue 051 T4).
    #[cfg(feature = "bandit")]
    pub(crate) shared_log: Option<SharedTrialLog>,
    /// LoRA adapter for learned action re-weighting (Issue 018 follow-up).
    /// When `Some`, replaces pure heuristic base with LoRA-blended scores.
    #[cfg(feature = "bomber-wasm")]
    pub(crate) lora: Option<LoraAdapter>,
    /// WASM validator for sandboxed safety checks (Issue 018 follow-up).
    /// When `Some`, replaces native `is_safe_action` with WASM-backed check.
    #[cfg(feature = "bomber-wasm")]
    pub(crate) wasm: Option<BomberWasmPruner>,
    /// Reusable LoRA scratch buffer (rank-sized, zero-alloc across calls).
    #[cfg(feature = "bomber-wasm")]
    pub(crate) lora_buf: Vec<f32>,
}

impl HLPlayer {
    pub fn new(id: u8) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            compressed: [false; ACTION_COUNT],
            round_actions: Vec::new(),
            round_rewards: Vec::new(),
            #[cfg(any(
                feature = "contextual_bandit",
                feature = "binned_blend",
                feature = "kernel_blend"
            ))]
            round_contexts: Vec::new(),
            #[cfg(feature = "contextual_bandit")]
            contextual_bandit: ContextualBandit::default(),
            #[cfg(feature = "binned_blend")]
            binned_blend: BinnedBlendEstimator::default(),
            #[cfg(feature = "kernel_blend")]
            kernel_blend: KernelBlendEstimator::default(),
            last_dir: None,
            #[cfg(feature = "bandit")]
            shared_stats: None,
            #[cfg(feature = "bandit")]
            shared_log: None,
            #[cfg(feature = "bomber-wasm")]
            lora: None,
            #[cfg(feature = "bomber-wasm")]
            wasm: None,
            #[cfg(feature = "bomber-wasm")]
            lora_buf: Vec::new(),
        }
    }

    /// Create HLPlayer with LoRA + WASM artifacts loaded (the "Full HL" stack).
    ///
    /// Mirrors `LoraWasmPlayer::new_with_secrets`: loads the LoRA adapter and
    /// WASM validator from file paths. On any load failure, silently falls
    /// back to heuristic-only mode (the player still works, just without the
    /// model delta and sandboxed safety).
    ///
    /// Only loads the first LoRA adapter — multi-adapter L2+ files have layers
    /// 1+ silently dropped. See `LoraAdapter::load_first` for the limitation.
    #[cfg(feature = "bomber-wasm")]
    pub fn new_with_secrets(id: u8, lora_path: &str, wasm_path: &str) -> Self {
        let lora = LoraAdapter::load_first(std::path::Path::new(lora_path)).ok();
        let wasm = BomberWasmPruner::load_from_file(wasm_path).ok();
        let buf_size = lora.as_ref().map_or(0, |l| l.rank);
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            compressed: [false; ACTION_COUNT],
            round_actions: Vec::new(),
            round_rewards: Vec::new(),
            #[cfg(any(
                feature = "contextual_bandit",
                feature = "binned_blend",
                feature = "kernel_blend"
            ))]
            round_contexts: Vec::new(),
            #[cfg(feature = "contextual_bandit")]
            contextual_bandit: ContextualBandit::default(),
            #[cfg(feature = "binned_blend")]
            binned_blend: BinnedBlendEstimator::default(),
            #[cfg(feature = "kernel_blend")]
            kernel_blend: KernelBlendEstimator::default(),
            last_dir: None,
            #[cfg(feature = "bandit")]
            shared_stats: None,
            #[cfg(feature = "bandit")]
            shared_log: None,
            lora,
            wasm,
            lora_buf: vec![0.0; buf_size],
        }
    }

    /// Create HLPlayer sharing bandit stats with other agents.
    ///
    /// Multiple agents sharing one `SharedBanditStats` learn cooperatively:
    /// Q-values and visit counts are shared, but each agent still has
    /// its own heuristic scoring and RNG for action selection.
    ///
    /// Optionally pass a `SharedTrialLog` to record episodes with `player_id`
    /// for multi-agent post-hoc analysis (Issue 051 T4).
    #[cfg(feature = "bandit")]
    pub fn with_shared_stats(
        id: u8,
        stats: Arc<SharedBanditStats>,
        shared_log: Option<SharedTrialLog>,
    ) -> Self {
        Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            q_values: [0.0; ACTION_COUNT],
            visits: [0; ACTION_COUNT],
            total_pulls: 0,
            compressed: [false; ACTION_COUNT],
            round_actions: Vec::new(),
            round_rewards: Vec::new(),
            #[cfg(any(
                feature = "contextual_bandit",
                feature = "binned_blend",
                feature = "kernel_blend"
            ))]
            round_contexts: Vec::new(),
            #[cfg(feature = "contextual_bandit")]
            contextual_bandit: ContextualBandit::default(),
            #[cfg(feature = "binned_blend")]
            binned_blend: BinnedBlendEstimator::default(),
            #[cfg(feature = "kernel_blend")]
            kernel_blend: KernelBlendEstimator::default(),
            last_dir: None,
            shared_stats: Some(stats),
            shared_log,
            #[cfg(feature = "bomber-wasm")]
            lora: None,
            #[cfg(feature = "bomber-wasm")]
            wasm: None,
            #[cfg(feature = "bomber-wasm")]
            lora_buf: Vec::new(),
        }
    }

    // ── Shared Stats Accessors ─────────────────────────────────

    /// Whether an arm is compressed (hard-blocked).
    ///
    /// Delegates to shared stats when present, else uses local field.
    #[cfg(feature = "bandit")]
    pub fn arm_compressed(&self, arm: usize) -> bool {
        self.shared_stats
            .as_ref()
            .map_or(self.compressed[arm], |s| s.is_compressed(arm))
    }

    #[cfg(not(feature = "bandit"))]
    pub fn arm_compressed(&self, arm: usize) -> bool {
        self.compressed[arm]
    }

    /// Visit count for an arm.
    #[cfg(feature = "bandit")]
    pub fn arm_visits(&self, arm: usize) -> u32 {
        self.shared_stats
            .as_ref()
            .map_or(self.visits[arm], |s| s.visits(arm))
    }

    #[cfg(not(feature = "bandit"))]
    pub fn arm_visits(&self, arm: usize) -> u32 {
        self.visits[arm]
    }

    /// Q-value estimate for an arm.
    #[cfg(feature = "bandit")]
    pub fn arm_q(&self, arm: usize) -> f32 {
        self.shared_stats
            .as_ref()
            .map_or(self.q_values[arm], |s| s.q_value(arm))
    }

    #[cfg(not(feature = "bandit"))]
    pub fn arm_q(&self, arm: usize) -> f32 {
        self.q_values[arm]
    }

    /// Total pulls across all arms.
    ///
    /// Delegates to shared stats when present, else uses local field.
    #[cfg(feature = "bandit")]
    pub fn arm_total_pulls(&self) -> u32 {
        self.shared_stats
            .as_ref()
            .map_or(self.total_pulls, |s| s.total_pulls())
    }

    #[cfg(not(feature = "bandit"))]
    pub fn arm_total_pulls(&self) -> u32 {
        self.total_pulls
    }

    /// Update Q-value for an arm with observed reward.
    ///
    /// Delegates to shared stats when present, else updates local fields.
    #[cfg(feature = "bandit")]
    fn update_arm_q(&mut self, arm: usize, reward: f32) {
        if let Some(stats) = &self.shared_stats { stats.update(arm, reward) } else {
                self.visits[arm] += 1;
                self.total_pulls += 1;
                let n = self.visits[arm] as f32;
                self.q_values[arm] += (reward - self.q_values[arm]) / n;
            }
    }

    #[cfg(not(feature = "bandit"))]
    fn update_arm_q(&mut self, arm: usize, reward: f32) {
        self.visits[arm] += 1;
        self.total_pulls += 1;
        let n = self.visits[arm] as f32;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;
    }

    /// Mark an arm as compressed (hard-blocked).
    #[cfg(feature = "bandit")]
    fn mark_compressed(&mut self, arm: usize) {
        match &self.shared_stats {
            Some(stats) => stats.compress_arm(arm),
            None => self.compressed[arm] = true,
        }
    }

    #[cfg(not(feature = "bandit"))]
    fn mark_compressed(&mut self, arm: usize) {
        self.compressed[arm] = true;
    }

    /// Update bandit Q-values based on round outcome.
    ///
    /// Distributes reward across ALL actions taken this round (not just the last).
    /// This prevents misattribution where only the final action gets blamed for death.
    pub fn update_outcome(
        &mut self,
        survived: bool,
        killed_opponent: bool,
        collected_powerups: u32,
    ) {
        if self.round_actions.is_empty() {
            return;
        }

        // Base reward shaping
        let base_reward = if survived { 1.0 } else { -1.0 }
            + if killed_opponent { 0.5 } else { 0.0 }
            + collected_powerups as f32 * 0.2;

        // Decay-based credit assignment: recent actions get more weight.
        //
        // Issue 371 Option 2 — per-tick reward shaping. Instead of distributing
        // one `base_reward` uniformly across all actions, each tick carries its
        // own blast-zone-shaped reward (`round_rewards[i]`, computed at decision
        // time in `select_action`). This fixes the credit-assignment dilution
        // (T3 evidence #3): a move that walked into a blast zone gets a direct
        // negative signal regardless of whether HL survived the round, so the
        // bandit can learn context-dependent safety without a contextual
        // (state-conditioned) Q implementation.
        //
        // Issue 371 Option 1 (T6) — contextual bandit update. When active,
        // each (action, context) pair updates θ_a via online LMS instead of
        // the n-armed running-average update below. This is the principled
        // fix: the same action gets different θ-updates depending on the board
        // state it was taken in.
        #[cfg(feature = "contextual_bandit")]
        {
            for (i, action) in self.round_actions.iter().enumerate() {
                let tick_reward = self.round_rewards.get(i).copied().unwrap_or(0.0);
                let phi = match self.round_contexts.get(i) {
                    Some(p) => *p,
                    None => continue,
                };
                let combined = base_reward + tick_reward;
                let idx = action_index(action);
                self.contextual_bandit.update(idx, &phi, combined);
            }
        }

        // Plan 436 / Issue 428 — nonlinear blend estimator updates. Same
        // pattern as the contextual bandit: iterate this round's (action,
        // context, reward) triples and update the estimator per-tick.
        #[cfg(feature = "binned_blend")]
        {
            for (i, action) in self.round_actions.iter().enumerate() {
                let tick_reward = self.round_rewards.get(i).copied().unwrap_or(0.0);
                let phi = match self.round_contexts.get(i) {
                    Some(p) => *p,
                    None => continue,
                };
                let combined = base_reward + tick_reward;
                let idx = action_index(action);
                self.binned_blend.update(&phi, idx, combined);
            }
        }
        #[cfg(feature = "kernel_blend")]
        {
            for (i, action) in self.round_actions.iter().enumerate() {
                let tick_reward = self.round_rewards.get(i).copied().unwrap_or(0.0);
                let phi = match self.round_contexts.get(i) {
                    Some(p) => *p,
                    None => continue,
                };
                let combined = base_reward + tick_reward;
                let idx = action_index(action);
                self.kernel_blend.update(&phi, idx, combined);
            }
        }

        let total = self.round_actions.len();
        let mut action_rewards = [0.0f32; ACTION_COUNT];
        let mut action_weights = [0.0f32; ACTION_COUNT];

        for (i, action) in self.round_actions.iter().enumerate() {
            // Exponential decay: later actions get exponentially more credit
            let recency = 0.5_f32.powi((total - 1 - i) as i32);
            // Per-tick shaped reward (defensive: default 0.0 if Vec lengths
            // ever diverge — they should always be parallel).
            let tick_reward = self.round_rewards.get(i).copied().unwrap_or(0.0);
            let idx = action_index(action);
            action_rewards[idx] += (base_reward + tick_reward) * recency;
            action_weights[idx] += recency;
        }

        // Update Q-values with weighted rewards (delegates to shared stats when present)
        for idx in 0..ACTION_COUNT {
            if action_weights[idx] == 0.0 {
                continue;
            }
            let reward = action_rewards[idx] / action_weights[idx];
            self.update_arm_q(idx, reward);
        }

        // Record trial data for shared log (Issue 051 T4)
        #[cfg(feature = "bandit")]
        if let Some(ref log) = self.shared_log {
            let episode = match &self.shared_stats {
                Some(stats) => stats.total_pulls() as usize,
                None => self.total_pulls as usize,
            };
            for idx in 0..ACTION_COUNT {
                if action_weights[idx] == 0.0 {
                    continue;
                }
                let reward = action_rewards[idx] / action_weights[idx];
                let record = TrialRecord {
                    episode,
                    player_id: self._id as u32,
                    arm: idx,
                    reward,
                    q_value: self.arm_q(idx),
                    cumulative_reward: 0.0,
                    cumulative_regret: 0.0,
                    config: "bomber_hl".into(),
                    note: format!("survived={survived},killed={killed_opponent}"),
                    base_correct: None,
                    reviewed_correct: None,
                    anchors: None,
                };
                let _ = log.append(&record);
            }
        }
    }

    /// Run absorb-compress cycle. Returns newly compressed arm indices.
    pub fn compress_cycle(&mut self) -> Vec<usize> {
        let min_visits = 20u32;
        let threshold = 0.1f32;
        let mut newly_compressed = Vec::with_capacity(ACTION_COUNT);

        for i in 0..ACTION_COUNT {
            if self.arm_compressed(i) {
                continue;
            }
            if self.arm_visits(i) >= min_visits && self.arm_q(i) < threshold {
                self.mark_compressed(i);
                newly_compressed.push(i);
            }
        }

        newly_compressed
    }

    /// Generate a compression report string.
    pub fn compress_report(&self) -> String {
        #[cfg(feature = "bandit")]
        if let Some(ref stats) = self.shared_stats {
            let compressed_count = (0..ACTION_COUNT)
                .filter(|&i| stats.is_compressed(i))
                .count();
            let compressed_names: Vec<String> = (0..ACTION_COUNT)
                .filter(|&i| stats.is_compressed(i))
                .map(|i| format!("{}({:.2})", index_to_action(i), stats.q_value(i)))
                .collect();
            return format!(
                "Pulls={} Compressed={}/{} [{}] Q=[{}]",
                stats.total_pulls(),
                compressed_count,
                ACTION_COUNT,
                compressed_names.join(","),
                (0..ACTION_COUNT)
                    .map(|i| format!("{}:{:.2}", index_to_action(i), stats.q_value(i)))
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }

        let compressed_count = self.compressed.iter().filter(|&&c| c).count();
        let compressed_names: Vec<String> = self
            .compressed
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .map(|(i, _)| format!("{}({:.2})", index_to_action(i), self.q_values[i]))
            .collect();

        format!(
            "Pulls={} Compressed={}/{} [{}] Q=[{}]",
            self.total_pulls,
            compressed_count,
            ACTION_COUNT,
            compressed_names.join(","),
            self.q_values
                .iter()
                .enumerate()
                .map(|(i, q)| format!("{}:{:.2}", index_to_action(i), q))
                .collect::<Vec<_>>()
                .join(" "),
        )
    }

    /// Freeze bandit knowledge into a `repr(C)` struct for disk persistence.
    ///
    /// Only captures learned knowledge (Q-values, visits, compressed flags).
    /// Transient game state (bombs, positions, opponents) is NOT included.
    pub fn freeze(&self) -> BomberFrozenBandit {
        let mut compressed = [0u8; 7];
        for (i, &c) in self.compressed.iter().enumerate() {
            compressed[i] = if c { 1 } else { 0 };
        }
        BomberFrozenBandit {
            magic: BomberFrozenBandit::MAGIC,
            version: BomberFrozenBandit::VERSION,
            q_values: self.q_values,
            visits: self.visits,
            total_pulls: self.total_pulls,
            compressed,
            reserved: [0; 16],
        }
    }

    /// Thaw a player from frozen bandit knowledge.
    ///
    /// Creates a fresh player (no transient state) with pre-loaded bandit knowledge.
    /// Validates magic bytes and version before reconstruction.
    pub fn thaw(frozen: &BomberFrozenBandit, id: u8) -> Result<Self, String> {
        frozen.validate()?;
        Ok(Self {
            _id: id,
            known_bombs: Vec::new(),
            known_powerups: Vec::new(),
            known_opponents: Vec::new(),
            q_values: frozen.q_values,
            visits: frozen.visits,
            total_pulls: frozen.total_pulls,
            compressed: frozen.compressed.map(|c| c != 0),
            round_actions: Vec::new(),
            round_rewards: Vec::new(),
            #[cfg(any(
                feature = "contextual_bandit",
                feature = "binned_blend",
                feature = "kernel_blend"
            ))]
            round_contexts: Vec::new(),
            #[cfg(feature = "contextual_bandit")]
            contextual_bandit: ContextualBandit::default(),
            #[cfg(feature = "binned_blend")]
            binned_blend: BinnedBlendEstimator::default(),
            #[cfg(feature = "kernel_blend")]
            kernel_blend: KernelBlendEstimator::default(),
            last_dir: None,
            #[cfg(feature = "bandit")]
            shared_stats: None,
            #[cfg(feature = "bandit")]
            shared_log: None,
            #[cfg(feature = "bomber-wasm")]
            lora: None,
            #[cfg(feature = "bomber-wasm")]
            wasm: None,
            #[cfg(feature = "bomber-wasm")]
            lora_buf: Vec::new(),
        })
    }

    /// Check if action is safe — WASM validator if loaded, native otherwise.
    ///
    /// When `self.wasm` is `Some`, delegates to the sandboxed WASM validator
    /// (stricter, external-process isolation). Otherwise falls back to the
    /// native `is_safe_action` check. Mirrors `LoraWasmPlayer::is_action_safe`.
    #[cfg(feature = "bomber-wasm")]
    fn check_safety(&self, action: &BomberAction, grid: &ArenaGrid, pos: GridPos) -> bool {
        match &self.wasm {
            Some(wasm) => wasm.is_safe_action(
                action_index(action),
                grid,
                pos.x,
                pos.y,
                self._id,
                &self.known_bombs,
            ),
            None => is_safe_action(action, grid, pos, &self.known_bombs),
        }
    }

    /// Native-only safety check (no WASM feature compiled).
    #[cfg(not(feature = "bomber-wasm"))]
    fn check_safety(&self, action: &BomberAction, grid: &ArenaGrid, pos: GridPos) -> bool {
        is_safe_action(action, grid, pos, &self.known_bombs)
    }

    /// Export the kernel estimator's learned state for sharing between NPCs.
    ///
    /// Story 3 (Plan 437): Cohort B exports its kernel state after the learning
    /// phase. Cohort C imports it to inherit B's accumulated experience
    /// without playing those rounds itself (sharing substitutes for learning).
    ///
    /// Returns `None` if `kernel_blend` is not active.
    #[cfg(feature = "kernel_blend")]
    pub fn export_kernel_state(&self) -> Option<KernelState> {
        Some(self.kernel_blend.export_state())
    }

    /// Import a shared kernel state, replacing this player's learned state.
    ///
    /// Story 3 (Plan 437): Cohort C calls this after receiving Cohort B's
    /// exported state. The receiver immediately benefits from B's experience.
    #[cfg(feature = "kernel_blend")]
    pub fn import_kernel_state(&mut self, state: &KernelState) {
        self.kernel_blend.import_state(state);
    }
}

impl BomberPlayer for HLPlayer {
    fn select_action(
        &mut self,
        grid: &ArenaGrid,
        pos: GridPos,
        events: &[GameEvent],
        rng: &mut Rng,
    ) -> BomberAction {
        update_bombs(&mut self.known_bombs, events);
        update_powerups(&mut self.known_powerups, events);
        update_opponents(&mut self.known_opponents, events, self._id);

        // O(bombs) linear helper — replaces per-call HashSet allocation.
        let is_blocked = |x: i32, y: i32| {
            self.known_bombs
                .iter()
                .any(|(p, _, _)| p.0 == x && p.1 == y)
        };

        // Find nearest opponent and their predicted trajectory
        let nearest_info = self
            .known_opponents
            .iter()
            .filter(|(_, op, _)| grid.is_walkable(op.0, op.1))
            .min_by_key(|(_, op, _)| (pos.x - op.0).abs() + (pos.y - op.1).abs());

        let nearest_opponent = nearest_info.map(|(_, op, _)| *op);
        let predicted_opponent =
            nearest_info.and_then(|(_, op, prev)| predict_direction(*op, *prev));

        // Issue 018 follow-up: LoRA-blended base scores (if adapter loaded).
        // When `Some`, replaces pure heuristic with 70% heuristic + 30% LoRA
        // correction (see `lora_score_actions`). Strategy bonus is added later.
        #[cfg(feature = "bomber-wasm")]
        let lora_scores = self.lora.as_ref().and_then(|lora| {
            lora_score_actions(
                lora,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
                &mut self.lora_buf,
            )
        });

        // Compute action scores: heuristic (+ LoRA blend when loaded) + strategy bonus
        // + centered bandit Q-value blend (Issue 371: re-enabled, weight 2.0).
        let mut scores: [(BomberAction, f32); ACTION_COUNT] = ALL_ACTIONS.map(|a| (a, 0.0));

        // Compute the board-state context vector φ(s) once per tick. Used by all
        // blend estimators (contextual_bandit, binned_blend, kernel_blend) so
        // the same action gets different Q-values in safe vs dangerous board
        // states — the principled fix for T3 evidence #3.
        #[cfg(any(
            feature = "contextual_bandit",
            feature = "binned_blend",
            feature = "kernel_blend"
        ))]
        let phi = blend_context::compute_phi(
            pos,
            grid,
            &self.known_bombs,
            &self.known_powerups,
            nearest_opponent,
        );

        // Plan 436 / Issue 428: compute blend estimator Qs once per tick when
        // the nonlinear estimators are active. These replace the per-arm
        // n-armed Q lookup with a context-conditioned Q.
        #[cfg(feature = "kernel_blend")]
        let blend_q = self.kernel_blend.predict_all(&phi);
        #[cfg(all(feature = "binned_blend", not(feature = "kernel_blend")))]
        let blend_q = self.binned_blend.predict_all(&phi);

        for (i, action) in ALL_ACTIONS.iter().enumerate() {
            // Skip compressed (hard-blocked) arms
            if self.arm_compressed(i) {
                scores[i] = (*action, f32::NEG_INFINITY);
                continue;
            }

            let h = score_action(
                action,
                grid,
                pos,
                &self.known_bombs,
                &self.known_powerups,
                self.last_dir,
            );

            // Issue 018 follow-up: use LoRA-blended score as base if available
            #[cfg(feature = "bomber-wasm")]
            let h = match &lora_scores {
                Some(s) => s[i],
                None => h,
            };

            // Domain hard block (unwalkable, unsafe bomb) overrides everything
            if h == f32::NEG_INFINITY {
                scores[i] = (*action, h);
                continue;
            }

            // Safety validation — hard-block unsafe Bomb/Wait only;
            // let score_action handle movement (it uses escape_distance in blast zones)
            let is_move = matches!(
                action,
                BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
            );
            if !is_move && !self.check_safety(action, grid, pos) {
                scores[i] = (*action, f32::NEG_INFINITY);
                continue;
            }

            // Strategic bonus: hunt, intercept, ambush, and trap
            let mut strategy_bonus = 0.0f32;
            match action {
                BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right => {
                    if let Some((ox, oy)) = nearest_opponent {
                        let target = move_target(action, pos);
                        let current_dist = (pos.x - ox).abs() + (pos.y - oy).abs();
                        let target_dist = (target.x - ox).abs() + (target.y - oy).abs();

                        // Hunt: move toward opponent
                        if target_dist < current_dist {
                            strategy_bonus += 1.5;
                        }

                        // Intercept: move toward predicted position
                        strategy_bonus +=
                            intercept_score((target.x, target.y), (ox, oy), predicted_opponent);

                        // Chokepoint: prefer moving where opponent has fewer escapes
                        if target_dist <= 3 {
                            let routes = count_escape_routes((target.x, target.y), grid);
                            if routes <= 1 {
                                strategy_bonus += 1.0;
                            }
                        }
                    }
                }
                BomberAction::Bomb => {
                    // Strategic value: more adjacent walls = better bomb placement
                    let wall_count = [(0i32, -1), (0, 1), (-1, 0), (1, 0)]
                        .iter()
                        .filter(|&&(dx, dy)| {
                            matches!(
                                grid.get(pos.x + dx, pos.y + dy),
                                Cell::DestructibleWall | Cell::PowerUpHidden(_)
                            )
                        })
                        .count();
                    strategy_bonus += wall_count as f32 * 0.5;

                    // Attack: trap scoring when opponent is nearby
                    if let Some((ox, oy)) = nearest_opponent {
                        strategy_bonus +=
                            trap_score((pos.x, pos.y), (ox, oy), grid, DEFAULT_BLAST_RANGE);
                    }
                }
                BomberAction::Wait => {}
                BomberAction::Detonate => {
                    // Strategic detonation: bonus when own bombs are near opponents
                    // (future: remote bombs only; currently all player bombs are Timed)
                    if let Some((ox, oy)) = nearest_opponent {
                        for &((bx, by), _range, _fuse) in &self.known_bombs {
                            let bomb_to_opp = (bx - ox).abs() + (by - oy).abs();
                            if bomb_to_opp <= DEFAULT_BLAST_RANGE as i32 {
                                strategy_bonus += 2.0; // Own bomb threatens opponent
                            }
                        }
                    }
                    // Safety penalty: detonating while in own blast zone is fatal
                    if in_blast_zone(pos, grid, &self.known_bombs) {
                        strategy_bonus -= 5.0;
                    }
                }
            }

            // Bandit Q-value blend (Issue 371: re-enabled, centered, weight 2.0).
            // Reward arms with Q > 0.5, penalize Q < 0.5. Unvisited arms are neutral
            // (treated as Q = 0.5) so the bandit doesn't suppress early exploration.
            //
            // Priority chain (mutually exclusive in practice):
            //   1. kernel_blend  (Plan 436 / Issue 428) — Nadaraya-Watson
            //   2. binned_blend  (Plan 436 / Issue 428) — per-bin empirical Q
            //   3. contextual_bandit (Issue 371 T6) — linear per-arm model
            //   4. n-armed bandit (default) — context-independent average Q
            #[cfg(feature = "kernel_blend")]
            let bandit_term = if !self.kernel_blend.is_cold(i) {
                (blend_q[i] - 0.5) * 2.0
            } else {
                0.0
            };
            #[cfg(all(feature = "binned_blend", not(feature = "kernel_blend")))]
            let bandit_term = if !self.binned_blend.is_cold(i) {
                (blend_q[i] - 0.5) * 2.0
            } else {
                0.0
            };
            #[cfg(all(
                feature = "contextual_bandit",
                not(feature = "binned_blend"),
                not(feature = "kernel_blend")
            ))]
            let bandit_term = if !self.contextual_bandit.is_cold(i) {
                let q = self.contextual_bandit.predict(i, &phi);
                (q - 0.5) * 2.0
            } else {
                0.0
            };
            #[cfg(not(any(
                feature = "contextual_bandit",
                feature = "binned_blend",
                feature = "kernel_blend"
            )))]
            let bandit_term = if self.arm_visits(i) > 0 {
                (self.arm_q(i) - 0.5) * 2.0
            } else {
                0.0
            };

            scores[i] = (*action, h + strategy_bonus + bandit_term);
        }

        // Per-tick reward shaping (Issue 371 Option 2).
        //
        // Computes a blast-zone-shaped reward for a candidate action at the
        // current board state. This is recorded alongside the action in
        // `round_rewards` and consumed by `update_outcome` so the n-armed
        // bandit learns per-tick danger instead of a uniform round-level
        // reward. Fixes the credit-assignment dilution (T3 evidence #3):
        // a move that walks into a blast zone now gets a direct negative
        // signal, regardless of whether HL survives the round.
        //
        // `bombs` is passed as a parameter (not captured) so the closure
        // doesn't hold an immutable borrow of `self` that would conflict
        // with the later `self.known_bombs.push(...)` / `self.round_rewards.push(...)`.
        let currently_in_blast = in_blast_zone(pos, grid, &self.known_bombs);
        let shape_tick = |action: BomberAction, bombs: &[KnownBomb]| -> f32 {
            match action {
                BomberAction::Up
                | BomberAction::Down
                | BomberAction::Left
                | BomberAction::Right => {
                    let target = move_target(&action, pos);
                    let target_in_blast = in_blast_zone(target, grid, bombs);
                    if target_in_blast && !currently_in_blast {
                        -0.5 // entered danger
                    } else if !target_in_blast && currently_in_blast {
                        0.5 // escaped danger
                    } else {
                        0.0 // neutral
                    }
                }
                BomberAction::Bomb => {
                    // Placing a bomb with no escape route is dangerous;
                    // `has_escape_route` does a BFS for a safe cell reachable
                    // within blast_range+1 steps. Penalty when trapped.
                    if !has_escape_route(grid, pos, (pos.x, pos.y), DEFAULT_BLAST_RANGE, bombs) {
                        -0.3
                    } else {
                        0.0
                    }
                }
                BomberAction::Wait => {
                    // Waiting inside a blast zone is fatal.
                    if currently_in_blast { -0.5 } else { 0.0 }
                }
                BomberAction::Detonate => {
                    // Detonating while in own blast zone is fatal.
                    if currently_in_blast { -0.5 } else { 0.0 }
                }
            }
        };

        // ε-greedy: 10% explore (only safe moves — less random than Greedy's 20%)
        if rng.f32() < 0.10 {
            // Pick a random non-compressed, non-hard-blocked, safe action
            let safe_explore: Vec<usize> = (0..ACTION_COUNT)
                .filter(|&i| {
                    if self.arm_compressed(i) || scores[i].1 <= f32::NEG_INFINITY {
                        return false;
                    }
                    let action = ALL_ACTIONS[i];
                    match action {
                        BomberAction::Up
                        | BomberAction::Down
                        | BomberAction::Left
                        | BomberAction::Right => {
                            let target = move_target(&action, pos);
                            grid.is_walkable(target.x, target.y)
                                && !is_blocked(target.x, target.y)
                                && !in_blast_zone(target, grid, &self.known_bombs)
                        }
                        _ => false, // Don't randomly explore Bomb/Wait
                    }
                })
                .collect();
            if !safe_explore.is_empty() {
                let pick = safe_explore[rng.usize(0..safe_explore.len())];
                let action = scores[pick].0;
                self.round_actions.push(action);
                self.round_rewards
                    .push(shape_tick(action, &self.known_bombs));
                #[cfg(any(
                    feature = "contextual_bandit",
                    feature = "binned_blend",
                    feature = "kernel_blend"
                ))]
                self.round_contexts.push(phi);
                self.last_dir = Some(action);
                return action;
            }
        }

        // Pick best action
        let best = scores
            .iter()
            .max_by(|a, b| katgpt_core::float_order::cmp_for_max(a.1, b.1)).map_or(BomberAction::Wait, |(a, _)| *a);

        // Track own bomb placement (critical: prevents walking back into own bomb)
        if best == BomberAction::Bomb {
            self.known_bombs
                .push(((pos.x, pos.y), DEFAULT_BLAST_RANGE, BOMB_FUSE_TICKS));
        }

        self.round_actions.push(best);
        self.round_rewards.push(shape_tick(best, &self.known_bombs));
        #[cfg(any(
            feature = "contextual_bandit",
            feature = "binned_blend",
            feature = "kernel_blend"
        ))]
        self.round_contexts.push(phi);
        if matches!(
            best,
            BomberAction::Up | BomberAction::Down | BomberAction::Left | BomberAction::Right
        ) {
            self.last_dir = Some(best);
        }
        best
    }

    fn name(&self) -> &str {
        "HL"
    }

    fn emoji(&self) -> &str {
        "🐵"
    }

    fn reset(&mut self) {
        self.known_bombs.clear();
        self.known_powerups.clear();
        self.known_opponents.clear();
        self.round_actions.clear();
        self.round_rewards.clear();
        #[cfg(any(
            feature = "contextual_bandit",
            feature = "binned_blend",
            feature = "kernel_blend"
        ))]
        self.round_contexts.clear();
        self.last_dir = None;
        // NOTE: Q-values, visits, compressed persist across rounds (bandit memory)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
