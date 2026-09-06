//! Shared bandit statistics (extracted from mod.rs by Issue 177).
//!
//! SharedBanditStats + BanditSnapshot — thread-safe multi-agent cooperative
//! bandit state (Mutex-wrapped Q-values, UCB1 scoring, snapshot under single lock).

// ── Shared Bandit Stats ────────────────────────────────────────

/// Thread-safe shared bandit statistics for multi-agent cooperative learning.
/// Wraps bandit state in Mutex so multiple agents share one learning process.
///
/// Contention is minimal — ~1 update per ~200 tick round per agent.
/// Use `Arc<SharedBanditStats>` to share across agents.
#[cfg(feature = "bandit")]
pub struct SharedBanditStats {
    inner: std::sync::Mutex<BanditStatsInner>,
}

#[cfg(feature = "bandit")]
struct BanditStatsInner {
    q_values: Vec<f32>,
    visits: Vec<u32>,
    total_pulls: u32,
    compressed: Vec<bool>,
}

#[cfg(feature = "bandit")]
impl SharedBanditStats {
    /// Create shared stats with optimistic initialization (Q=1.0 for all arms).
    pub fn new(n_arms: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(BanditStatsInner {
                q_values: vec![1.0; n_arms],
                visits: vec![0; n_arms],
                total_pulls: 0,
                compressed: vec![false; n_arms],
            }),
        }
    }

    /// Update Q-value for `arm` after observing `reward`.
    ///
    /// Uses incremental mean: `Q(a) += (reward - Q(a)) / n(a)`.
    pub fn update(&self, arm: usize, reward: f32) {
        let mut inner = self.inner.lock().unwrap();
        if arm >= inner.q_values.len() {
            return;
        }
        inner.visits[arm] += 1;
        inner.total_pulls += 1;
        let n = inner.visits[arm] as f32;
        inner.q_values[arm] += (reward - inner.q_values[arm]) / n;
    }

    /// UCB1 score: `Q(a) + sqrt(2 * ln(N) / n(a))`.
    ///
    /// Returns `f32::MAX` for unvisited arms (must explore first).
    pub fn ucb1_score(&self, arm: usize) -> f32 {
        let inner = self.inner.lock().unwrap();
        if arm >= inner.q_values.len() || inner.visits[arm] == 0 || inner.total_pulls == 0 {
            return f32::MAX;
        }
        let q = inner.q_values[arm];
        let n = inner.visits[arm] as f32;
        let total = inner.total_pulls as f32;
        q + (2.0 * total.ln() / n).sqrt()
    }

    /// Index of the arm with highest Q-value.
    pub fn best_arm(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner
            .q_values
            .iter()
            .enumerate()
            // float_order: a NaN q-value must never win best-arm.
            .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
            .map_or(0, |(i, _)| i)
    }

    /// Whether an arm has been compressed (hard-blocked).
    pub fn is_compressed(&self, arm: usize) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.compressed.get(arm).copied().unwrap_or(false)
    }

    /// Mark an arm as compressed (hard-blocked).
    pub fn compress_arm(&self, arm: usize) {
        let mut inner = self.inner.lock().unwrap();
        if arm < inner.compressed.len() {
            inner.compressed[arm] = true;
        }
    }

    /// Total pulls across all arms.
    pub fn total_pulls(&self) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.total_pulls
    }

    /// Snapshot of (q_value, visits) for an arm under a single lock acquisition.
    ///
    /// Prefer this over calling `q_value()` + `visits()` separately to avoid
    /// acquiring the lock twice.
    pub fn arm_snapshot(&self, arm: usize) -> (f32, u32) {
        let inner = self.inner.lock().unwrap();
        let q = inner.q_values.get(arm).copied().unwrap_or(0.0);
        let v = inner.visits.get(arm).copied().unwrap_or(0);
        (q, v)
    }

    /// Visit count for an arm.
    pub fn visits(&self, arm: usize) -> u32 {
        let inner = self.inner.lock().unwrap();
        inner.visits.get(arm).copied().unwrap_or(0)
    }

    /// Q-value estimate for an arm.
    pub fn q_value(&self, arm: usize) -> f32 {
        let inner = self.inner.lock().unwrap();
        inner.q_values.get(arm).copied().unwrap_or(0.0)
    }

    /// Snapshot of all bandit state — single lock acquisition.
    ///
    /// Prefer this over calling individual accessors when you need
    /// multiple fields, to avoid repeated lock acquisitions.
    pub fn snapshot(&self) -> BanditSnapshot {
        let inner = self.inner.lock().unwrap();
        BanditSnapshot {
            q_values: inner.q_values.clone(),
            visits: inner.visits.clone(),
            total_pulls: inner.total_pulls,
            compressed: inner.compressed.clone(),
        }
    }

    /// Compute UCB1 scores for ALL arms under a single lock acquisition.
    ///
    /// Returns `f32::MAX` for unvisited arms (must explore first).
    /// Prefer this over calling `ucb1_score(arm)` N times.
    pub fn batch_ucb1(&self) -> Vec<f32> {
        let inner = self.inner.lock().unwrap();
        let n_arms = inner.q_values.len();
        if inner.total_pulls == 0 {
            return vec![f32::MAX; n_arms];
        }
        let total = inner.total_pulls as f32;
        let ln_total = 2.0_f32 * total.ln();
        inner
            .q_values
            .iter()
            .zip(inner.visits.iter())
            .map(|(&q, &n)| {
                if n == 0 {
                    f32::MAX
                } else {
                    q + (ln_total / n as f32).sqrt()
                }
            })
            .collect()
    }
}

/// Snapshot of all bandit state — single lock acquisition.
#[cfg(feature = "bandit")]
pub struct BanditSnapshot {
    pub q_values: Vec<f32>,
    pub visits: Vec<u32>,
    pub total_pulls: u32,
    pub compressed: Vec<bool>,
}
