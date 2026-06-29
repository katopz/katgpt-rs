//! β warmup-decay schedule from SDPG paper.
//!
//! Warmup: linearly ramp β from 0 to β_base over warmup_steps.
//! Decay: linearly decay β from β_base to 0 over decay_steps.
//! After decay_steps: β = 0 (teacher fully phased out).

/// β warmup-decay schedule from SDPG paper.
///
/// Warmup: linearly ramp β from 0 to β_base over warmup_steps.
/// Decay: linearly decay β from β_base to 0 over decay_steps.
/// After decay_steps: β = 0 (teacher fully phased out).
#[derive(Clone, Debug)]
pub struct BetaSchedule {
    /// Maximum teacher influence strength.
    pub beta_base: f32,
    /// Steps to ramp up from 0 to β_base.
    pub warmup_steps: usize,
    /// Steps to decay from β_base to 0 (starts after warmup).
    pub decay_steps: usize,
    /// Current training step.
    pub current_step: usize,
}

impl BetaSchedule {
    pub fn new(beta_base: f32, warmup_steps: usize, decay_steps: usize) -> Self {
        Self {
            beta_base,
            warmup_steps,
            decay_steps,
            current_step: 0,
        }
    }

    /// Default schedule: β_base=0.1, warmup=100, decay=1000.
    pub fn default_schedule() -> Self {
        Self::new(0.1, 100, 1000)
    }

    /// Get current β value.
    pub fn beta(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Warmup phase: linear ramp from 0 to β_base
            if self.warmup_steps == 0 {
                return self.beta_base;
            }
            self.beta_base * (self.current_step as f32 / self.warmup_steps as f32)
        } else if self.current_step < self.warmup_steps + self.decay_steps {
            // Decay phase: linear decay from β_base to 0
            if self.decay_steps == 0 {
                return 0.0;
            }
            let progress =
                (self.current_step - self.warmup_steps) as f32 / self.decay_steps as f32;
            self.beta_base * (1.0 - progress)
        } else {
            // Post-decay: teacher fully phased out
            0.0
        }
    }

    /// Advance to next step.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Reset to step 0.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }

    /// Whether the schedule is past the warmup phase.
    pub fn is_warmed_up(&self) -> bool {
        self.current_step >= self.warmup_steps
    }

    /// Whether the schedule is fully decayed (teacher phased out).
    pub fn is_decayed(&self) -> bool {
        self.current_step >= self.warmup_steps + self.decay_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_at_step_zero() {
        let schedule = BetaSchedule::new(0.1, 100, 1000);
        assert!(
            (schedule.beta() - 0.0).abs() < 1e-6,
            "step 0 should be 0, got {}",
            schedule.beta()
        );
    }

    #[test]
    fn test_beta_at_warmup_end() {
        let schedule = BetaSchedule::new(0.1, 100, 1000);
        let mut s = schedule;
        s.current_step = 100;
        assert!(
            (s.beta() - 0.1).abs() < 1e-6,
            "at warmup end should be β_base, got {}",
            s.beta()
        );
    }

    #[test]
    fn test_beta_at_decay_end() {
        let schedule = BetaSchedule::new(0.1, 100, 1000);
        let mut s = schedule;
        s.current_step = 1100; // warmup + decay
        assert!(
            (s.beta() - 0.0).abs() < 1e-6,
            "after decay should be 0, got {}",
            s.beta()
        );
    }

    #[test]
    fn test_beta_mid_decay() {
        let schedule = BetaSchedule::new(0.1, 100, 1000);
        let mut s = schedule;
        s.current_step = 600; // 100 warmup + 500 decay = midpoint
        let beta = s.beta();
        assert!(
            beta > 0.0 && beta < 0.1,
            "mid-decay should be 0 < β < β_base, got {}",
            beta
        );
    }

    #[test]
    fn test_step_advances() {
        let mut schedule = BetaSchedule::new(0.1, 100, 1000);
        assert_eq!(schedule.current_step, 0);
        schedule.step();
        assert_eq!(schedule.current_step, 1);
    }

    #[test]
    fn test_reset() {
        let mut schedule = BetaSchedule::new(0.1, 100, 1000);
        schedule.step();
        schedule.step();
        schedule.reset();
        assert_eq!(schedule.current_step, 0);
    }
}
