//! Edit budget schedule — controls how many edits are allowed per optimization step.

/// Schedule for the number of edits allowed at each optimization step.
#[derive(Debug, Clone, Copy)]
pub enum EditBudgetSchedule {
    /// Fixed budget every step.
    Constant { budget: usize },
    /// Linear interpolation from `start` to `end` over `total_steps`.
    Linear {
        start: usize,
        end: usize,
        total_steps: usize,
    },
    /// Cosine annealing from `start` down to `floor` over `total_steps`.
    Cosine {
        start: usize,
        floor: usize,
        total_steps: usize,
    },
    /// No budget limit — optimizer decides.
    Autonomous,
}

impl EditBudgetSchedule {
    /// Return the edit budget at the given optimization step.
    pub fn budget_at_step(&self, step: usize) -> usize {
        match self {
            Self::Constant { budget } => *budget,
            Self::Linear {
                start,
                end,
                total_steps,
            } => {
                if step >= *total_steps {
                    return *end;
                }
                let t = step as f64 / (*total_steps as f64);
                let budget = *start as f64 + t * (*end as f64 - *start as f64);
                budget.round() as usize
            }
            Self::Cosine {
                start,
                floor,
                total_steps,
            } => {
                if step >= *total_steps {
                    return *floor;
                }
                let t = step as f64 / (*total_steps as f64);
                let cosine_factor = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                let budget = *floor as f64 + (*start as f64 - *floor as f64) * cosine_factor;
                budget.round() as usize
            }
            Self::Autonomous => usize::MAX,
        }
    }
}
