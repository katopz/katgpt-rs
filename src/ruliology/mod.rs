//! Ruliology Bandit — Simple Program Strategies as Bandit Arms (Plan 188).
//!
//! Wolfram's ruliology proves that exhaustive enumeration of simple programs
//! finds winning strategies that hand-design misses. This module enumerates
//! all FSM(N) strategies as bandit arms — zero training, inference-time only.
//!
//! # Architecture
//!
//! - [`SimpleProgram`] — trait for any enumerable strategy (FSM, CA, TM)
//! - [`FsmStrategy`] — deterministic finite-state machine
//! - [`FsmEnumerator`] — exhaustive FSM enumeration + round-robin tournament
//! - [`WinMatrix`] — tournament result with Pareto front
//! - [`RuliologyPruner`] — filter to Pareto-optimal arms
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use katgpt_rs::ruliology::{FsmEnumerator, matching_pennies};
//!
//! let strategies = FsmEnumerator::enumerate(2); // ~22 distinct FSMs
//! let matrix = FsmEnumerator::tournament(&strategies, 100, &matching_pennies);
//! println!("Winner: {:?}", matrix.rankings[0]);
//! ```

pub mod bandit;
pub mod ca;
pub mod fsm;
pub mod payoff;
pub mod tm;
pub mod types;

pub use bandit::{RuliologyArm, RuliologyBandit};
pub use ca::CaStrategy;
pub use fsm::{FsmEnumerator, FsmStrategy, MAX_STATES};
pub use payoff::{matching_pennies, prisoners_dilemma};
pub use tm::TmStrategy;
pub use types::{RuliologyPruner, SimpleProgram, WinMatrix};

// TL;DR: Ruliology module — exhaustive simple-program enumeration (FSM/CA/TM) as bandit arms. Phase 3: FSM + CA + TM strategies.
