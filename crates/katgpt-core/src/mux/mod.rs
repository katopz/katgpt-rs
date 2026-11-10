//! MUX — vocabulary-simplex superposition tree search modules.
//!
//! Each submodule is gated behind a Cargo feature:
//! - `mux_pruner`   — `MuxSpanPruner` + `extract_top_k_peaks`
//! - `mux_ddtree`   — `MuxDdTree` superposition DD-tree (implies `mux_pruner`)
//! - `mux_bfs`      — `MuxBfs` dynamic-width BFS frontier (implies `mux_ddtree`)
//! - `mux_demux`    — `mux_demux` verifier
//! - `mux_bandit_width` — `MuxBanditWidth` arm selector
//! - `mux_freeze_thaw`  — `MuxTarget` / `MuxPatternStore`
//! - `mux_height_gate`  — `HeightGate` commit-timing + `GapTrendNarrower`
//!   (Issue 688, Coconut §4.4/Fig. 6; implies `mux_bfs`)

#[cfg(feature = "mux_pruner")]
pub mod span_pruner;
#[cfg(feature = "mux_pruner")]
pub mod top_k;

#[cfg(feature = "mux_pruner")]
pub mod dd_tree;

#[cfg(feature = "mux_pruner")]
pub mod bfs;

#[cfg(feature = "mux_pruner")]
pub mod height_gate;

#[cfg(feature = "mux_demux")]
pub mod demux;

#[cfg(feature = "mux_bandit_width")]
pub mod bandit_width;

#[cfg(feature = "mux_pruner")]
pub mod freeze_thaw;
