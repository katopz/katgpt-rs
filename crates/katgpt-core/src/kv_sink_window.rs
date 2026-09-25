//! Permanent attention sinks + a bounded KV window — a **deterministic RAM
//! ceiling** for decode (Issue 841, Research 571 §B-8).
//!
//! # What was actually missing
//!
//! The lead recorded this as *"grep-clean today (nothing ships)"* and that is
//! wrong in the direction that invites a parallel system. [`crate::kv_eviction`]
//! ships the whole eviction apparatus (Plan 585, Research 523): usage-rate
//! `mass/age` scoring, `UsageScoreTable`, `select_evict_into(scores, k,
//! pinned, out)` — **which already takes a pin mask** — and `runaway_gate`,
//! the promotion gate every lossy KV policy must pass.
//!
//! What is genuinely absent is (a) anything that CONSTRUCTS that pin mask, and
//! (b) a ceiling. `select_evict_into` answers *"given a budget of k, which k
//! rows go?"*; nothing answered *"what is k, and what is the bound on live
//! rows that makes decode-time memory a function of the POLICY rather than of
//! the sequence length?"* This module is that half, and it **composes with**
//! `select_evict_into` rather than replacing it.
//!
//! # The ceiling
//!
//! ```text
//! live_slots <= n_sink + window        for every step, every sequence length
//! ```
//!
//! Positions `[0, n_sink)` — the system prompt, the tool schema — are
//! **never** evicted at any usage score. Everything outside the trailing
//! `window` positions is [`SlotClass::Stale`] and goes first. What is left is
//! ordered by whatever score the caller supplies, which is
//! [`crate::kv_eviction::score`] in the shipped case.
//!
//! # ⛔ Two fidelity regimes, and only one of them can ever be promoted
//!
//! A bounded window is **not** automatically lossy. `katgpt-attn`'s
//! `alibi_entmax_window_1p5` (Issue 747, Research 549) derives a distance
//! `d_max` beyond which entmax-1.5 attention mass is **exactly** zero for an
//! ALiBi head — evicting past it is bit-identical, not approximate. So:
//!
//! - `window >= d_max` → [`WindowFidelity::Lossless`]. Every evicted row
//!   carried exactly 0.0 mass. Not a lossy surface; AGENTS.md's
//!   lossy-promotion rule does not apply.
//! - `window < d_max`, or no `d_max` known → [`WindowFidelity::Lossy`]. Rows
//!   with real mass are dropped. **The Plan-585 `runaway_gate` on a sealed
//!   long-context eval is mandatory before any promotion** — aggregate
//!   perplexity is flat while output length runs to the cap, which is the
//!   whole point of that canary.
//!
//! `d_max` is **caller-supplied**, not imported: `katgpt-attn` is downstream
//! of this crate, and the alternative is a dependency cycle. That is the same
//! house pattern `kv_eviction` already uses for attention mass
//! (`causal_head_importance::suspect_indices`).
//!
//! # Sync boundary
//!
//! None. Pure classification over caller-owned state. Zero allocation on
//! every path that takes an `out` buffer; no RNG, no globals, no `Instant`.

use crate::float_order;

pub mod policy;

#[cfg(test)]
mod tests;

pub use policy::{SinkWindowPolicy, SlotClass, WindowFidelity};

/// Eviction-priority order under a sink + window policy, reusing `out`.
///
/// Returns up to `k` slot indices, **lowest priority to keep first**:
///
/// 1. every [`SlotClass::Stale`] row, oldest position first — these are
///    outside the window and go regardless of how much mass they once got;
/// 2. then [`SlotClass::Window`] rows by ascending `scores`, ties by
///    ascending index — the [`crate::kv_eviction::select_evict_into`] order;
/// 3. **never** a [`SlotClass::Sink`] row, at any score, for any `k`.
///
/// `positions[i]` is slot `i`'s absolute token position. `scores` and
/// `positions` are indexed alike; `scores` shorter than `positions` reads the
/// missing entries as `0.0` (evict-first), which is the conservative
/// direction for a caller that has not scored a row yet.
///
/// NaN-safe through [`float_order::cmp_for_min`] — a corrupt score orders
/// LAST under min-ordering, so it can never be evicted first.
///
/// ZERO allocation: `out` is the workspace.
///
/// # Why stale-before-score, rather than one blended key
///
/// A row outside the window is not *less useful*; it is **out of policy**,
/// and mixing the two into one comparable number means a high-scoring stale
/// row can survive and silently break the ceiling. Two ordered phases keep
/// the ceiling a structural property instead of a numerical one.
pub fn select_evict_windowed(
    policy: &SinkWindowPolicy,
    positions: &[u64],
    current_pos: u64,
    scores: &[f32],
    k: usize,
    out: &mut Vec<usize>,
) {
    out.clear();
    if k == 0 || positions.is_empty() {
        return;
    }

    // Phase 1 — stale, oldest first. Collected into `out` so the workspace is
    // the output buffer, exactly as `select_evict_into` does it.
    out.extend(
        positions
            .iter()
            .enumerate()
            .filter(|&(_, &p)| policy.classify(p, current_pos) == SlotClass::Stale)
            .map(|(i, _)| i),
    );
    let n_stale = out.len();
    if n_stale > 1 {
        // Unstable: total order (index tie-break) ⇒ identical sequence, and
        // no merge-scratch heap allocation at large n (Bench 894 G4).
        out.sort_unstable_by(|&a, &b| positions[a].cmp(&positions[b]).then(a.cmp(&b)));
    }

    // Phase 2 — window rows by score. Appended after the stale block, then
    // sorted WITHIN that block only, so phase 1's ordering is preserved.
    if n_stale < k {
        out.extend(
            positions
                .iter()
                .enumerate()
                .filter(|&(_, &p)| policy.classify(p, current_pos) == SlotClass::Window)
                .map(|(i, _)| i),
        );
        let tail = &mut out[n_stale..];
        if tail.len() > 1 {
            tail.sort_unstable_by(|&a, &b| {
                let sa = scores.get(a).copied().unwrap_or(0.0);
                let sb = scores.get(b).copied().unwrap_or(0.0);
                float_order::cmp_for_min(sa, sb).then(a.cmp(&b))
            });
        }
    }

    if out.len() > k {
        out.truncate(k);
    }
}

/// Fill `out` with the pin mask [`crate::kv_eviction::select_evict_into`]
/// consumes — `true` exactly for [`SlotClass::Sink`] slots.
///
/// The bridge for a caller that wants this module's sink rule and the shipped
/// scorer's selection, without this module's window phase. Reuses `out`; zero
/// allocation once its capacity is reached.
pub fn sink_pin_mask_into(
    policy: &SinkWindowPolicy,
    positions: &[u64],
    current_pos: u64,
    out: &mut Vec<bool>,
) {
    out.clear();
    out.extend(
        positions
            .iter()
            .map(|&p| policy.classify(p, current_pos) == SlotClass::Sink),
    );
}
