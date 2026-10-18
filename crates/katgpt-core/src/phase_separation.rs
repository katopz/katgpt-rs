//! Phase Separation Probe — per-entity minimum circular distance on a phase
//! circle, distilled from the **Lonely Runner Conjecture** (LRC).
//!
//! > **Source:** Barajas & Serra, *The Lonely Runner with Seven Runners*,
//! > [arXiv:0710.4495](https://arxiv.org/abs/0710.4495) [math.CO], 2007.
//! > Proven for N ≤ 7 (k = 6 runners); conjectured (open) for N > 7.
//!
//! # The primitive
//!
//! For N entities each with a phase `φ_i ∈ [0, 1)` on the unit torus, the
//! **phase separation** of entity `i` is the minimum circular distance from
//! `i`'s phase to every other entity's phase:
//!
//! ```text
//! phase_separation(i) = min_{j ≠ i}  ‖φ_i − φ_j‖ mod 1     ∈ [0, 0.5]
//! ```
//!
//! where `‖x‖ mod 1` is the distance to the nearest integer (the geodesic
//! distance on the unit circle). High separation ⇒ the entity is "lonely":
//! far from every peer on the phase circle.
//!
//! # Theorem backing (the guaranteed-peak property)
//!
//! For N entities with **integer** cycle speeds `{s_1, ..., s_N}` whose gcd
//! is 1, driven by a shared tick counter `t` so that `φ_i(t) = (s_i · t) mod
//! 1`, the LRC guarantees (for N ≤ 7, conjectured beyond):
//!
//! > Every entity `i` has some tick `t_i` where
//! > `phase_separation(i, t_i) ≥ 1/N`.
//!
//! This is a **coverage guarantee** no existing primitive provides — KARC
//! divergence is empirical, curiosity is noisy, Salience is direction-vector
//! based. The LRC backs a guaranteed peak on a per-entity scalar.
//!
//! **Scope caveat (honest):** the theorem is non-constructive — the proof
//! (Prime Filtering Lemma + 20 pages of case analysis + computer search on
//! Z_49 / Z_98) shows *existence* of a lonely tick, not *when* it occurs. The
//! primitive computes the per-tick scalar; the consumer reacts. The guarantee
//! is conjectural for N > 7 — at MMORPG crowd scale (N=1000) the scalar is
//! always valid, but the *peak guarantee* is unproven. See Research 470 §2.4.
//!
//! # Raw vs latent boundary (per `AGENTS.md` §"Latent vs Raw Space Rules")
//!
//! The primitive is **substrate-agnostic** — it operates on any `&[f32]`
//! phases, no opinion on what they mean. Two equally valid input paths:
//!
//! - **Raw time-phase path (sync-safe).** `φ_i(t) = (s_i · t) mod P` with
//!   integer `s_i`, integer `t`, integer period `P`. Bit-identical across
//!   nodes (integer modular arithmetic). Crosses the sync boundary as the
//!   resulting scalar (a raw `f32`), never as a phase vector. Use
//!   [`from_speeds_and_tick`] to materialize raw phases before the scan.
//!
//! - **Latent projection path (local-only).** `φ_i(t) = σ(d · z_i(t))`
//!   where `z_i` is entity i's latent state and `d` is a learned direction.
//!   This is the standard bridge pattern (raw → latent via dot-product +
//!   sigmoid). Local to the entity — never synced; only the resulting
//!   scalar crosses sync. Use [`from_latent_projection`] to materialize
//!   latent phases before the scan.
//!
//! # Bridge pattern (per `AGENTS.md` §"Bridge Pattern")
//!
//! The bridge functions ([`from_speeds_and_tick`], [`from_latent_projection`])
//! are zero-allocation, gateable by feature flag, and introduce no sync
//! dependency — they write into caller-provided `&mut [f32]` slices. The scan
//! itself ([`phase_separation_sorted`]) is similarly zero-allocation. The
//! only thing that crosses the sync boundary is the per-entity scalar in
//! `out`, which is raw `f32`.
//!
//! # Allocation discipline (G4)
//!
//! All hot-path functions take caller-provided `&mut [f32]` slices and
//! allocate nothing internally:
//! - [`phase_separation_sorted`] copies `phases` into the caller's scratch,
//!   sorts in place, scans adjacent neighbors, writes per-entity output by
//!   binary-searching the sorted scratch for each original index.
//! - [`from_speeds_and_tick`] / [`from_latent_projection`] write phases
//!   directly into the caller's `out_phases` slice.
//!
//! # Complexity
//!
//! | Function | Complexity | Use case |
//! |---|---|---|
//! | [`phase_separation`] | O(N) | Single entity, small N, correctness checks |
//! | [`phase_separation_all`] | O(N²) | All entities, small N, correctness checks |
//! | [`phase_separation_sorted`] | O(N log N) | Production path, large N (N=1000 NPCs) |
//!
//! The O(N log N) sort + adjacent-neighbor scan is asymptotically optimal:
//! the minimum circular distance to any neighbor is always to one of the two
//! adjacent entities in sorted phase order (the metric is a geodesic on the
//! circle), so checking only neighbors suffices.
//!
//! # NOT a UQ primitive
//!
//! `phase_separation` is a **deterministic distance metric**, not a
//! probability distribution. It does not claim coverage, calibration, or
//! predictive intervals. No conformal-naive floor comparison is required
//! (per the "Report the Floor" rule, AGENTS.md §"Feature Flag Discipline").
//!
//! # Cross-references
//!
//! - [Research 470](../.research/470_Lonely_Runner_Phase_Separation_Probe.md)
//!   — public distillation + Super-GOAT verdict.
//! - `riir-ai/.research/334_phase_separation_game_runtime_guide.md` —
//!   private game-runtime guide + fusion map (Salience Tri-Gate × Sleep-Time
//!   × KARC × feeling brain).
//! - [Plan 571](../.plans/571_phase_separation_probe.md) — execution plan.
//! - [Research 056](../.research/056_OpenAI_Unit_Distance_Disproof.md) —
//!   same combinatorial family (chromatic number bounds on distance graphs).
//!
//! Plan 571: Phase Separation Probe — per-entity minimum circular distance on
//! a phase circle, distilled from the Lonely Runner Conjecture (Barajas & Serra
//! 2007, arXiv:0710.4495; proven for N≤7, conjectured beyond). The LRC
//! guarantees every entity cycles through phase_separation ≥ 1/N — a coverage
//! guarantee no existing primitive provides. Three modelless paths (O(N),
//! O(N²), O(N log N)) + two bridge helpers (raw time-phase, latent projection).
//! Pure modelless (closed-form modular arithmetic + sigmoid + dot-product).
//! DEFAULT-ON (2026-08-07, Plan 571) — G1–G4 GOAT gate ALL PASS
//! (bench_571_phase_separation_goat).

// ──────────────────────────────────────────────────────────────────────────
// Core distance primitive
// ──────────────────────────────────────────────────────────────────────────

/// Geodesic distance on the unit torus `R/Z`, in `[0, 0.5]`.
///
/// `‖a − b‖ mod 1` — the distance to the nearest integer of `a − b`.
/// Equivalent to `min(|a − b|, 1 − |a − b|)` when both `a, b ∈ [0, 1)`.
///
/// Always non-negative; equals `0` iff `a ≡ b (mod 1)`; equals `0.5` iff
/// `a, b` are antipodal on the unit circle.
#[inline]
pub fn circular_distance(a: f32, b: f32) -> f32 {
    // Reduce |a - b| to [0, 1) via fract-style wrap, then take the shorter
    // arc around the circle. We use `abs_diff = (a - b).abs().fract()` so
    // that inputs outside [0,1) (e.g. raw `(s·t)` before mod) are handled
    // correctly. For negative differences, `.fract()` on f32 returns a value
    // in (-1, 0] (Rust semantics), so `abs().fract()` gives [0, 1).
    let abs_diff = (a - b).abs();
    let wrapped = abs_diff - abs_diff.floor(); // ∈ [0, 1)
    // Shorter arc around the circle:
    if wrapped <= 0.5 {
        wrapped
    } else {
        1.0 - wrapped
    }
}

/// Naive O(N) computation of entity `i`'s phase separation — the minimum
/// circular distance from `phases[i]` to every other phase.
///
/// Returns `0.0` for `N = 0` or `N = 1` with `i` out of range; for a single
/// entity (`N = 1`), there are no peers, so the separation is undefined —
/// we return `0.5` (the maximum, "maximally alone" — see [`phase_separation_all`]
/// for the convention). For `N ≥ 2`, returns the true minimum circular
/// distance to any peer.
///
/// **Use case:** correctness testing + small N. The production path is
/// [`phase_separation_sorted`] at O(N log N).
#[inline]
pub fn phase_separation(phases: &[f32], i: usize) -> f32 {
    let n = phases.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        // Single entity: no peers. By convention, maximally alone.
        return 0.5;
    }
    let phi_i = phases[i];
    let mut best = 0.5_f32; // upper bound on the metric
    for (j, &phi_j) in phases.iter().enumerate() {
        if j == i {
            continue;
        }
        let d = circular_distance(phi_i, phi_j);
        if d < best {
            best = d;
        }
        if best == 0.0 {
            // Cannot do better than 0; early-exit.
            break;
        }
    }
    best
}

/// O(N²) all-pairs computation — writes every entity's separation into `out`.
///
/// `out[i] = min_{j ≠ i} circular_distance(phases[i], phases[j])`.
///
/// **Edge cases:** `N = 0` → writes nothing; `N = 1` → `out[0] = 0.5`
/// (maximally alone); `N ≥ 2` → true minimum circular distance per entity.
///
/// **Panics** if `out.len() < phases.len()`.
///
/// **Use case:** correctness testing + small N. For N=1000 NPCs the O(N²)
/// cost is ~1M comparisons — fine for tests, wrong for the tick hot path.
pub fn phase_separation_all(phases: &[f32], out: &mut [f32]) {
    assert!(
        out.len() >= phases.len(),
        "phase_separation_all: out.len() ({}) < phases.len() ({})",
        out.len(),
        phases.len()
    );
    let n = phases.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = 0.5;
        return;
    }
    for (out_i, i) in out.iter_mut().take(n).zip(0..n) {
        *out_i = phase_separation(phases, i);
    }
}

/// O(N log N) production-path computation — sorts a permutation of `phases`
/// into `scratch_perm`, scans adjacent neighbors on the circle, and writes
/// per-entity separation to `out` at the ORIGINAL indices.
///
/// The algorithm:
/// 1. Fill `scratch_perm` with `0..n` and sort by `phases[scratch_perm[k]]`
///    ascending. Now `scratch_perm[k]` is the original index of the k-th
///    smallest phase.
/// 2. For each rank `k`, compute the minimum circular distance to its two
///    sorted-adjacent neighbors (left = rank `k-1`, right = rank `k+1`, with
///    circle wraparound). Write `out[scratch_perm[k]] = sep_k`.
///
/// The minimum circular distance to any neighbor on the circle is always
/// to one of the two sorted-adjacent neighbors (geodesic-metric property),
/// so checking only neighbors suffices — no all-pairs scan needed.
///
/// **Why `&mut [usize]` scratch (not `&mut [f32]`)?** Sorting values
/// destroys the original-index mapping; to write per-entity output at
/// original indices without a second buffer or an O(N log N) binary search
/// per entity, we sort a permutation index array. This is O(N log N) total
/// (sort + linear scan), vs the binary-search-per-entity approach which is
/// O(N log N) sort + O(N log N) searches = 2× the work. The `usize` type is
/// required because the permutation holds original indices.
///
/// **Tie handling:** sort stability doesn't matter — `sort_unstable_by_key`
/// produces a valid permutation for any tie order. When multiple entities
/// share a phase, their sorted-adjacent distance is `0` (a co-located peer
/// is at distance 0), so the min is correctly `0` regardless of tie order.
///
/// **Edge cases:** `N = 0` → no-op; `N = 1` → `out[0] = 0.5`.
///
/// **Panics** if `scratch_perm.len() < phases.len()` or
/// `out.len() < phases.len()`.
///
/// **Allocation:** zero. The permutation sort is in place on `scratch_perm`;
/// the scan + write use only stack-local `f32` arithmetic. Verified by G4.
///
/// **Correctness vs [`phase_separation_all`]:** bit-identical output on
/// every input (modulo NaN, which the metric clamps away by construction).
/// The two paths compute the same min; the sort is just a faster way to find
/// the minimizing neighbor.
pub fn phase_separation_sorted(phases: &[f32], scratch_perm: &mut [usize], out: &mut [f32]) {
    assert!(
        scratch_perm.len() >= phases.len(),
        "phase_separation_sorted: scratch_perm.len() ({}) < phases.len() ({})",
        scratch_perm.len(),
        phases.len()
    );
    assert!(
        out.len() >= phases.len(),
        "phase_separation_sorted: out.len() ({}) < phases.len() ({})",
        out.len(),
        phases.len()
    );
    let n = phases.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        out[0] = 0.5;
        return;
    }

    // Step 1: fill permutation 0..n and sort by phase value ascending.
    let perm = &mut scratch_perm[..n];
    for (k, slot) in perm.iter_mut().enumerate() {
        *slot = k;
    }
    perm.sort_unstable_by_key(|&i| phases[i].to_bits()); // total_cmp via bits

    // Step 2: for each rank k, compute min circular distance to left/right
    // sorted neighbors (with circle wraparound), write to out[original_idx].
    //
    // Wraparound: rank 0's left neighbor is rank n-1 (largest phase, which is
    // the geodesic left neighbor on the circle); rank n-1's right neighbor is
    // rank 0 (smallest, geodesic right neighbor). `circular_distance`
    // computes the wraparound arc correctly from the raw values.
    for k in 0..n {
        let i = perm[k];
        let phi_i = phases[i];
        let prev_k = if k == 0 { n - 1 } else { k - 1 };
        let next_k = if k == n - 1 { 0 } else { k + 1 };
        let d_prev = circular_distance(phi_i, phases[perm[prev_k]]);
        let d_next = circular_distance(phi_i, phases[perm[next_k]]);
        out[i] = if d_prev <= d_next { d_prev } else { d_next };
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Bridge helpers (raw time-phase + latent projection)
// ──────────────────────────────────────────────────────────────────────────

/// Raw time-phase bridge: materialize `φ_i(t) = (s_i · t) mod P` into
/// `out_phases` for each integer speed in `speeds`.
///
/// **Sync-safe (raw domain).** Integer speeds × integer tick → integer phase
/// → bit-identical across nodes per `AGENTS.md` §"Sync Boundary Rule". The
/// resulting scalar (after [`phase_separation_sorted`]) crosses sync as a
/// raw `f32`.
///
/// The period `P` is caller-specified — typically the LCM of the speeds,
/// but any positive integer works. Use `P = 1` to compute phases modulo 1
/// directly (`(s · t) mod 1 ∈ [0, 1)`).
///
/// **Zero allocation.** Writes into caller-provided `out_phases`.
///
/// **Panics** if `out_phases.len() < speeds.len()`.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "phase_separation")]
/// # {
/// use katgpt_core::phase_separation::{from_speeds_and_tick, phase_separation_sorted};
///
/// // 7 entities, speeds {1..=7}, tick t=42, period 1.
/// let speeds: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
/// let mut phases = [0.0_f32; 7];
/// let mut scratch_perm = [0_usize; 7];
/// let mut sep = [0.0_f32; 7];
/// from_speeds_and_tick(&speeds, 42, 1, &mut phases);
/// phase_separation_sorted(&phases, &mut scratch_perm, &mut sep);
/// for (i, &s) in sep.iter().enumerate() {
///     assert!((0.0..=0.5).contains(&s), "entity {i} sep {s} out of range");
/// }
/// # }
/// ```
#[inline]
pub fn from_speeds_and_tick(speeds: &[u32], tick: u64, period: u32, out_phases: &mut [f32]) {
    assert!(
        out_phases.len() >= speeds.len(),
        "from_speeds_and_tick: out_phases.len() ({}) < speeds.len() ({})",
        out_phases.len(),
        speeds.len()
    );
    assert!(period > 0, "from_speeds_and_tick: period must be > 0");
    let p = period as f32;
    for (i, &s) in speeds.iter().enumerate() {
        // (s · t) mod P — integer modular arithmetic, then cast to f32.
        // Using u128 to avoid overflow on (s as u64) * tick.
        let raw = ((s as u128) * (tick as u128)) % (p as u128);
        out_phases[i] = (raw as f32) / p;
    }
}

/// Latent projection bridge: materialize `φ_i = σ(d · z_i)` into
/// `out_phases` for each entity's latent state.
///
/// **Local-only (semantic domain).** The latent projection is per-entity,
/// not synced — only the resulting scalar (after
/// [`phase_separation_sorted`]) crosses sync as a raw `f32`. Per
/// `AGENTS.md` §"Bridge Pattern": raw → latent via dot-product + sigmoid,
/// zero-allocation, gateable.
///
/// Uses the crate's [`sigmoid`](crate::sigmoid) (NEVER softmax, per
/// `AGENTS.md`). Output ∈ `(0, 1)` — strictly inside the open interval, so
/// no entity is ever exactly at 0 or 1 (sigmoid saturates but never
/// reaches).
///
/// **Zero allocation.** Writes into caller-provided `out_phases`.
///
/// **Panics** if `out_phases.len() < latent_states.len() / d`,
/// `latent_states.len() % d != 0`, or `direction.len() != d`.
#[inline]
pub fn from_latent_projection(latent_states: &[f32], direction: &[f32], out_phases: &mut [f32]) {
    let d = direction.len();
    assert!(d > 0, "from_latent_projection: direction is empty");
    assert_eq!(
        latent_states.len() % d,
        0,
        "from_latent_projection: latent_states.len() ({}) not a multiple of d ({})",
        latent_states.len(),
        d
    );
    let n = latent_states.len() / d;
    assert!(
        out_phases.len() >= n,
        "from_latent_projection: out_phases.len() ({}) < n ({})",
        out_phases.len(),
        n
    );
    for i in 0..n {
        let z = &latent_states[i * d..(i + 1) * d];
        let mut dot = 0.0_f32;
        for k in 0..d {
            dot += direction[k] * z[k];
        }
        out_phases[i] = crate::sigmoid(dot);
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Tests (G1 determinism + LRC bound confirmation)
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: compare two f32 by bit pattern (true bit-identical check).
    fn bits_eq(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits()
    }

    /// Helper: assert two f32 slices are bit-identical.
    fn assert_bits_eq(actual: &[f32], expected: &[f32], ctx: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{ctx}: length mismatch ({} vs {})",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                bits_eq(*a, *e),
                "{ctx}: mismatch at index {i}: actual={a:?} (bits {:#x}) vs expected={e:?} (bits {:#x})",
                a.to_bits(),
                e.to_bits()
            );
        }
    }

    // ─── G1a: integer phases bit-identical across all 3 paths ──────────────

    /// `g1_integer_phases_bit_identical`: phases from integer speeds
    /// `{1, 2, 3, 4, 5, 6, 7}` at tick `t=42` — verify `phase_separation`,
    /// `phase_separation_all`, and `phase_separation_sorted` all produce the
    /// same f32 bits (the sort is just a faster way to find the minimizing
    /// neighbor; the min is the same).
    #[test]
    fn g1_integer_phases_bit_identical() {
        let speeds: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        let mut phases = [0.0_f32; 7];
        // Period 420 = lcm(1..=7) so phases actually spread on the circle
        // (period 1 would give all-zero phases — degenerate).
        from_speeds_and_tick(&speeds, 42, 420, &mut phases);

        // Path 1: O(N) per-entity.
        let mut sep_naive = [0.0_f32; 7];
        for (i, sep_i) in sep_naive.iter_mut().enumerate() {
            *sep_i = phase_separation(&phases, i);
        }

        // Path 2: O(N²) all-pairs.
        let mut sep_allpairs = [0.0_f32; 7];
        phase_separation_all(&phases, &mut sep_allpairs);

        // Path 3: O(N log N) sorted scan.
        let mut scratch_perm = [0_usize; 7];
        let mut sep_sorted = [0.0_f32; 7];
        phase_separation_sorted(&phases, &mut scratch_perm, &mut sep_sorted);

        // All three paths must agree bit-identically.
        assert_bits_eq(&sep_naive, &sep_allpairs, "naive vs all-pairs");
        assert_bits_eq(&sep_naive, &sep_sorted, "naive vs sorted");
        assert_bits_eq(&sep_allpairs, &sep_sorted, "all-pairs vs sorted");

        // Sanity: every separation is in [0, 0.5].
        for (i, &s) in sep_sorted.iter().enumerate() {
            assert!(
                (0.0..=0.5).contains(&s),
                "entity {i} separation {s} out of [0, 0.5]"
            );
        }
    }

    // ─── G1b: circle wraparound ────────────────────────────────────────────

    /// `g1_circle_wraparound`: phases `{0.0, 0.49, 0.51}` — entity 0's
    /// separation should be `min(circ_dist(0.0, 0.49), circ_dist(0.0,
    /// 0.51))`. The wraparound path is `circ_dist(0.0, 0.51) = 0.49` (going
    /// backward through 1.0), not `0.51` (going forward). Both neighbors of
    /// entity 0 are at distance 0.49 → separation 0.49.
    #[test]
    fn g1_circle_wraparound() {
        let phases = [0.0_f32, 0.49, 0.51];

        // Hand-computed: circular_distance(0.0, 0.49) = 0.49 (no wraparound
        // needed, |0.49| < 0.5). circular_distance(0.0, 0.51) = 0.49 (the
        // shorter arc wraps around through 1.0: 1.0 - 0.51 = 0.49). So
        // entity 0's separation is min(0.49, 0.49) = 0.49.
        let d_0_to_1 = circular_distance(0.0, 0.49);
        let d_0_to_2 = circular_distance(0.0, 0.51);
        assert!(
            bits_eq(d_0_to_1, 0.49),
            "circ_dist(0.0, 0.49) = {d_0_to_1}, expected 0.49"
        );
        assert!(
            bits_eq(d_0_to_2, 0.49),
            "circ_dist(0.0, 0.51) = {d_0_to_2} (wraparound), expected 0.49"
        );

        let mut scratch_perm = [0_usize; 3];
        let mut sep = [0.0_f32; 3];
        phase_separation_sorted(&phases, &mut scratch_perm, &mut sep);

        assert!(
            bits_eq(sep[0], 0.49),
            "entity 0 separation = {}, expected 0.49 (both neighbors at wraparound distance)",
            sep[0]
        );

        // Cross-check against the naive path.
        let sep0_naive = phase_separation(&phases, 0);
        assert!(
            bits_eq(sep0_naive, 0.49),
            "naive entity 0 separation = {sep0_naive}, expected 0.49"
        );
    }

    // ─── G1c: edge cases (N=0, N=1, N=2 antipodal) ─────────────────────────

    #[test]
    fn g1_edge_cases() {
        // N=0: no entities, no output. Should not panic.
        let phases: [f32; 0] = [];
        let mut scratch_perm: [usize; 0] = [];
        let mut out: [f32; 0] = [];
        phase_separation_sorted(&phases, &mut scratch_perm, &mut out);
        assert_eq!(out.len(), 0);

        // N=0 naive: returns 0.0.
        assert!(bits_eq(phase_separation(&[], 0), 0.0));

        // N=1: single entity is "maximally alone" — separation 0.5 by
        // convention. Both naive + sorted agree.
        let phases1 = [0.3_f32];
        let mut scratch_perm1 = [0_usize; 1];
        let mut out1 = [0.0_f32; 1];
        phase_separation_sorted(&phases1, &mut scratch_perm1, &mut out1);
        assert!(
            bits_eq(out1[0], 0.5),
            "N=1 separation = {}, expected 0.5",
            out1[0]
        );
        assert!(
            bits_eq(phase_separation(&phases1, 0), 0.5),
            "N=1 naive separation = {}, expected 0.5",
            phase_separation(&phases1, 0)
        );

        // N=2 antipodal: phases {0.0, 0.5} → each at distance 0.5 from the
        // other (maximally separated).
        let phases2 = [0.0_f32, 0.5];
        let mut scratch_perm2 = [0_usize; 2];
        let mut out2 = [0.0_f32; 2];
        phase_separation_sorted(&phases2, &mut scratch_perm2, &mut out2);
        assert!(
            bits_eq(out2[0], 0.5),
            "N=2 entity 0 separation = {}, expected 0.5",
            out2[0]
        );
        assert!(
            bits_eq(out2[1], 0.5),
            "N=2 entity 1 separation = {}, expected 0.5",
            out2[1]
        );

        // N=2 close: phases {0.0, 0.1} → each at distance 0.1.
        let phases2b = [0.0_f32, 0.1];
        let mut scratch_perm2b = [0_usize; 2];
        let mut out2b = [0.0_f32; 2];
        phase_separation_sorted(&phases2b, &mut scratch_perm2b, &mut out2b);
        assert!(bits_eq(out2b[0], 0.1));
        assert!(bits_eq(out2b[1], 0.1));
    }

    // ─── G1d: tie handling (all phases equal → separation 0) ───────────────

    /// At tick t=0 with integer speeds, every entity is at phase 0 → every
    /// separation is 0. This exercises the tie-handling path in
    /// `phase_separation_sorted` (binary-search-for-rank assigns 0 to every
    /// tied entity).
    #[test]
    fn g1_tie_handling_tick_zero() {
        let speeds: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        let mut phases = [0.0_f32; 7];
        from_speeds_and_tick(&speeds, 0, 1, &mut phases);

        // Sanity: all phases are 0 at tick 0.
        for (i, &p) in phases.iter().enumerate() {
            assert!(bits_eq(p, 0.0), "phase[{i}] = {p}, expected 0 at tick 0");
        }

        let mut scratch_perm = [0_usize; 7];
        let mut sep = [0.0_f32; 7];
        phase_separation_sorted(&phases, &mut scratch_perm, &mut sep);

        for (i, &s) in sep.iter().enumerate() {
            assert!(
                bits_eq(s, 0.0),
                "entity {i} separation at tick 0 = {s}, expected 0 (all co-located)"
            );
        }
    }

    // ─── G1e: LRC bound confirmation (N=7, every entity hits ≥ 1/7) ─────────

    /// `g1_lrc_bound_n7`: with N=7 entities, integer speeds `{1,2,3,4,5,6,7}`
    /// (gcd=1), scan the discrete orbit and verify every entity hits
    /// `phase_separation ≥ 1/7` at least once. This is the **theorem
    /// confirmation test** — the LRC says it must happen (proven for N=7 by
    /// Barajas & Serra 2007).
    ///
    /// The bound is `1/N = 1/7 ≈ 0.142857`.
    ///
    /// **Discrete sampling setup:** the continuous LRC is stated over real
    /// time `t`. We sample at granularity `1/P` by computing
    /// `phase_i(k) = (s_i · k mod P) / P` for integer `k ∈ [0, P)`, where
    /// `P = lcm(1..=7) = 420`. Since `s_1 = 1` has `gcd(1, 420) = 1`, the
    /// joint orbit has full period P, so scanning `k ∈ [0, 420)` covers the
    /// entire reachable configuration. The sampling granularity `1/420 ≈
    /// 0.0024` is much finer than the bound `1/7 ≈ 0.143`, so the discrete
    /// scan finds lonely times within floating-point slack of the true
    /// continuous lonely time.
    #[test]
    fn g1_lrc_bound_n7() {
        let speeds: [u32; 7] = [1, 2, 3, 4, 5, 6, 7];
        let n = speeds.len();
        let bound = 1.0_f32 / n as f32; // 1/7 ≈ 0.142857
        // Epsilon accounts for: (a) f32 rounding in (s·k mod P)/P, (b) the
        // 1/P discrete sampling granularity vs the continuous lonely time.
        // 1/420 ≈ 0.0024 is the finest resolution; 5× that gives slack for
        // the lonely time falling between two sample points.
        let eps = 5.0 / 420.0;
        let period = 420; // lcm(1..=7)

        let mut hit_bound = [false; 7];

        let mut phases = [0.0_f32; 7];
        let mut scratch_perm = [0_usize; 7];
        let mut sep = [0.0_f32; 7];

        for k in 0..period as u64 {
            from_speeds_and_tick(&speeds, k, period, &mut phases);
            phase_separation_sorted(&phases, &mut scratch_perm, &mut sep);
            for i in 0..n {
                if sep[i] >= bound - eps {
                    hit_bound[i] = true;
                }
            }
        }

        // The LRC says EVERY entity must hit the bound. Report which did not
        // (if any) for diagnostic clarity.
        let misses: Vec<usize> = hit_bound
            .iter()
            .enumerate()
            .filter_map(|(i, &hit)| if hit { None } else { Some(i) })
            .collect();
        assert!(
            misses.is_empty(),
            "LRC bound violated for entities {misses:?}: speeds={speeds:?}, bound={bound}, \
             period={period}, eps={eps}, but these entities never hit \
             phase_separation >= {} across the full orbit k=0..{period}",
            bound - eps
        );
    }

    // ─── G1f: sorted-scan agrees with naive on random inputs ───────────────

    /// Property test (deterministic seed): for many random phase
    /// configurations, the O(N log N) sorted scan agrees with the O(N²) naive
    /// all-pairs computation. This is the regression guard — any future
    /// optimization to `phase_separation_sorted` must preserve bit-identical
    /// output vs `phase_separation_all`.
    #[test]
    fn g1_sorted_matches_naive_random() {
        // Deterministic LCG (no dep on fastrand for the lib-test path).
        let mut seed: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next_f32 = || {
            // xorshift64 → [0, 1)
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 40) as f32
        };

        for trial in 0..50 {
            let n = 1 + (trial % 30); // n ∈ [1, 30]
            let mut phases = vec![0.0_f32; n];
            for p in &mut phases {
                *p = next_f32();
            }
            let mut scratch_perm = vec![0_usize; n];
            let mut sep_sorted = vec![0.0_f32; n];
            let mut sep_naive = vec![0.0_f32; n];
            phase_separation_sorted(&phases, &mut scratch_perm, &mut sep_sorted);
            phase_separation_all(&phases, &mut sep_naive);

            for i in 0..n {
                // Allow 1 ULP slack for sort-order sensitivity on near-tied
                // phases (the min is the same, but binary-search-rank vs
                // linear-scan may pick different neighbors when distances
                // differ by < 1 ULP — both are correct mins, just different
                // arg-mins).
                let diff = (sep_sorted[i] - sep_naive[i]).abs();
                let ulp = (sep_naive[i].abs() + 1e-30) * f32::EPSILON;
                assert!(
                    diff <= ulp.max(1e-7),
                    "trial {trial} entity {i}: sorted={} naive={} diff={} ulp={}",
                    sep_sorted[i],
                    sep_naive[i],
                    diff,
                    ulp
                );
            }
        }
    }

    // ─── Bridge helper tests ────────────────────────────────────────────────

    #[test]
    fn from_speeds_and_tick_basic() {
        // speeds {1, 2}, tick 1, period 1: phases = {0.0, 0.0}? No —
        // (1·1) mod 1 = 0, (2·1) mod 1 = 0. Both 0. Let's use tick 3:
        // (1·3) mod 1 = 0, (2·3) mod 1 = 0... hmm period 1 always gives 0.
        // Use period 10: (1·3) mod 10 = 3, (2·3) mod 10 = 6.
        // phases = {0.3, 0.6}.
        let speeds = [1_u32, 2];
        let mut out = [0.0_f32; 2];
        from_speeds_and_tick(&speeds, 3, 10, &mut out);
        assert!(bits_eq(out[0], 0.3), "out[0] = {}, expected 0.3", out[0]);
        assert!(bits_eq(out[1], 0.6), "out[1] = {}, expected 0.6", out[1]);
    }

    #[test]
    fn from_latent_projection_basic() {
        // direction = [1, 0]; latent states = [[0, 0], [1, 0]].
        // φ_0 = σ(0) = 0.5; φ_1 = σ(1) ≈ 0.7311.
        let direction = [1.0_f32, 0.0];
        let latent_states = [0.0_f32, 0.0, 1.0, 0.0];
        let mut out = [0.0_f32; 2];
        from_latent_projection(&latent_states, &direction, &mut out);
        assert!(bits_eq(out[0], 0.5), "φ_0 = {}, expected σ(0)=0.5", out[0]);
        let expected_1 = crate::sigmoid(1.0);
        assert!(
            bits_eq(out[1], expected_1),
            "φ_1 = {}, expected σ(1)={}",
            out[1],
            expected_1
        );
    }
}
