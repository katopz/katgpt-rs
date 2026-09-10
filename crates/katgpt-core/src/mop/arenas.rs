//! Shared MOP arenas — deterministic domain builders for tests, benches,
//! and downstream parity harnesses (riir-ai Plan 538 consumes these to
//! compare against its private riir-poc oracle).
//!
//! Both arenas are paper-derived (Fig. 2a 4-room gridworld; the ring world)
//! but the exact placements are OUR representative constructions — the
//! golden-parity gate compares the solver against a structurally-different
//! reference on the SAME arena, not against paper figures.

/// 4-room gridworld (paper Fig. 2a shape): 9×9 grid = 81 cells + 1 DEAD.
///
/// - `N = 82` (index 81 = DEAD), `A = 4` (UP, DOWN, LEFT, RIGHT).
/// - The cross walls (row 4 / col 4) are unreachable states with no
///   available actions, EXCEPT the four door cells adjacent to the center
///   — (3,4), (5,4), (4,3), (4,5) — which connect the four 4×4 rooms
///   in a ring.
/// - 2 traps (2,2), (6,6) + 2 food (2,6), (6,2): absorbing ON ENTRY
///   (single deterministic self-loop → pinned `V = 0` exactly — the
///   paper's episode-ends-at-the-trap model).
/// - DEAD (index 81) is the reserved absorbing slot (single stay action,
///   pinned); deterministic moves never route to it in this variant.
/// - Border moves that would leave the grid stay in place.
///
/// Wall/trap/food placements are OUR representative construction of the
/// paper's Fig. 2a shape (the golden gate compares solver vs reference on
/// the SAME arena; paper-figure number matching is the riir-ai PoC's job).
pub const GRID_N: usize = 82;
pub const GRID_A: usize = 4;
/// DEAD absorbing state index.
pub const GRID_DEAD: usize = 81;

#[inline]
fn cell(r: usize, c: usize) -> usize {
    r * 9 + c
}

#[inline]
fn is_wall(r: usize, c: usize) -> bool {
    // Cross walls: full row 4 / full col 4, minus the four door cells
    // adjacent to the center. The center (4,4) itself IS a wall — the doors
    // connect the four rooms in a ring around it.
    let cross = r == 4 || c == 4;
    let door = (c == 4 && (r == 3 || r == 5)) || (r == 4 && (c == 3 || c == 5));
    cross && !door
}

#[inline]
fn is_trap_or_food(r: usize, c: usize) -> bool {
    (r == 2 || r == 6) && (c == 2 || c == 6)
}

/// Build the 4-room gridworld kernel + mask.
pub fn four_room_gridworld() -> (
    [[[f32; GRID_N]; GRID_A]; GRID_N],
    [[u8; GRID_A]; GRID_N],
) {
    let mut p = [[[0.0f32; GRID_N]; GRID_A]; GRID_N];
    let mut mask = [[0u8; GRID_A]; GRID_N];

    let moves: [(isize, isize); GRID_A] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for r in 0..9usize {
        for c in 0..9usize {
            let s = cell(r, c);
            if is_wall(r, c) {
                // Wall: unreachable, terminal mask (pinned V = 0).
                continue;
            }
            if is_trap_or_food(r, c) {
                // Trap/food: absorbing ON ENTRY (the paper's model — the
                // episode ends at the trap/food state, V = 0 exactly via
                // the single deterministic self-loop pin).
                mask[s] = [1, 0, 0, 0];
                p[s][0][s] = 1.0;
                continue;
            }
            // Room / door / border cell: deterministic moves; border exits
            // and wall hits stay in place.
            mask[s] = [1, 1, 1, 1];
            for (k, &(dr, dc)) in moves.iter().enumerate() {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                let target = if !(0..9).contains(&nr) || !(0..9).contains(&nc) {
                    s
                } else {
                    let (nr, nc) = (nr as usize, nc as usize);
                    if is_wall(nr, nc) {
                        s
                    } else {
                        cell(nr, nc)
                    }
                };
                p[s][k][target] = 1.0;
            }
        }
    }
    (p, mask)
}

/// Ring world: 16 positions + DEAD. `N = 17`, `A = 3` (CW, CCW, STAY).
///
/// Deterministic variant: every action is a δ move — `H(S'|s,a) = 0`, so
/// the β term vanishes and the closed-form fixed point is uniform:
/// `V* = α·ln 3 / (1 − γ)` for every ring state (an analytic check the
/// tests assert).
pub const RING_N: usize = 17;
pub const RING_A: usize = 3;
/// DEAD absorbing state index.
pub const RING_DEAD: usize = 16;

pub fn ring_world() -> (
    [[[f32; RING_N]; RING_A]; RING_N],
    [[u8; RING_A]; RING_N],
) {
    ring_world_noisy(0.0)
}

/// Noisy ring: with slip probability `slip ∈ [0, 0.5)`, the CW action
/// instead moves CCW (and vice versa); STAY stays deterministic. `slip = 0`
/// is the deterministic ring. Exercises `H(S'|s,a) > 0` (the β term).
pub fn ring_world_noisy(
    slip: f32,
) -> (
    [[[f32; RING_N]; RING_A]; RING_N],
    [[u8; RING_A]; RING_N],
) {
    let mut p = [[[0.0f32; RING_N]; RING_A]; RING_N];
    let mut mask = [[1u8; RING_A]; RING_N];
    // DEAD: single stay action.
    mask[RING_DEAD] = [1, 0, 0];
    p[RING_DEAD][0][RING_DEAD] = 1.0;
    for (i, row) in p.iter_mut().enumerate().take(16) {
        let cw = (i + 1) % 16;
        let ccw = (i + 15) % 16;
        row[0][cw] = 1.0 - slip;
        row[0][ccw] = slip;
        row[1][ccw] = 1.0 - slip;
        row[1][cw] = slip;
        row[2][i] = 1.0;
    }
    (p, mask)
}

/// Lethal zone on the ring (Plan 590 T2.4): states 4, 5, 6. Entering one
/// ends the episode (mask all-zero → MOP pin). Positions 2, 3, 7, 8 are
/// death-adjacent (their CW/CCW action crosses into the zone with mass
/// `1 − slip`); the arc 9..15, 0, 1 is death-free.
pub const RING_LETHAL: [usize; 3] = [4, 5, 6];

/// Ring kernel type (shared by [`ring_world_terminal`]'s triple return).
pub type RingKernel = [[[f32; RING_N]; RING_A]; RING_N];
/// Ring mask type.
pub type RingMask = [[u8; RING_A]; RING_N];
/// Ring continuation-probability table type (Plan 590).
pub type RingPsafe = [[f32; RING_A]; RING_N];

/// Terminal-zone ring: [`ring_world_noisy`] + the [`RING_LETHAL`] states
/// made terminal (mask all-zero, kernel rows zeroed) + the per-(s, a)
/// continuation-probability table `psafe[s][a] = 1 − (mass the action
/// flows into the lethal zone)` — the Plan 590 G3 arena (paper Eqs. 16-18,
/// arXiv:2609.07508). Fractional psafe arises from slip: from state 3, CW
/// enters state 4 with mass `1 − slip` → `psafe[3][CW] = slip`.
///
/// The psafe rows of terminal states are meaningless (no action leaves
/// them; MOP pins ζ = 0 regardless) and read as 1.0 by construction.
pub fn ring_world_terminal(slip: f32) -> (RingKernel, RingMask, RingPsafe) {
    let (mut p, mut mask) = ring_world_noisy(slip);
    for &s in RING_LETHAL.iter() {
        mask[s] = [0; RING_A];
        p[s] = [[0.0f32; RING_N]; RING_A];
    }
    let mut psafe = [[1.0f32; RING_A]; RING_N];
    for (i, row_i) in p.iter().enumerate() {
        for (k, row_ik) in row_i.iter().enumerate() {
            let into_lethal: f32 = RING_LETHAL.iter().map(|&l| row_ik[l]).sum();
            psafe[i][k] = 1.0 - into_lethal;
        }
    }
    (p, mask, psafe)
}
