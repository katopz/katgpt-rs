//! Shared Go board helpers (Issue 001 H-20).
//!
//! Previously, `board_neighbors` and `flood_group` were copy-pasted across
//! `players.rs`, `g_zero_player.rs`, and `autoresearch.rs`. This module
//! centralizes them so all three call sites share one implementation.
//!
//! These operate on raw `&[GoCell]` board slices (not `&GoState`) because
//! callers need to run them on the *post-advance* board (`new_state.board`)
//! without constructing a full `GoState`. `GoState::neighbors()` +
//! `get_group_and_liberties()` are the higher-level API for in-place state.

use super::types::GoCell;

/// Compute 4-connected neighbor flat indices for a board position.
///
/// Returns up to 4 indices. Allocates only when the caller needs an owned
/// `Vec`; prefer the `extend_board_neighbors` helper when filling a reusable
/// scratch buffer.
#[inline]
pub fn board_neighbors(idx: usize, size: usize) -> Vec<usize> {
    let row = idx / size;
    let col = idx % size;
    let mut result = Vec::with_capacity(4);
    if row > 0 {
        result.push(idx - size);
    }
    if row + 1 < size {
        result.push(idx + size);
    }
    if col > 0 {
        result.push(idx - 1);
    }
    if col + 1 < size {
        result.push(idx + 1);
    }
    result
}

/// Append 4-connected neighbor flat indices into `out` without allocating.
///
/// Use this in hot loops where a scratch buffer can be `clear()`-ed and reused.
/// `out` is **not** cleared — callers can accumulate across multiple positions.
#[inline]
pub fn extend_board_neighbors(out: &mut Vec<usize>, idx: usize, size: usize) {
    let row = idx / size;
    let col = idx % size;
    if row > 0 {
        out.push(idx - size);
    }
    if row + 1 < size {
        out.push(idx + size);
    }
    if col > 0 {
        out.push(idx - 1);
    }
    if col + 1 < size {
        out.push(idx + 1);
    }
}

/// BFS flood fill to find a connected group and its liberties.
///
/// Returns `(group_indices, liberty_indices)`. Both empty if `board[start]` is
/// not a stone.
///
/// **Hot-path note**: this allocates a `visited: Vec<bool>` of length `size*size`
/// per call. For per-move scoring loops, prefer batch-processing neighbors with
/// `extend_board_neighbors` + a caller-managed visited bitset. This helper
/// remains for code paths that call it a bounded number of times per turn.
pub fn flood_group(board: &[GoCell], start: usize, size: usize) -> (Vec<usize>, Vec<usize>) {
    let color = board[start];
    if !color.is_stone() {
        return (Vec::new(), Vec::new());
    }

    let total = size * size;
    let mut group = Vec::new();
    let mut liberties = Vec::new();
    let mut visited = vec![false; total];
    let mut stack = vec![start];

    while let Some(pos) = stack.pop() {
        if visited[pos] {
            continue;
        }
        visited[pos] = true;

        match board[pos] {
            c if c == color => {
                group.push(pos);
                for n in board_neighbors(pos, size) {
                    if !visited[n] {
                        stack.push(n);
                    }
                }
            }
            GoCell::Empty => {
                liberties.push(pos);
            }
            _ => {} // Opponent boundary
        }
    }

    (group, liberties)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors_center() {
        let ns = board_neighbors(40, 9); // (4,4) center of 9x9
        assert_eq!(ns.len(), 4);
        assert!(ns.contains(&31)); // up
        assert!(ns.contains(&49)); // down
        assert!(ns.contains(&39)); // left
        assert!(ns.contains(&41)); // right
    }

    #[test]
    fn neighbors_corner() {
        let ns = board_neighbors(0, 9); // (0,0) top-left
        assert_eq!(ns.len(), 2);
        assert!(ns.contains(&1)); // right
        assert!(ns.contains(&9)); // down
    }

    #[test]
    fn extend_matches_allocating_version() {
        for &idx in &[0usize, 40, 80, 8, 72] {
            let mut a = board_neighbors(idx, 9);
            let mut b = Vec::new();
            extend_board_neighbors(&mut b, idx, 9);
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "idx={idx}");
        }
    }

    #[test]
    fn flood_single_stone() {
        let size = 9;
        let mut board = vec![GoCell::Empty; size * size];
        board[40] = GoCell::Black;
        let (group, libs) = flood_group(&board, 40, size);
        assert_eq!(group.len(), 1);
        assert_eq!(group[0], 40);
        assert_eq!(libs.len(), 4);
    }

    #[test]
    fn flood_two_stones() {
        let size = 9;
        let mut board = vec![GoCell::Empty; size * size];
        board[40] = GoCell::Black;
        board[41] = GoCell::Black;
        let (group, libs) = flood_group(&board, 40, size);
        assert_eq!(group.len(), 2);
        assert!(group.contains(&40));
        assert!(group.contains(&41));
        // Liberties: up(31), down(49), left(39), up(32), down(50), right(42) = 6
        assert_eq!(libs.len(), 6);
    }

    #[test]
    fn flood_empty_start_returns_empty() {
        let size = 9;
        let board = vec![GoCell::Empty; size * size];
        let (group, libs) = flood_group(&board, 40, size);
        assert!(group.is_empty());
        assert!(libs.is_empty());
    }
}
