//! Tournament scheduling — matchup generation.

/// A scheduled matchup between player indices.
#[derive(Clone, Debug)]
pub struct Matchup {
    pub player_indices: Vec<usize>,
}

/// Generates all round-robin pairs from N players.
pub fn round_robin_pairs(n: usize) -> Vec<Matchup> {
    let mut matchups = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            matchups.push(Matchup {
                player_indices: vec![i, j],
            });
        }
    }
    matchups
}

/// Generates full-field matchups where all players compete simultaneously.
/// For Bomber: 4 players per match. For FFT: 8 units per match.
pub fn full_field_matchups(n: usize, field_size: usize) -> Vec<Matchup> {
    match n {
        0 => Vec::new(),
        n if n <= field_size => vec![Matchup {
            player_indices: (0..n).collect(),
        }],
        n => {
            let mut matchups = Vec::new();
            for start in 0..n {
                let indices: Vec<usize> = (0..field_size).map(|i| (start + i) % n).collect();
                matchups.push(Matchup {
                    player_indices: indices,
                });
            }
            matchups
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_pairs_zero() {
        let result = round_robin_pairs(0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_round_robin_pairs_one() {
        let result = round_robin_pairs(1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_round_robin_pairs_two() {
        let result = round_robin_pairs(2);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].player_indices, vec![0, 1]);
    }

    #[test]
    fn test_round_robin_pairs_four() {
        let result = round_robin_pairs(4);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0].player_indices, vec![0, 1]);
        assert_eq!(result[1].player_indices, vec![0, 2]);
        assert_eq!(result[2].player_indices, vec![0, 3]);
        assert_eq!(result[3].player_indices, vec![1, 2]);
        assert_eq!(result[4].player_indices, vec![1, 3]);
        assert_eq!(result[5].player_indices, vec![2, 3]);
    }

    #[test]
    fn test_round_robin_pairs_formula() {
        // n*(n-1)/2 — use checked arithmetic to avoid underflow at 0
        let expected = |n: usize| match n {
            0 | 1 => 0,
            _ => n * (n - 1) / 2,
        };
        for n in [0, 1, 2, 3, 5, 8, 10] {
            assert_eq!(round_robin_pairs(n).len(), expected(n));
        }
    }

    #[test]
    fn test_full_field_matchups_zero() {
        let result = full_field_matchups(0, 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_full_field_matchups_below_field_size() {
        let result = full_field_matchups(3, 4);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].player_indices, vec![0, 1, 2]);
    }

    #[test]
    fn test_full_field_matchups_equal_field_size() {
        let result = full_field_matchups(4, 4);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].player_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_full_field_matchups_rotation() {
        // 6 players, field_size 4 → 6 matchups with rotation
        let result = full_field_matchups(6, 4);
        assert_eq!(result.len(), 6);
        assert_eq!(result[0].player_indices, vec![0, 1, 2, 3]);
        assert_eq!(result[1].player_indices, vec![1, 2, 3, 4]);
        assert_eq!(result[2].player_indices, vec![2, 3, 4, 5]);
        assert_eq!(result[3].player_indices, vec![3, 4, 5, 0]);
        assert_eq!(result[4].player_indices, vec![4, 5, 0, 1]);
        assert_eq!(result[5].player_indices, vec![5, 0, 1, 2]);
    }

    #[test]
    fn test_full_field_matchups_fft_eight() {
        let result = full_field_matchups(8, 8);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].player_indices.len(), 8);
    }
}
