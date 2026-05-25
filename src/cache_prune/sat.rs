//! Summed-Area Table (integral image) for O(1) rectangular region sum queries on attention matrices.
//!
//! Part of the **CachePrune** module (Plan 140).
//!
//! Reference: *CachePrune: KV-Cache Compression via Attention-Based Segment Pruning*
//! arXiv:2605.23640
//!
//! The summed-area table (SAT), also known as an integral image, is a data structure that
//! enables O(1) sum queries over arbitrary rectangular regions of a 2D matrix after O(n²)
//! preprocessing. This is used to efficiently compute intra- and inter-segment attention
//! scores for identifying reusable KV-cache segments.

#![allow(clippy::too_many_lines)]

/// Summed-area table (integral image) for O(1) rectangular region sum queries.
///
/// Preprocesses an n×n attention matrix in-place in O(n²) time.
/// After preprocessing, any rectangular region sum is O(1).
pub struct SummedAreaTable<'a> {
    data: &'a mut [Vec<f32>], // n×n, modified in-place
    n: usize,
}

impl<'a> SummedAreaTable<'a> {
    /// Build SAT in-place from an n×n attention matrix.
    ///
    /// Uses the standard SAT recurrence:
    /// ```text
    /// sat[i][j] = attention[i][j] + sat[i-1][j] + sat[i][j-1] - sat[i-1][j-1]
    /// ```
    ///
    /// Time: O(n²)
    pub fn build(attention: &'a mut [Vec<f32>]) -> Self {
        let n = attention.len();
        let sat = Self { data: attention, n };

        for i in 0..sat.n {
            for j in 0..sat.n {
                let val = sat.data[i][j];
                let up = if i > 0 { sat.data[i - 1][j] } else { 0.0 };
                let left = if j > 0 { sat.data[i][j - 1] } else { 0.0 };
                let up_left = if i > 0 && j > 0 {
                    sat.data[i - 1][j - 1]
                } else {
                    0.0
                };
                sat.data[i][j] = val + up + left - up_left;
            }
        }

        sat
    }

    /// Query sum of rectangular region `[x1..=x2] × [y1..=y2]`.
    ///
    /// Uses inclusion-exclusion:
    /// ```text
    /// sum = sat[x2][y2] - sat[x1-1][y2] - sat[x2][y1-1] + sat[x1-1][y1-1]
    /// ```
    ///
    /// Time: O(1)
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds or `x1 > x2` or `y1 > y2`.
    pub fn region_sum(&self, x1: usize, x2: usize, y1: usize, y2: usize) -> f32 {
        assert!(x1 <= x2, "x1 must be <= x2");
        assert!(y1 <= y2, "y1 must be <= y2");
        assert!(x2 < self.n, "x2 out of bounds");
        assert!(y2 < self.n, "y2 out of bounds");

        let full = self.data[x2][y2];
        let top = if x1 > 0 { self.data[x1 - 1][y2] } else { 0.0 };
        let left = if y1 > 0 { self.data[x2][y1 - 1] } else { 0.0 };
        let top_left = if x1 > 0 && y1 > 0 {
            self.data[x1 - 1][y1 - 1]
        } else {
            0.0
        };

        full - top - left + top_left
    }

    /// Sum of attention from positions `l..r` to positions `l..r` (intra-segment).
    ///
    /// This is the total attention weight that positions within `[l, r)` assign to
    /// each other, forming a square block on the attention matrix diagonal.
    ///
    /// Time: O(1)
    pub fn intra_attention(&self, l: usize, r: usize) -> f32 {
        assert!(l < r, "l must be < r");
        // Rows [l, r-1], columns [l, r-1]
        self.region_sum(l, r - 1, l, r - 1)
    }

    /// Sum of attention from positions `l..r` to prefix `0..l` (inter-segment).
    ///
    /// This is the total attention weight that positions within `[l, r)` assign to
    /// all positions before `l`.
    ///
    /// Time: O(1)
    pub fn inter_attention(&self, l: usize, r: usize) -> f32 {
        if l == 0 {
            return 0.0;
        }
        assert!(l < r, "l must be < r");
        // Rows [l, r-1], columns [0, l-1]
        self.region_sum(l, r - 1, 0, l - 1)
    }

    /// Contextualization score for segment `[l, r)`.
    ///
    /// Returns `intra_attention - inter_attention`. A positive score means the
    /// segment is self-contained and a good candidate for reuse (KV-cache pruning).
    pub fn contextualization_score(&self, l: usize, r: usize) -> f32 {
        self.intra_attention(l, r) - self.inter_attention(l, r)
    }

    /// Find the optimal reusable substring within `[start..=end]`.
    ///
    /// Returns `(l, r)` maximizing `contextualization_score`, subject to the
    /// segment length `r - l >= min_length`.
    ///
    /// Returns `None` if no valid segment exists (e.g., range too small).
    ///
    /// Time: O(n²) where n = end - start + 1.
    pub fn best_reusable_segment(
        &self,
        start: usize,
        end: usize,
        min_length: usize,
    ) -> Option<(usize, usize)> {
        if end < start || end - start + 1 < min_length {
            return None;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best: Option<(usize, usize)> = None;

        let mut l = start;
        loop {
            let mut r = l + min_length;
            loop {
                if r > end + 1 {
                    break;
                }
                let score = self.contextualization_score(l, r);
                if score > best_score {
                    best_score = score;
                    best = Some((l, r));
                }
                r += 1;
            }
            l += 1;
            if l + min_length > end + 1 {
                break;
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a 4×4 test matrix with known values.
    ///
    /// ```text
    /// 1  2  3  4
    /// 5  6  7  8
    /// 9  10 11 12
    /// 13 14 15 16
    /// ```
    fn test_matrix_4x4() -> Vec<Vec<f32>> {
        let mut m = Vec::with_capacity(4);
        for row in 0..4u32 {
            m.push((0..4).map(|col| (row * 4 + col + 1) as f32).collect());
        }
        m
    }

    /// Naive region sum by iterating over all elements.
    fn naive_region_sum(matrix: &[Vec<f32>], x1: usize, x2: usize, y1: usize, y2: usize) -> f32 {
        let mut sum = 0.0;
        for i in x1..=x2 {
            for j in y1..=y2 {
                sum += matrix[i][j];
            }
        }
        sum
    }

    #[test]
    fn test_sat_build_correctness() {
        // The SAT of the 4×4 test matrix should be:
        // 1   3   6   10
        // 6   14  24  36
        // 15  33  54  78
        // 28  60  96  136
        let mut data = test_matrix_4x4();
        let _sat = SummedAreaTable::build(&mut data);

        let expected = [
            [1.0, 3.0, 6.0, 10.0],
            [6.0, 14.0, 24.0, 36.0],
            [15.0, 33.0, 54.0, 78.0],
            [28.0, 60.0, 96.0, 136.0],
        ];

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (data[i][j] - expected[i][j]).abs() < 1e-6,
                    "SAT[{i}][{j}]: got {}, expected {}",
                    data[i][j],
                    expected[i][j]
                );
            }
        }
    }

    #[test]
    fn test_region_sum_matches_naive() {
        let mut data = test_matrix_4x4();
        // Keep a copy for naive computation (before build mutates in-place).
        let original = data.clone();
        let sat = SummedAreaTable::build(&mut data);

        // Test all possible rectangular regions in the 4×4 matrix.
        for x1 in 0..4 {
            for x2 in x1..4 {
                for y1 in 0..4 {
                    for y2 in y1..4 {
                        let sat_sum = sat.region_sum(x1, x2, y1, y2);
                        let naive = naive_region_sum(&original, x1, x2, y1, y2);
                        assert!(
                            (sat_sum - naive).abs() < 1e-4,
                            "region_sum({x1},{x2},{y1},{y2}): got {sat_sum}, expected {naive}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_intra_inter_attention() {
        // Use a custom 5×5 matrix where attention values make intra/inter obvious.
        //
        // Rows = query positions, cols = key positions.
        // Positions [2,3,4] attend mostly to each other:
        // ```text
        // 0.1  0.1  0.1  0.1  0.1
        // 0.1  0.1  0.1  0.1  0.1
        // 0.0  0.0  0.3  0.3  0.3
        // 0.0  0.0  0.3  0.3  0.3
        // 0.0  0.0  0.3  0.3  0.3
        // ```
        let mut data: Vec<Vec<f32>> = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
        ];

        let sat = SummedAreaTable::build(&mut data);

        // intra_attention(2, 5) = sum of rows [2,3,4] × cols [2,3,4] = 9 × 0.3 = 2.7
        let intra = sat.intra_attention(2, 5);
        assert!(
            (intra - 2.7).abs() < 1e-4,
            "intra_attention: got {intra}, expected 2.7"
        );

        // inter_attention(2, 5) = sum of rows [2,3,4] × cols [0,1] = 6 × 0.0 = 0.0
        let inter = sat.inter_attention(2, 5);
        assert!(
            (inter - 0.0).abs() < 1e-4,
            "inter_attention: got {inter}, expected 0.0"
        );

        // inter_attention(0, 2) = 0 (no prefix before l=0)
        assert!((sat.inter_attention(0, 2) - 0.0).abs() < 1e-6);

        // intra_attention(0, 2) = rows [0,1] × cols [0,1] = 4 × 0.1 = 0.4
        assert!((sat.intra_attention(0, 2) - 0.4).abs() < 1e-4);

        // inter_attention(1, 3) = rows [1,2] × cols [0,0]
        // = data[1][0] + data[2][0] = 0.1 + 0.0 = 0.1
        assert!((sat.inter_attention(1, 3) - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_contextualization_score() {
        // Using the same matrix as above.
        let mut data: Vec<Vec<f32>> = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
        ];

        let sat = SummedAreaTable::build(&mut data);

        // [2, 5): intra=2.7, inter=0.0 → score=2.7 (positive, self-contained)
        let score_self = sat.contextualization_score(2, 5);
        assert!(
            (score_self - 2.7).abs() < 1e-4,
            "score [2,5): got {score_self}"
        );

        // [0, 2): intra=0.4, inter=0.0 → score=0.4
        let score_prefix = sat.contextualization_score(0, 2);
        assert!(
            (score_prefix - 0.4).abs() < 1e-4,
            "score [0,2): got {score_prefix}"
        );

        // Verify the self-contained segment has a higher score
        assert!(score_self > score_prefix);
    }

    #[test]
    fn test_best_reusable_segment() {
        // 5×5 attention matrix where rows [2,3,4] attend only to themselves.
        //
        // ```text
        //     0   1   2   3   4
        // 0: 0.1 0.1 0.1 0.1 0.1
        // 1: 0.1 0.1 0.1 0.1 0.1
        // 2: 0.0 0.0 0.3 0.3 0.3
        // 3: 0.0 0.0 0.3 0.3 0.3
        // 4: 0.0 0.0 0.3 0.3 0.3
        // ```
        let mut data: Vec<Vec<f32>> = vec![
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.1, 0.1, 0.1, 0.1, 0.1],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
            vec![0.0, 0.0, 0.3, 0.3, 0.3],
        ];

        let sat = SummedAreaTable::build(&mut data);

        // In subrange [2..4] with min_length=3, only segment [2, 5) fits.
        // Its score: intra=2.7, inter=0.0 → 2.7
        let result = sat.best_reusable_segment(2, 4, 3);
        assert_eq!(result, Some((2, 5)), "best in [2..4] should be [2, 5)");

        // Full range [0..4] with min_length=5: only [0, 5) is possible.
        let result = sat.best_reusable_segment(0, 4, 5);
        assert_eq!(result, Some((0, 5)), "only full range with min_length=5");

        // Subrange [2..4] with min_length=2: several candidates exist.
        let result = sat.best_reusable_segment(2, 4, 2);
        assert!(result.is_some());
        let (l, r) = result.unwrap();
        assert!(l >= 2 && r <= 5 && r - l >= 2);

        // No valid segment when min_length exceeds range.
        assert!(sat.best_reusable_segment(0, 3, 5).is_none());
    }

    #[test]
    fn test_single_element() {
        // 1×1 matrix: SAT is just the value itself.
        let mut data = vec![vec![42.0]];
        let sat = SummedAreaTable::build(&mut data);

        assert!((sat.region_sum(0, 0, 0, 0) - 42.0).abs() < 1e-6);
        assert!((sat.intra_attention(0, 1) - 42.0).abs() < 1e-6);
        assert!((sat.inter_attention(0, 1) - 0.0).abs() < 1e-6);
        assert!((sat.contextualization_score(0, 1) - 42.0).abs() < 1e-6);

        let result = sat.best_reusable_segment(0, 0, 1);
        assert_eq!(result, Some((0, 1)));
    }

    #[test]
    fn test_edge_cases_no_valid_segment() {
        let mut data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let sat = SummedAreaTable::build(&mut data);

        // Range too small for min_length
        assert!(sat.best_reusable_segment(0, 1, 3).is_none());

        // Empty range (end < start)
        assert!(sat.best_reusable_segment(1, 0, 1).is_none());
    }

    #[test]
    fn test_full_matrix_sum() {
        let mut data = test_matrix_4x4();
        let sat = SummedAreaTable::build(&mut data);

        // Sum of entire 4×4 matrix = sum(1..=16) = 136
        let total = sat.region_sum(0, 3, 0, 3);
        assert!(
            (total - 136.0).abs() < 1e-4,
            "full matrix sum: got {total}, expected 136"
        );

        // Intra of full range [0, 4) = total
        assert!((sat.intra_attention(0, 4) - 136.0).abs() < 1e-4);
    }
}
