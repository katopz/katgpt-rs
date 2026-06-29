//! Split tree builder from byte n-gram frequency counts.
//!
//! **Source:** Schmidt et al. (2026). Tokenization with Split Trees. arXiv:2605.22705

use std::collections::HashMap;

use super::toast_types::{SplitNode, SplitTree};

/// Builds split trees from byte n-gram frequency counts.
pub struct SplitTreeBuilder<'a> {
    /// Byte n-gram frequency counts.
    counts: &'a HashMap<Vec<u8>, u64>,
    /// Minimum count to consider a split.
    min_count: u64,
}

impl<'a> SplitTreeBuilder<'a> {
    /// Create a new builder with n-gram counts and minimum count threshold.
    pub fn new(counts: &'a HashMap<Vec<u8>, u64>, min_count: u64) -> Self {
        Self { counts, min_count }
    }

    /// Build a split tree for the given pretoken bytes.
    pub fn build(&self, pretoken: &[u8]) -> SplitTree {
        if pretoken.is_empty() {
            return SplitTree {
                pretoken: pretoken.to_vec(),
                nodes: Vec::new(),
            };
        }

        if pretoken.len() == 1 {
            return SplitTree {
                pretoken: pretoken.to_vec(),
                nodes: vec![SplitNode {
                    start: 0,
                    end: 1,
                    left: None,
                    right: None,
                }],
            };
        }

        let mut nodes = Vec::new();
        self.build_recursive(pretoken, 0, &mut nodes);
        SplitTree {
            pretoken: pretoken.to_vec(),
            nodes,
        }
    }

    fn build_recursive(&self, s: &[u8], offset: u16, nodes: &mut Vec<SplitNode>) -> u32 {
        let node_idx = nodes.len() as u32;

        if s.len() <= 1 {
            nodes.push(SplitNode {
                start: offset,
                end: offset + s.len() as u16,
                left: None,
                right: None,
            });
            return node_idx;
        }

        // Find best split point
        let split_pos = self.best_split(s);

        // Reserve space for this node (will fill children after recursive calls)
        nodes.push(SplitNode {
            start: offset,
            end: offset + s.len() as u16,
            left: None,
            right: None,
        });

        let (left_bytes, right_bytes) = (&s[..split_pos], &s[split_pos..]);

        let left_idx = self.build_recursive(left_bytes, offset, nodes);
        let right_idx = self.build_recursive(right_bytes, offset + split_pos as u16, nodes);

        // Update the node with children
        nodes[node_idx as usize].left = Some(left_idx);
        nodes[node_idx as usize].right = Some(right_idx);

        node_idx
    }

    /// Find the best split position using min(left_count, right_count) heuristic.
    fn best_split(&self, s: &[u8]) -> usize {
        let mut best: Option<(usize, u64)> = None; // (position, score)

        for i in 1..s.len() {
            let left_score = self.counts.get(&s[..i]).copied().unwrap_or(0);
            let right_score = self.counts.get(&s[i..]).copied().unwrap_or(0);

            if left_score < self.min_count || right_score < self.min_count {
                continue;
            }

            let score = left_score.min(right_score);
            match best {
                Some((_, best_score)) if score > best_score => {
                    best = Some((i, score));
                }
                None => {
                    best = Some((i, score));
                }
                _ => {}
            }
        }

        // If no good split found, try longest known prefix
        match best {
            Some((pos, _)) => pos,
            None => self.most_known(s),
        }
    }

    /// Find the longest prefix that appears in counts.
    fn most_known(&self, s: &[u8]) -> usize {
        (1..s.len())
            .rev()
            .find(|&i| self.counts.contains_key(&s[..i]))
            .unwrap_or(s.len() / 2)
    }
}
