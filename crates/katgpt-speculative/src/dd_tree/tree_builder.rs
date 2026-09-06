// TreeBuilder — extracted from dd_tree.rs (Issue 162 C2, 2026-07-17)
// Pre-allocated buffer pool + build methods for zero-alloc DDTree construction.

use super::*;

/// Pre-allocated buffers for zero-alloc DDTree building.
///
/// Create once with `TreeBuilder::new(config)`, reuse across calls.
/// `build()` clears and reuses internal buffers — no allocation on steady state.
pub struct TreeBuilder {
    heap: BinaryHeap<TreeNode>,
    pub(crate) tree: Vec<TreeNode>,
    chain_nodes: Vec<TreeNode>,
    chain_parent_tokens: Vec<usize>,
    parent_tokens_buf: Vec<usize>,
    candidates_buf: Vec<usize>,
    valid_buf: Vec<bool>,
    /// Reusable per-depth budget counter for the progressive / depth-budgeted
    /// build variants. Hoisted out of the per-build scope to avoid one Vec
    /// allocation per build call.
    #[cfg(any(
        feature = "dflare_progressive_budget",
        feature = "corr_budget",
        feature = "nf_flow_budget"
    ))]
    depth_used_buf: Vec<usize>,
    /// Cached `ln(marginals[d][i])` — computed once per build to avoid redundant
    /// `f32::ln()` calls in the Phase C expansion inner loop (called per token
    /// per heap-pop). Entries for `prob <= 0.0` are `0.0` (unused since those
    /// tokens are skipped before the lookup).
    log_marginals: Vec<Vec<f32>>,
    /// Plan 424 Phase 6 / paper §3.5 Figure 6: when `Some(t)`, tree expansion
    /// beyond depth `t` uses argmax-of-marginal (single greedy child per node)
    /// instead of pushing all valid tokens into the heap. Shallow depths keep
    /// full branching; deep depths go greedy where the draft marginal has
    /// degraded. Default `None` = full branching everywhere (current behavior).
    /// The paper observes a draft-length crossover (~2–4) past which argmax
    /// beats full-marginal on mean acceptance length.
    deep_argmax_threshold: Option<usize>,
    /// Issue 699 T2 — optional answer-space halt monitor (structural CoT,
    /// TRACE arXiv:2510.07880, `structural_cot_halt` feature). When `Some`,
    /// the `build()` expansion loop feeds each popped node's dominant token
    /// to the monitor and cuts the build on the first `Halt` vote — the
    /// answer-space analog of the score-based early-exit patience. `None`
    /// (default) = bit-identical legacy behavior.
    #[cfg(feature = "structural_cot_halt")]
    structural_halt: Option<katgpt_core::structural_cot_halt::StructuralTraceMonitor>,
}

impl TreeBuilder {
    /// Allocate all buffers from config dimensions.
    pub fn new(config: &katgpt_types::Config) -> Self {
        Self {
            heap: BinaryHeap::new(),
            tree: Vec::with_capacity(config.tree_budget),
            chain_nodes: Vec::with_capacity(config.draft_lookahead),
            chain_parent_tokens: Vec::with_capacity(config.draft_lookahead),
            parent_tokens_buf: vec![0usize; config.draft_lookahead + 1],
            candidates_buf: Vec::with_capacity(config.vocab_size),
            valid_buf: Vec::with_capacity(config.vocab_size),
            #[cfg(any(
                feature = "dflare_progressive_budget",
                feature = "corr_budget",
                feature = "nf_flow_budget"
            ))]
            depth_used_buf: Vec::new(),
            log_marginals: Vec::new(),
            deep_argmax_threshold: None,
            #[cfg(feature = "structural_cot_halt")]
            structural_halt: None,
        }
    }

    /// Issue 699 T2 — install/remove the answer-space halt monitor (opt-in
    /// `structural_cot_halt`). `None` restores the legacy behavior exactly.
    /// The monitor is episode-shaped: call
    /// `monitor.reset()` between builds (or install a fresh one) when one
    /// builder serves many queries.
    #[cfg(feature = "structural_cot_halt")]
    pub fn set_structural_halt_monitor(
        &mut self,
        monitor: Option<katgpt_core::structural_cot_halt::StructuralTraceMonitor>,
    ) -> &mut Self {
        self.structural_halt = monitor;
        self
    }

    /// Set the deep-argmax threshold (Plan 424 Phase 6, paper §3.5).
    /// When set, tree nodes at depth > threshold only expand the
    /// argmax-of-marginal token, not all valid tokens. Default `None`
    /// (full branching at all depths).
    #[inline]
    pub fn set_deep_argmax_threshold(&mut self, threshold: Option<usize>) -> &mut Self {
        self.deep_argmax_threshold = threshold;
        self
    }

    /// Pre-compute `ln(prob)` for every token in every marginal depth.
    ///
    /// Reuses inner `Vec` allocations across builds (clear + refill pattern).
    /// The Phase C expansion loop calls `prob.ln()` once per token per heap-pop;
    /// caching turns that O(budget × vocab) `ln` calls into O(depths × vocab).
    #[inline]
    fn cache_log_marginals(&mut self, marginals: &[&[f32]]) {
        // Grow the outer Vec if needed; existing inner Vecs are reused below.
        if self.log_marginals.len() < marginals.len() {
            self.log_marginals.resize_with(marginals.len(), Vec::new);
        } else {
            self.log_marginals.truncate(marginals.len());
        }
        for (log_m, &m) in self.log_marginals.iter_mut().zip(marginals) {
            log_m.clear();
            log_m.reserve(m.len());
            // Branch-free: `ln(0)` would be -inf, but those entries are never
            // read (the expansion loop skips `prob <= 0.0` before indexing).
            for &p in m {
                log_m.push(if p > 0.0 { p.ln() } else { 0.0 });
            }
        }
    }

    /// Build DDTree from marginals, reusing pre-allocated buffers.
    ///
    /// Clears and reuses `heap`, `tree`, `chain_nodes`, `chain_parent_tokens`.
    /// Returns a borrowed slice valid until the next `build()` call.
    pub fn build(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        pruner: &dyn ConstraintPruner,
        chain_seed: bool,
    ) -> &[TreeNode] {
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        self.cache_log_marginals(marginals);

        if chain_seed {
            // ── Phase A: Build greedy chain backbone ──────────────
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path = TreePath::default();

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                // Find argmax token at this depth
                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                // Chain breaks if argmax has zero prob or is pruned
                if prob <= 0.0 || !pruner.is_valid(depth, token_idx, &self.chain_parent_tokens) {
                    break;
                }

                cumulative_score += self.log_marginals[depth][token_idx];
                let node_path = parent_path.push(token_idx as u32, depth);

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // No chain built — fall back to original root seeding
                if config.vocab_size > 256 {
                    // Batch validate: collect candidates with prob>0, validate all
                    // in one batch_is_valid call, then create nodes.
                    // Reuse pre-allocated candidates_buf to avoid per-build allocation.
                    self.candidates_buf.clear();
                    self.candidates_buf.extend(
                        marginals[0]
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &prob)| if prob > 0.0 { Some(i) } else { None }),
                    );
                    self.valid_buf.clear();
                    self.valid_buf.resize(self.candidates_buf.len(), false);
                    pruner.batch_is_valid(0, &self.candidates_buf, &[], &mut self.valid_buf);
                    if self.candidates_buf.len() >= RAYON_CANDIDATE_THRESHOLD {
                        let nodes: Vec<TreeNode> = self
                            .candidates_buf
                            .par_iter()
                            .zip(self.valid_buf.par_iter())
                            .filter_map(|(&i, &v)| {
                                if v {
                                    Some(TreeNode {
                                        score: self.log_marginals[0][i],
                                        depth: 0,
                                        token_idx: i,
                                        parent_path: TreePath::root(i as u32),
                                    })
                                } else {
                                    None
                                }
                            })
                            .collect();
                        self.heap.extend(nodes);
                    } else {
                        for (&i, &v) in self.candidates_buf.iter().zip(self.valid_buf.iter()) {
                            if v {
                                self.heap.push(TreeNode {
                                    score: self.log_marginals[0][i],
                                    depth: 0,
                                    token_idx: i,
                                    parent_path: TreePath::root(i as u32),
                                });
                            }
                        }
                    }
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                            self.heap.push(TreeNode {
                                score: self.log_marginals[0][i],
                                depth: 0,
                                token_idx: i,
                                parent_path: TreePath::root(i as u32),
                            });
                        }
                    }
                }
            } else {
                // Seed siblings at each chain depth
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    // Parent tokens for pruning: chain tokens at depths 0..depth-1
                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path.parent(chain_node.depth),
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob > 0.0 && pruner.is_valid(depth, i, sibling_parent_tokens) {
                            let sibling_path = chain_node.parent_path.parent(chain_node.depth).push(i as u32, depth);

                            self.heap.push(TreeNode {
                                score: parent_chain_score + self.log_marginals[depth][i],
                                depth,
                                token_idx: i,
                                parent_path: sibling_path,
                            });
                        }
                    }
                }

                // Seed children of the last chain node
                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                            self.heap.push(TreeNode {
                                score: last.score + self.log_marginals[next_depth][i],
                                depth: next_depth,
                                token_idx: i,
                                parent_path: last.parent_path.push(i as u32, next_depth),
                            });
                        }
                    }
                }
            }
        } else {
            // Original behavior: seed heap with root's children, filtered by pruner
            if config.vocab_size > 256 {
                // Batch validate: collect candidates with prob>0, validate all
                // in one batch_is_valid call, then create nodes.
                // Reuse pre-allocated candidates_buf to avoid per-build allocation.
                self.candidates_buf.clear();
                self.candidates_buf.extend(
                    marginals[0]
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| if prob > 0.0 { Some(i) } else { None }),
                );
                self.valid_buf.clear();
                self.valid_buf.resize(self.candidates_buf.len(), false);
                pruner.batch_is_valid(0, &self.candidates_buf, &[], &mut self.valid_buf);
                if self.candidates_buf.len() >= RAYON_CANDIDATE_THRESHOLD {
                    let nodes: Vec<TreeNode> = self
                        .candidates_buf
                        .par_iter()
                        .zip(self.valid_buf.par_iter())
                        .filter_map(|(&i, &v)| {
                            if v {
                                Some(TreeNode {
                                    score: self.log_marginals[0][i],
                                    depth: 0,
                                    token_idx: i,
                                    parent_path: TreePath::root(i as u32),
                                })
                            } else {
                                None
                            }
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (&i, &v) in self.candidates_buf.iter().zip(self.valid_buf.iter()) {
                        if v {
                            self.heap.push(TreeNode {
                                score: self.log_marginals[0][i],
                                depth: 0,
                                token_idx: i,
                                parent_path: TreePath::root(i as u32),
                            });
                        }
                    }
                }
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob > 0.0 && pruner.is_valid(0, i, &[]) {
                        self.heap.push(TreeNode {
                            score: self.log_marginals[0][i],
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        });
                    }
                }
            }
        }

        // ── Phase C: Standard best-first expansion ────────────────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    // Not a new best — track running second best (degrades with heap)
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            // ── Issue 699 T2: answer-space halt (structural CoT, TRACE
            // arXiv:2510.07880). Optional monitor keyed on the popped node's
            // dominant token — cycles in the best-first token stream (the
            // answer-space analog of score patience) cut the expansion
            // early. `None` (default) = bit-identical legacy behavior; the
            // whole block is feature-gated and the call allocates nothing.
            #[cfg(feature = "structural_cot_halt")]
            if let Some(monitor) = self.structural_halt.as_mut() {
                match monitor.step_key(best.token_idx as u64) {
                    katgpt_core::structural_cot_halt::StructuralHaltDecision::Halt { .. } => {
                        break;
                    }
                    katgpt_core::structural_cot_halt::StructuralHaltDecision::Continue => {}
                }
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                // Extract parent tokens from path bitfield for path-aware pruning
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                let log_m = &self.log_marginals[next_depth];
                let depth_marginal = marginals[next_depth];

                if self.deep_argmax_threshold.is_some_and(|t| next_depth > t) {
                    // Plan 424 Phase 6 (paper §3.5): at deep positions the
                    // marginal is diluted (averages over many possible
                    // prefixes). Expanding only the argmax token concentrates
                    // tree budget on the most-likely path, matching the paper's
                    // finding that argmax-of-marginal outperforms full-marginal
                    // sampling at draft length > crossover (~2–4).
                    let mut best_idx = usize::MAX;
                    let mut best_prob = 0.0f32;
                    for (i, &prob) in depth_marginal.iter().enumerate() {
                        if prob > best_prob && pruner.is_valid(next_depth, i, parent_tokens) {
                            best_idx = i;
                            best_prob = prob;
                        }
                    }
                    if best_idx != usize::MAX {
                        self.heap.push(TreeNode {
                            score: best.score + log_m[best_idx],
                            depth: next_depth,
                            token_idx: best_idx,
                            parent_path: best.parent_path.push(best_idx as u32, next_depth),
                        });
                    }
                } else {
                    for (i, &prob) in depth_marginal.iter().enumerate() {
                        // NEURO-SYMBOLIC INTERCEPT: prune before adding to heap
                        if prob > 0.0 && pruner.is_valid(next_depth, i, parent_tokens) {
                            self.heap.push(TreeNode {
                                score: best.score + log_m[i],
                                depth: next_depth,
                                token_idx: i,
                                parent_path: best.parent_path.push(i as u32, next_depth),
                            });
                        }
                    }
                }
            }
        }

        &self.tree
    }

    /// Build tree and merge retrieved branches in one call.
    ///
    /// For REST feature: builds the DDTree, then calls `merge_retrieved_branches`
    /// on the internal tree buffer. Returns a borrowed slice valid until the
    /// next `build()` or `build_and_merge()` call.
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_merge(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        pruner: &dyn ConstraintPruner,
        chain_seed: bool,
        token_sequences: &[Vec<usize>],
        scores: &[f32],
        rest_weight: f32,
    ) -> &[TreeNode] {
        self.build(marginals, config, pruner, chain_seed);
        merge_retrieved_branches(
            &mut self.tree,
            marginals,
            config,
            token_sequences,
            scores,
            rest_weight,
        );
        &self.tree
    }

    /// Consume the builder and return the tree as an owned `Vec`.
    pub fn into_tree(self) -> Vec<TreeNode> {
        self.tree
    }

    /// Build DDTree with graded relevance screening (Plan 021).
    ///
    /// Like `build()` but uses [`ScreeningPruner`] for continuous relevance
    /// instead of binary [`ConstraintPruner`]. The relevance score `R ∈ [0.0, 1.0]`
    /// is blended into log-prob space: `score += ln(P_llm) + ln(R)`.
    ///
    /// Branches with `relevance <= config.screening_threshold` are hard-trimmed.
    pub fn build_screened(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
    ) -> &[TreeNode] {
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        self.cache_log_marginals(marginals);

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with screening ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path = TreePath::default();

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                // Blended score: ln(P_llm) + ln(R). Use the cached ln(prob).
                cumulative_score +=
                    self.log_marginals[depth][token_idx] + relevance.ln();
                let node_path = parent_path.push(token_idx as u32, depth);

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // Slice length is a sufficient condition for the rayon threshold:
                // if the whole marginal has <512 entries it certainly has <512
                // positive ones. Avoids a full O(vocab) counting pass that the
                // par_iter below redoes anyway via filter_map.
                if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: self.log_marginals[0][i] + relevance.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: TreePath::root(i as u32),
                            })
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: self.log_marginals[0][i] + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path.parent(chain_node.depth),
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = chain_node.parent_path.parent(chain_node.depth).push(i as u32, depth);

                        self.heap.push(TreeNode {
                            score: parent_chain_score + self.log_marginals[depth][i] + relevance.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + self.log_marginals[next_depth][i] + relevance.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: last.parent_path.push(i as u32, next_depth),
                        });
                    }
                }
            }
        } else {
            // Original seeding with screening. Slice length is a sufficient
            // condition for the rayon threshold (see Phase B note above).
            if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: self.log_marginals[0][i] + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        })
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: self.log_marginals[0][i] + relevance.ln(),
                        depth: 0,
                        token_idx: i,
                        parent_path: TreePath::root(i as u32),
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with screening ─────────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    // Not a new best — track running second best (degrades with heap)
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                let log_m = &self.log_marginals[next_depth];
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    // SCREENING: ln(P_llm) + ln(R) blended score
                    self.heap.push(TreeNode {
                        score: best.score + log_m[i] + relevance.ln(),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: best.parent_path.push(i as u32, next_depth),
                    });
                }
            }
        }

        &self.tree
    }

    /// Build tree with screening and merge retrieved branches in one call.
    #[allow(clippy::too_many_arguments)]
    pub fn build_and_merge_screened(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        token_sequences: &[Vec<usize>],
        scores: &[f32],
        rest_weight: f32,
    ) -> &[TreeNode] {
        self.build_screened(marginals, config, screener, chain_seed);
        merge_retrieved_branches(
            &mut self.tree,
            marginals,
            config,
            token_sequences,
            scores,
            rest_weight,
        );
        &self.tree
    }

    /// Build DDTree with GFlowNet backward-weighted scoring (Plan 052).
    ///
    /// Generalization of [`Self::build_screened`] with tunable backward weight
    /// and flow bonus. The paper's `single_state_beam_search` scores beams
    /// using ONLY backward logits. We blend because our WASM `relevance()`
    /// is coarser than a trained neural P_B.
    ///
    /// # Scoring Formula
    ///
    /// ```text
    /// score = ln(P_llm) + backward_weight × ln(R) + lambda_flow × (1 - stop_prob[depth])
    /// ```
    ///
    /// - `backward_weight = 1.0, lambda_flow = 0.0` → identical to `build_screened`
    /// - `backward_weight = 2.0` → backward relevance counts 2× more than forward
    /// - `backward_weight = 4.0` → near-pure backward (paper's approach)
    ///
    /// # Arguments
    ///
    /// * `marginals` — Per-depth token probability distributions
    /// * `config` — DDTree configuration
    /// * `screener` — Screening pruner for relevance scoring
    /// * `chain_seed` — Whether to build greedy chain backbone first
    /// * `stop_probs` — Per-depth EOS probability from marginals
    /// * `backward_weight` — Weight for backward relevance in scoring
    /// * `lambda_flow` — Flow regularization strength
    #[allow(clippy::too_many_arguments)]
    pub fn build_balanced(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        stop_probs: &[f32],
        backward_weight: f32,
        lambda_flow: f32,
    ) -> &[TreeNode] {
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        self.cache_log_marginals(marginals);

        // Helper: compute balanced score for a node
        // score = ln(P_llm) + backward_weight × ln(R) + lambda_flow × (1 - stop_prob[depth])
        let balanced_score = |prob: f32, relevance: f32, depth: usize| -> f32 {
            let r_safe = relevance.max(1e-10); // Avoid ln(0)
            let flow_bonus = lambda_flow * (1.0 - stop_probs.get(depth).copied().unwrap_or(0.5));
            prob.ln() + backward_weight * r_safe.ln() + flow_bonus
        };

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with balanced scoring ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path = TreePath::default();

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                cumulative_score += balanced_score(prob, relevance, depth);
                let node_path = parent_path.push(token_idx as u32, depth);

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // Slice length is a sufficient condition for the rayon threshold
                // (see build_screened Phase B note). Avoids an O(vocab) count.
                if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: balanced_score(prob, relevance, 0),
                                depth: 0,
                                token_idx: i,
                                parent_path: TreePath::root(i as u32),
                            })
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: balanced_score(prob, relevance, 0),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path.parent(chain_node.depth),
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = chain_node.parent_path.parent(chain_node.depth).push(i as u32, depth);

                        self.heap.push(TreeNode {
                            score: parent_chain_score + balanced_score(prob, relevance, depth),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + balanced_score(prob, relevance, next_depth),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: last.parent_path.push(i as u32, next_depth),
                        });
                    }
                }
            }
        } else {
            // Original seeding with balanced scoring. Slice length is a
            // sufficient condition for the rayon threshold (see Phase B note).
            if marginals[0].len() >= RAYON_CANDIDATE_THRESHOLD {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: balanced_score(prob, relevance, 0),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        })
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: balanced_score(prob, relevance, 0),
                        depth: 0,
                        token_idx: i,
                        parent_path: TreePath::root(i as u32),
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with balanced scoring ──
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                // Hoist flow_bonus: depends only on next_depth, not token `i`.
                let flow_bonus =
                    lambda_flow * (1.0 - stop_probs.get(next_depth).copied().unwrap_or(0.5));
                let log_m = &self.log_marginals[next_depth];
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    // BALANCED: ln(P_llm) + backward_weight × ln(R) + flow_bonus
                    let r_safe = relevance.max(1e-10); // Avoid ln(0)
                    self.heap.push(TreeNode {
                        score: best.score + log_m[i] + backward_weight * r_safe.ln() + flow_bonus,
                        depth: next_depth,
                        token_idx: i,
                        parent_path: best.parent_path.push(i as u32, next_depth),
                    });
                }
            }
        }

        &self.tree
    }

    // ── Plan 392 (2026-07-05): extended TreeBuilder methods moved from
    // katgpt-rs/src/speculative/dd_tree.rs. Compose leaf-resident siblings
    // and katgpt_core::speculative::types. Verbatim port with import rewrites.

    /// Build DDTree with progressive per-depth budget allocation (Plan 174 Task 3b).
    ///
    /// Like [`Self::build_screened`] but distributes `tree_budget` unevenly
    /// across depths using `PositionWeightedBudget`. Early depths get more
    /// nodes (higher weight), later depths get fewer (exponential decay).
    ///
    /// When `budget_config` is `None` or `budget_config.enabled == false`,
    /// delegates to [`Self::build_screened`] unchanged (zero overhead).
    ///
    /// The total node count stays within `config.tree_budget` regardless of
    /// the per-depth allocation.
    #[cfg(feature = "dflare_progressive_budget")]
    pub fn build_screened_progressive(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        budget_config: Option<&katgpt_core::speculative::types::PositionWeightedBudget>,
    ) -> &[TreeNode] {
        // Delegate to original when feature is not active
        let Some(bcfg) = budget_config else {
            return self.build_screened(marginals, config, screener, chain_seed);
        };
        if !bcfg.enabled {
            return self.build_screened(marginals, config, screener, chain_seed);
        }

        // Compute per-depth budget allocation
        let depth_budgets = bcfg.allocate(config.tree_budget, marginals.len());
        // Reuse the per-build depth_used buffer (clear + resize preserves the
        // inner allocation across builds). Accessed as `self.depth_used_buf`
        // throughout because the build loop interleaves `depth_used[d] += 1`
        // with `self.tree.push(...)` — a separate `let` binding would conflict
        // with the other self borrows.
        self.depth_used_buf.clear();
        self.depth_used_buf.resize(depth_budgets.len(), 0);

        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        self.cache_log_marginals(marginals);

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with progressive budget ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path = TreePath::default();

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }
                // Per-depth budget check for chain backbone
                if self.depth_used_buf[depth] >= depth_budgets[depth] {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                // Blended score: ln(P_llm) + ln(R). Use the cached ln(prob) from
                // cache_log_marginals (computed once per build).
                cumulative_score +=
                    self.log_marginals[depth][token_idx] + relevance.ln();
                let node_path = parent_path.push(token_idx as u32, depth);

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.depth_used_buf[depth] += 1;
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                // Seed depth 0 — only add tokens within depth 0 budget
                let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
                if config.vocab_size > 256 {
                    let mut nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: self.log_marginals[0][i] + relevance.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: TreePath::root(i as u32),
                            })
                        })
                        .collect();
                    nodes.truncate(budget_d0);
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if self.depth_used_buf[0] >= budget_d0 {
                            break;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: self.log_marginals[0][i] + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path.parent(chain_node.depth),
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = chain_node.parent_path.parent(chain_node.depth).push(i as u32, depth);

                        self.heap.push(TreeNode {
                            score: parent_chain_score + self.log_marginals[depth][i] + relevance.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + self.log_marginals[next_depth][i] + relevance.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: last.parent_path.push(i as u32, next_depth),
                        });
                    }
                }
            }
        } else {
            // Original seeding with progressive budget for depth 0
            let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
            if config.vocab_size > 256 {
                let mut nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: self.log_marginals[0][i] + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        })
                    })
                    .collect();
                nodes.truncate(budget_d0);
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if self.depth_used_buf[0] >= budget_d0 {
                        break;
                    }
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: self.log_marginals[0][i] + relevance.ln(),
                        depth: 0,
                        token_idx: i,
                        parent_path: TreePath::root(i as u32),
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with progressive per-depth budget ──
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };

            // Per-depth budget check: skip nodes whose depth is exhausted
            if best.depth < depth_budgets.len()
                && self.depth_used_buf[best.depth] >= depth_budgets[best.depth]
            {
                continue;
            }

            self.tree.push(best);
            self.depth_used_buf[best.depth] += 1;

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                // Skip expanding children into a depth that has exhausted its budget
                if next_depth < depth_budgets.len()
                    && self.depth_used_buf[next_depth] >= depth_budgets[next_depth]
                {
                    continue;
                }
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );
                let log_m = &self.log_marginals[next_depth];
                for (i, &prob) in marginals[next_depth].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: best.score + log_m[i] + relevance.ln(),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: best.parent_path.push(i as u32, next_depth),
                    });
                }
            }
        }

        &self.tree
    }

    /// Build DDTree with externally-provided per-depth budget caps (Plan 200).
    ///
    /// Identical to [`Self::build_screened_progressive`] but accepts pre-computed
    /// `depth_budgets` directly instead of computing them from `PositionWeightedBudget`.
    ///
    /// This is the integration point for `CorrelationBudgetAllocator` — the allocator
    /// produces `depth_budgets` from EMA-tracked agreement rates, and this method
    /// enforces them during tree expansion.
    #[cfg(any(feature = "corr_budget", feature = "nf_flow_budget"))]
    pub fn build_screened_with_depth_budgets(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        depth_budgets: &[usize],
    ) -> &[TreeNode] {
        if depth_budgets.is_empty() {
            return self.build_screened(marginals, config, screener, chain_seed);
        }

        // Reuse the per-build depth_used buffer (see build_screened_progressive).
        self.depth_used_buf.clear();
        self.depth_used_buf.resize(depth_budgets.len(), 0);
        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        self.cache_log_marginals(marginals);

        if chain_seed {
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path = TreePath::default();

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }
                if depth >= depth_budgets.len() || self.depth_used_buf[depth] >= depth_budgets[depth] {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                cumulative_score +=

                    self.log_marginals[depth][token_idx] + relevance.ln();
                let node_path = parent_path.push(token_idx as u32, depth);

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.depth_used_buf[depth] += 1;
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // Seed heap with siblings
            if self.chain_nodes.is_empty() {
                let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if self.depth_used_buf[0] >= budget_d0 {
                        break;
                    }
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: self.log_marginals[0][i] + relevance.ln(),
                        depth: 0,
                        token_idx: i,
                        parent_path: TreePath::root(i as u32),
                    });
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path.parent(chain_node.depth),
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = chain_node.parent_path.parent(chain_node.depth).push(i as u32, depth);

                        self.heap.push(TreeNode {
                            score: parent_chain_score + self.log_marginals[depth][i] + relevance.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + self.log_marginals[next_depth][i] + relevance.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: last.parent_path.push(i as u32, next_depth),
                        });
                    }
                }
            }
        } else {
            let budget_d0 = depth_budgets.first().copied().unwrap_or(config.tree_budget);
            for (i, &prob) in marginals[0].iter().enumerate() {
                if self.depth_used_buf[0] >= budget_d0 {
                    break;
                }
                if prob <= 0.0 {
                    continue;
                }
                let relevance = screener.relevance(0, i, &[]);
                if relevance <= threshold {
                    continue;
                }
                self.heap.push(TreeNode {
                    score: self.log_marginals[0][i] + relevance.ln(),
                    depth: 0,
                    token_idx: i,
                    parent_path: TreePath::root(i as u32),
                });
            }
        }

        // Best-first expansion with per-depth budget caps
        let mut best_score: Option<f32> = None;
        let mut _second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };

            if best.depth < depth_budgets.len()
                && self.depth_used_buf[best.depth] >= depth_budgets[best.depth]
            {
                continue;
            }

            self.tree.push(best);
            self.depth_used_buf[best.depth] += 1;

            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    let gap = bs - score;
                    if gap > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                        _second_best_score = Some(score);
                    }
                }
            }

            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
            {
                break;
            }

            // Expand children
            let next_depth = best.depth + 1;
            if next_depth >= marginals.len() {
                continue;
            }
            let parent_tokens = extract_parent_tokens_into(
                best.parent_path,
                next_depth,
                &mut self.parent_tokens_buf,
            );
            let log_m = &self.log_marginals[next_depth];
            for (i, &prob) in marginals[next_depth].iter().enumerate() {
                if prob <= 0.0 {
                    continue;
                }
                let relevance = screener.relevance(next_depth, i, parent_tokens);
                if relevance <= threshold {
                    continue;
                }
                self.heap.push(TreeNode {
                    score: score + log_m[i] + relevance.ln(),
                    depth: next_depth,
                    token_idx: i,
                    parent_path: best.parent_path.push(i as u32, next_depth),
                });
            }
        }

        &self.tree
    }

    /// Build DDTree with graded relevance screening AND RecFM cross-scale consistency.
    ///
    /// Identical to [`Self::build_screened`] but additionally filters branches whose
    /// probability velocity violates cross-scale consistency (RecFM Theorem 3.1).
    ///
    /// Branches are pruned when `|v₂ − α·v₁| > threshold`, where:
    /// - `v₁` = velocity at parent depth (change in top-1 probability)
    /// - `v₂` = velocity at current depth
    /// - `α` = scale factor from [`CrossScaleConfig::scale_alpha`]
    ///
    /// When `recfm_config.enable == false`, delegates to [`Self::build_screened`] (zero overhead).
    #[cfg(feature = "recfm")]
    pub fn build_screened_recfm(
        &mut self,
        marginals: &[&[f32]],
        config: &katgpt_types::Config,
        screener: &dyn ScreeningPruner,
        chain_seed: bool,
        recfm_config: &CrossScaleConfig,
    ) -> &[TreeNode] {
        if !recfm_config.enable {
            return self.build_screened(marginals, config, screener, chain_seed);
        }

        let threshold = config.screening_threshold;
        self.heap.clear();
        self.tree.clear();
        self.chain_nodes.clear();
        self.chain_parent_tokens.clear();

        if marginals.is_empty() {
            return &self.tree;
        }

        self.cache_log_marginals(marginals);

        // Track velocity at each depth for cross-scale consistency checks
        let mut prev_velocity: f32 = 0.0;

        if chain_seed {
            // ── Phase A: Build greedy chain backbone with screening + RecFM ──
            let mut cumulative_score: f32 = 0.0;
            let mut parent_path = TreePath::default();

            for (depth, marginal) in marginals.iter().enumerate() {
                if self.tree.len() >= config.tree_budget {
                    break;
                }

                let best_token = marginal
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| katgpt_core::float_order::cmp_for_max(**a, **b))
                    .map(|(i, _)| i);

                let Some(token_idx) = best_token else {
                    break;
                };
                let prob = marginal[token_idx];

                if prob <= 0.0 {
                    break;
                }

                let relevance = screener.relevance(depth, token_idx, &self.chain_parent_tokens);
                if relevance <= threshold {
                    break;
                }

                // RecFM cross-scale consistency check
                let marginal_prev = if depth > 0 { marginals[depth - 1] } else { &[] };
                let velocity = branch_velocity_at(depth, marginal, marginal_prev);
                if depth > 0
                    && !cross_scale_consistent(
                        prev_velocity,
                        velocity,
                        recfm_config.scale_alpha,
                        recfm_config.consistency_threshold,
                    )
                {
                    // Branch violates cross-scale consistency — prune
                    break;
                }
                prev_velocity = velocity;

                // Blended score: ln(P_llm) + ln(R). Use the cached ln(prob).
                cumulative_score +=
                    self.log_marginals[depth][token_idx] + relevance.ln();
                let node_path = parent_path.push(token_idx as u32, depth);

                let node = TreeNode {
                    score: cumulative_score,
                    depth,
                    token_idx,
                    parent_path: node_path,
                };

                self.tree.push(node);
                self.chain_nodes.push(node);
                parent_path = node_path;
                self.chain_parent_tokens.push(token_idx);
            }

            // ── Phase B: Seed heap with siblings + last chain children ──
            if self.chain_nodes.is_empty() {
                if config.vocab_size > 256 {
                    let nodes: Vec<TreeNode> = marginals[0]
                        .par_iter()
                        .enumerate()
                        .filter_map(|(i, &prob)| {
                            if prob <= 0.0 {
                                return None;
                            }
                            let relevance = screener.relevance(0, i, &[]);
                            if relevance <= threshold {
                                return None;
                            }
                            Some(TreeNode {
                                score: self.log_marginals[0][i] + relevance.ln(),
                                depth: 0,
                                token_idx: i,
                                parent_path: TreePath::root(i as u32),
                            })
                        })
                        .collect();
                    self.heap.extend(nodes);
                } else {
                    for (i, &prob) in marginals[0].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: self.log_marginals[0][i] + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        });
                    }
                }
            } else {
                for chain_node in &self.chain_nodes {
                    let depth = chain_node.depth;
                    let parent_chain_score = if depth == 0 {
                        0.0f32
                    } else {
                        self.chain_nodes[depth - 1].score
                    };

                    let sibling_parent_tokens = extract_parent_tokens_into(
                        chain_node.parent_path.parent(chain_node.depth),
                        depth,
                        &mut self.parent_tokens_buf,
                    );

                    for (i, &prob) in marginals[depth].iter().enumerate() {
                        if i == chain_node.token_idx {
                            continue;
                        }
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(depth, i, sibling_parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        let sibling_path = chain_node.parent_path.parent(chain_node.depth).push(i as u32, depth);

                        self.heap.push(TreeNode {
                            score: parent_chain_score + self.log_marginals[depth][i] + relevance.ln(),
                            depth,
                            token_idx: i,
                            parent_path: sibling_path,
                        });
                    }
                }

                let last = self.chain_nodes.last().unwrap();
                if last.depth + 1 < marginals.len() {
                    let next_depth = last.depth + 1;
                    let parent_tokens = extract_parent_tokens_into(
                        last.parent_path,
                        last.depth + 1,
                        &mut self.parent_tokens_buf,
                    );
                    for (i, &prob) in marginals[next_depth].iter().enumerate() {
                        if prob <= 0.0 {
                            continue;
                        }
                        let relevance = screener.relevance(next_depth, i, parent_tokens);
                        if relevance <= threshold {
                            continue;
                        }
                        self.heap.push(TreeNode {
                            score: last.score + self.log_marginals[next_depth][i] + relevance.ln(),
                            depth: next_depth,
                            token_idx: i,
                            parent_path: last.parent_path.push(i as u32, next_depth),
                        });
                    }
                }
            }
        } else {
            // Original seeding with screening (no chain seed)
            if config.vocab_size > 256 {
                let nodes: Vec<TreeNode> = marginals[0]
                    .par_iter()
                    .enumerate()
                    .filter_map(|(i, &prob)| {
                        if prob <= 0.0 {
                            return None;
                        }
                        let relevance = screener.relevance(0, i, &[]);
                        if relevance <= threshold {
                            return None;
                        }
                        Some(TreeNode {
                            score: self.log_marginals[0][i] + relevance.ln(),
                            depth: 0,
                            token_idx: i,
                            parent_path: TreePath::root(i as u32),
                        })
                    })
                    .collect();
                self.heap.extend(nodes);
            } else {
                for (i, &prob) in marginals[0].iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(0, i, &[]);
                    if relevance <= threshold {
                        continue;
                    }
                    self.heap.push(TreeNode {
                        score: self.log_marginals[0][i] + relevance.ln(),
                        depth: 0,
                        token_idx: i,
                        parent_path: TreePath::root(i as u32),
                    });
                }
            }
        }

        // ── Phase C: Best-first expansion with screening + RecFM ─────
        let mut best_score: Option<f32> = None;
        let mut second_best_score: Option<f32> = None;
        let mut consecutive_dominant: usize = 0;
        while self.tree.len() < config.tree_budget {
            let Some(best) = self.heap.pop() else {
                break;
            };
            self.tree.push(best);

            // Confidence-gap early exit (Plan 026: AutoTTS)
            let score = best.score;
            match best_score {
                None => {
                    best_score = Some(score);
                }
                Some(bs) if score > bs => {
                    second_best_score = Some(bs);
                    best_score = Some(score);
                    consecutive_dominant = 1;
                }
                Some(bs) => {
                    second_best_score = Some(score);
                    if bs - score > config.early_exit_gap {
                        consecutive_dominant += 1;
                    } else {
                        consecutive_dominant = 0;
                    }
                }
            }
            if config.early_exit_patience > 0
                && config.early_exit_gap > 0.0
                && consecutive_dominant >= config.early_exit_patience
                && best_score.unwrap_or(0.0) - second_best_score.unwrap_or(0.0)
                    > config.early_exit_gap
            {
                break;
            }

            if best.depth + 1 < marginals.len() {
                let next_depth = best.depth + 1;
                let parent_tokens = extract_parent_tokens_into(
                    best.parent_path,
                    best.depth + 1,
                    &mut self.parent_tokens_buf,
                );

                // RecFM: child velocity does not depend on token index `i` —
                // it's a property of the (parent_depth, child_depth) marginal
                // transition. Compute once, was per-token (O(V²) per expansion).
                let child_marginal = marginals[next_depth];
                let parent_marginal = marginals[best.depth];
                let parent_velocity = branch_velocity_at(
                    best.depth,
                    parent_marginal,
                    if best.depth > 0 {
                        marginals[best.depth - 1]
                    } else {
                        &[]
                    },
                );
                let child_velocity =
                    branch_velocity_at(next_depth, child_marginal, parent_marginal);

                // Hoist cross_scale_consistent: its inputs (parent_velocity,
                // child_velocity, recfm_config) are loop-invariant — the result
                // is identical for every token `i`. If inconsistent, skip the
                // entire inner loop (no children added at this depth).
                if !cross_scale_consistent(
                    parent_velocity,
                    child_velocity,
                    recfm_config.scale_alpha,
                    recfm_config.consistency_threshold,
                ) {
                    continue;
                }

                let log_m = &self.log_marginals[next_depth];
                for (i, &prob) in child_marginal.iter().enumerate() {
                    if prob <= 0.0 {
                        continue;
                    }
                    let relevance = screener.relevance(next_depth, i, parent_tokens);
                    if relevance <= threshold {
                        continue;
                    }

                    self.heap.push(TreeNode {
                        score: best.score + log_m[i] + relevance.ln(),
                        depth: next_depth,
                        token_idx: i,
                        parent_path: best.parent_path.push(i as u32, next_depth),
                    });
                }
            }
        }

        &self.tree
    }
}
