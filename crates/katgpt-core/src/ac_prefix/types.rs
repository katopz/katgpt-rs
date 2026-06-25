//! Core types for the AC-GPT arbitrary-conditional prefix primitive.

/// AC-GPT-style arbitrary-conditional prefix. Borrowed; zero owning
/// allocations.
///
/// See the [module docs](super) for the three-region attention rule and the
/// leakage-prevention argument.
pub struct AcPrefix<'a> {
    base_tokens: &'a [u32],
    /// Sorted ascending; each entry indexes INTO `base_tokens`.
    conditioning_positions: &'a [usize],
}

impl<'a> AcPrefix<'a> {
    /// Empty conditioning set — degenerates to a vanilla causal forward (this
    /// is the G3 invariant: `AcPrefix::empty(tokens)` must be bit-identical to
    /// a forward without `AcPrefix` at all).
    pub fn empty(base_tokens: &'a [u32]) -> Self {
        Self {
            base_tokens,
            conditioning_positions: &[],
        }
    }

    /// Construct from a sorted, in-range conditioning-position slice.
    ///
    /// `debug_assert`s (cheap, stripped in release):
    /// - `conditioning_positions` is sorted strictly ascending.
    /// - Every entry is `< base_tokens.len()`.
    pub fn new(base_tokens: &'a [u32], conditioning_positions: &'a [usize]) -> Self {
        debug_assert!(
            conditioning_positions.windows(2).all(|w| w[0] < w[1]),
            "conditioning_positions must be strictly ascending"
        );
        debug_assert!(
            conditioning_positions
                .iter()
                .all(|&p| p < base_tokens.len()),
            "conditioning_positions must index into base_tokens"
        );
        Self {
            base_tokens,
            conditioning_positions,
        }
    }

    #[inline]
    pub fn base_tokens(&self) -> &'a [u32] {
        self.base_tokens
    }

    #[inline]
    pub fn conditioning_positions(&self) -> &'a [usize] {
        self.conditioning_positions
    }

    /// Number of conditioning copies placed at the front.
    #[inline]
    pub fn xc_len(&self) -> usize {
        self.conditioning_positions.len()
    }

    /// Length of the augmented sequence: `|xc|` copies at the front + `|x|`
    /// original tokens.
    #[inline]
    pub fn augmented_len(&self) -> usize {
        self.base_tokens.len() + self.conditioning_positions.len()
    }

    /// Original position lookup for augmented slot `k`:
    ///   - `k < |xc|`               → `conditioning_positions[k]` (the copy
    ///     carries its source position so RoPE applies the correct rotation).
    ///   - `|xc| <= k < augmented`  → `k - |xc|` (identity position in the
    ///     original sequence).
    ///
    /// Branch-free, zero-allocation. Used by [`Self::attends`] in the r1-r1
    /// case (where it collapses to `k - |xc|`) and by
    /// [`Self::original_positions_into`].
    #[inline]
    pub fn original_pos(&self, k: usize) -> usize {
        let xc = self.conditioning_positions.len();
        if k < xc {
            // SAFETY-equivalent: bounds-checked indexing; debug builds catch OOB.
            self.conditioning_positions[k]
        } else {
            k - xc
        }
    }

    /// Write the original position for each augmented slot into `out`.
    ///
    /// The first `|xc|` slots are the conditioning copies (carry their source
    /// position `conditioning_positions[k]`); the remaining `|x|` slots are
    /// the original tokens (carry identity positions `0..|x|`).
    ///
    /// `debug_assert`s `out.len() == augmented_len()`.
    pub fn original_positions_into(&self, out: &mut [usize]) {
        let xc = self.conditioning_positions.len();
        let base_len = self.base_tokens.len();
        debug_assert_eq!(
            out.len(),
            xc + base_len,
            "out.len() must equal augmented_len"
        );
        out[..xc].copy_from_slice(self.conditioning_positions);
        for k in 0..base_len {
            out[xc + k] = k;
        }
    }

    /// Three-region attention rule — see the [module docs](super).
    ///
    /// Branch-free inner expression (boolean `&` / `|`, no short-circuit, no
    /// allocation, O(1)). In region r1 the `original_pos(k) = k - |xc|` offset
    /// cancels in the causal comparison, so `original_pos(i) >= original_pos(j)`
    /// collapses to `i >= j` — no conditioning_positions lookup needed on the
    /// hot path.
    #[inline]
    pub fn attends(&self, i: usize, j: usize) -> bool {
        // Region partition:
        //   r0 = [0, |xc|)         — conditioning copies
        //   r1 = [|xc|, augmented) — original sequence positions
        //
        // Truth table:
        //   (i ∈ r0, j ∈ r0) → true
        //   (i ∈ r1, j ∈ r0) → true
        //   (i ∈ r0, j ∈ r1) → false
        //   (i ∈ r1, j ∈ r1) → i >= j   (original_pos offset cancels in r1)
        //
        // Compact form: `j_in_r0 OR (both_in_r1 AND i >= j)`.
        // When j ∈ r0, the second clause is false (both_in_r1 requires j ∈ r1),
        // so the result is true regardless of i. When j ∈ r1, the first clause
        // is false; the result is then `both_in_r1 AND i >= j`, which is false
        // if i ∈ r0 (both_in_r1 = false) and `i >= j` if i ∈ r1.
        let xc = self.conditioning_positions.len();
        let j_in_r0 = j < xc;
        let i_in_r1 = i >= xc;
        let j_in_r1 = j >= xc;
        let both_in_r1 = i_in_r1 & j_in_r1;
        let causal_in_r1 = i >= j;
        j_in_r0 | (both_in_r1 & causal_in_r1)
    }

    /// Check whether original position `p` (an index into `base_tokens`) is a
    /// conditioning position. O(log |xc|) via binary search on the sorted
    /// `conditioning_positions` slice. Zero allocation.
    #[inline]
    pub fn is_xc_position(&self, p: usize) -> bool {
        self.conditioning_positions.binary_search(&p).is_ok()
    }

    /// **Deduplicated three-region rule** (§3.5 modelless unblock candidate,
    /// Issue 003 Phase 0 Path 2).
    ///
    /// Same as [`Self::attends`] **except** eval tokens in r1 do NOT attend to
    /// in-place conditioning tokens in r1 — they get all conditioning through
    /// the r0 copies only.
    ///
    /// # Why this exists — the doubled-signal bias
    ///
    /// The original [`Self::attends`] rule lets an eval token at original
    /// position `k` attend to an in-place `xc` token at original position `p <= k`
    /// **twice**: once via its r0 copy, once via its r1 in-place slot. On an
    /// untrained model both appearances contribute real signal, biasing the
    /// conditional likelihood. The paper resolves this via LoRA fine-tuning
    /// (→ riir-train). The modelless alternative (this method) eliminates the
    /// doubling by construction: eval tokens source ALL conditioning from r0
    /// copies, never from in-place r1 `xc`.
    ///
    /// # Correctness argument (single-layer)
    ///
    /// For a single attention layer, the K/V at any position depend only on the
    /// token embedding (not on other positions' attention). The r0 copy of `xc`
    /// at original position `p` has the **same** token, **same** RoPE rotation,
    /// **same** K/V as the in-place r1 `xc` at position `p`. Therefore:
    ///
    /// - Deduplicated AC-GPT attended set for eval at position `k`:
    ///   { all xc via r0 copies } ∪ { eval at positions <= k via r1 }
    /// - Iterative-MLM attended set for eval at position `k`:
    ///   { all xc in-place } ∪ { all positions <= k }
    ///   = { all xc } ∪ { eval at positions <= k }   (xc at <= k counted once)
    ///
    /// Both sets contain the same (token, original_position) pairs → same K/V
    /// → same attention scores → same softmax → same logprobs. The deduplicated
    /// mask makes single-pass AC-GPT **bit-identical** to iterative-MLM on a
    /// single-layer model, modellessly (no gradient descent).
    ///
    /// # Multi-layer caveat
    ///
    /// On multi-layer models the r0 copies' representations evolve through
    /// layers attending only to other r0 copies (r0→r1 is false), whereas in
    /// iterative-MLM the in-place xc attend bidirectionally to eval tokens too.
    /// The representations diverge from layer 2 onward. The G1 gate
    /// (Issue 003) uses a single-layer micro-GPT where this divergence does
    /// not arise; multi-layer equivalence remains a riir-train question.
    ///
    /// # Cost
    ///
    /// One `binary_search` (O(log |xc|)) when both `i` and `j` are in r1 and
    /// `i >= j`. This is more expensive than [`Self::attends`] (which is O(1))
    /// but still zero-allocation. Use [`Self::attends`] on the hottest paths;
    /// use this method when the modelless bias correction is required.
    #[inline]
    pub fn attends_dedup(&self, i: usize, j: usize) -> bool {
        // Same as attends, but in the (i ∈ r1, j ∈ r1) case additionally
        // require that j is NOT an in-place xc position.
        let xc = self.conditioning_positions.len();
        let j_in_r0 = j < xc;
        if j_in_r0 {
            return true;
        }
        // j ∈ r1.
        let i_in_r1 = i >= xc;
        if !i_in_r1 {
            return false; // i ∈ r0, j ∈ r1 → false (copies don't attend back)
        }
        // Both in r1. Standard causal, EXCEPT eval doesn't attend to in-place xc.
        if i < j {
            return false; // causal: i must be >= j
        }
        // i >= j, both in r1. Check if j is an in-place xc position.
        let j_original = j - xc;
        if self.is_xc_position(j_original) {
            return false; // deduplicated: eval doesn't attend to in-place xc
        }
        true
    }

    /// Write the augmented token sequence into `out`. Slot layout:
    ///   - `[0, xc_len)`         → copies: `base_tokens[conditioning_positions[k]]`
    ///   - `[xc_len, augmented)` → originals: `base_tokens` verbatim
    ///
    /// `debug_assert`s `out.len() == augmented_len()`.
    pub fn augmented_tokens_into(&self, out: &mut [u32]) {
        let xc = self.conditioning_positions.len();
        let base_len = self.base_tokens.len();
        debug_assert_eq!(
            out.len(),
            xc + base_len,
            "out.len() must equal augmented_len"
        );
        // Region 0: copies from conditioning positions.
        for (k, out_slot) in out[..xc].iter_mut().enumerate() {
            *out_slot = self.base_tokens[self.conditioning_positions[k]];
        }
        // Region 1: originals verbatim — straight slice copy.
        out[xc..xc + base_len].copy_from_slice(&self.base_tokens[..base_len]);
    }

    /// Write the loss mask into `out`:
    ///   - `0.0` for slots in region 0 (the copies — never part of the loss).
    ///   - `0.0` for slots in region 1 whose original position is in
    ///     `conditioning_positions` (these are the in-place conditioning
    ///     tokens, not eval).
    ///   - `1.0` for all other slots in region 1 (the eval positions `xe`).
    ///
    /// Membership check uses `slice::binary_search` on the sorted
    /// `conditioning_positions` — O(log |xc|) per slot, zero allocation.
    /// (Hot-path alternative would be a precomputed `Vec<bool>` lookup table;
    /// not used here because `loss_mask_into` runs once per forward, not per
    /// (i,j) pair.)
    ///
    /// `debug_assert`s `out.len() == augmented_len()`.
    pub fn loss_mask_into(&self, out: &mut [f32]) {
        let xc = self.conditioning_positions.len();
        let base_len = self.base_tokens.len();
        debug_assert_eq!(
            out.len(),
            xc + base_len,
            "out.len() must equal augmented_len"
        );
        // Region 0: copies are never in the loss.
        out[..xc].fill(0.0);
        // Region 1: original sequence positions.
        let xc_positions = self.conditioning_positions;
        for k in 0..base_len {
            let is_conditioning = xc_positions.binary_search(&k).is_ok();
            out[xc + k] = if is_conditioning { 0.0 } else { 1.0 };
        }
    }

    /// Single-pass arbitrary-conditional log-likelihood `log p(xe | xc)`.
    ///
    /// Builds the augmented sequence (`xc copies | base_tokens`), materializes
    /// the attention mask, calls `forward` once, and sums the per-position
    /// logprobs at loss_mask=1.0 positions (the eval tokens `xe`).
    ///
    /// `forward` receives:
    ///   - `augmented_tokens: &[u32]`  — the augmented sequence
    ///   - `augmented_positions: &[usize]` — original position per slot (for RoPE)
    ///   - `mask: &AcPrefixMask` — the materialized three-region attention mask
    ///   - `loss_mask: &[f32]` — 1.0 at eval positions, 0.0 elsewhere
    ///
    /// and returns per-position logprobs `Vec<f32>` (length = augmented_len).
    ///
    /// Returns the sum of logprobs at loss_mask=1.0 positions.
    pub fn conditional_logprob<F>(&self, mut forward: F) -> f32
    where
        F: FnMut(&[u32], &[usize], &AcPrefixMask, &[f32]) -> Vec<f32>,
    {
        let n = self.augmented_len();
        let mut augmented_tokens = vec![0u32; n];
        let mut augmented_positions = vec![0usize; n];
        let mut loss_mask = vec![0.0f32; n];
        self.augmented_tokens_into(&mut augmented_tokens);
        self.original_positions_into(&mut augmented_positions);
        self.loss_mask_into(&mut loss_mask);
        let mask = AcPrefixMask::materialize_from(self);
        let logprobs = forward(&augmented_tokens, &augmented_positions, &mask, &loss_mask);
        debug_assert_eq!(
            logprobs.len(),
            n,
            "forward must return one logprob per augmented slot"
        );
        let mut acc = 0.0f32;
        for (lp, m) in logprobs.iter().zip(loss_mask.iter()) {
            acc += *lp * *m;
        }
        acc
    }

    /// **Deduplicated single-pass conditional log-likelihood** — the §3.5
    /// modelless bias-correction variant (Issue 003 Phase 0 Path 2).
    ///
    /// Identical to [`Self::conditional_logprob`] except the materialized mask
    /// uses [`AcPrefixMask::materialize_dedup_from`] (eval tokens do not attend
    /// to in-place `xc` tokens in r1). See [`Self::attends_dedup`] for the
    /// correctness argument: on a single-layer model this makes AC-GPT
    /// bit-identical to iterative-MLM conditional logprob, modellessly.
    pub fn conditional_logprob_dedup<F>(&self, mut forward: F) -> f32
    where
        F: FnMut(&[u32], &[usize], &AcPrefixMask, &[f32]) -> Vec<f32>,
    {
        let n = self.augmented_len();
        let mut augmented_tokens = vec![0u32; n];
        let mut augmented_positions = vec![0usize; n];
        let mut loss_mask = vec![0.0f32; n];
        self.augmented_tokens_into(&mut augmented_tokens);
        self.original_positions_into(&mut augmented_positions);
        self.loss_mask_into(&mut loss_mask);
        let mask = AcPrefixMask::materialize_dedup_from(self);
        let logprobs = forward(&augmented_tokens, &augmented_positions, &mask, &loss_mask);
        debug_assert_eq!(
            logprobs.len(),
            n,
            "forward must return one logprob per augmented slot"
        );
        let mut acc = 0.0f32;
        for (lp, m) in logprobs.iter().zip(loss_mask.iter()) {
            acc += *lp * *m;
        }
        acc
    }

    /// Sample `xe` tokens conditionally on `xc`, left-to-right.
    ///
    /// For each eval position (loss_mask=1.0 slot) in left-to-right order:
    ///   - Forward the augmented sequence up to and including the current eval slot.
    ///   - The closure returns logits `[vocab]` at the current eval position.
    ///   - Sample via the **Gumbel-max trick** (`argmax(logit - log(-log(u)))`,
    ///     `u ~ Uniform(0,1)`). This is the cleanest sigmoid-respecting sampler:
    ///     it doesn't construct an explicit probability distribution and is
    ///     mathematically equivalent to sampling from the softmax-categorical.
    ///     The AGENTS.md "sigmoid not softmax" rule applies to blending/decision
    ///     gates, not the LM head; Gumbel-max is used here because it sidesteps
    ///     the explicit softmax while remaining exact.
    ///   - Write the sampled token into the augmented sequence at the eval slot
    ///     so later eval positions can attend to it.
    ///
    /// Conditioning copies and original conditioning positions stay fixed.
    /// Returns just the eval tokens in original order.
    pub fn conditional_sample<F>(&self, mut forward: F, rng: &mut fastrand::Rng) -> Vec<u32>
    where
        F: FnMut(&[u32], &[usize], &AcPrefixMask, &[f32], usize) -> Vec<f32>,
    {
        let n = self.augmented_len();
        let mut augmented_tokens = vec![0u32; n];
        let mut augmented_positions = vec![0usize; n];
        let mut loss_mask = vec![0.0f32; n];
        self.augmented_tokens_into(&mut augmented_tokens);
        self.original_positions_into(&mut augmented_positions);
        self.loss_mask_into(&mut loss_mask);
        let mask = AcPrefixMask::materialize_from(self);

        // Walk eval slots left-to-right. We forward the *entire* augmented
        // sequence each step (the closure may cache internally); the current
        // eval slot index is passed so the closure can return its logits.
        let mut sampled = Vec::with_capacity(n);
        for eval_slot in 0..n {
            if loss_mask[eval_slot] == 0.0 {
                continue;
            }
            let logits = forward(
                &augmented_tokens,
                &augmented_positions,
                &mask,
                &loss_mask,
                eval_slot,
            );
            let token = gumbel_max_sample(&logits, rng);
            augmented_tokens[eval_slot] = token;
            sampled.push(token);
        }
        sampled
    }
}

/// Gumbel-max sampler: `argmax_i (logit_i + g_i)` where `g_i = -log(-log(u_i))`,
/// `u_i ~ Uniform(0,1)`. Mathematically equivalent to sampling from
/// `softmax(logits)` without ever materializing the categorical distribution.
///
/// Returns `0` on an empty input. Redraws when `u_i == 0.0` to keep `log`
/// finite (matches the existing `sample_token` defensive redraw).
#[inline]
pub(crate) fn gumbel_max_sample(logits: &[f32], rng: &mut fastrand::Rng) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        let mut u = rng.f32();
        while u <= 0.0 || u >= 1.0 {
            u = rng.f32();
        }
        // Gumbel(0,1) sample: g = -ln(-ln(u)). Both -ln(u) and ln(-ln(u)) are
        // real-valued for u in (0,1), so no NaN risk after the redraw guard.
        let neg_ln_u = -u.ln();
        let g = -neg_ln_u.ln();
        let score = l + g;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Bit-packed attention mask for the augmented sequence.
///
/// Layout: `augmented_len × augmented_len` bits, row-major. The bit at offset
/// `(i * augmented_len + j)` encodes `attends(i, j)`. The row length
/// (`augmented_len`) is **not** stored — callers pass it back to [`Self::get`]
/// so the struct stays a single-field transparent wrapper.
#[repr(transparent)]
pub struct AcPrefixMask {
    bits: Box<[u64]>,
}

impl AcPrefixMask {
    /// Bit-pack the [`AcPrefix::attends`] rule over the full
    /// `augmented_len × augmented_len` grid into a `Box<[u64]>` of size
    /// `ceil(augmented_len² / 64)`.
    ///
    /// This is the only allocating call in the module — run it once per
    /// augmented sequence for batched attention kernels that want a
    /// materialized mask. Hot-path callers should prefer
    /// [`AcPrefix::attends`] directly.
    pub fn materialize_from(prefix: &AcPrefix<'_>) -> Self {
        Self::materialize_with(prefix, |p, i, j| p.attends(i, j))
    }

    /// Bit-pack the [`AcPrefix::attends_dedup`] rule — the §3.5 modelless
    /// bias-correction variant (Issue 003 Phase 0 Path 2). See
    /// [`AcPrefix::attends_dedup`] for the doubled-signal-bias argument.
    ///
    /// Same allocation profile as [`Self::materialize_from`] (one `Box<[u64]>`
    /// per augmented sequence).
    pub fn materialize_dedup_from(prefix: &AcPrefix<'_>) -> Self {
        Self::materialize_with(prefix, |p, i, j| p.attends_dedup(i, j))
    }

    fn materialize_with<F: Fn(&AcPrefix<'_>, usize, usize) -> bool>(
        prefix: &AcPrefix<'_>,
        rule: F,
    ) -> Self {
        let n = prefix.augmented_len();
        let total_bits = n.checked_mul(n).expect("augmented_len squared overflows");
        let words = total_bits.div_ceil(64);
        let mut bits = vec![0u64; words].into_boxed_slice();
        // Word-stride outer loop so the compiler can hoist the row base.
        for i in 0..n {
            let row_base = i * n;
            for j in 0..n {
                if rule(prefix, i, j) {
                    let bit = row_base + j;
                    // SAFETY: bit < n*n <= words*64, so bit/64 is in bounds.
                    bits[bit / 64] |= 1u64 << (bit % 64);
                }
            }
        }
        Self { bits }
    }

    /// Number of 64-bit words in the packed buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Read the `attends(i, j)` bit. `row_len` must equal the `augmented_len`
    /// passed to [`Self::materialize_from`].
    #[inline]
    pub fn get(&self, i: usize, j: usize, row_len: usize) -> bool {
        let bit = i * row_len + j;
        (self.bits[bit / 64] >> (bit % 64)) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Small fixture: base_len=4, xc_positions=[1,3].
    ///   augmented_len = 6
    ///   r0 = [0, 2)            — copies (original positions 1 and 3)
    ///   r1 = [2, 6)            — original tokens (original positions 0,1,2,3)
    fn small_prefix<'a>(base: &'a [u32]) -> AcPrefix<'a> {
        // base.len() must be >= 4 for [1,3] to be in-range.
        assert!(base.len() >= 4);
        AcPrefix::new(base, &[1, 3])
    }

    #[test]
    fn augmented_len_empty_and_nonempty() {
        let base = [10u32, 20, 30, 40];
        let empty = AcPrefix::empty(&base);
        assert_eq!(empty.augmented_len(), 4);
        assert_eq!(empty.xc_len(), 0);

        let p = small_prefix(&base);
        assert_eq!(p.augmented_len(), 6);
        assert_eq!(p.xc_len(), 2);
    }

    #[test]
    fn original_positions_into_matches_layout() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let mut out = [0usize; 6];
        p.original_positions_into(&mut out);
        // First 2 slots are copies: their source positions are conditioning_positions = [1, 3].
        // Remaining 4 slots are original tokens: identity positions 0..4.
        assert_eq!(out, [1, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn original_positions_into_empty_prefix_is_identity() {
        let base = [10u32, 20, 30];
        let p = AcPrefix::empty(&base);
        let mut out = [0usize; 3];
        p.original_positions_into(&mut out);
        assert_eq!(out, [0, 1, 2]);
    }

    #[test]
    fn attends_three_region_rule_small_example() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        // augmented_len = 6; r0 = [0,2); r1 = [2,6).

        // (i ∈ r0, j ∈ r0) → true  (copies bidirectional)
        assert!(p.attends(0, 0));
        assert!(p.attends(0, 1));
        assert!(p.attends(1, 0));
        assert!(p.attends(1, 1));

        // (i ∈ r1, j ∈ r0) → true  (eval attends to all copies)
        assert!(p.attends(2, 0));
        assert!(p.attends(2, 1));
        assert!(p.attends(5, 0));
        assert!(p.attends(5, 1));

        // (i ∈ r0, j ∈ r1) → false (copies don't attend back to originals)
        assert!(!p.attends(0, 2));
        assert!(!p.attends(0, 5));
        assert!(!p.attends(1, 2));
        assert!(!p.attends(1, 5));

        // (i ∈ r1, j ∈ r1) → i >= j (standard causal in r1; original_pos offset cancels)
        assert!(p.attends(2, 2)); // 2 >= 2
        assert!(p.attends(3, 2)); // 3 >= 2
        assert!(p.attends(5, 2)); // 5 >= 2
        assert!(p.attends(5, 5)); // 5 >= 5
        assert!(!p.attends(2, 3)); // 2 < 3
        assert!(!p.attends(2, 5)); // 2 < 5
        assert!(!p.attends(4, 5)); // 4 < 5
    }

    #[test]
    fn attends_dedup_eliminates_inplace_xc_attention() {
        // base_len=4, xc_positions=[1,3]. augmented_len=6.
        //   r0 = [0,2) — copies at original positions {1,3}
        //   r1 = [2,6) — original sequence; r1 slots map to original positions:
        //     aug 2 → orig 0 (eval), aug 3 → orig 1 (in-place xc),
        //     aug 4 → orig 2 (eval), aug 5 → orig 3 (in-place xc)
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);

        // ── Same as `attends` for r0-source columns ──
        // (i ∈ r0, j ∈ r0) → true; (i ∈ r1, j ∈ r0) → true; (i ∈ r0, j ∈ r1) → false.
        assert!(p.attends_dedup(0, 0)); // both r0
        assert!(p.attends_dedup(0, 1)); // both r0
        assert!(p.attends_dedup(1, 0)); // both r0
        assert!(p.attends_dedup(2, 0)); // r1 → r0 copy
        assert!(p.attends_dedup(5, 1)); // r1 → r0 copy
        assert!(!p.attends_dedup(0, 2)); // r0 → r1
        assert!(!p.attends_dedup(1, 5)); // r0 → r1

        // ── DIFFERENCE from `attends`: eval must NOT attend to in-place xc in r1 ──
        // aug 3 (orig 1) is an in-place xc → eval tokens in r1 must not attend to it.
        assert!(!p.attends_dedup(4, 3)); // eval at aug 4 → in-place xc at aug 3
        assert!(!p.attends_dedup(5, 3)); // eval at aug 5 → in-place xc at aug 3
        // aug 5 (orig 3) is an in-place xc → eval tokens must not attend to it.

        // Self-attention: aug 5 IS in-place xc. attends_dedup returns false for
        // (i ∈ r1, j ∈ r1, j is in-place xc). But aug 5 attending to aug 5 is
        // an in-place xc attending to ITSELF — this is the (i==j, both in-place xc)
        // corner case. Per the deduplicated rule, this is also false because the
        // row query is "eval doesn't attend to in-place xc". An in-place xc
        // row is NOT an eval token; it's a conditioning token. Its row in the
        // attention matrix is only consumed by the loss mask (which masks it
        // out). So the self-attention of in-place xc doesn't affect the eval
        // logprobs. The rule consistently returns false here.
        assert!(!p.attends_dedup(5, 5)); // in-place xc self-attn → false (irrelevant for eval)
        // aug 3 (orig 1, in-place xc) self-attention.
        assert!(!p.attends_dedup(3, 3));

        // ── eval → eval in r1 still causal ──
        // aug 2 (orig 0, eval) and aug 4 (orig 2, eval):
        assert!(p.attends_dedup(2, 2)); // self
        assert!(p.attends_dedup(4, 2)); // eval at orig 2 → eval at orig 0
        assert!(!p.attends_dedup(2, 4)); // causal: 2 < 4
    }

    #[test]
    fn attends_dedup_empty_prefix_is_standard_causal() {
        // With no conditioning, dedup must degenerate to standard causal
        // (same as the original `attends` — there are no in-place xc to skip).
        let base = [10u32, 20, 30, 40];
        let p = AcPrefix::empty(&base);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(
                    p.attends_dedup(i, j),
                    i >= j,
                    "empty-prefix dedup must match standard causal at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn materialize_dedup_matches_attends_dedup_for_all_pairs() {
        let base: Vec<u32> = (0..12).collect();
        let xc = vec![0usize, 3, 7, 10];
        let p = AcPrefix::new(&base, &xc);
        let mask = AcPrefixMask::materialize_dedup_from(&p);
        let n = p.augmented_len();
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    mask.get(i, j, n),
                    p.attends_dedup(i, j),
                    "dedup mask bit ({i}, {j}) mismatch"
                );
            }
        }
    }

    #[test]
    fn attends_empty_prefix_is_standard_causal() {
        let base = [10u32, 20, 30];
        let p = AcPrefix::empty(&base);
        // augmented_len = 3; r0 is empty so everything is r1.
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(p.attends(i, j), i >= j, "i={i}, j={j}");
            }
        }
    }

    #[test]
    fn loss_mask_into_marks_only_eval_positions() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let mut out = [0.0f32; 6];
        p.loss_mask_into(&mut out);
        // r0 copies: always 0.0.
        // r1 positions: original_pos 0 (not in xc) → 1.0
        //               original_pos 1 (in xc)     → 0.0
        //               original_pos 2 (not in xc) → 1.0
        //               original_pos 3 (in xc)     → 0.0
        assert_eq!(out, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn loss_mask_into_empty_prefix_all_ones() {
        let base = [10u32, 20, 30];
        let p = AcPrefix::empty(&base);
        let mut out = [0.0f32; 3];
        p.loss_mask_into(&mut out);
        assert_eq!(out, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn materialize_from_matches_attends_for_all_pairs() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let n = p.augmented_len();
        let mask = AcPrefixMask::materialize_from(&p);

        // Word count = ceil(n*n / 64) = ceil(36/64) = 1.
        assert_eq!(mask.len(), 1);
        assert!(!mask.is_empty());

        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    mask.get(i, j, n),
                    p.attends(i, j),
                    "materialized bit disagrees with attends at (i={i}, j={j})"
                );
            }
        }
    }

    #[test]
    fn materialize_from_empty_prefix_matches_causal() {
        let base = [10u32, 20, 30, 40, 50];
        let p = AcPrefix::empty(&base);
        let n = p.augmented_len();
        let mask = AcPrefixMask::materialize_from(&p);
        // ceil(25/64) = 1 word.
        assert_eq!(mask.len(), 1);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(mask.get(i, j, n), i >= j, "i={i}, j={j}");
            }
        }
    }

    #[test]
    fn materialize_from_large_prefix_spans_multiple_words() {
        // base_len=12, xc=4 → augmented=16 → 256 bits → 4 words.
        let base: Vec<u32> = (0..12).collect();
        let xc: Vec<usize> = vec![1, 4, 7, 10];
        let p = AcPrefix::new(&base, &xc);
        let n = p.augmented_len();
        assert_eq!(n, 16);
        let mask = AcPrefixMask::materialize_from(&p);
        assert_eq!(mask.len(), (16u32 * 16).div_ceil(64) as usize);
        for i in 0..n {
            for j in 0..n {
                assert_eq!(
                    mask.get(i, j, n),
                    p.attends(i, j),
                    "large-case mismatch at (i={i}, j={j})"
                );
            }
        }
    }

    #[test]
    fn augmented_tokens_into_matches_layout() {
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        // r0 copies come from positions [1, 3] → base[1]=20, base[3]=40
        // r1 originals: 10, 20, 30, 40
        let mut out = [0u32; 6];
        p.augmented_tokens_into(&mut out);
        assert_eq!(out, [20, 40, 10, 20, 30, 40]);
    }

    #[test]
    fn conditional_logprob_sums_loss_mask_slots_only() {
        // Stub forward: returns logprob[i] = (token[i] as f32) / 100.0 for
        // every slot. The conditional_logprob sum should pick only the
        // loss_mask=1.0 slots.
        //   augmented_tokens = [20, 40, 10, 20, 30, 40]
        //   loss_mask         = [ 0,  0,  1,  0,  1,  0]
        //   picked: slots 2 and 4 → logprobs 0.10 + 0.30 = 0.40
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let total = p.conditional_logprob(|tokens, _pos, _mask, _lm| {
            tokens.iter().map(|t| *t as f32 / 100.0).collect()
        });
        assert!((total - 0.40_f32).abs() < 1e-6, "got {total}");
    }

    #[test]
    fn conditional_sample_walks_eval_slots_left_to_right() {
        // Stub forward: always returns logits with a sharp peak at index 5.
        // Gumbel-max noise has variance π²/6 ≈ 1.64, so a peak of 1000 vs
        // trough -1000 makes the peak essentially deterministic.
        // The augmented sequence has 2 eval slots → sampled = [5, 5].
        let base = [10u32, 20, 30, 40];
        let p = small_prefix(&base);
        let mut rng = fastrand::Rng::with_seed(0);
        let sampled = p.conditional_sample(
            |_tokens, _pos, _mask, _lm, _eval_slot| {
                (0..27)
                    .map(|i| if i == 5 { 1000.0 } else { -1000.0 })
                    .collect()
            },
            &mut rng,
        );
        assert_eq!(sampled.len(), 2);
        assert_eq!(sampled, vec![5, 5], "all eval slots should pick peak=5");
    }
}
