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

/// Keyed Gumbel noise: `g(seed, position, token_id)` — a pure function of
/// its key. The same `(seed, position, token)` triple always yields the
/// same noise regardless of evaluation order or batch composition, which
/// is the order-invariance a speculative verify loop needs: a position's
/// sampled token is decided by its key alone (the position-keyed-noise
/// rule, promoted from the corpus rule to a runtime primitive for the
/// Plan-614 DFlash2 lane — drafter proposals and target verify samples
/// share one keyed stream, so a draft token coincides with the target's
/// own sample exactly when the two score orderings agree after the shared
/// noise).
///
/// Construction: the key parts are mixed (wrapping — overflow wraps by
/// contract, never panics) and pushed through the SplitMix64 finalizer
/// (Steele et al., full avalanche), then 53 uniform bits map to
/// `u = bits·2⁻⁵³ + 2⁻⁵⁴`, which lies in the OPEN interval (0, 1) by
/// construction — no redraw loop, which a pure function of the key could
/// not perform anyway. `g = -ln(-ln(u))` is finite for every key.
///
/// Platform note: `ln` is the platform libm's IEEE-754 log. Same-platform
/// determinism (the verify loop's requirement) is exact; cross-platform
/// bit-identity of the SAMPLED TOKEN is not claimed — a flip requires a
/// near-tie inside one ulp of noise.
#[inline]
pub fn keyed_gumbel_noise(seed: u64, position: u64, token_id: u32) -> f32 {
    let u = keyed_unit_interval(seed, position, token_id);
    (-(-u.ln()).ln()) as f32
}

/// The open-interval uniform underneath [`keyed_gumbel_noise`], exposed as
/// its own step so transforms that need `u` itself (the posterior sampler's
/// truncation) consume the SAME keyed stream instead of a parallel one.
/// Bit-identical to the arithmetic that lived inline in
/// `keyed_gumbel_noise` since Plan 614 (extraction only — the G1 pins on
/// the verify stream's sampled tokens are the regression tripwire).
#[inline]
fn keyed_unit_interval(seed: u64, position: u64, token_id: u32) -> f64 {
    #[inline(always)]
    fn fin(z: u64) -> u64 {
        // SplitMix64 finalizer (Steele et al.) — full avalanche.
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    // Nested finalize — insurance against structured collisions between
    // the key parts: the seed⊕position mix is finished BEFORE the token
    // joins, then the whole key is finished again.
    let z = fin(fin(seed ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        ^ (token_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let inv = 1.0f64 / (1u64 << 53) as f64;
    ((z >> 11) as f64) * inv + inv * 0.5
}

/// Keyed Gumbel-max sample: `argmax_i (logits[i] + g(seed, position, i))`
/// — the categorical sample whose randomness is a pure function of
/// `(seed, position, token_id)` with the vocab index standing in for the
/// token id (full-vocab sampling, the target side of a verify loop; index
/// == token id there). Deterministic given the key: re-running a position
/// reproduces its token bit-identically, and no other position's or
/// batch-row's evaluation can perturb it. Returns `0` on an empty input.
///
/// This samples from `softmax(logits)` at temperature 1; a caller decoding
/// at temperature `T` scales its logits by `1/T` first (the standard
/// compositional form — noise scale and temperature are redundant knobs).
#[inline]
pub fn keyed_gumbel_max_sample(logits: &[f32], seed: u64, position: u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        let score = l + keyed_gumbel_noise(seed, position, i as u32);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// The keep-mask for a truncated sampler: `true` = the token is eligible.
/// Composition order is the deployment incumbent's (llama.cpp's sampling
/// chain): **top-k first, then top-p over the SURVIVORS** — the top-p
/// denominator is the survivors' softmax mass at T=1, and the temperature
/// applies after truncation (the masked sampler scales survivors only, so
/// mask → scale is order-equivalent). Normalising against the full row
/// instead would make the nucleus strictly WIDER than the incumbent's —
/// the tail mass dilutes the survivors' shares. `top_k` keeps the `k`
/// highest finite logits (ties keep the EARLIER index first — the
/// `top_k_desc` convention); `top_p` keeps the smallest descending-logit
/// prefix of the survivors whose cumulative survivor mass reaches `p`
/// (inclusive of the crossing token; the first survivor is always kept, so
/// the mask is never empty when any finite logit exists). `None` on either
/// axis leaves it unfiltered. Documented divergences from the incumbent
/// (both shared-function, so losslessness is unaffected): a `+inf` logit
/// is DROPPED here while llama.cpp would rank it, and top-k ties break
/// earliest-index-first here while llama.cpp's sort order is unspecified.
///
/// NaN and ±inf logits are never eligible: they cannot rank, contribute no
/// mass, and are never marked (an infinite logit is never a legitimate
/// model output, and a NaN that ranked would poison the whole softmax —
/// `total_cmp` sorts +NaN above +inf, so an unguarded NaN would silently
/// degrade the mask to top-k-only). A row with no finite logits yields an
/// all-false mask; the masked sampler's first-index convention then picks
/// token 0 deterministically. The full-vocab sort is O(v log v) — the
/// Phase-2 correctness posture accepts it (the keyed sample itself is
/// already a full scan); the Phase-3 GPU sampler replaces both.
pub fn truncation_keep_mask(logits: &[f32], top_k: Option<usize>, top_p: Option<f32>) -> Vec<bool> {
    let v = logits.len();
    let mut mask = vec![false; v];
    if v == 0 {
        return mask;
    }
    let mut order: Vec<u32> = (0..v as u32)
        .filter(|&i| logits[i as usize].is_finite())
        .collect();
    if order.is_empty() {
        return mask;
    }
    order.sort_unstable_by(|&a, &b| {
        logits[b as usize]
            .total_cmp(&logits[a as usize])
            .then(a.cmp(&b))
    });
    let max = logits[order[0] as usize];
    let k_keep = top_k.map_or(order.len(), |k| k.clamp(1, order.len()));
    // The top-p denominator is the SURVIVORS' mass (the incumbent's
    // chain: top-k first, then top-p within).
    let mut sum = 0.0f32;
    for &i in order.iter().take(k_keep) {
        sum += (logits[i as usize] - max).exp();
    }
    let p = top_p.unwrap_or(1.0).clamp(0.0, 1.0);
    let mut kept = 0usize;
    let mut cum = 0.0f32;
    for &i in order.iter().take(k_keep) {
        mask[i as usize] = true;
        kept += 1;
        if top_p.is_some() {
            cum += (logits[i as usize] - max).exp() / sum;
        }
        // Stop at k survivors, or once the survivor mass is reached — the
        // LAST survivor is always kept, which also absorbs the f32 case
        // where the accumulated mass stalls a hair under p.
        if kept >= k_keep || (top_p.is_some() && cum >= p) {
            break;
        }
    }
    mask
}

/// [`keyed_gumbel_max_sample`] over a truncated distribution: `keep[i]
/// == false` tokens are masked BEFORE the temperature scale and the keyed
/// argmax (the Plan-614 Phase-2 contract order — mask, scale, sample), so
/// their keys are never consulted and the pick is always inside the
/// deployment nucleus. `temperature <= 0` is the greedy posture: the plain
/// argmax over the survivors, no noise at all. A `keep` whose length
/// differs from `logits` is a caller bug → returns `0` (the defensive
/// convention of the unmasked sampler's empty input), never a silent
/// full-vocab fallback that would sample outside the caller's intended
/// support.
#[inline]
pub fn keyed_gumbel_max_sample_masked(
    logits: &[f32],
    seed: u64,
    position: u64,
    temperature: f32,
    keep: &[bool],
) -> u32 {
    if logits.is_empty() || keep.len() != logits.len() {
        return 0;
    }
    if temperature <= 0.0 {
        let mut best_idx = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, (&l, &k)) in logits.iter().zip(keep.iter()).enumerate() {
            if k && l > best {
                best = l;
                best_idx = i;
            }
        }
        return best_idx as u32;
    }
    let inv_t = 1.0 / temperature;
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (i, (&l, &k)) in logits.iter().zip(keep.iter()).enumerate() {
        if !k {
            continue;
        }
        let score = l * inv_t + keyed_gumbel_noise(seed, position, i as u32);
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// The one-call composition the verify loop uses: [`truncation_keep_mask`]
/// then [`keyed_gumbel_max_sample_masked`] at the deployment temperature.
/// Sampling from `softmax(logits/T)` restricted to the (top-k, top-p)
/// nucleus; `temperature <= 0` is the greedy posture over the survivors.
pub fn keyed_gumbel_max_sample_truncated(
    logits: &[f32],
    seed: u64,
    position: u64,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> u32 {
    let keep = truncation_keep_mask(logits, top_k, top_p);
    keyed_gumbel_max_sample_masked(logits, seed, position, temperature, &keep)
}

/// The keyed uniform slot reserved for the posterior sampler's max draw.
/// Vocab indices occupy `0..len`; `u32::MAX` is unreachable for any real
/// vocabulary and keeps the max's randomness disjoint from every loser's.
const POSTERIOR_MAX_SLOT: u32 = u32::MAX;

/// Keyed posterior (truncated-Gumbel) inverse sampler — the MAX-FIRST
/// construction (Issue 918; the posterior construction is Zhang et al. 2026,
/// via arXiv:2610.00497 App E; the Gumbel trick is Gumbel 1954).
///
/// Given logits `ℓ` and a REALIZED pick `y` (e.g. a teacher-forced token
/// whose sampling seed is unknown), writes into `out` a noise vector `ξ*`
/// sampled from the exact posterior `p(ξ | argmax(ℓ + ξ) = y)`:
///
/// - `M ~ Gumbel(logsumexp(ℓ))` — under the Gumbel-max representation the
///   maximum score is Gumbel with that location and is INDEPENDENT of the
///   winner index, so `M`'s key (`(seed, position)`, slot
///   [`POSTERIOR_MAX_SLOT`]) does not involve `y`: two draws at the same key
///   with different winners share their max, which is exactly the coupling
///   the residual-ordering property holds on.
/// - `ξ*_y = M − ℓ_y` — the winner's noise is PINNED so `ℓ_y + ξ*_y = M`.
/// - each loser `k ≠ y`: `ξ*_k ~ Gumbel(0)` truncated ABOVE at `M − ℓ_k`
///   via the inverse CDF `ξ*_k = −log(−log(u_k · F(M − ℓ_k)))`,
///   `F(t) = exp(−exp(−t))` — coordinates conditionally independent given
///   the max. `u_k` is the SAME keyed uniform [`keyed_gumbel_noise`] maps
///   through the prior transform at that key (common random numbers:
///   prior and posterior at one key are one stream, two transforms).
///
/// This is NOT "draw losers freely, then pin the winner above their max" —
/// that reproduces the pick but leaves the loser marginals UNTRUNCATED, a
/// biased coupling that would corrupt a straightness consumer (the G1b
/// gate's red arm pins the difference).
///
/// Deterministic float tie-guard: scores are compared in f32
/// (`argmax(ℓ + ξ*)` with the substrate's strict-`>` first-index
/// convention); a loser that rounds up onto the winner's score is nudged
/// one ulp down until strictly below. The nudge is sub-precision — it fires
/// only on exact float ties and is invisible to the distribution gates.
///
/// Honest consumer note (the issue's retraction clause): this exists for
/// consumers that must RECONSTRUCT consistent noise from an outcome
/// (the GSF training coupling, riir-train Plan 437). It is NOT a replay
/// tool for our own keyed sampler — `keyed_gumbel_noise` is a pure
/// function of its key, so a recorded seed already replays exactly, and a
/// posterior sample is *a* consistent noise, never *the* noise used.
///
/// Defensive conventions (the substrate's): empty `logits`, `winner >=
/// len`, or `out.len() != logits.len()` write nothing and return — a
/// caller bug never becomes a silent full-vocab fallback. Assumes finite
/// logits (the verify-loop contract); the LSE is max-shifted, so
/// moderate magnitudes are exact to within f32 rounding.
/// Zero-allocation: writes only into `out`.
pub fn keyed_posterior_gumbel_noise(
    logits: &[f32],
    winner: u32,
    seed: u64,
    position: u64,
    out: &mut [f32],
) {
    let n = logits.len();
    if n == 0 || out.len() != n || winner as usize >= n {
        return;
    }
    // LSE(ℓ), max-shifted (f32 — the precision the consumer's argmax runs
    // at; deterministic on-platform per the keyed_noise platform note).
    let mut m = f32::NEG_INFINITY;
    for &l in logits {
        if l > m {
            m = l;
        }
    }
    let mut sum = 0.0f32;
    for &l in logits {
        sum += (l - m).exp();
    }
    let lse = m + sum.ln();
    // M ~ Gumbel(lse): keyed uniform at the reserved slot, winner-independent.
    // Gumbel(μ) = μ − ln(−ln(u)) = μ + g with g ~ Gumbel(0), so the max
    // ADDS the standard noise (the sign the CDF derivation pins:
    // P(max ≤ x) = exp(−exp(−(x − lse)))).
    let u_max = keyed_unit_interval(seed, position, POSTERIOR_MAX_SLOT);
    let gumbel_zero_max = (-(-u_max.ln()).ln()) as f32;
    let big_m = lse + gumbel_zero_max;
    let w = winner as usize;
    out[w] = big_m - logits[w];
    let score_w = logits[w] + out[w];
    for (k, &l) in logits.iter().enumerate() {
        if k == w {
            continue;
        }
        // Gumbel(0) truncated above at t = M − ℓ_k, inverse-CDF form.
        let t = (big_m - l) as f64;
        let u = keyed_unit_interval(seed, position, k as u32);
        let w_trunc = u * (-(-t).exp()).exp();
        let mut xi = (-(-w_trunc.ln()).ln()) as f32;
        // Deterministic tie-guard: the loser's f32 score must sit strictly
        // below the winner's. One ulp (`next_down` — the bit-pattern
        // decrement is direction-correct for negative values too) per pass;
        // bounded in the sane-logit regime (a tie needs |Δ| below one ulp
        // of the score magnitude). NaN scores compare false and exit at
        // once (the defensive posture for non-finite caller input).
        let mut score = l + xi;
        while score >= score_w {
            xi = xi.next_down();
            score = l + xi;
        }
        out[k] = xi;
    }
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

    #[test]
    fn keyed_gumbel_noise_is_a_pure_function_of_its_key() {
        // Same key → same noise, regardless of call order or interleaving.
        let a = keyed_gumbel_noise(42, 7, 1234);
        let _ = keyed_gumbel_noise(42, 7, 1235);
        let _ = keyed_gumbel_noise(43, 7, 1234);
        let _ = keyed_gumbel_noise(42, 8, 1234);
        assert_eq!(keyed_gumbel_noise(42, 7, 1234), a);
        // Distinct keys produce distinct noise (avalanche — spot-check a
        // dense sweep instead of asserting on any single pair).
        let mut distinct = 0u32;
        for k in 0..1000u64 {
            if keyed_gumbel_noise(k, k * 3, (k % 997) as u32)
                != keyed_gumbel_noise(k + 1, k * 3, (k % 997) as u32)
            {
                distinct += 1;
            }
        }
        assert!(
            distinct > 950,
            "adjacent-seed keys should almost never collide: {distinct}/1000"
        );
        // Finite for every key — the open-interval u construction means no
        // redraw and no NaN/inf, swept across the key space.
        for s in [0u64, 1, u64::MAX] {
            for p in [0u64, 1, u64::MAX / 2] {
                for t in [0u32, 1, u32::MAX] {
                    let g = keyed_gumbel_noise(s, p, t);
                    assert!(g.is_finite(), "noise not finite at ({s},{p},{t})");
                    assert!(g.abs() < 64.0, "noise out of Gumbel range: {g}");
                }
            }
        }
    }

    #[test]
    fn keyed_gumbel_max_sample_is_deterministic_and_greedy_in_the_limit() {
        let logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.37 - 8.0).collect();
        // Determinism: the same key reproduces the sample bit-identically.
        let first = keyed_gumbel_max_sample(&logits, 9, 55);
        assert_eq!(keyed_gumbel_max_sample(&logits, 9, 55), first);
        // The noise is bounded (|g| ≲ 40), so a logit gap far above that
        // bound forces the argmax — the greedy limit of the keyed sample.
        let mut peaked = vec![-1000.0f32; 64];
        peaked[17] = 1000.0;
        for p in 0..200u64 {
            assert_eq!(keyed_gumbel_max_sample(&peaked, 3, p), 17);
        }
        // Empty input → 0 (matches `gumbel_max_sample`).
        assert_eq!(keyed_gumbel_max_sample(&[], 1, 1), 0);
    }

    #[test]
    fn truncation_keep_mask_topk_topp_and_combination() {
        let logits = [1.0f32, 5.0, 5.0, 0.5, 3.0];
        // top-k=3: the three highest — ties keep the earlier index first, so
        // both 5.0s and the 3.0 survive, never the 1.0.
        let m = truncation_keep_mask(&logits, Some(3), None);
        assert_eq!(m, vec![false, true, true, false, true]);
        // top-k=1: exactly the argmax (earlier index on ties).
        let m1 = truncation_keep_mask(&logits, Some(1), None);
        assert_eq!(m1, vec![false, true, false, false, false]);
        // top-p alone: the two 5.0s carry e⁰+e⁰ / (sum) ≈ 0.924 of the
        // softmax mass — a p=0.9 nucleus stops there; p=0.95 pulls in the
        // 3.0 (cum 0.987) but never the 0.5/1.0 tail. The argmax always
        // keeps at least one token even at extreme p.
        let mp = truncation_keep_mask(&logits, None, Some(0.9));
        assert_eq!(mp, vec![false, true, true, false, false]);
        let mp95 = truncation_keep_mask(&logits, None, Some(0.95));
        assert_eq!(mp95, vec![false, true, true, false, true]);
        let mtiny = truncation_keep_mask(&logits, None, Some(1e-6));
        assert_eq!(mtiny, vec![false, true, false, false, false]);
        // Combined: top-k=2 first, then top-p over the survivors — the
        // first survivor already carries 0.5 of the SURVIVOR mass, so the
        // p=0.5 nucleus stops at one token (the crossing token inclusive).
        let mc = truncation_keep_mask(&logits, Some(2), Some(0.5));
        assert_eq!(mc, vec![false, true, false, false, false]);
        // No filters: everything eligible.
        let mall = truncation_keep_mask(&logits, None, None);
        assert!(mall.iter().all(|&b| b));
        // Empty row: empty mask (and the masked sampler's 0 convention).
        assert!(truncation_keep_mask(&[], Some(3), Some(0.9)).is_empty());
    }

    #[test]
    fn truncation_keep_mask_survivor_normalized_top_p_matches_the_incumbent_chain() {
        // The llama.cpp chain: top-k → top-p over the SURVIVORS at T=1.
        // This row discriminates the two denominators: survivors {5.0,
        // 4.0, 4.0} renormalise to 0.576/0.212/0.212, so p=0.7 keeps TWO
        // (cum 0.788 at the second); the full-row denominator (with the
        // 5×3.0 tail diluting) gives 0.414/0.152/0.152, cum 0.719 at the
        // third — a full-row mask would keep THREE. The survivor order is
        // the deployment incumbent's; pinning it.
        let logits = [5.0f32, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0];
        let m = truncation_keep_mask(&logits, Some(3), Some(0.7));
        assert_eq!(
            m,
            vec![true, true, false, false, false, false, false, false],
            "top-p must normalise over the top-k survivors, not the full row"
        );
        // p-alone (k = None): the survivor set is the whole finite row, so
        // the denominator is the full row's — unchanged behaviour there.
        let base = [1.0f32, 5.0, 5.0, 0.5, 3.0];
        let mp = truncation_keep_mask(&base, None, Some(0.9));
        assert_eq!(mp, vec![false, true, true, false, false]);
    }

    #[test]
    fn truncation_keep_mask_never_marks_non_finite_logits() {
        // A NaN that ranked would poison the softmax (total_cmp sorts +NaN
        // above +inf, so the unguarded max would be NaN and top-p would
        // never fire); a ±inf logit is never a legitimate output. Both are
        // ineligible: no mass, no mark.
        let logits = [f32::NAN, 5.0, 1.0, f32::NEG_INFINITY];
        let m = truncation_keep_mask(&logits, None, Some(0.5));
        assert_eq!(m, vec![false, true, false, false]);
        // All-non-finite: nothing eligible → all false (the masked
        // sampler's first-index convention then picks token 0).
        let bad = [f32::NAN, f32::NEG_INFINITY, f32::INFINITY];
        assert!(
            truncation_keep_mask(&bad, Some(2), Some(0.9))
                .iter()
                .all(|&b| !b)
        );
        // The masked sampler on the NaN row with the survivor mask: picks
        // the survivor (the NaN, even masked, could never win — NaN > x is
        // false — but the mask keeps the pick inside the nucleus).
        assert_eq!(keyed_gumbel_max_sample_masked(&logits, 9, 3, 0.6, &m), 1);
    }

    #[test]
    fn keyed_masked_sampler_matches_the_unmasked_on_a_full_mask() {
        let logits: Vec<f32> = (0..32).map(|i| (i as f32) * 0.41 - 6.0).collect();
        let all = vec![true; 32];
        for p in 0..50u64 {
            assert_eq!(
                keyed_gumbel_max_sample_masked(&logits, 77, p, 0.7, &all),
                // The unmasked sampler is the masked one at temperature 1;
                // pre-scaled logits reproduce it exactly (the composition
                // the Phase-2 target side and the drafter's walk share).
                keyed_gumbel_max_sample(
                    &logits.iter().map(|&l| l / 0.7).collect::<Vec<_>>(),
                    77,
                    p
                )
            );
        }
    }

    #[test]
    fn keyed_masked_sampler_respects_the_mask_and_greedy_posture() {
        let logits = [-1000.0f32, 1000.0, 999.0, -1000.0];
        // Masked argmax survives: token 1 masked out → token 2 (the argmax
        // of the survivors) is the keyed pick at every position, because the
        // logit gap to the rest dwarfs the noise bound.
        let keep = [false, false, true, true];
        for p in 0..100u64 {
            assert_eq!(keyed_gumbel_max_sample_masked(&logits, 5, p, 0.6, &keep), 2);
        }
        // temperature <= 0: the plain argmax over survivors, NO noise —
        // bit-stable for any seed/position and equal to the T→0 limit arm.
        assert_eq!(
            keyed_gumbel_max_sample_masked(&logits, 5, 42, 0.0, &keep),
            2
        );
        assert_eq!(
            keyed_gumbel_max_sample_masked(&logits, 9, 7, -1.0, &keep),
            2
        );
        // A keep row shorter than the logits is a caller bug → 0, never a
        // silent full-vocab fallback outside the intended support.
        assert_eq!(
            keyed_gumbel_max_sample_masked(&logits, 5, 42, 0.6, &[true]),
            0
        );
        // The one-call composition equals mask-then-sample.
        let truncated = keyed_gumbel_max_sample_truncated(&logits, 5, 41, 0.6, Some(1), None);
        assert_eq!(truncated, 1);
    }

    #[test]
    fn keyed_gumbel_max_sample_reproduces_the_categorical_distribution() {
        // The construction claim: argmax(logit + Gumbel(0,1)) samples from
        // softmax(logits). Three tokens with known probs (0.5, 0.3, 0.2);
        // 100k distinct positions = 100k independent keyed draws. At n=100k
        // the per-token binomial σ is 0.13–0.16%, so the ±1% band sits at
        // ≥6σ — tight enough to catch a ~10% temperature error (which
        // moves token 0 by ~1.4%) yet far from flake territory.
        let ln = |p: f64| (p as f32).ln();
        let logits = [ln(0.5), ln(0.3), ln(0.2)];
        let want = [0.5f32, 0.3, 0.2];
        let n = 100_000u64;
        let mut hits = [0u64; 3];
        for p in 0..n {
            hits[keyed_gumbel_max_sample(&logits, 0xDEADBEEF, p) as usize] += 1;
        }
        for k in 0..3 {
            let freq = hits[k] as f32 / n as f32;
            assert!(
                (freq - want[k]).abs() < 0.01,
                "token {k}: freq {freq} vs softmax {} (hits {})",
                want[k],
                hits[k]
            );
        }
    }

    #[test]
    fn keyed_unit_interval_extraction_is_bit_identical_to_the_plan614_arithmetic() {
        // Issue 918's refactor extracted the uniform construction from
        // `keyed_gumbel_noise` into `keyed_unit_interval`. The extraction
        // must be BIT-identical: the verify loop (Plan 614) consumes the
        // keyed stream, and downstream pins ride those exact bits. This
        // replicates the pre-extraction inline body verbatim and compares
        // over a key grid.
        fn old_inline_gumbel(seed: u64, position: u64, token_id: u32) -> f32 {
            #[inline(always)]
            fn fin(z: u64) -> u64 {
                let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            }
            let z = fin(fin(seed ^ position.wrapping_mul(0x9E37_79B9_7F4A_7C15))
                ^ (token_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
            let inv = 1.0f64 / (1u64 << 53) as f64;
            let u = ((z >> 11) as f64) * inv + inv * 0.5;
            (-(-u.ln()).ln()) as f32
        }
        for seed in [0u64, 1, 42, 0xDEAD_BEEF, u64::MAX] {
            for position in [0u64, 7, 1_000_003, u64::MAX / 3] {
                for token in [0u32, 1, 31999, u32::MAX] {
                    assert_eq!(
                        keyed_gumbel_noise(seed, position, token),
                        old_inline_gumbel(seed, position, token),
                        "({seed}, {position}, {token})"
                    );
                }
            }
        }
    }

    /// Two-sample KS statistic (sup |ECDF_a − ECDF_b|), total_cmp-sorted.
    fn ks_two_sample(a: &mut [f32], b: &mut [f32]) -> f64 {
        a.sort_by(|x, y| x.total_cmp(y));
        b.sort_by(|x, y| x.total_cmp(y));
        let mut i = 0usize;
        let mut j = 0usize;
        let mut d = 0.0f64;
        while i < a.len() && j < b.len() {
            if a[i] <= b[j] {
                i += 1;
            } else {
                j += 1;
            }
            let fa = i as f64 / a.len() as f64;
            let fb = j as f64 / b.len() as f64;
            d = d.max((fa - fb).abs());
        }
        d
    }

    fn mean_std(xs: &[f32]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = xs
            .iter()
            .map(|&x| {
                let d = x as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        (mean, var.sqrt())
    }

    #[test]
    fn posterior_reproduces_the_pick_exactly() {
        // G1: argmax(ℓ + ξ*) == winner on 100% of fixture picks, and the
        // draw is a pure function of its key. Red arm: noise drawn for
        // logits A must break the identity against a different row B —
        // the identity belongs to the CONSTRUCTION, not to any noise.
        let ln = |p: f64| (p as f32).ln();
        let logits = [ln(0.40), ln(0.30), ln(0.20), ln(0.10)];
        let shifted = [ln(0.10), ln(0.20), ln(0.35), ln(0.35)];
        let mut out = [0f32; 4];
        let mut other = [0f32; 4];
        let mut wrong = 0u32;
        for pos in 0..2_000u64 {
            for &winner in &[0u32, 1, 2, 3] {
                keyed_posterior_gumbel_noise(&logits, winner, 0x918, pos, &mut out);
                keyed_posterior_gumbel_noise(&logits, winner, 0x918, pos, &mut other);
                assert_eq!(out, other, "keyed draw must be bit-identical");
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for (k, (&l, &x)) in logits.iter().zip(out.iter()).enumerate() {
                    let s = l + x;
                    if s > best_s {
                        best_s = s;
                        best = k;
                    }
                }
                assert_eq!(best as u32, winner, "pos {pos} winner {winner}");
                let mut b2 = 0usize;
                let mut s2 = f32::NEG_INFINITY;
                for (k, (&l, &x)) in shifted.iter().zip(out.iter()).enumerate() {
                    let s = l + x;
                    if s > s2 {
                        s2 = s;
                        b2 = k;
                    }
                }
                if b2 as u32 != winner {
                    wrong += 1;
                }
            }
        }
        assert!(
            wrong > 0,
            "red arm never fired — the identity held for foreign logits"
        );
    }

    #[test]
    fn posterior_marginals_match_the_filtered_prior() {
        // G1b (joint-distribution gate): the sampler's winner AND loser
        // coordinate marginals must match the rejection-filtered prior
        // (draw ξ ~ prior, keep argmax == y). KS + moment checks on both
        // groups; the red arm is the BIASED construction (free losers,
        // winner pinned above their max) which must FAIL the loser checks —
        // the gate has to be able to catch the coupling the issue forbids.
        let ln = |p: f64| (p as f32).ln();
        const V: usize = 5;
        const W: usize = 1; // mid-probability winner: p ≈ 0.28
        const N: usize = 6_000;
        let logits = [ln(0.32), ln(0.28), ln(0.20), ln(0.12), ln(0.08)];

        // Reference: rejection-filter the prior.
        let mut ref_winner = Vec::with_capacity(N);
        let mut ref_losers = Vec::with_capacity(N * (V - 1));
        let mut ref_top_loser = Vec::with_capacity(N);
        let mut pos = 0u64;
        while ref_winner.len() < N {
            pos += 1;
            let mut best = 0usize;
            let mut best_s = f32::NEG_INFINITY;
            let mut draw = [0f32; V];
            for (k, &l) in logits.iter().enumerate() {
                let g = keyed_gumbel_noise(0x61B, pos, k as u32);
                draw[k] = g;
                let s = l + g;
                if s > best_s {
                    best_s = s;
                    best = k;
                }
            }
            if best == W {
                ref_winner.push(draw[W]);
                for (k, &g) in draw.iter().enumerate() {
                    if k != W {
                        ref_losers.push(g);
                        if k == 0 {
                            ref_top_loser.push(g);
                        }
                    }
                }
            }
        }

        // Sampler draws: same counts, distinct keys.
        let mut smp_winner = Vec::with_capacity(N);
        let mut smp_losers = Vec::with_capacity(N * (V - 1));
        let mut smp_top_loser = Vec::with_capacity(N);
        let mut out = [0f32; V];
        for i in 0..N {
            keyed_posterior_gumbel_noise(&logits, W as u32, 0x61B, 1_000_000 + i as u64, &mut out);
            smp_winner.push(out[W]);
            for (k, &x) in out.iter().enumerate() {
                if k != W {
                    smp_losers.push(x);
                    if k == 0 {
                        smp_top_loser.push(x);
                    }
                }
            }
        }

        // Red arm: the BIASED construction — free losers, winner pinned
        // above their max afterwards. Loser marginals lack the truncation
        // tilt, which the loser checks below must detect.
        let mut bias_losers = Vec::with_capacity(N * (V - 1));
        let mut bias_top_loser = Vec::with_capacity(N);
        for i in 0..N {
            let pos = 2_000_000 + i as u64;
            for k in 0..V {
                if k == W {
                    continue;
                }
                let g = keyed_gumbel_noise(0x61B, pos, k as u32);
                bias_losers.push(g);
                if k == 0 {
                    bias_top_loser.push(g);
                }
            }
        }

        // KS bounds: n=6k vs 6k per group → α=0.001 critical ≈ 0.036;
        // the pass bound 0.05 keeps ≥1.4× margin over the exact sampler
        // while sitting far below the biased arm's truncation gap.
        const KS_PASS: f64 = 0.05;
        let ks_w = ks_two_sample(&mut smp_winner, &mut ref_winner);
        assert!(ks_w < KS_PASS, "winner KS {ks_w}");
        let ks_l = ks_two_sample(&mut smp_losers, &mut ref_losers);
        assert!(ks_l < KS_PASS, "loser KS {ks_l}");
        let ks_tl = ks_two_sample(&mut smp_top_loser, &mut ref_top_loser);
        assert!(ks_tl < KS_PASS, "top-loser KS {ks_tl}");
        eprintln!("[G1b] KS winner {ks_w:.4} · pooled losers {ks_l:.4} · top-loser {ks_tl:.4}");

        // Moment checks (means within 4 joint SEs; the filtered prior and
        // the posterior share one law, so any real drift is a bug).
        let (mw, sw) = mean_std(&smp_winner);
        let (rw, tw) = mean_std(&ref_winner);
        let se = (sw * sw / N as f64 + tw * tw / N as f64).sqrt();
        assert!((mw - rw).abs() < 4.0 * se, "winner mean {mw} vs {rw}");
        let (ml, sl) = mean_std(&smp_losers);
        let (rl, tl) = mean_std(&ref_losers);
        let nl = smp_losers.len() as f64;
        let se = (sl * sl / nl + tl * tl / nl).sqrt();
        assert!((ml - rl).abs() < 4.0 * se, "loser mean {ml} vs {rl}");

        // The red arm must FAIL the loser checks (winner coordinate is
        // correct in both constructions — the discrimination lives in the
        // losers' truncation).
        let ks_bl = ks_two_sample(&mut bias_losers, &mut ref_losers);
        let ks_btl = ks_two_sample(&mut bias_top_loser, &mut ref_top_loser);
        eprintln!("[G1b-red] biased pooled {ks_bl:.4} · biased top-loser {ks_btl:.4}");
        assert!(
            ks_bl >= KS_PASS || ks_btl >= KS_PASS,
            "biased construction escaped: pooled {ks_bl}, top-loser {ks_btl}"
        );
    }

    #[test]
    fn posterior_residual_ordering_property() {
        // Theorem-1 mechanism (arXiv:2610.00497 App A.1): two posterior
        // draws from the SAME logits and key (hence a SHARED max M) with
        // distinct winners k1≠k2 satisfy ξ¹_k1 − ξ²_k1 > ξ¹_k2 − ξ²_k2
        // DETERMINISTICALLY — the coupled pair's residuals order by winner.
        // A literal path-intersection test cannot discriminate (measure-
        // zero in ℝ^V); the inequality is the falsifiable form.
        //
        // Sharper than the issue's spec, measured on the first run: ANY two
        // valid posterior draws satisfy the inequality — even at INDEPENDENT
        // keys — because pinned-winner-vs-truncated-loser alone implies it
        // (d1 > M¹−M² > d2). The discrimination arm therefore contrasts the
        // posterior construction against UNCONDITIONAL noise (iid Gumbel
        // coordinates), where the two differences are exchangeable and the
        // violation fraction is ~0.5 — the red arm that actually fires.
        let ln = |p: f64| (p as f32).ln();
        let logits = [ln(0.3), ln(0.3), ln(0.2), ln(0.2)];
        const K1: usize = 0;
        const K2: usize = 3;
        const PAIRS: u64 = 10_000;
        let mut a = [0f32; 4];
        let mut b = [0f32; 4];
        let mut coupled_violations = 0u32;
        let mut indep_posterior_violations = 0u32;
        let mut iid_noise_violations = 0u32;
        for pos in 0..PAIRS {
            // Coupled: shared key → shared max M.
            keyed_posterior_gumbel_noise(&logits, K1 as u32, 0x7EED, pos, &mut a);
            keyed_posterior_gumbel_noise(&logits, K2 as u32, 0x7EED, pos, &mut b);
            let (d1, d2) = (a[K1] - b[K1], a[K2] - b[K2]);
            if !matches!(d1.partial_cmp(&d2), Some(std::cmp::Ordering::Greater)) {
                coupled_violations += 1;
            }
            // Independent keys: still two valid posteriors — the ordering
            // is structural (recorded; NOT the discrimination arm).
            keyed_posterior_gumbel_noise(&logits, K2 as u32, 0x7EED, 5_000_000 + pos, &mut b);
            let (d1, d2) = (a[K1] - b[K1], a[K2] - b[K2]);
            if !matches!(d1.partial_cmp(&d2), Some(std::cmp::Ordering::Greater)) {
                indep_posterior_violations += 1;
            }
            // Unconditional control: BOTH sides iid Gumbel coordinates
            // (the prior noise, no posterior structure anywhere) — the four
            // coordinates are exchangeable, so the two differences are
            // iid and the violation fraction is ~0.5 (the measured
            // one-posterior-side variant reads ~0.23: the winner-pinning
            // asymmetry biases holds — not a clean control).
            for (k, slot) in a.iter_mut().enumerate() {
                *slot = keyed_gumbel_noise(0x7EED, 8_000_000 + pos, k as u32);
            }
            for (k, slot) in b.iter_mut().enumerate() {
                *slot = keyed_gumbel_noise(0x7EED, 9_000_000 + pos, k as u32);
            }
            let (d1, d2) = (a[K1] - b[K1], a[K2] - b[K2]);
            if !matches!(d1.partial_cmp(&d2), Some(std::cmp::Ordering::Greater)) {
                iid_noise_violations += 1;
            }
        }
        assert_eq!(
            coupled_violations, 0,
            "coupled pairs must satisfy the residual ordering deterministically"
        );
        assert_eq!(
            indep_posterior_violations, 0,
            "independent-key posteriors satisfy it too (structural) — if this reds, the winner-pinning/truncation invariants broke"
        );
        let frac = iid_noise_violations as f64 / PAIRS as f64;
        eprintln!(
            "[residual] iid-control violation fraction {frac:.4} (coupled 0, indep-posterior {indep_posterior_violations})"
        );
        assert!(
            (0.30..=0.70).contains(&frac),
            "unconditional-noise control violation fraction {frac} not ~0.5"
        );
    }

    #[test]
    fn posterior_defensive_conventions() {
        let logits = [0.1f32, 0.2, 0.3];
        let mut out = [f32::NAN; 3];
        // winner out of range → no write.
        keyed_posterior_gumbel_noise(&logits, 3, 1, 1, &mut out);
        assert!(out.iter().all(|x| x.is_nan()));
        // empty logits → no write, no panic.
        keyed_posterior_gumbel_noise(&[], 0, 1, 1, &mut out);
        assert!(out.iter().all(|x| x.is_nan()));
        // V=1: the posterior degenerates to the winner's own Gumbel(0)
        // noise, drawn at the reserved max slot — pin the construction.
        let mut one = [0f32; 1];
        keyed_posterior_gumbel_noise(&[2.0f32], 0, 5, 9, &mut one);
        let u_max = keyed_unit_interval(5, 9, u32::MAX);
        let expect = (-(-u_max.ln()).ln()) as f32;
        assert!((one[0] - expect).abs() < 1e-4, "{} vs {}", one[0], expect);
    }
}
