//! refinement_marginal — single-pass token→byte distribution conversion with
//! a closed-form terminal-mass error certificate (Plan 598 / Research 559;
//! arXiv:2609.12303, "Breaking the Token Ceiling", Marathe et al., Meta FAIR
//! + UW 2026).
//!
//! # The law (substrate-independent)
//!
//! Coarse-graining a categorical distribution over a variable-depth
//! refinement tree (each symbol refines into a child-symbol sequence of
//! variable length) is **lossy exactly at terminals**: the classic
//! Marginalize-It pass restricts the vocabulary to the tokens whose child
//! sequence matches the realized prefix and renormalizes — silently deleting
//! exactly the mass of the refinements that END within the filter window, a
//! signed bias toward longer continuations of magnitude `1/(1−M)` on the
//! surviving mass. The repair this module ships: an explicit **terminal bin**
//! absorbs the boundary mass, and the per-step dropped fraction `M` is
//! exposed as a closed-form certificate on the conversion error.
//!
//! # What ships here
//!
//! | Primitive | Role |
//! |---|---|
//! | [`RefinementTable`] | symbol→child-symbol sequences, flat (built once per checkpoint; the byte instantiation builds it from BPE `id_to_vocab` bytes) |
//! | [`CoarseRecord`] | the 257-bin record: 256 child-symbol bins + index [`TERMINAL_BIN`] |
//! | [`coarse_grain_first`] / [`coarse_grain_step`] | the streaming single-pass conversion: depth 0 is the EXACT first-symbol marginal; each later step consumes one realized child symbol and emits the next conditional record |
//! | [`CoarseGrainScratch`] | caller-owned frontier pair + depth — the streaming loop allocates nothing per step |
//! | [`error_bound`] | per-step certificate: TV ≤ `M/(1−M)` worst case (measured TV is typically ≈ M — see the derivation below) |
//! | [`expected_escalation_cost`] | `Σ_k M_k` over the emitted records |
//! | [`escalation_sigmoid`] | the serve-vs-escalate gate (sigmoid, never softmax) |
//!
//! # The streaming shape (why a frontier, not a per-call rescan)
//!
//! The paper's Marginalize-It re-scans the whole vocabulary per output byte:
//! `O(vocab)` per step, `O(vocab × L)` per token of length `L`. This module
//! carries the SURVIVING set forward in scratch ([`CoarseGrainScratch`]):
//! step `k` touches only the symbols still alive at depth `k`, so the whole
//! conversion costs one pass over the flattened child table
//! (`Σ_s len(s)`) worst case — and typically far less, a realized prefix
//! narrowing the frontier geometrically after the first couple of bytes.
//!
//! # The certificate derivation (why the bound never under-reports)
//!
//! Let `T` = terminal mass dropped at this step and `C` = surviving mass
//! (the relative dropped fraction is `M = T/(T+C)`). The approximate
//! renormalized readout is `P̃(b) = P(b)/C`; the exact continuation puts the
//! dropped mass back as unknown non-negative `m_b` with `Σ m_b = T`. Then
//!
//! ```text
//! TV = ½ Σ_b |P(b)/C − P(b) − m_b|  ≤  ½ (T + T) = M ≤ M/(1−M) = T/C.
//! ```
//!
//! The tight identity is `TV ≤ M`; the shipped bound `T/C` adds the
//! `1/(1−M)` slack the plan pins, so it can only OVER-report — the G1
//! "never under-reports" property holds by construction and is asserted on
//! adversarial fixtures (all-mass-on-one-byte allocations).
//!
//! # Precision boundary (kept honest)
//!
//! The 257-bin record is lossless **w.r.t. what the single pass computed**
//! — it is NOT the exact byte distribution. Exactness at a boundary crossing
//! requires the continuation distribution after the partial token (a second
//! pass). The record's value: nothing computed is discarded, the error is
//! quantified (`M_k`), and escalation is schedulable
//! ([`expected_escalation_cost`]).
//!
//! # Determinism + allocation posture
//!
//! No randomness; identical inputs ⇒ bit-identical outputs. The hot loop is
//! a flat scatter-add over the frontier with the byte index read at
//! `child[cursor[s] + k]` — O(1) per symbol, no allocation, no per-step
//! vocabulary rescan. Serve/escalation-seam adjacency, never a per-tick
//! signal.
//!
//! Plan 598 / Research 559 — single-pass token→byte marginal with the
//! terminal-mass certificate. OPT-IN (no-default-consumer rule).

#![allow(dead_code)]

/// The terminal bin index in [`CoarseRecord::bins`] (the 256 byte bins live
/// at `0..256`; the terminal absorbing bin is the last lane).
pub const TERMINAL_BIN: usize = 256;

/// Child-symbol domain width (the byte instantiation: one bin per byte).
const SYMBOL_BINS: usize = 256;

/// Symbol→child-symbol refinement sequences, built once per checkpoint.
///
/// Layout: symbol `s`'s child sequence is
/// `child[cursor[s] .. cursor[s+1]]` (so `cursor` has `n_symbols + 1`
/// entries and `child` is the flattening). `byte_at(s, k)` is O(1) — the
/// property the streaming step relies on.
///
/// The byte instantiation (`katgpt-tokenizer` bridge) fills each sequence
/// with the token's UTF-8 bytes; the generic law is indifferent to what the
/// child domain is, only that it is `u8`-indexed (≤ 256 bins + terminal).
pub struct RefinementTable {
    n_symbols: usize,
    cursor: Vec<u32>,
    child: Vec<u8>,
}

impl RefinementTable {
    /// Build from per-symbol child sequences (one slice per symbol).
    pub fn from_sequences(seqs: &[&[u8]]) -> Self {
        let n_symbols = seqs.len();
        let mut cursor = Vec::with_capacity(n_symbols + 1);
        let mut child = Vec::new();
        cursor.push(0u32);
        for s in seqs {
            child.extend_from_slice(s);
            cursor.push(child.len() as u32);
        }
        Self {
            n_symbols,
            cursor,
            child,
        }
    }

    #[inline]
    pub fn n_symbols(&self) -> usize {
        self.n_symbols
    }

    /// Length of symbol `s`'s child sequence.
    #[inline]
    pub fn symbol_len(&self, s: usize) -> usize {
        (self.cursor[s + 1] - self.cursor[s]) as usize
    }

    /// The `k`-th child symbol of `s` (caller guarantees `k < symbol_len`).
    #[inline]
    pub fn byte_at(&self, s: usize, k: usize) -> u8 {
        self.child[self.cursor[s] as usize + k]
    }
}

/// One depth-`k` conditional record: the 257-bin lossless single-pass
/// readout. `bins[0..256]` hold the surviving child-symbol mass;
/// `bins[TERMINAL_BIN]` == `terminal_mass` holds the boundary mass of the
/// refinements that END at this depth (the mass Marginalize-It silently
/// renormalizes away). `alive_mass` = `terminal_mass + Σ bins[0..256]` —
/// the prefix-matched denominator before any renormalization.
#[derive(Clone, Copy)]
pub struct CoarseRecord {
    pub bins: [f32; SYMBOL_BINS + 1],
    /// Cached copy of `bins[TERMINAL_BIN]` (the plan's `M_k` field — the
    /// certificate reads it without a second lane load).
    pub terminal_mass: f32,
    /// Total prefix-matched mass this step saw (`dropped + surviving`).
    pub alive_mass: f32,
}

impl CoarseRecord {
    #[inline]
    pub fn zeroed() -> Self {
        Self {
            bins: [0.0; SYMBOL_BINS + 1],
            terminal_mass: 0.0,
            alive_mass: 0.0,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.bins = [0.0; SYMBOL_BINS + 1];
        self.terminal_mass = 0.0;
        self.alive_mass = 0.0;
    }
}

/// Caller-owned frontier pair + depth for the streaming conversion. Reused
/// across every step and every position — the loop allocates nothing per
/// step.
///
/// Depth invariant: after `coarse_grain_first`, `depth == 1` and the
/// frontier holds exactly the symbols with `len ≥ 1`; after each
/// `coarse_grain_step`, `depth` increments and the frontier holds exactly
/// the symbols with `len ≥ depth` (every one of which HAS a byte at depth
/// `depth − 1` — the filter's lookup contract).
pub struct CoarseGrainScratch {
    frontier: Vec<u32>,
    next: Vec<u32>,
    depth: usize,
}

impl CoarseGrainScratch {
    pub fn new(capacity: usize) -> Self {
        Self {
            frontier: Vec::with_capacity(capacity),
            next: Vec::with_capacity(capacity),
            depth: 0,
        }
    }

    /// Reset to the depth-0 state (all symbols alive). Called by
    /// [`coarse_grain_first`]; a mid-stream restart goes through it too.
    pub fn reset(&mut self, n_symbols: usize) {
        self.frontier.clear();
        self.frontier.extend(0..n_symbols as u32);
        self.next.clear();
        self.depth = 0;
    }

    /// The depth index the NEXT emitted record will be for.
    #[inline]
    pub fn depth(&self) -> usize {
        self.depth
    }
}

/// Depth 0: the EXACT first-child-symbol marginal over the full vocabulary.
///
/// Every symbol scatters `probs[s]` into its first child's bin (nothing is
/// filtered — no prefix has been consumed, so this record is the exact
/// first-symbol marginal). A symbol with an EMPTY sequence (unusual but
/// representable) lands in the terminal bin — the one place a depth-0 drop
/// can exist.
///
/// The scratch resets here and is left holding the depth-1 frontier: feed
/// each realized byte to [`coarse_grain_step`] to continue the pass.
pub fn coarse_grain_first(
    probs: &[f32],
    table: &RefinementTable,
    record: &mut CoarseRecord,
    scratch: &mut CoarseGrainScratch,
) {
    assert_eq!(probs.len(), table.n_symbols(), "probs/table vocab mismatch");
    record.clear();
    scratch.reset(table.n_symbols());
    step_body(probs, table, record, scratch, 0);
    scratch.depth = 1;
}

/// Depth `k = scratch.depth()`: consume the realized child symbol `realized`
/// observed at depth `k − 1`, emit the depth-`k` conditional record, and
/// advance the frontier.
///
/// Symbols whose sequence ENDS at depth `k` (`len == k`) had their last
/// child consumed at `k − 1` — their mass drops into the terminal bin here,
/// which is the whole point: it is kept, counted, and bounded, not
/// renormalized away.
pub fn coarse_grain_step(
    probs: &[f32],
    table: &RefinementTable,
    realized: u8,
    record: &mut CoarseRecord,
    scratch: &mut CoarseGrainScratch,
) {
    assert!(
        scratch.depth >= 1,
        "coarse_grain_first must run before step"
    );
    assert_eq!(probs.len(), table.n_symbols(), "probs/table vocab mismatch");
    record.clear();
    let k = scratch.depth;
    // Prefix filter at depth k−1: the frontier invariant (every member has
    // len ≥ k > k−1) guarantees the lookup byte exists — no branch needed.
    let cursor = &table.cursor;
    let child = &table.child;
    let k_prev = k - 1;
    let frontier = &mut scratch.frontier;
    let mut w = 0usize;
    for r in 0..frontier.len() {
        let s = frontier[r] as usize;
        debug_assert!(table.symbol_len(s) >= k);
        if child[cursor[s] as usize + k_prev] == realized {
            frontier[w] = frontier[r];
            w += 1;
        }
    }
    frontier.truncate(w);
    step_body(probs, table, record, scratch, k);
    scratch.depth = k + 1;
}

/// The shared step body at depth `k` (frontier = `len ≥ k` guaranteed):
/// scatter surviving mass by the `k`-th child symbol, drop `len == k`
/// symbols into the terminal bin, advance `len > k+1` symbols into `next`.
fn step_body(
    probs: &[f32],
    table: &RefinementTable,
    record: &mut CoarseRecord,
    scratch: &mut CoarseGrainScratch,
    k: usize,
) {
    let cursor = &table.cursor;
    let child = &table.child;
    let bins = &mut record.bins;
    let mut alive = 0.0_f32;
    scratch.next.clear();
    for &s in &scratch.frontier {
        let s = s as usize;
        let p = probs[s];
        let start = cursor[s] as usize;
        let len = (cursor[s + 1] - cursor[s]) as usize;
        if len == k {
            // Terminal at this depth: absorb, never renormalize away.
            bins[TERMINAL_BIN] += p;
            record.terminal_mass += p;
        } else {
            bins[child[start + k] as usize] += p;
            // Advance everything that can still participate at depth k+1 —
            // BOTH contributors (len > k+1) AND the len == k+1 symbols,
            // which must survive into the next frontier to take their
            // terminal drop there.
            if len > k {
                scratch.next.push(s as u32);
            }
        }
        alive += p;
    }
    record.alive_mass = alive;
    // Swap: `next` (len ≥ k+1) becomes the frontier for step k+1.
    core::mem::swap(&mut scratch.frontier, &mut scratch.next);
    scratch.next.clear();
}

/// Per-step certificate: worst-case total-variation bound between the
/// renormalized approximate readout and the exact continuation, from this
/// record's boundary mass.
///
/// With `T = terminal_mass`, `C = alive_mass − T` (surviving mass), the
/// shipped bound is `T/C` (the plan's `M/(1−M)` form, `M = T/(T+C)`).
/// Degenerate cells: no surviving mass and `T > 0` → every refinement ended
/// here, the approximate readout is vacuous → 1.0 (TV's ceiling); nothing
/// at all (`T == 0, C == 0`) → 0.0 (empty record, no error to bound).
#[inline]
pub fn error_bound(record: &CoarseRecord) -> f32 {
    let t = record.terminal_mass;
    let c = record.alive_mass - t;
    if c <= 0.0 {
        return if t > 0.0 { 1.0 } else { 0.0 };
    }
    (t / c).min(1.0)
}

/// `Σ_k M_k` over the per-step records of one position — the expected cost
/// (in extra passes) of escalating at every boundary crossing. R559: the
/// crossings are mutually independent after pass 1, so batched-exact
/// escalation amortizes this to ~1.2× on batched serving.
#[inline]
pub fn expected_escalation_cost(records: &[CoarseRecord]) -> f32 {
    let mut sum = 0.0_f32;
    for r in records {
        sum += r.terminal_mass;
    }
    sum
}

/// The serve-vs-escalate gate (sigmoid, never softmax): returns the
/// escalation probability. Below `threshold` the cost is fine (→ < 0.5);
/// above it the certificate says the approximate readout is too lossy
/// (→ > 0.5). `sharpness` is the sigmoid's inverse temperature in cost
/// units — 1.0 is the neutral default.
#[inline]
pub fn escalation_sigmoid(cost: f32, threshold: f32, sharpness: f32) -> f32 {
    let x = (cost - threshold) * sharpness;
    // Bench-844 substrate delegation (Issue 861) — the pre-substrate body was
    // expression-identical (shared exponent shape either way).
    crate::exact_sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG prob vector (house fixture pattern — no RNG dep).
    fn lcg_probs(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                0.25 + (s >> 33) as f32 / (1u64 << 31) as f32
            })
            .collect()
    }

    /// Mixed-length fixture: 8 symbols, lens 0/1/1/2/3/3/1/4, shared
    /// prefixes (three symbols start with byte 0x61), one empty sequence
    /// (the only depth-0 terminal the law allows).
    fn table_a() -> RefinementTable {
        RefinementTable::from_sequences(&[
            &[],                       // s0: empty — depth-0 terminal
            &[0x61],                   // s1: "a"
            &[0x62],                   // s2: "b"
            &[0x61, 0x63],             // s3: "ac"
            &[0x61, 0x63, 0x64],       // s4: "acd"
            &[0x61, 0x65],             // s5: "ae"
            &[0x62, 0x66],             // s6: "bf" (len 1? no — len 2)
            &[0x61, 0x63, 0x64, 0x65], // s7: "acde"
        ])
    }

    /// The paper's Marginalize-It + terminal bin, written independently as
    /// a full-vocab rescan — the reference the streaming form must match.
    fn naive_marginal(probs: &[f32], table: &RefinementTable, prefix: &[u8]) -> CoarseRecord {
        let mut rec = CoarseRecord::zeroed();
        let k = prefix.len();
        for (s, &p) in probs.iter().enumerate().take(table.n_symbols()) {
            let len = table.symbol_len(s);
            if len < k {
                continue;
            }
            if !(0..k).all(|i| table.byte_at(s, i) == prefix[i]) {
                continue;
            }
            if len == k {
                rec.bins[TERMINAL_BIN] += p;
                rec.terminal_mass += p;
            } else {
                rec.bins[table.byte_at(s, k) as usize] += p;
            }
            rec.alive_mass += p;
        }
        rec
    }

    fn assert_records_close(a: &CoarseRecord, b: &CoarseRecord, tol: f32, what: &str) {
        assert_eq!(a.terminal_mass, b.terminal_mass, "{what}: terminal");
        assert!((a.alive_mass - b.alive_mass).abs() <= tol, "{what}: alive");
        for i in 0..SYMBOL_BINS + 1 {
            assert!(
                (a.bins[i] - b.bins[i]).abs() <= tol,
                "{what}: bin {i} {} vs {}",
                a.bins[i],
                b.bins[i]
            );
        }
    }

    /// T3(a): the depth-0 record is the exact first-symbol marginal —
    /// equal to an independent brute-force scatter over the vocab.
    #[test]
    fn first_symbol_marginal_exact_vs_brute_force() {
        let table = table_a();
        let probs = lcg_probs(table.n_symbols(), 0x5EED_0598);
        let mut rec = CoarseRecord::zeroed();
        let mut scratch = CoarseGrainScratch::new(table.n_symbols());
        coarse_grain_first(&probs, &table, &mut rec, &mut scratch);

        // Independent brute force: one scatter loop over the sequences.
        let mut brute = vec![0.0_f32; SYMBOL_BINS + 1];
        for (s, &p) in probs.iter().enumerate().take(table.n_symbols()) {
            let len = table.symbol_len(s);
            let b = if len == 0 {
                TERMINAL_BIN
            } else {
                table.byte_at(s, 0) as usize
            };
            brute[b] += p;
        }
        for (i, (rec_bin, brute_bin)) in rec.bins.iter().zip(brute.iter()).enumerate() {
            assert!((rec_bin - brute_bin).abs() < 1e-6, "bin {i}");
        }
        // The only depth-0 terminal is the empty symbol s0 — its mass is
        // genuinely unresolved continuation, so the certificate is T/C > 0
        // (the marginal itself is still exact — the bins above match).
        let t = rec.terminal_mass;
        let c = rec.alive_mass - t;
        assert!((t - probs[0]).abs() < 1e-6);
        assert!((error_bound(&rec) - t / c).abs() < 1e-6);
    }

    /// T3(b): the streaming records match the independent full-rescan
    /// reference per depth along a realized path (KL = 0 to fp tolerance),
    /// and the alive mass reconstructs the prefix-matched token mass.
    #[test]
    fn round_trip_streaming_matches_naive_along_realized_path() {
        let table = table_a();
        let probs = lcg_probs(table.n_symbols(), 0x5EED_0599);
        let mut rec = CoarseRecord::zeroed();
        let mut scratch = CoarseGrainScratch::new(table.n_symbols());
        coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
        assert_records_close(&rec, &naive_marginal(&probs, &table, &[]), 1e-5, "depth 0");

        // Walk the "ac" prefix (symbols s3/s4/s7 share it), then "d".
        for (step, realized) in [0x61u8, 0x63, 0x64].iter().enumerate() {
            let prefix: Vec<u8> = [0x61u8, 0x63, 0x64][..=step].to_vec();
            coarse_grain_step(&probs, &table, *realized, &mut rec, &mut scratch);
            let want = naive_marginal(&probs, &table, &prefix);
            assert_records_close(&rec, &want, 1e-5, &format!("depth {}", step + 1));
            // alive_mass == Σ prefix-matched token probs (the round trip).
            let matched: f32 = (0..table.n_symbols())
                .filter(|&s| {
                    table.symbol_len(s) >= prefix.len()
                        && (0..prefix.len()).all(|i| table.byte_at(s, i) == prefix[i])
                })
                .map(|s| probs[s])
                .sum();
            assert!((rec.alive_mass - matched).abs() < 1e-4);
            // KL(streaming || naive) = 0 to fp tolerance (identical shape).
            let mut kl = 0.0_f32;
            for i in 0..SYMBOL_BINS + 1 {
                let q = rec.bins[i];
                if q > 0.0 {
                    let p = want.bins[i];
                    kl += q * (q / p.max(1e-30)).ln();
                }
            }
            assert!(kl.abs() < 1e-4, "KL at depth {}: {kl}", step + 1);
        }
    }

    /// T3(c): the certificate NEVER under-reports. Adversarial ground
    /// truths place ALL dropped mass on the single byte the surviving
    /// distribution likes least — the measured TV identities are M (spread)
    /// and ≤ M (concentrated); the shipped bound T/C ≥ M must cover both.
    #[test]
    fn certificate_never_underreports_on_adversarial_fixtures() {
        // Terminal-heavy vocab: six len-1 tokens + two len-3 long tail.
        let table = RefinementTable::from_sequences(&[
            &[0x10],
            &[0x11],
            &[0x12],
            &[0x13],
            &[0x14],
            &[0x15],
            &[0x10, 0x20, 0x30],
            &[0x10, 0x20, 0x31],
        ]);
        let n = table.n_symbols();
        // Normalized to a proper distribution — the TV identity below
        // compares two normalized readouts (unnormalized mass breaks it).
        let mut probs = lcg_probs(n, 0x5EED_059A);
        let total: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= total;
        }
        let mut rec = CoarseRecord::zeroed();
        let mut scratch = CoarseGrainScratch::new(n);
        coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
        // Step to depth 1 (realized 0x10): len-1 symbols s0..s5 TERMINATE
        // here — heavy boundary mass by construction.
        coarse_grain_step(&probs, &table, 0x10, &mut rec, &mut scratch);
        assert!(rec.terminal_mass > 0.0, "fixture must drop mass at depth 1");

        let t = rec.terminal_mass;
        let c = rec.alive_mass - t;
        let bound = error_bound(&rec);
        let m = t / (t + c);
        assert!(bound >= m - 1e-6, "bound {bound} must cover M {m}");

        // Adversarial true continuation: BOTH sides normalized — the
        // dropped mass all placed on the byte the survivors like least.
        let a = rec.alive_mass;
        let argmin = (0..SYMBOL_BINS)
            .min_by(|&a, &b| rec.bins[a].total_cmp(&rec.bins[b]))
            .unwrap();
        let mut tv = 0.0_f32;
        for b in 0..SYMBOL_BINS {
            let approx = rec.bins[b] / c;
            let m_b = if b == argmin { t } else { 0.0 };
            let exact = (rec.bins[b] + m_b) / a;
            tv += (approx - exact).abs();
        }
        tv *= 0.5;
        assert!(
            bound >= tv - 1e-5,
            "certificate {bound} under-reports adversarial TV {tv}"
        );
        // The tight identity: concentrated allocation measures TV ≤ M.
        assert!(tv <= m + 1e-5, "TV {tv} exceeded M {m}");
    }

    /// T3(d): an all-terminal vocabulary (every symbol length 1) — depth 0
    /// exact with certificate 0; depth 1 absorbs everything (the vacuous
    /// readout cell); a further step is a clean empty record.
    #[test]
    fn all_terminal_vocab_certificate_zero_then_absorbs() {
        let table = RefinementTable::from_sequences(&[&[0x01], &[0x02], &[0x03]]);
        let probs = vec![0.5, 0.3, 0.2];
        let mut rec = CoarseRecord::zeroed();
        let mut scratch = CoarseGrainScratch::new(3);
        coarse_grain_first(&probs, &table, &mut rec, &mut scratch);
        assert_eq!(rec.terminal_mass, 0.0);
        assert_eq!(error_bound(&rec), 0.0);
        assert!((rec.alive_mass - 1.0).abs() < 1e-6);

        coarse_grain_step(&probs, &table, 0x02, &mut rec, &mut scratch);
        // The record is CONDITIONAL on the realized byte: only the matching
        // token (prob 0.3) is in play, and it ends here — its mass is the
        // entire terminal bin, and the vacuous-readout cell fires (C = 0).
        assert!((rec.terminal_mass - 0.3).abs() < 1e-6);
        assert!((rec.alive_mass - 0.3).abs() < 1e-6);
        assert_eq!(error_bound(&rec), 1.0, "vacuous readout cell");

        // Frontier is empty now: a further step is a clean zero record.
        coarse_grain_step(&probs, &table, 0x02, &mut rec, &mut scratch);
        assert_eq!(rec.alive_mass, 0.0);
        assert_eq!(error_bound(&rec), 0.0, "empty record, nothing to bound");
    }

    /// T2 helpers: the escalation cost is the Σ of per-step terminal mass;
    /// the gate is sigmoid-monotone around the threshold (never softmax).
    #[test]
    fn escalation_cost_and_sigmoid_gate() {
        let r = [
            CoarseRecord {
                terminal_mass: 0.1,
                ..CoarseRecord::zeroed()
            },
            CoarseRecord {
                terminal_mass: 0.2,
                ..CoarseRecord::zeroed()
            },
        ];
        assert!((expected_escalation_cost(&r) - 0.3).abs() < 1e-6);
        assert!((escalation_sigmoid(0.5, 0.5, 1.0) - 0.5).abs() < 1e-6);
        assert!(escalation_sigmoid(0.4, 0.5, 1.0) < 0.5, "below → serve");
        assert!(escalation_sigmoid(0.6, 0.5, 1.0) > 0.5, "above → escalate");
        // Monotone in the cost.
        let a = escalation_sigmoid(0.45, 0.5, 10.0);
        let b = escalation_sigmoid(0.55, 0.5, 10.0);
        assert!(a < 0.5 && b > 0.5 && a < b);
    }
}
