//! Bounded template decode over CLOSED sentence grammars — katgpt-rs
//! Plan 607 T2 (opt-in `template_decode`).
//!
//! A closed grammar is a fixed table of templates; a template alternates
//! literal segments with slots, and every slot draws from a closed fill
//! vocabulary. [`Grammar::decode`] parses a sentence back into
//! (template, fill indices) — never strings, never a guess: a sentence no
//! template generates is [`DecodeError::Unknown`], and a sentence the table
//! generates MORE THAN ONE WAY is [`DecodeError::Ambiguous`] and is
//! refused (an ambiguous table would make decode a silent choice, which is
//! the failure class this module exists to exclude).
//!
//! # Scope (Plan 607 T2, R6 — the fairness argument)
//!
//! In the laya protocol the game code computes the features and renders
//! English ONLY because the reference model cannot read numbers — the
//! sentence is the model's input requirement, NOT the task's; rendering is
//! pure loss over state the code already holds. This module therefore has
//! exactly two jobs: (a) the **losslessness measurement arm** — decode a
//! fixture's sentences and score the decoded arm against the structured
//! arm, reported as an AGREEMENT DELTA where a non-zero delta is a finding
//! about the RENDER, not automatically a decode bug; (b) **third-party
//! laya-format traffic intake** — the durable consumer justification.
//! Decode-only, corpus-limited: tables are pinned per protocol version
//! (`laya-tetris-v2`, `laya-flappy-v2`, `laya-lanes-v1`), and nothing here
//! renders free text. Provenance: the `Lz4FlexDrafter` lineage (Plan 285 —
//! corpus-limited, bounded, loud-refusal pattern); `quest_grammar` is
//! riir-ai's wrapper and is never a dep.
//!
//! # Boundedness
//!
//! Fill indices are `u8` (each vocabulary holds ≤ 255 fills), a template
//! holds ≤ [`MAX_SLOTS`] slots, and a grammar ≤ 256 vocabularies — all
//! asserted at build. [`Grammar::verify_closed`] walks a template's whole
//! fill product (capped by the caller): render → decode must return the
//! identical template and fills for EVERY combination, which proves the
//! table ambiguity-free over its full closed space — the guarantee is
//! checked, never assumed. Decode itself is a backtracking segment walk:
//! zero-alloc (a fixed `u8` fill buffer on the stack), deterministic
//! (vocab order is match order), and its worst case is bounded by the same
//! small products.
//!
//! # Generic by law (R4)
//!
//! Nothing arena-specific lives here: the surface is (grammar table,
//! sentence) → (template, fills). The game vocabulary — templates, fills,
//! and the fill→feature mappings — lives in the consumer.
//!
//! template_decode — bounded template decode over CLOSED sentence grammars
//! (Plan 607 T2). A table of literal/slot templates with closed fill
//! vocabularies; decode returns (template, fill indices) and REFUSES
//! anything else (Unknown / Ambiguous — never a guess). Decode-only,
//! corpus-limited per protocol version; `verify_closed` proves each table
//! ambiguity-free over its full fill product. The losslessness-measurement
//! + laya-traffic-intake arm of the game-decision lane (Lz4FlexDrafter
//! lineage, Plan 285). Opt-in; zero deps; zero-alloc decode.

/// Maximum slots per template (`u8` fill indices keep the match buffer
/// stack-local; the build refuses anything wider).
pub const MAX_SLOTS: usize = 8;

/// One slot's closed fill vocabulary. Order is CONTRACT: the fill index is
/// the decoded value, so consumers pin ordinals to this order.
pub type Vocab = &'static [&'static str];

/// One template segment: even positions are literals, odd positions are
/// slots naming a vocabulary index (validated at [`Grammar::new`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seg {
    /// Verbatim bytes.
    Lit(&'static str),
    /// A slot filled from vocabulary `u8`.
    Slot(u8),
}

/// One closed template: alternating literal/slot segments, starting and
/// ending with a literal (either may be empty).
#[derive(Debug, Clone, Copy)]
pub struct Template(pub &'static [Seg]);

/// A validated closed grammar: the decode table IS the protocol pin.
#[derive(Debug, Clone, Copy)]
pub struct Grammar {
    vocabs: &'static [Vocab],
    templates: &'static [Template],
}

/// One successful decode: which template generated the sentence and which
/// fill each slot took (indices into the slot's vocabulary, in template
/// slot order). Only `fills[..n_slots]` is meaningful.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeMatch {
    pub template: usize,
    pub fills: [u8; MAX_SLOTS],
    pub n_slots: usize,
}

/// Why a sentence did not decode. Both variants are LOUD refusals — decode
/// never guesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// No template in the grammar generates this sentence.
    Unknown,
    /// The table generates this sentence more than one way — ambiguous
    /// grammar; refused rather than silently choosing one reading.
    Ambiguous,
}

impl Grammar {
    /// Build + validate a closed grammar. Malformed tables are programmer
    /// errors (the tables are pinned code, not data) and panic LOUD.
    pub fn new(vocabs: &'static [Vocab], templates: &'static [Template]) -> Self {
        assert!(
            !vocabs.is_empty() && vocabs.len() <= 256,
            "grammar: needs 1..=256 vocabularies (u8 slot indices), got {}",
            vocabs.len()
        );
        for (i, v) in vocabs.iter().enumerate() {
            assert!(
                !v.is_empty() && v.len() <= 255,
                "grammar: vocab {i} needs 1..=255 fills, got {}",
                v.len()
            );
        }
        assert!(
            !templates.is_empty(),
            "grammar: needs at least one template"
        );
        for (ti, t) in templates.iter().enumerate() {
            let segs = t.0;
            assert!(
                !segs.is_empty() && segs.len() % 2 == 1,
                "template {ti}: segments must alternate literal/slot and start+end \
                 with a literal (odd count), got {}",
                segs.len()
            );
            let mut slots = 0usize;
            for (si, seg) in segs.iter().enumerate() {
                match (si % 2, seg) {
                    (0, Seg::Lit(_)) => {}
                    (1, Seg::Slot(v)) => {
                        assert!(
                            (*v as usize) < vocabs.len(),
                            "template {ti} seg {si}: vocab index {v} out of range"
                        );
                        slots += 1;
                    }
                    (pos, other) => panic!(
                        "template {ti} seg {si}: segments must alternate \
                         (even=Lit, odd=Slot); position {pos} holds {other:?}"
                    ),
                }
            }
            assert!(
                slots <= MAX_SLOTS,
                "template {ti}: {slots} slots overflows MAX_SLOTS({MAX_SLOTS})"
            );
        }
        Self { vocabs, templates }
    }

    /// Vocabulary count.
    pub fn vocab_count(&self) -> usize {
        self.vocabs.len()
    }

    /// Template count.
    pub fn template_count(&self) -> usize {
        self.templates.len()
    }

    /// The vocabulary a template's `slot` draws from.
    fn slot_vocab(&self, template: usize, slot: usize) -> Vocab {
        let mut seen = 0usize;
        for seg in self.templates[template].0 {
            if let Seg::Slot(v) = seg {
                if seen == slot {
                    return self.vocabs[*v as usize];
                }
                seen += 1;
            }
        }
        unreachable!("slot {slot} out of range (validated at build)");
    }

    /// Number of slots in a template.
    pub fn template_slot_count(&self, template: usize) -> usize {
        self.templates[template]
            .0
            .iter()
            .filter(|s| matches!(s, Seg::Slot(_)))
            .count()
    }

    /// The decoded string of slot `slot` of match `m`.
    pub fn fill_str(&self, m: &DecodeMatch, slot: usize) -> &'static str {
        assert!(slot < m.n_slots, "fill_str: slot {slot} out of range");
        self.slot_vocab(m.template, slot)[m.fills[slot] as usize]
    }

    /// Decode a sentence against the closed table. Zero-alloc, backtrack-
    /// free of guesses: exactly one derivation must exist, or the sentence
    /// is refused ([`DecodeError::Ambiguous`] / [`DecodeError::Unknown`]).
    pub fn decode(&self, sentence: &str) -> Result<DecodeMatch, DecodeError> {
        let text = sentence.as_bytes();
        let mut total = 0usize;
        let mut first: Option<DecodeMatch> = None;
        for (ti, t) in self.templates.iter().enumerate() {
            let mut fills = [0u8; MAX_SLOTS];
            let mut walker = Walker {
                g: self,
                text,
                first: &mut first,
                template: ti,
            };
            total += walker.walk(t.0, 0, 0, 0, &mut fills);
            if total > 1 {
                return Err(DecodeError::Ambiguous);
            }
        }
        match first {
            Some(m) => Ok(m),
            None => Err(DecodeError::Unknown),
        }
    }

    /// Render a template with the given fills into `out` (append). The
    /// inverse of [`Self::decode`] — the pair is what makes a table
    /// checkable against its renderer without sharing code with it.
    pub fn render_into(&self, template: usize, fills: &[u8], out: &mut String) {
        assert!(template < self.templates.len(), "render: bad template");
        let mut slot = 0usize;
        for seg in self.templates[template].0 {
            match seg {
                Seg::Lit(l) => out.push_str(l),
                Seg::Slot(v) => {
                    assert!(
                        slot < fills.len(),
                        "render: fills shorter than the template's slots"
                    );
                    let vocab = self.vocabs[*v as usize];
                    let fi = fills[slot] as usize;
                    assert!(
                        fi < vocab.len(),
                        "render: fill {fi} out of range for vocab {v}"
                    );
                    out.push_str(vocab[fi]);
                    slot += 1;
                }
            }
        }
        assert_eq!(
            slot,
            self.template_slot_count(template),
            "render: fills longer than the template's slots"
        );
    }

    /// [`Self::render_into`] into a fresh String (validation paths only —
    /// the decode hot path never allocates).
    pub fn render(&self, template: usize, fills: &[u8]) -> String {
        let mut s = String::new();
        self.render_into(template, fills, &mut s);
        s
    }

    /// The bounded-grammar proof: walk EVERY fill combination of every
    /// template (capped at `max_combinations` per template — a product over
    /// the cap means the "bounded" claim is unchecked and must not pass),
    /// render, decode, and require the identical (template, fills) back.
    /// Catches within-template ambiguity (duplicate/prefix-overlapping
    /// fills), cross-template collisions, and render/decode drift — over
    /// the FULL closed space, not just an observed corpus. Validation-only
    /// (allocates one reused scratch String); decode itself never does.
    pub fn verify_closed(&self, max_combinations: usize) -> Result<(), String> {
        let mut scratch = String::new();
        for ti in 0..self.templates.len() {
            let n = self.template_slot_count(ti);
            let mut lens = [0usize; MAX_SLOTS];
            let mut product = 1usize;
            for (s, slot) in lens.iter_mut().enumerate().take(n) {
                let l = self.slot_vocab(ti, s).len();
                *slot = l;
                product = product.saturating_mul(l);
            }
            if product > max_combinations {
                return Err(format!(
                    "template {ti}: fill product {product} exceeds the cap \
                     {max_combinations} — the bounded claim is unchecked"
                ));
            }
            let mut fills = [0u8; MAX_SLOTS];
            loop {
                scratch.clear();
                self.render_into(ti, &fills, &mut scratch);
                match self.decode(&scratch) {
                    Ok(m) => {
                        if m.template != ti || m.fills[..n] != fills[..n] {
                            return Err(format!(
                                "template {ti}: round-trip mismatch on {scratch:?} \
                                 (decoded template {}, fills {:?})",
                                m.template,
                                &m.fills[..m.n_slots]
                            ));
                        }
                    }
                    Err(e) => {
                        return Err(format!(
                            "template {ti}: rendered {scratch:?} refused: {e:?}"
                        ));
                    }
                }
                // Odometer: next fill combination (row-major, vocab order).
                let mut s = 0usize;
                while s < n {
                    fills[s] += 1;
                    if fills[s] as usize == lens[s] {
                        fills[s] = 0;
                        s += 1;
                    } else {
                        break;
                    }
                }
                if s == n {
                    break; // wrapped every slot — this template is done
                }
            }
        }
        Ok(())
    }
}

/// The recursive segment walker: grammar + sentence + the first-derivation
/// recorder. One per decode; the recursion carries only cursors and the
/// stack-local fill buffer.
struct Walker<'a> {
    g: &'a Grammar,
    text: &'a [u8],
    first: &'a mut Option<DecodeMatch>,
    template: usize,
}

impl Walker<'_> {
    fn walk(
        &mut self,
        segs: &'static [Seg],
        seg_i: usize,
        slot_i: usize,
        pos: usize,
        fills: &mut [u8; MAX_SLOTS],
    ) -> usize {
        match segs[seg_i] {
            Seg::Lit(lit) => {
                let b = lit.as_bytes();
                if self.text.len() < pos + b.len() || &self.text[pos..pos + b.len()] != b {
                    return 0;
                }
                let next = pos + b.len();
                if seg_i + 1 == segs.len() {
                    if next != self.text.len() {
                        return 0;
                    }
                    if self.first.is_none() {
                        *self.first = Some(DecodeMatch {
                            template: self.template,
                            fills: *fills,
                            n_slots: slot_i,
                        });
                    }
                    1
                } else {
                    self.walk(segs, seg_i + 1, slot_i, next, fills)
                }
            }
            Seg::Slot(v) => {
                let vocab = self.g.vocabs[v as usize];
                let mut count = 0usize;
                for (fi, fill) in vocab.iter().enumerate() {
                    let fb = fill.as_bytes();
                    if self.text.len() >= pos + fb.len() && &self.text[pos..pos + fb.len()] == fb {
                        fills[slot_i] = fi as u8;
                        count += self.walk(segs, seg_i + 1, slot_i + 1, pos + fb.len(), fills);
                        if count > 1 {
                            return count; // ambiguous already — stop early
                        }
                    }
                }
                count
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static T_ONE: [Seg; 3] = [Seg::Lit("L "), Seg::Slot(0), Seg::Lit("")];
    static TEMPLATES_ONE: [Template; 1] = [Template(&T_ONE)];

    fn one_lit_grammar(vocab: &'static [&'static str]) -> Grammar {
        let vocabs: &'static [Vocab] = Box::leak(vec![vocab].into_boxed_slice());
        Grammar::new(vocabs, &TEMPLATES_ONE)
    }

    #[test]
    fn decode_and_render_round_trip() {
        let g = one_lit_grammar(&["alpha", "beta"]);
        for (i, fill) in ["alpha", "beta"].iter().enumerate() {
            let s = format!("L {fill}");
            let m = g.decode(&s).expect("decodes");
            assert_eq!(m.template, 0);
            assert_eq!(m.n_slots, 1);
            assert_eq!(m.fills[0], i as u8);
            assert_eq!(g.fill_str(&m, 0), *fill);
            assert_eq!(g.render(0, &[i as u8]), s);
        }
    }

    #[test]
    fn unknown_sentence_is_a_loud_refusal() {
        let g = one_lit_grammar(&["alpha"]);
        assert_eq!(g.decode("L gamma"), Err(DecodeError::Unknown));
        assert_eq!(g.decode(""), Err(DecodeError::Unknown));
        assert_eq!(g.decode("L alpha extra"), Err(DecodeError::Unknown));
    }

    #[test]
    fn duplicate_fills_are_ambiguous() {
        let g = one_lit_grammar(&["same", "same"]);
        assert_eq!(g.decode("L same"), Err(DecodeError::Ambiguous));
        assert!(g.verify_closed(100).is_err());
    }

    #[test]
    fn prefix_overlapping_fills_backtrack_without_guessing() {
        // "falling" is a prefix of "falling fast": each sentence must have
        // exactly ONE derivation, reached by backtracking in vocab order.
        let g = one_lit_grammar(&["falling fast", "falling"]);
        assert_eq!(g.decode("L falling fast").unwrap().fills[0], 0);
        assert_eq!(g.decode("L falling").unwrap().fills[0], 1);
        assert_eq!(g.decode("L falling.").err(), Some(DecodeError::Unknown));
    }

    #[test]
    fn empty_fill_and_empty_literals_decode() {
        // Template: "" slot "" — the tetris clears slot is exactly this
        // shape ("" = "no clear").
        static SEGS: [Seg; 3] = [Seg::Lit(""), Seg::Slot(0), Seg::Lit("")];
        static VOCAB: [&str; 2] = ["", "x"];
        static VOCABS: [&[&str]; 1] = [&VOCAB];
        static TEMPLATES: [Template; 1] = [Template(&SEGS)];
        let g = Grammar::new(&VOCABS, &TEMPLATES);
        assert_eq!(g.decode("").unwrap().fills[0], 0);
        assert_eq!(g.decode("x").unwrap().fills[0], 1);
        assert_eq!(g.decode("y"), Err(DecodeError::Unknown));
    }

    #[test]
    fn multi_template_dispatch() {
        static SEGS_A: [Seg; 3] = [Seg::Lit("The "), Seg::Slot(0), Seg::Lit(" runs.")];
        static SEGS_B: [Seg; 3] = [Seg::Lit("The "), Seg::Slot(1), Seg::Lit(" walks.")];
        static VOCAB_A: [&str; 2] = ["fox", "dog"];
        static VOCAB_B: [&str; 1] = ["goose"];
        static VOCABS: [&[&str]; 2] = [&VOCAB_A, &VOCAB_B];
        static TEMPLATES: [Template; 2] = [Template(&SEGS_A), Template(&SEGS_B)];
        let g = Grammar::new(&VOCABS, &TEMPLATES);
        assert_eq!(g.decode("The fox runs.").unwrap().template, 0);
        assert_eq!(g.decode("The goose walks.").unwrap().template, 1);
        assert_eq!(g.decode("The goose runs."), Err(DecodeError::Unknown));
    }

    #[test]
    fn verify_closed_proves_the_full_space() {
        static SEGS: [Seg; 5] = [
            Seg::Lit("<"),
            Seg::Slot(0),
            Seg::Lit("|"),
            Seg::Slot(1),
            Seg::Lit(">"),
        ];
        static VOCAB_A: [&str; 3] = ["a", "b", "c"];
        static VOCAB_B: [&str; 2] = ["x", "y"];
        static VOCABS: [&[&str]; 2] = [&VOCAB_A, &VOCAB_B];
        static TEMPLATES: [Template; 1] = [Template(&SEGS)];
        let g = Grammar::new(&VOCABS, &TEMPLATES);
        assert_eq!(g.verify_closed(6), Ok(()));
        // Below the product (3*2 = 6) the claim is UNCHECKED — refuse.
        assert!(g.verify_closed(5).is_err());
    }

    #[test]
    fn verify_closed_catches_a_cross_template_collision() {
        // Two templates that generate the same sentence: decode must refuse
        // it, and verify_closed must find the collision over the space.
        static SEGS_A: [Seg; 1] = [Seg::Lit("dup")];
        static SEGS_B: [Seg; 1] = [Seg::Lit("dup")];
        static VOCAB: [&str; 1] = ["z"];
        static VOCABS: [&[&str]; 1] = [&VOCAB];
        static TEMPLATES: [Template; 2] = [Template(&SEGS_A), Template(&SEGS_B)];
        let g = Grammar::new(&VOCABS, &TEMPLATES);
        assert_eq!(g.decode("dup"), Err(DecodeError::Ambiguous));
        assert!(g.verify_closed(10).is_err());
    }

    #[test]
    #[should_panic(expected = "overflows MAX_SLOTS")]
    fn slot_overflow_is_refused_at_build() {
        static VOCAB: [&str; 1] = ["a"];
        static VOCABS: [&[&str]; 1] = [&VOCAB];
        let segs: Vec<Seg> = (0..2 * MAX_SLOTS + 3)
            .map(|i| {
                if i % 2 == 0 {
                    Seg::Lit("")
                } else {
                    Seg::Slot(0)
                }
            })
            .collect();
        let segs: &'static [Seg] = Box::leak(segs.into_boxed_slice());
        let templates: &'static [Template] = Box::leak(vec![Template(segs)].into_boxed_slice());
        Grammar::new(&VOCABS, templates);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn bad_vocab_index_is_refused_at_build() {
        static SEGS: [Seg; 3] = [Seg::Lit(""), Seg::Slot(7), Seg::Lit("")];
        static VOCAB: [&str; 1] = ["a"];
        static VOCABS: [&[&str]; 1] = [&VOCAB];
        static TEMPLATES: [Template; 1] = [Template(&SEGS)];
        Grammar::new(&VOCABS, &TEMPLATES);
    }

    #[test]
    #[should_panic(expected = "segments must alternate")]
    fn non_alternating_segments_are_refused_at_build() {
        static SEGS: [Seg; 2] = [Seg::Slot(0), Seg::Lit("x")];
        static VOCAB: [&str; 1] = ["a"];
        static VOCABS: [&[&str]; 1] = [&VOCAB];
        static TEMPLATES: [Template; 1] = [Template(&SEGS)];
        Grammar::new(&VOCABS, &TEMPLATES);
    }
}
