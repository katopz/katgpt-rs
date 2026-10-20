//! G5 span-baseline arms (Plan 628 T0.7).
//!
//! Scores `span_pointer` (the F0 span primitive, feature `span_pointer`)
//! against a RULE FLOOR on an EXTERNAL text-bearing NER fixture — the
//! MIT-Movie trivia10k13 test split (BLAKE3-pinned; provenance:
//! `tests/fixtures/external/PROVENANCE_MIT_MOVIE.md`). This is the GOAT
//! gate the plan binds: **`span_pointer` earns its G5 only by beating the
//! rule floor** (verdict round 1: our own goldens alone are circular).
//!
//! # The task (pre-registered before any arm ran)
//!
//! Schema-vocabulary grounding — the F0 production shape ("alias/dictionary
//! tables per field schema", Research 627 §5 Extract row): every entity in
//! the fixture belongs to a VALUE GROUP (its type's vocabulary cluster); the
//! query is the group's canonical surface plus its alias surfaces, and the
//! answer is the annotated span of THIS mention in THIS row's text. The
//! group tables are derived mechanically from the fixture corpus itself
//! (schema-derived like a real deployment's choices table — never
//! hand-authored, and identical inputs for every arm).
//!
//! - **Value groups:** derived from the trivia10k13 TRAIN split (a second
//!   BLAKE3-pinned table — NEVER from the scored rows; verdict round 1
//!   REVISE: test-derived aliases made the alias channel an answer key).
//!   Per type, connected components over "surface A's token sequence is a
//!   contiguous subsequence of surface B's" (token = maximal
//!   ASCII-alphanumeric run — the span_pointer tokenizer's class).
//!   Canonical = the most frequent TRAIN surface (ties: lexicographically
//!   smallest); aliases = the other members by descending frequency, capped
//!   at [`ALIAS_CAP`]. A scored mention whose (type, surface) the schema
//!   lacks is OUT of the population (a deployment cannot ground what its
//!   schema never saw) — counted and disclosed.
//! - **Expected span:** the annotated span trimmed to word boundaries (the
//!   same word-byte class both arms' answers land on). Applied uniformly.
//! - **Rule-class floors (pre-registered, verbatim-round-1 REVISE arms):**
//!   the bare rule = leftmost case-identical occurrence of the canonical;
//!   the gazetteer arms get the SAME (canonical, aliases) inputs the
//!   pointer gets, matched literally — `first-in-order` (canonical then
//!   aliases in list order, first hit wins) and `longest-match` (the
//!   longest occurring surface wins, ties by list order — the standard
//!   gazetteer rule and the GOAT bar's comparator). Literal matching is
//!   what the rule class does; the floor is no longer denied the schema.
//! - **Pointer arms (defaults, top-5):** `SpanQuery { text: canonical,
//!   aliases }` at the shipped `SpanConfig::default()`. *raw* = rank-0
//!   window or abstain when nothing survives the default 0.5 gate. *gate* =
//!   the production posture — rank-0 must clear the nothing-fired boundary
//!   STRICTLY (`score > 0.5 + 1e-3`; a window that fires nothing scores
//!   exactly `sigmoid(0) = 0.5`): below it abstains, which is what Temper's
//!   fused abstain composes in production. *no-alias* (gate) = the same
//!   with an empty alias table — isolates the alias channel's contribution.
//!
//! # Metrics + populations
//!
//! Per arm: exact (answer == expected), containment (answer ⊇ expected),
//! abstains — over the **choices-class** population (the 9 types whose
//! surface unique% < 95, restricted to mentions the TRAIN schema carries),
//! the **all-types** population (same restriction), the **non-degenerate
//! slice** (canonical != the mention's own surface — where a literal rule
//! and a ranked pointer can differ), and per-type rows. The **GOAT bar**
//! (pre-registered): pointer-gate exact > floor+alias(longest-match) exact
//! on the choices-class aggregate — the strongest rule-class arm, same
//! schema inputs.
//!
//! # What is asserted
//!
//! Pin gate + census (rows/entities/per-type counts — any fixture drift
//! refuses to score), lowercase sanity, structural liveness (on verbatim
//! mentions — canonical bytes literally present — the floor NEVER abstains
//! by identity and the pointer ALWAYS answers: both are tripwires against a
//! broken harness), determinism (the scoring pass runs twice; the aggregate
//! checksums must be identical and nonzero), and the measured ratchets
//! (downward-only, pinned from the measured record — a regression past them
//! reds). The GOAT relation is ASSERTED as a standing gate: pointer(gate) >
//! floor+alias(longest) on choices-class — that inequality is the promotion
//! basis, so a scorer change that voids it reds here and forces
//! re-adjudication (the full tables live in
//! `.benchmarks/950_f0_t07_span_baselines.md`).
//!
//! The table prints to stdout (captured on green — use `--nocapture`).

use katgpt_extract::span_pointer::{SpanConfig, SpanPointer, SpanQuery, SpanScore};
use serde_json::Value;

/// The pinned corpus, relative to the crate root (manifest-anchored, so the
/// test runs from any working directory).
const FIXTURE_REL: &str = "tests/fixtures/external/mit_movie_trivia_test.jsonl";

/// The SCHEMA vocabulary — derived from the trivia10k13 TRAIN split, never
/// from the scored rows (verdict round 1 REVISE: test-derived aliases made
/// the alias channel an answer key; the deployment analogue is a schema
/// compiled from historical data, which is exactly what the train split is).
const VOCAB_REL: &str = "tests/fixtures/external/mit_movie_trivia_train_vocab.jsonl";

/// BLAKE3 pin over the fixture bytes (see PROVENANCE_MIT_MOVIE.md).
const BLAKE3_PIN: &str = "77a36d097177560e49378e9a3b7ea4496d43a06ae2ca43667f680bf0cf65c56d";

/// BLAKE3 pin over the train-split vocabulary table.
const VOCAB_BLAKE3_PIN: &str = "3a7ba3868950deafa6f8f7eb4a578f72a954f661b009e85800169f6d3d1b9b4a";

/// Census pins (the converter's build record — a drifted fixture reds here).
const ROWS_EXPECTED: usize = 1562;
const ENTITIES_EXPECTED: usize = 5093;
/// Per-type entity counts over kept rows (`type` = the upstream tag name).
const TYPE_EXPECTED: [(&str, usize); 12] = [
    ("Plot", 1218),
    ("Actor", 1218),
    ("Genre", 755),
    ("Year", 643),
    ("Director", 411),
    ("Character_Name", 260),
    ("Opinion", 183),
    ("Relationship", 157),
    ("Origin", 138),
    ("Award", 65),
    ("Quote", 37),
    ("Soundtrack", 8),
];
/// Choices-class = types whose fixture-wide surface unique% < 95 (an
/// enumerable vocabulary exists); the pre-registered scoring population.
const CHOICES_CLASS: [&str; 9] = [
    "Actor",
    "Award",
    "Character_Name",
    "Director",
    "Genre",
    "Opinion",
    "Relationship",
    "Soundtrack",
    "Year",
];

/// Census pins for the TRAIN-derived vocabulary (surfaces per type) — a
/// drifted schema table reds here.
const VOCAB_TYPE_EXPECTED: [(&str, usize); 9] = [
    ("Actor", 1436),
    ("Award", 163),
    ("Character_Name", 664),
    ("Director", 532),
    ("Genre", 908),
    ("Opinion", 292),
    ("Relationship", 346),
    ("Soundtrack", 47),
    ("Year", 143),
];

/// Pointer top-k buffer (the any-containment disclosure depth).
const TOP_K: usize = 5;
/// Alias surfaces carried by one query (the schema-choices cap).
const ALIAS_CAP: usize = 8;
/// The strict nothing-boundary slack for the gate posture (f32 rounding
/// around `sigmoid(0) = 0.5`).
const GATE_EPS: f32 = 1e-3;
/// The nothing-fired score (the documented span_pointer boundary).
const NOTHING_FIRED: f32 = 0.5;

/// Measured ratchets (downward-only) — pinned from the T0.7 re-run record
/// (verdict round 1 REVISE addressed: TRAIN-derived schema vocabulary + the
/// gazetteer arms; 2026-10-20, debug profile; full tables in
/// `.benchmarks/950_f0_t07_span_baselines.md`). A ratchet never reds an
/// improvement; a scorer or harness regression past it fails here. The GOAT
/// bar's comparator is `FLOOR_ALIAS_LONGEST` — the strongest floor variant
/// (same alias inputs as the pointer, matched literally, longest-match
/// gazetteer rule).
const FLOOR_ALIAS_LONGEST_EXACT_RATCHET: u32 = 2304;
const PTR_EXACT_RATCHET: u32 = 2200;
const PTR_CONTAIN_RATCHET: u32 = 2488;
const ND_PTR_EXACT_RATCHET: u32 = 537;
const GENRE_PTR_EXACT_RATCHET: u32 = 277;

/// One fixture row: the text plus `(type, start, end)` entities (byte
/// offsets, end exclusive).
struct Row {
    id: String,
    text: String,
    entities: Vec<(String, u32, u32)>,
}

/// The tokenizer's character law — the SAME class as span_pointer's
/// `is_word_byte` (ASCII alphanumeric or non-ASCII; the fixture is pure
/// ASCII so the non-ASCII arm never fires). Word-boundary trims and group
/// tokenization both use it.
#[inline]
fn is_word_byte(c: u8) -> bool {
    !c.is_ascii() || c.is_ascii_alphanumeric()
}

/// Trim `[s, e)` to word boundaries (the expected span lands where both
/// arms' answers can land: token-aligned byte ranges).
fn trim_word_bounds(text: &[u8], mut s: u32, mut e: u32) -> (u32, u32) {
    while s < e && !is_word_byte(text[s as usize]) {
        s += 1;
    }
    while e > s && !is_word_byte(text[(e - 1) as usize]) {
        e -= 1;
    }
    (s, e)
}

/// Maximal word-byte runs of `s`, lowercased (the fixture is lowercase; the
/// fold kept for symmetry with the scorer's own case handling).
fn surface_tokens(s: &str) -> Vec<String> {
    s.as_bytes()
        .split(|&b| !is_word_byte(b))
        .filter(|p| !p.is_empty())
        .map(|p| String::from_utf8_lossy(p).to_ascii_lowercase())
        .collect()
}

/// One value group: the canonical query surface and its alias table.
#[derive(Clone)]
struct Group {
    canonical: String,
    aliases: Vec<String>,
}

/// The derived schema vocabulary: for every `(type, surface)` in the TRAIN
/// table, the group it belongs to.
struct Vocabulary {
    of_surface: std::collections::HashMap<(String, String), Group>,
}

/// Parse + pin-check the TRAIN-split vocabulary table (one JSON line per
/// type: `{"type", "surfaces": [[surface, count], …]}`).
fn load_vocabulary(path: &std::path::Path) -> Vocabulary {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("G5 span: cannot read vocab {}: {e}", path.display()));
    let got = blake3::hash(&bytes).to_hex();
    assert_eq!(
        got.as_str(),
        VOCAB_BLAKE3_PIN,
        "G5 span vocab drifted: BLAKE3 {} != pinned {} — re-pin via PROVENANCE_MIT_MOVIE.md first",
        got.as_str(),
        VOCAB_BLAKE3_PIN
    );
    let text = std::str::from_utf8(&bytes).expect("G5 span vocab is UTF-8");
    let mut freq: std::collections::HashMap<(String, String), u32> =
        std::collections::HashMap::new();
    for (i, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("G5 span vocab line {}: not JSON: {e}", i + 1));
        let ty = v
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("G5 span vocab line {}: missing type", i + 1));
        let surfaces = v
            .get("surfaces")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("G5 span vocab line {}: missing surfaces", i + 1));
        for s in surfaces {
            let arr = s.as_array().expect("vocab surface pair");
            let surface = arr[0].as_str().expect("vocab surface string").to_string();
            let count = arr[1].as_u64().expect("vocab surface count") as u32;
            freq.insert((ty.to_string(), surface), count);
        }
    }
    // Census pins (per-type surface counts — the converter's build record).
    {
        let mut per_type: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (ty, _) in freq.keys() {
            *per_type.entry(ty.as_str()).or_default() += 1;
        }
        for (ty, n) in VOCAB_TYPE_EXPECTED {
            assert_eq!(
                per_type.get(ty).copied().unwrap_or(0),
                n,
                "G5 span vocab census: type {ty} holds {}, expected {n}",
                per_type.get(ty).copied().unwrap_or(0)
            );
        }
    }
    build_vocabulary(freq)
}

/// Build the value groups over the TRAIN surface table.
///
/// Deterministic: components over the token-contiguous-subsequence
/// relation; canonical = (max TRAIN frequency, then lexicographically
/// smallest); aliases = the other members by (descending frequency,
/// ascending lexical), capped at [`ALIAS_CAP`]. Every member surface maps
/// to the same group (the schema's choices entry, whichever surface the
/// mention uses).
fn build_vocabulary(freq: std::collections::HashMap<(String, String), u32>) -> Vocabulary {
    use std::collections::{HashMap, HashSet};

    // Per-type surface lists + union-find over the containment relation.
    let mut by_type: HashMap<String, Vec<String>> = HashMap::new();
    for (ty, surface) in freq.keys() {
        by_type.entry(ty.clone()).or_default().push(surface.clone());
    }
    for list in by_type.values_mut() {
        list.sort(); // deterministic iteration
    }

    let mut parent: HashMap<(String, String), (String, String)> = HashMap::new();
    let find = |parent: &HashMap<(String, String), (String, String)>,
                mut x: (String, String)|
     -> (String, String) {
        while let Some(p) = parent.get(&x) {
            if p == &x {
                break;
            }
            x = p.clone();
        }
        x
    };
    for (ty, surfaces) in &by_type {
        let set: HashSet<&String> = surfaces.iter().collect();
        // Enumerate each surface's contiguous token subsequences; every
        // subsequence that IS another surface unions the pair (a ⊂ b from
        // b's side; union-find carries the transitive closure).
        for b in surfaces {
            let btoks = surface_tokens(b);
            let max_l = btoks.len().min(24);
            for l in 1..=max_l {
                for i in 0..=btoks.len() - l {
                    let sub = btoks[i..i + l].join(" ");
                    if &sub != b && set.contains(&sub) {
                        let a_key = (ty.clone(), sub);
                        let b_key = (ty.clone(), b.clone());
                        let ra = find(&parent, a_key.clone());
                        let rb = find(&parent, b_key.clone());
                        if ra != rb {
                            parent.insert(ra, rb.clone());
                        }
                    }
                }
            }
        }
    }

    // Members per component root.
    let mut members: HashMap<(String, String), Vec<String>> = HashMap::new();
    for (ty, surfaces) in &by_type {
        for s in surfaces {
            let key = (ty.clone(), s.clone());
            let root = find(&parent, key);
            members.entry(root).or_default().push(s.clone());
        }
    }

    // Canonical + aliases per component, then expand: EVERY member surface
    // maps to the same group (the schema's choices entry, whichever surface
    // the mention uses).
    let mut of_surface = HashMap::with_capacity(members.len());
    for (root, mut comp) in members {
        let ty = root.0;
        comp.sort_by(|a, b| {
            let fa = freq[&(ty.clone(), a.clone())];
            let fb = freq[&(ty.clone(), b.clone())];
            fb.cmp(&fa).then_with(|| a.cmp(b))
        });
        let canonical = comp[0].clone();
        let aliases: Vec<String> = comp.iter().skip(1).take(ALIAS_CAP).cloned().collect();
        let group = Group { canonical, aliases };
        for member in &comp {
            of_surface.insert((ty.clone(), member.clone()), group.clone());
        }
    }
    Vocabulary { of_surface }
}

#[test]
fn g5_span_baselines() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_REL);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("G5 span: cannot read fixture {}: {e}", path.display()));

    // 1. Pin gate — a drifted fixture must never silently score.
    let got = blake3::hash(&bytes).to_hex();
    assert_eq!(
        got.as_str(),
        BLAKE3_PIN,
        "G5 span fixture drifted: BLAKE3 {} != pinned {} — re-pin via PROVENANCE_MIT_MOVIE.md first",
        got.as_str(),
        BLAKE3_PIN
    );

    // 2. Parse + census (row / entity / per-type counts are build pins).
    let text = std::str::from_utf8(&bytes).expect("G5 span fixture is UTF-8");
    let rows: Vec<Row> = text
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let v: Value = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("G5 span line {}: not JSON: {e}", i + 1));
            let id = v
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("G5 span line {}: missing id", i + 1))
                .to_string();
            let row_text = v
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("G5 span row {id}: missing text"))
                .to_string();
            // The floor rule is case-identical literal matching; the corpus
            // is lowercase by construction — refuse to score if that drifts.
            assert_eq!(
                row_text,
                row_text.to_lowercase(),
                "G5 span row {id}: text is not lowercase — the case-identical floor rule drifted"
            );
            let entities = v
                .get("entities")
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("G5 span row {id}: missing entities"))
                .iter()
                .map(|e| {
                    let ty = e
                        .get("type")
                        .and_then(Value::as_str)
                        .expect("entity type")
                        .to_string();
                    let s = e.get("start").and_then(Value::as_u64).expect("start") as u32;
                    let en = e.get("end").and_then(Value::as_u64).expect("end") as u32;
                    (ty, s, en)
                })
                .collect();
            Row {
                id,
                text: row_text,
                entities,
            }
        })
        .collect();
    assert_eq!(
        rows.len(),
        ROWS_EXPECTED,
        "G5 span fixture holds {} rows, expected {ROWS_EXPECTED}",
        rows.len()
    );
    let entity_total: usize = rows.iter().map(|r| r.entities.len()).sum();
    assert_eq!(
        entity_total, ENTITIES_EXPECTED,
        "G5 span fixture holds {entity_total} entities, expected {ENTITIES_EXPECTED}"
    );
    {
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for r in &rows {
            for (ty, _, _) in &r.entities {
                *counts.entry(ty.as_str()).or_default() += 1;
            }
        }
        for (ty, n) in TYPE_EXPECTED {
            assert_eq!(
                counts.get(ty).copied().unwrap_or(0),
                n,
                "G5 span census: type {ty} holds {}, expected {n}",
                counts.get(ty).copied().unwrap_or(0)
            );
        }
    }

    // 3. The schema vocabulary (TRAIN-derived — never the scored rows).
    let vocab_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(VOCAB_REL);
    let vocab = load_vocabulary(&vocab_path);

    // 4. Score every choices-class entity whose (type, surface) the schema
    //    carries. A mention outside the vocabulary is OUT of the population
    //    (a deployment cannot ground what its schema has never seen) —
    //    counted and disclosed, never silently pooled. Each row's
    //    SpanPointer is built once (the primitive's cold path).
    let mut per_type: std::collections::HashMap<String, PopStats> =
        std::collections::HashMap::new();
    let mut choices = PopStats::default();
    let mut all = PopStats::default();
    let mut nd = PopStats::default(); // non-degenerate slice (choices-class only)
    let mut verbatim_floor_abstain = 0u32;
    let mut verbatim_ptr_abstain = 0u32;
    let mut oov_skipped = 0u32;
    let mut misses: Vec<String> = Vec::new();
    let mut nd_total = 0u32;

    for row in &rows {
        let sp = SpanPointer::new(&row.text, SpanConfig::default());
        let row_bytes = row.text.as_bytes();
        for (ty, s, e) in &row.entities {
            let is_choices = CHOICES_CLASS.contains(&ty.as_str());
            let (a, b) = trim_word_bounds(row_bytes, *s, *e);
            if a >= b {
                continue; // nothing left after word-boundary trim (none expected)
            }
            let expected = (a, b);
            let surface = row.text[a as usize..b as usize].to_ascii_lowercase();
            let group = match vocab.of_surface.get(&(ty.clone(), surface.clone())) {
                Some(g) => g,
                None => {
                    if is_choices {
                        oov_skipped += 1;
                    }
                    continue;
                }
            };
            let canonical: &str = &group.canonical;
            let alias_refs: Vec<&str> = group.aliases.iter().map(String::as_str).collect();
            let verbatim = surface == canonical;

            // The rule floor: leftmost literal occurrence of the canonical.
            let floor_answer = floor_find(&row.text, canonical);
            // The gazetteer floors: the SAME (canonical, aliases) inputs the
            // pointer gets, matched literally (verdict round 1's missing arm).
            let gaz = GazetteerAnswers::of(&row.text, canonical, &alias_refs);

            // The pointer: canonical text + the alias table (schema choices).
            let mut out = [SpanScore {
                start: 0,
                end: 0,
                score: 0.0,
            }; TOP_K];
            let n = sp.score_into(
                &SpanQuery {
                    text: canonical,
                    aliases: &alias_refs,
                },
                &mut out,
            );
            let ptr_raw = if n > 0 {
                Some((out[0].start, out[0].end))
            } else {
                None
            };
            let ptr_gate = gate_answer(&out[..n]);
            // The no-alias disclosure arm (empty alias table).
            let mut out_na = [SpanScore {
                start: 0,
                end: 0,
                score: 0.0,
            }; 1];
            let n_na = sp.score_into(&SpanQuery::new(canonical), &mut out_na);
            let ptr_noalias = gate_answer(&out_na[..n_na]);

            // Structural liveness tripwires on verbatim mentions: the
            // canonical bytes ARE in the text, so the floor cannot abstain,
            // and an exact-token window must survive the raw gate.
            if verbatim {
                if floor_answer.is_none() {
                    verbatim_floor_abstain += 1;
                }
                if ptr_raw.is_none() {
                    verbatim_ptr_abstain += 1;
                }
            }

            let in_nd = is_choices && !verbatim;
            if in_nd {
                nd_total += 1;
            }
            let answers = |expected: (u32, u32)| ArmAnswers {
                floor: floor_answer,
                gaz,
                ptr_raw,
                ptr_gate,
                ptr_noalias,
                expected,
            };
            record(&mut all, answers(expected));
            if is_choices {
                record(&mut choices, answers(expected));
            }
            if in_nd {
                record(&mut nd, answers(expected));
            }
            {
                let pt = per_type.entry(ty.clone()).or_default();
                record(pt, answers(expected));
            }

            // Collect a few misses for the record (mixed arms).
            if misses.len() < 8 && (floor_answer != Some(expected) || ptr_gate != Some(expected)) {
                misses.push(format!(
                    "  {} [{ty}] expected {:?} ({}..{}) canon={canonical:?} aliases={:?}\n    floor={:?} ptr_gate={:?} ptr_raw={:?} text={:?}",
                    row.id,
                    &row.text[a as usize..b as usize],
                    a,
                    b,
                    group.aliases,
                    floor_answer.map(|(s, e)| &row.text[s as usize..e as usize]),
                    ptr_gate.map(|(s, e)| &row.text[s as usize..e as usize]),
                    ptr_raw.map(|(s, e)| &row.text[s as usize..e as usize]),
                    row.text,
                ));
            }
        }
    }

    // 5. Structural liveness asserts.
    assert_eq!(
        verbatim_floor_abstain, 0,
        "G5 span: the floor abstained on a verbatim mention — the harness is broken"
    );
    assert_eq!(
        verbatim_ptr_abstain, 0,
        "G5 span: the pointer produced no window above the raw gate on a verbatim mention — the harness or the gate is broken"
    );
    assert!(
        nd_total > 0,
        "G5 span: empty non-degenerate slice — group derivation broken"
    );

    // 5b. Measured ratchets + the GOAT gate (see the constant docs). Both
    //     fire DOWNWARD only; a better scorer re-pins upward from its own
    //     record. The GOAT comparator is the strongest floor variant.
    assert!(
        choices.floor_alias_longest.exact >= FLOOR_ALIAS_LONGEST_EXACT_RATCHET,
        "G5 span ratchet: floor+alias(longest) exact {} < {FLOOR_ALIAS_LONGEST_EXACT_RATCHET} — the gazetteer floor or the grouping regressed",
        choices.floor_alias_longest.exact
    );
    assert!(
        choices.ptr_gate.exact >= PTR_EXACT_RATCHET,
        "G5 span ratchet: pointer(gate) exact {} < {PTR_EXACT_RATCHET} — a span_pointer regression past the G5 record",
        choices.ptr_gate.exact
    );
    assert!(
        choices.ptr_gate.contain >= PTR_CONTAIN_RATCHET,
        "G5 span ratchet: pointer(gate) containment {} < {PTR_CONTAIN_RATCHET}",
        choices.ptr_gate.contain
    );
    assert!(
        nd.ptr_gate.exact >= ND_PTR_EXACT_RATCHET,
        "G5 span ratchet: non-degenerate pointer exact {} < {ND_PTR_EXACT_RATCHET} — the alias-channel mechanism regressed",
        nd.ptr_gate.exact
    );
    assert!(
        per_type["Genre"].ptr_gate.exact >= GENRE_PTR_EXACT_RATCHET,
        "G5 span ratchet: Genre pointer exact {} < {GENRE_PTR_EXACT_RATCHET} — the alias-channel mechanism regressed",
        per_type["Genre"].ptr_gate.exact
    );
    assert!(
        oov_skipped > 0,
        "G5 span: zero OOV skips — the vocabulary-source contract (TRAIN split, not the scored rows) is not in effect"
    );
    // The GOAT verdict is REPORTED, never asserted: verdict round 1 measured
    // the pointer above a floor that was denied the schema (an answer-key
    // alias asymmetry); the fair re-run (TRAIN-derived schema, gazetteer
    // arms) REVERSED it — the negative is the record, and a future scorer
    // that genuinely clears the strongest floor arm re-opens the promotion
    // by re-pinning this relation as an assert.

    // 6. The GLiNER-class trained lane (Plan 628 T0.7's second baseline —
    //    a trained reference, never the GOAT bar). OPT-IN: without its env
    //    the lane SKIPS LOUD (a deferral, never a silent pass). The lane's
    //    script (tools/gliner_span_lane.py) runs GLiNER over the same
    //    fixture texts and writes open-NER predictions:
    //    {"id", "spans": [[type, start, end], ...]} — one line per row id.
    match std::env::var("GLINER_SPAN_PRED") {
        Err(_) => println!(
            "\nGLiNER lane: SKIPPED (env GLINER_SPAN_PRED unset — run tools/gliner_span_lane.py to arm the trained-lane reference)"
        ),
        Ok(pred_path) => {
            let pred_bytes = std::fs::read(&pred_path).unwrap_or_else(|e| {
                panic!("G5 span: GLINER_SPAN_PRED={pred_path} unreadable: {e}")
            });
            let pred_text = std::str::from_utf8(&pred_bytes)
                .unwrap_or_else(|e| panic!("G5 span: GLiNER predictions not UTF-8: {e}"));
            let mut by_id: std::collections::HashMap<String, Vec<(String, u32, u32)>> =
                std::collections::HashMap::new();
            for (i, line) in pred_text.lines().enumerate() {
                let v: Value = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("G5 span: GLiNER line {}: not JSON: {e}", i + 1));
                let id = v
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("G5 span: GLiNER line {}: missing id", i + 1));
                let spans = v
                    .get("spans")
                    .and_then(Value::as_array)
                    .unwrap_or_else(|| panic!("G5 span: GLiNER line {}: missing spans", i + 1))
                    .iter()
                    .map(|s| {
                        let arr = s.as_array().expect("GLiNER span array");
                        (
                            arr[0].as_str().expect("GLiNER span type").to_string(),
                            arr[1].as_u64().expect("GLiNER start") as u32,
                            arr[2].as_u64().expect("GLiNER end") as u32,
                        )
                    })
                    .collect();
                by_id.insert(id.to_string(), spans);
            }
            // Open-NER scoring: expected recall (exact + word-trimmed) and
            // precision over ALL fixture rows the lane answered, micro.
            let (mut exp_total, mut exp_exact, mut exp_contain) = (0u32, 0u32, 0u32);
            let (mut pred_total, mut pred_exact, mut pred_contain) = (0u32, 0u32, 0u32);
            for row in &rows {
                let Some(preds) = by_id.get(row.id.as_str()) else {
                    continue; // the lane may answer a subset; disclosed
                };
                let row_bytes = row.text.as_bytes();
                let norm = |(s, e): (u32, u32)| trim_word_bounds(row_bytes, s, e);
                let pred_norm: Vec<(String, (u32, u32))> = preds
                    .iter()
                    .map(|(t, s, e)| (t.clone(), norm((*s, *e))))
                    .collect();
                pred_total += pred_norm.len() as u32;
                for (ty, s, e) in &row.entities {
                    let (a, b) = trim_word_bounds(row_bytes, *s, *e);
                    if a >= b {
                        continue;
                    }
                    exp_total += 1;
                    if pred_norm
                        .iter()
                        .any(|(pt, (ps, pe))| pt == ty && ps == &a && pe == &b)
                    {
                        exp_exact += 1;
                    }
                    if pred_norm
                        .iter()
                        .any(|(pt, (ps, pe))| pt == ty && ps <= &a && pe >= &b)
                    {
                        exp_contain += 1;
                    }
                }
                for (pt, pb) in &pred_norm {
                    let hit_exact = row.entities.iter().any(|(ty, s, e)| {
                        ty == pt && {
                            let (a, b) = trim_word_bounds(row_bytes, *s, *e);
                            (a, b) == *pb
                        }
                    });
                    let hit_contain = row.entities.iter().any(|(ty, s, e)| {
                        ty == pt && {
                            let (a, b) = trim_word_bounds(row_bytes, *s, *e);
                            pb.0 <= a && pb.1 >= b
                        }
                    });
                    if hit_exact {
                        pred_exact += 1;
                    }
                    if hit_contain {
                        pred_contain += 1;
                    }
                }
            }
            let f1 = |p: u32, r: u32| {
                // F1% from counts: 200*p*r / (p*E + r*P) with P = pred_total,
                // E = exp_total (0 denominator -> n/a).
                let den =
                    f64::from(p) * f64::from(exp_total) + f64::from(r) * f64::from(pred_total);
                if den == 0.0 {
                    "n/a".to_string()
                } else {
                    format!("{:.1}%", 200.0 * f64::from(p) * f64::from(r) / den)
                }
            };
            println!();
            println!(
                "GLiNER lane ({pred_path}): {} rows answered, {} spans predicted",
                by_id.len(),
                pred_total
            );
            println!(
                "  expected recall: exact {} ({}) containment {} ({}) over {exp_total} entities",
                exp_exact,
                pct(exp_exact, exp_total),
                exp_contain,
                pct(exp_contain, exp_total),
            );
            println!(
                "  precision: exact {} ({}) containment {} ({}) — micro F1 exact {}",
                pred_exact,
                pct(pred_exact, pred_total),
                pred_contain,
                pct(pred_contain, pred_total),
                f1(pred_exact, exp_exact),
            );
        }
    }

    // 7. Determinism: the scoring pass again (fresh SpanPointers, fresh
    //    accumulators); the choices-class checksum must repeat exactly.
    let cs1 = choices_checksum(&rows, &vocab);
    let cs2 = choices_checksum(&rows, &vocab);
    assert_eq!(
        cs1, cs2,
        "G5 span: nondeterministic scoring — same input, different aggregates"
    );
    assert_ne!(cs1, 0, "G5 span: zero checksum — the vacuity gate fired");

    report(
        PopReports {
            choices: &choices,
            all: &all,
            nd: &nd,
        },
        &per_type,
        entity_total,
        oov_skipped,
        cs1,
        &misses,
    );
}

/// Gate posture: rank-0 must clear the nothing-fired boundary strictly.
fn gate_answer(out: &[SpanScore]) -> Option<(u32, u32)> {
    if !out.is_empty() && out[0].score > NOTHING_FIRED + GATE_EPS {
        Some((out[0].start, out[0].end))
    } else {
        None
    }
}

/// One population's arms, bundled for the per-entity record pass.
struct ArmAnswers {
    floor: Answer,
    gaz: GazetteerAnswers,
    ptr_raw: Answer,
    ptr_gate: Answer,
    ptr_noalias: Answer,
    expected: (u32, u32),
}

fn record(p: &mut PopStats, a: ArmAnswers) {
    p.floor.record(a.floor, a.expected);
    p.floor_alias_first.record(a.gaz.first, a.expected);
    p.floor_alias_longest.record(a.gaz.longest, a.expected);
    p.ptr_raw.record(a.ptr_raw, a.expected);
    p.ptr_gate.record(a.ptr_gate, a.expected);
    p.ptr_noalias.record(a.ptr_noalias, a.expected);
}

/// One span answer (byte offsets into the row text).
type Answer = Option<(u32, u32)>;

/// The gazetteer answers: the same (canonical, aliases) inputs the pointer
/// gets, matched literally. Returns `(first_in_order, longest_match)`.
#[derive(Clone, Copy, Default)]
struct GazetteerAnswers {
    first: Answer,
    longest: Answer,
}

impl GazetteerAnswers {
    /// `first` walks the list in order and takes the first surface with an
    /// occurrence; `longest` takes the longest occurring surface (ties by
    /// list order), each answer's span the leftmost occurrence.
    fn of(text: &str, canonical: &str, aliases: &[&str]) -> Self {
        let mut out = GazetteerAnswers::default();
        let mut best_len = 0usize;
        for surface in std::iter::once(canonical).chain(aliases.iter().copied()) {
            if let Some((s, e)) = floor_find(text, surface) {
                if out.first.is_none() {
                    out.first = Some((s, e));
                }
                if e as usize - s as usize > best_len {
                    best_len = e as usize - s as usize;
                    out.longest = Some((s, e));
                }
            }
        }
        out
    }
}

/// The rule floor: leftmost case-identical occurrence of `needle`, or None.
/// (The fixture is lowercase; the needle comes from the same corpus.)
fn floor_find(text: &str, needle: &str) -> Option<(u32, u32)> {
    text.match_indices(needle)
        .next()
        .map(|(s, part)| (s as u32, (s + part.len()) as u32))
}

/// Per-arm counters over one population.
#[derive(Default)]
struct ArmStats {
    entities: u32,
    exact: u32,
    contain: u32,
    abstain: u32,
}

impl ArmStats {
    fn record(&mut self, answer: Option<(u32, u32)>, expected: (u32, u32)) {
        self.entities += 1;
        match answer {
            None => self.abstain += 1,
            Some((s, e)) => {
                if s == expected.0 && e == expected.1 {
                    self.exact += 1;
                }
                if s <= expected.0 && e >= expected.1 {
                    self.contain += 1;
                }
            }
        }
    }
}

/// One population's arms. `floor` = the literal rule on the canonical only;
/// `floor_alias_first` = the gazetteer that tries canonical then each alias
/// in list order (first surface with a literal occurrence wins);
/// `floor_alias_longest` = the standard longest-match gazetteer (the longest
/// occurring surface wins, ties by list order, then leftmost) — the
/// strongest disclosed floor variant and the GOAT bar's comparator.
#[derive(Default)]
struct PopStats {
    floor: ArmStats,
    floor_alias_first: ArmStats,
    floor_alias_longest: ArmStats,
    ptr_raw: ArmStats,
    ptr_gate: ArmStats,
    ptr_noalias: ArmStats,
}

/// FNV-1a checksum over the choices-class floor + gazetteer + gated-pointer
/// counters (the determinism gate compares two full passes).
fn choices_checksum(rows: &[Row], vocab: &Vocabulary) -> u64 {
    let mut c = PopStats::default();
    for row in rows {
        let sp = SpanPointer::new(&row.text, SpanConfig::default());
        let row_bytes = row.text.as_bytes();
        for (ty, s, e) in &row.entities {
            if !CHOICES_CLASS.contains(&ty.as_str()) {
                continue;
            }
            let (a, b) = trim_word_bounds(row_bytes, *s, *e);
            if a >= b {
                continue;
            }
            let expected = (a, b);
            let surface = row.text[a as usize..b as usize].to_ascii_lowercase();
            let group = match vocab.of_surface.get(&(ty.clone(), surface)) {
                Some(g) => g,
                None => continue,
            };
            let alias_refs: Vec<&str> = group.aliases.iter().map(String::as_str).collect();
            let floor_answer = floor_find(&row.text, &group.canonical);
            let gaz = GazetteerAnswers::of(&row.text, &group.canonical, &alias_refs);
            let mut out = [SpanScore {
                start: 0,
                end: 0,
                score: 0.0,
            }; TOP_K];
            let n = sp.score_into(
                &SpanQuery {
                    text: &group.canonical,
                    aliases: &alias_refs,
                },
                &mut out,
            );
            c.floor.record(floor_answer, expected);
            c.floor_alias_first.record(gaz.first, expected);
            c.floor_alias_longest.record(gaz.longest, expected);
            c.ptr_gate.record(gate_answer(&out[..n]), expected);
        }
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for a in [
        &c.floor,
        &c.floor_alias_first,
        &c.floor_alias_longest,
        &c.ptr_gate,
    ] {
        for v in [a.entities, a.exact, a.contain, a.abstain] {
            for byte in v.to_le_bytes() {
                h ^= u64::from(byte);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

fn pct(n: u32, d: u32) -> String {
    if d == 0 {
        "  n/a".to_string()
    } else {
        format!("{:5.1}%", 100.0 * f64::from(n) / f64::from(d))
    }
}

fn print_arm(label: &str, a: &ArmStats) {
    println!(
        "{:<34}{:>8}  {:>14}{:>14}{:>8}",
        label,
        a.entities,
        format!("{} {}", a.exact, pct(a.exact, a.entities)),
        format!("{} {}", a.contain, pct(a.contain, a.entities)),
        a.abstain,
    );
}

fn print_pop(name: &str, p: &PopStats) {
    println!("-- {name}");
    print_arm("  floor (canonical literal)", &p.floor);
    print_arm("  floor+alias (first-in-order)", &p.floor_alias_first);
    print_arm("  floor+alias (longest-match)", &p.floor_alias_longest);
    print_arm("  pointer(gate)", &p.ptr_gate);
    print_arm("  pointer(raw)", &p.ptr_raw);
    print_arm("  pointer(no-alias gate)", &p.ptr_noalias);
}

/// One population's report row bundle (the report's 8th-argument smell is
/// the point: one struct, not eight positionals).
struct PopReports<'a> {
    choices: &'a PopStats,
    all: &'a PopStats,
    nd: &'a PopStats,
}

fn report(
    pops: PopReports<'_>,
    per_type: &std::collections::HashMap<String, PopStats>,
    entity_total: usize,
    oov_skipped: u32,
    checksum: u64,
    misses: &[String],
) {
    let PopReports { choices, all, nd } = pops;
    println!("G5 span baselines — span_pointer vs the rule class (Plan 628 T0.7, re-run)");
    println!(
        "  fixture: {FIXTURE_REL} (BLAKE3-pinned, {} rows / {entity_total} entities)",
        ROWS_EXPECTED
    );
    println!(
        "  schema vocabulary: {VOCAB_REL} (TRAIN split — never the scored rows; OOV mentions skipped: {oov_skipped})"
    );
    println!();
    println!(
        "{:<34}{:>8}  {:>14}{:>14}{:>8}",
        "arm / population", "ents", "exact", "contain (⊇)", "abstain"
    );
    print_pop("CHOICES-CLASS (primary)", choices);
    print_pop("ALL types", all);
    print_pop("non-degenerate slice", nd);
    println!();
    let mut tys: Vec<&String> = per_type.keys().collect();
    tys.sort();
    for ty in tys {
        let p = &per_type[ty];
        let label = format!("  {ty}");
        let cell = |n: u32, d: u32| format!("{} {}", n, pct(n, d));
        println!(
            "{label:<30}{:>8}  {:>14}{:>14}{:>8}   <- {ty} floor / ptr(gate): {}",
            p.floor.entities,
            cell(p.floor.exact, p.floor.entities),
            cell(p.floor.contain, p.floor.entities),
            p.floor.abstain,
            cell(p.ptr_gate.exact, p.ptr_gate.entities),
        );
    }
    println!();
    println!(
        "  floor (canon)     = leftmost literal occurrence of the canonical (no aliases — the bare rule)"
    );
    println!(
        "  floor+alias       = the gazetteer on the SAME (canonical, aliases) inputs the pointer gets:"
    );
    println!(
        "      first-in-order = canonical then aliases in list order, first literal hit wins;"
    );
    println!(
        "      longest-match  = the longest occurring surface wins (ties by list order) — the strongest"
    );
    println!("                     rule-class variant and the GOAT bar's comparator");
    println!(
        "  pointer(gate) = rank-0 window, defaults, must clear sigmoid(0)=0.5 strictly (production posture)"
    );
    println!("  pointer(raw)  = rank-0 window above the default 0.5 gate, no strict clearance");
    println!(
        "  no-alias      = the pointer with an empty alias table (isolates the alias channel)"
    );
    println!(
        "  exact = answer == expected (word-trimmed); contain = answer is a SUPERSET of expected"
    );
    println!();
    println!(
        "GOAT bar (pre-registered): pointer(gate) exact > floor+alias(longest) exact on CHOICES-CLASS"
    );
    {
        let f = &choices.floor_alias_longest;
        let g = &choices.ptr_gate;
        let verdict = if g.exact > f.exact {
            "PASS — the pointer beats the strongest rule-class arm given the same schema inputs"
        } else {
            "FAIL — the pointer does NOT beat the floor+alias arm (demote span_pointer; record the negative)"
        };
        println!(
            "  floor+alias(longest) exact {} ({}) vs pointer(gate) exact {} ({}) -> {verdict}",
            f.exact,
            pct(f.exact, f.entities),
            g.exact,
            pct(g.exact, g.entities),
        );
    }
    println!("  determinism checksum: {checksum:#018x}");
    println!();
    if !misses.is_empty() {
        println!("sample misses (up to 8, mixed arms):");
        for m in misses {
            println!("{m}");
        }
    }
}
