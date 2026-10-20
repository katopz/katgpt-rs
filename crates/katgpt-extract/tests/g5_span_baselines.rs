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
//! - **Value groups:** per type, connected components over "surface A's
//!   token sequence is a contiguous subsequence of surface B's" (token =
//!   maximal ASCII-alphanumeric run — the span_pointer tokenizer's class).
//!   Canonical = the most frequent surface in the fixture (ties:
//!   lexicographically smallest); aliases = the other members by descending
//!   frequency, capped at [`ALIAS_CAP`].
//! - **Expected span:** the annotated span trimmed to word boundaries (the
//!   same word-byte class both arms' answers land on). Applied uniformly.
//! - **Rule floor:** the Duckling/Recognizers-class literal rule — leftmost
//!   case-identical occurrence of the canonical surface in the row text;
//!   abstains when none. It gets NO aliases: literal matching is exactly
//!   what the rule class does; the alias table is the pointer's schema
//!   surface (the delta under test). Disclosed asymmetry.
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
//! surface unique% < 95: an enumerable vocabulary exists), the
//! **all-types** population, the **non-degenerate slice** (canonical != the
//! mention's own surface — where a literal rule and a ranked pointer can
//! differ), and per-type rows. The **GOAT bar** (pre-registered):
//! pointer-gate exact > floor exact on the choices-class aggregate.
//!
//! # What is asserted
//!
//! Pin gate + census (rows/entities/per-type counts — any fixture drift
//! refuses to score), lowercase sanity, structural liveness (on verbatim
//! mentions — canonical bytes literally present — the floor NEVER abstains
//! by identity and the pointer ALWAYS answers: both are tripwires against a
//! broken harness), determinism (the scoring pass runs twice; the aggregate
//! checksums must be identical and nonzero), and the measured ratchets
//! (downward-only, pinned from the first measured record — a regression
//! past them reds). The GOAT verdict itself is REPORTED, not asserted: the
//! verdict lives in `.benchmarks/950_f0_t07_span_baselines.md` and drives
//! the span lane's promotion question, so a fixture-independent scorer
//! change cannot silently flip it either way.
//!
//! The table prints to stdout (captured on green — use `--nocapture`).

use katgpt_extract::span_pointer::{SpanConfig, SpanPointer, SpanQuery, SpanScore};
use serde_json::Value;

/// The pinned corpus, relative to the crate root (manifest-anchored, so the
/// test runs from any working directory).
const FIXTURE_REL: &str = "tests/fixtures/external/mit_movie_trivia_test.jsonl";

/// BLAKE3 pin over the fixture bytes (see PROVENANCE_MIT_MOVIE.md).
const BLAKE3_PIN: &str = "77a36d097177560e49378e9a3b7ea4496d43a06ae2ca43667f680bf0cf65c56d";

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

/// Pointer top-k buffer (the any-containment disclosure depth).
const TOP_K: usize = 5;
/// Alias surfaces carried by one query (the schema-choices cap).
const ALIAS_CAP: usize = 8;
/// The strict nothing-boundary slack for the gate posture (f32 rounding
/// around `sigmoid(0) = 0.5`).
const GATE_EPS: f32 = 1e-3;
/// The nothing-fired score (the documented span_pointer boundary).
const NOTHING_FIRED: f32 = 0.5;

/// Measured ratchets (downward-only) — pinned from the FIRST measured
/// record of the T0.7 landing (2026-10-20, debug profile, checksum
/// 0x21a83a3683003ce7; full tables in
/// `.benchmarks/950_f0_t07_span_baselines.md`). A ratchet never reds an
/// improvement; a scorer or harness regression past it fails here:
/// - `FLOOR_EXACT_RATCHET` — the choices-class rule floor must keep finding
///   its literal hits (a broken grouping/trim silently deflates it);
/// - `PTR_EXACT_RATCHET` / `PTR_CONTAIN_RATCHET` — the pointer's G5 win is
///   the promotion basis (span_pointer is DEFAULT-ON since this record);
/// - `PTR_NOALIAS_EXACT` — the no-alias disclosure arm (reported, not the
///   gate — attribution only, so it is pinned loosely);
/// - `ND_PTR_EXACT_RATCHET` / `GENRE_PTR_EXACT_RATCHET` — the mechanism
///   tripwires: the alias channel's win concentrates exactly there.
const FLOOR_EXACT_RATCHET: u32 = 2658;
const PTR_EXACT_RATCHET: u32 = 3055;
const PTR_CONTAIN_RATCHET: u32 = 3258;
const ND_PTR_EXACT_RATCHET: u32 = 509;
const GENRE_PTR_EXACT_RATCHET: u32 = 407;

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

/// The derived schema vocabulary: for every `(type, surface)` in the
/// fixture, the group it belongs to.
struct Vocabulary {
    of_surface: std::collections::HashMap<(String, String), Group>,
}

/// Build the per-type value groups from the fixture's entity surfaces.
///
/// Deterministic: components over the token-contiguous-subsequence
/// relation; canonical = (max frequency, then lexicographically smallest);
/// aliases = the other members by (descending frequency, ascending
/// lexical), capped at [`ALIAS_CAP`]. Every member surface maps to the same
/// group (the schema's choices entry, whichever surface the mention uses).
fn build_vocabulary(rows: &[Row]) -> Vocabulary {
    use std::collections::{HashMap, HashSet};

    // Surface frequency per type (entity occurrences, not rows).
    let mut freq: HashMap<(String, String), u32> = HashMap::new();
    for row in rows {
        let bytes = row.text.as_bytes();
        for (ty, s, e) in &row.entities {
            let (a, b) = trim_word_bounds(bytes, *s, *e);
            if a < b {
                let surface = row.text[a as usize..b as usize].to_ascii_lowercase();
                *freq.entry((ty.clone(), surface)).or_insert(0) += 1;
            }
        }
    }

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

    // 3. Value groups (the schema vocabulary, fixture-derived).
    let vocab = build_vocabulary(&rows);

    // 4. Score every choices-class entity. Each row's SpanPointer is built
    //    once (the primitive's cold path) and queried per entity.
    let mut per_type: std::collections::HashMap<String, PopStats> =
        std::collections::HashMap::new();
    let mut choices = PopStats::default();
    let mut all = PopStats::default();
    let mut nd = PopStats::default(); // non-degenerate slice (choices-class only)
    let mut verbatim_floor_abstain = 0u32;
    let mut verbatim_ptr_abstain = 0u32;
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
            let group = vocab
                .of_surface
                .get(&(ty.clone(), surface.clone()))
                .unwrap_or_else(|| {
                    panic!(
                        "G5 span: row {} entity {ty} surface {surface:?} has no vocabulary group",
                        row.id
                    )
                });
            let canonical: &str = &group.canonical;
            let alias_refs: Vec<&str> = group.aliases.iter().map(String::as_str).collect();
            let verbatim = surface == canonical;

            // The rule floor: leftmost literal occurrence of the canonical.
            let floor_answer = floor_find(&row.text, canonical);

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
            record(
                &mut all,
                floor_answer,
                ptr_raw,
                ptr_gate,
                ptr_noalias,
                expected,
            );
            if is_choices {
                record(
                    &mut choices,
                    floor_answer,
                    ptr_raw,
                    ptr_gate,
                    ptr_noalias,
                    expected,
                );
            }
            if in_nd {
                record(
                    &mut nd,
                    floor_answer,
                    ptr_raw,
                    ptr_gate,
                    ptr_noalias,
                    expected,
                );
            }
            {
                let pt = per_type.entry(ty.clone()).or_default();
                record(pt, floor_answer, ptr_raw, ptr_gate, ptr_noalias, expected);
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
    //     record.
    assert!(
        choices.floor.exact >= FLOOR_EXACT_RATCHET,
        "G5 span ratchet: floor exact {} < {FLOOR_EXACT_RATCHET} — the rule floor or the grouping regressed",
        choices.floor.exact
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
        choices.ptr_gate.exact > choices.floor.exact,
        "G5 span GOAT gate: pointer(gate) exact {} <= floor exact {} — span_pointer no longer beats the rule floor; its default-on promotion basis is void and must be re-adjudicated",
        choices.ptr_gate.exact,
        choices.floor.exact
    );

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

    report(&choices, &all, &nd, &per_type, entity_total, cs1, &misses);
}

/// Gate posture: rank-0 must clear the nothing-fired boundary strictly.
fn gate_answer(out: &[SpanScore]) -> Option<(u32, u32)> {
    if !out.is_empty() && out[0].score > NOTHING_FIRED + GATE_EPS {
        Some((out[0].start, out[0].end))
    } else {
        None
    }
}

fn record(
    p: &mut PopStats,
    floor: Option<(u32, u32)>,
    ptr_raw: Option<(u32, u32)>,
    ptr_gate: Option<(u32, u32)>,
    ptr_noalias: Option<(u32, u32)>,
    expected: (u32, u32),
) {
    p.floor.record(floor, expected);
    p.ptr_raw.record(ptr_raw, expected);
    p.ptr_gate.record(ptr_gate, expected);
    p.ptr_noalias.record(ptr_noalias, expected);
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

/// One population's four arms.
#[derive(Default)]
struct PopStats {
    floor: ArmStats,
    ptr_raw: ArmStats,
    ptr_gate: ArmStats,
    ptr_noalias: ArmStats,
}

/// FNV-1a checksum over the choices-class floor + gated-pointer counters
/// (the determinism gate compares two full passes).
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
            let group = vocab
                .of_surface
                .get(&(ty.clone(), surface))
                .expect("vocabulary group");
            let alias_refs: Vec<&str> = group.aliases.iter().map(String::as_str).collect();
            let floor_answer = floor_find(&row.text, &group.canonical);
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
            c.ptr_gate.record(gate_answer(&out[..n]), expected);
        }
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for a in [&c.floor, &c.ptr_gate] {
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
        "{:<30}{:>8}  {:>14}{:>14}{:>8}",
        label,
        a.entities,
        format!("{} {}", a.exact, pct(a.exact, a.entities)),
        format!("{} {}", a.contain, pct(a.contain, a.entities)),
        a.abstain,
    );
}

fn print_pop(name: &str, p: &PopStats) {
    println!("-- {name}");
    print_arm("  floor", &p.floor);
    print_arm("  pointer(gate)", &p.ptr_gate);
    print_arm("  pointer(raw)", &p.ptr_raw);
    print_arm("  pointer(no-alias gate)", &p.ptr_noalias);
}

fn report(
    choices: &PopStats,
    all: &PopStats,
    nd: &PopStats,
    per_type: &std::collections::HashMap<String, PopStats>,
    entity_total: usize,
    checksum: u64,
    misses: &[String],
) {
    println!("G5 span baselines — span_pointer vs rule floor (Plan 628 T0.7)");
    println!(
        "  fixture: {FIXTURE_REL} (BLAKE3-pinned, {} rows / {entity_total} entities)",
        ROWS_EXPECTED
    );
    println!();
    println!(
        "{:<30}{:>8}  {:>14}{:>14}{:>8}",
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
        "  floor         = leftmost literal occurrence of the canonical (no aliases — the rule class)"
    );
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
    println!("GOAT bar (pre-registered): pointer(gate) exact > floor exact on CHOICES-CLASS");
    {
        let f = &choices.floor;
        let g = &choices.ptr_gate;
        let verdict = if g.exact > f.exact {
            "PASS — the pointer beats the rule floor"
        } else {
            "FAIL — the pointer does NOT beat the rule floor (stays opt-in; record the negative)"
        };
        println!(
            "  floor exact {} ({}) vs pointer(gate) exact {} ({}) -> {verdict}",
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
