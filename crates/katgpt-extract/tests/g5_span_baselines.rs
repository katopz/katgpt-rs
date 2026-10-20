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
//! reds). The GOAT relation is ASSERTED as a standing gate for the ROUND-3
//! hybrid (`pointer+snap` > floor+alias(longest) on choices-class — the
//! promotion basis, so a scorer change that voids it reds here and forces
//! re-adjudication); the RAW pointer's round-2 failure stays REPORTED, never
//! asserted (the full tables live in
//! `.benchmarks/950_f0_t07_span_baselines.md`).
//!
//! # Round 3 — the boundary-snap arm (`ptr_snap`, PRE-REGISTERED BEFORE
//! SCORING — the round-2 lesson)
//!
//! The round-2 verdict's re-open candidate: the lane's problem is exact
//! boundaries, not entity finding (containment 84.1% vs the gazetteer's
//! 78.7%; exact 74.3% vs 77.8%). `ptr_snap` = the pointer LOCALIZES, the
//! schema vocabulary RESOLVES the boundaries:
//!
//! 1. The pointer(gate) rank-0 window (the production posture) is the
//!    candidate region.
//! 2. Among the group's (canonical, aliases) — the SAME inputs every arm
//!    gets — only surface occurrences FULLY INSIDE the window qualify
//!    (`ws <= os && oe <= we`): what the pointer fully saw.
//! 3. The longest qualifying occurrence wins; ties by list order (canonical
//!    first, then aliases in list order — strictly-greater replacement); per
//!    surface the leftmost occurrence. The gazetteer's exact longest-match
//!    law, constrained to the window — the constraint is the only
//!    difference from the `floor+alias (longest-match)` comparator.
//! 4. No qualifying occurrence → the pointer window ITSELF (the hybrid
//!    stays pointer-derived). Deliberately NOT the global gazetteer: a
//!    pointer∨gazetteer ensemble would trivially floor the floor; the
//!    scientific question is whether the pointer's localization + the
//!    schema's boundary vocabulary beats the rule class alone.
//!
//! The GOAT bar is UNCHANGED (pointer-gate exact > floor+alias(longest)
//! exact, choices-class); `ptr_snap` is reported against the same bar. A
//! `ptr_snap` that clears it is the re-open basis the round-2 record
//! pre-registered.
//!
//! # Round 4 — the verdict's fix list (REVISE, session ea31bb50): the
//! control arm + McNemar, PRE-REGISTERED BEFORE SCORING
//!
//! Round 3's promotion was REVISED. The verdict's arithmetic: of the snap
//! hybrid's 2311 exact hits, 2294 came via an in-window surface while the
//! floor scored 2304 overall — the window CONSTRAINT nets ≈ −10 vs
//! unconstrained longest-match, and the 17 fallback exacts (the pointer
//! covering gazetteer abstentions) are the entire +7 win. The hybrid's
//! honest description is therefore "gazetteer, minus window-constraint
//! losses, plus the pointer covering gazetteer misses" — NOT
//! "pointer-derived". Three additions, all pre-registered here:
//!
//! 1. **`floor+ptr_fallback` (the missing control, report-only):** the
//!    floor+alias(longest-match) answer; when it would ABSTAIN, the
//!    pointer(gate) window. The pointer is used ONLY where the floor has no
//!    answer — the exact arm the verdict named (and NOT the forbidden
//!    ensemble: no floor override of the pointer, ever). If this arm beats
//!    the snap hybrid, the promotion basis moves to IT.
//! 2. **McNemar (snap vs floor, choices-class):** the discordant counts
//!    (b = snap-hit/floor-miss, c = snap-miss/floor-hit) + the two-sided
//!    exact binomial p — the verdict's minimum for a +7 claim.
//! 3. **Held-out confirmation** (a fixture that played no part in designing
//!    the arm) — a separate lane, same TRAIN-derived vocabulary protocol;
//!    recorded in the bench when it lands.
//!
//! Re-promotion rule (pre-registered from the verdict): only after (a) the
//! control arm is measured and does NOT beat the snap, (b) the McNemar
//! stands, (c) a held-out fixture confirms snap ≥ floor, and (d) release-
//! profile G2 with a quiet-box preflight line — then re-land the promotion
//! (or promote the control instead if IT wins, recorded why).
//!
//! # Round 5 — the verdict's AGREE conditions (pre-promotion gates)
//!
//! The round-4 execution was AGREED (session 37660519) with four conditions
//! before any promotion lands (alongside release G2):
//!
//! 1. **Production composite API** — DONE: `SpanPointer::floor_first_span`
//!    with `longest_match` and `GroundedSpan`/`SpanAnswerSource` (the
//!    control pattern shipped, zero-alloc G4-gated, unit-tested — no
//!    consumer rebuilds it by hand).
//! 2. **Assert the control's relations** — DONE: control >= floor is a
//!    construction identity (asserted per entity); control >= snap is
//!    asserted wherever both ratchets are pinned (a flip reds and forces
//!    the promotion-basis re-adjudication).
//! 3. **McNemar on the control** — DONE: control vs snap printed (the
//!    informative comparison); control vs floor counted and labeled
//!    one-sided-by-construction, never quoted as a test.
//! 4. **Stale prose** — DONE: the crate README + Cargo.toml comment name
//!    the one real remaining gate (release G2 + quiet-box preflight) and
//!    the composite API.
//!
//! The table prints to stdout (captured on green — use `--nocapture`).

use katgpt_extract::span_pointer::{SpanConfig, SpanPointer, SpanQuery, SpanScore};
use serde_json::Value;

/// One pinned fixture: the corpus + its TRAIN-derived schema vocabulary +
/// every census/ratchet pin measured from the record. A drifted byte reds at
/// the pin gate; a regressed arm reds at its ratchet. Same pipeline, two
/// domains (Plan 628 T0.7; the held-out-confirmation law from Bench 950
/// round 4: a fixture that played no part in designing the arms).
struct Fixture {
    label: &'static str,
    fixture_rel: &'static str,
    blake3_pin: &'static str,
    rows_expected: usize,
    entities_expected: usize,
    /// Per-type entity counts over kept rows (`type` = the upstream tag).
    type_expected: &'static [(&'static str, usize)],
    /// Types whose fixture-wide surface unique% < 95 (an enumerable
    /// vocabulary exists) — the pre-registered scoring population.
    choices_class: &'static [&'static str],
    vocab_rel: &'static str,
    vocab_blake3_pin: &'static str,
    /// Census pins for the TRAIN-derived vocabulary (surfaces per type).
    vocab_type_expected: &'static [(&'static str, usize)],
    /// The OOV skip census pinned EXACT when measured (the round-2
    /// hardening: catches schema drift in size, not just source). None =
    /// not yet pinned — disclosed, never asserted.
    oov_expected: Option<u32>,
    /// Measured ratchets (downward-only); 0 = unmeasured — the arm is
    /// report-only until its record pins it.
    ratchets: Ratchets,
    /// Whether the round-3 GOAT relation (pointer+snap > floor+alias(longest)
    /// on choices-class exact) is ASSERTED for this fixture. True only where
    /// measured true; a fixture where the relation fails records the
    /// negative and keeps the assert off pending re-adjudication.
    assert_goat: bool,
    /// Per-type exact ratchets — measured pins (type, min exact).
    per_type_ratchets: &'static [(&'static str, u32)],
    /// Whether the env-armed GLiNER predictions hook scores this fixture
    /// (the shipped predictions file is movie-domain).
    gliner_lane: bool,
}

/// Measured ratchets for one fixture (downward-only; 0 = unmeasured).
#[derive(Clone, Copy)]
struct Ratchets {
    floor_alias_longest_exact: u32,
    ptr_exact: u32,
    ptr_contain: u32,
    nd_ptr_exact: u32,
    snap_exact: u32,
    nd_snap_exact: u32,
    floor_ptr_fallback_exact: u32,
    nd_floor_ptr_fallback_exact: u32,
}

/// MIT-Movie trivia10k13 test split (Plan 628 T0.7; provenance:
/// `tests/fixtures/external/PROVENANCE_MIT_MOVIE.md`, Liu et al. ASRU 2013).
/// Ratchets: rounds 2-4 of the Bench-950 record (debug profile, 2026-10-20).
const MOVIE: Fixture = Fixture {
    label: "MIT-Movie trivia10k13 test",
    fixture_rel: "tests/fixtures/external/mit_movie_trivia_test.jsonl",
    blake3_pin: "77a36d097177560e49378e9a3b7ea4496d43a06ae2ca43667f680bf0cf65c56d",
    rows_expected: 1562,
    entities_expected: 5093,
    type_expected: &[
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
    ],
    choices_class: &[
        "Actor",
        "Award",
        "Character_Name",
        "Director",
        "Genre",
        "Opinion",
        "Relationship",
        "Soundtrack",
        "Year",
    ],
    vocab_rel: "tests/fixtures/external/mit_movie_trivia_train_vocab.jsonl",
    vocab_blake3_pin: "3a7ba3868950deafa6f8f7eb4a578f72a954f661b009e85800169f6d3d1b9b4a",
    vocab_type_expected: &[
        ("Actor", 1436),
        ("Award", 163),
        ("Character_Name", 664),
        ("Director", 532),
        ("Genre", 908),
        ("Opinion", 292),
        ("Relationship", 346),
        ("Soundtrack", 47),
        ("Year", 143),
    ],
    oov_expected: Some(740),
    ratchets: Ratchets {
        floor_alias_longest_exact: 2304,
        ptr_exact: 2200,
        ptr_contain: 2488,
        nd_ptr_exact: 537,
        snap_exact: 2311,
        nd_snap_exact: 575,
        floor_ptr_fallback_exact: 2321,
        nd_floor_ptr_fallback_exact: 581,
    },
    assert_goat: true,
    per_type_ratchets: &[("Genre", 277)],
    gliner_lane: true,
};

/// MIT-Restaurant test split — the ROUND-4 HELD-OUT confirmation fixture (a
/// domain that played no part in designing the arms; same SLS downloads
/// tree, provenance: `tests/fixtures/external/PROVENANCE_MIT_RESTAURANT.md`).
/// All 8 types are choices-class (train unique% <= 54.1%). First-run record
/// (2026-10-20, debug): the round-4 picture CONFIRMS cross-domain — snap
/// clears the floor with real significance (McNemar b=58 c=29, p=0.002), and
/// the control (floor+ptr_fallback) still wins (1493 vs 1464): the pointer's
/// value concentrates where the gazetteer abstains. Ratchets pinned from
/// that record; the GOAT relation asserted (measured true).
const RESTAURANT: Fixture = Fixture {
    label: "MIT-Restaurant test",
    fixture_rel: "tests/fixtures/external/mit_restaurant_test.jsonl",
    blake3_pin: "427c3a51d464ee6dc8788d0c24a3de695835284ebd2dad4ff9d51a5ad7746763",
    rows_expected: 1517,
    entities_expected: 3151,
    type_expected: &[
        ("Amenity", 533),
        ("Cuisine", 532),
        ("Dish", 288),
        ("Hours", 212),
        ("Location", 812),
        ("Price", 171),
        ("Rating", 201),
        ("Restaurant_Name", 402),
    ],
    choices_class: &[
        "Amenity",
        "Cuisine",
        "Dish",
        "Hours",
        "Location",
        "Price",
        "Rating",
        "Restaurant_Name",
    ],
    vocab_rel: "tests/fixtures/external/mit_restaurant_train_vocab.jsonl",
    vocab_blake3_pin: "517554fb436ef2e12a71e60a6855e0ced8543ca16d30a050840f5cc793c5411e",
    vocab_type_expected: &[
        ("Amenity", 1126),
        ("Cuisine", 475),
        ("Dish", 662),
        ("Hours", 361),
        ("Location", 1126),
        ("Price", 182),
        ("Rating", 228),
        ("Restaurant_Name", 1028),
    ],
    oov_expected: Some(813),
    ratchets: Ratchets {
        floor_alias_longest_exact: 1435,
        ptr_exact: 1195,
        ptr_contain: 1613,
        nd_ptr_exact: 580,
        snap_exact: 1464,
        nd_snap_exact: 747,
        floor_ptr_fallback_exact: 1493,
        nd_floor_ptr_fallback_exact: 752,
    },
    assert_goat: true,
    per_type_ratchets: &[],
    gliner_lane: false,
};

/// Pointer top-k buffer (the any-containment disclosure depth).
const TOP_K: usize = 5;
/// Alias surfaces carried by one query (the schema-choices cap).
const ALIAS_CAP: usize = 8;
/// The strict nothing-boundary slack for the gate posture (f32 rounding
/// around `sigmoid(0) = 0.5`).
const GATE_EPS: f32 = 1e-3;
/// The nothing-fired score (the documented span_pointer boundary).
const NOTHING_FIRED: f32 = 0.5;

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
fn load_vocabulary(path: &std::path::Path, fx: &Fixture) -> Vocabulary {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("G5 span: cannot read vocab {}: {e}", path.display()));
    let got = blake3::hash(&bytes).to_hex();
    assert_eq!(
        got.as_str(),
        fx.vocab_blake3_pin,
        "G5 span vocab drifted: BLAKE3 {} != pinned {} — re-pin via the provenance doc first",
        got.as_str(),
        fx.vocab_blake3_pin
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
        for (ty, n) in fx.vocab_type_expected {
            let n = *n;
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

/// The two #[test] drivers (one per fixture; the SAME pipeline, pre-registered
/// arms, and population rules — only the pins differ).
#[test]
fn g5_span_baselines_movie() {
    run_suite(&MOVIE);
}

#[test]
fn g5_span_baselines_restaurant() {
    run_suite(&RESTAURANT);
}

fn run_suite(fx: &Fixture) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(fx.fixture_rel);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("G5 span: cannot read fixture {}: {e}", path.display()));

    let got = blake3::hash(&bytes).to_hex();
    assert_eq!(
        got.as_str(),
        fx.blake3_pin,
        "G5 span fixture drifted: BLAKE3 {} != pinned {} — re-pin via the provenance doc first",
        got.as_str(),
        fx.blake3_pin
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
        fx.rows_expected,
        "G5 span fixture holds {} rows, expected {}",
        rows.len(),
        fx.rows_expected
    );
    let entity_total: usize = rows.iter().map(|r| r.entities.len()).sum();
    assert_eq!(
        entity_total, fx.entities_expected,
        "G5 span fixture holds {entity_total} entities, expected {}",
        fx.entities_expected
    );
    {
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for r in &rows {
            for (ty, _, _) in &r.entities {
                *counts.entry(ty.as_str()).or_default() += 1;
            }
        }
        for (ty, n) in fx.type_expected {
            let n = *n;
            assert_eq!(
                counts.get(ty).copied().unwrap_or(0),
                n,
                "G5 span census: type {ty} holds {}, expected {n}",
                counts.get(ty).copied().unwrap_or(0)
            );
        }
    }

    // 3. The schema vocabulary (TRAIN-derived — never the scored rows).
    let vocab_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(fx.vocab_rel);
    let vocab = load_vocabulary(&vocab_path, fx);

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
    // Round-3 mechanism attribution: of the snap arm's exact hits, how many
    // came from a qualifying in-window surface (the snap mechanism) vs a
    // window fallback (no surface inside — the window itself was exact).
    let mut snap_from_hit = 0u32;
    let mut snap_from_fallback = 0u32;
    // Round-4 McNemar discordants (choices-class): snap vs floor-longest.
    let mut snap_vs_floor = Discordants { b: 0, c: 0 };
    // Round-5 (verdict AGREE conditions): the control arm's own McNemar —
    // control vs snap is the informative comparison; control vs floor is
    // one-sided BY CONSTRUCTION (the control's answer IS the floor's answer
    // wherever the floor answers) — counted, printed, never read as a test.
    let mut control_vs_snap = Discordants { b: 0, c: 0 };

    for row in &rows {
        let sp = SpanPointer::new(&row.text, SpanConfig::default());
        let row_bytes = row.text.as_bytes();
        for (ty, s, e) in &row.entities {
            let is_choices = fx.choices_class.contains(&ty.as_str());
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
            // Round 3: the boundary-snap arm — the pointer localizes, the
            // schema vocabulary resolves the boundaries (pre-registered in
            // the module docs above: longest fully-in-window occurrence,
            // gazetteer tie law, pointer-window fallback).
            let ptr_snap = ptr_gate
                .map(|w| snap_longest_in_window(&row.text, w, canonical, &alias_refs).unwrap_or(w));
            // Round 4 control: the floor's answer; when it would ABSTAIN,
            // the pointer(gate) window. The pointer ONLY where the floor
            // has no answer (pre-registered round 4).
            let floor_ptr_fallback = gaz.longest.or(ptr_gate);
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
            if is_choices && ptr_snap == Some(expected) {
                // ptr_snap == Some(expected) implies ptr_gate is Some (the
                // snap composes over it), so the fallback question is well
                // posed: did a qualifying surface produce the exact span?
                let via_surface = ptr_gate
                    .and_then(|w| snap_longest_in_window(&row.text, w, canonical, &alias_refs))
                    .is_some();
                if via_surface {
                    snap_from_hit += 1;
                } else {
                    snap_from_fallback += 1;
                }
            }
            if is_choices {
                let snap_hit = ptr_snap == Some(expected);
                let floor_hit = gaz.longest == Some(expected);
                let control_hit = floor_ptr_fallback == Some(expected);
                if snap_hit && !floor_hit {
                    snap_vs_floor.b += 1;
                } else if !snap_hit && floor_hit {
                    snap_vs_floor.c += 1;
                }
                if control_hit && !snap_hit {
                    control_vs_snap.b += 1;
                } else if !control_hit && snap_hit {
                    control_vs_snap.c += 1;
                }
                // Construction invariant: the control never loses to the
                // floor (it IS the floor where the floor answers).
                assert!(
                    control_hit || !floor_hit,
                    "G5 span ({}): control missed where the floor hit — the control composition is broken",
                    fx.label
                );
            }
            let answers = |expected: (u32, u32)| ArmAnswers {
                floor: floor_answer,
                gaz,
                ptr_raw,
                ptr_gate,
                ptr_snap,
                floor_ptr_fallback,
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
            if misses.len() < 8 && (floor_answer != Some(expected) || ptr_snap != Some(expected)) {
                misses.push(format!(
                    "  {} [{ty}] expected {:?} ({}..{}) canon={canonical:?} aliases={:?}\n    floor={:?} ptr_gate={:?} ptr_snap={:?} text={:?}",
                    row.id,
                    &row.text[a as usize..b as usize],
                    a,
                    b,
                    group.aliases,
                    floor_answer.map(|(s, e)| &row.text[s as usize..e as usize]),
                    ptr_gate.map(|(s, e)| &row.text[s as usize..e as usize]),
                    ptr_snap.map(|(s, e)| &row.text[s as usize..e as usize]),
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

    // 5b. Measured ratchets + the GOAT gate (see the Fixture docs). Ratchets
    //     fire DOWNWARD only (0 = unmeasured — report-only until pinned); a
    //     better scorer re-pins upward from its own record. The GOAT bar's
    //     comparator is the strongest floor variant.
    let rat = fx.ratchets;
    let ratchet = |actual: u32, floor: u32, what: &str| {
        if floor > 0 {
            assert!(
                actual >= floor,
                "G5 span ratchet ({}): {what} {actual} < {floor} — a scorer or harness regression past the record",
                fx.label
            );
        }
    };
    ratchet(
        choices.floor_alias_longest.exact,
        rat.floor_alias_longest_exact,
        "floor+alias(longest) exact",
    );
    ratchet(choices.ptr_gate.exact, rat.ptr_exact, "pointer(gate) exact");
    ratchet(
        choices.ptr_gate.contain,
        rat.ptr_contain,
        "pointer(gate) containment",
    );
    ratchet(
        nd.ptr_gate.exact,
        rat.nd_ptr_exact,
        "non-degenerate pointer exact",
    );
    ratchet(choices.ptr_snap.exact, rat.snap_exact, "pointer+snap exact");
    ratchet(
        nd.ptr_snap.exact,
        rat.nd_snap_exact,
        "non-degenerate pointer+snap exact",
    );
    ratchet(
        choices.floor_ptr_fallback.exact,
        rat.floor_ptr_fallback_exact,
        "floor+ptr_fallback exact",
    );
    ratchet(
        nd.floor_ptr_fallback.exact,
        rat.nd_floor_ptr_fallback_exact,
        "non-degenerate floor+ptr_fallback exact",
    );
    for (ty, floor) in fx.per_type_ratchets {
        ratchet(
            per_type[*ty].ptr_gate.exact,
            *floor,
            &format!("{ty} pointer exact"),
        );
    }
    // The control arm's GOAT relations (the verdict's AGREE condition 2):
    // control >= floor is a CONSTRUCTION identity (asserted per-entity in
    // the loop above); control >= snap is a MEASURED relation — asserted
    // wherever both records are pinned, so a scorer change that flips it
    // reds and forces the promotion-basis re-adjudication.
    if rat.floor_ptr_fallback_exact > 0 && rat.snap_exact > 0 {
        assert!(
            choices.floor_ptr_fallback.exact >= choices.ptr_snap.exact,
            "G5 span ({}): control exact {} < snap exact {} — the promotion basis (the control arm) is void; re-adjudicate",
            fx.label,
            choices.floor_ptr_fallback.exact,
            choices.ptr_snap.exact
        );
    }
    // The round-3 GOAT relation, ASSERTED where the fixture's record measured
    // it true (the promotion basis — a scorer or fixture change that voids it
    // reds here and forces re-adjudication). The raw pointer alone still does
    // NOT clear the floor — that verdict stays reported, never asserted.
    if fx.assert_goat {
        assert!(
            choices.ptr_snap.exact > choices.floor_alias_longest.exact,
            "G5 span GOAT ({}): pointer+snap exact {} no longer clears floor+alias(longest) {} — the promotion basis is void; re-adjudicate",
            fx.label,
            choices.ptr_snap.exact,
            choices.floor_alias_longest.exact
        );
    }
    if let Some(oov_pin) = fx.oov_expected {
        assert_eq!(
            oov_skipped, oov_pin,
            "G5 span ({}): OOV skip count {} != pinned {oov_pin} — the schema vocabulary drifted (size or source); re-pin from the re-measured record",
            fx.label, oov_skipped
        );
    }
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
    //    Movie-domain only: the shipped predictions file is movie's; the
    //    restaurant fixture is always SKIPPED-loud here.
    match std::env::var("GLINER_SPAN_PRED") {
        Err(_) => println!(
            "\nGLiNER lane ({}): SKIPPED (env GLINER_SPAN_PRED unset — run tools/gliner_span_lane.py to arm the trained-lane reference)",
            fx.label
        ),
        Ok(pred_path) if fx.gliner_lane => {
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
        Ok(pred_path) => println!(
            "\nGLiNER lane ({}): SKIPPED — GLINER_SPAN_PRED={pred_path} is set, but this fixture's lane is not enabled (the shipped predictions are movie-domain)",
            fx.label
        ),
    }

    // 7. Determinism: the scoring pass again (fresh SpanPointers, fresh
    //    accumulators); the choices-class checksum must repeat exactly.
    let cs1 = choices_checksum(&rows, fx, &vocab);
    let cs2 = choices_checksum(&rows, fx, &vocab);
    assert_eq!(
        cs1, cs2,
        "G5 span: nondeterministic scoring — same input, different aggregates"
    );
    assert_ne!(cs1, 0, "G5 span: zero checksum — the vacuity gate fired");

    report(
        PopReports {
            fx,
            choices: &choices,
            all: &all,
            nd: &nd,
        },
        &per_type,
        entity_total,
        oov_skipped,
        cs1,
        &misses,
        RoundFourReports {
            snap_from_hit,
            snap_from_fallback,
            snap_vs_floor: Discordants {
                b: snap_vs_floor.b,
                c: snap_vs_floor.c,
            },
            control_vs_snap: Discordants {
                b: control_vs_snap.b,
                c: control_vs_snap.c,
            },
        },
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
    ptr_snap: Answer,
    floor_ptr_fallback: Answer,
    ptr_noalias: Answer,
    expected: (u32, u32),
}

fn record(p: &mut PopStats, a: ArmAnswers) {
    p.floor.record(a.floor, a.expected);
    p.floor_alias_first.record(a.gaz.first, a.expected);
    p.floor_alias_longest.record(a.gaz.longest, a.expected);
    p.ptr_raw.record(a.ptr_raw, a.expected);
    p.ptr_gate.record(a.ptr_gate, a.expected);
    p.ptr_snap.record(a.ptr_snap, a.expected);
    p.floor_ptr_fallback
        .record(a.floor_ptr_fallback, a.expected);
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

/// `floor_find` constrained to `[ws, we)` — leftmost occurrence fully
/// inside the window. Both bounds are token-aligned (pointer windows and
/// word-trimmed spans land on word bytes), so the slice is at char
/// boundaries.
fn floor_find_in(text: &str, needle: &str, ws: u32, we: u32) -> Option<(u32, u32)> {
    if needle.is_empty() || ws >= we {
        return None;
    }
    let hay = &text[ws as usize..we as usize];
    hay.match_indices(needle)
        .next()
        .map(|(s, part)| (ws + s as u32, ws + (s + part.len()) as u32))
}

/// The round-3 snap: among (canonical, aliases), the LONGEST occurrence
/// fully inside `window` wins; ties by list order (strictly-greater
/// replacement, `GazetteerAnswers::of`'s law); per surface the leftmost
/// occurrence. None when no surface occurs inside the window — the caller
/// falls back to the pointer window itself.
fn snap_longest_in_window(
    text: &str,
    window: (u32, u32),
    canonical: &str,
    aliases: &[&str],
) -> Option<(u32, u32)> {
    let mut best: Option<(u32, u32)> = None;
    let mut best_len = 0usize;
    for surface in std::iter::once(canonical).chain(aliases.iter().copied()) {
        if let Some((s, e)) = floor_find_in(text, surface, window.0, window.1)
            && e as usize - s as usize > best_len
        {
            best_len = e as usize - s as usize;
            best = Some((s, e));
        }
    }
    best
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
    ptr_snap: ArmStats,
    floor_ptr_fallback: ArmStats,
    ptr_noalias: ArmStats,
}

/// FNV-1a checksum over the choices-class floor + gazetteer + gated-pointer
/// counters (the determinism gate compares two full passes).
fn choices_checksum(rows: &[Row], fx: &Fixture, vocab: &Vocabulary) -> u64 {
    let mut c = PopStats::default();
    for row in rows {
        let sp = SpanPointer::new(&row.text, SpanConfig::default());
        let row_bytes = row.text.as_bytes();
        for (ty, s, e) in &row.entities {
            if !fx.choices_class.contains(&ty.as_str()) {
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
            let ptr_gate = gate_answer(&out[..n]);
            let ptr_snap = ptr_gate.map(|w| {
                snap_longest_in_window(&row.text, w, &group.canonical, &alias_refs).unwrap_or(w)
            });
            // Round 4 control: the floor's answer; when it would ABSTAIN,
            // the pointer(gate) window. The pointer ONLY where the floor
            // has no answer (pre-registered round 4).
            // has no answer (pre-registered round 4).
            let floor_ptr_fallback = gaz.longest.or(ptr_gate);
            c.floor.record(floor_answer, expected);
            c.floor_alias_first.record(gaz.first, expected);
            c.floor_alias_longest.record(gaz.longest, expected);
            c.ptr_gate.record(ptr_gate, expected);
            c.ptr_snap.record(ptr_snap, expected);
            c.floor_ptr_fallback.record(floor_ptr_fallback, expected);
        }
    }
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for a in [
        &c.floor,
        &c.floor_alias_first,
        &c.floor_alias_longest,
        &c.ptr_gate,
        &c.ptr_snap,
        &c.floor_ptr_fallback,
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
    print_arm("  pointer+snap (round 3)", &p.ptr_snap);
    print_arm("  floor+ptr_fallback (r4)", &p.floor_ptr_fallback);
    print_arm("  pointer(raw)", &p.ptr_raw);
    print_arm("  pointer(no-alias gate)", &p.ptr_noalias);
}

/// One population's report row bundle + the fixture context (the report's
/// positional-argument smell is the point: structs, not positionals).
struct PopReports<'a> {
    fx: &'a Fixture,
    choices: &'a PopStats,
    all: &'a PopStats,
    nd: &'a PopStats,
}

/// The snap arm's exact-hit mechanism split (disclosure: where the hybrid's
/// score comes from — resolved surfaces vs pointer-window fallbacks), the
/// McNemar discordants (snap vs floor), and the round-4 control-arm reads.
/// One bundle — the report takes structs, never eight positionals.
struct RoundFourReports {
    snap_from_hit: u32,
    snap_from_fallback: u32,
    snap_vs_floor: Discordants,
    control_vs_snap: Discordants,
}

/// McNemar discordant pairs (one comparison), + the two-sided exact
/// binomial p over b+c at 0.5.
struct Discordants {
    b: u32,
    c: u32,
}

impl Discordants {
    fn p(&self) -> f64 {
        let n = self.b + self.c;
        if n == 0 {
            return 1.0;
        }
        let m = self.b.min(self.c);
        // Two-sided exact: 2 * P(X <= m), X ~ Bin(n, 0.5). Iterative pmf —
        // f64 is exact enough at n ~ 10^3 (report to 3 decimals).
        let mut pmf = (0.5f64).powi(n as i32);
        let mut cum = pmf;
        for k in 1..=m {
            pmf *= (n - k + 1) as f64 / k as f64;
            cum += pmf;
        }
        (2.0 * cum).min(1.0)
    }
}

fn report(
    pops: PopReports<'_>,
    per_type: &std::collections::HashMap<String, PopStats>,
    entity_total: usize,
    oov_skipped: u32,
    checksum: u64,
    misses: &[String],
    round_four: RoundFourReports,
) {
    let PopReports {
        fx,
        choices,
        all,
        nd,
    } = pops;
    println!("G5 span baselines — {} ({})", fx.fixture_rel, fx.label);
    println!(
        "  fixture: BLAKE3-pinned, {} rows / {entity_total} entities",
        fx.rows_expected
    );
    println!(
        "  schema vocabulary: {} (TRAIN split — never the scored rows; OOV mentions skipped: {oov_skipped})",
        fx.vocab_rel
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
        "  pointer+snap  = round 3: the pointer localizes, the schema vocabulary resolves the"
    );
    println!(
        "                  boundaries — longest fully-in-window (canonical|alias) occurrence,"
    );
    println!(
        "                  gazetteer tie law, pointer-window fallback (pre-registered round 3)"
    );
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
        let s = &choices.ptr_snap;
        let verdict = if s.exact > f.exact {
            "PASS — the snap hybrid CLEARS the strongest floor arm: the pre-registered re-open basis"
        } else {
            "FAIL — the snap hybrid does not clear the floor+alias arm (the negative stands)"
        };
        println!(
            "  floor+alias(longest) exact {} ({}) vs pointer+snap exact {} ({}) -> {verdict}",
            f.exact,
            pct(f.exact, f.entities),
            s.exact,
            pct(s.exact, s.entities),
        );
        println!(
            "    snap mechanism: {} exact via in-window surface, {} via window fallback (no surface inside)",
            round_four.snap_from_hit, round_four.snap_from_fallback,
        );
        println!(
            "    McNemar snap vs floor: b={}(snap-hit/floor-miss) c={}(snap-miss/floor-hit) — exact two-sided p={:.3}",
            round_four.snap_vs_floor.b,
            round_four.snap_vs_floor.c,
            round_four.snap_vs_floor.p(),
        );
        println!(
            "    McNemar control vs snap: b={}(control-hit/snap-miss) c={}(control-miss/snap-hit) — exact two-sided p={:.3} (the informative comparison)",
            round_four.control_vs_snap.b,
            round_four.control_vs_snap.c,
            round_four.control_vs_snap.p(),
        );
        println!(
            "    control vs floor: one-sided BY CONSTRUCTION (the control IS the floor where the floor answers) — not an informative test, never quoted as one"
        );
        let fb = &choices.floor_ptr_fallback;
        println!(
            "  floor+ptr_fallback (round 4 control) exact {} ({}) contain {} ({}) abstain {} — the pointer ONLY where the floor abstains",
            fb.exact,
            pct(fb.exact, fb.entities),
            fb.contain,
            pct(fb.contain, fb.entities),
            fb.abstain,
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
