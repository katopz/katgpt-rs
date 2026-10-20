# Bench 950 — F0 T0.7 span baselines: span_pointer vs the rule floor (GOAT PASS)

**Status:** DONE — the span lane's G5 landed; `span_pointer` PROMOTED to default per the GOAT rule.

## What ran

Plan 628 T0.7 — the G5 span-baseline arms Research 627 §8 named as the sole remaining
re-score condition (c):

- **Fixture:** `crates/katgpt-extract/tests/fixtures/external/mit_movie_trivia_test.jsonl`
  — the MIT-Movie trivia10k13 TEST split converted mechanically (1,562 rows / 5,093
  entities / 479,734 B; BLAKE3 `77a36d09…65c56d`; provenance +
  converter rules: `PROVENANCE_MIT_MOVIE.md`). External corpus, nothing authored for the
  test; the Ritter-class domain-ontology NER corpus the plan named.
- **Harness:** `tests/g5_span_baselines.rs` (feature `span_pointer`; required-features row
  landed same-commit). Protocol pre-registered in the module docs BEFORE any arm ran:
  schema-vocabulary grounding (the F0 production shape), value groups derived mechanically
  from the fixture (per-type connected components over token-contiguous-substring;
  canonical = most-frequent surface, aliases = other members, cap 8), expected spans
  word-boundary-trimmed uniformly, identical queries for every arm.
- **Arms:** the Duckling/Recognizers-class RULE FLOOR (leftmost literal occurrence of the
  canonical; no aliases — the rule class does literal matching; the alias table is exactly
  the pointer's schema surface, disclosed asymmetry) vs `span_pointer` at
  `SpanConfig::default()` (raw = rank-0 above the 0.5 gate; gate = the production posture,
  rank-0 must clear `sigmoid(0) = 0.5` strictly; no-alias = empty alias table, isolating
  the alias channel). GLiNER-class trained lane shipped as
  `tools/gliner_span_lane.py` + an env-gated harness hook (`GLINER_SPAN_PRED`) that skips
  LOUD when unarmed (trained reference, never the GOAT bar).

## Results (debug profile, M3 Max, 4 sibling agent sessions active — accuracy metrics,
## not latency; the determinism checksum `0x21a83a3683003ce7` repeated exactly ×2)

| arm (population) | ents | exact | contain (⊇) | abstain |
|---|---|---|---|---|
| **CHOICES-CLASS (primary) · floor** | 3700 | 2658 (**71.8%**) | 2662 (71.9%) | 778 |
| **CHOICES-CLASS · pointer(gate)** | 3700 | **3055 (82.6%)** | **3258 (88.1%)** | 1 |
| CHOICES-CLASS · pointer(raw) | 3700 | 3055 (82.6%) | 3258 (88.1%) | 0 |
| CHOICES-CLASS · pointer(no-alias gate) | 3700 | 2712 (73.3%) | 2725 (73.6%) | 349 |
| ALL types · floor | 5093 | 3186 (62.6%) | 3190 (62.6%) | 1191 |
| ALL types · pointer(gate) | 5093 | 3417 (67.1%) | 3653 (71.7%) | 1 |
| non-degenerate · floor | 1032 | 0 (0.0%) | 4 (0.4%) | 778 |
| non-degenerate · pointer(gate) | 1032 | 509 (49.3%) | 621 (60.2%) | 1 |
| non-degenerate · pointer(no-alias) | 1032 | 56 (5.4%) | 69 (6.7%) | 349 |

Per-type (floor exact → pointer-gate exact): Director 96.8→99.0, Year 97.7→99.1,
Character_Name 86.5→92.7, Actor 83.4→90.3, Opinion 77.6→85.2, **Genre 20.7→53.9**,
Award 29.2→47.7, Relationship 42.7→43.9, Soundtrack 87.5→87.5. Out-of-population
disclosure: Quote 97.3→78.4 (36 verbatim mentions; the pointer's rank-0 over-extends to a
longer window on some), **Plot 35.8→22.2** (free-text descriptions; the transitive
Plot component collapses onto a stopword canonical "and" — both arms degenerate there,
the pointer loses MORE because its rank-0 lands on alias-noise windows; Plot is excluded
from the scoring population precisely because no enumerable vocabulary exists).

## GOAT verdict

**PASS — pointer(gate) 82.6% > floor 71.8% exact on choices-class (+10.8 pts), and the
mechanism is the alias channel, measured, not narrated:** without aliases the pointer is
a wash against the floor (73.3% vs 71.8% — BM25/embedding ranking alone buys ~1.5 pts);
with the schema's alias tables it clears the floor by 10.8. On the non-degenerate slice
(canonical ≠ the mention's own surface — where a literal rule CANNOT produce the extended
span by construction, hence the floor's 0.0%) the alias channel is worth +43.9 pts
(49.3% vs 5.4%). This is the product story the primitive was built for: literal matching
is what rules give you; alias/dictionary tables per field schema are what `span_pointer`
adds.

**`span_pointer` PROMOTED to default** (`crates/katgpt-extract/Cargo.toml`): G1 = this
floor win (asserted as a STANDING gate in the test — a future regression below the floor
reds and voids the promotion basis), G2/G4 = Benches 948's green rows (composite p99
402–409 µs vs the 10 ms bar; zero-alloc span paths), G3 = the crate's suite green at
default/no-default/all postures, modelless gain by construction. Same promotion rule that
promoted `extraction_estimate` at T0.6.

## Pinned in the harness (downward-only ratchets from THIS record)

`FLOOR_EXACT_RATCHET` 2658 · `PTR_EXACT_RATCHET` 3055 · `PTR_CONTAIN_RATCHET` 3258 ·
`ND_PTR_EXACT_RATCHET` 509 · `GENRE_PTR_EXACT_RATCHET` 407 (the mechanism tripwires) ·
the GOAT relation assert (pointer > floor on choices-class) · verbatim liveness (floor
never abstains by identity; pointer always answers) · pin gate + full census.

## GLiNER lane — FIRST ARMED RUN (reference row, never the gate)

`tools/gliner_span_lane.py` @ `urchade/gliner_multi-v2.1`, CPU (M3), threshold 0.5,
batch 8, ~90 s wall over all 1,562 rows (3,290 spans predicted). Harness hook
(`GLINER_SPAN_PRED`) scores it against the same fixture, ALL-types micro:

- expected recall: exact 2261 (**44.4%**), containment 2320 (45.6%) over 5,093 entities
- precision: exact 68.7%, containment 70.5% — **micro F1 exact 53.9%**

Reading (honest): this is the OPEN task (type labels only, no query, no schema vocabulary)
while span_pointer's 82.6% is the GROUNDING task (schema canonical + aliases given) — the
two numbers are company, not a matchup. What the pair says: a ~200M trained zero-shot NER
model lands at 44% recall / 54% F1 on this domain ontology, while the weightless
schema-vocabulary grounding of the SAME texts reaches 82.6% when the field schema carries
the vocabulary — and the schema is exactly what a deployment has. The lane ships opt-in
(venv + model, `GLINER_SPAN_PRED`); no CI lane runs it.

## Honest caveats

- The floor's disclosed asymmetry is the POINT of the benchmark (aliases are the
  primitive's schema surface) but it IS an asymmetry: a rule engine WITH a gazetteer of
  the same aliases (longest-match) would close part of the gap. That engine is a
  dictionary build, not a rule class — recorded here so nobody quotes the +10.8 as
  "pointer vs all rules".
- The value groups' transitive closure chains surfaces through shared tokens (the Genre
  component is one 254-surface cluster via "comedy"). That makes queries HARDER (generic
  canonicals), never easier — it cannot manufacture a pointer win; disclosed as the
  grouping heuristic's shape.
- DEBUG-profile accuracy metrics (counting, not timing); the ratchets pin this record.
- The GLiNER lane's first armed run is recorded above (CPU posture, reference-only);
  venv notes: gliner on transformers 5.x needs `sentencepiece` + `protobuf` alongside
  (the mdeberta spm.model otherwise falls into a tiktoken parse and dies).
- Box state: 4 sibling agent sessions active (accuracy work, load-immune; the G2 latency
  numbers stay owned by Benches 948 + the quiet-box re-run debt).
