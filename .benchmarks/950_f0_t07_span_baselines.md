# Bench 950 — F0 T0.7 span baselines: span_pointer vs the rule class (GOAT FAILED on the fair re-run)

**Status:** DONE — two rounds. Round 1 measured a GOAT PASS on an asymmetric comparison;
the verdict-round REVISE forced the fair re-run; **the fair result is an honest NEGATIVE —
the longest-match gazetteer beats the pointer on the primary exact metric — and
`span_pointer` was DEMOTED back to opt-in** per the plan's own rule.

## Round 1 (superseded — kept for the record)

Fixture `mit_movie_trivia_test.jsonl` (MIT-Movie trivia10k13 TEST split, 1,562 rows /
5,093 entities / 479,734 B, BLAKE3 `77a36d09…65c56d`; ASRU-2013 Liu/Pasupat/Wang/Cyphers/
Glass; provenance + converter rules in `PROVENANCE_MIT_MOVIE.md`). Harness
`tests/g5_span_baselines.rs` (required-features row same-commit). First result: pointer
82.6% vs floor 71.8% exact on the choices-class population → GOAT PASS declared,
`span_pointer` promoted to default.

**That result was inflated by two defects the verdict round caught (REVISE):**

1. **The alias table was derived from the scored test rows** (`build_vocabulary` over the
   fixture itself) — a mention's own surface usually rode its group's alias list (cap 8),
   so the pointer's alias channel was close to an answer key.
2. **The floor was denied the schema** (canonical only, no aliases) while the pointer got
   aliases — the comparison measured "an oracle alias table beats no alias table", not
   "span_pointer beats the rule class". The honest pointer-only edge was +1.5 pts
   (no-alias 73.3% vs floor 71.8%).

## Round 2 — the fair protocol (pre-registered in the harness module docs before scoring)

1. **Schema vocabulary from the TRAIN split** — `mit_movie_trivia_train_vocab.jsonl`
   (4,531 unique surfaces / 9 choices-class types / 103,550 B, BLAKE3
   `3a7ba386…3d1b9b4a`; surfaces + train-occurrence counts, converter rules in the
   provenance doc). The scored test rows never feed it. A scored mention whose
   (type, surface) the schema lacks is OUT of the population (a deployment cannot ground
   what its schema never saw): **740 skipped, 2,960 scored** (80.0% coverage).
2. **The gazetteer arms get the SAME (canonical, aliases) inputs as the pointer:**
   `first-in-order` (canonical then aliases in list order, first literal hit wins) and
   `longest-match` (the longest occurring surface wins, ties by list order — the standard
   gazetteer rule, the GOAT bar's comparator). Both reported, neither cherry-picked.
3. **GOAT bar (re-registered):** pointer(gate) exact > floor+alias(longest) exact on the
   choices-class aggregate.

## Round 2 results (debug profile, M3 Max, accuracy metrics — not latency;
## determinism checksum `0x51b0ec82f1bb2768` repeated exactly ×2)

| arm (CHOICES-CLASS primary, 2,960 ents) | exact | contain (⊇) | abstain |
|---|---|---|---|
| floor (canonical literal) | 1778 (60.1%) | 1780 (60.1%) | 1003 |
| floor+alias (first-in-order) | 2260 (76.4%) | 2262 (76.4%) | 313 |
| **floor+alias (longest-match)** | **2304 (77.8%)** | 2329 (78.7%) | 313 |
| pointer(gate) | 2200 (74.3%) | **2488 (84.1%)** | 1 |
| pointer(raw) | 2200 (74.3%) | 2488 (84.1%) | 0 |
| pointer(no-alias gate) | 1832 (61.9%) | 1838 (62.1%) | 504 |

Non-degenerate slice (canonical ≠ the mention's own surface, 1,178 ents): floor 0.0% (by
construction) · floor+alias(first) 40.9% · **floor+alias(longest) 47.9%** · pointer(gate)
45.6% (containment **61.3%** vs the gazetteer's 48.4%) · pointer(no-alias) 4.6%.

Per-type (floor-longest exact → pointer-gate exact): Director 93.4→97.6, Year 96.7→97.9,
Character_Name 72.7→86.7, Actor 55.6→71.5, Opinion 70.7→82.0, Genre 11.9→44.6,
Relationship 43.8→44.9, Award 15.8→34.2, Soundtrack 100→100.

## GOAT verdict (the pre-registered bar)

**FAILED — pointer(gate) 2200 (74.3%) <= floor+alias(longest) 2304 (77.8%) on the
choices-class exact metric.** The rule class, given the same schema inputs and a standard
longest-match gazetteer rule, beats the pointer's ranked windows by 3.5 pts. The verdict
round's bet ("it closes most or all of the gap") was correct and then some — it flipped
the sign.

**`span_pointer` DEMOTED back to opt-in** (`crates/katgpt-extract/Cargo.toml`): the
promotion's G1 basis was the round-1 asymmetry; the fair arm voids it. Per the plan's own
rule — "span_pointer earns GOAT only by beating the rule floor" — it does not, so it does
not promote. Re-open condition: a scorer that clears floor+alias(longest) on choices-class
exact under THIS pinned protocol (same fixture, same TRAIN schema, same metric), at which
point the standing ratchets re-pin upward and the promotion re-lands.

## What the pointer still wins (disclosed, not the gate)

- **Containment:** 84.1% vs the gazetteer's 78.7% — the pointer's windows over-cover the
  answer more often than they exactly hit; on the non-degenerate slice the containment
  gap is +12.9 pts (61.3% vs 48.4%). For a downstream consumer that can shrink a
  containing window, the pointer's errors are a different (milder) class than the
  gazetteer's.
- **Abstention honesty:** the pointer answers 2,959/2,960 with a scored window or nothing
  (1 gated abstain); the gazetteer abstains 313 (surfaces that never occur literally —
  case/punct variants the windows still catch).
- **The alias mechanism is real but not exclusive:** aliases lift the pointer +12.4 pts
  (61.9→74.3) and the gazetteer +16.3-17.7 pts — a literal matcher exploits the same
  schema better on exact spans.

## GLiNER lane — FIRST ARMED RUN (reference row, never the gate)

`tools/gliner_span_lane.py` @ `urchade/gliner_multi-v2.1`, CPU (M3), threshold 0.5,
batch 8, ~90 s wall over all 1,562 rows (3,290 spans predicted). Harness hook
(`GLINER_SPAN_PRED`) scores it against the same fixture, ALL-types micro:

- expected recall: exact 2261 (**44.4%**), containment 2320 (45.6%) over 5,093 entities
- precision: exact 68.7%, containment 70.5% — **micro F1 exact 53.9%**

Reading (honest): this is the OPEN task (type labels only, no query, no schema
vocabulary) while the grounding arms pose the GROUNDING task (schema canonical + aliases
given) — the two numbers are company, not a matchup. What the pair says: a ~200M trained
zero-shot NER model lands at 44% recall / 54% F1 on this domain ontology, while the
weightless schema-vocabulary grounding of the SAME texts reaches 74–78% when the field
schema carries the vocabulary. The lane ships opt-in (venv + model, `GLINER_SPAN_PRED`);
no CI lane runs it.

## Pinned in the harness (downward-only ratchets from the ROUND-2 record)

`FLOOR_ALIAS_LONGEST_EXACT_RATCHET` 2304 · `PTR_EXACT_RATCHET` 2200 ·
`PTR_CONTAIN_RATCHET` 2488 · `ND_PTR_EXACT_RATCHET` 537 · `GENRE_PTR_EXACT_RATCHET` 277 ·
vocab pin + census (4,531 surfaces / per-type counts) · OOV-skip > 0 (the vocabulary-
source contract is live) · verbatim liveness (floor never abstains by identity; pointer
always answers) · double-run determinism.

## Honest caveats

- Round 1's numbers (82.6/71.8) are INFLATED and superseded; they stay in git history and
  in this record marked as such — nobody should quote them.
- The gazetteer's longest-match rule is the STRONGEST simple rule over the same inputs; a
  weaker rule (first-in-order) also beats the pointer (76.4 vs 74.3), so the negative
  does not hinge on the rule choice.
- The TRAIN schema limits every arm to mentions the schema carries (80% of choices-class
  mentions); the OOV 20% is unmeasurable by any schema-driven arm, disclosed.
- The transitive grouping chains surfaces through shared tokens (Genre's cluster); it
  shapes ALL arms' inputs identically.
- DEBUG-profile counting metrics (not timing); box state: sibling agent sessions active —
  accuracy work, load-immune; the G2 latency numbers stay owned by Benches 948.
- Verdict provenance: round 1 landed as GOAT PASS + default promotion (commit `41e6d3419`);
  the REVISE verdict round (claude-sub-agent-verdict, session 6c9ef478) named both
  defects; this re-run is the addressed verdict, landed in the SAME working session.
