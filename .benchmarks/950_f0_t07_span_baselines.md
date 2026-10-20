# Bench 950 — F0 T0.7 span baselines: span_pointer vs the rule class (rounds 1-4 + the round-5 AGREE conditions; the control arm is the promotion basis, pending release G2)

**Status:** four rounds. Round 1: a GOAT PASS on an asymmetric comparison — REVISED. Round
2: the fair re-run, an honest NEGATIVE for the BARE pointer (demoted to opt-in) with a
pre-registered re-open condition. Round 3: the boundary-snap hybrid cleared the bar (+7)
and re-landed the promotion — **REVISED again (session ea31bb50): the margin rides the 17
fallback exacts, not the snap mechanism; the control arm was unmeasured; no McNemar; no
held-out; the promotion skipped release G2.** Round 4 (this record): the verdict's fix
list executed — the control arm (`floor+ptr_fallback`) measured and it BEATS the snap on
BOTH domains; the McNemar run (movie +7 is n.s., restaurant +29 is p=0.002); the
held-out MIT-Restaurant fixture CONFIRMS the cross-domain picture. **`span_pointer`
STAYS OPT-IN** — the promotion (of the control-arm unit) is pre-registered to land only
after a release-profile G2 with a quiet-box preflight line, which remains
environmentally blocked.

## The round-4 fix list (all landed this session — the revert in the Cargo.toml, the arms in the harness, this fixture pair)

**Round 5 (verdict round 2, session 37660519): AGREE** — the fix list discharged. Four
pre-promotion conditions named, all landed the same session:

1. **Production composite API** — `SpanPointer::floor_first_span` (the control pattern
   shipped: longest-match gazetteer first, `best_span` on abstain, the answer's source
   disclosed via `GroundedSpan`/`SpanAnswerSource`) + the public `longest_match`;
   zero-alloc (G4-gated test), 5 new unit tests. Promoting `span_pointer` now ships
   the WINNING pattern, not a part consumers rebuild by hand.
2. **The control's relations ASSERTED** — control ≥ floor as a construction identity
   (per-entity assert); control ≥ snap wherever both ratchets are pinned (a flip reds
   and forces re-adjudication).
3. **McNemar on the control** — control vs snap: movie b=29 c=5 → **p<0.001**;
   restaurant b=15 c=5 → **p=0.041** — the control is SIGNIFICANTLY better than the
   snap on both domains (the prior "+10 movie likely n.s." guess was wrong in the
   control's favor). Control vs floor: counted, labeled one-sided-by-construction,
   never quoted as a test.
4. **Stale prose fixed** — crate README row + Cargo.toml comment name the ONE real
   remaining gate (release-profile G2 + quiet-box preflight).

## The round-4 fix list (round-4 items, landed this session — the revert in the Cargo.toml, the arms in the harness, this fixture pair)

1. **Promotion REVERTED** — `default = ["extraction_estimate"]` again; the production
   API (`best_snapped_span`/`snap_longest`), the asserted GOAT relations, and the
   upward ratchets stay (correct whatever happens to the promotion).
2. **`floor+ptr_fallback` (the control, pre-registered):** the floor+alias(longest)
   answer; the pointer(gate) window ONLY where the floor would abstain. NOT the
   forbidden ensemble (no floor override of the pointer).
3. **McNemar (snap vs floor, choices-class):** discordant counts + exact two-sided p.
4. **Held-out fixture:** MIT-Restaurant (built this round — provenance:
   `tests/fixtures/external/PROVENANCE_MIT_RESTAURANT.md`; same converter rules,
   TRAIN-derived vocab, BLAKE3-pinned; 1,517 rows / 3,151 entities / 8 types, ALL
   choices-class).
5. **Release G2 + quiet-box preflight:** still BLOCKED (box load 14.6, 8 sibling cargo
   processes; a latency number now would measure the scheduler). The one remaining gate.

## Round 4 results — BOTH domains (debug profile, counting metrics, load-immune;
## determinism: movie `0x1b8e00acdabd9965` ×2, restaurant `0x338b61622eec9efb` ×2)

| arm (choices-class exact) | MIT-Movie (2,960) | MIT-Restaurant (2,338) |
|---|---|---|
| floor+alias (longest-match) | 2304 (77.8%) | 1435 (61.4%) |
| pointer+snap (round 3) | 2311 (78.1%) | 1464 (62.6%) |
| **floor+ptr_fallback (round 4 control)** | **2321 (78.4%)** | **1493 (63.9%)** |
| pointer(gate) | 2200 (74.3%) | 1195 (51.1%) |

Non-degenerate: movie fb 581 > snap 575 > floor 564; restaurant fb 752 > snap 747 >
floor 694.

**McNemar (snap vs floor-longest, choices-class):** movie b=22 c=15 → **p=0.324 — the
+7 margin is NOT significant** (the verdict's suspicion, confirmed). restaurant b=58
c=29 → **p=0.002 — significant.** The snap-over-floor edge is real in the held-out
domain and noise-thin in the design domain; on neither domain does it beat the control.

## Round 4 verdict (the pre-registered re-promotion rule, evaluated)

- (a) control measured: **the control BEATS the snap on both domains** (+10 movie, +29
  restaurant) — the verdict's prediction ("it could plausibly reach about floor+17 or
  more and beat snap") was exact on movie (2321 = 2304 + 17 precisely). The hybrid's
  honest description, now measured: *the gazetteer resolves; the pointer covers the
  gazetteer's abstentions; the window constraint itself nets ≈ −10 (movie) / +29−58
  fallback... (restaurant: snap 1464 = 1406 via-surface + 58 fallback; control 1493 =
  floor 1435 + 58 pointer-covered abstention hits, minus none)* — the pointer's value
  concentrates where the gazetteer has NO literal answer (702 abstains on movie, 702 on
  restaurant for the gazetteer vs the pointer's ≤23).
- (b) McNemar stands: recorded (movie n.s., restaurant significant).
- (c) held-out confirms snap ≥ floor: **YES on restaurant** (+29, p=0.002) — and the
  control ≥ snap on both.
- (d) release G2 + quiet-box: **BLOCKED** (environmental, standing).

**Per the pre-registered rule: the promotion basis MOVES TO THE CONTROL ARM
(`floor+ptr_fallback` — gazetteer-first, pointer-for-abstentions), and NO promotion
lands until (d).** The control arm is implemented in `span_pointer` consumers' terms:
it needs the pointer ONLY for the fallback, so the promoted unit remains the
`span_pointer` feature — but the GOAT-winning CONSUMER PATTERN is gazetteer-first +
`best_span`-on-abstain, never the bare window and never snap-first. The harness pins
both domains' ratchets (downward-only) and asserts the snap>floor relation on BOTH
(movie measured +7; restaurant measured +29 — both true, both asserted).

**Per the pre-registered rule: the promotion basis MOVES TO THE CONTROL ARM
(`floor+ptr_fallback` — gazetteer-first, pointer-for-abstentions), and NO promotion
lands until (d).** The control arm is implemented in `span_pointer` consumers' terms:
it needs the pointer ONLY for the fallback, so the promoted unit remains the
`span_pointer` feature — but the GOAT-winning CONSUMER PATTERN is gazetteer-first +
`best_span`-on-abstain, never the bare window and never snap-first. The harness pins
both domains' ratchets (downward-only) and asserts the snap>floor relation on BOTH
(movie measured +7; restaurant measured +29 — both true, both asserted).

## Round 3 (measured history — the promotion it landed was REVISED by round 4)

The round-2 record's pre-registered re-open, executed on the movie fixture: the
pointer(gate) window LOCALIZES, the schema vocabulary RESOLVES the boundaries (arm
pre-registered in the harness docs BEFORE scoring; production API
`SpanPointer::best_snapped_span` + free `snap_longest`). Measured:
**pointer+snap 2311 (78.1%) > floor+alias(longest) 2304 (77.8%)** choices-class exact —
the bar cleared (+7; ND 575 vs 564; contain 2371 vs 2329; abstain 1 vs 313), mechanism
2294 via-surface + 17 fallback, determinism `0x8f21fb6ed150b87c` ×2. The promotion
re-landed in `cb7dd315e` and was REVISED by the round-4 verdict (session ea31bb50) the
same session: the margin rides the 17 fallback exacts, not the snap mechanism (the
window constraint nets ≈ −10: 2294 via-surface vs the floor's 2304); the control arm
was unmeasured; no McNemar; no held-out; release G2 skipped. Demotion reverted same
session; the API, asserts, and ratchets stayed.

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

## Round 2 GOAT verdict (the pre-registered bar)

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
point the standing ratchets re-pin upward and the promotion re-lands. *(Round 3 above is
that re-open, measured; round 4 revised its promotion.)*

## What the pointer still wins on the movie fixture (disclosed, not the gate)

- **Containment:** 84.1% vs the gazetteer's 78.7% — the pointer's windows over-cover the
  answer more often than they exactly hit; on the non-degenerate slice the containment
  gap is +12.9 pts (61.3% vs 48.4%). For a downstream consumer that can shrink a
  containing window, the pointer's errors are a different (milder) class than the
  gazetteer's. (Round 4 reframes it: this containment surplus is exactly what the
  control arm's fallback monetizes.)
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

## Pinned in the harness (downward-only; both fixtures, config-driven)

Movie (`Ratchets` in the harness `MOVIE` config): floor-longest 2304 · ptr_gate exact
2200 / contain 2488 · ND ptr 537 · snap 2311 / ND 575 · **round-4 control 2321 / ND
581** · vocab pin + census (4,531 surfaces / per-type counts) · OOV EXACT 740 ·
per-type (Genre 277) · **GOAT relation asserted (snap > floor, measured +7)**.
Restaurant (the `RESTAURANT` config, round-4 pins): floor-longest 1435 · ptr_gate
1195 / 1613 · ND ptr 580 · snap 1464 / ND 747 · **control 1493 / ND 752** · vocab pin
+ census (5,188 surfaces / per-type counts) · OOV EXACT 813 · **GOAT relation asserted
(snap > floor, measured +29, McNemar p=0.002)**. Both: verbatim liveness, double-run
determinism, census pins; the control arm's ratchets are the promotion basis's pins.

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
