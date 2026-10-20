# PROVENANCE — mit_restaurant_test.jsonl (Bench 950 round 4, the held-out confirmation fixture)

**Status:** COMPLETE — 1,517 rows / 3,151 entities, BLAKE3-pinned; converter kept out of
the repo (in /tmp, `scripts/tmp_convert_restaurant.py` deleted at close) by design, full
rules below. Built 2026-10-20 as the verdict's held-out domain (Bench 950 round 4 fix
list item 3): a fixture that played no part in designing the round-3 snap arm or the
round-4 control arm.

## Source

- **Corpus:** MIT Restaurant Corpus — the MIT Spoken Language Systems Group downloads
  tree (the same SLS releases as the movie corpus; the restaurant domain of the
  Liu/Pasupat/Wang/Cyphers/Glass ASRU-2013 query-understanding corpus family; original
  collection credited to the MIT SLS semantic-query parsing work).
- **Files:**
  `restauranttest.bio` — <https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio>
  `restauranttrain.bio` — <https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio>
- **Downloaded:** 2026-10-20 from the MIT SLS downloads tree.
- **SHA-256 (upstream file pins):**
  - test: `09809dd126a3781c4205d06e721752054e51a10b1c98f3495ef17cd723f25ce2`
  - train: `f51794247426c975e7701011d4dba668ac6b881358f162e2c2c10709f2d17b9b`
- **License:** distributed for research from the MIT Spoken Language Systems Group; cite
  the SLS query-understanding corpus family. No fixture row is authored by us — every
  token and tag is upstream's.

## Fixture

- **Path:** `crates/katgpt-extract/tests/fixtures/external/mit_restaurant_test.jsonl`
- **Format:** identical to the movie fixture — one JSON object per line:
  `{"id","source","text","entities":[{"type","start","end"},...]}` with BYTE offsets
  (`[start, end)`); the corpus is lowercase ASCII so byte == char offsets; every span
  satisfies `text[start..end]` == the annotated surface (asserted at build, 0 defects).
- **Rows:** 1,517 (every test sentence carrying >= 1 choices-class entity, file order,
  480 KB cap) / **3,151 entities** / 305,263 bytes.
- **BLAKE3 pin:** `427c3a51d464ee6dc8788d0c24a3de695835284ebd2dad4ff9d51a5ad7746763`
- **SHA-256 (secondary pin):** `1e0a64d46e1c45247254ab8fabe9d3cab512a6db03d4d76de7a80c4a05b19d9f`
- **Per-type census pins (the harness re-asserts):** Amenity 533 · Cuisine 532 · Dish
  288 · Hours 212 · Location 812 · Price 171 · Rating 201 · Restaurant_Name 402.

## Types — ALL 8 are choices-class

The choices-class rule (TRAIN surface unique% < 95) admits every type here — the
restaurant ontology is far more enumerable than the movie's (no Plot/Quote/Origin
free-text types):

| type | train unique% | vocab surfaces | test entities |
|---|---|---|---|
| Amenity | 44.3 | 1126 | 533 |
| Cuisine | 16.7 | 475 | 532 |
| Dish | 44.9 | 662 | 288 |
| Hours | 36.5 | 361 | 212 |
| Location | 29.5 | 1126 | 812 |
| Price | 24.9 | 182 | 171 |
| Rating | 21.3 | 228 | 201 |
| Restaurant_Name | 54.1 | 1028 | 402 |

## Companion fixture — the TRAIN-split schema vocabulary

`mit_restaurant_train_vocab.jsonl` — same rules as the movie vocab (the split that is
never scored; surfaces + TRAIN occurrence counts, sorted (count desc, surface asc)).

- **Rows:** 8 types / 5,188 unique surfaces; **size:** 115,112 bytes.
- **BLAKE3 pin:** `517554fb436ef2e12a71e60a6855e0ced8543ca16d30a050840f5cc793c5411e`
- **SHA-256:** `9cb8df435c63875104ea049d8af2edacd56a4f2bc3711f0a2d3294d7f780f960`

## Population rule

Identical to the movie protocol: a scored mention whose (type, surface) the TRAIN
vocabulary lacks is OUT of the population — **813 skipped, 2,338 scored** (74.2%
coverage; the movie fixture's coverage was 80.0%). The harness pins the OOV census
exact at 813.

## First-run record (2026-10-20, debug profile — Bench 950 round 4)

The held-out domain CONFIRMS the round-4 movie picture (full tables in
`.benchmarks/950_f0_t07_span_baselines.md` §round 4):

- floor+alias(longest) 1435 (61.4%) < pointer+snap **1464 (62.6%)** — the snap clears
  the floor with REAL significance here (McNemar b=58 c=29, **p=0.002**; the movie
  margin was nonsignificant at p=0.324).
- **floor+ptr_fallback (the control) 1493 (63.9%) — wins again**, over the snap by +29.
  Cross-domain verdict: the pointer's value concentrates where the gazetteer abstains;
  the promotion basis is the control arm, not the snap.
- Mechanism: 1406 exact via in-window surface, 58 via window fallback.
- Determinism checksum `0x338b61622eec9efb` (double-run identical).
