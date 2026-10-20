# PROVENANCE — mit_movie_trivia_test.jsonl (Plan 628 T0.7, G5 span-baseline fixture)

**Status:** COMPLETE — 1,562 rows / 5,093 entities, BLAKE3-pinned; converter kept out of the
repo (in /tmp) by design, full rules below.

## Source

- **Corpus:** MIT Movie Corpus (trivia10k13) — J. Liu, P. Pasupat, Y. Wang, S. Cyphers,
  J. Glass, *"Query Understanding Enhanced by Hierarchical Parsing Structures,"* Proc. ASRU
  2013, pp. 72-77 (MIT SLS; the Ritter-class domain-ontology NER corpus the plan names).
- **File:** `trivia10k13test.bio` — <https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13test.bio>
- **Downloaded:** 2026-10-20 from the MIT SLS downloads tree (last-modified 2014-06-17 upstream).
- **SHA-256 (upstream file pin):**
  `244807ba0006744a103a9449a12503641af287a869887b6d9cbb75b749e92d3d`
- **License:** distributed for research from the MIT Spoken Language Systems Group;
  cite Liu et al. (ACL 2013 short, query-understanding movie corpus). No fixture row is
  authored by us — every token and tag is upstream's.

## Fixture

- **Path:** `crates/katgpt-extract/tests/fixtures/external/mit_movie_trivia_test.jsonl`
- **Format:** one JSON object per line:
  `{"id", "source", "text", "entities": [{"type", "start", "end"}, ...]}`
  — `start`/`end` are BYTE offsets into `text` (`[start, end)`, end exclusive). The corpus
  is pure printable ASCII lowercase, so byte offsets == char offsets; every span satisfies
  `text[start..end]` == the annotated surface (verified 0 defects at build).
- **Rows:** 1,562 (every test-split sentence carrying >= 1 choices-class entity, file order,
  stopped at the 480 KB cap).
- **Entities:** 5,093 across 12 types (the counts below are per-type census pins the harness
  re-asserts).
- **Size:** 479,734 bytes (< 500 KB cap).
- **BLAKE3 pin (b3sum, /opt/homebrew/bin/b3sum):**
  `77a36d097177560e49378e9a3b7ea4496d43a06ae2ca43667f680bf0cf65c56d`
- **SHA-256 (secondary pin, shasum -a 256):**
  `990474be902b04fdfb82ca8b30ca036e58be3f3d1db8eb0bdda1b9478f1df749`

## Companion fixture — the TRAIN-split schema vocabulary

`mit_movie_trivia_train_vocab.jsonl` (Bench 950 round 2, verdict-round REVISE: the
round-1 alias table was derived from the scored test rows — the mention's own surface
usually rode its alias list, an answer key; the schema now comes from the split that is
never scored, the deployment analogue of a schema compiled from historical data).

- **Source:** `trivia10k13train.bio` — <https://groups.csail.mit.edu/sls/downloads/movie/trivia10k13train.bio>
- **SHA-256 (upstream file pin):**
  `3208087913814190b5ac6bf7a7c926d23b7cf0ab4a1734311badec4d8eaa5a82`
- **Format:** one JSON line per choices-class type:
  `{"type":"Actor","surfaces":[["johnny depp",16],...]}` — surface + TRAIN occurrence
  count, sorted by (count desc, surface asc).
- **Rows:** 9 types / 4,531 unique surfaces; **size:** 103,550 bytes.
- **BLAKE3 pin:** `3a7ba3868950deafa6f8f7eb4a578f72a954f661b009e85800169f6d3d1b9b4a`
- **SHA-256:** `def8ceed0cee1f0dda7d623dfe1aa9be9e344e7d7ccf66fd18f9279f686a0ecc`
- **Extraction:** the same IOB2 entity walk as the test fixture (same word-boundary trim,
  same B-/I- same-type continuation), counts per trimmed surface, choices-class types only
  (Plot/Quote/Origin excluded — no enumerable vocabulary; the same pre-registered rule as
  the fixture's population).
- **Harness census pins (surfaces per type):** Actor 1436 · Award 163 · Character_Name
  664 · Director 532 · Genre 908 · Opinion 292 · Relationship 346 · Soundtrack 47 ·
  Year 143.
- **Population rule (round 2):** a scored mention whose (type, surface) is absent from
  this table is OUT of the scoring population — 740 of 3,700 choices-class test mentions
  (80.0% coverage, 2,960 scored), counted and disclosed by the harness (an OOV-skip > 0
  assert keeps the vocabulary-source contract live).

## Upstream format

`TAG<TAB>token` lines; a blank line ends one sentence. Tags are IOB2 (`B-Type`, `I-Type`, `O`).
`text` is RECONSTRUCTED as the sentence's tokens joined by single spaces (the annotation
offsets are computed on that reconstruction — self-consistent by construction; the original
whitespace is not recoverable and not needed).

## Converter rules (in order)

1. Parse `trivia10k13test.bio` into sentences of `(tag, token)`.
2. Reconstruct `text = " ".join(tokens)`; token start offsets = prefix sums of
   `len(token) + 1`; an entity = a `B-` run (consecutive `I-` of the SAME type extends it;
   any other tag closes it). Entity span = `[start of first token, end of last token)`.
3. **Choices-class rule (the pre-registered scoring population):** a type is choices-class
   iff its fixture-wide surface unique% < 95% (an enumerable vocabulary exists — a choices
   table is representable). Measured unique%: Plot 99.0, Quote 97.9, Origin 97.4,
   Soundtrack 87.5, Character_Name 86.2, Award 71.2, Relationship 61.4, Director 50.8,
   Actor 49.3, Opinion 43.6, Genre 42.0, Year 15.9 → choices-class = {Actor, Award,
   Character_Name, Director, Genre, Opinion, Relationship, Soundtrack, Year}.
   Plot (free-text plot descriptions), Quote (quotations) and Origin (nearly-unique place
   mentions) are carried in the rows but are OUTSIDE the choices-class scoring population —
   a schema choices table cannot enumerate them.
4. **Row filter:** keep sentences with >= 1 choices-class entity, in file order; stop when
   the file would exceed 480,000 bytes (not reached — all 1,562 qualifying sentences fit).
5. `id` = `M-0001`.. sequential over emitted rows; `source` = `trivia10k13test.bio#<1-based
   upstream sentence index>`.

## Per-type entity census (harness pins)

| type | entities | choices-class |
|---|---|---|
| Plot | 1,218 | no |
| Actor | 1,218 | YES |
| Genre | 755 | YES |
| Year | 643 | YES |
| Director | 411 | YES |
| Character_Name | 260 | YES |
| Opinion | 183 | YES |
| Relationship | 157 | YES |
| Origin | 138 | no |
| Award | 65 | YES |
| Quote | 37 | no |
| Soundtrack | 8 | YES |
| **total** | **5,093** | **3,700** |

## Honest caveats

- The reconstruction joins tokens with single spaces — punctuation attaches to tokens
  upstream exactly as tagged; spans are exact against the reconstruction, not against any
  original string.
- The choices-class threshold (95%) is a mechanical rule over annotation statistics only —
  it was fixed before any scoring arm ran, and it excludes exactly the three types whose
  surfaces are >= 97% unique.
- The fixture is the corpus's own test split; the vocabulary/alias tables the harness
  derives (Plan 628 T0.7 protocol) are computed FROM this fixture — schema-derived like a
  real deployment's choices table, never hand-authored.
