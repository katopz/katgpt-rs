# Issue 953 — typed parsers: word-form + composition coverage vs the Recognizers surface

**Status:** OPEN — measured input from the G5 fixture run (Bench 948); the accuracy bar for `typed_parsers` gets pinned from the first coverage landing, not before.

**Date:** 2026-10-20
**Plan:** [628](../.plans/628_fathom_extraction_substrate.md) T0.6 · **Bench:** [948](../.benchmarks/948_f0_extraction_g1_g2_g4_g5.md) (G5 table) · **Fixture:** `crates/katgpt-extract/tests/fixtures/external/recognizers_core.jsonl` (BLAKE3-pinned, Recognizers-Text @ `da7edcff59f6`)

## Finding

The G5 external fixture measures `typed_parsers` at **12.7% rank-0 value accuracy / 33.9% liveness** against 351 Microsoft-authored rows. The dark mass is NOT parse bugs — direct probes verified every miss class is a deliberate T0.3 scope boundary now colliding with the full Recognizers surface:

| class | example | dims darkened |
|---|---|---|
| word numbers/units | "two weeks", "eight forty", "twenty percent", "nineteen zero one" | duration (39/56 rows), time (29/67), number (25/105) |
| comma-composed datetimes | "04/21/2016, 8:00pm"; time-first "19:00, 2016-12-22" | datetime (1/43 alive) |
| 2-digit years | "21/04/16", "05/05/89" | date |
| separator set | "16. Nov. 2016", "3-7-2017", "2019-sep-1", "Oct. 2" | date |
| ordinal suffixes | "January 12th", "21st", "04th Jan" | date |
| unit coverage | "75ml" (ml not in unit set; word-adjacent-digit block) | number |
| spaced currency rank | "$ 14 million" parses right VALUE at rank 1, not rank 0 | number |
| documented non-forms | ".08" leading-dot, "7 o'clock" | number, time |

## Ask

1. **Coverage levers** (each lands with its G5 delta measured — the fixture harness re-runs in seconds): word-number grammar (Duckling's number grammar is the translated source), comma/space datetime composition, 2-digit-year window rule, ordinal suffix stripping, spaced-currency rank-0 promotion, unit-set widening (ml/mls/g/kg class).
2. **Pin the accuracy ratchet** after the first coverage landing (measured-then-pinned — the floors are liveness-only today by design).
3. Rank-0 promotion rule: when the same value parses at rank 0 from one surface form and rank 1 from another (spaced "$ 14 million"), the rank rule — not the grammar — is the defect; check the tie-break ordering before widening grammar.

## Non-goals

- Full Recognizers parity is NOT the goal — the F0 floor serves digit-form + the composition classes above; word-heavy long-tail rows can remain measured misses if the pinned bar says so.
- No fixture edits (the corpus is BLAKE3-pinned upstream data; widening scope = a NEW fixture add with its own provenance, never a mutation).
