# Bench 949 — F0 typed parsers: the issue-953 coverage levers (G5 before/after)

**Status:** DONE — all six issue-953 lever families landed; G5 re-measured; the accuracy ratchet pinned from the post-landing record; the load-disclosed G2 re-run PASS ×2 (checksum-stable).

**Date:** 2026-10-20 · **Crate:** `katgpt-extract` · **Plan:** [628](../.plans/628_fathom_extraction_substrate.md) T0.6 follow-up · **Baseline:** [Bench 948](948_f0_extraction_g1_g2_g4_g5.md) · **Issue 953:** RESOLVED + removed per the noise rule — full text in git history (`git log --follow -- .issues/953_parser_word_form_coverage.md`); the lever list + miss classes live in this record

## The G5 delta (351 pinned Recognizers-Text rows, 275 wire rows)

| dim | rows | wire | value (before → after) | lenient | exact any | cont any | live (before → after) |
|---|---|---|---|---|---|---|---|
| number | 105 | 85 | 9.4% → **87.1%** | 0.0% | 74.8% | 76.2% | 73.3% → **98.1%** |
| date | 80 | 65 | 12.3% → **60.0%** | 0.0% | 46.2% | 46.2% | 10.0% → **48.8%** |
| time | 67 | 50 | 12.0% → **16.0%** | **62.0%** | 55.2% | 55.2% | 31.3% → **74.6%** |
| datetime | 43 | 23 | 4.3% → **13.0%** | **17.4%** | 16.3% | 16.3% | 2.3% → **16.3%** |
| duration | 56 | 52 | 23.1% → **100.0%** | 0.0% | 91.1% | 91.1% | 21.4% → **94.6%** |
| **TOTAL** | 351 | 275 | **12.7% → 64.0%** | 12.7% | 61.4% | 62.0% | **33.9% → 71.8%** |

Time/datetime primaries stay low because the wire expects `HH:MM:00` while the
parsers emit the minimal `HH:MM` — those rows land in the LENIENT bucket
(reported, never folded in). Duration is 52/52 exact.

## What landed (the six lever families, all Recognizers/Duckling-translated shapes)

1. **Word-number grammar** — new `parsers/words.rs`: the magnitude grammar
   (`twenty and four` = 24, `one hundred and three`, `hundred thousand`,
   `two thousand three hundred`), one 0-99 groups, written years (thousand
   form + digit-smash: `nineteen zero one` = 1901), ordinal day words, `a/an`
   = 1 before a magnitude/dozen/tiny-magnitude word. Additive groups compose
   only at strictly decreasing magnitude (`one two` is not a number).
2. **Duration quantifiers** — `(all|full|whole) [-]unit` = 1, `(a|an|another)
   unit` = 1, `(a) couple of unit` = 2, `half (an) unit` = 0.5, `a
   fortnight` = 2 w, `N decades` = 10N y (spelling-keyed factor table), the
   `and [a] half` chain/term tail, hyphenated units (`3-min`, `30-minutes`),
   `hrs` (a real pre-existing UNITS-table gap). Duration wire accuracy 23.1%
   → **100%**.
3. **Number levers** — spaced currency (`$ 14 million`), percent WORDS in the
   spaced suffix table (`12 percent/percentage/percents` — a pre-existing
   gap: `"100 percent"` never parsed), prefix-percent forms (`per cent of
   210`), chained multipliers incl. `dozen(s)` ×12 / `lakh` ×1e5 /
   `trillionth(s)` ×1e-12 / `third(s)` ×1/3, digit fractions (attached
   `/den` = the core IS the numerator; spaced `10 1/2` = 10.5; `- 1 4/5` =
   -1.8), unit-set widening (ml/liters/gallons/tonnes/ounces/yen/francs/
   pence/cents).
4. **Rank-0 law** (the issue's ask 3 — the rank rule, not the grammar): the
   BEST candidate always carries rank 0 in `finish()` (a lone bare `192`
   answers a rank-0 read), and rank ties break by composition DEPTH first
   (`$ 254 million` — currency+multiplier — over an earlier `27 %`).
5. **Date levers** — separators `/ - .` + year-gap `,`, year-first numerics
   (`2015/08/12`, `2019-sep-1`, `2019/sep/01`), 2-digit-year window (< 50 →
   2000+, else 1900+), dotted/dashed month names (`16. Nov. 2016`,
   `05-Aug-2016`, `Dec. 31 , 1994`, attached `Dec31`), ordinal suffixes
   (any of st/nd/rd/th — Recognizers accepts `02nd`), weekday-prefixed
   absolutes (`monday january 12th, 2016`, `friday 2018-7-6`), word days +
   written years in month-day (`jan twenty two, 2018`, `September 10,
   nineteen zero one`, `the tenth of September, two thousand`,
   `twenty second of June 2017`), word numbers in relative offsets.
6. **Time levers** — the written clock (`eight forty` = 08:40, `seven thirty
   pm`, `fifteen fifteen p m`), the loose meridiem (`p. m.`, `p . m .`),
   day-part modifiers both orders (`afternoon 7:00`, `five thirty in the
   night`, `in the morning at 7`, `circa 5 in the morning`), attached
   minutes before a meridiem (`340pm` = 15:40, `1140 a.m.` = 11:40);
   composed datetime separators `,` `-` `@` both orders (`04/21/2016,
   8:00pm`, time-first `19:00, 2016-12-22`, `5/3/18 @ 17:49:19`).

## Defects the fixture caught in-flight (fixed the same session)

- Magnitude fold: `hundred thousand` computed 1100 (hundred must multiply
  the PENDING sum, not fold it).
- Scaled-unit factor landed in the DENOMINATOR (`two decades` = 2 y).
- Attached `/den` ADDED 1/den instead of dividing (`1/4 mile` = 1.25).
- The comma separator check ran after the whitespace guard — a comma
  against the date (`2016, 8:00pm`) read as "no separator".
- The rank-0 promotion initially ran BEFORE dedup, splitting rank-equal
  duplicate pairs into rank-0 + rank-2 ghosts.
- `cargo refine`'s map-entry rewrite broke `span_pointer`'s `Box<str>` key
  (the documented divergence class) — caught by clippy, fixed by hand.

## Gates

- **G5 ratchet (NEW, issue 953 ask 2):** `VALUE_RATCHETS` pinned at the
  exact post-landing fractions (74/85, 39/65, 8/50, 3/23, 52/52); liveness
  floors re-pinned (103/105, 39/80, 50/67, 7/43, 53/56). Both fire DOWNWARD
  only; the next coverage lever re-pins from its own record. GREEN.
- **G4:** all scanners zero-alloc at `--release --features
  typed_parsers,alloc_tracking` GREEN (word paths are stack-only).
- **G2 (load-disclosed, PROVISIONAL):** composite F0 request p99 **402 µs /
  409 µs** (two runs, work checksum `14233999` identical — deterministic;
  the checksum moved from 948's `14156999` because the richer candidate set
  extracts more from the bench state) vs the pinned 10 ms bar — PASS ×2,
  ~24× headroom. `parse_number` p50 2.0 → 5.3 µs (the word tables' scan
  cost; trivial against the bar). **Preflight REFUSED at both runs** (load
  9.28 vs 6.0, powermode 2 High Power, 4 sibling sessions) — the quiet-box
  preflight PASS line stays OWED before any final composite quote (inherited
  from Bench 948's disclosure).
- **Tests/clippy:** 132 tests green (typed_parsers+span_pointer+temper),
  clippy `-D warnings` clean at default/all/no-default postures; workspace
  `cargo check` green.

## Disclosed remainder (the measured misses below the ratchet)

Number: leading-dot decimals (`.08`, documented non-form), mixed
integer+fraction sums (`one hundred and three and two thirds`), compound
units (`5 ft 2 in`), the `$`-vs-`%` equal-depth tie (`2 % or $ 1.40 a
share` — depth ties still break by position), Recognizers' `three hundred
and two dozens` = 324 quirk, `half a  dozen` double-space. Time: `to/past`
forms, `7h01`, `8.10 pm`, `noonish`/`11ish`, `9.am.`, `mid night`. Datetime:
`end of the day`/`eod` idioms, ctime format (`Wed Oct 26 15:50:06 2016`),
`monday the twenty seventh at six pm` (anchored — no reference in the
harness). Date: the anchored families (`Thursday the 21st`, `second Sunday`,
`in four business days`, `tmrw`, `the day`) — all need `reference_date`,
which the harness passes as None by design.
