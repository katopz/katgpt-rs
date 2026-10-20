# PROVENANCE — recognizers_core.jsonl (Plan 628 T0.6, G5 external extraction fixture)

**Status:** COMPLETE — 351 rows, BLAKE3-pinned; extraction script kept out of the repo (in /tmp) by design.

## Source

- **Repo:** Microsoft/Recognizers-Text — <https://github.com/microsoft/Recognizers-Text>
- **Commit:** `da7edcff59f669b2a460ab9d400e36298f0d658e` (short `da7edcff59f6`), shallow clone (`git clone --depth 1`) taken 2026-10-20 from `master`.
- **License:** MIT (repo root `LICENSE`). Every fixture row is Microsoft-authored test data for THEIR test suite — nothing here is authored by us for ours.
- **Layout note:** the upstream spec tree is `Specs/<dimension>/<Language>/<File>.json` (e.g. `Specs/Number/English/NumberModel.json`) — NOT `Specs/<lang>/<dimension>/`. All `source` fields carry the real upstream paths.
- **BOM note:** `Specs/DateTime/English/DateTimeModel.json` carries a UTF-8 BOM upstream; the extractor reads with `utf-8-sig`.

## Fixture

- **Path:** `crates/katgpt-extract/tests/fixtures/external/recognizers_core.jsonl`
- **Format:** one JSON object per line: `{"id", "source", "dim", "text", "expected_raw"[, "expected_wire"]}`.
- **Rows:** 351 (275 with `expected_wire`, 76 span-only).
- **Size:** 144,241 bytes (< 500KB cap).
- **BLAKE3 pin (b3sum, /opt/homebrew/bin/b3sum):**
  `f3e1dfc3dae9d417b7e93e7005857a12ba3963be9085cac93adc35665a988163`
- **SHA-256 (secondary pin, shasum -a 256):**
  `f2ea2b8ba11a9e1ec858044ef3c186611ed1f168fd98807a8e9c84f61abf8f13`
- The extraction script itself is deliberately NOT committed (task scope: output + provenance only); it lived at `/tmp/rtx/extract_rtx.py` and this document is the full record of its rules.

## Extraction method

Fields read per upstream row: `Input` (row text), `Results` (carried verbatim), and inside each result either `TypeName` + `Resolution.value` (number family) or `Type` + `Value.{Timex, FutureResolution, PastResolution}` (DateTime parser files), or `TypeName` + `Resolution.values[]` with `{timex, type, value}` (DateTimeModel supplement).

Source files (per dim):

| dim | files (upstream path) |
|---|---|
| number | `Specs/Number/English/NumberModel.json`, `Specs/Number/English/PercentModel.json`, `Specs/NumberWithUnit/English/CurrencyModel.json`, `Specs/NumberWithUnit/English/DimensionModel.json` — interleaved round-robin so plain numbers, "$5"-class currency, "12%"-class percent and "5 km"-class dimensions are all represented |
| date | `Specs/DateTime/English/DateParser.json` + `Specs/DateTime/English/DateTimeModel.json` (`datetimeV2.date` rows) |
| time | `Specs/DateTime/English/TimeParser.json` + DateTimeModel (`datetimeV2.time`) |
| datetime | `Specs/DateTime/English/DateTimeParser.json` + DateTimeModel (`datetimeV2.datetime`) |
| duration | `Specs/DateTime/English/DurationParser.json` + DateTimeModel (`datetimeV2.duration`; deduped against parser rows) |

Row-selection rules (in order):

1. Row must have non-empty `Results` (negative/empty-result test rows dropped).
2. **Type purity:** every result in a row shares one type — number family: `TypeName` uniform and in {number, percentage, currency, dimension}; DateTime family: `Type` equals the dim exactly (ranges/sets/timezones excluded).
3. **Realism filter (supported-subset):** text ≤ 90 chars, printable ASCII only.
4. **Relative-text drop (reference-date rule):** rows whose TEXT itself needs a reference date are dropped entirely — blocklist: today, tomorrow, yesterday, tonight, now, later, recently, ago, "next/last/this/coming/past X", "the day after/before", "as soon as possible", "in <n> <unit>", "<n> <units> later/from now".
5. **Value-mechanical gate for `expected_wire`:** single-result rows only, and only when the value resolves without any reference:
   - parser files: `FutureResolution` == `PastResolution` AND fully concrete `Timex` (no `X` placeholders — an `X` component means the value moves with the reference date; symbolic `PRESENT_REF`/`FUTURE_REF`/`PAST_REF` never pass);
   - DateTimeModel supplement: single result, single resolution value, concrete timex (no `X`), so model rows carrying the future/past candidate pair are all dropped as reference-dependent;
   - multi-result rows never carry wire (multi-entity rows serve the span lane).
6. **Caps:** per-dim (wire_cap, span_cap) = number (85, 20), date (65, 15), time (55, 20), datetime (25, 40), duration (52, 8); wire-eligible rows fill first, span-only rows fill the reserve (multi-result rows first).
7. **Dedupe:** exact text (case-sensitive), global across the file, first in emission order wins — this is why the DateTimeModel supplement contributes little for duration/datetime (upstream reuses the same Input sentences across parser and model files).
8. Emission order: number, date, time, datetime, duration; ids `R-0001`… sequential.
9. `"NotSupported"` annotations on upstream rows (57 among emitted rows — marks other-language ports) were ignored: the C# results are the reference and the annotation does not affect them.

`source` field format: `microsoft/Recognizers-Text@<12-char sha> <upstream-path>#<1-based row index in the file's JSON array>`.

## expected_wire vocabulary (mechanical translations only)

| dim | wire shape | transform from their result |
|---|---|---|
| number | `{"value": <f64>}` | `Resolution.value` parsed as float; for percentage rows ONE trailing `%` is stripped first (e.g. `"100%"` → `100.0`); currency/dimension rows carry only the numeric value (their `unit` string is upstream vocabulary, not ours) |
| date | `{"iso_date": "YYYY-MM-DD"}` | `FutureResolution.date` when equal to `PastResolution.date` and to the concrete Timex |
| time | `{"iso_time": "HH:MM:SS"}` | `FutureResolution.time` (== past), requires the seconds form |
| datetime | `{"iso_datetime": "YYYY-MM-DDTHH:MM:SS"}` | `FutureResolution.dateTime` with the space separator replaced by `T`; a seconds-less upstream value is zero-padded with `:00` |
| duration | `{"unit": "s|min|h|d|w|mo|y", "value": <f64>}` | single-component Timex only: `P<n>Y→y`, `P<n>M→mo`, `P<n>W→w`, `P<n>D→d`, `PT<n>H→h`, `PT<n>M→min`, `PT<n>S→s`; multi-component Timex (e.g. `P1DT2H`) → wire omitted |

**Honest caveat:** `expected_raw` is Microsoft's `Results` array for the row, verbatim (key order preserved, no normalization, no answer-morphing). `expected_wire` is OUR mechanical translation layered on top for typed-lane scoring — it is not Microsoft's claim, and where their shape was ambiguous or reference-dependent the wire field is OMITTED entirely and the row serves span-level G5 only. Fidelity was verified by a full census: all 351 rows' `text` + `expected_raw` re-read from the pinned upstream commit and compared structurally — 0 mismatches.

## Row counts

| dim | rows | with wire | span-only | multi-result |
|---|---|---|---|---|
| number | 105 | 85 | 20 | 20 |
| date | 80 | 65 | 15 | 0 |
| time | 67 | 50 | 17 | 0 |
| datetime | 43 | 23 | 20 | 0 |
| duration | 56 | 52 | 4 | 0 |
| **total** | **351** | **275** | **76** | **20** |

Notes: datetime is below the ~80-row target because upstream's plain-form absolute datetime rows are finite (36 parser rows + 43 model rows survived filters before dedupe). Duration unit coverage across wire rows: h=14, d=9, w=8, y=8 (incl. 0.5/3.5 fractionals), mo=6, min=5, s=2.

## Skipped rows, by reason class (counted at the point of classification, upstream file order)

| class | where | count |
|---|---|---|
| empty `Results` (their negative tests) | number files | 18 + mode-file rows |
| relative-text blocklist (rule 4) | number 19 · date 102 · time 25 · datetime 69 · duration 3 | 218 |
| > 90 chars / non-printable (rule 3) | mostly long CurrencyModel news sentences | 159 (number family) |
| type impurity / other-type rows (rule 2) | e.g. daterange/timerange/set/timezone model rows, NumberRangeModel, OrdinalModel not sourced at all | counts per dim printed by the extractor run; see the extractor log excerpt in git history of this file's companion commit message if needed |
| reference-dependent value (rule 5) | rows kept as span-only with wire omitted: 76 (table above) | — |
| duplicate text after all filters (rule 7) | mostly DateTimeModel rows duplicating parser-file Inputs | resolved first-wins |
| cap truncation | remaining wire-eligible candidates beyond caps (number had ~370 candidates for 105 slots) | not tabulated per-row |

DateTimeModel rows of other TypeName families (daterange 233, datetimerange 41, timerange 40, set 45, timezone 1) are out of the five-dim scope by construction.
