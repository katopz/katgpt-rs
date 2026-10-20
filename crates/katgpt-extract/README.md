# katgpt-extract

The modelless **extraction substrate** — the Fathom F0 floor (Plan 628 /
Proposal 016): typed field extraction from text with zero weights, zero
output tokens, µs–ms latency, and first-class abstention.

| module | feature | one-liner |
|---|---|---|
| `estimate`, `forecast` | `extraction_estimate` (**default-on** — G1 floor comparison beats the conformal-naive floor at --release, Plan 628 T0.6) | conformal estimate + forecast heads over the Plan-340 substrate (Report-the-Floor gated) |
| `span_pointer` | `span_pointer` (opt-in — the round-3 hybrid's +7 is pending the verdict's fix list: control arm + McNemar + held-out fixture + release G2) | candidate-window span scoring (BM25 + hashed-feature embedding + aliases, sigmoid, byte-exact spans, zero-alloc) + `snap_longest`/`best_snapped_span` boundary resolution |
| `parsers` | `typed_parsers` | deterministic typed parsers (number/date/time/datetime/duration/list; Duckling/Recognizers-Text grammar lineage; constraints by construction) |
| `temper` | `temper` | calibration/abstention gate (sigmoid confidence, fused abstain, null-with-guess) |

`extraction_estimate` is default-on (the GOAT floor gate passed); the rest
are opt-in (`--no-default-features` compiles to the empty crate). One dep:
`katgpt-core` (feature-gated, `default-features = false`). Pure std —
wasm-safe by construction.

The wire contract (`extraction_wire`) lives in `katgpt-core` beside
`decision_wire`; this crate is the engine-side substrate that speaks it.
The assembled serving engine is the consumer repo (Phase 1, owner-gated).

License: MIT.
