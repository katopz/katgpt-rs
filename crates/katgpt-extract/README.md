# katgpt-extract

The modelless **extraction substrate** — the Fathom F0 floor (Plan 628 /
Proposal 016): typed field extraction from text with zero weights, zero
output tokens, µs–ms latency, and first-class abstention.

| module | feature | one-liner |
|---|---|---|
| `span_pointer` | `span_pointer` | candidate-window span scoring (BM25 + hashed-feature embedding + aliases, sigmoid, byte-exact spans, zero-alloc) |
| `parsers` | `typed_parsers` | deterministic typed parsers (number/date/time/datetime/duration/list; Duckling/Recognizers-Text grammar lineage; constraints by construction) |
| `estimate`, `forecast` | `extraction_estimate` | conformal estimate + forecast heads over the Plan-340 substrate (Report-the-Floor gated) |
| `temper` | `temper` | calibration/abstention gate (sigmoid confidence, fused abstain, null-with-guess) |

Everything opt-in (`default = []`). One dep: `katgpt-core`
(feature-gated, `default-features = false`). Pure std — wasm-safe by
construction.

The wire contract (`extraction_wire`) lives in `katgpt-core` beside
`decision_wire`; this crate is the engine-side substrate that speaks it.
The assembled serving engine is the consumer repo (Phase 1, owner-gated).

License: MIT.
