//! Deterministic typed parsers (Plan 628 T0.3).
//!
//! PLACEHOLDER — implemented by the T0.3 lane. Contract: grammar-first
//! parsers over the wire's field kinds — number, date/time/datetime
//! (ISO + `reference_date`-anchored relatives), duration, list-of-typed —
//! returning candidates ranked best-first, each with its exact byte span
//! into the state. Constraints (integer/decimals/unit/range/element kind)
//! enforced BY CONSTRUCTION: a candidate that cannot satisfy them is
//! never produced; the engine abstains rather than clamps.
//!
//! Grammar design translated from Meta Duckling (Apache-2.0) and
//! Microsoft Recognizers-Text (MIT) — credited per-module; never
//! reinvented.
