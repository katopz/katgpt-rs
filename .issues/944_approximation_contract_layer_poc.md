# Issue 944: Consolidate the three ad-hoc ULP helpers into one `ulp_distance` / `ApproxContract` (from ink-lang Research 475)

**Status:** Open — P2, DRY consolidation with three live consumers; opt-in until gates pass

**Claimed:** Zed-agent (GLM), epoch 1792147000, expires 2026-10-18T14:00:00+00:00

**Source:** ink-lang (MIT, @ `1a5c025fe86e92e7729edb1af093aa1ceb5e1abb`) checked approximation contracts, distilled in `riir-train/.research/475_Ink_Lang_Learning_Lifecycle_Proof_Checked_Optimization.md` (**private repo** — essentials inlined here). External prior art for the *metric*: Goldberg ULP, Arm libamath ULP practice, Herbie/Gappa lineage — the claim is only the **declarative contract form + the non-composition rule**, not the metric.

## Corrected premise — the substrate already ships ULP checks, three times (copy-pasted)

One formula (`(a.to_bits() as i32)` two's-complement bit distance) exists as three unshared copies:

| Copy | Site | What it asserts |
|---|---|---|
| `ulp_diff` | `crates/katgpt-core/benches/bench_844_exact_sigmoid_ordered_dot.rs:72` | **G1a gate** (L148–160): `exact_sigmoid` ≤ 4 ULP vs the f64-computed-and-narrowed reference, abs ≤ 1e-6, and strictly closer than `fast_sigmoid` |
| `ulp_diff` + `ulp_scan` | `crates/katgpt-types/src/simd/activations.rs:1844` + `:1884` | Tests: `exact_sigmoid` ≤ 2 ULP vs f64 reference over a [-90, 90] sweep + edges; `exact < fast` over a coarse stride |
| `ulp_delta` | `benches/bench_871_algebraic_dot_ab.rs:119` | Max-ULP A/B gate over algebraic-dot outputs; doc honestly states the same-sign-finite precondition ("±0 never occurs with a random mantissa") |

## What ink actually adds (claim only this)

1. **A declarative contract surface** — `approximates ref within ulps K` as a checkable assertion bound to a named reference, instead of hand-rolled sweep loops per bench.
2. **Two semantics decisions our copies punt on:**
   - *Signed zeros share ONE position.* Our two's-complement copies give ±0 a garbage huge distance and work only under a same-sign-finite precondition (bench_871's doc says so).
   - *NaN/Inf fail closed.* Ink refuses non-finite inputs; our copies produce undefined bit-distance garbage.
3. **"Two local tolerances do not silently become one tolerance"** — compositional bound discipline (analogous to our lossy-law rule that aggregate metrics can be flat while family-conditional behavior flips; **riir-ai** Issue 750 T3).

## Work

- One `ulp_distance(a, b) -> u32` (+ a tiny `ApproxContract` assertion helper) in `katgpt-types`, behind opt-in feature `approx_contract`; zero-alloc, test/bench-only consumers — no hot-path code changes.
- **Settle the semantics split** (the real design decision): parameterize signed-zero handling (poison / one-position) and non-finite handling (fail-closed `Option`/`Result` vs documented precondition) — the three copies currently disagree with each other *and* with ink; pick one law, state it at the definition.
- Migrate the three call sites (`bench_844` G1a, `activations.rs` tests + `ulp_scan`, `bench_871`).
- **Green-zero trap (verdict-reviewer note):** if `ulp_distance` sits behind the opt-in `approx_contract` feature, every migrated target either enables that feature or its gated body compiles away to a silent `ok. 0 passed`. Either leave the helper UNGATED in `katgpt-types` (test-only, zero hot path — the simpler law) or add a `required-features` row per migrated target in the SAME commit as the migration.
- Optional, cheap: a Lean face for the analytic bound (e.g. `exact_sigmoid` ≤ 4 ULP vs the f64 reference) in `KatgptProof`, mirroring the boundedness-theorem pattern.

## Gates

- Migrated gates assert **identical verdicts** on their existing domains (bit-identical metric outputs for finite same-sign inputs — the consolidation changes no measurement).
- Negative test: a seeded bound violation is caught (the gate can fail).
- No allocation, no hot-path code touched; feature stays opt-in until the above pass.

## Non-goals

- Does NOT apply to model-level lossy transforms (PQ2_0 quant, GDN state-half) — those stay on behavioral retention gates (**riir-ai** Issue 750 T3 law); a ULP contract cannot express family-conditional behavior.
- Modelless only; consumes no training.
