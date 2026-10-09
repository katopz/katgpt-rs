# Issue 927 — katgpt-assign::rng::SplitMix64 duplicates katgpt-types::rng::SplitMix64 (same name, one day apart; adjudication-gated on the zero-dep law)

**Status:** RESOLVED 2026-10-09 — menu option 1 (the cross-pin) landed in `36823de3f`: the copy-gate convention fully met (twin cited in-source + the below divergence named + `katgpt-core/tests/splitmix64_twin_pin.rs` under `required-features = ["assignment"]` pinning both streams bit-identical from 6 seeds × 10k draws — proven to fire by perturbation). Option 2 (delegate) DECLINED: it would cost katgpt-assign's documented zero-dep standalone posture. Filed by the 10-09 substrate-first Mode 2 audit (detection-only; no fix in the filing commit)

- **Date:** 2026-10-09
- **Found by:** substrate-first Mode 2 fresh-wave audit (window 10-07 03:51 → 10-09 12:2x)
- **Skill:** `substrate-first` — run log row of 2026-10-09

## The finding

Two `pub struct SplitMix64` stream types in this workspace, the later one apparently unaware of the earlier:

| | `katgpt-types/src/rng.rs:39` | `katgpt-assign/src/rng.rs:13` |
|---|---|---|
| landed | 10-06, `514989664` (the reflex-Issue-071 substrate export — the designated workspace home) | `65a6b1b71` (Plan 620 Phase 1) — in the 10-07+ wave, i.e. AFTER |
| stream core | `new(seed)` + `next_u64` = γ-add + finalizer | identical (same constants, same order) |
| `below(n)` | **53-bit Lemire** (`((u64>>11) as u128 * n) >> 53`) | **64-bit Lemire** (`(u64 as u128 * n) >> 64`) |
| extras | `next_f64` | `shuffle`, `shuffle_u32`, `const fn new` |
| exported via | `katgpt_core::types::rng::SplitMix64` | `katgpt_core::assign::SplitMix64` (core takes assign as an optional dep behind `assignment`) |

- Both are re-exported through the root crate under different paths — two same-named pub stream types in the public workspace.
- **Not drop-in duplicates:** `below(n)` differs between them — different indices for the same stream state, a silent semantic difference wearing the same name.
- The in-source rationale argues against pulling `rand` ("pulling `rand` would break the zero-dep posture, so the 8-line SplitMix64 … ships here instead") but **never mentions the in-workspace twin that shipped the day before** — the miss class.
- Both carry reference-vector tests, so each pins its own algorithm bit-exactly; nothing pins the twin-relationship.

## The structural defense (why this is adjudication-gated, not a straight fix)

katgpt-assign's manifest documents a deliberate **zero-dependency posture**: "a pure-combinatorics substrate that sits UPSTREAM of katgpt-core". No cycle blocks `assign → types` (types does not dep assign), but the dep would cost assign's standalone extractability — the crate's stated role.

## Menu

1. **Cross-pin (zero posture change):** keep assign's copy, extend its doc to cite `katgpt_types::rng::SplitMix64` as the twin (and name the `below` divergence), and add a **katgpt-core test under `feature = "assignment"`** (the one place both crates are visible) asserting both structs' `next_u64` streams are bit-identical from the same seed — divergence fails the pin. The copy-gate convention ("a justified copy needs in-source rationale + a divergence-failing test") becomes fully met.
2. **Delegate:** assign deps katgpt-types (one in-repo path dep), re-exports `types::rng::SplitMix64`, keeps `below64`/shuffles as free functions or an extension impl. Costs the zero-dep law.

Classification per the skill: **DRY violation candidate with a real structural defense** — the splitmix family census counted ~20 substrate-side module-local finalizers as below-gate, but this is the first **same-named full pub STREAM pair** in one workspace, and the later arrival postdates the designated home's export.

## Related (same audit, other repos)

- riir-ai Issue 1044 — the same wave's consumer-side splitmix/sigmoid re-statements (hero_progress_drive vs luck.rs; chord_drive vs exact_sigmoid).
- seal-remake Issue 057 — seal-view's three inline splitmix64 constructions (intra-crate DRY).
