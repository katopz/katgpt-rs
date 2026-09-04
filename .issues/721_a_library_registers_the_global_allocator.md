# Issue 721 — the root crate registers a `#[global_allocator]` **as a library**, so every downstream alloc gate is either a conflict or a silent zero

> **Renumber note (2026-09-03):** originally filed as 720 at `b1340fe8` (20:41);
> a concurrent session dual-allocated 720 sixteen minutes later (`a119bb0a`,
> the convergence-cadence gate — which read `.highwater` before the first
> bump landed, or skipped the write-time re-scan). Tie-breaker per the
> Bench-677/697 + Issue-087 precedents: baked-refs wins — the cadence issue
> carries 5 live committed references (Research 529 ×3, riir-ai 858 ×2) vs
> zero for this one, so IT keeps 720 and this file renumbered → 721
> (highwater+1, highwater bumped). No external file referenced this issue at
> renumber time — the cheapest possible moment to move it.

**Status:** OPEN — T1/T2/T4 DONE 2026-09-03 (audit + full sentinel sweep + the xhc_train_phase7 interim, riir-train `e2373492`); **T3 remains: the owner sequencing call** (move the registration out of `src/lib.rs`). T2's precondition is discharged. Found 2026-09-03 while closing riir-ai `.issues/855` Class 3. **Owner sequencing decision 2026-09-04: T3 APPROVED as an opportunistic move, not a standalone campaign** — the registration moves out of the library when each dependent target is next touched for real work (the per-target list below is the checklist); until then the interim state stands, which is deliberately safe-and-loud (the xhc guard + the sentinel sweep make any new consumer conflict a compile error naming the fix, never a fabricated zero). Do not re-sweep or re-audit for T3's sake; the guard fires first.

## The defect

`src/lib.rs:257`:

```rust
#[cfg(debug_assertions)]
#[global_allocator]
static GLOBAL_ALLOC: katgpt_core::alloc::TrackingAllocator = katgpt_core::alloc::TrackingAllocator;
```

That is a **library** choosing the process allocator for every binary that
links it. Rust permits exactly one `#[global_allocator]` per binary, and the
choice belongs to the binary crate. `katgpt-core` gets this right two lines of
reasoning away — its own copy is `#[cfg(all(test, debug_assertions))]` and its
comment says why:

> Downstream consumers (katgpt-rs root, riir-engine, etc.) install their OWN
> `#[global_allocator]`; this static is `cfg(test)` so it does not exist when
> katgpt-core is consumed as a library — no double-declare conflict.

The root crate's is **not** `cfg(test)`, and cannot be: integration tests link
the lib as a dependency, where `cfg(test)` is false, and
`tests/kimi_k3_g4_alloc_free.rs` documents that it relies on exactly this
static. So the current shape is load-bearing *and* wrong, which is why it has
survived.

## What it costs downstream — two failure modes, opposite directions

**1. A conflict, visible only in debug.** Any downstream test that installs its
own allocator fails to compile — but only when `debug_assertions` is on AND the
feature that pulls the root crate is enabled. Measured instances:

| repo | target | shape |
|---|---|---|
| riir-ai | `riir-games-quest/tests/issue847_tpr_goat.rs` | FIXED in `b35f8b901` (riir-ai `.issues/855` Class 3) |
| riir-train | `riir-train-engine/tests/xhc_train_phase7.rs` | OPEN — identical shape, `#[cfg(debug_assertions)] #[global_allocator]` at line 320 |

The riir-ai fix guards on `not(feature = "quest_compression_draft")` because
that repo has exactly ONE feature pulling `dep:katgpt-rs`. **That fix does not
generalise:** `riir-train-engine` has at least five (`kimi_k3_train`,
`go-latent-steering`, `go-data-tools`, `bonsai-go`, and the `katgpt-rs/go`
edge at line 1122), so the equivalent guard is a five-term disjunction that
goes stale the next time someone adds a feature. That is the argument for
fixing it at the source rather than once per consumer.

**2. A silent zero, in every profile.** A consumer that does NOT install one
and does NOT link the root crate calls `get_alloc_stats()` against no tracking
allocator and gets a real `0` — indistinguishable from an allocation-free hot
path. riir-ai `.issues/856` was exactly this (a release stub returning a
fabricated `0`); this is the unobserved-zero sibling of it.

## Blast radius — measured 2026-09-03, not estimated

Across the 4 repos that consume `katgpt_core::alloc::{get,reset}_alloc_stats`:

```
consumers = 67    register their own = 49    RELY ON THE LIBRARY'S = 18
```

Per-FILE grep, so it under-counts consumers whose allocator lives in a
`tests/common/` module linked into the same target — treat 49 as a floor and
18 as a ceiling. Of the 18, most are `#[cfg(test)]` blocks inside
`katgpt-core/src/*.rs`, which are served by katgpt-core's own `cfg(all(test,
debug_assertions))` static and are unaffected. The ones that are not:

- `katgpt-rs/examples/kimi_k3_hello_world.rs` — links the root lib, so it is
  served by the static this issue proposes to remove.
- `riir-ai/crates/riir-poc/src/behavior_gate_poc.rs`
- `riir-ai/crates/riir-games-civ/src/civ/map_tick/mod.rs`
- `riir-train/crates/riir-train-gpu/tests/bench_558_issue490_t2_incremental_staging_goat.rs`
- `riir-train/crates/riir-train-gpu/tests/bench_490_anchor_accumulation_goat.rs`

**Those last four are the ones to check FIRST, and not because of this fix.**
A unit test inside `riir-games-civ` linking `katgpt-core` as a plain dependency
gets **no** tracking allocator today — katgpt-core's is `cfg(test)` on
*katgpt-core*, not on the consumer. If any of them asserts `allocs == 0`
without installing an allocator, it is passing over nothing **right now**, and
removing the root static changes nothing for it.

## Tasks

- [x] **T1** — audit the five files above: does each install an allocator
      (directly or via a linked `tests/common`), and does each assert on the
      counter? Any that asserts without one is a live vacuous gate, independent
      of T2/T3, and should be fixed first. *(DONE 2026-09-03 — see the audit
      table below.)*
- [x] **T2** — the non-negotiable half, cheap and independent of the rest:
      every consumer that asserts on `get_alloc_stats()` gets a **liveness
      sentinel** — force a known heap allocation, assert the counter saw it,
      *before* the measurement. `riir-games-quest/tests/issue847_tpr_goat.rs`
      (riir-ai `b35f8b901`) is the reference implementation. This makes both
      failure modes loud and is worth doing even if T3 never happens.
      *(CLOSED 2026-09-03 — full-workspace sentinel sweep, see the T2 audit
      table below. Verdict: every asserting consumer is now covered by a
      sentinel, a self-protecting nonzero assert, or a stronger type-level
      mechanism; the one true gap was xhc_train_phase7, fixed under T4 the
      same day, riir-train `e2373492`.)*
- [ ] **T3** — owner call on sequencing: move the registration out of
      `src/lib.rs` and into each target that needs it. Correct, and the only
      thing that removes the conflict class rather than guarding it per
      consumer. Blocked on T2 across all repos first — without sentinels, T3
      converts a compile error into a silent zero, which is strictly worse
      and is the trade riir-ai `.issues/856` already refused once. *(T2 now
      closed — the blocker is discharged; awaiting the owner's sequencing
      call.)*
- [x] **T4** — `riir-train-engine/tests/xhc_train_phase7.rs` needs *something*
      today; it is the one known live conflict. If T3 is deferred, it needs the
      five-term feature guard plus a sentinel. Prefer T3. *(DONE 2026-09-03 as
      the sanctioned interim — riir-train `e2373492`: the static guards on
      `not(any(kimi_k3_train, go_training_arena, go-latent-steering,
      go-data-tools, bonsai-go))` (grep-verified as the manifest's COMPLETE
      `dep:katgpt-rs` set — L472/1122/1524/1534/1535), and the G4 test gained
      the liveness sentinel. Validated BOTH arms: own-static arm G4 PASS;
      kimi_k3_train arm COMPILES (was the duplicate-allocator error) + G4
      PASS under the root-lib static. The guard is T3-fodder — when T3 lands
      it is deleted with the root static.*

## T1 audit — DONE 2026-09-03, one gap found + fixed

| # | file | installs an allocator? | asserts on the counter? | verdict |
|---|---|---|---|---|
| 1 | `katgpt-rs/examples/kimi_k3_hello_world.rs` | via the root-lib static (debug) | **no — print-only**, honest in BOTH profiles (release prints the "not measured" notice, never a zero-alloc claim) | fine; T3 consequence only |
| 2 | `riir-ai/crates/riir-poc/src/behavior_gate_poc.rs` `g4_alloc_steady_state` | **via the root-lib static** (riir-poc deps the katgpt-rs ROOT crate — line 150; see the corrected finding below) | artifact half: `assert!(first > 0)` sentinel (loud red without allocator); **greedy half: NO sentinel** — `assert_eq!(gfirst, gsecond)` passes vacuously at 0==0 | **gap → FIXED** (greedy-half sentinel added, riir-ai commit) |
| 3 | `riir-ai/crates/riir-games-civ/src/civ/map_tick/mod.rs` | n/a | **no — diagnostic, not a gate**: `trace_phase!` asserts nothing; release fallback reports `None` + one-time notice, never a fabricated 0 (explicitly documented in-source, 855 Class 1) | fine |
| 4 | `riir-train-gpu/tests/bench_558_issue490_t2_incremental_staging_goat.rs` | served via the katgpt-rs shim under the kimi features | `assert!(joint_allocs > 0)` sentinel present | fine |
| 5 | `riir-train-gpu/tests/bench_490_anchor_accumulation_goat.rs` | served via the katgpt-rs shim under the kimi features | `assert!(legacy_allocs > 0)` sentinel present | fine |

**The environment finding (row 2) — CORRECTED 2026-09-03, second pass:** the
first pass claimed riir-poc never links an allocator. **Wrong** — the audit's
own grep was truncated (`head -5`) and missed line 150 of riir-poc's
Cargo.toml: `katgpt-rs = { path = "../../../katgpt-rs", default-features =
false, features = ["dllm", "flashar_consensus", "bandit", "ropd_rubric"] }`
— a NORMAL dep on the ROOT crate. So the root static DOES install in the
riir-poc debug lib-test binary, the counters ARE live, and
`g4_alloc_steady_state` is runnable today (the exact same serving shape as
row 1). This is the katgpt-rs Issue 724 Phase 1 lesson re-hit: an unanchored
(or here, truncated) dep-line grep misattributed the dep graph. **Consequence
for T3:** riir-poc is a root-static consumer — when the registration moves
out of the root lib, this gate reds (both halves, loudly, thanks to the
sentinels) until riir-poc installs its own per-target allocator. That is the
designed T3 failure mode, not a defect. The grep lesson generalizes: classify
dep edges with `grep -n "^<crate>\s*="` (anchored, untruncated), never a
name substring with a head limit.

**The fix shipped (row 2):** a greedy-half liveness sentinel before the
equality asserts (riir-ai `crates/riir-poc/src/behavior_gate_poc.rs`):

```rust
assert!(
    gfirst > 0,
    "greedy-decode liveness: counters saw 0 allocations — is a TrackingAllocator installed in this test binary?"
);
```

so the T2 doctrine ("force a known allocation, assert the counter saw it,
BEFORE the measurement") now covers BOTH halves of that gate, and weakening
the artifact half can no longer silently re-vacuous the greedy half.

## T2 sweep audit — DONE 2026-09-03 (the full consumer set, not just the 5 root-static files)

Method: `grep -rl get_alloc_stats` across riir-ai/crates, riir-train/crates,
riir-neuron-db, katgpt-rs {tests,crates,examples,src} minus the
sentinel-vocabulary files, then per-file reads of every remaining candidate.
Verdict per candidate:

| file | verdict |
|---|---|
| riir-games-quest `issue847_tpr_goat.rs` | **the T2 reference impl itself** — guard + LIVENESS SENTINEL + doctrine text |
| riir-engine `cgsp_runtime/evpi_gate.rs` tests | covered — Issue 830 probe ("exact counts when live, capacity witness when not") |
| riir-games-civ `tests/common/alloc_delta.rs` + its 13 targets | covered STRONGER — Issue 856 `AllocCount` newtype (invalid state unrepresentable; release loudly "NOT EVALUATED", never a fabricated 0) + `extern crate katgpt_rs` force-link doctrine |
| riir-poc `benches/archetype_trajectory_consolidation_poc.rs` | covered — force-link + honest `None` release policy |
| riir-train-gpu `bench_490` / `bench_558` / `probe_684` | covered — `> 0` asserts ARE sentinels (inert-0 reds loudly) |
| riir-agents `goat_phase6_g2_g4.rs` | covered de facto — exact-count assert against a NONZERO expectation (`count == 2*MEASURED`) + byte floor red at inert-0 |
| katgpt-rs tests (`bench_271/272/274/275/280`, …) | covered — `assert_alloc_tracking_live` helpers (Issue 682) |
| riir-neuron-db `freeze_lineage_gates.rs` / `steering_g5_zero_alloc.rs` | covered — sentinels (Issue 604) |
| katgpt-core `src/*.rs` cfg(test) blocks (incl. the new `convergence_cadence.rs`) | house style — served by the crate's own `cfg(all(test, debug_assertions))` static; the release-profile axis is the documented Issue 716/855 territory, not a per-test gap |
| `examples/kimi_k3_hello_world.rs`, riir-games-civ `map_tick/mod.rs` | audit rows 1/3 — print-only / diagnostic, honest in both profiles |
| riir-train-engine `xhc_train_phase7.rs` | **THE GAP → fixed** (T4, riir-train `e2373492`): equality asserts `c1==c2`/`b1==b2` passed vacuously at 0==0 under inert linkage, and the own static conflicted under the 5 root-pulling features |

Also worth noting: the idle-loop sibling landed the riir-poc
`behavior_gate_poc.rs` greedy-half sentinel (row 2 of the T1 audit) in
riir-ai in parallel — the last open sentinel gap from the T1 table.

**T4 guard-staleness note (the issue's own prediction, now written down):**
the five-term `not(any(...))` disjunction goes stale the next time a
riir-train-engine feature starts pulling `dep:katgpt-rs`. The detector is
cheap and mechanical: `grep -n "dep:katgpt-rs\|katgpt-rs/"
crates/riir-train-engine/Cargo.toml` must return exactly the five guarded
lines (472/1122/1524/1534/1535) + the two dep rows (37/1599). A new feature
edge that misses the guard re-creates the duplicate-allocator compile error
— LOUD, the failure direction this issue prefers — and the fix is one more
`feature = "…"` term in the disjunction until T3 deletes the whole shape.

## Related

- riir-ai `.issues/855` Class 3 — the first instance, fixed `b35f8b901`
- riir-ai `.issues/856` — the fabricated-zero sibling (release stub returning 0)
- `.docs/10_audits/debug_release_profile_axis.md` — why this is invisible in one profile
- `.docs/10_audits/cfg_gated_silent_zero_pass.md` — the PROFILE dimension; the 3 debug-only
  load-bearing targets it reports are all alloc gates, i.e. this same tension
