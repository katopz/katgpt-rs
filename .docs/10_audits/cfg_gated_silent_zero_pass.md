# `#![cfg]`-gated targets that report a green `0 passed` — measurement record (Issue 713, closed 2026-09-03, file removed 2026-09-03)

Status: **historical record.** Every katgpt-rs task of Issue 713 landed; the
one remaining row (**T3**, arming sibling repos' load-bearing targets) is an
owner call per repo and is carried here as the table those owners should work
from. Recover the full narrative with `git log --all -- '.issues/713_*.md'`
(last revision `651f118a`).

**Fix commits:** `4509b7d8` (instrument + first measurement) · `a2c8aa71`
(over-count by 48 fixed, pinned) · `180be9c5` + `1e4a52a` (T2: 39 katgpt-rs
gates armed) · `b59b20b3` → `83cb1d56` (T2b run, then its perf reds
**retracted** — sweep was in debug) · `6406f710` (T4: `cfg_gated_floor_gate.py`
+ the trigger-list fix) · `52eef429` (T6: `all_ignored_target_audit.py`; found
Issue 715) · `f84f5c7e` (T6 pin: reasonless `#[ignore]` at 0) · `345e301f` (T5:
the 21 "platform" targets were three classes) · `2272b262` (T4c: classifier
token set widened, 17 more targets armed) · `651f118a` (T3 table re-measured).
Same defect fixed one target at a time earlier: riir-train `5821cba9`,
riir-clippy `19beece`.

**Instruments (all still live):** `scripts/cfg_gated_target_audit.py` (report),
`scripts/cfg_gated_floor_gate.py` + `scripts/cfg_gated_floors.txt` (the gate,
a `docs_gate.sh` check), `scripts/all_ignored_target_audit.py` (report).

## The PROFILE dimension — a third kind, added 2026-09-03 (riir-ai `.issues/855` Class 2)

The report modelled two kinds of gate: **feature**-expressible (fixable with a
`required-features` row) and `target_os` / `miri` / `unix` / … (which cargo
cannot express). `debug_assertions` sat in the second list, pooled with the
platform predicates, and that pooling hid the worst case in the set.

**It is not a third bucket, it is a second DIMENSION**, and it is reported as
one — overlapping every existing class, `covered` included, so the partition
assertion is untouched. Two properties make it different from everything else
in `NON_FEATURE_PREDICATES`:

1. **Every other predicate is silent only in a configuration somebody CHOSE.**
   `target_os` needs the wrong platform; `miri` needs miri; a default-off
   feature needs `--no-default-features` typed. `not(debug_assertions)` is
   silent under **`cargo test`** — the default invocation, on the right
   machine, with no flags.
2. **It survives the fix.** Adding a `required-features` row moves a target
   into the `w/ req-f` column, which reads as protected. A profile-gated target
   is still a green zero in dev afterwards. Measured: **11 of the 133** already
   have their row.

### Read the DIRECTION, never the pooled total

Pooling the two halves would repeat the exact error this report already
documents for platform gates one axis over — they differ by the single token
`not(` and are opposite in severity. Measured over all 19 contract repos:

| shape | count | load-bearing | what it means |
|---|---:|---:|---|
| `not(debug_assertions)` | **130** | 26 | RELEASE-only: green zero on plain `cargo test` |
| `debug_assertions` | **3** | **3** | DEBUG-only: green zero under `--release` |

130 of the 133 are riir-ai, overwhelmingly `riir-gpu/tests/bench_*_g1.rs` —
GPU benches deliberately release-gated, which is defensible. What is not
defensible is that they are invisible to every dev-profile gate, which is how
riir-ai `.issues/855` Class 2's 18 release diagnostics sat unnoticed.

**The 3 debug-only ones are the sharper finding: all three are load-bearing,
and all three already carry `required-features`.** They are
`kimi_k3_g4_alloc_free.rs`, `steering_g5_zero_alloc.rs` and
`probe_684_topology_hotpath_alloc.rs` — alloc gates, necessarily debug-only
because the counters are. They vanish **exactly when someone follows
`.docs/10_audits/debug_release_profile_axis.md`'s rule to run gates in release**. The instruction and
the gate are in direct conflict, and nothing said so.

The fix pattern for that conflict is riir-ai `.issues/855` Class 3's: an alloc
budget and a wall-clock budget cannot be asserted in the same test, because no
single profile can observe both. Split them, `#[cfg_attr(…, ignore)]` each into
its own profile, and never stub the absent half to a fabricated zero (Issue
856). Measured there: 3 ms/cycle in debug against 14.35 µs in release, a 209×
profile artefact on a bar the code clears by 70×.

### How it is detected

`NOT_DEBUG_ASSERTIONS` matches the parenthesised form, so
`not(all(debug_assertions, …))` and a bare `debug_assertions` elsewhere in the
same cfg cannot be confused. `selftest()` pins it in **both** directions plus a
`not(target_os = …)` non-claim, because both failure modes are silent: a
matcher that stops seeing the term takes the column to a confident zero, and
one that fires on any file mentioning it makes the column unreadable.


## The shape

A test file opening with `#![cfg(feature = "x")]` compiles to an **empty
binary** when `x` is off. Cargo prints `running 0 tests` / `ok. 0 passed` and
**exits 0** — byte-for-byte a real pass. `required-features` changes that to
`error: target … requires the features: x` (exit 101). **The `#![cfg]` protects
the COUNT; `required-features` protects the READER.** Both are needed.

Verified by execution, not inferred: riir-dapps `content_vessel` (a GOAT gate)
and riir-game-sdk `game_anticheat` both printed a green zero; on the fixed side
riir-clippy `t063_tpr_structure` errored without its feature and ran `1 passed`
with it.

## Measured — 19 repos, derived population (corrected figures)

| | count |
|---|---|
| targets scanned | 2,897 |
| carrying a whole-file `#![cfg]` | 1,740 |
| declaring `required-features` | 1,016 |
| **SILENT-NOW** — zeroes on a plain `cargo test` | **382** |
| latent — zeroes only under `--no-default-features` | 320 |
| platform `cfg` (cannot be `required-features`) | 21 |
| `any(...)` of features (cargo is AND-only) | 1 |

The five classes **partition** the 1,740 — and the partition held under a
*wrong* classifier (the first cut read 430 SILENT-NOW), so it proves the classes
are exhaustive, not that each is right. **Read the severity split, never the
pooled total:** default-on-gated targets still run on a plain `cargo test`;
default-off-gated ones report a green zero every time anyone names them.

## The lessons that generalise

1. **The auditor over-counted by 48** by keying declared targets on `name`
   against the filename stem, so a `[[test]]` row with an explicit `path` under a
   different name read as undeclared. "Fixing" those added a SECOND target for
   the same file. Found because the T2b sweep returned `NO-RESULT-LINE` on
   exactly those four. Pinned in `selftest()` with a temp crate.
2. **Run perf gates in `--release`.** T2b's first pass reported four reds and
   nearly filed two as perf regressions after three agreeing quiet re-runs
   (`fast_bpe_goat`: 388 s debug vs 15.6 s release). Ruling out the confounder
   you thought of says nothing about the one you didn't. All 39 pass in release.
   The one real find was Issue 714 (an alloc gate counting a sibling test),
   which reproduces in release — see
   [`alloc_gate_per_thread_counter.md`](alloc_gate_per_thread_counter.md).
3. **Two-sided pins.** `cfg_gated_floors.txt` carries ceilings
   (`max_load_bearing = 0`, `max_silent_now`, `max_reasonless_ignores = 0`) AND
   blindness floors (`min_targets`, `min_gated`, `min_ignore_targets`): a
   ceiling alone cannot fail once the instrument goes blind and reports zero.
   Canaried with a throwaway unarmed `*_goat.rs` (red on both ceilings, green
   on removal) and with the all-zeroes synthetic report (must fail both floors).
4. **The trigger list was the real hazard.** `docs_gate.yml`'s `paths` filter
   carried no `.rs` glob, so the gate could not have fired on the one push it
   exists for. Both hand-duplicated lists widened and asserted identical.
5. **A zero ceiling is only as wide as its classifier's vocabulary.** T4b built
   the load-bearing token matcher independently of the published ad-hoc grep;
   five of six disagreements were the matcher's misses (`g16f`/`g2p`/`g9gov`
   variant suffixes, the plural `drills`, the compound `regate`). Agreement
   licensed the pin. **T4c then showed they agreed on the wrong population:**
   the set did not know `*_correctness` / `*_alloc_check` / `*_determinism` /
   `*_equivalence` / `*_floor` / `*_grad_check`. Every candidate token was
   measured against all 2,157 workspace target names first:

   | token | newly classified | verdict |
   |---|---|---|
   | `alloc` | 36 | added (`*_alloc_check` is G4 here) |
   | `floor` | 15 | added (Report-the-Floor mandate) |
   | `grad` | 9 | added |
   | `determinism` | 6 | added |
   | `correctness` | 5 | added |
   | `equivalence` | 3 | added |
   | `soundness` | 1 | added |
   | `budget` | 5 | **rejected** — admits a sweep and a config |
   | `calibration` | 1 | **rejected** — names a record, not an assertion |
   | `check` | 13 | **rejected** — admits any smoke test |

   17 more katgpt-rs targets appeared, were armed, and ran in release: 45/45.
   Silently *unverified*, not broken. **Re-run this table when a new naming
   dialect appears** — a vocabulary gap is indistinguishable from a clean repo.
6. **T5:** the "21 platform-cfg targets" were 11 `not(wasm32)` (the inverse of a
   hazard), 8 `#![cfg(test)]` on integration targets (a **no-op**, confirmed by
   execution — cargo passes `--test`), and **2** riir-ai `wasm32`-only targets
   that no CI platform compiles. No new instrument needed; a two-row riir-ai
   owner call.
7. **T6, the second silent-zero shape:** a target whose every test is
   `#[ignore]`d prints `ok. 0 passed; N ignored` — and `required-features`
   cannot touch it. 244 such targets across 19 repos, 60 load-bearing by name,
   two of them (`bench_octopus_goat`, `bench_block_diagonal_goat`) not
   feature-gated at all. Deliberately a **report**: `#[ignore]` is the correct
   marker for a slow or hardware-gated test. The one defensible pin is *whether
   the source says why*: 27 reasonless `#[ignore]`s in katgpt-rs were given
   their file's already-documented reason (none invented), pinned at 0. The
   instrument had to be built twice — per-test attribute blocks resolved against
   the default closure, with an unsatisfied `any()` whole-file gate reported as
   its own ambiguous class rather than guessed. Executing `test_120_vpd_arena_goat`
   to confirm its count is what found Issue 715's two-day release break.

## T6 follow-up (2026-09-04) — the count moved and nothing said so

Item 7 above records "60 load-bearing" and `scripts/cfg_gated_floors.txt`
recorded "19 ALL-IGNORED (3 load-bearing)" for katgpt-rs. Re-measured the
next evening: **66 workspace-wide, 5 in katgpt-rs.** Neither number was
pinned, so neither moved.

The two katgpt-rs arrivals are not new `#[ignore]`s. They are
`velocity_field_disagreement_uq_floor.rs` and
`velocity_field_ensemble_uq_floor.rs`, which became load-bearing when T4c
widened `is_load_bearing` to the `floor` dialect — **the instrument working,
with nothing to report that it had.** Same shape as the T4c finding itself,
one axis over: the classifier got wider and the prose describing its output
did not.

### Both `*_uq_floor` rows were checked against their sources, and both are correct

They are 0-assertion, 41-println comparison tables — a "Report the Floor"
(Issue 010) *pre-validation*, not a gate — and neither rests a promotion on
an unasserted verdict:

| target | feature state | recorded verdict | promotion |
|---|---|---|---|
| `velocity_field_ensemble_uq_floor.rs` | `velocity_field_ensemble` **default-on** | `.benchmarks/376_uq_floor.md`: **BEATS FLOOR** on AR(1) | legitimate — Bench 376 states the primitive "currently makes NO UQ claim … this benchmark is a pre-validation, not a claim addition", so the UQ rule is not the gate its promotion rested on |
| `velocity_field_disagreement_uq_floor.rs` | `velocity_field_disagreement` **opt-in** | `.benchmarks/432_vfd_uq_floor.md`: **G2 FAILS** for the epistemic-UQ claim (λ\*=0 on both corpora; the AR(1) `BeatsFloor` is inherited from the ensemble's point forecast) | correct — Plan 432 T3.2 promotion **NOT EXECUTED**, ships as an opt-in non-UQ disagreement score |

The wider conformal-floor suite is genuinely armed — **11
`crates/katgpt-core/tests/conformal_*.rs` files, 50 tests, 108 assertions, and
exactly ONE `#[ignore]`** (in `conformal_karc_no_regression.rs`, i.e. a
`partial` row, not an ALL-IGNORED one). The two print-only tables are the
exception, and the exception is documented in both plans. Same conclusion as `.issues/714` T3
and T4c: **silently unverified, not silently broken.**

The other three katgpt-rs rows are equally justified, read out of their own
`#[ignore = "..."]` strings: two "pure measurement benchmark (no assertions),
slow in debug", and `test_120_vpd_arena_goat` at "1000 Bomber games per test —
minutes, not seconds".

### The pin: MEMBERSHIP, because the count genuinely is not gateable

`cfg_gated_floors.txt` argues that the ALL-IGNORED **count** cannot be gated —
`#[ignore]` is the right marker for a slow or hardware-gated test — and that is
correct. It is also incomplete: **a set is gateable where its cardinality is
not.** `scripts/all_ignored_load_bearing.txt` lists the five paths with each
one's own reason string, and `check_membership` in `cfg_gated_floor_gate.py`
reds on drift in *either* direction (a removal is good news and still needs the
pin updated — `scripts/repo_set.txt` discipline).

Three properties a count ceiling does not have, all canaried before landing
(arrival → exit 1, removal → exit 1, empty allowlist → exit 1, restored → 0):

1. A **sixth** row arriving reds the push that adds it, before its green
   `ok. 0 passed` is cited as promotion evidence.
2. A same-size **swap** fails. That is the AGENTS.md repo-set incident's lesson
   applied one axis over, and it is pinned as its own selftest case.
3. It is **self-protecting against blindness**: if the auditor's `#[test]` or
   `#[ignore]` recogniser regresses, the measured set empties and the gate reds
   on five removals. Every ceiling in this family passes when the instrument
   dies; this one cannot. An empty *allowlist* is refused for the mirror-image
   reason — otherwise a truncated pin file and a broken auditor would cancel.

Not added to `REQUIRED_PINS`: it is not an integer, and the int-pin selftest
drives every pin over a boundary a set does not have.

## T3 — DONE across every contract repo (2026-09-03). Load-bearing zero is now real.

No longer an owner call pending per sibling: the sweep was taken. **97 targets
armed across 6 repos**, every one classified LOAD-BEARING by
`scripts/cfg_gated_target_audit.py` — a name carrying `goat` / `gate` / `g<N>` /
`drill` / `proof` / `correctness` / `alloc_check` / `determinism` /
`equivalence` / `floor` / `grad_check`, i.e. a promotion or a safety claim rests
on its verdict.

| repo | load-bearing before | after | commit |
|---|---|---|---|
| riir-ai | 40 | **0** | `e1fa3858f` |
| riir-clippy | 18 | **0** | `883d536` |
| riir-chain | 16 | **0** | `0da394c9` |
| riir-train | 15 | **0** | `f998d552` |
| riir-game-sdk | 5 | **0** | `aa5d8af` + `e160e34` |
| riir-neuron-db | 3 | **0** | `6839a3b` |
| katgpt-rs | 0 | **0** | `180be9c5` (earlier) |

SILENT-NOW does **not** go to zero and should not: the residual 233 are targets
whose green zero nobody cites as evidence. Arming those is churn. The column
that mattered was always the load-bearing one.

**Verified, not assumed** — the two failure modes here are symmetric, and both
are silent. A row naming the wrong feature *keeps* the green zero; a row naming
a feature that merely compiles the file could still gate nothing.

1. **The row fires.** All 97 error with exit 101 and `requires the features: …`
   when named without them. The one apparent miss was instructive rather than
   real: `riir-chain-engine-bridge` is a **workspace-`exclude`d** crate, so
   `-p <name>` reports "did not match any packages" — the same rc=101, a
   completely different reason. Re-verified via its own manifest path. An
   exit code is not a diagnosis.
2. **The row names features that arm the gate, not features that silence it.**
   One armed target per repo re-run WITH its features in `--release`:

   | repo | target | result |
   |---|---|---|
   | riir-neuron-db | `lifecycle_goat_gates` | 3 passed |
   | riir-clippy | `sec_boundary_gate` | 2 passed |
   | riir-chain | `sufficiency_gates` | 12 passed |
   | riir-train | `schur_correctness` | 5 passed |
   | riir-ai | `rrf_fusion_goat` | 6 passed |

   Non-zero and green in every case — which is the whole point, since a
   *zero* here would mean the row had swapped one silent pass for another.

**Safety, checked before the batch rather than after:** this cannot red an
existing CI. `cargo test --workspace` silently SKIPS a target whose
required-features are off. riir-chain was the sharpest case — 9 of its 16
targets are also named explicitly in `scripts/test_gate.sh` rows carrying pinned
counts, and every one of those rows already passes a feature set that is a
superset of the row now declared, so no pinned count moves.

**One manifest was committed as a blob, not a worktree add.** A concurrent
session had an uncommitted `game_budgets` row in
`riir-game-sdk/crates/riir-e2e/Cargo.toml`. The commit is HEAD + my four rows
(`git hash-object -w` + `git update-index --cacheinfo`); their row stays
uncommitted and the worktree carries both, so nothing of theirs was swept and
nothing of mine reverts when they commit. Their own comment in that file had
explicitly deferred these four to this sweep.

## Why this is not `feature_isolation_gate.py` or `ci_feature_guard.sh`

Both mention `required-features` only incidentally; neither asks whether a
`#![cfg]`-gated target declares one. Related family: `.issues/705` (a gate that
passed over zero compiled units) and `.issues/706` (a compile surface in a
workflow nothing started) — **an instrument that cannot fail is not passing.**
