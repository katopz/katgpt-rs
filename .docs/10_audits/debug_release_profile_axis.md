# The profile is part of the claim — the dev/release axis (Issues 715 + 716, closed 2026-09-03, files removed 2026-09-03)

Status: **historical record, both CLOSED.** Issue 715 (an orphaned `#[cfg]`
binding across a blank line) is fixed and gated at zero; Issue 716 (the full
gate never ran `--release`) is fixed and the axis is `scripts/full_gate.sh`
**Layer 6**. The one open row — 716 **T3**, whether each of the 19 sibling repos
wants a release pass — is an owner call per repo, same shape as Issue 713 T3.
Recover the narratives with `git log --all -- '.issues/715_*.md' '.issues/716_*.md'`
(last revisions `52eef429`, `4b96e0e6`).

**Commits:** 715 — `26d055c6` (introduced: dropped an import, left its attribute
behind) · `7e34ccef` (erased the evidence by deleting the blank line) ·
`a08376a0` (fix, verified in both profiles) · `52eef429`
(`scripts/orphaned_attr_gate.py`, a `docs_gate.sh` check). 716 — `f0696986`
(two release-only breaks fixed) · `f84f5c7e` (Layer 6 liveness signal) ·
`4b96e0e6` (cost figures qualified as M3 Max numbers).

## Three findings on one axis, pointing in different directions

| record | what the profile did |
|---|---|
| Issue 713 T2b | debug **manufactured** four false perf reds (a latency gate on an unoptimised binary) |
| Issue 715 | debug **hid** a two-day release build break |
| Issue 716 | the full gate compiled `debug_assertions` code **only in the profile where it works** |

**Neither profile is the safe default.** A green result is a claim about its
literal command *including the profile*.

## Issue 715 — a `#[cfg]` separated from its item by a blank line still applies

Rust binds an attribute to the next item **across a blank line**. `26d055c6`
dropped `use std::cmp::Ordering;` from `katgpt-pruners/src/sdar/sdar_absorb.rs`
and left its `#[cfg(debug_assertions)]` behind, where it silently re-bound to
the next import. In every release build the import vanished while its five
usages stayed unconditional: 5 × `E0425`, `katgpt-pruners (lib)` uncompilable
under `sdar_gate`. The introducing commit's own validation line was a **debug**
run. Blast radius audited: `git show -U4` over all seven imports that commit
dropped — only this one had an attribute above it.

**The narrowing is the instrument.** Any attribute + blank line + item is
**2,044** sites across 19 repos and is NOT a defect class (dominated by
whole-file inner `#![cfg(...)]`, which bind to the enclosing module and are
conventionally followed by a blank). Restricting to **outer** `#[cfg]` /
`#[cfg_attr]` takes 2,044 to **0**, so the pin is 0 with no floor to negotiate.
Canaried against the real bug (pre-fix tree reconstructed; reported at
`sdar_absorb.rs:47`, exit 1; green on restore). `selftest()` also asserts the
walk **prunes** `target/` rather than filtering after `rglob` (the 556 s trap).
Found only because Issue 713 T6 insisted on *executing* a target rather than
reading it.

## Issue 716 — the full gate's fifth blind spot

`cargo clippy --workspace --all-targets --all-features --keep-going` runs in the
**dev** profile, so `debug_assertions` is always ON. Measured 2026-09-03 with
`--release`: **2 errors, both `E0432`**, 277 units compiled (not vacuous) —
`latent_confounder_audit.rs` and `orthogonal_factorization.rs` had `#[cfg(test)]`
blocks importing `crate::alloc`'s counters, which `alloc.rs` gates on
`debug_assertions` *by design* and documents at length. **Consequence:**
`cargo test --release -p katgpt-core --lib` did not compile at all — the very
command Issue 713 T2b tells everyone to run.

14 files reference those counters; the compiler proved exactly 2 were wrong. A
`grep -B6` for an enclosing `#[cfg]` missed 7 of the 12 correct ones, so the
narrow instrument would have said "7 more are broken".

**Fixed** by gating both tests `#[cfg(debug_assertions)]`, joining 12 siblings
in the crate already gated that way. **Deliberately NOT** release no-op stubs
returning 0: a zero-alloc assertion against a stubbed counter passes vacuously
(`.issues/705` and Issue 714 exactly). The cost, enumerated by name rather than
inferred from the delta: release runs **26 fewer** katgpt-core lib tests
(4,609 → 4,583) — 6 `alloc.rs` own tests, 14 `g4_*` alloc gates, 6
`*_panics_in_debug`; the reverse set is empty.

### Layer 6 — `cargo check --release`, deliberately not clippy

The axis is **compilation** with `debug_assertions` off; the lint surface is
covered by the dev pass. Not folded into `GATE_ARGS`, because Layer 5 asserts
AGENTS.md quotes that string verbatim. Cost measured before deciding: warm 27 s,
cold 65 s (394 units, 622 MB) — ~8% of a >13 min weekly gate, **both M3 Max /
16-core figures**; a GitHub macOS runner has ~1/4 the cores, so expect minutes.

**Its first in-situ run found a bug in itself.** Counting `Compiling`/`Checking`
lines reported INCONCLUSIVE on a warm tree (cargo compiled 0 units, printed
nothing) — *freshness must not decide a liveness verdict*. Fixed by counting
`--message-format=json` `compiler-artifact` records, emitted for fresh units too
(3 `Checking` lines vs 1,423 artifacts on the same tree), and immune to the
ANSI-colour trap that zeroed every anchored counter in `.issues/705`. Canaried
two-sided on the real tree: 1,423 / 0 → PASSED; one break reintroduced → 1,422
/ 1 → FAILED with the rendered `E0432`; restored → PASSED.

### 716 T3 — MEASURED across 6 siblings (2026-09-03). One repo is RED.

No longer an open owner call in the abstract: the sweep was run. Same literal
command per repo, sequential, each in its own scratch `CARGO_TARGET_DIR`
removed afterwards (the box was at 100% disk when this started):

```
cargo check --workspace --all-targets --all-features --keep-going --release
```

and, wherever release was non-zero, the identical command **without**
`--release` — because an error count alone cannot tell a profile break from a
plain all-features break.

| repo | units (rel) | rel errs | dev errs | REL-only | DEV-only | verdict |
|---|---|---|---|---|---|---|
| riir-neuron-db | 214 | 0 | — | — | — | clean |
| riir-clippy | 530 | 0 | — | — | — | clean |
| riir-game-sdk | 854 | 0 | — | — | — | clean |
| riir-chain | 678 | 12 | 12 | **0** | 0 | all-features break, profile-NEUTRAL — **FIXED** `feadd573` |
| riir-train | 1139 | 30 | 31 | **0** | **1** | profile break, DEV-only |
| riir-ai | 1766 | 282 | 6 | **30** | 1 | **RELEASE BROKEN** |

Counts are raw diagnostics; REL-only / DEV-only are **distinct** `(package,
target, message)` triples, which is the column that means anything — riir-chain's
12-vs-12 is the same six triples twice over, i.e. the profile changed nothing
and its errors are a pre-existing `--all-features` break (`riir_ffi` unresolved
in three `riir-chaind` tests). Filed with its owner as riir-chain `.issues/120`
and **CLOSED there 2026-09-03** (`feadd573`): Plan 040 moved `LatentSidecar`
from `riir-ffi` to `riir-wasm` and missed three files, so the fix was a 12-token
repoint — and it un-hid **8 green tests** (four latency GOAT gates, four
wire-protocol E2Es) that had been unbuildable for 12 days while reporting
`ok. 0 passed`, because the three files are `#![cfg(feature =
"chain_mcp_latent")]` and that feature is default-off. Worth noting *which*
instrument found it: this sweep's contribution was `--all-features`, and Issue
713 T3's `required-features` rows were what turned the resulting
`ok. 0 passed` into an exit-101. Neither alone was enough.

**The axis cuts BOTH ways, which this repo had not yet recorded.** Every prior
finding was release-only. riir-train's single DEV-only error is a
`#[global_allocator]` conflict in `xhc_train_phase7` — visible **only in debug**,
because the allocator katgpt-rs installs is itself `debug_assertions`-gated. So
"run it in release" is not a safe blanket instruction either; it is a second
configuration, not a better one. riir-ai carries the same shape in
`riir-games-quest`'s `issue847_tpr_goat`.

**riir-ai is the find.** 282 release diagnostics against 6 in dev, and the
release run produced **1,364 artifacts to dev's 1,757** — so the release figure
is an *undercount*: `--keep-going` still cannot build a failed unit's dependents.
232 of the 282 are one class, and its mechanism is **verified, not inferred** —
same package selection, both profiles:

```
cargo check -p riir-games-civ --all-features --lib --tests            → 0
cargo check -p riir-games-civ --all-features --lib --tests --release  → 232
```

all of them `katgpt_core::alloc::get_alloc_stats`, which this very document
records as `debug_assertions`-only **by design**. It is Issue 716 T1's exact
class, reproduced in a consumer at 232 call sites in a single crate, and it
means riir-games-civ's alloc assertions can never run in the profile
`.docs/10_audits/cfg_gated_silent_zero_pass.md` tells everyone to run gates in.

The remaining riir-gpu classes are **real but NOT root-caused**, and the
distinction is worth keeping. A first probe (`-p riir-gpu --all-features
--bench …`) returned 0 errors in *both* profiles and looked like a refutation;
it changed **two** variables at once, package selection and profile, so it
refuted nothing. The decisive check was to ask the logs whether those targets
were *built* in the dev run rather than whether they were silent in it:

| target | dev | release |
|---|---|---|
| `bench_734_arm6_mma_cuda_ab` | built ✓ | errored ✗ |
| `bench_734_arm10_mma_v2_ab` | built ✓ | errored ✗ |
| `bench_606_t3c_dot4i8_probe` | built ✓ | errored ✗ |

Silence is not success — a target absent from an error list may simply never
have been attempted. These reproduce only under workspace-wide feature
unification, so their cause is the profile axis *interacting* with AGENTS.md's
third axis (`-p` vs `--workspace` at the same nominal features). Filed for
root-cause with the owner: **riir-ai `.issues/855`** (pointer now historical:
no `.issues/855_*` file exists on develop — recover the disposition with
`git log --all -- '../riir-ai/.issues/855_*.md` if needed; the durable record
is this table).

**2026-09-11 owner call (per-repo T3 disposition, subagent re-verdicted):**
riir-ai's release pass is **DEFERRED, not signed off**. The 232-call-site
`get_alloc_stats` class stays the known, verified surface — the fix (gate the
`riir-games-civ` call sites or route them through the existing gated accessor)
is a riir-ai hygiene item, not a katgpt-rs one. The three riir-gpu
release-only bench errors remain the second class, whose issue home no longer
exists on develop (see above). No other repo's disposition changes: chain
fixed (`feadd573`), train's DEV-only allocator conflict and riir-ai's
`issue847_tpr_goat` twin remain recorded above as the both-ways caveat.

The three clean repos are a real result too, not an absence of one: each
compiled hundreds of units in release with zero diagnostics, so their gates can
be run in the profile the perf rule requires.
