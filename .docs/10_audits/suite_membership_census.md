# Suite membership: which test targets does anything name?

**Instrument:** `scripts/suite_membership_audit.py` (report, always exit 0;
`--selftest` pins the classifiers). First census 2026-09-04.

## The axis

Three standing reports already cover adjacent questions and this one completes
the set:

| question | instrument |
|---|---|
| does CI gate the compile/lint surface? | `ci_gate_coverage.py` |
| does CI EXECUTE tests (vs compile them)? | `ci_test_execution_report.py` (per-REPO verdict) |
| **is a specific test target named by ANY committed suite?** | **this report** (per-TARGET) |

The motivating class is the recurring ad-hoc "gate nobody runs" find:
riir-ai 865 (a default-feature GOAT gate executable by nothing automatic for
55 days — found because it FAILED in a sweep), riir-ai 868 (all three
`goat_290_*` targets named by no pinned suite — found by a hand grep),
riir-chain 09-03 (`chain_engram_commit` suite-less until a percentile
per-site read touched it). Each was discovered by accident; this report makes
the census one command.

## Method (and its honest approximations)

- Population derived (`BOUNDARY.md` + `.git` walk), never typed. Test targets
  only (`[[test]]` explicit + implicit `tests/*.rs`); benches excluded on
  purpose (834's standing skip class).
- Pin detection = target-name substring over the repo's `scripts/` +
  `.github/` text (the faithful systematization of the 868 hand grep).
  **"Pinned" reads as "named", not "run"** — the corpus deliberately includes
  prose/comments (documented intent names a target), so a floors `.txt` or an
  audit script's docstring counts as naming. The RUN question is answered
  separately by the broad-run detection below.
- A BROAD run (`cargo test` with no `--test` filter, no `--lib` suppressor)
  executes every default-compiled integration target without naming any, so
  unpinned-by-name only implies never-executed for repos with NO broad run.
  Real-invocation extraction is imported from `ci_test_execution_report`
  (quoted-label stripping + `$()` lifting); the integration-filter layer
  (`--tests` broad, `--lib`/`--bins`/`--doc` never execute integration
  targets) is this report's own. Shell backslash continuations join before
  classification (riir-ai's Layer-1.16 named `--test goat_290_*` filters live
  on continuation lines).
- Load-bearing split imports `is_load_bearing` from `cfg_gated_target_audit`
  — one committed classifier across both reports.
- ACTIONABLE = load-bearing + unpinned + default-visible (no
  required-features) + no broad run in the repo. Feature-gated opt-in gates
  are the standing expected state (only named feature runs execute them;
  arming each is the 868-bespoke-layer decision, owner's call per gate).

## First census (2026-09-04, Windows 4090 box — 10 of 16 contract repos present)

| repo | targets | pinned (named) | sibling-named | UNPINNED | LB unpinned | broad run | ACTIONABLE |
|---|---:|---:|---:|---:|---:|---|---:|
| katgpt-rs | 652 | 30 | 0 | 622 | 402 | NO | 125 |
| riir-ai | 889 | 34 | 1 | 854 | 369 | YES | 0 |
| riir-chain | 205 | 128 | 3 | 74 | 44 | YES | 0 |
| riir-clippy | 87 | 4 | 10 | 73 | 34 | YES | 0 |
| riir-dapps | 30 | 1 | 2 | 27 | 1 | YES | 0 |
| riir-game-sdk | 64 | 50 | 0 | 14 | 4 | YES | 0 |
| riir-neuron-db | 51 | 0 | 1 | 50 | 18 | YES | 0 |
| riir-train | 458 | 6 | 31 | 421 | 221 | NO | 221 |
| riir-viewbridge | 4 | 0 | 0 | 4 | 0 | YES | 0 |
| seal-remake | 5 | 3 | 0 | 2 | 2 | YES | 0 |

Totals: 2445 targets / 2141 unpinned / 1095 LB unpinned / **346 actionable**.

## Reading the 346 — recorded populations, not new findings

Both actionable repos ALREADY document exactly this state:

- **katgpt-rs (125):** the AGENTS.md full-gate block and Issue 723 record that
  the scoped executed core is `katgpt-rs + katgpt-core --lib` (Issue 718
  T3b), that 477 integration targets + 176 benches "remain executed by
  nothing automatic", and that the full-workspace `--all-features` run is
  PRICED but not a supported TEST configuration (per-feature fixture RNG/GOAT
  calibrations; 45 reds in six classes await per-target triage pins). The
  triage is 723's — sibling-owned; do not collide.
- **riir-train (221):** Issue 507's CI gate documents the same split
  explicitly ("Explicitly NOT covered": 302 of 487 targets cfg-gated, 277
  carrying required-features, the few default-runnable ones including
  multi-minute single tests — `goat_247` 433 s — priced out of a weekly
  gate).

The load-bearing column (1095) is NOT actionable noise to burn down: most are
opt-in feature gates whose GOAT verdicts were measured at landing and recorded
in `.benchmarks/`. The 865/868 lesson is not "pin everything" — it is that a
gate nobody names is invisible to every sweep, so NEW arrivals should be
checked here (or armed via a suite row) at landing time.

## What the first census adds beyond the two documented populations

1. **riir-ai's integration surface is genuinely covered** — Layer 3 runs
   `cargo test --workspace --all-features` and the per-crate loop runs
   `--all-targets`, so its 369 LB-unpinned targets execute automatically.
   Earlier prose ("executed by nothing") referred to katgpt-rs, not riir-ai;
   this report is the measurement that pins the distinction.
2. **riir-neuron-db pins nothing by name yet runs everything** — its
   `cargo test --all-features` broad run covers even feature-gated targets.
   Name-pinning is a riir-chain-shaped habit (128 pinned), not a universal
   one; absence of pins is not a defect where a broad run exists.
3. The cross-repo sibling-named column (48) is real: league/regate harnesses
   in one repo name another repo's targets. A per-repo-only corpus would have
   reported those 48 as orphans.

## Repeat usage

```bash
python3 scripts/suite_membership_audit.py             # all present repos
python3 scripts/suite_membership_audit.py ../riir-ai  # one repo
python3 scripts/suite_membership_audit.py --selftest  # classifier pins
```

Repos not checked out on the running box are invisible to the walk — absent
is UNVERIFIABLE, not clean (the link-sweep lesson). The 6 absent repos on the
census box above are the M3-resident ones.
