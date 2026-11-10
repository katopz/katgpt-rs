# Issue 992 — crates.io refuses katgpt-core: 414 features > the 300 limit (P1 cascade blocker, 8/9 landed)

**Status:** OPEN — owner decision required (reduce-to-300 vs limit-increase email vs other); 8/9 cascade crates PUBLISHED, katgpt-core alone refused.

## What happened

The issue-221 P1 publish cascade (riir-refine verdict D2; bumps `85c2a2edf` + lock `a230cc5d4`) ran 2026-11-10 via the kat-secret Touch-ID lane from a clean detached worktree at `origin/develop`:

| crate | published | registry now |
|---|---|---|
| katgpt-types | ✅ 0.3.0 | 0.3.0 |
| katgpt-dec | ✅ 0.1.4 | 0.1.4 |
| katgpt-assign | ✅ 0.1.0 (FIRST) | 0.1.0 |
| katgpt-hla | ✅ 0.1.4 | 0.1.4 |
| katgpt-micro-belief | ✅ 0.2.2 | 0.2.2 |
| katgpt-personality | ✅ 0.1.4 | 0.1.4 |
| katgpt-sleep | ✅ 0.1.4 | 0.1.4 |
| katgpt-sense | ✅ 0.3.0 | 0.3.0 |
| **katgpt-core** | ❌ **REFUSED** | still 0.4.1 |

The refusal (hard registry rule, [the 2023-10 badges blog](https://blog.rust-lang.org/2023/10/26/broken-badges-and-23k-keywords.html)):

```
error: failed to publish katgpt-core v0.5.0 to registry at https://crates.io
Caused by: the remote server responded with an error (status 400 Bad Request):
crates.io only allows a maximum number of 300 features, but your crate is
declaring 414 features.
```

## The census (2026-11-10, this issue)

414 feature rows in `crates/katgpt-core/Cargo.toml`:

| class | count | note |
|---|---|---|
| EMPTY rows (`feature = []`) | 79 | pure decl aliases (cfg vocabulary) |
| pure feature-forwards | 111 | e.g. `lattice_memory_3d = "lattice_memory"` |
| katgpt-* dep forwards | 85 | `"katgpt-types/maxsim"`-class |
| `dep:` syntax rows | 70 | opt-in dep activators |
| (overlap between classes) | — | classes overlap; the 414 total is crates.io's count |

Downstream usage (grep `katgpt-core/<feature>` across the workspace, `*.toml`):

| consumer | distinct features spelled |
|---|---|
| riir-ai | 173 |
| riir-train | 40 |
| riir-neuron-db | 35 |
| riir-infer | 33 |
| riir-refine / riir-reflex | 19 each |
| riir-instinct | 13 |
| riir-chain | 8 |
| riir-game-sdk | 4 |
| riir-dapps | 1 |
| **UNION across all 10 repos** | **234 distinct** |
| **NEVER spelled downstream** | **180 of 414** |

## The decision (owner)

Three defensible roads; all are katgpt-rs-surface changes, so none is agent-unilateral:

1. **Reduce to ≤300 in-repo** (no external dependency): the 180 never-spelled features are the cut pool — candidates: collapse the 79 empty aliases into their cfg consumers' own feature spellings (downstream spells the underlying feature or a `katgpt-core/<underlying>` that EXISTS — the alias rows are `feature = []`, so they gate code via `#[cfg(feature = ...)]` INSIDE katgpt-core only; removing one requires renaming those cfg sites OR keeping the name but... no: every row counts, so removal = the cfg sites gate on a survivor feature instead). The 111 pure forwards can stay if the count fits after the alias cut; else collapse forwards whose consumers can spell the target directly (breaking-spelling audit required per removal). **Effort: a scored sweep, semver-checks re-run (feature REMOVAL is a major bump → 0.6.0, not 0.5.0), every downstream `Cargo.toml` spelling audited.** The 8 published crates stay put (their version rows against core `version = "0.5.0"` must bump to 0.6.0 too — ANOTHER 8 publishes or wait for their next natural bump; the dep rows' version requirements are what core 0.6.0 will declare, so the already-published 0.1.4/0.3.0 crates remain valid registry neighbors — they don't need republishing unless core's dep rows demand their newer versions, which they already carry).
2. **Email help@crates.io** for a limit increase (the error's own suggestion) — slowest, zero code churn; the crate genuinely HAS 414 features and the registry rule exists to stop badge-spam crates, not real feature matrices.
3. **Hybrid**: cut the mechanically-free rows (the 79 empty aliases whose inside-core cfg sites can fold into survivors) to get under 300 without touching any downstream spelling; keep everything ever spelled.

## Constraint that binds the choice

`riir-refine` issue 221's P1 verdict explicitly REJECTED running the publish posture against katgpt-core **0.4.1** (it lacks the `rating`/`step_noise` modules that default-on machinery consumes). Until katgpt-core 0.5.0+ is on the registry, the refine crates.io lane stays at the sibling-mode verdict (`green (sibling: dry-run UNSEEN)`).

**Claimed:** zed-352d06e6-m3-221, 1794209100, expires 2026-11-12T09:30:00+00:00
