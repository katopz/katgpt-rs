# Issue 992 — crates.io refuses katgpt-core: 414 features > the 300 limit (P1 cascade blocker, 8/9 landed)

**Status:** UNBLOCKING — 2026-11-10 owner order superseded ROAD 2 ("we won't mail and wait forever — bake or group"): Plan 635 Tranche 1 (BAKE 74 default-closure features flagless + GROUP 62 single-parent additive children into their parent) takes katgpt-core **414 → 278** features (< 300). Next: land the path-dep sibling rewrites, re-run the P1 publish cascade (kat-secret lane); riir-chain/riir-dapps (git `branch = "main"`) rewrite at the develop→main merge. Previous status (ROAD 2 email, delegated verdict) kept below as history.

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

## The census (2026-11-10, this issue) — SUPERSEDED by the corrected census below

⛔ The tables in THIS section came from a forward-form-only grep (`katgpt-core/<feature>` across 10 downstream repos); the corrected census in the next section adds the lanes that grep missed and reverses its conclusion. Kept for the record.

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

## Corrected census (2026-11-10 second sweep — exact TOML parse + multi-lane grep; full tables in Plan 635)

The original grep missed three lanes; re-derived by a classification subagent (reads+greps only, tree at `origin/develop` `3963a7d39`) and spot-verified by the orchestrator:

| lane | distinct features | detail |
|---|---|---|
| parsed `[features]` rows | **414** | matches crates.io exactly; `default` array = 83 members (KEEP-IMMUTABLE) |
| internal spellings | **352** | fwd `katgpt-core/<x>` 204 · dep-table `features = [...]` 3 · **core's own `[[bench]]`/`[[test]]`/`[[example]]` `required-features`: 214 distinct across 376 rows** (160 benches + 164 tests) · CLI `--features decision_wire` (`scripts/test_gate.sh:155`) |
| downstream spellings | **250** across **13 repos** | fwd 239 / dep-table 56; three repos the original missed entirely: riir-dao (3), riir-reflexer (1), seal-online-remaster (1) |
| spelled union (internal ∪ downstream) | **391** | |
| kept after closure (default ∪ spelled ∪ transitively implied) | **399** | |
| never-spelled candidates | **19** | **ALL 19 are gain/GOAT-documented** (e.g. `laprop`: gates real code in `crates/katgpt-core/src/lib.rs`, GOAT record `.benchmarks/678_laprop_normalized_momentum_goat.md`; `simd_sigmoid` is implied by DEFAULT member `bom_sampling`) |
| **CUT** | **0** | under the recorded rule (never spelled · no gain/GOAT · not default · not implied) |
| resulting count | **414** | required ≤ 300 → **shortfall 114; even an override cutting all 19 leaves 395 > 300** |

**Conclusion:** the "180 never spelled" premise was an artifact of the incomplete grep. The matrix is a live, near-fully-consumed research-primitive surface — the GOAT bench/test gates themselves bind 214 features. In-repo reduction to ≤300 requires deleting documented primitives and/or bench targets (the opposite of "demote stale"); it is NOT the road. Verified examples of the missed lanes: riir-dao `crates/riir-dao/Cargo.toml:18` dep-table `features = ["rating", "hint_regret"]` + `:76` forward `evidence_budget = ["katgpt-core/step_noise"]`.

Also closed by the verdict reviewer's own re-parse: katgpt-core has 11 optional deps and **0 implicit dep-features**, so switching to explicit `dep:` syntax saves zero rows — the cheap mechanical route does not exist.

## The decision (owner)

⛔ SUPERSEDED 2026-11-10 — the owner chose ROAD 1 on the (wrong) 180-never-spelled premise; the corrected census above falsifies it. The DECIDED road is in "Road (decided)" below. Original text kept for the record.

Three defensible roads; all are katgpt-rs-surface changes, so none is agent-unilateral:

1. **Reduce to ≤300 in-repo** (no external dependency): the 180 never-spelled features are the cut pool — candidates: collapse the 79 empty aliases into their cfg consumers' own feature spellings (downstream spells the underlying feature or a `katgpt-core/<underlying>` that EXISTS — the alias rows are `feature = []`, so they gate code via `#[cfg(feature = ...)]` INSIDE katgpt-core only; removing one requires renaming those cfg sites OR keeping the name but... no: every row counts, so removal = the cfg sites gate on a survivor feature instead). The 111 pure forwards can stay if the count fits after the alias cut; else collapse forwards whose consumers can spell the target directly (breaking-spelling audit required per removal). **Effort: a scored sweep, semver-checks re-run (feature REMOVAL is a major bump → 0.6.0, not 0.5.0), every downstream `Cargo.toml` spelling audited.** The 8 published crates stay put (their version rows against core `version = "0.5.0"` must bump to 0.6.0 too — ANOTHER 8 publishes or wait for their next natural bump; the dep rows' version requirements are what core 0.6.0 will declare, so the already-published 0.1.4/0.3.0 crates remain valid registry neighbors — they don't need republishing unless core's dep rows demand their newer versions, which they already carry).
2. **Email help@crates.io** for a limit increase (the error's own suggestion) — slowest, zero code churn; the crate genuinely HAS 414 features and the registry rule exists to stop badge-spam crates, not real feature matrices.
3. **Hybrid**: cut the mechanically-free rows (the 79 empty aliases whose inside-core cfg sites can fold into survivors) to get under 300 without touching any downstream spelling; keep everything ever spelled.

## Road (DECIDED 2026-11-10 — delegated-owner verdict, `claude_code` session `09880baf`, round 1 REVISE → round 2 AGREE)

- **Main road — ROAD 2:** the owner emails `help@crates.io` requesting a per-crate feature-limit override ≈ **600** (headroom; primitives land continuously). Draft below. The reviewer's note: crates.io has a per-crate feature-limit override for exactly this case; send from the publishing account.
- **Interim (a) — WAIT (default):** riir-refine Issue 221 stays sibling-mode; nothing degrades (its G8 forbids deprecating anything before a crates.io install of the published version actually works). No code moves.
- **Fallback (b) — leaf extraction, decision gate 2026-11-24** (only if crates.io has not answered by then): carve `rating` + `step_noise` into a katgpt-dec-style leaf crate, publish it, refine depends leaf@registry + katgpt-core 0.4.1@registry. TWO PRECONDITIONS from the verdict round 2:
  1. **First build refine's publish posture against registry katgpt-core 0.4.1 with `rating`/`step_noise` stubbed out.** semver-checks recorded **8 majors between 0.4.1 and the tree's 0.5.0** (pub field/const removals, exhaustive enum variants, fn/method arity changes, a `LatentSpace` trait method added — riir-refine Issue 221 P1 prep row, katgpt-rs `85c2a2edf`); refine's other features (`entropic_tilt`, `best_belief`, `plasma_path`, `poincare_navigator`, `subspace_phase_gate`, `variable_rank_domain_expert`) may call items 0.4.1 lacks — a red build means (b) must extract those breaking items too and gets much bigger.
  2. **katgpt-core must RE-EXPORT the leaf crate's types, never keep its own copies** — otherwise core 0.5.x and the leaf carry two incompatible versions of the same types and anything passing them between the crates fails to compile.
- **ROAD 1′ — Plan 635 is the FALLBACK MAP ONLY** (never the default): the tranche tables (19 KEEP-EXCEPTIONS ranked most-stale-first, 5 prune-implication pairs, cfg-site files) are execution-ready if the owner ever orders the breaking collapse campaign — which must also scrub 376 bench/test required-features rows and re-point 250 downstream spellings across 13 repos. It contradicts the GOAT-gate doctrine; not recommended.
- **REJECTED during the verdict: the git-dep interim** (publish cargo-refine with a git dep on katgpt-rs). Evidence that killed it: rust-lang/cargo **#16161** — publishing REWRITES git deps to registry deps (the git source is dropped); a version-less git dep fails publish outright ("all dependencies must have a version specified when publishing"); crates.io policy — published crates cannot depend on other repositories. Net effect identical to no interim: refine's published manifest would resolve katgpt-core from the registry → **0.4.1**, which lacks `rating`/`step_noise` entirely (riir-refine Issue 221 P1 parity check).
- **Hybrid C (tranche cuts now, ~30 rows → ~384) DEFERRED**: does not clear 300; only meaningful if crates.io counters with a specific partial limit.
- **Growth guard (new task, in Plan 635):** a feature-count ceiling check in the publish lane — warn as the count approaches the granted limit, hard-fail BEFORE the registry refuses.

## Email draft (👤 owner sends from the publishing account to help@crates.io)

```
Subject: Feature-limit increase request for katgpt-core (414 features vs the 300 cap)

Hello,

`cargo publish` of katgpt-core v0.5.0 was refused on 2026-11-10:

  crates.io only allows a maximum number of 300 features, but your crate is
  declaring 414 features.

We would like to request a per-crate feature-limit override, ideally to 600
(headroom — this is a research-primitives crate and the matrix grows), for
katgpt-core. Facts we believe make this a legitimate case rather than a
badge/keyword-spam one:

- All 414 features are real rows in a single [features] table: ~79 empty
  declaration aliases, ~111 pure forwards, ~85 forwards into our own
  published katgpt-* sibling crates, ~70 optional-dependency (dep:) rows.
  The classes overlap; the 414 total is what the registry counted.
- 391 of the 414 features are referenced by at least one consumer: the
  manifests of 13 repositories in our workspace, and the crate's own 160
  [[bench]] + 164 [[test]] targets, whose required-features rows bind 214
  distinct features — these are the crate's benchmark and correctness gates,
  not vanity switches.
- katgpt-core 0.4.1 was published 2026-08-22. Since then the tree has moved
  and 8 sibling crates from the same account were published in a cascade
  (katgpt-types 0.3.0, katgpt-dec 0.1.4, katgpt-assign 0.1.0,
  katgpt-hla 0.1.4, katgpt-micro-belief 0.2.2, katgpt-personality 0.1.4,
  katgpt-sleep 0.1.4, katgpt-sense 0.3.0) — katgpt-core alone is blocked,
  and with it the crates.io install path of a real downstream application
  (cargo-refine). Reducing the feature count in-repo is not honest here:
  cutting every feature not referenced anywhere would still leave 395.
- Happy to share the manifest, the census, or anything else useful.

Thank you for considering it.
```

### Registry-side pointers (researched 2026-11-10 — cite in the email)

- **The official exception path is DOCUMENTED**: the Cargo Book's [Features page](https://doc.rust-lang.org/cargo/reference/features.html) states the 300-feature limit and adds "Exceptions are granted on a case-by-case basis" — the request is procedure, not favor.
- **Precedent is on record**: the announcing [blog post](https://blog.rust-lang.org/2023/10/26/broken-badges-and-23k-keywords/) says "We are aware of a couple of crates that also have legitimate reasons for having more than 300 features, and we have granted them appropriate exceptions to this rule."
- **The limit is a pending mitigation, not a permanent law**: the blog frames it "… for now, or at least until we have found solutions for the above problems" — the underlying problems (unpaginated API responses + unpaginated package index) are what the request actually works around.
- **Rationale + scale counter-fact**: the cap exists because feature rows bloat the unpaginated API response and the index file (the abuse case: ~23,000 features — 77× ours; see rust-lang/crates.io #7269 for the metadata-cost report). Our 414 rows are ~1.7% of the abusive crate's metadata; sparse-index clients (the cargo default for years, per the blog itself) download only this crate's own index entry.
- **Strengthening line for the email**: note the limit was announced as temporary pending index/API pagination work, and cite the Cargo Book's exceptions sentence as the documented procedure.

## Tasks

- [ ] 👤 owner: send the email above to help@crates.io from the publishing account (this unblocks everything else; agent cannot send it)
- [ ] 2-week gate **2026-11-24**: if no crates.io answer, run fallback (b)'s precondition 1 (build riir-refine's publish posture against registry katgpt-core 0.4.1 with `rating`/`step_noise` stubbed) and then decide the leaf-extraction fallback (record in riir-refine Issue 221 P1)
- [ ] Growth guard: feature-count ceiling check in the publish lane (task recorded in Plan 635; wire into the publish cascade script lane)
- [x] Corrected census recorded (2026-11-10, this revision; classification subagent + orchestrator spot-checks)
- [x] Road decided via delegated-owner verdict (2026-11-10, round 1 REVISE → round 2 AGREE)
- [-] Hybrid C tranche cuts — DEFERRED (does not clear 300; re-open only if crates.io counters with a specific partial limit)

**Claimed:** zed-352d06e6-m3-221, 1794209100, expires 2026-11-12T09:30:00+00:00
