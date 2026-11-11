---
name: boundary-guard
description: Audit + enforce game-stack boundary rules across the multi-repo workspace. Use when adding a System impl, game logic, vocabulary type, FFI surface, or view-layer code; when reviewing PRs touching game systems or view/FFI boundaries; when a violation is suspected; or quarterly as a boundary-hygiene gate. Reads each repo's BOUNDARY.md as the contract — owns, not-owns, dep allowlist, drift ledger — enforced via ci_boundary_contract.sh (workspace dep graph + contract honesty, every repo with a root BOUNDARY.md, derived never counted) + ci_boundary_guard.sh (per-repo code logic) + grep checks. Sibling to feature-gate-audit + goat-audit + doc-sync.
---

# Boundary Guard

Generic game logic → substrate (`riir-games`). View renders state, doesn't compute it. FFI moves raw bytes only.

## Spec source — each repo's `BOUNDARY.md`

Every repo ships a root [`BOUNDARY.md`](../../riir-ai/BOUNDARY.md) — the per-repo
contract this skill audits against: **Owns** / **Does not own** / **May depend
on** (crate-granular allowlist with Location) / **Inherited** (links) / **Drift
ledger**. The 8-surface table below is the workspace *methodology*; the per-repo
rules + exceptions come from that repo's BOUNDARY.md, not from this file's prose.
Findings are therefore two classes: **code-vs-contract violations** and
**contract rot** (drift row without an open issue, allowlist row without a gate,
by-design row whose decision record is gone).

**Two scripts, one job each** (added 2026-08-21, Issue 737 T-CI/T-GATE):

| script | scope | question it answers |
|---|---|---|
| `riir-ai/scripts/ci_boundary_contract.sh` | workspace (every repo with a root `BOUNDARY.md` — **18 again since 2026-09-10**: 16 after the 2026-09-04 retirements, then +`riir-esp32` 2026-09-06, +`riir-kat` 2026-09-10; the script enumerates, so prefer its banner over this cell) | Is the dep graph what the contracts say — and are the contracts still honest? Parses every `May depend on` table + drift ledger, checks the riir-ai CANONICAL matrix against the measured graph, pins the 4 split-prep invariants. `--list-deps` prints the measured edge set; `--repo X` narrows. |
| `riir-mmorpg-examples/scripts/ci_boundary_guard.sh` | that repo's `src/` | Is the CODE in the right repo? Checks A–E: System impls with game logic, duplicated geometry, hardcoded behavior constants, generic logic in free functions, facade leaks. |

The contract script replaces prose-only allowlists; it is the successor of
Issue 724 Phase 2's "document the contract and review by hand". `EXEMPT_LEAKS`
in the per-repo guard deliberately STAYS file-granular — the ledger is
surface-granular, and collapsing the two would lose the file:line precision
Check E needs.

**Partial-clone boxes** (riir-ai Issue 939, the katgpt-rs Issue 765 idiom
one repo over): C0d (a path dep whose target sibling is absent) and C0e (a
`../<repo>` routing reference that does not resolve) both decide by asking
whether a DIRECTORY exists under the workspace root, so a box carrying a
subset of the workspace reds on every edge and route into an un-cloned
sibling. Export `BOUNDARY_PARTIAL_CLONE=1` there and those two ABSENT-TARGET
verdicts report as loud instrument-alive DEFERRALS instead; the rest of the
gate runs at full strength and the exit code is untouched. Never
auto-detected and never set in CI — from the walk alone a genuinely DELETED
sibling is set-identical to an un-cloned one. Deliberately narrow: a target
that IS present and merely routed wrong (C0a's missing contract, a C0e route
into `obsolete/`) stays a hard finding in every posture. `--self-test` pins
both postures plus that narrowing (10 arms); it is two-sided, so jamming the
marker on or off reds it.

**Drift-ledger semantics** (scripts consume this): Disposition ∈ `fixable` |
`owner-call` | `by-design`. `fixable`/`owner-call` rows REQUIRE an open issue;
`by-design` rows cite a decision record instead. Issue closes → row removed in
the same commit. Exit semantics: ledger unparseable → hard error; finding mapped
to a row → exit 0 with a LOUD known-drift count; unmapped finding → exit 1;
row-without-open-issue → exit 1 (rot). Never silently fail open or closed.

## Eight surfaces

| # | Surface | Rule | Grep check |
|---|---------|------|------------|
| 1 | Consumer `src/` (riir-mmorpg-examples, mmorpg-remake) | Thin glue only — no `impl System` with loops/math; no hardcoded constants; no duplicated helpers | `grep -rn 'fn distance_2d\|const.*FEAR' src/` |
| 2 | SDK root crate (`riir-game-sdk/src/`) | Facade only — no engine/chain/db deps in the DEFAULT build. Sanctioned opt-in exceptions (Issue 053 Part 2 + `auth_impl`/`gm` pattern; all `optional = true`, heavy ones target-gated native): `auth_impl`/`identity_impl` (riir-auth, riir-chain/ssh_key), `gm` (katgpt-core, hoisted `InferenceBackend`), `static_data_impl`/`warm_tier_impl` (riir-neuron-db, neuron-db-sdk). Scan BOTH `[dependencies]` AND `[target.'cfg(...)'.dependencies]` — the Issue 053 deps live in the target-gated section | main + target sections, filter out `optional = true` lines |
| 3 | SDK root vs workspace members | Root crate clean; members (`crates/riir-viz`, etc.) MAY depend on engine | `sed -n '/\[dependencies\]/,/^\[/p' riir-game-sdk/Cargo.toml \| grep katgpt` |
| 4 | Leaf-clean vocabulary (`riir-games-shared`) | No engine deps unless feature-gated — **at the dep level too**: dep line `optional = true` AND its feature carries `dep:<name>`. Half-gated (module cfg-gated, dep non-optional) = violation — every no-features build pays the engine tree (Issue 682: katgpt-core non-optional pulled rustfft/postcard/half into a default-`[]` crate) | `grep -E 'katgpt-core\|riir-engine' riir-ai/crates/riir-games-shared/Cargo.toml \| grep -v 'optional = true'` |
| 5 | View consumers — **live: `mmorpg-remake` (`mmorpg-view` Bevy/wasm + `mmorpg-node`)**; the C# that remains in the workspace is `riir-viewbridge/csharp/` (3 files, surface 6's own side) | Rendering + input only — NO game logic (AI, combat, physics, sync). Documented deliberate debt in an OPEN issue + cross-language contract doc = record as such, don't re-file | `grep -rnE 'sigmoid\|dot_product\|impl .*System for' mmorpg-remake/crates/*/src/ \| grep -viE 'showcase\|benchmark\|camera'` (measured 2026-09-04: **empty**) |
| 6 | FFI bridge (`riir-viewbridge` — repo PARKED 2026-09-03, Unity lane frozen; checks stay live for the unfreeze path) | Raw physical only (`pos[3]`, `rot[4]`) — NO latent state crosses FFI | `grep -rn 'emotion\|fear\|mood\|curiosity' riir-viewbridge/crates/*/src/` |
| 7 | Dev tools + KAT client plane (`riir-refine`, `riir-kat` — the client/protocol half spun out of clippy 2026-09-10) | Zero game-domain coupling in the DEFAULT build (clippy's only public dep is `katgpt-core`; kat's deps are riir-auth `account_key`-only + the katgpt patch pin). Sanctioned opt-in arms where reimplementation would duplicate whole substrates: `ternary_inference` (riir-engine + riir-gpu), `latent_retrieval` (riir-rag) | `grep -nE 'riir-games\|riir-chain' riir-refine/Cargo.toml riir-kat/Cargo.toml \| grep -v ':[0-9]*:#'` (dep lines only — the `-n` + comment filter matter: bare grep returns clippy's prose comments; verified empty 2026-09-11) |
| 8 | dApp layer (`riir-dapps`) | One-way **game → dapps → chain** — never a game dep here (`scripts/direction_gate.sh`); a `Settlement` stays chain vocabulary (never quest/kill/recipe — gate check 3 of direction_gate.sh); anything a game wants on-chain passes the **three-test rule** (product / value / rate) on the **agreement axis** — the chain hosts what mutually distrusting parties must agree on; neuron-db hosts authenticated durability. A paid quest still needs NO chain program (neuron-db template + the one generic `MultiClaimEscrow`) | `grep -E 'riir-games\|riir-game-sdk\|riir-engine' riir-dapps/Cargo.toml` + game-vocab scan in `scripts/direction_gate.sh` |

### The dApp three-test rule (surface 8 detail — riir-chain Issues 096/097 + riir-dapps Proposal 001)

1. **Product** — would a commerce customer of this chain want it in their dependency? (An NFT is a token → yes. A quest/recipe/kill-credit predicate → no.)
2. **Value** — BigInt fungible currency, a token, or an authority binding? Not FAME/XP/items/reputation/karma/quest progress.
3. **Rate** — quorum-coordination ops are Glacial (≤0.1 Hz); settlement transactions are **capacity-share bound** (783 req/s measured floor) — different limits, both binding.

The defining axis is **agreement** (Research 003 §"The Second Axis" "Must agree on" column) — value and rate are the disqualifying tests, not the definition. Full argument + failure-mode matrix: `riir-dapps/.proposals/001_agreement_boundary_and_tiered_durability.md`.

All greps should return **empty** (clean), modulo the sanctioned opt-in exceptions noted per-surface.

<!-- retired surfaces, kept as lineage: `riir-unity` (C# Unity host) and
`mmorpg-remake-unity` were moved to `git/obsolete/` by owner act on 2026-09-04,
`riir-armageddon` on 2026-09-02. Surface 5's old command globbed
`riir-unity/**/*.cs` and could no longer run — worse than stale, because a
non-matching zsh glob aborts the whole command line rather than returning
nothing, so the check would have looked skipped rather than broken. The
`WireProtocol.cs` precedent it cited (riir-unity Issue 002 Phase A, a
cross-language contract documented on both sides instead of re-filed) is the
part worth keeping and applies to `riir-viewbridge/csharp/` under surface 6. -->

**Methodology lesson (2026-08-15 run):** exclusion filters can hide exactly what you're looking for — the "who enables feature X" grep returned zero because the forwarder lines contain `katgpt-core` and were killed by `grep -v katgpt-core`. Vocabulary-translation care applies to filters, not just search terms.

## Failure pattern

"Helper" in consumer → wrapped in `System impl` → grows loops + math → stuck in consumer. Same applies to C# view code reimplementing substrate logic.

## Extraction checklist

Before adding to consumer `src/` or view C#/Bevy:

1. Is this generic game behavior? → substrate (`riir-games`)
2. Does substrate already have it? → grep `riir-games/src/{swarm,motivation,combat}/`
3. Can it be parameterized? → trait (`ThreatSource`) or config struct
4. Is the consumer/view just data + wiring? → if loops/math/constants present, STOP

If unsure → file an issue, don't add the code.

## Filing violations

1. `.issues/NNN_boundary_*.md` in the repo
2. Reference which surface (1–7)
3. Include file:line + grep output
4. Propose extraction target (substrate module + trait)
5. **Issue BEFORE fix** — every fixable finding gets its `.issues/NNN_boundary_*.md`
   filed BEFORE any fix commit, even trivially-fixable ones. The fix commit
   references the issue; closing the issue removes the drift row in the SAME
   commit. Only the guard/script tooling itself may be fixed in-run — boundary
   CODE never.

## Move by script, never regenerate

Any relocation of boundary content — extracting superseded sections from
AGENTS.md into BOUNDARY.md, removing drift rows, linking READMEs, or any future
crate/repo move — is done **by script** (`git mv` + anchored sed/python) with a
before/after **grep-parity check** (every rule sentence present exactly once
post-move). Re-typing or regenerating the content is forbidden. Grounding:
the AGENTS.md section silently dropped by a concurrent session's stale-buffer
commit (`88e5f98`), and the edit-fuzzy-match that ate a raw-string `#`
terminator — both would have been caught by parity checks.

## Running

```bash
# workspace contract (dep graph + contract honesty + split-prep gates)
cd riir-ai && ./scripts/ci_boundary_contract.sh          # exit 0 = clean
./scripts/ci_boundary_contract.sh --list-deps            # measured edge set
./scripts/ci_boundary_contract.sh --repo riir-chain      # one repo

# per-repo code logic (surface 1)
cd riir-mmorpg-examples && ./scripts/ci_boundary_guard.sh # exit 0 = clean
```

Exit codes (contract script): 0 clean or all-findings-mapped (LOUD known-drift
count), 1 unmapped finding or contract rot, **2 hard error** — a missing or
unparseable contract never fails open. For other repos' code-logic checks,
adapt the guard's SRC_DIR + patterns. Or as pre-commit: `exec ./scripts/ci_boundary_guard.sh`

**Run boundary checks VIA this skill** — not as ad-hoc greps. The skill reads
each repo's BOUNDARY.md as the contract, applies the methodology below, and
records the run in the log. The T-CI/T-GATE wiring landed 2026-08-21, so the
full via-skill run is now available (and the first one is logged below).

## Run log

**Compacted 2026-09-08 (user-directed — the file crossed 100 KB of context);
re-compacted 2026-09-11 (post-09-08 rows had regressed to multi-clause
narratives).**
Rows carried full narratives until these compactions; `git log -p -- .agents/skills/boundary-guard/SKILL.md` (the katgpt-rs repo) recovers any of them verbatim. New rows append ONE line each. **Pruned 2026-09-21: 105 → 15 rows, 71KB → 30KB; again 2026-10-26: 32 → 15 rows, 65KB → 42KB (243rd run); again 2026-10-29: 18 → 15 rows, 63.3KB → 56.1KB (312th run); again 2026-10-29: 18 → 15 rows, 60.5KB → 54.8KB (315th run)** — the one-line convention held but the cadence didn't (~10 rows/day re-bloated it since the 09-11 re-compaction; third prune). **Maintenance rule: whenever this file exceeds 60KB, prune the run table to the newest 15 rows** — enforced mechanically by `scripts/skill_size_gate.py` (80KB hard ceiling, all `.agents/skills/*/SKILL.md`); recovery via `git log -p` as above.

### Standing lessons (distilled from the compacted rows — load-bearing process rules)

- **Read the exit UNPIPED.** `cmd | tail; echo $?` reads TAIL's exit (re-proven twice); outputs go to /tmp, read the real exit or PIPESTATUS. A piped gate summary cannot vouch for an exit code — and a green summary over an unstaged body is worse: the 20th run's "exit 0" was FALSE while C0e's implementation sat unstaged (`a6ca9ccbd` docs-only; body landed `10977dd23`).
- **Verify the script not-mid-edit before running**: mtime vs the last recorded run + `bash -n` + clean in `git status`. A sibling rewriting the script mid-run truncated a measurement after SP1 while the first attempt still reported exit 0 (the 2026-09-04 membership row) — bash re-reads by byte offset. The script anchors ROOT from its own path, so /tmp snapshot-copy immunization CANNOT work; diff-vs-snapshot before running the live script instead.
- **Parser replay for stale shared checkouts**: run the script's own allowlist awk against `git show origin/<default>:BOUNDARY.md` — a sibling branch predating a row reads exit-1 on disk while main is CLEAN (the riir-train until-merge artifact class); touching the sibling branch is forbidden.
- **Detection-only discipline**: file `.issues/NNN_boundary_*.md` BEFORE any fix; sibling WIP (staged sets, mid-flight sweeps, dirty manifests) is measured as-found and never fixed by the idle unit — the sweep owner re-pins C6 at closeout (the `2575b6519` precedent). A landed growth WITHOUT its re-pin, though, is repaired by whoever finds it once the landing session is gone (the honest up-pin class: verify zero training semantics, pin UP with the reason).
- **The C6 ledger row text must be apostrophe-free** and the real detector is `bash -n`, not running the guard — bash 3.2's quote tracker reports the wrong line (~20 below the guilty one).
- **Expected shapes, not findings**: 0 contract-rot — the historical 2-row mmorpg-remaster pair is RESOLVED upstream (mmorpg-remaster = seal-online-remaster under the alias codec, `maxion-game/seal-online-remaster`; its ledger now carries only the by-design D1 row, decision-recorded, fires nothing — verified by the 313th + 314th full runs); C7's no-lockfile set and the C8/C9 pin counts MOVE — read each run's own lines, never a count pinned here (the old bullet's "C7 = 4" went stale at 7); scoped `--repo` runs print C7/C8 out-of-scope notes only.
- **C8/C9 standing advisory (owner call, never an idle-unit act)**: consumer lockfiles pin katgpt-rs `@6f392727` and drift further behind develop each window — 326 (09-03) → 739 commits behind with 7 security fixes in the gap (09-11, the 70th run) across 3 lockfiles, plus a second divergent pin in riir-chain (167/1). Recorded every run; the bump is the owner's. **Gap ENUMERATED 09-10 with the script's own predicate (689 behind at the 37th run, `@c478ab9f` chain pin 0 security): the 6 = 2× SIMD soundness (`99afbab9` safe-code OOB read + heap-corrupting write in `katgpt-types/src/simd/` — CWE-125, `simd_dot_f32` unchecked-`len`; `5b028c00` the near-verbatim katgpt-dec twin) + 2× FFI/panic-UB hardening (`af2e48e1` catch_unwind on 10 wasmi extern fns; `e854958e`/`96543f4f` the 208-site NaN-comparator sec heal) + `2dec22a1` (a filed, not-fixed, EngramHotSwap nested-swap soundness hazard) — the sharp half is that the two SIMD holes are MEMORY-SAFETY bugs in the exact substrate the three deploy lockfiles compile today, so the bump is a soundness call, not a freshness preference. **RESOLVED 2026-09-11 (owner call executed, subagent re-verdicted): all three consumer locks bumped `6f392727 → c478ab9f`, and the discovery corrected the advisory's own premises.** The 3 tracked lockfiles are riir-dapps `cloudflare/kat-service`, riir-mmorpg-examples `cloudflare/warm-tier-do`, and riir-esp32 `crates/riir-satellite-probe` — NOT riir-kat/riir-dao, which carry no tracked lock at all (the C7 class). The consumers pin `branch = "main"`, so the reachable target was main tip `c478ab9f`, which already contains ALL 6 enumerated fixes; the 7th grep hit in the develop-only range is a docs commit (the advisory's own enumeration) — a **predicate false-positive class**: the sec-subject grep matches docs commits whose subject contains "soundness". riir-chain's "divergent pin" re-verdicted to the same answer: it already sits at main tip `c478ab9f`, zero action needed. All builds/tests green per consumer (kat-service 143 tests + wasm32; warm-tier-do 4 + wasm32; satellite-probe esp32c6 release). Commits: dapps `c9203a2`+`6342519`, mmorpg `0afb52b`, esp32 `2bfb795`. Going forward the "behind develop" count is a branch-policy artifact (`branch=main`), not a security gap — the advisory's remaining value is the C9 one-rev agreement, which this bump achieved (all 4 build roots on `c478ab9f`).**
- **Guard output truncates in `tail` views** — capture the FULL violation list when filing (the "8 = 6 unique" enumeration error).
- **The mmorpg-examples guard is bash-shebang with process substitution — invoke `./` or `bash`, never `sh`** (macOS /bin/sh is POSIX-mode bash 3.2: `done < <(find …)` is a syntax error, and the completion sentinel correctly reports the mis-invocation as an abort-exit-1, not a pass — 333rd run).
- **Never suppress fetch output you act on** — a silenced fetch failed and produced a false empty-divergence read while the push error was right (the 59th run, 4090 box).
- **`git show HEAD:<dir>` on a directory prints the tree LISTING** — honest dir LOC is an `ls-tree -r` + `cat-file` sum. `rustfmt --emit stdout` prints a 2-line filename header — a round-trip harness must strip it.
- **The staged-set check must be a SEPARATE command** (or commit `git commit -- <paths>`): chained `add && diff --cached && commit` swept a sibling's entire staged set once (the `30127c4` incident); recovery anatomy lives in that row's history — read the reflog before any second corrective action.
- **A prose contract cannot catch an inverted measurement** — every "repo X may depend on Y" row is a measurement with a direction; that is why the contract script parses tables instead of prose (the first full run's founding lesson).

### Run table

| Date | Run | Verdict | Record |
| 2026-11-12 | 409th — SCOPED delta verification, NO script run (amo box, the riir-ai-touching lane — the deferred AGENTS league-row refresh; trigger = the 1165 flip arc landed in the sibling [riir-infer `351f5f62` substrate default + riir-ai issue file status + Bench 1158 posture cell, ALL md/one-env-seam .rs in riir-infer, zero dep-edge surface] + this lane's own md-only commit `adf8e6cf3` [AGENTS.md league-row clause + three .issues lease folds]) — the 408th full-run verdict (34 repos, 535 edges, exit 0, same day) + the 354th–408th delta verdicts stand; C8/C9 owner-call advisories stand; en-route coordination: the checkout was 2 behind with the 1162 peer's UNCOMMITTED T2 WIP blocking the pull — checked via merge-tree (clean) + a detached-worktree patch replay (byte-identical apply) BEFORE `git pull --ff-only` (no stash of sibling files); the peer landed `53026fb44` minutes after — their line, their call, hands off | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md + lease folds only) | riir-ai `adf8e6cf3` + katgpt-rs (this commit) |
| 2026-11-12 | 408th — FULL script run (amo box, the doc-sync 990-fill housekeeping lane; trigger = routine unit boundary check-up, this cycle's own windows md-only: katgpt-rs `70347c95a` [README + overview + catalog .md rows only, zero dep surface]) — workspace `riir-ai/scripts/ci_boundary_contract.sh` exit 0 (34 repos, 535 cross-repo dep edges — matches the 404th/407th graphs, no drift); C9 katgpt-rs multi-rev pins advisory stands; C8 not-checked list unchanged; run-log prune fired same unit (34→15 rows incl. mine, 61,058→36,063B) | landed-state CLEAN — 0 violations / 0 rot | katgpt-rs `70347c95a` + (this commit) |
| 2026-11-12 | 407th — SCOPED delta verification, NO script run (m3 box, the idle-loop hygiene lane; trigger = this cycle's Protocol-H windows ×3: riir-infer 50031ed0..1a5a778b [the 1165 delegated-verdict env-default promotion — ONE in-crate .rs (ternary_deltanet_gpu_forward.rs), no manifest row per the env-switch law; the 1205 land+close (Bench 969 + HISTORY); the 1206/plan-631/1190/1199 claim + corpus-script md/txt; the i1199 fused-batch layer interleave — in-crate riir-infer-dist .rs + G1 test + Bench 970; en-route MY hygiene commit f83a5bd2 = 2 claim-line folds + 1 AGENTS.md md row, zero dep surface] + riir-refine 8f9411e45..59f8cf506 [2 × issue-227 md-only] + riir-ai c079e320f [plan-083 T8d feat, in-crate; verified AT ORIGIN — the tree is dirty with the live 648 peer's WIP, not pulled per the dirty-tree law]) — `*.toml`/`*.lock`/BOUNDARY.md diff-stat EMPTY across all three ranges (verified unpiped) — the 404th full-run verdict (34 repos, 535 edges, exit 0) + the 402nd scoped script verdict (riir-train, 31 edges, exit 0) + the 403rd–406th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + md/bench/plan data + one env-default flip inside an in-crate file) | riir-infer `c6696efe`+`1a5a778b`+`f83a5bd2` + riir-refine `59f8cf506` + riir-ai `c079e320f` + (this commit) |
| 2026-11-12 | 406th — SCOPED delta verification, NO script run (amo box, the plan-373 housekeeping lane; trigger = post-405th windows: riir-train 8c86e7ef [docs(373) HISTORY row, md-only] + the sibling P0.5c lane's UNCOMMITTED in-tree WIP measured as-found [late_state_loop_gpu backward .rs + tests + ONE riir-train-gpu Cargo.toml whose 11 added lines are the intra-package [[test]] plan373_p05c_gpu_backward required-features row + its comment — zero dep-edge added lines, zero BOUNDARY hunks, diff verified]) — the 404th full-run verdict (34 repos, 535 edges, exit 0) + the 402nd scoped script verdict (riir-train, 31 edges, exit 0) stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md + intra-package [[test]] row as-found) | riir-train `8c86e7ef` + (this commit) |
| 2026-11-12 | 405th — SCOPED delta verification, NO script run (amo box, the 932-canary-verdict lane zed-c7ae9307; trigger = canary ops only since the 404th: the guard-script brew-PATH pre-stage on bmo `~/riir932/10k11_guard.sh` [sha `c45f74f6…`, OUTSIDE any git repo] + two timed-wake self-mails, ZERO repo windows — added-line dep-edge grep vacuous by construction) — the 404th full-run verdict (34 repos, 535 edges, exit 0, same day) + the 354th–404th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (zero repo changes) | (this commit) |
| 2026-11-12 | 404th — routine double script verdict (amo box, the idle-lane boundary-guard unit; NO trigger window — queue item 6 routine check; the bash-vs-sh trap hit and corrected en route: `sh ci_boundary_guard.sh` dies at the process-substitution `done < <(find …)` (line 192), re-run under its bash shebang clean) — repo-local `riir-mmorpg-examples/scripts/ci_boundary_guard.sh` exit 0 (all five checks green, no violations) + workspace `riir-ai/scripts/ci_boundary_contract.sh` exit 0 (34 repos, 535 cross-repo edges — matches the 397th/401st graphs, no drift) + pre-flight clean (script committed state, not mid-edit) | landed-state CLEAN — 0 violations / 0 rot; C8/C9 owner-call advisories stand (katgpt-rs multi-rev pins unchanged); NO issue filed (both gates green; the sh-invocation abort was the caller's shell, not a script defect) | (this commit) |
| 2026-11-12 | 403rd — SCOPED delta verification, NO script run (M3 box, the plan-648 handoff housekeeping lane; trigger = this cycle's Protocol-H ff-merge windows ×3: seal-remake 76d9d7fb..0aec04b9 [plan-083 T8b/T8d — in-crate seal-view/hero_lane.rs + src/tests + guard-script + md + ONE Cargo.toml whose single added line is the scenario_rig forward "riir-games-mmorpg/authority_heroes" on the ALREADY-DECLARED riir-games-mmorpg edge, diff verified] + riir-refine d6eb7e21..8f9411e45 [the B351 foxguard landing 15cbe3a56 + claim/yield/pin docs — in-crate sec corpus/fixtures/proposers + pins_manifest/score_history data + md, zero Cargo.toml/lock/BOUNDARY hunks by merge diffstat] + katgpt-rs 70d5b6278..40e58595b [988 T5 + docs + the 402nd row — katgpt-core Cargo.toml added lines are the intra-package grafts feature row + comments + ONE same-package [[test]] required-features row, diff verified]) — BOUNDARY.md hunks ZERO across all three ranges; the 402nd scoped script verdict (riir-train, 31 edges, exit 0, same day) + the 397th full-run verdict (34 repos, 535 edges) + the 354th–402nd delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (one feature forward on a declared edge + intra-package rows + in-crate .rs + md/data) | seal-remake `0aec04b9` + riir-refine `8f9411e45` + katgpt-rs `40e58595b` + (this commit) |
| 2026-11-12 | 402nd — SCOPED delta verification WITH scoped script run (amo box, the doc-sync continuation lane; trigger = post-401st windows ×6: katgpt-rs dc348756b [i988 T1+T2a+T3+T6 — in-crate katgpt-core graft.rs/lib.rs/tests/graft_g4_alloc.rs + ONE Cargo.toml whose added lines are the intra-package `grafts` feature row + comments, zero dep lines] + 3dc2b49a4 [T5 — in-crate graft_artifact.rs + Cargo.toml comments-only] + riir-train bd2ccd53/361b74e2/62290ad0 [plan-373: in-crate eval.rs fix + npc_memory rustfmt zero-content + P0.5b GPU forward — in-crate riir-train-gpu .rs ×4 + parity test + plan md + ONE riir-train-gpu Cargo.toml whose added lines are ONE intra-workspace feature row + ONE same-package [[test]] required-features row; the `riir-gpu/cubecl_runtime` forward rides the ALREADY-DECLARED riir-gpu edge, path=../ grep 0 in the whole window] + riir-refine 86a5e7b2 [plan-361 md only]) + pre-flight clean (bash -n, script not-mid-edit) — `ci_boundary_contract.sh --repo riir-train` exit 0 (31 cross-repo edges, SP5b ✓, vendored-patch note as printed); this morning's full-run verdict (33 repos, 535 edges, exit 0) + the 354th–401st delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + intra-package/intra-workspace manifest rows + md) | katgpt-rs dc348756b+3dc2b49a4 + riir-train 62290ad0 + riir-refine 86a5e7b2 + (this commit) |
| 2026-11-12 | 401st — SCOPED delta verification, NO script run (amo box, the boss-run housekeeping continuation lane; trigger = the post-400th windows ×2: riir-ai e87ce8770 [plan-648 T1+T2 — 14 in-crate quest_combat .rs files, added-line dep-edge grep ZERO: the only `path = "../"`-adjacent hits are in-crate `+++` diff headers, zero Cargo.toml/lock/BOUNDARY hunks via diff-stat] + llama.cpp +2 [external opponent mirror, ff-merged, OUT of scope]; en-route doc-sync fills in katgpt-rs [README/overview/catalog §164 md-only] + riir-ai [AGENTS crate-table row, md-only] carry zero dep surface by construction) + THIS morning's full-run verdict stands (33 repos, 535 edges, exit 0, amo box, ran post-sync ~03:35 +07 in the boss housekeeping lane — same graph the 397th measured) | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges; C8/C9 owner-call advisories stand | riir-ai `e87ce8770` + katgpt-rs (this commit's doc fills) + (this commit) |
| 2026-11-12 | 400th — SCOPED delta verification, NO script run (amo box, the foxguard staged-distill lane zed-c21902cf; trigger = this lane's OWN window: riir-refine d6eb7e21 [5 files: snapshot md + research md + plan md + two .highwater counters — added-line dep-edge grep ZERO: no `path = "../"`, no `git+`, zero BOUNDARY.md hunks]) — the 398th scoped script verdict (riir-refine, 20 edges, exit 0, same day) + the 397th full-run verdict (34 repos, 535 edges) + the 354th–399th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md/highwater-only); en-route housekeeping: doc-sync SKILL.md 60KB prune fired (run log 25→15, 62,051→57,511 bytes) | riir-refine `d6eb7e21` + (this commit) |
| 2026-11-12 | 399th — SCOPED delta verification, NO script run (amo box, the idle-loop i1190 lane zed-c21902cf; trigger = this lane's OWN windows ×4: riir-infer 2b54e46 [claim md + plan-631 md + highwater + scripts/i1188_corpus_unit.txt] + 9711ce0 [plan md 1-line] + katgpt-rs 6cc2c63d5+24117f82a [987 issue-md checkbox/trigger flips]) — added-line dep-edge grep EMPTY across all four windows (md/txt/highwater only; zero `path = "../"`, zero `git+`, zero BOUNDARY.md hunks) | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges; the 398th scoped script verdict (riir-refine, 20 edges) + the 354th–398th delta verdicts stand; C8/C9 owner-call advisories stand | riir-infer `2b54e46`+`9711ce0` + katgpt-rs `6cc2c63d5`+`24117f82a` + (this commit) |
| 2026-11-12 | 398th — SCOPED delta verification WITH scoped script run (amo box, the idle-loop i222/1197 lane zed-a6fadf68; trigger = this cycle's OWN windows ×2: riir-refine 73f2aca5 [claim md] + e2d8744b [in-crate src/domains/batch_diversity.rs + issue md — the 222 axis-b readout, zero manifest/lock/BOUNDARY hunks, verified unpiped: diff-stat over '*.toml'/'*.lock'/BOUNDARY EMPTY + path=../ count 0] + riir-ai 3c57d2e04 [in-repo scripts/sr1098/run_campaign.sh + issue md — the 1197 P4 harness, same greps EMPTY]; pre-flight clean bash -n + not-mid-edit) — `ci_boundary_contract.sh --repo riir-refine` exit 0 (20 cross-repo edges, C8 n/a no-lockfile); the 397th full-run verdict (34 repos, 535 edges) + the 354th–397th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + in-repo script + md); the 397th full-run + 354th–397th delta verdicts stand | riir-refine `73f2aca5`+`e2d8744b` + riir-ai `3c57d2e04` + (this commit) |
| 2026-11-11 | 385th — SCOPED delta verification, NO script run (amo box, the 932-canary-wait housekeeping lane zed-9665f823; trigger = post-384th ff-merges ×2 [riir-rethink 7d2db75 — .issues/063 md ×3±3, the paraphrase-shadow-read rows; seal-remake 887edbdb — .issues/145 md +1, the strategy-ask claim line] + canary ops [job 01a1cbd1 cancelled un-started and re-queued as 01a1cbd8 with the notify re-addressed zed-0b69b81d→zed-9665f823 — the overflowed old session was still alive-idle and would have forked the verdict handling; zero repo changes] + katgpt-rs Cargo.lock stat-dirty content-identical as-found [empty diff, build mtime touch, NOT committed]) — added-line dep-edge grep ZERO hits in both windows (md-only by construction) | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only); the 379th full-run verdict + the m3-box full 397th (34 repos, 535 edges, exit 0) stand; C8/C9 owner-call advisories stand | (this commit) |
| 2026-11-10 | 384th — SCOPED delta verification, NO script run (amo box, the 932-canary lane zed-0b69b81d; trigger = this lane's OWN windows + canary ops: riir-ai 6030ee7dd [the 932 canary enqueue-1 no-run + enqueue-2 + guard-script PATH warning, .issues md only] — added-line dep-edge grep EMPTY (md-only); CANARY OPS: enqueue 1 (01a1c339) died at exit 101 BEFORE the test — cmake missing from the rig job PATH (a fresh target dir compiles sentencepiece-sys via CMake; brew's cmake at /opt/homebrew/bin is on NO default PATH on bmo — the 10h lease's "brew PATH export per the rev-5 posture" was exactly this, masked all prior attempts by the warm 10g dir's cached artifact); enqueue 2 (01a1cbd1) re-queued with the brew PATH + a cmake presence refusal, partial compile artifacts KEPT (fingerprints verify content; the kg test binary was never built, freshness semantics intact), exclusive; the guard script carries the same PATH warning in the 932 file). The 379th full-script run verdict stands (34 repos, 530 edges) + the 380th–383rd delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only + canary ops); the 379th full-run + 380th–383rd delta verdicts stand | 6030ee7dd |
| 2026-11-11 | 397th — FULL script run (M3 box, the zed-m3-i1204 lane — riir-infer issue 1204's resume-replay fix cycle; trigger = a full-workspace re-run after the previous full attempt timed out under load 80+; `ci_boundary_contract.sh` exit 0 — **34 repos, 535 cross-repo dep edges measured**, summary line clean; en-route this lane landed riir-infer `698fc5b2` [1204 claim] + `be7fa79a` [the replay fix wip] + `9a4b11f7` [cell-validated close + issue removal + HISTORY row + the 1200 resume-anchor re-anchor] + `bb63d1c9` [the selection_sample free fn — the committed closure form was E0503-broken, the cell ran against the corrected tree; release + clippy clean at HEAD] — ALL in-crate .rs + md/deletion, zero new cross-repo edges) + `cargo refine --oracle-census` ran (report-only, the priced-not-run examples/poc rows as standing) | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + md); C8/C9 owner-call advisories stand | riir-infer `698fc5b2`+`be7fa79a`+`9a4b11f7`+`bb63d1c9` + (this commit) |
| 2026-11-11 | 396th — SCOPED script run (M3 box, the riir-refine idle-loop housekeeping lane; trigger = this cycle's Protocol-H windows: ff-merges riir-refine fda089429 [bench-150 md+data + .distill + 3 in-crate proposer .rs] + riir-reflexer f07c629 [in-crate vessel_apply.rs] + riir-train 2298201e [plan-373 in-crate .rs + data + the ONE Cargo.toml whose added lines are ONE intra-package [[test]] row + ONE same-package [[bin]] required-features row, verified in the diff] + katgpt-rs/riir-neuron-db/seal-render/seal-std/sealm-toolkit [manifest/BOUNDARY grep EMPTY across all their windows] + llama.cpp [external mirror, out of scope]) — pre-flight clean (bash -n, script not-mid-edit), `ci_boundary_contract.sh --repo riir-refine` exit 0 (20 cross-repo edges, SP5a hard-zero ✓, C7/C8 n/a no-lockfile, the vendored-patch note as printed); en-route hygiene: the landed-but-unshaped sdk 050 claim reshaped, the 118 fix-at-birth file removal completed in examples | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges; the 353rd full-run + 354th–395th verdicts stand; C8/C9 owner-call advisories stand | riir-refine fda089429+ + riir-train 2298201e + (this commit) |
| 2026-11-11 | 395th — SCOPED delta verification, NO script run (M3 box, the zed-m3-i223t4 claim-holder close-out lane; trigger = the 223-T4 landing windows landed by the yielded continuation lane zed-a647854c: riir-refine scripts/fmt_gate.py + fmt_exclusions.txt + fmt_pending_expected.txt [all in-repo python/txt, zero deps] + scripts/ci_feature_guard.sh Layer 8 [16 lines, in-repo wiring, sentinel untouched — GUARD_COMPLETED=1 still the last line] + AGENTS/README md + the 9 formatted in-repo src files + the b345 provenance byte-restore [pre-sweep a95e8c26 content re-landed, net-zero dep surface] + .issues/223 close + HISTORY row; diffstats verified) — gate reproduced by the holder: PASS 830 checked / 0 pinned / 7 tripwired / 22 excluded, exit 0; self-test arms green (unconditional pre-scan design — a failure exits 2 before any scan); lease check item 223 clean (the Claimed line folded to Lease-closed by the landing lane); the landing lane's doc-sync row 09a3aa0cb covers the content sweep | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-repo scripts + md + byte-restore); the 353rd full-run + 354th–394th delta verdicts stand; C8/C9 owner-call advisories stand; riir-refine HEAD 1e08a36ff is the sibling lane's local unpushed docs commit, as-found NOT touched | riir-refine `186661be8`+`44536a2a9` + (this commit) |
