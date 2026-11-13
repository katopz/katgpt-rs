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
Rows carried full narratives until these compactions; `git log -p -- .agents/skills/boundary-guard/SKILL.md` (the katgpt-rs repo) recovers any of them verbatim. New rows append ONE line each. **Pruned 2026-09-21: 105 → 15 rows, 71KB → 30KB; again 2026-10-26: 32 → 15 rows, 65KB → 42KB (243rd run); again 2026-10-29: 18 → 15 rows, 63.3KB → 56.1KB (312th run); again 2026-10-29: 18 → 15 rows, 60.5KB → 54.8KB (315th run) · 11-13 (41→15 incl. the 433rd row, 63.3KB trip → pruned, the 11-13 landing wave re-bloated it)** — the one-line convention held but the cadence didn't (~10 rows/day re-bloated it since the 09-11 re-compaction; third prune). **Maintenance rule: whenever this file exceeds 60KB, prune the run table to the newest 15 rows** — enforced mechanically by `scripts/skill_size_gate.py` (80KB hard ceiling, all `.agents/skills/*/SKILL.md`); recovery via `git log -p` as above.

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
| 2026-11-13 | 436th — SCOPED delta verification, NO script run (M3 box, the cross-repo delta-sweep housekeeping lane zed-m3-hk-436; trigger = the sibling windows the lane-scoped 426th–435th rows do not name: riir-train c29fa1b0+edfc969d [487 T0.2 claim+landing — in-crate zoo.rs +301 + plan md, added-line dep-edge grep over '*.toml'/'*.lock'/BOUNDARY.md EMPTY verified unpiped; lease zed-93595a2b live to 11-14T21:25Z] + riir-rethink f80fbda+4a7457b + riir-chain 368a9899 + riir-dapps 2848bc1 + riir-neuron-db e521e3c + riir-refab 940cfff [md-only ×5 — records/lease-folds/highwaters, stat-verified] + riir-reflex 052a7e6 [HISTORY rows, md] + seal-game-editor 31e851e16+b3cfde6de+03d2f08d4 [rust-developer skill md + in-repo vibe fix ×4 files + merge, dep-edge grep EMPTY verified unpiped]) — the same-day 423rd FULL verdict (33 repos, 514 edges) + the 424th–435th delta verdicts stand; en-route Protocol H: 7 clean-behind repos ff-synced (chain/dapps/reflex/neuron-db/rethink/refab/train), riir-ai dirty as-found NOT pulled (standing sibling-WIP law), riir-reflexer Cargo.lock local-dirty as-found; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only ×6 repos, in-crate .rs ×2) | riir-train `edfc969d` + riir-rethink `4a7457b` + riir-chain `368a9899` + riir-dapps `2848bc1` + riir-neuron-db `e521e3c` + riir-refab `940cfff` + riir-reflex `052a7e6` + seal-game-editor `03d2f08d4` + (this commit) |
| 2026-11-13 | 435th — SCOPED delta verification, NO script run (M3 box, the 1191/1213 continuation lane zed-d2d18aff-c2; trigger = this lane's OWN windows since the 434th: riir-refine 76aea958d+27500a69a [md-only: the 11-13 HISTORY doc-sync rows + the 218 CI-watch poll row — 27500a69a this lane's own landing]; added-line dep-edge grep over d76ec2737..HEAD '*.toml'/'*.lock'/BOUNDARY.md = 0 bytes, verified unpiped; en-route this unit: riir-instinct + seal-remake clean-behind ff-synced, riir-ai dirty-39 as-found NOT pulled, katgpt-rs ff-merged over the 432nd-434th lane race) — the 434th/433rd/432nd rows + the 423rd full-run verdict (33 repos, 514 edges) + the 403rd–434th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only ×2) | riir-refine `76aea958d`+`27500a69a` + (this commit) |
| 2026-11-13 | 434th — SCOPED delta verification, NO script run (amo box, the plan-646 D2 second-frozen-seed lane amo-idle-646d2; trigger = this lane's OWN windows ×2: riir-ai f111d6856 [646 D2 claim + pre-registration — plan md ±1] + 7e5e51161 [the D2 cell — in-crate riir-engine tests/cgsp_diffusion_div_batch_ab.rs +106 (the d2_second_frozen_seed test: three arms at the pre-registered seed 0x0646_00A3, report-only, determinism pins) + plan md record; added-line dep-edge grep EMPTY verified unpiped: no path=../, no git+, zero Cargo.toml/lock/BOUNDARY hunks — the test joined the file's existing required-features row, zero manifest change] — pre-flight clean (test 3/3 green incl. both frozen reads; lib 3004; clippy clean default + feature-on; numbering gate PASSED fresh riir-ai) — the same-day 423rd FULL verdict (33 repos, 514 edges, exit 0) + the 424th–433rd verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate test + plan md) | riir-ai `7e5e51161` + (this commit) |
| 2026-11-13 | 433rd — SCOPED delta verification, NO script run (amo box, the seal-remake sprint!023 overclock-campaign continuation housekeeping lane; trigger = this lane's OWN windows ×2: seal-remake e4d3b11d^..5d43b0f7 [fix e4d3b11d warp-chain patcher + feat c82d88c6 the sim-overclock env arm SEAL_SIM_OVERCLOCK/SEAL_SIM_CLOCK + docs ×3 — shell scripts + in-crate seal-view .rs + md, added-line dep-edge grep EMPTY verified unpiped: no path=../, no git+, zero Cargo.toml/lock/BOUNDARY hunks] + seal-sprints f01c2e1..4485b6c [sprint!023 close + sprint!027/028 filings + TASKS.md — md-only ×2]) — the same-day 423rd FULL verdict (33 repos, 514 edges, exit 0) + the 424th–432nd verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + shell scripts + md ×2 repos) | seal-remake `5d43b0f7` + seal-sprints `4485b6c` + (this commit) |
| 2026-11-13 | 432nd — SCOPED delta verification, NO script run (M3 box, the 1191 resume/verification lane zed-c6a1462a; trigger = this lane's OWN window since the 431st: riir-infer d5f3ebc5 [docs(1203) prior-art census — .issues md only, +31 lines, web+vendor grep, no GPU]; added-line dep-edge grep over f43cc6c3..HEAD over '*.toml'/'*.lock'/BOUNDARY.md EMPTY, verified unpiped; numbering gate PASS fresh exit 0, legacy-collision advisories only) — the 431st scoped verdict (same day) + the 430th scoped script verdict (13 cross-repo edges) + the 423rd full-run verdict (33 repos, 514 edges) + the 403rd–431st delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only) | riir-infer `d5f3ebc5` + (this commit) |
| 2026-11-13 | 431st — SCOPED delta verification, NO script run (M3 box, the 1191/1213 supervision lane zed-ed7bd295 → continuation zed-d2d18aff, E3-close windows; trigger = this lane's OWN windows since the 430th: riir-infer 4c69ea71 [ONE .rs line — the i1203_counters_probe import path fix, E0603 root-path spelling, in-crate only] + 2a2fd916 [ONE .issues md line — the E3 compile-VERIFIED record] + f43cc6c3 [AGENTS.md 1182 lane-row synced to the measured Bench 966 verdict — md-only]; added-line dep-edge grep over 1e384d9d..HEAD over '*.toml'/'*.lock'/BOUNDARY.md EMPTY, verified unpiped) — the 430th scoped script verdict (13 cross-repo edges, same day) + the 423rd full-run verdict (33 repos, 514 edges) + the 403rd–430th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + md) | riir-infer `4c69ea71`+`2a2fd916`+`f43cc6c3` + (this commit) |
| 2026-11-13 | 430th — SCOPED delta verification WITH scoped script run (M3 box, the 1191/1213 supervision + 1203 E3-arming lane zed-ed7bd295; renumbered 422nd→430th at rebase per-append monotonicity — upstream 423rd–429th landed while this row sat local; trigger = this lane's OWN windows: riir-infer 9e0714b9+f3ccb167 [md-only: 1191 claim scope + 1213 status refresh + claim line] + 4439b1bf+1e384d9d [the i1203_counters_probe bin + ONE riir-infer-gpu Cargo.toml whose added lines are the intra-package [[bin]] required-features row + the optional dep:cubecl-wgpu row on the ALREADY-IN-GRAPH vendored fork (root [patch], one lockfile instance — zero cross-repo surface: no path=../, no git+, zero BOUNDARY hunks) + the runner's E3 build step (in-repo script) + 1203 issue md]) — scoped script `ci_boundary_contract.sh --repo riir-infer` exit 0 (13 cross-repo edges, matches the 417th graph); the 423rd full-run verdict (33 repos, 514 edges, exit 0) + the 403rd–429th delta verdicts stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (one optional dep row on an in-repo vendored crate + intra-package bin row + md + in-repo script) | riir-infer `4439b1bf`+`1e384d9d` + (this commit) |
| 2026-11-13 | 429th — SCOPED delta verification, NO script run (4090 box, the 679-wait continuation housekeeping lane shikuwa-4090-hygiene; trigger = this lane's OWN windows since the 425th: riir-refine d76ec2737 + e26233353 [plan-283 terminal-park truth-up + the 7-plan checkbox reconciliation — md-only ×7, added-line dep-edge grep over '*.toml'/'*.lock'/BOUNDARY.md EMPTY verified unpiped; the windows are NOT covered by the intervening 426th–428th rows, which carry riir-train/riir-infer/amo-refine windows]) — same-day FULL verdicts stand: the 423rd (33 repos, 514 edges, exit 0) + the 428th lane's own full run; C8/C9 owner-call advisories stand; en-route Protocol H on this box: gitsync SYNC DONE (all repos current), the 11-12 21:27 orphan git+ssh tree re-verified parent-dead + ELEVATED (kill stays the owner call), disk C 123.8 / E 103.8 GiB free, riir-refine tree clean at origin, numbering gate PASSED fresh (939 files, 0 duplicates), linkcheck clean on all 7 touched plan files | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only) | riir-refine `d76ec2737`+`e26233353` + (this commit) |
| 2026-11-13 | 428th — SCOPED delta verification, NO script run (amo box, the b352-lone continuation housekeeping lane; trigger = this lane's OWN windows since the 423rd full + the same-session full run: riir-refine f0d93666 [frontier-snapshot row append, data-only] + 3c6359ea [Layer 9 bin-target combo gate — TWO scripts + guard wiring + AGENTS.md + heal.rs cfg attrs, added-line dep-edge grep ZERO: no path=../, no git+, zero Cargo.toml/lock/BOUNDARY hunks, verified unpiped] + katgpt-rs be37c3835 [issue 1208 filing — md + highwater only]) — pre-flight clean (script committed state, bash -n, exit read unpiped); same-day FULL verdicts stand: the 423rd (33 repos, 514 edges, exit 0) AND this session's own full run (ci_boundary_contract.sh exit 0, 33 repos, 514 edges, boundary contract clean, run pre-rebase at the riir-refine 3c6359ea landing); C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (scripts + in-crate cfg attrs + md + data rows) | riir-refine `f0d93666`+`3c6359ea` + katgpt-rs `be37c3835` + (this commit) |
| 2026-11-13 | 427th — SCOPED delta verification WITH scoped script run (amo box, the issue-1209 weight-audit landing lane; renumbered 425th→427th at rebase — the 4090 lanes’ own 425th (679-T3) and 426th (plan-487 T0.1) landed first, per-append monotonicity kept; trigger = this lane’s OWN window: riir-infer 61da4d8^..faf918f [feat 3dc77e2 — the weight_audit feature + weight_audit.rs + weight_audit_fixture.rs + the weight-audit bin + ONE Cargo.toml whose 18 added lines are the pure-local weight_audit feature row + the intra-package [[bin]] weight-audit required-features row + comments, path=../ 0, BOUNDARY hunks 0, diff verified; NO Cargo.lock change (blake3 already in-tree, zero new deps) + docs 61da4d8+faf918f md-only] — pre-flight clean (bash -n, script committed state, exit read unpiped); scoped script ci_boundary_contract.sh --repo riir-infer exit 0 (13 cross-repo edges, unchanged graph); the same-day full-run verdicts (33 repos, 514 edges) stand; benchmark routine N/A (offline report-only lane, zero serving-path cost, no perf claim); C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (pure-local feature + intra-package [[bin]] row + in-crate .rs/tests + md) | riir-infer \`3dc77e2\`+\`61da4d8\`+\`faf918f\` + (this commit) |
| 2026-11-13 | 426th — SCOPED delta verification, NO script run (4090 box, the plan-487 T0.1 landing lane zed-93595a2b; trigger = this lane’s OWN windows: riir-train d407da0c [486 claim-line reshape to the parseable form (epoch= key=value → bare seconds epoch, scope to its own line, fields preserved) + 487 claim — md-only ×2] + dfe4e9ae [487 T0.1 LANDED: in-crate crates/riir-train-engine/src/zoo.rs (332 lines, new file) + ONE engine Cargo.toml whose 8 added lines are the intra-package wsl_warm_start = [] feature row + comments + lib.rs gated mod decl + plan md T0.1 [x]/Status/lease-fold — added-line dep-edge grep EMPTY verified (no path=../, no git+, zero BOUNDARY hunks; serde/serde_json/blake3 already declared; crate-internal atomic_replace reuse)]) — the same-day 425th scoped (md-only) + 424th scoped-script + 423rd FULL verdicts (33 repos, 514 edges, exit 0) stand; the 487 lane is zero-GPU infrastructure under the live 679 run 5 (builds on an isolated /tmp/plan487 target dir, no GPU cells); C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + intra-package feature row + md) | riir-train `d407da0c`+`dfe4e9ae` + (this commit) |
| 2026-11-13 | 425th — SCOPED delta verification, NO script run (4090 box, the 679 T3 continuation housekeeping lane zed-d6dc12b6; trigger = this lane's OWN windows since the 424th: riir-train 2b855764 [sibling docs(679) attribution-correction — issue md ±1, diff verified] + e3e9f490 [docs(679) record correction + lease extension — issue md ±3, diff verified; the md-only predicate ran with the live trainer holding 38.65 GB commit / 17 GiB VRAM, so the 420th no-script precedent applies — box at ~55/60 GB commit]) — the same-day 423rd FULL verdict (33 repos, 514 edges, exit 0) + the 424th scoped verdict stand; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md-only ×2) | riir-train `2b855764`+`e3e9f490` + (this commit) |
| 2026-11-13 | 424th — SCOPED delta verification WITH scoped script runs ×2 (amo box, the plan-373 continuation housekeeping lane; trigger = windows since the 423rd: katgpt-rs 52aa55349+43fe4b9e2 [plan-634 T2 batteries — ONE Cargo.toml whose 36 added lines are intra-package [[test]] required-features rows on the EXISTING diffusion_diversity feature + comments, path=../ 0, BOUNDARY hunks 0, diff read directly] + riir-infer d4b4b67..497d540 [issue-1201 T3 GDS retention + 1167 re-anchor — in-crate riir-infer-gpu .rs/tests + md + bench highwater + one issue-file removal, dep-surface diff EMPTY] + riir-refine b47ded60^..fcd21290 [Batches 354+355 — corpus/fixtures/router/tests + pins_manifest data + md, '*.toml'/'*.lock'/BOUNDARY diff EMPTY]) — pre-flight clean (bash -n, committed state); scoped script --repo katgpt-rs exit 0 (0 cross-repo edges) + --repo riir-infer exit 0 (13 edges); the 423rd FULL verdict (33 repos, 514 edges, same day) stands; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs/tests + intra-package [[test]] rows + md) | katgpt-rs 52aa55349 + riir-infer 497d540 + riir-refine fcd21290 + (this commit) |
| 2026-11-13 | 423rd — FULL double script verdict (amo box, the idle-loop housekeeping lane; trigger = routine unit — no dep-edge windows since the 422nd: the 11-12/11-13 workspace windows reviewed via commit scan are in-crate .rs + md/claim folds + merges carrying their own gate records; no Cargo.toml/BOUNDARY surface spotted) — workspace ci_boundary_contract.sh exit 0 (33 repos, 514 cross-repo dep edges — the set/edge counts MOVED down from the 416th's 34/535 with the mmorpg-remaster retirement; SP1–SP5 green, C6 clean, C7/C8 not-checked lists as printed) + riir-mmorpg-examples ci_boundary_guard.sh exit 0 (5 checks green, invoked via bash per the sh-trap lesson); pre-flight clean (script committed state, mtime Nov 8, bash -n) | landed-state CLEAN — 0 violations / 0 rot; C8/C9 owner-call advisories stand (2 pins, multi-rev disagreement unchanged) | (this commit) |
| 2026-11-12 | 422nd — SCOPED delta verification WITH scoped script run (amo box, the plan-373 overflow-continuation housekeeping lane; trigger = this lane's OWN windows since the 417th: riir-train 89e6e4cc..8d584857 [P2.0 feat 9e647e90 — in-crate riir-train-gpu .rs x4 + engine late_state_loop/train.rs hoist + TWO test files + ONE riir-train-gpu Cargo.toml whose added lines are the intra-package [[bin]] plan373_gpu_train row + comments, path=../ count 0 verified unpiped; docs 8c7bdf6b+8d584857 md+ps1-only] + en-route sibling window NOT mine: 285e049a [688 md + highwater]) — pre-flight clean (script committed state, mtime Nov 8, bash -n); scoped script ci_boundary_contract.sh --repo riir-train exit 0 (31 cross-repo edges, matches the 402nd/406th/411th/413th graph) + the m3 421st full-run verdict (33 repos, 514 edges, same day) stands; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (in-crate .rs + intra-package [[bin]] row + one lockfile metadata line + md/ps1) | riir-train `9e647e90`+`8c7bdf6b`+`8d584857` + (this commit) |
| 2026-11-12 | 419th — SCOPED delta verification, NO script run (4090 box, the boss-run housekeeping lane zed-642f7c51; trigger = this lane's OWN windows since the 418th: riir-train 5db257c6..edb4a7f9 [my 8 commits: docs(679) ×5 md-only + docs(486) claim md + feat(486) in-crate riir-train-engine .rs ×2 new + 2 edited + tests ×1 new — per-commit manifest/lock/BOUNDARY grep verified EMPTY across all 8; the sibling plan-373 lane's riir-train-gpu Cargo.toml +26 rows in the wider range are THEIR in-crate feature rows, already graphed by their own 412th/413th records] + fetch-only syncs of riir-auth/deployer/game-sdk/reflexer/rethink/infer [zero tree changes]) — the feat(486) landing adds run_digest.rs (ungated, std+blake3 base deps, no new dep) + the lambda seam behind the EXISTING degrpo feature (no Cargo.toml change) — zero new cross-repo edges by construction; C8/C9 owner-call advisories stand | landed-state CLEAN — 0 violations / 0 rot; zero new cross-repo edges (md + in-crate .rs/tests, no manifest rows) | riir-train `dc6c3bc9`+`edb4a7f9` + (this commit) |
| 2026-11-12 | 420th — FULL script run, PARTIAL capture (4090 box, the 679 handover idle lane zed-bcb28b85; trigger = queue-item-6 routine check under the Plan-337 zero-contention carve-out while the 679 arm-1 trainer holds the GPU + ~28/32 GB RAM; ×3 attempts — first captured C0 / C0e 587 refs / C1+C2 34 contracts / C3 / C3b / SP1–SP4 ALL GREEN plus the standing vendored-patch notes + 5 C10 LOUD gh-skips; C7–C9 tail + exit code NOT captured — the box's RAM pressure outlasted the 300 s and 600 s budgets and a detached-log attempt died with the tool's process-group kill, third run captured nothing before kill) — C7/C8/C9 advisories stand per the SAME-DAY 416th full run (amo, exit 0, 34 repos/535 edges) + 418th scoped run (this box); NO issue filed (every captured check green, zero findings) | majors CLEAN on this box this hour; zero new cross-repo edges since the 419th (only this lane's own md rows) | (this commit) |

