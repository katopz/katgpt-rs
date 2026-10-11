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
Rows carried full narratives until these compactions; `git log -p -- .agents/skills/boundary-guard/SKILL.md` (the katgpt-rs repo) recovers any of them verbatim. New rows append ONE line each. **Pruned 2026-09-21: 105 → 15 rows, 71KB → 30KB** — the one-line convention held but the cadence didn't (~10 rows/day re-bloated it since the 09-11 re-compaction; third prune). **Maintenance rule: whenever this file exceeds 60KB, prune the run table to the newest 15 rows** — enforced mechanically by `scripts/skill_size_gate.py` (80KB hard ceiling, all `.agents/skills/*/SKILL.md`); recovery via `git log -p` as above.

### Standing lessons (distilled from the compacted rows — load-bearing process rules)

- **Read the exit UNPIPED.** `cmd | tail; echo $?` reads TAIL's exit (re-proven twice); outputs go to /tmp, read the real exit or PIPESTATUS. A piped gate summary cannot vouch for an exit code — and a green summary over an unstaged body is worse: the 20th run's "exit 0" was FALSE while C0e's implementation sat unstaged (`a6ca9ccbd` docs-only; body landed `10977dd23`).
- **Verify the script not-mid-edit before running**: mtime vs the last recorded run + `bash -n` + clean in `git status`. A sibling rewriting the script mid-run truncated a measurement after SP1 while the first attempt still reported exit 0 (the 2026-09-04 membership row) — bash re-reads by byte offset. The script anchors ROOT from its own path, so /tmp snapshot-copy immunization CANNOT work; diff-vs-snapshot before running the live script instead.
- **Parser replay for stale shared checkouts**: run the script's own allowlist awk against `git show origin/<default>:BOUNDARY.md` — a sibling branch predating a row reads exit-1 on disk while main is CLEAN (the riir-train until-merge artifact class); touching the sibling branch is forbidden.
- **Detection-only discipline**: file `.issues/NNN_boundary_*.md` BEFORE any fix; sibling WIP (staged sets, mid-flight sweeps, dirty manifests) is measured as-found and never fixed by the idle unit — the sweep owner re-pins C6 at closeout (the `2575b6519` precedent). A landed growth WITHOUT its re-pin, though, is repaired by whoever finds it once the landing session is gone (the honest up-pin class: verify zero training semantics, pin UP with the reason).
- **The C6 ledger row text must be apostrophe-free** and the real detector is `bash -n`, not running the guard — bash 3.2's quote tracker reports the wrong line (~20 below the guilty one).
- **Expected shapes, not findings**: the 2 known-drift rows (the ledgered mmorpg-remaster pair); C7 = 4 repos with no tracked lockfile; scoped `--repo` runs print C7/C8 out-of-scope notes only.
- **C8/C9 standing advisory (owner call, never an idle-unit act)**: consumer lockfiles pin katgpt-rs `@6f392727` and drift further behind develop each window — 326 (09-03) → 739 commits behind with 7 security fixes in the gap (09-11, the 70th run) across 3 lockfiles, plus a second divergent pin in riir-chain (167/1). Recorded every run; the bump is the owner's. **Gap ENUMERATED 09-10 with the script's own predicate (689 behind at the 37th run, `@c478ab9f` chain pin 0 security): the 6 = 2× SIMD soundness (`99afbab9` safe-code OOB read + heap-corrupting write in `katgpt-types/src/simd/` — CWE-125, `simd_dot_f32` unchecked-`len`; `5b028c00` the near-verbatim katgpt-dec twin) + 2× FFI/panic-UB hardening (`af2e48e1` catch_unwind on 10 wasmi extern fns; `e854958e`/`96543f4f` the 208-site NaN-comparator sec heal) + `2dec22a1` (a filed, not-fixed, EngramHotSwap nested-swap soundness hazard) — the sharp half is that the two SIMD holes are MEMORY-SAFETY bugs in the exact substrate the three deploy lockfiles compile today, so the bump is a soundness call, not a freshness preference. **RESOLVED 2026-09-11 (owner call executed, subagent re-verdicted): all three consumer locks bumped `6f392727 → c478ab9f`, and the discovery corrected the advisory's own premises.** The 3 tracked lockfiles are riir-dapps `cloudflare/kat-service`, riir-mmorpg-examples `cloudflare/warm-tier-do`, and riir-esp32 `crates/riir-satellite-probe` — NOT riir-kat/riir-dao, which carry no tracked lock at all (the C7 class). The consumers pin `branch = "main"`, so the reachable target was main tip `c478ab9f`, which already contains ALL 6 enumerated fixes; the 7th grep hit in the develop-only range is a docs commit (the advisory's own enumeration) — a **predicate false-positive class**: the sec-subject grep matches docs commits whose subject contains "soundness". riir-chain's "divergent pin" re-verdicted to the same answer: it already sits at main tip `c478ab9f`, zero action needed. All builds/tests green per consumer (kat-service 143 tests + wasm32; warm-tier-do 4 + wasm32; satellite-probe esp32c6 release). Commits: dapps `c9203a2`+`6342519`, mmorpg `0afb52b`, esp32 `2bfb795`. Going forward the "behind develop" count is a branch-policy artifact (`branch=main`), not a security gap — the advisory's remaining value is the C9 one-rev agreement, which this bump achieved (all 4 build roots on `c478ab9f`).**
- **Guard output truncates in `tail` views** — capture the FULL violation list when filing (the "8 = 6 unique" enumeration error).
- **Never suppress fetch output you act on** — a silenced fetch failed and produced a false empty-divergence read while the push error was right (the 59th run, 4090 box).
- **`git show HEAD:<dir>` on a directory prints the tree LISTING** — honest dir LOC is an `ls-tree -r` + `cat-file` sum. `rustfmt --emit stdout` prints a 2-line filename header — a round-trip harness must strip it.
- **The staged-set check must be a SEPARATE command** (or commit `git commit -- <paths>`): chained `add && diff --cached && commit` swept a sibling's entire staged set once (the `30127c4` incident); recovery anatomy lives in that row's history — read the reflog before any second corrective action.
- **A prose contract cannot catch an inverted measurement** — every "repo X may depend on Y" row is a measurement with a direction; that is why the contract script parses tables instead of prose (the first full run's founding lesson).

### Run table

| Date | Run | Verdict | Record |
|---|---|---|---|
| 2026-10-11 | 190th — SCOPED `--repo riir-reflex` (4090 box, riir-refine idle-loop unit 6 post-Protocol-H; trigger = manifest movement since the 189th: riir-reflex `658e920` the code_fixtures retirement — Cargo.toml 19 lines: comment rewording + ONE [[example]] gen_code_frozen row removal, zero dep-table hunks) + this session's own reflex `9e517d2` / reflexer `68cdaae` (md + rust-toolchain.toml only, non-manifest); riir-rethink `4413bc8` file-verified arsenal.toml-only (zero contract surface); script verified not-mid-edit: Oct-7 18:50 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg190_reflex.log; Plan-337 note: manifest-parse only, zero cargo, under the riir-train 623 banking77 capture (~72% CPU)) | **scoped exit 0 — 1 repo, 5 cross-repo dep edges, 0 violations / 0 rot — the code_fixtures retirement is dep-graph-neutral (target-row removal only); C0e 15 refs 0 unresolved; C1/C2 green; C3b armed-quiet; C7/C8 green** | — |
| 2026-10-11 | 189th — SCOPED landed-state verification, NO script run (M3 box, riir-refine idle-loop overflow-handoff unit 6; trigger = the 188th's own deferral — "landed-state of the incoming range deferred to the owning lane" — riir-ai now behind-42 and STILL dirty (combo lane live), so a script re-run would re-measure the same stale local manifests; the landed state measured by the 161st honest-read shape instead: `git diff HEAD...origin/develop` over ALL `*Cargo.toml`/`*BOUNDARY.md`/`*.lock` across the FULL 42-commit incoming range = exactly ONE contract-surface file, `crates/riir-games-mmorpg/Cargo.toml` +47 (the Issue-1049 `sync_relation_gate`/`sync_relation_ltm` feature rows + two [[test]] registrations), both features forward PRE-DECLARED optional deps (`riir-engine` manifest line 1092, in-repo path dep — zero cross-repo edge; `riir-neuron-db` line 1084 with BOUNDARY.md row-179 naming `riir-games-mmorpg` PRE-DECLARED) — zero `[dependencies]`-table hunks, zero new cross-repo edges; the 42nd commit `1a1830dce` (quest-grammar determinism) .rs-only; the other three repos' incoming +1s file-verified zero manifest surface (riir-infer `376644f` staging note, riir-refine `0981c3a2` claim row, riir-reflex `ff27637` bench artifacts); riir-ai 1052 verdict CLAIMED by the shikuwa/4090 lane (48h) — yielded per the claim protocol; Plan-337 note: diff/parse only, zero cargo, under the 929 chat probe (~9 cores)) | **landed-state CLEAN — the 188th's deferral DISCHARGED: the 42-commit incoming range adds zero contract surface beyond the already-C3-declared 1049 feature forwards; 0 violations / 0 rot in the delta; stale-local-checkout disclosure stands (as-found, combo lane live)** | — |
| 2026-10-11 | 188th — full workspace (M3 box, riir-refine idle-loop unit 6 post-Protocol-H; trigger = manifest movement since the 183rd M3-full view: riir-ai +40 incoming (HEAD..origin, combo lane landed Issue-1049/1055 series incl. `crates/riir-games-mmorpg/Cargo.toml` surface) + riir-reflex +1 + the 187th's three scoped repos' landings now in the walk; as-found per the 183rd precedent: riir-ai DIRTY behind-40 (live combo lane: .issues/971 + riir-engine test + .worktrees/) → stale-manifest disclosure, landed-state of the incoming range deferred to the owning lane per the parser-replay rule; riir-ai-combo worktree detached-dirty quest_combat manifests as-found; riir-reflex behind-1 dirty = bench tables + pycache only, zero manifest surface; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg188_full.log; Plan-337 note: manifest-parse only, zero cargo, under the 929 chat probe (~9 cores) + a plan454 cargo) | **exit 0 — 28 repos / 367 cross-repo dep edges (+2 vs the 183rd's 365 — the 187th's local landing set: seal-remake warm-mirror feature row + riir-infer bench staging + riir-train plan-404 lane, all C3-declared), 0 violations / 0 rot — the 183rd's 26 SP5a work-t5dump findings SELF-RESOLVED (the dapps session's nested worktrees are gone); C0e 428 refs 0 unresolved; SP1–SP5b green; C6 0 LOC; C7 = 7 no-lock (expected shape); C8 four-rev split + C9 multi-rev stand owner call (documented branch-policy class) | — |
| 2026-10-11 | 187th — SCOPED ×3 (M3 box, riir-refine idle-loop overflow-handoff unit 6; trigger = post-186th manifest movement: seal-remake `c550895` warm-mirror feature row + `cf43146` BOUNDARY.md D1-fragment drop, riir-infer `32fd838` multicol_ab bench staging, riir-train `1bfe023b` plan-404 GPU lane — all 09:32–09:53, postdating the 183rd M3-full view; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exits unpiped; Plan-337 note: manifest-parse only, zero cargo, under the 929 probe at 893% CPU / 24 GB RSS) | **all three scoped exit 0 — seal-remake 35 edges, riir-infer 10 edges, riir-train 26 edges; 0 violations / 0 rot across the moved set — the warm-mirror row, bench staging and plan-404 lane are all contract-clean** | — |
| 2026-10-11 | 186th — SCOPED `--repo riir-refine` + S7 grep (M3 box, Decision-ordered housekeeping unit, riir-refine idle-loop overflow-handoff continuation; trigger = manifest movement since the 185th full view ran 08:1x: riir-refine `ed42dd7f` 08:53 (v0.3.0 release prep — manual_release.sh re-pointed at the release.yml set + dist profile; ONE Cargo.toml line, measured graph-neutral below) + `c4fc0a4c` 09:02 (T16b consent line, zero manifest surface); script verified not-mid-edit: Oct-7 18:42 mtime + bash -n + clean scripts/ status; foreground + </dev/null, exit unpiped to /tmp/bg186_refine.log; Plan-337 note: manifest-parse only, zero cargo, under ~16 sibling rustc at CPU 100%) | **scoped exit 0 — 1 repo, 15 cross-repo dep edges (== the 182nd's post-fix count), 0 violations / 0 rot — the release-prep Cargo line is dep-graph-neutral; C3b armed-quiet; SP5a green; C7/C8 out-of-scope notes only (no tracked lockfile, expected shape); S7 grep clean — zero game/chain dep lines in riir-refine + riir-kat** | — |
| 2026-10-11 | 185th — full workspace (4090 box, riir-refine idle-loop unit 6; DOUBLE NUMBERING REPAIR: (1) the run self-numbered "182nd" at run time from a STALE LOCAL LOG — katgpt-rs sat behind 27 + dirty, the M3's 178th–183rd rows invisible; (2) the first landing attempt (50f7e45e6, never pushed) numbered itself 184th and was rejected by the push arbiter — the M3's SCOPED riir-dapps 184th (ce0a88400) landed in the window; true landing-order position = 185th; ran 08:1x +07, before both same-day rows — landing order wins; FIRST 4090 run since the 162nd (09-30); script verified not-mid-edit: Oct-7 18:50 mtime + bash -n + clean scripts/ status; runs foreground + </dev/null, exits read unpiped from /tmp/bg182_4090.log + /tmp/bg182b_4090.log) | FIRST RUN exit 2 — seal-remake's drift ledger UNPARSEABLE (the sibling issue-060 cycle's detection commit 8897823 swapped the sanctioned "None." statement for a "| Row |"-headered table the parser's ID-only header-skip read as a data row; the close-out commit c5b5d4c then dropped the D2 row, leaving header+separator with ZERO live rows — neither-table-nor-None) → ROOT-CAUSE REPAIR in-run (issue-first): seal-remake 6aca284 (.issues/061 filed+closed same commit, highwater 060→061, the sanctioned "None." restored — live rows empty by the sibling's own close-out); RE-RUN with BOUNDARY_PARTIAL_CLONE=1 (this box legitimately lacks seal-std, born 10-07, never cloned here) → exit 0 — 27 repos / 367 cross-repo dep edges (+1 vs the 181st's 366 = the declared issue-060 seal-anim DEV-ONLY edge), 0 violations / 0 rot; 2 known-drift (the ledgered seal-game-editor pair) + 1 loud partial-clone DEFERRAL (seal-remake's ../seal-std route — box-local, resolves on the M3's full view); SP1–SP5b green, C3 ✓, C0e 420 refs scanned, C6 0 LOC, C7 = 7 no-lock (expected shape), C8 four-rev split + C9 multi-rev stand owner call; NOT repaired (side observation, recorded in 061): pre-existing duplicated dangling "(D1 —" prose fragment under seal-remake's ledger (cosmetic, parser-invisible) | seal-remake 6aca284 |
| 2026-10-11 | 184th — SCOPED `--repo riir-dapps` + direction_gate (M3 box, dapps issue-119-T16b session's post-landing housekeeping unit; trigger = OWN landing `c80cd9c` (docs-only, `.issues/119` T16b tick) + riir-refine `c4fc0a4c` (bin heal.rs + test — zero manifest surface, file-list verified); script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg_dapps_t16b.log; Plan-337 note: manifest-parse only, zero cargo, under the v0.3.0 release build + 5 live sibling lanes) | **scoped exit 0 — 1 repo, 17 cross-repo dep edges, 0 violations / 0 rot; SP5a green (dapps declares zero games deps); C7 lockfile green; C8 out-of-scope note (scoped run); direction_gate.sh exit 0 — game → dapps → chain holds** | — |
| 2026-10-11 | 183rd — full workspace (M3 box, riir-refine idle-loop unit 6 post-Protocol-H; trigger = manifest movement since the 182nd: riir-ai +36 incoming (HEAD..origin) touching `crates/riir-games-mmorpg/Cargo.toml` — the Issue-1049 T2/T3 `sync_relation_gate`/`sync_relation_ltm` feature rows forwarding already-declared edges (dep:riir-engine / riir-games/crowd_mcgs / dep:riir-neuron-db per its own comment); riir-ai DIRTY with live sibling work (issue-971 md + a test rs + .worktrees/) → checkout left as-found pre-pull, the stale-manifest disclosure rides the verdict; riir-infer/refine incoming = .md/.distill only; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; detached nohup + polled log (the 180th timeout-kill lesson) → /tmp/bg183_full.log; Plan-337 note: manifest-parse only, zero cargo, under 5 live sibling lanes at CPU 90%) | **exit 1 as-found, adjudicated IN-RUN — 28 repos (unchanged; the riir-chain `.work-t5dump/*` entries are `git worktree`s with .git FILES, never discovered as repos), C0/C0e (428 refs)/C1/C2/C3/SP1–SP4 all green on the canonical surface; the 26 SP5a findings are 100% nested-worktree artifacts — the live dapps dapp-vessel session's `.work-t5dump/{riir-train,riir-ai}` manifests read by the walk and attributed to riir-chain's game-free domain (the 162nd `dq614-wt` scratch-clone class) — LEFT AS-FOUND per detection-only, owning lane LIVE, self-resolves at their cleanup; 0 rot; C8 four-rev split unchanged, C9 owner call | — |
| 2026-10-10 | 182nd — SCOPED `--repo riir-refine` (M3 box, riir-refine idle-loop overflow-handoff continuation; trigger = manifest movement since the 181st: riir-refine `b41e1403` issue-157 P0 landing added the [[example]] Cargo row (+10, the p157 census instrument) + dapps `4628637` rustfmt pass + riir-ai `ab9315760` quarry demotion + seal-remake `298d3ea` — all three file-list-verified .rs/.md-only, zero manifest surface; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg182_refine.log; Plan-337 note: manifest-parse only, zero cargo, under 4 live sibling lanes at CPU 100%) | **scoped exit 0 — 1 repo, 15 cross-repo dep edges (== the 172nd post-fix count — the [[example]] row is dep-graph-neutral, the required-features-target-row shape), 0 violations / 0 rot; C3b armed-quiet; C7/C8 out-of-scope notes only (no tracked lockfile, expected shape)** | — |
| 2026-10-10 | 181st — full workspace (M3 box, riir-refine idle-loop unit 6 post-Protocol-H; trigger = manifest movement since the 180th (same day): riir-ai plan-629 graft phases `4975bd469`+`fa2cf4e76` (16:03/16:30, riir-games-shared feature/[[test]]/[[bench]] rows) + riir-ai `386e60f2d`/`7509e9029` hero_rest_spot + raw_wake game-boot feature rows + riir-infer desc_len `483f2ec` (16:18) + dapps `19af631` curve25519-dalek optional (external, non-edge) + seal-remake `7345c44`/`75d6ae5` rest-spot/raw_wake feature forwards + katgpt-rs event_state_windows default-promote `f27a031c2`; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg181_full.log; Plan-337 note: manifest-parse only, zero cargo, under 5 live sibling lanes at load ~12) | **exit 0 — 28 repos / 366 cross-repo dep edges (+1 vs the 180th's 365 — today's feature-row landing set, ALL C3-declared; per-commit attribution of the single-edge delta not uniquely resolvable without the 180th's edge list — every candidate is feature-forward/internal shape), 0 violations / 0 rot; C0e 427 refs 0 unresolved; SP1–SP5b green; C6 0 LOC; C7 = 7 no-lock repos (expected shape); C8 four-rev split (@3c844aeb 1446/0 + @6a9d4d5a 85/0 + @c478ab9f 1716/1-sec + @ecce691c 531/0 — documented branch-policy class), C9 multi-rev stands owner call | — |
| 2026-10-10 | 180th — full workspace (M3 box, this session's queue unit 6 after the raw_wake handoff (1050-lane residue); trigger = manifest movement since the 179th: the riir-infer Issue-929 T1.2b probe lane `9d67e67` (feature-gated, composes existing edges) + the RUN-lane siblings' seal-remake/game-sdk commits + docs-only self commits — whether any added dep surface was unresolvable from the sweep alone; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; detached nohup + polled log (the foreground attempt was killed by a tool timeout mid-SP1 under the 929 probe's 9-core load — first run of the timeout-kill class, log was a partial, relaunched); Plan-337 note: manifest-parse only, zero cargo, under the 929 probe measurement) | **clean — 28 repos / 365 cross-repo dep edges (== the 179th's 365, ZERO new edges — the probe lane + RUN-lane commits are dep-graph-neutral), 0 violations / 0 rot, C6 0 LOC; C7 = 7 no-lock repos (expected shape); C8 four-rev split + C9 multi-rev stand owner call (documented branch-policy class, unchanged) | — |
| 2026-10-10 | 179th — full workspace (M3 box, riir-refine idle-loop unit 6; trigger = OWN sync: Protocol-H ff-pulled riir-ai `fbdfbb2e7` 23:52 (postdates the 178th's 22:44 log) — the Issue-1046 wgpu-hal `[patch]` re-point at the canonical fork home `../riir-infer/vendor/wgpu-hal-30.0.0` (drift-dead twin forks, S6b precedent) = the +1 edge; riir-ai `9a61db9ab` + riir-infer `48d598e`/`193744c` are .rs/docs-only, zero manifest surface; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg179_full.log; Plan-337 note: manifest-parse only, zero cargo, under the nice-19 hyperthink census at load ~11) | **exit 0 — 28 repos / 365 cross-repo dep edges (+1 vs the 178th's 364 — the wgpu-hal patch re-point, a declared riir-ai → riir-infer vendor edge, same direction as the carve-era edges), 0 violations / 0 rot; C7 = 7 no-lock repos (expected shape); C8 four-rev split + C9 multi-rev stand owner call (documented branch-policy class, unchanged) | — |
| 2026-10-09 | 178th — full workspace (M3 box, riir-refine idle-loop unit 6 post-Protocol-H; trigger = post-177th manifest movement of ambiguous ordering: the katgpt-rs-928 green-zero arming series (required-features rows: katgpt-rs `5313ed1fe`+`32908a05f`, riir-train `1892808`, riir-kat `7afdd40`, seal-game-editor `54dd734`) + the riir-chain/riir-dapps dapp-vessel tail (`88311b03`+`e6cfe466`) + seal-remake plan-021 sidecar (`d6482d7`) — whether any postdate the 177th was unresolvable from the sweep alone, so the script ran for the definitive graph; script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg178_full.log; Plan-337 note: manifest-parse only, zero cargo, under the hyperthink census at load ~12) | **exit 0 — 28 repos / 364 cross-repo dep edges (== the 177th's 364 — the arming series touches [[test]] rows only, ZERO new dep edges), 0 violations / 0 rot; C7 = 7 no-lock repos (expected shape); C8 four-rev split + C9 multi-rev stand owner call (documented branch-policy class, unchanged) | — |
| 2026-10-09 | 177th — full workspace (M3 box, dedicated dapp-vessel session; trigger = OWN manifest movement: riir-chain `chain_dapp_vessel` feature + riir-dapps `dapp_vessel`/`dapp_vessel_bridge` features (all composing EXISTING allowlisted deps — ndb secure_vessel/art_vessel, ed25519-dalek, blake3, chain_domain_anchor, domain_backend) + the artb assembler re-home (zero dep surface); script verified not-mid-edit: bash -n + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg_dappvessel_177.log) | **exit 0 — 28 repos / 364 cross-repo dep edges (== the 176th's 364, ZERO new edges — the lane composes existing allowlisted deps only, the no-new-dep-edges design claim measured), 0 violations / 0 rot; C8 four-rev split + C9 multi-rev stand owner call (documented branch-policy class); C7 = 7 no-lock repos (expected shape)** | — |
| 2026-10-09 | 176th — full workspace (M3 box, riir-refine idle-loop unit 6 post-Protocol-H; trigger = contract-surface movement since the 175th its trigger row did NOT name: katgpt-rs runetrace module `f5bb0c535` + **BOUNDARY.md owns-row edit** `90352a856` (the contract-edit-owes-verification class, 166th precedent) + riir-ai `900d0e810` Plan-626 substrate landing; seal-remake's runetrace pair `604c5a5`/`721b7c8` PUSHED by the sibling mid-unit (origin/develop..HEAD empty at run time — LANDED state measured, no as-found deferral; sibling now live on fresh shop.rs WIP, untouched); script verified not-mid-edit: Oct-7 18:42 mtime + `bash -n` + clean scripts/ status; foreground + `</dev/null`, exit unpiped to /tmp/bg176_full.log; Plan-337 note: manifest-parse only, zero cargo, under the hyperthink census + 2 live sibling lanes) | **exit 0 — 28 repos / 364 cross-repo dep edges (−1 vs the 175th's 365, sibling landing, all declared — C3 green), 0 violations / 0 rot — the katgpt-rs owns-row edit parses as valid contract; C0e 427 refs 0 unresolved; SP1–SP5b green; C6 0 LOC; C7 = 7 no-lock repos (expected shape); C8 four-rev split (@3c844aeb 1398/0 + @6a9d4d5a 37/0 + @c478ab9f 1668/1-sec esp32 + @ecce691c 483/0 — documented branch-policy class), C9 multi-rev stands owner call | — |

