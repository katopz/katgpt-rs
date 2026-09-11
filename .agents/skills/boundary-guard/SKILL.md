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

**Drift-ledger semantics** (scripts consume this): Disposition ∈ `fixable` |
`owner-call` | `by-design`. `fixable`/`owner-call` rows REQUIRE an open issue;
`by-design` rows cite a decision record instead. Issue closes → row removed in
the same commit. Exit semantics: ledger unparseable → hard error; finding mapped
to a row → exit 0 with a LOUD known-drift count; unmapped finding → exit 1;
row-without-open-issue → exit 1 (rot). Never silently fail open or closed.

## Eight surfaces

| # | Surface | Rule | Grep check |
|---|---------|------|------------|
| 1 | Consumer `src/` (riir-mmorpg-examples, seal-remake) | Thin glue only — no `impl System` with loops/math; no hardcoded constants; no duplicated helpers | `grep -rn 'fn distance_2d\|const.*FEAR' src/` |
| 2 | SDK root crate (`riir-game-sdk/src/`) | Facade only — no engine/chain/db deps in the DEFAULT build. Sanctioned opt-in exceptions (Issue 053 Part 2 + `auth_impl`/`gm` pattern; all `optional = true`, heavy ones target-gated native): `auth_impl`/`identity_impl` (riir-auth, riir-chain/ssh_key), `gm` (katgpt-core, hoisted `InferenceBackend`), `static_data_impl`/`warm_tier_impl` (riir-neuron-db, neuron-db-sdk). Scan BOTH `[dependencies]` AND `[target.'cfg(...)'.dependencies]` — the Issue 053 deps live in the target-gated section | main + target sections, filter out `optional = true` lines |
| 3 | SDK root vs workspace members | Root crate clean; members (`crates/riir-viz`, etc.) MAY depend on engine | `sed -n '/\[dependencies\]/,/^\[/p' riir-game-sdk/Cargo.toml \| grep katgpt` |
| 4 | Leaf-clean vocabulary (`riir-games-shared`) | No engine deps unless feature-gated — **at the dep level too**: dep line `optional = true` AND its feature carries `dep:<name>`. Half-gated (module cfg-gated, dep non-optional) = violation — every no-features build pays the engine tree (Issue 682: katgpt-core non-optional pulled rustfft/postcard/half into a default-`[]` crate) | `grep -E 'katgpt-core\|riir-engine' riir-ai/crates/riir-games-shared/Cargo.toml \| grep -v 'optional = true'` |
| 5 | View consumers — **live: `seal-remake` (`seal-view` Bevy/wasm + `seal-node`)**; the C# that remains in the workspace is `riir-viewbridge/csharp/` (3 files, surface 6's own side) | Rendering + input only — NO game logic (AI, combat, physics, sync). Documented deliberate debt in an OPEN issue + cross-language contract doc = record as such, don't re-file | `grep -rnE 'sigmoid\|dot_product\|impl .*System for' seal-remake/crates/*/src/ \| grep -viE 'showcase\|benchmark\|camera'` (measured 2026-09-04: **empty**) |
| 6 | FFI bridge (`riir-viewbridge` — repo PARKED 2026-09-03, Unity lane frozen; checks stay live for the unfreeze path) | Raw physical only (`pos[3]`, `rot[4]`) — NO latent state crosses FFI | `grep -rn 'emotion\|fear\|mood\|curiosity' riir-viewbridge/crates/*/src/` |
| 7 | Dev tools + KAT client plane (`riir-clippy`, `riir-kat` — the client/protocol half spun out of clippy 2026-09-10) | Zero game-domain coupling in the DEFAULT build (clippy's only public dep is `katgpt-core`; kat's deps are riir-auth `account_key`-only + the katgpt patch pin). Sanctioned opt-in arms where reimplementation would duplicate whole substrates: `ternary_inference` (riir-engine + riir-gpu), `latent_retrieval` (riir-rag) | `grep -nE 'riir-games\|riir-chain' riir-clippy/Cargo.toml riir-kat/Cargo.toml \| grep -v ':[0-9]*:#'` (dep lines only — the `-n` + comment filter matter: bare grep returns clippy's prose comments; verified empty 2026-09-11) |
| 8 | dApp layer (`riir-dapps`) | One-way **game → dapps → chain** — never a game dep here (`scripts/direction_gate.sh`); a `Settlement` stays chain vocabulary (never quest/kill/recipe — gate check 3 of direction_gate.sh); anything a game wants on-chain passes the **three-test rule** (product / value / rate) on the **agreement axis** — the chain hosts what mutually distrusting parties must agree on; neuron-db hosts authenticated durability. A paid quest still needs NO chain program (neuron-db template + the one generic `MultiClaimEscrow`) | `grep -E 'riir-games\|riir-game-sdk\|riir-engine' riir-dapps/Cargo.toml` + game-vocab scan in `scripts/direction_gate.sh` |

### The dApp three-test rule (surface 8 detail — riir-chain Issues 096/097 + riir-dapps Proposal 001)

1. **Product** — would a commerce customer of this chain want it in their dependency? (An NFT is a token → yes. A quest/recipe/kill-credit predicate → no.)
2. **Value** — BigInt fungible currency, a token, or an authority binding? Not FAME/XP/items/reputation/karma/quest progress.
3. **Rate** — quorum-coordination ops are Glacial (≤0.1 Hz); settlement transactions are **capacity-share bound** (783 req/s measured floor) — different limits, both binding.

The defining axis is **agreement** (Research 003 §"The Second Axis" "Must agree on" column) — value and rate are the disqualifying tests, not the definition. Full argument + failure-mode matrix: `riir-dapps/.proposals/001_agreement_boundary_and_tiered_durability.md`.

All greps should return **empty** (clean), modulo the sanctioned opt-in exceptions noted per-surface.

<!-- retired surfaces, kept as lineage: `riir-unity` (C# Unity host) and
`seal-remake-unity` were moved to `git/obsolete/` by owner act on 2026-09-04,
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
Rows carried full narratives until these compactions; `git log -p -- .agents/skills/boundary-guard/SKILL.md`
(the katgpt-rs repo) recovers any of them verbatim. New rows append ONE line each.

### Standing lessons (distilled from the compacted rows — load-bearing process rules)

- **Read the exit UNPIPED.** `cmd | tail; echo $?` reads TAIL's exit (re-proven twice); outputs go to /tmp, read the real exit or PIPESTATUS. A piped gate summary cannot vouch for an exit code — and a green summary over an unstaged body is worse: the 20th run's "exit 0" was FALSE while C0e's implementation sat unstaged (`a6ca9ccbd` docs-only; body landed `10977dd23`).
- **Verify the script not-mid-edit before running**: mtime vs the last recorded run + `bash -n` + clean in `git status`. A sibling rewriting the script mid-run truncated a measurement after SP1 while the first attempt still reported exit 0 (the 2026-09-04 membership row) — bash re-reads by byte offset. The script anchors ROOT from its own path, so /tmp snapshot-copy immunization CANNOT work; diff-vs-snapshot before running the live script instead.
- **Parser replay for stale shared checkouts**: run the script's own allowlist awk against `git show origin/<default>:BOUNDARY.md` — a sibling branch predating a row reads exit-1 on disk while main is CLEAN (the riir-train until-merge artifact class); touching the sibling branch is forbidden.
- **Detection-only discipline**: file `.issues/NNN_boundary_*.md` BEFORE any fix; sibling WIP (staged sets, mid-flight sweeps, dirty manifests) is measured as-found and never fixed by the idle unit — the sweep owner re-pins C6 at closeout (the `2575b6519` precedent). A landed growth WITHOUT its re-pin, though, is repaired by whoever finds it once the landing session is gone (the honest up-pin class: verify zero training semantics, pin UP with the reason).
- **The C6 ledger row text must be apostrophe-free** and the real detector is `bash -n`, not running the guard — bash 3.2's quote tracker reports the wrong line (~20 below the guilty one).
- **Expected shapes, not findings**: the 2 known-drift rows (the ledgered seal-online-remaster pair); C7 = 4 repos with no tracked lockfile; scoped `--repo` runs print C7/C8 out-of-scope notes only.
- **C8/C9 standing advisory (owner call, never an idle-unit act)**: consumer lockfiles pin katgpt-rs `@6f392727` and drift further behind develop each window — 326 (09-03) → 739 commits behind with 7 security fixes in the gap (09-11, the 70th run) across 3 lockfiles, plus a second divergent pin in riir-chain (167/1). Recorded every run; the bump is the owner's. **Gap ENUMERATED 09-10 with the script's own predicate (689 behind at the 37th run, `@c478ab9f` chain pin 0 security): the 6 = 2× SIMD soundness (`99afbab9` safe-code OOB read + heap-corrupting write in `katgpt-types/src/simd/` — CWE-125, `simd_dot_f32` unchecked-`len`; `5b028c00` the near-verbatim katgpt-dec twin) + 2× FFI/panic-UB hardening (`af2e48e1` catch_unwind on 10 wasmi extern fns; `e854958e`/`96543f4f` the 208-site NaN-comparator sec heal) + `2dec22a1` (a filed, not-fixed, EngramHotSwap nested-swap soundness hazard) — the sharp half is that the two SIMD holes are MEMORY-SAFETY bugs in the exact substrate the three deploy lockfiles compile today, so the bump is a soundness call, not a freshness preference. **RESOLVED 2026-09-11 (owner call executed, subagent re-verdicted): all three consumer locks bumped `6f392727 → c478ab9f`, and the discovery corrected the advisory's own premises.** The 3 tracked lockfiles are riir-dapps `cloudflare/kat-service`, riir-mmorpg-examples `cloudflare/warm-tier-do`, and riir-esp32 `crates/riir-satellite-probe` — NOT riir-kat/riir-dao, which carry no tracked lock at all (the C7 class). The consumers pin `branch = "main"`, so the reachable target was main tip `c478ab9f`, which already contains ALL 6 enumerated fixes; the 7th grep hit in the develop-only range is a docs commit (the advisory's own enumeration) — a **predicate false-positive class**: the sec-subject grep matches docs commits whose subject contains "soundness". riir-chain's "divergent pin" re-verdicted to the same answer: it already sits at main tip `c478ab9f`, zero action needed. All builds/tests green per consumer (kat-service 143 tests + wasm32; warm-tier-do 4 + wasm32; satellite-probe esp32c6 release). Commits: dapps `c9203a2`+`6342519`, mmorpg `0afb52b`, esp32 `2bfb795`. Going forward the "behind develop" count is a branch-policy artifact (`branch=main`), not a security gap — the advisory's remaining value is the C9 one-rev agreement, which this bump achieved (all 4 build roots on `c478ab9f`).**
- **Guard output truncates in `tail` views** — capture the FULL violation list when filing (the "8 = 6 unique" enumeration error).
- **Never suppress fetch output you act on** — a silenced fetch failed and produced a false empty-divergence read while the push error was right (the 59th run, 4090 box).
- **`git show HEAD:<dir>` on a directory prints the tree LISTING** — honest dir LOC is an `ls-tree -r` + `cat-file` sum. `rustfmt --emit stdout` prints a 2-line filename header — a round-trip harness must strip it.
- **The staged-set check must be a SEPARATE command** (or commit `git commit -- <paths>`): chained `add && diff --cached && commit` swept a sibling's entire staged set once (the `30127c4` incident); recovery anatomy lives in that row's history — read the reflog before any second corrective action.
- **A prose contract cannot catch an inverted measurement** — every "repo X may depend on Y" row is a measurement with a direction; that is why the contract script parses tables instead of prose (the first full run's founding lesson).

### Run table

| Date | Run | Verdict | Record |
|---|---|---|---|
| 2026-09-12 | 74th — full (post-census-19 fix in kat `7f58578e`; script verified not-mid-edit: Sep-7 mtime + `bash -n` + clean status; exit read unpiped to /tmp; 4 sibling lanes live: kat 393-regen · riir-shader vgpu · sdk shop-chaos · ai shared-context) | **exit 2 — 3 sibling-lane findings, detection-only by discipline**: HARD ERROR riir-shader/BOUNDARY.md missing `## Owns`/`## Does not own`/`## May depend on` (vgpu scaffold `464e95a` gap — the riir-kat 67th-run class, enroll-without-contract-shape) + CONTRACT ROT riir-ai CANONICAL matrix has no riir-shader row (the 68th-run fix pattern, producer's obligation) + 2 VIOLATIONS seal-remake → neuron-db-sdk + riir-neuron-db dev edges undeclared (landed `265dec63` shop T3.2 front, Plan 192 lane ACTIVE — and seal-remake's own riir-scenelab row says "seal-remake never deps ndb/rag directly", so the reconcile is the shop lane's design call, not a mechanical row-add); 2 known-drift expected pair (seal-online-remaster absent); C8 220-behind/1-security (branch-policy artifact post-bump, advisory stands); census fix validated: banner now derives **19 repos** | — |
| 2026-09-11 | 73rd — full + S1 (fresh idle window, zero-cargo posture under sibling seq-job load; script verified not-mid-edit: Sep-7 mtime + `bash -n` + clean status; exits read unpiped to /tmp) | **exit 0 — 18 repos / 273 edges unchanged**; 2 known-drift expected pair; S1 mmorpg all five clean; S1-analog seal-remake + S7 empty; S2/S4/S6/S8 hits = comments + sanctioned feature-gated shapes; direction_gate exit 0; C8 760-behind/7-security + chain pin 188/1 (both worsened) — soundness-bump advisory stands, owner call | — |
| 2026-09-11 | 72nd — mmorpg per-repo S1 (post-`943dce3` consumer landing; script verified not-mid-edit: Sep-8 mtime + `bash -n` + tree clean; exit read from /tmp) | **exit 0** — all five checks clean (System impls thin, no geometry dupes, no hardcoded constants, no generic logic, no facade leaks beyond the 053 exemptions); contract half fresh via the 71st (same day, exit 0, 18/273) | — |
| 2026-09-11 | 71st — full (the 088 follow-up adoption landing: dapps + dao consume the riir-kat protocol tier) | exit 0, 0 violations, 0 rot — **18 repos / 273 edges** (the 70th's count, now with both new edges DECLARED in BOUNDARY.md same-commit: dapps `kat_ledger`-gated protocol-only row, dao units row); 2 known-drift expected pair; C8 unchanged | kat `3482a3c` · dapps `f2498d2` · dao `7823daa` |
| 2026-09-11 | 70th — full (idle unit; script verified not-mid-edit; exit read unpiped to /tmp) | first pass read 1 transient violation (riir-train wsdep token) that did NOT reproduce on re-run — measured mid-landing state, the as-found class; current truth **exit 0, 0 violations, 0 rot — 18 repos / 273 edges**; 2 known-drift expected pair; C8 739-behind/7-security + chain pin 167/1 — soundness-bump advisory stands, owner call | — |
| 2026-09-11 | 69th — mmorpg per-repo S1 | exit 0, all five checks clean (the 053 exemption set holding) | — |
| 2026-09-10 | 68th — full (kat lane closeout) | exit 1 → the 67th's findings all closed in-run: riir-kat declares the 9 katgpt-* patch edges (`a80985e`) + riir-ai CANONICAL row for riir-kat (`18c878a32`) → re-run **exit 0, 18 repos / 271 edges**; same window: repo_set.txt + AGENTS count pins 17→18 | riir-kat `a80985e` · ai `18c878a32` |
| 2026-09-10 | 67th — full + S1 | exit 1 → 10 violations + 1 rot, ALL one root cause: riir-kat landed (`90a990e`) with 9 undeclared patch edges + clippy edge + missing CANONICAL row; lane MID-FLIGHT → detection-only; mmorpg S1 ✓ exit 0 | — |
| 2026-09-10 | 66th — full + S1 | exit 0, 17 repos / 260 edges unchanged (docs-only landings); C8 704/7 + 132/1 | ai `1b2e86a60` |
| 2026-09-10 | 65th — full + S1 | exit 0, 17 / 260 unchanged; C8 701/7 + 129/1 | clippy `2718e7a3` |
| 2026-09-10 | 64th — full + S1 | exit 0, 17 / 260 unchanged; C8 699/7 + 127/1 | dapps `c54b0f8` |
| 2026-09-10 | 63rd — full | exit 1 → 1 violation: game-sdk→chaind landed (`7f6114d`) with the edge prose-sanctioned but the crate token missing from column 1 → one-token amendment → re-run **exit 0, 17 / 260** | game-sdk `11b9c95` |
| 2026-09-10 | 60th–62nd — full ×3 (the 029 fix landing: seal-online-remaster's first BOUNDARY.md + this repo's CANONICAL row) | 60th exit 1 → the NEW canonical row itself was rot twice — the C4 parser admits any `riir-*`/`katgpt-*` word in row prose as a dep claim → reworded to plain-English repo descriptions → **62nd exit 0, 14 repos / 263 edges** (4090 box, population 12→13 after cloning seal-game-editor) | seal `b832fd2` · ai `63ab6d963` |
| 2026-09-10 | 59th — full (4090 box) | exit 1 → 4 rot: 3× missing seal-game-editor + C0c seal-online-remaster no-contract → cloned the missing sibling (27th-run precedent, 3 cleared) + filed seal `.issues/029` → re-run 0 violations / 2 mapped rot; lesson: a fetch with suppressed output failed silently → false empty-divergence read — never suppress fetch output you act on | seal-online-remaster `6979784` |
| 2026-09-10 | 39th — full (user-ordered hygiene unit; repos FF'd first — both were behind the commits that make the run honest) | exit 1 → 1 violation (game-sdk→chaind, the sibling's uncommitted riir-e2e WIP — as-found) + 1 rot (my behind-2 stale checkout, fixed by FF) → re-run 1 violation / 0 rot, 17 repos | — |
| 2026-09-10 | 38th — full | exit 0, 17 / 258 unchanged (m32-pipe landing dep-free); C8 691/7 + 119/1 | — |
| 2026-09-10 | 37th — full (zero-cargo form) | CLEAN 17/258; C8 688/5 + 116/0; standing advisory unchanged, owner call | — |
| 2026-09-09 | 58th — full (fresh idle window; script verified not-mid-edit: Sep-7 mtime + bash -n + clean status; box quiet except sibling UI lanes) | **exit 0, 17 repos / 258 edges** — unchanged from the 57th; 2 known-drift expected pair; C8 677-behind/6-security + 105-behind chain pin recorded (worsened); doc-sync same window CLEAN — zero new landings workspace-wide since the 57th (only pushes of previously-measured commits + sibling WIP dirt, incl. ndb `421008c` now pushed, dapps ahead-1 sibling lane) | — |
| 2026-09-09 | 57th — full (fresh idle window; script verified not-mid-edit: Sep-7 mtime + bash -n + clean status; sibling bevy compile lane busy on box but no manifest work) | **exit 0, 17 repos / 258 edges** — unchanged from the 56th; 2 known-drift expected pair; C8 675-behind/6-security + 103-behind chain pin recorded (worsened); doc-sync same window CLEAN (only landing since baseline `3115f7f4` self-documenting); hygiene: alloc-gates 741 resolved filing removed + HISTORY row added (`8f4697eb`) | kat `8f4697eb` |
| 2026-09-09 | 56th — full (fresh idle window; script verified not-mid-edit: Sep-7 mtime + bash -n + clean status; 1st exit-1) | exit 1 → 1 unmapped C3: seal-remake → editor-widgets NOT in 'May depend on' — the row LANDED at `101b7f58` with column 1 BARE, and the allowlist parser admits a non-family dep only in backtick form (its own documented convention; family-prefix rows never hit this) → one-token contract fix `` `editor-widgets` ``, re-run **exit 0, 17 repos / 258 edges** (+2 = the scenario-dag-panel edges now recognized); 2 known-drift expected pair; C8 672-behind/6-security + 100-behind chain pin recorded; lesson: a landed allowlist row the parser cannot SEE is a violation, not a declaration — replay the row through the parser's own awk before assuming the contract is honest | seal-remake `ab4cbd9d` |
| 2026-09-09 | 54th — full (fresh idle window; script verified not-mid-edit; migration agent's adopt-visibility change NOT in the tree — riir-auth clean at the doc-sync `bcf543c` landing) | **exit 0, 17 repos / 256 edges** — unchanged; 2 known-drift expected pair; C8 655-behind/6-security + 83-behind chain pin recorded | — |
| 2026-09-09 | 55th — mmorpg-examples per-repo code-logic half (zero-build quiet-gate window; script `bash -n` clean, repo clean at `HEAD`) | **exit 0** — all five checks clean (System impls thin, no geometry dupes, no hardcoded constants, no generic logic, no facade leaks outside the 053 exemption); contract gate re-verified clean this session (17 repos / 256 edges) | — |
| 2026-09-09 | 53rd — full (fresh idle window; script verified not-mid-edit: clean + syntax + Sep-7 mtime) | **exit 0, 17 repos / 256 edges** — unchanged from the 52nd (the kat_account agent's declared riir-clippy→riir-auth edge still uncommitted WIP, measured as-found; their lane now in G9/bin verification); 2 known-drift expected pair; C6 green; C8 654-behind/6-security + 82-behind chain pin (still worsening) recorded | — |
| 2026-09-09 | 52nd — full (fresh idle session; script verified not-mid-edit: clean + syntax + Sep-7 mtime) | **exit 0, 17 repos / 256 edges** (+1 = the kat_account agent's IN-FLIGHT uncommitted riir-clippy→riir-auth optional dep, declared, measured as-found — their lane); 2 known-drift expected pair; C8 652-behind/6-security (worsened from 648) recorded | — |
| 2026-09-09 | 51st — full (trigger: the ndb CLI agent's landing cleared the 50th's violation — ndb `7dda44e` CLI crate + BOUNDARY row (riir-auth scoped to `crates/neuron-db-cli` ONLY, root-crate leaf protection), auth `c2274ca` account_key substrate, ndb `25ee879` Plan 327 COMPLETE; script verified not-mid-edit) | **exit 0, 17 repos / 255 edges** (+1 = the declared ndb→auth CLI edge); 2 known-drift expected pair; C6 green; C8 648-behind/6-security recorded | — |
| 2026-09-09 | 50th — full (idle-loop unit after the 327 T2/T3 landings; script verified not-mid-edit: clean + syntax + Sep-7 mtime) | exit 1 → **1 violation: riir-neuron-db → riir-auth** via the sibling CLI agent's UNTRACKED `crates/neuron-db-cli/Cargo.toml` (undeclared allowlist edge — their in-flight lane, measured as-found, not fixed by this run); 2 known-drift expected pair; C8 645-behind/6-security (worsened from 632) recorded | — |
| 2026-09-09 | 49th — scoped seal-remake contract + S5 (trigger: Issue 015 inventory screen landed in seal-view — new view-layer module, zero new deps) | contract exit 0, 19 edges unchanged; S5 empty on inventory.rs/equip_view.rs (the screen renders sim state + writes the existing HeroWeapon desired state — no game logic); closure-build validation recorded in seal HISTORY | — |
| 2026-09-09 | 48th — S1–S8 grep sweep only, zero-cargo posture (851 v8 quiet-gate live; contract half = the 47th today, exit 0) | all 8 surfaces CLEAN — S2/S3/S7/S8 surviving lines are comments/sanctioned opt-in features; S4 optional+dep: shape holds (L217/L222); S6 hits are the contract doc comments themselves; direction_gate OK exit 0; lesson re-hit: first attempt ran sibling-relative paths from the katgpt-rs anchor with 2>/dev/null — false empties, re-run with absolute paths before any verdict | — |
| 2026-09-09 | 47th — full (trigger: landings since the 46th incl. the 12-commit sweep-hygiene fan-out; script verified not-mid-edit) | exit 1 → 8 UNMAPPED riir-auth → katgpt-* edges (the 5856d61 `[patch]` unification landed 02:39, 41 min AFTER the 46th, skipping its allowlist) → declared in riir-auth BOUNDARY.md (issue 003 row-lands-and-closes) → re-run **exit 0, 17 repos / 245 edges** (+9 = 8 patch + satellite-probe-era edges), 2 known-drift expected | auth `b71f213` |
| 2026-09-09 | 46th — full + S1 + S5 (trigger: Plan 575 landings — ai `1099c9e65`/`8b4c4f80e`/`1a5dfdff4` + seal `f6460477` attack_reasoning forward; box busy with a sibling pipeline, contract script verified not-mid-edit first) | exit 0, 17 repos / 236 edges unchanged; 2 known-drift (expected pair); S1 clean; S5 empty; C8 632-behind/6-security recorded | — |
| 2026-09-08 | 45th — full-workspace contract (quiet-box window; trigger: landings since the 44th) | exit 0, 17 repos / 236 edges unchanged; 2 known-drift (expected pair) | — (this compaction commit logs it) |
| 2026-09-08 | 44th — scoped seal-remake + riir-ai + S5 (box saturated by the p335 Bonsai arm) | riir-ai exit 1 = C6 lora ratchet 3082→3088 (888-T2 landing skipped its re-pin; +6 = docs + ctor, zero training semantics) → re-pinned, re-run exit 0 / 80 edges; seal-remake CLEAN 19 edges; S5 empty on the post-014 view code | riir-ai `644f3012a` · kat `2c57a4d0` |
| 2026-09-08 | 43rd — full + S1 (trigger: sibling docs landings) | CLEAN 17/236 unchanged; C8 620-behind recorded; hygiene: ai 852+891 resolved files removed (15→13 index) | — |
| 2026-09-08 | 42nd ×2 — full + S1 (±S2–S8) (trigger: the 894 lane + kat hygiene batch `bad86f9d`) | CLEAN 17/236 unchanged both; C8 617/6-security; doc-sync: viewbridge 006/007 log entries added | viewbridge `2243dee` |
| 2026-09-07 | 41st — full + S1 + S4–S8 (zero-compute posture, 851 autofire live) | CLEAN 17/236; C8 595/6 worsened; 6 resolved-but-present files found in sibling repos — left to owners | — |
| 2026-09-07 | 40th — scoped clippy + S7 | CLEAN 10 edges; run-count chain reconciled (37 uncommitted → 38/39 snapshot-only → 40) | — |
| 2026-09-07 | 37th — scoped clippy + full + S1 (trigger: 883 completion + Batch 125 + release lane) | CLEAN 17/236 (+1 consumer-seam edge, declared) | — |
| 2026-09-07 | 36th — full + S1 (both mid-flight lanes LANDED) | CLEAN 17/235; esp32 enrollment confirmed by measurement; C8 567/5 escalation recorded | — |
| 2026-09-06 | 35th — full (trigger: the ffi lane LANDED) | exit 1 → 2 C4 rots (chain row prose naming `riir-ai`/`riir-ffi` post-dissolution) → C4 exemption list extended, canaried BOTH directions → exit 0, 16/234 | riir-ai `0af26c2b5` |
| 2026-09-06 | 34th — full + scoped ai | exit 1 = C4 rot inside the ffi lane's own UNCOMMITTED BOUNDARY.md WIP (HEAD replay clean); mechanism + fix options pinned for their lane | — (their `0af26c2b5` landed it) |
| 2026-09-06 | 33rd — full + S1 | CLEAN 16/236; uncommitted `think_brain_wasm` WIP measured as by-design note | — |
| 2026-09-06 | 32nd / 31st — full (×2 windows) | CLEAN 16/236 unchanged both; C8 516→523 | — |
| 2026-09-06 | 30th / 29th / 28th — scoped kat / clippy / full+ S1 (Plan-005 closeout) | all CLEAN; Plan-005 boundary rows verified landed; 27th-run fresh-clone note discharged (scenelab + panel pushed) | — |
| 2026-09-05 | 27th — first 4090-box run; 3 scoped (auth/mmorpg/clippy) | CLEAN 3/29/10 edges; cloned riir-auth + mmorpg on that box to end the divergence | — |
| 2026-09-04 | 26th — full + S1 | C6 worktree-only artifact (sibling sweep mid-flight); S1 4 violations → mmorpg Issue 095 → FIXED same session (SDK `warm_tier_impl` + merkle_freeze re-export) | game-sdk `11b97d5` · mmorpg `339311e` · `90a67b7` |
| 2026-09-04 | 25th / 24th / 23rd — scoped seal + full ×2 (092 landmine-6 ndb-sdk row PRE-declared) | CLEAN/scoped; C6 worktree artifact proven formatting-only via rustfmt(HEAD) round-trip; sweep owner re-pins at closeout | — |
| 2026-09-04 | 21st + parked-item run — retirement sweep + two 20th-run debts corrected | canonical none-row de-named retired repos; C0e body landed; tail-pipe + unstaged-body lessons recorded | kat `10977dd23` · seal `0cd11f0` · mmorpg `1e83e00` |
| 2026-09-04 | C0e landing (Issue 862) + membership change (19→16) | C0e dangling-route check landed, canaried both directions; PARTIAL run during a sibling's mid-edit = truncated-but-exit-0 hazard recorded; repo_set/AGENTS/S5/doc-sync routing updated | kat `10977dd23` · viewbridge `b406025`/`3c6bafa` |
| 2026-09-04 | early — scoped riir-train post-00:22-fix | CLEAN 23 edges; ROOT-anchored script → snapshot-copy immunization impossible (lesson above) | — |
| 2026-09-03 | 19th — full + S1 | CLEAN 19/225; C8 advisory first quantified (326 behind / 3 named security fixes) | — |
| 2026-09-02 | 18th / 17th — full ×2 (Plan-031 split; post-844) | 18th: CLEAN 19/225, seal-remake-unity enrolled 0 edges, armageddon de-enroll confirmed; 17th: 7 C7 on armageddon → fixed, owner verdict OBSOLETE | armageddon `c2e50fa` |
| 2026-09-01 | 16th / 15th — 3-scoped spot-check / full + S1 + S2–S8 | 16th CLEAN; 15th: C6 RED → repaired (bench_672 +132 honest up-pin 218→350 + R3 shrinks), bash-3.2 apostrophe lesson | riir-ai `2575b6519` |
| 2026-08-29 | 14th — full + S1 | CLEAN 17/202; 4090 yield note (478-T3 contention class) | — |
| 2026-08-28 | 13th / 12th — S1 / games-dep audit | 13th: 3 violations → Issue 086 → resolved same day (dapps re-export); 12th: SP5 landed, zero unconditional games deps workspace-wide | dapps `6671eb5` · mmorpg `627800e` · riir-ai `8da5e187e` |
| 2026-08-27 | 11th / 10th / 9th / 8th / 7th — scoped per-repo deltas | train until-merge artifact via parser replay (×2); ai D4 rot re-anchored to Proposal 041 + 762 collision yielded; dapps D1 rot fixed; gm_server edge promotion verified rowed | riir-ai `12284bf6c` · dapps `f704a4d` |
| 2026-08-26 | 6th / 5th / 4th / 3rd + 2 later runs | kat 0-edges upstream check; train parser-replay ×2; S1 4 violations → Issue 085 → resolved same day (SDK hint_regret facade; first green S1 since 08-22); katgpt-device-verify patch rows → allowlist rows after Issue-759 arbitration closed | game-sdk `0a65703` · mmorpg `baab368`/`0aba62b` · ai `538a7aad0` · sdk `7f3b6aa` · mmorpg `91b33e5` · train `9100cfe4` |
| 2026-08-24 | scoped riir-train | CLEAN 21 edges | — |
| 2026-08-22 | full + S1 | CONTRACT CLEAN 15/177; 4 violations → Issue 081 → resolved (SDK art_vessel_impl facade; vessel_impl is a SIBLING not a superset) | game-sdk `1a6fe62` · mmorpg `71d3c10` |
| 2026-08-21 | routine + FIRST FULL CONTRACT RUN (Issue 737 T-RUN) | 1 real leak (chain root riir-wasm missing `default-features = false`) + 5 contract bugs fixed; C0 derives membership from OUTSIDE the set — 15 repos not 11; 173 edges | riir-chain `f19c6ee5` |
| 2026-08-19 | S1 | 1 violation → mmorpg Issue 075 (remote_smoothing dead-reckoning; substrate-vs-exempt = owner call) | mmorpg `4961be2` |
| 2026-08-17 | 6 runs — S1 detection/fix arc + guard-semantics change | Issues 061/069 arc: 8 facade leaks detected → SDK karma_history re-export + consumer swap; Check D cfg(test)-mod awareness (221 test fns exempted, 0 prod lost); pet_ai position_toward tasked | game-sdk `94f3198` · mmorpg `70a5aac`/`ca50093`/`f3ac749` |
| 2026-08-15 | quarterly gate + 2 idle runs | 6/7 surfaces clean; Issue 682 half-gated dep fixed (no-features check 15.45s→5.76s); S2/S4/S7 grep notes encoded the sanctioned exceptions | riir-ai `e982ef7d2` · `20abe832d` |
