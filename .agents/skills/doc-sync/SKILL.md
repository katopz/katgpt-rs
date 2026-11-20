---
name: doc-sync
description: Synchronize each repo's `.docs/` and `README.md` with recently succeeded plans/issues/benchmarks by diffing git history against the last documented entry. Use after landing a GOAT-passing plan, closing a batch of issues, or quarterly as a doc-hygiene gate. Covers every workspace repo and knows each repo's doc layout + where to record what, including each repo's root BOUNDARY.md contract (drift rows vs issue state).
---

# doc-sync — Keep `.docs/` + `README.md` in sync with landed work

This skill brings a repo's documentation up to date with the work that has
**landed in git but not yet been written up**. It is the doc equivalent of a
`cargo doc` rebuild: the code shipped, now make the narrative match.

## When to use

- After a plan closes with a GOAT/gain verdict (promote, keep-opt-in, or honest fail).
- After a batch of issues resolves (especially negative-result issues that move a
  primitive's status line).
- After a feature is promoted to default-on OR demoted to opt-in.
- Quarterly as a doc-hygiene gate.
- **Any numbered-file work session (issue 1205 fix path 3):** run that repo's numbering gate before leaving — `py ../katgpt-rs/scripts/numbering_gate.py <repo>` (exit 1 = an un-bumped `.highwater`, a collision, or a stale/malformed allocator; fix in the same commit). riir-ai's guard layer 1.39 carries the same check every guard run; this bullet is the standing lane for repos without that layer.
- **NOT** for speculative work — only landed, committed work counts.

## `BOUNDARY.md` — the 4th doc surface (added 2026-08-21)

Every repo ships a root `BOUNDARY.md`: Owns / Does not own / May depend on
(crate-granular allowlist) / Inherited (links) / **Drift ledger**. It is a
doc-sync surface because its drift ledger is a *claim about issue state*, and
claims rot:

- **Row ⟺ open issue.** A `fixable` / `owner-call` row REQUIRES an existing
  issue file; a `by-design` row cites a decision record instead. When an issue
  closes, its row must be removed **in the same commit** — the noise-reduction
  rule extended to BOUNDARY.md. A row whose issue is gone is the boundary
  equivalent of a stale README claim.
- **Flag row-without-issue** and **issue-without-row** (a boundary issue that
  landed with no ledger row is invisible to the guard).
- **Don't hand-verify the dep tables** — run
  `riir-ai/scripts/ci_boundary_contract.sh`. It fails on an undeclared
  cross-repo dep, a stale allowlist row, an unparseable ledger, and on the 4
  split-prep invariants. `--list-deps` prints the measured graph.
- **Numbers in a contract are measurements**, so they carry a date. If a row
  cites "N symbols" or "N packages", re-measure before trusting it in a new
  decision (`riir-ai/BOUNDARY.md` D2/D3 are the pattern).
- The contract is per-repo; cross-repo rules live in ONE canonical home
  (chain admission → `riir-chain`, dep matrix + split-prep → `riir-ai`) and
  every other repo LINKS. Never copy a cross-repo rule into a second file —
  that is the duplication doc-sync exists to catch.

## The workspace repos and their doc shapes (**derived, never counted** — de-counted 2026-09-04)

Hand-typed counts went wrong twice within days and missing rows were skipped silently — **never
re-type the count or trust the census below as current**: it is a dated snapshot (provenance notes
and shape narratives live in this file's git history — `git log -p -- <this file>`); derive the
membership:

```bash
cd /Users/katopz/git && for d in */; do
  [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
done
```

Canonical count + product-vs-workspace split: `katgpt-rs/AGENTS.md` §"Repo count". Shape classes:
numbered folders (count) · flat numbered FILES · NAMED subfolders · no `.docs/` (AGENTS-centric) ·
empty/README-only `.docs/`. "same" = the four `.highwater` files; worktrees (`.git` FILE) never
enter the census; a repo absent from the running box is UNVERIFIABLE, not corrected.

| Repo | `.docs/` shape | README | Highwater | Branch |
|---|---|---|---|---|
| `katgpt-rs` | 10 numbered folders (01_orientation…10_audits), unnumbered files inside — the public book | large showcase + feature tables | same | develop |
| `riir-ai` | 12 numbered folders (01…12; 02_inference renumbered 12_inference) — the private book | large showcase + crate table | same | develop |
| `riir-chain` | 7 numbered folders (01…07_formal_verification; canonical chain self-description — riir-ai's `.docs/07_neuro_symbolic_chain/` is the fusion view, links here; 2 members: lib + chaind) | build + flags + wallet/RPC trust + merkle lesson | same | develop |
| `riir-neuron-db` | 11 numbered folders (01…11_cli) | feature gates + FV summary + license | same | develop |
| `riir-train` | 6 numbered folders (01…06_cross_cutting) | minimal: role + sibling layout | same | develop (default; `main` frozen) |
| `riir-game-sdk` | 10 numbered folders (01…10_multiplayer_topology) | boundary + leaf constraint + spatial canonical + phases + feature gates | same | develop |
| `riir-mmorpg-examples` | no `.docs/` — AGENTS.md-centric (+ README + .plans/.issues/.benchmarks) | yes | same | develop |
| `riir-refine` | 12 numbered folders (01…11_domains + 12_ane) | status + quick start + usage + feature gates | same | develop |
| `riir-unity` | RETIRED 2026-09-04 → `git/obsolete/`; lineage only | yes | same | develop |
| `riir-viewbridge` | 1 numbered folder (01_orientation) + AGENTS.md-centric remainder | yes | same | develop |
| `mmorpg-remake` | 1 numbered folder (01_orientation) | yes | same | develop |
| `riir-dapps` | 1 numbered folder (`.docs/11_kat_service/` — numbered to MIRROR the kat-service group, not a tenth sibling) + AGENTS.md remainder | yes | same | develop |
| `riir-dao` | flat numbered FILES (001_kat_account_key_authority_census) + AGENTS.md remainder | yes | same | develop |
| `riir-armageddon` | RETIRED 2026-09-02 → `git/obsolete/`; lineage only | yes | same | main (check before branching) |
| `riir-auth` | `.docs/` holds only `.highwater` — AGENTS.md-centric | yes | same | develop |
| `riir-burner` | RETIRED 2026-09-04 → `git/obsolete/`; lineage only | yes | same | develop |
| `riir-deployer` | 2 numbered folders (01_orientation, 02_runbooks) + `.docs/README.md` index | yes | same | develop |
| `katgpt-web` | no `.docs/` — AGENTS.md-centric | yes | none | main (trunk) |
| `mmorpg-editor` | NAMED `.docs/` subfolders (new-game-schema/, registry/) + ARCHITECTURE.md (internal layering) + DESIGN.md; BOUNDARY.md covers only the outer edge | yes | same | develop |
| `riir-esp32` | no `.docs/` — AGENTS.md/BOUNDARY.md-centric | yes | `.issues` 110, `.proposals` 006 | develop |
| `riir-kat` | no `.docs/`, no README — AGENTS.md/BOUNDARY.md/HISTORY.md-centric | none | same | develop |
| `riir-shader` | 4 numbered folders (01_orientation…04_size_budget) | yes | same | develop |
| `mmorpg-remaster` | flat numbered FILES (00_principal…10_quest_system) — RETIRED 2026-11-12 → `git/obsolete/`; READ-ONLY lineage | yes | same | develop |
| `seal-remake` | 6 numbered folders (01…06_tooling) + top-level `.docs/README.md` | yes | same | develop |
| `riir-reflex` | 5 numbered folders (01…05_resources) + top README | yes | same | develop |
| `riir-reflexer` | 6 numbered folders (01…06_resources) | yes | same | develop |
| `riir-instinct` | 3 numbered folders (01_serving, 03_decision_flow, 05_resources — the number gaps are the seed's own) | yes | same | develop |
| `riir-rethink` | numbered folders with gaps (03/05/06/07/08/09_human_scale_ai — the seed's own) | yes | same | develop |
| `riir-refab` | 1 numbered folder (001_orientation) + README | yes | same | develop |
| `riir-infer` | flat numbered FILES (001…004) + README | yes | same | develop |
| `seal-game-editor` | 4 numbered folders (01_assets…04_signoffs) + top README | yes | same | develop (tickets per the git-workflow skill) |
| `seal-online-remaster` | flat numbered FILES — DEPRECATED 2026-09-16: migration source only, read-only upstream | yes | same | develop |
| `sealm-toolkit` | `.docs/` EMPTY (the riir-auth class) | yes | same | develop |
| `mirror-twin-lint` | 4 numbered folders (the seal-game-editor shape) | yes | same | develop |
| `jup-tool` | no `.docs/` — RETIRED 2026-11-02 → `git/obsolete/` | yes | same | develop |
| `reflex-site` | no `.docs/` | yes | same | develop |
| `riir-llm` | no `.docs/` | yes | same | develop |
| `seal-std` | flat numbered FILES (001_capped_supply_vocabulary) | yes | same | develop |
| `seal-render` | no `.docs/` | yes | same | develop |
| `riir-ledger` | `.docs/` holds only `README.md` (no numbered files yet) | yes | same | develop |
| `zed` | flat numbered FILES (001–018, 033–038 — the vendored-editor lane records; number gaps are its own) | yes | same | develop |
| `riir-grams` | no `.docs/` — AGENTS.md/BOUNDARY.md-centric | yes | same | develop |
| `seal-sprints` | no `.docs/` — TASKS/DONE/HISTORY tracker, AGENTS.md-centric | yes | same | develop |

## The sync workflow (per repo)

### Step 0 — Enumerate the code surface (owner mandate, 2026-10-24 — do this FIRST, every run)

Baseline-diffing (Step 1) covers TEXT, never SURFACE: a book that never grew passes every
baseline range green while the code underneath tripled. seal-remake held a 2-doc book over an
8-crate / ~550-file workspace for weeks — every run green, nothing doc-shaped ever committed.
Before diffing anything:

1. **Derive the surface** — workspace `members` + `[[bin]]` rows from the root manifests, and
   the crate file counts (`find crates/*/src -name '*.rs' | wc -l`). Never trust a count typed
   in any doc (including this one).
2. **Ask what no doc row covers.** A crate/bin with zero book coverage is a GAP even when every
   existing doc is accurate — the book UNDER the code is the same rot as a stale claim, just quieter.
3. **Fill gaps from code facts**: workspace members → a crates map (role from each crate's own
   lib doc); a big uncovered crate → its own numbered folder (family format: bare slugs, a README
   index per folder, a top-level `.docs/README.md` entry point). Structural maps only — module
   roles from `//!` headers, features from Cargo.toml, bins from `[[bin]]` + bin headers. Anything
   not derivable this session is "—" or deleted, never guessed. NEVER restate the repo's
   AGENTS.md lane prose — that duplication is the drift this skill exists to catch; link it.
4. **Then run the Shape-change contract below** if the fill added top-level folders — the census
   row update is part of the fill, not an afterthought.

### Step 1 — Find the last documented commit

```sh
git --no-pager log --oneline <branch> -- ".docs/**" "README.md" | head -20
```

The most recent `docs:` commit is your baseline. Everything after it is **undocumented work**.

### Step 2 — List landed-but-undocumented work

```sh
git --no-pager log --oneline <baseline>..<branch>
```

Filter for:
- `feat:` / `fix:` commits that close a plan or issue (grep the message for `Plan NNN` / `Issue NNN`).
- `docs:` commits that close research notes or benchmarks (these may already be half-documented).
- Promotions / demotions (search for `promote`, `demote`, `default-on`, `opt-in`).

Cross-reference against the repo's `.plans/`, `.issues/`, `.benchmarks/`,
`.research/` folders — read the highwater files to know the current max number.

### Step 3 — Classify each landed item

For each undocumented plan/issue, classify it:

| Verdict | What to write |
|---|---|
| **GOAT PASS + promoted to default-on** | Add to the default-features list in README. Add/update the feature table row in `.docs/01_orientation/overview.md` (or equivalent). Mark the plan's TL;DR with the promotion date. |
| **GOAT PASS + stays opt-in** | Add to the opt-in features table in README. Update the `.docs/` feature catalog. Honest about why it stays opt-in (heavy, fusion-pending, diagnostic-only). |
| **GOAT FAIL / negative result** | Add to the negative-results section (`09_feature_catalog/negative_results.md` for katgpt-rs, equivalent elsewhere). Mark the plan with the failure mode. **Keep the entry** — negative results are load-bearing. |
| **Issue closed (investigation)** | If it changes a primitive's status (e.g. "map-fidelity hypothesis exhausted"), update that primitive's README/docs entry. If it's pure investigation with no status change, it may not need a doc writeup — judge case by case. |
| **Research note (PASS/Gain/GOAT)** | If it led to a plan, the plan entry is the writeup. If it's a standalone PASS verdict with no plan (e.g. "already shipped"), add a one-liner to the relevant `.docs/` group README. |

### Step 4 — Write the updates

Apply the repo-specific rules:

#### katgpt-rs (the public engine)
- **README.md**: feature showcase entries (one `###` section per primitive with a GOAT gate table), the opt-in features table, the default-features list, the Documentation Index.
- **`.docs/01_orientation/overview.md`**: the full feature-flag table (one row per flag).
- **`.docs/09_feature_catalog/`**: opt-in features + negative results.
- **`.docs/<group>/README.md`**: the group's fusion map + file list.
- Numbering: never reuse a plan/issue/benchmark/research number. Read the `.highwater` file, use `value + 1`, write it back.

#### riir-ai (the private runtime)
- **README.md**: crate table + feature showcase.
- **`.docs/`**: 12 numbered folders — drop new docs in the right group, add one line to the group README.
- Cross-repo: if a katgpt-rs primitive was consumed, note the fusion in the riir-ai doc AND the katgpt-rs doc (bidirectional cross-refs).

#### riir-chain (7-folder `.docs/` book, reindexed 2026-08-08)
- **README.md**: build surface, feature flags, consumers, drift notes.
- **`.docs/`**: 7 numbered folders mirroring the `riir-ai/.docs/` format — `01_orientation` (what it is + feature surface + module map + how the ledger works), `02_consensus`, `03_economics`, `04_daemon` (incl. the operator runbook), `05_wallet` (trust boundaries, SIWR, node certificates), `06_operations` (rolling upgrade across protocol versions, e2e coverage, failure scenarios), `07_formal_verification` (pointer — the invariant table stays in `AGENTS.md`). Drop new docs in the right group folder and add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point.
- **Division of labour with riir-ai (set 2026-08-08):** riir-chain holds the canonical chain docs; `riir-ai/.docs/07_neuro_symbolic_chain/` is a **fusion map + feature highlights** that links here and keeps only what is riir-ai's own (the Egg/Shell raw-vs-latent boundary, latent precision realms, game-layer sync strategy, CF Workers edge topology). Do not re-centralize chain internals in riir-ai — that duplication is what drifted before. Cross-link bidirectionally.
- **Module map discipline:** `01_orientation/overview.md` claims to list every `src/` and `crates/riir-chaind/src/` subtree. If a plan adds a module, add the row — a map that silently omits modules reads as "these do not exist".
- **AGENTS.md**: the FV (Lean 4) invariant table lives here (mirrored in `.proofs/README.md`), NOT in `.docs/`. Plan 016 spec self-tests live next to each spec module under `.proofs/RiirChainProof/`.

#### riir-neuron-db (11-folder `.docs/` book; 10th added 2026-07-30, 11th `11_cli/` added 2026-09-09)
- **README.md**: build surface — feature gates (default-on / transitive / opt-in / per-feature prose sections for promoted primitives) + Formal Verification summary + License. Prose sections are reserved for promoted default-on features; opt-in features get table rows only.
- **`.docs/`**: 11 numbered folders mirroring the `riir-ai/.docs/` format. Drop new docs in the right group folder (by capability: shard substrate / freeze-thaw / consolidation / vessel / specialized / zone / examples / FV / **local-kv Warm tier** / **CLI**), add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point. The `05_secure_vessel/vessel_primitive.md` doc is the restored home of the old `15_vessel.md` (corrected: riir-neuron-db is "this crate", NOT katgpt-rs per Plan 006). The `10_local_kv/` folder covers the `LocalKvStore` + `CommitLevel`/`CommitBatch` + WAL compaction + BM25 (the Warm tier substrate backing per-player state recovery in riir-mmorpg-examples Plan 013, added Issue 043). The `11_cli/` folder covers the `ndb` binary (Plan 327: Warm-tier CRUD, identity login via riir-auth `account_key`, multi-node WAL-mirror sync).
- **AGENTS.md**: the FV (Lean 4) invariant table lives here (mirrored in `.proofs/README.md`), NOT in `.docs/`. The `.docs/09_formal_verification/` folder is the narrative overview; `AGENTS.md` is the authoritative invariant table.
- Cross-repo: if a primitive was consumed by `riir-ai` or `riir-chain`, the fusion is documented bidirectionally.

#### riir-train (6-folder `.docs/` book, reindexed 2026-07-15)
- 6 numbered folders (`01_orientation/` … `06_cross_cutting/`), unnumbered `.md` files inside — mirrors the `riir-ai/.docs/` format.
- Training-method research vault: adapter training, distillation/RL, data filtering, cross-cutting audits.
- `README.md` is minimal — role + sibling layout.
- Branch: `develop` (flipped from `main` 2026-09-04, `main` frozen — the census table's row wins over this section's older text).

#### riir-game-sdk (10-folder `.docs/` book; 10th added for multiplayer topology)
- **README.md**: build surface — boundary rule, leaf constraint, spatial canonical, feature gates, Phase 2/3 status table.
- **`.docs/`**: 10 numbered folders mirroring the `riir-ai/.docs/` / `riir-neuron-db/.docs/` format. Drop new docs in the right group folder (by capability: spatial-entity / tick-world / rules-ai / game-builder / zone-living-world / gm-dashboard / examples / lessons / **multiplayer-topology**), add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point. The `10_multiplayer_topology/` folder covers the two-binary authority/player production model + avatar/game sync facade (the consumer pattern for the documented C1/C2/C4 chain topologies).
- **AGENTS.md**: authoritative repo-local context (phase status, boundary rule rationale, leaf-constraint argument, the canonical-failure lessons). The `09_lessons/` folder is the narrative mirror of those lessons.
- `examples/`: showcase examples are part of the doc surface (Issue 517 rule) AND documented in `.docs/08_examples/`.
- **Leaf constraint reminder**: this crate has zero sibling path deps. Docs that reference sibling repos use relative links only — never imply a code dependency.

#### riir-mmorpg-examples (no `.docs/` folder — AGENTS.md-centric)
- POC consumer of `riir-game-sdk` (orchard multiplayer: 1000-NPC swarm + cross-target Bevy binary).
- **No `.docs/` folder** — documentation lives in:
  - `AGENTS.md` — the authoritative narrative (role, topology, plans/issues/benchmarks index, canonical-failure lessons, honest POC-grade caveats).
  - `README.md` — build surface (status, build commands, env vars, Plan/Issue index).
  - `.plans/` / `.issues/` / `.benchmarks/` — individual plan/issue/benchmark files.
- The `AGENTS.md` is large (~1000+ lines) and IS the doc surface — `doc-sync` for this repo means keeping `AGENTS.md` sections current with landed plans.

#### riir-refine (12-folder `.docs/` book)
- **`.docs/`**: 12 numbered folders mirroring the `riir-ai/.docs/` format — corpus / drafter / pruner / verify / ruliology / examples / benchmarks / lessons / self-evolve / domains / **ANE substrate knowledge** (`12_ane/`, riir-ai Issue 726 T0 distillation). Drop new docs in the right group folder, add one line to that folder's `README.md` index table.
- **`AGENTS.md`**: the batch-mining progress notes + sweep records live here (the cross-repo clippy-fix record home). A landed fix slice in a sibling repo (katgpt-rs, riir-train, riir-ai) gets its progress note in the SAME commit as the fix — a later `doc-sync` run defers to the fixing session (never write progress notes for someone else's in-flight sweep).
- **README.md**: Status + Quick Start + Usage + feature gates.

#### riir-unity — RETIRED 2026-09-04 (`git/obsolete/`), lineage only
- **`AGENTS.md`**: domain boundary (no Rust crates here; UPM package is build output; no engine substrate in C#) + the Unity MCP rules + the issue log. Doc-sync = issue-log sections for resolved issues + module-map freshness (the `Packages/com.riir.viewbridge/` population + scene wiring notes).
- The Rust side of any feature lives in `riir-viewbridge` — cross-repo arcs (e.g. Issue 004) document on BOTH sides at arc close.

#### riir-viewbridge (`.docs/01_orientation/` + an AGENTS.md-centric remainder)
- **`AGENTS.md`**: workspace layout (core/derive/abi/xtask) + boundary rules (latent/raw wall, generated-bindings rule, catch_unwind) + the issue log.
- **`.benchmarks/`**: GOAT records (e.g. Bench 002 node GOAT). Doc-sync = issue-log resolution entries + benchmark cross-refs.

#### Every repo not subsectioned above (riir-dapps, riir-dao, riir-auth, riir-deployer, riir-esp32, riir-kat, katgpt-web, mmorpg-remake, mmorpg-editor)
- Follow the census-table row — these are AGENTS.md/BOUNDARY.md-centric: doc-sync = AGENTS.md/BOUNDARY.md status sections + numbering highwater + README freshness (riir-kat has no README; its AGENTS.md is the surface). The repo's own AGENTS.md supersedes this skill. The no-docs rows added 2026-10-24 (jup-tool, reflex-site, riir-llm, seal-std, seal-render) and the empty-book `sealm-toolkit` ride the same treatment.
- `katgpt-web` checkout may sit on a feature branch (see census row) — sync the branch you find, and say which one in the run log.

### Step 5 — Verify

- **No broken links**: every `[...](.plans/NNN_*.md)` must point to a file that exists.
- **No stale numbers**: if a README entry says "ratio 0.01" but the benchmark says "0.27", the README is wrong — update it.
- **Numbering discipline**: `.highwater` files must be bumped when new plans/issues land.
- **Honesty**: a GOAT FAIL stays a GOAT FAIL in the docs. A "stays opt-in" primitive is documented as opt-in with the reason. Never upgrade a verdict in the docs without the benchmark to back it.

### Step 5.5 — Semantic drift lane (OPTIONAL stage — T4, Issue 977 / EGM `egm-drift-report`)

For each doc the quarter's sync actually touched (not the whole book — the touch set bounds the cost): verify the doc's **named symbols and checkable claims** against the CURRENT code state — does `fn name` still exist at the cited path, does the named feature still carry the cited default, does the named struct still own the cited field? Report ONLY differences (a clean sweep prints "No drift detected", never a per-doc recital). `feature-gate-audit` keeps ownership of the feature-gate CLAIM subset — this lane covers the general case: any named symbol/claim in a touched doc that the code no longer agrees with.

- The lane REPORTS, never auto-edits: a drift finding lands as a listed difference for the sync step (Step 4) to resolve, because the fix direction is a judgment (update the doc vs the code drifted).
- **Proof fixture (the zero-finding trap):** a deliberately drifted doc pair seeds the lane — a doc claiming a `fn drift_probe_fixture` at a path where it does not exist MUST be flagged; a run that reports nothing on the fixture fails the lane, never reads as clean docs (fixture seeded 2026-11-04, Issue 977; recovery via `git log -S drift_probe_fixture -- <this file>`).

### Step 6 — Commit (ONLY through the gate — never bare `git add` + `git commit`)

Per the global `AGENTS.md` rule: **always commit at task completion**. Use `docs:`
prefix. Stay on the repo's working branch (per the census table — `develop` for most; `main`
for katgpt-web; riir-train flipped to `develop` 2026-09-04). Do not push.

**Commit through `tools/sync_commit.sh`** — the mechanical form of the
staged-index standing lesson below (the #42 incident swept a stale sibling
index through a bare `git add` + `git commit`; prose did not stop it, this
does). The gate refuses a non-empty index up front (stale sibling state —
adjudicate it first, never auto-clear), stages only the named paths, refuses
anything staged beyond them, and commits via the partial form
(`git commit -- <paths>` — a temporary index from HEAD + the named paths), so
content staged concurrently never enters the commit and the real index is
never disturbed:

```sh
sh <workspace>/katgpt-rs/.agents/skills/doc-sync/tools/sync_commit.sh <repo> \
  "docs: sync .docs + README with recent plans (NNN, NNN, NNN)" \
  .docs README.md
```

Name exactly the paths THIS run edited — the file when the edit is one file,
the smallest containing dir otherwise; never a directory you did not touch.
`.plans/ .issues/ .benchmarks/ .research/` join the list only when the run
actually wrote there.

## Cross-repo coordination

The 5-repo (now 10-repo) family shares numbering namespaces for
plans/issues/benchmarks/research **within each repo** but NOT across repos.
When a katgpt-rs primitive is consumed by riir-ai, the fusion is documented
**bidirectionally**: the katgpt-rs doc notes "consumed by riir-ai/NNN", and the
riir-ai doc notes "consumes katgpt-rs/NNN".

Formal verification (Lean 4) has its own cross-repo pattern (Research 351):
each repo's `.proofs/` instance is self-documenting via its invariant table in
`AGENTS.md`. The `doc-sync` skill does NOT cross-port Lean files between repos
(coordinator rule C4: private proofs stay private).

## Shape-change contract (when `.docs/` grows a new top-level `NNN_*` folder)

**This is the root-cause guard for skill drift.** The recurring failure mode: a
plan adds a top-level `.docs/NNN_*/` folder to a repo, lands the commit, and
nobody updates this skill file — so the next `doc-sync` run operates on a
stale folder-count assumption (canonical drifts: `riir-neuron-db` 9→10 via
Issue 043, `riir-game-sdk` 9→10 via the multiplayer-topology docs, `riir-chain`
flat→7 via the 2026-08-08 reindex, `riir-ai` 11→12 via Plan 455's orphaned
`02_inference/` folder discovered 2026-08-08). This contract makes the update a
grep-able checklist instead of an implicit expectation.

**Trigger:** any plan/issue/commit that adds a new top-level `.docs/NNN_*/`
folder to any repo that already has numbered folders — or a doc-sync run's
Step 0 surface fill doing the same (seal-remake 2026-10-24: 1 → 6 folders +
the top-level entry point, from one run's enumeration). **Or** giving a `.docs/`
to a repo with none, or a first document to an empty-`.docs/` repo — creating
the folder is itself a shape change and requires this contract. Derive the
split per run (`ls */.docs`), never from this file: past hand-typed lists here
named repos as "no docs" while their folders already shipped, which is exactly
the hiding-a-shape-change failure this contract exists to kill.

**Checklist (run in the SAME pass as the folder-adding commit):**

- [ ] **Verify ground truth.** `ls <repo>/.docs/` and count the `NNN_*`
  folders. Do not trust the skill's current number — it may already be stale.
- [ ] **Update the table row** in `## The workspace repos and their
  doc shapes` above: bump the folder count, extend the range
  (`…NN_<new-folder>/`), and add a short provenance note
  (plan/issue number + one-phrase capability description).
- [ ] **Update the Step 4 section** for that repo: change the header
  count, add the new folder's name to the capability list, and add a
  one-sentence description of what the folder covers.
- [ ] **Grep-verify zero stale counts.** After the edit, run
  `grep -nE "<old_count> (numbered|folder)" ~/.agents/skills/doc-sync/SKILL.md`
  for the repo you touched — it MUST return zero hits. (Example: after
  bumping riir-neuron-db from 9 to 10, `grep -nE "9 (numbered|folder)"`
  filtered to the neuron-db rows must be empty.)
- [ ] **Commit.** This file is NOT in a git repo (`~/.agents/` is on-disk
  only), so the update lands by saving — but the repo-side commit that adds
  the folder should reference this contract in its message (e.g.
  `docs: add .docs/10_local_kv/ (shape-change contract: doc-sync SKILL.md
  updated)`).

**Who runs this:** the agent executing the plan that adds the folder — NOT a
later `doc-sync` run. `doc-sync` is the consumer of the skill; the contract is
the producer-side obligation. A `doc-sync` run that discovers a stale count
(row says 9, disk says 10) is a SIGNAL that the producer skipped this
contract — fix the skill then, but also note the gap.

## Anti-patterns

- **Do not** write a doc entry for a plan that hasn't landed yet. Speculative docs go in `.proposals/`.
- **Do not** remove a negative-result entry when closing its issue — the negative result is load-bearing documentation.
- **Do not** upgrade a GOAT FAIL to a PASS in the docs without the benchmark file to back it.
- **Do not impose a `.docs/` shape that differs from the repo's existing convention — respect the FORMAT you find.** Numbered vs flat vs named folders, slugs vs prefixes: follow what's there. **FORMAT is not COVERAGE (owner correction, 2026-10-24, twice-paid):** respect-for-shape was once read as "the book is 1 folder, keep it 1 folder" while the code grew to 8 crates / ~550 files — the convention decides HOW docs are shaped, never WHETHER the code is covered; Step 0's surface gap overrides it, and the fill lands in the family format. The workspace shape split (numbered-folder books / no-`.docs/` AGENTS-centric / empty-`.docs/` / NAMED subfolders / flat numbered FILES) is DERIVED per run from the census trigger's one-liner — never re-typed here; every hand-typed version of that split in this file's history was wrong within days. A shape change is a deliberate, committed decision governed by the **Shape-change contract** above.
- **Do not** renumber existing docs — the numbering discipline is monotonic and never reused.
- **Do not** document trivial mechanical commits (lockfile bumps, clippy fixes) unless they close a tracked issue.

## Standing lessons (distilled from the run log — the load-bearing process rules)

- **Per-file logs + pickaxe are the reliable baseline tools.** `git log -- .docs README.md AGENTS.md` once reported `5a1330b` as game-sdk's newest doc-touching commit while `git log -- AGENTS.md` + `git log -S '<landed-string>' -- <file>` proved `1977d83` (two days newer) had touched AGENTS.md. Always re-verify a suspicious baseline per-file before declaring a range clean.
- **Narrow producer-side syncs leave holes the baseline heuristic can't see.** A narrow sync fixes its own feature and skips everything else, so the strict `baseline..HEAD` range can be empty while the book still misses older landings. Gate runs GREP the book for recently-landed headline features; don't just trust the range. The inverse gap exists too: code landing to MATCH an already-documented row (mmorpg `?transport=local`) is invisible to any baseline heuristic — nothing to fix, but don't declare a gap either.
- **Grep extraction regex must include digits.** `^[a-z_]+ =` over `[features]` silently dropped `avatar_sync_ed25519` and `game_sync_p2p` — nearly filed phantom "documented-but-dead feature" findings. Use `^[a-z0-9_]+ =`.
- **The staged-index check must GATE the commit, not precede it.** One run saw two sibling-staged files in `git diff --cached --name-only` output and then sailed past them because the check was `;`-chained ahead of `git add` + `git commit` — the sibling's WIP landed inside the run-log commit (benign outcome, luck not process). The correct form: run the check as its OWN command and read it, or use `git commit -- <paths>` (a partial commit builds a temporary index from HEAD + the named paths, leaving anyone else's staged hunks intact — the safe form on shared checkouts). `scripts/staged_set_audit.py` (katgpt-rs) reports the same signal class pre-commit. **Mechanical since the 10-26 gate unit: every doc-sync commit goes through `tools/sync_commit.sh` (refuse-nonempty-index → add named → verify staged ⊆ named → partial commit); the bare form is retired for this lane — the #42 incident proved the prose alone doesn't hold.**
- **A doc claim of determinism is a claim about a proof.** When a fix refutes the proof (the ndb Bm25 tie-truncation class), grep the book for the CLAIM, not just for coverage of the fix — the fix landed with in-source comments only and the book kept asserting the falsified invariant.
- **False positives: grep the repo's OWN `.benchmarks/` before declaring coverage.** Different repos (and even different series in one repo) reuse bench numbers — the overview's only "Bench 025" hit was a different numbering series.
- **Broken links: prose citations survive file removal, markdown links do not.** The noise-reduction rule removes record files but no link-fix pass followed, so every closed issue left `](.issues/NNN…)` links dangling. Standing tools (committed in katgpt-rs **`.agents/skills/doc-sync/tools/`** — moved out of a repo-root `tools/` that no longer exists; the 2026-09-04 landing row below still says `tools/` and is history, not a path): `python3 .agents/skills/doc-sync/tools/linkcheck_sweep.py` (census) + `python3 .agents/skills/doc-sync/tools/link_fix.py <repo>` (auto R1-repoint / R3-delink), re-sweep to verify, commit pathspec'd `.md` only. Guards baked into the tools: **absent-repo** (a link into a workspace repo not checked out on the running box is UNVERIFIABLE, not broken — `exists()` is box-relative), **backtick** (link text already code-marked is emitted unwrapped), **EOL** (the fixer reads/writes with `newline=""` since 2026-09-22 — byte-exact round-trip; the pre-09-22 LF-normalization history and its workaround are retired), and a markdown link split across lines is invisible to one-line fix regexes (NOMATCH → manual two-line edit).
- **Never cite "Plan/Research NNN" without naming the repo** when the number exists in more than one namespace — and never link a path you haven't verified (`ls` it first; per-repo numbering namespaces collide constantly).
- **Classification against dirty trees uses `git show <rev>:` content, never the working tree.** Sync remotes first; `git show -1 <sha>` mis-parses (the `-1` overrides the sha — pass the sha alone). The reliable per-file baseline is the 4-surface set (AGENTS.md / README.md / `.docs/` / the feature file) + a since-count.
- **"0 passed" in a baseline run is the reliable gate detector** — grep on attribute FORMS lies (whole-file `#![cfg]` gates sit below doc-comment headers, mod-level `cfg(any(feature…))` compiles empty under default).
- **Baseline-diff is coverage of TEXT, never of SURFACE (owner correction, 2026-10-24).** A clean `baseline..HEAD -- .docs README.md` range says nothing about landings the book NEVER covered: seal-remake's book held 2 orientation docs (one retired) over an 8-crate workspace, and every run passed it green because nothing doc-shaped was committed. Each run ENUMERATES the code surface once (workspace members + bins from the root manifests; derive, never trust a stale count) and asks what no doc row covers; the crates map it writes is role-from-lib-doc only (facts read that session), never a restatement of AGENTS.md lanes — that duplication is the drift this gate exists to catch.

## Run log (compact — full narratives live in git history)

Each row's durable record is the named `docs:`/fix commit(s) in the repo it
touched, plus this file's own history
(`git log -p -- .agents/skills/doc-sync/SKILL.md` — rows carried full
narratives until the 2026-09-05 compaction; `git log -S '<date>' -- <this
file>` recovers any of them). **Re-compacted 2026-09-11** — the rows appended
after 09-05 had regressed to full narratives again; same recovery applies.
**Re-compacted 2026-11-10 (amo, the idle-loop lane)** — the 11-09/11-10 rows
regressed to narratives a third time (the documented cycle: rows born compact,
grown narrative, compacted); all 15 compacted to the one-line form, 62.4→50.1KB
— the row-prune lever plus the narrative-compact lever together, so the
"prune lever exhausted" notes of 11-10 are superseded by this run.
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`. Prune events: 09-21 · 09-23 · 09-28 · 10-12 · 10-26 (two lanes) · 10-30 · 11-12 (25→15, 62,051→57,511B, trip) · 11-13 (trip-preempt) · 11-15 ×3 (trip-preempts; the 11-15 census-unit run also disk-clean ASSESSED list-only — /private/tmp ~126Gi of >48h per-plan target gates DELETION OWES OWNER APPROVAL) · 11-15 ×2 more (the census-unit row and this row; 14 rows standing) · 11-16 ×3 (the sdk055 row's trip + two more trip-preempts; 15 rows standing) · 11-17 (the boundary-494 pairing row; 15 rows standing) · 11-20 (m3 delta lane: trip TRIED at 60,694B; the prune-to-15 lever EMPTY — 13 rows standing, 14 after the append) · 11-20 again (the SAME lane's body compaction + census drift fixes, DISCHARGED: 62,101→53,304B — the census table compacted to shape facts with provenance moved to git history, and the drift fixes landed with it: riir-dao + seal-std shapes corrected to flat numbered FILES (disk had `.docs/001_*.md` each), 4 missing workspace rows added (riir-ledger README-only / zed flat 001–018+033–038 / riir-grams no-.docs / seal-sprints no-.docs), mmorpg-remaster + jup-tool retirement flags added per katgpt-rs AGENTS.md, the stale riir-train `main` branch lines in Step 4 + Step 6 corrected to the table's develop row)

| Date | Scope | Verdict | Record (primary fix commits) |
| 2026-11-20 | delta unit + ONE lease repair + the 60KB trip (m3, the disk-block reverification + doc-sync delta lane zed-m3-cont@m3 — the boundary-531 pairing row): disk blocks re-verified ALL HOLD (escrow-authority PID 6952 live 1h23m → seal-remake/target 327G; the /tmp set all today-mtime, the s038-gate×2 Nov-16 pair left to the active sprint038 lane's own teardown; df 394→373 GiB — lanes compiling, expected); 3rd 1226-class MALFORMED claim reshaped seal-sprints `dabfa11` (.plans/063 claude-code@pawsengineers-macbook-pro's epoch slot held a session-uuid → 1795240800, the line's own expiry, the 30f116e conversion; gate PASSED fresh 92 files after); 60KB trip TRIED at 60,694B — prune-to-15 lever EMPTY (13 rows standing, 14 after this append); body compaction LANDED later this same unit (62,101→53,304B: the census table compacted to shape facts, provenance moved to git history — see the Prune events line); book surface clean on today's landing repos (self-recording: riir-ai i1245 rows landed same-commit, seal-sprints tracker edits are the record, refine/katgpt-rs housekeeping lanes own theirs, seal-remake merges in flight); benches NOT owed (md-only) | numbering gate PASSED fresh on seal-sprints (exit 0 after the reshape) | seal-sprints dabfa11 + katgpt-rs (this row, this commit) |
| 2026-11-20 | delta unit + ONE repo write (m3, the seal-sprints hygiene lane zed-m3@m3 — the boundary-530 pairing row): sprint!039 CLOSED `38c381e`+`2f46027` (the B′ landing + acceptance + the 177 close were all on origin but the lapsed amo lease never recorded them; HISTORY/DONE/TASKS rows folded, plan 033 → DONE, the distance-hunt residual filed `sprint!085` highwater 84→85, the lease fold in-row); 2 MALFORMED claim lines reshaped `30f116e` (084's epoch held a session-uuid, 061's a git hash — the Issue-1226 blind class; claim data preserved, the numbering gate's 2 WALL findings cleared to PASSED); seal-sprints numbering gate PASSED fresh (92 files, claim lint clean); en-route adjudications: sprint!045/069/075 verified CLAIMED-live (the 075fin lane holds all three — untouched), seal-remake lobby_smoke WIP = issue 050's active-by-work lane (skip), issue 177 verified CLOSED GREEN `64b2c2a1` + plan 052 DONE; doc-sync 60KB trip NOT tripped (~53KB at entry); benches NOT owed (md-only) | numbering gate PASSED fresh on seal-sprints (exit 0; claim lint clean ×2 repaired files) | seal-sprints 38c381e+2f46027+30f116e + katgpt-rs (this row, this commit) |
| 2026-11-20 | delta unit + TWO repo acts (m3, the post-close idle housekeeping lane zed-08a67a53 — the boundary-531 pairing row): book surface verified complete for the cycle's delta (the 1245 arc's records landed in-arc by the closing lane: bench 1176 + HISTORY trail 0566f8dd6+5c28975f4+649030017; fetch 33 repos current, only riir-train upstream +1); riir-reflex i125 CONTENDED mid-cycle — the lease `1240090` (zed-m3-i125floor@m3) landed while this lane ran the corpus forensics, YIELDED per the lease law with findings mailed to the holder: the issue's per-workflow-gold-key premise REFUTED on this box's bytes (test-000..003 all score golds in range of their own criteria; `security_incidents.severity` the only 5-level head per the manifest table and heterogeneous WITHIN test-003 — crit_len 4×100 + 5×100; the port spec §3.1 + the Rust loader agree), so the 4090 panic needs the stale-dataset-bytes digest check + an engine-side probs-length assert BEFORE any loader change (the Score arm's unguarded `int(label)` hardening is worth landing regardless); riir-game-sdk i050 closed + removed `967a74b` (the landed mmap ladder verified in-tree + upstream 646 fix verified at HEAD; sdk numbering gate PASSED fresh; sibling Cargo.lock drift stashed as-found, never committed); 60KB trip NOT tripped (54.1KB at entry); benches NOT owed (md-only) | numbering gate PASSED fresh on riir-game-sdk (exit 0, claim lint clean); no numbered files moved in katgpt-rs (skill rows only) | riir-game-sdk 967a74b + katgpt-rs (this row, this commit) |
| 2026-11-20 | delta unit, ZERO book writes (m3, the i1245 verdict-closure lane zed-1245-cont — the boundary-529 pairing row): the 1245 arc IS its own record (riir-ai 0566f8dd6 the bench-side verdict corrections + 5c28975f4 the verdict trail naming both concurrent adjudication sessions — HISTORY.md rows landed in the same commits; bench 1176 the record of record; issue file removed at close with the HISTORY fold, lease folded in-row, the re-open trigger recorded); book surface verified complete; numbering gate PASSED fresh on riir-ai (exit 0, 1451 files, claim lint clean over 506); 60KB trip NEAR (58.6KB at entry, ~59.3KB after this append — one append from the prune); boundary pairing row = the 529th; benches NOT owed (md-only) | numbering gate PASSED fresh on riir-ai (exit 0) | riir-ai 0566f8dd6+5c28975f4 + katgpt-rs (this row, this commit) |
| 2026-11-20 | delta unit + ONE book write (4090, the i677-T1/1197-adjudication lane zed-2cd66a64 — the boundary-510 pairing row): the T0 census instrument's 16 subprocess calls repaired (encoding=utf-8 errors=replace, the Issue-778 class — the cp874 box's reader thread died on text=True with no encoding; the holder's landed artifact, the repair is the gate's own prescription, landed riir-train b4cb7b74 with the T1 disclosure); the 1197 poisoned-boot adjudication IS its own record (riir-ai aa13a4e82: CALIBRATION INVALID ×78 + stalled 4760/100k, P0 reader PASS banked, the re-queue plan recorded — issue-is-the-doc); the i695 race repair acknowledged to the repairing lane (my empty pre-fix artifact deleted; the real freeze gates on the join + the 12-row mutation battery); seal-sprints census ROUTED to m3's sprint lane (the expired leases there are that lane's continuous re-claim cadence — 075 is literally another 4090 lane's live claim; interloping collides); benches NOT owed (py + md only) | the i677 encoding repair + the 1197 adjudication are the book surface; no .docs/README debt found | riir-train b4cb7b74 + riir-ai aa13a4e82 + katgpt-rs (this row, this commit) |
| 2026-11-20 | seal-sprints closed-issue hygiene, ONE file (m3, zed-25e6be16@m3 — the boundary-528 pairing row): sprint!080 (the Quests tab reads journal quests, merged sge `758c28838` 2026-11-19, DONE.md row predating) file removed with its compact HISTORY row folded `a5491dd` (named-path commit via sync_commit, pushed clean FF); numbering gate PASSED fresh on seal-sprints pre-commit (92 numbered files, exit 0, claim lint clean, 2 historical collisions ratcheted); the fleet's `--lint .` invocation on claim_lease_check is per-FILE, not a repo finding; closed sprint#NNN PLANS stay by convention (only issue files ride the noise-reduction removal); doc-sync 60KB trip NOT tripped (56,619B at entry); benches NOT owed (md-only) | numbering gate PASSED (seal-sprints, exit 0) | seal-sprints a5491dd + katgpt-rs (this row, this commit) |
| 2026-11-19 | seal-remake HISTORY backfill — the sprint!010 + sprint!042 batch merge `26716c9e` landing row (m3, zed-m3-031cell — this cycle's doc-sync unit; the gap the prior session parked as sibling-owned-at-check-time, re-adjudicated: the sibling tree still carries staged vibe.rs WIP so the row landed via a DETACHED worktree at origin/develop + push HEAD:develop, the 512th worktree-push path, main checkout untouched): gap CONFIRMED first (zero `26716c9e`/branch-name hits in origin:HISTORY.md + no HISTORY commit in the 11-14..17 window while every sibling merge got its row); row f1f3220c lands the four legs (gate-timing d154e6a8 / one-log 2a6b6574 / shop-chaos d7f43180 / quest-lane-chaos 7d502bd2), the gate verdicts (56-layer guard ALL LAYERS GREEN + p9 census PASS 2026-11-15T16:36Z at the merge), plan 013 G1/T6 discharge + the canonical pointer to seal-sprints HISTORY; docs-only carve-out = reading the diff IS the verification (4-line .md insertion, byte-safe python splice, worktree removed + pruned same act); boundary pairing row = the 527th; benches NOT owed (md-only) | docs-only push clean FF (47538cf6..f1f3220c), no race | seal-remake f1f3220c + katgpt-rs (this row, this commit) |
| 2026-11-19 | riir-ai plan-637 T4 GOAT cell + T5 close-the-loop (amo, zed-plan019t4p-hygiene@amo — the boundary-525 pairing row; the same lane's T3 unit closed earlier today): Bench 1173 landed riir-ai `9bca8fc16` (G1 catch 1.0 = 1.0 at 2.36× fewer obs, G2 parity 1.023/1.021, G4 zero-alloc own-target, G3 default green; no-flip disposition = the plan's own T3 consumer law, no verdict needed) + issue 1223 bookkeeping `02dcc44f6` (T2/T2b/T4-here/T5-here folded); katgpt-rs family commit lands plan 637 T3/T4/T5 folds (plan COMPLETE) + README GOAT-table seam row + belief_contract module doc seam section + issue 1000 UNBLOCK-line close; the cycle's doc debts verified clean at entry (katgpt-rs HEAD b135af369 documented, HISTORY current) and the cycle's own landings carry their records in the same commits (riir-ai HISTORY row + bench record + AGENTS line landed with the cell); numbering gate + claim lint NOT yet run on this commit's landing — they ride the commit's own gate sweep below; doc-sync 60KB trip NOT tripped (~54KB at entry); boundary pairing row = the 525th; benches OWED AND RUN (Bench 1173) | Bench 1173 four-gate sweep GREEN; numbering gate + claim lint to be run before push | riir-ai 9bca8fc16+02dcc44f6 + katgpt-rs (this row, this commit) |
| 2026-11-19 | riir-ai plan-637 T3 stalemate-drive primitive (amo, zed-plan019t4p-hygiene@amo — the boundary-523 pairing row; the same lane's issue-1230 execution closed earlier today): claim-first across both repos (katgpt-rs plan 637 lease `a80422d71` + the riir-ai issue-1223 expired-lease takeover `4222f3008`, lint-clean ×2 after the 1226-shape reshape); T3 primitive landed riir-engine `973595efd` gated `gate_contraction_derived` (armed 20/0 + default 14/0 + clippy −D both postures + rustfmt; sibling quest_combat WIP measured as-found and untouched — staged-set audit one episode, named-path staging); plan 637's stale "T2 next" status line corrected (T2 landed `faa643b1a`); benches NOT owed (feature-gated primitive, no measurement claims — the T4 paired cell is the quiet-rig unit and amo held a heavy job); dep-edge surface '*.toml'/'*.lock'/BOUNDARY.md over this lane's landings = 0 lines (the katgpt-types edge rode T2's commit, pre-existing in-graph) | armed 20/0 + default 14/0 + clippy both postures exit 0 | riir-ai 4222f3008+973595efd + katgpt-rs a80422d71 (this row, this commit) |
| 2026-11-19 | katgpt-rs issue-1230 stale-dirt disposition (amo, zed-plan019t4p-hygiene@amo — the boundary-522 pairing row, absorbing the boundary-521 post-hoc row): the shared-checkout 9-file recipe executed claim-first (lease `73d0d097f` lint-clean pushed before the discard), 9 files re-verified then discarded (8 ancestor-stale + doc-sync mixed-buffer, census author disclaimed in-session), ff-only restored `319d0b052..73d0d097f` — 4 upstream landings + the landed run rows back on disk, the ff-block discharged; step-0 buffer source dormant (mtimes stable 7.5h+, no re-dirty post-discard); issue closed+removed with the HISTORY fold in the same commit; numbering gate PASSED fresh on katgpt-rs (exit 0, claim lint clean — the removed file carries no stale-claim residue); doc-sync 60KB trip NOT tripped (~52KB at entry); boundary pairing rows = the 522nd (this unit) + the 521st (the i932 lane's owed riir-ai run, recorded post-hoc); benches NOT owed (md-only) | numbering gate PASSED fresh on katgpt-rs (exit 0; claim lint clean) | katgpt-rs 73d0d097f (this row, this commit) |
| 2026-11-18 | riir-rethink housekeeping delta (amo, the plan-019 T4p wake lane — the boundary-519 pairing row): Step-0 surface re-derived (single crate, 22 src modules + 3 bins; maps 1:1 onto the AGENTS.md lane prose, no book gap); post-baseline window 102f59b..HEAD = plan-019 staging md only, NO landed feat/fix debt; census drift FOUND+FIXED — the 9th book folder `09_human_scale_ai` (grant lane ae19b6d ~10-31) postdated the 10-24 census row, shape-change contract backstop filled the row in THIS commit; numbering gate PASSED fresh on riir-rethink (exit 0, 63 files, claim lint clean over 20; legacy-collision advisory = the shared katgpt-rs scalar, gate green); doc-sync 60KB trip NOT tripped (~47KB upstream at entry); boundary pairing row = the 519th; benches NOT owed (md-only unit, riir-rethink tree clean at c24aac8) | numbering gate PASSED fresh on riir-rethink (exit 0, 63 files, claim lint clean) | riir-rethink c24aac8 + katgpt-rs (this row, this commit) |
| 2026-11-18 | riir-refab book sync + 5-repo residue batch (4090, the plan-635 housekeeping lane — the boundary-520 pairing row): riir-refab Step-0 surface check caught the uncovered new bin commands → README Lanes row gains rig-verify/rig-author + 005's stale P1 status line re-anchored to the landed lane decision, e11cba7; riir-infer residue folds 5588f10→299ad02 (1222 lease fold, plan 634 → DEAD per Bench-1217 FAIL, 1208 stale-claim dup dropped, 1223 embedded claim reshaped lint-clean); katgpt-rs 1000 stale claim folded 34a5383a2; grams 001 T1 ticked 3ba7247; mmorpg-examples 120 five landed boxes ticked b53bfc4; train residue b2210b8c (678 lease fold + 302 _DEFERRED filename/title drop + 397 status → terminal verdict); 347/498/466 verified cosmetic (prose-box false positives / closed-form lint-clean); numbering gates PASSED fresh on refab/infer/train (13/245/862 files, exit 0, claim lint clean); 60KB trip NOT tripped (50.8KB at entry); boundary pairing row = the 520th | numbering gates PASSED fresh (refab 13, infer 245, train 862 — all exit 0) | riir-refab e11cba7 + riir-infer 299ad02 + riir-train b2210b8c + katgpt-rs (this row, this commit) |
| 2026-11-18 | riir-refine issue-245 NEGATIVE close + pulled-window debt check (4090, shikuwa-4090-i245-poc0): Bench 161 (PoC-0 co-fire census, kill-fired NEGATIVE-sparse) + Research 420 §7 demotion + issue 245 removed + HISTORY row `7b52087a475`; debt check over the six pulled windows found NO book debts (sprint!067 closed with its own seal-sprints ledger 456b487; infer 1183 AGENTS row landed b9de7d8/507a91f; refine 386/387 verdict-only landings, no count-web movement); numbering gate PASSED fresh on riir-refine (exit 0, 990 files, claim lint clean over 392 incl. the en-route plan-310 malformed-claim reshape `7b52087a475`); doc-sync 60KB trip NOT tripped (48.7KB at entry); boundary pairing row = the 518th; benches NOT owed (parse-only instrument) | numbering gate PASSED fresh (riir-refine 990 files, exit 0; claim lint clean over 392) | riir-refine 7b52087a475 + katgpt-rs (this row, this commit) |

| 2026-11-18 | seal-sprints + riir-game-sdk closed-issue hygiene (m3, zed-i215closeout-hygiene — owner-directed "find close issue and doc and rm it all"): 11 CLOSED sprint!NNN files removed with 9 compact HISTORY rows folded (051/063 already had rows; 049's row disambiguated from the burned-049 header above it) `51476b2` + 2 MALFORMED claim lines reshaped en-route `3504eeb` (the gate's repo-wide lease check exits 1 = BLIND until reshaped; both gates re-run exit 0 after) + sdk issue-060 preserved-record folded to a HISTORY row + removed `9360401` (the file's own "preserved record" note superseded by the noise-reduction law, fold provenance named in the row) + GOLF.md filed (5 owner-gated questions: riir-ai 1176 signed dist_stage, riir-infer 1183/1179 beat-lane verdict, refine 221 retire-vs-keep ripe, riir-ai 1095 lever-3 standing, zed 068 key roll); numbering gates PASSED fresh on both repos (sdk's 055 collision re-pinned UP 0→1 in katgpt-rs + seal-sprints' own channel file landed, both in this commit); skipped: seal-remake (active agent + files modified today), riir-train 558 (routing restriction), zed 061/062/066/067/068 (outside roots; 066-068 carry open tails — 061/062 routed in the summary) | numbering gate PASSED fresh on seal-sprints (exit 0, claim lint clean over 75) + riir-game-sdk (exit 0 after the honest up-pin); doc-sync's 60KB trip NOT tripped (47,218B) | seal-sprints 51476b2+3504eeb + riir-game-sdk 9360401 + katgpt-rs (this row, this commit) |

| 2026-11-17 | delta unit, ZERO book writes (amo, the sdk055 arming-verdict lane — the boundary-492 pairing row): the verdict IS recorded in-tree (seal-remake Cargo.toml comment cites session b2f6686f AGREE + the merge/flip hashes) and sdk HISTORY.md carries the closing row (99e363d — file 055 removed, lease folded; .issues/060 keeps the collision record); the flip's doc surface = the drift test's ALIAS_MEMBERS (the pin travels with the code); riir-game-sdk numbering gate PASSED fresh (39 files, exit 0); benches NOT owed (feature-alias + md window; the parity gate ran twice as the armed posture's bench); mining GATED fresh (plan 383 lease exp 2026-11-18T07:28:22Z) | numbering gate PASSED fresh on riir-game-sdk (exit 0, 39 files; owed: numbered file removed in-window — the gate + the re-pin DOWN both discharged in-commit); doc-sync's own 60KB trip DISCHARGED (61,552B at entry — this append then prune 15, the 11-15 class) | seal-remake 3ca8b60e/6d34c6d3 + riir-game-sdk c381157/99e363d + katgpt-rs bbe17fc4f (this row, this commit) |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
