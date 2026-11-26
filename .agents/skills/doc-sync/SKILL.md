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
| `riir-ai` | 13 numbered folders (01…12_quest_style + 13_web_family; 02_inference renumbered 12_inference, then ebf30247a renamed 12_inference→12_quest_style 09-26 and dc9540e7f added 13_web_family 10-03 — the 11-19 census check caught both shape changes; this row is their contract backstop) — the private book | large showcase + crate table | same | develop |
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
| `riir-reframe` | flat numbered FILES (001_how_it_works_flow + its SVG figures) | yes | same | develop |

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
- **`.docs/`**: 13 numbered folders — drop new docs in the right group, add one line to the group README (`12_quest_style/` holds the Quest Style Bridge; `13_web_family/` the riir.ai family records).
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
**Re-compacted 2026-11-10 (amo, the idle-loop lane)** — the 11-09/11-10 rows · 11-25b (this lane: trip FOUND at 59,739B entry / 21 rows; prune-to-15 21→15, + this append = 16 standing)
regressed to narratives a third time (the documented cycle: rows born compact,
grown narrative, compacted); all 15 compacted to the one-line form, 62.4→50.1KB
— the row-prune lever plus the narrative-compact lever together, so the
"prune lever exhausted" notes of 11-10 are superseded by this run.
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`. Prune events: 09-21 · 09-23 · 09-28 · 10-12 · 10-26 (two lanes) · 10-30 · 11-12 (25→15, 62,051→57,511B, trip) · 11-13 (trip-preempt) · 11-15 ×3 (trip-preempts; the 11-15 census-unit run also disk-clean ASSESSED list-only — /private/tmp ~126Gi of >48h per-plan target gates DELETION OWES OWNER APPROVAL) · 11-15 ×2 more (the census-unit row and this row; 14 rows standing) · 11-16 ×3 (the sdk055 row's trip + two more trip-preempts; 15 rows standing) · 11-17 (the boundary-494 pairing row; 15 rows standing) · 11-20 (m3 delta lane: trip TRIED at 60,694B; the prune-to-15 lever EMPTY — 13 rows standing, 14 after the append) · 11-20 again (the SAME lane's body compaction + census drift fixes, DISCHARGED: 62,101→53,304B — the census table compacted to shape facts with provenance moved to git history, and the drift fixes landed with it: riir-dao + seal-std shapes corrected to flat numbered FILES (disk had `.docs/001_*.md` each), 4 missing workspace rows added (riir-ledger README-only / zed flat 001–018+033–038 / riir-grams no-.docs / seal-sprints no-.docs), mmorpg-remaster + jup-tool retirement flags added per katgpt-rs AGENTS.md, the stale riir-train `main` branch lines in Step 4 + Step 6 corrected to the table's develop row) · 11-25 (the auto-prompt continuation lane: trip at 60,063B; 22→15, + this append = 16 standing) · 11-25 again (the 602nd pairing's trip: 60,188B entry; 18→15 rows standing) · 11-26 (this lane: trip at 59,765B entry; 19 rows + this = 20, prune-to-15 → 15 standing, the 627/626/625/634/622 pairing rows dropped)

| Date | Scope | Verdict | Record (primary fix commits) |
|---|---|---|---|
| 2026-11-27 | delta unit, ZERO book writes (amo, zed-3fbc05dd@amo — the boundary-648 pairing row, DEFERRED-HELD at the 648th landing [sibling-uncommitted prune edit held on this file] and landed at the continuation pass): the 648th pass's writes are RECORDS, all self-recording — riir-train 624+485 lapsed-lease claims routed c8249149 + 494 7b2948e4 (folds of lapsed holders zed-3fc4f42e/zed-072d9b4c; T2/P4/ramp routed to the post-i1186/cq3-repair 4090 window), the predecessor wakes proved PHANTOM and re-armed RFC3339-UTC (mails, not writes), riir-ai 1260 resolved-upstream (read); the continuation's acts are reads (the 20:45Z wake checklist discharged: 494/624/485 claims verified LIVE on origin/develop, 4090 QUIET + cq3 arc still broken → hands-off per 678 T8b) + the ff-pull of the sibling 649th (0c37c695d, its pairing row already landed) + this prune (the held 629th edit applied + the 635th/633rd rows dropped, recoverable via git log; 16→15 standing — the 649th row's over-cap note resolved); numbering gate not owed (skill rows only); benches NOT owed (md-only) | numbering gate not owed (skill rows only) | riir-train c8249149/7b2948e4 + katgpt-rs (this row, this commit) |
| 2026-11-27 | delta unit + record landings, ZERO book writes (m3, zed-ada727bc@m3 — the boundary-649 pairing row [renumbered 644→649 at the collision with the 645th–648th landing wave, the per-lane precedent AGAIN; the 648th's held pairing = this lane's uncommitted edit, landed here]): the pass's landings are RECORDS, all self-recording — riir-ai 86e32c844 (the combo compile fix; commit message + issue-1266 file carried the record, then the file closed+removed 7ce7488aa with the HISTORY row), riir-ai d2783e704 (the latch-pair fixture fix), seal-sprints 51fee20 (073 plan T2 GREEN + the TASKS sprint!085 row) + ac39878 (GOLF.md status); seal-remake f9a7cd749's merge message names the sprint; no feature/verdict/status change in any .docs book; numbered-file session gates RUN OWED AND PASSED FRESH (claim lint clean ×4 files; riir-ai numbering gate ×3: 1451/1450 files, seal-sprints ×2: 127 files, all exit 0); no numbered files allocated beyond issue 1266 (bumped .highwater 1265→1266 same commit, closed+removed same day — the counter's monotonic duty done); 60KB trip NOT tripped (boundary 58.3KB post-append + the 635th row dropped per the maintenance rule, 15 standing; doc-sync 57.7KB + this row ≈ 59.5KB, 17 rows over the 15 cap but under the trip line, noted for the next tripped landing) | numbering gate PASSED fresh (riir-ai ×3 + seal-sprints ×2, all exit 0) | riir-ai 86e32c844/d2783e704/7ce7488aa + seal-sprints 51fee20/ac39878 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE record close, ZERO book writes (amo, zed-263bb5c2@amo — the boundary-647 pairing row; the i258-close housekeeping pass): the lane's landing is the riir-refine 5904670b i258 close (issue file removed per its own close instruction + the HISTORY record — record-not-book, no feature/verdict/status change in any .docs book; the 6401ef1a guard change is documented in the record itself); dangling-ref sweep for the removed file CLEAN (zero hits repo-wide outside the file); numbered-file session gates RUN OWED AND PASSED (numbering gate exit 0, 1060 files, claim lint clean 425; the lease fold lives in the HISTORY record per the i254 precedent); no numbered files moved (zero allocations — one removal, one record); the pass's other acts are reads (fresh scoped riir-refine boundary gate, heavy-repo claim sweep, mining CONCUR at source, rig census + 4090 gitsync kick); 60KB trip TRIED at this append (59,765B entry) → prune-to-15 in the SAME commit (the 627/626/625/634/622 pairing rows dropped, recoverable via git log) | numbering gate PASSED fresh (riir-refine, exit 0) | riir-refine 5904670b + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE record landing, ZERO book writes (amo, zed-7efbf864@amo — the boundary-646 pairing row; the i1186 gate-sweep resumption lane's doc pass): the lane's landing is riir-infer 654288c9 (i1221 T0-extension PRICED — the record IS the research, SELF-RECORDING issue row; no feature/verdict/status change in any .docs book) + the routing mail 01a21923 (mail, not a write); the pass's acts are the fresh scoped riir-infer boundary gate (13 edges fresh, evidence not a write; the sh-vs-bash runner note recorded in the boundary row), the claim lint + numbering gate at both i1221 edits (both green), and the 1263-lane collision avoidance (i1253 T6 routed not claimed — the league is zed-f3b519c3@amo's live lease); no numbered files moved (zero allocations — the claim rode existing 1221); numbered-file session gates RUN OWED AND PASSED (claim lint clean + numbering gate exit 0); 60KB trip NOT tripped on this file (58.6KB entry + row); benches NOT owed (md-only + no-bench-on-a-busy-box) | numbering gate PASSED fresh (riir-infer, exit 0) | riir-infer 654288c9 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE route mail, ZERO book writes (amo, zed-amo-i1186-hk@amo — the boundary-647 pairing row [renumbered 646→647 at the collision with the 7efbf864 lane's own 646th landed mid-pass]; the i1186 storm's housekeeping pass): the pass landed NO book surface — its acts are the FIRST full workspace boundary gate since the i131 mirrors (evidence, not a write; 0 violations, 8 rot all box-artifact class), the fresh 42-repo fetch + ff-merges + seal-remake dirty-hold, the 16-repo claim sweep with ONE lapse adjudication (riir-ai 1260 holder lapse-notice mail 01a21927 — mail not a write), the mining-snapshot CONCUR read (B424 landing row 2981 verified present, head routed), and the 4090 gitsync kick; no numbered files moved (zero allocations); numbering gate not owed (skill rows only); 60KB trip TRIED on THIS file at the landing (59,765B entry + row ≈ 61.5KB) → prune-to-15 landed in the SAME commit (the 5 oldest 11-26 rows dropped, recoverable via git log); benches NOT owed (md-only + no-bench-on-a-busy-box) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE record repair, ZERO book writes (amo, zed-amo-i1186-hk@amo — the boundary-645 pairing row; the i1186 gate-sweep recovery lane s boss-Decision housekeeping pass): the lane s landing is the riir-refine 76f570d3 snapshot backfill (the B425 landing row the ThunderKittens batch commit 89e753d0 skipped — record-not-book, the .distill snapshot IS its own record; no feature/verdict/status change in any .docs book); the pass s other acts are reads (fresh 42-repo fetch + ff-merges, the fresh scoped riir-infer boundary gate, the 10-repo claim sweep, the mining-head census — B425 landed, elders resin-294 + hunch-311 routed by mail); no numbered files moved (zero allocations — the edit rode .distill prose only); numbered-file session gate not owed (no numbered-file edits; claim lint n/a outside .issues/.plans); 60KB trip NOT tripped (57,507B entry + this row ≈ 58.7KB; 18 rows over the 15 cap but under the trip line, prune NOT owed, noted for the next tripped landing) | numbering gate not owed (skill + .distill rows only) | riir-refine 76f570d3 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE table repair, ZERO book writes (amo, zed-8d70c2db@amo — the boundary-644 pairing row; the i1186 housekeeping pass's doc half): the pass landed NO book surface — the lane's landings (A2v2f verdict c2d61542, forensics 70df0116, T3-prep 97333746, handover 3872b8f1) are SELF-RECORDING .issues/1186 rows by the lease-holder zed-fb390152; the pass's acts are the fresh scoped boundary gate (evidence, not a write), the 15-repo claim sweep + the 46-file claim-lint battery (reads), the mining-snapshot CONCUR read (B425 head owned), and the 4090 rig probe (read); the ONE write = the L393 table repair (the boundary-625 row carried a stray `|---|` glued after its last cell — a 5th bogus cell breaking the render; trailing separator deleted, rows verified intact); no numbered files moved (zero allocations); numbering gate not owed (skill rows only); benches NOT owed (md-only + amo 4-shard busy); 60KB trip NOT tripped (boundary 56.9KB + row ≈ 58.5KB 16 rows standing; doc-sync 56.4KB + row ≈ 58KB) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE record repair, ZERO book writes (amo, zed-ff1f0cec@amo — the boundary-641 pairing row; the i1186 housekeeping pass): the lane's landing is the riir-ai 646 STALE-CLAIM repair (edc08bb19 — record-not-book, the plan file IS its own record; no feature/verdict/status change in any .docs book); the pass's other acts are reads (fresh 42-repo fetch, the 16-repo claim enumeration, the mining-gate B422/B423 verification at the snapshot tail, the rig census — 4090 loaded under 2 implants_cell contenders); numbered-file session gate RUN OWED AND PASSED (claim lint clean at the 646 edit; riir-ai numbering gate PASSED fresh 1450 files exit 0); no numbered files moved (zero allocations — the repair rode existing 646); 60KB trip NOT tripped (doc-sync 58,564B entry + this row ≈ 59.7KB, 19 rows over the 15-row cap but under the trip line, prune NOT owed, noted for the next tripped landing; boundary sibling pruned to 15 at its landing) | numbering gate PASSED fresh (riir-ai, exit 0) | riir-ai edc08bb19 + katgpt-rs (this row, this commit) |
| 2026-11-26 | 643rd pairing — delta unit, ZERO book writes (amo, zed-49c373fd@amo — the boundary-643 pairing row; the i067 Arm C fold-check-backstop lane's doc pass): the pass landed NO book surface — its acts are the fresh scoped riir-rethink boundary gate (evidence, not a write), the 19-repo claim sweep (reads; zero expired-unadjudicated, the mining head plan 419 verified CLAIMED-in-execution at source), the Arm C queue verification (rig status; the prior turn's pseudo-pipe void-check lesson recorded in the boundary row), and the 4090 liveness probe (Tailscale nvidia-smi); rethink's recent lane commits are bench/issue SELF-RECORDING (067 mid-lane, HISTORY rows land at close — the standing no-book-write verdict); no numbered files moved (zero allocations); numbering gate not owed (skill rows only); benches NOT owed (md-only + amo 4-shard busy) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + FOUR record repairs, ZERO book writes (amo, zed-33c67184@amo — the boundary-642 pairing row; the i1186 A2-lane overflow-successor's doc pass): the lane's landings are the lease-grammar record repairs themselves (reflex e276afa — 114 ChessBench/GPQA EMBEDDED → Lease closed folds; refine bd462862 — p266+p417 claim fields extracted from the Status lines to own-line Lease closed rows), each SELF-RECORDING (the fold row IS the record); numbered-file session gates RUN OWED AND PASSED FRESH (claim lint clean ×3 files; numbering gate reflex 154 files + refine 1057 files, both exit 0, 0 malformed after the repairs); no numbered files moved (zero allocations); 60KB trip TRIED at both appends → prune-to-15 landed in the SAME commit (boundary + doc-sync each crossed 60KB, the oldest rows dropped, recoverable via git log); benches NOT owed (md-only + amo 4-shard busy) | numbering gate PASSED fresh (reflex + refine, exit 0) | reflex e276afa + refine bd462862 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit, ZERO book writes (amo, zed-1ec414e8@amo — the boundary-640 pairing row; the i067 Arm C overflow-successor's doc pass): the pass landed NO book surface — its acts are the fresh post-pull queue sweep + census (reads; train 21 live / 0 expired after the upstream fold wave cc22e139), the dep-edge neutrality verification over the three since-ranges (evidence, not a write — the only Cargo.toml touch read in full = the i1256 implants intra-repo feature family), and ONE coordination mail to the 4090 resident lane (mail not a write); no numbered files moved (zero allocations — skill rows only); numbering gate not owed (the 635th precedent); 60KB trip NOT tripped (boundary 57.3KB + this pass's row ≈ 58.5KB, 15 rows standing; doc-sync 57.6KB + this row ≈ 58.6KB — 18 rows over the 15-row cap but under the 60KB size-triggered trip line, prune NOT owed, noted for the next tripped landing) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE executing training claim, ZERO book writes (4090/shikuwa, zed-4090-hk@4090 — the boundary-639 pairing row; the box housekeeping lane's boss-Decision pass): the lane's landings are riir-train records (1ce049d03 the plan-488 claim + cc22e1391 the 7-fold sweep — claim/lease rows ARE the records, self-recording; the executing train run's bench record lands at cell end); numbered-file session gates RUN OWED AND PASSED (claim lint clean ×8 at every edit; riir-train numbering gate PASSED fresh 895 files exit 0); no numbered files moved (zero allocations); 60KB trip TRIED on the boundary sibling at the landing → prune in the SAME commit (the 634th renumber-anomaly row dropped, 14 rows standing) | numbering gate PASSED fresh (riir-train, exit 0) | riir-train 1ce049d03 + cc22e1391 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE claim repair, ZERO book writes (m3, zed-5a977e1c@m3 — the boundary-638 pairing row; the idle-loop housekeeping pass): the lane's landing is seal-sprints 6ea0a25 (the .plans/109 MALFORMED claim-line reshape — record-not-book, the issue file IS its own record; no feature/verdict/status change in any .docs book); the pass's other acts are reads (fresh 27-repo fetch sweep, the 18-repo claim enumeration, the mining-gate B422/B423 landing verification at the snapshot tail); numbered-file session gates RUN OWED AND PASSED (claim lint clean at the 109 edit; seal-sprints numbering gate PASSED fresh 126 files, 0 stale allocators, 0 malformed); no numbered files moved (zero allocations — the repair rode existing 109); 60KB trip NOT tripped (doc-sync 55,707B entry + this row ≈ 57KB; boundary sibling pruned to 14 rows at its landing) | numbering gate PASSED fresh (seal-sprints, exit 0) | seal-sprints 6ea0a25 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit + ONE claim landing, ZERO book writes (amo, zed-a9773ebb@amo — the boundary-636 pairing row; the i067 ESC T4 overflow-successor's doc pass): the lane's landing is the riir-infer 1255 CLAIM (6213ae24 — the claim line + T0-prep note ARE the records, self-recording issue file; no book surface owed — no feature/verdict/status change in any book); the pass's other acts are reads (fresh boss-entry enumeration post-fetch, the mining-gate fresh verification that the B422 AND B423 landing rows are BOTH present in the snapshot — the 635th's routed tranche B confirmed EXECUTED, nothing owed); numbered-file session gate RUN OWED AND PASSED (claim lint clean at the 1255 edit; numbering gate ran in the detached claim worktree — the pre-existing 1151 unpinned collision + the two ratchet re-pin advisories held AS-FOUND, foreign pre-edit state verified by the stash-run, never this lane's); no numbered files moved (zero allocations — the claim rode existing 1255); 60KB trip TRIED on the boundary sibling at the landing → prune-to-15 in the SAME commit (the 620th + 623rd rows dropped); doc-sync 17 rows standing over the 15 cap → prune-to-15 landed alongside this append (the 3 oldest 11-25 rows dropped) | numbering gate ran fresh in the claim worktree (claim lint clean; pre-existing advisories as-found) | riir-infer 6213ae24 + katgpt-rs (this row, this commit) |
| 2026-11-26 | delta unit, ZERO book writes (amo, zed-4f3bcc3d@amo — the boundary-631 pairing row; the i470-T4 retry continuation lane's doc pass): the lane landed NO book surface — its acts are the lease stand-down (Path-1 acceptance to the live i470 holder zed-b586a370@amo, mail not a write), the TDR-precedent heads-up relay (mail), the fresh scoped riir-train boundary gate (evidence), the queue CONCUR (the riir-train +1 = the i666 claim lane's own landing; 470/513 claim lint clean), teardown reads (/tmp/plan470 5.0G + /tmp/i470_gate 0B HELD till landing, df 164Gi); no numbered files moved (zero allocations — the pairing rows are the only writes, both skills); numbering gate not owed; benches NOT owed (md-only + zero perf claims) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
