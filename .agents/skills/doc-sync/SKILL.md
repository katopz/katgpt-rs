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
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`. Prune events: 09-21 · 09-23 · 09-28 · 10-12 · 10-26 (two lanes) · 10-30 · 11-12 (25→15, 62,051→57,511B, trip) · 11-13 (trip-preempt) · 11-15 ×3 (trip-preempts; the 11-15 census-unit run also disk-clean ASSESSED list-only — /private/tmp ~126Gi of >48h per-plan target gates DELETION OWES OWNER APPROVAL) · 11-15 ×2 more (the census-unit row and this row; 14 rows standing) · 11-16 ×3 (the sdk055 row's trip + two more trip-preempts; 15 rows standing) · 11-17 (the boundary-494 pairing row; 15 rows standing) · 11-20 (m3 delta lane: trip TRIED at 60,694B; the prune-to-15 lever EMPTY — 13 rows standing, 14 after the append) · 11-20 again (the SAME lane's body compaction + census drift fixes, DISCHARGED: 62,101→53,304B — the census table compacted to shape facts with provenance moved to git history, and the drift fixes landed with it: riir-dao + seal-std shapes corrected to flat numbered FILES (disk had `.docs/001_*.md` each), 4 missing workspace rows added (riir-ledger README-only / zed flat 001–018+033–038 / riir-grams no-.docs / seal-sprints no-.docs), mmorpg-remaster + jup-tool retirement flags added per katgpt-rs AGENTS.md, the stale riir-train `main` branch lines in Step 4 + Step 6 corrected to the table's develop row) · 11-25 (the auto-prompt continuation lane: trip at 60,063B; 22→15, + this append = 16 standing) · 11-25 again (the 602nd pairing's trip: 60,188B entry; 18→15 rows standing)

| Date | Scope | Verdict | Record (primary fix commits) |
| 2026-11-25 | delta unit, ZERO book writes (amo, the sprint!020 mist-bars leftover-sweep lane's housekeeping pass — the boundary-603 pairing row): the pass landed NO book surface — its acts are the fresh amo gate run (evidence, not a write: exit 1 = the live sprint!131 mirror-family enumeration, adjudicated NOT-ROT in the boundary 603rd), 2 holder mails + 1 timed wake (mail, not a write), the queue sweep (reads); the sprint!020 wave-3 seal-remake landing (5dd5fcc8 → ff43e841) is the LANDING LANE'S own close-out (condition 8, lease live, nudged 01a210a6 — deferred to the fixing session, never this lane's write); no numbered files moved (zero allocations; numbering gate not owed); 60KB trip NOT tripped (53.7KB entry + this row ≈ 55.3KB); benches NOT owed (md + reads + mails, zero perf claims) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (m3, zed-25c6d37b@m3 — the boundary-602 pairing row; the plan-077 housekeeping continuation's doc pass): the pass landed NO book surface — its acts are the fresh full contract gate (evidence, not a write), 3 Protocol-H ffs into sibling checkouts (riir-train plan-504 wave / riir-auth winhello wave / katgpt-rs upstream skill rows — all SELF-RECORDING by their own lanes: train's PIN mds + fixtures, auth's HISTORY +3, the skill rows themselves), 2 route mails + 1 probe-update mail (mail, not a write), the queue sweep (reads); no numbered files moved (zero allocations; numbering gate not owed); 60KB trip NOT tripped (57.5KB entry + this row ≈ 59KB — the NEXT append trips, prune-to-15 owed then); benches NOT owed (md + reads + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit + data/fixture writes (amo, zed-52ad70c3@amo — the boundary-601 pairing row; the plan-504 engagement pass's doc surface): riir-train landed T2+T2b (a9d3d9a8): the two PIN mds (EVAL_SUITE_PIN.md = the frozen eval composition, written BEFORE any rung per the issue's law; SOURCE_PIN.md = the provenance authority with hash-verified source rows + the measured coverage table) + the eval fixtures (data/hnet_probe/eval/ — the frozen reads, blake3-manifested) + the two uv scripts; no .docs book surface owed (the plan + issue files ARE the records; the data dir carries no book convention); no numbered files moved (zero allocations; numbering gate PASSED fresh, claim lint clean ×2); benches NOT owed (python data tooling + JSON fixtures, zero perf claims); 60KB trip NOT tripped (58.3KB entry + this row ≈ 59.7KB — the NEXT append trips, prune-to-15 owed then) | numbering gate PASSED fresh (riir-train 888 files) | riir-train a9d3d9a8 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (amo, zed-52ad70c3@amo — the boundary-600 pairing row [numbered 600 at the collision with the m3 lane's 599th]; the fleet-queue-sweep continuation's doc pass): the pass landed NO book surface — its acts are Protocol H ffs into sibling checkouts (seal-sprints 64bf579, riir-dapps 29a7c86 — SELF-RECORDING by their own lanes, never this lane's writes), the orphan-daemon kill (runtime debris, not a write), the queue sweep (reads); no numbered files moved (zero allocations; numbering gate not owed); 60KB trip NOT tripped (57.5KB entry + this row ≈ 58.7KB); benches NOT owed (md + reads, zero perf claims) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (m3, zed-adf0c34a@m3 — the boundary-599 pairing row [RENUMBERED 595→599 with its boundary row at the collision with the amo lane's 595th]; the leftover-sweep continuation's doc pass): the pass landed NO book surface — its acts are the plan-095 record correction (seal-sprints 7af5ff5 — the plan md IS the record, lease lint clean at the edit) + the rig resubmission (mail/rig routing, not a write); the sibling sprint!117/128 landing (64bf579) is SELF-RECORDING by its own lane; skill rows self-record (this + the boundary 599th); no numbered files moved (zero allocations; numbering gate not owed); 60KB trip NOT tripped (58.3KB entry + their rows merged ≈ 59KB, checked post-resolution); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | seal-sprints 7af5ff5 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit + ONE issue-file status pass (amo, zed-e0e8d7f0@amo — the boundary-598 pairing row; the i656 expired-lease resolution's doc pass): the resolution IS self-recording (riir-train 379e057c+78ddd052 — the lease fold + the commit-or-revert verdict ride the issue file's own Status/Lease-closed lines, claim lint clean + numbering gate PASSED fresh at the landing, 888 files); no book surface owed (no feature/verdict/status change in any .docs book; the compute_multiplier delta was fmt-only); no numbered files moved (zero allocations); 60KB trip NOT tripped (55.8KB entry + this row ≈ 57.5KB); benches NOT owed (fmt-only + md, zero perf claims) | numbering gate PASSED fresh on riir-train (exit 0, at 379e057c) | riir-train 379e057c+78ddd052 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (amo, zed-b2bd6193@amo — the boundary-597 pairing row; the i1220/709 continuation lane's post-handoff housekeeping pass): the lane's acts are reads + the fresh 01:00Z wake (01a20fae, mail not a write) + this row — the 709 T4 bench doc allocates at the wake per the runbook (in-flight, the claimant lane's or this lane's successor act); the cell queue + the probe landing are SELF-RECORDING (riir-train 54a84783 status line + 9073d489 commit message, the 549th precedent); no book surface owed (no feature/verdict/status change anywhere); no numbered files moved (zero allocations; numbering gate not owed by this lane); 60KB trip FOUND at entry (58,111B / 19 rows) → prune-to-15 landed alongside (19→15, + this append = 16 standing, 54.9KB post); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (amo, zed-e0e8d7f0@amo — the boundary-595 pairing row; the 709-lane thread-family successor's housekeeping pass): the lane's landings are SELF-RECORDING (riir-train 54a84783+a9ea14da — the issue-709 status update IS the record; the T4 bench doc allocates at the 01:00Z wake per the runbook, in-flight until then, the 549th precedent; the 9073d489 example landing was already adjudicated self-recording by the 593rd's delta); no book surface owed (no feature/verdict/status change anywhere); no numbered files moved (zero allocations; numbering gate not owed by this lane — riir-train's gate PASSED at the lane's own landing a9ea14da); 60KB trip NOT tripped (57.5KB entry + this row ≈ 59KB — the next append likely trips; prune-to-15 owed then); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | riir-train 54a84783 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (m3, zed-adf0c34a@m3 — the boundary-594 pairing row; the leftover-sweep lane's doc pass): the lane landed NO book surface — its acts were the wp3p2 worktree rm (recorded in the sweep close-out, no doc surface) + the amo heads-up mail + the two timed wakes (mail, not writes); the sibling landings this hour are SELF-RECORDING (refine 52a94c540 Batch-414 forward claim row in the mining snapshot, the 549th precedent); scope-repo delta: all 5 fetched repos HEAD == origin, seal-sprints/seal-remake plans adjudicated in the 594th boundary row — zero doc writes owed; no numbered files moved (zero allocations; numbering gate not owed by this lane); 60KB trip NOT tripped (56,642B entry + this row ≈ 57.5KB); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (m3, zed-bcf297c1@m3 — the boundary-592 pairing row; the sprint!074-close continuation's successor lane's doc pass): the pass landed NO book surface — the two skill rows (this + the boundary 592nd) ARE the records, self-recording; the delta check: fetch_contract_repos 34/34 all-current zero-moved (no unseen upstream landings; recent landings all self-recording — the predecessor's ten-repo linkcheck row landed 11-25, refine AGENTS count-web current at B413), zero doc writes owed; no numbered files moved (zero allocations; numbering gates not owed by this lane); 60KB trip FOUND on this file (59,739B entry, 21 rows) → prune-to-15 landed alongside (21→15, + this append = 16 standing); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit + TEN-REPO linkcheck sweep (m3, zed-314b606a@m3 — the sprint!074-close continuation's housekeeping lane): 101 dangling findings → 59 fixed across katgpt-rs/riir-ai/riir-chain/riir-infer/riir-neuron-db/riir-refine/riir-reflex/riir-rethink/riir-train/seal-sprints (41 .md files, R3 de-links on removed issues 031/037/040/106/1165/1167/1176/1189/119/1201/1214/1217-1218/1222/1230/1235/1259/065/1117/558/221/plan-019/plan-302 + R1/R2 repoints: missing-`../` benches 521/714/1235-1238/1215/plan-636, over-deep train 503/508 + league 1250, TASKS.md plan-074 stray `../`); census residue 44 = zed's 42 fixture/example links (EXCLUDED by design — intentional content) + refine 378a's 2 `file:///` owner-scratch N/A; the sweep is the FIRST linkcheck run in the recent rows — this week's issue removals left exactly the predicted residue; commits per repo via sync_commit.sh partial form (sibling WIP in riir-ai ×24 .rs + rethink 108/067 untouched; 3 non-FF pushes rebased clean — zero allocations so no dual-allocation risk); no numbered files moved (link fixes only, zero allocations); 60KB trip NOT tripped (58.1KB entry + this row ≈ 59.5KB — the next append likely trips; prune-to-15 owed then); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (link fixes only, zero allocations) | katgpt-rs 92fd134e8+6b413b0f3 + riir-ai 55f6122d1 + riir-chain 529b71f5 + riir-infer 015e920f + riir-neuron-db 0043953 + riir-refine b7bf9867b + riir-reflex 6127989b + riir-rethink 64bb2b5 + riir-train b2fa3cc52 + seal-sprints 8d758e3 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit + ONE issue-file removal (amo, zed-aa5dd964@amo — the boundary-591 pairing row; the plan-485-P4 forwarding lane's doc pass): the removal IS the record (riir-train 5defaed8 — closed issue 692's file removed per the fixed-issue law; the GOAT FAIL verdict + hashes b80ae247/331b5f19/5a023f20 were ALREADY in HISTORY row 590 + Bench 715 at the closure, so nothing new to write — the file was pure residue, the predecessor overflow session's missed teardown step); link check clean (only a prose katgpt-rs-namespace citation in train plan 356, not our file); numbering gate PASSED fresh on riir-train (exit 0, 886 files, claim lint clean over 365); the 1259 closure landed upstream with its own HISTORY row (466fd88c6) — no book surface owed by this lane; no numbered allocations (one removal, zero moves); 60KB trip NOT tripped (57.0KB entry + this row ≈ 58.5KB); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate PASSED fresh on riir-train (exit 0) | riir-train 5defaed8 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (m3, zed-2b94512c@m3 — the boundary-590 pairing row [renumbered 589→590 at the collision with the amo lane's 589th]; the riir-refine lease-pickup continuation's doc pass): the predecessor lane's landing is SELF-RECORDING (riir-refine 400dc0190 — the commit message + the P126 snapshot intake row ARE the record, the 549th precedent; no book surface owed — count-neutral corpus text, zero feature/verdict/status changes anywhere); the pass's acts are reads + the 07:20Z 11-25 mining-gate wake (mail, not a write); no numbered files moved (zero allocations); 60KB trip NOT tripped (bg 50.5KB / ds 55.4KB at entry, rows land ~53/58KB); benches NOT owed (md-class + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | riir-refine 400dc0190 + katgpt-rs (this row, this commit) |
| 2026-11-24 | delta unit, ZERO book writes (amo, zed-3d94a4e1@amo — the boundary-589 pairing row; renumbered 582->589 with its boundary row at the upstream 582nd-588th collision wave): the lane's landings are SELF-RECORDING (riir-infer 28d05caf/e316e559 — the .issues/1220 staging rows ARE the records; the T2/T5 landing records ride the green adjudications per the issue's own plan); no book surface owed (riir-infer flat numbered FILES shape, none touched); no numbered files moved (zero allocations); numbering gate PASSED fresh on riir-infer (exit 0, 270 files, claim lint clean over 49); benches NOT owed (md-only, zero perf claims) | numbering gate PASSED fresh on riir-infer (exit 0) | riir-infer 28d05caf + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit + FOUR issue-file removals (m3, zed-docsync-prune@m3 — the boundary-588 pairing row; the resolved-issue hygiene pass, the rejected-stop continuation): the rm lanes landed — riir-ai 70c471e39 (1257 golfktp-closed c82ae503c, the HISTORY row had already claimed the rm; 1097 closed under (c) 11-24, re-open instrument = seal-sprints sprint!126, new closure row added) + riir-infer bf6e797e (1248 census LANDED + verdict AGREE, ndb Plan 338 correction applied) + riir-infer 13bfc864 (1201 TERMINAL-DEMOTE by its own record, Bench 1253 T3 FAIL; T4/T5 dead with the demote, boxes lagged unflipped); numbering gates PASSED fresh + claim lint clean at each landing (riir-ai 1444 files / riir-infer 270→269); NOT removed after adjudication: 1164 (contract-gated on 932's green, live lease), 1183/1207/1181 (paused/parked/open with re-open triggers — tracking homes), refine 250/218/232/243 (passive accrual / standing checklist / data-gated / event-gated); earlier this session: the doc-sync SKILL.md prune-to-15 (60,063→53,983B, 22→15 rows, 832a8aa57); no numbered files moved (removals only, zero allocations); 60KB trip NOT tripped (ds ~55.6KB after this row); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gates PASSED fresh on riir-ai + riir-infer (exit 0 both, claim lint clean) | riir-ai 70c471e39 + riir-infer bf6e797e+13bfc864 + katgpt-rs 832a8aa57 + katgpt-rs (this row, this commit) |
| 2026-11-25 | delta unit, ZERO book writes (m3, zed-docsync-prune@m3 — the boundary-587 pairing row; the auto-prompt continuation lane picking up the 586th pairing's what-remains): the lane's act is the doc-sync SKILL.md prune-to-15 (60,063B entry, over the file's own 60KB prose line; 22→15 rows standing, + this append = 16; removed rows recoverable via `git log -S` per the maintenance rule; boundary SKILL.md checked 50.6KB, under); the plan-412 mining gate re-checked at pickup — still lease-gated (the 405 lease lapses 11-25T07:05Z, ~12h out at the 19:03Z pass; re-fire then); no numbered files moved (zero allocations); benches NOT owed (md-only + the no-bench-on-a-busy-box law) | numbering gate not owed (skill rows only) | katgpt-rs (this row, this commit) |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
