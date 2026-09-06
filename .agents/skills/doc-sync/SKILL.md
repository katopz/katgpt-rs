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
>
> The header carried a hand-typed count (**18**, measured 2026-09-01) while the
> workspace moved to 16 live repos on 2026-09-04 (three to `git/obsolete/`) — the
> same rot class the boundary-guard skill de-counted in its 21st run. The census
> below is a snapshot; the one-liner derives the membership.

> The header said *"14 as of 2026-08-28"* over a table of **12** rows until
> 2026-09-01 — wrong twice, and six repos had no row at all, so a sync run that
> walked this table skipped them silently. Don't re-type the count; derive the set
> the same way the boundary gate does:
>
> ```bash
> cd /Users/katopz/git && for d in */; do
>   [ -f "$d/BOUNDARY.md" ] && [ -d "$d/.git" ] && echo "${d%/}"
> done
> ```
>
> Canonical count + product-vs-workspace split: `katgpt-rs/AGENTS.md` §"Repo count".

Each repo has a different doc layout. **Read the repo's `AGENTS.md` first** —
it documents the canonical layout and the numbering discipline.

| Repo | `.docs/` shape | `README.md` | Numbering highwater | Working branch |
|---|---|---|---|---|
| `katgpt-rs` | 10 numbered folders (`01_orientation/` … `10_audits/`), unnumbered files inside. The **public** selling-point book. | Large showcase + feature tables + getting-started. | `.plans/.highwater`, `.issues/.highwater`, `.benchmarks/.highwater`, `.research/.highwater` | `develop` |
| `riir-ai` | **12 numbered folders** (`01_orientation/` … `12_inference/`; `02_inference/` was renumbered to `12_inference/` on 2026-08-08 to resolve a `02_` prefix collision with `02_crates/`). The **private** consolidated selling-point book. | Large showcase + crate table. | same `.highwater` files | `develop` |
| `riir-chain` | **7 numbered folders** (`01_orientation/` … `07_formal_verification/`, reindexed from flat on 2026-08-08), unnumbered files inside. Canonical self-description — the chain owns the truth about the chain; `riir-ai/.docs/07_neuro_symbolic_chain/` is the consumer/fusion view and links here. Two workspace members (`riir-chain` lib + `riir-chaind` daemon). Build surface still lives in `README.md`; FV invariants in `AGENTS.md` + `.proofs/README.md`. | Build commands + feature flags + the wallet/RPC trust surface + the `merkle_root` lesson. | same | `develop` |
| `riir-neuron-db` | **10 numbered folders** (`01_orientation/` … `10_local_kv/`; the 10th was added 2026-07-30 for the Warm tier substrate, Issue 043), unnumbered files inside. Covers all `src/` modules + `examples/`. Matches the `riir-ai/.docs/` format. | What the crate owns + feature gates (default-on / opt-in) + feature→chain mapping + `merkle_root`/`can_freeze` lessons. | same | `develop` |
| `riir-train` | **6 numbered folders** (`01_orientation/` … `06_cross_cutting/`, reindexed from flat on 2026-07-15), unnumbered files inside. Training-method research vault. | Role + sibling layout. | same | `develop` (**flipped from `main` 2026-09-04** — develop created from the `main` tip and made the default branch, the Issue 704 convention; `main` frozen) |
| `riir-game-sdk` | **10 numbered folders** (`01_orientation/` … `10_multiplayer_topology/`; the 10th was added for the two-binary production topology + avatar/game sync facade), unnumbered files inside. Covers all `src/` modules + `examples/`. Matches the `riir-ai/.docs/` / `riir-neuron-db/.docs/` format. | Boundary rule + leaf constraint + spatial canonical + Phase 2/3 status + feature gates. | same | `develop` |
| `riir-mmorpg-examples` | **No `.docs/` folder** — docs live in `AGENTS.md` (extensive: role, topology, plans/issues/benchmarks index, canonical-failure lessons) + `README.md` (status + build commands + env vars) + `.plans/` / `.issues/` / `.benchmarks/` files. POC consumer of `riir-game-sdk`. | Status + build commands + Plan/Issue index. | same | `develop` |
| `riir-clippy` | **12 numbered folders** (`01_orientation/` … `11_domains/` + `12_ane/`; the 12th added for the Apple Neural Engine substrate knowledge — private runtime API, MIL/blob formats, M3 Max findings, the Rust-bridge negative result + working ObjC substrate — riir-ai Issue 726 T0 distillation), unnumbered files inside. The code-healer vault — corpus/drafter/pruner/verify/self-evolve/domains narrative. `AGENTS.md` carries the batch-mining progress notes (the sweep record home for cross-repo clippy heals). | Status + Quick Start + Usage + feature gates. | same | `develop` |
| `riir-unity` | **RETIRED 2026-09-04 → `git/obsolete/` (owner act). Lineage only; do not route work here.** No `.docs/` folder — AGENTS.md-centric (domain boundary, Unity MCP rules, issue log) + `.benchmarks/`. The Unity host; Rust work belongs in riir-viewbridge, so doc-sync here = AGENTS.md issue-log sections + module-map freshness. | Role + boundary + sibling layout. | same | `develop` |
| `riir-viewbridge` | **1 numbered folder** (`.docs/01_orientation/` — `README.md` + `crate_role.md`) + a `.docs/README.md` index; AGENTS.md still carries the workspace layout, boundary rules (latent/raw wall, generated-bindings, catch_unwind) + issue log, and `.benchmarks/` the node GOAT. The Rust FFI side of the Unity bridge. **This row said "No `.docs/` folder" until 2026-09-04** — the folder arrived in the workspace scaffold `2f0257c` (Plan 532 P0 part 2), i.e. a shape change that never came back to this contract, which is the failure the Shape-change contract below exists to prevent. | Role + boundary + build commands. | same | `develop` |
| `seal-remake` | **1 numbered folder** (`.docs/01_orientation/` — `README.md` + `unity_host.md`, from Plan 031 Phase 5) + AGENTS.md/README.md/BOUNDARY.md. The Bevy/wasm viewer half after the Unity host split out. **Row ADDED 2026-09-04** — the repo was enrolled 2026-09-03 and this table never got a row for it, while carrying three retired ones: 18 rows over a 16-repo workspace, wrong in BOTH directions. | Role + boundary + build/run commands + the vessel-texture + present-mode records. | same | `develop` (highwater: `.issues` 005, `.benchmarks` 002) |
| `riir-dapps` | **No `.docs/` folder** — AGENTS.md-centric (the one-way game → dapps → chain invariant, the three-test rule, tiered-durability record) + `.plans/` / `.issues/` / `.benchmarks/`. The settlement-composition layer. | Boundary + build + the `direction_gate` + kat rail status. | same | `develop` |
| `riir-dao` | **No `.docs/` folder** — AGENTS.md-centric (the KAT tokenomics agent: signals → strategy → guard → advisory → commit; the G5 advisory-only verdict) + `.plans/` / `.benchmarks/`. | Boundary + build + the direction gate. | same | `develop` |
| `riir-armageddon` | **RETIRED 2026-09-02 → `git/obsolete/` (owner act). Lineage only; do not route work here.** `.docs/` exists but is EMPTY — AGENTS.md-centric in practice (arena/game-product domain types). Added 2026-09-01 | yes | `.issues` 005, `.plans` 008 | **`main`** — not `develop`; check before branching |
| `riir-auth` | **`.docs/` exists but holds only `.highwater`** — i.e. no docs at all, AGENTS.md-centric in practice (the numbering file was created ahead of the folder's first document). Added 2026-09-01 | yes | `.issues` 002, `.benchmarks` 4, `.plans`/`.docs`/`.research` at 0 | `develop` |
| `riir-burner` | **RETIRED 2026-09-04 → `git/obsolete/` (owner act). Lineage only; do not route work here.** Flat numbered FILES, no folders — `.docs/001_model_verdict.md` … `016_*.md` (7 files; two share 016 — the numbering discipline is not enforced here). Added 2026-09-01 | yes | `.issues` 015, `.plans` 019 | `develop` |
| `riir-deployer` | **2 numbered folders** (`01_orientation/`, `02_runbooks/`) + a `.docs/README.md` index — the smallest numbered shape in the workspace. No `CLAUDE.md`. Added 2026-09-01 | yes | `.issues` 003, `.plans` 002, `.benchmarks` 001 | `develop` |
| `katgpt-web` | **No `.docs/` folder** — AGENTS.md-centric. Added 2026-09-01 | yes | none | **`feat/percepta-arch-diagrams`** — the only repo whose checkout is not on its trunk; sync the branch you find, and say which one in the run log |
| `seal-game-editor` | **NAMED (not numbered) `.docs/` subfolders** — `new-game-schema/`, `registry/`, plus loose `SEALM_ASSETS.md`. Carries `ARCHITECTURE.md` + `DESIGN.md` alongside AGENTS/README, and `ARCHITECTURE.md` is where the internal layering lives (`BOUNDARY.md` covers only the outer edge). Added 2026-09-01 | yes | `.issues` 141, `.plans` 140 | `develop` |

## The sync workflow (per repo)

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

#### riir-neuron-db (10-folder `.docs/` book; 10th added 2026-07-30)
- **README.md**: build surface — feature gates (default-on / transitive / opt-in / per-feature prose sections for promoted primitives) + Formal Verification summary + License. Prose sections are reserved for promoted default-on features; opt-in features get table rows only.
- **`.docs/`**: 10 numbered folders mirroring the `riir-ai/.docs/` format. Drop new docs in the right group folder (by capability: shard substrate / freeze-thaw / consolidation / vessel / specialized / zone / examples / FV / **local-kv Warm tier**), add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point. The `05_secure_vessel/vessel_primitive.md` doc is the restored home of the old `15_vessel.md` (corrected: riir-neuron-db is "this crate", NOT katgpt-rs per Plan 006). The `10_local_kv/` folder covers the `LocalKvStore` + `CommitLevel`/`CommitBatch` + WAL compaction + BM25 (the Warm tier substrate backing per-player state recovery in riir-mmorpg-examples Plan 013, added Issue 043).
- **AGENTS.md**: the FV (Lean 4) invariant table lives here (mirrored in `.proofs/README.md`), NOT in `.docs/`. The `.docs/09_formal_verification/` folder is the narrative overview; `AGENTS.md` is the authoritative invariant table.
- Cross-repo: if a primitive was consumed by `riir-ai` or `riir-chain`, the fusion is documented bidirectionally.

#### riir-train (6-folder `.docs/` book, reindexed 2026-07-15)
- **6 numbered folders** (`01_orientation/` … `06_cross_cutting/`), unnumbered `.md` files inside — mirrors the `riir-ai/.docs/` format.
- Training-method research vault: adapter training, distillation/RL, data filtering, cross-cutting audits.
- `README.md` is minimal — role + sibling layout.
- `main` branch (no `develop`).

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

#### riir-clippy (12-folder `.docs/` book)
- **`.docs/`**: 12 numbered folders mirroring the `riir-ai/.docs/` format — corpus / drafter / pruner / verify / ruliology / examples / benchmarks / lessons / self-evolve / domains / **ANE substrate knowledge** (`12_ane/`, riir-ai Issue 726 T0 distillation). Drop new docs in the right group folder, add one line to that folder's `README.md` index table.
- **`AGENTS.md`**: the batch-mining progress notes + sweep records live here (the cross-repo clippy-heal record home). A landed heal slice in a sibling repo (katgpt-rs, riir-train, riir-ai) gets its progress note in the SAME commit as the heal — a later `doc-sync` run defers to the healing session (never write progress notes for someone else's in-flight sweep).
- **README.md**: Status + Quick Start + Usage + feature gates.

#### riir-unity — RETIRED 2026-09-04 (`git/obsolete/`), lineage only
- **`AGENTS.md`**: domain boundary (no Rust crates here; UPM package is build output; no engine substrate in C#) + the Unity MCP rules + the issue log. Doc-sync = issue-log sections for resolved issues + module-map freshness (the `Packages/com.riir.viewbridge/` population + scene wiring notes).
- The Rust side of any feature lives in `riir-viewbridge` — cross-repo arcs (e.g. Issue 004) document on BOTH sides at arc close.

#### riir-viewbridge (`.docs/01_orientation/` + an AGENTS.md-centric remainder)
- **`AGENTS.md`**: workspace layout (core/derive/abi/xtask) + boundary rules (latent/raw wall, generated-bindings rule, catch_unwind) + the issue log.
- **`.benchmarks/`**: GOAT records (e.g. Bench 002 node GOAT). Doc-sync = issue-log resolution entries + benchmark cross-refs.

### Step 5 — Verify

- **No broken links**: every `[...](.plans/NNN_*.md)` must point to a file that exists.
- **No stale numbers**: if a README entry says "ratio 0.01" but the benchmark says "0.27", the README is wrong — update it.
- **Numbering discipline**: `.highwater` files must be bumped when new plans/issues land.
- **Honesty**: a GOAT FAIL stays a GOAT FAIL in the docs. A "stays opt-in" primitive is documented as opt-in with the reason. Never upgrade a verdict in the docs without the benchmark to back it.

### Step 6 — Commit

Per the global `AGENTS.md` rule: **always commit at task completion**. Use `docs:`
prefix. Stay on the repo's working branch (`develop` for most, `main` for
riir-train). Do not push.

```sh
git add .docs/ README.md .plans/ .issues/ .benchmarks/ .research/
git commit -m "docs: sync .docs + README with recent plans (NNN, NNN, NNN)"
```

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
folder to any repo that already has numbered folders — measured 2026-09-04 as
**10 of the 16**: `katgpt-rs`, `riir-ai`, `riir-chain`, `riir-clippy`,
`riir-deployer`, `riir-game-sdk`, `riir-neuron-db`, `riir-train`,
`riir-viewbridge`, `seal-remake`. **Or** that gives a `.docs/` to one of the
four with none (`katgpt-web`, `riir-dao`, `riir-dapps`,
`riir-mmorpg-examples`) or the one whose `.docs/` holds no document
(`riir-auth`) — creating the folder is itself a shape change and requires this
contract. Derive the split rather than reading it here (`ls */.docs`); the
previous version of this trigger named `riir-viewbridge` as having none while
its `.docs/01_orientation/` had shipped in the repo's own scaffold commit.

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
- **Do not** impose a `.docs/` shape that differs from the repo's existing convention — respect the shape you find. **Re-measured 2026-09-04 over the live 16** (three repos retired to `git/obsolete/` the same day; the previous text measured 18 and named two of them as routing targets): **10** numbered folders (katgpt-rs 10, riir-ai 12, riir-chain 7, riir-clippy 12, riir-deployer 2, riir-game-sdk 10, riir-neuron-db 10, riir-train 6, riir-viewbridge 1, seal-remake 1), **4** with no `.docs/` at all (katgpt-web, riir-dao, riir-dapps, riir-mmorpg-examples), **1** whose `.docs/` holds no document (riir-auth — only `.highwater`, which reads as a shape and is not one), **1** NAMED subfolders (seal-game-editor). The **flat numbered FILES** shape left the workspace with `riir-burner`. Don't re-type this census: the one-liner in the trigger above derives it, and every hand-written version of it in this file's history has been wrong within days — the previous one said "8 numbered / 8 AGENTS.md-centric" over a table whose membership differed from the workspace in BOTH directions. A shape change is a deliberate, committed decision governed by the **Shape-change contract** above.
- **Do not** renumber existing docs — the numbering discipline is monotonic and never reused.
- **Do not** document trivial mechanical commits (lockfile bumps, clippy fixes) unless they close a tracked issue.


## Standing lessons (distilled from the run log — the load-bearing process rules)

- **Per-file logs + pickaxe are the reliable baseline tools.** `git log -- .docs README.md AGENTS.md` once reported `5a1330b` as game-sdk's newest doc-touching commit while `git log -- AGENTS.md` + `git log -S '<landed-string>' -- <file>` proved `1977d83` (two days newer) had touched AGENTS.md. Always re-verify a suspicious baseline per-file before declaring a range clean.
- **Narrow producer-side syncs leave holes the baseline heuristic can't see.** A narrow sync fixes its own feature and skips everything else, so the strict `baseline..HEAD` range can be empty while the book still misses older landings. Gate runs GREP the book for recently-landed headline features; don't just trust the range. The inverse gap exists too: code landing to MATCH an already-documented row (mmorpg `?transport=local`) is invisible to any baseline heuristic — nothing to fix, but don't declare a gap either.
- **Grep extraction regex must include digits.** `^[a-z_]+ =` over `[features]` silently dropped `avatar_sync_ed25519` and `game_sync_p2p` — nearly filed phantom "documented-but-dead feature" findings. Use `^[a-z0-9_]+ =`.
- **The staged-index check must GATE the commit, not precede it.** One run saw two sibling-staged files in `git diff --cached --name-only` output and then sailed past them because the check was `;`-chained ahead of `git add` + `git commit` — the sibling's WIP landed inside the run-log commit (benign outcome, luck not process). The correct form: run the check as its OWN command and read it, or use `git commit -- <paths>` (a partial commit builds a temporary index from HEAD + the named paths, leaving anyone else's staged hunks intact — the safe form on shared checkouts). `scripts/staged_set_audit.py` (katgpt-rs) reports the same signal class pre-commit.
- **A doc claim of determinism is a claim about a proof.** When a fix refutes the proof (the ndb Bm25 tie-truncation class), grep the book for the CLAIM, not just for coverage of the fix — the fix landed with in-source comments only and the book kept asserting the falsified invariant.
- **False positives: grep the repo's OWN `.benchmarks/` before declaring coverage.** Different repos (and even different series in one repo) reuse bench numbers — the overview's only "Bench 025" hit was a different numbering series.
- **Broken links: prose citations survive file removal, markdown links do not.** The noise-reduction rule removes record files but no link-fix pass followed, so every closed issue left `](.issues/NNN…)` links dangling. Standing tools (committed in katgpt-rs `tools/`): `python3 tools/linkcheck_sweep.py` (census) + `python3 tools/link_fix.py <repo>` (auto R1-repoint / R3-delink), re-sweep to verify, commit pathspec'd `.md` only. Guards baked into the tools: **absent-repo** (a link into a workspace repo not checked out on the running box is UNVERIFIABLE, not broken — `exists()` is box-relative), **backtick** (link text already code-marked is emitted unwrapped), **CRLF** (`Path.write_text` on Windows translates LF→CRLF — restore per-file EOL to the HEAD convention before committing), and a markdown link split across lines is invisible to one-line fix regexes (NOMATCH → manual two-line edit).
- **Never cite "Plan/Research NNN" without naming the repo** when the number exists in more than one namespace — and never link a path you haven't verified (`ls` it first; per-repo numbering namespaces collide constantly).
- **Classification against dirty trees uses `git show <rev>:` content, never the working tree.** Sync remotes first; `git show -1 <sha>` mis-parses (the `-1` overrides the sha — pass the sha alone). The reliable per-file baseline is the 4-surface set (AGENTS.md / README.md / `.docs/` / the feature file) + a since-count.
- **"0 passed" in a baseline run is the reliable gate detector** — grep on attribute FORMS lies (whole-file `#![cfg]` gates sit below doc-comment headers, mod-level `cfg(any(feature…))` compiles empty under default).

## Run log (compact — full narratives live in git history)

Each row's durable record is the named `docs:`/fix commit(s) in the repo it
touched, plus this file's own history
(`git log -p -- .agents/skills/doc-sync/SKILL.md` — rows carried full
narratives until the 2026-09-05 compaction; `git log -S '<date>' -- <this
file>` recovers any of them).

| Date | Scope | Verdict | Record (primary fix commits) |
|---|---|---|---|
| 2026-09-06 | ffi-dissolution book sweep (35th boundary run's companion; the dissolution landed 12:28-12:29 with its own partial doc-sync — overview/02_crates/08_wasm/AGENTS/BOUNDARY done in-landing; this run swept the MISSED surfaces workspace-wide via a `riir-ffi` grep over the 4 book surfaces of 14 repos) | 4 repos had gaps, 7 clean: **riir-ai** 4 surfaces (README crate table '⚠️ Blocked' + composition count; architecture.md Mermaid FFI node (+ VIZ/GMTOOL EXTERNAL annotations, same staleness class); 03_pillars T3/T4 + risk row; 10_browser/latent_state_framework Status/Byte-Bridge/feature-table/hot-swap/References — incl. TWO PRE-EXISTING broken links to riir-engine/src/wasm_bridge/memory_layout.rs, the canonical file moved to riir-wasm + engine re-exports); **riir-train** sibling-layout line; **katgpt-rs** chunked_content_store consumer paths (overview row + 03_memory section incl. broken link to the e2e test, now riir-chain/tests); chain/game-sdk/mmorpg/auth/ndb/dapps/deployer/dao/viewbridge/seal-remake CLEAN (chain books already dissolution-aware, game-sdk fixed its own viz ref `b21cc93`). En-route find recorded-not-fixed: riir-wasm latent_sidecar.rs in-source doc comments still say 'wasm_compute feature' — 570's own residue (their code file; the feature was collapsed, Plan 570 L58) | ai `d500c4a31` · train `b0fec904` · kat `030a0137` |
| 2026-09-06 | 34th-run handoff unit (post-morning-sweep window; survey of all 13 checked-out repos) | CLEAN — no book gaps: post-morning landings all self-documenting or active-lane-owned (ai `d3cb10292` 851-prewarm + `e479e30a7` 846 header — league lane's own docs; clippy `97639a4` unit-8 record + `2a60e92` Issue-072 challenger — NaN agent's own; train `e8c7e1b7` Research 450 docs; chain `8b5d371e` tests/scripts-only (assessed at the morning row); deployer `b88932e`/`73159a0` T4.4a/T2.0 — AGENTS documents both; dapps `dd6261e` — Issue-013 AGENTS row landed with the fix). One note carried to the boundary-guard 34th row instead: the riir-chain→riir-ffi dev-dep now dangles against the ffi lane's deleted working-tree dir — their landing-window concern, not a docs gap | — (no commit; this row + the boundary-guard 34th row in the same commit) |
| 2026-09-06 | idle-loop delta check (post quiet-sweep window; active lanes untouched: NaN sweep in katgpt-rs, ffi sidecar in riir-ai, 098 parity across seal-remake/mmorpg/game-sdk, 851 monitor, riir-clippy mining held-4090) | CLEAN — no book gaps: landings since the last rows are self-documenting (chain `8b5d371e` stat = tests/scripts only; clippy `2a60e92`/`9f113fe` docs; ai league/851 docs; mmorpg `89232e3` closed the 092/097 sections; katgpt-rs `6afbbde1`/`422588a9` docs+highwater); seal-remake `eb1df32c` (098 S1) + the 098 issue = the active parity agent's lane, deferred by the never-write-someone-else's-sweep rule. 1 hygiene fix executed in quiet riir-train instead: 514 resolved row added + file removed, 518 recorded-removal executed (AGENTS had claimed it removed since `3317f3b4`; no deletion ever landed on develop) | kat-rs this commit · train `32f72e77` |
| 2026-09-06 | quiet-repo 8-repo sweep via 2 parallel subagents (chain/ndb/dapps/dao + deployer/auth/viewbridge/sdk; kat/ai/clippy/train/seal/mmorpg deferred — 6 active sibling lanes own them); safety gates passed all 8 (fetch + clean + HEAD==origin) | 4 CLEAN (chain, deployer, auth, viewbridge) + 4 gaps: ndb Issue-610 resolved-entry missing from the AGENTS.md issue log (the noise-reduction convention's one hole); dapps AGENTS.md kat-service wire contract missing the `/sync` `recorded` field (`9396402`); dao treasury-floor sim mirror (`bc09019`) + README status pointer; sdk overview flag-table missing `panel_trace` row + stale `clr_collective_threat` row missing the Issue-874 D1 forward record | ndb `0e88705` · dapps `214e652` · dao `60d17a2` · sdk `271ecbc` |
| 2026-09-06 | katgpt-rs (post-Plan-585-closeout window; 513 sibling lane live in the shared checkout — partial-commit form) | 1 stale status + 3 catalog gaps: plan-585 header ADDENDUM OPEN→CLOSED; catalog §91 usage_rate_eviction (Bench 697 MIXED GOAT), §92 contrastive_scope (DEFAULT-ON 2026-08-20 — the doc-sync default-on-promotion trigger), §93 anti_common_mode+anchored_reach RVM pair (consumer facts verified against riir-ai source: anti_common_mode's live consumer is riir-poc 696-T3 PoC, NOT the Cargo-comment's named future consumer); README usage_rate_eviction row | kat `adbdc943` |
| 2026-09-06 | riir-clippy post-Issue-071 window | CLEAN — no gaps: the fix landed its own docs (`9ebac59` carries the `.docs/11_domains/docker.md` header + degradation § + tier-2 agreement row AND the AGENTS.md issue-071 row); book grep verified (registry-blocked / `classify()` / `Check complete` all present); 12-folder `.docs/` shape unchanged; no landed-but-undocumented commits (HEAD == origin/develop, clean tree). No commit. | — |
| 2026-09-06 | riir-ai league-doc lane (851-wait window; window scan of the other 14 repos found all landings self-documented or sibling-lane-owned — 513/097/030/batch-116 arcs carried their own rows) | 2 fixes: the 09-06 DESTROYED league-doc edit re-derived + re-landed from the incident marker (fa-vec upstream-completion watch pulled out of the giant M3 row cell into a standalone paragraph beside the PrismML revival note — content verbatim, placement restored; marker file removed on re-derivation) + the same-day watcher coverage-gap extension (ds4 + oMLX joined, 6 sources) recorded on the harness lane row | ai `7614de21c` |
| 2026-09-05 eve | 15-repo sweep | 4 gaps: kat `incidence_algebra` book sync, sdk `stealth_agreement_contagion` row, dapps `item_nft`/Issue-122 bullet, clippy Issue-067 digest; riir-ai deferred (69 dirty, live arc) | kat `dbc82a0f` · sdk `8596e6e` · dapps `c02f3b2` · clippy `8c16e88` |
| 2026-09-05 | post-substrate-first-audit 15-repo sweep | 1 gap: game-sdk scenelab book sync (Proposal 029/030 six surfaces + riir-e2e enumeration rot) | sdk `8839f0a` |
| 2026-09-05 | clippy rust_perf ledger catch-up (Batch-112 handoff window) | 1 gap: highlights table + category ledger + corpus.rs module header ~9 batches stale behind their own prose | clippy `dfd069e` |
| 2026-09-04 night | workspace markdown link-integrity sweep (16 repos; `tools/linkcheck_sweep.py` + `tools/link_fix.py` landed) | 116 links repaired in 8 repos; 650 deferred in 4 sibling-hot repos (static debt, tools are the tracker) | ndb/game-sdk/chain/dapps/dao/deployer/auth/seal link commits + `tools/` |
| 2026-09-04 | 4090-box link sweep of the 4 hot repos + re-check | 627 links repaired: train 158→0, clippy 80→3, ai 270→34, kat 155→1, seal 1→0; remainder = absent-repo refs | train `8d3ba1a9` · clippy `8e94fd7` · ai `cac3f7d8e` · kat `f4e17beb` · seal `2b2f644` |
| 2026-09-04 morn | mmorpg link sweep (cooled tree; `git commit -- <paths>` partial-commit form) | 37 links repaired, repo at zero | mmorpg `a7db1b1` |
| 2026-09-04 eve | post-S3 quiet sweep + quarterly goat-audit | ALL CLEAN; goat-audit filed ai Issue 867 + kat `contrastive_scope` comment-drift fix | kat `75ff04cc` |
| 2026-09-04 | post-853 sweep | 2 gaps: sdk `deliberation_cadence` + swarm.md section; dapps kat-service `deploy.yaml` paragraph | sdk `d203daf` · dapps `e4d8963` |
| 2026-09-03 eve | sdk 023/024 arc-close window | 1 gap: broken-link class (bench 027 → removed Issue-023 file repointed to Bench 029) | sdk `8d0a2fa` |
| 2026-09-03 | percentile per-site discharge window | 2 gaps: auth CI section (the morning sweep's own miss); chain gate-coverage 13th row (engram GOAT) | auth `38678b2` · chain `ffb39061` |
| 2026-09-03 | post-P18/706 fan-out sweep (6 repos) | 6 book-silent CI landings documented; staged-sweep incident (benign, sibling validated) | unity `edb211f` · viewbridge `1d056f2` · dao `d1e73d6` · chain `825fb9a9` · ndb `c3638a0` · seal `ef17d68` |
| 2026-09-02 | post-Issue-608 ndb window | 1 gap: Bm25 false-invariant claim (falsified proof grep, not coverage) | ndb `36b61ea` |
| 2026-09-02 ×3 | T0.3 + cluster charter + m32-league + viewbridge arcs | 5 gaps: chain `pay_request` page (+ wrong-citation fix in the T0.3 doc itself), sdk cluster charter terms, auth `derive_all_account_slots` + README Status, viewbridge multi-game `cargo unity`, unity per-game policy | chain `844de308` · sdk `d1a42b6` · auth `8b74651` · viewbridge `703c629` · unity `31f217c` |
| 2026-09-02 | post-KAT-landing sweep | 2 gaps: clippy kat-011 T0.2 onboarding + corpus-stats provenance; dapps mint-anchor Status bullet | clippy `7217f39` · dapps `810598a` |
| 2026-09-01 | handoff decision-item sweep | 1 gap: kat `float_order` (headline primitive, zero book coverage) | kat `b06bd2ba` |
| 2026-08-31 | riir-chain (+ full 13-repo sweep) | 1 gap: Plan-044 bounded tx-log retention section (README had it, the book didn't) | chain `a44932d9` |
| 2026-08-30 | riir-train + ai broken-link sweep | 2 gaps: runbook tense rot + 06-folder coverage line; 3 broken md links to removed train Issue-493 repointed | train `1fbc88ca` · ai `d6f622466` |
| 2026-08-28 ×2 | train 490-T1–T3 + 6-repo window sweep | 7 gaps: train edge_lora incremental page; ndb anchor_root; sdk hint_regret + anchor_root re-exports; chain Warm-tier blockquote; dao 2 status bullets; mmorpg stale deployer note | train `f14c0119` + ndb/sdk/chain/dao/mmorpg commits |
| 2026-08-27 ×2 | train dispatch/decision-item units | CLEAN / deferred-by-rule; no commit | — |
| 2026-08-26 ×3 | train adjudication units + kat Plans-575/576/577 handoff | train `syre` adjudication annotation; kat 4 gaps (catalog §87/§88, README showcase ×2, negative_results §41, highwater 682→683 regression fixed) | train annotation · kat handoff commit |
| 2026-08-24 | riir-train scoped unit | CLEAN / deferred-by-rule; no commit | — |
| 2026-08-23 | full 11-repo gate | 4 gaps: kat `verdict_margin`+`gaussianity_probe` (3 surfaces each), ai `social_pressure`, train edge_lora dist-guard page + stale location fix, sdk `art_vessel_impl` | kat `ad221606` · ai `fdcc462f2` · train `77d23662` · sdk `6766191` |
| 2026-08-22 | idle-queue unit | 2 gaps: sdk e2e drills + spectral/gm_asset/vessel_pack bullets (false-positive catch: repo-own Bench 025 vs mmorpg's); kat spectral_pencil 3 flag surfaces | sdk `a0ddb83` · kat `6575ef21` |
| 2026-08-19 ×2 | full 10-repo gate + verification scan | 5 gaps: kat §81-83 + stale untracked drafts, ai Issue-725 untracked commit, sdk target_coordination, ndb storage-challenger negatives, viewbridge adapter surface; scan all clean | kat `f8f194e0` · ai `a57c3d394` · sdk `33ad206` · ndb `ba386e8` · viewbridge `8bb0a04` |
| 2026-08-18 | full 10-repo gate | 3 gaps: sdk forage forwards, ndb ED-challenge negative, mmorpg Plan-539 section + viewbridge-node stub; 3 deferred-by-rule | sdk `7766fba` · ndb `67c1715` · mmorpg `894a703` |
| 2026-08-17 ×4 | idle-queue units | 5-feature kat sync (cluster_lm_head/bigram_markov/switch_cost/ugc_schedule/gpu_inference), sdk stealth member + entity_sync + Plan-022 forwards, kat Bench-688 verdict propagation, sdk karma_history extension | kat `75a634ac` · sdk `3687de6` · kat `fc802ff6` · sdk `b885d38` |
| 2026-08-16 | idle-queue sweep | 1 gap: sdk vocabulary enumeration (`types`/`social`) | sdk `40f0971` |
| 2026-08-15 ×2 | full quarterly gate + train follow-up | 3 gaps: sdk coverage_curiosity + 15 feature rows, ai swarm-family cluster, mmorpg Issue-667 section; train flashmemory + lm_head pages | sdk `1460448` · ai `5aea89969` · mmorpg `6cbe97e` · train docs commit |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
