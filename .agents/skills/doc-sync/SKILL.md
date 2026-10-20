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
> same rot class the boundary-guard skill de-counted in its 21st run. It is 18
> again since 2026-09-10 (`riir-esp32` moved out of riir-chain 2026-09-06;
> `riir-kat` spun out of riir-clippy 2026-09-10) — same count, different
> membership, which is exactly why the count is derived and the census below is
> a snapshot; the one-liner derives the membership.

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
| `riir-neuron-db` | **11 numbered folders** (`01_orientation/` … `11_cli/`; the 11th added 2026-09-09 for the `ndb` CLI — Plan 327, Warm-tier CRUD + identity login + multi-node sync), unnumbered files inside. Covers all `src/` modules + `examples/`. Matches the `riir-ai/.docs/` format. | What the crate owns + feature gates (default-on / opt-in) + feature→chain mapping + `merkle_root`/`can_freeze` lessons. | same | `develop` |
| `riir-train` | **6 numbered folders** (`01_orientation/` … `06_cross_cutting/`, reindexed from flat on 2026-07-15), unnumbered files inside. Training-method research vault. | Role + sibling layout. | same | `develop` (**flipped from `main` 2026-09-04** — develop created from the `main` tip and made the default branch, the Issue 704 convention; `main` frozen) |
| `riir-game-sdk` | **10 numbered folders** (`01_orientation/` … `10_multiplayer_topology/`; the 10th was added for the two-binary production topology + avatar/game sync facade), unnumbered files inside. Covers all `src/` modules + `examples/`. Matches the `riir-ai/.docs/` / `riir-neuron-db/.docs/` format. | Boundary rule + leaf constraint + spatial canonical + Phase 2/3 status + feature gates. | same | `develop` |
| `riir-mmorpg-examples` | **No `.docs/` folder** — docs live in `AGENTS.md` (extensive: role, topology, plans/issues/benchmarks index, canonical-failure lessons) + `README.md` (status + build commands + env vars) + `.plans/` / `.issues/` / `.benchmarks/` files. POC consumer of `riir-game-sdk`. | Status + build commands + Plan/Issue index. | same | `develop` |
| `riir-refine` | **12 numbered folders** (`01_orientation/` … `11_domains/` + `12_ane/`; the 12th added for the Apple Neural Engine substrate knowledge — private runtime API, MIL/blob formats, M3 Max findings, the Rust-bridge negative result + working ObjC substrate — riir-ai Issue 726 T0 distillation), unnumbered files inside. The code-healer vault — corpus/drafter/pruner/verify/self-evolve/domains narrative. `AGENTS.md` carries the batch-mining progress notes (the sweep record home for cross-repo clippy heals). | Status + Quick Start + Usage + feature gates. | same | `develop` |
| `riir-unity` | **RETIRED 2026-09-04 → `git/obsolete/` (owner act). Lineage only; do not route work here.** No `.docs/` folder — AGENTS.md-centric (domain boundary, Unity MCP rules, issue log) + `.benchmarks/`. The Unity host; Rust work belongs in riir-viewbridge, so doc-sync here = AGENTS.md issue-log sections + module-map freshness. | Role + boundary + sibling layout. | same | `develop` |
| `riir-viewbridge` | **1 numbered folder** (`.docs/01_orientation/` — `README.md` + `crate_role.md`) + a `.docs/README.md` index; AGENTS.md still carries the workspace layout, boundary rules (latent/raw wall, generated-bindings, catch_unwind) + issue log, and `.benchmarks/` the node GOAT. The Rust FFI side of the Unity bridge. **This row said "No `.docs/` folder" until 2026-09-04** — the folder arrived in the workspace scaffold `2f0257c` (Plan 532 P0 part 2), i.e. a shape change that never came back to this contract, which is the failure the Shape-change contract below exists to prevent. | Role + boundary + build commands. | same | `develop` |
| `mmorpg-remake` | **1 numbered folder** (`.docs/01_orientation/` — `README.md` + `unity_host.md`, from Plan 031 Phase 5) + AGENTS.md/README.md/BOUNDARY.md. The Bevy/wasm viewer half after the Unity host split out. **Row ADDED 2026-09-04** — the repo was enrolled 2026-09-03 and this table never got a row for it, while carrying three retired ones: 18 rows over a 16-repo workspace, wrong in BOTH directions. | Role + boundary + build/run commands + the vessel-texture + present-mode records. | same | `develop` (highwater: `.issues` 005, `.benchmarks` 002) |
| `riir-dapps` | **1 numbered folder** (`.docs/11_kat_service/`, numbered to MIRROR the kat-service group elsewhere — there is no 01–10 here, so don't read the prefix as a tenth sibling) holding dated evidence bundles (`2026-09-10_domain_swap/`: a README narrative + 5 manifest/health JSONs), plus the AGENTS.md-centric remainder (the one-way game → dapps → chain invariant, the three-test rule, tiered-durability record) + `.plans/` / `.issues/` / `.benchmarks/`. The settlement-composition layer. **This row said "No `.docs/` folder" until 2026-09-11** — the folder landed tracked in `eb85243` (2026-09-10, Plans 023+024, cited from AGENTS.md §Status) and never came back to this contract: the Shape-change contract's exact failure, caught by a doc-sync run deriving the census instead of reading it. | Boundary + build + the `direction_gate` + kat rail status. | same | `develop` |
| `riir-dao` | **No `.docs/` folder** — AGENTS.md-centric (the KAT tokenomics agent: signals → strategy → guard → advisory → commit; the G5 advisory-only verdict) + `.plans/` / `.benchmarks/`. | Boundary + build + the direction gate. | same | `develop` |
| `riir-armageddon` | **RETIRED 2026-09-02 → `git/obsolete/` (owner act). Lineage only; do not route work here.** `.docs/` exists but is EMPTY — AGENTS.md-centric in practice (arena/game-product domain types). Added 2026-09-01 | yes | `.issues` 005, `.plans` 008 | **`main`** — not `develop`; check before branching |
| `riir-auth` | **`.docs/` exists but holds only `.highwater`** — i.e. no docs at all, AGENTS.md-centric in practice (the numbering file was created ahead of the folder's first document). Added 2026-09-01 | yes | `.issues` 002, `.benchmarks` 4, `.plans`/`.docs`/`.research` at 0 | `develop` |
| `riir-burner` | **RETIRED 2026-09-04 → `git/obsolete/` (owner act). Lineage only; do not route work here.** Flat numbered FILES, no folders — `.docs/001_model_verdict.md` … `016_*.md` (7 files; two share 016 — the numbering discipline is not enforced here). Added 2026-09-01 | yes | `.issues` 015, `.plans` 019 | `develop` |
| `riir-deployer` | **2 numbered folders** (`01_orientation/`, `02_runbooks/`) + a `.docs/README.md` index — the smallest numbered shape in the workspace. No `CLAUDE.md`. Added 2026-09-01 | yes | `.issues` 003, `.plans` 002, `.benchmarks` 001 | `develop` |
| `katgpt-web` | **No `.docs/` folder** — AGENTS.md-centric. Added 2026-09-01 | yes | none | `main` — the `feat/percepta-arch-diagrams` checkout note is history: Issue 002 (2026-09-09, `01a9c51`) merged + ff'd main to the working branch and the local feat branch is deleted; the repo sits on its trunk |
| `mmorpg-editor` | **NAMED (not numbered) `.docs/` subfolders** — `new-game-schema/`, `registry/`, plus loose `GAME_ASSETS.md`. Carries `ARCHITECTURE.md` + `DESIGN.md` alongside AGENTS/README, and `ARCHITECTURE.md` is where the internal layering lives (`BOUNDARY.md` covers only the outer edge). Added 2026-09-01 | yes | `.issues` 141, `.plans` 140 | `develop` |
| `riir-esp32` | **No `.docs/` folder** — AGENTS.md/BOUNDARY.md-centric (the ESP32 Satellite device-tier POC: `crates/riir-satellite-probe`, emulator recipes; explicitly not prod). **Row ADDED 2026-09-11** — the repo moved out of riir-chain 2026-09-06 and this census never gained a row (the same silent-skip class the de-counted header warns about). | Role + boundary + domain test. | `.issues` 110, `.proposals` 006 | `develop` |
| `riir-kat` | **No `.docs/` folder, no README** — AGENTS.md/BOUNDARY.md/HISTORY.md-centric (the KAT network CLIENT + wire-protocol plane, spun out of riir-clippy issue 088 on 2026-09-10; single crate). **Row ADDED 2026-09-11** — born 2026-09-10, censused a day late. | none — AGENTS.md §Status is the surface | `.issues` 1 | `develop` |
| `riir-shader` | **4 numbered folders** (`01_orientation/`, `02_substrate/`, `03_porting_method/`, `04_size_budget/`), unnumbered files inside — the WebGPU visual-effect substrate as Bevy plugins (ports every vgpu.sh example to wasm32+native as composable plugins; leaf, zero riir-* deps; upstream `vercel-labs/vgpu` @ 42bc4bc, MIT, attribution header per shader) + the `crates/riir-shader-graph` serde-only graph medium (Proposal 001 Phase 1, `3aa01ce`). **Row ADDED 2026-09-12; CORRECTED 2026-09-14** — the 09-12 row said "No `.docs/` folder" and the book landed 7h later (`2b78bb5`, 09-12 07:48, "Phase 2 doc-sync — 4 folders, 10 docs") with the Shape-change contract never run by the producer; the 09-14 run re-derived the census and caught it. | yes (README) | `.issues` 16 · `.plans` 002 · `.benchmarks` 3 · `.proposals` 001 | `develop` |
| `mmorpg-remaster` | **Flat numbered FILES** (`.docs/00_principal.md` … `10_quest_system.md` + `lessons_code_smell_audit_023.md` + `task_index.md`; design docs from the remaster gap analysis, `c0954dc`). **READ-ONLY to agents** (main + develop, owner rule). **Row ADDED 2026-09-14** — the repo joined the contract set at katgpt-rs Issue 760 making the workspace 20, but this census table never gained a row (the 09-12 anti-pattern census counted 19), and the flat-numbered-FILES shape it carries was wrongly believed to have left the workspace with `riir-burner`. | yes (README) | `.issues` 30 · `.plans` 045 · `.benchmarks` 042 | `develop` (read-only) |

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

#### riir-neuron-db (11-folder `.docs/` book; 10th added 2026-07-30, 11th `11_cli/` added 2026-09-09)
- **README.md**: build surface — feature gates (default-on / transitive / opt-in / per-feature prose sections for promoted primitives) + Formal Verification summary + License. Prose sections are reserved for promoted default-on features; opt-in features get table rows only.
- **`.docs/`**: 11 numbered folders mirroring the `riir-ai/.docs/` format. Drop new docs in the right group folder (by capability: shard substrate / freeze-thaw / consolidation / vessel / specialized / zone / examples / FV / **local-kv Warm tier** / **CLI**), add one line to that folder's `README.md` index table. The top-level `.docs/README.md` is the entry point. The `05_secure_vessel/vessel_primitive.md` doc is the restored home of the old `15_vessel.md` (corrected: riir-neuron-db is "this crate", NOT katgpt-rs per Plan 006). The `10_local_kv/` folder covers the `LocalKvStore` + `CommitLevel`/`CommitBatch` + WAL compaction + BM25 (the Warm tier substrate backing per-player state recovery in riir-mmorpg-examples Plan 013, added Issue 043). The `11_cli/` folder covers the `ndb` binary (Plan 327: Warm-tier CRUD, identity login via riir-auth `account_key`, multi-node WAL-mirror sync).
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

#### riir-refine (12-folder `.docs/` book)
- **`.docs/`**: 12 numbered folders mirroring the `riir-ai/.docs/` format — corpus / drafter / pruner / verify / ruliology / examples / benchmarks / lessons / self-evolve / domains / **ANE substrate knowledge** (`12_ane/`, riir-ai Issue 726 T0 distillation). Drop new docs in the right group folder, add one line to that folder's `README.md` index table.
- **`AGENTS.md`**: the batch-mining progress notes + sweep records live here (the cross-repo clippy-heal record home). A landed heal slice in a sibling repo (katgpt-rs, riir-train, riir-ai) gets its progress note in the SAME commit as the heal — a later `doc-sync` run defers to the healing session (never write progress notes for someone else's in-flight sweep).
- **README.md**: Status + Quick Start + Usage + feature gates.

#### riir-unity — RETIRED 2026-09-04 (`git/obsolete/`), lineage only
- **`AGENTS.md`**: domain boundary (no Rust crates here; UPM package is build output; no engine substrate in C#) + the Unity MCP rules + the issue log. Doc-sync = issue-log sections for resolved issues + module-map freshness (the `Packages/com.riir.viewbridge/` population + scene wiring notes).
- The Rust side of any feature lives in `riir-viewbridge` — cross-repo arcs (e.g. Issue 004) document on BOTH sides at arc close.

#### riir-viewbridge (`.docs/01_orientation/` + an AGENTS.md-centric remainder)
- **`AGENTS.md`**: workspace layout (core/derive/abi/xtask) + boundary rules (latent/raw wall, generated-bindings rule, catch_unwind) + the issue log.
- **`.benchmarks/`**: GOAT records (e.g. Bench 002 node GOAT). Doc-sync = issue-log resolution entries + benchmark cross-refs.

#### Every repo not subsectioned above (riir-dapps, riir-dao, riir-auth, riir-deployer, riir-esp32, riir-kat, katgpt-web, mmorpg-remake, mmorpg-editor)
- Follow the census-table row — these are AGENTS.md/BOUNDARY.md-centric: doc-sync = AGENTS.md/BOUNDARY.md status sections + numbering highwater + README freshness (riir-kat has no README; its AGENTS.md is the surface). The repo's own AGENTS.md supersedes this skill.
- `katgpt-web` checkout may sit on a feature branch (see census row) — sync the branch you find, and say which one in the run log.

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
folder to any repo that already has numbered folders — measured 2026-09-14 as
**12 of the 20**: `katgpt-rs`, `riir-ai`, `riir-chain`, `riir-clippy`,
`riir-dapps`, `riir-deployer`, `riir-game-sdk`, `riir-neuron-db`, `riir-shader`,
`riir-train`, `riir-viewbridge`, `mmorpg-remake`. **Or** that gives a `.docs/` to
one of the five with none (`katgpt-web`, `riir-dao`, `riir-esp32`,
`riir-kat`, `riir-mmorpg-examples`) or the one whose `.docs/` holds no document
(`riir-auth`) — creating the folder is itself a shape change and requires this
contract. Derive the split rather than reading it here (`ls */.docs`); the
previous version of this trigger named `riir-viewbridge` as having none while
its `.docs/01_orientation/` had shipped in the repo's own scaffold commit —
and the version before that kept `riir-shader` in the "none" list for two
days after its 4-folder book landed (`2b78bb5`), because a census row nobody
re-derived beats no row at all for hiding a shape change.

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
- **Do not** impose a `.docs/` shape that differs from the repo's existing convention — respect the shape you find. **Re-measured 2026-09-14 over the live 20** (the 09-12 measurement read 19 — it had no row for `mmorpg-remaster`, which joined at Issue 760): **12** numbered folders (katgpt-rs 10, riir-ai 12, riir-chain 7, riir-clippy 12, riir-dapps 1, riir-deployer 2, riir-game-sdk 10, riir-neuron-db 11, riir-shader 4, riir-train 6, riir-viewbridge 1, mmorpg-remake 1), **5** with no `.docs/` at all (katgpt-web, riir-dao, riir-esp32, riir-kat, riir-mmorpg-examples), **1** whose `.docs/` holds no document (riir-auth — only `.highwater`, which reads as a shape and is not one), **1** NAMED subfolders (mmorpg-editor), **1** flat numbered FILES (mmorpg-remaster — the shape did NOT leave the workspace with `riir-burner`; the 09-12 version of this sentence was wrong because that repo had no census row to be counted through). Don't re-type this census: the one-liner in the trigger above derives it, and every hand-written version of it in this file's history has been wrong within days — the previous one said "8 numbered / 8 AGENTS.md-centric" over a table whose membership differed from the workspace in BOTH directions. A shape change is a deliberate, committed decision governed by the **Shape-change contract** above.
- **Do not** renumber existing docs — the numbering discipline is monotonic and never reused.
- **Do not** document trivial mechanical commits (lockfile bumps, clippy fixes) unless they close a tracked issue.


## Standing lessons (distilled from the run log — the load-bearing process rules)

- **Per-file logs + pickaxe are the reliable baseline tools.** `git log -- .docs README.md AGENTS.md` once reported `5a1330b` as game-sdk's newest doc-touching commit while `git log -- AGENTS.md` + `git log -S '<landed-string>' -- <file>` proved `1977d83` (two days newer) had touched AGENTS.md. Always re-verify a suspicious baseline per-file before declaring a range clean.
- **Narrow producer-side syncs leave holes the baseline heuristic can't see.** A narrow sync fixes its own feature and skips everything else, so the strict `baseline..HEAD` range can be empty while the book still misses older landings. Gate runs GREP the book for recently-landed headline features; don't just trust the range. The inverse gap exists too: code landing to MATCH an already-documented row (mmorpg `?transport=local`) is invisible to any baseline heuristic — nothing to fix, but don't declare a gap either.
- **Grep extraction regex must include digits.** `^[a-z_]+ =` over `[features]` silently dropped `avatar_sync_ed25519` and `game_sync_p2p` — nearly filed phantom "documented-but-dead feature" findings. Use `^[a-z0-9_]+ =`.
- **The staged-index check must GATE the commit, not precede it.** One run saw two sibling-staged files in `git diff --cached --name-only` output and then sailed past them because the check was `;`-chained ahead of `git add` + `git commit` — the sibling's WIP landed inside the run-log commit (benign outcome, luck not process). The correct form: run the check as its OWN command and read it, or use `git commit -- <paths>` (a partial commit builds a temporary index from HEAD + the named paths, leaving anyone else's staged hunks intact — the safe form on shared checkouts). `scripts/staged_set_audit.py` (katgpt-rs) reports the same signal class pre-commit.
- **A doc claim of determinism is a claim about a proof.** When a fix refutes the proof (the ndb Bm25 tie-truncation class), grep the book for the CLAIM, not just for coverage of the fix — the fix landed with in-source comments only and the book kept asserting the falsified invariant.
- **False positives: grep the repo's OWN `.benchmarks/` before declaring coverage.** Different repos (and even different series in one repo) reuse bench numbers — the overview's only "Bench 025" hit was a different numbering series.
- **Broken links: prose citations survive file removal, markdown links do not.** The noise-reduction rule removes record files but no link-fix pass followed, so every closed issue left `](.issues/NNN…)` links dangling. Standing tools (committed in katgpt-rs **`.agents/skills/doc-sync/tools/`** — moved out of a repo-root `tools/` that no longer exists; the 2026-09-04 landing row below still says `tools/` and is history, not a path): `python3 .agents/skills/doc-sync/tools/linkcheck_sweep.py` (census) + `python3 .agents/skills/doc-sync/tools/link_fix.py <repo>` (auto R1-repoint / R3-delink), re-sweep to verify, commit pathspec'd `.md` only. Guards baked into the tools: **absent-repo** (a link into a workspace repo not checked out on the running box is UNVERIFIABLE, not broken — `exists()` is box-relative), **backtick** (link text already code-marked is emitted unwrapped), **EOL** (the fixer reads/writes with `newline=""` since 2026-09-22 — CRLF/mixed files round-trip byte-exact; before that the read_text/write_text pair LF-normalized whole files, measured on a CRLF `.research/200`, and the old 'restore EOL to HEAD convention manually' workaround is retired), and a markdown link split across lines is invisible to one-line fix regexes (NOMATCH → manual two-line edit).
- **Never cite "Plan/Research NNN" without naming the repo** when the number exists in more than one namespace — and never link a path you haven't verified (`ls` it first; per-repo numbering namespaces collide constantly).
- **Classification against dirty trees uses `git show <rev>:` content, never the working tree.** Sync remotes first; `git show -1 <sha>` mis-parses (the `-1` overrides the sha — pass the sha alone). The reliable per-file baseline is the 4-surface set (AGENTS.md / README.md / `.docs/` / the feature file) + a since-count.
- **"0 passed" in a baseline run is the reliable gate detector** — grep on attribute FORMS lies (whole-file `#![cfg]` gates sit below doc-comment headers, mod-level `cfg(any(feature…))` compiles empty under default).

## Run log (compact — full narratives live in git history)

Each row's durable record is the named `docs:`/fix commit(s) in the repo it
touched, plus this file's own history
(`git log -p -- .agents/skills/doc-sync/SKILL.md` — rows carried full
narratives until the 2026-09-05 compaction; `git log -S '<date>' -- <this
file>` recovers any of them). **Re-compacted 2026-09-11** — the rows appended
after 09-05 had regressed to full narratives again; same recovery applies.
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Pruned 2026-09-21: 119 → 15 rows, 104KB → 51KB** — this time the one-line convention HELD but the cadence didn't: ~8 rows/day × ~1KB/row re-bloated the file in 10 days (67 file-touching commits 09-11→09-19).  **Pruned 2026-09-23: 25 → 15 rows, 59.1KB → 49.2KB** — trip-preempting at 59.1KB, ~2 idle-pass rows under the 60KB threshold (the 04:49 idle-pass proximity note; the ~8-rows/day cadence lesson stands). **Pruned 2026-09-28: 25 → 15 rows, 56.9KB → 50.4KB** — same trip-preempt precedent at a full-gate run (fetch sweep 26/26 + linkcheck 0 breaks / 5894 files + dirty sweep all sibling-WIP). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`. **Pruned 2026-10-12: 17 → 14 rows + the new unit = 15** (60,152 → 56506 bytes) — trip at the 60KB line during the sge proposal-008 doc-sync unit.

| Date | Scope | Verdict | Record (primary fix commits) |
|---|---|---|
| 2026-10-20 | delta unit #25 (M3 ~14:0x +07, seal-remake issue-104/105 continuation session; sibling-hot untouched: the seal-remake skill-fx lane (main-checkout WIP), the katgpt-rs Fathom 628 lane, the riir-train Plan-465 sweep, the riir-ai wanderlust lane) | 2 wall-calibration landings self-doc'd in-commit (083 §7 — POST_KILL_STILL_WALL 46→60, the 53 mechanism ceiling derived + two-content-state draws, probe green; 105 interim 06:53Z read recorded in-file) + the claim-line canonicalization class swept workspace-wide: 3 malformed lines FIXED (seal-remake 105 + plan 035 `f61ceaed`, riir-train 462 `453760fd`, seal-std 001 `6cb83c2` — prose-after-ISO / ISO-in-epoch-slot / +07-offset shapes blind claim_lease_check; katgpt-rs 627/628 Z-suffix verified PARSING, 950 strikethrough correctly unseen) · verified NO-gap: seal-remake AGENTS carries no wall constant, ae84a8a9 wanderlust = the live lane's own doc duty · prune 19→15 rows (58.2KB trip-preempt) | seal-remake `e034be46`+`f2ab50cc` · katgpt-rs (this commit) |
| 2026-10-20 | delta unit #24 (4090/shikuwa, riir-refine idle loop — overflow-continuation session, bout still GPU-gated: llama-server StartLux resident, window to ~13:30Z; Protocol H: 9-repo fetch, riir-ai ff'd 4f76a637d→99cbee613 around the sibling's dirty lock; my AGENTS push rode one rebase --autostash over the seal lane's disjoint 4321db091 — overlap verified zero before the autostash) | delta since #23: 1 gap FIXED — riir-ai 99cbee613 (`hero_survival_stoop` + `hero_spin_break`, the owner's RUN report, both opt-in, tests 24/24 both postures) landed doc-silent 5 min after #23's commit — the mmorpg AGENTS row gains both features with mechanism + Schmitt/budget constants (the buffs-row precedent, narrow-producer-sync hole class) · verified NO-gap: refine 726c0ef2c + katgpt-rs 8cbbec896 self-doc'd (the cycle row + the Research 577 PASS-Redirects) | riir-ai `9179ea2bc` · katgpt-rs (this commit) |
| 2026-10-20 | delta unit #23 (4090/shikuwa, riir-refine idle loop — GPU-gated day: llama-server StartLux resident to ~13:30Z so the decode re-pin bout + item-8 benches skipped; Protocol H at session start: 9-repo fetch, katgpt-rs ff'd 990d975→e91c32a (the sibling-leased Fathom lane's 2 docs commits), 0 zombies, E: 53G / C: 40G BELOW the 100Gi line — owner lists at `.distill/002` stand, E: dropped 67→53G under the StartLux lane's model writes, reported not deleted; sibling-hot untouched: riir-train 635 StartLux lane live (routing-restricted, report-only), katgpt-rs 628 Fathom lease to 10-22) | delta since #22: 2 gaps FIXED — refine B262+B264 count-neutral riders landed research+plan+corpus in-commit but skipped the `.docs/11_domains/kernel_opt.md` corpus ledger (260/261/263 present, 262/264 absent) — both rows backfilled + the AGENTS kernel_opt bullet gains the count-neutral rider mention (the rust_perf bullet's own list pattern) · instinct plan-225 T2/T4 examples (705 lines, `instinct_heal_train` + `heal_specialist_server`) landed doc-silent — Build-commands block added with the real arg surface (τ digest-checked sidecar verified at source before documenting) · verified NO-gap: katgpt-rs 950 gated_summary row (opt_in_features §152) + 949 path_kernel_sim row (§153, T9 KEEP-OFF verdict in-file), 628 plan ticks in-commit (Fathom lane's own docs commits), rethink Research 011 / infer 619+621 / reflexer E18 all docs-commits themselves, reflex zero commits, ai knock_force tweak = catalog data with the owning record in the seal-remake lane · linkcheck 7198 files: TOTAL 2 = the standing sr079→ai-1078 deferral only | riir-refine `f15420da0` · riir-instinct `ac4aa60` · katgpt-rs (this commit) |
| 2026-10-19b | delta unit #22 (M3, riir-refine idle-loop continuation post-substrate-first unit; Protocol H: workspace fetch all in-sync except riir-refine behind-49 + dirty-181 and seal-remake behind-2 + dirty-83 (both sibling lanes' WIP, left as-found); disk 203Gi healthy, 0 zombies, 0 cargo procs; sibling-hot untouched: ai-1095 gatefinding, sge plan-392 quest cutover) | delta since the plan-033 arc run: 1 gap FIXED — dapps issue 130 T3 sponsor lane `4f5ef03` landed plan-file + posture note in-commit but no AGENTS §Status row (claim released by its lane; the narrow-producer-sync hole class) — row added top-of-list · verified NO-gap: katgpt-rs ×2 skill rows, ai-1095/1093 ×4 docs records, sge plan-391 arc (issue 026 close carried HISTORY row, 218 + 232/233 closes self-doc'd), seal-remake lava pair + docs(079) all self-documented in-commit · linkcheck 7132 files: TOTAL 2 = the standing sr079→ai-1078 deferral only (live lane's closeout) | dapps `46bba81` · katgpt-rs (this commit) |
| 2026-10-19 | plan-033 arc doc-sync (M3, own-lane follow-up post slice F `105e3ce`; Protocol H light: seal-remake/sge/riir-ai fetched in-sync, sge clean at the slice-D tip, sibling-hot seal-remake dirty files untouched) | 2 doc-silent rows closed + 1 verified self-doc'd: riir-ai AGENTS mmorpg row gains the `buffs` machine (`3db3c1202` — 532b33b84 landed plan-recorded but row-less; the is_idle zero-buff skip IS the landed cost check) · sge AGENTS RUN-presenter bullet names the four fx wire docs (`ca1ec37b` — 67ffca0d+7efaef2a doc-silent) · seal-remake plan 033 self-doc'd in the plan file (a61d2fd + the P7 amendment; HISTORY row deferred to plan completion per the plan-029 pattern) · prune 19→15 rows | riir-ai `3db3c1202` · sge `ca1ec37b` · seal-remake `105e3ce` · katgpt-rs (this commit) |
| 2026-10-18c | delta unit (M3, 623-tail watch continuation; Protocol H: 15-repo fetch + selective ff-pull (riir-refine behind-44 + heavy sibling WIP left as-found = the B250 landing lane's checkout), 4090 gitsync SYNC DONE 19:41Z clean, disk 191Gi M3 / 62Gi E:; sibling-hot untouched: refine B250 slice-1 (claim b52a4874, combo test running), katgpt-rs epoch lane, train 577 corpus, seal-remake stall lane, plan-621 window claim (shikuwa, expires 10-20)) | delta CLEAN 0 book gaps: all +since landings self-doc'd (katgpt-rs docs rows ×5, ai-971 arc ×8 docs, train 2cc443e0, infer plan-621 docs arc + issue-929 code covered by katgpt-rs HISTORY 1689 with sibling hashes — cross-repo citation law satisfied, sge batch carries plan-378 status + tracked issues 232/233) · linkcheck 5→2: 3 FIXED (the 2f2602d04 issue-929 removal's bench links → prose citations) + 2 DEFERRED (sr079→ai-1078, the live lane's closeout) | katgpt-rs `eed47fbbe` (this commit) |
| 2026-10-18b | delta unit (4090/shikuwa, riir-refine idle loop — substrate-first 18c unit same session; sibling-hot untouched: ai-1091 live lane rebased-under cleanly, B250 M3 claim to 21:45Z, plan-436 GPU chain) | 1 gap FIXED: ai-1081 T1–T4 landed `1999fff01` doc-silent — riir-games feature table gains the crowd_state_windows row (HISTORY 1165 close verified self-doc'd; the row landed against the JUST-closed issue — filed post-close, no conflict) · katgpt-rs QRR row verified PRESENT (README L2885, self-doc'd at P6) · B249 slices 2/3 verified self-doc'd (AGENTS + .docs/11_domains in-commit) | riir-ai `a9ef6fa9d` · katgpt-rs (this commit) |
| 2026-10-18 | delta unit #21 (M3, idle-loop continuation post-reflex 085+093 closes — the closes self-documented in-commit; Protocol H: 30-repo fetch (non-repo dirs skipped), no zombies owed, disk / 57Gi BELOW the 100Gi line -> disk-clean audit owed owner approval, SD 182Gi healthy; sibling-hot untouched: ai-1091 combo lane, refine Batch-250 wt, clef CpuOnly lane, dapps §9 lane, the katgpt-rs 947 split lane (180-file WIP as-found), sge lava lanes) | delta since #20: 1 gap FIXED (seal-std README seal-anim row gains FORMAT_VERSION 8 — Take.length + the MIN_FORMAT_VERSION additive-reader law; 326ef71 landed doc-silent) + 1 one-liner (instinct AGENTS arena --winners-raw-v1 flag — 9a0232f doc-silent, beside the fake-quant line) + 2 delinks (rethink research-001 removed-028 links, R3) · verified NO-gap: sdk hunt-reach STRANDED refinement (BOUNDARY charter carries it), ndb 5adac74 (mechanical, no doc owed), all other landings self-documented in-commit · linkcheck 4->2 (the standing sr079 deferral only) | seal-std e87e6a0 · riir-instinct d27252c · riir-rethink 7b5c8ee · katgpt-rs (this commit) |
| 2026-10-17 | delta unit #20 (M3, idle-loop continuation cycle post-#19; Protocol H full: 37-repo fetch clean, 0 zombies, disk 129Gi/217Gi, 4090 gitsync SYNC DONE 12:11Z via Tailscale; sibling-hot untouched: the ai-1083 lane, the seal-remake post-soak lane ×74, sge release.rs, the b917 4090 finetune, Batch-249's 4090 claim to 10-18T00:20Z, riir-infer 048 claimed to 10-19T10:00Z) | delta since #19: 4 self-doc claims grep-verified (shader HISTORY 055, kat HISTORY 006, reflexer HISTORY 005, refine AGENTS P74 counts) + linkcheck 7113 files 4→2: 2 FIXED (ai plan-637's two closed sge-219 links R1-repointed to sge HISTORY — parked plan, named-file commit over the 1083 lane) + 2 DEFERRED (seal-remake 079→ai-1078, the LIVE sr079 lane's closeout — standing) | riir-ai `5b9a716d2` · katgpt-rs (this commit) |
| 2026-10-17 | delta unit #19 (M3, same idle-loop chain as unit #18, cycle continuation post-boundary-213th; Protocol H re-verified: 30-repo fetch 0 moved, riir-train behind-8->0 resolved since #18, disk 183Gi/217Gi healthy, load 10-20 = the sibling capture lanes so Plan-337 kept this unit git-only; sibling-hot untouched: the b917 finetune lane (4090), the 619 clef ANE lane, the seal-viewer leak check, Batch-249's 4090 claim to 10-18T00:20Z) | delta since unit #18 CLEAN 0 doc gaps in the claimable set: the only landings are this chain's self-documented idle units (katgpt-rs `2e2782831` boundary-guard 213th row, refine `e6d95838` post-mining + `5bdd194c` substrate-first - snapshot/skill rows, zero book .md) - decode re-arm trigger FIRED on prism-b10777-e3de715 recorded in refine `ade38d6b` (the re-pin cell waits b917-exit + B249, no doc action) - linkcheck not owed (zero book .md landings since #17) - TRIP-PREEMPT prune 21->15 rows (59.5KB pre-row, the 2026-09-23 precedent at 59.1KB) | katgpt-rs (this commit) |
| 2026-10-17 | delta unit #18 (M3, new idle-loop session post-handoff; Decision-ordered re-verification; Protocol H: 38-repo fetch behind=0 everywhere except riir-train behind-8 = the live 4090 lane's own pushes, left as-found; sibling-hot untouched: seal-remake ×76 (RAM-leak lane, bot-host procs live), sge `release.rs` + `warm_tier.db/`, mmorpg-examples Cargo.lock churn, the b917 finetune lane, the 619 clef ANE lane, Batch-249's 4090 claim to 10-18T00:20Z) | delta since unit #17 CLEAN 0 doc gaps in the claimable set: the only landings are riir-train +3 (c0b8d897 archive row + 0bfc2d6c RUN chain + ab11a13c wiring gates — the plan-445/issue-633 lane's own self-documented pushes; lane-owned doc duty, routing restriction) + this session's boundary-guard 212th row (self-documented) · seal-remake/sge movement all UNCOMMITTED WIP (not landed, no doc duty yet) · linkcheck not owed (zero book .md landings since #17) | katgpt-rs (this commit) |
| 2026-10-17 | delta unit #17 (M3, same idle-loop session as unit #16, sge records-hygiene follow-up; Protocol H light: trees committed, disk healthy, sibling-hot untouched) | 1 stale close-out recovered + worktrees adjudicated: sge issue 226's T2b/T3 fix had landed on develop (`8c307518`, FRAME_NAV_STRANDS emptied) but the landing session's close commit `bc0f1c5f` stayed on its detached `sge-226` worktree — the issue file sat tracked with a stale Open status and HISTORY had no 226 row; close-out re-landed on develop (`79be4ce7`, file removed + the recovered row citing the develop hashes + the unmerged-close-out note; zero inbound links grep-verified) · worktrees: sge-226 + srx-sge removed (clean, superseded), lavacap prunable metadata pruned, sge-105698 (dirty walk_export.rs) + sge-wt-b1/b2/verify (dirty, unknown lanes) + run-net-selector (branch-holding) LEFT as-found | sge `79be4ce7` · katgpt-rs (this commit) |
| 2026-10-17 | delta unit #16 (M3, idle-loop housekeeping post-issue-227 handoff; Protocol H: fetch sweep in-sync except riir-train behind-5 = the live 4090 lane's own pushes, left as-found; sibling-hot untouched: the b917 chain-watch, the seal-viewer RAM-leak probe, the 619 clef lane, Batch-249's 4090 claim) | delta since unit #15 CLEAN 0 doc gaps in the claimable set: sge issue-227 arc self-doc'd (eaac3880 carries the HISTORY entry; b5fe96c4 the SCHEMA fix) · seal-remake 096 close self-doc'd (27d9ba9 + 8f712b8 docs commits) · sge plan-380 T7/T8 + seal-remake 8e9344b lava = the plan's own in-commit records · the 211th boundary run's 2 seal-asset findings repaired in-run with HISTORY rows in-commit both repos (sge 228 `70fef1d5`, remake 097 `9bfe556`) · linkcheck not owed (zero book .md landings; BOUNDARY/HISTORY edits carry no links) · IN-RUN SKILL REPAIR: unit #15's row found mis-placed inside the census table (the 209th's insert-anchor misfire class) — moved into this run table with parity, newest-first | sge eb91cf0b+70fef1d5 · seal-remake 0b17b9d+9bfe556 · katgpt-rs (this commit) |
| 2026-10-16 | delta unit #15 (M3, katgpt-rs idle continuation after the ai-1079 close-out; Protocol H: 33-repo fetch — reflex behind-4+dirty ff'd to `aea23a9` via the temp-worktree pattern (the pin re-seed), sealm-toolkit behind-5 left (consumed by the live sge agent), disk 233G/245G healthy, no zombies, worktrees all KEEP (2 live lanes + dirty WIP + unmerged branch); sibling-hot untouched: seal-remake sr079 preload lane, sge bridge-walk-guards lane, refine Batch-249 (4090 claim), 626/627 claims live) | delta CLEAN: ai-1079 CLOSED in-commit (M3 golden+genesis legs GREEN, consumer legs realized-by-import re-runs GREEN) · katgpt-rs 941+942 hygiene (`21843a1fb`, landings verified before removal) · seal-remake 087-093 hygiene (`f550c44`, 087-090 status lines were stale from filing) · rethink 035 T1 done (reflex `aea23a9` + rethink `1811613`) · linkcheck 7081 files: TOTAL 2 = the known sr079→ai-1078 deferral only (live lane's closeout), zero new breaks from the 9 removals (inbound-link grep + census verified) | katgpt-rs `21843a1fb` (this commit) · seal-remake `f550c44` · riir-reflex `aea23a9` · riir-rethink `1811613` · riir-ai `01b345a52` |
| 2026-10-16 | delta unit #14 (4090/shikuwa, LOWO-chain-watch continuation; Protocol H: 8 workspace repos fetched all in-sync, chain healthy (arm 2/4 customer_service GPU 100% since 07:10, arm 1 rc=0 7.21h, 623 tail staged --wait), league watch polled: prism tags unchanged past b10777 — no-fire; sibling-hot untouched: refine Batch-249 claimed 07:20, train 632 jeva lane (M3), seal-remake sr079 live) | delta since unit #13 CLEAN 0 doc gaps (train 445-P1 `156556e7` self-doc'd plan+bench+highwater; ai `57cb34e05` posture test needs no doc — zero SHINE claims in the book, grep-verified; reflex 092 + refine B248 + instinct 023/623-T5 self-doc'd in-commit); linkcheck 9→2 (7078 files): katgpt-rs bench-938 R3 delink · ai league 633/984 R2 repoint ×2 + plan-634 R3 delink ×2 (gsdk 046 removed 10-15) · rethink 067↔032 R2 repoint ×2 (depth, both files exist); 2 DEFERRED: seal-remake 079 → ai-1078 (removed) = the LIVE sr079 lane's closeout | katgpt-rs (this commit) · riir-ai (this commit) · riir-rethink (this commit) |
## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
