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
| 2026-10-16 | delta unit #13 (M3 ~08:1x +07, riir-refine idle-loop continuation during the 632-T2 jeva capture; Protocol H: 37-repo fetch all in-sync, riir-refine behind-3 + dirty = the live Batch-248 lane as-found untouched, disk 144Gi/245Gi healthy, no zombies, 6 worktrees adjudicated leave (sr079-headcheck clean+touched-today live, run-net-selector branch-holding, 4 dirty 10-14 reported); sibling-hot untouched: riir-ai Wanderlust/1079 + plan-626 lanes, seal-remake 079 warp-swap lane, refine Batch-248) | 2 doc gaps + 1 stale chain closed: riir-infer AGENTS build block gains the entropy_bounded_commit lane row (44ca3cb landed doc-silent 10-16) · riir-kat HISTORY gains the issue-126 follow-up arc (6 client-plane commits 10-14/15 doc-silent: fec926c/192badc/1a48fc3/3c21880/a70fbad/6384e65) · riir-auth HISTORY gains the SSH-direct posture (146ed0b+386b021 doc-silent; the refine-side citation re-verified to 42048b49/2b556584 before commit — the memory-quoted hash was wrong) · dapps issue 117 D8 clock chain synced with bench 018's 6th reset (dc398f6); artb 512 verified NO gap (LEGACY_MAX_ART_ENTRIES self-doc in-code, zero stale claims); clean deltas: katgpt-rs/reflex/rethink/instinct/reflexer/chain/sdk/ndb self-doc'd or sibling-lane | riir-infer `0e50c61` · riir-kat `a57c8df` · riir-auth `560a87f` · riir-dapps `dc398f6` · katgpt-rs (this commit) |
| 2026-10-15 | delta unit #12 (M3, riir-refine idle-loop Decision-ordered continuation; boundary 207th landed-state + 208th scoped-gate rows in the same session; sibling-hot untouched: riir-ai hero-in-blast 1070 + plan-032 lanes, sge RUN lane; the 4090's P65+P66 claim respected — mining stayed off this session) | THE PLAN-619 ARC (riir-infer, self-doc'd in-commit each step): T1.0 the shared ANE host crate (`2ca7f30`+lock `415c43d`) carries its plan tick + evidence + BOUNDARY rows in-commit · T1.1 the clef skeleton (`af00df7`) same · the C0e rot repair + AGENTS workspace-layout row (`49ab6d1`) — the one DOC GAP this unit: riir-infer AGENTS §layout named neither new crate, closed in-commit · reflex consumer lock rode the substrate landing (`7aaf0e4`, fix: lock — its manifest graph genuinely moved); katgpt-rs skill rows = this commit; linkcheck not owed (zero book .md landings) | riir-infer 2ca7f30/415c43d/af00df7/49ab6d1 · riir-reflex 7aaf0e4 · katgpt-rs (this commit) |
| 2026-10-15 | delta unit #11 (M3, riir-refine idle-loop Decision-ordered continuation; boundary 207th landed-state row in the same unit; Protocol H: 37-repo fetch all in-sync (riir-ai ahead-1 = a live sibling lane's unpushed commit, as-found untouched), disk 172Gi/261Gi healthy, no zombies, 4 dirty detached worktrees (sge-wt-b1/b2/verify + remake-wt-stones) reported-not-removed (live sge/remake lanes); sibling-hot untouched: riir-ai hero-in-blast 1070 verdict lane + plan-032 force-knockback lane, sge RUN lane) | delta since unit #10 CLEAN 0 gaps: refine B237 `c3f51cae` self-doc'd (AGENTS rust_perf-201 row + .docs/11_domains/rust_perf.md + README in-commit) · refine B238 `26300ffc` self-doc'd (AGENTS kernel_opt-813 rider row in-commit) · the 4090's claim/envelope/disk/docs commits (d983e23a/af914ee7/e2ecb1bc/2b556584/42048b49) ARE docs commits · katgpt-rs `55f2ec73c` = the skill rows themselves; linkcheck not owed (zero book .md landings — corpus .rs + skill rows only) | katgpt-rs (this commit) |
| 2026-10-15 | delta unit #10 (M3 ~11:0x +07, riir-refine idle-loop Decision-ordered continuation; boundary 206th full-view row in the same unit; Protocol H: 30-repo fetch — seal-std ff'd +7 (the seal-stat/seal-formula landing, fully self-doc'd), all other repos in-sync, disk 250Gi, katgpt-rs clean; sibling-hot untouched: the seal-game-editor spawn/equip debug lane (other agent)) | delta since unit #9 CLEAN 0 gaps: reflex issue-089-T1 `cb6f5ff` self-doc'd (commit message + analyzer script + bench record ARE the record) · seal-remake issues 075+077 removal `2769f7f` carries one hash-pinned HISTORY row each in-commit, zero inbound links to the removed files (grep-verified) · riir-ai C6 trim `467739939` provenance in-message + the 1069 record · seal-std seal-stat landing self-doc'd (README crate-table rows + BOUNDARY.md Owns rows in-commit); linkcheck not owed (zero book .md landings — removals only, zero inbound links) | katgpt-rs (this commit) |
| 2026-10-14 | delta unit #9 (M3 ~23:1x +07, riir-refine idle-loop Decision-ordered continuation; boundary 202nd landed-state row in the same unit; sibling-hot untouched: the 4090 pplx/bout lane (rethink 032 T3.5 claim `22:4x`), the torch-flame gate re-run, the rethink UltraBERT lane (0 pushed at scan time)) | delta since unit #8 CLEAN 0 gaps: refine `768503bd` IS the newest docs commit (the cycle row documents the t32 re-pin in-commit — baseline == HEAD, empty range by construction) · `529c2083` self-doc'd (the assert-comment re-measure history + the .distill row) · riir-ai +2 = docs/issue records in-commit (the b10776 re-pin row `460a04780` + the ai-1066 dispatch file `c4b1b16bc`) · linkcheck not owed (zero book .md landings since the baseline — the .distill row is a record file, no links) | katgpt-rs (this commit) |
| 2026-10-14 | delta unit #8 (4090/shikuwa, riir-refine idle-loop cycle 74; boundary 200th landed-state row in the same unit; Protocol H: 39 repos fetched — riir-kat ff'd +3 (the 163 follow-ups), riir-train behind-15 sibling WIP as-found, 1 inert ssh.exe standing, worktrees clean, disk C:82G/E:109G standing owner-gated; Watch 971 polled: #29353 OPEN, head `dde6a4d` unchanged, ggerganov APPROVED 10-13 (1 of 2 approvals) — no-fire, re-armed cycle 75) | delta since unit #7 CLEAN 0 gaps in the claimable set: refine P51a `3ff0013fe` self-doc'd (AGENTS clippy_lints counts 112/90/19 + the P51a narrative in-commit, .distill P51 row) · sibling refine-163 `9ca1f90cf` self-doc'd (AGENTS+HISTORY in-commit) · kat +3 self-doc'd (`f579e1a` docs commit; HISTORY +24 in `03f4b62`) · train +15 self-recording (issue removals + bench files; routing-restricted, no action either way); 1 DEFERRED finding: reflex AGENTS L346 "9 dataset suites" is behind the dataset_manifest's 15-row numbering (the JDI four = rows 12–15, `8d1a9f7`) — the LIVE 087 lane's closeout owns it (B2 clef tranche in flight, partition rule) | katgpt-rs (this commit) |
| 2026-10-13 | delta unit #7 (M3 ~10:5x +07, riir-refine idle-loop Decision-ordered continuation; boundary 199th scoped run in the same unit (zero-movement verdict); Protocol H: 36-repo fetch all in-sync zero incoming, 4090 gitsync via Tailscale (LAN timed out), disk 166Gi/261Gi, no zombies — the only compute is the live 938 lattice GOAT bench, Plan-337 kept this unit grep/git-only; sibling-hot untouched: Beggar lane (riir-ai/seal-remake), 938 lattice (katgpt-rs), SubQ (rethink)) | workspace delta since unit #6 CLEAN 0 gaps in the claimable set: katgpt-rs 936 arc (T5 bdccce0e9 / T4-PROMOTE cbaeac926 / closure 7652638ac) verified self-doc'd — HISTORY row carries the full arc + promotion measurements, .docs/10_audits q2 record in place, T4 is a code default NOT a Cargo feature (1-file stat, tables unchanged), zero inbound links to the removed .issues/936 · seal-std ComboClock merge f139635 self-doc'd (README +2 / BOUNDARY +5 in-commit) · dapps plan-042 closure f133e6d edited the plan in-place (no removal) · refine batches 226/227 + walk/dual-walk records self-doc'd (AGENTS kernel_opt 803 in-commit) · train/rethink/sge landings are docs commits themselves; linkcheck NOT owed (only removal has 0 inbound links; other .md landings verified targeted); live-lane landings (Beggar/938/SubQ) partition-deferred to their owners | katgpt-rs `7bcd0df4e` (this commit) |
| 2026-10-13 | delta unit #6 (M3 ~08:4x +07, riir-refine overflow-handoff continuation; boundary 198th scoped run — contract clean 29 repos/371 edges, C8/C9 advisories informational; Protocol H: 29-repo fetch 0 moved/0 failed all behind=0, 0 zombies, 190Gi disk, 4090 gitsync already fired by the sibling lane (f4800cd2 08:22); sibling-hot untouched: refine Batch-226 claim (262b10d7) + live G2 lib suites; handoff's stale 926 item verified ALREADY LANDED by a sibling (6d1add192 CLOSE+PROMOTE — mla_absorbed in default features + kill-switch + bench record, in-tree verified) | 5-repo delta CLEAN 0 gaps since 06:00 (katgpt-rs 17 / infer 17 / refine 8 / ai 2 / train 1): katgpt-rs issues 933/936/937 closed-with-records (937 record relocated to .docs/10_audits/q2_matvec_bandwidth_levers.md; decode_spin_pool + ternary_q2_q8 pair/N-way = kernel functions NOT features — 454 features unchanged, catalog untouched, retired q2_prefetch zero residue); infer 043/044/045 resolved via Bench 879 (spin pool promoted, Bonsai ~100→72 ms/tok); refine batch 225 = the AGENTS kernel_opt row (801) in-commit; ai/train mechanical fixes owe no docs; linkcheck 0 (6880 files — owed because the 08:03 q2 audit .md landed after unit #5's not-owed verdict) | katgpt-rs (this commit) |
| 2026-10-13 | delta unit #5 (M3 ~06:5x +07, riir-refine idle-loop continuation; boundary 197th scoped run in the same unit; Protocol H light: M3 35-repo fetch all in-sync, 4090 gitsync fired via Tailscale (LAN timed out), disk healthy, sibling-hot untouched: riir-infer release-test lane (live sibling cargo) + katgpt-rs 935-session WIP) | riir-refine delta CLEAN 0 gaps: every landing since the last docs commit self-doc'd (f0718269 cubecl vendor mirror = the AGENTS.md Batch-224 rider text "our vendored 0.11.0-pre.2 pin-bump carries the fix"; e8f76fa1 corpus-pr = the AGENTS.md CLI contract row; b6f2310b v0.3.1 = the release tag itself); linkcheck not owed (zero .md book landings since the baseline — the 3 dirty files are uncommitted dashboard artifacts) | katgpt-rs (this commit) |
| 2026-10-13 | delta unit #4 (M3, 29-repo fetch all behind=0; delta since `2408f4b32` over 7 repos — ai 20, infer 8, refine 3, remake 2, dapps 1; infer 038/041/620 + refine/dapps plan-050 lanes self-doc'd in-commit, dapps S1 mid-plan = its plan file) | 2 gaps + 1 link closed: riir-ai AGENTS mmorpg row still said the style think-brain feed "waits Issues 1043/1045" (both closed) — now names `hero_style_prior` (Plan 631, G2 FAIL, opt-in) + `hero_rest_readiness` (Plan 630, promoted into seal-view run) + `hero_hold_band_commit` (opt-in) · riir-ai HISTORY row for 1043/1045/1061 closures + Plans 630/631 (1058 row already in-commit) · seal-remake proposal 006 dead link to removed riir-ai Issue 1045 delinked; linkcheck 1→0 (6865 files) | riir-ai `111a31bb6` · seal-remake `3772d31` · katgpt-rs (this commit) |
| 2026-10-12 | delta unit #3 (M3 ~00:0x +07 10-13 local, riir-refine idle-loop continuation; Protocol H this session: 3 repos ff-merged (katgpt-rs e70e0bdb9 / dapps 03f61c9 / refine 53299212 — all self-doc'd), disk healthy, sibling-hot untouched: seal-remake vessel_boot lane + plan-027 code-only landing a48a113fb = ITS lane's records at closeout) | 1 doc gap closed: riir-ai AGENTS basis_purity_audit row still claimed the 931 blockage — updated for the post-931 re-read (37b029ae7: FEAR_DIR 0.295→0.264, style S1 = 0/77 honest negative, T5 owner-gated); katgpt-rs sync_bank promotion verified SELF-doc'd (605bc3d2b carried README counts + catalog §147 + HISTORY qualified citation in-commit) | riir-ai `a65276811` · katgpt-rs (this commit) |
| 2026-10-12 | delta unit #2 (M3 ~07:1x +07, idle-loop overflow-handoff continuation; boundary-guard 193rd scoped run in the same unit; sibling-hot untouched: seal-remake hero-combo lane dirty manifests + a live cargo test --workspace, riir-ai plan-027 lane) | 1 doc gap closed: riir-refine AGENTS feature table gains `basis_cert` (e6c02dae landed 06:59 doc-silent — Cargo feature + example row, zero AGENTS mention; grep-verified); issue-162 closure (421b3e7a) already carried its HISTORY row in-commit; katgpt-rs BasisSplit row accurate as-written (reflex = first consumer, refine = second, no false claim) | riir-refine `e1ec5c40` · katgpt-rs (this commit) |
| 2026-10-12 | delta unit (M3 ~05:1x +07, sge proposal-008 handoff continuation; Protocol H light: 17-repo fetch, disk 77Gi, no zombies; sge ff'd fcc61f6c→02bf0cf7 — origin already held the whole plan-347 merge; sibling-hot untouched: riir-ai DIVERGED ahead-7/behind-7 with FRESH plan-027 WIP (7 files, 03:5x-04:1x mtimes; local 52a4cbe87/4fb12b8d9 are patch-id-identical duplicates of origin's 43347b74e/88212f0f9 — the owning session's pull --rebase drops them), refine behind-19 + 87 dirty, seal-remake 7-file hero-combo/safe-zone lane, katgpt-rs README pair) | Proposal-008 arc doc-sync, 3 doc-silent landings closed: riir-ai AGENTS mmorpg row gains the duel/party lane (issue 1055 arc: 0e0916496/72e970ef8/501bbda49/43347b74e/378f94cdf; landed via an origin/develop detached worktree — the local checkout is sibling-hot + diverged) · seal-remake AGENTS dead-phase FINDING → the landed fix record + the HISTORY row for batches 1+2 · sge AGENTS routing rows (RUN Duo orchestrator, the SEAL_SPAWN_INTENT_DIR seam, seeded random equip); owner GUI smoke skipped (owner-gated) | riir-ai `4a5ac1f1f` (via worktree) · seal-remake `9ff20d2`+`7312909` · sge `fc185041` · katgpt-rs (this commit) |
| 2026-10-11 | delta unit #3 (M3 ~13:1x +07, issue-929 run-watch lane's bounded unit while the Bonsai-2 full run cooks; delta window since the 09:4x unit over 8 repos — ~36 commits, all self-doc'd or sibling-lane: dapps plan-050-S0 + plan-223 W5/W6/W7W8 AGENTS bullets, train bench-633/634 + plan-454 rows + README catalog, refine close-out-audit backfill dc97a554 + plan-224 lane, kat issue-003 closed+removed (HISTORY record landed in the closure commit), seal-remake issues 052/054/058 rows, katgpt-rs bench 933/934 records; DEFERRALS by partition rule: riir-infer 038/039 = the live 11-file-WIP lane's, seal-online-site = the wiki-migration lane's) | 9 broken links → 0 (6818 files): riir-ai proposal-055 ×3 depth R1 + chain bench-346 issue-165 R3 + dapps plans 045/048/050 riir-kat-003 R3 ×4 + seal-remake plan-021 issue-058 R3 — the 6 delinks are THIS WINDOW's own removals (kat-003 b699612, remake-058 cf43146) landing ahead of any link-fix pass; ⛔ NEW FALSE-POSITIVE CLASS (Issue-825 cousin): the riir-ai ×3 "fixes" duplicated upstream 60875d7ec (the 09:4x unit's own fix) because the local checkout was 45 behind — linkcheck reads the WORKING TREE, so a stale checkout re-finds fixed-upstream breaks; local duplicate commit af86fd599 reset away (HEAD~1, sibling commit + dirty WIP byte-untouched), the 3 stay upstream-fixed; RULE: before committing a linkcheck fix, grep origin/<branch> for an equivalent fix when the repo is behind | chain `f9823f59` · dapps `4441acb` · seal-remake `75a3d95` · katgpt-rs (this commit) |---|
| 2026-10-11 | delta unit #2 (M3 ~09:4x +07, riir-refine overflow-handoff continuation; Protocol H clean — 28-repo fetch sweep 13.8s/0 failed, disk 338Gi internal + 261Gi SDXC, no zombies, worktrees clean, katgpt-rs ff'd 29e2e54f7→9f481346a; sibling-hot untouched: refine bench-artifact WIP ×4 = the idle-loop lane's, riir-ai behind-40 + dirty 971 lane) | 2 riir-refine AGENTS.md gaps closed in one commit: the `rescue_probe` feature row (893f4df8 landed 03:32 doc-silent — Cargo feature + rescue_mine.rs +233, its issue lane closed+removed, grep-verified zero AGENTS mention) + the `--mine` consent screen's MIT-licence line (riir-dapps 119 T16b, c4fc0a4c — the arc bullet lives in dapps AGENTS only, the CLI surface doc had nothing); boundary 186th row landed in the same katgpt-rs commit | riir-refine (this lane's docs commit) · katgpt-rs (this commit) |
| 2026-10-11 | delta unit (4090/shikuwa idle, riir-refine loop continuation; delta window 10-10→10-11 over the 27 box repos — Protocol H: 36 fetched, sibling-hot left as-found: katgpt-rs behind-24 + dirty bench-621 lane, riir-infer 22-file GPU WIP; THIS ROW landed via an origin/develop detached worktree because the local katgpt-rs checkout is sibling-hot — zero touch to its tree) | 4 broken links → 0 (6817 files): riir-ai proposal-055 ×3 depth errors · reflex-site dev_flow mirror link fixed AT THE CANONICAL SOURCE (instinct .docs/05_resources, the resources-figcaption GitHub-URL convention; byte-identical mirror law holds, sync_mirror 32/32 green, manifest sha re-recorded) · refine kernel_opt.md ledger-gap disclosure re-derived (208–214 + 216–223); delta landings verified self-doc'd (ai 1049 arc in-commit docs row, reflex probe-delta-ab AGENTS bullet, seal-remake plan-025/issue-052 rows, chain/dapps/deployer/rethink/reflexer docs-or-mechanical); SKILL.md 60KB trip → pruned 18 → 15 rows | ai `60875d7ec3` · instinct `bb57ace` · reflex-site `c7e4441` · refine `c98142ef` · katgpt-rs (this commit) |
| 2026-10-10 | delta unit #4 (M3 ~2x:xx +07, riir-refine idle-loop overflow-handoff continuation; Protocol H light: 34-repo fetch all behind=0 in-sync (riir-train behind=1 = the live sibling lane, left as-found), zombies none, worktrees clean, disk healthy (SDXC 282Gi); sibling-hot untouched: refine mining lane, train Plan-454, seal-remake 052, sge plan lane, ai dirty-8) | 6+-repo delta scan CLEAN 0 gaps — every landing since the 19:10 unit self-doc'd or is a scan unit itself (chain 70a3be8d = delta unit #2, sge 2997cf90 + train 4f72f6e8 docs in-commit); 2 DEFERRALS: the seal-remake issue-052 arc (998fa49eb→e0200b4ed→ab9315760 quarry/rest lane = the LIVE 052 session's lane, partition rule) + riir-train behind-1 sibling lane; linkcheck not owed (morning full sweep TOTAL 0; today's post-scan .md landings are record files, not link-bearing book pages) | katgpt-rs (this commit) |
| 2026-10-10 | delta unit #3 (M3 ~20:1x +07, sge proposal-008 thread fall-through — plan 347 owner-go gated so the unit is housekeeping; sibling-hot untouched: infer deltanet 3-file WIP, game-sdk Cargo.lock) | 6-repo delta scan CLEAN 0 gaps (infer/instinct/ndb/chain/dapps/game-sdk — every landing since the morning scan self-doc'd: infer Issue-040 HISTORY+docs-001-§19 verified by grep, chain dapp-vessel rows = the morning unit itself, dapps cd84dbb consolidated record, sdk 1892808/74c11f9 fix-commits-are-records) + PR #29353 (GDN chunked prefill) re-checked: still OPEN/unmerged, riir-ai Issue 971 re-arm NOT FIRED — NEW intel: #30087 merged upstream 10-07 (already distilled in refine Batch 213) and the reviewer's re-baselined expectation is ~5–10% residual (not +13.3%), kernel won't fire under MTP/spec-dec (K>1) — the ~1.04× margin-compression estimate from the 10-07 row softens to ~1.06–1.12× when it lands | katgpt-rs (this commit) |
| 2026-10-10 | delta unit #2 (M3 ~19:0x +07, dapp-vessel "do remains" handoff continuation; the lane's verdict had completed in a later session — plan 067 CLOSED AGREE, transport wiring c0f1cde7 landed; Protocol H light: disk 224Gi, no zombies/worktree debt, chain+ai fetched in-sync; sibling-hot untouched: ai quest_combat 6-file WIP) | riir-chain dapp-vessel content gap closed — the 10-10 morning run fixed links only and the book still had ZERO lane mentions: overview module map gained the 'Dapp-vessel lane' section (src/dapp_vessel.rs) + daemon rows mcp_dapp_vessel.rs / dapp_vessel_serve.rs · overview transport.rs row un-rotted (0xA2 latent read + 0xA3 dapp-vessel; 'unassigned 0xA1–0xAF' → RESERVED 0xA4–0xAF) · MCP matrix 0x40..=0x4F row names 0x43–0x45 · README gained the 'Dapp-wasm vessel lane (default OFF)' subsection (Plan-066 shape) + chain_mcp_* family row · chaind doc 2 feature rows (066 precedent). All claims verified against code (Cargo.toml implication lists, with_dapp_vessel_ingress, drain cadence, config flags); linkcheck chain 0 (workspace 4 = standing sibling-hot deferrals) | chain `70a3be8d` · katgpt-rs (this commit) |
| 2026-10-10 | full-workspace delta scan + git sync (M3, 35 repos fetched all in-sync behind=0; 4090 gitsync fired; sibling-hot untouched: ai raw_wake lane src, dapps 32-file src WIP, refine 5-file WIP, reflex bench-137 artifacts — all doc surfaces clean, all commits pathspec'd) | 5 doc gaps + 1 soft note + 16→0 links: riir-ai AGENTS crate row + HISTORY row for the riir-mcp-client `dapp_vessel` envelope (2669592a8+49f3f13b9 landed doc-silent; records only in chain/dapps) · seal-remake HISTORY rows for plan-341 runtime strike-timing (a4387ec) + balloon polish (b466b1b) · seal-std README names seal-anim FORMAT_VERSION 2/Take.hop (74b623d) · riir-kat AGENTS feature table gains pay-wire index mirror (b195b9e) + kat_ua (b1eb3a8) · dapps AGENTS §Status gains issue-122 webhook bullet (T4 accepted live, docs-silent); 16 broken links across 10 repos — mostly depth errors (written as if from repo root) + removed-issue delinks — linkcheck TOTAL 0 after | ai `2778c6195`+`68c921db8` · seal-remake `fa99b54` · seal-std `ffefe6b` · riir-kat `4aa2947` · dapps `4c54b0c`+`04186d7` · katgpt-rs `3ad211b14` · reflex-site `7c27266` · chain `a42de6af` · infer `dd7261f` · ndb `ef67648` · refine `d6bca08c` · train `b85e5871` · katgpt-rs (this commit) |
| 2026-10-09 | delta unit #2 (M3, Runetrace Phase 6 continuation — the session LANDED seal-remake Plan 023 `cf3c1b4`, the move-family rows, then ran this doc-sync; sibling-WIP untouched: riir-ai Cargo.lock, katgpt-rs .research/614) | Runetrace arc doc-sync — 3 silent landings fixed: seal-remake HISTORY gained the Plans 021/022/023 player-style consumer row (landed HISTORY-silent) · riir-ai HISTORY gained the Plans 626/627/628 substrate row + AGENTS mmorpg row gains `runetrace` + `player_style_shard` (both doc-silent) · katgpt-rs §145 runetrace catalog entry (Plan 624: README/.docs/HISTORY all ZERO-mention; count_features green — opt-in rides the '120 more' bucket, no gate red, the gap was catalog-coverage only) | seal-remake `cf3c1b4`+docs · ai docs · katgpt-rs (this commit) |
| 2026-10-09 | delta unit (M3, dapps issue-121 overflow-handoff continuation; repo fetched synced-clean; sibling-hot untouched: the issue-122 verdict lane landed docs-only `388eb21`/`f5bd194`/`4288ea9`/`7cb4e29`, no overlap; the 121 remainder is owner-gated — SLIP-39 seed, per-lane drain windows, env arming, deploy) | dapps D8 3rd-reset Day-0 baseline RECORDED (bench 018 §3rd reset, read 05:51Z same-UTC-day as the `b0d359b`→`027bbaa3` deploy: reconcile all-seven-green, deposit leg `up_to_date` armed+idle counters 0, GONE arms pinned 410 devnet/308 apex, per-PASS-bar day-0 verdicts, Δ vs the 2nd window = only the arming's own diff) + the queued 4th reset DISCLOSED (undeployed `a0e2121` = issue 121's pay-lane diff — the next devnet deploy restarts the window; owner-timing input recorded in bench 018 + issue 117 §D8) + 2 doc gaps: AGENTS §Status gained the missing Issue-121 line (landed 05:47Z docs-silent — HISTORY had it, AGENTS didn't) + README module table gained the `kat/rails.rs` row (rails claims verified against `RAIL_DOMAINS` + `PAY_INDEX_MOD` before writing) | dapps `9fd94c5`+`9e13c1f` · katgpt-rs (this commit) |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
