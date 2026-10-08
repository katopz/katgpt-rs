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
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Pruned 2026-09-21: 119 → 15 rows, 104KB → 51KB** — this time the one-line convention HELD but the cadence didn't: ~8 rows/day × ~1KB/row re-bloated the file in 10 days (67 file-touching commits 09-11→09-19).  **Pruned 2026-09-23: 25 → 15 rows, 59.1KB → 49.2KB** — trip-preempting at 59.1KB, ~2 idle-pass rows under the 60KB threshold (the 04:49 idle-pass proximity note; the ~8-rows/day cadence lesson stands). **Pruned 2026-09-28: 25 → 15 rows, 56.9KB → 50.4KB** — same trip-preempt precedent at a full-gate run (fetch sweep 26/26 + linkcheck 0 breaks / 5894 files + dirty sweep all sibling-WIP). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`.

| Date | Scope | Verdict | Record (primary fix commits) |
|---|---|---|---|
| 2026-10-08 | delta unit #2 (M3 idle ~09:4x +07, riir-refine idle-loop continuation; Protocol H clean — all repos fetched in-sync, no zombies, no prunable worktrees, 4090 gitsync DONE 09:27; census still running — zero-cargo units only; hygiene: refine Issues 153+148 closed into HISTORY rows, files removed per the noise rule) | 3 doc gaps in riir-chain (landed after the 10-07 chain unit): overview module-map row for `shard_assignment_solver.rs` + README `shard_solver` umbrella row (Issue 164 item 4, 992cbf0b) · chaind doc `riir-chaind token` CLI block + `chain_token_cli` feature row + SDK `token-registry` crate-table mention (Plan 041 tails, 56f790f4); linkcheck 2 = the standing refine plan-202 deferral, 0 new; issues triaged: 089/150 owner-gated, 133/139/140 deferred-trigger trackers, ai pool all gated (1033 stays = the only T3–T6/T8 tracker), rethink 028 T7 owner-gated tracker | chain `599b4ea4` · refine `4101b30b`+`06fb1ea2` · katgpt-rs (this commit) |
| 2026-10-08 | delta unit (M3 idle ~08:1x +07, katgpt-rs 917/920 handoff continuation; Protocol H clean — 33 repos fetched all in-sync, no zombies (the 07:44 riir-esp32 fetch pair completed on its own), no prunable worktrees, 4090 gitsync DONE 07:50; disk 76Gi internal BELOW the 100Gi gate — top incremental slices (sge 41G, refine 35G, chain 28G) all in LIVE sibling sessions' repos, deletion deferred with reason, disk-clean contract needs user approval anyway; queue triage: issues 1–2 EXHAUSTED everywhere — owner-gated (dapps 111/117/119-T10, rethink 025/028-T7, neuron-db 627-T4b, infer 1003-record, auth-005 audio-deferred, katgpt 620 consumer-gated, ai 607/602/528 Ultra-gated, ndb-333 owner window), contended (riir-kat 003 T13 lane, infer 1005 Drex session, refine plan-202), or M3-measurement-gated (the 920 run holds ~3 cores nice-19 for ~21h; instinct 008 + ai-1004 league are measurement units); zero-contention unit picked: doc-sync) | 3 link/coverage repairs: katgpt-rs feature-catalog §141 gained the 917-T2 decode-wiring row (cousins + eligibility join landed 10-08 docs-silent by the landing session's own lineage — this closes it) + pending line now names only the lane A/B; katgpt-rs bench 923 owner delink (removed issue, HISTORY record); neuron-db riir-rag carve link tail — 524/526/523/575 + crate doc repointed cross-repo to riir-ai, pre-carve 325/324 back-links localized (8 links, the Issue-1038 session's residue); riir-shader proposal 002 sge plan-241 citation goes folder-level (file removed + number REUSED — delink, never repoint). linkcheck 12→2 (the 2 = refine plan-202, sibling-hot, deferred). Verified self-doc'd: rethink 030 egemma NO-GO (full HISTORY row), deployer 014 sleep_after (full HISTORY row), dao 008+B5-mirror (both rows), dapps v4 ingest (AGENTS row). 920 run healthy: probe 75/2000, 25658 tokens byte-identical checkpoints ×3, deltas_f32.bin 250MB, manifest still 0 (buffered, known) | katgpt-rs `b3464a45e` · neuron-db `d35acd2` · shader `8278716` · katgpt-rs (this commit) |
| 2026-10-07 | delta unit (M3, riir-chain Plan-066 overflow-handoff continuation; the doc-sync trigger IS the 066 landing — chain book had 2 audit rows only; sibling-hot left as-found: ai + .wprobe + reflex-site per prior rows) | riir-chain: overview module-map row un-rotted (dead chain_prog_nft dropped; escrow_multi/token_registry/domain_registry/delegation added; header 20→19 + the two 066 features named) · architecture.md program table gained the 7 missing live programs (Curator 20…Subscription 26) + NftProgram row dropped (Issue 125) + src tree current + audit filename fixed (escrow_multi.rs, was multi_claim_escrow.rs) · README Feature Surface section for programs 25/26 (wire 97–103, G1–G4, default-off, consumer = dapps 044-E owner-gated, v2 k-of-n additive) · AGENTS + audit pin counts 20→22 live · chaind doc 2 forward rows · 03_economics KAT-issuance (flow-vs-stock) paragraph · linkcheck: chain 0; train 6→0 delink R3 (removed Issues 487/570) — prior rows deferred those 6 as routing-restricted; train found quiet this run (clean + origin-synced, no listed sibling) so the delinks landed · deferred: ai 4 + wprobe 4 (sibling-hot), reflex-site 1 (claimed file) | chain `5d858c73` · train `ade74e26` |
| 2026-10-07 | delta unit ×2 (M3 idle; two overflow-handoff sessions back to back; 8 quiet repos fetched all in-sync — chain/dapps/sdk/infer/reflex/neuron-db/reflexer/katgpt-rs; sibling-hot left as-found: ai ×2 lanes, rethink ×2, refine ×2, seal) | unit 1: katgpt-rs feature-catalog §141 updated for the 917-T3 landing (catalog row still said "promotion pending"; T3 shipped 10-06) · unit 2: 8-repo delta scan CLEAN — every landing since 10-05 self-doc'd in-commit (HISTORY rows/AGENTS compactions/README bullets; reflex 623-T5 artifact adoption carries its full HISTORY row incl. the SDXC1TB durability finding); riir-train Plan 416 T2.5 close-out landed (Bench 626: no-harm proven, plasticity_lr STAYS OPT-IN; issue 570 resolved+removed per its own contract, 604 closed); llama.cpp PR #29353 (GDN chunked prefill) re-checked — still OPEN/unmerged, riir-ai Issue 971 re-arm NOT FIRED; when merged the +13.3% 4090 pp2048 Qwen3.8-27B claim compresses our league prefill margins (~1.176× → ~1.04×) — next fork re-pin should expect it | katgpt-rs `a2adb8797` · train `e393c216` · katgpt-rs (this commit) |
| 2026-10-07 | delta unit (M3 idle ~10:0x +07, riir-refine idle-loop continuation; Protocol H clean — 32 repos fetched all in-sync, no zombies/worktree debt, disk 211G/502G, 4090 gitsync DONE 08:58; sibling-hot untouched: ai 5d + train 2d + rethink 1d + mmorpg 2d + sge 1d dirty, reflex-site dev_flow.md claimed by its WIP session, refine/dapps Plan-066 lane) | 3 doc gaps + 5 linkcheck repairs: game-sdk riir-e2e README gate rows re-derived from test_gate.sh (13 rows — was stale ~08-28 at 6 rows/every count moved: default 21→29, adaptive 30→42, game 36→45, settlement 45→54, prod 172→187, item 180→195; +7 missing: aoi:47, backstab:38, backstab-anchored:41, backstab-arm-link:43, nft:240, shop-chaos:35, market-gold-chaos:37) + aoi/backstab runbook lanes + AGENTS member row gains the Plan 617 Phase 3 backstab chaind link · riir-kat AGENTS feature table gains the cegal-only wire (114 T6, dapps cut-over 4f2a8da) + purge domain (119 T12) + kat_cegel_client row · sealm-toolkit README sprite paragraph six→thirteen (rain 311 + RUN attack 314 sets) · linkcheck 16→11: dapps delink removed 116/120 + repoint 117 relative depth, refine delink removed 146 in Proposal 020; deferred with reasons: train 6 (routing-restricted), ai 4 (sibling-hot), reflex-site 1 (claimed file) | game-sdk `809db28` · riir-kat `aa82302` · sealm-toolkit `ceba59d` · dapps `b9c7a32` · refine `4cd02276` |
| 2026-10-06 | delta unit (M3 idle ~09-11 +07, riir-refine overflow handoff continuation; 27 repos fetched all in-sync; sibling-hot deferred: dapps B7 live-smoke lane + kat wire (the live pack/pay session), refine B7, train GGUF-hunt, ai 1017/1020/1022 open-issue lanes, instinct 013 re-baseline (b) pending) | 2 doc gaps fixed: katgpt-rs kv_compression.md book copy followed the 883 promotion late (`fitted_v_reconstruct` DEFAULT-ON since 34145bfd5, book still said BOTH OPT-IN) · neuron-db 10_local_kv gained the 10th capability (scan_after_prefix from-cursor twins + scan_rows_returned read-cost instrument, dapps 116 T2 — folder count 9→10, store-doc section, index row, AGENTS folder row). THE UNIT'S REAL WORK: reflex issue-number dual allocation caught + remediated — the substrate-first audit (50ddbb7) filed SplitMix64-duplication as issue 070 while 070 was already allocated (1d2befe eval-seat alloc surface, closed e0c43c7, ~10 cross-repo mentions); first holder keeps the number, audit issue renumbered 070→071 (reflex 66a195e: file mv + highwater 70→71 + in-code comment repoints) + katgpt-rs run-log row repointed (a53ee7bb5) — lesson: ls-only allocation misses a CLOSED-AND-REMOVED same-number holder; read highwater AND the removal log (git log --diff-filter=D) before allocating | reflex 479b042+673ce4c+66a195e · katgpt-rs a53ee7bb5+c2ee7dcba · neuron-db 8c7b781 |
| 2026-10-05 | delta unit #3 (M3 ~12:3x +07; web-half of the ESC/typed-head additions per owner ask "new addition should update both doc-sync and website especially /resources, also check rethink.gist.rs"; repos fetched in-sync; sibling-hot left as-found: riir-rethink mid-flight Issue-023-T2 XNLI-ESC-demotion lane (arsenal.toml + escalating_backend.rs + esc tests dirty — the demotion my copy is written to survive) + reflex-site docs/instinct/dev_flow.md link-repair WIP + katgpt-rs src WIP) | 3 site/doc gaps fixed: (1) riir-rethink .docs/05_resources/resources.md said storefront "incoming / link may be dark" — live since 10-03; reworded + gained concept-level "How escalation works" (per-suite arming + rate guard + receipt-named demotion; no suite names, no numbers — mirror-fence-safe, survives T2) → mirrored docs/rethink/resources.md + manifest sha in reflex-site 06922a5 · (2) reflex-site /resources Rethink section gained the same fact ("Escalation is built, not just drawn"), moat-law-clean | reflex-site `06922a5` · rethink `22e1157` · (3) the rethink storefront feed was STALE — site/data/why.json built from bench sha aa4bc12e while the published feed moved to f82f648b at the Bench-123 publish (reflex-site 3529710); rebuilt via build_why_data.py, Rethink cells byte-unchanged (only provenance stamps moved), deployed rethink fc52b2a4 + reflex-site 5e556e4e; en-route gate repair: resources smoke's numbers-law scan flagged the footer licence line "Apache-2.0" (pre-existing since 007-T1, licence name ≠ measurement) — digitFindings now strips licence identifiers by name; gates: sync_mirror 32/32 --check + resources smoke PASS + public_copy_gate PASS; rethink.gist.rs check verdict: current, feed-driven tiles, no ESC/typed-head copy drift — rethink `35b3296` |
| 2026-10-05 | delta unit #2 (M3 ~05:2x-07:1x +07, overflow-handoff continuation; Protocol H fetch 27 repos 12.2s / disk 124Gi free; queue triage: every handoff item owner-gated or sibling — 627 T4 owner-adjacent (in-issue), 625 ND1/ND2 owner, reflex-site 007 T5 owner commitment, refine 146/147 sibling lane, 089/102/133/139/140 trigger-gated untriggered, dapps 112 owner tag, katgpt-rs 917 blocked-on-train-437; clean-behind riir-infer FF'd fe75497 (sibling 013-T4 lane — its doc notes deferred to that session) + katgpt-rs ff'd 618d7ec71 (fitted_value_table.rs only, zero overlap with the sibling's 5 dirty src files)) | 3 doc gaps fixed: neuron-db README raw_wake_ring feature row (the 626/627 arc + Benches 500/501 landed README-silent) · refine AGENTS stale arc-swap fork-row claim (dropped 10-03 fd9aead7/fa904dde7 — prose still asserted it stays) · reflexer HISTORY vessel-mint genesis row (vmint-keygen + stub worker + GENESIS.md keys, e81d66d+d29de0b+4b40003 record-silent); THE UNIT'S REAL WORK: workspace linkcheck 167→0 — the 10-01 riir-clippy→riir-refine rename left 95 inbound links stale across 7 repos, the 10-03 rethink split moved ~25 targets (moat/ arsenal proposal, benches 039-041 stayed instinct-side), + depth classes + removed-issue delinks (898/914/106/571/55/56/64/30); 15 commits across 15 repos via link_fix R1/R2/R3 + hand fixes; lesson: pre-translating finding targets breaks the tool's line anchor — sed the files first, re-run linkcheck, then fix | neuron-db 9150198+6659521 · refine feffe68e+545a835a · reflexer b5acd6c+bb18884 · ai 1dc8398a5 · dapps fdcb29a · train 25e9b580 · dao b4d6170 · reflex 6365fab · chain 2efd8192 · deployer fc25053 · infer 3f67035 · instinct e196573 · rethink 0c2c68b · reflex-site 3baffcc · katgpt-rs 10345b6cc |
| 2026-10-03 | delta unit (M3 idle ~11:0x +07, riir-refine idle unit post-Protocol-H; trigger = Plan 618 T2 landed 04:5x today (neuron-db `2b53ed7` + ai `16276e3e2` + the vessel-attest wiring) with the landing session's coverage verified partial — crate-module row only; boundary 166th scoped row committed alongside; sibling-hot untouched: katgpt-rs 915 lane, train Plan-614 lane (routing restriction), reflex Bench-111 head self-doc'd, seal-remake HISTORY rows in-commit) | 2 gaps fixed: riir-neuron-db `abstraction_freeze_gate` (README feature row + consolidation_pipeline methods row + classified-capture section + folder index; claims verified against `phase_gate.rs` MIN_WAKE_ABSTRACTION 0.5 / G2 750 ns) · riir-ai (AGENTS engine row — the whole 618 arc was doc-silent: engram_runtime_store T1 + abstraction_producer T2.3; AGENTS mmorpg row — vessel_attest_abstraction T2 wiring; cognitive_branches_runtime.md module table + count 15→17) · 1026 arc verified self-doc'd (closed+HISTORY) · chain 161 closed, 160 defer-marked self-doc'd | neuron-db `015850d` · ai `fb7657afc` |
| 2026-10-01 | delta unit #3 (M3 idle ~23:1x +07, overflow-handoff continuation; riir-refine ff'd +3 → `141c6109` — the live batch session's Batch 198 PrismML few-row landing, all four commits self-doc'd by construction: `.research/231` + `.plans/196` + AGENTS kernel_opt 734→739 / rust_perf 167→168 re-pins + README + `.distill` snapshot + domain docs; session LIVE per Protocol A — progress notes deferred to it, never written by another session; sibling-hot untouched: seal-game-editor map-tab build, surveying-sibling grammar-tests lane; boundary-guard 164th row committed alongside) | CLEAN — 0 gaps in riir-refine; no other repo deltas this window (all 0/0 synced except sibling worktrees left as-found) | katgpt-rs (this commit) |
| 2026-10-01 | delta unit #2 (M3 idle ~18:5x +07, post-handoff; all repos fetched — every landing since the 07:2x baseline is the Plan-192/Proposal-018 rename wave carrying per-repo docs companions, perf-lane self-docs (ai 1004, infer 032/998/1004, instinct 017/035, reflex 057/058), dapps owner-ask HISTORY rows, or mechanical chores; riir-train read-only per routing restriction — its 9 landings all docs-companioned; sibling-hot untouched: refine Batch-197 + driver wiring, reflex-site edits, ai perf lane live at 18:14) | CLEAN — 0 gaps; game-sdk `a3a82b5` (backstab e2e) verified covered cross-repo via ai-1020 status + Plan 617 5.3; run log 57.3KB < 60KB no prune | katgpt-rs (this commit) |
| 2026-10-01 | delta unit (M3 idle ~07:2x +07, handoff carry-over observation; 32 repos fetched in-sync except riir-infer behind-1 sibling-owned (dq614 matrix in flight) + reflex 058 sibling-claimed per partition rule) | katgpt-rs "894+ window under-documented" observation REFUTED — every closed issue 894..912 has a HISTORY row (the top newest-first section, lines 4–248; the prior session's tail-read mistook file position for coverage). REAL defect found instead: the 912 row had been appended MID-ENTRY inside Issue 893's body (between its intro and its task bullets) and at file-end against the newest-first shape — relocated to the top, 893's entry re-joined | katgpt-rs (this commit) |
| 2026-09-30 | delta unit #2 (M3, post-C1-handoff continuation; 26 repos fetched/synced, instinct+train behind-1 FF'd — the sibling's 426-T4 4090 cross-check bench 030 + the C2 sibling's untracked 602/surrogate files in train, both in-flight deferred; the prior unit's "picked up NEXT: reflex 055" thread landed CLOSED by a later session — 055 null + 056 spun+closed, HISTORY-only) | 3 gaps fixed: reflex AGENTS gained the 055/056 arc (mc_ensemble opt-in + --mc-ab + the --gate-fit-calibrated DEFAULT posture, Benches 092–095 + katgpt-rs 909-911) · instinct AGENTS gained the 014 C1 row (Bench 029 encoder lane serve:✗ live, the T8 width-guard regression fix) + 008's typed board refreshed to the SERVED H2 0.6475 (0.6300/+5.8/−14.1 → +7.5/−12.4, site-verified) · katgpt-rs §113 sigmoid_calibration gained the 909/910/911 solver-repair record; DEFERRED: riir-ai 326e720f8 (045-arc half-landed, seal-remake sibling re-running) + mmorpg soak boot #3 (issue 1002 in-flight); run log 55.3KB < 60KB no prune | reflex `2f6b58c` · instinct `6218dbc` · katgpt-rs (this commit) |
| 2026-09-30 | delta unit (M3 ~01:0x +07, post-reflexer-002 handoff; 15 tips checked vs the 09-29 baseline — delta CLEAN: reflexer 60408d2 self-doc'd (HISTORY row + issue removal in-commit), clippy 99515251/katgpt-rs 0fffc3d7b docs-class, seal-remake f1e5099 + infer befcaa2 both carry HISTORY.md in-commit, sge de18b1ae follows its own docs(279) ledger, ai/train/instinct tips = the sibling's ACTIVE tetris round-5 arc (invalid-build amendment + warm-epochs fix) — deferred by rule; run log 54.6KB < 60KB no prune; picked up NEXT: reflex Issue 055 DRM distributional layer (unclaimed, ungated) | CLEAN — 0 gaps; every landing docs-class, self-doc'd, or sibling-in-flight | — |
| 2026-09-29 | idle unit (M3, ~04-05 +07; box LOADED throughout — twt sweep bursty 956→95→1070% + infer sibling benchmark arc, so Bench 961 re-run + Plan 182 T18 stayed preflight-gated; 32 fetched all in-sync; doc delta CLEAN — tips newer than the prior unit are all docs-class or mechanical: mmorpg 0695cd6, train 49e8d0f4, seal-remake 733e53a, reflex cfa132a, clippy 95e5a14d) | the unit's real work: **riir-instinct Issue 012 PICKED UP + LANDED** — the frozen-artifact staleness probe (Plan 005 / Bench 022): `src/staleness.rs` + fixture (192 items, digest 98a68396…) + example + 6 gates; MEASURED NOT dead-by-domination (banking77 v1→v2 bridge drift 13/64 flips mean |Δgold| 0.240 FIRED; identical pairs exact 0.0); report-only, swap-hook + refresh arm deferred; en-route: stale `.benchmarks/.highwater` 0015→0022 + pre-existing server.rs needless_borrow in passing; code-smell sweep 6 more quiet repos (sdk/seal-remake/kat/viewbridge/esp32/llm-adjacent) — 0 TPs, seal-remake's 3 C2 = the known marker-downgraded trio | instinct `d866047` |
| 2026-09-29 | delta unit #2 (M3, quiet-box queue still gated load ~13 — twt sibling sweep + riir-infer benchmark arc; work units: instinct Issue 010 closed+removed — site half verified reflex-site `74b49e4`, residual owner decision → pickup 013, AGENTS stale open-half row fixed (`b6eb4c2`) · reflex doc-sync — openthai + 426-T5 corpus-synthesis lanes into AGENTS Build Commands + HISTORY 09-29 row (`021486b`) · reflex-site 003 T5 hybrid half resolved by SHA mapping — `0959928` = instinct Bench 002 in-repo, `634093f` = reflex HEAD (`2a5ed55`) · ai 971 09-29 re-check still NOT FIRED — +15 prism commits incl. sycl PQ2_0/PTQ1_0 FP16 dequant + metal FWHT-SwiGLU fold + DFlash2 spec, ZERO GDN-prefill-family/rebase, 09-24 CUDA cells stand (`a81a2fc60`) · code-smell hunt completed over all quiet repos — chain/dapps/dao/deployer/neuron-db/mmorpg/sdk-lite/shader/llm clean, seal-online-remaster 2[B]+7[C1]+29 stillborn REPORT-ONLY (owner-DEFERRED repo, no fixes filed) | 2 doc gaps fixed (reflex lanes; instinct 010 hygiene) + 1 stale residual relocated; trigger verdicts re-confirmed, no state flips | instinct `b6eb4c2` · reflex `021486b` · reflex-site `2a5ed55` · ai `a81a2fc60` |
| 2026-09-29 | delta unit (M3, overflow handoff; quiet-box items preflight-gated — load 12-13, sibling twt sweep pegging ~10 cores; 32 checkouts fetched, all in-sync except seal-game-editor behind 30 known sibling WIP; incoming landings all self-doc'd by construction: 1016-pickup wave ×6 repos, plan-426 arc (train docs + reflex 8426cef), instinct bench-021 + docs row, seal-remake plan-015 arc + plan-016 amend, riir-infer 022 T5.0 sibling-hot untouched, riir-clippy mining walk sibling-hot) | 2 gaps fixed: riir-ai deltanet showcase gained the f32 CPU-reference row (Bench 961 NOT-PROMOTED verdict — the landed negative was showcase-silent, GPU rows only) + sdk vessel_pipeline A2 basis-gate row resolved-by-retirement (D11 `0df869d`, was still listed open) | ai `c43b5ebb2` · game-sdk `22948a1` |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
