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
| `riir-clippy` | **12 numbered folders** (`01_orientation/` … `11_domains/` + `12_ane/`; the 12th added for the Apple Neural Engine substrate knowledge — private runtime API, MIL/blob formats, M3 Max findings, the Rust-bridge negative result + working ObjC substrate — riir-ai Issue 726 T0 distillation), unnumbered files inside. The code-healer vault — corpus/drafter/pruner/verify/self-evolve/domains narrative. `AGENTS.md` carries the batch-mining progress notes (the sweep record home for cross-repo clippy heals). | Status + Quick Start + Usage + feature gates. | same | `develop` |
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
New rows append ONE compact line each (hashes + one phrase — never narratives; the 09-05 and 09-11 compactions both followed regressions to full narratives). **Pruned 2026-09-21: 119 → 15 rows, 104KB → 51KB** — this time the one-line convention HELD but the cadence didn't: ~8 rows/day × ~1KB/row re-bloated the file in 10 days (67 file-touching commits 09-11→09-19).  **Pruned 2026-09-23: 25 → 15 rows, 59.1KB → 49.2KB** — trip-preempting at 59.1KB, ~2 idle-pass rows under the 60KB threshold (the 04:49 idle-pass proximity note; the ~8-rows/day cadence lesson stands). **Maintenance rule: whenever this file exceeds 60KB, prune the run log to the newest 15 rows** (checked at any full-gate run); every removed row is recoverable via `git log -S '<date>' -- <this file>`.

| Date | Scope | Verdict | Record (primary fix commits) |
|---|---|---|---|
| 2026-09-24 | delta unit (M3 idle ~02:3x +07; orphan-check re-verified closed first — issue-131 dapps half landed `82933de`/`eefd8cd` + Plan 037/038; accepted the 01:2x "Plan-038 arc self-doc'd" verdict, zero new rows) | 2 stale-claim fixes instead (Step-5 class, both passed over by the row-coverage scans): clippy AGENTS `--stats` row's "auto-domain runs leave `by_domain` honest-empty" false since `5f00d2ac` → pin > compiled (`fold_rule_domains`) > honest-empty-when-compiled-out; dapps AGENTS Plan-037 row still described the live card as per-gen small-multiples + created-rate → struck + Plan-038 supersession amendment (day-keyed live shape, plan link) | clippy `c63386ec` · dapps `e8e769f` |
| 2026-09-24 | delta unit (4090 idle ~01:2x +07; post-21:0x-M3 baseline; syncs pulled katgpt-rs +8 / dapps +7 / chain +1 / mmorpg +2 / ndb +2; sibling-hot untouched by rule: shader vfx-tornado WIP · ndb+katgpt-rs+dapps+chain owner-gates-m2 wave 23:4x · katgpt-rs Plan-607 lane active to 01:19) | CLEAN — zero gaps: katgpt-rs 8 all docs-class/self-doc (Plan-607 T13–T16 addenda ARE docs; CI re-arm doc'd in AGENTS trigger-health; alias-seam + citation fixes mechanical); dapps Plan-038 arc self-doc'd + eefd8cd covered by clippy HISTORY row 47 (the issue's home repo — clippy e4dd85f8); chain 3ee5a25 + mmorpg fe485b6 docs-class; ndb install.sh hardening c5c11b5/6f457a6 in-commit self-doc (dist Phase 1 lane); riir-ai tip 00:15 linkcheck docs-class | — |
| 2026-09-23 | delta unit (M3 idle-continuation window ~21:0x +07; post-20:4x baseline; the only landings since were riir-clippy e4829f6f ops_nightly leg 4d + 9e72b760/39667f8d docs commits; dapps Plan-037 lane sibling-hot → Devnet healer owns) | CLEAN — zero gaps: e4829f6f self-doc'd (.proposals/016 status → LANDED with the live-verification narrative; dapps AGENTS.md carries the rider bullet consumer-side; .docs/11_domains sweep fully restored byte-complete by 39667f8d — empty diff vs e4829f6f^); nightly-leg convention (script + proposal status + consumer-side docs, no .docs row) matches the adjudicated 4c corpus leg — not a gap | — |
| 2026-09-23 | delta unit (M3 idle-continuation window ~20:4x +07; ff'd riir-clippy ed7bf3aa→ada0a96b first — the two incoming rows were the 4090 idle passes touching only .distill; sibling-hot deferred by rule: riir-dapps 12-dirty healqual+front-page lanes · riir-ai/train/reflex/esp32 dirty · riir-shader behind-1) | CLEAN — zero gaps: riir-clippy arc fully self-doc'd (74954cfc Batch-173 feat carries the snapshot row; b619e670 .docs sync; ed7bf3aa Issue-131 docs; ada0a96b/8f1cbee8 idle passes ARE docs commits); quiet repos all 0/0 synced, no undocumented landings since the three earlier 09-23 units; esp32 110 board-blocked (owner-deferred), not a doc gap | — |
| 2026-09-23 | delta unit (4090 idle ~20:1x +07, post-Batch-173-landing window; the M3 batch session self-doc'd AGENTS/HISTORY/.distill/.research/.plans in-commit — the .docs book was the gap) | 1 gap fixed: .docs untouched by the landing — rust_perf.md top-block ledger 149→151 (B173 serde-family lead narrative + "Previously 149" transition on B172), Source-section corpus row (7 slices `entries_01..07`, 151 rules / 95 BenchSpecs re-derived from source `bench_spec: Some(` counts 22+20+23+12+10+6+2), 11_domains README index row | clippy `b619e670` |
| 2026-09-23 | delta unit (zed-4090 session ~15:5x +07, post-1001-closeout window; sibling-hot deferred by rule: riir-clippy idle loop 15:30 · riir-infer carve 15:13 · shader vfx-tornado WIP · katgpt-rs 607 T8 15:15) | mmorpg re-adjudicated over the earlier 09-23 "no rows owed": HISTORY rows for the 992 tripwire lane (`2eb70b5`+`4ad7860`), gpu-allocator 0.62.2 (`7889881`), bench drain (`f91885b`), CI provisions (`ccc5f73`) + the 09-16 arc-swap row relocated to chronological position; 2 real stragglers the provisioning commit left: rust.yml inline + AGENTS.md still said seven private siblings (loop has 8) — both fixed; en-route: clippy-sweep dry-runs 7 uncontended repos all 0 oracle-anchored (clean) | mmorpg `ba3e459` |
| 2026-09-23 | delta unit (4090, post-875-closeout window ~07:10 +07; deferred by rule: riir-ai pp-waiter lane · shader 042/043 sibling-hot · riir-infer carve incl. its riir-train Cargo.lock +16 residue · seal-remake/-editor behind-36/-169) | CLEAN — zero gaps: riir-train 569-C9 (`2036acc0`) + 875-t3 (`3dc2b951`) self-doc'd in-commit (HISTORY + issue rows verified); katgpt-rs 875 closeout doc-complete; rest covered by the earlier 09-23 unit or docs-class | — |
| 2026-09-23 | delta unit (riir-clippy idle continuation window; sibling-hot deferred by rule: riir-shader 042/043 texture lane · katgpt-rs 607 · riir-ai/riir-infer consolidation · seal-game-editor worktrees) | CLEAN — zero gaps across 16 quiet repos: dao/deployer/dapps/sdk/sealm/kat/llm/web/seal-remake delta-0; ndb `docs(330)` is the grep's paren-form false-positive (docs-class); chain heal+intake · auth+viewbridge same guard zero-ok-line fix (closes no tracked issue) · mmorpg 7 hygiene commits incl. f91885b prior-adjudicated · seal-online-remaster 7 sweeps/tracked fixes — all mechanical/self-doc class, no rows owed | — |
| 2026-09-23 | delta unit (the 998-handoff session, ~11:0x +07; post-07:10 window) — includes the 09-23 shader 042/043 deferral DISCHARGED as CLEAN (both in HISTORY: `e3c4bdd`/`3b12963`) + katgpt-rs docs_gate 35/35 PASS run en-route | CLEAN — zero gaps: katgpt-rs 3 docs-class (`f50c18ba2`/`c4338be3a`/`59a8be397`); riir-shader `b05a06f` + editor `8b170311` fix commits carry full narrative bodies, close no issue files (self-doc class, no rows owed); seal-remake `6bf7bd7` docs-class | — |
| 2026-09-22 | linkcheck unit (ndb-release-continuation session; P1 re-verified sibling-blocked — the reflex Issue-009 wave holds BOUNDARY.md + src/harness/mod.rs dirty while P1's own law requires the boundary row in the same commit) | 1 finding fixed: ndb HISTORY katgpt-rs 844 removed-issue link → R2 repoint to katgpt-rs HISTORY.md record (line 1166/8477 verified before citing); census 22 → 21 standing (katgpt-rs 1 fresh-removal sibling lane · train 6 routing-restricted · seal-online-remaster 14 deprecated); ndb Phase 1 self-doc delta verified CLEAN by grep (ndb_cli.md --json + build-release/UNSHIPPABLE + BOUNDARY:30) | ndb `8d09268` |
| 2026-09-22 | riir-clippy delta unit (idle window post-handoff; baseline f3816c9b plan-165 close-out; 26 commits scanned — idle passes, distill records, walk11 opening balance, P35 + Issue-128 fixes) | CLEAN — zero gaps: Issue-128 T1–T7 feat carries its HISTORY row (line 3); Research 201 + walk10/11 + snapshot commits ARE the records; highwaters consistent (plans 165 / issues 128 / bench 98 / research 201); no feat without doc coverage | — |
| 2026-09-22 | linkcheck extension (same idle window ~03:05 +07; shader/editor/seal-remake went CLEAN — agents committed + moved on) | 4 more findings fixed in the 3 newly-quiet repos: removed-issue delinks ×3 with dispositions read from HISTORY.md (sge 003 → seal-remake Issue 029 landed; seal-remake 004 → Issue 030 resolved + Issue 026 CLOSED `5613f46` — stale "open, complement" claim corrected) + 1 rename repoint (shader 002 → sge `241_glb_delivery_lane.md`, the tool would have delinked a rename — manual-repoint class); census 25 → 21, remainder = ndb 1 (sibling-dirty) · train 6 (routing-restricted) · seal-online-remaster 14 (deprecated) | sge `5e08950d` · seal-remake `9e1eaf8` · shader `5e33c6c` |
| 2026-09-22 | linkcheck re-sweep (riir-clippy idle window ~02:50 +07; walk-#10 close-out still held to ~07:00 gate) | the census unit's riir-ai deferral DISCHARGED: tree clean + synced, re-census 24 true findings at HEAD → fixed with the EOL-safe fixer — 23 removed-issue R3 delinks (incl. cross-repo katgpt-rs 844 + ndb 622) + 1 too-deep HISTORY R2 repoint (`.benchmarks/948_attn_fa_promotion.md`); the CRLF/mixed trio 383/604/941 CR-counts byte-identical HEAD↔worktree (the `newline=""` repair's live field proof); workspace census 49 → 25, remainder all standing-blocked (ndb/shader/seal-remake/editor sibling-dirty · train routing-restricted · seal-online-remaster 14 deprecated) | riir-ai `e08146808` |
| 2026-09-22 | `link_fix.py` EOL repair (riir-clippy idle window 02:31 +07; walk-#10 close-out still held to ~07:00 gate; tree clean, no sibling collision) | the recorded EOL guard class fixed at the tool: read+write now `newline=""` — CRLF/mixed files round-trip byte-exact; fixture-proven (CRLF + mixed + LF byte-identical outside the fixed line), red-arm proven against the HEAD version (whole-file LF normalization reproduced by the old pair), idempotence second-pass NOMATCH bytes-stable; locale_io/subprocess_encoding/console_encoding gates PASS (0 findings, floors held); live census re-run TOTAL=49 unchanged | katgpt-rs (this commit) |
| 2026-09-21 | delta unit (post-Issue-865-closeout idle window, M3; sibling-hot deferred by rule: katgpt-rs 868 engram lane · riir-reflex birth · seal-remake tornado · editor 186-C2 · riir-train Plan-415 routing-restricted) | quiet-repo scan CLEAN — chain 158 (HISTORY line 2071) · deployer 009 (HISTORY row) · sealm-toolkit 257 (README rows) · ndb/sdk/dapps/dao/auth/kat/shader/esp32/llm/viewbridge/web docs-class or mechanical at tip; **2 riir-ai candidates DEFERRED by rule**: `bonsai2_hadamard` riir-engine default promotion (`e0f5da490`, plan-602 C6 owns the league-doc fold; tracker-is-record convention — the 09-19 riir-gpu promotion landed book-silent and multiple adjudicated units accepted it) + `attn_mass_tap` (`88b571eff`, 1h old, riir-train Plan-415 lane) | — |
| 2026-09-22 | linkcheck census unit (riir-clippy idle window 02:15 +07; walk-#10 close-out still held to ~07:00 gate) | census 68 → 49 across 22 repos; 19 dangles fixed in 3 quiet repos — katgpt-rs 16 (Plan-603 845 rename repoints ×2 manual — the bench exists as `845_distance_abstain_goat.md`, tool's R3 delink would lose the pointer; 5 too-deep HISTORY R2 repoints; 9 removed-issue R3 delinks) · riir-clippy 2 · riir-dapps 1 — census re-run clean on all three; **EOL guard earned its keep twice**: the fixer's read_text/write_text normalized i/crlf `.research/200` (riir-clippy) to LF → reverted + re-applied byte-level (CRLF preserved), same class caught pre-commit in riir-ai (604/383 crlf + 941 mixed) — **riir-ai ABANDONED mid-unit**: 24 findings invalid (census read a moving tree; repo ff-advanced behind the sweep by a sibling's git ops; 941:5 finding does not reproduce at HEAD) → all edits reverted, re-sweep when quiet; untouched-by-rule: ndb/shader/seal-remake/editor (sibling-dirty) · train (routing-restricted) · seal-online-remaster 14 (deprecated standing); sync sweep: riir-train ff-pulled (89799358+15d2a47a, 452 T4/T7 wave) · riir-ai self-resolved by sibling | katgpt-rs `81a42bd92` · clippy `d1f96f11` · dapps `a9cb514` |
| 2026-09-22 | delta unit (riir-clippy idle window 00:59 +07; walk-#10 close-out held to ~07:00 gate; sibling-hot untouched: shader tornado · seal-remake 037/038+editor 186 · katgpt-rs 089/102 measurement lane) | 1 gap fixed: sdk issue 038 close book-silent (file removed `02ba9fe` with NO HISTORY row — the repo's own Issues-033/035 shape) → HISTORY row + the Byte determinism claim scoped (byte-for-byte = the single-image fixture population; multi-image GLBs are the deliberate `38abdf9` divergence — Python silently clobbered same-name textures) + the packer table cell scoped; en-route riir-train ff-pulled to origin (`058531fd` issue452 T7, other-box push); CLEAN: chain (158 HISTORY'd + lint-heal mechanical w/ intake miss) · sealm issue 002 (fix-commit-self-doc, its own 001 convention) · ndb/mmorpg/dapps/dao/auth/deployer/kat/esp32/viewbridge/web delta-0 · train plan415/452 wave tracker-owned | sdk `c680a36` |
| 2026-09-21 | delta unit (riir-clippy only — the Decision-ordered doc-sync; post-delta-unit-#7 window: 2 commits) | CLEAN — zero gaps: `7fc14393` highwater 164→165 numbering repair (mechanical class, no row owed) + `8195d50a` .distill sweep verdict (the snapshot IS the record); highwaters verified consistent (plans 165 / research 199 / issues 127 / bench 98); batches 169-171 + plans 160-165 all self-doc'd in-commit | — |
| 2026-09-21 | full gate, 10 repos (4090 idle unit; clippy+train deferred-by-rule — sibling B168 closeout + Plan-402 WIP) | 10/10 CLEAN, no commit | baseline scan (last `docs:` vs code-after): katgpt-rs/kat/auth/mmorpg/shader/ai zero-after (ai baseline = HEAD `d13203d6b`); ndb `6899333` self-doc (HISTORY + dense_embed_index.md format table in-commit); dapps `ce8b428` + sdk `f9e77e9` mechanical; chain 156/157 HISTORY-covered (`f7eb85e`/`7facc3c`) | 
| 2026-09-21 | delta unit #7 (the clippy idle loop's recorded doc-sync candidate, discharged; sibling-hot: clippy 089/102 lane mid-commit — 2 unpushed develop commits + .distill/001 + fix_verify.rs dirty, untouched, push left to that lane) | 1 gap + census 7→1: clippy seed_corpus.md wholesale stale — header "53 entries" vs 90 actual, narrative stopped at P12/Issue-120 while the Batch-150+ packs landed → de-counted + re-pointed to the compiled truth (corpus-stats law) instead of a third hand-typed count; en-route: corpus.rs module doc proved the class twice ("72 entries" prose + "86-entry" comment vs 90) — enumeration restored through Batch 168, 02_corpus index broken `src/seed_corpus.rs` link fixed, 3 one-../-too-deep depth fixes (AGENTS Research-182 + HISTORY Research-186/Plan-150, all targets verified on disk), 2 removed-issue repoints to HISTORY (Benches 098/099, the Issue-125 RESOLVED/126 COMPLETE rows), 1 late delink (HISTORY Issue-103, surfaced by the post-commit re-run) | clippy `d4669e38`+`1838af74` |
| 2026-09-20 | delta unit #5 (discharging unit #4's deferrals now cooled; sibling-hot persisting: editor plan-251 publish console · clippy idle loop · katgpt-rs 859 lane (1ee1f8cb8 05:47) · train routing-restricted) | 1 gap fixed + deferrals DISCHARGED as CLEAN: seal-remake — the vessel-boot lane (a14c1ed, issue-034 close 592203c HISTORY-only) book-silent on BOTH boot surfaces while every sibling env was documented → AGENTS house-style block (G1c/A0/PUBKEY semantics) + README env rows (`be5900f`). CLEAN: riir-ai DFlash2 (unit #4's behind-7 deferral — Benches 942/943/944 + issue-989 close self-doc on HISTORY ×4 + scripts/dflash2_fidelity/README.md; book has no dflash2 status claim to flip, qwen38_dflash2 never a book feature — investigation class, no row owed) + 5704b8440 numbering repair docs-class (940 dual-allocation → 945; citation-weight read 11-0 for the rotation GOAT keeping 940) · shader BackLight 5613f46 self-doc (un-enumerated-term precedent) · chain 157-arc in-flight tracker-owned · train 7173ac6e docs-class | seal-remake `be5900f` · kat (this commit) |
| 2026-09-21 | delta unit #6 (post-605/991/601-landing window scan; sibling-hot: clippy idle (arxiv walk, 2 dirty) · ndb sibling WIP (18) · train routing-restricted) | 1 arc fixed (riir-static-ai, Plan 601 Shape B / Issue 990, landed `51badf2de`) book-silent on BOTH crate-map surfaces → AGENTS repo-layout row + README Crates row added; the games-shared `static_ai` clauses on BOTH surfaces corrected to forward-only (the vocabulary moved; the feature re-exports) — plus the same-pass count rot: AGENTS "21 in-tree" (last touched 07-12) + README "21 active" → 26, manifest-derived, dated; AGENTS also missing riir-games-mmorpg / riir-net / riir-simloop / riir-host / riir-rag rows (467ac6550 fixed the README family row, the AGENTS table kept the hole) + README missing simloop/rag rows — all one-liners fact-verified against src (manifest_is_self_contained, run_headless/tick_once, Plan 524). CLEAN: chain 156/157 (HISTORY in-commit, files removed) · seal-remake 029/035 (HISTORY rows) · deployer 009 (HISTORY row IS the record) · sdk f9e77e9 write_atomic = hygiene class, commit-message self-doc · dapps 099–104 arc (docs: companions + AGENTS bullets) · katgpt-rs/602-859-860-861 lanes self-doc · leaves zero-commits. Left alone: .proposals/025 "21-crate" (dated record) + prose_proxy.md (test fixture) | ai `e0bf65395` · kat (this commit) |
| 2026-09-20 | delta unit #4 (post-residue window scan ~14h; sibling-hot deferred by rule: katgpt-rs 841/arm_reach lane actively committing (e85f9169f 01:55) · riir-ai DFlash2 behind-7 · seal-remake+sdk+editor vessel-CAS lane (ndb 623's cross-repo halves ride it) · shader 72e0037 blend-state = seal-remake lane's closeout; train routing-restricted) | **CLEAN — zero gaps**: ndb 623-close self-doc (0808796 HISTORY row + file removed; artb wire halves 14fee59/b657a4b inside the arc) · dapps 4ebcf0d/f956e3e/60513b9 docs-class · clippy 09-20 docs-class wave (distill verdicts + Issue 125) · train 35b44973 docs() record · kat 56f79c6 trivial-test class pulled (no row owed) · chain/deployer/auth/viewbridge/esp32/web delta-0. En-route own work: editor residue — issue 146 removed (FIXED 57048290; 164 verified genuinely OPEN, T1/T2 remain, kept) + HISTORY truncation polish ×44 completed from removed issues' statuses + 2 dangling-path fixes (07ffe869/3fd245cb/50cd518c) | editor `07ffe869`+`3fd245cb`+`50cd518c` · kat (this commit) |

## TL;DR

Diff git history against the last `docs:` commit. For each landed plan/issue,
write the matching README/docs entry using the repo's existing shape and the
verdict from its benchmark file. Commit with `docs:` prefix on the working
branch. Never upgrade a verdict without proof; never delete a negative result.
