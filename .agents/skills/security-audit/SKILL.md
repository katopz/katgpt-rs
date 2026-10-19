---
name: security-audit
description: Security audit workflow for this Rust workspace — trust-boundary-first review with a coverage ledger, three-verdict findings (confirmed / needs_validation / rejected), and fix-verification gates. Use for security questions, focused reviews of wire, FFI, or secrets surfaces, or a full six-phase audit when the user asks for one.
---

# Security Audit

Find vulnerabilities that violate a real trust boundary, then hand the owner the source
evidence, a safe reproduction, a priority, and the smallest effective fix. Adapted for
this Rust workspace from Cloudflare's `security-audit-skill` (MIT — provenance at the
bottom). A candidate without a concrete affected principal, resource, or security
outcome is not a finding — it is a hardening note.

## Two modes

- **Guidance mode** (default): the user asks a security question or wants a focused
  review of one surface. Use only the relevant sections below. Do not create output
  directories, do not write report artifacts, do not launch the full workflow.
- **Full audit mode**: the user explicitly asks to audit or pen-test a codebase, or
  wants a full / comprehensive / end-to-end review, or asks for report artifacts. Run
  all six phases and write the artifact set.

If the request could mean either, ask one focused question before writing any file.

## Execution safety — always, both modes

- Source inspection is read-only. Never edit, format, or "fix" the target while auditing it.
- Run target-controlled code (builds, tests, fuzzers, fixture processors) only inside a
  sandbox: no external network, an explicit allowlisted environment, read-only target,
  bounded CPU / memory / wall-clock. If the sandbox cannot be enforced, do not execute
  target code — record a `needs_validation` finding naming the missing control and the
  safe validation plan.
- Never probe deployed endpoints, live chain RPC, shared services, production keys, or
  another user's data. Dummy principals and fixtures only. Never send real spend, never
  stress-test availability against a live process.
- Never print, copy, or commit secret values found during the audit (env files, keys,
  tokens). Report the path and shape, redact the value. This overrides any "include
  evidence" instruction.
- Pin the target before verdict work — record repo + commit sha (and dirty state) in
  the run metadata. Quotes against an unpinned tree are worthless; a dirty worktree can
  make an uncommitted edit look like a finding.

## The six phases (full audit mode)

1. **Reconnaissance** — map architecture, trust boundaries, input surfaces, and prior
   evidence into `architecture.md`; build the deterministic coverage plan as
   `coverage-ledger.json` (one unit per surface × trust boundary × attack class).
   Read the repo's own `BOUNDARY.md` and `AGENTS.md` first — they state what the repo
   owns, its dependency contract, and its security-relevant conventions (hash choices,
   password KDF, secret storage, FFI surfaces). A repo violating its own stated
   contract is a finding; a repo following a contract you did not read is not.
2. **Coverage-led hunting** — assign one isolated hunter per ledger unit, each
   returning checks performed + candidates. Run a coverage critic after each wave to
   find gaps before trusting "clean". A check nobody recorded is a check that did not
   happen.
3. **Candidate validation** — give every unique candidate to a fresh verifier whose
   job is to DISPROVE it. The verifier needs no loyalty to the candidate.
4. **Structured output** — write `confirmed`, `needs_validation`, and `rejected`
   records to `findings.json`; validate shape before reporting.
5. **Independent record verification** — a fresh agent re-verifies every `confirmed`
   record's source claims (file, line, reachable path) before the report. Replacements
   re-enter verification.
6. **Reporting** — derive `REPORT.md` (owner-facing summary), `FINDINGS-DETAIL.md`
   (per-finding trace + reproduction), `NEEDS-VALIDATION.md` (exact unresolved facts)
   from the verified records and the coverage ledger. State the coverage statement
   honestly — one pass is never complete; a scoped run is partial by definition.

Multiple runs against the same target are additive: carry prior confirmed findings
forward only after re-checking the source still matches; changed source re-opens the
unit; prior `rejected` suppresses only the exact failed claim, never the unit.

## The three verdicts — the evidence bar

- **confirmed** — complete source trace (file → line → reachable path from an
  attacker-controlled input) PLUS a bounded local reproduction (fixture, unit test, or
  sandboxed run). Names the affected principal, resource, and outcome.
- **needs_validation** — a real candidate with ONE named unresolved fact (a proxy
  behavior, a deployed config, an upstream cap, a library version). No severity is
  assigned. Name the exact check that would resolve it.
- **rejected** — a disproved candidate, kept with the disproof. Rejections are
  evidence too; they stop the same candidate from being re-raised next run.

Never upgrade by severity inflation: a missing header, a missing rate limit, or a
missing flag is NOT a finding by itself — show an accepted invalid request, a
cross-principal impact, a downgrade, or a disclosure. Verify framework and library
defaults before claiming a defect; an unknown version makes it `needs_validation`.

## Rust attack-class quick reference

Pick classes by surface, not by mood. (Full per-domain checklists live upstream —
see provenance.)

- **Wire parsing** — length fields validated BEFORE allocation; attacker arithmetic
  through bare `+ - * <<` (checked ops + error path); truncating casts on wire ints;
  unbounded reads bounded with `Read::take`; bounds re-derived after every expansion
  (decompression, nested decode); duplicate/unknown/zero enum variants reaching
  default-privileged arms. This workspace's healer carries matching sec rules
  (`length-prefix-validate-before-alloc`, `unchecked-wire-arithmetic`,
  `unbounded-read-into-vec`) — `cargo refine --secret-scan`-adjacent lanes and the
  corpus are first-pass detectors, not the audit itself.
- **FFI / unsafe** — panics unwinding across `extern "C"` (catch_unwind); reference
  params in extern fns (a C caller passing NULL is instant UB); raw-pointer deref
  without null handling; `from_raw_parts` with `len == 0` on a possibly-null pointer;
  `unsafe impl Send/Sync` granting thread-safety the field types deny; pointer-length
  and ownership contracts across the boundary (who frees, bytes vs elements).
- **Secrets** — timing-safe comparison for tokens/MACs (never `==` on secrets);
  credential files created world-readable (0600 at create); secrets in logs, error
  paths, URLs, or client bundles; weak digests (MD5/SHA1) and bare-digest password
  "hashing" (memory-hard KDF only); constant values standing in for key material.
- **Webhooks and signed messages** — signatures verified over the EXACT received
  bytes (never a re-serialization of a parsed body); untrusted producers treated as
  control plane (admin/callback/migration claims need authenticated producer +
  audience binding); replay and ordering (nonce/sequence checked-and-set consumed
  atomically); envelope vs payload identity (authorization on envelope fields a
  client can override while the handler acts on body fields).
- **Egress and URLs** — user-influenced URLs reaching outbound clients without a
  scheme/host allowlist (SSRF — cloud metadata, loopback admin edges, internal
  peers); redirect targets from request input without same-origin/allowlist checks;
  forwarded headers (`Host`, `X-Forwarded-*`) feeding auth or callback decisions.
- **Paths and files** — untrusted path segments joined to a trusted base then used
  for fs access (`..`, absolute, or Windows-prefix components escape — walk
  `components()` and refuse `ParentDir`/`RootDir`/`Prefix`); check-then-use TOCTOU on
  filesystem decisions; temp files with predictable names in shared directories.
- **Resource exhaustion** — one peer controlling allocations (per-item AND aggregate
  caps); expensive work before authentication; superlinear behavior on accepted
  input (note — the `regex` crate is linear-time by construction; this class applies
  to hand-rolled matchers, not `regex`); reachable panics/aborts in shared processes;
  fail-open recovery paths that discard poison/corruption signals.
- **Isolation** — tenant/peer identifiers taken from request bodies instead of
  authenticated channel identity; one tenant's identifier selecting another tenant's
  resource; batch/stream operations authorized once at the front and never re-checked
  per item.
- **Supply chain and release** — unpinned or unpinned-verifiable dependencies;
  build scripts fetching at compile time; release artifacts without checksums;
  signing keys in CI-reachable env; generated code compiled without review.

## Workspace specifics

- One repo per audit unit; the workspace is multi-root. Scope paths explicitly in the
  run metadata; a sibling repo is out of scope unless named (its interface is part of
  the target's trust boundary though — read its contract, do not audit it).
- Default output location is OUTSIDE the target repo (`~/security-audits/<repo>/run-N/`).
  Raw findings carry exploit detail — never commit them into a repo without an
  explicit owner decision, and NEVER into a public repo. A sanitized post-fix record
  (what the class was, what the fix is) may be landed later as docs, owner-approved.
- Existing detectors are inputs, not conclusions: `cargo clippy` (pedantic, suspicious),
  `cargo audit` / `cargo deny` where available, the healer's `--secret-scan`, and the
  sec-corpus suggest lanes. A clean detector pass proves the DETECTED classes are
  absent, never the audit complete.
- Perf-sensitive lanes: a proposed fix that touches a hot path must respect the
  repo's own budget conventions (alloc-free cores, deterministic replay, bit-identity
  pins). A security fix that silently breaks a determinism gate is a new finding.

## Anti-patterns

- Reciting a checklist without tracing a single input to a sink.
- Severity from a label ("this is CWE-79") instead of from a demonstrated path.
- Treating "internal network" or "trusted peer" as authentication.
- Claiming completeness for one pass; implying a scoped run covered the rest.
- Probing live services to "confirm" availability impact.
- Fixing findings mid-audit — report first; fixes are the owner's call (and invalidate
  the evidence trail when mixed into the audited tree).

## Provenance

Adapted from `cloudflare/security-audit-skill`
(https://github.com/cloudflare/security-audit-skill),
MIT License, Copyright (c) 2025-2026 Cloudflare, Inc., inspected at pin
`c1c8a8c1471069fb0e188eeaff69b8e8db6564a8` (2026-10-19). This file is an original
workspace adaptation (Rust-shaped classes, workspace conventions, artifact policy) —
consult the upstream repo for the full per-domain hunting checklists, the coverage
ledger schema, and the findings validators when running a deep audit.
