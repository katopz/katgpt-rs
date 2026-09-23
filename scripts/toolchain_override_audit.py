#!/usr/bin/env python3
"""The toolchain-override decay class — a hardcoded toolchain override that
survives a workspace toolchain-pin bump and silently builds at a different
toolchain than the repo's rust-toolchain.toml declares.

E0658 on fresh code, benchmarks measured on the wrong compiler, and a perf
bout that dies pre-measurement. riir-clippy mining-queue intake P14 item (k),
2026-09-07. Live specimens found 2026-09-15, all against the workspace pin
1.98.1 (six repos carry the pin file: katgpt-rs, riir-ai, riir-chain,
riir-clippy, riir-kat, riir-train):

    riir-ai/scripts/g2_netem_docker.sh:212          docker -e, value 1.95.0
    riir-game-sdk/scripts/t36_netem_partition_docker.sh:91   same shape
    katgpt-rs .github/workflows/full_gate.yml:99    value stable (rot gate)
    katgpt-rs .github/workflows/wasm32_gate.yml:75  value stable (rot gate)
    riir-chain .github/workflows/toolchain_drift.yml:117     a token

The two netem rows predate the 2026-09-04 owner-directed pin bump and their
docker images bake the old toolchain; the two rot-gate rows and the drift-test
row were deliberate by PROSE only (AGENTS.md, the rust-toolchain.toml header,
workflow comments) until 2026-09-15, when the in-source markers landed in the
owning repos. Prose is not an in-source marker: between the pin bump and the
marker landing, those rows classified DRIFT / UNRESOLVED and the sweep red.
That mid-state was the instrument working, not a defect — DRIFT is walled at
0 and the spec forbids papering over an unmarked override with an expectation
row.

# The rule

Scan TRACKED files (tracked_walk — git ls-files where git can answer) whose
extension is .sh .yml .yaml .toml .py, or any path component matching
Dockerfile* or *.dockerfile, across the contract repos (derived: sibling
directories of this repo's parent carrying both a BOUNDARY.md and a .git
DIRECTORY — the derive_repos predicate; a worktree's .git FILE never counts).

An OCCURRENCE is the token RUSTUP_TOOLCHAIN followed by `:` (YAML env
mapping) or `=` (shell assignment, export, docker -e/--env, TOML/quoted
values), or the Dockerfile space form `ENV RUSTUP_TOOLCHAIN <value>` (an
extension beyond the intake grammar — Dockerfiles are explicitly in the file
scope and the ENV space form is exactly how an image bakes a toolchain).
Comment lines (leading `#` in every scanned type) are skipped as
occurrences but still CARRY markers for the line below. At most one
occurrence per line (v1; the first match wins).

# Verdicts, in precedence order

    DELIBERATE         carries the marker (below), literal value
    UNRESOLVED-MARKED  carries the marker, TOKEN value
    UNRESOLVED         TOKEN value (${{ ... }}, ${VAR}, $VAR) — a non-literal
                       cannot be evaluated statically; never folded into clean
    NO-PIN-OVERRIDE    literal in a repo whose root rust-toolchain.toml is
                       absent (or declares no channel) — the override IS the
                       de facto authority, drift relative to WHAT is
                       unanswerable; INFO, listed and counted, never walled.
                       A marker documents it but never flips it: DELIBERATE
                       without a pin would mean "deliberate divergence from
                       the pin", which is meaningless. Pin-absence dominates
                       the marker in classify().
    MATCH              literal equals the repo pin
    DRIFT              literal differs from the pin, unmarked — THE finding

The repo-level class UNPINNED-ROPO is spelled UNPINNED-REPO: a repo with a
root Cargo.toml and no rust-toolchain.toml — version-sensitive code with no
declared pin (intake item (k)(ii)). One INFO row per repo.

The MARKER is the literal string `toolchain-override-deliberate` in the
shape `# toolchain-override-deliberate: <reason>`, accepted at THREE sites:

    (a) the occurrence's own line (shell trailing comment, YAML inline
        comment);
    (b) the contiguous run of `#` comment lines immediately above the
        occurrence line;
    (c) the contiguous run of `#` comment lines immediately above the HEAD
        of the logical command carrying the occurrence — the head found by
        walking up while the PRECEDING physical line ends with a backslash
        continuation (odd trailing-backslash count: `\\` at EOL is an
        escaped literal, not a continuation).

Rule (c) exists because a `-e RUSTUP_TOOLCHAIN=... \\` deep inside a
multi-line `docker run` cannot carry a comment of its own: a `#` inside a
continuation chain terminates the logical command early — a real shell bug,
not a style slip — so shell scripts document the whole command above its
head line, and a marker there marks every override that command carries.
A code line or a blank line breaks any comment run, and a comment line
inside a continuation chain breaks the head walk (it ends the logical
command in real bash too) — an arm below pins each boundary, because a
marker that leapt blank lines would mark overrides its author never saw.
A marker is an in-source CLAIM of intent, not proof; the report lists every
marked row so the claim stays auditable.

# Honest caveats (v1)

- ROOT PIN ONLY. A workspace with member-level rust-toolchain.toml files is
  unmodeled; the root channel is the only pin this classifier reads. A pin
  file present but channel-less reads as no pin (deliberate: without a
  declared channel, "drift relative to what" is unanswerable).
- TOKEN blindness is structural, not fixable by more regex: a parameter
  expansion or an Actions expression resolves at runtime. UNRESOLVED is its
  own counted ceiling in the sweep and is never folded into clean.
- Value classes: TOKEN is anything starting with `$`; every OTHER static
  value is LITERAL and compared to the pin by string equality. The canonical
  channels are stable/beta/nightly/N.N[.N], but a static value like
  min-toolchain-spin is ALSO a different toolchain than a numeric pin — it
  lands DRIFT, which is the honest verdict, so no whitelist of channel
  shapes gates comparability.
- One occurrence per line; case-sensitive extensions; no other comment
  syntax (# only — correct for all six scanned types). STRING LITERALS:
  for .py, triple-quoted regions (either three-quote delimiter — docstrings
  are the 99% case) are MASKED by a minimal line-oriented state machine
  before any matching, because this file's own docstring prose fired three
  false DRIFT rows on 2026-09-15 (the fourth instrument in this family to
  meet the shape — see the landing record below). Single-quoted one-line
  string literals in .py still count as occurrences: documented, not
  solved, and measured-absent in the workspace today. Shell heredocs and
  YAML block scalars are NOT modeled — measured-absent too (the only .sh
  occurrences workspace-wide are the two marked -e lines; no block scalar
  carries the token). The house answer for Rust is the AST in
  platform_dead_code_audit.py, which does not transfer to YAML/sh.
- This file's own self-test fixtures are BUILT AT RUNTIME (the token is
  assembled from the TRIGGER constant), and its own docstrings are MASKED
  by the same triple-quote state machine the scanner applies to every .py
  file — the scanner learned the language shape rather than the docs being
  obfuscated or the file exempting itself (the AGENTS.md law: an instrument
  that certifies nothing exempts nothing). The katgpt-rs self-scan shows
  exactly the repo's real surface.

# Exits

    0  report printed, every walked repo above its floor
    2  the instrument itself is untrustworthy — self-test MISS, an unreadable
       floors file, an empty derived population, or a walked-file count below
       the floors file's min_files for the repo (a blind walk reporting a
       green zero is the one verdict this class must never print)

--prove-fires: DELIBERATELY ABSENT — no frozen known-answer commit exists
for this class yet. The marker-landing commits that clear the first DRIFT
rows become it; add the flag against one of those shas then, on the
platform_dead_code_audit.py pattern.

First-run measurement (2026-09-15, all 20 contract repos on this box):
748 scannable tracked files · 0 MATCH · 0 DELIBERATE · 3 DRIFT (katgpt-rs
full_gate.yml:99 + wasm32_gate.yml:75, riir-ai g2_netem_docker.sh:212 —
all three prose-deliberate with in-source markers pending) · 1 UNRESOLVED
(riir-chain toolchain_drift.yml:117, marker pending) · 1 NO-PIN-OVERRIDE
(riir-game-sdk t36_netem_partition_docker.sh:91 — the repo carries NO pin
file, so there is nothing to drift FROM; the intake's "same shape" reading
held for the line, not for the verdict — the real repair there is a pin
file, the marker documents intent) · 13 UNPINNED-REPO. Per-repo walked
counts and the floors they pinned:
toolchain_override_drift_floors.txt.

Marker-landing re-run (same day, rule (c) added): the five markers landed
(g2/t36 as comment blocks above the docker-run HEAD lines — rules (c);
the other three directly above their occurrences — rule (b)). Result:
0 DRIFT workspace-wide · 3 DELIBERATE (katgpt-rs 2, riir-ai 1) ·
1 UNRESOLVED-MARKED (riir-chain; the counted UNRESOLVED ceiling re-pinned
to 0) · riir-game-sdk stays NO-PIN-OVERRIDE 1 by design (pin-absence
dominates the marker) · 13 UNPINNED-REPO unchanged. Sweep green.

Self-scan incident (same day, after that green): the follow-up docstring
prose planted three false DRIFT rows in THIS file (lines 38, 79, 243 —
grammar examples and the rule-(c) rationale, all inside triple-quoted
docstrings). Fourth instrument in the family to meet the shape, per
AGENTS.md's lineage: platform_dead_code_audit (literal masking dropped
inline format args), subprocess_encoding_gate (its own selftest fixtures
were the four offenders; it moved to an AST), wasm32_surface_audit (a raw
string's cfg read as surface), now this file. The repair is the scanner
learning the shape — a minimal line-oriented triple-quote state machine
for .py, UNPARSED on a runaway region (the trap-sentinel law) — not
docstring obfuscation, not a self-exemption. Post-repair: the three
self-hits gone, every other repo's verdict byte-identical to the pre-
incident green, sweep green.

Alias-seam repair (2026-09-23): this audit's own main() was the one
production walk the Issue-842 batch missed — the SWEEP opened repos
through `sweep_population.open_repo` from the start, but the audit's
derived mode built `WORKSPACE / <contract-name>` handles directly, and
on this box (repo_alias.local.txt active, 3 mappings) that opened
directories that DO NOT EXIST for the three aliased mmorpg-* repos:
measured pre-fix, files=0 pin=none for all three with ✓ glyphs over the
zeros and exit 2 only because their floors rows caught it (a box without
those rows would have read a confident green over nothing). The repair
is the one seam, not a third walk spelling: derived mode opens through
`open_repo(n, WORKSPACE)` while the LABEL stays the contract handle, so
pins, floors and stdout keep the contract vocabulary and the alias
content never prints. Post-fix, same run: 70/39/41 files for the three
repos (floors 30/13/25 — held), pin=1.98.1 read for real (the row-4f
pins became visible to this instrument the same day they landed), 23/23
selftest arms green, the sweep byte-identical, and the explicit-path
invocation unchanged. Family grep for the raw `WORKSPACE / n` shape: no
other production walk carries it — the remaining hits are selftest
fixtures, where repo_alias.real's identity mapping is what keeps them
fixture-safe.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from skill_repo_set_gate import derive_repos  # noqa: E402
from sweep_population import open_repo  # noqa: E402
from tracked_walk import tracked_files  # noqa: E402

# Issue 804: this instrument is documented as directly invokable, and its
# verdict glyphs (✓ ✗ ⛔ ⚠) kill it on a non-UTF-8 console — no verdict at
# all, findings unread. docs_gate.sh's PYTHONIOENCODING only covers runs
# that go through the wrapper.
import console_safe  # noqa: E402

console_safe.apply()

REPO_ROOT = HERE.parent
WORKSPACE = REPO_ROOT.parent
PINS = HERE / "toolchain_override_drift_floors.txt"

TRIGGER = "RUSTUP_TOOLCHAIN"
MARKER = "toolchain-override-deliberate"
SCAN_SUFFIXES = (".sh", ".yml", ".yaml", ".toml", ".py")

VERDICTS = ("MATCH", "DELIBERATE", "DRIFT", "UNRESOLVED",
            "UNRESOLVED-MARKED", "NO-PIN-OVERRIDE")
# The actionable rows a default report prints verbatim; -v adds the rest.
ACTION_VERDICTS = ("DRIFT", "UNRESOLVED", "UNRESOLVED-MARKED")

PIN_RE = re.compile(r'channel\s*=\s*"([^"]+)"')

# Value alternation, most-specific first: an Actions expression, a shell
# ${VAR} expansion, a bare $VAR, a quoted string, then any bare token
# (`#` excluded so a trailing comment never becomes part of the value).
_TOKEN_VAL = (r"\$\{\{.*?\}\}|\$\{[^{}]*\}|\$\w+"
              + "|"
              + r'"[^"\n]*"|' + r"'[^'\n]*'|"
              + r"[^\s#]+")
_VAL_GRP = "(?P<val>" + _TOKEN_VAL + ")"
# The name must not be a suffix of a longer identifier (MY_TOOLCHAIN-ish),
# and must sit directly before the assignment/mapping punctuation.
OCC_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-])" + TRIGGER + r"[ \t]*[:=][ \t]*" + _VAL_GRP)
# Dockerfile space form — checked only when the assignment form missed, so
# `ENV NAME=value` is never counted twice (one occurrence per line, v1).
DOCKER_ENV_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-])ENV[ \t]+" + TRIGGER + r"[ \t]+" + _VAL_GRP)

# Triple-quote delimiters, built without a literal triple in THIS source —
# the file scans itself, and a `"""` literal here would open a false region
# in its own masker (the self-scan law: the scanner reads its own file).
_TRIPLE_RE = re.compile('"' * 3 + "|" + "'" * 3)


def _py_code_slices(lines: list[str]) -> tuple[list[str], str | None]:
    """Per-line CODE text — the parts of each physical line outside
    triple-quoted string regions — and the delimiter still open at EOF.

    Minimal line-oriented state machine (v1): while OUTSIDE, the first
    three-quote delimiter of either kind opens a region; while INSIDE,
    only the MATCHING delimiter closes it (the other triple is plain
    text — Python agrees).
    A full `#` comment line never changes state (Python agrees: it is a
    comment). Lines whose content is string come back as empty code, so
    docstring prose never fires an occurrence.

    A region still open at EOF returns it as `state` — the file is
    UNPARSED, never silently half-skipped: a runaway docstring swallowing
    the rest of the file must not read as clean (the trap-sentinel UNPARSED
    law), and must not read as DRIFT either. Occurrences in the readable
    prefix before the runaway still count — they are real text the scanner
    did read; the UNPARSED flag is what keeps the file from ever reading
    as a pass.

    Documented v1 limits: escape sequences (`\"\"\"` inside a string),
    single-quoted NORMAL strings containing a triple, and a triple inside
    a trailing `#` comment can false-open a region. Measured absent in the
    workspace today (the only .py occurrences ever seen were this file's
    own docstring prose); re-measure before relying on the masking for a
    file that plays games with quotes.
    """
    code: list[str] = []
    state: str | None = None
    for line in lines:
        if state is None and line.lstrip().startswith("#"):
            code.append(line)
            continue
        buf: list[str] = []
        pos = 0
        while pos < len(line):
            if state is None:
                m = _TRIPLE_RE.search(line, pos)
                if m is None:
                    buf.append(line[pos:])
                    break
                buf.append(line[pos:m.start()])
                state = m.group(0)
                pos = m.end()
            else:
                end = line.find(state, pos)
                if end == -1:
                    pos = len(line)
                    break
                pos = end + len(state)
                state = None
        code.append("".join(buf))
    return code, state


def is_scannable(rel: str) -> bool:
    return rel.endswith(SCAN_SUFFIXES) or _dockerfile_p(rel)


def _dockerfile_p(rel: str) -> bool:
    return any(part.startswith("Dockerfile") or part.endswith(".dockerfile")
               for part in rel.replace("\\", "/").split("/"))


def norm_value(val: str) -> str:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    return val


def _marker_in_run_above(lines: list[str], idx: int) -> bool:
    j = idx - 1
    while j >= 0 and lines[j].lstrip().startswith("#"):
        if MARKER in lines[j]:
            return True
        j -= 1
    return False


def _ends_continuation(line: str) -> bool:
    """A trailing backslash continuation — `\\` plus optional whitespace
    before the end of the line. ODD trailing-backslash count: a `\\\\` at
    end-of-line is an escaped literal backslash, not a continuation."""
    s = line.rstrip()
    return (len(s) - len(s.rstrip("\\"))) % 2 == 1


def _command_head(lines: list[str], idx: int) -> int:
    """The first line of the logical command carrying line `idx` — walk up
    while the PRECEDING physical line ends with a continuation."""
    j = idx
    while j > 0 and _ends_continuation(lines[j - 1]):
        j -= 1
    return j


def has_marker(lines: list[str], idx: int) -> bool:
    """Three clauses, in order:
    (a) the marker on the occurrence's own line;
    (b) the contiguous run of `#` comment lines immediately above it;
    (c) the same run above the HEAD of the logical command carrying the
        occurrence. A `-e RUSTUP_TOOLCHAIN=... \\` deep inside a multi-line
        `docker run` cannot carry a comment of its own — a `#` inside a
        continuation chain terminates the logical command early (a real
        shell bug, not a style slip) — so shell scripts document the whole
        command above its head line, and a marker there marks every
        override that command carries.
    A code line or a blank line breaks any comment run; a comment line
    inside a continuation chain breaks the HEAD walk (it ends the logical
    command in real bash too). Both boundaries are pinned by self-test
    arms."""
    if MARKER in lines[idx]:
        return True
    if _marker_in_run_above(lines, idx):
        return True
    head = _command_head(lines, idx)
    return head != idx and _marker_in_run_above(lines, head)


def classify(value: str, marked: bool, pin: str | None) -> str:
    """Pin-absence dominates the marker: in a repo with no declared pin the
    override IS the de facto authority and drift relative to WHAT is
    unanswerable regardless of intent — the marker documents, it does not
    re-classify. (The floors file's recorded repair for that shape is a pin
    file, not prose.) DELIBERATE without a pin would mean "deliberate
    divergence from the pin", which is meaningless; UNRESOLVED-MARKED would
    overpay an intent claim that answers only half the question."""
    token = value.startswith("$")
    if pin is None:
        return "UNRESOLVED" if token else "NO-PIN-OVERRIDE"
    if marked:
        return "UNRESOLVED-MARKED" if token else "DELIBERATE"
    return "UNRESOLVED" if token else ("MATCH" if value == pin else "DRIFT")


@dataclass(frozen=True)
class Occ:
    rel: str        # repo-relative, forward slashes
    lineno: int     # 1-based
    value: str      # normalized (quotes stripped)
    verdict: str
    form: str       # assign | docker-env
    line: str       # the raw line, rstrip'd — printed verbatim

    def addr(self) -> str:
        return f"{self.rel}:{self.lineno}"


def scan_lines(rel: str, raw_lines: list[str], pin: str | None,
               code_lines: list[str] | None = None) -> list[Occ]:
    """Occurrence scan over one file. `code_lines` is the triple-quote-
    masked text for .py files (docstring content removed); every decision —
    comment skip, occurrence match, marker context — runs on it, while the
    printed `line` stays the RAW source line so rows stay auditable."""
    if code_lines is None:
        code_lines = raw_lines
    occs: list[Occ] = []
    for idx, code in enumerate(code_lines):
        if code.lstrip().startswith("#"):
            continue
        m = OCC_RE.search(code)
        form = "assign"
        if m is None:
            m = DOCKER_ENV_RE.search(code)
            form = "docker-env"
        if m is None:
            continue
        value = norm_value(m.group("val"))
        occs.append(Occ(rel, idx + 1, value,
                        classify(value, has_marker(code_lines, idx), pin),
                        form, raw_lines[idx].rstrip()))
    return occs


PIN_REL = "rust-toolchain.toml"


def head_source(p: Path, root: Path, overlay: dict | None) -> str | None:
    """One file's bytes, taken from the OVERLAY when it names this path.

    Issue 822 T5f. A pin is a claim about the repo and a repo's state is its
    COMMITS, but this walk reads a working tree several concurrent sessions
    write into. `worktree_state.head_overlay()` answers `{rel: HEAD bytes}`
    for the dirty files; this is the one place a path becomes text.

    ⚠ `None` in the overlay is a VALUE — "tracked but absent from HEAD", a
    staged-but-never-committed file — and it must SKIP the file rather than
    fall through to the worktree copy, or another session's in-flight source
    is classified as committed. `in` and `.get()` say different things here.
    """
    if overlay is not None:
        rel = p.relative_to(root).as_posix()
        if rel in overlay:
            return overlay[rel]
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def repo_pin(root: Path, overlay: dict | None = None) -> str | None:
    """The ROOT rust-toolchain.toml channel — v1 scope, documented caveat.

    ⛔ This is why the sweep's Issue 822 instrument is the CROSS-FILE one.
    Every occurrence in the repo is judged against this ONE value, so a dirty
    `rust-toolchain.toml` moves every verdict in the repo at once and the
    per-file `head_delta` shortcut — whose premise is that a row depends only
    on its own file's bytes — would invent rows wholesale.
    """
    p = root / PIN_REL
    if overlay is not None and PIN_REL in overlay:
        src = overlay[PIN_REL]
        if src is None:
            return None             # absent from HEAD: no pin to read
        m = PIN_RE.search(src)
        return m.group(1) if m else None
    if not p.is_file():
        return None
    m = PIN_RE.search(p.read_text(encoding="utf-8", errors="replace"))
    return m.group(1) if m else None


@dataclass
class RepoScan:
    name: str
    root: Path
    pin: str | None
    pin_file: bool
    walked: int
    vendored: int          # vendor/ files the walk excluded, ALL file types
    occs: list
    unpinned_repo: bool
    unparsed: list = field(default_factory=list)   # files the masker could
    # not fully read (runaway triple-quote) — never folded into clean

    def count(self, verdict: str) -> int:
        return sum(1 for o in self.occs if o.verdict == verdict)


def _present(root: Path, rel: str, overlay: dict | None) -> bool:
    """Does this path exist in the tree being classified?

    With an overlay the answer is HEAD's, not the worktree's: a staged-only
    `rust-toolchain.toml` does not exist in any commit, and one deleted in the
    worktree but still committed does.
    """
    if overlay is not None and rel in overlay:
        return overlay[rel] is not None
    return (root / rel).is_file()


def scan_repo(root: Path, name: str, overlay: dict | None = None) -> RepoScan:
    """Scan one repo. With `overlay`, scan what HEAD would produce.

    The default is `None` — every existing caller is byte-identical, the
    `instrument_reachability_gate.reachable(read=)` precedent (Issue 822 T5a).
    """
    pin = repo_pin(root, overlay)
    files, vendored = tracked_files(root, "*")
    occs: list[Occ] = []
    unparsed: list[str] = []
    walked = 0
    for f in files:
        rel = f.relative_to(root).as_posix()
        if not is_scannable(rel):
            continue
        text = head_source(f, root, overlay)
        if text is None:
            # Absent from HEAD (or unreadable): nothing committed to scan.
            # Deliberately NOT counted into `walked` either — the walk figure
            # has to describe the tree that was actually read.
            continue
        walked += 1
        raw = text.splitlines()
        if rel.endswith(".py"):
            code, open_q = _py_code_slices(raw)
            if open_q is not None:
                unparsed.append(rel)
        else:
            code = raw
        occs.extend(scan_lines(rel, raw, pin, code))
    return RepoScan(name, root, pin, _present(root, PIN_REL, overlay),
                    walked, vendored, occs,
                    _present(root, "Cargo.toml", overlay) and pin is None,
                    unparsed)


# ── floors (shared with the sweep — one parser, one file, two consumers) ────

FLOOR_FIELDS = ("min_files", "max_drift", "max_unresolved")


def parse_pins(path: Path) -> dict:
    rows: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 1 + len(FLOOR_FIELDS):
            raise ValueError(
                f"malformed pin row (want {1 + len(FLOOR_FIELDS)} fields): "
                f"{raw!r}")
        rows[parts[0]] = dict(zip(FLOOR_FIELDS, (int(v) for v in parts[1:])))
    return rows


# ── report ──────────────────────────────────────────────────────────────────

def _pin_txt(scan: RepoScan) -> str:
    if scan.pin:
        return scan.pin
    return "UNREAD" if scan.pin_file else "none"


def report(scans: list, verbose: bool = False) -> None:
    tot = {v: 0 for v in VERDICTS}
    tot_walked = tot_unpinned = tot_unparsed = 0
    for s in scans:
        for v in VERDICTS:
            tot[v] += s.count(v)
        tot_walked += s.walked
        tot_unpinned += 1 if s.unpinned_repo else 0
        tot_unparsed += len(s.unparsed)
        hot = s.count("DRIFT") or s.count("UNRESOLVED")
        status = "⛔" if s.count("DRIFT") else ("·" if hot else "✓")
        print(f"{status} {s.name:22s} pin={_pin_txt(s):7s} files={s.walked:<5d} "
              f"occ={len(s.occs):<3d} (match {s.count('MATCH')} · "
              f"delib {s.count('DELIBERATE')} · drift {s.count('DRIFT')} · "
              f"unres {s.count('UNRESOLVED')}+{s.count('UNRESOLVED-MARKED')} · "
              f"no-pin {s.count('NO-PIN-OVERRIDE')})"
              + (f" vendored={s.vendored}" if s.vendored else ""))
        shown = VERDICTS if verbose else ACTION_VERDICTS
        for o in s.occs:
            if o.verdict not in shown:
                continue
            mark = "⛔" if o.verdict == "DRIFT" else "⚠"
            form = f" [{o.form}]" if verbose else ""
            print(f"      {mark} {o.verdict:<18s} {o.addr():<58s}{form} "
                  f"{o.line}")
        if s.unpinned_repo:
            print("      ℹ UNPINNED-REPO — root Cargo.toml with no "
                  "rust-toolchain.toml: version-sensitive code, no declared "
                  "pin (intake P14 (k)(ii))")
        for rel in s.unparsed:
            print(f"      ⚠ UNPARSED          {rel} — a triple-quote region "
                  f"never closed; the file never reads as clean")
    print()
    print(f"{len(scans)} repo(s) · {tot_walked} scannable tracked file(s) · "
          + " · ".join(f"{v.lower().replace('_', '-')} {tot[v]}"
                       for v in VERDICTS)
          + f" · {tot_unpinned} UNPINNED-REPO · {tot_unparsed} UNPARSED")
    print("  scope: hardcoded toolchain overrides in tracked "
          ".sh/.yml/.yaml/.toml/.py + Dockerfile paths; one occurrence per "
          "line; # comments skipped in every scanned type")
    print("  markers are an in-source CLAIM of intent, not proof — run with "
          "-v to list every marked row and audit the claim")
    print("  caveats: ROOT pin only (member-level rust-toolchain.toml "
          "unmodeled); TOKEN values are UNRESOLVED, never clean")
    print("  --prove-fires: none yet — no frozen known-answer commit exists "
          "for this class; the marker-landing commits that clear the first "
          "DRIFT rows become it")


# ── self-test (on EVERY invocation — a classifier MISS exits 2, never a
#    confident zero; the platform_dead_code_audit precedent) ─────────────────

def _assign(value: str, prefix: str = "", suffix: str = "\n") -> str:
    """A shell-style occurrence line. The token is assembled at RUNTIME so
    this file's own source carries no occurrence."""
    return f"{prefix}{TRIGGER}={value}{suffix}"


def _yaml_map(value: str, indent: str = "") -> str:
    return f"{indent}{TRIGGER}: {value}\n"


def _marker(reason: str = "self-test fixture — not a workspace override",
            nl: str = "\n") -> str:
    return f"# {MARKER}: {reason}{nl}"


# ── continuation-chain fixtures (rule (c)) ──────────────────────────────────
# A multi-line docker run: the head and the occurrence are joined by
# backslash continuations, so no comment can sit on or directly above the
# occurrence INSIDE the chain — which is exactly why a shell script
# documents the command above its head.

def _chain_head(pre: str = "") -> str:
    return pre + "docker run --rm \\\n"


_CHAIN_MID = "    -v /tmp/work:/work \\\n"
_CHAIN_TAIL = "    --network none\n"


def _chain_occ(value: str) -> str:
    return "    -e " + TRIGGER + "=" + value + " \\\n"


# ── .py triple-quote fixtures (the self-scan incident shape) ─────────────
# Built without a literal triple in THIS source: the file scans itself, and
# a `"""` literal here would open a false region in its own masker.

def _py_docstring(inner: str, close: bool = True) -> str:
    q = '"' * 3
    return q + "\n" + inner + (q + "\n" if close else "")


def _py_oneliner(inner: str) -> str:
    q = '"' * 3
    return q + inner + q + "\n"


# Files expected to come back UNPARSED, keyed by case label (every other
# case must read fully — an unexpected UNPARSED file fails its arm).
WANT_UNPARSED = {
    "py unterminated triple-quote is UNPARSED, never half-clean":
        ("doc.py",),
}


PIN_TOML = 'channel = "1.98.1"\n'
CARGO_TOML = '[package]\nname = "t"\nversion = "0.0.0"\n'


def _write_tree(root: Path, files: dict, with_pin: bool = True,
                with_cargo: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    if with_pin:
        (root / "rust-toolchain.toml").write_text(PIN_TOML, encoding="utf-8")
    if with_cargo:
        (root / "Cargo.toml").write_text(CARGO_TOML, encoding="utf-8")


# (label, files, write kwargs, want_walked, want {(verdict, value)},
#  want_unpinned_repo). walked counts every scannable file the tree carries —
# rust-toolchain.toml and Cargo.toml are .toml themselves, so they walk too;
# an arm whose walked count silently collapsed fails here, not as a green.
SELFTEST_CASES = (
    ("shell export MATCH",
     {"a.sh": _assign("1.98.1", "export ")}, {},
     3, {("MATCH", "1.98.1")}, False),
    ("yaml env-map MATCH (colon grammar)",
     {"w.yml": _yaml_map("1.98.1", "  ")}, {},
     3, {("MATCH", "1.98.1")}, False),
    ("docker -e DRIFT (the P14 specimen shape)",
     {"a.sh": _assign("1.95.0", "docker run -e ") + "    --rm\n"}, {},
     3, {("DRIFT", "1.95.0")}, False),
    ("same-line marker DELIBERATE",
     {"a.sh": _assign("1.95.0",
                      suffix=" # " + MARKER + ": image bakes it\n")}, {},
     3, {("DELIBERATE", "1.95.0")}, False),
    ("comment-block-above marker DELIBERATE",
     {"a.sh": _marker("image bakes it") + _assign("1.95.0")}, {},
     3, {("DELIBERATE", "1.95.0")}, False),
    ("marker separated from occurrence by code is NOT carried",
     {"a.sh": _marker("above an unrelated line")
              + "cargo build --release\n" + _assign("1.95.0")}, {},
     3, {("DRIFT", "1.95.0")}, False),
    ("rule (c): marker above the chain HEAD marks the mid-list occurrence",
     {"a.sh": _marker("the image bakes it") + _chain_head() + _CHAIN_MID
              + _chain_occ("1.95.0") + _CHAIN_TAIL}, {},
     3, {("DELIBERATE", "1.95.0")}, False),
    ("rule (c): marker separated from the head by code is NOT carried",
     {"a.sh": _marker("above an unrelated line") + "cargo build\n"
              + _chain_head() + _CHAIN_MID + _chain_occ("1.95.0")
              + _CHAIN_TAIL}, {},
     3, {("DRIFT", "1.95.0")}, False),
    ("rule (b) stays literal: a comment directly above a mid-list line marks",
     {"a.sh": _chain_head() + _CHAIN_MID
              + _marker("directly above the occurrence")
              + _chain_occ("1.95.0") + _CHAIN_TAIL}, {},
     3, {("DELIBERATE", "1.95.0")}, False),
    ("rule (c) boundary: a stray mid-chain comment breaks the head walk",
     {"a.sh": _marker("above a head the chain no longer reaches")
              + _chain_head() + _CHAIN_MID
              + "# stray mid-chain comment (a bash syntax hazard)\n"
              + _chain_occ("1.95.0") + _CHAIN_TAIL}, {},
     3, {("DRIFT", "1.95.0")}, False),
    ("comment line is not an occurrence (the rust-toolchain.toml:20 shape)",
     {"a.sh": "# see " + TRIGGER + " docs before bumping\n",
      "b.py": "x = 1\n"}, {},
     4, set(), False),
    ("token UNRESOLVED (Actions expression)",
     {"w.yml": _yaml_map("${{ matrix.channel }}", "  ")}, {},
     3, {("UNRESOLVED", "${{ matrix.channel }}")}, False),
    ("token + marker UNRESOLVED-MARKED",
     {"w.yml": _marker() + _yaml_map("${{ matrix.channel }}", "  ")}, {},
     3, {("UNRESOLVED-MARKED", "${{ matrix.channel }}")}, False),
    ("token UNRESOLVED (shell expansion)",
     {"a.sh": _assign("${" + TRIGGER + ":-1.95.0}")}, {},
     3, {("UNRESOLVED", "${" + TRIGGER + ":-1.95.0}")}, False),
    ("no pin file: NO-PIN-OVERRIDE + UNPINNED-REPO",
     {"a.sh": _assign("stable")}, {"with_pin": False},
     2, {("NO-PIN-OVERRIDE", "stable")}, True),
    ("no pin file: a marker documents but does NOT flip to DELIBERATE",
     {"a.sh": _marker("documents intent; the real repair is a pin file")
              + _assign("1.95.0")}, {"with_pin": False},
     2, {("NO-PIN-OVERRIDE", "1.95.0")}, True),
    ("Dockerfile ENV space-form DRIFT + ARG assignment MATCH",
     {"deploy/Dockerfile": "ENV " + TRIGGER + " 1.95.0\n"
                           + "ARG " + TRIGGER + "=1.98.1\n"}, {},
     3, {("DRIFT", "1.95.0"), ("MATCH", "1.98.1")}, False),
    ("toml quoted value (quotes stripped before compare)",
     {"cargo-config.toml": "[env]\n" + TRIGGER + ' = "1.98.1"\n'}, {},
     3, {("MATCH", "1.98.1")}, False),
    ("py docstring prose is NOT an occurrence (the self-scan incident shape)",
     {"doc.py": _py_docstring("doc prose: " + TRIGGER + "=1.0.0 inside\n")
                + "x = 1\n"}, {},
     3, set(), False),
    ("py one-line triple-quoted string is NOT an occurrence",
     {"doc.py": _py_oneliner("one-liner with " + TRIGGER + "=x inside")
                + "x = 1\n"}, {},
     3, set(), False),
    ("py occurrence AFTER a closed docstring re-arms the scanner",
     {"doc.py": _py_docstring("harmless doc\n")
                + "env " + TRIGGER + "=1.0.0\n"}, {},
     3, {("DRIFT", "1.0.0")}, False),
    ("py unterminated triple-quote is UNPARSED, never half-clean",
     {"doc.py": "env " + TRIGGER + "=1.0.0\n"
                + _py_docstring("runaway prose mentioning "
                                + TRIGGER + "=1.0.0\n", close=False)
                + TRIGGER + "=1.98.1\n"}, {},
     3, {("DRIFT", "1.0.0")}, False),
    ("clean pinned repo: zero occurrences, not UNPINNED",
     {"a.sh": "cargo build --release\n", "b.py": "print(1)\n"}, {},
     4, set(), False),
)


def selftest() -> int:
    """Pin the classifier in BOTH directions. Exit 2, never 1: an
    untrustworthy instrument is a different verdict from drift, and a report
    that cannot classify must not be read as `0 findings`.

    Runs over plain temp trees (the fallback walk): the CLASSIFIER is the
    thing under test here; the tracked branch of the walk is exercised by the
    sweep's self-test with real `git init` repos.
    """
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        for idx, (label, files, kwargs, want_walked, want_occs,
                  want_unpinned) in enumerate(SELFTEST_CASES):
            root = Path(td) / f"c{idx}"
            _write_tree(root, files, **kwargs)
            res = scan_repo(root, f"c{idx}")
            got_occs = sorted((o.verdict, o.value) for o in res.occs)
            want_unparsed = WANT_UNPARSED.get(label, ())
            ok = (res.walked == want_walked and got_occs == sorted(want_occs)
                  and res.unpinned_repo == want_unpinned
                  and tuple(res.unparsed) == want_unparsed)
            mark = "✓" if ok else "✗"
            if not ok:
                bad += 1
                print(f"  {mark} {label}")
                print(f"      walked {res.walked} (want {want_walked}), occs "
                      f"{got_occs} (want {sorted(want_occs)}), unpinned "
                      f"{res.unpinned_repo} (want {want_unpinned}), unparsed "
                      f"{res.unparsed} (want {list(want_unparsed)})")
            else:
                print(f"  {mark} {label}")
    print()
    print(f"  {len(SELFTEST_CASES) - bad}/{len(SELFTEST_CASES)} arm(s) pinned")
    if bad:
        print("  ⛔ classifier MISS — the report is not trustworthy")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="toolchain-override decay audit: hardcoded "
                    "RUSTUP_TOOLCHAIN overrides vs the repo pin")
    ap.add_argument("repos", nargs="*", help="repo paths (default: derived)")
    ap.add_argument("--self-test", action="store_true",
                    help="run only the classifier self-test")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also list MATCH/DELIBERATE/NO-PIN-OVERRIDE rows")
    args = ap.parse_args()

    if args.self_test:
        print("toolchain-override audit — classifier self-test\n")
        return selftest()

    rc = selftest()
    if rc:
        return rc
    print()

    if args.repos:
        paths = [Path(a).resolve() for a in args.repos]
        scans = [scan_repo(p, p.name) for p in paths]
    else:
        names = derive_repos(WORKSPACE)
        if not names:
            print("⛔ derived population is EMPTY — refusing to report a "
                  "green over zero repos")
            return 2
        # Issue 842 seam: derive_repos returns CONTRACT names, and on an
        # alias box the on-disk directory spells differently — opening
        # WORKSPACE / name measured three repos at zero files with every
        # verdict judged against a directory that does not exist (only the
        # floors rows kept it from reading as a green). Open through the
        # seam; the LABEL stays the contract handle, so pins, floors and
        # stdout keep the contract vocabulary and the alias content never
        # prints (repo_alias's own rule).
        scans = [scan_repo(open_repo(n, WORKSPACE), n) for n in names]

    floors: dict = {}
    if PINS.is_file():
        try:
            floors = parse_pins(PINS)
        except ValueError as e:
            print(f"⛔ floors file unreadable: {e}")
            return 2
    else:
        print(f"⚠ floors file absent ({PINS.name}) — walk-floor assertion "
              "skipped (first run on a fresh clone)\n")

    report(scans, args.verbose)

    blind = []
    for s in scans:
        row = floors.get(s.name)
        if row is not None and s.walked < row["min_files"]:
            blind.append(f"{s.name}: walked {s.walked} scannable file(s) < "
                         f"floored {row['min_files']} — the counts above are "
                         f"a green over a blind walk")
    if blind:
        print()
        for b in blind:
            print(f"⛔ BLIND WALK — {b}")
        return 2
    return 0                        # a REPORT: the verdict half is the sweep


if __name__ == "__main__":
    sys.exit(main())
