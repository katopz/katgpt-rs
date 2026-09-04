#!/usr/bin/env python3
"""Which test targets are named by NO pinned suite?

`ci_test_execution_report.py` answers "does anything in this repo's CI
automatically EXECUTE tests" — a per-REPO verdict. This report answers the
per-TARGET question underneath it: for each `[[test]]` target (explicit or
implicit), does any committed suite (the repo's own `scripts/` + `.github/`,
then every present sibling's as a cross-check) name it at all?

The motivating class is the recurring "gate nobody runs" find, each previously
discovered ad hoc by hand-grepping one suspicious name:

  - riir-ai 865: a default-feature GOAT gate executable by nothing automatic
    for 55 days (`bench_336 g6e`).
  - riir-ai 868: all three `goat_290_*` targets named by NO pinned suite
    ("the twelfth-plus instance") — found by `grep goat_290 scripts/ .github/`
    returning empty.
  - riir-chain (09-03): `chain_engram_commit` was suite-less until a
    percentile per-site read happened to touch it.

A **report, not a gate** (always exit 0): most unpinned targets are
deliberately unpinned — one-shot probes, historical benches, hardware-gated
suites, 17-minute fixtures. The actionable column is the LOAD-BEARING split
(`is_load_bearing`, imported from `cfg_gated_target_audit` so both reports
share one committed vocabulary — a classifier disagreement between them is
the finding, not a tie to break by hand).

Scope: TEST targets only. Benches are excluded on purpose — Issue 834's
triage dispositioned one-time/historical benches as a standing skip class
(measurement records; the league harness owns live re-measurement), and
including them floods the report with known-noise.

Pin detection is a plain substring of the target name over the corpus text —
the faithful systematization of the hand method. Target names are long and
specific, so coincidental hits are rare; when one happens it errs toward
"pinned" (under-reporting), which is the safe direction for a report whose
readers act on the unpinned column.

Vocabulary is COMMITTED (imported), population is DERIVED (BOUNDARY.md + a
`.git` dir), per the workspace rule that deriving both from one walk is what
makes a cross-repo report permanently empty. Repos not checked out on the
running box are invisible to the walk — an ABSENT repo is unverifiable, not
clean (the link-sweep lesson).

    scripts/suite_membership_audit.py             # all present contract repos
    scripts/suite_membership_audit.py ../riir-ai  # or one, by path
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# cfg_gated_target_audit imports tomllib at module level (3.11+); this box runs
# 3.10. The vocabulary functions used here never parse TOML, so a raising stub
# keeps ONE committed classifier instead of a duplicated drifting copy.
try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover (3.10 only)
    import types

    _stub = types.ModuleType("tomllib")
    _stub.loads = lambda *_a, **_k: (_ for _ in ()).throw(
        NotImplementedError("tomllib stub: Python 3.10 box, TOML parsing unavailable"))
    sys.modules["tomllib"] = _stub

import cfg_gated_target_audit as cga  # noqa: E402  (shared vocabulary)
import ci_test_execution_report as cit  # noqa: E402  (shared invocation_texts)

# Files that can pin a suite, per repo. The hand method grepped
# `scripts/ .github/` — that boundary is kept. These are text-scanned, not
# parsed: a target name inside a comment still names the target (a suite that
# refuses to run it "yet" is documented intent, and losing that signal would
# re-hide the 55-day class behind a comment).
PIN_DIRS = ("scripts", ".github")


@dataclass
class Target:
    name: str
    kind: str              # "explicit" | "implicit"
    manifest: Path
    path: str | None = None
    required_features: list[str] = field(default_factory=list)
    harness_false: bool = False


@dataclass
class RepoReport:
    repo: str
    targets: list[Target] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    pinned_elsewhere: list[tuple[str, str]] = field(default_factory=list)
    unpinned: list[Target] = field(default_factory=list)
    broad_run: bool = False

    @property
    def unpinned_load_bearing(self) -> list[Target]:
        return [t for t in self.unpinned if cga.is_load_bearing(t.name)]

    def actionable(self) -> list[Target]:
        """The 865 shape: load-bearing, unpinned by name, default-visible
        (no required-features), and this repo has no broad test run that would
        execute it anyway. Feature-gated opt-in benches are the standing
        expected state, not findings."""
        if self.broad_run:
            return []
        return [
            t for t in self.unpinned_load_bearing
            if not t.required_features
        ]


# ── manifest scanning (regex, not tomllib: this box runs Python 3.10) ──────

SECTION_HDR = re.compile(r"^\s*\[\[?(test|bench)\]\]", re.M)
KEY_RE = re.compile(r'^\s*(name|path|required-features|harness)\s*=\s*(.+)$', re.M)
AUTOTESTS_OFF = re.compile(r"^\s*autotests\s*=\s*false", re.M)


def _toml_scalar(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith(('"', "'")):
        return raw[1:raw.rfind(raw[0])] if raw[1:].find(raw[0]) != -1 else raw[1:]
    return raw


def _toml_strlist(raw: str) -> list[str]:
    return re.findall(r'"([^"]+)"', raw)


def scan_manifest(repo: Path, manifest: Path, rep: RepoReport) -> None:
    text = manifest.read_text(encoding="utf-8", errors="replace")
    explicit_paths: set[str] = set()

    # Split into [[test]] / [[bench]] sections; each runs to the next header.
    hdr_spans = [(m.group(1), m.start(), m.end()) for m in SECTION_HDR.finditer(text)]
    for idx, (kind, start, end) in enumerate(hdr_spans):
        body_end = hdr_spans[idx + 1][1] if idx + 1 < len(hdr_spans) else len(text)
        body = text[end:body_end]
        # The body ends at the FIRST bare `[`-header too (e.g. `[features]`
        # following a lone [[test]]). Any line starting with `[` that is not
        # a continuation of a multi-line array terminates the section.
        cut = None
        depth = 0
        for ln in body.splitlines():
            stripped = ln.strip()
            if depth == 0 and stripped.startswith("[") and not stripped.startswith("[["):
                cut = body.find(ln)
                break
            depth += ln.count("[") - ln.count("]")
        if cut is not None:
            body = body[:cut]

        name = path = None
        req: list[str] = []
        harness_false = False
        for k, raw in KEY_RE.findall(body):
            if k == "name":
                name = _toml_scalar(raw)
            elif k == "path":
                path = _toml_scalar(raw)
            elif k == "required-features":
                req = _toml_strlist(raw)
            elif k == "harness":
                harness_false = _toml_scalar(raw).strip().lower() == "false"
        if kind != "test":
            continue
        if name is None:
            # `[[test]]` without a name derives it from `path`'s stem.
            if path:
                name = Path(path).stem
            else:
                continue
        rep.targets.append(Target(name, "explicit", manifest, path, req, harness_false))
        if path:
            explicit_paths.add(path.replace("\\", "/"))

    if AUTOTESTS_OFF.search(text):
        return
    tests_dir = manifest.parent / "tests"
    if not tests_dir.is_dir():
        return
    for f in sorted(tests_dir.glob("*.rs")):
        rel = f.relative_to(manifest.parent).as_posix()
        if rel in explicit_paths:
            continue
        rep.targets.append(Target(f.stem, "implicit", manifest, rel))


def manifests(repo: Path) -> list[Path]:
    return cga.manifests(repo)


def corpus_text(repo: Path) -> str:
    chunks: list[str] = []
    for d in PIN_DIRS:
        base = repo / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.stat().st_size >= 2_000_000:
                continue
            head = f.open("rb").read(8192)
            if b"\x00" in head:  # binary (icons, lockfiles) — NUL survives replace
                continue
            chunks.append(f.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


# A BROAD run (`cargo test` with no target filter) executes every
# default-compiled integration target without naming any, so "unpinned by
# name" only implies "never executed automatically" for repos with no broad
# run. This keeps riir-neuron-db (`cargo test --all-features`, pinned=0 by
# name) from reading as 51 orphans.
#
# Real-invocation detection is IMPORTED from ci_test_execution_report
# (quoted-label stripping + $(...) lifting — that report's selftest pins the
# `echo "── L4: cargo test ..."` label shapes and the audit scripts' own
# docstring mentions, all of which a raw line match fabricates into broad
# runs). The integration-target filter below is this report's own layer:
# named `--test` runs are not broad, and `--lib`/`--bins`/`--doc`/`--example`
# never execute integration targets (the katgpt-rs scoped core is
# `-p X --lib` — cit.classify() calls that "exec" and is right about the lib
# targets, but wrong for THIS report's population).
NOT_INTEGRATION_FILTERS = ("--lib", "--bins", "--doc", "--example", "--bench")


# Backtick prose (`a plain `cargo test` compiles to nothing`) defeats
# quote-stripping, and audit scripts' own source is full of it — so broad-run
# detection scans only the files where suites actually live (shell + workflow
# YAML), and only non-comment lines. Pin detection keeps the full corpus:
# documented intent in prose/comments still names a target.
SUITE_SUFFIXES = {".sh", ".yml", ".yaml"}


def command_lines(repo: Path) -> list[str]:
    raw: list[str] = []
    for d in PIN_DIRS:
        base = repo / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in SUITE_SUFFIXES:
                continue
            if f.stat().st_size >= 2_000_000:
                continue
            if b"\x00" in f.open("rb").read(8192):
                continue
            for ln in f.read_text(encoding="utf-8", errors="replace").splitlines():
                if not ln.strip().startswith("#"):
                    raw.append(ln)
    # Shell backslash continuations join before classification: the named
    # `--test goat_290_*` filters of riir-ai's Layer 1.16 live on the lines
    # AFTER `G290_OUT=$(cargo test -p riir-gpu --features spec_adapter \` —
    # per-line splitting would misread the head as a broad run.
    joined: list[str] = []
    buf = ""
    for ln in raw:
        buf = f"{buf} {ln.strip()}" if buf else ln
        if buf.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1]
            continue
        joined.append(buf)
        buf = ""
    if buf:
        joined.append(buf)
    return joined


def line_is_broad_test(line: str) -> bool:
    for text in cit.invocation_texts(line):
        if "cargo test" not in text:
            continue
        if any(sup in text for sup in cit.SUPPRESS):  # --no-run/--list compile only
            continue
        if "--tests" in text:  # BEFORE the --test check: "--test" prefixes "--tests"
            return True
        if "--test" in text:
            continue
        if any(f in text for f in NOT_INTEGRATION_FILTERS):
            continue
        return True
    return False


def has_broad_run(repo: Path) -> bool:
    return any(line_is_broad_test(ln) for ln in command_lines(repo))


def audit(repo: Path, workspace_corpus: str | None = None) -> RepoReport:
    rep = RepoReport(repo=repo.name)
    rep.broad_run = has_broad_run(repo)
    for m in manifests(repo):
        scan_manifest(repo, m, rep)
    own = corpus_text(repo)
    for t in rep.targets:
        if t.name in own:
            rep.pinned.append(t.name)
        elif workspace_corpus and t.name in workspace_corpus:
            where = "sibling"
            rep.pinned_elsewhere.append((t.name, where))
        else:
            rep.unpinned.append(t)
    return rep


def derive_repos(workspace: Path) -> list[Path]:
    """A root BOUNDARY.md AND a `.git` DIR — never a typed list."""
    return sorted(
        d for d in workspace.iterdir()
        if d.is_dir() and (d / "BOUNDARY.md").is_file() and (d / ".git").is_dir()
    )


def selftest() -> None:
    import tempfile

    # 1. Explicit + implicit enumeration, autotests=off, common/ excluded.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "tests").mkdir(parents=True)
        (repo / "tests" / "goat_alpha_gate.rs").write_text("#[test] fn a() {}\n")
        (repo / "tests" / "plain_helper.rs").write_text("pub fn h() {}\n")
        (repo / "tests" / "common").mkdir()
        (repo / "tests" / "common" / "mod.rs").write_text("pub const X: u32 = 1;\n")
        (repo / "Cargo.toml").write_text(
            '[package]\nname = "t"\n'
            '[[test]]\nname = "named_bench"\npath = "tests/named_bench.rs"\n'
            'required-features = ["f1"]\n'
            '[[test]]\npath = "tests/derived_name.rs"\n'
            '[[bench]]\nname = "some_bench"\nharness = false\n'
        )
        rep = RepoReport(repo="t")
        scan_manifest(repo, repo / "Cargo.toml", rep)
        names = {t.name: t for t in rep.targets}
        assert set(names) == {"named_bench", "derived_name", "goat_alpha_gate", "plain_helper"}, names
        assert names["named_bench"].kind == "explicit"
        assert names["named_bench"].required_features == ["f1"]
        assert names["derived_name"].kind == "explicit"  # name derived from path stem
        assert names["goat_alpha_gate"].kind == "implicit"
        assert names["plain_helper"].kind == "implicit"  # enumeration lists it; LB split decides
        assert "some_bench" not in names  # benches out of scope by policy
        assert cga.is_load_bearing("goat_alpha_gate")
        assert not cga.is_load_bearing("plain_helper")

    # 2. autotests = false suppresses implicit enumeration.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "tests").mkdir()
        (repo / "tests" / "orphan.rs").write_text("#[test] fn a() {}\n")
        (repo / "Cargo.toml").write_text('[package]\nname = "t"\nautotests = false\n')
        rep = RepoReport(repo="t")
        scan_manifest(repo, repo / "Cargo.toml", rep)
        assert rep.targets == []

    # 3. Pin detection: own corpus pins; section-terminator correctness.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "scripts").mkdir()
        (repo / "scripts" / "gate.sh").write_text("cargo test --test pinned_target -- --nocapture\n")
        (repo / "Cargo.toml").write_text(
            '[package]\nname = "t"\n'
            '[[test]]\nname = "pinned_target"\npath = "tests/p.rs"\n\n'
            '[features]\nf1 = []\n'
            '[[test]]\nname = "orphan_target"\npath = "tests/o.rs"\n'
        )
        rep = audit(repo)
        names = {t.name for t in rep.targets}
        assert names == {"pinned_target", "orphan_target"}, names
        assert rep.pinned == ["pinned_target"]
        assert {t.name for t in rep.unpinned} == {"orphan_target"}

    # 4. Vocabulary delegation: the two reports must agree on the classifier.
    for probe, want in (
        ("goat_290_hybrid_router", True),
        ("certified_frontier_correctness", True),
        ("integration_smoke", False),
        ("aggregate_delegate", False),  # substring false-positive guard
        ("g16f", True),
    ):
        assert cga.is_load_bearing(probe) is want, probe

    # 5. Broad-run line classifier (integration-target layer over cit's
    # invocation extraction): named runs are not broad; --lib is not broad
    # (the scoped-core shape); --tests is; quoted labels are not invocations.
    assert not line_is_broad_test('echo "── L4: cargo test (default features) ──"')
    assert not line_is_broad_test('ok "cargo test --workspace"')
    assert line_is_broad_test('cargo test --workspace --quiet || fail "cargo test --workspace"')
    assert line_is_broad_test('cargo test --all-features --tests')
    assert line_is_broad_test('out="$(cargo test --workspace)"')
    assert line_is_broad_test("cargo test -p seal-view --features texture_vessel")
    assert not line_is_broad_test("cargo test -p riir-gpu --features x --test bench_831 -- --ignored")
    assert not line_is_broad_test("cargo test -p katgpt-core --lib")
    assert not line_is_broad_test("cargo test --doc")

    # 6. command_lines: suite files only, comment lines dropped. The backtick
    # prose that survives line_is_broad_test must never reach it.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "scripts").mkdir()
        (repo / "scripts" / "gate.sh").write_text(
            "cargo test --workspace\n"
            "# a plain `cargo test` compiles the file to nothing\n"
            "echo 'cargo test --workspace'\n"
        )
        (repo / "scripts" / "notes.py").write_text(
            'x = "cargo test --workspace"  # prose in a non-suite file\n'
        )
        (repo / "scripts" / "bin.lock").write_bytes(b"\x00\x01cargo test --workspace")
        lines = command_lines(repo)
        joined = "\n".join(lines)
        assert "cargo test --workspace" in joined
        assert "plain `cargo test`" not in joined  # comment line dropped
        assert len(lines) == 2, lines  # the echo label survives HERE but must
        # classify as non-broad below — the label neutralization lives in
        # line_is_broad_test's quote-stripping, not in the line extractor.
        assert not line_is_broad_test(lines[1])
        assert has_broad_run(repo)

    # 7. Backslash continuations join before classification: a named --test
    # on a continuation line must not leave the head reading as broad.
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td)
        (repo / "scripts").mkdir()
        (repo / "scripts" / "gate.sh").write_text(
            "OUT=$(cargo test -p pkg --features f \\\n"
            "    --test named_gate \\\n"
            "    -- --nocapture)\n"
            "cargo test --workspace --quiet\n"
        )
        lines = command_lines(repo)
        assert len(lines) == 2, lines
        assert not line_is_broad_test(lines[0]), lines[0]  # named via continuation
        assert line_is_broad_test(lines[1])

    print("selftest: ok")


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        selftest()
        return 0
    ws = HERE.parent.parent  # the dir CONTAINING this repo (E:\git on Windows)
    repos = [Path(a) for a in argv if not a.startswith("-")] or derive_repos(ws)
    reports = [audit(r) for r in repos]

    # Cross-repo corpus pass: a target pinned by a SIBLING's suite is not an
    # orphan. Built once from every repo's own corpus, then re-checked only
    # against the unpinned names (cheap: the class is a small minority).
    all_corpus = "\n".join(corpus_text(r) for r in repos)
    for rep in reports:
        still = []
        for t in rep.unpinned:
            if t.name in all_corpus:
                rep.pinned_elsewhere.append((t.name, "sibling"))
            else:
                still.append(t)
        rep.unpinned = still

    total = sum(len(r.targets) for r in reports)
    pinned = sum(len(r.pinned) for r in reports)
    elsewhere = sum(len(r.pinned_elsewhere) for r in reports)
    unpinned = sum(len(r.unpinned) for r in reports)
    lb = sum(len(r.unpinned_load_bearing) for r in reports)
    actionable = [t for r in reports for t in r.actionable()]

    print(f"suite membership audit — repos: {len(reports)}, test targets: {total}")
    print(f"  pinned (own repo): {pinned}   pinned (sibling suite): {elsewhere}   UNPINNED: {unpinned}   unpinned load-bearing: {lb}")
    print(f"  ACTIONABLE (LB + unpinned + default-visible + no broad run in repo): {len(actionable)}")
    print()
    for rep in reports:
        if not rep.targets:
            continue
        act = rep.actionable()
        broad = "broad-run" if rep.broad_run else "NO broad run"
        print(f"[{rep.repo}] targets={len(rep.targets)} pinned={len(rep.pinned)} sibling={len(rep.pinned_elsewhere)} unpinned={len(rep.unpinned)} (LB {len(rep.unpinned_load_bearing)}) [{broad}]")
        for t in act:
            print(f"    [ACTIONABLE] {t.name}  (kind={t.kind})")
        for t in rep.unpinned_load_bearing:
            if t in act:
                continue
            feat = ",".join(t.required_features) or "-"
            print(f"    [LB] {t.name}  (kind={t.kind}, req-feat={feat})")
        for t in rep.unpinned:
            if t in rep.unpinned_load_bearing:
                continue
            feat = ",".join(t.required_features) or "-"
            print(f"         {t.name}  (kind={t.kind}, req-feat={feat})")
    print()
    print("report only (exit 0) — unpinned is not a defect by itself; the")
    print("load-bearing rows are the ones a suite SHOULD name (Issue 865/868 class).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
