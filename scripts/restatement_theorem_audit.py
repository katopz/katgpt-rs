#!/usr/bin/env python3
"""Audit Lean spec theorems that RESTATE a definition and so cannot fail.

A REPORT, not a gate (always exit 0) — same standing as
`percentile_index_audit.py`: most `=` theorems in these repos are not
arithmetic at all (Mathlib-bearing analysis, hypothesis-carrying lemmas), and
a report that exits 1 on the unresolvable majority is a report nobody runs.

# The defect (riir-neuron-db Issue 617, resolved 2026-09-12 `24957a2`)

    def sidecarHeaderSize : Nat :=
      magicSize + versionSize + envelopeRefSize + marginSize + variantSize + nFactsSize

    theorem sidecarHeaderSize_eq_sum : sidecarHeaderSize =
        magicSize + versionSize + envelopeRefSize + marginSize + variantSize + nFactsSize
      := by decide

The RHS is the definition, verbatim. `decide` discharges it no matter what any
`*Size` constant holds, so the theorem is GREEN on every transcription typo it
was written to catch — while its doc comment claimed it was "the `merkle_root`
guard". Four of these shipped in riir-neuron-db for months. `lake build` is
green, `#print axioms` says axiom-free, the proof gate counts it in the audited
surface: nothing in the stack could tell it apart from a theorem that proves
something.

# The criterion: symbolic equality over LEAF constants

"Cannot fail on a constant being wrong" means precisely: **the identity holds
for ANY values of the transcribed constants**. So unfold every *composite*
nullary `def` (one whose body is an expression) down to the **leaf** constants
(bodies that are a bare numeral — these are the transcribed values that can be
wrong), keep the leaves SYMBOLIC, and compare the two sides as polynomials.

Unfolding all the way to numerals instead would be the classic instrument bug:
`neuronShardSize = 464` would compare `464` to `464`, and the audit would
report every sound literal pin as vacuous. The leaf boundary IS the classifier.

# Why `CROSS-DEF` is a separate bucket and not a finding

`commitmentOffset = RAW_PREFIX_LEN` is symbolically equal too — the offset
chain unfolds to the same 13-term sum as the prefix-length definition. It is
NOT the same defect: those are two *independently maintained definitions*, so
the theorem pins one against the other and reds when either chain is
restructured (measured — riir-neuron-db negative-test arm 3 breaks exactly
this theorem). The removed four had an RHS that existed *only* inside the
theorem: delete the theorem and its duplicate goes with it, so nothing is left
to drift. Authored-once-inline vs pinned-across-two-definitions is the whole
distinction, and pooling them would have condemned a load-bearing theorem.

`IDENTITY` (both sides inline, symbolically equal) is reported apart again: it
is an algebra lemma, not a spec claim about a definition.

# Buckets

    RESTATEMENT-INLINE  the finding: NAME = <expression written only here>,
                        symbolically equal. True for any constant values.
    CROSS-DEF           NAME_A = NAME_B, both composite defs, symbolically
                        equal. Vacuous on VALUES, load-bearing on STRUCTURE.
    IDENTITY            both sides inline and symbolically equal (algebra).
    LITERAL             one side a bare numeral, sides NOT symbolically equal
                        — a value pin, the sound shape.
    VALUE-DEPENDENT     sides not symbolically equal, neither a bare numeral.
    STRUCTURAL          the statement is not a bare `=` (∧ → ∀ ↔ < ≤ …).
    UNRESOLVED          a bare `=` this pass cannot reduce to arithmetic
                        (function application, Mathlib, hypotheses, ℝ/ℚ ops).

**UNRESOLVED is not clean** and is never folded into a neighbour — it is
"needs a per-site read", exactly as in the percentile and wasm32 audits.

# First measurement (2026-09-12) — and the oracle it was validated against

    HEAD, 4 repos / 68 .lean / 255 theorems / 85 composite defs / 284 examples:
      RESTATEMENT-INLINE 0 · CROSS-DEF 1 · IDENTITY 0 · LITERAL 13 ·
      VALUE-DEPENDENT 0 · STRUCTURAL 23 · HYPOTHETICAL 199 · UNRESOLVED 19

A zero is only worth what its classifier can see, so it was run against a tree
whose answer was already known by OTHER means — riir-neuron-db at
`24957a2^`, before Issue 617 removed the four restatements
(`git archive 24957a2^ .proofs | tar -x -C /tmp/ndb-pre`). It reports **exactly
those four** and nothing else, and it puts `commitmentOffset_eq_RAW_PREFIX_LEN`
— the theorem an independent perturbation arm proves load-bearing — in
CROSS-DEF rather than with them. Two-sided: 4 before the removal, 0 after.

That run is also what found the classifier's own soundness bug. The first
version pooled every `def` in a repo into ONE table, and riir-neuron-db defines
`zoneHashOffset` in BOTH `Shard/Layout.lean` (`:= 0`) and
`ExperienceGraph/Layout.lean` (a composite): the Shard theorems were unfolding
through ExperienceGraph's chain. The verdict survived by luck; what gave it
away was the caveat NAMING the wrong repo's symbol. Defs are scoped per module
+ transitive imports now, with shadowing applied BOTH ways.

# The UNRESOLVED census (2026-09-12) — 19 rows + 6 conjuncts, read one by one

Adjudicated by reading the source, NOT by an independent instrument — so this
bounds reader error only, in the direction somebody thought to look. Every row
is a value pin the arithmetic model cannot reduce, not a restatement:
`is_wire_safe KeyRole.alphaSeed = false` and `(fresh).initialized = false`
(function application / record projection over a pattern-match definition),
`FREEZE_FLATNESS_THRESHOLD = 3/10` (ℚ division), `tdUpdatePosScaled … = 20800`
(concrete instance), `Module.finrank ℝ (EuclideanSpace ℝ (Fin D)) = D` and
`((√2)⁻¹)² = 1/2` (Mathlib). None can be vacuous in this class's sense: a
restatement needs both sides to be the same expression, and a bare literal RHS
is not the LHS. Re-read when the population moves.

# Still open (deliberately not landed here)

No verdict half. A per-push gate would need a floors file and a CHECKS row in
`docs_gate.py`, and that array is under concurrent edit (Issue 750) — a
membership pin written into a moving array is how the count drift this family
exists to catch gets re-introduced. The report is the deliverable; the gate
follows once that settles.

# Two floors, not one

The finding count is a ceiling over whatever the walk can SEE, so it is
meaningless without the population underneath it: `.lean` files walked,
theorems parsed, defs resolved. A tokenizer regression takes the population to
~0 and every ceiling passes. `selftest()` REFUSES to print a report if the
classifier disagrees with the riir-neuron-db ground truth encoded below.
"""

import os
import re
import sys

MAX_UNFOLD_ROUNDS = 32
MAX_TOKENS = 4000

# ── Lean surface ──────────────────────────────────────────────────────────

# A top-level command starts at column 0. Modifiers may precede the keyword.
CMD_KEYWORDS = (
    "def", "theorem", "lemma", "abbrev", "example", "instance", "structure",
    "inductive", "namespace", "end", "open", "import", "section", "variable",
    "macro", "notation", "attribute", "deriving", "class", "axiom", "mutual",
)
MODIFIERS = ("private", "protected", "noncomputable", "partial", "unsafe",
             "nonrec", "scoped", "local", "@[")

TOKEN_RE = re.compile(
    r"""(?P<assign>:=)
      | (?P<num>\d+)
      | (?P<ident>[A-Za-z_Ͱ-Ͽᵉ-ᵪ][A-Za-z0-9_'.Ͱ-Ͽ₀-₉]*)
      | (?P<op>[+*()\-/=<>^,:;\[\]{}|]|≤|≥|≠|∧|∨|→|↔|∀|∃|·|⁻|√)
      | (?P<ws>\s+)
      | (?P<other>.)
    """,
    re.VERBOSE,
)

# A statement carrying any of these is not a bare arithmetic `=`.
STRUCTURAL_TOKENS = {"∧", "∨", "→", "↔", "∀", "∃", "<", ">", "≤", "≥", "≠",
                     ",", "λ", "fun", "if", "then", "else", "match"}


def strip_comments(src):
    """Remove `--` line comments and `/- … -/` (incl. `/-- … -/`) blocks.

    Nesting matters: Lean block comments nest, and these files embed `/-!`
    module docs that quote code. Column positions are preserved by replacing
    comment bodies with spaces, so the column-0 command split still works."""
    out = []
    i, n, depth = 0, len(src), 0
    while i < n:
        if src.startswith("/-", i):
            depth += 1
            out.append("  ")
            i += 2
        elif src.startswith("-/", i) and depth:
            depth -= 1
            out.append("  ")
            i += 2
        elif depth:
            out.append("\n" if src[i] == "\n" else " ")
            i += 1
        elif src.startswith("--", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def top_level_blocks(src):
    """Split into (keyword, body) blocks at column-0 command keywords."""
    lines = strip_comments(src).splitlines()
    blocks, cur, kw = [], [], None
    for line in lines:
        stripped = line.lstrip()
        starts = line[:1] not in ("", " ", "\t")
        head = stripped.split(":")[0].split()
        first = head[0] if head else ""
        if starts and (first in CMD_KEYWORDS or stripped.startswith(MODIFIERS)):
            if kw:
                blocks.append((kw, "\n".join(cur)))
            words = stripped.split()
            kw = next((w for w in words if w in CMD_KEYWORDS), words[0])
            cur = [line]
        elif kw:
            cur.append(line)
    if kw:
        blocks.append((kw, "\n".join(cur)))
    return blocks


def tokenize(text):
    toks = []
    for m in TOKEN_RE.finditer(text):
        if m.lastgroup == "ws":
            continue
        toks.append(m.group())
        if len(toks) > MAX_TOKENS:
            break
    return toks


def split_at_depth0(toks, needle):
    """Index of the first `needle` token at bracket depth 0, else -1."""
    depth = 0
    for i, t in enumerate(toks):
        if t in "([{":
            depth += 1
        elif t in ")]}":
            depth -= 1
        elif depth == 0 and t == needle:
            return i
    return -1


def parse_def(body):
    """`def NAME : T := BODY` → (name, body_tokens) for NULLARY defs only.

    A def taking binders cannot be textually substituted, so it is skipped —
    which is also why an applied identifier later reads as UNRESOLVED rather
    than being silently mis-unfolded."""
    toks = tokenize(body)
    if not toks:
        return None
    try:
        kw = next(i for i, t in enumerate(toks) if t in ("def", "abbrev"))
    except StopIteration:
        return None
    if kw + 1 >= len(toks):
        return None
    name = toks[kw + 1]
    rest = toks[kw + 2:]
    assign = split_at_depth0(rest, ":=")
    if assign < 0:
        return None
    sig, rhs = rest[:assign], rest[assign + 1:]
    # Nullary: the signature is either empty or `: TYPE` — no binders.
    if sig and sig[0] != ":":
        return None
    if not rhs:
        return None
    return name, rhs


def parse_theorem(body):
    """`theorem NAME : STATEMENT := PROOF` → (name, statement_tokens, proof)."""
    toks = tokenize(body)
    if not toks:
        return None
    try:
        kw = next(i for i, t in enumerate(toks) if t in ("theorem", "lemma"))
    except StopIteration:
        return None
    if kw + 1 >= len(toks):
        return None
    name = toks[kw + 1]
    rest = toks[kw + 2:]
    colon = split_at_depth0(rest, ":")
    if colon < 0:
        return None
    if colon != 0:  # binders before the `:` — hypotheses, not this class
        return name, None, None
    after = rest[colon + 1:]
    assign = split_at_depth0(after, ":=")
    stmt = after[:assign] if assign >= 0 else after
    proof = after[assign + 1:] if assign >= 0 else []
    return name, stmt, proof


# ── Symbolic arithmetic over leaf constants ───────────────────────────────
#
# A polynomial is {monomial: coefficient} with monomial a sorted tuple of leaf
# symbol names (() = the constant term). Only `+` and `*` over ℕ are modeled:
# `-` and `/` are Nat-truncating and NOT polynomial-safe, so any statement
# carrying one lands in UNRESOLVED rather than being reduced wrongly.

class NotArithmetic(Exception):
    pass


def poly_norm(p):
    """Drop zero-coefficient monomials — `0 + x` and `x` must compare equal.

    Without this the offset-chain origin (`zoneHashOffset := 0`) leaves a
    literal `(): 0` term on one side only, and every chain-vs-sum theorem
    reads VALUE-DEPENDENT. A canonical form is not optional for an equality
    classifier."""
    return {m: c for m, c in p.items() if c != 0}


def poly_add(a, b):
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0) + c
    return poly_norm(out)


def poly_mul(a, b):
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(sorted(ma + mb))
            out[m] = out.get(m, 0) + ca * cb
    return poly_norm(out)


class Parser:
    """expr := term (('+') term)* ; term := atom (('*') atom)* ; atom := num | ident | '(' expr ')'"""

    def __init__(self, toks, defs, depth=0):
        self.toks, self.i, self.defs, self.depth = toks, 0, defs, depth

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def expr(self):
        p = self.term()
        while self.peek() == "+":
            self.i += 1
            p = poly_add(p, self.term())
        return p

    def term(self):
        p = self.atom()
        while self.peek() == "*":
            self.i += 1
            p = poly_mul(p, self.atom())
        return p

    def atom(self):
        t = self.peek()
        if t is None:
            raise NotArithmetic("truncated")
        if t == "(":
            self.i += 1
            p = self.expr()
            if self.peek() != ")":
                raise NotArithmetic("unbalanced")
            self.i += 1
        elif re.fullmatch(r"\d+", t):
            self.i += 1
            p = {(): int(t)}
        elif re.fullmatch(r"[A-Za-z_].*", t):
            self.i += 1
            p = self.symbol(t)
        else:
            raise NotArithmetic(f"token {t!r}")
        # Juxtaposition = function application; not modeled, and must not be
        # silently dropped.
        nxt = self.peek()
        if nxt is not None and (re.fullmatch(r"[A-Za-z_].*|\d+", nxt) or nxt == "("):
            raise NotArithmetic("application")
        return p

    def symbol(self, name):
        body = self.defs.get(name)
        if body is None or self.depth >= MAX_UNFOLD_ROUNDS:
            return {(name,): 1}  # leaf (or unresolvable) — stays symbolic
        sub = Parser(body, self.defs, self.depth + 1)
        p = sub.expr()
        if sub.i != len(sub.toks):
            raise NotArithmetic("trailing in def body")
        return poly_norm(p)


def is_leaf_body(toks):
    """A def body that is a bare numeral — the transcribed value itself."""
    core = [t for t in toks if t not in "()"]
    return len(core) == 1 and re.fullmatch(r"\d+", core[0]) is not None


def to_poly(toks, composites):
    p = Parser(toks, composites)
    out = p.expr()
    if p.i != len(p.toks):
        raise NotArithmetic("trailing tokens")
    return poly_norm(out)


def is_single_ident(toks):
    core = [t for t in toks if t not in "()"]
    return len(core) == 1 and re.fullmatch(r"[A-Za-z_].*", core[0]) is not None


def is_bare_numeral(toks):
    core = [t for t in toks if t not in "()"]
    return len(core) == 1 and re.fullmatch(r"\d+", core[0]) is not None


# ── Classification ────────────────────────────────────────────────────────

RESTATEMENT = "RESTATEMENT-INLINE"
CROSS_DEF = "CROSS-DEF"
IDENTITY = "IDENTITY"
LITERAL = "LITERAL"
VALUE_DEP = "VALUE-DEPENDENT"
STRUCTURAL = "STRUCTURAL"
HYPOTHETICAL = "HYPOTHETICAL"
UNRESOLVED = "UNRESOLVED"

ORDER = [RESTATEMENT, CROSS_DEF, IDENTITY, LITERAL, VALUE_DEP, STRUCTURAL,
         HYPOTHETICAL, UNRESOLVED]


def split_conjuncts(toks):
    """Split a token stream on top-level `∧`."""
    parts, cur, depth = [], [], 0
    for t in toks:
        if t in "([{":
            depth += 1
        elif t in ")]}":
            depth -= 1
        if depth == 0 and t == "∧":
            parts.append(cur)
            cur = []
        else:
            cur.append(t)
    parts.append(cur)
    return [p for p in parts if p]


def classify(stmt, composites, zero_leaves=()):
    """→ (bucket, note). `composites` maps a composite def name to its body.

    Two passes, and the second one is REPORTED rather than folded in. Pass 1
    keeps every numeral-bodied leaf symbolic. Pass 2 additionally substitutes
    the `:= 0` leaves — an offset chain anchored at `zoneHashOffset := 0`
    otherwise carries that symbol on the LHS and not on the RHS, so the two
    sides differ by a term that is structurally zero.

    A `:= 0` origin is a structural anchor, not a transcribed measurement, so
    pass 2 is the honest reading of "true for any constant values". But it can
    only ever WIDEN the finding set — a theorem that genuinely depends on some
    constant being zero would be caught by it too — so whichever zero-leaves
    the verdict needed are NAMED in the note instead of disappearing into it.
    """
    if stmt is None:
        return HYPOTHETICAL, "binders / hypotheses — quantified, not this class"
    if any(t in STRUCTURAL_TOKENS for t in stmt):
        # A top-level `∧` chain of equations is the one shape that CAN be
        # wholly vacuous while carrying a connective, and it is the blind spot
        # a bare "not a bare `=`" reject would keep: an offset-chain theorem is
        # a conjunction. It is vacuous only if EVERY conjunct is — one
        # value-dependent conjunct still reds — so the rule is `all`, never
        # `any`, and the per-conjunct tally is printed either way.
        parts = split_conjuncts(stmt)
        if len(parts) > 1 and all(split_at_depth0(p, "=") >= 0
                                  and not any(t in STRUCTURAL_TOKENS - {"∧"}
                                              for t in p) for p in parts):
            sub = [classify(p, composites, zero_leaves) for p in parts]
            kinds = [s[0] for s in sub]
            if all(k in (RESTATEMENT, IDENTITY) for k in kinds):
                return RESTATEMENT, (f"all {len(parts)} conjuncts restate a "
                                     f"definition")
            tally = " ".join(f"{kinds.count(k)}×{k}" for k in ORDER
                             if kinds.count(k))
            return STRUCTURAL, f"∧ of {len(parts)} equations: {tally}"
        return STRUCTURAL, "not a bare `=`"
    eq = split_at_depth0(stmt, "=")
    if eq < 0:
        return STRUCTURAL, "no top-level `=`"
    lhs, rhs = stmt[:eq], stmt[eq + 1:]
    if not lhs or not rhs:
        return UNRESOLVED, "empty side"
    try:
        pl, pr = to_poly(lhs, composites), to_poly(rhs, composites)
    except NotArithmetic as e:
        return UNRESOLVED, str(e)
    caveat = ""
    if pl != pr and zero_leaves:
        widened = dict(composites)
        widened.update({z: ["0"] for z in zero_leaves})
        try:
            pl2, pr2 = to_poly(lhs, widened), to_poly(rhs, widened)
        except NotArithmetic:
            pl2 = pr2 = None
        if pl2 is not None and pl2 == pr2:
            used = sorted({s for m in set(pl) ^ set(pr) for s in m}
                          & set(zero_leaves))
            pl = pr = pl2
            caveat = f" [modulo zero-origin {', '.join(used)}]" if used else ""
    if pl != pr:
        if is_bare_numeral(lhs) or is_bare_numeral(rhs):
            return LITERAL, "pins a value"
        return VALUE_DEP, "holds only for the transcribed values"
    l_def = is_single_ident(lhs) and lhs[-1] in composites
    r_def = is_single_ident(rhs) and rhs[-1] in composites
    if l_def and r_def:
        return CROSS_DEF, "two independently maintained definitions" + caveat
    if l_def or r_def:
        return RESTATEMENT, "RHS is an expression authored only here" + caveat
    return IDENTITY, "algebra over inline expressions" + caveat


def collect_defs(blocks):
    """→ (composites, zero_leaves) for ONE file's blocks."""
    composites, zero_leaves = {}, set()
    for kw, body in blocks:
        if kw not in ("def", "abbrev"):
            continue
        d = parse_def(body)
        if d and not is_leaf_body(d[1]):
            composites[d[0]] = d[1]
        elif d and [t for t in d[1] if t not in "()"] == ["0"]:
            zero_leaves.add(d[0])
    return composites, zero_leaves


def module_of(path, proofs):
    rel = os.path.relpath(path, proofs)
    return rel[:-len(".lean")].replace(os.sep, ".")


def imports_of(blocks):
    out = []
    for kw, body in blocks:
        if kw != "import":
            continue
        for line in body.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "import":
                out.append(parts[1])
    return out


def audit_repo(repo_root):
    """Scope every def table to its OWN module plus its transitive imports.

    A single repo-wide table is UNSOUND and was measurably so: riir-neuron-db
    defines `zoneHashOffset` in BOTH `Shard/Layout.lean` (`:= 0`, the chain
    origin) and `ExperienceGraph/Layout.lean` (a composite mid-chain offset).
    Pooled, the Shard theorems unfolded through ExperienceGraph's chain — the
    verdict survived by luck, and the wrong repo's symbol was named in the
    note. Lean itself scopes by module + `open`; so does this."""
    proofs = os.path.join(repo_root, ".proofs")
    files = []
    for dp, dns, fns in os.walk(proofs):
        dns[:] = [d for d in dns if d not in (".lake", "build")]
        files += [os.path.join(dp, f) for f in sorted(fns) if f.endswith(".lean")]

    per_file, examples = {}, 0
    for path in sorted(files):
        try:
            src = open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        blocks = top_level_blocks(src)
        per_file[path] = {
            "blocks": blocks,
            "module": module_of(path, proofs),
            "imports": imports_of(blocks),
            "defs": collect_defs(blocks),
        }
        examples += sum(1 for kw, _ in blocks if kw == "example")

    by_module = {v["module"]: k for k, v in per_file.items()}

    def scope(path, seen=None):
        """Own defs win over imported ones, as `open` + shadowing would."""
        seen = seen if seen is not None else set()
        info = per_file[path]
        if info["module"] in seen:
            return {}, set()
        seen.add(info["module"])
        comp, zero = {}, set()
        for imp in info["imports"]:
            src_path = by_module.get(imp)
            if src_path:
                c, z = scope(src_path, seen)
                comp.update(c)
                zero |= z
        c, z = info["defs"]
        comp.update(c)
        zero |= z
        # Shadowing runs BOTH ways: a name redefined here as a composite is no
        # longer a zero leaf, and one redefined here as `:= 0` must drop the
        # imported composite — otherwise the import silently wins and the
        # unfold walks a chain this module never had.
        zero -= set(c)
        for name in z:
            comp.pop(name, None)
        return comp, zero

    rows, counts = [], {b: 0 for b in ORDER}
    all_defs = 0
    for path in sorted(per_file):
        composites, zero_leaves = scope(path)
        all_defs = max(all_defs, 0) + len(per_file[path]["defs"][0])
        for kw, body in per_file[path]["blocks"]:
            if kw not in ("theorem", "lemma"):
                continue
            t = parse_theorem(body)
            if t is None:
                counts[UNRESOLVED] += 1
                rows.append((UNRESOLVED, path, "?", "unparsed block"))
                continue
            name, stmt, _ = t
            bucket, note = classify(stmt, composites, zero_leaves)
            counts[bucket] += 1
            rows.append((bucket, path, name, note))
    return {
        "files": len(files),
        "defs": all_defs,
        "examples": examples,
        "theorems": sum(counts.values()),
        "counts": counts,
        "rows": rows,
    }


# ── Ground truth: the riir-neuron-db theorems Issue 617 adjudicated ───────
#
# Every one of these was decided by an INDEPENDENT instrument (`lake build`
# deletion probes, planted-canary dependency checks, and the 17-arm
# perturbation harness that measures which file reds), not by this classifier.
# That is what makes them usable as an oracle: a bucket boundary is only
# testable against cases whose answer is known some other way.

GROUND_TRUTH = [
    # (source, expected bucket) — the four REMOVED restatements
    ("""def styleWeightsSize : Nat := 4 * STYLE_DIM
def STYLE_DIM : Nat := 64
def zoneHashSize : Nat := 32
def zoneHashOffset : Nat := 0
def styleWeightsOffset : Nat := zoneHashOffset + zoneHashSize
def neuronShardSize : Nat := styleWeightsOffset + styleWeightsSize
theorem neuronShardSize_eq_sum : neuronShardSize =
    zoneHashSize + styleWeightsSize := by decide""",
     "neuronShardSize_eq_sum", RESTATEMENT),
    # the LITERAL sibling that carries the real coverage — must NOT be flagged
    ("""def STYLE_DIM : Nat := 64
def zoneHashSize : Nat := 32
def styleWeightsSize : Nat := 4 * STYLE_DIM
def neuronShardSize : Nat := zoneHashSize + styleWeightsSize
theorem neuronShardSize_eq : neuronShardSize = 288 := rfl""",
     "neuronShardSize_eq", LITERAL),
    # two independently maintained definitions — vacuous on values, NOT a finding
    ("""def aSize : Nat := 8
def bSize : Nat := 4
def aOff : Nat := 0
def bOff : Nat := aOff + aSize
def commitmentOffset : Nat := bOff + bSize
def RAW_PREFIX_LEN : Nat := aSize + bSize
theorem commitmentOffset_eq_RAW_PREFIX_LEN : commitmentOffset = RAW_PREFIX_LEN := rfl""",
     "commitmentOffset_eq_RAW_PREFIX_LEN", CROSS_DEF),
    # the transitive case: the RHS sum hides one level down (RAW_PREFIX_LEN)
    ("""def aSize : Nat := 8
def bSize : Nat := 4
def cSize : Nat := 2
def RAW_PREFIX_LEN : Nat := aSize + bSize
def experienceNodeSize : Nat := RAW_PREFIX_LEN + cSize
theorem experienceNodeSize_eq_sum_fields : experienceNodeSize =
    aSize + bSize + cSize := rfl""",
     "experienceNodeSize_eq_sum_fields", RESTATEMENT),
    # a conjunction is never this class
    ("""def x : Nat := 1
def y : Nat := 2
theorem offset_chain_monotone : x < y ∧ y < 3 := by decide""",
     "offset_chain_monotone", STRUCTURAL),
    # hypothesis-carrying: binders before the `:` — its OWN bucket, because
    # pooling it with the connective rejects would hide how few statements the
    # arithmetic pass ever sees (202 of 222 in the first workspace run).
    ("""theorem tamper (h : a ≠ b) : f a = f b := by simp""",
     "tamper", HYPOTHETICAL),
    # a top-level ∧ of equations IS reachable — all conjuncts vacuous ⇒ vacuous
    ("""def a : Nat := 3
def b : Nat := 4
def s : Nat := a + b
def t : Nat := a + b + b
theorem both_restate : s = a + b ∧ t = a + b + b := by decide""",
     "both_restate", RESTATEMENT),
    # …but ONE value-dependent conjunct means the theorem can still red
    ("""def a : Nat := 3
def b : Nat := 4
def s : Nat := a + b
theorem mixed : s = a + b ∧ s = 7 := by decide""",
     "mixed", STRUCTURAL),
    # function application is NOT arithmetic — must not reduce to a false equal
    ("""def k : Nat := 3
theorem app_is_unresolved : f k = f k := rfl""",
     "app_is_unresolved", UNRESOLVED),
    # Nat subtraction is truncating — never claim symbolic equality over it
    ("""def a : Nat := 5
def b : Nat := 2
def d : Nat := a - b
theorem sub_unresolved : d = a - b := rfl""",
     "sub_unresolved", UNRESOLVED),
    # a doc comment quoting the removed theorem must not be parsed as code
    ("""/-- ⚠ A former `sidecarHeaderSize_eq_sum` proved
    theorem sidecarHeaderSize_eq_sum : sidecarHeaderSize = magicSize -/
def magicSize : Nat := 4
def versionSize : Nat := 4
def sidecarHeaderSize : Nat := magicSize + versionSize
theorem sidecarHeaderSize_eq : sidecarHeaderSize = 8 := rfl""",
     "sidecarHeaderSize_eq", LITERAL),
]


def selftest():
    """Refuse to print a report if the classifier disagrees with the oracle.

    A silent classifier regression is the failure mode this whole family has
    been burned by: the finding count drops, the summary still prints, and the
    zero reads as good news."""
    bad = []
    for src, want_name, want_bucket in GROUND_TRUTH:
        composites, zero_leaves = {}, set()
        blocks = top_level_blocks(src)
        for kw, body in blocks:
            if kw in ("def", "abbrev"):
                d = parse_def(body)
                if d and not is_leaf_body(d[1]):
                    composites[d[0]] = d[1]
                elif d and [t for t in d[1] if t not in "()"] == ["0"]:
                    zero_leaves.add(d[0])
        got = None
        for kw, body in blocks:
            if kw not in ("theorem", "lemma"):
                continue
            t = parse_theorem(body)
            if t and t[0] == want_name:
                got = classify(t[1], composites, zero_leaves)[0]
        if got != want_bucket:
            bad.append(f"  {want_name}: expected {want_bucket}, got {got}")
    if bad:
        print("✗ selftest FAILED — the classifier moved; report suppressed")
        print("\n".join(bad))
        sys.exit(1)


# The module-scoping oracle: two files defining the SAME name differently.
# Pooled into one table this reads VALUE-DEPENDENT; scoped per module it is the
# restatement it actually is. Measured on the live tree before it was written —
# riir-neuron-db defines `zoneHashOffset` in both Shard and ExperienceGraph.
SCOPING_FIXTURE = {
    "A.lean": """import B
def sz : Nat := 4
def off : Nat := 0
def total : Nat := off + sz
theorem total_eq_sum : total = sz := rfl
""",
    "B.lean": """def other : Nat := 7
def off : Nat := other + other
""",
}


def selftest_scoping():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="restatement-selftest-")
    try:
        proofs = os.path.join(tmp, ".proofs")
        os.makedirs(proofs)
        for name, src in SCOPING_FIXTURE.items():
            with open(os.path.join(proofs, name), "w", encoding="utf-8") as fh:
                fh.write(src)
        r = audit_repo(tmp)
        got = [(b, n, note) for b, _, n, note in r["rows"] if n == "total_eq_sum"]
        if len(got) != 1 or got[0][0] != RESTATEMENT or "off" not in got[0][2]:
            print("✗ selftest FAILED — module scoping regressed; "
                  "report suppressed")
            print(f"  total_eq_sum: expected {RESTATEMENT} caveating A's own "
                  f"`off`, got {got}")
            sys.exit(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def repos(root):
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "BOUNDARY.md"))
        and os.path.isdir(os.path.join(root, d, ".git"))
        and os.path.isdir(os.path.join(root, d, ".proofs"))
    )


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    selftest()
    selftest_scoping()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(here)
    targets = [os.path.abspath(a) for a in args] or \
        [os.path.join(root, d) for d in repos(root)]

    print("=== Lean restatement-theorem audit "
          "(riir-neuron-db Issue 617 class) ===\n")
    tot = {b: 0 for b in ORDER}
    files = defs = thms = examples = 0
    findings = []
    for repo in targets:
        if not os.path.isdir(os.path.join(repo, ".proofs")):
            continue
        r = audit_repo(repo)
        files += r["files"]
        defs += r["defs"]
        thms += r["theorems"]
        examples += r["examples"]
        for b in ORDER:
            tot[b] += r["counts"][b]
        name = os.path.basename(repo)
        print(f"── {name}: {r['files']} .lean · {r['theorems']} theorems · "
              f"{r['defs']} composite defs · {r['examples']} examples")
        print("   " + " · ".join(f"{b} {r['counts'][b]}"
                                 for b in ORDER if r["counts"][b]))
        for bucket, path, tname, note in r["rows"]:
            rel = os.path.relpath(path, repo)
            if bucket == RESTATEMENT:
                findings.append((name, rel, tname, note))
                print(f"   ⛔ {RESTATEMENT}  {rel} :: {tname}  — {note}")
            elif bucket in (CROSS_DEF, IDENTITY) or verbose:
                print(f"      {bucket}  {rel} :: {tname}  — {note}")
        print()

    print(f"POPULATION: {len(targets)} repo(s) (derived: BOUNDARY.md + .git + "
          f".proofs) · {files} .lean files · {thms} theorems · "
          f"{defs} composite defs · {examples} examples")
    print("VERDICT:    " + " · ".join(f"{b} {tot[b]}" for b in ORDER))
    print(f"\n⛔ {len(findings)} RESTATEMENT-INLINE finding(s) — a theorem true "
          f"for ANY constant values, whose RHS exists only inside it.")
    print(f"   {tot[CROSS_DEF]} CROSS-DEF are NOT findings (two definitions "
          f"pinned against each other — keep).")
    print(f"   {tot[UNRESOLVED]} UNRESOLVED need a per-site read — never a pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
