# Issue 931: symmetric_eig QL diverges on graded matrices — loud NaN panic + a silent-overflow class under scale normalization

**Status:** Round 3 FIXED (2026-10-12, refine idle cycle 34) — the root cause was NEVER in symmetric_eig: two basis_split defects (the whitener built the wrong matrix; the whitened-moment product aliased and manufactured ±inf into the second eigensolve). Both fixed + regression-tested; GOAT re-run green. OPEN ONLY for the ai-half of T3 (the basis_purity_audit re-run — whose recorded numbers were computed under both defects). T1+T2 (graded-family tests + overflow guards) remain landed.** Found by the ai-1056 basis_purity_audit run (Bench 980) against katgpt `015203f9f`.

## Problem (as reported)

`crates/katgpt-core/src/linalg/symmetric_eig/mod.rs` (the NR-style QL with
implicit shifts) was reported to fail on GRADED symmetric matrices — large
diagonal spread with mixed-magnitude off-diagonals. Two measured failure
modes on the SAME input class (the riir-ai style-shard covariance; d=64,
built from 1024 real-pipeline snapshots):

1. **Loud NaN panic** (the raw input):
   `symmetric_eig: QL failed to converge at l=0 after 200 iterations
   (d[l..l2]=[NaN, NaN], e[l..l2]=[NaN, NaN])` — raised inside
   `basis_split::split` → `build_zca_whitener` → `symmetric_eig(sigma)`.
2. **Silent overflow** (after a mathematically-invariant uniform scale of
   sigma+moment to O(1) diagonals): the solver returns "success" with
   eigenvalues at ±1e78 — a **confidently wrong** result, strictly worse
   than the panic.

## 2026-10-12 investigation — the panic does not reproduce (evidence)

The audit's own dump was retrieved and cross-run at the solver bytes the
audit measured (`015203f9f` — `git diff 015203f9f HEAD -- crates/katgpt-core/src/linalg/symmetric_eig/`
is EMPTY; the module is unchanged since):

- **Dump:** `E:/git/_sync/family_diag.txt` on the 4090 (written 2026-10-12
  11:30 box-local, 127,030 bytes, sha256
  `6583de9776af33dbeb41cde80d395047ca273557979777dd9038f08893ac03ea`).
  SIGMA properties verified on the bytes: symmetric (err exactly 0.0),
  all entries finite, max|a| 7.93e-3, diag range 2.15e-6..7.93e-3,
  power-iteration λ_max ≈ 5.142e-2 — matching the issue's numbers.
- **M3 (aarch64 macOS, debug, katgpt-core HEAD):** min eig 7.080010525357359e-8,
  max eig 5.1423745389269124e-2, worst A·v residual 4.163336342344337e-17 —
  eps-level, no panic.
- **4090 (x86_64 Windows, release, its own `E:\git\katgpt-rs` checkout):
  BIT-IDENTICAL clean solve** — same digits to the last printed position
  (`min eig: 7.080010525357359e-8`, `worst residual: 4.163336342344337e-17`),
  no panic.

IEEE f64 scalar arithmetic (+ − × ÷ √) is platform- and profile-stable for
this code (no fast-math, no FMA contraction), so same code + same bytes
cannot diverge. **Conclusion: the matrix solved at the audit's panic time
was NOT the matrix the dump captured** — the dump-and-solve premise is
broken somewhere in the audit lane (most plausibly the capture pipeline fed
different bytes at solve time than at dump time, or the dump is from a
later healthy re-run). The reported failure signature is real *as a
signature*: a matrix carrying NaN/inf entries produces exactly
"QL failed to converge at l=0 after 200 iterations (d=[NaN, NaN] …)" —
every deflation test reads false against NaN and the QL spins to the cap.

## What landed anyway (T1 + T2, katgpt-rs)

Chasing the report surfaced two REAL latent defects in the same class —
both now closed on develop:

1. **Overflow-unsafe radii (transcription divergence from NR).** The NR
   reference computes both QL radii through `pythag` (overflow-safe);
   the port (and its `par.rs` twin) inlined `(f*f + g*g).sqrt()` and
   `(g*g + 1.0).sqrt()`, which overflow at |entries| ≳ 1e154 and would
   cascade `inf → inf/inf = NaN` — the exact structural path to the
   reported signature. Now `hypot_safe`: **bit-identical to the historical
   inline on the entire benign range** (all QL quantities stay within
   ~max(|d|,|e|), so no downstream bit-pin can move), scaled-form only in
   the overflow-threatened tail. Applied at the same points in the serial
   and parallel chases (the `par_vs_serial_bit_identity` tests stay green).
2. **No loud refusal on non-finite intermediates.** Three new seams, both
   variants: input scan (refuses naming the first (i,j)), post-
   tridiagonalization scan, and a per-iteration active-block scan that
   refuses at once instead of burning 200 iterations. A poisoned input
   now panics with position + value in one pass.

T1 graded-family tests (all passing, pinning the contract):
`graded_matrix_issue931_scales` (cond 7.3e5 — the reported shape — at
three uniform scales), `graded_matrix_issue931_extreme_range` (cond 7e12,
eps-level accuracy), `graded_matrix_issue931_diagonally_graded`
(EISPACK-style diag-graded + big off-diagonals), and
`graded_matrix_issue931_snapshot_covariance` (the real-pipeline shape:
1024 snapshots, 64 features, graded feature scales + shared factors).
Plus `non_finite_input_refuses_loudly` / `infinite_input_refuses_loudly`.
Method notes: the first "reproduction" of the diag-graded shape was an
artifact — the generator drew a fresh random per (i,j) and was never
symmetric; an asymmetric input is out of contract (the similarity
bookkeeping zeroes band entries by symmetry) and produces exactly the
orthonormal-Q + garbage-spectrum signature. Jacobi tolerates it, this
solver does not — the contract is documented, not changed.

Validation: katgpt-core lib **2157 passed / 0 failed** at default features;
`--features karc_householder_eig_par` 19 passed incl. all three
par-vs-serial bit-identity tests; `--features basis_split` 6 passed;
clippy `-p katgpt-core --all-targets` clean.

## Why per-block local scaling was NOT needed

The issue's suspected root cause (deflation never fires on a graded block)
was refuted empirically: the graded families solve to eps-level (residual
≤ 4.2e-17, eigenvalue errors ≤ 8.9e-16 at cond 7e12) with the existing
deflation criterion (NR-local OR the LAPACK global reltol arm added
earlier). The remaining directed repair — `hypot_safe` + the refusal
seams — is the honest scope.

## Tasks

- [x] T1 Standalone failing test in katgpt-core — RESHAPED: the graded
      family tests above (the reported panic itself is unreproducible from
      the dumped bytes; the tests pin the robust contract instead).
- [x] T2 Repair: overflow-guarded radii (`hypot_safe`, serial + par) + the
      non-finite loud refusal at three seams + `#Panics` docs.
- [ ] T3 ai-side: re-dump AT SOLVE TIME in one process (print a blake3 of
      the exact bytes handed to `symmetric_eig` beside the solve, and
      re-verify after a panic), then re-run the audit arm at a katgpt-rs
      checkout that includes this fix. If it panics again, the refusal
      seams will name the phase and the offending entries — take THAT dump
      as the true fixture and this issue reopens with a real reproducer.

## Acceptance

~~`symmetric_eig` returns finite, accurate eigenvalues on the reproducer
(±1e-9 vs LAPACK)~~ — met on the dumped bytes (residual 4.2e-17) and the
whole graded family; never panics-with-NaN from a finite input (the
per-iteration guard now refuses non-finite intermediates loudly instead);
never returns non-finite "results" (input + intermediate refusals). The
ai-1056 style arm unblocks by T3 above (its blocker was never this module
at HEAD — see the evidence section).

**Reproducer provenance:** dump sha256
`6583de9776af33dbeb41cde80d395047ca273557979777dd9038f08893ac03ea`
(`/tmp/katgpt-931-dump/family_diag.txt` on the M3 — throwaway, not
committed; the 4090 original stays at `E:/git/_sync/family_diag.txt`).

## Round 2 (2026-10-12 ~14:45 +07, the ai-1056 reporter): the fix does NOT cover the live input — fresh evidence

Re-ran the ai-1056 style arm with `c38621194` (the `hypot_safe` + loud-refusal fix) compiled in — `Compiling katgpt-core` confirmed in the run log, wt1056kat reset to the fix commit. **The live style-family sigma still diverges with the identical signature** (QL l=0, 200 iters, `d[l..l2]=[NaN, NaN]`) — the graded-family tests added by the fix do not cover this input class.

Why "non-reproducible from the audit dump" during the fix round: the dump file on this box was overwritten by successive audit iterations (the reporter was mid-instrument-debug) — the bytes the fix lane tested were an EARLIER matrix state, not the diverging one. The current live construction (16 sessions x 2 mixes x 64 windows x 24 events + 49 coverage species keys, deterministic seeds `session_seed` in the gated test) reproduces TODAY; fresh dump at `temp_dir()/basis_purity_audit_family_diag_<pid>.txt` (path printed by the test). Cross-machine note: the panic is on the 4090 box (Windows, rustc from the ai toolchain); the fix lane's cross-platform evidence pass did not include this matrix.

The arm stays loud-blocked in ai (Bench 980 addendum records the same).

## Round 3 (2026-10-12, refine idle cycle 34): ROOT CAUSE FOUND + FIXED — the bug was never in symmetric_eig

The probe the fix round never ran: the dumped sigma+moment through the FULL
`split` path (the fix round only ever solved SIGMA standalone —
`build_zca_whitener`'s QL — which is benign on these bytes). The full path
panicked instantly:

```
symmetric_eig: non-finite input at (i, j) = (58, 0) = -inf
```

— the T2 input scan catching a matrix POISONED UPSTREAM inside basis_split
itself. Two defects, both invisible to the module's entire test surface
(which only ever used DIAGONAL sigmas, where they vanish):

1. **`build_zca_whitener` returned the wrong matrix.** It scaled eigvecs
   ROW j by λ_j^{-1/2} and multiplied the scaled buffer against itself —
   Zᵀ·diag(λ⁻¹)·Z in the WRONG basis (symmetric_eig's eigenvectors are
   COLUMNS: `eigvecs[i*n+j] = V[i,j]`, column k paired with `eigvals[k]`).
   On diagonal sigmas Z≈I, rows≈cols, and the result ≈ Σ⁻¹ — already
   wrong vs the documented ZCA Σ^{-1/2} (the `Whitening::Whitener`
   contract `W Σ Wᵀ = I` fails); on the live ROTATED style-shard Σ it
   whitened nothing.
2. **The whitened-moment product aliased.** `M' = W·(M·Wᵀ)` wrote M'
   into `moment_scratch` while still reading it as T: row i read rows
   0..i-1 already overwritten with output, compounding W
   factor-on-factor down the rows. With defect (1)'s magnitudes the
   compounded values overflowed f64 by row ~41 → the second QL call
   received ±inf → BOTH historical signatures: the original pre-guard
   NaN QL spin at l=0, and the post-T2 `non-finite input` refusal at
   (58,0).

This explains every prior mystery: the fix-round "non-reproducible from
the dump" (they solved sigma alone — benign); round-2's "same signature
post-fix" (the panic text changed to the refusal — same failure site);
the dump was never stale — the premise was fine, the tested FUNCTION was
wrong.

Fixes (`basis_split.rs`):
- `build_zca_whitener`: `W = V Λ^{-1/2} Vᵀ` accumulated per eigenvector
  COLUMN (one sqrt + one scale per j); the eigvec buffer is no longer
  mutated (stays the caller's diagnostic).
- `split`: the M' product now runs across three DISTINCT buffers (T lands
  in `eigvecs` — free scratch between the whitener build and the
  eigensolve that overwrites it).
- test hygiene: the g4 alloc canary serialized — its two pipeline tests
  share ONE global allocator counter and libtest runs them in parallel;
  d16 was counting d64's SETUP allocations (250/1 depending on
  scheduling — latent since T2.4, surfaced by the timing shift).
- mechanical: 2 unused_mut + 1 identity_op in the 931 T1/T2 tests
  (the full gate's -D list would have red).

Regression tests (all two-sided verified RED on the pre-fix code):
- `zca_whitener_is_sigma_inv_half_not_inv`: `W Σ Wᵀ = I` on a rotated
  graded (cond ~1e6) sigma — pre-fix error 6.8e9.
- `total_covariance_whitened_moment_matches_w_m_wt`: post-split
  `moment_scratch` == naive `W M Wᵀ` — pre-fix divergence 3.1e-2 (the
  aliasing is silent at benign scales — it corrupts without the
  overflow).
- `issue931_graded_style_shard_split_end_to_end`: the synthetic
  style-shard class at d=64 — pre-fix panics `non-finite input at
  (50,0)=inf`; post-fix finite verdict.

GOAT re-run post-fix: **G1** planted recovery PASS (d16/64/256);
**G2** split latency 2.081 ms (d=64/N=2048; quiet box, AC, 6% CPU — vs
1.923 ms recorded under the bug; the corrected whitener costs ~8%, gate
<10 ms); **G4** alloc PASS ×5 runs (post-serialization); adversarial gates
PASS. Full lib: 2166 (basis_split) / 2157 (default) passed; clippy clean
in both postures.

The dump bytes now SPLIT cleanly end-to-end:
`SufficientVariation { k: 63, gap_ratio: 18.03 }` — the style arm's
substrate blockage is lifted pending the ai-side re-run.

Tasks update:
- [x] T3 (katgpt half): the reproducer class lands as regression tests;
      the fix is the substrate repair the arm waited on.
- [ ] T3 (ai half): re-run the basis_purity_audit at this fix. ⚠ THE
      AFFECT-NUMBERS WARNING: the affect arm's recorded numbers (k=5, gap
      337.81, the A1 FEAR_DIR ~29.5% identity-mass finding) were computed
      UNDER both defects (the aliased M' corrupts at ANY d — silently at
      d=8) — the audit's findings must be re-read before they stand.
