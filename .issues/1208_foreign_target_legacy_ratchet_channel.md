**Status:** open — the per-repo legacy_ratchet channel for a foreign `numbering_gate` target; advisory note today, never a failure. Measured: riir-refine walks 9 legacy collisions against the SHARED `number_collisions_expected.txt` `legacy_ratchet = 64` (katgpt-rs's own pre-gate archive value) on every gate run, printing a permanent `re-pin DOWN in this commit` note the target repo cannot act on.

## 2026-11-13 — Issue 1208: foreign-target legacy_ratchet is the shared file's value — the re-pin-DOWN remedy is unexpressable for every non-katgpt-rs repo

Found live in riir-refine (session zed-amo-b352-lone, epoch 1794518415, during the issue-231 Layer-9 gate landing): `py ../katgpt-rs/scripts/numbering_gate.py .` prints

```
⚠ legacy collisions 9 < ratchet 64 — re-pin DOWN in this commit
```

every run. The 9 is real (measured). The 64 is not riir-refine's — it is the SHARED `scripts/number_collisions_expected.txt` `legacy_ratchet = 64` (katgpt-rs's own pre-gate `.issues/121`-archive count), because:

1. For a foreign target, `collision_scalars` (`numbering_gate.py` ~763-835) overrides ONLY `max_hand_read` ← row `max_hand` and `max_rename_residue` ← row `max_rename` (the ONE-RECORD-TWO-MEDIA block, Issue 1206). `legacy_ratchet` is never taken from the drift-floors ROW (the row has no legacy column) nor from a per-repo `number_collisions_<repo>.txt` scalar.
2. riir-refine has no `number_collisions_riir-refine.txt`; even if it had one, its scalars are consulted only for the two caps above.
3. Lowering the SHARED `legacy_ratchet = 64` would break katgpt-rs's own ratchet (its 64 legacy collisions are the adjudicated backlog — Issue 785 forbids ratcheting an unread bucket down to a lie).

So the note's remedy — "re-pin DOWN in this commit" — is UNEXPRESSABLE for every foreign repo with a legacy count below 64: the channel does not exist, and the shared value must not move. The note is permanent advisory noise in exactly the shape the workspace's own rules warn about (a gate that teaches a remedy the target cannot perform).

**Proposed shape (one mechanism, family-wide — the rule-in-one-instrument law):**

- `parse_collision_pins` already reads `legacy_ratchet` from a pins file; the per-repo file's scalars block is the natural home. Extend the foreign-target branch of `collision_scalars` to ALSO override `legacy_ratchet` from `number_collisions_<repo>.txt` when present (row has no column to equality-check, so the per-repo file is authoritative alone; the sweep's sync check needs no new axis — it does not currently sync this scalar).
- riir-refine then lands `number_collisions_riir-refine.txt` with `legacy_ratchet = 9` (measured 2026-11-13 at develop 3c6359ea; 9 historical collisions over 1012 numbers, 0 above the boundary, families 0 collision / 0 rename-residue / 2 hand-read) — and the note disappears until the count actually moves (growth still reds via the same ratchet).
- Gate self-test arm: a per-repo file carrying `legacy_ratchet` overrides the shared value (and a STALE per-repo value that the measured count exceeds still fails upward).
- Audit every other contract repo's gate output for the same permanent note before/after (the fix is family-wide, not riir-refine-scoped).

**Adjudication note:** the note is ADVISORY (never exit-1) — this issue is an instrument-honesty work item, not a red-gate repair. No security or correctness exposure; the ratchet's GROWTH direction already works for foreign repos (9 > 64 is impossible, but a repo whose shared ratchet undercounts would miss real growth — e.g. any repo starting clean at 0 legacy that accumulates its first re-spend would compare against 64 and stay green forever — that direction IS a live blind spot, and is the load-bearing reason this channel should exist: today a foreign repo's legacy-collision growth is UNWATCHED below 64).

Claimed by: session zed-amo-b352-lone (epoch 1794518415) as FILED-ONLY — implementation is a katgpt-rs instrument change with its own self-test battery, open for the numbering lane.
