# Latent vs raw space rules (critical for game AI)

- **Physical** (position, velocity, HP, wallet): MUST be raw exact. Deterministic replay, quorum sync, anti-cheat require bit-identical reconstruction.
- **Semantic** (emotion, mood, curiosity, style, habit): SHOULD be latent via dot-product + sigmoid onto learned direction vectors.
- **Social** (encounters, relationships, factions): SHOULD produce KG triples from latent/embedding proximity, not raw coordinate distance.

**Sync boundary:** if data flows through `SyncBlock → ChainConsensus` quorum → Cold tier, it MUST be raw + deterministic. If consumed locally (emotion projection, shard retrieval, consolidation sleep-cycle), it SHOULD be latent. Bridge functions (raw→latent projection, latent→raw scalar clamp) MUST be zero-alloc, gateable, sync-invariant.

**KG triple emission:** semantic encounters → KG triple from latent similarity. Physical events → TxDelta with raw values, NOT KG triple. Never substitute latent embedding for raw position in anti-cheat.

**Spatial cognition (two-brain model):** info brain = real `MapPos` (synced, ground truth). Think brain = per-NPC `SpatialBelief` (zone KG triple + stale last_known_pos, fog-of-war gated, NOT synced). Bridge is one-way: real position → belief update only when within `visible_radius`. Confidence decay: `sigmoid(-λ * (current_tick - last_observed_tick))`. Two brains MUST exist independently — divergence is emergent, not a bug.

Moved verbatim out of `SKILL.md` on 2026-10-28 to keep that file under the skill-size gate. The global agent rules (`~/.claude/CLAUDE.md` §"Latent vs Raw Space Rules") are the canonical, fuller copy.
