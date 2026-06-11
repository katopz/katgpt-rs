//! Bomberman HL Arena — ECS game systems (bevy_ecs 0.15, World-based).
//!
//! All systems operate on `&mut World` for tick-based game loop control.
//! Called in deterministic order by [`run_tick`]; no ECS schedule is used.

use std::collections::{HashMap, HashSet};

use bevy_ecs::prelude::*;

use super::arena::ArenaGrid;
use super::{
    Alive, BOMB_FUSE_TICKS, Blast, Bomb, BombCount, BombFuse, BombRange, BombType, BomberAction,
    Cell, DEFAULT_BLAST_RANGE, DEFAULT_MAX_BOMBS, DEFAULT_SPEED, GameEvent, GameRng, GridPos,
    Player, PlayerEntities, PowerUp, PowerUpKind, SPAWN_POSITIONS, ScoreBoard, Speed, TICK_LIMIT,
    TickCounter,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// A bomb whose fuse expired and is ready to explode.
pub struct PendingExplosion {
    pub pos: (i32, i32),
    pub range: u32,
    /// Player entity that placed the bomb (for active-count tracking).
    pub owner: Entity,
    /// Bomb type — determines blast behavior (e.g. piercing continues through walls).
    pub bomb_type: BombType,
}

/// Four cardinal directions used for blast propagation.
const DIRECTIONS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

// ---------------------------------------------------------------------------
// World initialisation
// ---------------------------------------------------------------------------

/// Create a fresh `World` with all resources required for one bomberman round.
pub fn init_world(seed: u64) -> World {
    let mut world = World::new();
    world.insert_resource(ArenaGrid::generate(seed));
    world.insert_resource(GameRng { seed });
    world.insert_resource(TickCounter::default());
    world.insert_resource(ScoreBoard::default());
    world.init_resource::<Events<GameEvent>>();
    world
}

/// Create a fresh `World` with a pre-built [`ArenaGrid`] (e.g. from a fixed template).
pub fn init_world_with_arena(arena: ArenaGrid) -> World {
    let mut world = World::new();
    world.insert_resource(arena);
    world.insert_resource(GameRng { seed: 0 });
    world.insert_resource(TickCounter::default());
    world.insert_resource(ScoreBoard::default());
    world.init_resource::<Events<GameEvent>>();
    world
}

/// Spawn 4 player entities at the corner spawn positions.
///
/// Returns the 4 player [`Entity`] ids and inserts a [`PlayerEntities`] resource.
pub fn spawn_players(world: &mut World) -> [Entity; 4] {
    let entities: Vec<Entity> = SPAWN_POSITIONS
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| {
            world
                .spawn((
                    Player { id: i as u8 },
                    GridPos { x, y },
                    BombCount {
                        max: DEFAULT_MAX_BOMBS,
                        active: 0,
                    },
                    BombRange {
                        cells: DEFAULT_BLAST_RANGE,
                    },
                    Speed {
                        cells_per_tick: DEFAULT_SPEED,
                    },
                    Alive,
                ))
                .id()
        })
        .collect();

    let [a, b, c, d] = entities.try_into().expect("exactly 4 spawn positions");
    world.insert_resource(PlayerEntities {
        entities: [a, b, c, d],
    });
    [a, b, c, d]
}

// ---------------------------------------------------------------------------
// Tick orchestration
// ---------------------------------------------------------------------------

/// Run one game tick. Returns `true` while the round continues, `false` when ended.
///
/// Processing order:
/// 1. Fuse countdown (skips Remote/Landmine bombs)
/// 2. Remote detonation (player-triggered)
/// 3. Process explosions (timed + remote)
/// 4. Apply movement
/// 5. Landmine trigger (proximity-based, after movement)
/// 6. Process landmine explosions
/// 7. Bomb placement
/// 8. Power-up collection
/// 9. Cleanup + round-end check
pub fn run_tick(world: &mut World, actions: [Option<BomberAction>; 4]) -> bool {
    // 1–2. Fuse countdown + remote detonation
    let mut pending = tick_bomb_fuses(world);
    pending.extend(detonate_remote_bombs(world, actions));

    // 3. Process all explosions (timed + remote)
    let mut blast_cells = process_explosions(world, pending);

    // 4. Movement
    apply_movement(world, actions);

    // 5–6. Landmine trigger (after movement — stepping on one triggers it)
    let landmine_explosions = trigger_landmines(world);
    if !landmine_explosions.is_empty() {
        blast_cells.extend(process_explosions(world, landmine_explosions));
    }

    // 7–9. Placement, collection, cleanup
    place_bombs(world, actions);
    collect_powerups(world);
    cleanup_and_check(world, blast_cells)
}

// ---------------------------------------------------------------------------
// 1. Bomb fuse countdown
// ---------------------------------------------------------------------------

fn tick_bomb_fuses(world: &mut World) -> Vec<PendingExplosion> {
    #[allow(clippy::type_complexity)]
    let mut to_explode: Vec<(Entity, (i32, i32), u32, Entity, BombType)> = Vec::new();

    {
        let mut q = world.query::<(Entity, &Bomb, &mut BombFuse, &GridPos, &BombRange)>();
        for (entity, bomb, mut fuse, pos, range) in q.iter_mut(world) {
            // Remote and Landmine bombs use trigger-based detonation, not fuse countdown
            let bomb_type = bomb.bomb_type;
            match bomb_type {
                BombType::Remote | BombType::Landmine => continue,
                BombType::Timed | BombType::Piercing => {}
            }
            let owner = fuse.owner;
            fuse.ticks_remaining = fuse.ticks_remaining.saturating_sub(1);
            if fuse.ticks_remaining == 0 {
                to_explode.push((entity, (pos.x, pos.y), range.cells, owner, bomb_type));
            }
        }
    }

    let mut result = Vec::with_capacity(to_explode.len());
    for (entity, pos, range, owner, bomb_type) in to_explode {
        world.entity_mut(entity).despawn();
        world.send_event(GameEvent::BombExploded { pos, range });
        result.push(PendingExplosion {
            pos,
            range,
            owner,
            bomb_type,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// 2. Remote detonation
// ---------------------------------------------------------------------------

/// Process `Detonate` actions: all `Remote` bombs owned by the detonating player
/// immediately explode.
fn detonate_remote_bombs(
    world: &mut World,
    actions: [Option<BomberAction>; 4],
) -> Vec<PendingExplosion> {
    // Collect player entities that issued Detonate action
    let detonating_owners: Vec<Entity> = {
        let Some(pe) = world.get_resource::<PlayerEntities>() else {
            return Vec::new();
        };
        actions
            .iter()
            .enumerate()
            .filter_map(|(i, action)| match action {
                Some(BomberAction::Detonate) => Some(pe.entities[i]),
                _ => None,
            })
            .collect()
    };

    if detonating_owners.is_empty() {
        return Vec::new();
    }

    // Find all Remote bombs owned by detonating players
    let mut to_explode: Vec<(Entity, (i32, i32), u32, Entity)> = Vec::new();
    {
        let mut q = world.query::<(Entity, &Bomb, &BombFuse, &BombRange, &GridPos)>();
        for (entity, bomb, fuse, range, pos) in q.iter(world) {
            if bomb.bomb_type != BombType::Remote {
                continue;
            }
            if !detonating_owners.contains(&fuse.owner) {
                continue;
            }
            to_explode.push((entity, (pos.x, pos.y), range.cells, fuse.owner));
        }
    }

    let mut result = Vec::with_capacity(to_explode.len());
    for (entity, pos, range, owner) in to_explode {
        world.entity_mut(entity).despawn();
        world.send_event(GameEvent::BombExploded { pos, range });
        result.push(PendingExplosion {
            pos,
            range,
            owner,
            bomb_type: BombType::Remote,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// 3. Blast propagation
// ---------------------------------------------------------------------------

fn process_explosions(world: &mut World, queue: Vec<PendingExplosion>) -> Vec<(i32, i32)> {
    if queue.is_empty() {
        return Vec::new();
    }

    // ── Snapshot current state (immutable reads) ────────────────────
    let bomb_map: HashMap<(i32, i32), (Entity, u32, Entity, BombType)> = {
        let mut q =
            world.query_filtered::<(Entity, &GridPos, &BombRange, &BombFuse, &Bomb), With<Bomb>>();
        q.iter(world)
            .map(|(e, p, r, f, b)| ((p.x, p.y), (e, r.cells, f.owner, b.bomb_type)))
            .collect()
    };

    // Vec to handle multiple players at the same position
    let player_map: HashMap<(i32, i32), Vec<(u8, Entity)>> = {
        let mut q = world.query_filtered::<(Entity, &Player, &GridPos), With<Alive>>();
        let mut map: HashMap<(i32, i32), Vec<(u8, Entity)>> = HashMap::new();
        for (e, p, pos) in q.iter(world) {
            map.entry((pos.x, pos.y)).or_default().push((p.id, e));
        }
        map
    };

    // Map player entity → player id (for killer tracking)
    let player_id_map: HashMap<Entity, u8> = player_map
        .values()
        .flat_map(|v| v.iter())
        .copied()
        .map(|(id, e)| (e, id))
        .collect();

    // ── Compute blast propagation (no world mutation) ───────────────
    let mut blast_cells: Vec<(i32, i32)> = Vec::new();
    let mut walls_destroyed: HashSet<(i32, i32)> = HashSet::new();
    let mut powerups_revealed: Vec<(PowerUpKind, (i32, i32))> = Vec::new();
    let mut players_killed: Vec<(u8, Entity, Option<u8>)> = Vec::new();
    let mut bombs_to_despawn: Vec<Entity> = Vec::new();
    let mut owners_to_decrement: HashSet<Entity> = HashSet::new();
    let mut processed: HashSet<(i32, i32)> = HashSet::new();

    let mut explosion_queue: Vec<PendingExplosion> = queue;

    while let Some(exp) = explosion_queue.pop() {
        let killer_id = player_id_map.get(&exp.owner).copied();
        processed.insert(exp.pos);
        blast_cells.push(exp.pos);
        owners_to_decrement.insert(exp.owner);

        if let Some(players) = player_map.get(&exp.pos) {
            for &(pid, pe) in players {
                players_killed.push((pid, pe, killer_id));
            }
        }

        for (dx, dy) in DIRECTIONS {
            for dist in 1..=exp.range as i32 {
                let cx = exp.pos.0 + dx * dist;
                let cy = exp.pos.1 + dy * dist;
                let cell = world.resource::<ArenaGrid>().get(cx, cy);

                match cell {
                    Cell::FixedWall => break,
                    Cell::DestructibleWall => {
                        blast_cells.push((cx, cy));
                        walls_destroyed.insert((cx, cy));
                        if let Some(players) = player_map.get(&(cx, cy)) {
                            for &(pid, pe) in players {
                                players_killed.push((pid, pe, killer_id));
                            }
                        }
                        match exp.bomb_type {
                            BombType::Piercing => {} // continue through wall
                            _ => break,
                        }
                    }
                    Cell::PowerUpHidden(kind) => {
                        blast_cells.push((cx, cy));
                        walls_destroyed.insert((cx, cy));
                        powerups_revealed.push((kind, (cx, cy)));
                        if let Some(players) = player_map.get(&(cx, cy)) {
                            for &(pid, pe) in players {
                                players_killed.push((pid, pe, killer_id));
                            }
                        }
                        break;
                    }
                    Cell::Floor => {
                        blast_cells.push((cx, cy));
                        if let Some(players) = player_map.get(&(cx, cy)) {
                            for &(pid, pe) in players {
                                players_killed.push((pid, pe, killer_id));
                            }
                        }
                        if processed.insert((cx, cy))
                            && let Some(&(be, br, bo, bt)) = bomb_map.get(&(cx, cy))
                        {
                            bombs_to_despawn.push(be);
                            owners_to_decrement.insert(bo);
                            explosion_queue.push(PendingExplosion {
                                pos: (cx, cy),
                                range: br,
                                owner: bo,
                                bomb_type: bt,
                            });
                        }
                    }
                }
            }
        }
    }

    // ── Apply mutations ─────────────────────────────────────────────
    {
        let mut grid = world.resource_mut::<ArenaGrid>();
        for &(x, y) in &walls_destroyed {
            grid.set(x, y, Cell::Floor);
        }
    }
    for &(x, y) in &walls_destroyed {
        world.send_event(GameEvent::WallDestroyed { pos: (x, y) });
    }

    for (kind, (x, y)) in powerups_revealed {
        world.spawn((PowerUp { kind }, GridPos { x, y }));
        world.send_event(GameEvent::PowerUpRevealed { pos: (x, y), kind });
    }

    for be in bombs_to_despawn {
        world.entity_mut(be).despawn();
    }

    let mut killed_entities: HashSet<Entity> = HashSet::new();
    for (pid, pe, killer) in players_killed {
        if killed_entities.insert(pe) && world.get::<Alive>(pe).is_some() {
            world.entity_mut(pe).remove::<Alive>();
            world.send_event(GameEvent::PlayerKilled {
                victim: pid,
                killer,
            });

            // Update ScoreBoard resource for real-time TUI display
            let mut sb = world.resource_mut::<ScoreBoard>();
            sb.add(pid, -3);
            match killer {
                Some(k) if k != pid => sb.add(k, 3),
                _ => sb.add(pid, -2), // Suicide penalty
            }
        }
    }

    for owner in owners_to_decrement {
        if let Some(mut c) = world.get_mut::<BombCount>(owner) {
            c.active = c.active.saturating_sub(1);
        }
    }

    blast_cells
}

// ---------------------------------------------------------------------------
// 4. Movement
// ---------------------------------------------------------------------------

fn apply_movement(world: &mut World, actions: [Option<BomberAction>; 4]) {
    // Landmines do NOT block movement — players can walk onto them
    let bomb_pos: HashSet<(i32, i32)> = {
        let mut q = world.query_filtered::<(&GridPos, &Bomb), ()>();
        q.iter(world)
            .filter(|(_, bomb)| bomb.bomb_type != BombType::Landmine)
            .map(|(p, _)| (p.x, p.y))
            .collect()
    };

    // Collect phase — immutable query
    #[allow(clippy::type_complexity)]
    let mut moves: Vec<(u8, Entity, (i32, i32), (i32, i32))> = Vec::new();
    {
        let mut q = world.query_filtered::<(Entity, &Player, &GridPos), With<Alive>>();
        for (entity, player, pos) in q.iter(world) {
            let action = match actions.get(player.id as usize).copied().flatten() {
                Some(a) => a,
                None => continue,
            };
            let (dx, dy) = match action {
                BomberAction::Up => (0, -1),
                BomberAction::Down => (0, 1),
                BomberAction::Left => (-1, 0),
                BomberAction::Right => (1, 0),
                BomberAction::Bomb | BomberAction::Wait | BomberAction::Detonate => continue,
            };
            let tx = pos.x + dx;
            let ty = pos.y + dy;

            if !world.resource::<ArenaGrid>().is_walkable(tx, ty) {
                continue;
            }
            if bomb_pos.contains(&(tx, ty)) {
                continue;
            }
            moves.push((player.id, entity, (pos.x, pos.y), (tx, ty)));
        }
    }

    // Apply phase — write new positions
    for (pid, entity, from, to) in moves {
        if let Some(mut pos) = world.get_mut::<GridPos>(entity) {
            pos.x = to.0;
            pos.y = to.1;
        }
        world.send_event(GameEvent::PlayerMoved {
            player: pid,
            from,
            to,
        });
    }
}

// ---------------------------------------------------------------------------
// 5. Landmine trigger
// ---------------------------------------------------------------------------

/// After movement, check if any alive player is standing on a `Landmine` bomb.
/// Triggered landmines explode with range 1 regardless of their `BombRange` component.
fn trigger_landmines(world: &mut World) -> Vec<PendingExplosion> {
    // Collect alive player positions into fixed-size array (max 4 players)
    let mut player_positions = [None; 4];
    let mut player_count = 0usize;
    {
        let mut q = world.query_filtered::<&GridPos, With<Alive>>();
        for p in q.iter(world) {
            if player_count < 4 {
                player_positions[player_count] = Some((p.x, p.y));
                player_count += 1;
            }
        }
    }

    if player_count == 0 {
        return Vec::new();
    }

    // Find landmines at player positions
    let mut to_explode: Vec<(Entity, (i32, i32), Entity)> = Vec::new();
    {
        let mut q = world.query::<(Entity, &Bomb, &BombFuse, &GridPos)>();
        for (entity, bomb, fuse, pos) in q.iter(world) {
            if bomb.bomb_type != BombType::Landmine {
                continue;
            }
            let pos_tuple = (pos.x, pos.y);
            for i in 0..player_count {
                if player_positions[i] == Some(pos_tuple) {
                    to_explode.push((entity, pos_tuple, fuse.owner));
                    break;
                }
            }
        }
    }

    let mut result = Vec::with_capacity(to_explode.len());
    for (entity, pos, owner) in to_explode {
        world.entity_mut(entity).despawn();
        // Landmine always has range 1 regardless of BombRange
        let range = 1;
        world.send_event(GameEvent::BombExploded { pos, range });
        result.push(PendingExplosion {
            pos,
            range,
            owner,
            bomb_type: BombType::Landmine,
        });
    }
    result
}

// ---------------------------------------------------------------------------
// 6. Bomb placement
// ---------------------------------------------------------------------------

fn place_bombs(world: &mut World, actions: [Option<BomberAction>; 4]) {
    let bomb_pos: HashSet<(i32, i32)> = {
        let mut q = world.query_filtered::<&GridPos, With<Bomb>>();
        q.iter(world).map(|p| (p.x, p.y)).collect()
    };

    let mut to_place: Vec<(Entity, (i32, i32), u32)> = Vec::new();
    {
        let mut q = world
            .query_filtered::<(Entity, &Player, &GridPos, &BombCount, &BombRange), With<Alive>>();
        for (entity, player, pos, count, range) in q.iter(world) {
            match actions.get(player.id as usize).copied().flatten() {
                Some(BomberAction::Bomb) => {}
                _ => continue,
            }
            if count.active >= count.max {
                continue;
            }
            if bomb_pos.contains(&(pos.x, pos.y)) {
                continue;
            }
            to_place.push((entity, (pos.x, pos.y), range.cells));
        }
    }

    for (owner, (x, y), range) in to_place {
        world.spawn((
            Bomb::new(),
            GridPos { x, y },
            BombFuse {
                owner,
                ticks_remaining: BOMB_FUSE_TICKS,
            },
            BombRange { cells: range },
        ));
        if let Some(mut c) = world.get_mut::<BombCount>(owner) {
            c.active += 1;
        }
        let pid = world.get::<Player>(owner).map(|p| p.id).unwrap_or(0);
        world.send_event(GameEvent::BombPlaced {
            player: pid,
            pos: (x, y),
        });
    }
}

// ---------------------------------------------------------------------------
// 7. Power-up collection
// ---------------------------------------------------------------------------

fn collect_powerups(world: &mut World) {
    let pu_map: HashMap<(i32, i32), (Entity, PowerUpKind)> = {
        let mut q = world.query_filtered::<(Entity, &PowerUp, &GridPos), ()>();
        q.iter(world)
            .map(|(e, pu, pos)| ((pos.x, pos.y), (e, pu.kind)))
            .collect()
    };

    if pu_map.is_empty() {
        return;
    }

    let mut to_collect: Vec<(Entity, Entity, PowerUpKind)> = Vec::new();
    {
        let mut pq = world.query_filtered::<(Entity, &Player, &GridPos), With<Alive>>();
        for (pe, _player, pos) in pq.iter(world) {
            if let Some(&(pue, kind)) = pu_map.get(&(pos.x, pos.y)) {
                to_collect.push((pe, pue, kind));
            }
        }
    }

    // Track already-despawned entities to prevent double-despawn when
    // two players land on the same power-up cell in the same tick.
    let mut collected_entities: HashSet<bevy_ecs::entity::Entity> = HashSet::new();

    for (player_entity, pu_entity, kind) in to_collect {
        // Apply power-up effect to player
        match kind {
            PowerUpKind::BombUp => {
                if let Some(mut c) = world.get_mut::<BombCount>(player_entity) {
                    c.max += 1;
                }
            }
            PowerUpKind::FireUp => {
                if let Some(mut r) = world.get_mut::<BombRange>(player_entity) {
                    r.cells += 1;
                }
            }
            PowerUpKind::SpeedUp => {
                if let Some(mut s) = world.get_mut::<Speed>(player_entity) {
                    s.cells_per_tick = (s.cells_per_tick + 1).min(2);
                }
            }
        }
        let pid = world
            .get::<Player>(player_entity)
            .map(|p| p.id)
            .unwrap_or(0);

        // Only first player to reach this entity emits event and despawns
        if collected_entities.insert(pu_entity) {
            let pu_pos = world
                .get::<GridPos>(pu_entity)
                .map(|g| (g.x, g.y))
                .unwrap_or((0, 0));
            world.send_event(GameEvent::PowerUpCollected {
                player: pid,
                kind,
                pos: pu_pos,
            });

            // Update ScoreBoard resource for real-time TUI display
            world.resource_mut::<ScoreBoard>().add(pid, 1);

            world.entity_mut(pu_entity).despawn();
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Cleanup + round-end check
// ---------------------------------------------------------------------------

fn cleanup_and_check(world: &mut World, blast_cells: Vec<(i32, i32)>) -> bool {
    // Remove previous tick's blast visuals
    {
        let old: Vec<Entity> = {
            let mut q = world.query_filtered::<Entity, With<Blast>>();
            q.iter(world).collect()
        };
        for e in old {
            world.entity_mut(e).despawn();
        }
    }

    // Spawn blast visuals for this tick
    for (x, y) in blast_cells {
        world.spawn((Blast, GridPos { x, y }));
    }

    // Advance tick counter
    world.resource_mut::<TickCounter>().tick += 1;

    // Count survivors
    let alive: Vec<u8> = {
        let mut q = world.query_filtered::<(Entity, &Player), With<Alive>>();
        q.iter(world).map(|(_, p)| p.id).collect()
    };

    let tick = world.resource::<TickCounter>().tick;
    let round_over = alive.len() <= 1 || tick >= TICK_LIMIT;

    if round_over {
        world.send_event(GameEvent::RoundEnd { survivors: alive });
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_world_creates_required_resources() {
        let world = init_world(42);
        assert!(world.contains_resource::<ArenaGrid>());
        assert!(world.contains_resource::<TickCounter>());
        assert!(world.contains_resource::<ScoreBoard>());
        assert!(world.contains_resource::<Events<GameEvent>>());
    }

    #[test]
    fn spawn_players_creates_four_entities() {
        let mut world = init_world(42);
        let entities = spawn_players(&mut world);
        assert_eq!(entities.len(), 4);

        // Verify PlayerEntities resource was inserted
        let pe = world.resource::<PlayerEntities>();
        assert_eq!(pe.entities, entities);

        // Verify each player has correct id and spawn position
        for (i, &entity) in entities.iter().enumerate() {
            let player = world.get::<Player>(entity).unwrap();
            assert_eq!(player.id, i as u8);
            let pos = world.get::<GridPos>(entity).unwrap();
            assert_eq!((pos.x, pos.y), SPAWN_POSITIONS[i]);
        }
    }

    #[test]
    fn run_tick_advances_counter() {
        let mut world = init_world(42);
        spawn_players(&mut world);
        let before = world.resource::<TickCounter>().tick;
        let ongoing = run_tick(&mut world, [None; 4]);
        assert!(ongoing, "round should continue with no bombs");
        assert_eq!(world.resource::<TickCounter>().tick, before + 1);
    }

    #[test]
    fn bomb_explodes_after_fuse_ticks() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place a bomb at player 0 position
        let pos = *world.get::<GridPos>(p0).unwrap();
        world.spawn((
            Bomb::new(),
            GridPos { x: pos.x, y: pos.y },
            BombFuse {
                owner: p0,
                ticks_remaining: BOMB_FUSE_TICKS,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        // Track the owner's active count
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Run ticks until bomb explodes
        for _ in 0..BOMB_FUSE_TICKS {
            let _ = run_tick(&mut world, [None; 4]);
        }

        // Bomb should be gone, active count decremented
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);

        // No bomb entities left
        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(bomb_count, 0);
    }

    // ── A6: Remote detonation tests ─────────────────────────────────

    #[test]
    fn remote_bomb_detonates_on_action() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place a remote bomb at a position away from player 0
        let pos = *world.get::<GridPos>(p0).unwrap();
        let bomb_pos = (pos.x + 1, pos.y);
        world.spawn((
            Bomb::with_type(BombType::Remote),
            GridPos {
                x: bomb_pos.0,
                y: bomb_pos.1,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Issue Detonate action for player 0
        let _ = run_tick(&mut world, [Some(BomberAction::Detonate), None, None, None]);

        // Bomb should be gone, active count decremented
        assert_eq!(
            world.get_mut::<BombCount>(p0).unwrap().active,
            0,
            "remote bomb should be despawned after detonate"
        );
        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(bomb_count, 0, "no bomb entities should remain");
    }

    #[test]
    fn remote_bomb_ignores_fuse_countdown() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place a remote bomb with fuse = 1 (would normally explode next tick)
        let pos = *world.get::<GridPos>(p0).unwrap();
        let bomb_pos = (pos.x + 1, pos.y);
        world.spawn((
            Bomb::with_type(BombType::Remote),
            GridPos {
                x: bomb_pos.0,
                y: bomb_pos.1,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: 1,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Run a tick without Detonate action
        let _ = run_tick(&mut world, [None; 4]);

        // Remote bomb should still exist (fuse was not decremented)
        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(
            bomb_count, 1,
            "remote bomb should not explode via fuse countdown"
        );
        assert_eq!(
            world.get_mut::<BombCount>(p0).unwrap().active,
            1,
            "active count should remain unchanged"
        );
    }

    #[test]
    fn remote_detonate_only_affects_owned_bombs() {
        let mut world = init_world(42);
        let [p0, p1, ..] = spawn_players(&mut world);

        // Place a remote bomb for player 0
        let pos0 = *world.get::<GridPos>(p0).unwrap();
        world.spawn((
            Bomb::with_type(BombType::Remote),
            GridPos {
                x: pos0.x + 1,
                y: pos0.y,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Place a remote bomb for player 1
        let pos1 = *world.get::<GridPos>(p1).unwrap();
        world.spawn((
            Bomb::with_type(BombType::Remote),
            GridPos {
                x: pos1.x + 1,
                y: pos1.y,
            },
            BombFuse {
                owner: p1,
                ticks_remaining: u32::MAX,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        world.get_mut::<BombCount>(p1).unwrap().active = 1;

        // Player 0 detonates — only their bomb should explode
        let _ = run_tick(&mut world, [Some(BomberAction::Detonate), None, None, None]);

        // Player 0's bomb gone, player 1's bomb still present
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);
        assert_eq!(world.get_mut::<BombCount>(p1).unwrap().active, 1);

        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(bomb_count, 1, "only player 1's remote bomb should remain");
    }

    #[test]
    fn remote_detonate_with_no_remote_bombs_is_noop() {
        let mut world = init_world(42);
        spawn_players(&mut world);

        // No bombs placed — Detonate should be a no-op
        let ongoing = run_tick(&mut world, [Some(BomberAction::Detonate), None, None, None]);
        assert!(ongoing, "round should continue");
    }

    // ── A7: Landmine trigger tests ──────────────────────────────────

    #[test]
    fn landmine_triggers_on_step() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Player 0 is at (1,1). Place landmine at (2,1) — within safe zone, should be floor.
        let landmine_pos = (2, 1);
        world.spawn((
            Bomb::with_type(BombType::Landmine),
            GridPos {
                x: landmine_pos.0,
                y: landmine_pos.1,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Move player 0 right onto the landmine
        let _ = run_tick(&mut world, [Some(BomberAction::Right), None, None, None]);

        // Landmine should have exploded
        assert_eq!(
            world.get_mut::<BombCount>(p0).unwrap().active,
            0,
            "landmine should be despawned after trigger"
        );
        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(bomb_count, 0, "no bomb entities should remain");
    }

    #[test]
    fn landmine_ignores_fuse_countdown() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place a landmine with fuse = 1 (would normally explode next tick)
        // Use position away from player so it doesn't trigger by proximity
        let landmine_pos = (2, 1);
        world.spawn((
            Bomb::with_type(BombType::Landmine),
            GridPos {
                x: landmine_pos.0,
                y: landmine_pos.1,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: 1,
            },
            BombRange {
                cells: DEFAULT_BLAST_RANGE,
            },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Run a tick — player doesn't move onto landmine
        let _ = run_tick(&mut world, [Some(BomberAction::Wait), None, None, None]);

        // Landmine should still exist (fuse was not decremented, player not on it)
        let bomb_count = {
            let mut q = world.query_filtered::<Entity, With<Bomb>>();
            q.iter(&world).count()
        };
        assert_eq!(
            bomb_count, 1,
            "landmine should not explode via fuse countdown"
        );
        assert_eq!(
            world.get_mut::<BombCount>(p0).unwrap().active,
            1,
            "active count should remain unchanged"
        );
    }

    #[test]
    fn landmine_always_range_1() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place landmine with high BombRange — should still only blast range 1
        let landmine_pos = (2, 1);
        world.spawn((
            Bomb::with_type(BombType::Landmine),
            GridPos {
                x: landmine_pos.0,
                y: landmine_pos.1,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange { cells: 5 }, // high range, but landmine ignores it
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Move player onto landmine
        let _ = run_tick(&mut world, [Some(BomberAction::Right), None, None, None]);

        // Verify the BombExploded event has range 1
        let events = world.resource::<Events<GameEvent>>();
        let mut cursor = events.get_cursor();
        let mut found_landmine_explosion = false;
        for event in cursor.read(events) {
            match event {
                GameEvent::BombExploded { pos, range } if *pos == landmine_pos => {
                    assert_eq!(*range, 1, "landmine should always have range 1");
                    found_landmine_explosion = true;
                }
                _ => {}
            }
        }
        assert!(
            found_landmine_explosion,
            "should find landmine explosion event"
        );
    }

    #[test]
    fn landmine_friendly_fire() {
        let mut world = init_world(42);
        let [p0, _p1, ..] = spawn_players(&mut world);

        // Player 0 places landmine at (2,1)
        let landmine_pos = (2, 1);
        world.spawn((
            Bomb::with_type(BombType::Landmine),
            GridPos {
                x: landmine_pos.0,
                y: landmine_pos.1,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange { cells: 1 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Player 0 moves away first, then player 1 steps on landmine
        // Tick 1: Player 0 moves right to (2,1)... wait, that's where the landmine is.
        // Actually, let's place the landmine where player 1 can reach it.
        // Player 1 spawns at (11,1). Let me place landmine at (10,1).
        // But first, let me just verify the owner can trigger it.

        // Simpler test: place landmine at player 0's current position (1,1)
        // Player 0 stays → triggers own landmine
        let bomb_entity = world
            .query_filtered::<Entity, With<Bomb>>()
            .iter(&world)
            .next()
            .unwrap();
        world.despawn(bomb_entity);
        world.get_mut::<BombCount>(p0).unwrap().active = 0;

        let player_pos = *world.get::<GridPos>(p0).unwrap();
        world.spawn((
            Bomb::with_type(BombType::Landmine),
            GridPos {
                x: player_pos.x,
                y: player_pos.y,
            },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange { cells: 1 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Player 0 doesn't move — already on landmine → triggers
        let _ = run_tick(&mut world, [Some(BomberAction::Wait), None, None, None]);

        // Landmine should explode, player 0 should die (friendly fire)
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);
        assert!(
            world.get::<Alive>(p0).is_none(),
            "player 0 should be killed by own landmine"
        );
    }

    #[test]
    fn landmine_does_not_block_movement() {
        let mut world = init_world(42);
        let [p0, ..] = spawn_players(&mut world);

        // Place landmine at (2,1) — should NOT block player from moving there
        world.spawn((
            Bomb::with_type(BombType::Landmine),
            GridPos { x: 2, y: 1 },
            BombFuse {
                owner: p0,
                ticks_remaining: u32::MAX,
            },
            BombRange { cells: 1 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Player 0 at (1,1) moves right to (2,1) — landmine should not block
        let _ = run_tick(&mut world, [Some(BomberAction::Right), None, None, None]);

        // Player should have moved to (2,1)
        let pos = world.get::<GridPos>(p0).unwrap();
        assert_eq!((pos.x, pos.y), (2, 1), "player should move onto landmine");

        // Landmine should have triggered
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);
    }

    // ── A5: Piercing blast tests ────────────────────────────────────

    /// Piercing blast should destroy a destructible wall AND continue propagating
    /// through it, hitting cells behind the wall that a Timed bomb would not reach.
    #[test]
    fn piercing_blast_destroys_wall_and_continues() {
        let mut world = init_world(42);
        let [p0, p1, ..] = spawn_players(&mut world);

        // Set up controlled layout:
        //   (1,1) bomb origin (player 0 spawn)
        //   (2,1) destructible wall — should be destroyed
        //   (3,1) floor with player 1 — blast should reach here
        {
            let mut grid = world.resource_mut::<ArenaGrid>();
            grid.set(1, 1, Cell::Floor);
            grid.set(2, 1, Cell::DestructibleWall);
            grid.set(3, 1, Cell::Floor);
        }

        // Move player 1 behind the wall
        *world.get_mut::<GridPos>(p1).unwrap() = GridPos { x: 3, y: 1 };

        // Place a piercing bomb at (1,1) with range 3
        world.spawn((
            Bomb::with_type(BombType::Piercing),
            GridPos { x: 1, y: 1 },
            BombFuse {
                owner: p0,
                ticks_remaining: BOMB_FUSE_TICKS,
            },
            BombRange { cells: 3 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Run ticks until bomb explodes
        for _ in 0..BOMB_FUSE_TICKS {
            let _ = run_tick(&mut world, [None; 4]);
        }

        // Wall destroyed — grid should now be Floor
        assert_eq!(
            world.resource::<ArenaGrid>().get(2, 1),
            Cell::Floor,
            "piercing blast should destroy wall at (2,1)"
        );

        // Blast continued through wall — player 1 behind wall should be dead
        assert!(
            world.get::<Alive>(p1).is_none(),
            "piercing blast should continue through wall and kill player 1 at (3,1)"
        );

        // Bomb cleaned up
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);
    }

    /// Timed (default) blast should NOT continue through a destructible wall.
    #[test]
    fn timed_blast_stops_at_destructible_wall() {
        let mut world = init_world(42);
        let [p0, p1, ..] = spawn_players(&mut world);

        // Same layout: wall at (2,1), player 1 at (3,1)
        {
            let mut grid = world.resource_mut::<ArenaGrid>();
            grid.set(1, 1, Cell::Floor);
            grid.set(2, 1, Cell::DestructibleWall);
            grid.set(3, 1, Cell::Floor);
        }

        *world.get_mut::<GridPos>(p1).unwrap() = GridPos { x: 3, y: 1 };

        // Place a default Timed bomb
        world.spawn((
            Bomb::new(),
            GridPos { x: 1, y: 1 },
            BombFuse {
                owner: p0,
                ticks_remaining: BOMB_FUSE_TICKS,
            },
            BombRange { cells: 3 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        for _ in 0..BOMB_FUSE_TICKS {
            let _ = run_tick(&mut world, [None; 4]);
        }

        // Wall destroyed
        assert_eq!(
            world.resource::<ArenaGrid>().get(2, 1),
            Cell::Floor,
            "timed blast should still destroy wall at (2,1)"
        );

        // Player 1 behind wall should be alive — blast stopped at wall
        assert!(
            world.get::<Alive>(p1).is_some(),
            "timed blast should stop at wall, player 1 at (3,1) should survive"
        );
    }

    // ── A11: Additional bomb type tests ─────────────────────────────

    /// Piercing blast should NOT pass through FixedWall — only DestructibleWall.
    #[test]
    fn piercing_blast_stops_at_fixed_wall() {
        let mut world = init_world(42);
        let [p0, p1, ..] = spawn_players(&mut world);

        // Layout: FixedWall pillar at (2,2), player 1 at (1,2)
        // Piercing bomb at (3,2) going left — blast stops at pillar, player 1 survives
        {
            let mut grid = world.resource_mut::<ArenaGrid>();
            grid.set(1, 2, Cell::Floor);
            grid.set(2, 2, Cell::FixedWall); // pillar
            grid.set(3, 2, Cell::Floor);
        }

        // Move player 1 behind the pillar
        *world.get_mut::<GridPos>(p1).unwrap() = GridPos { x: 1, y: 2 };

        // Place a piercing bomb at (3,2) with range 3
        world.spawn((
            Bomb::with_type(BombType::Piercing),
            GridPos { x: 3, y: 2 },
            BombFuse {
                owner: p0,
                ticks_remaining: 1,
            },
            BombRange { cells: 3 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active = 1;

        // Run one tick — bomb explodes
        let _ = run_tick(&mut world, [None; 4]);

        // FixedWall should remain intact
        assert_eq!(
            world.resource::<ArenaGrid>().get(2, 2),
            Cell::FixedWall,
            "piercing blast should not destroy FixedWall"
        );

        // Player 1 behind the pillar should survive — blast stopped at FixedWall
        assert!(
            world.get::<Alive>(p1).is_some(),
            "piercing blast should stop at FixedWall, player 1 at (1,2) should survive"
        );

        // Bomb cleaned up
        assert_eq!(world.get_mut::<BombCount>(p0).unwrap().active, 0);
    }

    /// Chain reaction across different bomb types: Timed triggers Piercing,
    /// which continues through a destructible wall to hit a target behind it.
    #[test]
    fn chain_reaction_different_bomb_types() {
        let mut world = init_world(42);
        // Layout:
        //   (1,1) Timed bomb (player 0's, fuse=1)
        //   (2,1) Piercing bomb (player 1's, fuse=99 — only chain-explodes)
        //   (3,1) Destructible wall
        //   (4,1) player 2 — should die from piercing chain through wall
        let [p0, p1, p2, ..] = spawn_players(&mut world);
        {
            let mut grid = world.resource_mut::<ArenaGrid>();
            grid.set(1, 1, Cell::Floor);
            grid.set(2, 1, Cell::Floor);
            grid.set(3, 1, Cell::DestructibleWall);
            grid.set(4, 1, Cell::Floor);
        }

        // Move player 2 behind the wall as the victim
        *world.get_mut::<GridPos>(p2).unwrap() = GridPos { x: 4, y: 1 };

        // Place Timed bomb at (1,1) — will explode after 1 tick (owner: p0)
        world.spawn((
            Bomb::new(), // default Timed
            GridPos { x: 1, y: 1 },
            BombFuse {
                owner: p0,
                ticks_remaining: 1,
            },
            BombRange { cells: 3 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active += 1;

        // Place Piercing bomb at (2,1) — won't self-explode (fuse=99, owner: p1)
        world.spawn((
            Bomb::with_type(BombType::Piercing),
            GridPos { x: 2, y: 1 },
            BombFuse {
                owner: p1,
                ticks_remaining: 99,
            },
            BombRange { cells: 3 },
        ));
        world.get_mut::<BombCount>(p1).unwrap().active += 1;

        // Run one tick — Timed bomb explodes, triggers Piercing chain
        let _ = run_tick(&mut world, [None; 4]);

        // Both bombs gone — each owner's active count decremented
        assert_eq!(
            world.get_mut::<BombCount>(p0).unwrap().active,
            0,
            "timed bomb should be despawned"
        );
        assert_eq!(
            world.get_mut::<BombCount>(p1).unwrap().active,
            0,
            "piercing chain bomb should be despawned"
        );

        // Wall destroyed by piercing blast
        assert_eq!(
            world.resource::<ArenaGrid>().get(3, 1),
            Cell::Floor,
            "piercing chain should destroy wall at (3,1)"
        );

        // Player 2 behind wall should be dead — piercing continued through wall
        assert!(
            world.get::<Alive>(p2).is_none(),
            "piercing chain should continue through wall and kill player 2 at (4,1)"
        );
    }

    #[test]
    fn player_moving_into_blast_cell_survives() {
        // Blast only kills during the explosion phase (step 3 of run_tick).
        // Movement happens AFTER explosions (step 4), so a player who moves
        // into a blast cell post-explosion correctly survives — the blast is over.
        let mut world = init_world(42);
        let [p0, p1, ..] = spawn_players(&mut world);

        // Clear a horizontal corridor: (3,1) through (8,1)
        {
            let mut grid = world.resource_mut::<ArenaGrid>();
            for x in 3..=8 {
                grid.set(x, 1, Cell::Floor);
            }
        }

        // Place player 1 at (8,1) — outside the blast zone
        *world.get_mut::<GridPos>(p1).unwrap() = GridPos { x: 8, y: 1 };

        // Place bomb at (5,1) with range 2, fuse=1 (explodes this tick)
        // Blast will cover: (5,1), (6,1), (7,1) going right; (4,1), (3,1) going left
        world.spawn((
            Bomb::new(),
            GridPos { x: 5, y: 1 },
            BombFuse {
                owner: p0,
                ticks_remaining: 1,
            },
            BombRange { cells: 2 },
        ));
        world.get_mut::<BombCount>(p0).unwrap().active += 1;

        // Player 1 moves LEFT from (8,1) to (7,1) — INTO a blast cell
        let actions = [None, Some(BomberAction::Left), None, None];

        // Run tick: bomb explodes → player 1 moves into blast → survives (blast is over)
        let _ = run_tick(&mut world, actions);

        assert!(
            world.get::<Alive>(p1).is_some(),
            "player 1 should survive — movement is after explosion, blast is instantaneous"
        );
    }
}
