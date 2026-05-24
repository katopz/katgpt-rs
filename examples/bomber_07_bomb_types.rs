//! Bomber Arena — Bomb Types Demo (Issue 052, Task A12)
//!
//! Demonstrates each bomb type behavior in isolation:
//! 1. Timed   — normal fuse countdown + explosion
//! 2. Piercing — destroys wall and continues through
//! 3. Remote  — sits until Detonate action triggers it
//! 4. Landmine — explodes when player walks over it
//!
//! Run: `cargo run --example bomber_07_bomb_types --features bomber`

use bevy_ecs::event::Events;
use bevy_ecs::prelude::{Entity, World};
use katgpt_rs::pruners::bomber::arena::EMPTY_ARENA;
use katgpt_rs::pruners::bomber::*;

fn main() {
    println!("Bomber Arena — Bomb Types Demo");
    println!("═══════════════════════════════\n");
    demo_timed_bomb();
    demo_piercing_bomb();
    demo_remote_bomb();
    demo_landmine();
}

fn fresh_world() -> World {
    let arena = ArenaGrid::fixed(EMPTY_ARENA).unwrap();
    let mut world = init_world_with_arena(arena);
    spawn_players(&mut world);
    world
}

fn drain_events(world: &mut World) -> Vec<GameEvent> {
    world.resource_mut::<Events<GameEvent>>().drain().collect()
}

fn find_explosions(events: &[GameEvent]) -> Vec<(i32, i32, u32)> {
    events
        .iter()
        .filter_map(|e| match e {
            GameEvent::BombExploded { pos, range } => Some((pos.0, pos.1, *range)),
            _ => None,
        })
        .collect()
}

fn print_boom(events: &[GameEvent], label: &str) {
    match find_explosions(events).as_slice() {
        [(x, y, r)] => println!("   {label}: BOOM! at ({x},{y}) range={r}"),
        other => println!("   {label}: {other:?}"),
    }
}

fn print_fuse(world: &World, bomb: Entity, label: &str) {
    match world.get::<BombFuse>(bomb) {
        Some(f) => println!("   {label}: fuse={}", f.ticks_remaining),
        None => println!("   {label}: BOOM! (unexpected)"),
    }
}

fn owner(world: &World) -> Entity {
    world.resource::<PlayerEntities>().entities[0]
}

fn demo_timed_bomb() {
    println!("1. Timed Bomb (fuse=2, range=3) at (5,5)");
    let mut world = fresh_world();
    let p = owner(&world);
    let bomb = world
        .spawn((
            Bomb::new(),
            GridPos { x: 5, y: 5 },
            BombFuse {
                owner: p,
                ticks_remaining: 2,
            },
            BombRange { cells: 3 },
        ))
        .id();
    world.get_mut::<BombCount>(p).unwrap().active = 1;

    let _ = run_tick(&mut world, [None; 4]);
    drain_events(&mut world);
    print_fuse(&world, bomb, "Tick 1");

    let _ = run_tick(&mut world, [None; 4]);
    print_boom(&drain_events(&mut world), "Tick 2");
    assert_eq!(world.get_mut::<BombCount>(p).unwrap().active, 0);
    println!();
}

fn demo_piercing_bomb() {
    println!("2. Piercing Bomb (fuse=2, range=3) at (5,5) with wall at (6,5)");
    let mut world = fresh_world();
    let p = owner(&world);
    world
        .resource_mut::<ArenaGrid>()
        .set(6, 5, Cell::DestructibleWall);
    let bomb = world
        .spawn((
            Bomb::with_type(BombType::Piercing),
            GridPos { x: 5, y: 5 },
            BombFuse {
                owner: p,
                ticks_remaining: 2,
            },
            BombRange { cells: 3 },
        ))
        .id();
    world.get_mut::<BombCount>(p).unwrap().active = 1;

    let _ = run_tick(&mut world, [None; 4]);
    drain_events(&mut world);
    print_fuse(&world, bomb, "Tick 1");

    let _ = run_tick(&mut world, [None; 4]);
    print_boom(&drain_events(&mut world), "Tick 2");
    match world.resource::<ArenaGrid>().get(6, 5) {
        Cell::Floor => println!("   Wall at (6,5) destroyed — blast continued to (7,5)!"),
        other => println!("   Wall at (6,5) still standing: {other:?}"),
    }
    assert_eq!(world.get_mut::<BombCount>(p).unwrap().active, 0);
    println!();
}

fn demo_remote_bomb() {
    println!("3. Remote Bomb at (5,5)");
    let mut world = fresh_world();
    let p = owner(&world);
    // fuse=1 would explode on Timed — Remote ignores fuse countdown
    let bomb = world
        .spawn((
            Bomb::with_type(BombType::Remote),
            GridPos { x: 5, y: 5 },
            BombFuse {
                owner: p,
                ticks_remaining: 1,
            },
            BombRange { cells: 3 },
        ))
        .id();
    world.get_mut::<BombCount>(p).unwrap().active = 1;

    let _ = run_tick(&mut world, [None; 4]);
    drain_events(&mut world);
    match world.get::<BombFuse>(bomb) {
        Some(f) => println!(
            "   Tick 1: fuse={} (unchanged — waiting for detonate)",
            f.ticks_remaining
        ),
        None => println!("   Tick 1: BOOM! (unexpected — remote should ignore fuse)"),
    }

    let _ = run_tick(&mut world, [None; 4]);
    drain_events(&mut world);
    match world.get::<BombFuse>(bomb) {
        Some(f) => println!("   Tick 2: fuse={} (unchanged)", f.ticks_remaining),
        None => println!("   Tick 2: BOOM! (unexpected)"),
    }

    let _ = run_tick(&mut world, [Some(BomberAction::Detonate), None, None, None]);
    print_boom(&drain_events(&mut world), "Detonate");
    assert_eq!(world.get_mut::<BombCount>(p).unwrap().active, 0);
    println!();
}

fn demo_landmine() {
    println!("4. Landmine at (3,1)");
    let mut world = fresh_world();
    let p = owner(&world);
    // Player 0 starts at (1,1). Landmine at (3,1). BombRange=5 ignored — always 1.
    world.spawn((
        Bomb::with_type(BombType::Landmine),
        GridPos { x: 3, y: 1 },
        BombFuse {
            owner: p,
            ticks_remaining: u32::MAX,
        },
        BombRange { cells: 5 },
    ));
    world.get_mut::<BombCount>(p).unwrap().active = 1;

    let _ = run_tick(&mut world, [Some(BomberAction::Right), None, None, None]);
    drain_events(&mut world);
    let pos = world.get::<GridPos>(p).unwrap();
    println!("   Tick 1: player walks to ({},{}) — safe", pos.x, pos.y);

    let _ = run_tick(&mut world, [Some(BomberAction::Right), None, None, None]);
    match find_explosions(&drain_events(&mut world)).as_slice() {
        [(x, y, r)] => println!("   Tick 2: player walks to ({x},{y}) — BOOM! range={r}"),
        other => println!("   Tick 2: {other:?}"),
    }
    match world.get::<Alive>(p).is_some() {
        true => println!("   Player survived (unexpected)"),
        false => println!("   Player killed by landmine (range=1, ignored BombRange=5)"),
    }
    println!();
}
