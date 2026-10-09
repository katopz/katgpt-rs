//! Runetrace — the scene-as-text format (seal-remake Proposal 005, Phase 2
//! T2.1): every entity's strategy rendered as a glyph-led pseudo-code
//! condition + DAG block.
//!
//! The SIBLING of [`crate::decision_wire`] for the not-usually-multiple-choice
//! shape: `decision_wire` carries typed questions OUT to a decision lane;
//! `runetrace` carries the live scene IN as deterministic text — for human
//! debug (entity inspector / scene-tree panel), for LLM reasoning lanes
//! (instinct/rethink consume scene text and answer on `decision_wire`), and
//! for sleep-time consolidation digests.
//!
//! ## The digest law (what every design decision serves)
//!
//! The canonical text render is BYTE-DETERMINISTIC for a given doc: fixed
//! field order, pinned float spelling (Rust's shortest-round-trip `Display`
//! for `f32`), closed stage/kind vocabularies. Two renders of one doc are
//! byte-identical, so a [`blake3::Hash`] over the text is a stable identity
//! for capture/diff/consolidation (the `lab_trace` NDJSON discipline, one
//! layer up).
//!
//! ## Raw stays raw (the sync-boundary law)
//!
//! `pos`/vitals are RAW exact values carried verbatim — this format OBSERVES
//! the scene; it is never a sync surface, and nothing downstream may parse
//! the text back into game truth (the ConvexTok law). Raw pods/sync stay
//! canonical.
//!
//! ## Cold path by construction
//!
//! The renderer ALLOCATES (a `String`) — Runetrace is a panel/lane/debug
//! artifact at belt cadence, NEVER the per-tick hot path (Proposal 005
//! caveat 2). `render_into(&mut String)` exists so callers reuse one buffer
//! across renders. Producers (cognition pods) stay alloc-free; this crate
//! only defines the format.
//!
//! ## Fail-closed
//!
//! [`RunetraceDoc::validate`] pins the contract (version, unique ids, DAG
//! parent ordering, sigmoid-drive ranges, vital finiteness); `render` /
//! `render_into` / `content_hash` validate FIRST — an unvalidated doc never
//! produces text or a digest.
//!
//! ## Vocabulary (closed, additive-only after adoption)
//!
//! `DagStage` — the six-step decision path (perceive → believe → drive →
//! goal → transition → action); `EntityKind` — player / monster / npc / pet.
//! Free text lives in producer-authored fields (ids, conditions, notes) — a
//! v2 widening is a versioned change, never a silent one.
//!
//! Opt-in (`runetrace`) per the no-default-consumer rule — the consumers are
//! the seal-remake panels and the instinct/rethink lanes (Proposal 005
//! Phases 3–4); promotion rides the GOAT gate.

use serde::{Deserialize, Serialize};

/// The wire format version this module speaks (`runetrace/v1`).
pub const RUNETRACE_VERSION: u16 = 1;

/// The closed entity-kind vocabulary (renders as its lowercase name).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum EntityKind {
    Player,
    Monster,
    Npc,
    Pet,
}

impl EntityKind {
    /// The canonical spelling (also the serde tag + render form).
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Player => "player",
            Self::Monster => "monster",
            Self::Npc => "npc",
            Self::Pet => "pet",
        }
    }
}

/// The closed decision-path stage vocabulary — the six steps every block's
/// DAG rows walk (perception → belief → drives → goal → FSM transition →
/// action). Depth comes from parent indices, not stage order; a producer
/// may branch (two goals from one belief) — stages NAME a row, they do not
/// constrain the shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "lowercase")]
pub enum DagStage {
    Perceive,
    Believe,
    Drive,
    Goal,
    Transition,
    Action,
}

impl DagStage {
    /// The canonical spelling (also the serde tag + render form).
    #[inline]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Perceive => "perceive",
            Self::Believe => "believe",
            Self::Drive => "drive",
            Self::Goal => "goal",
            Self::Transition => "transition",
            Self::Action => "action",
        }
    }
}

/// One named raw vital (HP, MP, stamina…) — generic across games; the
/// producer owns the name and the scale.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vital {
    /// Short lowercase name as it renders (`hp`, `mp`, `sta`). Non-empty.
    pub name: String,
    /// Current value (raw exact — never latent-projected).
    pub value: f32,
    /// Maximum (the denominator when rendering `value/max`). Finite, > 0.
    pub max: f32,
}

/// One FIRED condition this belt, with the sigmoid drive that carried it.
/// Presence in `think` means the condition fired — producers omit what did
/// not fire (Proposal 005 §4.1); the ordered `dag` section is the decision
/// path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditionRow {
    /// The condition text (`hp_frac < 0.25`). Non-empty, producer-authored.
    pub condition: String,
    /// What the condition selects (`drink_red`). Non-empty.
    pub action: String,
    /// The named sigmoid drive value in `[0, 1]` when one carried this
    /// condition; `None` for threshold-only conditions.
    #[serde(default)]
    pub drive: Option<f32>,
}

/// One node of the tick's decision-path DAG. Rows are parent-indexed:
/// `parent: None` roots a path; `parent: Some(i)` MUST name an EARLIER row
/// (`i < own_index` — forward refs and self-refs are refused by
/// [`RunetraceDoc::validate`]). Depth at render time walks the chain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DagRow {
    /// Index into the owning `EntityBlock::dag`, or `None` for a root.
    #[serde(default)]
    pub parent: Option<u32>,
    /// Which of the six decision-path steps this row is.
    pub stage: DagStage,
    /// The row body (`bee_warrior dist 4.1`, `threat high`). Non-empty.
    pub text: String,
}

/// One entity's strategy block (the hero balloon generalized).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EntityBlock {
    /// Stable entity id within the scene. Non-empty; unique per doc.
    pub id: String,
    /// The closed kind vocabulary.
    pub kind: EntityKind,
    /// The lead glyph (one grapheme — the balloon's icon-led rows are the
    /// seed). Non-empty.
    pub glyph: String,
    /// RAW position (the `[f32; 3]` f32 canonical — MapPos3D's shape);
    /// `None` when the producer has none. Never latent-encoded.
    #[serde(default)]
    pub pos: Option<[f32; 3]>,
    /// Raw named vitals, rendered `name value/max` in order.
    #[serde(default)]
    pub vitals: Vec<Vital>,
    /// The conditions that fired this belt (rendered `if c → a`).
    #[serde(default)]
    pub think: Vec<ConditionRow>,
    /// The tick's decision path, parent-indexed, rendered in order.
    #[serde(default)]
    pub dag: Vec<DagRow>,
}

/// The scene header — one line of game-agnostic context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneHeader {
    /// The scene label as it renders (`Z1 rain`, `market · dusk`). Free
    /// producer text. Non-empty.
    pub label: String,
}

/// The whole scene: header + every entity block, rendered under one
/// `runetrace/vN` banner line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunetraceDoc {
    /// Format version; must equal [`RUNETRACE_VERSION`].
    pub version: u16,
    /// The tick the doc captures (raw exact).
    pub tick: u64,
    /// The scene context line.
    pub scene: SceneHeader,
    /// Entity blocks, rendered in order (the panel sorts by its own key;
    /// the format preserves what it is given).
    pub entities: Vec<EntityBlock>,
}

impl RunetraceDoc {
    /// A doc at the current format version.
    pub fn new(tick: u64, scene_label: impl Into<String>) -> Self {
        Self {
            version: RUNETRACE_VERSION,
            tick,
            scene: SceneHeader {
                label: scene_label.into(),
            },
            entities: Vec::new(),
        }
    }

    /// Fail-closed structural validation (the `decision_wire` discipline:
    /// pins the CONTRACT — arity, ranges, finiteness, ordering — not scene
    /// quality).
    pub fn validate(&self) -> Result<(), RunetraceError> {
        if self.version != RUNETRACE_VERSION {
            return Err(RunetraceError::UnsupportedVersion {
                got: self.version,
                want: RUNETRACE_VERSION,
            });
        }
        if self.scene.label.is_empty() {
            return Err(RunetraceError::EmptySceneLabel);
        }
        let mut seen_ids: std::collections::HashSet<&str> =
            std::collections::HashSet::with_capacity(self.entities.len());
        for entity in &self.entities {
            if entity.id.is_empty() {
                return Err(RunetraceError::EmptyEntityId);
            }
            if !seen_ids.insert(entity.id.as_str()) {
                return Err(RunetraceError::DuplicateEntityId {
                    id: entity.id.clone(),
                });
            }
            if entity.glyph.is_empty() {
                return Err(RunetraceError::EmptyGlyph {
                    id: entity.id.clone(),
                });
            }
            if let Some([x, y, z]) = entity.pos
                && (!x.is_finite() || !y.is_finite() || !z.is_finite())
            {
                return Err(RunetraceError::NonFinitePosition {
                    id: entity.id.clone(),
                });
            }
            for vital in &entity.vitals {
                if vital.name.is_empty() {
                    return Err(RunetraceError::EmptyVitalName {
                        id: entity.id.clone(),
                    });
                }
                if !vital.value.is_finite() || !vital.max.is_finite() {
                    return Err(RunetraceError::NonFiniteVital {
                        id: entity.id.clone(),
                        vital: vital.name.clone(),
                    });
                }
                if vital.max <= 0.0 {
                    return Err(RunetraceError::InvalidVitalRange {
                        id: entity.id.clone(),
                        vital: vital.name.clone(),
                    });
                }
            }
            for (row_index, row) in entity.think.iter().enumerate() {
                if row.condition.is_empty() || row.action.is_empty() {
                    return Err(RunetraceError::EmptyCondition {
                        id: entity.id.clone(),
                        row: row_index,
                    });
                }
                let Some(drive) = row.drive else {
                    continue;
                };
                if !drive.is_finite() {
                    return Err(RunetraceError::NonFiniteDrive {
                        id: entity.id.clone(),
                        row: row_index,
                    });
                }
                if !(0.0..=1.0).contains(&drive) {
                    return Err(RunetraceError::DriveOutOfRange {
                        id: entity.id.clone(),
                        row: row_index,
                        value: drive,
                    });
                }
            }
            for (row_index, row) in entity.dag.iter().enumerate() {
                if row.text.is_empty() {
                    return Err(RunetraceError::EmptyDagText {
                        id: entity.id.clone(),
                        row: row_index,
                    });
                }
                if let Some(parent) = row.parent {
                    // `dag rows are parent-indexed … MUST name an EARLIER
                    // row` — u32 vs usize: a dag longer than u32::MAX is
                    // absurd, but the comparison stays overflow-free.
                    if parent >= row_index as u32 {
                        return Err(RunetraceError::DagParentOutOfRange {
                            id: entity.id.clone(),
                            row: row_index,
                            parent,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Render the canonical text into `out`, REPLACING its contents (the
    /// belt-cadence reuse loop keeps one buffer; append semantics would
    /// silently accumulate docs). Validates first; fail-closed.
    pub fn render_into(&self, out: &mut String) -> Result<(), RunetraceError> {
        self.validate()?;
        out.clear();
        out.reserve(
            48 + self.scene.label.len()
                + self
                    .entities
                    .iter()
                    .map(|e| 64 + e.id.len() + 16 * (e.vitals.len() + e.think.len() + e.dag.len()))
                    .sum::<usize>(),
        );
        out.push_str("runetrace/v");
        push_u64(out, u64::from(self.version));
        out.push_str(" tick ");
        push_u64(out, self.tick);
        out.push_str(" scene ");
        out.push_str(&self.scene.label);
        out.push('\n');
        for entity in &self.entities {
            render_entity(out, entity);
        }
        Ok(())
    }

    /// Render the canonical text into a fresh `String`. Validates first.
    pub fn render(&self) -> Result<String, RunetraceError> {
        let mut out = String::new();
        self.render_into(&mut out)?;
        Ok(out)
    }

    /// BLAKE3 over the canonical text — the stable capture/diff identity
    /// (the digest law). Validates first; fail-closed.
    pub fn content_hash(&self) -> Result<blake3::Hash, RunetraceError> {
        let mut text = String::new();
        self.render_into(&mut text)?;
        Ok(blake3::hash(text.as_bytes()))
    }
}

fn render_entity(out: &mut String, entity: &EntityBlock) {
    out.push_str(&entity.glyph);
    out.push(' ');
    out.push_str(&entity.id);
    out.push_str(" · ");
    out.push_str(entity.kind.as_str());
    out.push('\n');
    if let Some([x, y, z]) = entity.pos {
        out.push_str("  pos ");
        push_f32(out, x);
        out.push(' ');
        push_f32(out, y);
        out.push(' ');
        push_f32(out, z);
        out.push('\n');
    }
    for vital in &entity.vitals {
        out.push_str("  ");
        out.push_str(&vital.name);
        out.push(' ');
        push_f32(out, vital.value);
        out.push('/');
        push_f32(out, vital.max);
        out.push('\n');
    }
    for row in &entity.think {
        out.push_str("  if ");
        out.push_str(&row.condition);
        out.push_str(" → ");
        out.push_str(&row.action);
        if let Some(drive) = row.drive {
            out.push_str(" (drive ");
            push_f32(out, drive);
            out.push(')');
        }
        out.push('\n');
    }
    // Depth per row walks the parent chain; parents precede (validated), so
    // a flat Vec<usize> of depths computes in one pass.
    let mut depths: Vec<u32> = Vec::with_capacity(entity.dag.len());
    for row in &entity.dag {
        let depth = match row.parent {
            None => 0,
            Some(parent) => depths[parent as usize] + 1,
        };
        depths.push(depth);
        // Entity-level base indent (2) + 2 per depth — dag rows sit under
        // the entity's field rows, one depth level per parent hop.
        out.push_str("  ");
        for _ in 0..depth {
            out.push_str("  ");
        }
        out.push_str(row.stage.as_str());
        out.push(' ');
        out.push_str(&row.text);
        out.push('\n');
    }
}

/// f32 in Rust shortest-round-trip `Display` form — THE pinned float
/// spelling; any renderer change that alters these bytes breaks the digest
/// law and the G1 fixtures catch it.
fn push_f32(out: &mut String, v: f32) {
    // Finiteness is validation-refused upstream; NaN/inf never render. The
    // unreachable arm keeps `debug_assert` honest if validation regresses.
    debug_assert!(v.is_finite(), "runetrace render: non-finite f32");
    out.push_str(&fmt_f32(v));
}

// Shortest-round-trip via std Display — isolated here so the spelling is
// pinned at ONE site (the G1 fixtures bite any change to this fn).
fn fmt_f32(v: f32) -> String {
    format!("{v}")
}

fn push_u64(out: &mut String, v: u64) {
    out.push_str(&fmt_u64(v));
}

fn fmt_u64(v: u64) -> String {
    format!("{v}")
}

/// Fail-closed validation failures (one variant per contract clause).
/// Not a wire type — the payload never rides the format; callers log it.
#[derive(Clone, Debug, PartialEq)]
pub enum RunetraceError {
    UnsupportedVersion {
        got: u16,
        want: u16,
    },
    EmptySceneLabel,
    EmptyEntityId,
    DuplicateEntityId {
        id: String,
    },
    EmptyGlyph {
        id: String,
    },
    NonFinitePosition {
        id: String,
    },
    EmptyVitalName {
        id: String,
    },
    NonFiniteVital {
        id: String,
        vital: String,
    },
    /// `max <= 0` — the `value/max` render needs a positive denominator.
    InvalidVitalRange {
        id: String,
        vital: String,
    },
    EmptyCondition {
        id: String,
        row: usize,
    },
    NonFiniteDrive {
        id: String,
        row: usize,
    },
    /// Drives are sigmoid outputs — `[0, 1]`, never outside.
    DriveOutOfRange {
        id: String,
        row: usize,
        value: f32,
    },
    EmptyDagText {
        id: String,
        row: usize,
    },
    /// Parent must name an EARLIER row (`parent < own index`); self-refs
    /// and forward refs both land here.
    DagParentOutOfRange {
        id: String,
        row: usize,
        parent: u32,
    },
}

impl std::fmt::Display for RunetraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedVersion { got, want } => {
                write!(f, "runetrace: unsupported version {got} (want {want})")
            }
            Self::EmptySceneLabel => write!(f, "runetrace: scene label is empty"),
            Self::EmptyEntityId => write!(f, "runetrace: entity id is empty"),
            Self::DuplicateEntityId { id } => {
                write!(f, "runetrace: duplicate entity id `{id}`")
            }
            Self::EmptyGlyph { id } => {
                write!(f, "runetrace: entity `{id}` has an empty glyph")
            }
            Self::NonFinitePosition { id } => {
                write!(f, "runetrace: entity `{id}` position is non-finite")
            }
            Self::EmptyVitalName { id } => {
                write!(f, "runetrace: entity `{id}` has an empty vital name")
            }
            Self::NonFiniteVital { id, vital } => {
                write!(f, "runetrace: entity `{id}` vital `{vital}` is non-finite")
            }
            Self::InvalidVitalRange { id, vital } => {
                write!(
                    f,
                    "runetrace: entity `{id}` vital `{vital}` max must be > 0"
                )
            }
            Self::EmptyCondition { id, row } => {
                write!(
                    f,
                    "runetrace: entity `{id}` think row {row} has an empty condition or action"
                )
            }
            Self::NonFiniteDrive { id, row } => {
                write!(
                    f,
                    "runetrace: entity `{id}` think row {row} drive is non-finite"
                )
            }
            Self::DriveOutOfRange { id, row, value } => {
                write!(
                    f,
                    "runetrace: entity `{id}` think row {row} drive {value} outside [0,1]"
                )
            }
            Self::EmptyDagText { id, row } => {
                write!(f, "runetrace: entity `{id}` dag row {row} text is empty")
            }
            Self::DagParentOutOfRange { id, row, parent } => write!(
                f,
                "runetrace: entity `{id}` dag row {row} parent {parent} must name an earlier row"
            ),
        }
    }
}

impl std::error::Error for RunetraceError {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical fixture — a hero mid-fight and a monster giving chase.
    /// Every field shape appears once: glyph-led rows, raw pos, vitals, a
    /// drive-less condition, a driven one, a two-branch DAG.
    fn canonical_doc() -> RunetraceDoc {
        let mut doc = RunetraceDoc::new(1204, "Z1 rain");
        doc.entities.push(EntityBlock {
            id: "hero".into(),
            kind: EntityKind::Player,
            glyph: "🧭".into(),
            pos: Some([12.5, 3.2, 0.0]),
            vitals: vec![
                Vital {
                    name: "hp".into(),
                    value: 12.0,
                    max: 40.0,
                },
                Vital {
                    name: "mp".into(),
                    value: 3.0,
                    max: 20.0,
                },
            ],
            think: vec![
                ConditionRow {
                    condition: "hp_frac < 0.25".into(),
                    action: "drink_red".into(),
                    drive: Some(0.81),
                },
                ConditionRow {
                    condition: "potion bag empty".into(),
                    action: "flee".into(),
                    drive: None,
                },
            ],
            dag: vec![
                DagRow {
                    parent: None,
                    stage: DagStage::Perceive,
                    text: "bee_warrior dist 4.1".into(),
                },
                DagRow {
                    parent: Some(0),
                    stage: DagStage::Believe,
                    text: "threat high".into(),
                },
                DagRow {
                    parent: Some(1),
                    stage: DagStage::Drive,
                    text: "fear 0.71".into(),
                },
                DagRow {
                    parent: Some(2),
                    stage: DagStage::Goal,
                    text: "flee_to_safety".into(),
                },
                DagRow {
                    parent: Some(2),
                    stage: DagStage::Goal,
                    text: "finish_quest".into(),
                },
                DagRow {
                    parent: Some(3),
                    stage: DagStage::Transition,
                    text: "Fight→Flee".into(),
                },
                DagRow {
                    parent: Some(5),
                    stage: DagStage::Action,
                    text: "move_to 8.2 9.9".into(),
                },
            ],
        });
        doc.entities.push(EntityBlock {
            id: "bee_1".into(),
            kind: EntityKind::Monster,
            glyph: "🐝".into(),
            pos: None,
            vitals: vec![Vital {
                name: "hp".into(),
                value: 30.0,
                max: 30.0,
            }],
            think: vec![ConditionRow {
                condition: "target in reach".into(),
                action: "sting".into(),
                drive: Some(1.0),
            }],
            dag: vec![DagRow {
                parent: None,
                stage: DagStage::Action,
                text: "sting hero".into(),
            }],
        });
        doc
    }

    /// The golden canonical text — the byte-pin every renderer change must
    /// reproduce EXACTLY (the digest law's G1 fixture).
    const GOLDEN_DOC_TEXT: &str = concat!(
        "runetrace/v1 tick 1204 scene Z1 rain\n",
        "🧭 hero · player\n",
        "  pos 12.5 3.2 0\n",
        "  hp 12/40\n",
        "  mp 3/20\n",
        "  if hp_frac < 0.25 → drink_red (drive 0.81)\n",
        "  if potion bag empty → flee\n",
        "  perceive bee_warrior dist 4.1\n",
        "    believe threat high\n",
        "      drive fear 0.71\n",
        "        goal flee_to_safety\n",
        "        goal finish_quest\n",
        "          transition Fight→Flee\n",
        "            action move_to 8.2 9.9\n",
        "🐝 bee_1 · monster\n",
        "  hp 30/30\n",
        "  if target in reach → sting (drive 1)\n",
        "  action sting hero\n",
    );

    #[test]
    fn golden_canonical_text_is_byte_stable() {
        let doc = canonical_doc();
        let text = doc.render().expect("canonical doc renders");
        assert_eq!(text, GOLDEN_DOC_TEXT);
    }

    #[test]
    fn render_is_deterministic_and_digest_stable() {
        let doc = canonical_doc();
        let a = doc.render().expect("first render");
        let b = doc.render().expect("second render");
        assert_eq!(a, b, "two renders of one doc must be byte-identical");
        assert_eq!(
            doc.content_hash().expect("hash a"),
            doc.content_hash().expect("hash b"),
            "digest must be stable across renders"
        );
        // render_into reuses a buffer without leaking prior contents.
        let mut reused = String::from("stale buffer contents");
        doc.render_into(&mut reused).expect("render_into");
        assert_eq!(reused, a);
    }

    #[test]
    fn digest_is_over_the_canonical_text() {
        let doc = canonical_doc();
        let text = doc.render().expect("render");
        assert_eq!(
            doc.content_hash().expect("hash"),
            blake3::hash(text.as_bytes()),
            "content_hash is blake3 over the canonical text"
        );
    }

    #[test]
    fn json_round_trip_preserves_doc() {
        let doc = canonical_doc();
        let json = serde_json::to_string(&doc).expect("serialize");
        let back: RunetraceDoc = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, doc);
        // Byte-stability: two serializations identical (field order + the
        // always-emit law — no skip_serializing_if anywhere).
        assert_eq!(serde_json::to_string(&doc).expect("serialize again"), json);
    }

    #[test]
    fn postcard_round_trip_preserves_doc() {
        let doc = canonical_doc();
        let bytes = postcard::to_allocvec(&doc).expect("serialize");
        let back: RunetraceDoc = postcard::from_bytes(&bytes).expect("deserialize");
        assert_eq!(back, doc);
    }

    #[test]
    fn validate_accepts_the_canonical_doc() {
        canonical_doc().validate().expect("canonical doc is valid");
    }

    #[test]
    fn validate_refuses_version_mismatch() {
        let mut doc = canonical_doc();
        doc.version = 2;
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::UnsupportedVersion { got: 2, want: 1 })
        );
        assert!(doc.render().is_err(), "render is fail-closed");
        assert!(doc.content_hash().is_err(), "digest is fail-closed");
    }

    #[test]
    fn validate_refuses_empty_scene_label() {
        let mut doc = canonical_doc();
        doc.scene.label = String::new();
        assert_eq!(doc.validate(), Err(RunetraceError::EmptySceneLabel));
    }

    #[test]
    fn validate_refuses_duplicate_entity_id() {
        let mut doc = canonical_doc();
        let clone = doc.entities[0].clone();
        doc.entities.push(clone);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::DuplicateEntityId { id: "hero".into() })
        );
    }

    #[test]
    fn validate_refuses_empty_id_and_glyph() {
        let mut doc = canonical_doc();
        doc.entities[0].id = String::new();
        assert_eq!(doc.validate(), Err(RunetraceError::EmptyEntityId));
        let mut doc = canonical_doc();
        doc.entities[0].glyph = String::new();
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::EmptyGlyph { id: "hero".into() })
        );
    }

    #[test]
    fn validate_refuses_non_finite_position() {
        let mut doc = canonical_doc();
        doc.entities[0].pos = Some([f32::NAN, 0.0, 0.0]);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::NonFinitePosition { id: "hero".into() })
        );
    }

    #[test]
    fn validate_refuses_bad_vitals() {
        let mut doc = canonical_doc();
        doc.entities[0].vitals[0].name = String::new();
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::EmptyVitalName { id: "hero".into() })
        );
        let mut doc = canonical_doc();
        doc.entities[0].vitals[0].value = f32::INFINITY;
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::NonFiniteVital {
                id: "hero".into(),
                vital: "hp".into(),
            })
        );
        let mut doc = canonical_doc();
        doc.entities[0].vitals[0].max = 0.0;
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::InvalidVitalRange {
                id: "hero".into(),
                vital: "hp".into(),
            })
        );
    }

    #[test]
    fn validate_refuses_empty_condition() {
        let mut doc = canonical_doc();
        doc.entities[0].think[0].condition = String::new();
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::EmptyCondition {
                id: "hero".into(),
                row: 0,
            })
        );
    }

    #[test]
    fn validate_refuses_bad_drives() {
        let mut doc = canonical_doc();
        doc.entities[0].think[0].drive = Some(f32::NAN);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::NonFiniteDrive {
                id: "hero".into(),
                row: 0,
            })
        );
        let mut doc = canonical_doc();
        doc.entities[0].think[0].drive = Some(1.2);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::DriveOutOfRange {
                id: "hero".into(),
                row: 0,
                value: 1.2,
            })
        );
    }

    #[test]
    fn validate_refuses_empty_dag_text() {
        let mut doc = canonical_doc();
        doc.entities[0].dag[0].text = String::new();
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::EmptyDagText {
                id: "hero".into(),
                row: 0,
            })
        );
    }

    #[test]
    fn validate_refuses_self_forward_and_far_parents() {
        // Self-reference.
        let mut doc = canonical_doc();
        doc.entities[0].dag[1].parent = Some(1);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::DagParentOutOfRange {
                id: "hero".into(),
                row: 1,
                parent: 1,
            })
        );
        // Forward reference.
        let mut doc = canonical_doc();
        doc.entities[0].dag[1].parent = Some(3);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::DagParentOutOfRange {
                id: "hero".into(),
                row: 1,
                parent: 3,
            })
        );
        // Out-of-bounds reference.
        let mut doc = canonical_doc();
        doc.entities[0].dag[1].parent = Some(99);
        assert_eq!(
            doc.validate(),
            Err(RunetraceError::DagParentOutOfRange {
                id: "hero".into(),
                row: 1,
                parent: 99,
            })
        );
    }

    #[test]
    fn float_spelling_is_shortest_round_trip() {
        let mut doc = RunetraceDoc::new(1, "spell");
        doc.entities.push(EntityBlock {
            id: "f".into(),
            kind: EntityKind::Npc,
            glyph: "·".into(),
            pos: Some([0.1, -2.35, 1e10]),
            vitals: vec![Vital {
                name: "hp".into(),
                value: 40.0,
                max: 40.0,
            }],
            think: vec![],
            dag: vec![],
        });
        let text = doc.render().expect("render");
        let expected = concat!(
            "runetrace/v1 tick 1 scene spell\n",
            "· f · npc\n",
            "  pos 0.1 -2.35 10000000000\n",
            "  hp 40/40\n",
        );
        assert_eq!(text, expected);
    }

    #[test]
    fn empty_scene_renders_only_the_banner() {
        let doc = RunetraceDoc::new(7, "void");
        let text = doc.render().expect("render");
        assert_eq!(text, "runetrace/v1 tick 7 scene void\n");
        assert!(doc.content_hash().is_ok());
    }

    #[test]
    fn deep_dag_indents_two_spaces_per_depth() {
        let mut doc = RunetraceDoc::new(1, "deep");
        let mut dag = Vec::with_capacity(4);
        for i in 0..4u32 {
            dag.push(DagRow {
                parent: if i == 0 { None } else { Some(i - 1) },
                stage: DagStage::Action,
                text: format!("step {i}"),
            });
        }
        doc.entities.push(EntityBlock {
            id: "chain".into(),
            kind: EntityKind::Pet,
            glyph: "*".into(),
            pos: None,
            vitals: vec![],
            think: vec![],
            dag,
        });
        let text = doc.render().expect("render");
        assert!(text.contains("\n  action step 0\n"));
        assert!(text.contains("\n    action step 1\n"));
        assert!(text.contains("\n      action step 2\n"));
        assert!(text.contains("\n        action step 3\n"));
    }

    #[test]
    fn error_display_names_the_entity_and_row() {
        let e = RunetraceError::DagParentOutOfRange {
            id: "bee_1".into(),
            row: 2,
            parent: 7,
        };
        let msg = e.to_string();
        assert!(msg.contains("bee_1") && msg.contains("row 2") && msg.contains("parent 7"));
    }
}
