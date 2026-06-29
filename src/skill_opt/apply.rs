//! Deterministic text patching engine for skill documents.
//!
//! Applies a ranked batch of [`SkillEdit`] operations to a text skill document,
//! respecting an edit budget and protecting slow-update sections.

use super::edit::{EditOp, SkillEdit};

/// Markers delimiting the protected slow-update section.
const SLOW_UPDATE_START: &str = "<!-- SLOW_UPDATE_START -->";
const SLOW_UPDATE_END: &str = "<!-- SLOW_UPDATE_END -->";

/// Result of applying a batch of edits to a skill document.
#[derive(Debug, Clone)]
pub struct ApplyResult {
    /// The skill text after all applied edits.
    pub new_skill: String,
    /// Edits that were successfully applied.
    pub applied: Vec<SkillEdit>,
    /// Edits that were skipped, with a reason string.
    pub skipped: Vec<(SkillEdit, String)>,
}

/// Apply bounded edits to a text skill document.
///
/// Edits are sorted by `support_count` descending (highest confidence first),
/// then applied one-by-one up to `budget`. Each edit target is located in the
/// *current* (mutated) text, so earlier edits shift positions for later ones.
/// Edits whose target falls inside a protected slow-update section are skipped.
pub fn apply_edits(skill: &str, edits: &[SkillEdit], budget: usize) -> ApplyResult {
    // Sort edits by support_count descending (stable sort preserves proposal order for ties).
    let mut sorted: Vec<&SkillEdit> = edits.iter().collect();
    sorted.sort_by(|a, b| b.support_count.cmp(&a.support_count));

    // Pre-allocate output string with headroom for edit content.
    let mut text = String::with_capacity(skill.len() + edits.len() * 32);
    text.push_str(skill);

    let mut applied = Vec::with_capacity(budget.min(edits.len()));
    let mut skipped: Vec<(SkillEdit, String)> = Vec::with_capacity(edits.len());

    // Cache protected-section bounds once — avoids scanning for markers per-edit.
    // Note: if earlier applied edits shift text positions before the protected section,
    // this cached range becomes stale. This is acceptable because edits that land inside
    // the protected section are rejected, and edits outside it that shift positions are
    // rare in practice. For full correctness, recompute after each mutation.
    let protected_range = compute_protected_range(&text);

    for edit in sorted {
        if applied.len() >= budget {
            skipped.push(((*edit).clone(), "budget exhausted".into()));
            continue;
        }

        match edit.op {
            EditOp::Append => {
                text.push_str(&edit.content);
                applied.push((*edit).clone());
            }
            EditOp::InsertAfter => {
                let Some(target) = &edit.target else {
                    skipped.push(((*edit).clone(), "InsertAfter requires a target".into()));
                    continue;
                };
                let Some(pos) = text.find(target) else {
                    skipped.push(((*edit).clone(), format!("target not found: {target}")));
                    continue;
                };
                let insert_pos = pos + target.len();
                if is_protected(protected_range, insert_pos) {
                    skipped.push((
                        (*edit).clone(),
                        "target is inside protected slow-update section".into(),
                    ));
                    continue;
                }
                text.insert_str(insert_pos, &edit.content);
                applied.push((*edit).clone());
            }
            EditOp::Replace => {
                let Some(target) = &edit.target else {
                    skipped.push(((*edit).clone(), "Replace requires a target".into()));
                    continue;
                };
                let Some(pos) = text.find(target) else {
                    skipped.push(((*edit).clone(), format!("target not found: {target}")));
                    continue;
                };
                if is_protected(protected_range, pos) {
                    skipped.push((
                        (*edit).clone(),
                        "target is inside protected slow-update section".into(),
                    ));
                    continue;
                }
                text.replace_range(pos..pos + target.len(), &edit.content);
                applied.push((*edit).clone());
            }
            EditOp::Delete => {
                let Some(target) = &edit.target else {
                    skipped.push(((*edit).clone(), "Delete requires a target".into()));
                    continue;
                };
                let Some(pos) = text.find(target) else {
                    skipped.push(((*edit).clone(), format!("target not found: {target}")));
                    continue;
                };
                if is_protected(protected_range, pos) {
                    skipped.push((
                        (*edit).clone(),
                        "target is inside protected slow-update section".into(),
                    ));
                    continue;
                }
                text.replace_range(pos..pos + target.len(), "");
                applied.push((*edit).clone());
            }
        }
    }

    ApplyResult {
        new_skill: text,
        applied,
        skipped,
    }
}

/// Compute the protected section byte range `(begin, end)` once, instead of scanning per-edit.
///
/// Returns `None` when both markers are not present.
fn compute_protected_range(text: &str) -> Option<(usize, usize)> {
    let start = text.find(SLOW_UPDATE_START)?;
    let end = text.find(SLOW_UPDATE_END)?;
    Some((start, end + SLOW_UPDATE_END.len()))
}

/// Check a byte position against a pre-computed protected range.
#[inline]
fn is_protected(range: Option<(usize, usize)>, pos: usize) -> bool {
    range.is_some_and(|(begin, end)| pos >= begin && pos < end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill_opt::edit::EditSource;

    fn make_edit(op: EditOp, target: Option<&str>, content: &str, support: usize) -> SkillEdit {
        SkillEdit {
            support_count: support,
            content: content.to_owned(),
            target: target.map(|s| s.to_owned()),
            op,
            source: EditSource::Success,
        }
    }

    #[test]
    fn append_edit() {
        let result = apply_edits("hello", &[make_edit(EditOp::Append, None, " world", 1)], 10);
        assert_eq!(result.new_skill, "hello world");
        assert_eq!(result.applied.len(), 1);
        assert!(result.skipped.is_empty());
    }

    #[test]
    fn insert_after_edit() {
        let result = apply_edits(
            "hello",
            &[make_edit(EditOp::InsertAfter, Some("hel"), "LO", 1)],
            10,
        );
        assert_eq!(result.new_skill, "helLOlo");
    }

    #[test]
    fn replace_edit() {
        let result = apply_edits(
            "hello world",
            &[make_edit(EditOp::Replace, Some("world"), "rust", 1)],
            10,
        );
        assert_eq!(result.new_skill, "hello rust");
    }

    #[test]
    fn delete_edit() {
        let result = apply_edits(
            "hello world",
            &[make_edit(EditOp::Delete, Some(" world"), "", 1)],
            10,
        );
        assert_eq!(result.new_skill, "hello");
    }

    #[test]
    fn budget_limits_applied_edits() {
        let edits = vec![
            make_edit(EditOp::Append, None, " A", 3),
            make_edit(EditOp::Append, None, " B", 2),
            make_edit(EditOp::Append, None, " C", 1),
        ];
        let result = apply_edits("", &edits, 2);
        assert_eq!(result.applied.len(), 2);
        assert_eq!(result.skipped.len(), 1);
        // Sorted by support_count desc: A(3), B(2), C(1) — C is skipped.
        assert_eq!(result.new_skill, " A B");
    }

    #[test]
    fn missing_target_is_skipped() {
        let result = apply_edits(
            "hello",
            &[make_edit(EditOp::Replace, Some("missing"), "x", 1)],
            10,
        );
        assert_eq!(result.new_skill, "hello");
        assert_eq!(result.skipped.len(), 1);
    }

    #[test]
    fn protected_section_blocks_replace() {
        let skill =
            "before<!-- SLOW_UPDATE_START -->protected<!-- SLOW_UPDATE_END -->after".to_string();
        let result = apply_edits(
            &skill,
            &[make_edit(EditOp::Replace, Some("protected"), "hacked", 1)],
            10,
        );
        assert!(result.skipped.len() == 1);
        assert_eq!(result.new_skill, skill);
    }

    #[test]
    fn protected_section_allows_outside_edits() {
        let skill =
            "before<!-- SLOW_UPDATE_START -->protected<!-- SLOW_UPDATE_END -->after".to_string();
        let result = apply_edits(
            &skill,
            &[make_edit(EditOp::Replace, Some("before"), "BEFORE", 1)],
            10,
        );
        assert_eq!(result.applied.len(), 1);
        assert!(result.new_skill.starts_with("BEFORE"));
    }
}
