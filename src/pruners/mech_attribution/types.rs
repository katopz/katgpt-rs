//! Core types for Mechanistic Data Attribution.

/// Influence score combining catalyst overlap with activation pattern match.
#[derive(Debug, Clone)]
pub struct MechInfluenceScore {
    /// How much structural catalyst overlap this sample has [0, 1].
    pub catalyst_overlap: f32,
    /// Which catalyst pattern was detected.
    pub pattern: CatalystPattern,
    /// Whether this sample is in the top-K high-influence set.
    pub is_high_influence: bool,
}

/// Structural catalyst patterns that drive circuit formation in LLMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CatalystPattern {
    /// XML-like `<tag>...</tag>` repetition with consistent structure.
    XmlRepetition,
    /// Function signature / type annotation repetition.
    CodeSignature,
    /// LaTeX `\command{...}` repetition.
    LatexFormula,
    /// CSV/row-like field repetition.
    DatabaseRow,
    /// Same substring repeated ≥3 times.
    PureRepetition,
    /// No structural catalyst detected.
    None,
}

impl std::fmt::Display for CatalystPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::XmlRepetition => write!(f, "xml"),
            Self::CodeSignature => write!(f, "code"),
            Self::LatexFormula => write!(f, "latex"),
            Self::DatabaseRow => write!(f, "db_row"),
            Self::PureRepetition => write!(f, "pure_rep"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Configuration for influence scoring.
#[derive(Debug, Clone)]
pub struct InfluenceConfig {
    /// Fraction of top-K samples to mark as high-influence. Default: 0.1
    pub top_k_fraction: f32,
    /// Minimum catalyst score to be considered a catalyst. Default: 0.5
    pub catalyst_threshold: f32,
    /// Minimum repetition length for pure repetition detection. Default: 3
    pub min_repetition_length: usize,
}

impl Default for InfluenceConfig {
    fn default() -> Self {
        Self {
            top_k_fraction: 0.1,
            catalyst_threshold: 0.5,
            min_repetition_length: 3,
        }
    }
}
