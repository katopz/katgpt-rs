#[cfg(feature = "data_gate")]
use katgpt_core::{DataGate, GateDecision, ProposerTask, TaskType};

#[test]
#[cfg(feature = "data_gate")]
fn test_gate_decision_variants() {
    let admit = GateDecision::Admit;
    let reject = GateDecision::Reject("bad task".to_string());
    assert!(admit != reject);
}

#[test]
#[cfg(feature = "data_gate")]
fn test_proposer_task_construction() {
    let task = ProposerTask {
        id: 1,
        query: "test".to_string(),
        program: Some("print(1)".to_string()),
        program_input: None,
        task_type: TaskType::CodeIO,
    };
    assert_eq!(task.id, 1);
    assert_eq!(task.task_type, TaskType::CodeIO);
}

#[test]
#[cfg(feature = "data_gate")]
fn test_task_type_variants() {
    assert_ne!(TaskType::CodeIO, TaskType::GameAction);
    assert_ne!(TaskType::DslExpr, TaskType::OpenEnded);
}

// Implement a simple always-admit gate for testing
struct AlwaysAdmitGate;

#[cfg(feature = "data_gate")]
impl DataGate for AlwaysAdmitGate {
    fn admit(&self, _task: &ProposerTask) -> GateDecision {
        GateDecision::Admit
    }
    fn leak_rate(&self) -> f32 {
        1.0
    }
}

#[test]
#[cfg(feature = "data_gate")]
fn test_always_admit_gate() {
    let gate = AlwaysAdmitGate;
    let task = ProposerTask {
        id: 0,
        query: String::new(),
        program: None,
        program_input: None,
        task_type: TaskType::OpenEnded,
    };
    assert_eq!(gate.admit(&task), GateDecision::Admit);
    assert_eq!(gate.leak_rate(), 1.0);
}

// Implement a strict gate for testing
struct StrictGate;

#[cfg(feature = "data_gate")]
impl DataGate for StrictGate {
    fn admit(&self, _task: &ProposerTask) -> GateDecision {
        GateDecision::Reject("strict".to_string())
    }
    fn leak_rate(&self) -> f32 {
        0.0
    }
}

#[test]
#[cfg(feature = "data_gate")]
fn test_strict_gate() {
    let gate = StrictGate;
    let task = ProposerTask {
        id: 0,
        query: String::new(),
        program: None,
        program_input: None,
        task_type: TaskType::OpenEnded,
    };
    assert!(matches!(gate.admit(&task), GateDecision::Reject(_)));
    assert_eq!(gate.leak_rate(), 0.0);
}
