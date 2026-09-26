//! Planner: goal tracking and multi-step task planning.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct PlanStep {
    pub step_id: String,
    pub action_intent: String,
    pub target_name: String,
    pub description: String,
    pub parameters: Value,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct Goal {
    pub goal_id: String,
    pub actor_id: String,
    pub objective: String,
    pub steps: Vec<PlanStep>,
    pub status: String,
    pub step_results: Vec<Value>,
}

impl Goal {
    pub fn to_dict(&self) -> Value {
        json!({
            "goal_id": self.goal_id,
            "actor_id": self.actor_id,
            "objective": self.objective,
            "status": self.status,
            "steps_count": self.steps.len(),
            "step_results": self.step_results
        })
    }

    pub fn record_step_result(&mut self, step_id: &str, result: Value, ok: bool) {
        self.step_results.push(json!({ "step_id": step_id, "result": result, "ok": ok }));
    }

    pub fn mark_failed(&mut self, reason: &str) {
        self.status = "FAILED".to_string();
        self.step_results.push(json!({ "failure_reason": reason }));
    }
}

pub struct GoalTracker {
    goals: Arc<Mutex<Vec<Goal>>>,
}

impl GoalTracker {
    pub fn new() -> Self {
        Self { goals: Arc::new(Mutex::new(Vec::new())) }
    }

    pub fn create_goal(&self, actor_id: &str, objective: &str, steps: Vec<PlanStep>) -> Goal {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();
        let goal = Goal {
            goal_id: format!("goal_{}", ts),
            actor_id: actor_id.to_string(),
            objective: objective.to_string(),
            steps,
            status: "ACTIVE".to_string(),
            step_results: Vec::new(),
        };
        self.goals.lock().unwrap().push(goal.clone());
        goal
    }

    pub fn get_goal(&self, goal_id: &str) -> Option<Goal> {
        self.goals.lock().unwrap().iter().find(|g| g.goal_id == goal_id).cloned()
    }
}

/// Plans multi-step composite tasks from user input.
pub struct TaskPlanner;

impl TaskPlanner {
    pub fn plan_composite_task(
        input: &str,
        slots: &HashMap<String, String>,
        _context: &HashMap<String, Value>,
    ) -> Vec<PlanStep> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis();

        // Parse "step N: <action>" patterns from input
        let mut steps = Vec::new();
        for (i, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let (intent, target) = if line.to_lowercase().contains("execute") || line.to_lowercase().contains("run") {
                ("EXECUTE_TOOL".to_string(), "unknown_tool".to_string())
            } else if line.to_lowercase().contains("skill") {
                ("EXECUTE_SKILL".to_string(), "unknown_skill".to_string())
            } else {
                ("CONVERSATIONAL".to_string(), String::new())
            };
            steps.push(PlanStep {
                step_id: format!("step_{}_{}", ts, i),
                action_intent: intent,
                target_name: target,
                description: line.to_string(),
                parameters: json!(slots),
                status: "PENDING".to_string(),
            });
        }

        if steps.is_empty() {
            steps.push(PlanStep {
                step_id: format!("step_{}", ts),
                action_intent: "CONVERSATIONAL".to_string(),
                target_name: String::new(),
                description: input.to_string(),
                parameters: json!({}),
                status: "PENDING".to_string(),
            });
        }

        steps
    }
}
