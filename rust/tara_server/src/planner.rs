//! Planner: goal tracking, multi-step task planning, and durable goal persistence.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalOrigin {
    UserRequest {
        actor_id: String,
    },
    AutonomousInternal {
        generator: String,
        trigger_reason: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub step_id: String,
    pub action_intent: String,
    pub target_name: String,
    pub description: String,
    pub parameters: Value,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub goal_id: String,
    pub actor_id: String,
    pub objective: String,
    pub steps: Vec<PlanStep>,
    pub status: String, // "ACTIVE", "IN_PROGRESS", "COMPLETED", "FAILED"
    pub priority: f64,  // [0.0, 1.0] (1.0 = highest)
    pub origin: GoalOrigin,
    pub step_results: Vec<Value>,
    pub created_at_ms: u64,
    pub completed_at_ms: Option<u64>,
}

impl Goal {
    pub fn to_dict(&self) -> Value {
        json!({
            "goal_id": self.goal_id,
            "actor_id": self.actor_id,
            "objective": self.objective,
            "status": self.status,
            "priority": self.priority,
            "origin": self.origin,
            "steps_count": self.steps.len(),
            "step_results": self.step_results,
            "created_at_ms": self.created_at_ms,
            "completed_at_ms": self.completed_at_ms
        })
    }

    pub fn record_step_result(&mut self, step_id: &str, result: Value, ok: bool) {
        self.step_results
            .push(json!({ "step_id": step_id, "result": result, "ok": ok }));
    }

    pub fn mark_completed(&mut self) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.status = "COMPLETED".to_string();
        self.completed_at_ms = Some(ts);
    }

    pub fn mark_failed(&mut self, reason: &str) {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.status = "FAILED".to_string();
        self.completed_at_ms = Some(ts);
        self.step_results.push(json!({ "failure_reason": reason }));
    }
}

pub struct GoalTracker {
    storage_path: PathBuf,
    goals: Arc<Mutex<Vec<Goal>>>,
}

impl Default for GoalTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl GoalTracker {
    pub fn new() -> Self {
        Self::with_storage("storage/persistence/goals.jsonl")
    }

    pub fn with_storage<P: AsRef<Path>>(path: P) -> Self {
        let storage_path = path.as_ref().to_path_buf();
        if let Some(parent) = storage_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let mut goals = Vec::new();
        if storage_path.exists() {
            if let Ok(content) = fs::read_to_string(&storage_path) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if !trimmed.is_empty() {
                        if let Ok(g) = serde_json::from_str::<Goal>(trimmed) {
                            goals.push(g);
                        }
                    }
                }
            }
        }

        Self {
            storage_path,
            goals: Arc::new(Mutex::new(goals)),
        }
    }

    fn current_ts_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Create standard user goal (maintains backward compatibility).
    pub fn create_goal(&self, actor_id: &str, objective: &str, steps: Vec<PlanStep>) -> Goal {
        self.create_prioritized_goal(
            actor_id,
            objective,
            steps,
            0.5,
            GoalOrigin::UserRequest {
                actor_id: actor_id.to_string(),
            },
        )
    }

    /// Create goal with explicit priority and origin tracking (User vs Autonomous).
    pub fn create_prioritized_goal(
        &self,
        actor_id: &str,
        objective: &str,
        steps: Vec<PlanStep>,
        priority: f64,
        origin: GoalOrigin,
    ) -> Goal {
        let ts = Self::current_ts_ms();
        let goal = Goal {
            goal_id: format!(
                "goal_{}_{}",
                ts,
                actor_id.replace(|c: char| !c.is_alphanumeric(), "_")
            ),
            actor_id: actor_id.to_string(),
            objective: objective.to_string(),
            steps,
            status: "ACTIVE".to_string(),
            priority: priority.clamp(0.0, 1.0),
            origin,
            step_results: Vec::new(),
            created_at_ms: ts,
            completed_at_ms: None,
        };

        {
            let mut g_list = self.goals.lock().unwrap();
            g_list.push(goal.clone());
        }
        let _ = self.persist_goal(&goal);
        goal
    }

    /// Retrieve active goals sorted by descending priority.
    pub fn get_prioritized_active_goals(&self) -> Vec<Goal> {
        let goals = self.goals.lock().unwrap();
        let mut active: Vec<Goal> = goals
            .iter()
            .filter(|g| g.status == "ACTIVE" || g.status == "IN_PROGRESS")
            .cloned()
            .collect();
        active.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        active
    }

    pub fn get_goal(&self, goal_id: &str) -> Option<Goal> {
        self.goals
            .lock()
            .unwrap()
            .iter()
            .find(|g| g.goal_id == goal_id)
            .cloned()
    }

    pub fn update_goal_status(&self, goal_id: &str, new_status: &str) -> bool {
        let mut goals = self.goals.lock().unwrap();
        if let Some(g) = goals.iter_mut().find(|g| g.goal_id == goal_id) {
            g.status = new_status.to_string();
            if new_status == "COMPLETED" || new_status == "FAILED" {
                g.completed_at_ms = Some(Self::current_ts_ms());
            }
            drop(goals);
            let _ = self.rewrite_all_goals();
            true
        } else {
            false
        }
    }

    fn persist_goal(&self, goal: &Goal) -> Result<(), String> {
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.storage_path)
        {
            if let Ok(line) = serde_json::to_string(goal) {
                let _ = writeln!(f, "{}", line);
            }
        }
        Ok(())
    }

    fn rewrite_all_goals(&self) -> Result<(), String> {
        let goals = self.goals.lock().unwrap();
        let mut content = String::new();
        for g in goals.iter() {
            if let Ok(line) = serde_json::to_string(g) {
                content.push_str(&line);
                content.push('\n');
            }
        }
        fs::write(&self.storage_path, content).map_err(|e| e.to_string())
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
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();

        // Parse steps from input lines
        let mut steps = Vec::new();
        let lines: Vec<&str> = input
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .collect();

        for (i, raw_line) in lines.iter().enumerate() {
            let mut line = *raw_line;
            let lower_line = line.to_lowercase();
            if let Some(rest) = line.strip_prefix(|c: char| c.is_ascii_digit()) {
                let rest = rest.trim_start_matches(['.', ')', ':', ' ']);
                line = rest.trim();
            } else if lower_line.starts_with("step") {
                let rest = &line[4..];
                let rest = rest.trim_start_matches(|c: char| {
                    c.is_ascii_digit() || c == ':' || c == '.' || c == ' '
                });
                line = rest.trim();
            } else if line.starts_with('-') || line.starts_with('*') {
                line = line[1..].trim();
            }

            let mut step_params = slots.clone();
            // Parse inline key=val or key="val" from the line
            for token in line.split_whitespace() {
                if let Some((k, v)) = token.split_once('=') {
                    let clean_k = k.trim().trim_start_matches('-');
                    let clean_v = v.trim().trim_matches('"').trim_matches('\'');
                    if !clean_k.is_empty() && !clean_v.is_empty() {
                        step_params.insert(clean_k.to_string(), clean_v.to_string());
                    }
                }
            }

            let lower = line.to_lowercase();
            let mut target = String::new();
            let mut intent = "CONVERSATIONAL".to_string();

            // Check slots or context for explicit tool/skill designation
            if let Some(t) = step_params
                .get("tool")
                .or_else(|| step_params.get("tool_name"))
            {
                intent = "EXECUTE_TOOL".to_string();
                target = t.clone();
            } else if let Some(s) = step_params
                .get("skill")
                .or_else(|| step_params.get("skill_name"))
            {
                intent = "EXECUTE_SKILL".to_string();
                target = s.clone();
            } else if lower.contains("tool") || lower.contains("execute") || lower.contains("run") {
                // Determine if it's a skill or tool
                let is_skill = lower.contains("skill");
                intent = if is_skill {
                    "EXECUTE_SKILL".to_string()
                } else {
                    "EXECUTE_TOOL".to_string()
                };

                // Dynamic extraction of target name
                let words: Vec<&str> = line.split_whitespace().collect();
                for (w_idx, &word) in words.iter().enumerate() {
                    let w_clean = word.to_lowercase();
                    let w_clean = w_clean
                        .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.');
                    if (w_clean == "tool"
                        || w_clean == "skill"
                        || w_clean == "execute"
                        || w_clean == "run")
                        && w_idx + 1 < words.len()
                    {
                        let next_word = words[w_idx + 1]
                            .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '.');
                        let next_lower = next_word.to_lowercase();
                        if next_lower != "tool"
                            && next_lower != "skill"
                            && next_lower != "a"
                            && next_lower != "the"
                            && !next_word.contains('=')
                        {
                            target = next_word.to_string();
                            break;
                        }
                    }
                }

                // If target not resolved from keyword following, check for built-in or known names
                if target.is_empty() {
                    for candidate in &[
                        "file_inspector",
                        "hash_verifier",
                        "knowledge_retriever",
                        "provenance_tracker",
                    ] {
                        if lower.contains(candidate) {
                            target = candidate.to_string();
                            intent = "EXECUTE_TOOL".to_string();
                            break;
                        }
                    }
                }
            } else {
                // Check if any built-in tool is mentioned directly
                for candidate in &[
                    "file_inspector",
                    "hash_verifier",
                    "knowledge_retriever",
                    "provenance_tracker",
                ] {
                    if lower.contains(candidate) {
                        target = candidate.to_string();
                        intent = "EXECUTE_TOOL".to_string();
                        break;
                    }
                }
            }

            steps.push(PlanStep {
                step_id: format!("step_{}_{}", ts, i),
                action_intent: intent,
                target_name: target,
                description: raw_line.to_string(),
                parameters: json!(step_params),
                status: "PENDING".to_string(),
            });
        }

        if steps.is_empty() {
            steps.push(PlanStep {
                step_id: format!("step_{}", ts),
                action_intent: "CONVERSATIONAL".to_string(),
                target_name: String::new(),
                description: input.to_string(),
                parameters: json!(slots),
                status: "PENDING".to_string(),
            });
        }

        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goal_tracking_prioritization_and_origin() {
        let tmp = std::env::temp_dir().join(format!("tara_goals_test_{}", uuid::Uuid::new_v4()));
        let tracker = GoalTracker::with_storage(tmp.join("goals.jsonl"));

        let step1 = PlanStep {
            step_id: "s1".into(),
            action_intent: "EXECUTE_TOOL".into(),
            target_name: "math_engine".into(),
            description: "Solve ODE".into(),
            parameters: json!({}),
            status: "PENDING".into(),
        };

        // User goal (Priority 0.6)
        let g_user = tracker.create_prioritized_goal(
            "operator",
            "Calculate planetary orbit",
            vec![step1.clone()],
            0.6,
            GoalOrigin::UserRequest {
                actor_id: "operator".into(),
            },
        );

        // Autonomous curiosity goal (Priority 0.9)
        let g_auto = tracker.create_prioritized_goal(
            "tara_curiosity",
            "Explore quantum chromodynamics gap",
            vec![step1],
            0.9,
            GoalOrigin::AutonomousInternal {
                generator: "CuriosityDriveEngine".into(),
                trigger_reason: "High epistemic entropy in quarks domain".into(),
            },
        );

        // Verify prioritized active goals ordering (0.9 > 0.6)
        let active = tracker.get_prioritized_active_goals();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].goal_id, g_auto.goal_id);
        assert_eq!(active[1].goal_id, g_user.goal_id);

        // Mark auto goal completed
        tracker.update_goal_status(&g_auto.goal_id, "COMPLETED");
        let active_after = tracker.get_prioritized_active_goals();
        assert_eq!(active_after.len(), 1);
        assert_eq!(active_after[0].goal_id, g_user.goal_id);

        // Verify reload from disk
        let tracker_reloaded = GoalTracker::with_storage(tmp.join("goals.jsonl"));
        let reloaded_auto = tracker_reloaded.get_goal(&g_auto.goal_id).unwrap();
        assert_eq!(reloaded_auto.status, "COMPLETED");
        assert!(reloaded_auto.completed_at_ms.is_some());

        let _ = fs::remove_dir_all(&tmp);
    }
}
