//! Autonomous Online Learning Engine for TARA.
//!
//! Workflow:
//! Search -> Read -> Quality check -> Multi-source verification -> Compare existing ->
//! Deduplicate -> Update/merge -> Provenance save -> TARA/KNOWLEDGE/.

use crate::knowledge::GlobalKnowledgeBase;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningDurationMode {
    Off,
    Duration1Hour,
    Duration2Hours,
    Duration6Hours,
    Duration12Hours,
    Duration24Hours,
    Loop,
}

impl LearningDurationMode {
    pub fn as_seconds(&self) -> Option<u64> {
        match self {
            Self::Off => None,
            Self::Duration1Hour => Some(3600),
            Self::Duration2Hours => Some(7200),
            Self::Duration6Hours => Some(21600),
            Self::Duration12Hours => Some(43200),
            Self::Duration24Hours => Some(86400),
            Self::Loop => None,
        }
    }

    pub fn parse_mode(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

impl std::str::FromStr for LearningDurationMode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "OFF" => Ok(Self::Off),
            "1 HOUR" | "1_HOUR" | "1H" => Ok(Self::Duration1Hour),
            "2 HOURS" | "2_HOURS" | "2H" => Ok(Self::Duration2Hours),
            "6 HOURS" | "6_HOURS" | "6H" => Ok(Self::Duration6Hours),
            "12 HOURS" | "12_HOURS" | "12H" => Ok(Self::Duration12Hours),
            "24 HOURS" | "24_HOURS" | "24H" => Ok(Self::Duration24Hours),
            "LOOP" => Ok(Self::Loop),
            _ => Err(()),
        }
    }
}

pub struct AutonomousOnlineLearner {
    pub repo_root: String,
    pub mode: LearningDurationMode,
    pub is_running: bool,
    pub start_time: Option<Instant>,
    pub target_duration_seconds: Option<u64>,
    pub cycle_count: usize,
    pub items_learned: usize,
}

impl Default for AutonomousOnlineLearner {
    fn default() -> Self {
        Self::new(".")
    }
}

impl AutonomousOnlineLearner {
    pub fn new(repo_root: &str) -> Self {
        Self {
            repo_root: repo_root.to_string(),
            mode: LearningDurationMode::Off,
            is_running: false,
            start_time: None,
            target_duration_seconds: None,
            cycle_count: 0,
            items_learned: 0,
        }
    }

    pub fn set_learning_mode(
        &mut self,
        mode_str: &str,
        is_creator: bool,
        custom_duration: Option<u64>,
    ) -> Result<Value, String> {
        if !is_creator {
            return Err(
                "Unauthorized: Only ROOT_OPERATOR can configure autonomous learning".into(),
            );
        }

        let mode: LearningDurationMode = mode_str
            .parse()
            .map_err(|_| format!("Invalid learning mode '{}'", mode_str))?;

        self.mode = mode;
        if mode == LearningDurationMode::Off {
            self.is_running = false;
            self.start_time = None;
            self.target_duration_seconds = None;
            return Ok(json!({ "status": "STOPPED", "mode": "OFF" }));
        }

        self.is_running = true;
        self.start_time = Some(Instant::now());
        self.target_duration_seconds = custom_duration.or_else(|| mode.as_seconds());

        Ok(json!({
            "status": "RUNNING",
            "mode": mode_str,
            "target_duration_seconds": self.target_duration_seconds,
            "started_at": crate::now_iso()
        }))
    }

    pub fn check_duration_and_auto_stop(&mut self) -> bool {
        if !self.is_running {
            return false;
        }

        if let (Some(start), Some(target)) = (self.start_time, self.target_duration_seconds) {
            if start.elapsed().as_secs() >= target {
                self.is_running = false;
                self.mode = LearningDurationMode::Off;
                self.start_time = None;
                self.target_duration_seconds = None;
                return true; // auto-stopped
            }
        }
        false
    }

    pub fn step_cycle(
        &mut self,
        kb: &GlobalKnowledgeBase,
        topic: &str,
        subject: &str,
        content: &str,
        confidence: f32,
    ) -> Result<Value, String> {
        if self.check_duration_and_auto_stop() {
            return Err("Learning session duration expired and automatically stopped".into());
        }
        if !self.is_running {
            return Err("Autonomous learner is OFF".into());
        }

        self.cycle_count += 1;
        let entry = kb.store_or_update_knowledge(topic, subject, content, confidence);
        self.items_learned += 1;

        Ok(json!({
            "cycle": self.cycle_count,
            "items_learned": self.items_learned,
            "stored_entry": entry
        }))
    }

    pub fn get_status(&self) -> Value {
        json!({
            "is_running": self.is_running,
            "mode": format!("{:?}", self.mode),
            "cycle_count": self.cycle_count,
            "items_learned": self.items_learned,
            "elapsed_seconds": self.start_time.map(|s| s.elapsed().as_secs()).unwrap_or(0)
        })
    }
}
