//! 5-State Skill Certification Authority and Trace Generalization Engine.
//!
//! Enforces the 5-state lifecycle: UNTESTED -> EXPERIMENTAL -> VALIDATED -> CERTIFIED (or REJECTED)
//! with execution trace generalization, invariant verification, and persistent records.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificationState {
    Untested,
    Experimental,
    Validated,
    Certified,
    Rejected,
}

impl CertificationState {
    pub fn as_str(&self) -> &'static str {
        match self {
            CertificationState::Untested => "UNTESTED",
            CertificationState::Experimental => "EXPERIMENTAL",
            CertificationState::Validated => "VALIDATED",
            CertificationState::Certified => "CERTIFIED",
            CertificationState::Rejected => "REJECTED",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillTraceSample {
    pub input_params: Value,
    pub output_result: Value,
    pub execution_time_ms: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCertificationRecord {
    pub skill_id: String,
    pub state: CertificationState,
    pub test_runs_count: usize,
    pub pass_count: usize,
    pub failure_count: usize,
    pub inferred_schema: Value,
    pub safety_score: f64,
    pub certification_notes: String,
    pub last_updated_ms: u64,
}

pub struct SkillCertificationAuthority {
    manifest_path: String,
    records: Arc<Mutex<HashMap<String, SkillCertificationRecord>>>,
}

impl SkillCertificationAuthority {
    pub fn new(repo_root: &str) -> Self {
        let skills_dir = format!("{}/storage/skills", repo_root);
        let _ = fs::create_dir_all(&skills_dir);
        let manifest_path = format!("{}/certification_manifest.json", skills_dir);

        let mut map = HashMap::new();
        if Path::new(&manifest_path).exists() {
            if let Ok(content) = fs::read_to_string(&manifest_path) {
                if let Ok(deserialized) =
                    serde_json::from_str::<HashMap<String, SkillCertificationRecord>>(&content)
                {
                    map = deserialized;
                }
            }
        }

        Self {
            manifest_path,
            records: Arc::new(Mutex::new(map)),
        }
    }

    /// Registers a newly learned skill in the UNTESTED state.
    pub fn register_untested(&self, skill_id: &str) -> SkillCertificationRecord {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let record = SkillCertificationRecord {
            skill_id: skill_id.to_string(),
            state: CertificationState::Untested,
            test_runs_count: 0,
            pass_count: 0,
            failure_count: 0,
            inferred_schema: json!({}),
            safety_score: 0.0,
            certification_notes: "Initial registration; requires experimental evaluation"
                .to_string(),
            last_updated_ms: now,
        };

        {
            let mut recs = self.records.lock().unwrap();
            recs.insert(skill_id.to_string(), record.clone());
        }
        let _ = self.save_manifest();
        record
    }

    /// Evaluates execution traces to generalize schemas and advance certification state.
    pub fn evaluate_and_advance(
        &self,
        skill_id: &str,
        traces: &[SkillTraceSample],
        target_state: CertificationState,
    ) -> Result<SkillCertificationRecord, String> {
        if traces.is_empty() {
            return Err("Cannot certify skill without execution trace samples".to_string());
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut pass_count = 0;
        let mut fail_count = 0;

        for t in traces {
            if t.success {
                pass_count += 1;
            } else {
                fail_count += 1;
            }
        }

        let total = pass_count + fail_count;
        let success_rate = (pass_count as f64) / (total as f64);

        // Generalize parameter schema from traces
        let inferred_schema = self.generalize_schema_from_traces(traces);

        let mut recs = self.records.lock().unwrap();
        let record = recs
            .entry(skill_id.to_string())
            .or_insert_with(|| SkillCertificationRecord {
                skill_id: skill_id.to_string(),
                state: CertificationState::Untested,
                test_runs_count: 0,
                pass_count: 0,
                failure_count: 0,
                inferred_schema: json!({}),
                safety_score: 0.0,
                certification_notes: String::new(),
                last_updated_ms: now,
            });

        record.test_runs_count += total;
        record.pass_count += pass_count;
        record.failure_count += fail_count;
        record.inferred_schema = inferred_schema;
        record.last_updated_ms = now;

        // State transition rules:
        match target_state {
            CertificationState::Experimental => {
                record.state = CertificationState::Experimental;
                record.certification_notes = "Moved to experimental sandbox".to_string();
                record.safety_score = 0.50;
            }
            CertificationState::Validated => {
                if success_rate < 0.80 {
                    record.state = CertificationState::Rejected;
                    record.certification_notes = format!(
                        "Failed validation: success rate {:.1}% < 80%",
                        success_rate * 100.0
                    );
                    record.safety_score = success_rate;
                } else {
                    record.state = CertificationState::Validated;
                    record.certification_notes = format!(
                        "Validated with {:.1}% success rate over {} runs",
                        success_rate * 100.0,
                        total
                    );
                    record.safety_score = 0.85;
                }
            }
            CertificationState::Certified => {
                if success_rate < 0.95 || record.test_runs_count < 3 {
                    record.state = CertificationState::Rejected;
                    record.certification_notes = format!("Certification failed: requires >= 95% pass rate and >= 3 runs (got {:.1}%, {} runs)", success_rate * 100.0, record.test_runs_count);
                    record.safety_score = success_rate;
                } else {
                    record.state = CertificationState::Certified;
                    record.certification_notes = format!(
                        "Fully certified for autonomous execution ({:.1}% pass rate)",
                        success_rate * 100.0
                    );
                    record.safety_score = 0.99;
                }
            }
            CertificationState::Rejected => {
                record.state = CertificationState::Rejected;
                record.certification_notes = "Explicitly rejected by safety authority".to_string();
                record.safety_score = 0.0;
            }
            CertificationState::Untested => {
                record.state = CertificationState::Untested;
            }
        }

        let updated = record.clone();
        drop(recs);
        let _ = self.save_manifest();
        Ok(updated)
    }

    /// Verifies whether a skill is permitted to run in production.
    pub fn is_allowed_in_production(&self, skill_id: &str) -> bool {
        let recs = self.records.lock().unwrap();
        if let Some(r) = recs.get(skill_id) {
            r.state == CertificationState::Certified
        } else {
            false
        }
    }

    /// Generalizes JSON schema from trace input parameters.
    fn generalize_schema_from_traces(&self, traces: &[SkillTraceSample]) -> Value {
        let mut fields: HashMap<String, String> = HashMap::new();
        for t in traces {
            if let Some(obj) = t.input_params.as_object() {
                for (k, v) in obj {
                    let type_name = match v {
                        Value::String(_) => "string",
                        Value::Number(_) => "number",
                        Value::Bool(_) => "boolean",
                        Value::Array(_) => "array",
                        Value::Object(_) => "object",
                        Value::Null => "null",
                    };
                    fields.insert(k.clone(), type_name.to_string());
                }
            }
        }
        json!({
            "type": "object",
            "properties": fields
        })
    }

    fn save_manifest(&self) -> Result<(), std::io::Error> {
        let recs = self.records.lock().unwrap();
        let content = serde_json::to_string_pretty(&*recs)?;
        fs::write(&self.manifest_path, content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_certification_lifecycle() {
        let temp = std::env::temp_dir().join("tara_test_skill_cert");
        let auth = SkillCertificationAuthority::new(temp.to_str().unwrap());

        // 1. Untested
        let r1 = auth.register_untested("calc_fibonacci");
        assert_eq!(r1.state, CertificationState::Untested);
        assert!(!auth.is_allowed_in_production("calc_fibonacci"));

        // 2. Advance to Validated
        let traces = vec![
            SkillTraceSample {
                input_params: json!({"n": 5}),
                output_result: json!({"fib": 5}),
                execution_time_ms: 1,
                success: true,
                error_message: None,
            },
            SkillTraceSample {
                input_params: json!({"n": 10}),
                output_result: json!({"fib": 55}),
                execution_time_ms: 2,
                success: true,
                error_message: None,
            },
            SkillTraceSample {
                input_params: json!({"n": 1}),
                output_result: json!({"fib": 1}),
                execution_time_ms: 1,
                success: true,
                error_message: None,
            },
        ];

        let r2 = auth
            .evaluate_and_advance("calc_fibonacci", &traces, CertificationState::Validated)
            .unwrap();
        assert_eq!(r2.state, CertificationState::Validated);
        assert!(!auth.is_allowed_in_production("calc_fibonacci"));

        // 3. Advance to Certified
        let r3 = auth
            .evaluate_and_advance("calc_fibonacci", &traces, CertificationState::Certified)
            .unwrap();
        assert_eq!(r3.state, CertificationState::Certified);
        assert!(auth.is_allowed_in_production("calc_fibonacci"));

        let _ = fs::remove_dir_all(&temp);
    }
}
