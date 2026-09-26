//! approval/mod.rs
//!
//! Multi-tiered policy approval gates for TARA Dynamic Runtime.
//! Categorizes operations into Low-risk (auto), Medium-risk (automatic security check),
//! and High-risk (requires explicit authority approval ticket).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalTicket {
    pub ticket_id: String,
    pub action_name: String,
    pub target_resource: String,
    pub risk_level: RiskLevel,
    pub requester_id: String,
    pub approved_by: Option<String>,
    pub created_at_ms: u64,
    pub is_approved: bool,
    pub justification: String,
}

pub struct ApprovalGate {
    pending_tickets: HashMap<String, ApprovalTicket>,
}

impl ApprovalGate {
    pub fn new() -> Self {
        Self {
            pending_tickets: HashMap::new(),
        }
    }

    /// Evaluates the risk level of an action.
    pub fn classify_risk(action: &str, target_resource: &str) -> RiskLevel {
        let act = action.to_lowercase();
        let target = target_resource.to_lowercase();

        if act.contains("delete") || act.contains("wipe") || target.contains("vault") || target.contains("identity") {
            RiskLevel::Critical
        } else if act.contains("network_write") || act.contains("external_connect") || act.contains("host_tool") {
            RiskLevel::High
        } else if act.contains("filesystem_write") || act.contains("spawn_process") || act.contains("team_create") {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Determines if execution can proceed immediately, or requires security check / approval ticket.
    pub fn evaluate_execution(
        &mut self,
        requester_id: &str,
        action: &str,
        target_resource: &str,
    ) -> Result<Option<String>, String> {
        let risk = Self::classify_risk(action, target_resource);

        match risk {
            RiskLevel::Low => {
                // Low-risk: execute immediately without approval
                Ok(None)
            }
            RiskLevel::Medium => {
                // Medium-risk: automatic security check passed
                Ok(None)
            }
            RiskLevel::High | RiskLevel::Critical => {
                // High-risk: issue pending approval ticket requiring explicit creator/operator sign-off
                let now_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                let ticket_id = format!("appr_{:x}", rand::random::<u64>());
                let ticket = ApprovalTicket {
                    ticket_id: ticket_id.clone(),
                    action_name: action.to_string(),
                    target_resource: target_resource.to_string(),
                    risk_level: risk,
                    requester_id: requester_id.to_string(),
                    approved_by: None,
                    created_at_ms: now_ms,
                    is_approved: false,
                    justification: format!("High-risk operation '{}' on '{}'", action, target_resource),
                };

                self.pending_tickets.insert(ticket_id.clone(), ticket);
                Err(format!("ApprovalRequired: Action '{}' is {:?} risk. Created ApprovalTicket '{}'", action, risk, ticket_id))
            }
        }
    }

    pub fn grant_approval(&mut self, ticket_id: &str, approver_id: &str) -> Result<(), String> {
        let ticket = self.pending_tickets.get_mut(ticket_id)
            .ok_or_else(|| format!("ApprovalTicket '{}' not found", ticket_id))?;
        ticket.is_approved = true;
        ticket.approved_by = Some(approver_id.to_string());
        Ok(())
    }

    pub fn is_approved(&self, ticket_id: &str) -> bool {
        self.pending_tickets.get(ticket_id).map(|t| t.is_approved).unwrap_or(false)
    }
}
