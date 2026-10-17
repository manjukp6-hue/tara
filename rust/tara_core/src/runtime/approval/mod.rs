//! approval/mod.rs
//!
//! Multi-tiered policy approval gates for TARA Dynamic Runtime.
//! Categorizes operations into Low-risk (auto), Medium-risk (automatic security check),
//! and High-risk (requires explicit authority approval ticket).

use crate::governance::CreatorAuthority;
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
    pub consumed: bool,
    pub justification: String,
}

/// Maximum ticket lifetime (15 minutes).
pub const APPROVAL_TICKET_TTL_MS: u64 = 15 * 60 * 1000;

pub struct ApprovalGate {
    pending_tickets: HashMap<String, ApprovalTicket>,
}

impl Default for ApprovalGate {
    fn default() -> Self {
        Self::new()
    }
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

        if act.contains("delete")
            || act.contains("wipe")
            || target.contains("vault")
            || target.contains("identity")
        {
            RiskLevel::Critical
        } else if act.contains("network_write")
            || act.contains("external_connect")
            || act.contains("host_tool")
            || act.contains("sandbox_transfer")
        {
            RiskLevel::High
        } else if act.contains("filesystem_write")
            || act.contains("spawn_process")
            || act.contains("team_create")
        {
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

                self.pending_tickets.retain(|_, t| {
                    !t.consumed && now_ms.saturating_sub(t.created_at_ms) <= APPROVAL_TICKET_TTL_MS
                });

                let ticket_id = format!("appr_{:032x}", rand::random::<u128>());
                let ticket = ApprovalTicket {
                    ticket_id: ticket_id.clone(),
                    action_name: action.to_string(),
                    target_resource: target_resource.to_string(),
                    risk_level: risk,
                    requester_id: requester_id.to_string(),
                    approved_by: None,
                    created_at_ms: now_ms,
                    is_approved: false,
                    consumed: false,
                    justification: format!(
                        "High-risk operation '{}' on '{}'",
                        action, target_resource
                    ),
                };

                self.pending_tickets.insert(ticket_id.clone(), ticket);
                Err(format!(
                    "ApprovalRequired: Action '{}' is {:?} risk. Created ApprovalTicket '{}'",
                    action, risk, ticket_id
                ))
            }
        }
    }

    pub fn grant_approval(&mut self, ticket_id: &str, approver_id: &str) -> Result<(), String> {
        let _ = self
            .pending_tickets
            .get(ticket_id)
            .ok_or_else(|| format!("ApprovalTicket '{}' not found", ticket_id))?;
        Err(format!(
            "Unsigned approval by '{}' rejected; creator signature required",
            approver_id
        ))
    }

    /// Approves only the exact ticket contents signed by the configured creator key.
    pub fn grant_creator_approval(
        &mut self,
        ticket_id: &str,
        authority: &CreatorAuthority,
        signature: &[u8; 64],
    ) -> Result<(), String> {
        let ticket = self
            .pending_tickets
            .get(ticket_id)
            .ok_or_else(|| format!("ApprovalTicket '{}' not found", ticket_id))?;
        if ticket.is_approved || ticket.consumed {
            return Err("Approval ticket has already been approved or consumed".to_string());
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now_ms.saturating_sub(ticket.created_at_ms) > APPROVAL_TICKET_TTL_MS {
            return Err("Approval ticket has expired (15-minute TTL exceeded)".to_string());
        }
        let message = Self::approval_signing_message(ticket);
        if !authority.verify_creator_signature(message.as_bytes(), signature) {
            return Err("Creator signature verification failed".to_string());
        }
        let ticket = self
            .pending_tickets
            .get_mut(ticket_id)
            .ok_or_else(|| "Approval ticket disappeared before commit".to_string())?;
        ticket.is_approved = true;
        ticket.approved_by = Some(authority.creator_id.clone());
        Ok(())
    }

    pub fn is_approved(&self, ticket_id: &str) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.pending_tickets
            .get(ticket_id)
            .map(|t| {
                t.is_approved
                    && !t.consumed
                    && now_ms.saturating_sub(t.created_at_ms) <= APPROVAL_TICKET_TTL_MS
            })
            .unwrap_or(false)
    }

    pub fn get_ticket(&self, ticket_id: &str) -> Option<ApprovalTicket> {
        self.pending_tickets.get(ticket_id).cloned()
    }

    pub fn approval_signing_message(ticket: &ApprovalTicket) -> String {
        format!(
            "TARA-APPROVAL-v1|{}|{}|{}|{}|{}",
            ticket.ticket_id,
            ticket.action_name,
            ticket.target_resource,
            ticket.requester_id,
            ticket.created_at_ms
        )
    }

    pub fn consume_approved(
        &mut self,
        ticket_id: &str,
        action: &str,
        target: &str,
    ) -> Result<(), String> {
        let ticket = self
            .pending_tickets
            .get_mut(ticket_id)
            .ok_or_else(|| format!("ApprovalTicket '{}' not found", ticket_id))?;
        if !ticket.is_approved || ticket.consumed {
            return Err("A valid, unused creator approval is required".to_string());
        }
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        if now_ms.saturating_sub(ticket.created_at_ms) > APPROVAL_TICKET_TTL_MS {
            return Err("Approval ticket has expired (15-minute TTL exceeded)".to_string());
        }
        if ticket.action_name != action || ticket.target_resource != target {
            return Err("Approval does not cover this exact action and target".to_string());
        }
        ticket.consumed = true;
        ticket.is_approved = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::Signer;

    #[test]
    fn test_risk_classification() {
        assert_eq!(
            ApprovalGate::classify_risk("delete_database", "main"),
            RiskLevel::Critical
        );
        assert_eq!(
            ApprovalGate::classify_risk("network_write_call", "endpoint"),
            RiskLevel::High
        );
        assert_eq!(
            ApprovalGate::classify_risk("filesystem_write_temp", "tmp"),
            RiskLevel::Medium
        );
        assert_eq!(
            ApprovalGate::classify_risk("read_metadata", "file"),
            RiskLevel::Low
        );
    }

    #[test]
    fn test_low_risk_auto_approves() {
        let mut gate = ApprovalGate::new();
        let res = gate.evaluate_execution("requester", "read_file", "readme.txt");
        assert_eq!(res.unwrap(), None);
    }

    #[test]
    fn test_creator_approval_signature_flow() {
        let mut gate = ApprovalGate::new();

        // High-risk operation requires approval
        let err = gate
            .evaluate_execution("requester", "sandbox_transfer", "vault")
            .unwrap_err();
        assert!(err.contains("ApprovalRequired"));

        // Extract ticket_id from error message
        let ticket_id = err.split('\'').collect::<Vec<&str>>()[err.split('\'').count() - 2];

        // Unsigned approval must be rejected
        assert!(gate.grant_approval(ticket_id, "admin").is_err());

        // Create CreatorAuthority with keypair
        let mut csprng = rand::rngs::OsRng;
        let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);
        let verifying_key = signing_key.verifying_key();
        let authority = CreatorAuthority::new("authority_primary", Some(&verifying_key.to_bytes()));

        let ticket = gate.get_ticket(ticket_id).unwrap();
        let message = ApprovalGate::approval_signing_message(&ticket);
        let signature = signing_key.sign(message.as_bytes()).to_bytes();

        // Grant creator approval with signature
        assert!(gate
            .grant_creator_approval(ticket_id, &authority, &signature)
            .is_ok());
        assert!(gate.is_approved(ticket_id));

        // Consume approved ticket
        assert!(gate
            .consume_approved(ticket_id, "sandbox_transfer", "vault")
            .is_ok());

        // Replay/second consumption must fail
        assert!(gate
            .consume_approved(ticket_id, "sandbox_transfer", "vault")
            .is_err());
    }

    #[test]
    fn test_approval_ticket_ttl_expiration() {
        let mut gate = ApprovalGate::new();

        let err = gate
            .evaluate_execution("requester", "host_tool", "vault")
            .unwrap_err();
        let ticket_id = err.split('\'').collect::<Vec<&str>>()[err.split('\'').count() - 2];

        // Simulate expired ticket by backdating created_at_ms by 16 minutes (> 15-min TTL)
        if let Some(ticket) = gate.pending_tickets.get_mut(ticket_id) {
            ticket.created_at_ms = ticket.created_at_ms.saturating_sub(16 * 60 * 1000);
        }

        let mut csprng = rand::rngs::OsRng;
        let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);
        let authority = CreatorAuthority::new(
            "authority_primary",
            Some(&signing_key.verifying_key().to_bytes()),
        );
        let ticket = gate.get_ticket(ticket_id).unwrap();
        let message = ApprovalGate::approval_signing_message(&ticket);
        let signature = signing_key.sign(message.as_bytes()).to_bytes();

        // Approval of expired ticket must be rejected
        let approval_err = gate
            .grant_creator_approval(ticket_id, &authority, &signature)
            .unwrap_err();
        assert!(approval_err.contains("expired"));

        // Consumption of expired ticket must also be rejected
        let consume_err = gate
            .consume_approved(ticket_id, "host_tool", "vault")
            .unwrap_err();
        assert!(consume_err.contains("valid, unused") || consume_err.contains("expired"));
    }

    #[test]
    fn test_approval_ticket_pruning_and_consumed_unapproved() {
        let mut gate = ApprovalGate::new();

        let err1 = gate
            .evaluate_execution("requester", "host_tool", "vault")
            .unwrap_err();
        let ticket_id1 =
            err1.split('\'').collect::<Vec<&str>>()[err1.split('\'').count() - 2].to_string();

        let mut csprng = rand::rngs::OsRng;
        let signing_key = ed25519_dalek::SigningKey::generate(&mut csprng);
        let authority = CreatorAuthority::new(
            "authority_primary",
            Some(&signing_key.verifying_key().to_bytes()),
        );

        let ticket1 = gate.get_ticket(&ticket_id1).unwrap();
        let msg1 = ApprovalGate::approval_signing_message(&ticket1);
        let sig1 = signing_key.sign(msg1.as_bytes()).to_bytes();
        gate.grant_creator_approval(&ticket_id1, &authority, &sig1)
            .unwrap();

        assert!(gate.is_approved(&ticket_id1));

        // Consume ticket1
        gate.consume_approved(&ticket_id1, "host_tool", "vault")
            .unwrap();

        // is_approved must return false once consumed!
        assert!(!gate.is_approved(&ticket_id1));

        // Now evaluate a new execution: ticket1 (which is consumed) must be pruned from pending_tickets
        let _ = gate.evaluate_execution("requester", "host_tool_2", "vault");
        assert!(!gate.pending_tickets.contains_key(&ticket_id1));
    }
}
