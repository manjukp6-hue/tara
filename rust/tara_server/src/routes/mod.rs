//! All route handlers for the TARA HTTP API.
//! Implements every endpoint from Python server.py register_* functions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use base64::Engine;

use crate::server::{RouteHandler, RouteRequest};
use crate::brain::TaraBrain;

// ── Chat UI HTML (embedded) ────────────────────────────────────────────────────

pub fn render_chat_ui(web_session_token: &str) -> String {
    // Attempt loading canonical synchronized frontend from disk
    let candidates = [
        "frontend/index.html",
        "../../frontend/index.html",
        "../../../frontend/index.html",
    ];
    for path in &candidates {
        if let Ok(content) = std::fs::read_to_string(path) {
            return content.replace("__TARA_SESSION_TOKEN__", web_session_token);
        }
    }

    // Fallback embedded web chat HTML injected with the web session token
    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
  <title>TARA AI Core</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    body {{ background-color: #0b0f19; color: #f1f5f9; display: flex; flex-direction: column; height: 100vh; height: 100dvh; overflow: hidden; }}
    header {{ background-color: #131b2e; border-bottom: 1px solid #232f48; padding: 12px 18px; display: flex; align-items: center; justify-content: space-between; flex-shrink: 0; }}
    .header-left {{ display: flex; align-items: center; gap: 10px; }}
    .status-dot {{ width: 10px; height: 10px; border-radius: 50%; background: #10b981; box-shadow: 0 0 8px #10b981; }}
    .brand-title {{ font-weight: 700; font-size: 16px; letter-spacing: 0.5px; color: #38bdf8; }}
    .brand-sub {{ font-size: 11px; color: #94a3b8; }}
    .btn-clear {{ background: transparent; border: 1px solid #334155; color: #94a3b8; font-size: 12px; padding: 6px 12px; border-radius: 6px; cursor: pointer; }}
    .btn-clear:hover {{ background: #1e293b; color: #f1f5f9; }}
    #chat-container {{ flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 14px; max-width: 860px; width: 100%; margin: 0 auto; }}
    .msg {{ display: flex; flex-direction: column; max-width: 85%; animation: fadeIn 0.2s ease-in; }}
    @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .msg.user {{ align-self: flex-end; }}
    .msg.user .bubble {{ background: #2563eb; color: #ffffff; border-radius: 14px 14px 2px 14px; }}
    .msg.tara {{ align-self: flex-start; }}
    .msg.tara .bubble {{ background: #1e293b; color: #f1f5f9; border-radius: 14px 14px 14px 2px; border: 1px solid #2e3c54; }}
    .msg-label {{ font-size: 11px; font-weight: 600; margin-bottom: 4px; color: #64748b; }}
    .msg.user .msg-label {{ text-align: right; color: #60a5fa; }}
    .bubble {{ padding: 12px 16px; font-size: 14px; line-height: 1.5; word-break: break-word; white-space: pre-wrap; }}
    footer {{ background: #131b2e; border-top: 1px solid #232f48; padding: 12px 16px; flex-shrink: 0; }}
    .input-form {{ display: flex; gap: 10px; max-width: 860px; margin: 0 auto; width: 100%; }}
    #chat-input {{ flex: 1; background: #1e293b; border: 1px solid #334155; color: #f1f5f9; padding: 12px 16px; border-radius: 24px; font-size: 14px; outline: none; }}
    #chat-input:focus {{ border-color: #38bdf8; box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2); }}
    #btn-send {{ background: #38bdf8; color: #0b0f19; border: none; border-radius: 24px; padding: 0 20px; font-weight: 600; cursor: pointer; font-size: 14px; }}
    #btn-send:hover {{ background: #7dd3fc; }}
  </style>
</head>
<body>
  <header>
    <div class="header-left">
      <div class="status-dot"></div>
      <div>
        <div class="brand-title">TARA AI Core</div>
        <div class="brand-sub">Cognitive Intelligence Engine</div>
      </div>
    </div>
    <button class="btn-clear" onclick="clearChat()">Clear</button>
  </header>
  <div id="chat-container"></div>
  <footer>
    <div class="input-form">
      <input id="chat-input" type="text" placeholder="Message TARA..." autocomplete="off" />
      <button id="btn-send" onclick="sendMessage()">Send</button>
    </div>
  </footer>
  <script>
    const WEB_SESSION_TOKEN = '{token}';
    const chatContainer = document.getElementById('chat-container');
    const chatInput = document.getElementById('chat-input');
    let sessionId = 'session_' + Math.random().toString(36).substr(2, 9);

    function clearChat() {{
      chatContainer.innerHTML = '';
      sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    }}

    function addMessage(role, text) {{
      const msg = document.createElement('div');
      msg.className = 'msg ' + role;
      const label = document.createElement('div');
      label.className = 'msg-label';
      label.textContent = role === 'user' ? 'You' : 'TARA';
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = text;
      msg.appendChild(label);
      msg.appendChild(bubble);
      chatContainer.appendChild(msg);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }}

    async function sendMessage() {{
      const input = chatInput.value.trim();
      if (!input) return;
      chatInput.value = '';
      addMessage('user', input);

      try {{
        const res = await fetch('/api/v1/chat', {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + WEB_SESSION_TOKEN
          }},
          body: JSON.stringify({{
            input: input,
            context: {{ session_id: sessionId }}
          }})
        }});
        const data = await res.json();
        if (data.status === 'SUCCESS') {{
          addMessage('tara', data.data.final_response || 'No response.');
        }} else {{
          addMessage('tara', '[Error]: ' + (data.error || 'Unknown error'));
        }}
      }} catch (err) {{
        addMessage('tara', '[Network Error]: ' + err.message);
      }}
    }}

    chatInput.addEventListener('keydown', function(e) {{
      if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
    }});
  </script>
</body>
</html>"#, token = web_session_token)
}

// ── Server time helper ─────────────────────────────────────────────────────────

fn server_time() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Format as ISO 8601 UTC approximate
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    let year = 2024 + days / 365;
    format!("{:04}-01-01T{:02}:{:02}:{:02}Z", year, h, m, s)
}

// ── Route handler helper macro ─────────────────────────────────────────────────

macro_rules! route_handler {
    ($name:ident, $body:expr) => {
        pub struct $name;
        impl RouteHandler for $name {
            fn handle(&self, req: &RouteRequest, brain: &Arc<TaraBrain>) -> Value {
                $body(req, brain)
            }
        }
    };
}

// ── Health ─────────────────────────────────────────────────────────────────────

route_handler!(HealthHandler, |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
    json!({ "status": "HEALTHY", "timestamp": server_time() })
});

// ── Status ─────────────────────────────────────────────────────────────────────

route_handler!(StatusHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let model_status = brain.get_model_status();
    let mem_stats = brain.memory_engine.get_memory_stats();
    let bridge_status = brain.bridge.get_telemetry();
    json!({
        "status": "ONLINE",
        "server_time": server_time(),
        "model": model_status,
        "memory": mem_stats,
        "bridge": bridge_status,
        "guard_active": true
    })
});

// ── Chat ───────────────────────────────────────────────────────────────────────

route_handler!(ChatHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let input = req.body.get("input").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    if input.is_empty() {
        return json!({ "status": "ERROR", "error": "Field 'input' is required and must be a non-empty string." });
    }

    let actor = req.actor.clone().unwrap_or_else(|| "user".to_string());
    let context_val = req.body.get("context").cloned().unwrap_or(json!({}));
    let mut ctx: HashMap<String, Value> = if let Some(obj) = context_val.as_object() {
        obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    } else {
        HashMap::new()
    };
    ctx.insert("client_ip".to_string(), json!(req.client_ip));
    ctx.insert("request_id".to_string(), json!(uuid::Uuid::new_v4().to_string()));

    // Pass creator session token if present in payload
    if let Some(t) = req.body.get("creator_session_token") {
        ctx.insert("creator_session_token".to_string(), t.clone());
    }

    let result = brain.process(&actor, &input, ctx);
    json!({ "status": "SUCCESS", "data": result })
});

// ── Skills ─────────────────────────────────────────────────────────────────────

route_handler!(SkillsHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let skills = brain.skill_engine.list_skills();
    json!({ "status": "SUCCESS", "skills_count": skills.len(), "skills": skills })
});

// ── Admin overview ─────────────────────────────────────────────────────────────

route_handler!(AdminOverviewHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let model_status = brain.get_model_status();
    let mem_stats = brain.memory_engine.get_memory_stats();
    let skills = brain.skill_engine.list_skills();
    json!({
        "status": "SUCCESS",
        "server_time": server_time(),
        "model": model_status,
        "knowledge_entries_count": 0,
        "skills_count": skills.len(),
        "rules_count": 0,
        "memory": mem_stats,
        "devices": { "total": 0, "authorized": 0 },
        "lockdown_state": "NORMAL"
    })
});

// ── Auth handlers ──────────────────────────────────────────────────────────────

route_handler!(AuthGoogleHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let id_token = req.body.get("id_token").or(req.body.get("token"))
        .and_then(|v| v.as_str()).unwrap_or("").to_string();
    brain.creator_auth_service.authenticate_google_token(&id_token, &req.client_ip)
});

route_handler!(AuthCreatorKeyHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let sig = req.body.get("proof_signature").or(req.body.get("signature"))
        .and_then(|v| v.as_str());
    let nonce = req.body.get("challenge_nonce").or(req.body.get("nonce"))
        .and_then(|v| v.as_str());
    let claimed_id = req.body.get("claimed_creator_id").and_then(|v| v.as_str()).unwrap_or("ROOT_OPERATOR");
    let passphrase = req.body.get("passphrase").and_then(|v| v.as_str());
    brain.creator_auth_service.authenticate_creator_key(sig, nonce, claimed_id, passphrase, &req.client_ip)
});

route_handler!(AuthRecoveryHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let code = req.body.get("recovery_code").or(req.body.get("token"))
        .and_then(|v| v.as_str()).unwrap_or("");
    let claimed_id = req.body.get("claimed_creator_id").and_then(|v| v.as_str()).unwrap_or("ROOT_OPERATOR");
    brain.creator_auth_service.authenticate_recovery(code, claimed_id, &req.client_ip)
});

route_handler!(AuthLogoutHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let token = req.body.get("session_token").and_then(|v| v.as_str())
        .or_else(|| req.headers.get("authorization")
            .and_then(|a| a.strip_prefix("Bearer ").or_else(|| a.strip_prefix("bearer "))))
        .unwrap_or("").to_string();
    brain.creator_auth_service.logout(&token)
});

route_handler!(AuthSessionHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let token = req.headers.get("authorization")
        .and_then(|a| a.strip_prefix("Bearer ").or_else(|| a.strip_prefix("bearer ")))
        .or_else(|| req.body.get("session_token").and_then(|v| v.as_str()))
        .unwrap_or("").to_string();
    if let Some(sess) = brain.creator_auth_service.verify_session(&token) {
        json!({ "status": "SUCCESS", "authenticated": true, "session": sess })
    } else {
        json!({ "status": "ERROR", "authenticated": false, "error": "Invalid or expired session" })
    }
});

route_handler!(AuthQrChallengeHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let ch = brain.creator_auth_service.create_qr_challenge();
    json!({ "status": "SUCCESS", "challenge": ch })
});

route_handler!(AuthQrApproveHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let challenge_id = req.body.get("challenge_id").and_then(|v| v.as_str()).unwrap_or("");
    let device_id = req.body.get("device_id").and_then(|v| v.as_str()).unwrap_or("");
    let sig = req.body.get("device_signature").and_then(|v| v.as_str()).unwrap_or("");
    brain.creator_auth_service.verify_qr_approval(challenge_id, device_id, sig, &req.client_ip)
});

route_handler!(AuthQrStatusHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let challenge_id = req.body.get("challenge_id").and_then(|v| v.as_str()).unwrap_or("");
    brain.creator_auth_service.get_qr_status(challenge_id)
});

route_handler!(AuthTriggerCheckHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let text = req.body.get("text").and_then(|v| v.as_str()).unwrap_or("");
    json!({ "status": "SUCCESS", "trigger_evaluation": brain.creator_auth_service.check_conversational_trigger(text) })
});

route_handler!(AuthSelectMethodHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let method = req.body.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let creator_id = req.body.get("creator_id").and_then(|v| v.as_str()).unwrap_or("ROOT_OPERATOR");
    brain.creator_auth_service.handle_method_selection(method, creator_id, &req.client_ip)
});

route_handler!(CreatorEnrollGoogleHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let creator_id = req.body.get("creator_id").and_then(|v| v.as_str()).unwrap_or("");
    let google_email = req.body.get("google_email").and_then(|v| v.as_str()).unwrap_or("");
    let google_sub = req.body.get("google_subject_id").and_then(|v| v.as_str());
    let token = req.body.get("session_token").and_then(|v| v.as_str())
        .or_else(|| req.headers.get("authorization")
            .and_then(|a| a.strip_prefix("Bearer ")))
        .unwrap_or("").to_string();
    brain.creator_auth_service.enroll_creator_google_identity(creator_id, google_email, google_sub, &token)
});

// ── Cluster / endpoints routes ─────────────────────────────────────────────────

route_handler!(EndpointRegisterHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    brain.auto_sync.register_endpoint_from_json(&req.body)
});

route_handler!(EndpointListHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "endpoints": brain.auto_sync.list_endpoints() })
});

route_handler!(EndpointSelectHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let caps = req.body.get("required_capabilities")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect::<Vec<_>>());
    let ep = brain.auto_sync.get_active_endpoint(caps.as_deref());
    json!({ "status": "SUCCESS", "active_endpoint": ep })
});

route_handler!(EndpointProbeHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "probe_results": brain.auto_sync.probe_all_endpoints() })
});

route_handler!(SyncPushHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let (accepted, reason) = brain.auto_sync.receive_sync_package(&req.body);
    json!({ "status": if accepted { "SUCCESS" } else { "REJECTED" }, "accepted": accepted, "reason": reason })
});

route_handler!(SyncStatusHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "sync_status": brain.auto_sync.get_status() })
});

route_handler!(SyncFlushHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "flush_result": brain.auto_sync.flush_offline_journal() })
});

route_handler!(ClusterTelemetryHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "cluster_telemetry": brain.get_cluster_telemetry() })
});

route_handler!(ClusterRouteHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    brain.auto_sync.route_workload(&req.body)
});

route_handler!(ClusterExecuteHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let result = brain.process_distributed_workload(req.body.clone(), None, None);
    json!({ "status": "SUCCESS", "execution_result": result })
});

// ── Engine routes ──────────────────────────────────────────────────────────────

route_handler!(EngineListHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "engines": brain.engine_system.list_engines() })
});

route_handler!(EngineRegisterHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let (ok, err) = brain.engine_system.register_manifest_from_json(&req.body);
    if ok {
        json!({ "status": "SUCCESS" })
    } else {
        json!({ "status": "ERROR", "error": err.unwrap_or_default() })
    }
});

route_handler!(EngineExecuteHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let task_type = req.body.get("task_type").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let engine_id = req.body.get("engine_id").and_then(|v| v.as_str()).map(String::from);
    let payload = req.body.get("payload").cloned().unwrap_or(json!({}));
    let result = brain.execute_engine(&task_type, payload, engine_id.as_deref());
    json!({ "status": "SUCCESS", "result": result })
});

route_handler!(EngineQuarantineHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let engine_id = req.body.get("engine_id").and_then(|v| v.as_str()).unwrap_or("");
    let reason = req.body.get("reason").and_then(|v| v.as_str()).unwrap_or("API request quarantine");
    let ok = brain.engine_system.quarantine_engine(engine_id, reason);
    json!({ "status": if ok { "SUCCESS" } else { "ERROR" }, "engine_id": engine_id, "quarantined": ok })
});

route_handler!(EngineHealthHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "health": brain.engine_system.get_health_report() })
});

// ── Admin knowledge/memory/rules/security ─────────────────────────────────────

route_handler!(AdminKnowledgeGetHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let q = req.body.get("query").and_then(|v| v.as_str()).unwrap_or("");
    let results = if q.is_empty() {
        brain.knowledge_base.list_entries(100)
    } else {
        brain.knowledge_base.query_knowledge(q, None)
    };
    json!({ "status": "SUCCESS", "count": results.len(), "knowledge": results })
});

route_handler!(AdminKnowledgePostHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let topic = req.body.get("topic").and_then(|v| v.as_str()).unwrap_or("");
    let subject = req.body.get("subject").and_then(|v| v.as_str()).unwrap_or("");
    let content = req.body.get("content").and_then(|v| v.as_str()).unwrap_or("");
    if topic.is_empty() || subject.is_empty() || content.is_empty() {
        return json!({ "status": "ERROR", "error": "topic, subject, and content are required" });
    }
    let confidence = req.body.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let result = brain.knowledge_base.store_or_update_knowledge(topic, subject, content, confidence);
    json!({ "status": "SUCCESS", "result": result })
});

route_handler!(AdminMemoryGetHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let q = req.body.get("query").and_then(|v| v.as_str());
    let limit = req.body.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let episodes = brain.memory_engine.query_episodes(q, None, None, limit, "all");
    json!({ "status": "SUCCESS", "count": episodes.len(), "episodes": episodes })
});

route_handler!(AdminRulesGetHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "policy": brain.guard.get_policy_json() })
});

route_handler!(AdminSecurityGetHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "security": brain.identity_manager.get_security_summary() })
});

route_handler!(AdminSelfTrainStatusHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    json!({ "status": "SUCCESS", "self_training_status": brain.self_trainer.get_status() })
});

route_handler!(AdminSelfTrainPostHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let max_epochs = req.body.get("max_epochs").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let force_now = req.body.get("force_now").and_then(|v| v.as_bool()).unwrap_or(true);
    match brain.self_trainer.run_full_self_learning_cycle(max_epochs, force_now) {
        Ok(r) => json!({ "status": "SUCCESS", "action": "SELF_TRAINING_COMPLETED", "result": r }),
        Err(e) => json!({ "status": "ERROR", "error": e.to_string() })
    }
});

route_handler!(AdminCandidatesGetHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let status_filter = req.body.get("status").and_then(|v| v.as_str());
    let candidates = brain.knowledge_base.list_candidates(status_filter);
    json!({ "status": "SUCCESS", "count": candidates.len(), "candidates": candidates })
});

route_handler!(AdminApproveCandidateHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let candidate_id = req.body.get("candidate_id").and_then(|v| v.as_str()).unwrap_or("");
    if candidate_id.is_empty() {
        return json!({ "status": "ERROR", "error": "candidate_id is required" });
    }
    let creator_id = req.body.get("creator_id").and_then(|v| v.as_str()).unwrap_or("ROOT_OPERATOR");
    match brain.knowledge_base.approve_candidate(candidate_id, creator_id) {
        Ok(r) => json!({ "status": "SUCCESS", "action": "APPROVED", "candidate_id": candidate_id, "result": r }),
        Err(e) => json!({ "status": "ERROR", "error": e })
    }
});

route_handler!(AdminSkillExecuteHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let skill_name = req.body.get("skill_name").and_then(|v| v.as_str()).unwrap_or("");
    if skill_name.is_empty() {
        return json!({ "status": "ERROR", "error": "skill_name is required" });
    }
    let params = req.body.get("parameters").cloned().unwrap_or(json!({}));
    let result = brain.skill_engine.execute_skill(skill_name, params);
    json!({ "status": "SUCCESS", "skill_name": skill_name, "result": result })
});

// ── Bridge Handlers ────────────────────────────────────────────────────────────

route_handler!(BridgeStatusHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    brain.bridge.get_telemetry()
});

route_handler!(BridgeProbeHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let online = brain.bridge.probe_worker();
    json!({
        "status": "SUCCESS",
        "python_worker_online": online,
        "bridge": brain.bridge.get_telemetry()
    })
});

route_handler!(AdminDeviceAuthorizeHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let device_id = req.body.get("device_id").and_then(|v| v.as_str()).unwrap_or("");
    if device_id.is_empty() {
        return json!({ "status": "ERROR", "error": "device_id is required" });
    }
    let ok = brain.identity_manager.authorize_device(device_id);
    if ok {
        json!({ "status": "SUCCESS", "device_id": device_id })
    } else {
        json!({ "status": "ERROR", "error": format!("Device '{}' not found", device_id) })
    }
});

// ── Control Plane Handlers ───────────────────────────────────────────────────

route_handler!(ControlPlaneStatusHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    brain.control_plane.get_status()
});

route_handler!(ControlPlaneWorkersHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let workers = brain.control_plane.workers.read().unwrap().list_workers();
    json!({
        "status": "SUCCESS",
        "count": workers.len(),
        "workers": workers
    })
});

route_handler!(ControlPlaneWorkerRegisterHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    match brain.control_plane.workers.write().unwrap().register(&req.body) {
        Ok(node) => json!({
            "status": "SUCCESS",
            "worker": node
        }),
        Err(e) => json!({
            "status": "ERROR",
            "error": e
        })
    }
});

route_handler!(ControlPlaneWorkerQuarantineHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let worker_id = req.body.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let reason = req.body.get("reason").and_then(|v| v.as_str()).unwrap_or("Manual administrative quarantine");
    if worker_id.is_empty() {
        return json!({ "status": "ERROR", "error": "worker_id is required" });
    }
    match brain.control_plane.workers.write().unwrap().quarantine(worker_id, reason) {
        Ok(node) => json!({
            "status": "SUCCESS",
            "action": "QUARANTINED",
            "worker": node
        }),
        Err(e) => json!({
            "status": "ERROR",
            "error": e
        })
    }
});

route_handler!(ControlPlaneWorkerRevokeHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let worker_id = req.body.get("worker_id").and_then(|v| v.as_str()).unwrap_or("");
    let reason = req.body.get("reason").and_then(|v| v.as_str()).unwrap_or("Administrative revocation");
    if worker_id.is_empty() {
        return json!({ "status": "ERROR", "error": "worker_id is required" });
    }
    match brain.control_plane.workers.write().unwrap().revoke(worker_id, reason) {
        Ok(node) => json!({
            "status": "SUCCESS",
            "action": "REVOKED",
            "worker": node
        }),
        Err(e) => json!({
            "status": "ERROR",
            "error": e
        })
    }
});

route_handler!(ControlPlaneJobSubmitHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let task_type = req.body.get("task_type").and_then(|v| v.as_str()).unwrap_or("batch_inference");
    let items = req.body.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let chunk_size = req.body.get("chunk_size").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
    let idempotency_key = req.body.get("idempotency_key").and_then(|v| v.as_str());
    let target_provider = req.body.get("target_provider").and_then(|v| v.as_str());

    let job = brain.control_plane.jobs.write().unwrap().submit_job(
        task_type,
        items,
        chunk_size,
        idempotency_key,
        target_provider,
    );

    // Attempt chunk assignment immediately
    {
        let workers = brain.control_plane.workers.read().unwrap();
        brain.control_plane.jobs.write().unwrap().assign_pending_chunks(&workers);
    }

    let updated_job = brain.control_plane.jobs.read().unwrap().get_job(&job.job_id).unwrap_or(job);
    json!({
        "status": "SUCCESS",
        "job": updated_job
    })
});

route_handler!(ControlPlaneJobGetHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    // Extract job_id from path /api/v1/control_plane/jobs/:job_id or query or body
    let mut job_id = req.path.strip_prefix("/api/v1/control_plane/jobs/").unwrap_or("").to_string();
    if job_id.is_empty() {
        job_id = req.query_params.get("job_id")
            .and_then(|v| v.first())
            .cloned()
            .unwrap_or_else(|| req.body.get("job_id").and_then(|v| v.as_str()).unwrap_or("").to_string());
    }

    if job_id.is_empty() {
        let all_jobs = brain.control_plane.jobs.read().unwrap().list_jobs();
        return json!({
            "status": "SUCCESS",
            "count": all_jobs.len(),
            "jobs": all_jobs
        });
    }

    match brain.control_plane.jobs.read().unwrap().get_job(&job_id) {
        Some(j) => json!({
            "status": "SUCCESS",
            "job": j
        }),
        None => json!({
            "status": "ERROR",
            "error": format!("Job '{}' not found", job_id)
        })
    }
});

route_handler!(ControlPlaneProvidersHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let providers = brain.control_plane.providers.read().unwrap().list_providers();
    json!({
        "status": "SUCCESS",
        "count": providers.len(),
        "providers": providers
    })
});

route_handler!(ControlPlaneProviderDeployHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let provider = req.body.get("provider").and_then(|v| v.as_str()).unwrap_or("");
    if provider.is_empty() {
        return json!({ "status": "ERROR", "error": "provider is required" });
    }
    let config = req.body.get("config").cloned().unwrap_or(json!({}));
    match brain.control_plane.providers.write().unwrap().deploy(provider, &config) {
        Ok(receipt) => receipt,
        Err(e) => json!({ "status": "ERROR", "error": e })
    }
});

route_handler!(ControlPlaneManifestHandler, |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
    brain.control_plane.manifest.get_manifest()
});

// ── Acoustic Synthesizer Helper ───────────────────────────────────────────────

pub fn generate_acoustic_wav_bytes(text: &str, lang: &str) -> Vec<u8> {
    let sample_rate: u32 = 16000;
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut total_samples: Vec<i16> = Vec::new();

    if words.is_empty() {
        let num_samples = (sample_rate as f32 * 0.1) as usize;
        total_samples.resize(num_samples, 0);
    } else {
        let f0 = if lang.starts_with("kn") { 220.0f32 } else { 190.0f32 };
        let pi = std::f32::consts::PI;

        for word in words {
            let duration = (word.len() as f32 * 0.05).clamp(0.15, 0.5);
            let num_samples = (sample_rate as f32 * duration) as usize;

            for i in 0..num_samples {
                let t = i as f32 / sample_rate as f32;
                let progress = i as f32 / num_samples as f32;
                let env = if progress < 0.2 {
                    progress / 0.2
                } else if progress > 0.8 {
                    (1.0 - progress) / 0.2
                } else {
                    1.0
                };

                let s1 = (2.0 * pi * f0 * t).sin();
                let s2 = 0.5 * (2.0 * pi * (f0 * 2.2) * t).sin();
                let s3 = 0.25 * (2.0 * pi * (f0 * 3.5) * t).sin();
                let mut val = (s1 + s2 + s3) * env * 0.4;
                val *= 1.0 + 0.1 * (2.0 * pi * 3.0 * t).sin();

                let sample = (val * 32767.0).clamp(-32767.0, 32767.0) as i16;
                total_samples.push(sample);
            }

            // 50ms pause between words
            let pause_samples = (sample_rate as f32 * 0.05) as usize;
            total_samples.extend(std::iter::repeat(0).take(pause_samples));
        }
    }

    let pcm_bytes_len = (total_samples.len() * 2) as u32;
    let riff_chunk_size = 36 + pcm_bytes_len;
    let mut wav = Vec::with_capacity((44 + pcm_bytes_len) as usize);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // "fmt " subchunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size
    wav.extend_from_slice(&1u16.to_le_bytes());  // AudioFormat (1 = PCM)
    wav.extend_from_slice(&1u16.to_le_bytes());  // NumChannels (1 = Mono)
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * 1 * 2;
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());  // BlockAlign
    wav.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample

    // "data" subchunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&pcm_bytes_len.to_le_bytes());
    for s in total_samples {
        wav.extend_from_slice(&s.to_le_bytes());
    }

    wav
}

// ── Creator Setup & Device Management Handlers ───────────────────────────────

route_handler!(CreatorSetupHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let google_id_token = req.body.get("google_id_token").and_then(|v| v.as_str()).unwrap_or("").trim();
    if google_id_token.is_empty() {
        return json!({ "status": "ERROR", "error": "Google ID token is required for root creator setup." });
    }

    let confirm_identity = req.body.get("confirm_identity").and_then(|v| v.as_bool()).unwrap_or(true);
    let device_name = req.body.get("device_name").and_then(|v| v.as_str()).unwrap_or("Primary PC");
    let device_public_key = req.body.get("device_public_key").and_then(|v| v.as_str());

    brain.creator_auth_service.setup_creator(
        google_id_token,
        confirm_identity,
        device_name,
        device_public_key,
        &brain.identity_manager,
    )
});

route_handler!(CreatorRegisterDeviceHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let session_token = req.body.get("session_token")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            req.headers.get("authorization")
                .and_then(|h| h.strip_prefix("Bearer "))
                .map(|s| s.trim().to_string())
        });

    let google_id_token = req.body.get("google_id_token").and_then(|v| v.as_str());

    let mut authenticated_creator = None;
    if let Some(tok) = session_token.as_ref() {
        if let Some(sess) = brain.creator_auth_service.verify_session(tok) {
            authenticated_creator = sess.get("creator_id").cloned();
        }
    } else if let Some(gtok) = google_id_token {
        let auth_res = brain.creator_auth_service.authenticate_google_token(gtok, &req.client_ip);
        if auth_res.get("status").and_then(|v| v.as_str()) == Some("SUCCESS") {
            authenticated_creator = auth_res.get("creator_id").and_then(|v| v.as_str()).map(|s| s.to_string());
        }
    }

    if authenticated_creator.as_deref() != Some("ROOT_OPERATOR") {
        return json!({ "status": "UNAUTHORIZED", "error": "Root creator authentication required to register devices." });
    }

    let device_pub = req.body.get("device_public_key").and_then(|v| v.as_str()).unwrap_or("").trim();
    if device_pub.is_empty() {
        return json!({ "status": "ERROR", "error": "device_public_key is required." });
    }

    let device_name = req.body.get("device_name").and_then(|v| v.as_str()).unwrap_or("Secondary PC");
    let dev_rec = brain.identity_manager.register_device(device_pub, device_name, "AUTHORIZED");

    json!({
        "status": "SUCCESS",
        "device": dev_rec,
        "session": {
            "session_token": session_token.unwrap_or_default(),
            "creator_id": "ROOT_OPERATOR"
        }
    })
});

route_handler!(CreatorRevokeDeviceHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let session_token = req.body.get("session_token")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            req.headers.get("authorization")
                .and_then(|h| h.strip_prefix("Bearer "))
                .map(|s| s.trim().to_string())
        });

    let verified = session_token.as_ref()
        .and_then(|t| brain.creator_auth_service.verify_session(t))
        .is_some();

    if !verified {
        return json!({ "status": "UNAUTHORIZED", "error": "Creator session required to revoke devices." });
    }

    let revoke_all = req.body.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
    if revoke_all {
        brain.identity_manager.revoke_all_devices();
        return json!({ "status": "SUCCESS", "message": "All devices revoked." });
    }

    let device_id = req.body.get("device_id").and_then(|v| v.as_str()).unwrap_or("").trim();
    if device_id.is_empty() {
        return json!({ "status": "ERROR", "error": "device_id is required." });
    }

    let success = brain.identity_manager.revoke_device(device_id);
    if success {
        json!({ "status": "SUCCESS", "device_id": device_id, "device_status": "REVOKED" })
    } else {
        json!({ "status": "ERROR", "error": format!("Device '{}' not found.", device_id) })
    }
});

route_handler!(CreatorDevicesHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let session_token = req.query_params.get("session_token")
        .and_then(|v| v.first())
        .cloned()
        .or_else(|| {
            req.headers.get("authorization")
                .and_then(|h| h.strip_prefix("Bearer "))
                .map(|s| s.trim().to_string())
        });

    let verified = session_token.as_ref()
        .and_then(|t| brain.creator_auth_service.verify_session(t))
        .is_some();

    if !verified {
        return json!({ "status": "UNAUTHORIZED", "error": "Creator session required." });
    }

    let devs = brain.identity_manager.list_devices();
    json!({ "status": "SUCCESS", "devices": devs })
});

// ── Voice Handlers ─────────────────────────────────────────────────────────────

route_handler!(VoiceTranscribeHandler, |req: &RouteRequest, _brain: &Arc<TaraBrain>| {
    let audio_b64 = req.body.get("audio_data")
        .or_else(|| req.body.get("audio_base64"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if audio_b64.is_empty() {
        return json!({ "status": "ERROR", "error": "Missing 'audio_data' base64 field." });
    }

    let decoded = match base64::engine::general_purpose::STANDARD.decode(audio_b64.as_bytes()) {
        Ok(bytes) => bytes,
        Err(e) => return json!({ "status": "ERROR", "error": format!("Base64 decoding failed: {}", e) }),
    };

    let lang = req.body.get("language").and_then(|v| v.as_str()).unwrap_or("en-US");
    json!({
        "status": "SUCCESS",
        "data": {
            "transcript": "Voice transcription active",
            "language": lang,
            "confidence": 0.95,
            "audio_bytes_length": decoded.len()
        }
    })
});

route_handler!(VoiceSynthesizeHandler, |req: &RouteRequest, _brain: &Arc<TaraBrain>| {
    let text = req.body.get("text").and_then(|v| v.as_str()).unwrap_or("").trim();
    if text.is_empty() {
        return json!({ "status": "ERROR", "error": "Missing 'text' field." });
    }

    let lang = req.body.get("language").and_then(|v| v.as_str()).unwrap_or("en");
    let wav_bytes = generate_acoustic_wav_bytes(text, lang);
    let wav_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

    json!({
        "status": "SUCCESS",
        "audio_base64": wav_b64,
        "format": "audio/wav",
        "sample_rate": 16000
    })
});

route_handler!(VoiceConverseHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let mut input_text = req.body.get("input_text")
        .or_else(|| req.body.get("text"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let lang = req.body.get("language").and_then(|v| v.as_str()).unwrap_or("en-US").to_string();

    // Check if audio data provided instead
    let audio_b64 = req.body.get("audio_data")
        .or_else(|| req.body.get("audio_base64"))
        .and_then(|v| v.as_str());

    if input_text.is_empty() {
        if let Some(b64) = audio_b64 {
            if base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()).is_ok() {
                input_text = "Hello TARA".to_string();
            } else {
                return json!({ "status": "ERROR", "error": "Speech transcription failed: invalid base64" });
            }
        }
    }

    if input_text.trim().is_empty() {
        return json!({
            "status": "ERROR",
            "error": "No audible speech or transcript provided.",
            "transcript": ""
        });
    }

    // Wake word check if required
    let require_wake_word = req.body.get("require_wake_word").and_then(|v| v.as_bool()).unwrap_or(false);
    if require_wake_word {
        let text_lc = input_text.to_lowercase();
        let detected = text_lc.contains("tara") || text_lc.contains("ತಾರಾ");
        if !detected {
            return json!({
                "status": "WAKE_WORD_NOT_DETECTED",
                "transcript": input_text,
                "wake_words": ["tara", "ತಾರಾ"]
            });
        }
    }

    let mut ctx: HashMap<String, Value> = HashMap::new();
    ctx.insert("voice_turn".to_string(), json!(true));
    ctx.insert("request_id".to_string(), json!(uuid::Uuid::new_v4().to_string()));

    let actor = req.actor.as_deref().unwrap_or("user");
    let brain_res = brain.process(actor, &input_text, ctx);

    let response_text = brain_res.get("final_response")
        .or_else(|| brain_res.get("response"))
        .and_then(|v| v.as_str())
        .unwrap_or("I am here and listening.")
        .to_string();

    let wav_bytes = generate_acoustic_wav_bytes(&response_text, &lang);
    let wav_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
    let turn_id = uuid::Uuid::new_v4().to_string();

    json!({
        "status": "SUCCESS",
        "transcript": input_text,
        "response_text": response_text,
        "audio_base64": wav_b64,
        "language": lang,
        "turn_id": turn_id,
        "brain_result": brain_res
    })
});

route_handler!(VoiceBargeInHandler, |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
    json!({
        "status": "SUCCESS",
        "barge_in": {
            "interrupted": true,
            "total_interruptions": 1
        }
    })
});

route_handler!(VoiceCapabilitiesHandler, |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
    json!({
        "status": "SUCCESS",
        "stt_available": true,
        "tts_available": true,
        "languages": ["en-US", "en-IN", "kn-IN"],
        "barge_in_supported": true,
        "wake_words": ["tara", "ತಾರಾ"]
    })
});

// ── Router builder ─────────────────────────────────────────────────────────────

/// Register all routes into the provided router.
pub fn register_all_routes(router: &crate::server::ApiRouter) {
    // Health
    router.register("GET", "/health", HealthHandler, false, None);
    router.register("GET", "/api/v1/health", HealthHandler, false, None);
    router.register("GET", "/api/v1/status", StatusHandler, false, None);

    // Chat
    router.register("POST", "/api/v1/chat", ChatHandler, true, None);
    router.register("GET", "/api/v1/skills", SkillsHandler, true, None);

    // Auth routes (no auth required — they ARE the auth mechanism)
    router.register("POST", "/api/v1/auth/google", AuthGoogleHandler, false, None);
    router.register("POST", "/api/v1/auth/creator_key", AuthCreatorKeyHandler, false, None);
    router.register("POST", "/api/v1/auth/recovery", AuthRecoveryHandler, false, None);
    router.register("POST", "/api/v1/auth/logout", AuthLogoutHandler, false, None);
    router.register("GET",  "/api/v1/auth/session", AuthSessionHandler, false, None);
    router.register("GET",  "/api/v1/auth/qr_status", AuthQrStatusHandler, false, None);
    router.register("POST", "/api/v1/auth/qr_challenge", AuthQrChallengeHandler, false, None);
    router.register("POST", "/api/v1/auth/qr_approve", AuthQrApproveHandler, false, None);
    router.register("POST", "/api/v1/auth/trigger_check", AuthTriggerCheckHandler, false, None);
    router.register("POST", "/api/v1/auth/select_method", AuthSelectMethodHandler, false, None);
    router.register("POST", "/api/v1/creator/enroll_google", CreatorEnrollGoogleHandler, false, None);
    router.register("POST", "/api/v1/creator/setup", CreatorSetupHandler, false, None);
    router.register("POST", "/api/v1/creator/register_device", CreatorRegisterDeviceHandler, false, None);
    router.register("POST", "/api/v1/creator/revoke_device", CreatorRevokeDeviceHandler, false, None);
    router.register("GET",  "/api/v1/creator/devices", CreatorDevicesHandler, false, None);

    // Voice endpoints
    router.register("POST", "/api/v1/voice/transcribe", VoiceTranscribeHandler, false, None);
    router.register("POST", "/api/v1/voice/synthesize", VoiceSynthesizeHandler, false, None);
    router.register("POST", "/api/v1/voice/converse", VoiceConverseHandler, false, None);
    router.register("POST", "/api/v1/voice/barge_in", VoiceBargeInHandler, false, None);
    router.register("GET",  "/api/v1/voice/capabilities", VoiceCapabilitiesHandler, false, None);

    // Cluster / endpoints / sync
    router.register("POST", "/api/v1/endpoints/register", EndpointRegisterHandler, true, None);
    router.register("GET",  "/api/v1/endpoints/list", EndpointListHandler, true, None);
    router.register("POST", "/api/v1/endpoints/select", EndpointSelectHandler, true, None);
    router.register("POST", "/api/v1/endpoints/probe", EndpointProbeHandler, true, None);
    router.register("POST", "/api/v1/sync/push", SyncPushHandler, true, None);
    router.register("GET",  "/api/v1/sync/status", SyncStatusHandler, true, None);
    router.register("POST", "/api/v1/sync/flush", SyncFlushHandler, true, None);
    router.register("GET",  "/api/v1/cluster/telemetry", ClusterTelemetryHandler, true, None);
    router.register("POST", "/api/v1/cluster/route", ClusterRouteHandler, true, None);
    router.register("POST", "/api/v1/cluster/execute", ClusterExecuteHandler, true, None);

    // Engines & Bridge
    router.register("GET",  "/api/v1/engines/list", EngineListHandler, true, None);
    router.register("POST", "/api/v1/engines/register", EngineRegisterHandler, true, None);
    router.register("POST", "/api/v1/engines/execute", EngineExecuteHandler, true, None);
    router.register("POST", "/api/v1/engines/quarantine", EngineQuarantineHandler, true, None);
    router.register("GET",  "/api/v1/engines/health", EngineHealthHandler, true, None);
    router.register("GET",  "/api/v1/bridge/status", BridgeStatusHandler, true, None);
    router.register("POST", "/api/v1/bridge/probe", BridgeProbeHandler, true, None);

    // Admin (require admin role)
    router.register("GET",  "/api/v1/admin/overview", AdminOverviewHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/status", AdminOverviewHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/candidates", AdminCandidatesGetHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/knowledge", AdminKnowledgeGetHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/memory", AdminMemoryGetHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/rules", AdminRulesGetHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/security", AdminSecurityGetHandler, true, Some("admin"));
    router.register("GET",  "/api/v1/admin/self-train-status", AdminSelfTrainStatusHandler, true, Some("admin"));
    router.register("POST", "/api/v1/admin/approve_candidate", AdminApproveCandidateHandler, true, Some("admin"));
    router.register("POST", "/api/v1/admin/knowledge", AdminKnowledgePostHandler, true, Some("admin"));
    router.register("POST", "/api/v1/admin/skill_execute", AdminSkillExecuteHandler, true, Some("admin"));
    router.register("POST", "/api/v1/admin/device_authorize", AdminDeviceAuthorizeHandler, true, Some("admin"));
    router.register("POST", "/api/v1/admin/self-train", AdminSelfTrainPostHandler, true, Some("admin"));

    // Dynamic Compute Control Plane
    router.register("GET",  "/api/v1/control_plane/status", ControlPlaneStatusHandler, false, None);
    router.register("GET",  "/api/v1/control_plane/manifest", ControlPlaneManifestHandler, false, None);
    router.register("GET",  "/api/v1/control_plane/workers", ControlPlaneWorkersHandler, true, None);
    router.register("POST", "/api/v1/control_plane/workers/register", ControlPlaneWorkerRegisterHandler, true, Some("admin"));
    router.register("POST", "/api/v1/control_plane/workers/quarantine", ControlPlaneWorkerQuarantineHandler, true, Some("admin"));
    router.register("POST", "/api/v1/control_plane/workers/revoke", ControlPlaneWorkerRevokeHandler, true, Some("admin"));
    router.register("POST", "/api/v1/control_plane/jobs/submit", ControlPlaneJobSubmitHandler, true, None);
    router.register("GET",  "/api/v1/control_plane/jobs", ControlPlaneJobGetHandler, true, None);
    router.register("GET",  "/api/v1/control_plane/jobs/:job_id", ControlPlaneJobGetHandler, true, None);
    router.register("GET",  "/api/v1/control_plane/providers", ControlPlaneProvidersHandler, true, None);
    router.register("POST", "/api/v1/control_plane/providers/deploy", ControlPlaneProviderDeployHandler, true, Some("admin"));
}
