//! All route handlers for the TARA HTTP API.
//! Implements every endpoint from Python server.py register_* functions.

pub mod bridge;
pub use bridge::*;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use base64::Engine;
use serde_json::{json, Value};

use crate::brain::TaraBrain;
use crate::server::{RouteHandler, RouteRequest};

// ── Chat UI HTML (embedded) ────────────────────────────────────────────────────

/// Sanitize web session token against DOM-based XSS when injecting into HTML / JS blocks.
pub fn sanitize_token_for_js(token: &str) -> String {
    token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
        .take(512)
        .collect()
}

/// Extract Bearer token from headers (case-insensitive "Bearer " / "bearer ").
pub fn extract_bearer_token(headers: &HashMap<String, String>) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|h| {
            h.strip_prefix("Bearer ")
                .or_else(|| h.strip_prefix("bearer "))
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Verify if an actor has privileged operator or administrator roles.
pub fn is_privileged_actor(actor: Option<&str>) -> bool {
    match actor {
        Some(actor) => {
            let allowed = ["admin", "root", "system", "ROOT_OPERATOR", "OPERATOR_ROOT"];
            allowed.contains(&actor)
                || actor.starts_with("creator:")
                || actor.starts_with("api_key:admin")
                || actor.starts_with("api_key:ROOT_OPERATOR")
                || actor.starts_with("api_key:operator")
                || actor.starts_with("operator")
        }
        None => false,
    }
}

pub fn render_chat_ui(web_session_token: &str) -> String {
    let safe_token = sanitize_token_for_js(web_session_token);
    // Attempt loading canonical synchronized frontend from disk
    let candidates = [
        "frontend/index.html",
        "../../frontend/index.html",
        "../../../frontend/index.html",
    ];
    for path in &candidates {
        if let Ok(content) = std::fs::read_to_string(path) {
            return content.replace("__TARA_SESSION_TOKEN__", &safe_token);
        }
    }

    // Fallback embedded web chat HTML injected with the web session token
    format!(
        r#"<!DOCTYPE html>
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
</html>"#,
        token = safe_token
    )
}

fn server_time() -> String {
    crate::now_iso()
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

// ── API Discovery & Info ───────────────────────────────────────────────────────

route_handler!(
    ApiDiscoveryHandler,
    |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        json!({
            "status": "ONLINE",
            "api_version": "v1",
            "timestamp": server_time(),
            "endpoints": [
                "/api/v1/health",
                "/api/v1/status",
                "/api/v1/chat",
                "/api/v1/skills",
                "/api/v1/auth/google",
                "/api/v1/auth/session"
            ]
        })
    }
);

route_handler!(
    ChatInfoHandler,
    |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        json!({
            "status": "ONLINE",
            "endpoint": "/api/v1/chat",
            "method": "POST",
            "description": "TARA Neural Chat Endpoint. Send POST request with JSON payload: {\"input\": \"your prompt here\"}",
            "auth_required": true
        })
    }
);

// ── Health ─────────────────────────────────────────────────────────────────────

route_handler!(
    HealthHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let model_status = brain.get_model_status();
        let is_loaded = model_status
            .get("offline_ready")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let model_id = model_status
            .get("model_identity")
            .and_then(|s| s.as_str())
            .unwrap_or("TARA");
        json!({
            "status": "HEALTHY",
            "service": "TARA Neural Intelligence Engine",
            "version": env!("CARGO_PKG_VERSION"),
            "model_identity": model_id,
            "model_checksum": model_status.get("model_sha256"),
            "parameter_count": model_status.get("parameters"),
            "live_inference_ready": is_loaded,
            "engine_state": if is_loaded { "READY" } else { "STANDBY" },
            "timestamp": server_time(),
            "model_status": model_status
        })
    }
);

// ── Status ─────────────────────────────────────────────────────────────────────

route_handler!(
    StatusHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let model_status = brain.get_model_status();
        let mem_stats = brain.memory_engine.get_memory_stats();
        let bridge_status = brain.bridge.get_telemetry();
        json!({
            "status": "ONLINE",
            "server_time": server_time(),
            "model": model_status,
            "model_registry": {
                "active_version": brain.model_registry.get_active_version(),
                "versions": brain.model_registry.list_versions()
            },
            "memory": mem_stats,
            "bridge": bridge_status,
            "guard_active": true
        })
    }
);

// ── Chat ───────────────────────────────────────────────────────────────────────

route_handler!(ChatHandler, |req: &RouteRequest, brain: &Arc<TaraBrain>| {
    let input = req
        .body
        .get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
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
    let req_id = req
        .body
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    ctx.insert("request_id".to_string(), json!(req_id));

    // Pass creator session token if present in payload
    if let Some(t) = req.body.get("creator_session_token") {
        ctx.insert("creator_session_token".to_string(), t.clone());
    }

    let result = brain.process(&actor, &input, ctx);
    json!({
        "status": "SUCCESS",
        "request_id": req_id,
        "model_id": "tara-native-neural-core",
        "data": result
    })
});

route_handler!(
    DirectInferenceHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let prompt = req
            .body
            .get("prompt")
            .or_else(|| req.body.get("input"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        if prompt.is_empty() {
            return json!({ "status": "ERROR", "error": "Missing prompt" });
        }
        let max_new_tokens = match req.body.get("max_new_tokens") {
            Some(value) => match value.as_u64() {
                Some(value) => value as usize,
                None => {
                    return json!({ "status": "ERROR", "error": "max_new_tokens must be an unsigned integer" })
                }
            },
            None => brain.config.inference.default_max_new_tokens,
        };
        let temperature = match req.body.get("temperature") {
            Some(value) => match value.as_f64() {
                Some(value) => value as f32,
                None => {
                    return json!({ "status": "ERROR", "error": "temperature must be a number" })
                }
            },
            None => brain.config.inference.default_temperature,
        };
        let start = Instant::now();
        match brain.direct_inference(prompt, max_new_tokens, temperature) {
            Ok(result) => {
                let status = brain.get_model_status();
                json!({
                    "status": "SUCCESS",
                    "text": result.text,
                    "response": result.text,
                    "model_identity": status.get("model_identity"),
                    "model_checksum": status.get("model_sha256"),
                    "parameter_count": status.get("parameters"),
                    "token_count": result.token_count,
                    "tps": result.tps,
                    "latency_ms": start.elapsed().as_secs_f64() * 1000.0
                })
            }
            Err(error) => json!({ "status": "ERROR", "error": error }),
        }
    }
);

// ── Skills ─────────────────────────────────────────────────────────────────────

route_handler!(
    SkillsHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let skills = brain.skill_engine.list_skills();
        json!({ "status": "SUCCESS", "skills_count": skills.len(), "skills": skills })
    }
);

// ── Admin overview ─────────────────────────────────────────────────────────────

route_handler!(
    AdminOverviewHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let model_status = brain.get_model_status();
        let mem_stats = brain.memory_engine.get_memory_stats();
        let skills = brain.skill_engine.list_skills();
        let knowledge_entries_count = brain
            .knowledge_base
            .index
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .doc_cache
            .len();
        let rules_count = brain
            .guard
            .get_policy_json()
            .get("rules")
            .and_then(|r| r.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        let all_devs = brain.identity_manager.list_devices();
        let auth_devs = all_devs.iter().filter(|d| d.status == "AUTHORIZED").count();
        let lockdown_state = if brain.is_system_locked_down() {
            "LOCKDOWN"
        } else {
            "NORMAL"
        };
        json!({
            "status": "SUCCESS",
            "server_time": server_time(),
            "model": model_status,
            "knowledge_entries_count": knowledge_entries_count,
            "skills_count": skills.len(),
            "rules_count": rules_count,
            "memory": mem_stats,
            "devices": { "total": all_devs.len(), "authorized": auth_devs },
            "lockdown_state": lockdown_state
        })
    }
);

// ── Auth handlers ──────────────────────────────────────────────────────────────

route_handler!(
    AuthGoogleChallengeHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let nonce = brain.creator_auth_service.create_google_auth_nonce();
        json!({
            "status": "SUCCESS",
            "challenge_nonce": nonce
        })
    }
);

route_handler!(
    AuthGoogleHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let id_token = req
            .body
            .get("id_token")
            .or(req.body.get("token"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let nonce = req
            .body
            .get("nonce")
            .or(req.body.get("challenge_nonce"))
            .and_then(|v| v.as_str());
        brain
            .creator_auth_service
            .authenticate_google_token(&id_token, nonce, &req.client_ip)
    }
);

route_handler!(
    AuthCreatorKeyHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let sig = req
            .body
            .get("proof_signature")
            .or(req.body.get("signature"))
            .and_then(|v| v.as_str());
        let nonce = req
            .body
            .get("challenge_nonce")
            .or(req.body.get("nonce"))
            .and_then(|v| v.as_str());
        let claimed_id = req
            .body
            .get("claimed_creator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("ROOT_OPERATOR");
        let passphrase = req.body.get("passphrase").and_then(|v| v.as_str());
        brain.creator_auth_service.authenticate_creator_key(
            sig,
            nonce,
            claimed_id,
            passphrase,
            &req.client_ip,
        )
    }
);

route_handler!(
    AuthRecoveryHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let code = req
            .body
            .get("recovery_code")
            .or(req.body.get("token"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let claimed_id = req
            .body
            .get("claimed_creator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("ROOT_OPERATOR");
        brain
            .creator_auth_service
            .authenticate_recovery(code, claimed_id, &req.client_ip)
    }
);

route_handler!(
    AuthLogoutHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let token = req
            .body
            .get("session_token")
            .and_then(|v| v.as_str())
            .or_else(|| {
                req.headers.get("authorization").and_then(|a| {
                    a.strip_prefix("Bearer ")
                        .or_else(|| a.strip_prefix("bearer "))
                })
            })
            .unwrap_or("")
            .to_string();
        brain.revoke_session_unified(&token);
        brain.creator_auth_service.logout(&token)
    }
);

route_handler!(
    AuthSessionHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let token = req
            .headers
            .get("authorization")
            .and_then(|a| {
                a.strip_prefix("Bearer ")
                    .or_else(|| a.strip_prefix("bearer "))
            })
            .or_else(|| req.body.get("session_token").and_then(|v| v.as_str()))
            .unwrap_or("")
            .to_string();
        if let Some(sess) = brain.creator_auth_service.verify_session(&token) {
            json!({ "status": "SUCCESS", "authenticated": true, "session": sess })
        } else {
            json!({ "status": "ERROR", "authenticated": false, "error": "Invalid or expired session" })
        }
    }
);

route_handler!(
    AuthQrChallengeHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let ch = brain.creator_auth_service.create_qr_challenge();
        json!({ "status": "SUCCESS", "challenge": ch })
    }
);

route_handler!(
    AuthQrApproveHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let challenge_id = req
            .body
            .get("challenge_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let device_id = req
            .body
            .get("device_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let sig = req
            .body
            .get("device_signature")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        brain.creator_auth_service.verify_qr_approval(
            challenge_id,
            device_id,
            sig,
            &brain.identity_manager,
            &req.client_ip,
        )
    }
);

route_handler!(
    AuthQrStatusHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let challenge_id = req
            .body
            .get("challenge_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        brain.creator_auth_service.get_qr_status(challenge_id)
    }
);

route_handler!(
    AuthTriggerCheckHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let text = req.body.get("text").and_then(|v| v.as_str()).unwrap_or("");
        json!({ "status": "SUCCESS", "trigger_evaluation": brain.creator_auth_service.check_conversational_trigger(text) })
    }
);

route_handler!(
    AuthSelectMethodHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let method = req
            .body
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let creator_id = req
            .body
            .get("creator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("ROOT_OPERATOR");
        brain
            .creator_auth_service
            .handle_method_selection(method, creator_id, &req.client_ip)
    }
);

route_handler!(
    CreatorEnrollGoogleHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let creator_id = req
            .body
            .get("creator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let google_email = req
            .body
            .get("google_email")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let google_sub = req.body.get("google_subject_id").and_then(|v| v.as_str());
        let token = req
            .body
            .get("session_token")
            .and_then(|v| v.as_str())
            .or_else(|| {
                req.headers.get("authorization").and_then(|a| {
                    a.strip_prefix("Bearer ")
                        .or_else(|| a.strip_prefix("bearer "))
                })
            })
            .unwrap_or("")
            .to_string();
        brain.creator_auth_service.enroll_creator_google_identity(
            creator_id,
            google_email,
            google_sub,
            &token,
        )
    }
);

// ── Cluster / endpoints routes ─────────────────────────────────────────────────

route_handler!(
    EndpointRegisterHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        brain.auto_sync.register_endpoint_from_json(&req.body)
    }
);

route_handler!(
    EndpointListHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "endpoints": brain.auto_sync.list_endpoints() })
    }
);

route_handler!(
    EndpointSelectHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let caps = req
            .body
            .get("required_capabilities")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            });
        let ep = brain.auto_sync.get_active_endpoint(caps.as_deref());
        json!({ "status": "SUCCESS", "active_endpoint": ep })
    }
);

route_handler!(
    EndpointProbeHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "probe_results": brain.auto_sync.probe_all_endpoints() })
    }
);

route_handler!(
    SyncPushHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let (accepted, reason) = brain.auto_sync.receive_sync_package(&req.body);
        json!({ "status": if accepted { "SUCCESS" } else { "REJECTED" }, "accepted": accepted, "reason": reason })
    }
);

route_handler!(
    SyncStatusHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "sync_status": brain.auto_sync.get_status() })
    }
);

route_handler!(
    SyncFlushHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "flush_result": brain.auto_sync.flush_offline_journal() })
    }
);

route_handler!(
    ClusterTelemetryHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "cluster_telemetry": brain.get_cluster_telemetry() })
    }
);

route_handler!(
    ClusterRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| { brain.auto_sync.route_workload(&req.body) }
);

route_handler!(
    ClusterExecuteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        brain.process_distributed_workload(req.body.clone(), None, None)
    }
);

// ── Engine routes ──────────────────────────────────────────────────────────────

route_handler!(
    EngineListHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "engines": brain.engine_system.list_engines() })
    }
);

route_handler!(
    EngineRegisterHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let (ok, err) = brain.engine_system.register_manifest_from_json(&req.body);
        if ok {
            json!({ "status": "SUCCESS" })
        } else {
            json!({ "status": "ERROR", "error": err.unwrap_or_default() })
        }
    }
);

route_handler!(
    EngineExecuteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let task_type = req
            .body
            .get("task_type")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let engine_id = req
            .body
            .get("engine_id")
            .and_then(|v| v.as_str())
            .map(String::from);
        let payload = req.body.get("payload").cloned().unwrap_or(json!({}));
        brain.execute_engine(&task_type, payload, engine_id.as_deref())
    }
);

route_handler!(
    EngineQuarantineHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let engine_id = req
            .body
            .get("engine_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let reason = req
            .body
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("API request quarantine");
        let ok = brain.engine_system.quarantine_engine(engine_id, reason);
        json!({ "status": if ok { "SUCCESS" } else { "ERROR" }, "engine_id": engine_id, "quarantined": ok })
    }
);

route_handler!(
    EngineHealthHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "health": brain.engine_system.get_health_report() })
    }
);

// ── Admin knowledge/memory/rules/security ─────────────────────────────────────

route_handler!(
    AdminKnowledgeGetHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let q = req.body.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let results = if q.is_empty() {
            brain.knowledge_base.list_entries(100)
        } else {
            brain.knowledge_base.query_knowledge(q, None)
        };
        json!({ "status": "SUCCESS", "count": results.len(), "knowledge": results })
    }
);

route_handler!(
    AdminKnowledgePostHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let topic = req.body.get("topic").and_then(|v| v.as_str()).unwrap_or("");
        let subject = req
            .body
            .get("subject")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let content = req
            .body
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if topic.is_empty() || subject.is_empty() || content.is_empty() {
            return json!({ "status": "ERROR", "error": "topic, subject, and content are required" });
        }
        let confidence = req
            .body
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(1.0);
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return json!({"status":"ERROR","error":"confidence must be between 0 and 1"});
        }
        let result = brain.knowledge_base.store_or_update_knowledge(
            topic,
            subject,
            content,
            confidence as f32,
        );
        let status = result
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("ERROR");
        json!({ "status": status, "result": result })
    }
);

route_handler!(
    AdminKnowledgeImportHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let raw_input = if let Some(items) = req.body.get("items") {
            items.to_string()
        } else if let Some(raw) = req.body.get("raw").and_then(|v| v.as_str()) {
            raw.to_string()
        } else {
            req.body.to_string()
        };
        let default_license = req.body.get("default_license").and_then(|v| v.as_str());
        let default_source = req.body.get("default_source").and_then(|v| v.as_str());
        brain.knowledge_base.import_open_knowledge_batch(
            &raw_input,
            default_license,
            default_source,
        )
    }
);

route_handler!(
    AdminReasoningBridgeHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let min_outcome = req
            .body
            .get("min_outcome")
            .and_then(Value::as_f64)
            .unwrap_or(0.8);
        let min_confidence = req
            .body
            .get("min_confidence")
            .and_then(Value::as_f64)
            .unwrap_or(0.7);
        let subdir = req.body.get("subdir").and_then(Value::as_str);
        match brain
            .cognitive
            .experiential
            .export_approved_lessons_to_dataset(min_outcome, min_confidence, subdir)
        {
            Ok(v) => v,
            Err(e) => json!({ "status": "ERROR", "error": e.to_string() }),
        }
    }
);

route_handler!(
    AdminMemoryGetHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let q = req.body.get("query").and_then(|v| v.as_str());
        let limit = req.body.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        let episodes = brain
            .memory_engine
            .query_episodes(q, None, None, limit, "all");
        json!({ "status": "SUCCESS", "count": episodes.len(), "episodes": episodes })
    }
);

route_handler!(
    AdminRulesGetHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "policy": brain.guard.get_policy_json() })
    }
);

route_handler!(
    AdminSecurityGetHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "security": brain.identity_manager.get_security_summary() })
    }
);

route_handler!(
    AdminSelfTrainStatusHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        json!({ "status": "SUCCESS", "self_training_status": brain.get_self_training_status() })
    }
);

route_handler!(
    AdminSelfTrainPostHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let max_epochs = req
            .body
            .get("max_epochs")
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize;
        let force_now = req
            .body
            .get("force_now")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        match brain.run_self_training(max_epochs, force_now) {
            Ok(r) => {
                json!({ "status": "SUCCESS", "action": "SELF_TRAINING_COMPLETED", "result": r })
            }
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    AdminCandidatesGetHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let status_filter = req.body.get("status").and_then(|v| v.as_str());
        let candidates = brain.knowledge_base.list_candidates(status_filter);
        json!({ "status": "SUCCESS", "count": candidates.len(), "candidates": candidates })
    }
);

route_handler!(
    AdminApproveCandidateHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let candidate_id = req
            .body
            .get("candidate_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if candidate_id.is_empty() {
            return json!({ "status": "ERROR", "error": "candidate_id is required" });
        }
        let creator_id = req
            .body
            .get("creator_id")
            .and_then(|v| v.as_str())
            .unwrap_or("ROOT_OPERATOR");
        match brain
            .knowledge_base
            .approve_candidate(candidate_id, creator_id)
        {
            Ok(r) => {
                json!({ "status": "SUCCESS", "action": "APPROVED", "candidate_id": candidate_id, "result": r })
            }
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    AdminSkillExecuteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let skill_name = req
            .body
            .get("skill_name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if skill_name.is_empty() {
            return json!({ "status": "ERROR", "error": "skill_name is required" });
        }
        let params = req.body.get("parameters").cloned().unwrap_or(json!({}));
        let result = brain.skill_engine.execute_skill(skill_name, params);
        let status = result
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("ERROR");
        json!({ "status": status, "skill_name": skill_name, "result": result })
    }
);

route_handler!(
    AdminSkillAddHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let skill_name = req
            .body
            .get("skill_name")
            .and_then(Value::as_str)
            .unwrap_or("");
        let Some(definition) = req.body.get("definition") else {
            return json!({"status":"ERROR","error":"definition is required"});
        };
        match brain
            .skill_engine
            .add_executable_dynamic_skill(skill_name, definition.clone())
        {
            Ok(message) => json!({"status":"SUCCESS","skill_name":skill_name,"message":message}),
            Err(error) => json!({"status":"ERROR","error":error}),
        }
    }
);

route_handler!(
    AdminSkillRemoveHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let skill_name = req
            .body
            .get("skill_name")
            .and_then(Value::as_str)
            .unwrap_or("");
        match brain.skill_engine.remove_dynamic_skill(skill_name) {
            Ok(message) => json!({"status":"SUCCESS","skill_name":skill_name,"message":message}),
            Err(error) => json!({"status":"ERROR","error":error}),
        }
    }
);

// ── Bridge Handlers ────────────────────────────────────────────────────────────

route_handler!(
    BridgeStatusHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| { brain.bridge.get_telemetry() }
);

route_handler!(
    BridgeProbeHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let online = brain.bridge.probe_worker();
        json!({
            "status": "SUCCESS",
            "python_worker_online": online,
            "bridge": brain.bridge.get_telemetry()
        })
    }
);

route_handler!(
    AdminDeviceAuthorizeHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let device_id = req
            .body
            .get("device_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if device_id.is_empty() {
            return json!({ "status": "ERROR", "error": "device_id is required" });
        }
        let ok = brain.identity_manager.authorize_device(device_id);
        if ok {
            if let Some(dev) = brain.identity_manager.get_device(device_id) {
                if let Ok(dev_id) = tara_security::primitives::DeviceId::new(&dev.device_id) {
                    if let Ok(pub_bytes) = hex::decode(&dev.device_public_key) {
                        if let Ok(principal) =
                            tara_security::primitives::PrincipalId::new("creator_root_operator")
                        {
                            let zero_rec = tara_security::zero_trust::DeviceEnrollmentRecord {
                                device_id: dev_id,
                                enrolled_public_key: pub_bytes,
                                bound_principal_id: principal,
                                is_revoked: false,
                            };
                            let _ = brain.centralized_device_registry.enroll_device(zero_rec);
                        }
                    }
                }
            }
            json!({ "status": "SUCCESS", "device_id": device_id })
        } else {
            json!({ "status": "ERROR", "error": format!("Device '{}' not found", device_id) })
        }
    }
);

// ── Control Plane Handlers ───────────────────────────────────────────────────

route_handler!(
    ControlPlaneStatusHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| { brain.control_plane.get_status() }
);

route_handler!(
    ControlPlaneWorkersHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let workers: Vec<Value> = brain
            .control_plane
            .workers
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .list_workers()
            .into_iter()
            .map(|worker| {
                json!({
                    "worker_id": worker.worker_id,
                    "name": worker.name,
                    "provider": worker.provider,
                    "endpoint": worker.endpoint,
                    "capabilities": worker.capabilities,
                    "model_sha256": worker.model_sha256,
                    "status": worker.status,
                    "quarantine_reason": worker.quarantine_reason,
                    "registered_at": worker.registered_at,
                    "last_heartbeat": worker.last_heartbeat,
                    "active_jobs": worker.active_jobs,
                    "max_concurrency": worker.max_concurrency,
                    "latency_ms": worker.latency_ms,
                    "error_count": worker.error_count
                })
            })
            .collect();
        json!({
            "status": "SUCCESS",
            "count": workers.len(),
            "workers": workers
        })
    }
);

route_handler!(ControlPlaneWorkerRegisterHandler, |req: &RouteRequest,
                                                   brain: &Arc<
    TaraBrain,
>| {
    // Protected route check: requires authenticated creator session or admin API key
    let is_creator_session = req
        .headers
        .get("authorization")
        .and_then(|h| {
            h.strip_prefix("Bearer ")
                .or_else(|| h.strip_prefix("bearer "))
                .or(Some(h.as_str()))
        })
        .map(|t| {
            brain
                .creator_auth_service
                .verify_session(t.trim())
                .is_some()
        })
        .unwrap_or(false);

    let is_admin_key = req
        .headers
        .get("x-admin-key")
        .or_else(|| req.headers.get("x-api-key"))
        .map(|k| {
            if let Ok(expected) = std::env::var("TARA_ADMIN_API_KEY") {
                tara_security::tokens::verify_api_key(k, &expected)
            } else {
                false
            }
        })
        .unwrap_or(false);

    if !is_creator_session && !is_admin_key {
        return json!({
            "status": "ERROR",
            "error": "UNAUTHORIZED: Creator session or admin API key required to register worker nodes"
        });
    }

    match brain
        .control_plane
        .workers
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .register_with_token(&req.body)
    {
        Ok((node, worker_token)) => json!({
            "status": "SUCCESS",
            "worker": {
                "worker_id": node.worker_id,
                "name": node.name,
                "provider": node.provider,
                "endpoint": node.endpoint,
                "capabilities": node.capabilities,
                "model_sha256": node.model_sha256,
                "status": node.status,
                "quarantine_reason": node.quarantine_reason,
                "registered_at": node.registered_at,
                "last_heartbeat": node.last_heartbeat,
                "active_jobs": node.active_jobs,
                "max_concurrency": node.max_concurrency,
                "latency_ms": node.latency_ms,
                "error_count": node.error_count
            },
            "worker_token": worker_token
        }),
        Err(e) => json!({
            "status": "ERROR",
            "error": e
        }),
    }
});

route_handler!(ControlPlaneWorkerHeartbeatHandler, |req: &RouteRequest,
                                                    brain: &Arc<
    TaraBrain,
>| {
    let worker_id = req
        .body
        .get("worker_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    let token = req
        .headers
        .get("x-tara-worker-token")
        .map(String::as_str)
        .unwrap_or("");
    if worker_id.is_empty() || token.is_empty() {
        return json!({ "status": "UNAUTHORIZED", "error": "worker_id and worker token are required" });
    }
    let mut workers = match brain.control_plane.workers.write() {
        Ok(workers) => workers,
        Err(_) => return json!({ "status": "ERROR", "error": "Worker registry is unavailable" }),
    };
    if workers.validate_worker_token(token).as_deref() != Some(worker_id) {
        return json!({ "status": "UNAUTHORIZED", "error": "Worker token does not match worker_id" });
    }
    match workers.heartbeat(
        worker_id,
        req.body.get("latency_ms").and_then(Value::as_f64),
    ) {
        Ok(()) => json!({ "status": "SUCCESS", "worker_id": worker_id, "worker_status": "READY" }),
        Err(error) => json!({ "status": "ERROR", "error": error }),
    }
});

route_handler!(ControlPlaneWorkerQuarantineHandler, |req: &RouteRequest,
                                                     brain: &Arc<
    TaraBrain,
>| {
    let worker_id = req
        .body
        .get("worker_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let reason = req
        .body
        .get("reason")
        .and_then(|v| v.as_str())
        .unwrap_or("Manual administrative quarantine");
    if worker_id.is_empty() {
        return json!({ "status": "ERROR", "error": "worker_id is required" });
    }
    match brain
        .control_plane
        .workers
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .quarantine(worker_id, reason)
    {
        Ok(node) => json!({
            "status": "SUCCESS",
            "action": "QUARANTINED",
            "worker": node
        }),
        Err(e) => json!({
            "status": "ERROR",
            "error": e
        }),
    }
});

route_handler!(
    ControlPlaneWorkerRevokeHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let worker_id = req
            .body
            .get("worker_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let reason = req
            .body
            .get("reason")
            .and_then(|v| v.as_str())
            .unwrap_or("Administrative revocation");
        if worker_id.is_empty() {
            return json!({ "status": "ERROR", "error": "worker_id is required" });
        }
        match brain
            .control_plane
            .workers
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .revoke(worker_id, reason)
        {
            Ok(node) => json!({
                "status": "SUCCESS",
                "action": "REVOKED",
                "worker": node
            }),
            Err(e) => json!({
                "status": "ERROR",
                "error": e
            }),
        }
    }
);

route_handler!(
    ControlPlaneJobSubmitHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let task_type = req
            .body
            .get("task_type")
            .and_then(|v| v.as_str())
            .unwrap_or("batch_inference");
        let items = req
            .body
            .get("items")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let chunk_size = req
            .body
            .get("chunk_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;
        let idempotency_key = req.body.get("idempotency_key").and_then(|v| v.as_str());
        let target_provider = req.body.get("target_provider").and_then(|v| v.as_str());

        let job = match brain
            .control_plane
            .jobs
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .submit_job(
                task_type,
                items,
                chunk_size,
                idempotency_key,
                target_provider,
            ) {
            Ok(j) => j,
            Err(e) => {
                return json!({ "status": "ERROR", "error": format!("Failed to submit job: {}", e) })
            }
        };

        // Attempt chunk assignment immediately
        {
            let workers = brain
                .control_plane
                .workers
                .read()
                .unwrap_or_else(|p| p.into_inner());
            if let Err(e) = brain
                .control_plane
                .jobs
                .write()
                .unwrap_or_else(|p| p.into_inner())
                .assign_pending_chunks(&workers)
            {
                eprintln!("Warning: Failed to assign pending chunks: {}", e);
            }
        }

        let updated_job = brain
            .control_plane
            .jobs
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get_job(&job.job_id)
            .unwrap_or(job);
        json!({
            "status": "SUCCESS",
            "job": updated_job
        })
    }
);

route_handler!(
    ControlPlaneJobGetHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        // Extract job_id from path /api/v1/control_plane/jobs/:job_id or query or body
        let mut job_id = req
            .path
            .strip_prefix("/api/v1/control_plane/jobs/")
            .unwrap_or("")
            .to_string();
        if job_id.is_empty() {
            job_id = req
                .query_params
                .get("job_id")
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or_else(|| {
                    req.body
                        .get("job_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string()
                });
        }

        if job_id.is_empty() {
            let all_jobs = brain
                .control_plane
                .jobs
                .read()
                .unwrap_or_else(|p| p.into_inner())
                .list_jobs();
            return json!({
                "status": "SUCCESS",
                "count": all_jobs.len(),
                "jobs": all_jobs
            });
        }

        match brain
            .control_plane
            .jobs
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get_job(&job_id)
        {
            Some(j) => json!({
                "status": "SUCCESS",
                "job": j
            }),
            None => json!({
                "status": "ERROR",
                "error": format!("Job '{}' not found", job_id)
            }),
        }
    }
);

route_handler!(
    ControlPlaneProvidersHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let providers = brain
            .control_plane
            .providers
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .list_providers();
        json!({
            "status": "SUCCESS",
            "count": providers.len(),
            "providers": providers
        })
    }
);

route_handler!(ControlPlaneProviderDeployHandler, |req: &RouteRequest,
                                                   brain: &Arc<
    TaraBrain,
>| {
    let provider = req
        .body
        .get("provider")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if provider.is_empty() {
        return json!({ "status": "ERROR", "error": "provider is required" });
    }
    let config = req.body.get("config").cloned().unwrap_or(json!({}));
    match brain
        .control_plane
        .providers
        .write()
        .unwrap_or_else(|p| p.into_inner())
        .deploy(provider, &config)
    {
        Ok(receipt) => receipt,
        Err(e) => json!({ "status": "ERROR", "error": e }),
    }
});

route_handler!(
    ControlPlaneManifestHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| { brain.control_plane.manifest.get_manifest() }
);

// ── Acoustic Synthesizer Helper ───────────────────────────────────────────────

pub fn generate_acoustic_wav_bytes(text: &str, lang: &str) -> Result<Vec<u8>, String> {
    crate::skills::multimedia::synthesize_speech(text, lang, None)
}

// ── Creator Setup & Device Management Handlers ───────────────────────────────

route_handler!(
    CreatorSetupHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let google_id_token = req
            .body
            .get("google_id_token")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if google_id_token.is_empty() {
            return json!({ "status": "ERROR", "error": "Google ID token is required for root creator setup." });
        }

        let confirm_identity = req
            .body
            .get("confirm_identity")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let device_name = req
            .body
            .get("device_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Primary PC");
        let device_public_key = req.body.get("device_public_key").and_then(|v| v.as_str());
        let passphrase = req.body.get("passphrase").and_then(|v| v.as_str());
        let nonce = req
            .body
            .get("nonce")
            .or_else(|| req.body.get("challenge_nonce"))
            .and_then(|v| v.as_str());

        let setup_req = tara_security::creator::CreatorCeremonyRequest {
            google_id_token,
            confirm_identity,
            device_name,
            device_public_key,
            passphrase,
            nonce,
            client_ip: Some(&req.client_ip),
        };

        brain.creator_auth_service.setup_creator(
            &setup_req,
            &brain.identity_manager,
            Some(&brain.centralized_device_registry),
        )
    }
);

route_handler!(
    CreatorRegisterDeviceHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let session_token = req
            .body
            .get("session_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                req.headers
                    .get("authorization")
                    .and_then(|h| {
                        h.strip_prefix("Bearer ")
                            .or_else(|| h.strip_prefix("bearer "))
                    })
                    .map(|s| s.trim().to_string())
            });

        let google_id_token = req.body.get("google_id_token").and_then(|v| v.as_str());

        let mut authenticated_creator = None;
        if let Some(tok) = session_token.as_ref() {
            if let Some(sess) = brain.creator_auth_service.verify_session(tok) {
                authenticated_creator = sess.get("creator_id").cloned();
            }
        } else if let Some(gtok) = google_id_token {
            let google_nonce = req
                .body
                .get("google_nonce")
                .or(req.body.get("nonce"))
                .or(req.body.get("challenge_nonce"))
                .and_then(|v| v.as_str());
            let auth_res = brain.creator_auth_service.authenticate_google_token(
                gtok,
                google_nonce,
                &req.client_ip,
            );
            if auth_res.get("status").and_then(|v| v.as_str()) == Some("SUCCESS") {
                authenticated_creator = auth_res
                    .get("creator_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }
        }

        if authenticated_creator.as_deref() != Some("ROOT_OPERATOR") {
            return json!({ "status": "UNAUTHORIZED", "error": "Root creator authentication required to register devices." });
        }

        let device_pub = req
            .body
            .get("device_public_key")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if device_pub.is_empty() {
            return json!({ "status": "ERROR", "error": "device_public_key is required." });
        }

        let device_name = req
            .body
            .get("device_name")
            .and_then(|v| v.as_str())
            .unwrap_or("Secondary PC");
        let dev_rec = match brain.identity_manager.register_device(
            device_pub,
            device_name,
            "AUTHORIZED",
            &crate::now_iso(),
        ) {
            Ok(rec) => {
                if let Ok(dev_id) = tara_security::primitives::DeviceId::new(&rec.device_id) {
                    if let Ok(pub_bytes) = hex::decode(&rec.device_public_key) {
                        if let Ok(principal) =
                            tara_security::primitives::PrincipalId::new("creator_root_operator")
                        {
                            let zero_rec = tara_security::zero_trust::DeviceEnrollmentRecord {
                                device_id: dev_id,
                                enrolled_public_key: pub_bytes,
                                bound_principal_id: principal,
                                is_revoked: false,
                            };
                            let _ = brain.centralized_device_registry.enroll_device(zero_rec);
                        }
                    }
                }
                rec
            }
            Err(e) => return json!({ "status": "ERROR", "error": e }),
        };

        json!({
            "status": "SUCCESS",
            "device": dev_rec,
            "session": {
                "session_token": session_token.unwrap_or_default(),
                "creator_id": "ROOT_OPERATOR"
            }
        })
    }
);

route_handler!(
    CreatorRevokeDeviceHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let session_token = req
            .body
            .get("session_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                req.headers
                    .get("authorization")
                    .and_then(|h| {
                        h.strip_prefix("Bearer ")
                            .or_else(|| h.strip_prefix("bearer "))
                    })
                    .map(|s| s.trim().to_string())
            });

        let verified = session_token
            .as_ref()
            .and_then(|t| brain.creator_auth_service.verify_session(t))
            .is_some();

        if !verified {
            return json!({ "status": "UNAUTHORIZED", "error": "Creator session required to revoke devices." });
        }

        let revoke_all = req
            .body
            .get("all")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if revoke_all {
            for dev in brain.identity_manager.list_devices() {
                if let Ok(dev_id) = tara_security::primitives::DeviceId::new(&dev.device_id) {
                    let _ = brain.centralized_device_registry.revoke_device(&dev_id);
                }
            }
            brain.identity_manager.revoke_all_devices();
            return json!({ "status": "SUCCESS", "message": "All devices revoked." });
        }

        let device_id = req
            .body
            .get("device_id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if device_id.is_empty() {
            return json!({ "status": "ERROR", "error": "device_id is required." });
        }

        let success = brain.identity_manager.revoke_device(device_id);
        if success {
            if let Ok(dev_id) = tara_security::primitives::DeviceId::new(device_id) {
                let _ = brain.centralized_device_registry.revoke_device(&dev_id);
            }
            json!({ "status": "SUCCESS", "device_id": device_id, "device_status": "REVOKED" })
        } else {
            json!({ "status": "ERROR", "error": format!("Device '{}' not found.", device_id) })
        }
    }
);

route_handler!(
    CreatorDevicesHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let session_token = req
            .query_params
            .get("session_token")
            .and_then(|v| v.first())
            .cloned()
            .or_else(|| {
                req.headers
                    .get("authorization")
                    .and_then(|h| {
                        h.strip_prefix("Bearer ")
                            .or_else(|| h.strip_prefix("bearer "))
                    })
                    .map(|s| s.trim().to_string())
            });

        let verified = session_token
            .as_ref()
            .and_then(|t| brain.creator_auth_service.verify_session(t))
            .is_some();

        if !verified {
            return json!({ "status": "UNAUTHORIZED", "error": "Creator session required." });
        }

        let devs = brain.identity_manager.list_devices();
        json!({ "status": "SUCCESS", "devices": devs })
    }
);

// ── Voice Handlers ─────────────────────────────────────────────────────────────

route_handler!(
    VoiceTranscribeHandler,
    |req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        let audio_b64 = req
            .body
            .get("audio_data")
            .or_else(|| req.body.get("audio_base64"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if audio_b64.is_empty() {
            return json!({ "status": "ERROR", "error": "Missing 'audio_data' base64 field." });
        }

        let decoded = match base64::engine::general_purpose::STANDARD.decode(audio_b64.as_bytes()) {
            Ok(bytes) => bytes,
            Err(e) => {
                return json!({ "status": "ERROR", "error": format!("Base64 decoding failed: {}", e) })
            }
        };

        let lang = req
            .body
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en-US");
        let segments = match crate::skills::multimedia::transcribe_audio_bytes(&decoded, lang) {
            Ok(segments) => segments,
            Err(error) => return json!({"status":"ERROR","error":error}),
        };
        let transcript = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        json!({
            "status": "SUCCESS",
            "data": {
                "transcript": transcript,
                "language": lang,
                "segments": segments,
                "audio_bytes_length": decoded.len()
            }
        })
    }
);

route_handler!(
    VoiceSynthesizeHandler,
    |req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        let text = req
            .body
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim();
        if text.is_empty() {
            return json!({ "status": "ERROR", "error": "Missing 'text' field." });
        }

        let lang = req
            .body
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en");
        let wav_bytes = match generate_acoustic_wav_bytes(text, lang) {
            Ok(bytes) => bytes,
            Err(error) => return json!({"status":"ERROR","error":error}),
        };
        let wav_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

        json!({
            "status": "SUCCESS",
            "audio_base64": wav_b64,
            "format": "audio/wav",
            "sample_rate": 16000
        })
    }
);

route_handler!(
    VoiceConverseHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let mut input_text = req
            .body
            .get("input_text")
            .or_else(|| req.body.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let lang = req
            .body
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en-US")
            .to_string();

        // Check if audio data provided instead
        let audio_b64 = req
            .body
            .get("audio_data")
            .or_else(|| req.body.get("audio_base64"))
            .and_then(|v| v.as_str());

        if input_text.is_empty() {
            if let Some(b64) = audio_b64 {
                let audio = match base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()) {
                    Ok(audio) => audio,
                    Err(error) => {
                        return json!({"status":"ERROR","error":format!("Invalid audio base64: {error}")})
                    }
                };
                let segments =
                    match crate::skills::multimedia::transcribe_audio_bytes(&audio, &lang) {
                        Ok(segments) => segments,
                        Err(error) => return json!({"status":"ERROR","error":error}),
                    };
                input_text = segments
                    .iter()
                    .map(|s| s.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
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
        let require_wake_word = req
            .body
            .get("require_wake_word")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        if require_wake_word {
            let spot_res = brain.wake_word_spotter.spot_keyword(&input_text);
            if !spot_res.detected {
                return json!({
                    "status": "WAKE_WORD_NOT_DETECTED",
                    "transcript": input_text,
                    "wake_words": ["tara", "hey tara", "ತಾರಾ", "ಓ ತಾರಾ", "namaskara tara", "ನಮಸ್ಕಾರ ತಾರಾ"]
                });
            }
        }

        let mut ctx: HashMap<String, Value> = HashMap::new();
        ctx.insert("voice_turn".to_string(), json!(true));
        ctx.insert(
            "request_id".to_string(),
            json!(uuid::Uuid::new_v4().to_string()),
        );

        let actor = req.actor.as_deref().unwrap_or("user");
        let brain_res = brain.process(actor, &input_text, ctx);

        let response_text = brain_res
            .get("final_response")
            .or_else(|| brain_res.get("response"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if response_text.is_empty() {
            return json!({"status":"ERROR","error":"Brain did not return a response"});
        }

        brain.barge_in.start_speaking();

        let wav_bytes = match generate_acoustic_wav_bytes(&response_text, &lang) {
            Ok(bytes) => bytes,
            Err(error) => {
                brain.barge_in.stop_speaking();
                return json!({"status":"ERROR","error":error});
            }
        };
        let wav_b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let turn_id = uuid::Uuid::new_v4().to_string();

        brain.barge_in.stop_speaking();

        json!({
            "status": "SUCCESS",
            "transcript": input_text,
            "response_text": response_text,
            "audio_base64": wav_b64,
            "language": lang,
            "turn_id": turn_id,
            "brain_result": brain_res
        })
    }
);

route_handler!(
    VoiceBargeInHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let interrupted = brain.barge_in.trigger_barge_in();
        json!({
            "status": "SUCCESS",
            "interrupted": interrupted,
            "turn_state": format!("{:?}", brain.barge_in.get_state()),
            "total_interruptions": brain.barge_in.total_interruptions()
        })
    }
);

route_handler!(
    VoiceCapabilitiesHandler,
    |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        json!({
            "status": "SUCCESS",
            "stt_available": std::env::var_os("TARA_WHISPER_BIN").is_some() || std::process::Command::new("whisper").arg("--help").output().is_ok(),
            "tts_available": cfg!(any(target_os = "windows", target_os = "linux")),
            "languages": ["en-US", "en-IN", "kn-IN"],
            "barge_in_supported": true,
            "wake_words": ["tara", "hey tara", "ತಾರಾ", "ಓ ತಾರಾ", "namaskara tara", "ನಮಸ್ಕಾರ ತಾರಾ"]
        })
    }
);

// ── Robotics HAL Handlers ───────────────────────────────────────────────────

route_handler!(
    RoboticsStatusHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let estop = brain.robotics_hal.is_estop_active();
        let joints = brain.robotics_hal.list_joints();
        json!({
            "status": "SUCCESS",
            "estop_active": estop,
            "joints_count": joints.len(),
            "joints": joints
        })
    }
);

route_handler!(
    RoboticsEstopHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let action = req
            .body
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("trigger");

        if action == "clear" || action == "reset" {
            if !is_privileged_actor(req.actor.as_deref()) {
                return json!({
                    "status": "UNAUTHORIZED",
                    "error": "Clearing E-Stop requires operator or creator authority"
                });
            }
            brain.robotics_hal.clear_estop();
            json!({
                "status": "SUCCESS",
                "action": "E_STOP_CLEARED",
                "estop_active": false
            })
        } else {
            brain.robotics_hal.trigger_estop();
            json!({
                "status": "SUCCESS",
                "action": "E_STOP_TRIGGERED",
                "estop_active": true
            })
        }
    }
);

route_handler!(
    RoboticsJointHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let joint_id = req
            .body
            .get("joint_id")
            .and_then(|v| v.as_str())
            .unwrap_or("joint_base");

        let dt = req.body.get("dt").and_then(|v| v.as_f64()).unwrap_or(0.01);

        // Optional raw sensor feedback ingestion
        if let Some(measured_rad) = req.body.get("sensor_measurement").and_then(|v| v.as_f64()) {
            let _ = brain
                .robotics_hal
                .process_sensor_feedback(joint_id, measured_rad, dt);
        }

        if let Some(target) = req.body.get("target_position_rad").and_then(|v| v.as_f64()) {
            match brain.robotics_hal.command_and_step(joint_id, target, dt) {
                Ok((effort, joint)) => json!({
                    "status": "SUCCESS",
                    "joint_id": joint_id,
                    "effort_nm": effort,
                    "joint": joint
                }),
                Err(e) => json!({
                    "status": "ERROR",
                    "error": e
                }),
            }
        } else {
            match brain.robotics_hal.get_joint(joint_id) {
                Some(joint) => json!({
                    "status": "SUCCESS",
                    "joint": joint
                }),
                None => json!({
                    "status": "ERROR",
                    "error": format!("Joint '{}' not found", joint_id)
                }),
            }
        }
    }
);

// ── Biometric Handlers ────────────────────────────────────────────────────────

route_handler!(
    BiometricCapabilitiesHandler,
    |_req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        let caps = tara_security::hardware_root::BiometricFactorProvider::detect_capabilities();
        json!({
            "status": "SUCCESS",
            "capabilities": caps
        })
    }
);

route_handler!(
    BiometricChallengeHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let user_id = req
            .body
            .get("creator_id")
            .or_else(|| req.body.get("user_id"))
            .and_then(|v| v.as_str())
            .unwrap_or("ROOT_OPERATOR");
        let session_id = req
            .body
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let action = req
            .body
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("general_access");

        let challenge = brain
            .biometric_provider
            .issue_challenge(user_id, session_id, action);
        json!({
            "status": "SUCCESS",
            "challenge": challenge,
            "challenge_nonce": challenge.nonce,
            "challenge_token": challenge.nonce,
            "creator_id": user_id
        })
    }
);

route_handler!(
    BiometricVerifyHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let expected_session = req
            .body
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("default");
        let expected_action = req
            .body
            .get("action")
            .and_then(|v| v.as_str())
            .unwrap_or("general_access");
        let max_age = req
            .body
            .get("max_age_seconds")
            .and_then(|v| v.as_u64())
            .unwrap_or(60);

        // Path A: Gated issuance via platform biometric assertion (C-28)
        if let (Some(ch_val), Some(as_val)) = (req.body.get("challenge"), req.body.get("assertion"))
        {
            let ch_parsed: Result<tara_security::hardware_root::BiometricChallenge, _> =
                serde_json::from_value(ch_val.clone());
            let as_parsed: Result<tara_security::hardware_root::PlatformBiometricAssertion, _> =
                serde_json::from_value(as_val.clone());

            match (ch_parsed, as_parsed) {
                (Ok(ch), Ok(as_proof)) => {
                    match brain
                        .biometric_provider
                        .verify_platform_assertion_and_mint_token(&ch, &as_proof)
                    {
                        Ok(minted_token) => json!({
                            "status": "SUCCESS",
                            "verified": true,
                            "creator_id": ch.creator_id,
                            "factor_token": minted_token
                        }),
                        Err(e) => json!({
                            "status": "ERROR",
                            "verified": false,
                            "error": e
                        }),
                    }
                }
                _ => json!({
                    "status": "ERROR",
                    "verified": false,
                    "error": "Malformed challenge or assertion structure"
                }),
            }
        } else {
            // Path B: Verification of previously minted factor token
            let token = req
                .body
                .get("challenge_token")
                .or_else(|| req.body.get("token"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            match brain.biometric_provider.verify_challenge_token(
                token,
                expected_session,
                expected_action,
                max_age,
            ) {
                Ok(creator_id) => json!({
                    "status": "SUCCESS",
                    "verified": true,
                    "creator_id": creator_id
                }),
                Err(e) => json!({
                    "status": "ERROR",
                    "verified": false,
                    "error": e
                }),
            }
        }
    }
);

// ── Emergency Self-Destruction Handler ────────────────────────────────────────

route_handler!(
    EmergencyDestroyHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let passphrase = req
            .body
            .get("confirm_passphrase")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if passphrase != "PERMANENT_SELF_DESTRUCT_CONFIRMED" {
            return json!({
                "status": "UNAUTHORIZED",
                "error": "Emergency destruction requires explicit passphrase 'PERMANENT_SELF_DESTRUCT_CONFIRMED'"
            });
        }

        let keystore_path = std::path::Path::new(&brain.repo_root).join(".keys");
        let session_path = std::path::Path::new(&brain.repo_root).join(".sessions");
        let vault_path = std::path::Path::new(&brain.repo_root)
            .join("storage")
            .join("vault");
        let tara_access_path = std::path::Path::new(&brain.repo_root)
            .join("TARA")
            .join("ACCESS");

        let destroy_res = tara_core::SecureDestructionEngine::trigger_emergency_destruction(
            &keystore_path,
            &session_path,
        );
        let _ = std::fs::remove_dir_all(&vault_path);
        let _ = std::fs::remove_dir_all(&tara_access_path);

        match destroy_res {
            Ok(report) => json!({
                "status": "DESTROYED",
                "report": report
            }),
            Err(e) => json!({
                "status": "ERROR",
                "error": e
            }),
        }
    }
);

route_handler!(
    CreatorSetupInitializeHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let passphrase = req
            .body
            .get("passphrase")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let creator_id = req.body.get("creator_id").and_then(|v| v.as_str());
        if let Some(cid) = creator_id {
            if cid != tara_security::primitives::CANONICAL_CREATOR_ID {
                return json!({ "status": "ERROR", "error": "Only canonical ROOT_OPERATOR setup allowed." });
            }
        }
        let display_name = req
            .body
            .get("display_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let google_email = req
            .body
            .get("google_email")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let device_name = req
            .body
            .get("device_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let setup_req = tara_security::creator::CreatorSetupRequest {
            display_name,
            passphrase: passphrase.to_string(),
            google_email,
            device_name,
        };

        match brain.creator_setup.initialize_creator_trust_root(setup_req) {
            Ok(res) => json!({
                "status": "SUCCESS",
                "result": res
            }),
            Err(e) => json!({
                "status": "ERROR",
                "error": e
            }),
        }
    }
);

route_handler!(
    AccessQrGenerateHandler,
    |req: &RouteRequest, _brain: &Arc<TaraBrain>| {
        let text = req
            .body
            .get("data")
            .or_else(|| req.body.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if text.is_empty() {
            return json!({"status": "ERROR", "error": "data field is required"});
        }
        match tara_security::qr::QrCodeMatrix::encode(text) {
            Ok(matrix) => json!({
                "status": "SUCCESS",
                "size": matrix.size,
                "ascii": matrix.render_ascii(),
                "svg": matrix.render_svg(8)
            }),
            Err(e) => json!({
                "status": "ERROR",
                "error": e
            }),
        }
    }
);

route_handler!(
    SecurityJailbreakInspectHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let input = req.body.get("input").and_then(|v| v.as_str()).unwrap_or("");
        let verdict = brain.jailbreak_detector.inspect(input);
        json!({
            "status": "SUCCESS",
            "verdict": verdict
        })
    }
);

route_handler!(SecurityQuarantineEvaluateHandler, |req: &RouteRequest,
                                                   brain: &Arc<
    TaraBrain,
>| {
    let artifact_type = req
        .body
        .get("artifact_type")
        .and_then(|v| v.as_str())
        .unwrap_or("generic");
    let name = req
        .body
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("artifact");
    let content = req
        .body
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    match brain.quarantine_engine.evaluate_artifact(
        name,
        artifact_type,
        content.as_bytes(),
        epoch_secs,
    ) {
        Ok(record) => json!({
            "status": "SUCCESS",
            "record": record
        }),
        Err(e) => json!({
            "status": "FAILED",
            "error": e
        }),
    }
});

route_handler!(
    SecurityProvenanceCheckHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let target_license = req
            .body
            .get("target_license")
            .and_then(|v| v.as_str())
            .unwrap_or("MIT");
        let report = brain
            .license_provenance
            .check_copyleft_conflicts(target_license);
        json!({
            "status": "SUCCESS",
            "report": report
        })
    }
);

route_handler!(
    SkillsCertifyHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let skill_id = req
            .body
            .get("skill_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let target = req
            .body
            .get("target_state")
            .and_then(|v| v.as_str())
            .unwrap_or("VALIDATED");
        let target_state = match target {
            "CERTIFIED" => crate::skills::CertificationState::Certified,
            "EXPERIMENTAL" => crate::skills::CertificationState::Experimental,
            "REJECTED" => crate::skills::CertificationState::Rejected,
            _ => crate::skills::CertificationState::Validated,
        };

        let traces: Vec<crate::skills::SkillTraceSample> = req
            .body
            .get("traces")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        match brain
            .skill_certifier
            .evaluate_and_advance(skill_id, &traces, target_state)
        {
            Ok(rec) => json!({
                "status": "SUCCESS",
                "record": rec
            }),
            Err(e) => json!({
                "status": "ERROR",
                "error": e
            }),
        }
    }
);

route_handler!(LearningContinualEvaluateHandler, |req: &RouteRequest,
                                                  brain: &Arc<
    TaraBrain,
>| {
    let task_id = req
        .body
        .get("task_id")
        .and_then(|v| v.as_str())
        .unwrap_or("task_0");
    let current_loss = req
        .body
        .get("current_loss")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.5);
    let current_accuracy = req
        .body
        .get("current_accuracy")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.9);
    let param_dist = req
        .body
        .get("param_distance_sq")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01);

    let eval = brain.continual_governor.evaluate_task_degradation(
        task_id,
        current_loss,
        current_accuracy,
        param_dist,
    );
    json!({
        "status": "SUCCESS",
        "evaluation": eval
    })
});

route_handler!(
    CognitiveReasoningSolveHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let candidates_val = req.body.get("candidates").and_then(|v| v.as_array());
        let candidates: Vec<String> = candidates_val
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let results = brain
            .cognitive
            .constraint_solver
            .solve_and_rank(&candidates);
        json!({
            "status": "SUCCESS",
            "results": results
        })
    }
);

route_handler!(
    CognitiveCuriosityAgendaHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let agenda = brain.cognitive.curiosity.get_ranked_exploration_agenda();
        json!({
            "status": "SUCCESS",
            "agenda": agenda,
            "stats": brain.cognitive.curiosity.status_json()
        })
    }
);

route_handler!(CognitiveExperientialCycleHandler, |req: &RouteRequest,
                                                   brain: &Arc<
    TaraBrain,
>| {
    let category = req
        .body
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("general");
    let description = req
        .body
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("task execution");
    let context = req.body.get("context").cloned().unwrap_or(json!({}));

    let episode = brain.cognitive.experiential.execute_closed_loop(
        category,
        description,
        &context,
        |_strat, _ctx| true,
        |_strat| Ok(json!({"execution": "ok"})),
    );

    json!({
        "status": "SUCCESS",
        "episode": episode
    })
});

route_handler!(
    CognitiveOntologyPathHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let start = req
            .body
            .get("start_concept")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let end = req
            .body
            .get("end_concept")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let max_depth = req
            .body
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as usize;

        let path = brain.cognitive.ontology.find_path(start, end, max_depth);
        json!({
            "status": "SUCCESS",
            "path": path,
            "graph_stats": brain.cognitive.ontology.stats()
        })
    }
);

route_handler!(
    MathEngineRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let op = req
            .body
            .get("operation")
            .or_else(|| req.body.get("op"))
            .and_then(Value::as_str)
            .unwrap_or("add");
        match brain.math_engine.evaluate(op, &req.body) {
            Ok(res) => json!({ "status": "SUCCESS", "result": res }),
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    ScienceEngineRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let discipline = req
            .body
            .get("discipline")
            .and_then(Value::as_str)
            .unwrap_or("physics");
        let op = req
            .body
            .get("operation")
            .or_else(|| req.body.get("op"))
            .and_then(Value::as_str)
            .unwrap_or("force");
        match brain.science_engine.evaluate(discipline, op, &req.body) {
            Ok(res) => json!({ "status": "SUCCESS", "result": res }),
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    ProgrammingEngineRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let op = req
            .body
            .get("operation")
            .or_else(|| req.body.get("op"))
            .and_then(Value::as_str)
            .unwrap_or("analyze_code");

        if (op == "cargo_check" || op == "cargo_test") && !is_privileged_actor(req.actor.as_deref())
        {
            return json!({
                "status": "UNAUTHORIZED",
                "error": "Privileged operation (cargo_check/cargo_test) requires operator or admin role"
            });
        }

        match brain.programming_engine.evaluate(op, &req.body) {
            Ok(res) => json!({ "status": "SUCCESS", "result": res }),
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    SpecialistDispatchRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let domain = req
            .body
            .get("domain")
            .and_then(Value::as_str)
            .unwrap_or("math");
        let op = req
            .body
            .get("operation")
            .or_else(|| req.body.get("op"))
            .and_then(Value::as_str)
            .unwrap_or("evaluate");

        if domain == "programming"
            && (op == "cargo_check" || op == "cargo_test")
            && !is_privileged_actor(req.actor.as_deref())
        {
            return json!({
                "status": "UNAUTHORIZED",
                "error": "Privileged operation (cargo_check/cargo_test) requires operator or admin role"
            });
        }

        match brain.specialist_engines.dispatch(domain, op, &req.body) {
            Ok(res) => {
                json!({ "status": "SUCCESS", "domain": domain, "operation": op, "result": res })
            }
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    LanguageAnalyzeRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let text = req.body.get("text").and_then(Value::as_str).unwrap_or("");
        let profile = brain.language_engine.analyze(text);
        json!({ "status": "SUCCESS", "profile": profile })
    }
);

route_handler!(
    LanguageTerminologyRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let term = req.body.get("term").and_then(Value::as_str).unwrap_or("");
        match brain.language_engine.lookup_technical_term(term) {
            Some(translation) => {
                json!({ "status": "SUCCESS", "term": term, "translation": translation })
            }
            None => json!({ "status": "NOT_FOUND", "term": term }),
        }
    }
);

route_handler!(
    RewardEventRouteHandler,
    |req: &RouteRequest, brain: &Arc<TaraBrain>| {
        if !is_privileged_actor(req.actor.as_deref()) {
            return json!({
                "status": "UNAUTHORIZED",
                "error": "Recording reward events requires operator or creator authority"
            });
        }
        let actor_id = req.actor.as_deref().unwrap_or("creator");
        let category_str = req
            .body
            .get("category")
            .and_then(Value::as_str)
            .unwrap_or("task_completion");
        let category = category_str
            .parse()
            .unwrap_or(crate::reward::RewardCategory::TaskCompletion);
        let score = req.body.get("score").and_then(Value::as_f64).unwrap_or(0.0);
        let reason = req.body.get("reason").and_then(Value::as_str).unwrap_or("");
        match brain
            .reward_system
            .record_reward(actor_id, category, score, reason)
        {
            Ok(entry) => json!({ "status": "SUCCESS", "entry": entry }),
            Err(e) => json!({ "status": "ERROR", "error": e }),
        }
    }
);

route_handler!(
    RewardSummaryRouteHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let summary = brain.reward_system.get_summary();
        json!({ "status": "SUCCESS", "summary": summary })
    }
);

route_handler!(
    RewardVerifyRouteHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let (valid, err) = brain.reward_system.verify_ledger_integrity();
        json!({ "status": "SUCCESS", "ledger_valid": valid, "error": err })
    }
);

route_handler!(
    RewardExportRouteHandler,
    |_req: &RouteRequest, brain: &Arc<TaraBrain>| {
        let records = brain.reward_system.get_events();
        json!({ "status": "SUCCESS", "count": records.len(), "records": records })
    }
);
// ── Router builder ─────────────────────────────────────────────────────────────

/// Register all routes into the provided router.
pub fn register_all_routes(router: &crate::server::ApiRouter) {
    // Specialist Engines
    router.register(
        "POST",
        "/api/v1/engines/math",
        MathEngineRouteHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/engines/science",
        ScienceEngineRouteHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/engines/programming",
        ProgrammingEngineRouteHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/engines/dispatch",
        SpecialistDispatchRouteHandler,
        true,
        None,
    );

    // Language Engine
    router.register(
        "POST",
        "/api/v1/language/analyze",
        LanguageAnalyzeRouteHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/language/terminology",
        LanguageTerminologyRouteHandler,
        false,
        None,
    );

    // Cryptographic Reward Ledger
    router.register(
        "POST",
        "/api/v1/reward/event",
        RewardEventRouteHandler,
        true,
        None,
    );
    router.register(
        "GET",
        "/api/v1/reward/summary",
        RewardSummaryRouteHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/reward/verify",
        RewardVerifyRouteHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/reward/export",
        RewardExportRouteHandler,
        true,
        Some("admin"),
    );

    // API Discovery & Health
    router.register("GET", "/", ApiDiscoveryHandler, false, None);
    router.register("GET", "/api", ApiDiscoveryHandler, false, None);
    router.register("GET", "/api/v1", ApiDiscoveryHandler, false, None);
    router.register("GET", "/health", HealthHandler, false, None);
    router.register("GET", "/api/v1/health", HealthHandler, false, None);
    router.register("GET", "/api/v1/status", StatusHandler, false, None);

    // Chat
    router.register("POST", "/api/v1/chat", ChatHandler, true, None);
    router.register("GET", "/api/v1/chat", ChatInfoHandler, false, None);
    router.register("POST", "/v1/chat", ChatHandler, true, None);
    router.register("GET", "/v1/chat", ChatInfoHandler, false, None);
    router.register(
        "POST",
        "/api/v1/inference",
        DirectInferenceHandler,
        true,
        None,
    );
    router.register("GET", "/api/v1/skills", SkillsHandler, true, None);

    // Auth routes (no auth required — they ARE the auth mechanism)
    router.register(
        "GET",
        "/api/v1/auth/google/challenge",
        AuthGoogleChallengeHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/google",
        AuthGoogleHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/creator_key",
        AuthCreatorKeyHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/recovery",
        AuthRecoveryHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/logout",
        AuthLogoutHandler,
        false,
        None,
    );
    router.register(
        "GET",
        "/api/v1/auth/session",
        AuthSessionHandler,
        false,
        None,
    );
    router.register(
        "GET",
        "/api/v1/auth/qr_status",
        AuthQrStatusHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/qr_challenge",
        AuthQrChallengeHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/qr_approve",
        AuthQrApproveHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/trigger_check",
        AuthTriggerCheckHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/auth/select_method",
        AuthSelectMethodHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/creator/enroll_google",
        CreatorEnrollGoogleHandler,
        true,
        Some("creator"),
    );
    router.register(
        "POST",
        "/api/v1/creator/setup",
        CreatorSetupHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/creator/register_device",
        CreatorRegisterDeviceHandler,
        true,
        Some("creator"),
    );
    router.register(
        "POST",
        "/api/v1/creator/revoke_device",
        CreatorRevokeDeviceHandler,
        true,
        Some("creator"),
    );
    router.register(
        "GET",
        "/api/v1/creator/devices",
        CreatorDevicesHandler,
        true,
        Some("creator"),
    );

    // Voice endpoints
    router.register(
        "POST",
        "/api/v1/voice/transcribe",
        VoiceTranscribeHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/voice/synthesize",
        VoiceSynthesizeHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/voice/converse",
        VoiceConverseHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/voice/barge_in",
        VoiceBargeInHandler,
        false,
        None,
    );
    router.register(
        "GET",
        "/api/v1/voice/capabilities",
        VoiceCapabilitiesHandler,
        false,
        None,
    );

    // Cluster / endpoints / sync
    router.register(
        "POST",
        "/api/v1/endpoints/register",
        EndpointRegisterHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/endpoints/list",
        EndpointListHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/endpoints/select",
        EndpointSelectHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/endpoints/probe",
        EndpointProbeHandler,
        true,
        Some("admin"),
    );
    router.register("POST", "/api/v1/sync/push", SyncPushHandler, true, None);
    router.register("GET", "/api/v1/sync/status", SyncStatusHandler, true, None);
    router.register("POST", "/api/v1/sync/flush", SyncFlushHandler, true, None);
    router.register(
        "GET",
        "/api/v1/cluster/telemetry",
        ClusterTelemetryHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/cluster/route",
        ClusterRouteHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/cluster/execute",
        ClusterExecuteHandler,
        true,
        None,
    );

    // Engines & Bridge
    router.register("GET", "/api/v1/engines/list", EngineListHandler, true, None);
    router.register(
        "POST",
        "/api/v1/engines/register",
        EngineRegisterHandler,
        true,
        Some("creator"),
    );
    router.register(
        "POST",
        "/api/v1/engines/execute",
        EngineExecuteHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/engines/quarantine",
        EngineQuarantineHandler,
        true,
        Some("creator"),
    );
    router.register(
        "GET",
        "/api/v1/engines/health",
        EngineHealthHandler,
        true,
        None,
    );
    router.register(
        "GET",
        "/api/v1/bridge/status",
        BridgeStatusHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/bridge/probe",
        BridgeProbeHandler,
        true,
        None,
    );

    // Admin (require admin role)
    router.register(
        "GET",
        "/api/v1/admin/overview",
        AdminOverviewHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/status",
        AdminOverviewHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/candidates",
        AdminCandidatesGetHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/knowledge",
        AdminKnowledgeGetHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/memory",
        AdminMemoryGetHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/rules",
        AdminRulesGetHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/security",
        AdminSecurityGetHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/admin/self-train-status",
        AdminSelfTrainStatusHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/approve_candidate",
        AdminApproveCandidateHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/knowledge",
        AdminKnowledgePostHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/knowledge/import",
        AdminKnowledgeImportHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/reasoning/export_to_dataset",
        AdminReasoningBridgeHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/skill_execute",
        AdminSkillExecuteHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/skill_add",
        AdminSkillAddHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/skill_remove",
        AdminSkillRemoveHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/device_authorize",
        AdminDeviceAuthorizeHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/admin/self-train",
        AdminSelfTrainPostHandler,
        true,
        Some("admin"),
    );

    // Dynamic Compute Control Plane
    router.register(
        "GET",
        "/api/v1/control_plane/status",
        ControlPlaneStatusHandler,
        false,
        None,
    );
    router.register(
        "GET",
        "/api/v1/control_plane/manifest",
        ControlPlaneManifestHandler,
        false,
        None,
    );
    router.register(
        "GET",
        "/api/v1/control_plane/workers",
        ControlPlaneWorkersHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/control_plane/workers/register",
        ControlPlaneWorkerRegisterHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/control_plane/workers/heartbeat",
        ControlPlaneWorkerHeartbeatHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/control_plane/workers/quarantine",
        ControlPlaneWorkerQuarantineHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/control_plane/workers/revoke",
        ControlPlaneWorkerRevokeHandler,
        true,
        Some("admin"),
    );
    router.register(
        "POST",
        "/api/v1/control_plane/jobs/submit",
        ControlPlaneJobSubmitHandler,
        true,
        Some("admin"),
    );
    router.register(
        "GET",
        "/api/v1/control_plane/jobs",
        ControlPlaneJobGetHandler,
        true,
        None,
    );
    router.register(
        "GET",
        "/api/v1/control_plane/jobs/:job_id",
        ControlPlaneJobGetHandler,
        true,
        None,
    );
    router.register(
        "GET",
        "/api/v1/control_plane/providers",
        ControlPlaneProvidersHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/control_plane/providers/deploy",
        ControlPlaneProviderDeployHandler,
        true,
        Some("admin"),
    );

    // Robotics HAL & Hardware Actuators
    router.register(
        "GET",
        "/api/v1/robotics/status",
        RoboticsStatusHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/robotics/estop",
        RoboticsEstopHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/robotics/joint",
        RoboticsJointHandler,
        true,
        Some("operator"),
    );

    // Platform Biometrics & Multi-Factor Access
    router.register(
        "GET",
        "/api/v1/access/biometric/capabilities",
        BiometricCapabilitiesHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/access/biometric/challenge",
        BiometricChallengeHandler,
        true,
        None,
    );
    router.register(
        "POST",
        "/api/v1/access/biometric/verify",
        BiometricVerifyHandler,
        true,
        None,
    );

    // Emergency DoD 5220.22-M Secure Destruction
    router.register(
        "POST",
        "/api/v1/system/emergency_destroy",
        EmergencyDestroyHandler,
        true,
        Some("creator"),
    );

    // Creator Setup & QR Code
    router.register(
        "POST",
        "/api/v1/access/setup/initialize",
        CreatorSetupInitializeHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/access/qr/generate",
        AccessQrGenerateHandler,
        false,
        None,
    );

    // Security & Quarantine
    router.register(
        "POST",
        "/api/v1/security/jailbreak/inspect",
        SecurityJailbreakInspectHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/security/quarantine/evaluate",
        SecurityQuarantineEvaluateHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/security/provenance/check",
        SecurityProvenanceCheckHandler,
        false,
        None,
    );

    // Skills Certification
    router.register(
        "POST",
        "/api/v1/skills/certify",
        SkillsCertifyHandler,
        true,
        Some("admin"),
    );

    // Continual Learning Governance
    router.register(
        "POST",
        "/api/v1/learning/continual/evaluate",
        LearningContinualEvaluateHandler,
        true,
        None,
    );

    // Advanced Cognitive Engines
    router.register(
        "POST",
        "/api/v1/cognitive/reasoning/solve",
        CognitiveReasoningSolveHandler,
        false,
        None,
    );
    router.register(
        "GET",
        "/api/v1/cognitive/curiosity/agenda",
        CognitiveCuriosityAgendaHandler,
        false,
        None,
    );
    router.register(
        "POST",
        "/api/v1/cognitive/experiential/cycle",
        CognitiveExperientialCycleHandler,
        true,
        Some("operator"),
    );
    router.register(
        "POST",
        "/api/v1/cognitive/ontology/path",
        CognitiveOntologyPathHandler,
        false,
        None,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_token_for_js_removes_xss_vectors() {
        let malicious = "token123\"; alert('XSS'); //";
        let sanitized = sanitize_token_for_js(malicious);
        assert_eq!(sanitized, "token123alertXSS");

        let html_tags = "<script>evil()</script>secret_token-456:789.abc";
        let sanitized_html = sanitize_token_for_js(html_tags);
        assert_eq!(sanitized_html, "scriptevilscriptsecret_token-456:789.abc");
    }

    #[test]
    fn test_sanitize_token_for_js_length_bound() {
        let long_token = "a".repeat(1000);
        let sanitized = sanitize_token_for_js(&long_token);
        assert_eq!(sanitized.len(), 512);
    }

    #[test]
    fn test_is_privileged_actor_access_control() {
        assert!(is_privileged_actor(Some("admin")));
        assert!(is_privileged_actor(Some("root")));
        assert!(is_privileged_actor(Some("system")));
        assert!(is_privileged_actor(Some("ROOT_OPERATOR")));
        assert!(is_privileged_actor(Some("OPERATOR_ROOT")));
        assert!(is_privileged_actor(Some("creator:primary")));
        assert!(is_privileged_actor(Some("api_key:admin_secret")));
        assert!(is_privileged_actor(Some("api_key:ROOT_OPERATOR")));
        assert!(is_privileged_actor(Some("api_key:operator_key")));
        assert!(is_privileged_actor(Some("operator_agent")));

        assert!(!is_privileged_actor(None));
        assert!(!is_privileged_actor(Some("anonymous")));
        assert!(!is_privileged_actor(Some("guest")));
        assert!(!is_privileged_actor(Some("regular_user")));
        assert!(!is_privileged_actor(Some("untrusted")));
    }

    #[test]
    fn test_render_chat_ui_sanitizes_token() {
        let dirty_token = "secret'; alert(1); '";
        let html = render_chat_ui(dirty_token);
        assert!(!html.contains("alert(1)"));
        assert!(html.contains("const WEB_SESSION_TOKEN = 'secretalert1';"));
    }

    #[test]
    fn test_unified_bridge_latency_sanitization() {
        let bridge = bridge::UnifiedBridge::new();
        bridge.record_rust_dispatch();
        bridge.record_failover(45.5);

        let telem = bridge.get_telemetry();
        assert_eq!(telem["metrics"]["dispatched_native_inferences"], 1);
        assert_eq!(telem["metrics"]["automatic_failovers"], 1);
        assert_eq!(telem["metrics"]["last_failover_latency_ms"], 45.5);

        // Negative latency is clamped to 0.0
        bridge.record_failover(-12.0);
        let telem = bridge.get_telemetry();
        assert_eq!(telem["metrics"]["last_failover_latency_ms"], 0.0);

        // NaN latency is clamped to 0.0
        bridge.record_failover(f64::NAN);
        let telem = bridge.get_telemetry();
        assert_eq!(telem["metrics"]["last_failover_latency_ms"], 0.0);

        // Infinity is clamped to 0.0
        bridge.record_failover(f64::INFINITY);
        let telem = bridge.get_telemetry();
        assert_eq!(telem["metrics"]["last_failover_latency_ms"], 0.0);
    }

    #[test]
    fn test_extract_bearer_token() {
        let mut headers = HashMap::new();
        assert_eq!(extract_bearer_token(&headers), None);

        headers.insert("authorization".into(), "Bearer my_secret_token_123".into());
        assert_eq!(
            extract_bearer_token(&headers),
            Some("my_secret_token_123".into())
        );

        headers.insert("authorization".into(), "bearer lowercase_token_456".into());
        assert_eq!(
            extract_bearer_token(&headers),
            Some("lowercase_token_456".into())
        );

        headers.insert("authorization".into(), "Basic dXNlcjpwYXNz".into());
        assert_eq!(extract_bearer_token(&headers), None);

        headers.insert("authorization".into(), "Bearer   ".into());
        assert_eq!(extract_bearer_token(&headers), None);
    }
}
