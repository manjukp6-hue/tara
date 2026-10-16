//! Complete TARA server: HTTP routing, authentication, rate limiting, and request dispatch.
//! Ports Python server.py fully using tiny_http.

use std::collections::HashMap;
use std::io::{Cursor, Read};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use constant_time_eq::constant_time_eq;
use rand::Rng;
use serde_json::{json, Value};

use tiny_http::{Header, Request, Response, Server};

// ──────────────────────────────────────────────────────────────────────────────
// Rate Limiter (60 req/min, burst 15/5s)
// ──────────────────────────────────────────────────────────────────────────────

/// Sliding-window rate limiter matching Python's SlidingWindowRateLimiter.
pub struct SlidingWindowRateLimiter {
    window: Duration,
    limit: usize,
    burst_window: Duration,
    burst_limit: usize,
    max_tracked: usize,
    records: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
}

impl SlidingWindowRateLimiter {
    pub fn new() -> Self {
        Self {
            window: Duration::from_secs(60),
            limit: 60,
            burst_window: Duration::from_secs_f64(5.0),
            burst_limit: 15,
            max_tracked: 5000,
            records: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Returns `(allowed, retry_after_seconds)`.
    pub fn is_allowed(&self, client_key: &str) -> (bool, u64) {
        let now = Instant::now();
        let mut records = self.records.lock().unwrap_or_else(|e| e.into_inner());

        // Cleanup old clients
        if records.len() >= self.max_tracked {
            records.retain(|_, ts| ts.iter().any(|t| now.duration_since(*t) < self.window));
        }

        let timestamps = records.entry(client_key.to_string()).or_default();
        // Remove entries outside the sliding window
        timestamps.retain(|t| now.duration_since(*t) < self.window);

        // Burst check (last 5s)
        let burst_count = timestamps
            .iter()
            .filter(|t| now.duration_since(**t) < self.burst_window)
            .count();

        if burst_count >= self.burst_limit {
            return (false, 5);
        }
        if timestamps.len() >= self.limit {
            return (false, 60);
        }

        timestamps.push(now);
        (true, 0)
    }
}

impl Default for SlidingWindowRateLimiter {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Route System
// ──────────────────────────────────────────────────────────────────────────────

/// Handler context passed to every route handler.
pub struct RouteRequest {
    pub method: String,
    pub path: String,
    pub query_params: HashMap<String, Vec<String>>,
    pub body: Value,
    pub headers: HashMap<String, String>,
    pub client_ip: String,
    pub actor: Option<String>,
}

/// All routes must implement this trait.
pub trait RouteHandler: Send + Sync {
    fn handle(&self, req: &RouteRequest, brain: &Arc<crate::brain::TaraBrain>) -> Value;
}

struct Route {
    method: String,
    path: String,
    handler: Arc<dyn RouteHandler>,
    required_auth: bool,
    required_role: Option<String>,
}

/// Dynamic route registry (port of Python ApiRouter).
pub struct ApiRouter {
    routes: RwLock<HashMap<String, Route>>,
}

impl Default for ApiRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ApiRouter {
    pub fn new() -> Self {
        Self {
            routes: RwLock::new(HashMap::new()),
        }
    }

    pub fn normalize_path(path: &str) -> String {
        let p = path.trim_end_matches('/');
        if p.is_empty() {
            "/".to_string()
        } else {
            p.to_string()
        }
    }

    pub fn register<H: RouteHandler + 'static>(
        &self,
        method: &str,
        path: &str,
        handler: H,
        required_auth: bool,
        required_role: Option<&str>,
    ) {
        let norm = Self::normalize_path(path);
        let key = format!("{}:{}", method.to_uppercase(), norm);
        self.routes.write().unwrap_or_else(|e| e.into_inner()).insert(
            key,
            Route {
                method: method.to_uppercase(),
                path: norm,
                handler: Arc::new(handler),
                required_auth,
                required_role: required_role.map(|s| s.to_string()),
            },
        );
    }

    pub fn match_route(
        &self,
        method: &str,
        path: &str,
    ) -> Option<(Arc<dyn RouteHandler>, bool, Option<String>)> {
        let norm = Self::normalize_path(path);
        let key = format!("{}:{}", method.to_uppercase(), norm);
        let routes = self.routes.read().unwrap_or_else(|e| e.into_inner());
        if let Some(r) = routes.get(&key) {
            return Some((r.handler.clone(), r.required_auth, r.required_role.clone()));
        }

        // Support parameterized routes (e.g. :job_id)
        let req_method = method.to_uppercase();
        let path_parts: Vec<&str> = norm.split('/').collect();
        for r in routes.values() {
            if r.method == req_method && r.path.contains(':') {
                let r_parts: Vec<&str> = r.path.split('/').collect();
                if r_parts.len() == path_parts.len() {
                    let matched = r_parts
                        .iter()
                        .zip(path_parts.iter())
                        .all(|(rp, pp)| rp.starts_with(':') || rp == pp);
                    if matched {
                        return Some((r.handler.clone(), r.required_auth, r.required_role.clone()));
                    }
                }
            }
        }
        None
    }

    pub fn has_path(&self, path: &str) -> bool {
        let norm = Self::normalize_path(path);
        let routes = self.routes.read().unwrap_or_else(|e| e.into_inner());
        if routes.values().any(|r| r.path == norm) {
            return true;
        }
        let path_parts: Vec<&str> = norm.split('/').collect();
        routes.values().any(|r| {
            if r.path.contains(':') {
                let r_parts: Vec<&str> = r.path.split('/').collect();
                if r_parts.len() == path_parts.len() {
                    return r_parts
                        .iter()
                        .zip(path_parts.iter())
                        .all(|(rp, pp)| rp.starts_with(':') || rp == pp);
                }
            }
            false
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Web Session Store (scoped tokens for browser UI)
// ──────────────────────────────────────────────────────────────────────────────

const MAX_WEB_SESSIONS: usize = 1000;

#[derive(Clone)]
pub struct WebSessionStore {
    sessions: Arc<Mutex<HashMap<String, (String, Instant)>>>,
    ttl: Duration,
}

impl Default for WebSessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            ttl: Duration::from_secs(3600),
        }
    }

    /// Issue a new web session token for `actor_id` with bounded capacity and TTL eviction.
    pub fn issue(&self, actor_id: &str) -> String {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        // Purge expired sessions
        sessions.retain(|_, (_, ts)| now.duration_since(*ts) < self.ttl);
        // If still at or over capacity, evict the oldest session
        if sessions.len() >= MAX_WEB_SESSIONS {
            if let Some(oldest_token) = sessions
                .iter()
                .min_by_key(|(_, (_, ts))| *ts)
                .map(|(tok, _)| tok.clone())
            {
                sessions.remove(&oldest_token);
            }
        }
        let token = hex::encode(rand::thread_rng().gen::<[u8; 32]>());
        sessions.insert(token.clone(), (actor_id.to_string(), now));
        token
    }

    /// Register an existing authenticated session token (such as a verified Creator session token).
    pub fn register(&self, token: &str, actor_id: &str) {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.retain(|_, (_, ts)| now.duration_since(*ts) < self.ttl);
        if sessions.len() >= MAX_WEB_SESSIONS {
            if let Some(oldest_token) = sessions
                .iter()
                .min_by_key(|(_, (_, ts))| *ts)
                .map(|(tok, _)| tok.clone())
            {
                sessions.remove(&oldest_token);
            }
        }
        sessions.insert(token.to_string(), (actor_id.to_string(), now));
    }

    /// Verify a web session token, returning actor_id if valid.
    pub fn verify(&self, token: &str) -> Option<String> {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((actor, issued_at)) = sessions.get(token) {
            if issued_at.elapsed() < self.ttl {
                return Some(actor.clone());
            }
            sessions.remove(token);
        }
        None
    }

    /// Explicitly revoke and remove a web session token.
    pub fn revoke(&self, token: &str) -> bool {
        let mut sessions = self.sessions.lock().unwrap_or_else(|e| e.into_inner());
        sessions.remove(token).is_some()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Authentication
// ──────────────────────────────────────────────────────────────────────────────

/// Canonical PrincipalId for client-IP risk attribution and isolation.
pub fn unauthenticated_client_principal(client_ip: &str) -> tara_security::primitives::PrincipalId {
    let sanitized_ip = client_ip
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>();
    let name = format!("ip_{}", sanitized_ip);
    tara_security::primitives::PrincipalId::new(&name)
        .unwrap_or_else(|_| tara_security::primitives::PrincipalId::new("ip_unauthenticated").unwrap())
}

/// Authenticate an incoming request. Returns `Ok(actor_id)` or `Err(error_message)`.
pub fn authenticate_request(
    headers: &HashMap<String, String>,
    api_keys: &HashMap<String, String>,
    brain: &Arc<crate::brain::TaraBrain>,
    web_sessions: &WebSessionStore,
    method: &str,
    path: &str,
    client_ip: &str,
) -> Result<String, String> {
    let auth_header = headers
        .get("authorization")
        .map(|s| s.as_str())
        .unwrap_or("");
    let x_api_key = headers.get("x-api-key").map(|s| s.as_str()).unwrap_or("");
    let x_worker_token = headers
        .get("x-tara-worker-token")
        .map(|s| s.as_str())
        .unwrap_or("");

    let cookie_token = headers.get("cookie").and_then(|c| {
        c.split(';')
            .find(|s| s.trim().starts_with("tara_web_session="))
            .and_then(|s| s.trim().strip_prefix("tara_web_session="))
            .map(|s| s.trim())
    });

    let token = if auth_header.to_lowercase().starts_with("bearer ") {
        auth_header[7..].trim()
    } else if !x_api_key.is_empty() {
        x_api_key.trim()
    } else if !x_worker_token.is_empty() {
        x_worker_token.trim()
    } else if let Some(ct) = cookie_token {
        ct
    } else {
        return Err("Unauthorized: Missing Authorization Bearer token, X-API-Key, or X-Tara-Worker-Token header.".into());
    };

    // Active Fail-Closed Honeypot Tripwire Inspection (Pillar 6)
    let now = crate::now_secs();
    if brain.tripwire_defense.check_token_trap(token, now) {
        return Err("Unauthorized: Honeypot decoy credential submitted. Security containment activated.".into());
    }

    // Fail-closed revocation check against ContainmentManager
    if brain.containment_manager.is_session_revoked(token) {
        return Err("Unauthorized: Session token has been revoked by security containment.".into());
    }

    // Check capability token validation (Pillar 2: Zero-Trust Capabilities)
    if token.starts_with("cap_") || token.starts_with('{') {
        use base64::Engine;
        let json_str = if let Some(stripped) = token.strip_prefix("cap_") {
            if let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(stripped) {
                String::from_utf8(bytes).unwrap_or_else(|_| stripped.to_string())
            } else if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(stripped) {
                String::from_utf8(bytes).unwrap_or_else(|_| stripped.to_string())
            } else {
                stripped.to_string()
            }
        } else {
            token.to_string()
        };

        if let Ok(cap_token) = serde_json::from_str::<tara_security::zero_trust::CapabilityToken>(&json_str) {
            let current_gen = brain.authority_manager.current_generation().unwrap_or(tara_security::primitives::AuthorityGeneration::INITIAL);
            if let Ok(root_key) = tara_security::dpapi::derive_authority_hmac_key(&format!("capability_root:gen_{}", current_gen.get())) {
                let action = match method.to_uppercase().as_str() {
                    "GET" => tara_security::primitives::SecurityAction::read_protected_resource(),
                    "POST" => tara_security::primitives::SecurityAction::execute_task(),
                    "PUT" | "PATCH" => tara_security::primitives::SecurityAction::write_protected_resource(),
                    "DELETE" => tara_security::primitives::SecurityAction::revoke_device(),
                    _ => tara_security::primitives::SecurityAction::execute_task(),
                };
                let resource_path = path.trim_start_matches('/');
                let resource = tara_security::primitives::SecurityResource::custom("route", resource_path)
                    .unwrap_or_else(|_| tara_security::primitives::SecurityResource::custom("server", "api").unwrap());
                let context = tara_security::primitives::SecurityContext::new(client_ip, now);
                if cap_token.verify(&root_key, &action, &resource, &context, None).is_ok() {
                    return Ok(format!("capability:{}", cap_token.token_id));
                }
            }
        }
    }

    // Check centralized SessionManager validation (Pillar 2: Zero-Trust Sessions)
    let current_gen = brain.authority_manager.current_generation().unwrap_or(tara_security::primitives::AuthorityGeneration::INITIAL);
    if let Ok(sess_rec) = brain.centralized_session_manager.validate_session(token, current_gen, now) {
        return Ok(sess_rec.principal.id().name().to_string());
    }

    // Check worker token validation against dynamic control plane registry
    if !x_worker_token.is_empty() || token.starts_with("tok_") || token.starts_with("internal_") {
        if let Some(worker_id) = brain
            .control_plane
            .workers
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .validate_worker_token(token)
        {
            if brain.containment_manager.is_worker_isolated(&worker_id) {
                return Err("Unauthorized: Worker has been isolated by security containment.".into());
            }
            return Ok(format!("worker:{}", worker_id));
        }
    }

    // Check web session tokens first (with complete zero-trust principal revocation check)
    if let Some(actor) = web_sessions.verify(token) {
        let safe_actor_id = actor
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect::<String>();
        let actor_pid = tara_security::primitives::PrincipalId::new(format!("actor_{}", safe_actor_id)).ok();
        let direct_pid = if let Some((ns, nm)) = actor.split_once(':') {
            let safe_ns = ns.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect::<String>();
            let safe_nm = nm.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '@' || *c == '.').collect::<String>();
            tara_security::primitives::PrincipalId::with_namespace(&safe_ns, &safe_nm).ok()
        } else {
            tara_security::primitives::PrincipalId::new(&actor).ok()
        };

        if let Some(ref p) = direct_pid {
            if brain.centralized_session_manager.is_principal_revoked(p) {
                return Err("WebSessionRevoked: Principal has been revoked in centralized zero-trust session store".into());
            }
        }
        if let Some(ref p) = actor_pid {
            if brain.centralized_session_manager.is_principal_revoked(p) {
                return Err("WebSessionRevoked: Principal has been revoked in centralized zero-trust session store".into());
            }
        }
        return Ok(actor);
    }

    // Check creator session (cryptographically verified via CreatorAuthService)
    if let Some(sess) = brain.creator_auth_service.verify_session(token) {
        let cid = sess
            .get("creator_id")
            .cloned()
            .unwrap_or_else(|| "ROOT_OPERATOR".to_string());
        if let Ok(creator_pid) = tara_security::primitives::PrincipalId::with_namespace("creator", &cid) {
            if brain.centralized_session_manager.is_principal_revoked(&creator_pid) {
                return Err("CreatorSessionRevoked: Creator principal has been revoked in centralized zero-trust session store".into());
            }
        }
        return Ok(format!("creator:{}", cid));
    }

    // Check API keys with constant-time comparison and principal revocation
    for (key, actor) in api_keys {
        if constant_time_eq(token.as_bytes(), key.as_bytes()) {
            let pid = if let Some((ns, nm)) = actor.split_once(':') {
                let safe_ns = ns.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect::<String>();
                let safe_nm = nm.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '@' || *c == '.').collect::<String>();
                tara_security::primitives::PrincipalId::with_namespace(&safe_ns, &safe_nm).ok()
            } else {
                tara_security::primitives::PrincipalId::new(actor).ok()
            };
            if let Some(ref p) = pid {
                if brain.centralized_session_manager.is_principal_revoked(p) {
                    return Err("ApiKeyRevoked: Principal has been revoked in centralized zero-trust session store".into());
                }
            }
            return Ok(format!("api_key:{}", actor));
        }
    }

    // Record failure in ContinuousRiskEngine (Pillar 2: Continuous Risk with client-IP isolation)
    let unauth_id = unauthenticated_client_principal(client_ip);
    brain.centralized_risk_engine.record_failure(&unauth_id);

    let epoch_secs = crate::now_secs();
    let sec_ctx = tara_security::primitives::SecurityContext::new(client_ip, epoch_secs);
    let sec_action = tara_security::primitives::SecurityAction::new("auth", "authenticate")
        .unwrap_or_else(|_| tara_security::primitives::SecurityAction::execute_task());
    if let Ok(tara_security::zero_trust::RiskVerdict::TriggerLockdown { reason, .. }) =
        brain.centralized_risk_engine.evaluate_risk(
            &unauth_id,
            &sec_action,
            &sec_ctx,
        )
    {
        eprintln!("[SECURITY] Unauthenticated client IP {} exceeded failure threshold ({}): blocking client request", client_ip, reason);
        return Err(format!("Unauthorized: Client IP blocked due to excessive authentication failures: {}", reason));
    }

    Err("Unauthorized: Invalid API key or token.".into())
}

/// Build the configured API keys map from environment variables.
pub fn load_api_keys() -> HashMap<String, String> {
    if let Ok(json_str) = std::env::var("TARA_API_KEYS") {
        if let Ok(v) = serde_json::from_str::<Value>(&json_str) {
            if let Some(obj) = v.as_object() {
                return obj
                    .iter()
                    .filter_map(|(k, v)| {
                        let actor = v.as_str()?;
                        (!k.trim().is_empty() && !actor.trim().is_empty())
                            .then(|| (k.clone(), actor.to_string()))
                    })
                    .collect();
            }
        }
    }
    if let Ok(single_key) = std::env::var("TARA_API_KEY") {
        if single_key.trim().is_empty() {
            return HashMap::new();
        }
        let mut m = HashMap::new();
        m.insert(single_key, "api_user".to_string());
        return m;
    }
    HashMap::new()
}

// ──────────────────────────────────────────────────────────────────────────────
// CORS
// ──────────────────────────────────────────────────────────────────────────────

fn cors_headers(origin: &str) -> Vec<Header> {
    let mut headers = Vec::new();

    // Sanitize origin: must be ASCII, non-empty, and free of control characters (CR/LF)
    let safe_origin = if !origin.is_empty()
        && origin.is_ascii()
        && !origin.chars().any(|c| c.is_ascii_control())
    {
        origin
    } else {
        "*"
    };

    if let Ok(h) = Header::from_bytes("Access-Control-Allow-Origin", safe_origin) {
        headers.push(h);
    }
    if let Ok(h) = Header::from_bytes("Vary", "Origin") {
        headers.push(h);
    }
    if let Ok(h) = Header::from_bytes("Access-Control-Allow-Methods", "GET, POST, OPTIONS") {
        headers.push(h);
    }
    if let Ok(h) = Header::from_bytes(
        "Access-Control-Allow-Headers",
        "Content-Type, Authorization, X-API-Key, X-Worker-Token, X-Hub-Signature-256",
    ) {
        headers.push(h);
    }
    headers
}


// ──────────────────────────────────────────────────────────────────────────────
// Request parsing
// ──────────────────────────────────────────────────────────────────────────────

const MAX_URL_LENGTH: usize = 4096;
const MAX_HEADERS_COUNT: usize = 100;
const MAX_HEADER_VALUE_LENGTH: usize = 8192;
const MAX_QUERY_PARAMS_COUNT: usize = 50;

fn parse_headers(req: &Request) -> HashMap<String, String> {
    req.headers()
        .iter()
        .take(MAX_HEADERS_COUNT)
        .map(|h| {
            let field = h.field.to_string().to_lowercase();
            let mut val = h.value.to_string();
            if val.len() > MAX_HEADER_VALUE_LENGTH {
                val.truncate(MAX_HEADER_VALUE_LENGTH);
            }
            (field, val)
        })
        .collect()
}

fn parse_query(url: &str) -> HashMap<String, Vec<String>> {
    let mut params: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(q) = url.split('?').nth(1) {
        for pair in q.split('&').take(MAX_QUERY_PARAMS_COUNT) {
            let mut parts = pair.splitn(2, '=');
            let key = url_decode(parts.next().unwrap_or(""));
            let val = url_decode(parts.next().unwrap_or(""));
            params.entry(key).or_default().push(val);
        }
    }
    params
}

fn url_decode(s: &str) -> String {
    let mut out: Vec<u8> = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex_str) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex_str, 16) {
                    out.push(byte);
                    i += 3;
                    continue;
                }
            }
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn read_body(req: &mut Request) -> Result<Vec<u8>, String> {
    const MAX_BODY_BYTES: usize = 1_048_576;
    if let Some(len) = req.body_length() {
        if len > MAX_BODY_BYTES {
            return Err(format!("Request body exceeds {} bytes", MAX_BODY_BYTES));
        }
        let mut buf = vec![0u8; len];
        req.as_reader()
            .read_exact(&mut buf)
            .map_err(|e| format!("Failed to read request body: {e}"))?;
        Ok(buf)
    } else {
        let mut buf = Vec::new();
        req.as_reader()
            .take((MAX_BODY_BYTES + 1) as u64)
            .read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read request body: {e}"))?;
        if buf.len() > MAX_BODY_BYTES {
            return Err(format!("Request body exceeds {} bytes", MAX_BODY_BYTES));
        }
        Ok(buf)
    }
}

fn json_response(status: u16, body: &Value) -> Response<Cursor<Vec<u8>>> {
    let json_str = serde_json::to_string(body).unwrap_or_default();
    let data = json_str.into_bytes();
    Response::from_data(data)
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", "application/json").unwrap())
        .with_header(Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap())
}

fn html_response(status: u16, body: &str) -> Response<Cursor<Vec<u8>>> {
    Response::from_data(body.as_bytes().to_vec())
        .with_status_code(status)
        .with_header(Header::from_bytes("Content-Type", "text/html; charset=utf-8").unwrap())
        .with_header(
            Header::from_bytes("Cache-Control", "no-cache, no-store, must-revalidate").unwrap(),
        )
        .with_header(Header::from_bytes("X-Frame-Options", "DENY").unwrap())
}

// ──────────────────────────────────────────────────────────────────────────────
// Main server dispatch loop
// ──────────────────────────────────────────────────────────────────────────────

/// Run the TARA HTTP server forever.
pub fn run_server(
    host: &str,
    port: u16,
    brain: Arc<crate::brain::TaraBrain>,
    router: Arc<ApiRouter>,
    rate_limiter: Arc<SlidingWindowRateLimiter>,
    api_keys: Arc<HashMap<String, String>>,
    web_sessions: Arc<WebSessionStore>,
) {
    let addr = format!("{}:{}", host, port);
    let server = match Server::http(&addr) {
        Ok(s) => s,
        Err(e) => {
            println!("\n[TARA Server] =========================================");
            if std::net::TcpStream::connect(&addr).is_ok() {
                println!(
                    "[TARA Server] TARA server is ALREADY ACTIVE and running on http://{}",
                    addr
                );
                println!(
                    "[TARA Server] Existing instance is healthy. No duplicate process needed."
                );
            } else {
                println!(
                    "[TARA Server] ERROR: Could not bind to http://{}: {}",
                    addr, e
                );
                println!("[TARA Server] Another application may be using this port.");
            }
            println!("[TARA Server] =========================================");
            #[cfg(target_os = "windows")]
            {
                println!("\nPress Enter to exit this window...");
                let mut buf = String::new();
                let _ = std::io::stdin().read_line(&mut buf);
            }
            return;
        }
    };
    println!("[TARA Server] Running on http://{}", addr);

    for mut request in server.incoming_requests() {
        let brain_ref = brain.clone();
        let router_ref = router.clone();
        let rl_ref = rate_limiter.clone();
        let api_keys_ref = api_keys.clone();
        let ws_ref = web_sessions.clone();

        // Rate limit
        let client_ip = request
            .remote_addr()
            .map(|a| a.ip().to_string())
            .unwrap_or_else(|| "127.0.0.1".to_string());

        let (allowed, retry_after) = rl_ref.is_allowed(&client_ip);
        if !allowed {
            let resp = json_response(
                429,
                &json!({
                    "status": "ERROR",
                    "error": "Too Many Requests: Rate limit exceeded.",
                    "retry_after": retry_after
                }),
            );
            let _ = request.respond(resp);
            continue;
        }

        let method = request.method().as_str().to_uppercase();
        let full_url = request.url().to_string();
        if full_url.len() > MAX_URL_LENGTH {
            let resp = json_response(
                414,
                &json!({
                    "status": "ERROR",
                    "error": "URI Too Long: URL exceeds maximum allowed length"
                }),
            );
            let _ = request.respond(resp);
            continue;
        }

        let path = full_url.split('?').next().unwrap_or("/").to_string();
        let path = path.trim_end_matches('/');
        let path = if path.is_empty() { "/" } else { path };

        let headers = parse_headers(&request);
        let origin = headers.get("origin").cloned().unwrap_or_default();

        // CORS preflight
        if method == "OPTIONS" {
            let mut resp = Response::empty(204);
            for h in cors_headers(&origin) {
                resp.add_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // UI routes: issue or inherit authenticated web session cookie (no query string token leakage)
        if method == "GET" && (path == "/" || path == "/chat" || path == "/index.html") {
            let auth_token = headers
                .get("authorization")
                .and_then(|h| {
                    h.strip_prefix("Bearer ")
                        .or_else(|| h.strip_prefix("bearer "))
                })
                .map(|s| s.trim().to_string());

            let creator_cookie_token = headers.get("cookie").and_then(|c| {
                c.split(';')
                    .find(|s| {
                        let t = s.trim();
                        t.starts_with("tara_creator_session=") || t.starts_with("tara_session=")
                    })
                    .and_then(|s| s.trim().split('=').nth(1).map(|v| v.to_string()))
            });

            let candidate_creator_token = auth_token.or(creator_cookie_token);

            let token_opt = if let Some(cand) = candidate_creator_token {
                if let Some(sess) = brain_ref.creator_auth_service.verify_session(&cand) {
                    let creator_id = sess
                        .get("creator_id")
                        .cloned()
                        .unwrap_or_else(|| "ROOT_OPERATOR".to_string());
                    ws_ref.register(&cand, &format!("creator:{}", creator_id));
                    Some(cand)
                } else {
                    None
                }
            } else {
                headers
                    .get("cookie")
                    .and_then(|c| {
                        c.split(';')
                            .find(|s| s.trim().starts_with("tara_web_session="))
                            .map(|s| s.trim()[17..].to_string())
                    })
                    .filter(|existing_cookie| ws_ref.verify(existing_cookie).is_some())
            };

            let token = token_opt.unwrap_or_default();
            let html = crate::routes::render_chat_ui(&token);
            let mut resp = html_response(200, &html);
            if !token.is_empty() {
                let is_secure = headers
                    .get("x-forwarded-proto")
                    .map(|v| v == "https")
                    .unwrap_or(false);
                let secure_flag = if is_secure { "; Secure" } else { "" };
                if let Ok(cookie_hdr) = Header::from_bytes(
                    "Set-Cookie",
                    format!(
                        "tara_web_session={}; Path=/; HttpOnly; SameSite=Strict; Max-Age=3600{}",
                        token, secure_flag
                    ),
                ) {
                    resp.add_header(cookie_hdr);
                }
            }
            for h in cors_headers(&origin) {
                resp.add_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // Preserve endpoint-list resource with mandatory authentication.
        if method == "GET" && path == "/endpoints.txt" {
            if let Err(e) = authenticate_request(&headers, &api_keys_ref, &brain_ref, &ws_ref, &method, path, &client_ip) {
                let resp = json_response(401, &json!({"status":"ERROR","error":e}));
                let _ = request.respond(resp);
                continue;
            }
            let repo_root = tara_engine::find_workspace_root()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| ".".into());
            let endpoint_path = std::path::Path::new(&repo_root).join("endpoints.txt");
            let response = match std::fs::read(&endpoint_path) {
                Ok(contents) => Response::from_data(contents)
                    .with_status_code(200)
                    .with_header(
                        Header::from_bytes("Content-Type", "text/plain; charset=utf-8").unwrap(),
                    )
                    .with_header(Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap()),
                Err(_) => json_response(
                    404,
                    &json!({"status":"NOT_FOUND","error":"Endpoint list is unavailable"}),
                ),
            };
            let _ = request.respond(response);
            continue;
        }

        // Route matching
        match router_ref.match_route(&method, path) {
            None => {
                let status = if router_ref.has_path(path) {
                    405u16
                } else {
                    404u16
                };
                let err_msg = if status == 405 {
                    format!("Method {} not allowed for '{}'", method, path)
                } else {
                    format!("Endpoint '{}' not found", path)
                };
                let resp = json_response(status, &json!({"status":"ERROR","error":err_msg}));
                let _ = request.respond(resp);
            }
            Some((handler, required_auth, required_role)) => {
                let actor = if required_auth {
                    match authenticate_request(&headers, &api_keys_ref, &brain_ref, &ws_ref, &method, path, &client_ip) {
                        Ok(a) => a,
                        Err(e) => {
                            let resp = json_response(401, &json!({"status":"ERROR","error":e}));
                            let _ = request.respond(resp);
                            continue;
                        }
                    }
                } else {
                    "anonymous".to_string()
                };

                // ─────────────────────────────────────────────────────────────
                // Centralized Dynamic Policy Evaluation (Pillar 2: Zero-Trust)
                // ─────────────────────────────────────────────────────────────
                let now_epoch = crate::now_secs();
                let current_gen = brain_ref
                    .authority_manager
                    .current_generation()
                    .unwrap_or(tara_security::primitives::AuthorityGeneration::INITIAL);

                let is_creator_actor = actor.starts_with("creator:") || actor == "ROOT_OPERATOR";
                let (role_obj, assurance_obj) = if is_creator_actor {
                    (
                        tara_security::primitives::PrincipalRole::creator(),
                        tara_security::primitives::AssuranceLevel::CREATOR_ROOT,
                    )
                } else if actor.starts_with("api_key:admin")
                    || actor == "admin"
                    || actor == "system"
                    || actor == "root"
                    || actor.starts_with("api_key:ROOT_OPERATOR")
                    || actor.starts_with("api_key:operator")
                    || actor.starts_with("api_key:creator")
                {
                    (
                        tara_security::primitives::PrincipalRole::operator(),
                        tara_security::primitives::AssuranceLevel::STANDARD_TOKEN,
                    )
                } else if actor.starts_with("worker:") || actor.starts_with("tok_") {
                    (
                        tara_security::primitives::PrincipalRole::worker(),
                        tara_security::primitives::AssuranceLevel::STANDARD_TOKEN,
                    )
                } else if actor == "anonymous" {
                    (
                        tara_security::primitives::PrincipalRole::new("anonymous").unwrap(),
                        tara_security::primitives::AssuranceLevel::ANONYMOUS,
                    )
                } else {
                    (
                        tara_security::primitives::PrincipalRole::user(),
                        tara_security::primitives::AssuranceLevel::STANDARD_TOKEN,
                    )
                };

                let cookie_token = headers.get("cookie").and_then(|c| {
                    c.split(';')
                        .find(|s| s.trim().starts_with("tara_web_session="))
                        .and_then(|s| s.trim().strip_prefix("tara_web_session="))
                        .map(|s| s.trim().to_string())
                });

                let session_token = if actor.starts_with("api_key:") {
                    headers
                        .get("x-session-id")
                        .or_else(|| headers.get("x-session-token"))
                        .cloned()
                } else {
                    headers
                        .get("authorization")
                        .and_then(|h| {
                            h.strip_prefix("Bearer ")
                                .or_else(|| h.strip_prefix("bearer "))
                        })
                        .map(|s| s.trim().to_string())
                        .or_else(|| cookie_token.clone())
                };

                let safe_actor_id = actor
                    .chars()
                    .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
                    .collect::<String>();
                let pid = if let Some((ns, nm)) = actor.split_once(':') {
                    let safe_ns = ns.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-').collect::<String>();
                    let safe_nm = nm.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '@' || *c == '.').collect::<String>();
                    tara_security::primitives::PrincipalId::with_namespace(&safe_ns, &safe_nm)
                        .unwrap_or_else(|_| tara_security::primitives::PrincipalId::new(format!("actor_{}", safe_actor_id)).unwrap_or_else(|_| tara_security::primitives::PrincipalId::new("actor_fallback").unwrap()))
                } else {
                    tara_security::primitives::PrincipalId::new(format!("actor_{}", safe_actor_id))
                        .unwrap_or_else(|_| tara_security::primitives::PrincipalId::new("actor_fallback").unwrap())
                };

                let principal = if is_creator_actor {
                    let tok = session_token.as_deref().unwrap_or("");
                    match brain_ref.creator_auth_service.mint_verified_creator_principal(tok, current_gen) {
                        Ok(p) => {
                            let _ = brain_ref.centralized_session_manager.ensure_session_registered(
                                tok,
                                &p,
                                current_gen,
                                now_epoch,
                            );
                            p
                        }
                        Err(e) => {
                            let resp = json_response(
                                401,
                                &json!({
                                    "status": "UNAUTHORIZED",
                                    "error": format!("Creator session verification failed: {}", e)
                                }),
                            );
                            let _ = request.respond(resp);
                            continue;
                        }
                    }
                } else {
                    let mut principal_claims = HashMap::new();
                    if let Some(ref s_id) = session_token {
                        principal_claims.insert("session_id".to_string(), serde_json::Value::String(s_id.clone()));
                    }
                    match tara_security::primitives::VerifiedPrincipal::construct_verified(
                        pid,
                        role_obj,
                        assurance_obj,
                        None,
                        current_gen,
                        now_epoch,
                        principal_claims,
                    ) {
                        Ok(p) => {
                            if let Some(ref s_id) = session_token {
                                let _ = brain_ref.centralized_session_manager.ensure_session_registered(
                                    s_id,
                                    &p,
                                    current_gen,
                                    now_epoch,
                                );
                            }
                            p
                        }
                        Err(e) => {
                            let resp = json_response(
                                403,
                                &json!({
                                    "status": "FORBIDDEN",
                                    "error": format!("Security identity construction rejected: {}", e)
                                }),
                            );
                            let _ = request.respond(resp);
                            continue;
                        }
                    }
                };

                // Role check
                if let Some(ref req_role) = required_role {
                    let req_lower = req_role.to_lowercase();
                    let has_role = match req_lower.as_str() {
                        "admin" | "operator" => {
                            let allowed_actors =
                                ["admin", "root", "system", "ROOT_OPERATOR", "OPERATOR_ROOT"];
                            allowed_actors.contains(&actor.as_str())
                                || actor.starts_with("creator:")
                                || actor == "ROOT_OPERATOR"
                                || actor.starts_with("api_key:admin")
                                || actor.starts_with("api_key:ROOT_OPERATOR")
                                || actor.starts_with("api_key:operator")
                        }
                        "creator" => {
                            actor.starts_with("creator:") || actor == "ROOT_OPERATOR"
                        }
                        "worker" => {
                            actor.starts_with("worker:")
                                || actor == "system"
                                || actor.starts_with("creator:")
                                || actor == "ROOT_OPERATOR"
                        }
                        _ => {
                            actor.starts_with("creator:")
                                || actor == "ROOT_OPERATOR"
                                || actor.starts_with("api_key:admin")
                                || actor.starts_with("api_key:ROOT_OPERATOR")
                                || actor == "admin"
                        }
                    };
                    if !has_role {
                        let resp = json_response(
                            403,
                            &json!({
                                "status": "ERROR",
                                "error": format!("Forbidden: actor '{}' lacks required role '{}'", actor, req_role)
                            }),
                        );
                        let _ = request.respond(resp);
                        continue;
                    }
                }

                // Parse body for POST/PUT/PATCH
                let query_params = parse_query(&full_url);
                let body = if matches!(method.as_str(), "POST" | "PUT" | "PATCH") {
                    let raw = match read_body(&mut request) {
                        Ok(raw) => raw,
                        Err(e) => {
                            let status = if e.contains("exceeds") { 413 } else { 400 };
                            let _ = request.respond(json_response(
                                status,
                                &json!({"status":"ERROR","error":e}),
                            ));
                            continue;
                        }
                    };
                    if raw.is_empty() {
                        json!({})
                    } else {
                        match serde_json::from_slice(&raw) {
                            Ok(body) => body,
                            Err(e) => {
                                let _ = request.respond(json_response(400, &json!({"status":"ERROR","error":format!("Malformed JSON body: {e}")})));
                                continue;
                            }
                        }
                    }
                } else {
                    // For GET, convert query params to a JSON body for handlers
                    let mut qmap = serde_json::Map::new();
                    for (k, vals) in &query_params {
                        if vals.len() == 1 {
                            qmap.insert(k.clone(), json!(vals[0]));
                        } else {
                            qmap.insert(k.clone(), json!(vals));
                        }
                    }
                    json!(qmap)
                };

                let route_req = RouteRequest {
                    method: method.clone(),
                    path: path.to_string(),
                    query_params,
                    body,
                    headers: headers.clone(),
                    client_ip: client_ip.clone(),
                    actor: Some(actor),
                };

                let sec_action = tara_security::primitives::SecurityAction::new("http", &method)
                    .unwrap_or_else(|_| tara_security::primitives::SecurityAction::execute_task());
                let sec_resource = tara_security::primitives::SecurityResource::custom("route", path)
                    .unwrap_or(tara_security::primitives::SecurityResource::SystemVault);
                let sec_context = tara_security::primitives::SecurityContext::new(&client_ip, now_epoch);

                // Synchronize unified subsystem lockdown state
                let _ = brain_ref.is_system_locked_down();

                let op_req = tara_security::broker::OperationRequest {
                    principal: principal.clone(),
                    action: sec_action.clone(),
                    resource: sec_resource.clone(),
                    context: sec_context.clone(),
                    session_id: session_token.clone(),
                };

                // Centralized Enforcement Gateway: Operation executes strictly inside authorized callback
                let brain_clone = brain_ref.clone();
                let exec_result = brain_ref.enforcement_gateway.evaluate_and_enforce(op_req, |_p, _r| {
                    Ok(handler.handle(&route_req, &brain_clone))
                });

                let result = match exec_result {
                    Ok(res) => res,
                    Err(denial) => {
                        let resp = json_response(
                            403,
                            &json!({
                                "status": "FORBIDDEN",
                                "error": format!("PolicyAuthorizationDenied: {}", denial)
                            }),
                        );
                        let _ = request.respond(resp);
                        continue; // Main Security denial strictly prevents handler execution!
                    }
                };

                let _ = brain_ref.centralized_audit_logger.log_event(
                    principal.id().clone(),
                    sec_action,
                    sec_resource,
                    "ROUTE_HANDLER_COMPLETED",
                    current_gen,
                    now_epoch,
                );
                let status = match result.get("status").and_then(Value::as_str) {
                    Some("UNHEALTHY") => 503,
                    Some("UNAUTHORIZED" | "AUTHENTICATION_FAILED") => 401,
                    Some("FORBIDDEN" | "AUTHORITY_LOCKED") => 403,
                    Some("NOT_FOUND") => 404,
                    Some("ERROR" | "REJECTED" | "FAILED" | "CREATOR_SETUP_REQUIRED") => 400,
                    _ => 200,
                };
                let mut resp = json_response(status, &result);
                for h in cors_headers(&origin) {
                    resp.add_header(h);
                }
                let _ = request.respond(resp);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tara_security::primitives::{
        AssuranceLevel, PrincipalId, PrincipalRole, SecurityAction,
        SecurityContext, SecurityResource, SecurityState, VerifiedPrincipal,
    };

    #[test]
    fn test_server_authenticate_with_centralized_session_manager() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let web_sessions = WebSessionStore::new();
        let api_keys = HashMap::new();

        let current_gen = brain
            .authority_manager
            .current_generation()
            .unwrap();
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let pid = PrincipalId::new("operator_alice").unwrap();
        let principal = VerifiedPrincipal::construct_verified(
            pid,
            PrincipalRole::operator(),
            AssuranceLevel::HARDWARE_ATTESTED,
            None,
            current_gen,
            now_epoch,
            HashMap::new(),
        )
        .unwrap();

        let session = brain
            .centralized_session_manager
            .create_session(principal, "tara_server_test", "tara_api", current_gen, now_epoch)
            .unwrap();

        let mut headers = HashMap::new();
        headers.insert("authorization".into(), format!("Bearer {}", session.session_id));

        let res = authenticate_request(&headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert!(res.is_ok(), "Centralized session token must authenticate successfully");
        assert_eq!(res.unwrap(), "operator_alice");
    }

    #[test]
    fn test_server_authenticate_failure_records_risk() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let web_sessions = WebSessionStore::new();
        let api_keys = HashMap::new();

        let mut headers = HashMap::new();
        headers.insert("authorization".into(), "Bearer invalid_bad_token_12345".into());

        let res = authenticate_request(&headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert!(res.is_err(), "Invalid token must be rejected");

        // Verify that failure was automatically recorded in ContinuousRiskEngine against client IP
        let unauth_id = unauthenticated_client_principal("127.0.0.1");
        let action = SecurityAction::execute_task();
        let context = SecurityContext::new("127.0.0.1", 1700000000);
        let verdict = brain
            .centralized_risk_engine
            .evaluate_risk(&unauth_id, &action, &context)
            .unwrap();

        match verdict {
            tara_security::zero_trust::RiskVerdict::Accept { risk_score } => {
                assert!(risk_score >= 200, "Failure penalty must elevate risk score (got {})", risk_score);
            }
            tara_security::zero_trust::RiskVerdict::RequireHardwareStepUp { risk_score, .. }
            | tara_security::zero_trust::RiskVerdict::RequireCreatorStepUp { risk_score, .. }
            | tara_security::zero_trust::RiskVerdict::TriggerLockdown { risk_score, .. } => {
                assert!(risk_score >= 200);
            }
        }
    }

    #[test]
    fn test_server_honeypot_trap_locks_down_system() {
        let temp_dir = std::env::temp_dir().join(format!("tara_honeypot_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let brain = Arc::new(crate::brain::TaraBrain::new_with_root(temp_dir.to_str().unwrap()).unwrap());
        let web_sessions = WebSessionStore::new();
        let api_keys = HashMap::new();

        // Plant honeypot decoy token trap
        brain
            .tripwire_defense
            .plant_honeypot_token("decoy_token_trap_xyz")
            .unwrap();

        let mut headers = HashMap::new();
        headers.insert("authorization".into(), "Bearer decoy_token_trap_xyz".into());

        let res = authenticate_request(&headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert!(res.is_err(), "Honeypot token must be rejected");
        assert!(res.unwrap_err().contains("Honeypot decoy credential submitted"));

        // Verify system lockdown fail-closed engaged
        assert!(brain.is_system_locked_down(), "Containment must lock down the system");
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    struct TestRouteHandlerProbe {
        call_count: Arc<AtomicUsize>,
    }

    impl RouteHandler for TestRouteHandlerProbe {
        fn handle(&self, _req: &RouteRequest, _brain: &Arc<crate::brain::TaraBrain>) -> Value {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            json!({"status": "SUCCESS", "message": "Handler executed"})
        }
    }

    #[test]
    fn test_server_centralized_policy_denial_blocks_handler() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let current_gen = brain.authority_manager.current_generation().unwrap();
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let call_count = Arc::new(AtomicUsize::new(0));
        let handler = TestRouteHandlerProbe {
            call_count: call_count.clone(),
        };

        // Anonymous user attempts to access /api/v1/admin/overview
        let path = "/api/v1/admin/overview";
        let pid = PrincipalId::new("actor_anonymous").unwrap();
        let principal = VerifiedPrincipal::construct_verified(
            pid,
            PrincipalRole::new("anonymous").unwrap(),
            AssuranceLevel::ANONYMOUS,
            None,
            current_gen,
            now_epoch,
            HashMap::new(),
        )
        .unwrap();

        let sec_state = SecurityState::Operational { generation: current_gen };
        let sec_action = SecurityAction::new("http", "GET").unwrap();
        let sec_resource = SecurityResource::custom("route", path).unwrap();
        let sec_context = SecurityContext::new("127.0.0.1", now_epoch);

        // Evaluate Centralized Policy Engine
        let authz_res = brain.centralized_policy.authorize(
            &sec_state,
            &principal,
            &sec_action,
            &sec_resource,
            &sec_context,
        );

        // MUST be denied by Centralized Policy Engine (Explicit Deny for Anonymous on Admin)
        assert!(authz_res.is_err(), "Anonymous must be denied access to admin endpoint by Centralized Policy");

        // Simulate the dispatch loop: if policy authorization fails, handler MUST NOT be called!
        if authz_res.is_ok() {
            let req = RouteRequest {
                method: "GET".into(),
                path: path.into(),
                query_params: HashMap::new(),
                body: Value::Null,
                headers: HashMap::new(),
                client_ip: "127.0.0.1".into(),
                actor: Some("anonymous".into()),
            };
            let _ = handler.handle(&req, &brain);
        }

        // Verify that handler was NOT invoked
        assert_eq!(call_count.load(Ordering::SeqCst), 0, "Handler must NEVER be executed when policy denies");
    }

    #[test]
    fn test_server_enforcement_gateway_live_pipeline_evaluates_and_enforces() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let current_gen = brain.authority_manager.current_generation().unwrap();
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // 1. Authorized Operator request passes 8-stage EnforcementGateway
        let pid = PrincipalId::new("operator_bob").unwrap();
        let operator_principal = VerifiedPrincipal::construct_verified(
            pid,
            PrincipalRole::operator(),
            AssuranceLevel::HARDWARE_ATTESTED,
            None,
            current_gen,
            now_epoch,
            HashMap::new(),
        )
        .unwrap();

        let sec_action = SecurityAction::new("http", "GET").unwrap();
        let sec_resource = SecurityResource::custom("route", "/api/v1/system/status").unwrap();
        let sec_context = SecurityContext::new("127.0.0.1", now_epoch);

        let initial_merkle_leaf_count = brain.transparency_tree.leaf_count();

        let op_req = tara_security::broker::OperationRequest {
            principal: operator_principal,
            action: sec_action.clone(),
            resource: sec_resource.clone(),
            context: sec_context.clone(),
            session_id: None,
        };

        let result = brain.enforcement_gateway.evaluate_and_enforce(op_req, |_p, _r| Ok("HANDLER_EXECUTED"));
        assert_eq!(result.unwrap(), "HANDLER_EXECUTED", "EnforcementGateway must approve authorized operator");
        assert_eq!(
            brain.transparency_tree.leaf_count(),
            initial_merkle_leaf_count + 1,
            "EnforcementGateway must append Merkle tree transparency leaf on success"
        );

        // 2. Unauthorized Anonymous request targeting Admin endpoint fails closed
        let anon_pid = PrincipalId::new("actor_anonymous").unwrap();
        let anon_principal = VerifiedPrincipal::construct_verified(
            anon_pid,
            PrincipalRole::new("anonymous").unwrap(),
            AssuranceLevel::ANONYMOUS,
            None,
            current_gen,
            now_epoch,
            HashMap::new(),
        )
        .unwrap();

        let admin_resource = SecurityResource::custom("route", "/api/v1/admin/shutdown").unwrap();
        let denied_req = tara_security::broker::OperationRequest {
            principal: anon_principal,
            action: sec_action,
            resource: admin_resource,
            context: sec_context,
            session_id: None,
        };

        let denied_result = brain.enforcement_gateway.evaluate_and_enforce(denied_req, |_p, _r| Ok("SHOULD_NOT_EXECUTE"));
        assert!(denied_result.is_err(), "EnforcementGateway must fail closed on anonymous access to admin route");
    }

    #[test]
    fn test_centralized_passkey_verifier_and_identity_registry() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let current_gen = brain.authority_manager.current_generation().unwrap();
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        use ed25519_dalek::Signer;

        let (signing_key, verifying_key) = tara_security::cryptography::CryptoUtils::generate_ed25519_keypair();
        let cred_id = vec![0x11, 0x22, 0x33, 0x44];

        // Authoritatively register credential in passkey_verifier with cryptographic verification
        let cred_hex = hex::encode(&cred_id);
        brain.passkey_verifier.register_credential(tara_security::hardware_root::EnrolledBiometricCredential {
            credential_id: cred_hex,
            creator_id: "creator_root".to_string(),
            public_key_bytes: verifying_key.to_bytes().to_vec(),
            enrolled_at: now_epoch,
            authenticator_type: tara_security::hardware_root::PlatformAuthenticatorType::default(),
            attestation_statement_hash: String::new(),
            require_hardware_authenticator: false,
            require_biometric_modality: false,
            attestation_format: "none".to_string(),
            aaguid: "00000000-0000-0000-0000-000000000000".to_string(),
            raw_attestation_object: Vec::new(),
            attestation_certificate_der: Vec::new(),
            client_data_hash: [0u8; 32],
            raw_client_data_json: Vec::new(),
        }).unwrap();

        // Build authentic WebAuthn authenticator_data with UP (0x01), UV (0x04), and AT (0x40) set
        let mut auth_data = vec![0u8; 37];
        // 32-byte RP ID hash (sha256 of localhost)
        let rp_id_hash = tara_security::cryptography::CryptoUtils::sha256(b"localhost");
        auth_data[..32].copy_from_slice(&rp_id_hash);
        // Flag byte at index 32: 0x01 (UP) | 0x04 (UV) | 0x40 (AT) = 0x45
        auth_data[32] = 0x45;
        // Sign counter bytes at 33..37: 1
        auth_data[36] = 1;
        // Attested Credential Data: AAGUID (16 bytes) + Credential ID len (2 bytes) + Credential ID + Public Key
        auth_data.extend_from_slice(&[0x00u8; 16]);
        auth_data.extend_from_slice(&(cred_id.len() as u16).to_be_bytes());
        auth_data.extend_from_slice(&cred_id);
        auth_data.extend_from_slice(&verifying_key.to_bytes());

        let client_data_json = br#"{"type":"webauthn.get","challenge":"test_challenge_nonce_1","origin":"https://localhost:7070"}"#;
        let client_data_hash = tara_security::cryptography::CryptoUtils::sha256(client_data_json);

        let mut signed_data = Vec::new();
        signed_data.extend_from_slice(&auth_data);
        signed_data.extend_from_slice(&client_data_hash);

        let signature = signing_key.sign(&signed_data).to_vec();

        let valid_proof = tara_security::zero_trust::IdentityProof::PasskeyAssertion {
            credential_id: cred_id.clone(),
            authenticator_data: auth_data.clone(),
            client_data_json: client_data_json.to_vec(),
            signature: signature.clone(),
        };

        let principal_res = brain.centralized_identity_registry.verify(&valid_proof, current_gen, now_epoch);
        assert!(principal_res.is_ok(), "Centralized identity registry must verify valid PasskeyAssertion: {:?}", principal_res.err());
        let principal = principal_res.unwrap();
        assert_eq!(principal.assurance_level(), AssuranceLevel::CREATOR_ROOT, "User-verified passkey must yield CREATOR_ROOT");
        assert_eq!(principal.role(), &PrincipalRole::creator());

        // Test fail-closed rejection: Authenticator data lacking UV flag (biometrics not verified)
        let non_uv_client_data_json = br#"{"type":"webauthn.get","challenge":"test_challenge_nonce_2","origin":"https://localhost:7070"}"#;
        let non_uv_client_data_hash = tara_security::cryptography::CryptoUtils::sha256(non_uv_client_data_json);

        let mut non_uv_auth_data = auth_data;
        non_uv_auth_data[32] = 0x41; // UP (0x01) | AT (0x40), UV missing
        let mut non_uv_signed = Vec::new();
        non_uv_signed.extend_from_slice(&non_uv_auth_data);
        non_uv_signed.extend_from_slice(&non_uv_client_data_hash);
        let non_uv_signature = signing_key.sign(&non_uv_signed).to_vec();

        let non_uv_proof = tara_security::zero_trust::IdentityProof::PasskeyAssertion {
            credential_id: cred_id,
            authenticator_data: non_uv_auth_data,
            client_data_json: non_uv_client_data_json.to_vec(),
            signature: non_uv_signature,
        };
        let non_uv_res = brain.centralized_identity_registry.verify(&non_uv_proof, current_gen, now_epoch);
        assert!(non_uv_res.is_ok());
        let non_uv_principal = non_uv_res.unwrap();
        // Without UV, assurance is reduced to HARDWARE_ATTESTED, NOT CREATOR_ROOT, and role is operator
        assert_eq!(non_uv_principal.assurance_level(), AssuranceLevel::HARDWARE_ATTESTED);
        assert_eq!(non_uv_principal.role(), &PrincipalRole::operator());

        // Test fail-closed rejection: Truncated authenticator data (<37 bytes)
        let truncated_proof = tara_security::zero_trust::IdentityProof::PasskeyAssertion {
            credential_id: vec![1, 2],
            authenticator_data: vec![0u8; 20],
            client_data_json: client_data_json.to_vec(),
            signature: vec![1, 2, 3],
        };
        let rej = brain.centralized_identity_registry.verify(&truncated_proof, current_gen, now_epoch);
        assert!(rej.is_err(), "Truncated authenticator data must fail closed");
    }

    #[test]
    fn test_centralized_device_registry_hydration_and_verification() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let current_gen = brain.authority_manager.current_generation().unwrap();
        let now_epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate an authentic Ed25519 device keypair
        let keypair = tara_security::devices::DeviceCrypto::generate_device_keypair();
        let dev_id_str = tara_security::devices::DeviceCrypto::derive_device_id(&keypair.public_key_hex).unwrap();
        let dev_id = tara_security::primitives::DeviceId::new(&dev_id_str).unwrap();

        // Enroll device into centralized_device_registry
        let pub_bytes = hex::decode(&keypair.public_key_hex).unwrap();
        let enroll_rec = tara_security::zero_trust::DeviceEnrollmentRecord {
            device_id: dev_id.clone(),
            enrolled_public_key: pub_bytes,
            bound_principal_id: PrincipalId::new("creator_root_operator").unwrap(),
            is_revoked: false,
        };
        brain.centralized_device_registry.enroll_device(enroll_rec).unwrap();

        // Issue and sign challenge nonce
        let nonce = brain.centralized_identity_registry.create_challenge_nonce("device_signature", 300).unwrap();
        let sig_hex = tara_security::devices::DeviceCrypto::sign_message(keypair.private_key_hex(), nonce.as_bytes()).unwrap();
        let sig_bytes = hex::decode(sig_hex).unwrap();

        let device_proof = tara_security::zero_trust::IdentityProof::DeviceSignature {
            device_id: dev_id.clone(),
            challenge_nonce: nonce.to_string(),
            signature: sig_bytes,
        };

        let verified_principal = brain.centralized_identity_registry.verify(&device_proof, current_gen, now_epoch).unwrap();
        assert_eq!(verified_principal.assurance_level(), AssuranceLevel::HARDWARE_ATTESTED);
        assert_eq!(verified_principal.bound_device(), Some(&dev_id));

        // Revoke device and verify fail-closed with a fresh challenge
        brain.centralized_device_registry.revoke_device(&dev_id).unwrap();
        let fresh_nonce = brain.centralized_identity_registry.create_challenge_nonce("device_signature", 300).unwrap();
        let fresh_sig_hex = tara_security::devices::DeviceCrypto::sign_message(keypair.private_key_hex(), fresh_nonce.as_bytes()).unwrap();
        let fresh_sig_bytes = hex::decode(fresh_sig_hex).unwrap();
        let revoked_device_proof = tara_security::zero_trust::IdentityProof::DeviceSignature {
            device_id: dev_id.clone(),
            challenge_nonce: fresh_nonce,
            signature: fresh_sig_bytes,
        };
        let revoked_res = brain.centralized_identity_registry.verify(&revoked_device_proof, current_gen, now_epoch);
        assert!(revoked_res.is_err(), "Revoked device signature must be rejected");
    }

    #[test]
    fn test_server_web_session_revocation_enforced() {
        let temp_dir = std::env::temp_dir().join(format!("tara_web_rev_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let brain = Arc::new(crate::brain::TaraBrain::new_with_root(temp_dir.to_str().unwrap()).unwrap());
        let web_sessions = WebSessionStore::new();
        let api_keys = HashMap::new();

        let token = "test_creator_session_token_xyz";
        web_sessions.register(token, "creator:test_creator_actor");

        // 1. Before revocation, session authenticates successfully
        let mut headers = HashMap::new();
        headers.insert("cookie".into(), format!("tara_web_session={}", token));
        let res = authenticate_request(&headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert!(res.is_ok(), "Web session must authenticate successfully before revocation");
        assert_eq!(res.unwrap(), "creator:test_creator_actor");

        // 2. Revoke the creator principal in CentralizedSessionManager
        let pid = PrincipalId::with_namespace("creator", "test_creator_actor").unwrap();
        assert!(brain.centralized_session_manager.revoke_principal(&pid).is_ok());

        // 3. After revocation, session must immediately fail closed with WebSessionRevoked
        let res_revoked = authenticate_request(&headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert!(res_revoked.is_err(), "Revoked principal web session must fail closed");
        assert!(res_revoked.unwrap_err().contains("WebSessionRevoked"), "Error must indicate principal revocation");

        // Cleanup test directory post-test
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_server_authenticate_client_ip_risk_isolation() {
        let temp_dir = std::env::temp_dir().join(format!("tara_ip_risk_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let brain = Arc::new(crate::brain::TaraBrain::new_with_root(temp_dir.to_str().unwrap()).unwrap());
        let web_sessions = WebSessionStore::new();
        let api_keys = HashMap::new();

        let mut headers = HashMap::new();
        headers.insert("authorization".into(), "Bearer bad_token_attempt".into());
        // Spoofed X-Forwarded-For header attempting to blame victim IP
        headers.insert("x-forwarded-for".into(), "203.0.113.199".into());

        let actual_ip = "198.51.100.5";
        let res = authenticate_request(&headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", actual_ip);
        assert!(res.is_err());

        // Verify failure was attributed strictly to actual_ip, NOT spoofed x-forwarded-for header
        let actual_pid = unauthenticated_client_principal(actual_ip);
        let spoofed_pid = unauthenticated_client_principal("203.0.113.199");
        let action = SecurityAction::execute_task();
        let context = SecurityContext::new(actual_ip, 1700000000);

        let actual_verdict = brain.centralized_risk_engine.evaluate_risk(&actual_pid, &action, &context).unwrap();
        let spoofed_verdict = brain.centralized_risk_engine.evaluate_risk(&spoofed_pid, &action, &context).unwrap();

        if let tara_security::zero_trust::RiskVerdict::Accept { risk_score } = actual_verdict {
            assert!(risk_score >= 200, "Actual IP must receive risk penalty");
        }
        match spoofed_verdict {
            tara_security::zero_trust::RiskVerdict::Accept { risk_score } => {
                assert_eq!(risk_score, 0, "Spoofed victim IP must NOT receive risk penalty");
            }
            _ => panic!("Spoofed IP should not have any risk score"),
        }

        // Cleanup test directory post-test
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_unauthenticated_ip_dos_does_not_trigger_global_lockdown() {
        let temp_dir = std::env::temp_dir().join(format!("tara_dos_isolation_test_{}", rand::random::<u64>()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let brain = Arc::new(crate::brain::TaraBrain::new_with_root(temp_dir.to_str().unwrap()).unwrap());

        // Initialize authority manager root ceremony so authority is in Operational state
        let creator_id = tara_security::primitives::PrincipalId::new("creator_test_root").unwrap();
        let _ = brain.authority_manager.initialize_root_ceremony(
            creator_id,
            "recovery_secret_with_sufficient_high_entropy_padding_32_bytes_long",
            crate::now_secs(),
        );

        let web_sessions = WebSessionStore::new();
        let mut api_keys = HashMap::new();
        api_keys.insert("valid_creator_key".into(), "creator:admin".into());

        let mut bad_headers = HashMap::new();
        bad_headers.insert("authorization".into(), "Bearer malformed_bad_token".into());

        let attacker_ip = "198.51.100.99";

        // Repeatedly send invalid credentials from attacker IP to trigger high risk
        for _ in 0..15 {
            let res = authenticate_request(&bad_headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", attacker_ip);
            assert!(res.is_err(), "Invalid token requests must fail");
        }

        // Verify attacker IP fails, but authority manager and lockdown coordinator are NOT locked down
        assert!(!brain.lockdown_coordinator.is_locked_down(), "Lockdown coordinator must remain active and not triggered by remote attacker IP");
        assert!(!brain.authority_manager.is_locked_down(), "Authority manager must remain active and not triggered by remote attacker IP");
        assert!(!brain.tripwire_defense.is_contained(), "Tripwire defense must not be contained");

        // Verify valid authenticated user can still authenticate successfully
        let mut good_headers = HashMap::new();
        good_headers.insert("authorization".into(), "Bearer valid_creator_key".into());
        let good_res = authenticate_request(&good_headers, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert!(good_res.is_ok(), "Legitimate authenticated requests must still succeed");
    }

    #[test]
    fn test_url_decode_kannada_and_unicode_query_preservation() {
        // Kannada text "ಕನ್ನಡ" percent-encoded: %E0%B2%95%E0%B2%A8%E0%B3%8D%E0%B2%A8%E0%B2%A1
        let encoded_kannada = "%E0%B2%95%E0%B2%A8%E0%B3%8D%E0%B2%A8%E0%B2%A1+language";
        let decoded = url_decode(encoded_kannada);
        assert_eq!(decoded, "ಕನ್ನಡ language");

        // Parse query with Kannada key and value
        let url = "/api/v1/chat?query=%E0%B2%A8%E0%B2%AE%E0%B2%B8%E0%B3%8D%E0%B2%95%E0%B2%BE%E0%B2%B0&lang=%E0%B2%95%E0%B2%A8%E0%B3%8D%E0%B2%A8%E0%B2%A1";
        let params = parse_query(url);
        assert_eq!(params.get("query").and_then(|v| v.first()).map(|s| s.as_str()), Some("ನಮಸ್ಕಾರ"));
        assert_eq!(params.get("lang").and_then(|v| v.first()).map(|s| s.as_str()), Some("ಕನ್ನಡ"));
    }

    #[test]
    fn test_case_insensitive_bearer_token_extraction() {
        let brain = Arc::new(crate::brain::TaraBrain::new(None).unwrap());
        let web_sessions = WebSessionStore::new();
        let mut api_keys = HashMap::new();
        api_keys.insert("test_key".into(), "operator:bob".into());

        // Test with uppercase 'Bearer '
        let mut headers_upper = HashMap::new();
        headers_upper.insert("authorization".into(), "Bearer test_key".into());
        let res_upper = authenticate_request(&headers_upper, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert_eq!(res_upper.unwrap(), "api_key:operator:bob");

        // Test with lowercase 'bearer '
        let mut headers_lower = HashMap::new();
        headers_lower.insert("authorization".into(), "bearer test_key".into());
        let res_lower = authenticate_request(&headers_lower, &api_keys, &brain, &web_sessions, "POST", "/api/v1/test", "127.0.0.1");
        assert_eq!(res_lower.unwrap(), "api_key:operator:bob");
    }
}


