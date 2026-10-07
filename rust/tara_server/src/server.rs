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
        let mut records = self.records.lock().unwrap();

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
        self.routes.write().unwrap().insert(
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
        let routes = self.routes.read().unwrap();
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
        let routes = self.routes.read().unwrap();
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

    /// Issue a new web session token for `actor_id`.
    pub fn issue(&self, actor_id: &str) -> String {
        let token = hex::encode(rand::thread_rng().gen::<[u8; 32]>());
        self.sessions
            .lock()
            .unwrap()
            .insert(token.clone(), (actor_id.to_string(), Instant::now()));
        token
    }

    /// Verify a web session token, returning actor_id if valid.
    pub fn verify(&self, token: &str) -> Option<String> {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some((actor, issued_at)) = sessions.get(token) {
            if issued_at.elapsed() < self.ttl {
                return Some(actor.clone());
            }
            sessions.remove(token);
        }
        None
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Authentication
// ──────────────────────────────────────────────────────────────────────────────

/// Authenticate an incoming request. Returns `Ok(actor_id)` or `Err(error_message)`.
pub fn authenticate_request(
    headers: &HashMap<String, String>,
    api_keys: &HashMap<String, String>,
    brain: &Arc<crate::brain::TaraBrain>,
    web_sessions: &WebSessionStore,
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

    let token = if auth_header.to_lowercase().starts_with("bearer ") {
        auth_header[7..].trim()
    } else if !x_api_key.is_empty() {
        x_api_key.trim()
    } else if !x_worker_token.is_empty() {
        x_worker_token.trim()
    } else {
        return Err("Unauthorized: Missing Authorization Bearer token, X-API-Key, or X-Tara-Worker-Token header.".into());
    };

    // Check worker token validation against dynamic control plane registry
    if !x_worker_token.is_empty() || token.starts_with("tok_") || token.starts_with("internal_") {
        if let Some(worker_id) = brain
            .control_plane
            .workers
            .read()
            .unwrap()
            .validate_worker_token(token)
        {
            return Ok(format!("worker:{}", worker_id));
        }
    }

    // Check web session tokens first
    if let Some(actor) = web_sessions.verify(token) {
        return Ok(actor);
    }

    // Check creator session
    if let Some(sess) = brain.creator_auth_service.verify_session(token) {
        return Ok(sess
            .get("creator_id")
            .cloned()
            .unwrap_or_else(|| "ROOT_OPERATOR".to_string()));
    }

    // Check API keys with constant-time comparison
    for (key, actor) in api_keys {
        if constant_time_eq(token.as_bytes(), key.as_bytes()) {
            return Ok(actor.clone());
        }
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
    vec![
        Header::from_bytes("Access-Control-Allow-Origin", origin).unwrap(),
        Header::from_bytes("Vary", "Origin").unwrap(),
        Header::from_bytes("Access-Control-Allow-Methods", "GET, POST, OPTIONS").unwrap(),
        Header::from_bytes(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-API-Key",
        )
        .unwrap(),
    ]
}

// ──────────────────────────────────────────────────────────────────────────────
// Request parsing
// ──────────────────────────────────────────────────────────────────────────────

fn parse_headers(req: &Request) -> HashMap<String, String> {
    req.headers()
        .iter()
        .map(|h| (h.field.to_string().to_lowercase(), h.value.to_string()))
        .collect()
}

fn parse_query(url: &str) -> HashMap<String, Vec<String>> {
    let mut params: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(q) = url.split('?').nth(1) {
        for pair in q.split('&') {
            let mut parts = pair.splitn(2, '=');
            let key = url_decode(parts.next().unwrap_or(""));
            let val = url_decode(parts.next().unwrap_or(""));
            params.entry(key).or_default().push(val);
        }
    }
    params
}

fn url_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex_str) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(byte) = u8::from_str_radix(hex_str, 16) {
                    out.push(byte as char);
                    i += 3;
                    continue;
                }
            }
        } else if bytes[i] == b'+' {
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
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

        // UI routes: issue or inherit authenticated web session cookie
        if method == "GET" && (path == "/" || path == "/chat" || path == "/index.html") {
            let query_token = full_url.split('?').nth(1).and_then(|q| {
                q.split('&').find_map(|pair| {
                    let mut parts = pair.split('=');
                    match (parts.next(), parts.next()) {
                        (Some("session_token"), Some(val)) | (Some("token"), Some(val)) => {
                            Some(val.to_string())
                        }
                        _ => None,
                    }
                })
            });

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

            let candidate_creator_token = auth_token.or(query_token).or(creator_cookie_token);

            let token_opt = if let Some(cand) = candidate_creator_token {
                if let Some(sess) = brain_ref.creator_auth_service.verify_session(&cand) {
                    let creator_id = sess
                        .get("creator_id")
                        .cloned()
                        .unwrap_or_else(|| "ROOT_OPERATOR".to_string());
                    Some(ws_ref.issue(&creator_id))
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
                resp.add_header(
                    Header::from_bytes(
                        "Set-Cookie",
                        format!(
                            "tara_web_session={}; HttpOnly; SameSite=Strict; Max-Age=3600",
                            token
                        ),
                    )
                    .unwrap(),
                );
            }
            for h in cors_headers(&origin) {
                resp.add_header(h);
            }
            let _ = request.respond(resp);
            continue;
        }

        // Preserve the legacy public endpoint-list resource as plain text.
        if method == "GET" && path == "/endpoints.txt" {
            let repo_root = std::env::var("TARA_REPO_ROOT").unwrap_or_else(|_| ".".into());
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
                // Check web session cookie fallback
                let cookie_token = headers.get("cookie").and_then(|c| {
                    c.split(';')
                        .find(|s| s.trim().starts_with("tara_web_session="))
                        .and_then(|s| s.trim().strip_prefix("tara_web_session="))
                        .map(|s| s.to_string())
                });

                let actor = if required_auth {
                    // Try normal auth first, then cookie-based web session
                    let auth_result =
                        authenticate_request(&headers, &api_keys_ref, &brain_ref, &ws_ref);
                    match auth_result {
                        Ok(a) => a,
                        Err(e) => {
                            // Try cookie
                            if let Some(ref cookie_tok) = cookie_token {
                                if let Some(actor) = ws_ref.verify(cookie_tok) {
                                    actor
                                } else {
                                    let resp =
                                        json_response(401, &json!({"status":"ERROR","error":e}));
                                    let _ = request.respond(resp);
                                    continue;
                                }
                            } else {
                                let resp = json_response(401, &json!({"status":"ERROR","error":e}));
                                let _ = request.respond(resp);
                                continue;
                            }
                        }
                    }
                } else {
                    "anonymous".to_string()
                };

                // Role check
                if let Some(ref req_role) = required_role {
                    if req_role == "admin" {
                        let allowed_actors =
                            ["admin", "root", "system", "ROOT_OPERATOR", "OPERATOR_ROOT"];
                        if !allowed_actors.contains(&actor.as_str()) {
                            let resp = json_response(
                                403,
                                &json!({
                                    "status": "ERROR",
                                    "error": format!("Forbidden: actor '{}' lacks role '{}'", actor, req_role)
                                }),
                            );
                            let _ = request.respond(resp);
                            continue;
                        }
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

                let result = handler.handle(&route_req, &brain_ref);
                let status = match result.get("status").and_then(Value::as_str) {
                    Some("UNHEALTHY") => 503,
                    Some("UNAUTHORIZED") => 401,
                    Some("FORBIDDEN") => 403,
                    Some("NOT_FOUND") => 404,
                    Some("ERROR" | "REJECTED") => 400,
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
