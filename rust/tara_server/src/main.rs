//! TARA Server main entry point.

use std::sync::Arc;

fn main() {
    // Load .env manually
    load_dotenv();

    let host = std::env::var("TARA_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("TARA_PORT")
        .unwrap_or_else(|_| "8765".to_string())
        .parse()
        .unwrap_or(8765);

    println!("[TARA Server] Initializing TaraBrain...");

    let brain = match tara_server::brain::TaraBrain::new(None) {
        Ok(b) => {
            println!("[TARA Server] TaraBrain initialized. Model: {}",
                if b.model.is_some() { "LOADED" } else { "OFFLINE FALLBACK" });
            Arc::new(b)
        }
        Err(e) => {
            eprintln!("[TARA Server] FATAL: Failed to initialize TaraBrain: {}", e);
            std::process::exit(1);
        }
    };

    let router = Arc::new(tara_server::server::ApiRouter::new());
    tara_server::routes::register_all_routes(&router);

    let rate_limiter = Arc::new(tara_server::server::SlidingWindowRateLimiter::new());
    let api_keys = Arc::new(tara_server::server::load_api_keys());
    let web_sessions = brain.web_sessions.clone();

    println!("[TARA Server] Routes registered. Starting HTTP server on {}:{}", host, port);

    tara_server::server::run_server(
        &host, port, brain, router, rate_limiter, api_keys, web_sessions,
    );
}

/// Read `.env` file and export variables into the process environment.
fn load_dotenv() {
    let env_path = ".env";
    if let Ok(content) = std::fs::read_to_string(env_path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let val = line[eq_pos + 1..].trim().trim_matches('"').trim_matches('\'');
                if std::env::var(key).is_err() {
                    std::env::set_var(key, val);
                }
            }
        }
    }
}
