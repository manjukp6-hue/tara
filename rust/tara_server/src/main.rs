//! TARA Server main entry point.

use std::sync::Arc;

fn main() {
    // Load .env manually
    load_dotenv();

    let host = std::env::var("TARA_HOST")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "127.0.0.1".to_string());
    let port_value = std::env::var("TARA_PORT")
        .or_else(|_| std::env::var("PORT"))
        .unwrap_or_else(|_| "8765".to_string());
    let port: u16 = match port_value.parse() {
        Ok(port) if port != 0 => port,
        _ => {
            eprintln!("[TARA Server] Invalid listening port: {port_value}");
            std::process::exit(2);
        }
    };

    println!("[TARA Server] Initializing TaraBrain...");

    let brain = match tara_server::brain::TaraBrain::new(None) {
        Ok(b) => {
            println!(
                "[TARA Server] TaraBrain initialized. Model: {}",
                if b.is_model_loaded() {
                    "LOADED"
                } else {
                    "OFFLINE FALLBACK"
                }
            );
            Arc::new(b)
        }
        Err(e) => {
            eprintln!("[TARA Server] FATAL: Failed to initialize TaraBrain: {}", e);
            #[cfg(target_os = "windows")]
            {
                eprintln!("\nPress Enter to exit...");
                let mut buf = String::new();
                let _ = std::io::stdin().read_line(&mut buf);
            }
            std::process::exit(1);
        }
    };

    let router = Arc::new(tara_server::server::ApiRouter::new());
    tara_server::routes::register_all_routes(&router);

    let rate_limiter = Arc::new(tara_server::server::SlidingWindowRateLimiter::new());
    let api_keys = Arc::new(tara_server::server::load_api_keys());
    let web_sessions = brain.web_sessions.clone();

    println!(
        "[TARA Server] Routes registered. Starting HTTP server on {}:{}",
        host, port
    );

    tara_server::server::run_server(
        &host,
        port,
        brain,
        router,
        rate_limiter,
        api_keys,
        web_sessions,
    );
}

/// Read `.env` file and export variables into the process environment.
/// Only loads variables not already set. Validates key names for safe syntax.
fn load_dotenv() {
    let env_path = ".env";
    if let Ok(content) = std::fs::read_to_string(env_path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Reject excessively long lines (defense-in-depth)
            if line.len() > 4096 {
                eprintln!(
                    "[TARA Security] Skipping oversized .env line (>{} bytes)",
                    4096
                );
                continue;
            }
            if let Some(eq_pos) = line.find('=') {
                let key = line[..eq_pos].trim();
                let val = line[eq_pos + 1..]
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                // Validate key: only ASCII alphanumeric and underscores, non-empty
                if key.is_empty() || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    eprintln!(
                        "[TARA Security] Skipping .env variable with invalid key syntax: '{}'",
                        &key[..key.len().min(32)]
                    );
                    continue;
                }
                // Only set if not already present in environment (no override)
                if std::env::var(key).is_err() {
                    std::env::set_var(key, val);
                }
            }
        }
    }
}
