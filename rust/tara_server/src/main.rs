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
                if b.model.read().map(|model| model.is_some()).unwrap_or(false) {
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

    // Initialize Architecture Sync Engine: Layer 2 Full Reconciler + Layer 1 Live Watcher
    let arch_config = tara_server::runtime::architecture_sync::ArchitectureSyncConfig::default();
    let arch_engine = Arc::new(tara_server::runtime::architecture_sync::ArchitectureSyncEngine::new(arch_config));
    if let Ok(report) = arch_engine.reconcile_full() {
        println!(
            "[TARA Server] Architecture Sync Engine reconciled: {} files, {} folders in {}ms.",
            report.total_files, report.total_folders, report.duration_ms
        );
    }
    if let Err(e) = arch_engine.start_live_watcher() {
        eprintln!("[TARA Server] Warning: Could not start Architecture Live Watcher: {e}");
    }

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
                let val = line[eq_pos + 1..]
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                if std::env::var(key).is_err() {
                    std::env::set_var(key, val);
                }
            }
        }
    }
}
