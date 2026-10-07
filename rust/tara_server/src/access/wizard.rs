//! First-run creator bootstrap wizard and interactive terminal QR prompts.

use super::qr::QrCodeMatrix;
use super::setup::{CreatorSetupEngine, CreatorSetupRequest, CreatorSetupResult};

pub struct QrPrompt;

impl QrPrompt {
    pub fn display_terminal_qr(content: &str) -> String {
        match QrCodeMatrix::encode(content) {
            Ok(qr) => qr.render_ascii(),
            Err(e) => format!("Failed to generate QR matrix: {}", e),
        }
    }

    pub fn format_fallback_secret(secret: &str) -> String {
        format!(
            "================ FALLBACK ENROLLMENT CODE ================\n\
             {}\n\
             ==========================================================",
            secret
        )
    }
}

pub struct BootstrapWizard {
    pub setup_engine: CreatorSetupEngine,
}

impl BootstrapWizard {
    pub fn new(repo_root: &str) -> Self {
        Self {
            setup_engine: CreatorSetupEngine::new(repo_root),
        }
    }

    pub fn execute_bootstrap(
        &self,
        request: CreatorSetupRequest,
    ) -> Result<CreatorSetupResult, String> {
        self.setup_engine.initialize_creator_trust_root(request)
    }

    pub fn render_bootstrap_prompt(&self, challenge_url: &str, fallback_code: &str) -> String {
        let mut out = String::new();
        out.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        out.push_str("║             TARA CREATOR ROOT IDENTITY INITIALIZATION        ║\n");
        out.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");
        out.push_str("Scan the QR code below using your trusted creator device:\n\n");
        out.push_str(&QrPrompt::display_terminal_qr(challenge_url));
        out.push_str("\n\n");
        out.push_str(&QrPrompt::format_fallback_secret(fallback_code));
        out.push('\n');
        out
    }
}
