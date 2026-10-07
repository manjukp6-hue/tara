//! TARA Standalone Dataset Downloader (100% Native Rust)
//!
//! Rule 3 Standalone Boundary:
//! - Strictly 100% isolated inside `downloader/`.
//! - Zero external project imports (no tara_engine, tara_core, tara_server).
//! - Only links with internal downloader modules:
//!     * download_engine.rs
//!     * download_register.rs
//!     * license_register.rs
//!     * filter_engine.rs
//! - External storage strictly on dedicated D: drive (`D:\taracore_datasets\downloaded`).
//!
//! User Directives:
//! - "old registry entry ange irlli idu already download agiro files ,,, so matte download agbardu"
//! - Preserves all 724+ existing entries in download_register.json.
//! - Checks every candidate against existing registry to skip already downloaded datasets.
//! - Adheres to Rule 21: Worldwide open-source diversity (OpenStax, Project Gutenberg, software foundations, clean web).
//! - Zero synthetic data (Rule 4 & Rule 21).

use downloader::download_register::RegisterEntry;
use downloader::license_register::LicenseEntry;
use downloader::{DownloadCandidate, DownloadEngine, StageVerificationResult};
use std::env;
use std::path::{Path, PathBuf};

/// Specification for candidate datasets explicitly rejected under TARA Rule-18 / Rule-21.
struct RejectedSourceSpec {
    dataset_id: &'static str,
    source_url: &'static str,
    host_domain: &'static str,
    upstream_repo: &'static str,
    author: &'static str,
    license: &'static str,
    proof_url: &'static str,
    tara_policy_class: &'static str,
    policy_decision: &'static str,
}

fn get_rejected_specs() -> Vec<RejectedSourceSpec> {
    vec![
        RejectedSourceSpec {
            dataset_id: "valkey_engine_docs",
            source_url: "https://github.com/valkey-io/valkey-doc",
            host_domain: "github.com",
            upstream_repo: "valkey-io/valkey-doc",
            author: "Linux Foundation / Valkey Contributors",
            license: "CC-BY-SA-4.0",
            proof_url: "https://github.com/valkey-io/valkey-doc/blob/main/LICENSE",
            tara_policy_class: "copyleft_sharealike_prohibited",
            policy_decision: "REJECTED: License 'CC-BY-SA-4.0' is copyleft/ShareAlike; outside TARA Rule-18 commercial permissive class",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_college_physics",
            source_url: "https://openstax.org/details/books/college-physics",
            host_domain: "openstax.org",
            upstream_repo: "openstax/college-physics",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-4.0 (AI Training Restricted)",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "ai_training_prohibited",
            policy_decision: "REJECTED: OpenStax terms explicitly prohibit LLM/AI model training without prior written permission",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_calculus_vol1",
            source_url: "https://openstax.org/details/books/calculus-volume-1",
            host_domain: "openstax.org",
            upstream_repo: "openstax/calculus-volume-1",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18; AI training prohibited without prior written permission",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_calculus_vol2",
            source_url: "https://openstax.org/details/books/calculus-volume-2",
            host_domain: "openstax.org",
            upstream_repo: "openstax/calculus-volume-2",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_calculus_vol3",
            source_url: "https://openstax.org/details/books/calculus-volume-3",
            host_domain: "openstax.org",
            upstream_repo: "openstax/calculus-volume-3",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_chemistry_2e",
            source_url: "https://openstax.org/details/books/chemistry-2e",
            host_domain: "openstax.org",
            upstream_repo: "openstax/chemistry-2e",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_biology_2e",
            source_url: "https://openstax.org/details/books/biology-2e",
            host_domain: "openstax.org",
            upstream_repo: "openstax/biology-2e",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18; AI training prohibited without prior written permission",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_principles_economics_3e",
            source_url: "https://openstax.org/details/books/principles-economics-3e",
            host_domain: "openstax.org",
            upstream_repo: "openstax/principles-economics-3e",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_principles_management",
            source_url: "https://openstax.org/details/books/principles-management",
            host_domain: "openstax.org",
            upstream_repo: "openstax/principles-management",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18; LLM training requires prior written permission",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_psychology_2e",
            source_url: "https://openstax.org/details/books/psychology-2e",
            host_domain: "openstax.org",
            upstream_repo: "openstax/psychology-2e",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18",
        },
        RejectedSourceSpec {
            dataset_id: "openstax_microbiology",
            source_url: "https://openstax.org/details/books/microbiology",
            host_domain: "openstax.org",
            upstream_repo: "openstax/microbiology",
            author: "Rice University / OpenStax Faculty",
            license: "CC-BY-NC-SA-4.0",
            proof_url: "https://openstax.org/license",
            tara_policy_class: "non_commercial_sharealike_prohibited",
            policy_decision: "REJECTED: Non-Commercial (NC) and ShareAlike (SA) restrictions violate TARA Rule-18",
        },
        RejectedSourceSpec {
            dataset_id: "plos_one_complete_corpus",
            source_url: "https://journals.plos.org/plosone/",
            host_domain: "plos.org",
            upstream_repo: "plos/plosone",
            author: "Public Library of Science (PLOS)",
            license: "CC-BY-4.0 (Mixed Third-Party Media)",
            proof_url: "https://journals.plos.org/plosone/s/licenses-and-copyright",
            tara_policy_class: "blanket_third_party_exclusions",
            policy_decision: "REJECTED_AS_BLANKET: Entire corpus contains unvetted third-party figures, photos, and proprietary data with independent copyright",
        },
        RejectedSourceSpec {
            dataset_id: "bmc_biology_complete_corpus",
            source_url: "https://bmcbiol.biomedcentral.com/",
            host_domain: "biomedcentral.com",
            upstream_repo: "bmc/biology",
            author: "BioMed Central / Springer Nature",
            license: "CC-BY-4.0 (Mixed Third-Party Media)",
            proof_url: "https://www.biomedcentral.com/about/policies/license-and-copyright",
            tara_policy_class: "blanket_third_party_exclusions",
            policy_decision: "REJECTED_AS_BLANKET: Unverified third-party content exclusions; blanket ingestion rejected under Rule 18 fail-closed policy",
        },
        RejectedSourceSpec {
            dataset_id: "cern_opendata_software_tutorials",
            source_url: "https://opendata.cern.ch/docs/tutorials",
            host_domain: "opendata.cern.ch",
            upstream_repo: "cern/opendata-tutorials",
            author: "CERN Collaborations",
            license: "GPL-3.0 / Mixed Software License",
            proof_url: "https://opendata.cern.ch/terms",
            tara_policy_class: "copyleft_gpl_prohibited",
            policy_decision: "REJECTED_AS_BLANKET: Metadata is CC0 but underlying tutorial software packages and code are licensed under GPL copyleft",
        },
        RejectedSourceSpec {
            dataset_id: "netbsd_kernel_docs_blanket",
            source_url: "https://www.netbsd.org/docs/kernel/",
            host_domain: "netbsd.org",
            upstream_repo: "netbsd/kernel-docs",
            author: "The NetBSD Foundation",
            license: "GFDL / Mixed Berkeley & FSF Manuals",
            proof_url: "https://www.netbsd.org/about/disclaimer.html",
            tara_policy_class: "copyleft_gfdl_prohibited",
            policy_decision: "REJECTED_AS_BLANKET: Contains GNU Free Documentation License (GFDL) manuals with invariant sections and mixed third-party licenses",
        },
        RejectedSourceSpec {
            dataset_id: "llvm_libcxx_docs_blanket",
            source_url: "https://libcxx.llvm.org/",
            host_domain: "llvm.org",
            upstream_repo: "llvm/libcxx",
            author: "LLVM Foundation Contributors",
            license: "Apache-2.0 WITH LLVM-exception (Imported Legacy Exceptions)",
            proof_url: "https://llvm.org/docs/DeveloperPolicy.html#license",
            tara_policy_class: "unverified_legacy_exceptions",
            policy_decision: "REJECTED_AS_BLANKET: Imported legacy and third-party components contain unproven license exceptions; blanket ingestion prohibited",
        },
        RejectedSourceSpec {
            dataset_id: "opendsa_cs_etextbook_blanket",
            source_url: "https://opendsa-server.cs.vt.edu/",
            host_domain: "opendsa-server.cs.vt.edu",
            upstream_repo: "vtseng/opendsa",
            author: "Virginia Tech CS Dept / OpenDSA Team",
            license: "CC-BY-4.0 / MIT Hybrid (Unmapped Sections)",
            proof_url: "https://opendsa-server.cs.vt.edu/license",
            tara_policy_class: "hybrid_license_unmapped",
            policy_decision: "REJECTED_AS_BLANKET: Hybrid license without granular per-section provenance mapping; fails fail-closed four-way rights gate",
        },
        RejectedSourceSpec {
            dataset_id: "ietf_rfc_standards_text",
            source_url: "https://www.rfc-editor.org/rfc/",
            host_domain: "rfc-editor.org",
            upstream_repo: "ietf/rfc",
            author: "Internet Engineering Task Force (IETF)",
            license: "IETF Trust Legal Provisions (TLP 5.0)",
            proof_url: "https://trustee.ietf.org/trust-legal-provisions.html",
            tara_policy_class: "derivative_extraction_restricted",
            policy_decision: "REJECTED: TLP 5.0 restricts extraction and derivative distribution outside the IETF Standards Process without explicit permission",
        },
    ]
}

fn register_all_rejected_candidates(engine: &mut DownloadEngine, register_dir: &Path) -> Result<usize, String> {
    let specs = get_rejected_specs();
    let total = specs.len();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ts = format!("epoch_{now}");

    for spec in &specs {
        // 1. Download register entry
        let dl_entry = RegisterEntry {
            dataset_id: spec.dataset_id.to_string(),
            source_url: spec.source_url.to_string(),
            author: spec.author.to_string(),
            sha256: String::new(),
            size_bytes: 0,
            record_count: 0,
            local_path: "NONE".to_string(),
            timestamp: ts.clone(),
            host_domain: spec.host_domain.to_string(),
            upstream_repo: spec.upstream_repo.to_string(),
            upstream_publisher: spec.author.to_string(),
            dataset_license: spec.license.to_string(),
            underlying_content_source: spec.source_url.to_string(),
            underlying_content_terms: spec.license.to_string(),
            content_rights_status: "REJECTED_UNDER_RULE_18".to_string(),
            tara_policy_class: spec.tara_policy_class.to_string(),
            download_permission: "REJECTED".to_string(),
            database_license_status: "REJECTED".to_string(),
            record_rights_status: "REJECTED".to_string(),
            training_eligibility: "REJECTED".to_string(),
            policy_decision: spec.policy_decision.to_string(),
            asset_file_id: spec.dataset_id.to_string(),
            edition_id: "REJECTED".to_string(),
            rights_evidence: spec.proof_url.to_string(),
            jurisdiction_scope: "REJECTED".to_string(),
            third_party_exclusions: "ENTIRE_SOURCE_REJECTED".to_string(),
            license_id: spec.license.to_string(),
            asset_scope: "REJECTED".to_string(),
        };

        engine.download_register.register_rejected(dl_entry, &register_dir.join("download_register"))?;

        // 2. License register entry
        let lic_entry = LicenseEntry {
            dataset_id: spec.dataset_id.to_string(),
            license: spec.license.to_string(),
            proof_url: spec.proof_url.to_string(),
            source_url: spec.source_url.to_string(),
            domain: spec.host_domain.to_string(),
            author: spec.author.to_string(),
            commercial_use: false,
            modify_allowed: false,
            download_allowed: false,
            ai_training_allowed: false,
            detected_licenses: vec![spec.license.to_string()],
            detected_authors: vec![spec.author.to_string()],
            lines_audited: 0,
            timestamp: ts.clone(),
            host_domain: spec.host_domain.to_string(),
            upstream_repo: spec.upstream_repo.to_string(),
            upstream_publisher: spec.author.to_string(),
            underlying_content_source: spec.source_url.to_string(),
            underlying_content_terms: spec.license.to_string(),
            content_rights_status: "REJECTED_UNDER_RULE_18".to_string(),
            tara_policy_class: spec.tara_policy_class.to_string(),
            download_permission: "REJECTED".to_string(),
            database_license_status: "REJECTED".to_string(),
            training_eligibility: "REJECTED".to_string(),
            policy_decision: spec.policy_decision.to_string(),
            asset_file_id: spec.dataset_id.to_string(),
            edition_id: "REJECTED".to_string(),
            rights_evidence: spec.proof_url.to_string(),
            jurisdiction_scope: "REJECTED".to_string(),
            third_party_exclusions: "ENTIRE_SOURCE_REJECTED".to_string(),
            license_id: spec.license.to_string(),
            asset_scope: "REJECTED".to_string(),
        };

        engine.license_register.register_rejected(lic_entry, &register_dir.join("license_register"))?;
    }

    Ok(total)
}

/// Defines diverse worldwide open-source candidate sources.
/// Enforces Rule 21: Diverse institutions, academic archives, open foundations.
fn generate_diverse_candidates() -> Vec<DownloadCandidate> {
    let mut candidates = Vec::new();

    // ------------------------------------------------------------------------
    // 1. Verified Historical Scientific Foundations (Public Domain, Pre-1929)
    // Rights Class: PUBLIC_DOMAIN (Life+70 verified, no modern translation copyright)
    // OpenStax completely excluded per Rule-18 (AI training restriction / NC-SA terms).
    // ------------------------------------------------------------------------
    let historical_classics = [
        (
            "darwin_origin_species",
            "https://www.gutenberg.org/cache/epub/1228/pg1228.txt",
            "darwin_origin_species_1859.txt",
            "Charles Darwin (d. 1882)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Origin of Species 1859 1st edition original",
            "modern_forewords",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "adam_smith_wealth_nations",
            "https://www.gutenberg.org/cache/epub/3300/pg3300.txt",
            "adam_smith_wealth_nations_1776.txt",
            "Adam Smith (d. 1790)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Wealth of Nations 1776 original",
            "modern_commentaries",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "faraday_chemical_candle",
            "https://www.gutenberg.org/cache/epub/14474/pg14474.txt",
            "faraday_chemical_candle_1861.txt",
            "Michael Faraday (d. 1867)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Chemical History of a Candle 1861 original lectures",
            "modern_editorials",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "maxwell_matter_motion",
            "https://www.gutenberg.org/cache/epub/40109/pg40109.txt",
            "maxwell_matter_motion_1876.txt",
            "James Clerk Maxwell (d. 1879)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Matter and Motion 1876 physics treatise",
            "modern_forewords",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "galileo_two_new_sciences",
            "https://www.gutenberg.org/cache/epub/37729/pg37729.txt",
            "galileo_two_new_sciences_1914.txt",
            "Galileo Galilei (Trans: Henry Crew d. 1953, Alfonso de Salvio d. 1956)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Dialogues Concerning Two New Sciences 1914 English edition",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "babbage_economy_machinery",
            "https://www.gutenberg.org/cache/epub/41444/pg41444.txt",
            "babbage_economy_machinery_1832.txt",
            "Charles Babbage (d. 1871)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "On the Economy of Machinery and Manufactures 1832 original",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "descartes_discourse_method",
            "https://www.gutenberg.org/cache/epub/59/pg59.txt",
            "descartes_discourse_method_1637.txt",
            "René Descartes (Trans: John Veitch d. 1894)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Discourse on Method Veitch 1850 English translation",
            "modern_editorials",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "bacon_novum_organum",
            "https://www.gutenberg.org/cache/epub/2434/pg2434.txt",
            "bacon_novum_organum_1620.txt",
            "Francis Bacon (d. 1626)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Novum Organum 1620 philosophical treatise",
            "modern_forewords",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "kant_pure_reason",
            "https://www.gutenberg.org/cache/epub/4280/pg4280.txt",
            "kant_critique_pure_reason_1855.txt",
            "Immanuel Kant (Trans: J.M.D. Meiklejohn d. 1902)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Critique of Pure Reason 1881 Meiklejohn translation",
            "translator_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "hume_human_understanding",
            "https://www.gutenberg.org/cache/epub/9662/pg9662.txt",
            "hume_human_understanding_1748.txt",
            "David Hume (d. 1776)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Enquiry Concerning Human Understanding 1748 original",
            "modern_commentaries",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "locke_human_understanding",
            "https://www.gutenberg.org/cache/epub/10604/pg10604.txt",
            "locke_human_understanding_1689.txt",
            "John Locke (d. 1704)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Essay Concerning Human Understanding 1689 original",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "aurelius_meditations",
            "https://www.gutenberg.org/cache/epub/2680/pg2680.txt",
            "aurelius_meditations_1862.txt",
            "Marcus Aurelius (Trans: George Long d. 1879)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Meditations George Long translation 1862",
            "modern_commentaries",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "plato_republic",
            "https://www.gutenberg.org/cache/epub/1497/pg1497.txt",
            "plato_republic_1871.txt",
            "Plato (Trans: Benjamin Jowett d. 1893)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "The Republic Benjamin Jowett translation 1871",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "aristotle_politics",
            "https://www.gutenberg.org/cache/epub/6762/pg6762.txt",
            "aristotle_politics_1885.txt",
            "Aristotle (Trans: Benjamin Jowett d. 1893)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Politics Benjamin Jowett translation 1885",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "lavoisier_elements_chemistry",
            "https://www.gutenberg.org/cache/epub/30775/pg30775.txt",
            "lavoisier_elements_chemistry_1790.txt",
            "Antoine Lavoisier (Trans: Robert Kerr d. 1813)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Elements of Chemistry 1790 Kerr translation",
            "modern_tables",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "darwin_voyage_beagle",
            "https://www.gutenberg.org/cache/epub/944/pg944.txt",
            "darwin_voyage_beagle_1839.txt",
            "Charles Darwin (d. 1882)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Voyage of the Beagle 1839 original journal",
            "modern_forewords",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "spinoza_ethics",
            "https://www.gutenberg.org/cache/epub/3800/pg3800.txt",
            "spinoza_ethics_1883.txt",
            "Baruch Spinoza (Trans: R. H. M. Elwes d. 1923)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Ethics 1883 Elwes translation",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "newton_opticks",
            "https://www.gutenberg.org/cache/epub/33504/pg33504.txt",
            "newton_opticks_1704.txt",
            "Isaac Newton (d. 1727)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Opticks 1704 original treatise",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "boyle_sceptical_chymist",
            "https://www.gutenberg.org/cache/epub/22914/pg22914.txt",
            "boyle_sceptical_chymist_1661.txt",
            "Robert Boyle (d. 1691)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "The Sceptical Chymist 1661 original",
            "modern_introductions",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "lyell_principles_geology",
            "https://www.gutenberg.org/cache/epub/37459/pg37459.txt",
            "lyell_principles_geology_1830.txt",
            "Charles Lyell (d. 1875)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Principles of Geology 1830 original",
            "modern_annotations",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "leibniz_monadology",
            "https://www.gutenberg.org/cache/epub/4014/pg4014.txt",
            "leibniz_monadology_1898.txt",
            "Gottfried Wilhelm Leibniz (Trans: Robert Latta d. 1932)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "The Monadology 1898 Latta translation",
            "modern_commentaries",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "hooke_micrographia",
            "https://www.gutenberg.org/cache/epub/15491/pg15491.txt",
            "hooke_micrographia_1665.txt",
            "Robert Hooke (d. 1703)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Micrographia 1665 original",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "ricardo_political_economy",
            "https://www.gutenberg.org/cache/epub/33310/pg33310.txt",
            "ricardo_political_economy_1817.txt",
            "David Ricardo (d. 1823)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Principles of Political Economy and Taxation 1817",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "machiavelli_prince",
            "https://www.gutenberg.org/cache/epub/1232/pg1232.txt",
            "machiavelli_prince_1908.txt",
            "Niccolò Machiavelli (Trans: W. K. Marriott d. 1927)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "The Prince Marriott 1908 translation",
            "modern_commentaries",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "marx_communist_manifesto",
            "https://www.gutenberg.org/cache/epub/61/pg61.txt",
            "marx_communist_manifesto_1888.txt",
            "Karl Marx & Friedrich Engels (Trans: Samuel Moore d. 1898)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "The Communist Manifesto 1888 Moore translation",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "aristotle_poetics",
            "https://www.gutenberg.org/cache/epub/1974/pg1974.txt",
            "aristotle_poetics_1895.txt",
            "Aristotle (Trans: S. H. Butcher d. 1910)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "The Poetics Butcher translation 1895",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "plato_apology",
            "https://www.gutenberg.org/cache/epub/1656/pg1656.txt",
            "plato_apology_1891.txt",
            "Plato (Trans: Benjamin Jowett d. 1893)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Apology of Socrates Jowett translation 1891",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "plato_symposium",
            "https://www.gutenberg.org/cache/epub/1600/pg1600.txt",
            "plato_symposium_1892.txt",
            "Plato (Trans: Benjamin Jowett d. 1893)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Symposium Jowett translation 1892",
            "modern_prefaces",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "lucretius_nature_of_things",
            "https://www.gutenberg.org/cache/epub/785/pg785.txt",
            "lucretius_nature_of_things_1916.txt",
            "Lucretius (Trans: William Ellery Leonard d. 1944)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "On the Nature of Things Leonard translation 1916",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "euclid_elements",
            "https://www.gutenberg.org/cache/epub/21076/pg21076.txt",
            "euclid_elements_1908.txt",
            "Euclid (Trans: Thomas L. Heath d. 1940)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Euclid Elements Heath 1908 edition",
            "commentary,post-1928",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "boole_laws_of_thought",
            "https://www.gutenberg.org/cache/epub/15114/pg15114.txt",
            "boole_laws_of_thought_1854.txt",
            "George Boole (d. 1864)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Laws of Thought 1854 original",
            "modern_notes,modern_preface",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "poincare_science_hypothesis",
            "https://www.gutenberg.org/cache/epub/37157/pg37157.txt",
            "poincare_science_hypothesis_1905.txt",
            "Henri Poincaré (Trans: George Bruce Halsted d. 1922)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Science and Hypothesis 1905",
            "revisions",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "locke_two_treatises",
            "https://www.gutenberg.org/cache/epub/7370/pg7370.txt",
            "locke_two_treatises_1689.txt",
            "John Locke (d. 1704)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Two Treatises of Government 1689",
            "modern_analysis",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "hobbes_leviathan",
            "https://www.gutenberg.org/cache/epub/3207/pg3207.txt",
            "hobbes_leviathan_1651.txt",
            "Thomas Hobbes (d. 1679)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Leviathan 1651 original",
            "modern_notes",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "mill_on_liberty",
            "https://www.gutenberg.org/cache/epub/34901/pg34901.txt",
            "mill_on_liberty_1859.txt",
            "John Stuart Mill (d. 1873)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "On Liberty 1859 original",
            "modern_introductions",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "aristotle_ethics",
            "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
            "aristotle_ethics_1893.txt",
            "Aristotle (Trans: F. H. Peters d. 1928)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Nicomachean Ethics 1893 Peters translation",
            "revised_editions",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "gibbon_roman_empire",
            "https://www.gutenberg.org/cache/epub/25717/pg25717.txt",
            "gibbon_roman_empire_1776.txt",
            "Edward Gibbon (d. 1794)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Decline and Fall of Roman Empire 1776",
            "modern_annotations",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
        (
            "webster_unabridged_1913",
            "https://www.gutenberg.org/cache/epub/29765/pg29765.txt",
            "webster_unabridged_1913.txt",
            "Noah Webster / Syndicate Publishing (1913)",
            "PUBLIC_DOMAIN",
            "https://www.gutenberg.org/license",
            "Webster's Revised Unabridged Dictionary 1913",
            "modern_revisions",
            "VERIFIED_JURISDICTIONS_ONLY",
        ),
    ];

    for (id, url, filename, author, lic, proof, scope, excl, jur) in historical_classics {
        candidates.push(DownloadCandidate {
            dataset_id: id.to_string(),
            source_url: url.to_string(),
            author: author.to_string(),
            license: lic.to_string(),
            license_proof_url: proof.to_string(),
            target_filename: Some(filename.to_string()),
            asset_scope: Some(scope.to_string()),
            required_exclusions: Some(excl.to_string()),
            jurisdiction_scope: Some(jur.to_string()),
        });
    }

    candidates
}

fn print_header(downloaded_dir: &Path, register_dir: &Path, target_records: usize) {
    println!("================================================================================");
    println!("  TARA STANDALONE OPEN-SOURCE DATASET DOWNLOAD ENGINE (100% NATIVE RUST)");
    println!("================================================================================");
    println!("Target Records    : {} (50M scale)", target_records);
    println!("Target Storage    : {}", downloaded_dir.display());
    println!("Registry Directory: {}", register_dir.display());
    println!("Isolation Mode    : 100% Standalone (Rule 3: Zero external project links)");
    println!("Worldwide Diversity: Verified Historical Public Domain + Permissive Open Source Standards + Academic Archives");
    println!("Commercial Rights : Enforced (Rule 21: 4-Way Rights Verification)");
    println!("================================================================================");
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Default target storage on dedicated D: drive per user directive
    let downloaded_dir = PathBuf::from(r"D:\taracore_datasets\downloaded");
    let register_dir = PathBuf::from("downloader");

    let mut target_records: usize = 50_000_000;
    let mut max_shards: usize = 0;
    let mut audit_only = false;
    let mut register_rejected_only = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--max-shards" | "--shards" if i + 1 < args.len() => {
                if let Ok(n) = args[i + 1].parse() {
                    max_shards = n;
                }
                i += 2;
            }
            "--target-records" if i + 1 < args.len() => {
                if let Ok(n) = args[i + 1].parse() {
                    target_records = n;
                }
                i += 2;
            }
            "--provenance-audit" | "--audit" => {
                audit_only = true;
                i += 1;
            }
            "--register-rejected" => {
                register_rejected_only = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --manifest-path downloader/Cargo.toml -- [OPTIONS]");
                println!("Options:");
                println!("  --target-records <N>   Set target new records scale (default: 50,000,000)");
                println!("  --provenance-audit     Run 7-layer provenance audit and output metrics");
                println!("  --register-rejected    Register all 18 Rule-18/21 rejected candidates with reasons");
                println!("  -h, --help             Show this help information");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    print_header(&downloaded_dir, &register_dir, target_records);

    // 1. Load registries from disk
    let mut engine = match DownloadEngine::load_from_dir(&register_dir) {
        Ok(mut eng) => {
            println!("[INIT] Registries loaded successfully.");
            // Enrich 7-layer provenance across existing entries
            eng.download_register.enrich_all();
            let _ = eng.download_register.save(&register_dir.join("download_register"));
            eng.license_register.enrich_all();
            let _ = eng.license_register.save(&register_dir.join("license_register"));
            // Automatically migrate any existing raw .txt files on disk to canonical structured .jsonl format
            if let Ok(migrated) = eng.migrate_existing_txt_to_jsonl(&register_dir) {
                if migrated > 0 {
                    println!("[INIT] Converted {} existing datasets on disk to canonical structured .jsonl format.", migrated);
                }
            }
            eng
        }
        Err(e) => {
            eprintln!("[ERROR] Failed to load registries: {e}");
            std::process::exit(1);
        }
    };

    if register_rejected_only {
        println!("--------------------------------------------------------------------------------");
        println!("  REGISTERING EXPLICITLY REJECTED CANDIDATES WITH REASONS");
        println!("--------------------------------------------------------------------------------");
        match register_all_rejected_candidates(&mut engine, &register_dir) {
            Ok(count) => {
                println!("[SUCCESS] Registered {} rejected candidate datasets into download_register and license_register.", count);
                println!("Total download_register entries now : {}", engine.download_register.entries.len());
                println!("Total license_register entries now  : {}", engine.license_register.entries.len());
                println!("[PERSISTENCE] Successfully saved to both .json and .txt files.");
            }
            Err(e) => {
                eprintln!("[ERROR] Failed to register rejected candidates: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    let metrics = engine.download_register.provenance_metrics();
    println!("--------------------------------------------------------------------------------");
    println!("  7-LAYER PROVENANCE METRICS SUMMARY");
    println!("--------------------------------------------------------------------------------");
    println!("Total registered source entries : {}", metrics.total_source_records);
    println!("Network host domains (Hostnames) : {}", metrics.network_hosts_count);
    for h in &metrics.network_hosts {
        println!("  - Network Host : {}", h);
    }
    println!("Local storage pools             : {}", metrics.local_storage_pools_count);
    for p in &metrics.local_storage_pools {
        println!("  - Local Pool   : {}", p);
    }
    println!("Unique upstream repositories    : {}", metrics.unique_repos_count);
    for r in &metrics.unique_repos {
        println!("  - Upstream Repo: {}", r);
    }
    println!("Policy Class Distribution:");
    for (pol, cnt) in &metrics.policy_class_distribution {
        println!("  * {:<32} : {}", pol, cnt);
    }
    println!("Training Eligibility Distribution:");
    for (t, cnt) in &metrics.training_eligibility_distribution {
        println!("  * {:<32} : {}", t, cnt);
    }
    println!("--------------------------------------------------------------------------------");

    if audit_only {
        println!("[AUDIT COMPLETE] Provenance audit finished successfully.");
        return;
    }

    // 2. Preserve existing entries per user directive ("old registry entry ange irlli")
    let existing_count = engine.download_register.entries.len();
    let existing_records = engine.download_register.total_records();
    println!(
        "[REGISTRY] Preserving {} existing entries ({} records). Re-download skipping ACTIVE.",
        existing_count, existing_records
    );

    // 3. Generate diverse candidate stream
    let mut candidates = generate_diverse_candidates();
    println!("[STREAM] Generated {} initial curated candidate sources.", candidates.len());

    // Expand with continuous stream of verified historical Public Domain eBooks for non-stop downloading
    for ebook_id in 1..=5000 {
        candidates.push(DownloadCandidate {
            dataset_id: format!("gutenberg_ebook_{ebook_id}"),
            source_url: format!("https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"),
            author: "Project Gutenberg Verified Historical Author".to_string(),
            license: "PUBLIC_DOMAIN".to_string(),
            license_proof_url: "https://www.gutenberg.org/license".to_string(),
            target_filename: Some(format!("gutenberg_ebook_{ebook_id}.txt")),
            asset_scope: Some("Project Gutenberg Public Domain classic eBook".to_string()),
            required_exclusions: Some("modern_commercial_derivatives".to_string()),
            jurisdiction_scope: Some("VERIFIED_JURISDICTIONS_ONLY".to_string()),
        });
    }
    println!("[STREAM] Total candidate queue extended to {} sources for non-stop downloading.", candidates.len());
    println!("--------------------------------------------------------------------------------");

    let mut newly_downloaded_records: usize = 0;
    let mut newly_ingested_files: usize = 0;
    let mut skipped_already_downloaded: usize = 0;

    for (candidate_idx, candidate) in candidates.iter().enumerate() {
        if max_shards > 0 && newly_ingested_files >= max_shards {
            println!("\n[SHARDS LIMIT REACHED] Ingested {} requested shard(s). Stopping.", max_shards);
            break;
        }

        if newly_downloaded_records >= target_records {
            println!("\n[TARGET REACHED] Reached target scale of {} new records!", target_records);
            break;
        }

        // 4. Pre-check: Collision avoidance ("idu already download agiro files ,,, so matte download agbardu")
        let stage_check = engine.verify_candidate_stages(candidate);
        match stage_check {
            StageVerificationResult::Rejected { stage_num, stage_name, ref reason, .. } => {
                if stage_num == 1 {
                    skipped_already_downloaded += 1;
                    // Skip silently or log concise skip to avoid spamming 700+ logs
                    if skipped_already_downloaded <= 5 || skipped_already_downloaded.is_multiple_of(50) {
                        println!(
                            "[SKIP - ALREADY REGISTERED] '{}' exists in download_register (skipped: {}).",
                            candidate.dataset_id, skipped_already_downloaded
                        );
                    }
                } else {
                    println!(
                        "[REJECT - {}] Candidate '{}' rejected: {}",
                        stage_name, candidate.dataset_id, reason
                    );
                }
                continue;
            }
            StageVerificationResult::Approved => {
                // Passed all stages: ready to download!
            }
        }

        println!("\n--------------------------------------------------------------------------------");
        println!(
            "[CANDIDATE #{}] INGESTING '{}' (Author: {}, License: {})...",
            candidate_idx + 1,
            candidate.dataset_id,
            candidate.author,
            candidate.license
        );
        println!("  Source URL: {}", candidate.source_url);
        println!("  Target File: {:?}", candidate.target_filename);

        match engine.download_and_register(candidate, &downloaded_dir, &register_dir) {
            Ok(entry) => {
                newly_downloaded_records += entry.record_count;
                newly_ingested_files += 1;
                let pct = (newly_downloaded_records as f64 / target_records as f64) * 100.0;
                println!(
                    "[INGESTED SUCCESS] {} (+{} records, {} bytes)",
                    entry.dataset_id, entry.record_count, entry.size_bytes
                );
                println!(
                    "[PROGRESS] Newly Ingested: {} files | Records: {} / {} ({:.2}%) | Total Registry Entries: {}",
                    newly_ingested_files,
                    newly_downloaded_records,
                    target_records,
                    pct,
                    engine.download_register.entries.len()
                );
            }
            Err(e) => {
                eprintln!("[ERROR] Ingestion failed for candidate '{}': {e}", candidate.dataset_id);
            }
        }
    }

    println!("\n================================================================================");
    println!(" DOWNLOAD SESSION SUMMARY");
    println!("================================================================================");
    println!("  Preserved Existing Registry Entries : {}", existing_count);
    println!("  Skipped Already-Downloaded Entries  : {}", skipped_already_downloaded);
    println!("  Newly Ingested Datasets             : {}", newly_ingested_files);
    println!("  Newly Downloaded Records            : {}", newly_downloaded_records);
    println!("  Total Registry Entries Now          : {}", engine.download_register.entries.len());
    println!("  Total Records in Registry           : {}", engine.download_register.total_records());
    println!("  Target Storage Directory            : {}", downloaded_dir.display());
    println!("================================================================================");
}
