use serde_json::json;
use std::fs;
use std::path::PathBuf;
use tara_engine::dataset_classifier::{
    CanonicalHierarchicalRecord, DeterministicClassifier, HierarchicalPartitionManager,
    ProvenanceMetadata,
};

#[test]
fn test_deterministic_domain_math() {
    let input = "Solve the quadratic equation 2x^2 + 5x - 3 = 0.";
    let output = "The solutions are x = 1/2 and x = -3.";
    let res = DeterministicClassifier::classify(input, output, "math_dataset", "mathematics");
    assert_eq!(res.primary_domain, "Mathematics");
    assert_eq!(res.subdomain, "Algebra");
    assert_eq!(res.topic, "Equations & Expressions");
    assert_eq!(res.education_level, "Grade 10");
    assert_eq!(res.difficulty, "Medium");
}

#[test]
fn test_deterministic_domain_cybersecurity() {
    let input = "Analyze CVE-2023-1234: buffer overflow vulnerability in network stack.";
    let output =
        "The vulnerability allows remote code execution due to unchecked bounds in packet parser.";
    let res = DeterministicClassifier::classify(input, output, "cve_advisories", "cybersecurity");
    assert_eq!(res.primary_domain, "Cybersecurity");
    assert_eq!(res.education_level, "Professional / Advanced");
    assert_eq!(res.difficulty, "Expert");
}

#[test]
fn test_deterministic_domain_programming() {
    let input = "Write a Python function to check if a word is a palindrome.";
    let output = "def is_palindrome(s: str) -> bool:\n    return s == s[::-1]";
    let res = DeterministicClassifier::classify(input, output, "code_alpaca", "programming");
    assert_eq!(res.primary_domain, "Programming");
    assert_eq!(res.subdomain, "Python Programming");
}

#[test]
fn test_fallback_unspecified_no_fabrication() {
    let input = "What is the capital of France?";
    let output = "The capital of France is Paris.";
    let res = DeterministicClassifier::classify(input, output, "general_qa", "general");
    // Should NOT fabricate school level or university level
    assert_eq!(res.education_level, "Level-Unspecified");
    // Difficulty should be reasonable
    assert_eq!(res.difficulty, "Easy");
}

#[test]
fn test_parse_raw_record_zero_content_mutation() {
    let input_str = "Write a binary search algorithm in Rust.\nMust handle empty slices.";
    let output_str = "pub fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    arr.binary_search(&target).ok()\n}";

    let raw = json!({
        "id": "tara_v2_unit_test_01",
        "domain": "programming",
        "input": input_str,
        "output": output_str,
        "provenance": {
            "source_name": "modelscope:AI-ModelScope/test",
            "source_url": "https://modelscope.cn/test",
            "source_owner": "AI-ModelScope",
            "dataset_id": "9999",
            "license": "Apache-2.0",
            "license_source_url": "https://apache.org/licenses/LICENSE-2.0",
            "collection_timestamp": "1790850000",
            "observed_downloads": 500,
            "observed_likes": 25,
            "synthetic_origin": false
        }
    });

    let parsed =
        DeterministicClassifier::parse_raw_record(&raw).expect("Failed to parse raw record");
    assert_eq!(parsed.input, input_str, "Input content was mutated!");
    assert_eq!(parsed.output, output_str, "Output content was mutated!");
    assert_eq!(parsed.provenance.license, "Apache-2.0");
    assert_eq!(
        parsed.provenance.source_name,
        "modelscope:AI-ModelScope/test"
    );
    assert_eq!(parsed.primary_domain, "Programming");
}

#[test]
fn test_partition_manager_manifest_and_shards() {
    let temp_dir = PathBuf::from("target/test_tara_hierarchy");
    let _ = fs::remove_dir_all(&temp_dir);
    fs::create_dir_all(&temp_dir).unwrap();

    let mut manager = HierarchicalPartitionManager::new(&temp_dir, 2).expect("Init manager failed");

    let rec1 = CanonicalHierarchicalRecord {
        id: "rec_1".to_string(),
        primary_domain: "Mathematics".to_string(),
        secondary_domains: vec![],
        subdomain: "Arithmetic".to_string(),
        education_level: "Grade 4".to_string(),
        difficulty: "Easy".to_string(),
        topic: "Addition".to_string(),
        subtopic: "Basic".to_string(),
        classification_confidence: "High".to_string(),
        classification_method: "evidence_rule".to_string(),
        input: "2 + 2 = ?".to_string(),
        output: "4".to_string(),
        original_record_id: Some("rec_1".to_string()),
        original_sha256: "dummy_sha_1".to_string(),
        provenance: ProvenanceMetadata {
            source_name: "test_source".to_string(),
            source_url: "https://example.com".to_string(),
            source_owner: "tester".to_string(),
            dataset_id: "001".to_string(),
            license: "MIT".to_string(),
            license_source_url: "https://mit.edu".to_string(),
            collection_timestamp: "1790800000".to_string(),
            observed_downloads: None,
            observed_likes: None,
            synthetic_origin: false,
        },
    };

    let rec2 = CanonicalHierarchicalRecord {
        id: "rec_2".to_string(),
        primary_domain: "Mathematics".to_string(),
        secondary_domains: vec![],
        subdomain: "Arithmetic".to_string(),
        education_level: "Grade 4".to_string(),
        difficulty: "Easy".to_string(),
        topic: "Multiplication".to_string(),
        subtopic: "Basic".to_string(),
        classification_confidence: "High".to_string(),
        classification_method: "evidence_rule".to_string(),
        input: "3 * 3 = ?".to_string(),
        output: "9".to_string(),
        original_record_id: Some("rec_2".to_string()),
        original_sha256: "dummy_sha_2".to_string(),
        provenance: rec1.provenance.clone(),
    };

    let rec3 = CanonicalHierarchicalRecord {
        id: "rec_3".to_string(),
        primary_domain: "Cybersecurity".to_string(),
        secondary_domains: vec![],
        subdomain: "Vulnerabilities".to_string(),
        education_level: "Professional / Advanced".to_string(),
        difficulty: "Expert".to_string(),
        topic: "CVE".to_string(),
        subtopic: "Buffer Overflow".to_string(),
        classification_confidence: "High".to_string(),
        classification_method: "evidence_rule".to_string(),
        input: "Analyze vulnerability".to_string(),
        output: "Mitigate with bounds checks".to_string(),
        original_record_id: Some("rec_3".to_string()),
        original_sha256: "dummy_sha_3".to_string(),
        provenance: rec1.provenance.clone(),
    };

    manager.write_record(rec1).unwrap();
    manager.write_record(rec2).unwrap();
    manager.write_record(rec3).unwrap();

    let metrics = manager.finalize("1790860000").unwrap();
    assert_eq!(metrics.total_output_records, 3);
    assert_eq!(metrics.missing_records, 0);
    assert_eq!(metrics.content_mutations, 0);

    // Verify manifest exists and is valid JSON
    let manifest_path = temp_dir.join("manifest.json");
    assert!(manifest_path.exists());
    let manifest_content = fs::read_to_string(manifest_path).unwrap();
    let manifest_val: serde_json::Value = serde_json::from_str(&manifest_content).unwrap();
    assert_eq!(
        manifest_val["verification_status"],
        "VERIFIED_CANONICAL_HIERARCHICAL_SUCCESS"
    );
    assert!(
        manifest_val["curriculum_training_order"]["stage_1_foundational_cognitive"].is_object()
    );

    // Clean up
    let _ = fs::remove_dir_all(&temp_dir);
}
