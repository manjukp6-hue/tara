//! Structured biological knowledge and genetics reasoning for ScienceEngine.

use std::collections::HashMap;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum BiologyError {
    #[error("invalid nucleotide sequence: {0}")]
    InvalidNucleotide(char),
    #[error("sequence length not divisible by 3 for translation: len = {0}")]
    InvalidCodonLength(usize),
    #[error("allele frequencies do not sum to 1.0: p={p}, q={q}")]
    InvalidHardyWeinbergFrequencies { p: f64, q: f64 },
}

pub struct Biology;

static GENETIC_CODE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();

fn init_genetic_code() -> HashMap<&'static str, &'static str> {
    let mut map = HashMap::new();
    // RNA Codon table
    map.insert("AUG", "Methionine (Start)");
    map.insert("UUU", "Phenylalanine");
    map.insert("UUC", "Phenylalanine");
    map.insert("UUA", "Leucine");
    map.insert("UUG", "Leucine");
    map.insert("CUU", "Leucine");
    map.insert("CUC", "Leucine");
    map.insert("CUA", "Leucine");
    map.insert("CUG", "Leucine");
    map.insert("AUU", "Isoleucine");
    map.insert("AUC", "Isoleucine");
    map.insert("AUA", "Isoleucine");
    map.insert("GUU", "Valine");
    map.insert("GUC", "Valine");
    map.insert("GUA", "Valine");
    map.insert("GUG", "Valine");
    map.insert("UCU", "Serine");
    map.insert("UCC", "Serine");
    map.insert("UCA", "Serine");
    map.insert("UCG", "Serine");
    map.insert("CCU", "Proline");
    map.insert("CCC", "Proline");
    map.insert("CCA", "Proline");
    map.insert("CCG", "Proline");
    map.insert("ACU", "Threonine");
    map.insert("ACC", "Threonine");
    map.insert("ACA", "Threonine");
    map.insert("ACG", "Threonine");
    map.insert("GCU", "Alanine");
    map.insert("GCC", "Alanine");
    map.insert("GCA", "Alanine");
    map.insert("GCG", "Alanine");
    map.insert("UAU", "Tyrosine");
    map.insert("UAC", "Tyrosine");
    map.insert("UAA", "Stop");
    map.insert("UAG", "Stop");
    map.insert("UGA", "Stop");
    map.insert("CAU", "Histidine");
    map.insert("CAC", "Histidine");
    map.insert("CAA", "Glutamine");
    map.insert("CAG", "Glutamine");
    map.insert("AAU", "Asparagine");
    map.insert("AAC", "Asparagine");
    map.insert("AAA", "Lysine");
    map.insert("AAG", "Lysine");
    map.insert("GAU", "Aspartic Acid");
    map.insert("GAC", "Aspartic Acid");
    map.insert("GAA", "Glutamic Acid");
    map.insert("GAG", "Glutamic Acid");
    map.insert("UGU", "Cysteine");
    map.insert("UGC", "Cysteine");
    map.insert("UGG", "Tryptophan");
    map.insert("CGU", "Arginine");
    map.insert("CGC", "Arginine");
    map.insert("CGA", "Arginine");
    map.insert("CGG", "Arginine");
    map.insert("AGU", "Serine");
    map.insert("AGC", "Serine");
    map.insert("AGA", "Arginine");
    map.insert("AGG", "Arginine");
    map.insert("GGU", "Glycine");
    map.insert("GGC", "Glycine");
    map.insert("GGA", "Glycine");
    map.insert("GGG", "Glycine");
    map
}

impl Biology {
    /// Transcribe DNA coding strand to mRNA (replace T with U).
    pub fn transcribe_dna_to_mrna(dna: &str) -> Result<String, BiologyError> {
        let mut rna = String::with_capacity(dna.len());
        for ch in dna.to_uppercase().chars() {
            match ch {
                'A' => rna.push('A'),
                'T' => rna.push('U'),
                'C' => rna.push('C'),
                'G' => rna.push('G'),
                whitespace if whitespace.is_whitespace() => continue,
                invalid => return Err(BiologyError::InvalidNucleotide(invalid)),
            }
        }
        Ok(rna)
    }

    /// Complementary DNA strand: A <-> T, C <-> G.
    pub fn dna_complement(dna: &str) -> Result<String, BiologyError> {
        let mut comp = String::with_capacity(dna.len());
        for ch in dna.to_uppercase().chars() {
            match ch {
                'A' => comp.push('T'),
                'T' => comp.push('A'),
                'C' => comp.push('G'),
                'G' => comp.push('C'),
                whitespace if whitespace.is_whitespace() => continue,
                invalid => return Err(BiologyError::InvalidNucleotide(invalid)),
            }
        }
        Ok(comp)
    }

    /// Translate mRNA to polypeptide amino acid chain.
    pub fn translate_mrna(mrna: &str) -> Result<Vec<&'static str>, BiologyError> {
        let clean: String = mrna
            .to_uppercase()
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        if !clean.len().is_multiple_of(3) {
            return Err(BiologyError::InvalidCodonLength(clean.len()));
        }

        let code = GENETIC_CODE.get_or_init(init_genetic_code);
        let mut peptides = Vec::new();

        for chunk in clean.as_bytes().chunks(3) {
            let codon = std::str::from_utf8(chunk).unwrap();
            let amino_acid = code.get(codon).copied().unwrap_or("Unknown");
            peptides.push(amino_acid);
            if amino_acid == "Stop" {
                break;
            }
        }
        Ok(peptides)
    }

    /// Monohybrid cross Punnett square ratios for heterozygous parents (Aa x Aa):
    /// Genotypic ratio: 1 AA : 2 Aa : 1 aa (25% : 50% : 25%)
    /// Phenotypic ratio: 3 Dominant : 1 Recessive (75% : 25%)
    pub fn monohybrid_cross_ratios() -> (&'static str, &'static str) {
        ("1 AA : 2 Aa : 1 aa", "3 Dominant : 1 Recessive")
    }

    /// Hardy-Weinberg equilibrium: p^2 + 2pq + q^2 = 1.
    /// Given dominant allele frequency p and recessive q (p + q = 1.0).
    pub fn hardy_weinberg(p: f64, q: f64) -> Result<(f64, f64, f64), BiologyError> {
        if (p + q - 1.0).abs() > 1e-4 || p < 0.0 || q < 0.0 {
            return Err(BiologyError::InvalidHardyWeinbergFrequencies { p, q });
        }
        let p2 = p * p; // Homozygous dominant (AA)
        let two_pq = 2.0 * p * q; // Heterozygous (Aa)
        let q2 = q * q; // Homozygous recessive (aa)
        Ok((p2, two_pq, q2))
    }
}
