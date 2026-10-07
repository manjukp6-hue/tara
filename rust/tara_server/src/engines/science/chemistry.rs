//! Chemistry computation and periodic table engine for ScienceEngine.

use super::constants::{AVOGADRO_CONSTANT, GAS_CONSTANT};
use std::collections::HashMap;
use std::sync::OnceLock;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ChemistryError {
    #[error("unknown chemical element symbol '{0}'")]
    UnknownElement(String),
    #[error("chemical formula parse error in '{formula}': {details}")]
    FormulaParseError { formula: String, details: String },
    #[error("division by zero: {0}")]
    DivisionByZero(String),
    #[error("physical quantity cannot be negative: {param} = {val}")]
    NegativeQuantity { param: String, val: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalElement {
    pub atomic_number: u32,
    pub symbol: &'static str,
    pub name: &'static str,
    pub atomic_mass: f64,
    pub period: u32,
    pub group: u32,
}

static PERIODIC_TABLE: OnceLock<HashMap<&'static str, ChemicalElement>> = OnceLock::new();

fn init_periodic_table() -> HashMap<&'static str, ChemicalElement> {
    let elements = vec![
        ChemicalElement {
            atomic_number: 1,
            symbol: "H",
            name: "Hydrogen",
            atomic_mass: 1.008,
            period: 1,
            group: 1,
        },
        ChemicalElement {
            atomic_number: 2,
            symbol: "He",
            name: "Helium",
            atomic_mass: 4.0026,
            period: 1,
            group: 18,
        },
        ChemicalElement {
            atomic_number: 3,
            symbol: "Li",
            name: "Lithium",
            atomic_mass: 6.94,
            period: 2,
            group: 1,
        },
        ChemicalElement {
            atomic_number: 4,
            symbol: "Be",
            name: "Beryllium",
            atomic_mass: 9.0122,
            period: 2,
            group: 2,
        },
        ChemicalElement {
            atomic_number: 5,
            symbol: "B",
            name: "Boron",
            atomic_mass: 10.81,
            period: 2,
            group: 13,
        },
        ChemicalElement {
            atomic_number: 6,
            symbol: "C",
            name: "Carbon",
            atomic_mass: 12.011,
            period: 2,
            group: 14,
        },
        ChemicalElement {
            atomic_number: 7,
            symbol: "N",
            name: "Nitrogen",
            atomic_mass: 14.007,
            period: 2,
            group: 15,
        },
        ChemicalElement {
            atomic_number: 8,
            symbol: "O",
            name: "Oxygen",
            atomic_mass: 15.999,
            period: 2,
            group: 16,
        },
        ChemicalElement {
            atomic_number: 9,
            symbol: "F",
            name: "Fluorine",
            atomic_mass: 18.998,
            period: 2,
            group: 17,
        },
        ChemicalElement {
            atomic_number: 10,
            symbol: "Ne",
            name: "Neon",
            atomic_mass: 20.180,
            period: 2,
            group: 18,
        },
        ChemicalElement {
            atomic_number: 11,
            symbol: "Na",
            name: "Sodium",
            atomic_mass: 22.990,
            period: 3,
            group: 1,
        },
        ChemicalElement {
            atomic_number: 12,
            symbol: "Mg",
            name: "Magnesium",
            atomic_mass: 24.305,
            period: 3,
            group: 2,
        },
        ChemicalElement {
            atomic_number: 13,
            symbol: "Al",
            name: "Aluminium",
            atomic_mass: 26.982,
            period: 3,
            group: 13,
        },
        ChemicalElement {
            atomic_number: 14,
            symbol: "Si",
            name: "Silicon",
            atomic_mass: 28.085,
            period: 3,
            group: 14,
        },
        ChemicalElement {
            atomic_number: 15,
            symbol: "P",
            name: "Phosphorus",
            atomic_mass: 30.974,
            period: 3,
            group: 15,
        },
        ChemicalElement {
            atomic_number: 16,
            symbol: "S",
            name: "Sulfur",
            atomic_mass: 32.06,
            period: 3,
            group: 16,
        },
        ChemicalElement {
            atomic_number: 17,
            symbol: "Cl",
            name: "Chlorine",
            atomic_mass: 35.45,
            period: 3,
            group: 17,
        },
        ChemicalElement {
            atomic_number: 18,
            symbol: "Ar",
            name: "Argon",
            atomic_mass: 39.948,
            period: 3,
            group: 18,
        },
        ChemicalElement {
            atomic_number: 19,
            symbol: "K",
            name: "Potassium",
            atomic_mass: 39.098,
            period: 4,
            group: 1,
        },
        ChemicalElement {
            atomic_number: 20,
            symbol: "Ca",
            name: "Calcium",
            atomic_mass: 40.078,
            period: 4,
            group: 2,
        },
        ChemicalElement {
            atomic_number: 26,
            symbol: "Fe",
            name: "Iron",
            atomic_mass: 55.845,
            period: 4,
            group: 8,
        },
        ChemicalElement {
            atomic_number: 29,
            symbol: "Cu",
            name: "Copper",
            atomic_mass: 63.546,
            period: 4,
            group: 11,
        },
        ChemicalElement {
            atomic_number: 30,
            symbol: "Zn",
            name: "Zinc",
            atomic_mass: 65.38,
            period: 4,
            group: 12,
        },
        ChemicalElement {
            atomic_number: 35,
            symbol: "Br",
            name: "Bromine",
            atomic_mass: 79.904,
            period: 4,
            group: 17,
        },
        ChemicalElement {
            atomic_number: 47,
            symbol: "Ag",
            name: "Silver",
            atomic_mass: 107.87,
            period: 5,
            group: 11,
        },
        ChemicalElement {
            atomic_number: 53,
            symbol: "I",
            name: "Iodine",
            atomic_mass: 126.90,
            period: 5,
            group: 17,
        },
        ChemicalElement {
            atomic_number: 79,
            symbol: "Au",
            name: "Gold",
            atomic_mass: 196.97,
            period: 6,
            group: 11,
        },
        ChemicalElement {
            atomic_number: 80,
            symbol: "Hg",
            name: "Mercury",
            atomic_mass: 200.59,
            period: 6,
            group: 12,
        },
        ChemicalElement {
            atomic_number: 82,
            symbol: "Pb",
            name: "Lead",
            atomic_mass: 207.2,
            period: 6,
            group: 14,
        },
        ChemicalElement {
            atomic_number: 92,
            symbol: "U",
            name: "Uranium",
            atomic_mass: 238.03,
            period: 7,
            group: 3,
        },
    ];
    let mut map = HashMap::new();
    for el in elements {
        map.insert(el.symbol, el);
    }
    map
}

pub struct Chemistry;

impl Chemistry {
    /// Lookup element by chemical symbol (case-sensitive).
    pub fn get_element(symbol: &str) -> Option<&'static ChemicalElement> {
        let table = PERIODIC_TABLE.get_or_init(init_periodic_table);
        table.get(symbol)
    }

    /// Parse chemical formula (e.g. "H2O", "C6H12O6", "NaCl", "H2SO4") into element counts.
    pub fn parse_formula(formula: &str) -> Result<HashMap<String, u32>, ChemistryError> {
        let trimmed = formula.trim();
        if trimmed.is_empty() {
            return Err(ChemistryError::FormulaParseError {
                formula: formula.to_string(),
                details: "empty formula string".into(),
            });
        }

        let mut counts = HashMap::new();
        let chars: Vec<char> = trimmed.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if !chars[i].is_ascii_uppercase() {
                return Err(ChemistryError::FormulaParseError {
                    formula: formula.to_string(),
                    details: format!("expected uppercase element symbol at index {}", i),
                });
            }

            let mut symbol = String::new();
            symbol.push(chars[i]);
            i += 1;

            if i < chars.len() && chars[i].is_ascii_lowercase() {
                symbol.push(chars[i]);
                i += 1;
            }

            let mut count_str = String::new();
            while i < chars.len() && chars[i].is_ascii_digit() {
                count_str.push(chars[i]);
                i += 1;
            }

            let count = if count_str.is_empty() {
                1
            } else {
                count_str.parse::<u32>().unwrap_or(1)
            };

            *counts.entry(symbol).or_insert(0) += count;
        }

        Ok(counts)
    }

    /// Calculate molar mass (g/mol) from formula string.
    pub fn molar_mass(formula: &str) -> Result<f64, ChemistryError> {
        let counts = Self::parse_formula(formula)?;
        let mut total_mass = 0.0;

        for (symbol, count) in counts {
            let el = Self::get_element(&symbol)
                .ok_or_else(|| ChemistryError::UnknownElement(symbol.clone()))?;
            total_mass += el.atomic_mass * (count as f64);
        }

        Ok(total_mass)
    }

    /// Convert mass (grams) to moles given molar mass (g/mol): n = m / M.
    pub fn mass_to_moles(mass_g: f64, molar_mass: f64) -> Result<f64, ChemistryError> {
        if mass_g < 0.0 || molar_mass <= 0.0 {
            return Err(ChemistryError::NegativeQuantity {
                param: "mass/molar_mass".into(),
                val: mass_g.min(molar_mass),
            });
        }
        Ok(mass_g / molar_mass)
    }

    /// Convert moles to mass (grams): m = n * M.
    pub fn moles_to_mass(moles: f64, molar_mass: f64) -> Result<f64, ChemistryError> {
        if moles < 0.0 || molar_mass <= 0.0 {
            return Err(ChemistryError::NegativeQuantity {
                param: "moles/molar_mass".into(),
                val: moles.min(molar_mass),
            });
        }
        Ok(moles * molar_mass)
    }

    /// Number of particles: N = n * N_A.
    pub fn particles_from_moles(moles: f64) -> f64 {
        moles * AVOGADRO_CONSTANT
    }

    /// Ideal Gas Law: P * V = n * R * T.
    /// Solves for Pressure (Pa) given n (mol), T (K), V (m^3).
    pub fn ideal_gas_pressure(
        n_moles: f64,
        temp_kelvin: f64,
        volume_m3: f64,
    ) -> Result<f64, ChemistryError> {
        if volume_m3 <= 0.0 {
            return Err(ChemistryError::DivisionByZero("volume must be > 0".into()));
        }
        if n_moles < 0.0 || temp_kelvin < 0.0 {
            return Err(ChemistryError::NegativeQuantity {
                param: "moles/temp".into(),
                val: n_moles.min(temp_kelvin),
            });
        }
        Ok((n_moles * GAS_CONSTANT * temp_kelvin) / volume_m3)
    }

    /// Solution molarity: M = moles / volume_liters.
    pub fn molarity(moles: f64, volume_liters: f64) -> Result<f64, ChemistryError> {
        if volume_liters <= 0.0 {
            return Err(ChemistryError::DivisionByZero("volume must be > 0".into()));
        }
        if moles < 0.0 {
            return Err(ChemistryError::NegativeQuantity {
                param: "moles".into(),
                val: moles,
            });
        }
        Ok(moles / volume_liters)
    }

    /// Dilution calculation: C1 * V1 = C2 * V2. Solves for V2 given C1, V1, C2.
    pub fn dilution_final_volume(c1: f64, v1: f64, c2: f64) -> Result<f64, ChemistryError> {
        if c2 <= 0.0 {
            return Err(ChemistryError::DivisionByZero(
                "final concentration must be > 0".into(),
            ));
        }
        if c1 < 0.0 || v1 < 0.0 {
            return Err(ChemistryError::NegativeQuantity {
                param: "c1/v1".into(),
                val: c1.min(v1),
            });
        }
        Ok((c1 * v1) / c2)
    }

    /// pH from hydronium ion concentration [H+]: pH = -log10[H+].
    pub fn ph(h_concentration: f64) -> Result<f64, ChemistryError> {
        if h_concentration <= 0.0 {
            return Err(ChemistryError::NegativeQuantity {
                param: "[H+] concentration".into(),
                val: h_concentration,
            });
        }
        Ok(-h_concentration.log10())
    }

    /// Hydronium ion concentration from pH: [H+] = 10^(-pH).
    pub fn hydronium_from_ph(ph_val: f64) -> f64 {
        10.0f64.powf(-ph_val)
    }
}
