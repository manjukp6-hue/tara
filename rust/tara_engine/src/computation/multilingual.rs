//! Multilingual Mathematics & Science Terminology Engine (English / Kannada / Symbols).
//!
//! Provides bidirectional terminology translation, semantic mapping, and phonetic explanations
//! for foundational math and science concepts without loss of technical rigor.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultilingualTerm {
    pub english: &'static str,
    pub kannada: &'static str,
    pub kannada_transliteration: &'static str,
    pub symbol: &'static str,
    pub category: &'static str,
    pub definition_en: &'static str,
    pub definition_kn: &'static str,
}

static DICTIONARY: OnceLock<Vec<MultilingualTerm>> = OnceLock::new();
static LOOKUP_EN: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
static LOOKUP_KN: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();

fn init_dictionary() -> (
    Vec<MultilingualTerm>,
    HashMap<&'static str, usize>,
    HashMap<&'static str, usize>,
) {
    let terms = vec![
        // Arithmetic & Foundation
        MultilingualTerm {
            english: "Addition",
            kannada: "ಕೂಡಿಸುವುದು",
            kannada_transliteration: "Koodisuvudu",
            symbol: "+",
            category: "Arithmetic",
            definition_en: "The arithmetic operation of combining two or more numbers into a single sum.",
            definition_kn: "ಎರಡು ಅಥವಾ ಹೆಚ್ಚಿನ ಸಂಖ್ಯೆಗಳನ್ನು ಒಟ್ಟುಗೂಡಿಸಿ ಮೊತ್ತವನ್ನು ಪಡೆಯುವ ಗಣಿತ ಪ್ರಕ್ರಿಯೆ.",
        },
        MultilingualTerm {
            english: "Subtraction",
            kannada: "ಕಳೆಯುವುದು",
            kannada_transliteration: "Kaleyuvudu",
            symbol: "-",
            category: "Arithmetic",
            definition_en: "The arithmetic operation of taking one number away from another to find the difference.",
            definition_kn: "ಒಂದು ಸಂಖ್ಯೆಯಿಂದ ಇನ್ನೊಂದು ಸಂಖ್ಯೆಯನ್ನು ತೆಗೆದು ವ್ಯತ್ಯಾಸವನ್ನು ಕಂಡುಹಿಡಿಯುವ ಪ್ರಕ್ರಿಯೆ.",
        },
        MultilingualTerm {
            english: "Multiplication",
            kannada: "ಗುಣಾಕಾರ",
            kannada_transliteration: "Gunaakaara",
            symbol: "×",
            category: "Arithmetic",
            definition_en: "Repeated addition of a number a specified number of times; product of factors.",
            definition_kn: "ಒಂದು ಸಂಖ್ಯೆಯನ್ನು ನಿಗದಿತ ಸಂಖ್ಯೆಯಷ್ಟು ಬಾರಿ ಪುನರಾವರ್ತಿತವಾಗಿ ಕೂಡಿಸುವ ಪ್ರಕ್ರಿಯೆ.",
        },
        MultilingualTerm {
            english: "Division",
            kannada: "ಭಾಗಾಕಾರ",
            kannada_transliteration: "Bhaagaakaara",
            symbol: "÷",
            category: "Arithmetic",
            definition_en: "The process of determining how many times one number is contained within another.",
            definition_kn: "ಒಂದು ಸಂಖ್ಯೆಯಲ್ಲಿ ಇನ್ನೊಂದು ಸಂಖ್ಯೆ ಎಷ್ಟು ಬಾರಿ ಅಡಕವಾಗಿದೆ ಎಂಬುದನ್ನು ಲೆಕ್ಕಾಚಾರ ಮಾಡುವ ಪ್ರಕ್ರಿಯೆ.",
        },
        MultilingualTerm {
            english: "Remainder",
            kannada: "ಶೇಷ",
            kannada_transliteration: "Shesha",
            symbol: "mod",
            category: "Arithmetic",
            definition_en: "The amount left over after integer division.",
            definition_kn: "ಪೂರ್ಣಾಂಕ ಭಾಗಾಕಾರದ ನಂತರ ಉಳಿಯುವ ಭಾಗ.",
        },
        MultilingualTerm {
            english: "Fraction",
            kannada: "ಭಿನ್ನರಾಶಿ",
            kannada_transliteration: "Bhinnaraashi",
            symbol: "a/b",
            category: "Arithmetic",
            definition_en: "A numerical quantity representing a part of a whole, expressed as numerator over denominator.",
            definition_kn: "ಪೂರ್ಣಾಂಕದ ಒಂದು ಭಾಗವನ್ನು ಅಂಶ ಮತ್ತು ಛೇದಗಳ ರೂಪದಲ್ಲಿ ನಿರೂಪಿಸುವ ಸಂಖ್ಯೆ.",
        },
        MultilingualTerm {
            english: "Decimal",
            kannada: "ದಶಮಾಂಶ",
            kannada_transliteration: "Dashamaamsha",
            symbol: ".",
            category: "Arithmetic",
            definition_en: "A fraction whose denominator is a power of ten, written with a decimal point.",
            definition_kn: "ಹತ್ತರ ಅಪವರ್ತ್ಯಗಳನ್ನು ಛೇದವಾಗಿ ಹೊಂದಿರುವ ಬಿಂದುವಿನ ಮೂಲಕ ಬರೆಯಲಾಗುವ ಸಂಖ್ಯೆ.",
        },
        MultilingualTerm {
            english: "Percentage",
            kannada: "ಶೇಕಡಾವಾರು",
            kannada_transliteration: "Shekadaavaaru",
            symbol: "%",
            category: "Arithmetic",
            definition_en: "A rate, number, or amount in each hundred.",
            definition_kn: "ಪ್ರತಿ ನೂರರಲ್ಲಿ ಇರುವ ಭಾಗ ಅಥವಾ ಪ್ರಮಾಣ.",
        },
        MultilingualTerm {
            english: "Prime Number",
            kannada: "ಅವಿಭಾಜ್ಯ ಸಂಖ್ಯೆ",
            kannada_transliteration: "Avibhaajya Sankhye",
            symbol: "p",
            category: "Number Theory",
            definition_en: "A natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers.",
            definition_kn: "ಒಂದು ಮತ್ತು ಅದೇ ಸಂಖ್ಯೆಯನ್ನು ಬಿಟ್ಟು ಬೇರೆ ಯಾವುದೇ ಅಪವರ್ತನಗಳನ್ನು ಹೊಂದಿರದ ಸಂಖ್ಯೆ.",
        },
        MultilingualTerm {
            english: "Square Root",
            kannada: "ವರ್ಗಮೂಲ",
            kannada_transliteration: "Vargamoola",
            symbol: "√",
            category: "Algebra",
            definition_en: "A factor of a number that, when multiplied by itself, gives the original number.",
            definition_kn: "ಒಂದು ಸಂಖ್ಯೆಯನ್ನು ಅದೇ ಸಂಖ್ಯೆಯಿಂದ ಗುಣಿಸಿದಾಗ ಮೂಲ ಸಂಖ್ಯೆಯನ್ನು ಕೊಡುವ ಮೂಲಾಂಕ.",
        },
        MultilingualTerm {
            english: "Cube Root",
            kannada: "ಘನಮೂಲ",
            kannada_transliteration: "Ghanamoola",
            symbol: "∛",
            category: "Algebra",
            definition_en: "A number that, when multiplied by itself three times, produces the given number.",
            definition_kn: "ಒಂದು ಸಂಖ್ಯೆಯನ್ನು ಮೂರು ಬಾರಿ ಪರಸ್ಪರ ಗುಣಿಸಿದಾಗ ಬರುವ ಮೂಲ ಸಂಖ್ಯೆ.",
        },
        MultilingualTerm {
            english: "Average",
            kannada: "ಸರಾಸರಿ",
            kannada_transliteration: "Saraasari",
            symbol: "μ",
            category: "Statistics",
            definition_en: "The quotient obtained by dividing the sum total of a set of figures by the number of figures.",
            definition_kn: "ದತ್ತಾಂಶಗಳ ಒಟ್ಟು ಮೊತ್ತವನ್ನು ದತ್ತಾಂಶಗಳ ಸಂಖ್ಯೆಯಿಂದ ಭಾಗಿಸಿದಾಗ ಬರುವ ಮೌಲ್ಯ.",
        },
        MultilingualTerm {
            english: "Perimeter",
            kannada: "ಸುತ್ತಳತೆ",
            kannada_transliteration: "Suttalate",
            symbol: "P",
            category: "Geometry",
            definition_en: "The continuous line forming the boundary of a closed geometric figure.",
            definition_kn: "ಆವೃತ ಜ್ಯಾಮಿತೀಯ ಆಕೃತಿಯ ಹೊರ ಅಂಚಿನ ಒಟ್ಟು ಉದ್ದ.",
        },
        MultilingualTerm {
            english: "Area",
            kannada: "ವಿಸ್ತೀರ್ಣ",
            kannada_transliteration: "Visteerna",
            symbol: "A",
            category: "Geometry",
            definition_en: "The extent or measurement of a surface or bounded region.",
            definition_kn: "ಒಂದು ಸಮತಲ ಆಕೃತಿಯು ಆಕ್ರಮಿಸಿಕೊಂಡಿರುವ ಮೇಲ್ಮೈ ವಿಸ್ತಾರ.",
        },
        MultilingualTerm {
            english: "Volume",
            kannada: "ಘನಫಲ",
            kannada_transliteration: "Ghanaphala",
            symbol: "V",
            category: "Geometry",
            definition_en: "The amount of 3-dimensional space occupied by a closed surface or substance.",
            definition_kn: "ತ್ರಿಕೋನೀಯ ಅಥವಾ ತ್ರಿ-ಆಯಾಮದ ವಸ್ತು ಆಕ್ರಮಿಸುವ ಒಟ್ಟು ಆಕಾಶ.",
        },
        MultilingualTerm {
            english: "Ratio",
            kannada: "ಅನುಪಾತ",
            kannada_transliteration: "Anupaatha",
            symbol: ":",
            category: "Arithmetic",
            definition_en: "The quantitative relation between two amounts showing the number of times one contains another.",
            definition_kn: "ಎರಡು ಪರಿಮಾಣಗಳ ನಡುವಿನ ಹೋಲಿಕೆಯ ಸಂಬಂಧ.",
        },
        MultilingualTerm {
            english: "Equation",
            kannada: "ಸಮೀಕರಣ",
            kannada_transliteration: "Sameekarana",
            symbol: "=",
            category: "Algebra",
            definition_en: "A mathematical statement asserting the equality of two expressions.",
            definition_kn: "ಎರಡು ಬೀಜಗಣಿತೀಯ ಅಭಿವ್ಯಕ್ತಿಗಳು ಸಮನಾಗಿವೆ ಎಂದು ತೋರಿಸುವ ಗಣಿತ ಹೇಳಿಕೆ.",
        },
        MultilingualTerm {
            english: "Matrix",
            kannada: "ಮಾತೃಕೆ",
            kannada_transliteration: "Maatruke",
            symbol: "[A]",
            category: "Linear Algebra",
            definition_en: "A rectangular array of numbers arranged in rows and columns.",
            definition_kn: "ಅಡ್ಡಸಾಲು ಮತ್ತು ಕಂಬಸಾಲುಗಳಲ್ಲಿ ಸಂಖ್ಯೆಗಳನ್ನು ಆಯತಾಕಾರದಲ್ಲಿ ಜೋಡಿಸಿದ ವ್ಯವಸ್ಥೆ.",
        },
        MultilingualTerm {
            english: "Vector",
            kannada: "ಸದಿಶ",
            kannada_transliteration: "Sadisha",
            symbol: "v⃗",
            category: "Physics & Linear Algebra",
            definition_en: "A quantity having both magnitude and direction.",
            definition_kn: "ಪರಿಮಾಣ ಮತ್ತು ದಿಕ್ಕು ಎರಡನ್ನೂ ಹೊಂದಿರುವ ಭೌತ ಪರಿಮಾಣ.",
        },
        MultilingualTerm {
            english: "Scalar",
            kannada: "ಅದಿಶ",
            kannada_transliteration: "Adisha",
            symbol: "s",
            category: "Physics & Linear Algebra",
            definition_en: "A quantity that has magnitude but no direction.",
            definition_kn: "ಕೇವಲ ಪರಿಮಾಣವನ್ನು ಮಾತ್ರ ಹೊಂದಿದ್ದು ದಿಕ್ಕನ್ನು ಹೊಂದಿರದ ಪರಿಮಾಣ.",
        },
        MultilingualTerm {
            english: "Velocity",
            kannada: "ವೇಗೋತ್ಕರ್ಷ / ವೇಗ",
            kannada_transliteration: "Vega",
            symbol: "v",
            category: "Physics",
            definition_en: "The rate of change of displacement with respect to time.",
            definition_kn: "ಕಾಲಕ್ಕೆ ಅನುಗುಣವಾಗಿ ಸ್ಥಾನಪಲ್ಲಟ ಬದಲಾಗುವ ದರ.",
        },
        MultilingualTerm {
            english: "Acceleration",
            kannada: "ವೇಗೋತ್ಕರ್ಷ",
            kannada_transliteration: "Vegotkarsha",
            symbol: "a",
            category: "Physics",
            definition_en: "The rate of change of velocity with respect to time.",
            definition_kn: "ಕಾಲಕ್ಕೆ ಅನುಗುಣವಾಗಿ ವೇಗ ಬದಲಾಗುವ ದರ.",
        },
        MultilingualTerm {
            english: "Force",
            kannada: "ಬಲ",
            kannada_transliteration: "Bala",
            symbol: "F",
            category: "Physics",
            definition_en: "An interaction that, when unopposed, changes the motion of an object (F = ma).",
            definition_kn: "ವಸ್ತುವಿನ ಚಲನೆ ಅಥವಾ ವಿಶ್ರಾಂತ ಸ್ಥಿತಿಯನ್ನು ಬದಲಾಯಿಸುವ ಬಾಹ್ಯ ಪ್ರಭಾವ.",
        },
        MultilingualTerm {
            english: "Energy",
            kannada: "ಶಕ್ತಿ",
            kannada_transliteration: "Shakti",
            symbol: "E",
            category: "Physics",
            definition_en: "The quantitative property that is transferred to a body or to a physical system to perform work.",
            definition_kn: "ಕೆಲಸ ಮಾಡುವ ಸಾಮರ್ಥ್ಯ.",
        },
        MultilingualTerm {
            english: "Gravity",
            kannada: "ಗುರುತ್ವಾಕರ್ಷಣೆ",
            kannada_transliteration: "Gurutvaakarshane",
            symbol: "g / G",
            category: "Physics",
            definition_en: "The universal force of attraction acting between all matter.",
            definition_kn: "ದ್ರವ್ಯರಾಶಿಯನ್ನು ಹೊಂದಿರುವ ಎಲ್ಲಾ ಕಾಯಗಳ ನಡುವೆ ಇರುವ ಆಕರ್ಷಕ ಬಲ.",
        },
        MultilingualTerm {
            english: "Atom",
            kannada: "ಪರಮಾಣು",
            kannada_transliteration: "Paramaanu",
            symbol: "atom",
            category: "Chemistry",
            definition_en: "The basic unit of a chemical element consisting of a nucleus surrounded by electrons.",
            definition_kn: "ಧಾತುವಿನ ಅತ್ಯಂತ ಸಣ್ಣ ರಾಸಾಯನಿಕ ಕಣ.",
        },
        MultilingualTerm {
            english: "Molecule",
            kannada: "ಅಣು",
            kannada_transliteration: "Anu",
            symbol: "mol",
            category: "Chemistry",
            definition_en: "A group of atoms bonded together representing the smallest fundamental unit of a chemical compound.",
            definition_kn: "ರಾಸಾಯನಿಕ ಬಂಧಗಳಿಂದ ಒಟ್ಟಿಗೆ ಹಿಡಿದಿಡಲಾದ ಪರಮಾಣುಗಳ ಸಮೂಹ.",
        },
        MultilingualTerm {
            english: "Cell",
            kannada: "ಜೀವಕೋಶ",
            kannada_transliteration: "Jeevakosha",
            symbol: "cell",
            category: "Biology",
            definition_en: "The basic structural, functional, and biological unit of all known organisms.",
            definition_kn: "ಸಕಲ ಜೀವಿಗಳ ರಚನಾತ್ಮಕ ಮತ್ತು ಕಾರ್ಯಾತ್ಮಕ ಮೂಲ ಘಟಕ.",
        },
        MultilingualTerm {
            english: "Algorithm",
            kannada: "ಕ್ರಮಾವಳಿ",
            kannada_transliteration: "Kramaavali",
            symbol: "algo",
            category: "Computer Science",
            definition_en: "A finite, unambiguous sequence of steps for solving a computational problem.",
            definition_kn: "ಒಂದು ಸಮಸ್ಯೆಯನ್ನು ಪರಿಹರಿಸಲು ಹಂತ-ಹಂತವಾಗಿ ರಚಿಸಲಾದ ನಿಯಮಗಳ ಪಟ್ಟಿ.",
        },
        MultilingualTerm {
            english: "Variable",
            kannada: "ಚರಾಂಶ",
            kannada_transliteration: "Charaamsha",
            symbol: "x",
            category: "Mathematics & Programming",
            definition_en: "A symbol or storage location representing a value that can change or vary.",
            definition_kn: "ಬದಲಾಗುವ ಅಥವಾ ವಿವಿಧ ಮೌಲ್ಯಗಳನ್ನು ಸ್ವೀಕರಿಸಬಲ್ಲ ಗಣಿತೀಯ ಅಥವಾ ಪ್ರೋಗ್ರಾಮಿಂಗ್ ಸಂಕೇತ.",
        },
    ];

    let mut en_map = HashMap::new();
    let mut kn_map = HashMap::new();

    for (idx, term) in terms.iter().enumerate() {
        en_map.insert(term.english, idx);
        kn_map.insert(term.kannada, idx);
    }

    (terms, en_map, kn_map)
}

fn get_dict() -> &'static [MultilingualTerm] {
    DICTIONARY.get_or_init(|| {
        let (terms, en_map, kn_map) = init_dictionary();
        LOOKUP_EN.set(en_map).ok();
        LOOKUP_KN.set(kn_map).ok();
        terms
    })
}

pub struct MultilingualEngine;

impl MultilingualEngine {
    /// Lookup by English term (case-insensitive).
    pub fn lookup_english(term: &str) -> Option<&'static MultilingualTerm> {
        let dict = get_dict();
        let en_map = LOOKUP_EN.get()?;
        let query = term.trim().to_lowercase();
        for (&key, &idx) in en_map.iter() {
            if key.to_lowercase() == query {
                return Some(&dict[idx]);
            }
        }
        None
    }

    /// Lookup by Kannada term.
    pub fn lookup_kannada(term: &str) -> Option<&'static MultilingualTerm> {
        let dict = get_dict();
        let kn_map = LOOKUP_KN.get()?;
        let query = term.trim();
        kn_map.get(query).map(|&idx| &dict[idx])
    }

    /// Search across English, Kannada, and transliterations.
    pub fn search(query: &str) -> Vec<&'static MultilingualTerm> {
        let dict = get_dict();
        let q = query.trim().to_lowercase();
        dict.iter()
            .filter(|item| {
                item.english.to_lowercase().contains(&q)
                    || item.kannada.contains(&q)
                    || item.kannada_transliteration.to_lowercase().contains(&q)
                    || item.category.to_lowercase().contains(&q)
            })
            .collect()
    }

    /// Return all registered multilingual terms.
    pub fn all_terms() -> &'static [MultilingualTerm] {
        get_dict()
    }
}
