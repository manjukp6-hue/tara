//! Extensible Partition Storage and Domain Registry.
//!
//! Provides dynamic partitioning by domain and topic without any hardcoded 1M limit or fixed category ceiling.
//! Future domains (e.g. biology, chemistry, medicine, robotics, materials) can be registered at runtime or
//! dynamically discovered from filesystem partitions.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct PartitionMeta {
    pub domain: String,
    pub subpartition: String,
    pub relative_path: String,
    pub description: String,
}

pub struct DomainPartitionRegistry {
    pub base_dir: PathBuf,
    partitions: HashMap<String, Vec<PartitionMeta>>,
}

impl DomainPartitionRegistry {
    pub fn new(base_dir: &Path) -> Self {
        let mut registry = Self {
            base_dir: base_dir.to_path_buf(),
            partitions: HashMap::new(),
        };

        registry.initialize_core_partitions();
        registry.discover_existing_partitions();
        registry
    }

    /// Initialize the standard core partition tree required by Phase 1.
    fn initialize_core_partitions(&mut self) {
        // 1. Reasoning
        self.register_partition(
            "reasoning",
            "deduction",
            "Formal deductive reasoning and syllogisms",
        );
        self.register_partition(
            "reasoning",
            "induction",
            "Empirical induction and generalization",
        );
        self.register_partition(
            "reasoning",
            "abduction",
            "Diagnostic reasoning and inference to best explanation",
        );
        self.register_partition(
            "reasoning",
            "propositional_logic",
            "Propositional logic, syntax, semantics, and truth tables",
        );
        self.register_partition(
            "reasoning",
            "predicate_logic",
            "First-order logic, quantifiers, and resolution",
        );
        self.register_partition(
            "reasoning",
            "proof_methods",
            "Direct, contrapositive, contradiction, induction",
        );
        self.register_partition(
            "reasoning",
            "argument_structures",
            "Toulmin models, warrants, premises",
        );
        self.register_partition(
            "reasoning",
            "fallacies",
            "Formal and informal cognitive fallacies",
        );
        self.register_partition(
            "reasoning",
            "bayesian_reasoning",
            "Priors, likelihoods, Bayesian updating, posteriors",
        );
        self.register_partition(
            "reasoning",
            "causal_reasoning",
            "Pearl do-calculus, DAGs, confounding, counterfactuals",
        );
        self.register_partition(
            "reasoning",
            "decision_reasoning",
            "Utility theory, multi-criteria decision making",
        );
        self.register_partition(
            "reasoning",
            "uncertainty_handling",
            "Dempster-Shafer, epistemic/aleatoric uncertainty",
        );
        self.register_partition(
            "reasoning",
            "falsification",
            "Popperian hypothesis falsification and counterexamples",
        );

        // 2. Mathematics (All 34 sub-partitions)
        let math_subpartitions = [
            (
                "arithmetic",
                "Elementary operations, divisibility, fractions, percentages",
            ),
            (
                "algebra",
                "Polynomials, equations, quadratic formula, inequalities, factorizations",
            ),
            (
                "geometry",
                "Euclidean plane & solid geometry, circles, polygons, area, volume",
            ),
            (
                "trigonometry",
                "Trigonometric ratios, identities, laws of sines and cosines",
            ),
            (
                "coordinate_geometry",
                "Lines, conic sections, parabolas, ellipses, hyperbolas, 3D coordinates",
            ),
            (
                "vectors",
                "Vector operations, dot product, cross product, projection, basis",
            ),
            (
                "matrices",
                "Matrix multiplication, transpose, rank, inverse, trace",
            ),
            (
                "determinants",
                "Determinant properties, expansion, Cramer's rule",
            ),
            (
                "calculus",
                "Limits, continuity, derivatives, product/quotient/chain rules",
            ),
            (
                "differential_equations",
                "First/second order ODEs, separation of variables, integrating factors",
            ),
            (
                "integral_calculus",
                "Definite & indefinite integrals, integration by parts, substitution",
            ),
            (
                "multivariable_calculus",
                "Partial derivatives, gradients, directional derivatives, Lagrange multipliers",
            ),
            (
                "vector_calculus",
                "Divergence, curl, line integrals, Green's, Stokes', Divergence theorems",
            ),
            (
                "complex_numbers",
                "Euler's formula, polar form, De Moivre's theorem, roots of unity",
            ),
            (
                "sequences_series",
                "Arithmetic/geometric progressions, Taylor series, convergence tests",
            ),
            (
                "probability",
                "Conditional probability, Bayes' theorem, distributions, expectations",
            ),
            (
                "statistics",
                "Mean, variance, standard error, Student's t, hypothesis tests, ANOVA",
            ),
            (
                "discrete_mathematics",
                "Recurrence relations, boolean algebra, pigeonhole principle",
            ),
            (
                "number_theory",
                "Prime numbers, modular arithmetic, Fermat's Little Theorem, Euler's totient",
            ),
            (
                "combinatorics",
                "Permutations, combinations, binomial theorem, generating functions",
            ),
            (
                "graph_theory",
                "Eulerian & Hamiltonian graphs, trees, shortest paths, planar graphs",
            ),
            (
                "set_theory",
                "ZFC axioms, unions, intersections, cardinalities, Cantor's theorem",
            ),
            ("logic", "Model theory, Godel's incompleteness, compactness"),
            (
                "linear_algebra",
                "Vector spaces, eigenvalues, eigenvectors, SVD, diagonalization",
            ),
            (
                "abstract_algebra",
                "Groups, rings, fields, homomorphisms, ideals, Galois theory",
            ),
            (
                "real_analysis",
                "Metric spaces, Cauchy sequences, Lebesgue integration, compactness",
            ),
            (
                "complex_analysis",
                "Cauchy-Riemann equations, residue theorem, conformal mappings",
            ),
            (
                "numerical_methods",
                "Newton-Raphson, Runge-Kutta, numerical quadrature, interpolation",
            ),
            (
                "optimization",
                "Convex optimization, gradient descent, KKT conditions, linear programming",
            ),
            (
                "differential_geometry",
                "Curves, surfaces, Christoffel symbols, Riemannian curvature tensor",
            ),
            (
                "topology",
                "Open sets, homeomorphisms, fundamental group, manifolds",
            ),
            (
                "mathematical_physics",
                "Fourier transforms, Green's functions, spherical harmonics, wavelets",
            ),
            (
                "indian_mathematics",
                "Sulba Sutras, Aryabhata, Brahmagupta, Bhaskara I & II, Madhava, Kerala school",
            ),
            (
                "mathematical_constants",
                "Pi, e, golden ratio phi, Euler-Mascheroni gamma, Feigenbaum alpha",
            ),
        ];

        for (sub, desc) in math_subpartitions {
            self.register_partition("mathematics", sub, desc);
        }

        // 3. Physics
        let physics_sub = [
            (
                "classical_mechanics",
                "Newton's laws, kinematics, work-energy, rotation, gravitation",
            ),
            (
                "thermodynamics",
                "Laws of thermodynamics, heat engines, entropy, phase transitions",
            ),
            (
                "electromagnetism",
                "Maxwell's equations, electrostatics, magnetostatics, waves",
            ),
            (
                "optics",
                "Geometric & wave optics, diffraction, interference, polarization",
            ),
            (
                "quantum_mechanics",
                "Schrodinger equation, uncertainty principle, spin, operators",
            ),
            (
                "relativity",
                "Special and General relativity, Lorentz transforms, Schwarzschild metric",
            ),
            (
                "atomic_nuclear",
                "Bohr model, nuclear decay, binding energy, fission/fusion",
            ),
        ];
        for (sub, desc) in physics_sub {
            self.register_partition("physics", sub, desc);
        }

        // 4. Science
        let science_sub = [
            (
                "scientific_method",
                "Observation, hypothesis, experiment, measurement, replication",
            ),
            (
                "measurement_units",
                "SI base units, derived units, dimensional analysis, uncertainty",
            ),
            (
                "astronomy",
                "Stellar evolution, HR diagram, planetary science, telescopes",
            ),
        ];
        for (sub, desc) in science_sub {
            self.register_partition("science", sub, desc);
        }

        // 5. Space
        let space_sub = [
            (
                "orbital_mechanics",
                "Kepler's laws, vis-viva equation, Hohmann transfer, escape velocity",
            ),
            (
                "cosmology",
                "Big Bang, Friedmann equations, CMB, Hubble-Lemaitre law, dark matter, dark energy",
            ),
            (
                "celestial_bodies",
                "Stars, planets, black holes, neutron stars, galaxies",
            ),
        ];
        for (sub, desc) in space_sub {
            self.register_partition("space", sub, desc);
        }

        // 6. Particles
        let particle_sub = [
            (
                "standard_model",
                "Quarks, leptons, gauge bosons, Higgs mechanism",
            ),
            (
                "fundamental_forces",
                "Strong, weak, electromagnetic, gravitational interactions",
            ),
            (
                "antimatter",
                "Positrons, antiprotons, CPT symmetry, baryon asymmetry",
            ),
        ];
        for (sub, desc) in particle_sub {
            self.register_partition("particles", sub, desc);
        }

        // 7. Programming & Languages
        self.register_partition(
            "programming",
            "paradigms",
            "Imperative, functional, object-oriented, concurrent",
        );
        self.register_partition(
            "programming",
            "algorithms",
            "Sorting, searching, graph traversal, dynamic programming",
        );
        self.register_partition(
            "programming",
            "data_structures",
            "Arrays, trees, hash tables, heaps, graphs",
        );

        let languages = [
            "rust",
            "python",
            "go",
            "java",
            "javascript",
            "c",
            "cpp",
            "csharp",
            "kotlin",
            "swift",
            "sql",
            "bash",
            "html",
            "css",
            "webassembly",
        ];
        for lang in languages {
            self.register_partition(
                "languages",
                lang,
                &format!("{} programming language reference and specification", lang),
            );
        }

        // 8. Standards & Manuals
        self.register_partition(
            "standards",
            "w3c",
            "W3C specifications, HTML, CSS, DOM, WCAG",
        );
        self.register_partition(
            "standards",
            "ietf_rfc",
            "IETF RFC specifications, HTTP, TCP/IP, TLS, JSON",
        );
        self.register_partition(
            "standards",
            "nist",
            "NIST cryptographic and security standards",
        );

        self.register_partition(
            "manuals",
            "os_posix",
            "POSIX specifications and OS engineering manuals",
        );
        self.register_partition(
            "manuals",
            "developer_tools",
            "Compiler, linker, and version control manuals",
        );

        // 9. Geography & General
        self.register_partition(
            "geography",
            "physical",
            "Continents, oceans, mountains, rivers, climates",
        );
        self.register_partition(
            "geography",
            "political",
            "Countries, capitals, geopolitical entities, borders",
        );
        self.register_partition(
            "general",
            "constants_units",
            "Fundamental physical constants and conversion factors",
        );
        self.register_partition(
            "general",
            "encyclopedic",
            "General verified technological and scientific references",
        );
    }

    /// Extensible registration of a new partition without modifying core engine logic.
    pub fn register_partition(&mut self, domain: &str, subpartition: &str, description: &str) {
        let domain_clean = domain.trim().to_lowercase().replace(' ', "_");
        let sub_clean = subpartition.trim().to_lowercase().replace(' ', "_");
        let rel_path = format!("partitions/{}/{}", domain_clean, sub_clean);

        let meta = PartitionMeta {
            domain: domain_clean.clone(),
            subpartition: sub_clean,
            relative_path: rel_path,
            description: description.to_string(),
        };

        self.partitions.entry(domain_clean).or_default().push(meta);
    }

    /// Automatically discover existing partitions on disk.
    pub fn discover_existing_partitions(&mut self) {
        let partitions_dir = self.base_dir.join("partitions");
        if !partitions_dir.exists() {
            return;
        }

        if let Ok(domains) = fs::read_dir(&partitions_dir) {
            for dom_entry in domains.flatten() {
                if dom_entry.path().is_dir() {
                    let domain_name = dom_entry.file_name().to_string_lossy().to_string();
                    if let Ok(subs) = fs::read_dir(dom_entry.path()) {
                        for sub_entry in subs.flatten() {
                            if sub_entry.path().is_dir() {
                                let sub_name = sub_entry.file_name().to_string_lossy().to_string();
                                if !self.has_partition(&domain_name, &sub_name) {
                                    self.register_partition(
                                        &domain_name,
                                        &sub_name,
                                        "Dynamically discovered domain partition",
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn has_partition(&self, domain: &str, subpartition: &str) -> bool {
        if let Some(subs) = self.partitions.get(&domain.to_lowercase()) {
            subs.iter()
                .any(|m| m.subpartition == subpartition.to_lowercase())
        } else {
            false
        }
    }

    /// Resolve target directory path for a document ID within a domain/subpartition.
    /// Incorporates 2-character hexadecimal shard prefix to prevent filesystem inode bottlenecks.
    pub fn resolve_storage_path(&self, domain: &str, subpartition: &str, doc_id: &str) -> PathBuf {
        let domain_clean = domain.trim().to_lowercase().replace(' ', "_");
        let sub_clean = if subpartition.trim().is_empty() {
            "general"
        } else {
            subpartition.trim()
        }
        .to_lowercase()
        .replace(' ', "_");

        let shard = if doc_id.len() >= 2 {
            &doc_id[..2]
        } else {
            "shard"
        };
        self.base_dir
            .join("partitions")
            .join(&domain_clean)
            .join(&sub_clean)
            .join(shard)
            .join(format!("{}.json", doc_id))
    }

    /// List all registered domains.
    pub fn list_domains(&self) -> Vec<String> {
        let mut doms: Vec<String> = self.partitions.keys().cloned().collect();
        doms.sort();
        doms
    }

    /// List all subpartitions for a domain.
    pub fn list_subpartitions(&self, domain: &str) -> Vec<PartitionMeta> {
        self.partitions
            .get(&domain.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }
}
