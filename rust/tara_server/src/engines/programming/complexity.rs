//! Algorithmic Big-O time and space complexity knowledge and classification engine.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComplexityProfile {
    pub algorithm_or_operation: &'static str,
    pub category: &'static str,
    pub time_best: &'static str,
    pub time_average: &'static str,
    pub time_worst: &'static str,
    pub space_worst: &'static str,
    pub explanation: &'static str,
}

static KNOWLEDGE_BASE: OnceLock<HashMap<&'static str, ComplexityProfile>> = OnceLock::new();

fn init_knowledge() -> HashMap<&'static str, ComplexityProfile> {
    let mut map = HashMap::new();

    map.insert(
        "binary_search",
        ComplexityProfile {
            algorithm_or_operation: "Binary Search",
            category: "Searching",
            time_best: "O(1)",
            time_average: "O(log n)",
            time_worst: "O(log n)",
            space_worst: "O(1)",
            explanation: "Halves the search space of a sorted array in each comparison step.",
        },
    );

    map.insert(
        "linear_search",
        ComplexityProfile {
            algorithm_or_operation: "Linear Search",
            category: "Searching",
            time_best: "O(1)",
            time_average: "O(n)",
            time_worst: "O(n)",
            space_worst: "O(1)",
            explanation: "Sequentially inspects each element until target is located or list ends.",
        },
    );

    map.insert(
        "quicksort",
        ComplexityProfile {
            algorithm_or_operation: "QuickSort",
            category: "Sorting",
            time_best: "O(n log n)",
            time_average: "O(n log n)",
            time_worst: "O(n^2)",
            space_worst: "O(log n)",
            explanation: "Divide-and-conquer partition around pivot; worst-case occurs on unbalanced partitions.",
        },
    );

    map.insert(
        "mergesort",
        ComplexityProfile {
            algorithm_or_operation: "MergeSort",
            category: "Sorting",
            time_best: "O(n log n)",
            time_average: "O(n log n)",
            time_worst: "O(n log n)",
            space_worst: "O(n)",
            explanation: "Guaranteed stable O(n log n) divide-and-conquer sort requiring O(n) auxiliary buffer.",
        },
    );

    map.insert(
        "heapsort",
        ComplexityProfile {
            algorithm_or_operation: "HeapSort",
            category: "Sorting",
            time_best: "O(n log n)",
            time_average: "O(n log n)",
            time_worst: "O(n log n)",
            space_worst: "O(1)",
            explanation: "In-place comparison sort using a binary max-heap with guaranteed O(n log n) upper bound.",
        },
    );

    map.insert(
        "hash_table_lookup",
        ComplexityProfile {
            algorithm_or_operation: "Hash Table Lookup",
            category: "Data Structure Operation",
            time_best: "O(1)",
            time_average: "O(1)",
            time_worst: "O(n)",
            space_worst: "O(n)",
            explanation: "Constant expected time with uniform hashing; degrades to O(n) on catastrophic hash collisions.",
        },
    );

    map.insert(
        "bfs",
        ComplexityProfile {
            algorithm_or_operation: "Breadth-First Search (BFS)",
            category: "Graph Traversal",
            time_best: "O(V + E)",
            time_average: "O(V + E)",
            time_worst: "O(V + E)",
            space_worst: "O(V)",
            explanation: "Level-order queue traversal exploring all neighbors before advancing to next depth.",
        },
    );

    map.insert(
        "dfs",
        ComplexityProfile {
            algorithm_or_operation: "Depth-First Search (DFS)",
            category: "Graph Traversal",
            time_best: "O(V + E)",
            time_average: "O(V + E)",
            time_worst: "O(V + E)",
            space_worst: "O(V)",
            explanation: "Recursive or stack traversal exploring deepest reachable nodes first.",
        },
    );

    map.insert(
        "dijkstra",
        ComplexityProfile {
            algorithm_or_operation: "Dijkstra's Shortest Path",
            category: "Graph Algorithm",
            time_best: "O((V + E) log V)",
            time_average: "O((V + E) log V)",
            time_worst: "O((V + E) log V)",
            space_worst: "O(V)",
            explanation: "Greedy priority queue search for single-source shortest paths on non-negative weighted graphs.",
        },
    );

    map.insert(
        "matrix_multiplication_naive",
        ComplexityProfile {
            algorithm_or_operation: "Naive Matrix Multiplication",
            category: "Numerical / Algebraic",
            time_best: "O(n^3)",
            time_average: "O(n^3)",
            time_worst: "O(n^3)",
            space_worst: "O(n^2)",
            explanation: "Three nested loops computing dot products across n x n dimensions.",
        },
    );

    map
}

pub struct ComplexityAnalyzer;

impl ComplexityAnalyzer {
    pub fn lookup(algo_key: &str) -> Option<&'static ComplexityProfile> {
        let kb = KNOWLEDGE_BASE.get_or_init(init_knowledge);
        let key = algo_key.trim().to_lowercase().replace([' ', '-'], "_");
        kb.get(key.as_str())
    }

    /// Estimate asymptotic complexity from detected nested loop depth.
    pub fn estimate_from_loop_depth(loop_depth: usize) -> (&'static str, &'static str) {
        match loop_depth {
            0 => ("O(1)", "O(1)"),
            1 => ("O(n)", "O(1)"),
            2 => ("O(n^2)", "O(1)"),
            3 => ("O(n^3)", "O(1)"),
            _ => ("O(n^k) polynomial", "O(1)"),
        }
    }
}
