//! # tara_engine
//!
//! Neural model inference engine for the TARA AI system.
//!
//! Provides a pure-Rust implementation of the TaraForCausalLM transformer,
//! SafeTensors weight loading, tokenization, sampling, and training utilities.

pub mod config;
pub mod tokenizer;
pub mod safetensors;
pub mod model;
pub mod generate;
pub mod control_tokens;
pub mod model_expansion;
pub mod trainer;
pub mod train_candidate;
pub mod skills_evaluator;
pub mod dataset;

pub use config::TaraConfig;
pub use tokenizer::TaraTokenizer;
pub use generate::{generate_response, generate_stream, GenerateResult};
pub use control_tokens::{ControlTokenAction, ControlTokenActionParser};
pub use model::causal_lm::TaraForCausalLM;
