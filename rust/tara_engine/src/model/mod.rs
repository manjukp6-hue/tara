//! Model sub-module tree.
//!
//! Exposes the neural network building blocks and the top-level
//! [`TaraForCausalLM`] model.

pub mod rms_norm;
pub mod rope;
pub mod attention;
pub mod mlp;
pub mod decoder_layer;
pub mod causal_lm;

pub use causal_lm::TaraForCausalLM;
