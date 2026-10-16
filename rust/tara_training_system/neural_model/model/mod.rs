//! Model sub-module tree.
//!
//! Exposes the neural network building blocks and the top-level
//! [`TaraForCausalLM`] model.

pub mod attention;
pub mod causal_lm;
pub mod decoder_layer;
pub mod mlp;
pub mod rms_norm;
pub mod rope;

pub use causal_lm::TaraForCausalLM;
