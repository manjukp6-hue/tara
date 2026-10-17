//! Unified Stream Event Protocol and Flow Control Primitives for TARA-HD.
//!
//! Provides hardware-agnostic, typed stream events, thread-safe cancellation tokens,
//! and in-flight byte-bounded memory backpressure across CPU, GPU, and TPU execution.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Thread-safe cancellation token with zero-cost atomic checks.
/// Supports cooperative, token-boundary cancellation between CPU host and compute devices.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    /// Create a new, uncancelled token.
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Trigger cancellation across all threads observing this token.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Check if cancellation has been requested.
    #[inline(always)]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Reset token to uncancelled state.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }
}

/// In-flight memory tracker providing true byte-level backpressure.
/// Prevents large tensor buffers and token batches from overwhelming available memory.
#[derive(Debug, Clone)]
pub struct MemoryFlowGovernor {
    max_bytes: usize,
    in_flight_bytes: Arc<AtomicUsize>,
}

impl MemoryFlowGovernor {
    /// Create a new governor with explicit maximum in-flight byte capacity.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes: max_bytes.max(1024),
            in_flight_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Current bytes in-flight.
    pub fn current_in_flight_bytes(&self) -> usize {
        self.in_flight_bytes.load(Ordering::Relaxed)
    }

    /// Maximum in-flight byte allowance.
    pub fn max_in_flight_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Try to acquire a permit for the specified number of bytes.
    /// Returns true if within budget, false if capacity reached.
    pub fn try_acquire(&self, bytes: usize) -> bool {
        let mut current = self.in_flight_bytes.load(Ordering::Relaxed);
        loop {
            if current + bytes > self.max_bytes {
                return false;
            }
            match self.in_flight_bytes.compare_exchange_weak(
                current,
                current + bytes,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    /// Release bytes back to the budget when a buffer or chunk is consumed.
    pub fn release(&self, bytes: usize) {
        let mut current = self.in_flight_bytes.load(Ordering::Relaxed);
        loop {
            let new_val = current.saturating_sub(bytes);
            match self.in_flight_bytes.compare_exchange_weak(
                current,
                new_val,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

/// Unified typed stream event traversing the hardware-agnostic streaming pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    /// Incremental generated token string (e.g., chat streaming response).
    TokenChunk {
        sequence_id: u64,
        step: usize,
        token_id: u32,
        text: String,
        is_final: bool,
    },
    /// Raw tensor buffer chunk passing through compute stages (e.g. KV-cache, activations).
    TensorChunk {
        layer_index: usize,
        byte_size: usize,
        data: Vec<f32>,
    },
    /// Real-time training progress or execution metric.
    ProgressMetric {
        step: usize,
        total_steps: usize,
        loss: f32,
        tokens_per_second: f64,
        allocated_bytes: usize,
    },
    /// Periodic heartbeat keeping stream liveness alive during long compute passes.
    Heartbeat {
        timestamp_ms: u64,
    },
    /// Graceful stream completion marker.
    EndOfStream {
        sequence_id: u64,
        total_tokens: usize,
        elapsed_duration: Duration,
    },
    /// Structured stream failure without panicking.
    Error {
        code: String,
        message: String,
    },
}

impl StreamEvent {
    /// Estimate byte footprint of the event for memory backpressure accounting.
    pub fn estimated_bytes(&self) -> usize {
        match self {
            StreamEvent::TokenChunk { text, .. } => std::mem::size_of::<Self>() + text.len(),
            StreamEvent::TensorChunk { data, .. } => {
                std::mem::size_of::<Self>() + (data.len() * std::mem::size_of::<f32>())
            }
            StreamEvent::ProgressMetric { .. } => std::mem::size_of::<Self>(),
            StreamEvent::Heartbeat { .. } => std::mem::size_of::<Self>(),
            StreamEvent::EndOfStream { .. } => std::mem::size_of::<Self>(),
            StreamEvent::Error { code, message } => {
                std::mem::size_of::<Self>() + code.len() + message.len()
            }
        }
    }
}
