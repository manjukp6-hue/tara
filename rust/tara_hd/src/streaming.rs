//! On-Demand Chunked Streaming Helper for CPU Execution (`CpuStreamHelper`).
//!
//! When CPU is the active compute device, this helper prevents memory exhaustion (OOM),
//! thread hangs, and allocator churn by streaming sequences, layer tensors, and batch
//! samples on demand in small, bounded buffer windows instead of loading entire large
//! models or sequences into memory all at once.

use serde::{Deserialize, Serialize};

/// Dynamic configuration for on-demand streaming.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Token window size for chunked sequence computation (e.g., 256 or 512 tokens).
    pub chunk_size_tokens: usize,
    /// Maximum number of layer activation buffers kept resident in RAM simultaneously.
    pub max_resident_layers: usize,
    /// Maximum memory budget in megabytes before forced eviction.
    pub memory_budget_mb: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size_tokens: 256,
            max_resident_layers: 2,
            memory_budget_mb: 4096,
        }
    }
}

/// Token chunk window for streaming sequence execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamWindow {
    /// Starting token index in the full sequence.
    pub start_idx: usize,
    /// Ending token index (exclusive) in the full sequence.
    pub end_idx: usize,
    /// Sliced token IDs within this active stream window.
    pub tokens: Vec<u32>,
    /// Whether this is the final chunk in the sequence stream.
    pub is_last_chunk: bool,
}

impl StreamWindow {
    /// Number of tokens in this window.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if the window contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// On-demand chunked streaming engine for CPU-bounded execution.
#[derive(Debug, Clone)]
pub struct CpuStreamHelper {
    config: StreamConfig,
}

impl Default for CpuStreamHelper {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuStreamHelper {
    /// Create a new stream helper with default configuration.
    pub fn new() -> Self {
        Self::with_config(StreamConfig::default())
    }

    /// Create a new stream helper with explicit configuration.
    pub fn with_config(mut config: StreamConfig) -> Self {
        config.chunk_size_tokens = config.chunk_size_tokens.max(1);
        config.max_resident_layers = config.max_resident_layers.max(1);
        config.memory_budget_mb = config.memory_budget_mb.max(64);
        Self { config }
    }

    /// Access active streaming configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }

    /// Stream a long sequence of tokens (e.g., 1419 tokens) into smaller, bounded compute chunks.
    /// Never loads or computes quadratic attention over the full sequence simultaneously.
    pub fn stream_sequence_chunks<'a>(
        &'a self,
        full_tokens: &'a [u32],
    ) -> SequenceChunkIterator<'a> {
        SequenceChunkIterator {
            tokens: full_tokens,
            chunk_size: self.config.chunk_size_tokens.max(1),
            current_offset: 0,
        }
    }

    /// Calculate recommended micro-batch sequence partition length based on configured RAM budget.
    pub fn recommend_chunk_size(&self, total_seq_len: usize, hidden_size: usize) -> usize {
        if total_seq_len == 0 {
            return 0;
        }
        let resident_layers = self.config.max_resident_layers.max(1);
        let bytes_per_token_approx = hidden_size.max(1) * 4 * 4 * resident_layers;
        // Allocate up to 1/16th of the total stream memory budget per active chunk window
        let target_bytes_per_chunk = (self.config.memory_budget_mb * 1024 * 1024 / 16)
            .clamp(16 * 1024 * 1024, 256 * 1024 * 1024);

        let max_tokens = target_bytes_per_chunk / bytes_per_token_approx.max(1);
        max_tokens.clamp(64, 512).min(total_seq_len)
    }

    /// Memory-efficient on-demand buffer recycler to avoid heap allocator churn.
    pub fn create_buffer_pool(&self, buffer_len: usize) -> StreamBufferPool {
        StreamBufferPool::new(buffer_len)
    }
}

/// Iterator streaming sequential token chunks on demand.
#[derive(Debug, Clone)]
pub struct SequenceChunkIterator<'a> {
    tokens: &'a [u32],
    chunk_size: usize,
    current_offset: usize,
}

impl<'a> Iterator for SequenceChunkIterator<'a> {
    type Item = StreamWindow;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.tokens.len() {
            return None;
        }

        let start = self.current_offset;
        let end = (start + self.chunk_size).min(self.tokens.len());
        self.current_offset = end;

        Some(StreamWindow {
            start_idx: start,
            end_idx: end,
            tokens: self.tokens[start..end].to_vec(),
            is_last_chunk: end >= self.tokens.len(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining_tokens = self.tokens.len().saturating_sub(self.current_offset);
        let chunks = remaining_tokens.div_ceil(self.chunk_size);
        (chunks, Some(chunks))
    }
}

impl<'a> ExactSizeIterator for SequenceChunkIterator<'a> {}

/// Reusable buffer pool to eliminate continuous reallocation during CPU streaming.
#[derive(Debug)]
pub struct StreamBufferPool {
    buffer_len: usize,
    max_pool_size: usize,
    pool: Vec<Vec<f32>>,
}

impl StreamBufferPool {
    pub fn new(buffer_len: usize) -> Self {
        Self::with_capacity(buffer_len, 8)
    }

    pub fn with_capacity(buffer_len: usize, max_pool_size: usize) -> Self {
        Self {
            buffer_len,
            max_pool_size: max_pool_size.max(1),
            pool: Vec::with_capacity(max_pool_size.min(8)),
        }
    }

    /// Acquire a zeroed buffer from the pool or allocate a new one if pool is empty.
    pub fn acquire(&mut self) -> Vec<f32> {
        if let Some(mut buf) = self.pool.pop() {
            // Because `release()` clears the buffer to len 0 while preserving capacity,
            // `resize` zeroes the buffer in a single pass without redundant work.
            buf.resize(self.buffer_len, 0.0);
            buf
        } else {
            vec![0.0f32; self.buffer_len]
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, mut buf: Vec<f32>) {
        if self.pool.len() < self.max_pool_size && buf.capacity() >= self.buffer_len {
            buf.clear();
            self.pool.push(buf);
        }
    }

    /// Number of buffers currently available in the pool.
    pub fn available_buffers(&self) -> usize {
        self.pool.len()
    }
}

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use crate::event::{CancellationToken, MemoryFlowGovernor, StreamEvent};

/// Universal, hardware-agnostic stream engine for TARA-HD.
/// Connects chat token generation, training iterations, and data flow across CPU, GPU, and TPU
/// with thread-safe cancellation and bounded memory flow control.
#[derive(Debug, Clone)]
pub struct UniversalStreamEngine {
    governor: MemoryFlowGovernor,
    cancel_token: CancellationToken,
}

impl Default for UniversalStreamEngine {
    fn default() -> Self {
        Self::new(32 * 1024 * 1024) // 32 MB default in-flight byte capacity
    }
}

impl UniversalStreamEngine {
    /// Create a universal stream engine with custom in-flight byte budget.
    pub fn new(max_in_flight_bytes: usize) -> Self {
        Self {
            governor: MemoryFlowGovernor::new(max_in_flight_bytes),
            cancel_token: CancellationToken::new(),
        }
    }

    /// Access the cooperative cancellation token.
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancel_token
    }

    /// Access memory flow governor.
    pub fn governor(&self) -> &MemoryFlowGovernor {
        &self.governor
    }

    /// Open a bounded, synchronous event stream channel.
    /// Returns (Sender, Receiver) pair with backpressure enforcement.
    pub fn open_stream(
        &self,
        bounded_queue_depth: usize,
    ) -> (UniversalStreamSender, UniversalStreamReceiver) {
        let (tx, rx) = sync_channel(bounded_queue_depth.max(1));
        let sender = UniversalStreamSender {
            tx,
            governor: self.governor.clone(),
            cancel_token: self.cancel_token.clone(),
        };
        let receiver = UniversalStreamReceiver {
            rx,
            governor: self.governor.clone(),
            cancel_token: self.cancel_token.clone(),
        };
        (sender, receiver)
    }
}

/// Producer side of the universal stream pipeline.
#[derive(Debug, Clone)]
pub struct UniversalStreamSender {
    tx: SyncSender<StreamEvent>,
    governor: MemoryFlowGovernor,
    cancel_token: CancellationToken,
}

impl UniversalStreamSender {
    /// Send an event downstream.
    /// Blocks if queue is full (Backpressure) or fails immediately if cancelled.
    pub fn send(&self, event: StreamEvent) -> Result<(), String> {
        if self.cancel_token.is_cancelled() {
            return Err("Stream processing cancelled by consumer".to_string());
        }

        let bytes = event.estimated_bytes();
        // Track byte footprint in the flow governor
        self.governor.try_acquire(bytes);

        self.tx
            .send(event)
            .map_err(|e| format!("Stream destination closed: {}", e))
    }

    /// Check if stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }
}

/// Consumer side of the universal stream pipeline.
pub struct UniversalStreamReceiver {
    rx: Receiver<StreamEvent>,
    governor: MemoryFlowGovernor,
    cancel_token: CancellationToken,
}

impl UniversalStreamReceiver {
    /// Receive next event from the stream.
    /// Automatically releases memory permits back to the governor as events are consumed.
    pub fn recv(&self) -> Result<StreamEvent, String> {
        match self.rx.recv() {
            Ok(event) => {
                let bytes = event.estimated_bytes();
                self.governor.release(bytes);
                Ok(event)
            }
            Err(e) => Err(format!("Stream source disconnected: {}", e)),
        }
    }

    /// Cooperative cancellation triggered by the consumer.
    pub fn cancel(&self) {
        self.cancel_token.cancel();
    }

    /// Check if stream has been cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }
}

