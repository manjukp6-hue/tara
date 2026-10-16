//! On-Demand Chunked Streaming Helper for CPU Execution (`CpuStreamHelper`).
//!
//! When CPU is the active compute device, this helper prevents memory exhaustion (OOM),
//! thread hangs, and allocator churn by streaming sequences, layer tensors, and batch
//! samples on demand in small, bounded buffer windows instead of loading entire large
//! models or sequences into memory all at once.


/// Dynamic configuration for on-demand streaming.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

/// On-demand chunked streaming engine for CPU-bounded execution.
#[derive(Debug, Clone)]
pub struct CpuStreamHelper {
    config: StreamConfig,
}

impl CpuStreamHelper {
    /// Create a new stream helper with default or environment-tuned configuration.
    pub fn new() -> Self {
        Self::with_config(StreamConfig::default())
    }

    /// Create a new stream helper with explicit configuration.
    pub fn with_config(config: StreamConfig) -> Self {
        Self { config }
    }

    /// Stream a long sequence of tokens (e.g., 1419 tokens) into smaller, bounded compute chunks.
    /// Never loads or computes quadratic attention over the full sequence simultaneously.
    pub fn stream_sequence_chunks<'a>(&'a self, full_tokens: &'a [u32]) -> SequenceChunkIterator<'a> {
        SequenceChunkIterator {
            tokens: full_tokens,
            chunk_size: self.config.chunk_size_tokens.max(16),
            current_offset: 0,
        }
    }

    /// Calculate recommended micro-batch sequence partition length based on available RAM.
    pub fn recommend_chunk_size(&self, total_seq_len: usize, hidden_size: usize) -> usize {
        let bytes_per_token_approx = hidden_size * 4 * 4; // Activations + attention + projection
        let target_bytes_per_chunk = 64 * 1024 * 1024; // 64 MB per chunk target
        let max_tokens = target_bytes_per_chunk / bytes_per_token_approx.max(1);
        max_tokens.clamp(64, 512).min(total_seq_len)
    }

    /// Memory-efficient on-demand buffer recycler to avoid heap allocator churn.
    pub fn create_buffer_pool(&self, buffer_len: usize) -> StreamBufferPool {
        StreamBufferPool::new(buffer_len)
    }
}

/// Iterator streaming sequential token chunks on demand.
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
}

/// Reusable buffer pool to eliminate continuous reallocation during CPU streaming.
#[derive(Debug)]
pub struct StreamBufferPool {
    buffer_len: usize,
    pool: Vec<Vec<f32>>,
}

impl StreamBufferPool {
    pub fn new(buffer_len: usize) -> Self {
        Self {
            buffer_len,
            pool: Vec::with_capacity(4),
        }
    }

    /// Acquire a zeroed buffer from the pool or allocate a new one if pool is empty.
    pub fn acquire(&mut self) -> Vec<f32> {
        if let Some(mut buf) = self.pool.pop() {
            buf.resize(self.buffer_len, 0.0);
            buf.fill(0.0);
            buf
        } else {
            vec![0.0f32; self.buffer_len]
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&mut self, buf: Vec<f32>) {
        if self.pool.len() < 8 {
            self.pool.push(buf);
        }
    }
}
