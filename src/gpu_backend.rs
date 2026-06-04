//! GPU inference backend via Metal compute (Plan 176).
//!
//! Implements [`InferenceBackend`] using Apple Metal compute shaders for the
//! transformer forward pass. All weights are uploaded to GPU memory during
//! `compile()`, and the forward pass executes entirely on GPU with only the
//! final logits copied back to CPU.
//!
//! # Architecture
//!
//! The GPU forward pass mirrors the CPU algorithm:
//! 1. Embedding lookup: `x = wte[token] + wpe[pos]`
//! 2. Per-layer: RMSNorm → QKV matmul → KV cache store → attention →
//!    output projection → residual → RMSNorm → MLP → residual
//! 3. LM head matmul: `logits = lm_head @ x`
//!
//! # Metal Shaders
//!
//! Compute kernels are embedded as MSL (Metal Shading Language) source strings
//! and compiled at pipeline creation time. Each kernel is a simple parallel
//! operation: RMSNorm, matrix-vector multiply, element-wise add, ReLU, etc.

use std::ffi::c_void;
use std::mem;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};

use crate::inference_backend::{CompileError, InferenceBackend};
use crate::transformer::{ForwardContext, MultiLayerKVCache, TransformerWeights};
use crate::types::{Config, kv_dim};

// ---------------------------------------------------------------------------
// Metal shader source
// ---------------------------------------------------------------------------

/// All Metal compute shaders as a single MSL source string.
const SHADER_SOURCE: &str = r#"
using namespace metal;

// RMSNorm: out[i] = x[i] * gamma[i] / sqrt(mean(x^2) + eps)
// Single-threaded kernel — simple and correct for n_embd <= 4096.
kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        float ss = 0.0;
        for (uint i = 0; i < n; i++) {
            ss += x[i] * x[i];
        }
        ss = 1.0 / sqrt(ss / float(n) + eps);
        for (uint i = 0; i < n; i++) {
            float val = x[i] * ss;
            out[i] = gamma ? val * gamma[i] : val;
        }
    }
}

// Matmul: out = W @ x  where W is [out_dim, in_dim], x is [in_dim]
// Each thread computes one output element (one row of W dotted with x).
kernel void matmul(
    device const float* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0;
    for (uint i = 0; i < in_dim; i++) {
        sum += W[gid * in_dim + i] * x[i];
    }
    out[gid] = sum;
}

// Element-wise add: out[i] = a[i] + b[i]
kernel void elem_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = a[gid] + b[gid];
}

// ReLU: out[i] = max(0, x[i])
kernel void relu(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = x[gid] > 0.0 ? x[gid] : 0.0;
}

// Attention score: for one head, compute q_h @ K_cached[0:seq_len] * scale
// Each thread computes the score for one cached position.
kernel void attention_score(
    device const float* q [[buffer(0)]],          // [n_embd] full query vector
    device const float* key_cache [[buffer(1)]],  // [block_size * kv_dim]
    constant uint& q_offset [[buffer(2)]],        // h * head_dim
    constant uint& kv_offset [[buffer(3)]],       // kv_group * head_dim
    constant uint& kv_dim [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& scale [[buffer(6)]],
    device float* scores [[buffer(7)]],           // [block_size] output scores
    uint gid [[thread_position_in_grid]]
) {
    float sum = 0.0;
    for (uint d = 0; d < head_dim; d++) {
        sum += q[q_offset + d] * key_cache[gid * kv_dim + kv_offset + d];
    }
    scores[gid] = sum * scale;
}

// Softmax: online softmax over scores[0:seq_len]
// Single-threaded — correct for block_size <= 8192.
kernel void softmax(
    device float* scores [[buffer(0)]],
    constant uint& seq_len [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        float max_val = scores[0];
        for (uint i = 1; i < seq_len; i++) {
            if (scores[i] > max_val) max_val = scores[i];
        }
        float sum = 0.0;
        for (uint i = 0; i < seq_len; i++) {
            scores[i] = exp(scores[i] - max_val);
            sum += scores[i];
        }
        for (uint i = 0; i < seq_len; i++) {
            scores[i] /= sum;
        }
    }
}

// Attention value: weighted sum of value cache for one head.
// Single-threaded — writes head_dim outputs for one head.
kernel void attention_value(
    device const float* scores [[buffer(0)]],      // [seq_len] softmaxed scores
    device const float* value_cache [[buffer(1)]], // [block_size * kv_dim]
    device float* attn_out [[buffer(2)]],          // [n_embd] output
    constant uint& out_offset [[buffer(3)]],       // h * head_dim
    constant uint& kv_offset [[buffer(4)]],        // kv_group * head_dim
    constant uint& kv_dim [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& seq_len [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid == 0) {
        for (uint d = 0; d < head_dim; d++) {
            float sum = 0.0;
            for (uint t = 0; t < seq_len; t++) {
                sum += scores[t] * value_cache[t * kv_dim + kv_offset + d];
            }
            attn_out[out_offset + d] = sum;
        }
    }
}

// KV store: copy k/v vectors into cache at position pos.
kernel void kv_store(
    device const float* k [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* key_cache [[buffer(2)]],
    device float* value_cache [[buffer(3)]],
    constant uint& pos [[buffer(4)]],
    constant uint& kv_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < kv_dim) {
        key_cache[pos * kv_dim + gid] = k[gid];
        value_cache[pos * kv_dim + gid] = v[gid];
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU pipeline & buffer helpers
// ---------------------------------------------------------------------------

/// Compiled Metal compute pipelines for each kernel.
struct GpuPipelines {
    rmsnorm: ComputePipelineState,
    matmul: ComputePipelineState,
    elem_add: ComputePipelineState,
    relu: ComputePipelineState,
    attention_score: ComputePipelineState,
    softmax: ComputePipelineState,
    attention_value: ComputePipelineState,
    kv_store: ComputePipelineState,
}

impl GpuPipelines {
    fn create(device: &metal::DeviceRef) -> Result<Self, CompileError> {
        let library = device
            .new_library_with_source(SHADER_SOURCE, &CompileOptions::new())
            .map_err(|e| CompileError::DeviceError(format!("shader compilation failed: {e}")))?;

        let make_pipeline = |name: &str| -> Result<ComputePipelineState, CompileError> {
            let function = library.get_function(name, None).map_err(|e| {
                CompileError::DeviceError(format!("function '{name}' not found: {e}"))
            })?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    CompileError::DeviceError(format!("pipeline creation failed for '{name}': {e}"))
                })
        };

        Ok(Self {
            rmsnorm: make_pipeline("rmsnorm")?,
            matmul: make_pipeline("matmul")?,
            elem_add: make_pipeline("elem_add")?,
            relu: make_pipeline("relu")?,
            attention_score: make_pipeline("attention_score")?,
            softmax: make_pipeline("softmax")?,
            attention_value: make_pipeline("attention_value")?,
            kv_store: make_pipeline("kv_store")?,
        })
    }
}

/// All transformer weights uploaded to GPU Metal buffers.
struct GpuWeightBuffers {
    // Embeddings are gathered from CPU-side slices per token/pos.
    // These buffers are kept for potential future fused embedding kernels.
    #[allow(dead_code)]
    wte: Buffer, // [vocab_size * n_embd]
    #[allow(dead_code)]
    wpe: Buffer, // [block_size * n_embd]
    lm_head: Buffer, // [vocab_size * n_embd]
    // Per-layer weights (flat Vec of buffers, indexed by layer)
    attn_wq: Vec<Buffer>,         // [n_layer] each [n_embd * n_embd]
    attn_wk: Vec<Buffer>,         // [n_layer] each [kv_dim * n_embd]
    attn_wv: Vec<Buffer>,         // [n_layer] each [kv_dim * n_embd]
    attn_wo: Vec<Buffer>,         // [n_layer] each [n_embd * n_embd]
    mlp_w1: Vec<Buffer>,          // [n_layer] each [mlp_hidden * n_embd]
    mlp_w2: Vec<Buffer>,          // [n_layer] each [n_embd * mlp_hidden]
    attn_norm_gamma: Vec<Buffer>, // [n_layer] each [n_embd]
    mlp_norm_gamma: Vec<Buffer>,  // [n_layer] each [n_embd]
}

/// Upload a `Vec<f32>` to a new Metal buffer with shared memory.
fn upload_buffer(device: &metal::DeviceRef, data: &[f32]) -> Buffer {
    let byte_len = (data.len() * mem::size_of::<f32>()) as u64;
    let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let contents = buffer.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const c_void, contents, byte_len as usize);
    }
    buffer
}

/// Create a zero-initialized Metal buffer of `len` floats.
fn zero_buffer(device: &metal::DeviceRef, len: usize) -> Buffer {
    let byte_len = (len * mem::size_of::<f32>()) as u64;
    let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let contents = buffer.contents();
    unsafe {
        std::ptr::write_bytes(contents as *mut u8, 0, byte_len as usize);
    }
    buffer
}

/// Upload a small scalar value as a Metal buffer (for constant arguments).
fn scalar_buffer<T: Copy>(device: &metal::DeviceRef, value: &T) -> Buffer {
    let byte_len = mem::size_of::<T>() as u64;
    let buffer = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let contents = buffer.contents();
    unsafe {
        std::ptr::copy_nonoverlapping(
            value as *const T as *const c_void,
            contents,
            byte_len as usize,
        );
    }
    buffer
}

// ---------------------------------------------------------------------------
// GpuBackend
// ---------------------------------------------------------------------------

/// GPU inference backend using Apple Metal compute shaders.
///
/// Starts uncompiled. Call [`compile()`](InferenceBackend::compile) with the
/// current weights + config to upload weights to GPU memory and create compute
/// pipelines. Until compiled, [`forward()`](InferenceBackend::forward) falls
/// back to CPU.
pub struct GpuBackend {
    device: Device,
    command_queue: CommandQueue,
    compiled: bool,
    needs_recompile: bool,
    config_snapshot: Option<Config>,

    // Compiled GPU resources (populated by compile())
    pipelines: Option<GpuPipelines>,
    weight_buffers: Option<GpuWeightBuffers>,

    // Runtime scratch buffers (allocated on first compile, reused across forward passes)
    x_buf: Option<Buffer>,        // [n_embd]
    xr_buf: Option<Buffer>,       // [n_embd]
    xr2_buf: Option<Buffer>,      // [n_embd]
    q_buf: Option<Buffer>,        // [n_embd]
    k_buf: Option<Buffer>,        // [kv_dim]
    v_buf: Option<Buffer>,        // [kv_dim]
    attn_out_buf: Option<Buffer>, // [n_embd]
    hidden_buf: Option<Buffer>,   // [mlp_hidden]
    scores_buf: Option<Buffer>,   // [block_size]
    logits_buf: Option<Buffer>,   // [vocab_size]
    // GPU-side KV cache: Vec of (key_buffer, value_buffer) per layer
    kv_cache: Option<Vec<(Buffer, Buffer)>>,
}

impl GpuBackend {
    /// Create a new uncompiled GPU backend.
    ///
    /// Returns an error if no Metal device is available (should not happen on
    /// any Apple Silicon or Intel Mac with GPU).
    pub fn new() -> Result<Self, CompileError> {
        let device = Device::system_default()
            .ok_or_else(|| CompileError::DeviceError("no Metal device available".to_string()))?;
        let command_queue = device.new_command_queue();
        Ok(Self {
            device,
            command_queue,
            compiled: false,
            needs_recompile: false,
            config_snapshot: None,
            pipelines: None,
            weight_buffers: None,
            x_buf: None,
            xr_buf: None,
            xr2_buf: None,
            q_buf: None,
            k_buf: None,
            v_buf: None,
            attn_out_buf: None,
            hidden_buf: None,
            scores_buf: None,
            logits_buf: None,
            kv_cache: None,
        })
    }
}

impl InferenceBackend for GpuBackend {
    fn forward<'a>(
        &'a mut self,
        ctx: &'a mut ForwardContext,
        weights: &TransformerWeights,
        cache: &mut MultiLayerKVCache,
        token: usize,
        pos: usize,
        config: &Config,
    ) -> &'a mut [f32] {
        // Fall back to CPU if not compiled
        if !self.compiled {
            return crate::transformer::forward(ctx, weights, cache, token, pos, config);
        }

        let pipelines = self
            .pipelines
            .as_ref()
            .expect("pipelines missing after compile");
        let wb = self
            .weight_buffers
            .as_ref()
            .expect("weights missing after compile");
        let kv_cache = self
            .kv_cache
            .as_ref()
            .expect("kv_cache missing after compile");

        let n_embd = config.n_embd;
        let kv_dim = kv_dim(config);
        let head_dim = config.head_dim;
        let n_head = config.n_head;
        let n_kv_head = config.n_kv_head;
        let mlp_hidden = config.mlp_hidden;
        let vocab_size = config.vocab_size;
        let eps = config.rms_norm_eps as f32;
        let scale = (1.0f32 / (head_dim as f32).sqrt()) as f32;
        let seq_len = pos + 1; // number of cached positions

        // Shortcut references to scratch buffers
        let x_buf = self.x_buf.as_ref().unwrap();
        let xr_buf = self.xr_buf.as_ref().unwrap();
        let xr2_buf = self.xr2_buf.as_ref().unwrap();
        let q_buf = self.q_buf.as_ref().unwrap();
        let k_buf = self.k_buf.as_ref().unwrap();
        let v_buf = self.v_buf.as_ref().unwrap();
        let attn_out_buf = self.attn_out_buf.as_ref().unwrap();
        let hidden_buf = self.hidden_buf.as_ref().unwrap();
        let scores_buf = self.scores_buf.as_ref().unwrap();
        let logits_buf = self.logits_buf.as_ref().unwrap();

        let device = &*self.device;
        let command_queue = &self.command_queue;

        // Helper: dispatch a 1D compute kernel with `n` total threads
        let dispatch_1d =
            |encoder: &metal::ComputeCommandEncoderRef, pipeline: &ComputePipelineState, n: u64| {
                encoder.set_compute_pipeline_state(pipeline);
                let tg_size = pipeline.thread_execution_width();
                let threadgroup_count = MTLSize::new((n + tg_size - 1) / tg_size, 1, 1);
                let threads_per_tg = MTLSize::new(tg_size, 1, 1);
                encoder.dispatch_thread_groups(threadgroup_count, threads_per_tg);
            };

        // Helper: dispatch a single-thread kernel (for sequential ops like rmsnorm, softmax)
        let dispatch_single = |encoder: &metal::ComputeCommandEncoderRef,
                               pipeline: &ComputePipelineState| {
            encoder.set_compute_pipeline_state(pipeline);
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
        };

        // Helper constants as buffers (created once per forward pass, reused across encoder calls)
        let n_embd_buf = scalar_buffer(device, &(n_embd as u32));
        let kv_dim_buf = scalar_buffer(device, &(kv_dim as u32));
        let head_dim_buf = scalar_buffer(device, &(head_dim as u32));
        let eps_buf = scalar_buffer(device, &eps);
        let scale_buf = scalar_buffer(device, &scale);
        let seq_len_buf = scalar_buffer(device, &(seq_len as u32));
        let n_embd_in_dim_buf = scalar_buffer(device, &(n_embd as u32)); // in_dim for matmul
        let mlp_in_dim_buf = scalar_buffer(device, &(n_embd as u32)); // in_dim for mlp_w1 matmul
        let mlp_hidden_in_dim_buf = scalar_buffer(device, &(mlp_hidden as u32)); // in_dim for mlp_w2 matmul

        // ── Step 1: Embedding lookup x = wte[token] + wpe[pos] ──
        {
            // Copy wte[token*n_embd..(token+1)*n_embd] into x_buf
            let wte_offset = token * n_embd;
            let wte_slice = &weights.wte[wte_offset..wte_offset + n_embd];
            let x_upload = upload_buffer(device, wte_slice);
            // Copy wpe[pos*n_embd..(pos+1)*n_embd] into xr_buf (reuse as temp)
            let wpe_offset = pos * n_embd;
            let wpe_slice = &weights.wpe[wpe_offset..wpe_offset + n_embd];
            let wpe_upload = upload_buffer(device, wpe_slice);
            // x_buf = wte + wpe
            let cmd_buffer = command_queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();
            encoder.set_buffer(0, Some(&x_upload), 0);
            encoder.set_buffer(1, Some(&wpe_upload), 0);
            encoder.set_buffer(2, Some(x_buf), 0);
            dispatch_1d(encoder, &pipelines.elem_add, n_embd as u64);
            encoder.end_encoding();
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();
        }

        // ── Step 2: Per-layer transformer block ──
        for layer_idx in 0..config.n_layer {
            let cmd_buffer = command_queue.new_command_buffer();
            let encoder = cmd_buffer.new_compute_command_encoder();

            // 2a. RMSNorm: xr = rmsnorm(x, attn_norm_gamma)
            encoder.set_buffer(0, Some(x_buf), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(&wb.attn_norm_gamma[layer_idx]), 0);
            encoder.set_buffer(3, Some(&n_embd_buf), 0);
            encoder.set_buffer(4, Some(&eps_buf), 0);
            dispatch_single(encoder, &pipelines.rmsnorm);

            // 2b. QKV projection: q = wq @ xr, k = wk @ xr, v = wv @ xr
            // Query: q = attn_wq @ xr  [n_embd output, n_embd input]
            encoder.set_buffer(0, Some(&wb.attn_wq[layer_idx]), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(q_buf), 0);
            encoder.set_buffer(3, Some(&n_embd_in_dim_buf), 0);
            dispatch_1d(encoder, &pipelines.matmul, n_embd as u64);

            // Key: k = attn_wk @ xr  [kv_dim output, n_embd input]
            let kv_in_dim_buf = scalar_buffer(device, &(n_embd as u32));
            encoder.set_buffer(0, Some(&wb.attn_wk[layer_idx]), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(k_buf), 0);
            encoder.set_buffer(3, Some(&kv_in_dim_buf), 0);
            dispatch_1d(encoder, &pipelines.matmul, kv_dim as u64);

            // Value: v = attn_wv @ xr  [kv_dim output, n_embd input]
            encoder.set_buffer(0, Some(&wb.attn_wv[layer_idx]), 0);
            encoder.set_buffer(1, Some(xr_buf), 0);
            encoder.set_buffer(2, Some(v_buf), 0);
            encoder.set_buffer(3, Some(&kv_in_dim_buf), 0);
            dispatch_1d(encoder, &pipelines.matmul, kv_dim as u64);

            // 2c. KV cache store: write k, v into cache at position pos
            let (key_cache_buf, value_cache_buf) = &kv_cache[layer_idx];
            let pos_buf = scalar_buffer(device, &(pos as u32));
            encoder.set_buffer(0, Some(k_buf), 0);
            encoder.set_buffer(1, Some(v_buf), 0);
            encoder.set_buffer(2, Some(key_cache_buf), 0);
            encoder.set_buffer(3, Some(value_cache_buf), 0);
            encoder.set_buffer(4, Some(&pos_buf), 0);
            encoder.set_buffer(5, Some(&kv_dim_buf), 0);
            dispatch_1d(encoder, &pipelines.kv_store, kv_dim as u64);

            // 2d. Multi-head attention
            // For each head, compute attention scores, softmax, and weighted value sum
            // We process heads sequentially in separate command buffers because
            // each head needs its own softmax before the value sum.
            encoder.end_encoding();
            cmd_buffer.commit();
            cmd_buffer.wait_until_completed();

            for h in 0..n_head {
                let kv_group = h * n_kv_head / n_head;
                let q_offset = (h * head_dim) as u32;
                let kv_offset = (kv_group * head_dim) as u32;
                let out_offset = (h * head_dim) as u32;

                let q_off_buf = scalar_buffer(device, &q_offset);
                let kv_off_buf = scalar_buffer(device, &kv_offset);
                let out_off_buf = scalar_buffer(device, &out_offset);

                let (key_cache_buf, value_cache_buf) = &kv_cache[layer_idx];

                let cmd_buf = command_queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();

                // Attention scores: scores[t] = q[h*head_dim..] . key_cache[t*kv_dim + kv_group*head_dim..] * scale
                enc.set_buffer(0, Some(q_buf), 0);
                enc.set_buffer(1, Some(key_cache_buf), 0);
                enc.set_buffer(2, Some(&q_off_buf), 0);
                enc.set_buffer(3, Some(&kv_off_buf), 0);
                enc.set_buffer(4, Some(&kv_dim_buf), 0);
                enc.set_buffer(5, Some(&head_dim_buf), 0);
                enc.set_buffer(6, Some(&scale_buf), 0);
                enc.set_buffer(7, Some(scores_buf), 0);
                dispatch_1d(enc, &pipelines.attention_score, seq_len as u64);

                // Softmax over scores[0..seq_len]
                enc.set_buffer(0, Some(scores_buf), 0);
                enc.set_buffer(1, Some(&seq_len_buf), 0);
                dispatch_single(enc, &pipelines.softmax);

                // Weighted value sum: attn_out[h*head_dim..] = sum_t scores[t] * value_cache[t*kv_dim + kv_offset..]
                enc.set_buffer(0, Some(scores_buf), 0);
                enc.set_buffer(1, Some(value_cache_buf), 0);
                enc.set_buffer(2, Some(attn_out_buf), 0);
                enc.set_buffer(3, Some(&out_off_buf), 0);
                enc.set_buffer(4, Some(&kv_off_buf), 0);
                enc.set_buffer(5, Some(&kv_dim_buf), 0);
                enc.set_buffer(6, Some(&head_dim_buf), 0);
                enc.set_buffer(7, Some(&seq_len_buf), 0);
                dispatch_single(enc, &pipelines.attention_value);

                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // 2e. Output projection: x = wo @ attn_out  [n_embd output, n_embd input]
            {
                let cmd_buf = command_queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();
                enc.set_buffer(0, Some(&wb.attn_wo[layer_idx]), 0);
                enc.set_buffer(1, Some(attn_out_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                enc.set_buffer(3, Some(&n_embd_in_dim_buf), 0);
                dispatch_1d(enc, &pipelines.matmul, n_embd as u64);
                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // 2f. Residual: x += xr
            {
                let cmd_buf = command_queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();
                enc.set_buffer(0, Some(x_buf), 0);
                enc.set_buffer(1, Some(xr_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                dispatch_1d(enc, &pipelines.elem_add, n_embd as u64);
                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // 2g. Save pre-norm x to xr2, then RMSNorm x in-place
            // xr2 = x (save pre-norm value for residual)
            // x = rmsnorm(x, mlp_norm_gamma)
            {
                // Copy x_buf → xr2_buf via shared memory (zero-copy on Apple Silicon)
                let src_ptr = x_buf.contents() as *const f32;
                let dst_ptr = xr2_buf.contents() as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, n_embd);
                }
            }
            // RMSNorm x_buf in-place
            {
                let cmd_buf = command_queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();
                enc.set_buffer(0, Some(x_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0); // in-place: output = input
                enc.set_buffer(2, Some(&wb.mlp_norm_gamma[layer_idx]), 0);
                enc.set_buffer(3, Some(&n_embd_buf), 0);
                enc.set_buffer(4, Some(&eps_buf), 0);
                dispatch_single(enc, &pipelines.rmsnorm);
                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // 2h. MLP: hidden = relu(w1 @ x), then x = w2 @ hidden
            // hidden = w1 @ x  [mlp_hidden output, n_embd input]
            {
                let cmd_buf = command_queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();
                enc.set_buffer(0, Some(&wb.mlp_w1[layer_idx]), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(hidden_buf), 0);
                enc.set_buffer(3, Some(&mlp_in_dim_buf), 0);
                dispatch_1d(enc, &pipelines.matmul, mlp_hidden as u64);

                // ReLU on hidden
                enc.set_buffer(0, Some(hidden_buf), 0);
                enc.set_buffer(1, Some(hidden_buf), 0);
                dispatch_1d(enc, &pipelines.relu, mlp_hidden as u64);

                // x = w2 @ hidden  [n_embd output, mlp_hidden input]
                enc.set_buffer(0, Some(&wb.mlp_w2[layer_idx]), 0);
                enc.set_buffer(1, Some(hidden_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                enc.set_buffer(3, Some(&mlp_hidden_in_dim_buf), 0);
                dispatch_1d(enc, &pipelines.matmul, n_embd as u64);

                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }

            // 2i. Residual: x += xr2 (pre-norm value)
            {
                let cmd_buf = command_queue.new_command_buffer();
                let enc = cmd_buf.new_compute_command_encoder();
                enc.set_buffer(0, Some(x_buf), 0);
                enc.set_buffer(1, Some(xr2_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                dispatch_1d(enc, &pipelines.elem_add, n_embd as u64);
                enc.end_encoding();
                cmd_buf.commit();
                cmd_buf.wait_until_completed();
            }
        }

        // ── Step 3: LM head: logits = lm_head @ x ──
        {
            let cmd_buf = command_queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_buffer(0, Some(&wb.lm_head), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(logits_buf), 0);
            enc.set_buffer(3, Some(&n_embd_in_dim_buf), 0);
            dispatch_1d(enc, &pipelines.matmul, vocab_size as u64);
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        // ── Step 4: Copy logits from GPU back to CPU ──
        {
            let ptr = logits_buf.contents() as *const f32;
            let gpu_logits = unsafe { std::slice::from_raw_parts(ptr, vocab_size) };
            ctx.logits[..vocab_size].copy_from_slice(gpu_logits);
        }

        // Also update CPU KV cache for fallback consistency
        cache.advance_pos(pos);

        &mut ctx.logits[..vocab_size]
    }

    fn device_name(&self) -> &'static str {
        "GPU"
    }

    fn compile(
        &mut self,
        weights: &TransformerWeights,
        config: &Config,
    ) -> Result<(), CompileError> {
        let device = &*self.device;

        // Compile Metal shaders and create pipelines
        let pipelines = GpuPipelines::create(device)?;

        // Upload all weights to GPU buffers
        let wb = GpuWeightBuffers {
            wte: upload_buffer(device, &weights.wte),
            wpe: upload_buffer(device, &weights.wpe),
            lm_head: upload_buffer(device, &weights.lm_head),
            attn_wq: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.attn_wq))
                .collect(),
            attn_wk: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.attn_wk))
                .collect(),
            attn_wv: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.attn_wv))
                .collect(),
            attn_wo: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.attn_wo))
                .collect(),
            mlp_w1: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.mlp_w1))
                .collect(),
            mlp_w2: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.mlp_w2))
                .collect(),
            attn_norm_gamma: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.attn_norm_gamma))
                .collect(),
            mlp_norm_gamma: weights
                .layers
                .iter()
                .map(|l| upload_buffer(device, &l.mlp_norm_gamma))
                .collect(),
        };

        let n_embd = config.n_embd;
        let kvd = kv_dim(config);
        let block_size = config.block_size;

        // Allocate runtime scratch buffers
        let x_buf = zero_buffer(device, n_embd);
        let xr_buf = zero_buffer(device, n_embd);
        let xr2_buf = zero_buffer(device, n_embd);
        let q_buf = zero_buffer(device, n_embd);
        let k_buf = zero_buffer(device, kvd);
        let v_buf = zero_buffer(device, kvd);
        let attn_out_buf = zero_buffer(device, n_embd);
        let hidden_buf = zero_buffer(device, config.mlp_hidden);
        let scores_buf = zero_buffer(device, block_size);
        let logits_buf = zero_buffer(device, config.vocab_size);

        // Allocate GPU-side KV cache: per-layer (key, value) buffers
        let kv_cache: Vec<(Buffer, Buffer)> = (0..config.n_layer)
            .map(|_| {
                (
                    zero_buffer(device, block_size * kvd),
                    zero_buffer(device, block_size * kvd),
                )
            })
            .collect();

        // Store everything
        self.pipelines = Some(pipelines);
        self.weight_buffers = Some(wb);
        self.x_buf = Some(x_buf);
        self.xr_buf = Some(xr_buf);
        self.xr2_buf = Some(xr2_buf);
        self.q_buf = Some(q_buf);
        self.k_buf = Some(k_buf);
        self.v_buf = Some(v_buf);
        self.attn_out_buf = Some(attn_out_buf);
        self.hidden_buf = Some(hidden_buf);
        self.scores_buf = Some(scores_buf);
        self.logits_buf = Some(logits_buf);
        self.kv_cache = Some(kv_cache);
        self.config_snapshot = Some(config.clone());
        self.compiled = true;
        self.needs_recompile = false;

        Ok(())
    }

    fn is_compiled(&self) -> bool {
        self.compiled
    }

    fn recompile_hint(&mut self) {
        self.needs_recompile = true;
    }

    fn reset(&mut self) {
        // Mark for recompile so next compile() re-uploads weights
        if self.compiled {
            self.needs_recompile = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer;
    use crate::types::Rng;

    /// Create micro test fixtures (small model config + random weights).
    fn micro_fixtures() -> (
        Config,
        TransformerWeights,
        ForwardContext,
        MultiLayerKVCache,
    ) {
        let config = Config::micro();
        let mut rng = Rng::new(42);
        let weights = TransformerWeights::new(&config, &mut rng);
        let ctx = ForwardContext::new(&config);
        let cache = MultiLayerKVCache::new(&config);
        (config, weights, ctx, cache)
    }

    /// Cosine similarity between two f32 slices.
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (mag_a * mag_b + 1e-8)
    }

    #[test]
    fn test_gpu_backend_new_uncompiled() {
        let backend = GpuBackend::new();
        // May fail on CI without Metal, so just check the happy path
        if let Ok(b) = backend {
            assert!(!b.is_compiled(), "new backend should not be compiled");
        }
    }

    #[test]
    fn test_gpu_backend_compile_marks_compiled() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return, // Skip on non-Metal environments
        };
        assert!(backend.compile(&weights, &config).is_ok());
        assert!(backend.is_compiled());
    }

    #[test]
    fn test_gpu_backend_device_name() {
        let backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        assert_eq!(backend.device_name(), "GPU");
    }

    #[test]
    fn test_gpu_forward_matches_cpu() {
        let (config, weights, _, _) = micro_fixtures();

        // CPU reference
        let mut ctx_cpu = ForwardContext::new(&config);
        let mut cache_cpu = MultiLayerKVCache::new(&config);
        let cpu_logits =
            transformer::forward(&mut ctx_cpu, &weights, &mut cache_cpu, 0, 0, &config).to_vec();

        // GPU
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return; // Skip if compilation fails
        }
        let mut ctx_gpu = ForwardContext::new(&config);
        let mut cache_gpu = MultiLayerKVCache::new(&config);
        let gpu_logits = backend
            .forward(&mut ctx_gpu, &weights, &mut cache_gpu, 0, 0, &config)
            .to_vec();

        let sim = cosine_similarity(&cpu_logits, &gpu_logits);
        assert!(
            sim >= 0.999,
            "GPU logits should match CPU (cosine sim = {sim:.6}, expected >= 0.999)\n\
             CPU first 10: {:?}\n\
             GPU first 10: {:?}",
            &cpu_logits[..cpu_logits.len().min(10)],
            &gpu_logits[..gpu_logits.len().min(10)],
        );
    }

    #[test]
    fn test_gpu_forward_multi_token_matches_cpu() {
        let (config, weights, _, _) = micro_fixtures();
        let tokens = [0usize, 1, 5, 10, 3];

        // CPU reference: run multiple tokens
        let mut ctx_cpu = ForwardContext::new(&config);
        let mut cache_cpu = MultiLayerKVCache::new(&config);
        let mut cpu_logits_all = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let logits =
                transformer::forward(&mut ctx_cpu, &weights, &mut cache_cpu, token, pos, &config)
                    .to_vec();
            cpu_logits_all.push(logits);
        }

        // GPU: run same tokens
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }
        let mut ctx_gpu = ForwardContext::new(&config);
        let mut cache_gpu = MultiLayerKVCache::new(&config);
        let mut gpu_logits_all = Vec::new();
        for (pos, &token) in tokens.iter().enumerate() {
            let logits = backend
                .forward(&mut ctx_gpu, &weights, &mut cache_gpu, token, pos, &config)
                .to_vec();
            gpu_logits_all.push(logits);
        }

        // Compare each position
        for (pos, (cpu, gpu)) in cpu_logits_all.iter().zip(gpu_logits_all.iter()).enumerate() {
            let sim = cosine_similarity(cpu, gpu);
            assert!(
                sim >= 0.999,
                "GPU logits should match CPU at pos {pos} (cosine sim = {sim:.6})"
            );
        }
    }

    #[test]
    fn test_gpu_forward_deterministic() {
        let (config, weights, _, _) = micro_fixtures();
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        if backend.compile(&weights, &config).is_err() {
            return;
        }

        let mut ctx1 = ForwardContext::new(&config);
        let mut cache1 = MultiLayerKVCache::new(&config);
        let run1 = backend
            .forward(&mut ctx1, &weights, &mut cache1, 0, 0, &config)
            .to_vec();

        // Reset and recompile to get fresh KV cache
        backend.reset();
        assert!(backend.compile(&weights, &config).is_ok());

        let mut ctx2 = ForwardContext::new(&config);
        let mut cache2 = MultiLayerKVCache::new(&config);
        let run2 = backend
            .forward(&mut ctx2, &weights, &mut cache2, 0, 0, &config)
            .to_vec();

        assert_eq!(run1, run2, "same input must produce same GPU logits");
    }

    #[test]
    fn test_gpu_recompile_hint() {
        let mut backend = match GpuBackend::new() {
            Ok(b) => b,
            Err(_) => return,
        };
        assert!(!backend.needs_recompile);
        backend.recompile_hint();
        assert!(backend.needs_recompile);
    }

    #[test]
    fn test_gpu_fallback_on_no_device() {
        // This test verifies the error path when no Metal device is available.
        // On most Macs, Device::system_default() succeeds, so this test is
        // effectively a no-op. It validates the error type when Metal is absent.
        if Device::system_default().is_none() {
            match GpuBackend::new() {
                Err(CompileError::DeviceError(msg)) => {
                    assert!(msg.contains("Metal"), "error should mention Metal: {msg}");
                }
                Err(other) => panic!("expected DeviceError, got {other:?}"),
                Ok(_) => panic!("expected error when no Metal device"),
            }
        }
    }
}
