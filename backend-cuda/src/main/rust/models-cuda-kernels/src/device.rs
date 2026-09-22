// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Device entry points, compiled only for `nvptx64-nvidia-cuda`.
//!
//! Three kernels, and nothing else goes on the device:
//!
//! * [`models_q4k_decode_projection`] — fused Q4_K dequantise-and-multiply projection;
//! * [`models_q6k_decode_projection`] — the same for Q6_K, which Q4_K_M uses for the value and
//!   output projections;
//! * [`models_gqa_decode_attention`] — grouped-query attention for the single-token decode step.
//!
//! Every one of them calls the shared arithmetic in [`crate::kquant`] and [`crate::attention`],
//! which the host tests exercise directly. No arithmetic is written twice.
//!
//! # Decomposition
//!
//! The projections use **one warp per output row**. Lane `l` takes super-blocks
//! `l, l + 32, l + 64, ...` and writes their *exact integer* partials to shared memory indexed
//! by super-block. After a barrier, lane 0 folds the float scales over the super-blocks in
//! ascending order. This is what keeps the result bit-exact with the CPU path (see
//! [`crate::kquant`]) while letting the warp stream a contiguous `32 * 144` byte span of the
//! weight row, which is what makes it coalesce.
//!
//! The attention kernel uses one warp per query head, with the stage-by-stage ordering
//! documented in [`crate::attention`].
//!
//! # Shared memory
//!
//! `core::arch::nvptx` exposes no shared-memory declaration, and
//! `#[unsafe(link_section = ".shared")]` **silently places the static in global memory** —
//! verified by reading the emitted PTX, which contains `st.global.b32`. That would be both a
//! race across CUDA blocks and a large slowdown, and nothing warns about it. The working
//! route is a `global_asm!` declaration plus inline PTX accessors, below. Recorded as `CU-003`
//! in `backend-cuda/UPSTREAM.md`.

#![allow(clippy::missing_safety_doc)]

use crate::attention::{self, AttentionShape};
use crate::kquant::{
    self, Q4_K_BLOCK_BYTES, Q4KRowAccumulator, Q6_K_BLOCK_BYTES, Q6_K_LANES, Q6KBlockLaneSums,
    Q6KRowAccumulator, Q8_K_SUM_BLOCK, QK_K,
};
use core::arch::nvptx::*;
use core::arch::{asm, global_asm};

/// Super-blocks a single row may hold, bounding the shared scratch.
///
/// `cols <= MAX_BLOCKS_PER_ROW * 256 = 32,768`, which covers every projection width in the
/// catalog including the widest Gemma 4 26B-A4B expert. The host refuses wider tensors rather
/// than reading past the scratch; see `CudaGgufBatchedMatrixKernel`.
pub const MAX_BLOCKS_PER_ROW: usize = 128;

/// Lanes in a warp; the projection kernels require exactly this block width.
pub const WARP_LANES: u32 = 32;

// Shared scratch, sized for the larger of the two projection kernels:
// Q6_K needs MAX_BLOCKS_PER_ROW * Q6_K_LANES i32 = 128 * 8 * 4 = 4096 bytes.
// Q4_K needs MAX_BLOCKS_PER_ROW * 2 i32 = 1024 bytes and reuses the same array.
global_asm!(".shared .align 4 .b8 models_cuda_scratch[4096];");

/// Stores one `i32` into the shared scratch at `index`.
///
/// Takes the symbol's address into a local register rather than using `[symbol + reg]`
/// addressing, which PTX does not define for a register displacement.
#[inline(always)]
unsafe fn scratch_store(index: u32, value: i32) {
    unsafe {
        asm!(
            "{{",
            ".reg .u64 %scratch;",
            "mov.u64 %scratch, models_cuda_scratch;",
            "add.u64 %scratch, %scratch, {offset};",
            "st.shared.b32 [%scratch], {value};",
            "}}",
            offset = in(reg64) (index as u64) * 4,
            value = in(reg32) value,
        );
    }
}

/// Loads one `i32` from the shared scratch at `index`.
#[inline(always)]
unsafe fn scratch_load(index: u32) -> i32 {
    let value: i32;
    unsafe {
        asm!(
            "{{",
            ".reg .u64 %scratch;",
            "mov.u64 %scratch, models_cuda_scratch;",
            "add.u64 %scratch, %scratch, {offset};",
            "ld.shared.b32 {value}, [%scratch];",
            "}}",
            offset = in(reg64) (index as u64) * 4,
            value = out(reg32) value,
        );
    }
    value
}

/// Fused Q4_K dequantise-and-multiply projection for one batch row.
///
/// Launch shape: `grid = (rows, batchSize, 1)`, `block = (32, 1, 1)`. One warp computes one
/// output row of one batch row.
///
/// # Safety
///
/// Every pointer must address at least the element count the shapes imply, and `cols` must be a
/// multiple of 256 with `cols / 256 <= MAX_BLOCKS_PER_ROW`. The host checks all of this before
/// launching; the kernel does not re-check, because a device-side refusal has nowhere to go.
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn models_q4k_decode_projection(
    weights: *const u8,
    quantized: *const i8,
    activation_scales: *const f32,
    activation_sums: *const i16,
    output: *mut f32,
    rows: u32,
    cols: u32,
) {
    let row = unsafe { _block_idx_x() } as usize;
    if row >= rows as usize {
        return;
    }
    let lane = unsafe { _thread_idx_x() };
    let cols = cols as usize;
    // The batch row is the grid's second dimension. One launch covers the whole prefill batch;
    // the previous scalar parameter forced one launch per row, which measured 54,306 launches
    // for 890 projections on an A40.
    let batch = unsafe { _block_idx_y() } as usize;
    let blocks_per_row = cols / QK_K;

    let weight_bytes = rows as usize * blocks_per_row * Q4_K_BLOCK_BYTES;
    let weights = unsafe { core::slice::from_raw_parts(weights, weight_bytes) };
    let quantized = unsafe { core::slice::from_raw_parts(quantized, (batch + 1) * cols) };
    let activation_sums = unsafe {
        core::slice::from_raw_parts(activation_sums, (batch + 1) * cols / Q8_K_SUM_BLOCK)
    };

    // Every lane computes the exact integer partials of its own super-blocks. Integer
    // addition is exact, so this assignment is free to be anything.
    let mut block = lane as usize;
    while block < blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let partials = kquant::q4k_block_partials(
            weights,
            weight_offset,
            quantized,
            activation_offset,
            activation_sums,
            sum_offset,
        );
        unsafe {
            scratch_store(block as u32 * 2, partials.quantized_sum);
            scratch_store(block as u32 * 2 + 1, partials.minimum_sum);
        }
        block += WARP_LANES as usize;
    }
    unsafe { _syncthreads() };

    // The float fold is order-sensitive, so exactly one lane replays it in block order.
    if lane != 0 {
        return;
    }
    let mut accumulator = Q4KRowAccumulator::new();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_scale = unsafe { *activation_scales.add(batch * blocks_per_row + block) };
        let (d, d_min) = kquant::q4k_block_scales(weights, weight_offset, activation_scale);
        let partials = kquant::Q4KBlockPartials {
            quantized_sum: unsafe { scratch_load(block as u32 * 2) },
            minimum_sum: unsafe { scratch_load(block as u32 * 2 + 1) },
        };
        accumulator.accumulate(d, d_min, partials);
    }
    unsafe { *output.add(batch * rows as usize + row) = accumulator.finish() };
}

/// Fused Q6_K dequantise-and-multiply projection for one batch row.
///
/// Launch shape: `grid = (rows, batchSize, 1)`, `block = (32, 1, 1)`.
///
/// Q6_K carries no `dmin` term, so it needs no activation sums; it does need the eight float
/// lane accumulators described in [`crate::kquant`].
///
/// # Safety
///
/// As [`models_q4k_decode_projection`].
#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn models_q6k_decode_projection(
    weights: *const u8,
    quantized: *const i8,
    activation_scales: *const f32,
    output: *mut f32,
    rows: u32,
    cols: u32,
) {
    let row = unsafe { _block_idx_x() } as usize;
    if row >= rows as usize {
        return;
    }
    let lane = unsafe { _thread_idx_x() };
    let cols = cols as usize;
    // As Q4_K: the batch row comes from the grid, not from a parameter.
    let batch = unsafe { _block_idx_y() } as usize;
    let blocks_per_row = cols / QK_K;

    let weight_bytes = rows as usize * blocks_per_row * Q6_K_BLOCK_BYTES;
    let weights = unsafe { core::slice::from_raw_parts(weights, weight_bytes) };
    let quantized = unsafe { core::slice::from_raw_parts(quantized, (batch + 1) * cols) };

    let mut block = lane as usize;
    while block < blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let lane_sums =
            kquant::q6k_block_lane_sums(weights, weight_offset, quantized, activation_offset);
        for slot in 0..Q6_K_LANES {
            unsafe { scratch_store((block * Q6_K_LANES + slot) as u32, lane_sums[slot]) };
        }
        block += WARP_LANES as usize;
    }
    unsafe { _syncthreads() };

    if lane != 0 {
        return;
    }
    let mut accumulator = Q6KRowAccumulator::new();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_scale = unsafe { *activation_scales.add(batch * blocks_per_row + block) };
        let d = kquant::q6k_block_scale(weights, weight_offset, activation_scale);
        let mut lane_sums: Q6KBlockLaneSums = [0; Q6_K_LANES];
        for slot in 0..Q6_K_LANES {
            lane_sums[slot] = unsafe { scratch_load((block * Q6_K_LANES + slot) as u32) };
        }
        accumulator.accumulate(d, lane_sums);
    }
    unsafe { *output.add(batch * rows as usize + row) = accumulator.finish() };
}

/// Grouped-query attention for one decode step.
///
/// Launch shape: `grid = (num_heads, 1, 1)`, `block = (32, 1, 1)`. One warp per query head.
///
/// `scores` is device scratch of at least `num_heads * positions` floats. The keys and values
/// are the device-resident KV mirror for **one** sequence; the host refuses the accelerator
/// when the attended window is not a single contiguous span, which is how physical KV prefix
/// sharing stays intact (gate G6).
///
/// # Safety
///
/// Shapes must satisfy [`AttentionShape::is_valid`] and every buffer must be large enough for
/// them. Checked on the host.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "ptx-kernel" fn models_gqa_decode_attention(
    query: *const f32,
    keys: *const f32,
    values: *const f32,
    scores: *mut f32,
    output: *mut f32,
    num_heads: u32,
    num_kv_heads: u32,
    key_length: u32,
    value_length: u32,
    key_dim: u32,
    value_dim: u32,
    positions: u32,
    scale: f32,
) {
    let head = unsafe { _block_idx_x() } as usize;
    if head >= num_heads as usize {
        return;
    }
    let lane = unsafe { _thread_idx_x() };
    let shape = AttentionShape {
        num_heads: num_heads as usize,
        num_kv_heads: num_kv_heads as usize,
        key_length: key_length as usize,
        value_length: value_length as usize,
        key_dim: key_dim as usize,
        value_dim: value_dim as usize,
        positions: positions as usize,
        scale,
    };
    let kv = shape.kv_head(head);
    let query = unsafe {
        core::slice::from_raw_parts(query.add(head * shape.key_length), shape.key_length)
    };
    let row_scores = unsafe { scores.add(head * shape.positions) };

    // Stage 1: one lane per position, each dot in index order. Bit-exact per score.
    let mut position = lane as usize;
    while position < shape.positions {
        let base = position * shape.key_dim + kv * shape.key_length;
        let key = unsafe { core::slice::from_raw_parts(keys.add(base), shape.key_length) };
        unsafe { *row_scores.add(position) = attention::attention_dot(query, key) * shape.scale };
        position += WARP_LANES as usize;
    }
    unsafe { _syncthreads() };

    // Stage 2: the maximum is exact and order-free, so every lane may recompute it rather than
    // pay a barrier to share it.
    let mut maximum = f32::NEG_INFINITY;
    for index in 0..shape.positions {
        let score = unsafe { *row_scores.add(index) };
        if score > maximum {
            maximum = score;
        }
    }

    // Stage 3: exp is elementwise and independent.
    let mut position = lane as usize;
    while position < shape.positions {
        let value = unsafe { *row_scores.add(position) };
        unsafe { *row_scores.add(position) = attention::expf(value - maximum) };
        position += WARP_LANES as usize;
    }
    unsafe { _syncthreads() };

    // Stage 4: the sum is float addition, so one lane adds it in ascending position order to
    // match the CPU reference, then every lane recomputes the reciprocal identically.
    let mut sum = 0.0_f32;
    for index in 0..shape.positions {
        sum += unsafe { *row_scores.add(index) };
    }
    let inverse = 1.0_f32 / sum;

    // Stage 5: one lane per output dimension, accumulating over positions in order. Each
    // output element sees exactly the CPU's sequence of fused multiply-adds.
    let out_base = head * shape.value_length;
    let mut index = lane as usize;
    while index < shape.value_length {
        let mut accumulated = 0.0_f32;
        for position in 0..shape.positions {
            let weight = unsafe { *row_scores.add(position) } * inverse;
            let value = unsafe {
                *values.add(position * shape.value_dim + kv * shape.value_length + index)
            };
            accumulated = crate::float::fma(value, weight, accumulated);
        }
        unsafe { *output.add(out_base + index) = accumulated };
        index += WARP_LANES as usize;
    }
}
