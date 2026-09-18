// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Shared K-quant arithmetic.
//!
//! Every function here compiles unchanged for the host (`cargo test`) and for
//! `nvptx64-nvidia-cuda`. That is the whole point of the module: the device kernels in
//! [`crate::device`] call exactly these functions, so a host test of this module is a test of
//! the arithmetic that runs on the GPU, not of a separate re-implementation of it.
//!
//! # Numeric contract
//!
//! The arithmetic is transcribed from the CPU reference in
//! `backend-native/src/main/rust/model-kernels/src/lib.rs` (`dot_q4_k_q8_k_row_scalar`,
//! `dot_q6_k_q8_k_row_scalar`, `qk_scale`, `qk_min`, `f16_to_f32`), which is itself the twin of
//! the Vector API path in `vectors-core`. The decomposition below is chosen so that the device
//! result is **bit-exact** with that reference, not merely close:
//!
//! * Every per-super-block reduction is `i32`. Integer addition is associative and exact, so a
//!   device is free to compute the per-block partials in any order or assignment.
//! * The only floating-point operations are the per-block scale applications. Those are *not*
//!   order-free, so [`Q4KRowAccumulator`] and [`Q6KRowAccumulator`] fold them in ascending
//!   super-block order, with the same `fma` the CPU reference uses. PTX lowers
//!   [`crate::float::fma`] to `fma.rn.f32`, the same IEEE-754 fused operation with the same
//!   rounding, so the fold is bit-identical.
//! * Q6_K keeps **eight** float lane accumulators. That is a determinism contract, not an
//!   optimisation: the CPU scalar path maintains them purely so its float reduction order
//!   matches the 8-lane SIMD order of the Vector API path. Dropping them would change the low
//!   bits. See `dot_q6_k_q8_k_row_scalar` in the CPU crate.
//!
//! The consequence for a device kernel is the useful one: the expensive part (the integer
//! products) may be spread across a warp however the hardware likes, and only the cheap tail
//! (two `fma`s per block for Q4_K, eight for Q6_K) must be replayed in order by a single thread.

#![allow(clippy::needless_range_loop)]

/// Weights per K-quant super-block.
pub const QK_K: usize = 256;
/// Activation values covered by one Q8_K partial sum.
pub const Q8_K_SUM_BLOCK: usize = 16;
/// Bytes in one Q4_K super-block: `d`, `dmin`, 12 packed scale bytes, 128 nibble bytes.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Bytes in one Q6_K super-block: 128 `ql`, 64 `qh`, 16 signed scales, `d`.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Float lane accumulators Q6_K must keep to stay bit-exact with the CPU path.
pub const Q6_K_LANES: usize = 8;

/// The exact integer lane sums of one Q6_K super-block.
pub type Q6KBlockLaneSums = [i32; Q6_K_LANES];

/// Converts a little-endian IEEE binary16 to `f32`.
///
/// Transcribed from `f16_to_f32` in the CPU kernel crate. Written out rather than taken from a
/// crate because this must compile for `nvptx64` under `no_std`, and because the CPU reference
/// is the definition of correct here.
#[inline(always)]
pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let significand = (value & 0x03ff) as u32;
    if exponent == 0 {
        if significand == 0 {
            return f32::from_bits(sign);
        }
        let magnitude = significand as f32 * f32::from_bits(0x3380_0000);
        return if sign == 0 { magnitude } else { -magnitude };
    }
    let bits = if exponent == 31 {
        sign | 0x7f80_0000 | (significand << 13)
    } else {
        sign | (((exponent - 15 + 127) as u32) << 23) | (significand << 13)
    };
    f32::from_bits(bits)
}

/// Reads a little-endian `u16` from `bytes` at `offset`.
#[inline(always)]
pub fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    (bytes[offset] as u16) | ((bytes[offset + 1] as u16) << 8)
}

/// Unpacks the unsigned 6-bit scale of Q4_K/Q5_K sub-block `group` from the 12 packed bytes.
///
/// Transcribed from `qk_scale` in the CPU kernel crate.
#[inline(always)]
pub fn qk_scale(scales: &[u8], group: usize) -> i32 {
    if group < 4 {
        return (scales[group] & 0x3f) as i32;
    }
    let low = scales[group + 4] & 0x0f;
    let high = scales[group - 4] >> 6;
    (low | (high << 4)) as i32
}

/// Unpacks the unsigned 6-bit minimum of Q4_K/Q5_K sub-block `group`.
///
/// Transcribed from `qk_min` in the CPU kernel crate.
#[inline(always)]
pub fn qk_min(scales: &[u8], group: usize) -> i32 {
    if group < 4 {
        return (scales[group + 4] & 0x3f) as i32;
    }
    let low = scales[group + 4] >> 4;
    let high = scales[group] >> 6;
    (low | (high << 4)) as i32
}

/// The exact integer content of one Q4_K super-block against one Q8_K activation block.
///
/// Both members are exact: no rounding has happened yet. A device may compute these for the
/// super-blocks of a row in any order and on any thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Q4KBlockPartials {
    /// `sum(scale[g] * dot(quant[g], activation[g]))` over the eight sub-blocks.
    pub quantized_sum: i32,
    /// `sum(min[g] * (activation_sums[2g] + activation_sums[2g+1]))` over the eight sub-blocks.
    pub minimum_sum: i32,
}

/// Computes the exact integer partials of one Q4_K super-block.
///
/// `weights` is the whole quantised tensor; `weight_offset` is the first byte of this
/// super-block. `quantized`, `activation_sums` are the Q8_K activation planes, indexed from
/// `activation_offset` and `sum_offset` respectively.
///
/// Transcribed from the inner block body of `dot_q4_k_q8_k_row_scalar`.
#[inline]
pub fn q4k_block_partials(
    weights: &[u8],
    weight_offset: usize,
    quantized: &[i8],
    activation_offset: usize,
    activation_sums: &[i16],
    sum_offset: usize,
) -> Q4KBlockPartials {
    let scales = &weights[weight_offset + 4..weight_offset + 16];
    let quants = &weights[weight_offset + 16..weight_offset + Q4_K_BLOCK_BYTES];
    let mut quantized_sum = 0_i32;
    let mut minimum_sum = 0_i32;
    for group in 0..8 {
        let group_scale = qk_scale(scales, group);
        let group_min = qk_min(scales, group);
        let packed_offset = (group >> 1) * 32;
        let shift = (group & 1) * 4;
        let group_activation_offset = activation_offset + group * 32;
        let mut group_dot = 0_i32;
        for index in 0..32 {
            let quant = (quants[packed_offset + index] >> shift) & 0x0f;
            group_dot += quant as i32 * quantized[group_activation_offset + index] as i32;
        }
        quantized_sum += group_scale * group_dot;
        minimum_sum += group_min
            * (activation_sums[sum_offset + group * 2] as i32
                + activation_sums[sum_offset + group * 2 + 1] as i32);
    }
    Q4KBlockPartials {
        quantized_sum,
        minimum_sum,
    }
}

/// Reads the `d` and `dmin` super-block scales of a Q4_K block, already multiplied by the
/// activation scale exactly as the CPU reference does.
#[inline(always)]
pub fn q4k_block_scales(weights: &[u8], weight_offset: usize, activation_scale: f32) -> (f32, f32) {
    let d = f16_to_f32(read_u16_le(weights, weight_offset)) * activation_scale;
    let d_min = f16_to_f32(read_u16_le(weights, weight_offset + 2)) * activation_scale;
    (d, d_min)
}

/// Folds Q4_K super-block partials into a row dot product in ascending block order.
///
/// This is the order-sensitive tail. One thread must own it for the whole row.
#[derive(Clone, Copy, Debug, Default)]
pub struct Q4KRowAccumulator {
    sum: f32,
}

impl Q4KRowAccumulator {
    /// A zeroed accumulator.
    #[inline(always)]
    pub fn new() -> Self {
        Self { sum: 0.0 }
    }

    /// Applies one super-block's scales to its exact integer partials.
    ///
    /// Must be called for ascending `block` with no gaps: the two `fma`s are not commutative
    /// with those of the neighbouring blocks.
    #[inline(always)]
    pub fn accumulate(&mut self, d: f32, d_min: f32, partials: Q4KBlockPartials) {
        self.sum = crate::float::fma(d, partials.quantized_sum as f32, self.sum);
        self.sum = crate::float::fma(-d_min, partials.minimum_sum as f32, self.sum);
    }

    /// The finished row dot product.
    #[inline(always)]
    pub fn finish(self) -> f32 {
        self.sum
    }
}

/// Computes one Q4_K row dot product end to end, in the reference order.
///
/// The host reference used by the parity tests, and the fallback the device kernel degenerates
/// to when only one thread is available for a row.
#[inline]
pub fn q4k_row_dot(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut accumulator = Q4KRowAccumulator::new();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let activation_scale = activation_scales[batch * blocks_per_row + block];
        let (d, d_min) = q4k_block_scales(weights, weight_offset, activation_scale);
        let partials = q4k_block_partials(
            weights,
            weight_offset,
            quantized,
            activation_offset,
            activation_sums,
            sum_offset,
        );
        accumulator.accumulate(d, d_min, partials);
    }
    accumulator.finish()
}

/// Computes the exact integer lane sums of one Q6_K super-block.
///
/// The eight lanes are the determinism contract described in the module documentation: lane
/// assignment is `index & 7`, and it must be preserved.
///
/// Transcribed from the inner block body of `dot_q6_k_q8_k_row_scalar`.
#[inline]
pub fn q6k_block_lane_sums(
    weights: &[u8],
    weight_offset: usize,
    quantized: &[i8],
    activation_offset: usize,
) -> Q6KBlockLaneSums {
    let ql = &weights[weight_offset..weight_offset + 128];
    let qh = &weights[weight_offset + 128..weight_offset + 192];
    let scales = &weights[weight_offset + 192..weight_offset + 208];
    let mut integer_sums = [0_i32; Q6_K_LANES];
    for super_block in 0..2 {
        let ql_base = super_block * 64;
        let qh_base = super_block * 32;
        let scale_base = super_block * 8;
        let quant_base = activation_offset + super_block * 128;
        for index in 0..32 {
            let scale_index = index / 16;
            let ql1 = ql[ql_base + index];
            let ql2 = ql[ql_base + 32 + index];
            let high = qh[qh_base + index];
            let q1 = ((ql1 & 0x0f) | ((high & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql2 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((ql1 >> 4) | (((high >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((ql2 >> 4) | (((high >> 6) & 0x03) << 4)) as i32 - 32;
            let s1 = scales[scale_base + scale_index] as i8 as i32;
            let s2 = scales[scale_base + scale_index + 2] as i8 as i32;
            let s3 = scales[scale_base + scale_index + 4] as i8 as i32;
            let s4 = scales[scale_base + scale_index + 6] as i8 as i32;
            let lane = index & 7;
            integer_sums[lane] += s1 * q1 * quantized[quant_base + index] as i32;
            integer_sums[lane] += s2 * q2 * quantized[quant_base + index + 32] as i32;
            integer_sums[lane] += s3 * q3 * quantized[quant_base + index + 64] as i32;
            integer_sums[lane] += s4 * q4 * quantized[quant_base + index + 96] as i32;
        }
    }
    integer_sums
}

/// Reads the `d` super-block scale of a Q6_K block, multiplied by the activation scale.
#[inline(always)]
pub fn q6k_block_scale(weights: &[u8], weight_offset: usize, activation_scale: f32) -> f32 {
    f16_to_f32(read_u16_le(weights, weight_offset + 208)) * activation_scale
}

/// Folds Q6_K super-block lane sums into a row dot product, preserving the eight-lane contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct Q6KRowAccumulator {
    lane_sums: [f32; Q6_K_LANES],
}

impl Q6KRowAccumulator {
    /// A zeroed accumulator.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            lane_sums: [0.0; Q6_K_LANES],
        }
    }

    /// Applies one super-block's scale to its exact integer lane sums, in ascending block order.
    #[inline(always)]
    pub fn accumulate(&mut self, d: f32, integer_sums: Q6KBlockLaneSums) {
        for lane in 0..Q6_K_LANES {
            self.lane_sums[lane] =
                crate::float::fma(d, integer_sums[lane] as f32, self.lane_sums[lane]);
        }
    }

    /// Reduces the eight lanes in lane order with plain adds, exactly as the CPU path does.
    #[inline(always)]
    pub fn finish(self) -> f32 {
        let mut sum = 0.0_f32;
        for lane in 0..Q6_K_LANES {
            sum += self.lane_sums[lane];
        }
        sum
    }
}

/// Computes one Q6_K row dot product end to end, in the reference order.
#[inline]
pub fn q6k_row_dot(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut accumulator = Q6KRowAccumulator::new();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let activation_scale = activation_scales[batch * blocks_per_row + block];
        let d = q6k_block_scale(weights, weight_offset, activation_scale);
        let lane_sums = q6k_block_lane_sums(weights, weight_offset, quantized, activation_offset);
        accumulator.accumulate(d, lane_sums);
    }
    accumulator.finish()
}

/// GGML's round-to-nearest bit trick, reproduced exactly.
///
/// `ggml_nearest_int` adds `1.5 * 2^23` so the mantissa holds the rounded integer, then
/// recovers it. Any other rounding changes the quantised activation and therefore the result.
#[inline(always)]
pub fn ggml_nearest_int(value: f32) -> i32 {
    let bits = (value + 12_582_912.0_f32).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Quantises one batch-major float activation plane to Q8_K.
///
/// Writes `quantized` (`batch_size * cols` i8), `scales` (one f32 per 256-value block) and
/// `sums` (one i16 per 16-value sub-block). Transcribed from `quantize_q8_k_batch`.
///
/// This stays on the host: it is `O(cols)` per token against the projections' `O(rows * cols)`,
/// and keeping it in one place means the device never disagrees with the CPU about what the
/// activations were.
pub fn quantize_q8_k(
    input: &[f32],
    batch_size: usize,
    cols: usize,
    quantized: &mut [i8],
    scales: &mut [f32],
    sums: &mut [i16],
) {
    let blocks_per_row = cols / QK_K;
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let offset = batch * cols + block * QK_K;
            let mut absolute_max = 0.0_f32;
            let mut extremum = 0.0_f32;
            for index in 0..QK_K {
                let value = input[offset + index];
                let magnitude = value.abs();
                if magnitude > absolute_max {
                    absolute_max = magnitude;
                    extremum = value;
                }
            }
            let scale_index = batch * blocks_per_row + block;
            let sum_base = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
            if absolute_max == 0.0 {
                scales[scale_index] = 0.0;
                for index in 0..QK_K {
                    quantized[offset + index] = 0;
                }
                for sub in 0..QK_K / Q8_K_SUM_BLOCK {
                    sums[sum_base + sub] = 0;
                }
                continue;
            }
            let inverse_scale = -127.0_f32 / extremum;
            for index in 0..QK_K {
                let quant = ggml_nearest_int(inverse_scale * input[offset + index]);
                quantized[offset + index] = if quant > 127 { 127 } else { quant as i8 };
            }
            for sub in 0..QK_K / Q8_K_SUM_BLOCK {
                let mut total = 0_i32;
                for index in 0..Q8_K_SUM_BLOCK {
                    total += quantized[offset + sub * Q8_K_SUM_BLOCK + index] as i32;
                }
                sums[sum_base + sub] = total as i16;
            }
            scales[scale_index] = 1.0_f32 / inverse_scale;
        }
    }
}
