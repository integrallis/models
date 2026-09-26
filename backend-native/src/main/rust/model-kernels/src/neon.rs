// SPDX-License-Identifier: Apache-2.0

//! ARM NEON kernels for AArch64 CPUs such as Apple Silicon.
//!
//! Only the exact integer work is vectorized: the Q4_0, Q5_0 and Q8_0 block sums, the Q4_K group
//! dot products and the Q6_K per-lane sums. Every floating-point step keeps the scalar kernels'
//! operations and order, so each function here returns bit-identical results to the scalar
//! function it mirrors.
//!
//! `sdot` is emitted through inline assembly because the `vdotq_s32` intrinsic is not yet stable
//! (rust-lang/rust#117224). Setting `JMODELS_KERNELS_DISABLE_NEON` selects the scalar kernels,
//! which makes A/B comparisons possible with one library.

use super::{
    Q4_0_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_0_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES,
    Q8_K_SUM_BLOCK, QK_0, QK_K, STACK_BATCH_CAPACITY, f16_to_f32, qk_min, qk_scale,
    record_k_quant_weight_block_decode,
};
use std::arch::aarch64::*;
use std::sync::OnceLock;

/// Whether the NEON kernels may run: the CPU has the dot-product extension and
/// `JMODELS_KERNELS_DISABLE_NEON` is unset.
pub(super) fn available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::env::var_os("JMODELS_KERNELS_DISABLE_NEON").is_none()
            && std::arch::is_aarch64_feature_detected!("neon")
            && std::arch::is_aarch64_feature_detected!("dotprod")
    })
}

/// Adds the dot product of each group of four signed bytes of `a` and `b` to the matching lane of
/// `acc`. Every product and sum is exact.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let mut acc = acc;
    // SAFETY: `sdot` reads and writes only the named vector registers, and this function is
    // compiled with the dot-product extension enabled.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    acc
}

const BIT_SELECT: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];

/// Spreads 16 bits of a Q5_0 high-bit plane over 16 bytes: `0x10` where bit `i` is set in byte
/// `i`, zero elsewhere.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn q5_0_fifth_bits(bits: u16) -> uint8x16_t {
    let bytes = vcombine_u8(vdup_n_u8(bits as u8), vdup_n_u8((bits >> 8) as u8));
    // SAFETY: BIT_SELECT holds 16 bytes.
    let select = unsafe { vld1q_u8(BIT_SELECT.as_ptr()) };
    vandq_u8(vtstq_u8(bytes, select), vdupq_n_u8(0x10))
}

/// Exact integer dot product of one Q4_0 weight block with one Q8_0 activation block.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4_0_q8_0_block_sum(packed: *const u8, quantized: *const i8) -> i32 {
    // SAFETY: callers pass the 16 packed bytes of one Q4_0 block and one 32-byte Q8_0 block.
    let (packed, low_inputs, high_inputs) = unsafe {
        (
            vld1q_u8(packed),
            vld1q_s8(quantized),
            vld1q_s8(quantized.add(16)),
        )
    };
    let bias = vdupq_n_s8(8);
    let low = vsubq_s8(
        vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f))),
        bias,
    );
    let high = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(packed)), bias);
    // SAFETY: this function runs with the dot-product extension enabled.
    vaddvq_s32(unsafe { sdot(sdot(vdupq_n_s32(0), low, low_inputs), high, high_inputs) })
}

/// Exact integer dot product of one Q5_0 weight block with one Q8_0 activation block.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q5_0_q8_0_block_sum(high_bits: u32, packed: *const u8, quantized: *const i8) -> i32 {
    // SAFETY: callers pass the 16 packed bytes of one Q5_0 block and one 32-byte Q8_0 block.
    let (packed, low_inputs, high_inputs) = unsafe {
        (
            vld1q_u8(packed),
            vld1q_s8(quantized),
            vld1q_s8(quantized.add(16)),
        )
    };
    // SAFETY: NEON is enabled for this function.
    let (low_fifth, high_fifth) = unsafe {
        (
            q5_0_fifth_bits(high_bits as u16),
            q5_0_fifth_bits((high_bits >> 16) as u16),
        )
    };
    let bias = vdupq_n_s8(16);
    let low = vorrq_u8(vandq_u8(packed, vdupq_n_u8(0x0f)), low_fifth);
    let high = vorrq_u8(vshrq_n_u8::<4>(packed), high_fifth);
    let low = vsubq_s8(vreinterpretq_s8_u8(low), bias);
    let high = vsubq_s8(vreinterpretq_s8_u8(high), bias);
    // SAFETY: this function runs with the dot-product extension enabled.
    vaddvq_s32(unsafe { sdot(sdot(vdupq_n_s32(0), low, low_inputs), high, high_inputs) })
}

/// Exact integer dot product of one Q8_0 weight block with one Q8_0 activation block.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q8_0_q8_0_block_sum(weights: *const u8, quantized: *const i8) -> i32 {
    // SAFETY: callers pass the 32 weight bytes of one Q8_0 block and one 32-byte Q8_0 block.
    let (low_weights, high_weights, low_inputs, high_inputs) = unsafe {
        (
            vld1q_s8(weights.cast()),
            vld1q_s8(weights.add(16).cast()),
            vld1q_s8(quantized),
            vld1q_s8(quantized.add(16)),
        )
    };
    // SAFETY: this function runs with the dot-product extension enabled.
    vaddvq_s32(unsafe {
        sdot(
            sdot(vdupq_n_s32(0), low_weights, low_inputs),
            high_weights,
            high_inputs,
        )
    })
}

/// Exact integer dot products of the eight 32-element groups of one Q4_K block with one Q8_K
/// activation block. Group `2k` holds the low nibbles of bytes `32k..32k+32`, group `2k+1` the
/// high nibbles.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4_k_group_dots(quants: *const u8, activations: *const i8) -> [i32; 8] {
    let mut dots = [0_i32; 8];
    let mask = vdupq_n_u8(0x0f);
    for pair in 0..4 {
        // SAFETY: callers pass the 128 packed bytes of one Q4_K block and its 256 activations.
        let (first, second, low_inputs, high_inputs) = unsafe {
            let inputs = activations.add(pair * 64);
            (
                vld1q_u8(quants.add(pair * 32)),
                vld1q_u8(quants.add(pair * 32 + 16)),
                [vld1q_s8(inputs), vld1q_s8(inputs.add(16))],
                [vld1q_s8(inputs.add(32)), vld1q_s8(inputs.add(48))],
            )
        };
        let low = [
            vreinterpretq_s8_u8(vandq_u8(first, mask)),
            vreinterpretq_s8_u8(vandq_u8(second, mask)),
        ];
        let high = [
            vreinterpretq_s8_u8(vshrq_n_u8::<4>(first)),
            vreinterpretq_s8_u8(vshrq_n_u8::<4>(second)),
        ];
        // SAFETY: this function runs with the dot-product extension enabled.
        unsafe {
            let zero = vdupq_n_s32(0);
            dots[pair * 2] = vaddvq_s32(sdot(
                sdot(zero, low[0], low_inputs[0]),
                low[1],
                low_inputs[1],
            ));
            dots[pair * 2 + 1] = vaddvq_s32(sdot(
                sdot(zero, high[0], high_inputs[0]),
                high[1],
                high_inputs[1],
            ));
        }
    }
    dots
}

/// Exact per-lane integer sums of one Q6_K block with one Q8_K activation block, partitioned like
/// the scalar kernel: element `index` of each 32-element run goes to lane `index & 7`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q6_k_lane_sums(block: *const u8, activations: *const i8) -> [i32; 8] {
    let low_mask = vdupq_n_u8(0x0f);
    let high_mask = vdupq_n_u8(0x30);
    let bias = vdupq_n_s8(32);
    let mut first_lanes = vdupq_n_s32(0);
    let mut last_lanes = vdupq_n_s32(0);
    for super_block in 0..2 {
        for half in 0..2 {
            let offset = half * 16;
            // SAFETY: callers pass one complete 210-byte Q6_K block.
            let (ql1, ql2, high, scales) = unsafe {
                (
                    vld1q_u8(block.add(super_block * 64 + offset)),
                    vld1q_u8(block.add(super_block * 64 + 32 + offset)),
                    vld1q_u8(block.add(128 + super_block * 32 + offset)),
                    block.add(192 + super_block * 8 + half),
                )
            };
            let quadrants = [
                vorrq_u8(
                    vandq_u8(ql1, low_mask),
                    vandq_u8(vshlq_n_u8::<4>(high), high_mask),
                ),
                vorrq_u8(
                    vandq_u8(ql2, low_mask),
                    vandq_u8(vshlq_n_u8::<2>(high), high_mask),
                ),
                vorrq_u8(vshrq_n_u8::<4>(ql1), vandq_u8(high, high_mask)),
                vorrq_u8(
                    vshrq_n_u8::<4>(ql2),
                    vandq_u8(vshrq_n_u8::<2>(high), high_mask),
                ),
            ];
            for (quadrant, weights) in quadrants.into_iter().enumerate() {
                let weights = vsubq_s8(vreinterpretq_s8_u8(weights), bias);
                // SAFETY: callers pass the 256 activations of this block, and the scale bytes
                // sit inside the block.
                let (inputs, scale) = unsafe {
                    (
                        vld1q_s8(activations.add(super_block * 128 + quadrant * 32 + offset)),
                        *scales.add(quadrant * 2) as i8 as i16,
                    )
                };
                // Elements offset..offset+8 and offset+8..offset+16 share lanes 0..8. Each
                // product is at most 32 * 128, so their sum fits in 16 bits.
                let products = vaddq_s16(
                    vmull_s8(vget_low_s8(weights), vget_low_s8(inputs)),
                    vmull_high_s8(weights, inputs),
                );
                first_lanes = vmlal_n_s16(first_lanes, vget_low_s16(products), scale);
                last_lanes = vmlal_high_n_s16(last_lanes, products, scale);
            }
        }
    }
    let mut sums = [0_i32; 8];
    // SAFETY: `sums` holds eight lanes.
    unsafe {
        vst1q_s32(sums.as_mut_ptr(), first_lanes);
        vst1q_s32(sums.as_mut_ptr().add(4), last_lanes);
    }
    sums
}

macro_rules! q_0_row_and_range {
    ($row:ident, $range:ident, $block_bytes:expr, $block_sum:expr) => {
        /// Mirrors the scalar row kernel of the same format bit for bit.
        #[target_feature(enable = "neon,dotprod")]
        pub(super) unsafe fn $row(
            weights: &[u8],
            quantized: &[i8],
            activation_scales: &[f32],
            batch: usize,
            row: usize,
            cols: usize,
        ) -> f32 {
            let blocks_per_row = cols / QK_0;
            let mut sum = 0_f32;
            for block in 0..blocks_per_row {
                let weight_offset = (row * blocks_per_row + block) * $block_bytes;
                let input_offset = batch * cols + block * QK_0;
                let weight_block = &weights[weight_offset..weight_offset + $block_bytes];
                let input_block = &quantized[input_offset..input_offset + QK_0];
                // SAFETY: both slices hold one complete block.
                let integer_sum = unsafe { $block_sum(weight_block, input_block.as_ptr()) };
                let weight_scale =
                    f16_to_f32(u16::from_le_bytes([weight_block[0], weight_block[1]]));
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sum = scale.mul_add(integer_sum as f32, sum);
            }
            sum
        }

        /// Mirrors the scalar batched row-range kernel of the same format bit for bit.
        #[allow(clippy::too_many_arguments)]
        #[target_feature(enable = "neon,dotprod")]
        pub(super) unsafe fn $range(
            weights: &[u8],
            quantized: &[i8],
            activation_scales: &[f32],
            output: *mut f32,
            batch_size: usize,
            rows: usize,
            cols: usize,
            start_row: usize,
            end_row: usize,
        ) {
            let blocks_per_row = cols / QK_0;
            batch_scratch!(sums, batch_size, 0_f32, f32);
            for row in start_row..end_row {
                sums.fill(0.0);
                for block in 0..blocks_per_row {
                    let weight_offset = (row * blocks_per_row + block) * $block_bytes;
                    let weight_block = &weights[weight_offset..weight_offset + $block_bytes];
                    let weight_scale =
                        f16_to_f32(u16::from_le_bytes([weight_block[0], weight_block[1]]));
                    for batch in 0..batch_size {
                        let input_offset = batch * cols + block * QK_0;
                        let input_block = &quantized[input_offset..input_offset + QK_0];
                        // SAFETY: both slices hold one complete block.
                        let integer_sum = unsafe { $block_sum(weight_block, input_block.as_ptr()) };
                        let scale =
                            weight_scale * activation_scales[batch * blocks_per_row + block];
                        sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
                    }
                }
                for (batch, &sum) in sums.iter().enumerate() {
                    // SAFETY: each worker owns this row across all batch-major output planes.
                    unsafe {
                        output.add(batch * rows + row).write(sum);
                    }
                }
            }
        }
    };
}

/// Block sum over a complete Q4_0 block slice (scale bytes included).
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q4_0_block(block: &[u8], quantized: *const i8) -> i32 {
    // SAFETY: `block` is one complete Q4_0 block and callers pass 32 activations.
    unsafe { q4_0_q8_0_block_sum(block[2..].as_ptr(), quantized) }
}

/// Block sum over a complete Q5_0 block slice (scale and high-bit plane included).
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q5_0_block(block: &[u8], quantized: *const i8) -> i32 {
    let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    // SAFETY: `block` is one complete Q5_0 block and callers pass 32 activations.
    unsafe { q5_0_q8_0_block_sum(high_bits, block[6..].as_ptr(), quantized) }
}

/// Block sum over a complete Q8_0 block slice (scale bytes included).
#[inline]
#[target_feature(enable = "neon,dotprod")]
unsafe fn q8_0_block(block: &[u8], quantized: *const i8) -> i32 {
    // SAFETY: `block` is one complete Q8_0 block and callers pass 32 activations.
    unsafe { q8_0_q8_0_block_sum(block[2..].as_ptr(), quantized) }
}

q_0_row_and_range!(
    dot_q4_0_q8_0_row,
    compute_q4_batched_row_range,
    Q4_0_BLOCK_BYTES,
    q4_0_block
);
q_0_row_and_range!(
    dot_q5_0_q8_0_row,
    compute_q5_batched_row_range,
    Q5_0_BLOCK_BYTES,
    q5_0_block
);
q_0_row_and_range!(
    dot_q8_0_q8_0_row,
    compute_q8_batched_row_range,
    Q8_0_BLOCK_BYTES,
    q8_0_block
);

/// Mirrors `dot_q4_k_q8_k_row_scalar` bit for bit.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn dot_q4_k_q8_k_row(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let scale_offset = batch * blocks_per_row + block;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let weight_block = &weights[weight_offset..weight_offset + Q4_K_BLOCK_BYTES];
        let activations = &quantized[activation_offset..activation_offset + QK_K];
        let d = f16_to_f32(u16::from_le_bytes([weight_block[0], weight_block[1]]))
            * activation_scales[scale_offset];
        let d_min = f16_to_f32(u16::from_le_bytes([weight_block[2], weight_block[3]]))
            * activation_scales[scale_offset];
        let scales = &weight_block[4..16];
        // SAFETY: the slices hold one complete Q4_K block and its 256 activations.
        let dots = unsafe { q4_k_group_dots(weight_block[16..].as_ptr(), activations.as_ptr()) };
        let mut quantized_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for (group, &dot) in dots.iter().enumerate() {
            quantized_sum += qk_scale(scales, group) * dot;
            minimum_sum += qk_min(scales, group)
                * (activation_sums[sum_offset + group * 2] as i32
                    + activation_sums[sum_offset + group * 2 + 1] as i32);
        }
        sum = d.mul_add(quantized_sum as f32, sum);
        sum = (-d_min).mul_add(minimum_sum as f32, sum);
    }
    sum
}

/// Mirrors `compute_q4_k_batched_row_range_scalar` bit for bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn compute_q4_k_batched_row_range(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            // SAFETY: NEON and the dot-product extension are enabled for this function.
            let sum = unsafe {
                dot_q4_k_q8_k_row(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    0,
                    row,
                    cols,
                )
            };
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let sums_per_batch = cols / Q8_K_SUM_BLOCK;
    batch_scratch!(sums, batch_size, 0_f32, f32);
    let mut group_scales = [0_i32; 8];
    let mut group_mins = [0_i32; 8];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
            let weight_block = &weights[weight_offset..weight_offset + Q4_K_BLOCK_BYTES];
            let weight_scale = f16_to_f32(u16::from_le_bytes([weight_block[0], weight_block[1]]));
            let weight_min_scale =
                f16_to_f32(u16::from_le_bytes([weight_block[2], weight_block[3]]));
            let scales = &weight_block[4..16];
            for group in 0..8 {
                group_scales[group] = qk_scale(scales, group);
                group_mins[group] = qk_min(scales, group);
            }
            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let sum_offset = batch * sums_per_batch + block * QK_K / Q8_K_SUM_BLOCK;
                let activations = &quantized[activation_offset..activation_offset + QK_K];
                // SAFETY: the slices hold one complete Q4_K block and its 256 activations.
                let dots =
                    unsafe { q4_k_group_dots(weight_block[16..].as_ptr(), activations.as_ptr()) };
                let mut quantized_sum = 0_i32;
                let mut minimum_sum = 0_i32;
                for group in 0..8 {
                    quantized_sum += group_scales[group] * dots[group];
                    minimum_sum += group_mins[group]
                        * (activation_sums[sum_offset + group * 2] as i32
                            + activation_sums[sum_offset + group * 2 + 1] as i32);
                }
                let activation_scale = activation_scales[batch * blocks_per_row + block];
                sums[batch] =
                    (weight_scale * activation_scale).mul_add(quantized_sum as f32, sums[batch]);
                sums[batch] =
                    (-weight_min_scale * activation_scale).mul_add(minimum_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

/// Mirrors `dot_q6_k_q8_k_row_scalar` bit for bit.
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn dot_q6_k_q8_k_row(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut lane_sums = [0_f32; 8];
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let weight_block = &weights[weight_offset..weight_offset + Q6_K_BLOCK_BYTES];
        let activations = &quantized[activation_offset..activation_offset + QK_K];
        let d = f16_to_f32(u16::from_le_bytes([weight_block[208], weight_block[209]]))
            * activation_scales[batch * blocks_per_row + block];
        // SAFETY: the slices hold one complete Q6_K block and its 256 activations.
        let integer_sums = unsafe { q6_k_lane_sums(weight_block.as_ptr(), activations.as_ptr()) };
        for lane in 0..lane_sums.len() {
            lane_sums[lane] = d.mul_add(integer_sums[lane] as f32, lane_sums[lane]);
        }
    }
    let mut sum = 0_f32;
    for lane_sum in lane_sums {
        sum += lane_sum;
    }
    sum
}

/// Mirrors `compute_q6_k_batched_row_range_scalar` bit for bit.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon,dotprod")]
pub(super) unsafe fn compute_q6_k_batched_row_range(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            // SAFETY: NEON and the dot-product extension are enabled for this function.
            let sum =
                unsafe { dot_q6_k_q8_k_row(weights, quantized, activation_scales, 0, row, cols) };
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    batch_scratch!(lane_sums, batch_size, [0_f32; 8], [f32; 8]);
    for row in start_row..end_row {
        lane_sums.fill([0.0; 8]);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
            let weight_block = &weights[weight_offset..weight_offset + Q6_K_BLOCK_BYTES];
            let weight_scale =
                f16_to_f32(u16::from_le_bytes([weight_block[208], weight_block[209]]));
            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let activations = &quantized[activation_offset..activation_offset + QK_K];
                // SAFETY: the slices hold one complete Q6_K block and its 256 activations.
                let integer_sums =
                    unsafe { q6_k_lane_sums(weight_block.as_ptr(), activations.as_ptr()) };
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                for (lane_sum, &integer_sum) in lane_sums[batch].iter_mut().zip(integer_sums.iter())
                {
                    *lane_sum = scale.mul_add(integer_sum as f32, *lane_sum);
                }
            }
        }
        for (batch, batch_lane_sums) in lane_sums.iter().enumerate() {
            let sum = batch_lane_sums.iter().copied().sum();
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compute_q4_batched_row_range_scalar, compute_q4_k_batched_row_range_scalar,
        compute_q5_batched_row_range_scalar, compute_q6_k_batched_row_range_scalar,
        compute_q8_batched_row_range_scalar, dot_q4_0_q8_0_row_scalar, dot_q4_k_q8_k_row_scalar,
        dot_q5_0_q8_0_row_scalar, dot_q6_k_q8_k_row_scalar, dot_q8_0_q8_0_row_scalar, f32_to_f16,
    };

    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn byte(&mut self) -> u8 {
            self.next() as u8
        }

        fn scale(&mut self) -> f32 {
            let magnitude = 0.001 + (self.next() % 1000) as f32 * 0.00005;
            if self.next() & 1 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }
    }

    const ROWS: usize = 5;

    /// Random blocks whose half-precision scale fields sit at `scale_offsets`; every other byte is
    /// random, which covers all quant values, bit planes and packed K-quant scales.
    fn weights(
        random: &mut Random,
        blocks: usize,
        block_bytes: usize,
        scale_offsets: &[usize],
    ) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..blocks * block_bytes).map(|_| random.byte()).collect();
        for block in 0..blocks {
            for &offset in scale_offsets {
                let scale = f32_to_f16(random.scale().abs()).to_le_bytes();
                bytes[block * block_bytes + offset..block * block_bytes + offset + 2]
                    .copy_from_slice(&scale);
            }
        }
        bytes
    }

    /// Random activations including the -128 and 127 extremes.
    fn activations(random: &mut Random, len: usize) -> Vec<i8> {
        (0..len)
            .map(|index| match index % 97 {
                0 => -128,
                1 => 127,
                _ => random.byte() as i8,
            })
            .collect()
    }

    fn q_0_case(
        block_bytes: usize,
        batch_size: usize,
        seed: u64,
    ) -> (Vec<u8>, Vec<i8>, Vec<f32>, usize) {
        let cols = 8 * QK_0;
        let mut random = Random(seed);
        let weights = weights(&mut random, ROWS * cols / QK_0, block_bytes, &[0]);
        let quantized = activations(&mut random, batch_size * cols);
        let scales = (0..batch_size * cols / QK_0)
            .map(|_| random.scale())
            .collect();
        (weights, quantized, scales, cols)
    }

    fn k_case(
        block_bytes: usize,
        scale_offsets: &[usize],
        batch_size: usize,
        seed: u64,
    ) -> (Vec<u8>, Vec<i8>, Vec<f32>, Vec<i16>, usize) {
        let cols = 3 * QK_K;
        let mut random = Random(seed);
        let weights = weights(&mut random, ROWS * cols / QK_K, block_bytes, scale_offsets);
        let quantized = activations(&mut random, batch_size * cols);
        let scales = (0..batch_size * cols / QK_K)
            .map(|_| random.scale())
            .collect();
        let sums = (0..batch_size * cols / Q8_K_SUM_BLOCK)
            .map(|_| (random.next() % 4096) as i16 - 2048)
            .collect();
        (weights, quantized, scales, sums, cols)
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    macro_rules! q_0_test {
        ($name:ident, $block_bytes:expr, $row:ident, $row_scalar:ident, $range:ident, $range_scalar:ident) => {
            #[test]
            fn $name() {
                if !available() {
                    return;
                }
                for (batch_size, seed) in [(1, 0x9e37_79b9_7f4a_7c15), (3, 0x2545_f491_4f6c_dd1d)] {
                    let (weights, quantized, scales, cols) =
                        q_0_case($block_bytes, batch_size, seed);
                    for batch in 0..batch_size {
                        for row in 0..ROWS {
                            // SAFETY: the test checked that the NEON kernels are available.
                            let neon =
                                unsafe { $row(&weights, &quantized, &scales, batch, row, cols) };
                            let scalar =
                                $row_scalar(&weights, &quantized, &scales, batch, row, cols);
                            assert_eq!(neon.to_bits(), scalar.to_bits(), "batch {batch} row {row}");
                        }
                    }
                    let mut neon = vec![0_f32; batch_size * ROWS];
                    let mut scalar = vec![0_f32; batch_size * ROWS];
                    // SAFETY: the outputs hold every batch plane for all rows.
                    unsafe {
                        $range(
                            &weights,
                            &quantized,
                            &scales,
                            neon.as_mut_ptr(),
                            batch_size,
                            ROWS,
                            cols,
                            0,
                            ROWS,
                        );
                        $range_scalar(
                            &weights,
                            &quantized,
                            &scales,
                            scalar.as_mut_ptr(),
                            batch_size,
                            ROWS,
                            cols,
                            0,
                            ROWS,
                        );
                    }
                    assert_eq!(bits(&neon), bits(&scalar), "batch size {batch_size}");
                }
            }
        };
    }

    q_0_test!(
        q4_0_matches_scalar_bit_for_bit,
        Q4_0_BLOCK_BYTES,
        dot_q4_0_q8_0_row,
        dot_q4_0_q8_0_row_scalar,
        compute_q4_batched_row_range,
        compute_q4_batched_row_range_scalar
    );
    q_0_test!(
        q5_0_matches_scalar_bit_for_bit,
        Q5_0_BLOCK_BYTES,
        dot_q5_0_q8_0_row,
        dot_q5_0_q8_0_row_scalar,
        compute_q5_batched_row_range,
        compute_q5_batched_row_range_scalar
    );
    q_0_test!(
        q8_0_matches_scalar_bit_for_bit,
        Q8_0_BLOCK_BYTES,
        dot_q8_0_q8_0_row,
        dot_q8_0_q8_0_row_scalar,
        compute_q8_batched_row_range,
        compute_q8_batched_row_range_scalar
    );

    #[test]
    fn q4_k_matches_scalar_bit_for_bit() {
        if !available() {
            return;
        }
        for (batch_size, seed) in [(1, 0x6a09_e667_f3bc_c909), (3, 0xbb67_ae85_84ca_a73b)] {
            let (weights, quantized, scales, sums, cols) =
                k_case(Q4_K_BLOCK_BYTES, &[0, 2], batch_size, seed);
            for batch in 0..batch_size {
                for row in 0..ROWS {
                    // SAFETY: the test checked that the NEON kernels are available.
                    let neon = unsafe {
                        dot_q4_k_q8_k_row(&weights, &quantized, &scales, &sums, batch, row, cols)
                    };
                    let scalar = dot_q4_k_q8_k_row_scalar(
                        &weights, &quantized, &scales, &sums, batch, row, cols,
                    );
                    assert_eq!(neon.to_bits(), scalar.to_bits(), "batch {batch} row {row}");
                }
            }
            let mut neon = vec![0_f32; batch_size * ROWS];
            let mut scalar = vec![0_f32; batch_size * ROWS];
            // SAFETY: the outputs hold every batch plane for all rows.
            unsafe {
                compute_q4_k_batched_row_range(
                    &weights,
                    &quantized,
                    &scales,
                    &sums,
                    neon.as_mut_ptr(),
                    batch_size,
                    ROWS,
                    cols,
                    0,
                    ROWS,
                );
                compute_q4_k_batched_row_range_scalar(
                    &weights,
                    &quantized,
                    &scales,
                    &sums,
                    scalar.as_mut_ptr(),
                    batch_size,
                    ROWS,
                    cols,
                    0,
                    ROWS,
                );
            }
            assert_eq!(bits(&neon), bits(&scalar), "batch size {batch_size}");
        }
    }

    #[test]
    fn q6_k_matches_scalar_bit_for_bit() {
        if !available() {
            return;
        }
        for (batch_size, seed) in [(1, 0x3c6e_f372_fe94_f82b), (3, 0xa54f_f53a_5f1d_36f1)] {
            let (weights, quantized, scales, _, cols) =
                k_case(Q6_K_BLOCK_BYTES, &[208], batch_size, seed);
            for batch in 0..batch_size {
                for row in 0..ROWS {
                    // SAFETY: the test checked that the NEON kernels are available.
                    let neon = unsafe {
                        dot_q6_k_q8_k_row(&weights, &quantized, &scales, batch, row, cols)
                    };
                    let scalar =
                        dot_q6_k_q8_k_row_scalar(&weights, &quantized, &scales, batch, row, cols);
                    assert_eq!(neon.to_bits(), scalar.to_bits(), "batch {batch} row {row}");
                }
            }
            let mut neon = vec![0_f32; batch_size * ROWS];
            let mut scalar = vec![0_f32; batch_size * ROWS];
            // SAFETY: the outputs hold every batch plane for all rows.
            unsafe {
                compute_q6_k_batched_row_range(
                    &weights,
                    &quantized,
                    &scales,
                    neon.as_mut_ptr(),
                    batch_size,
                    ROWS,
                    cols,
                    0,
                    ROWS,
                );
                compute_q6_k_batched_row_range_scalar(
                    &weights,
                    &quantized,
                    &scales,
                    scalar.as_mut_ptr(),
                    batch_size,
                    ROWS,
                    cols,
                    0,
                    ROWS,
                );
            }
            assert_eq!(bits(&neon), bits(&scalar), "batch size {batch_size}");
        }
    }
}
