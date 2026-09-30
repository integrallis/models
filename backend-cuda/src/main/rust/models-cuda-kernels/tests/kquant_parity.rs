// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Off-device parity for the K-quant projection arithmetic.
//!
//! The oracles here are transcribed **verbatim** from the CPU path the accelerator is actually
//! gated against, rather than expressed in terms of the crate under test, so the comparison is
//! not circular. Every assertion is bit-exact equality, not a tolerance: the decomposition in
//! `crate::kquant` was designed to make that achievable, and a tolerance here would hide
//! exactly the kind of reordering it exists to prevent.
//!
//! # Which CPU path is the control
//!
//! This is the whole substance of the 2026-09-19 G1 defect, so it is spelled out. `models` has
//! **two** Q6_K row reductions on the CPU and they are not bit-identical to each other:
//!
//! * the **Panama** reduction in `vectors-core`
//!   (`PanamaVectorUtilSupport.ggufQ6_KQ8_KMatVecDot`, and `ggufQ6_KQ8_KVectorRowDot` /
//!   `ggufQ6_KQ8_KBatchedRowDot` beside it) reduces each super-block to **one** exact `i32` and
//!   folds it with **one** `fma` into **one** `f32` accumulator;
//! * the **scalar** fallback (`VectorUtilSupport.ggufQ6_KQ8_KScalarRowDot`, mirrored by
//!   `dot_q6_k_q8_k_row_scalar` in `backend-native`) keeps **eight** `f32` lane accumulators,
//!   does eight `fma`s per super-block, and adds the lanes at the end.
//!
//! Those two disagree in the last bit as soon as a row has more than one super-block;
//! `the_two_cpu_q6k_reductions_do_not_agree_bit_for_bit` measures that here rather than
//! asserting it from the source.
//!
//! The kernel is gated against the **Panama** one, because that is the path
//! `TensorOps.ggufMatmul` takes on any host that can run these kernels: `ggufQ6_KQ8_KMatVecDot`
//! only falls back to the scalar reduction when `VECTOR_BITSIZE < 256`, and every CUDA bench
//! host is x86-64 with at least AVX2. Gating against the scalar reduction instead is precisely
//! the defect that made G1 fail: it was bit-exact for a one-super-block row by coincidence and
//! one ULP wrong from two super-blocks on.
//!
//! Q4_K needs no such choice: its Panama and scalar reductions are the *same* two `fma`s per
//! super-block into one accumulator, which is why Q4_K was bit-exact on the device all along.
//!
//! The device decomposition is exercised too — `q4k_row_dot_as_device` and
//! `q6k_row_dot_as_device` compute the integer partials in a deliberately scrambled order, the
//! way 32 warp lanes striding over super-blocks would, then fold the float scales in block
//! order. That is the property the PTX kernel depends on.

// These loops index by position deliberately: the order of the float reductions is the numeric
// contract, so an iterator rewrite that looks equivalent is not. Clippy's idiomatic-indexing
// lints are wrong for this file specifically.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use models_cuda_kernels::kquant::{
    self, Q4_K_BLOCK_BYTES, Q4KRowAccumulator, Q6_K_BLOCK_BYTES, Q6KRowAccumulator, Q8_K_SUM_BLOCK,
    QK_K,
};

// ---------------------------------------------------------------------------------------------
// Oracle: the shipped CPU scalar path, transcribed unchanged.
// ---------------------------------------------------------------------------------------------

fn f16_to_f32_oracle(value: u16) -> f32 {
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
    let bits = match exponent {
        31 => sign | 0x7f80_0000 | (significand << 13),
        _ => sign | (((exponent - 15 + 127) as u32) << 23) | (significand << 13),
    };
    f32::from_bits(bits)
}

fn qk_scale_oracle(scales: &[u8], group: usize) -> i32 {
    if group < 4 {
        return (scales[group] & 0x3f) as i32;
    }
    let low = scales[group + 4] & 0x0f;
    let high = scales[group - 4] >> 6;
    (low | (high << 4)) as i32
}

fn qk_min_oracle(scales: &[u8], group: usize) -> i32 {
    if group < 4 {
        return (scales[group + 4] & 0x3f) as i32;
    }
    let low = scales[group + 4] >> 4;
    let high = scales[group] >> 6;
    (low | (high << 4)) as i32
}

fn dot_q4_k_q8_k_row_oracle(
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
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let scale_offset = batch * blocks_per_row + block;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let d = f16_to_f32_oracle(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ])) * activation_scales[scale_offset];
        let d_min = f16_to_f32_oracle(u16::from_le_bytes([
            weights[weight_offset + 2],
            weights[weight_offset + 3],
        ])) * activation_scales[scale_offset];
        let scales = &weights[weight_offset + 4..weight_offset + 16];
        let quants = &weights[weight_offset + 16..weight_offset + Q4_K_BLOCK_BYTES];
        let mut quantized_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for group in 0..8 {
            let group_scale = qk_scale_oracle(scales, group);
            let group_min = qk_min_oracle(scales, group);
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
        sum = d.mul_add(quantized_sum as f32, sum);
        sum = (-d_min).mul_add(minimum_sum as f32, sum);
    }
    sum
}

/// The CPU control: `PanamaVectorUtilSupport.ggufQ6_KQ8_KMatVecDot` in `vectors-core`,
/// transcribed unchanged.
///
/// One exact `i32` per super-block, one `fma` per super-block, one `f32` accumulator. This is
/// what a Q6_K projection produces on every host that can also run the CUDA kernels, and
/// therefore what the kernel has to reproduce bit for bit.
fn dot_q6_k_q8_k_row_oracle(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let d = f16_to_f32_oracle(u16::from_le_bytes([
            weights[weight_offset + 208],
            weights[weight_offset + 209],
        ])) * activation_scales[batch * blocks_per_row + block];
        sum = d.mul_add(
            q6k_block_integer_sum_oracle(weights, weight_offset, quantized, activation_offset)
                as f32,
            sum,
        );
    }
    sum
}

/// The *other* CPU reduction: `VectorUtilSupport.ggufQ6_KQ8_KScalarRowDot`, mirrored by
/// `dot_q6_k_q8_k_row_scalar` in `backend-native`. Eight `f32` lane accumulators.
///
/// Kept only so that
/// [`the_two_cpu_q6k_reductions_do_not_agree_bit_for_bit`] can measure the disagreement between
/// the two CPU paths instead of asserting it from reading the source. The kernel is **not**
/// gated against this one; see the module documentation.
fn dot_q6_k_q8_k_row_eight_lane_oracle(
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
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let d = f16_to_f32_oracle(u16::from_le_bytes([
            weights[weight_offset + 208],
            weights[weight_offset + 209],
        ])) * activation_scales[batch * blocks_per_row + block];
        let integer_sums =
            q6k_block_lane_sums_oracle(weights, weight_offset, quantized, activation_offset);
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

/// The eight per-lane integer sums of one Q6_K super-block, as both CPU paths compute them.
///
/// The lane split is irrelevant to the integers themselves — `i32` addition is exact and
/// associative — but the scalar oracle needs them separated, so the decode lives here once.
fn q6k_block_lane_sums_oracle(
    weights: &[u8],
    weight_offset: usize,
    quantized: &[i8],
    activation_offset: usize,
) -> [i32; 8] {
    let ql = &weights[weight_offset..weight_offset + 128];
    let qh = &weights[weight_offset + 128..weight_offset + 192];
    let scales = &weights[weight_offset + 192..weight_offset + 208];
    let mut integer_sums = [0_i32; 8];
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

/// The single exact `i32` the Panama control reduces one super-block to.
fn q6k_block_integer_sum_oracle(
    weights: &[u8],
    weight_offset: usize,
    quantized: &[i8],
    activation_offset: usize,
) -> i32 {
    q6k_block_lane_sums_oracle(weights, weight_offset, quantized, activation_offset)
        .iter()
        .sum()
}

/// Lanes in a warp: the launch shape the PTX projections require.
const WARP_LANES: usize = 32;

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

/// Deterministic, reproducible pseudo-random bytes. A fixed multiplicative generator rather
/// than a crate, so the fixtures are identical on every host and in CI.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
    fn next_u8(&mut self) -> u8 {
        self.next_u32() as u8
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Builds a Q4_K tensor of `rows * (cols / 256)` super-blocks with plausible field values.
///
/// The scale and minimum nibbles are packed with the inverse of `qk_scale` / `qk_min`, so the
/// six-bit unpacking is genuinely exercised rather than fed zeroes.
fn q4k_tensor(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = rows * (cols / QK_K);
    let mut rng = Rng::new(seed);
    let mut bytes = vec![0_u8; blocks * Q4_K_BLOCK_BYTES];
    for block in 0..blocks {
        let base = block * Q4_K_BLOCK_BYTES;
        // d and dmin as small positive binary16 values.
        let d = f32_to_f16(0.02 + (rng.next_u32() % 64) as f32 * 0.001);
        let d_min = f32_to_f16(0.005 + (rng.next_u32() % 32) as f32 * 0.0005);
        bytes[base..base + 2].copy_from_slice(&d.to_le_bytes());
        bytes[base + 2..base + 4].copy_from_slice(&d_min.to_le_bytes());
        let scales: [u32; 8] = core::array::from_fn(|_| rng.next_u32() % 64);
        let mins: [u32; 8] = core::array::from_fn(|_| rng.next_u32() % 64);
        for group in 0..4 {
            bytes[base + 4 + group] = scales[group] as u8;
            bytes[base + 8 + group] = mins[group] as u8;
        }
        for group in 4..8 {
            bytes[base + 4 + group] =
                (scales[group] as u8 & 0x0f) | ((mins[group] as u8 & 0x0f) << 4);
            bytes[base + 4 + group - 4] |= ((scales[group] >> 4) as u8) << 6;
            bytes[base + 4 + group] |= ((mins[group] >> 4) as u8) << 6;
        }
        for index in 0..128 {
            bytes[base + 16 + index] = rng.next_u8();
        }
    }
    bytes
}

/// Builds a Q6_K tensor of `rows * (cols / 256)` super-blocks.
fn q6k_tensor(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = rows * (cols / QK_K);
    let mut rng = Rng::new(seed);
    let mut bytes = vec![0_u8; blocks * Q6_K_BLOCK_BYTES];
    for block in 0..blocks {
        let base = block * Q6_K_BLOCK_BYTES;
        for index in 0..192 {
            bytes[base + index] = rng.next_u8();
        }
        // Signed sub-block scales, spanning both signs so `as i8` is exercised.
        for index in 0..16 {
            bytes[base + 192 + index] = ((rng.next_u32() % 127) as i32 - 63) as i8 as u8;
        }
        let d = f32_to_f16(0.02 + (rng.next_u32() % 64) as f32 * 0.001);
        bytes[base + 208..base + 210].copy_from_slice(&d.to_le_bytes());
    }
    bytes
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let significand = bits & 0x007f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | (significand >> 13) as u16;
    }
    let unbiased = exponent - 127 + 15;
    if unbiased >= 0x1f {
        return sign | 0x7c00;
    }
    if unbiased <= 0 {
        return sign;
    }
    sign | ((unbiased as u16) << 10) | ((significand >> 13) as u16)
}

/// Quantises a float activation plane with the crate's Q8_K path and returns the three planes.
fn quantized_activations(
    batch_size: usize,
    cols: usize,
    seed: u64,
) -> (Vec<i8>, Vec<f32>, Vec<i16>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let input: Vec<f32> = (0..batch_size * cols)
        .map(|_| rng.next_f32() * 3.0)
        .collect();
    let mut quantized = vec![0_i8; batch_size * cols];
    let mut scales = vec![0.0_f32; batch_size * (cols / QK_K)];
    let mut sums = vec![0_i16; batch_size * cols / Q8_K_SUM_BLOCK];
    kquant::quantize_q8_k(
        &input,
        batch_size,
        cols,
        &mut quantized,
        &mut scales,
        &mut sums,
    );
    (quantized, scales, sums, input)
}

// ---------------------------------------------------------------------------------------------
// The device decomposition, simulated on the host.
// ---------------------------------------------------------------------------------------------

/// Reproduces the PTX kernel's work split: 32 lanes striding over super-blocks compute the
/// exact integer partials in a scrambled order, then lane 0 folds the float scales in block
/// order.
fn q4k_row_dot_as_device(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
    lanes: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut partials = vec![kquant::Q4KBlockPartials::default(); blocks_per_row];
    for lane in 0..lanes {
        let mut block = lane;
        while block < blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
            let activation_offset = batch * cols + block * QK_K;
            let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
            partials[block] = kquant::q4k_block_partials(
                weights,
                weight_offset,
                quantized,
                activation_offset,
                activation_sums,
                sum_offset,
            );
            block += lanes;
        }
    }
    let mut accumulator = Q4KRowAccumulator::new();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_scale = activation_scales[batch * blocks_per_row + block];
        let (d, d_min) = kquant::q4k_block_scales(weights, weight_offset, activation_scale);
        accumulator.accumulate(d, d_min, partials[block]);
    }
    accumulator.finish()
}

fn q6k_row_dot_as_device(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
    lanes: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut partials = vec![0_i32; blocks_per_row];
    for lane in 0..lanes {
        let mut block = lane;
        while block < blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
            let activation_offset = batch * cols + block * QK_K;
            partials[block] =
                kquant::q6k_block_sum(weights, weight_offset, quantized, activation_offset);
            block += lanes;
        }
    }
    let mut accumulator = Q6KRowAccumulator::new();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_scale = activation_scales[batch * blocks_per_row + block];
        let d = kquant::q6k_block_scale(weights, weight_offset, activation_scale);
        accumulator.accumulate(d, partials[block]);
    }
    accumulator.finish()
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn f16_conversion_matches_the_cpu_reference_across_every_bit_pattern() {
    for bits in 0..=u16::MAX {
        let expected = f16_to_f32_oracle(bits);
        let actual = kquant::f16_to_f32(bits);
        if expected.is_nan() {
            assert!(actual.is_nan(), "bits {bits:#06x} should be NaN");
        } else {
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "bits {bits:#06x} diverged"
            );
        }
    }
}

#[test]
fn six_bit_scale_and_minimum_unpacking_matches_the_cpu_reference() {
    let mut rng = Rng::new(7);
    for _ in 0..4_096 {
        let scales: [u8; 12] = core::array::from_fn(|_| rng.next_u8());
        for group in 0..8 {
            assert_eq!(
                qk_scale_oracle(&scales, group),
                kquant::qk_scale(&scales, group)
            );
            assert_eq!(
                qk_min_oracle(&scales, group),
                kquant::qk_min(&scales, group)
            );
        }
    }
}

#[test]
fn q4k_row_dot_is_bit_exact_with_the_cpu_reference() {
    let (rows, cols) = (37, 1_024);
    let weights = q4k_tensor(rows, cols, 11);
    let (quantized, scales, sums, _) = quantized_activations(1, cols, 23);
    for row in 0..rows {
        let expected = dot_q4_k_q8_k_row_oracle(&weights, &quantized, &scales, &sums, 0, row, cols);
        let actual = kquant::q4k_row_dot(&weights, &quantized, &scales, &sums, 0, row, cols);
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "row {row}: {expected} != {actual}"
        );
    }
}

#[test]
fn q6k_row_dot_is_bit_exact_with_the_cpu_reference() {
    let (rows, cols) = (29, 1_024);
    let weights = q6k_tensor(rows, cols, 13);
    let (quantized, scales, _, _) = quantized_activations(1, cols, 29);
    for row in 0..rows {
        let expected = dot_q6_k_q8_k_row_oracle(&weights, &quantized, &scales, 0, row, cols);
        let actual = kquant::q6k_row_dot(&weights, &quantized, &scales, 0, row, cols);
        assert_eq!(
            expected.to_bits(),
            actual.to_bits(),
            "row {row}: {expected} != {actual}"
        );
    }
}

/// The property the PTX kernel rests on: spreading the integer partials across warp lanes
/// changes nothing, because the only float operations are replayed in block order afterwards.
#[test]
fn q4k_warp_decomposition_is_bit_exact_for_every_lane_count() {
    let (rows, cols) = (16, 2_048);
    let weights = q4k_tensor(rows, cols, 31);
    let (quantized, scales, sums, _) = quantized_activations(1, cols, 37);
    for lanes in [1_usize, 2, 3, 5, 8, 32, 64] {
        for row in 0..rows {
            let expected =
                dot_q4_k_q8_k_row_oracle(&weights, &quantized, &scales, &sums, 0, row, cols);
            let actual =
                q4k_row_dot_as_device(&weights, &quantized, &scales, &sums, 0, row, cols, lanes);
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "lanes {lanes}, row {row} diverged"
            );
        }
    }
}

#[test]
fn q6k_warp_decomposition_is_bit_exact_for_every_lane_count() {
    let (rows, cols) = (16, 2_048);
    let weights = q6k_tensor(rows, cols, 41);
    let (quantized, scales, _, _) = quantized_activations(1, cols, 43);
    for lanes in [1_usize, 2, 3, 5, 8, 32, 64] {
        for row in 0..rows {
            let expected = dot_q6_k_q8_k_row_oracle(&weights, &quantized, &scales, 0, row, cols);
            let actual = q6k_row_dot_as_device(&weights, &quantized, &scales, 0, row, cols, lanes);
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "lanes {lanes}, row {row} diverged"
            );
        }
    }
}

/// Measures, rather than asserts from reading the source, that `models` carries two Q6_K
/// reductions on the CPU that do not agree in the last bit.
///
/// This is the fact the G1 defect turned on. The kernel can only be bit-exact with one of them;
/// it is gated against the Panama one (see the module documentation), so this test exists to
/// make the *other* one's existence visible rather than letting a later reader assume the two
/// are interchangeable.
#[test]
fn the_two_cpu_q6k_reductions_do_not_agree_bit_for_bit() {
    // The disagreement is only *observable* on inputs where the two summation orders actually
    // differ in the last bits, which is not most of them. Scan a handful of fixtures and
    // require that at least one diverges, so this test cannot quietly stop testing anything.
    let cols = 4_096;
    let mut observed_divergence = 0;
    for seed in 0..64_u64 {
        let weights = q6k_tensor(1, cols, 1_000 + seed);
        let (quantized, scales, _, _) = quantized_activations(1, cols, 2_000 + seed);
        let panama = dot_q6_k_q8_k_row_oracle(&weights, &quantized, &scales, 0, 0, cols);
        let eight_lane =
            dot_q6_k_q8_k_row_eight_lane_oracle(&weights, &quantized, &scales, 0, 0, cols);
        if panama.to_bits() != eight_lane.to_bits() {
            observed_divergence += 1;
        }
    }
    assert!(
        observed_divergence > 0,
        "the Panama and scalar CPU Q6_K reductions agreed on all 64 fixtures, so this test no \
         longer observes the hazard it was written for; widen the fixtures rather than deleting it"
    );
}

/// The regression the G1 parity gate failed on, at the host level.
///
/// The kernel's Q6_K fold used to keep eight `f32` lane accumulators, which is the *scalar* CPU
/// reduction, not the one the accelerator is gated against. It was bit-exact for a row of one
/// super-block and one ULP wrong from two super-blocks on. Pin that: the crate's fold must be
/// the Panama control at every width, and must **not** be the eight-lane fold.
#[test]
fn q6k_row_dot_is_the_panama_control_and_not_the_eight_lane_fold() {
    let cols_widths = [1_usize, 2, 3, 32, 33, 48].map(|blocks| blocks * QK_K);
    let mut observed_divergence = 0;
    for cols in cols_widths {
        for seed in 0..8_u64 {
            let weights = q6k_tensor(1, cols, 3_000 + seed);
            let (quantized, scales, _, _) = quantized_activations(1, cols, 4_000 + seed);
            let panama = dot_q6_k_q8_k_row_oracle(&weights, &quantized, &scales, 0, 0, cols);
            let actual = kquant::q6k_row_dot(&weights, &quantized, &scales, 0, 0, cols);
            assert_eq!(
                panama.to_bits(),
                actual.to_bits(),
                "cols {cols}, seed {seed}: the crate's Q6_K fold is not the CPU control"
            );
            let eight_lane =
                dot_q6_k_q8_k_row_eight_lane_oracle(&weights, &quantized, &scales, 0, 0, cols);
            if eight_lane.to_bits() != panama.to_bits() {
                observed_divergence += 1;
            }
        }
    }
    assert!(
        observed_divergence > 0,
        "no fixture separated the eight-lane fold from the CPU control, so this test would pass \
         with the defect still in place; widen the fixtures rather than deleting it"
    );
}

/// The Java device bisection (`CudaQ6KDeviceParityTest`) at the arithmetic level, for both
/// formats: a one-super-block row proves almost nothing, so walk the widths that matter,
/// including both sides of the 32-lane warp boundary.
#[test]
fn both_formats_are_bit_exact_at_every_bisection_width() {
    for blocks in [1_usize, 2, 3, 32, 33, 48] {
        let cols = blocks * QK_K;
        let rows = 3;

        let q4_weights = q4k_tensor(rows, cols, 5_000 + blocks as u64);
        let (quantized, scales, sums, _) = quantized_activations(1, cols, 6_000 + blocks as u64);
        for row in 0..rows {
            let expected =
                dot_q4_k_q8_k_row_oracle(&q4_weights, &quantized, &scales, &sums, 0, row, cols);
            let direct = kquant::q4k_row_dot(&q4_weights, &quantized, &scales, &sums, 0, row, cols);
            let as_device = q4k_row_dot_as_device(
                &q4_weights,
                &quantized,
                &scales,
                &sums,
                0,
                row,
                cols,
                WARP_LANES,
            );
            assert_eq!(
                expected.to_bits(),
                direct.to_bits(),
                "Q4_K {blocks} super-blocks, row {row}: direct fold diverged"
            );
            assert_eq!(
                expected.to_bits(),
                as_device.to_bits(),
                "Q4_K {blocks} super-blocks, row {row}: warp decomposition diverged"
            );
        }

        let q6_weights = q6k_tensor(rows, cols, 7_000 + blocks as u64);
        let (quantized, scales, _, _) = quantized_activations(1, cols, 8_000 + blocks as u64);
        for row in 0..rows {
            let expected = dot_q6_k_q8_k_row_oracle(&q6_weights, &quantized, &scales, 0, row, cols);
            let direct = kquant::q6k_row_dot(&q6_weights, &quantized, &scales, 0, row, cols);
            let as_device =
                q6k_row_dot_as_device(&q6_weights, &quantized, &scales, 0, row, cols, WARP_LANES);
            assert_eq!(
                expected.to_bits(),
                direct.to_bits(),
                "Q6_K {blocks} super-blocks, row {row}: direct fold diverged"
            );
            assert_eq!(
                expected.to_bits(),
                as_device.to_bits(),
                "Q6_K {blocks} super-blocks, row {row}: warp decomposition diverged"
            );
        }
    }
}

#[test]
fn q8k_quantisation_round_trips_the_ggml_rounding_rule() {
    // The bit trick must agree with an explicit round-half-away-from-zero over the range the
    // quantiser actually produces.
    for step in -400_000..400_000_i32 {
        let value = step as f32 / 1_000.0;
        let expected = if value >= 0.0 {
            (value + 0.5).floor() as i32
        } else {
            -((-value + 0.5).floor() as i32)
        };
        let actual = kquant::ggml_nearest_int(value);
        // The trick rounds halfway cases to even; only compare where that cannot bite.
        if (value - value.trunc()).abs() != 0.5 {
            assert_eq!(expected, actual, "value {value} rounded to {actual}");
        }
    }
}

#[test]
fn q8k_quantisation_keeps_scales_sums_and_quants_consistent() {
    let (batch_size, cols) = (3, 512);
    let (quantized, scales, sums, input) = quantized_activations(batch_size, cols, 59);
    let blocks_per_row = cols / QK_K;
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let offset = batch * cols + block * QK_K;
            let sum_base = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
            for sub in 0..QK_K / Q8_K_SUM_BLOCK {
                let expected: i32 = (0..Q8_K_SUM_BLOCK)
                    .map(|index| quantized[offset + sub * Q8_K_SUM_BLOCK + index] as i32)
                    .sum();
                assert_eq!(expected as i16, sums[sum_base + sub]);
            }
            // Dequantising must land within one quantisation step of the input.
            let scale = scales[batch * blocks_per_row + block];
            for index in 0..QK_K {
                let restored = quantized[offset + index] as f32 * scale;
                let original = input[offset + index];
                assert!(
                    (restored - original).abs() <= scale.abs() * 1.5 + 1e-4,
                    "batch {batch} block {block} index {index}: {restored} vs {original}"
                );
            }
        }
    }
}

#[test]
fn an_all_zero_activation_block_produces_a_zero_scale_and_a_zero_dot() {
    let cols = 256;
    let weights = q4k_tensor(4, cols, 61);
    let input = vec![0.0_f32; cols];
    let mut quantized = vec![0_i8; cols];
    let mut scales = vec![0.0_f32; cols / QK_K];
    let mut sums = vec![0_i16; cols / Q8_K_SUM_BLOCK];
    kquant::quantize_q8_k(&input, 1, cols, &mut quantized, &mut scales, &mut sums);
    assert_eq!(scales, vec![0.0_f32; cols / QK_K]);
    for row in 0..4 {
        assert_eq!(
            0.0_f32.to_bits(),
            kquant::q4k_row_dot(&weights, &quantized, &scales, &sums, 0, row, cols).to_bits()
        );
    }
}

/// The widest row the device kernel admits, so the shared-scratch bound is exercised rather
/// than assumed.
#[test]
fn the_widest_supported_row_stays_bit_exact() {
    let cols = 128 * QK_K; // MAX_BLOCKS_PER_ROW super-blocks.
    let weights = q4k_tensor(2, cols, 67);
    let (quantized, scales, sums, _) = quantized_activations(1, cols, 71);
    for row in 0..2 {
        let expected = dot_q4_k_q8_k_row_oracle(&weights, &quantized, &scales, &sums, 0, row, cols);
        let actual = q4k_row_dot_as_device(&weights, &quantized, &scales, &sums, 0, row, cols, 32);
        assert_eq!(expected.to_bits(), actual.to_bits(), "row {row} diverged");
    }
}

#[test]
fn batched_rows_select_the_right_activation_plane() {
    let (rows, cols, batch_size) = (8, 512, 4);
    let weights = q4k_tensor(rows, cols, 73);
    let (quantized, scales, sums, _) = quantized_activations(batch_size, cols, 79);
    for batch in 0..batch_size {
        for row in 0..rows {
            let expected =
                dot_q4_k_q8_k_row_oracle(&weights, &quantized, &scales, &sums, batch, row, cols);
            let actual =
                kquant::q4k_row_dot(&weights, &quantized, &scales, &sums, batch, row, cols);
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "batch {batch} row {row} diverged"
            );
        }
    }
}
