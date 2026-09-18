// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Shared grouped-query attention arithmetic for the single-token decode step.
//!
//! As with [`crate::kquant`], every function compiles unchanged for the host and for
//! `nvptx64-nvidia-cuda`, so the host tests exercise the arithmetic the device runs.
//!
//! # Numeric contract, stated
//!
//! Attention is **not** bit-exact with the CPU path, and cannot be made so without also owning
//! the CPU's `expf`. Everything else is:
//!
//! | stage | device order | matches CPU bit-for-bit |
//! | --- | --- | --- |
//! | score dot product | one thread per position, `fma` over `key_length` in index order | yes |
//! | score scaling | one multiply per score | yes |
//! | softmax maximum | parallel reduction | yes — `max` is exact and order-free |
//! | `exp` | [`expf`] below | **no** — see below |
//! | softmax sum | one thread, ascending position order | yes, given identical inputs |
//! | normalisation | one multiply per score | yes |
//! | value accumulation | one thread per output dimension, `fma` over positions in order | yes |
//!
//! The single divergence is `exp`. `core` has no floating-point transcendentals, rustc emits no
//! libdevice linkage for `nvptx64`, and PTX's own `ex2.approx.f32` is documented at about two
//! ulp. So [`expf`] below is our own, and it is used on **both** sides — host and device run the
//! same polynomial, which is why the host parity test is meaningful. Its measured agreement with
//! the platform `expf` is asserted in `tests/attention_parity.rs`
//! ([`MAX_EXP_RELATIVE_ERROR`]), and the end-to-end head contract against the CPU reference is
//! [`MAX_HEAD_RELATIVE_L2`]. Both are checked, not asserted.
//!
//! [`MAX_EXP_RELATIVE_ERROR`]: crate::attention::MAX_EXP_RELATIVE_ERROR
//! [`MAX_HEAD_RELATIVE_L2`]: crate::attention::MAX_HEAD_RELATIVE_L2

#![allow(clippy::needless_range_loop)]

/// Relative error of [`expf`] against the platform `expf`, over the softmax input range.
///
/// Tested in `tests/attention_parity.rs`; it is a measured bound, not a target.
pub const MAX_EXP_RELATIVE_ERROR: f32 = 1.0e-6;

/// Relative L2 error contract of one attention head against the CPU reference.
///
/// Matches the contract the sibling TornadoVM attention branch adopted, so the two accelerator
/// arms are held to the same standard.
pub const MAX_HEAD_RELATIVE_L2: f32 = 2.0e-5;

/// `log2(e)`, from `core` so the value is the platform's rather than a transcribed literal.
const LOG2_E: f32 = core::f32::consts::LOG2_E;
/// High part of `ln(2)`, chosen with trailing mantissa zeros so `x - n * LN2_HI` stays exact.
const LN2_HI: f32 = 0.693_359_4_f32;
/// Low part of `ln(2)`, carrying the remainder of the split.
const LN2_LO: f32 = -2.121_944_4e-4_f32;

/// `exp` for the softmax, identical on host and device.
///
/// Cody-Waite argument reduction to `r` in about `[-ln2/2, ln2/2]`, a degree-5 minimax
/// polynomial, then scaling by `2^n` through direct exponent construction. Uses only add,
/// multiply, `fma` and integer bit manipulation, all of which PTX provides exactly.
///
/// Softmax only ever passes `x <= 0` (scores minus their maximum), so the underflow path
/// matters and the overflow path does not; both are handled anyway.
#[inline]
pub fn expf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    // exp(-104) is already below the smallest subnormal binary32.
    if x < -104.0 {
        return 0.0;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }
    // n = round(x * log2(e)), computed without a rounding intrinsic.
    let scaled = x * LOG2_E;
    let n = if scaled >= 0.0 {
        (scaled + 0.5) as i32
    } else {
        (scaled - 0.5) as i32
    };
    let nf = n as f32;
    // r = x - n*ln2, split so the subtraction stays exact in the high part.
    let r = crate::float::fma(nf, -LN2_HI, x);
    let r = crate::float::fma(nf, -LN2_LO, r);
    // Minimax polynomial for exp(r) on [-ln2/2, ln2/2].
    let mut poly = 1.986_124_5e-4_f32;
    poly = crate::float::fma(poly, r, 1.390_073_5e-3_f32);
    poly = crate::float::fma(poly, r, 8.333_346e-3_f32);
    poly = crate::float::fma(poly, r, 4.166_663e-2_f32);
    poly = crate::float::fma(poly, r, 1.666_666_6e-1_f32);
    poly = crate::float::fma(poly, r, 5.0e-1_f32);
    poly = crate::float::fma(poly, r * r, r + 1.0);
    scale_by_power_of_two(poly, n)
}

/// Multiplies `value` by `2^n` without calling `ldexp`.
///
/// Splits the exponent in two steps so subnormal results stay representable instead of
/// flushing through an out-of-range intermediate.
#[inline(always)]
fn scale_by_power_of_two(value: f32, n: i32) -> f32 {
    if (-126..=127).contains(&n) {
        return value * f32::from_bits(((n + 127) as u32) << 23);
    }
    if n > 127 {
        return value
            * f32::from_bits((254_u32) << 23)
            * f32::from_bits(((n - 127 + 127) as u32) << 23);
    }
    // n < -126: two halves, each in range, so the product underflows gradually.
    let half = n / 2;
    let rest = n - half;
    value
        * f32::from_bits(((half.max(-126) + 127) as u32) << 23)
        * f32::from_bits(((rest.max(-126) + 127) as u32) << 23)
}

/// Dot product of a query head against one cached key row, in index order with `fma`.
///
/// Matches `attention_dot_scalar` in the CPU kernel crate exactly.
#[inline]
pub fn attention_dot(query: &[f32], key: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for index in 0..query.len() {
        sum = crate::float::fma(query[index], key[index], sum);
    }
    sum
}

/// The maximum of a score row. Exact and order-free, so a device may reduce it in any shape.
#[inline]
pub fn score_maximum(scores: &[f32]) -> f32 {
    let mut maximum = f32::NEG_INFINITY;
    for &score in scores.iter() {
        if score > maximum {
            maximum = score;
        }
    }
    maximum
}

/// In-place softmax over one head's score row.
///
/// Matches `attention_softmax` in the CPU kernel crate, with [`expf`] in place of the platform
/// `exp`. The summation is deliberately sequential: float addition is not associative and the
/// CPU reference sums in ascending position order.
#[inline]
pub fn softmax(scores: &mut [f32]) {
    let maximum = score_maximum(scores);
    let mut sum = 0.0_f32;
    for index in 0..scores.len() {
        scores[index] = expf(scores[index] - maximum);
        sum += scores[index];
    }
    let inverse = 1.0_f32 / sum;
    for index in 0..scores.len() {
        scores[index] *= inverse;
    }
}

/// Describes one grouped-query attention decode step over a contiguous cached window.
///
/// The device kernel takes a single span. The CPU kernel takes two, because the host-side KV
/// cache can present a shared prefix and a private suffix as separate spans; the Java side
/// linearises that before dispatch, and refuses the accelerator when it cannot. See
/// `CudaGgufBatchedMatrixKernel` for the refusal and gate G6.
#[derive(Clone, Copy, Debug)]
pub struct AttentionShape {
    /// Query heads.
    pub num_heads: usize,
    /// Key/value heads; `num_heads` must be a multiple of it.
    pub num_kv_heads: usize,
    /// Values per query/key head.
    pub key_length: usize,
    /// Values per value head.
    pub value_length: usize,
    /// Stride between cached key rows, in floats.
    pub key_dim: usize,
    /// Stride between cached value rows, in floats.
    pub value_dim: usize,
    /// Cached positions attended by this step.
    pub positions: usize,
    /// Score scaling applied before the softmax. Passed in rather than derived: Granite
    /// persists a scale that is not `1/sqrt(key_length)`.
    pub scale: f32,
}

impl AttentionShape {
    /// Whether the shape is self-consistent and supportable.
    pub fn is_valid(&self) -> bool {
        self.num_heads > 0
            && self.num_kv_heads > 0
            && self.num_heads.is_multiple_of(self.num_kv_heads)
            && self.key_length > 0
            && self.value_length > 0
            && self.positions > 0
            && self.key_dim >= self.num_kv_heads * self.key_length
            && self.value_dim >= self.num_kv_heads * self.value_length
            && self.scale.is_finite()
    }

    /// KV head serving query head `head`.
    #[inline(always)]
    pub fn kv_head(&self, head: usize) -> usize {
        head / (self.num_heads / self.num_kv_heads)
    }
}

/// Computes one query head of a decode step, writing `value_length` floats into `output`.
///
/// `scores` is scratch of at least `shape.positions`. The host reference used by the parity
/// tests; the device kernel performs the same stages with the decomposition documented at the
/// top of this module.
pub fn attend_head(
    shape: &AttentionShape,
    head: usize,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    scores: &mut [f32],
    output: &mut [f32],
) {
    let kv = shape.kv_head(head);
    let q = &query[head * shape.key_length..(head + 1) * shape.key_length];
    let row_scores = &mut scores[..shape.positions];
    for position in 0..shape.positions {
        let base = position * shape.key_dim + kv * shape.key_length;
        let k = &keys[base..base + shape.key_length];
        row_scores[position] = attention_dot(q, k) * shape.scale;
    }
    softmax(row_scores);
    for index in 0..shape.value_length {
        output[index] = 0.0;
    }
    for position in 0..shape.positions {
        let base = position * shape.value_dim + kv * shape.value_length;
        let weight = row_scores[position];
        for index in 0..shape.value_length {
            output[index] = crate::float::fma(values[base + index], weight, output[index]);
        }
    }
}
