// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Shared grouped-query attention arithmetic for the single-token decode step.
//!
//! As with [`crate::kquant`], every function compiles unchanged for the host and for
//! `nvptx64-nvidia-cuda`, so the host tests exercise the arithmetic the device runs.
//!
//! # Numeric contract, stated
//!
//! Attention **is** bit-exact with the CPU path. It was not, until 2026-09-26, and the reason is
//! worth keeping: `expf` below was an independently written minimax polynomial, while the CPU
//! kernel (`GroupedQueryAttentionKernel.expScalar`) uses a Taylor-coefficient polynomial with a
//! different Cody-Waite split and a different rounding step. Two implementations of the same idea,
//! neither wrong, that disagreed -- and the host test bounded `expf` against the *platform* `exp`
//! rather than against the CPU path it actually has to match, so the disagreement was invisible.
//! That is the same mistake the Q6_K fold made one layer down. `expf` is now an exact
//! transcription of the CPU's, so every stage agrees:
//!
//! | stage | device order | matches CPU bit-for-bit |
//! | --- | --- | --- |
//! | score dot product | one thread per position, `fma` over `key_length` in index order | yes |
//! | score scaling | one multiply per score | yes |
//! | softmax maximum | parallel reduction | yes — `max` is exact and order-free |
//! | `exp` | [`expf`] below | yes — an exact transcription of the CPU's |
//! | softmax sum | one thread, ascending position order | yes, given identical inputs |
//! | normalisation | one multiply per score | yes |
//! | value accumulation | one thread per output dimension, `fma` over positions in order | yes |
//!
//! `core` has no floating-point transcendentals, rustc emits no libdevice linkage for `nvptx64`,
//! and PTX's own `ex2.approx.f32` is documented at about two ulp, so [`expf`] has to be ours. The
//! requirement is not that it be accurate — it is that it be **the same function the CPU runs**.
//! It therefore uses the CPU's clamp, magic-constant rounding, Cody-Waite split, Taylor
//! coefficients and single-step exponent construction, in that order, with `fma` exactly where the
//! CPU has one.
//!
//! One host-dependency remains, and it is not ours: `MathUtil.fma` falls back to `a * b + c` when
//! the JVM reports no fast scalar FMA, so on such a host the CPU path changes and this kernel would
//! no longer match it. That is the same class of hazard as the Q6_K reduction split, it belongs in
//! `vectors`, and it is recorded rather than worked around here.
//!
//! [`MAX_EXP_RELATIVE_ERROR`] still bounds agreement with the platform `exp`, kept as a sanity
//! check that the transcription is a real exponential and not merely self-consistent.
//! [`MAX_HEAD_RELATIVE_L2`] remains the end-to-end head contract. Both are checked, not asserted.
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
const LOG2_E: f32 = 1.442_695_04_f32;
/// High part of `ln(2)`, chosen with trailing mantissa zeros so `x - n * LN2_HI` stays exact.
const LN2_HI: f32 = 0.693_145_752_f32;
/// Low part of `ln(2)`, carrying the remainder of the split.
const LN2_LO: f32 = 1.428_606_77e-6_f32;

/// Softmax input clamp, matching the CPU kernel. An infinity is clamped to an endpoint, not
/// mapped to 0 or infinity, and the clamp is what bounds the constructed exponent.
const EXP_LOWER: f32 = -87.0_f32;
/// Upper clamp; see [`EXP_LOWER`].
const EXP_UPPER: f32 = 88.0_f32;
/// `1.5 * 2^23`. Adding then subtracting it rounds a float to an integral value at f32
/// precision without a rounding intrinsic.
const ROUND_MAGIC: f32 = 12_582_912.0_f32;

/// Taylor coefficients `1/720 .. 1/2`, in Horner order.
const P0: f32 = 1.0 / 720.0;
/// See [`P0`].
const P1: f32 = 1.0 / 120.0;
/// See [`P0`].
const P2: f32 = 1.0 / 24.0;
/// See [`P0`].
const P3: f32 = 1.0 / 6.0;
/// See [`P0`].
const P4: f32 = 0.5;

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
    // NaN propagates through the Java expression too (Math.max/Math.min return NaN, and the
    // polynomial carries it), so an early return is observably identical and cheaper.
    if x.is_nan() {
        return x;
    }
    // Java clamps FIRST, so an infinity becomes a finite endpoint rather than 0 or inf. The
    // clamp is also what keeps `n + 127` inside [1, 254], which is why no subnormal two-step
    // scaling is needed here and none exists on the Java side.
    let x = if x < EXP_LOWER {
        EXP_LOWER
    } else if x > EXP_UPPER {
        EXP_UPPER
    } else {
        x
    };
    // n = round(x * log2 e), via the 1.5 * 2^23 magic constant rather than a rounding
    // intrinsic. Plain `*` and `+`, never an fma: contracting these would change the result,
    // and Rust does not contract without fast-math.
    let n = (x * LOG2_E + ROUND_MAGIC) - ROUND_MAGIC;
    // r = x - n*ln2, Cody-Waite split into a high and low part.
    let r = crate::float::fma(n, -LN2_HI, x);
    let r = crate::float::fma(n, -LN2_LO, r);
    // Taylor coefficients, in Horner order, ending in two fma-by-one steps. These are exact
    // reciprocals of factorials -- not a minimax fit -- because that is what the CPU uses.
    let mut p = P0;
    p = crate::float::fma(p, r, P1);
    p = crate::float::fma(p, r, P2);
    p = crate::float::fma(p, r, P3);
    p = crate::float::fma(p, r, P4);
    p = crate::float::fma(p, r, 1.0);
    p = crate::float::fma(p, r, 1.0);
    // Single-step exponent construction. `n` is already integral, so the cast is exact.
    p * f32::from_bits((((n as i32) + 127) as u32) << 23)
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
