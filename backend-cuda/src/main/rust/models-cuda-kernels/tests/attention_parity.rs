// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Off-device parity for grouped-query decode attention.
//!
//! Two different standards apply here, and the tests keep them apart on purpose:
//!
//! * against the **device decomposition** — bit-exact. Spreading the work across warp lanes
//!   must change nothing, and `attend_head_as_device` reproduces the PTX kernel's stage
//!   ordering to prove it.
//! * against the **CPU kernel** — a stated relative-L2 contract, because `core` has no
//!   floating-point `exp` for `nvptx64` and the crate ships its own. That is the only
//!   divergence, and `expf_agrees_with_the_platform_exp` bounds it directly rather than letting
//!   it hide inside the end-to-end number.
//!
//! The CPU oracle below is transcribed from `execute_attention_partition`,
//! `attention_dot_scalar`, `attention_softmax` and `attention_axpy` in
//! `backend-native/src/main/rust/model-kernels/src/lib.rs`.

// These loops index by position deliberately: the order of the float reductions is the numeric
// contract, so an iterator rewrite that looks equivalent is not. Clippy's idiomatic-indexing
// lints are wrong for this file specifically.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use models_cuda_kernels::attention::{
    self, AttentionShape, MAX_EXP_RELATIVE_ERROR, MAX_HEAD_RELATIVE_L2,
};

// ---------------------------------------------------------------------------------------------
// Oracle: the shipped CPU attention path, using the platform `exp`.
// ---------------------------------------------------------------------------------------------

fn attention_dot_oracle(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        sum = x.mul_add(*y, sum);
    }
    sum
}

fn attention_softmax_oracle(scores: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &score in scores.iter() {
        if score > max {
            max = score;
        }
    }
    let mut sum = 0.0_f32;
    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }
    let inverse = 1.0_f32 / sum;
    for score in scores.iter_mut() {
        *score *= inverse;
    }
}

fn attend_head_oracle(
    shape: &AttentionShape,
    head: usize,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
) {
    let group = shape.num_heads / shape.num_kv_heads;
    let kv = head / group;
    let q = &query[head * shape.key_length..(head + 1) * shape.key_length];
    let mut row_scores = vec![0.0_f32; shape.positions];
    for row in 0..shape.positions {
        let base = row * shape.key_dim + kv * shape.key_length;
        let k = &keys[base..base + shape.key_length];
        row_scores[row] = attention_dot_oracle(q, k) * shape.scale;
    }
    attention_softmax_oracle(&mut row_scores);
    for value in output.iter_mut() {
        *value = 0.0;
    }
    for row in 0..shape.positions {
        let base = row * shape.value_dim + kv * shape.value_length;
        let weight = row_scores[row];
        for index in 0..shape.value_length {
            output[index] = values[base + index].mul_add(weight, output[index]);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The device decomposition, simulated on the host.
// ---------------------------------------------------------------------------------------------

/// Reproduces the PTX kernel's stage ordering: lanes stride over positions for the score dots,
/// every lane recomputes the (exact, order-free) maximum, exp is elementwise, the softmax sum
/// is replayed in position order, and each output dimension accumulates over positions in order
/// on its own lane.
fn attend_head_as_device(
    shape: &AttentionShape,
    head: usize,
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    output: &mut [f32],
    lanes: usize,
) {
    let kv = shape.kv_head(head);
    let q = &query[head * shape.key_length..(head + 1) * shape.key_length];
    let mut scores = vec![0.0_f32; shape.positions];

    for lane in 0..lanes {
        let mut position = lane;
        while position < shape.positions {
            let base = position * shape.key_dim + kv * shape.key_length;
            let k = &keys[base..base + shape.key_length];
            scores[position] = attention::attention_dot(q, k) * shape.scale;
            position += lanes;
        }
    }
    let maximum = attention::score_maximum(&scores);
    for lane in 0..lanes {
        let mut position = lane;
        while position < shape.positions {
            scores[position] = attention::expf(scores[position] - maximum);
            position += lanes;
        }
    }
    let mut sum = 0.0_f32;
    for index in 0..shape.positions {
        sum += scores[index];
    }
    let inverse = 1.0_f32 / sum;
    for lane in 0..lanes {
        let mut index = lane;
        while index < shape.value_length {
            let mut accumulated = 0.0_f32;
            for position in 0..shape.positions {
                let weight = scores[position] * inverse;
                let value = values[position * shape.value_dim + kv * shape.value_length + index];
                accumulated = value.mul_add(weight, accumulated);
            }
            output[index] = accumulated;
            index += lanes;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0
    }
}

fn fixture(shape: &AttentionShape, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut rng = Rng::new(seed);
    let query = (0..shape.num_heads * shape.key_length)
        .map(|_| rng.next_f32())
        .collect();
    let keys = (0..shape.positions * shape.key_dim)
        .map(|_| rng.next_f32())
        .collect();
    let values = (0..shape.positions * shape.value_dim)
        .map(|_| rng.next_f32())
        .collect();
    (query, keys, values)
}

fn relative_l2(actual: &[f32], expected: &[f32]) -> f32 {
    let mut error = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for (a, e) in actual.iter().zip(expected) {
        let difference = (*a as f64) - (*e as f64);
        error += difference * difference;
        magnitude += (*e as f64) * (*e as f64);
    }
    if magnitude == 0.0 {
        return error.sqrt() as f32;
    }
    (error.sqrt() / magnitude.sqrt()) as f32
}

/// Gemma 4 26B-A4B has two attention shapes and both must work: 25 sliding-window layers with
/// 8 kv heads of head dimension 256, and 5 full-attention layers with 2 kv heads of 512.
/// Sources: `Gemma4ConfigTest` via `benchmark-results/2026-09-18-gpu-large-model/
/// LARGE-MODEL-MEMORY.md` on `feat/tornado-large-model-plan`.
fn gemma4_sliding(positions: usize) -> AttentionShape {
    AttentionShape {
        num_heads: 16,
        num_kv_heads: 8,
        key_length: 256,
        value_length: 256,
        key_dim: 2_048,
        value_dim: 2_048,
        positions,
        scale: 0.062_5,
    }
}

fn gemma4_full(positions: usize) -> AttentionShape {
    AttentionShape {
        num_heads: 16,
        num_kv_heads: 2,
        key_length: 512,
        value_length: 512,
        key_dim: 1_024,
        value_dim: 1_024,
        positions,
        scale: 0.044_194_173,
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[test]
fn expf_agrees_with_the_platform_exp_over_the_softmax_range() {
    // Softmax only ever evaluates exp on `score - max`, which is <= 0. Cover well past the
    // point where the result stops mattering, plus the positive side for completeness.
    let mut worst = 0.0_f32;
    let mut worst_at = 0.0_f32;
    let mut sample = -120.0_f32;
    while sample <= 40.0 {
        let expected = sample.exp();
        let actual = attention::expf(sample);
        if expected > f32::MIN_POSITIVE {
            let relative = ((actual - expected) / expected).abs();
            if relative > worst {
                worst = relative;
                worst_at = sample;
            }
        }
        sample += 0.000_37;
    }
    assert!(
        worst <= MAX_EXP_RELATIVE_ERROR,
        "expf relative error {worst} at x={worst_at} exceeds the declared contract \
         {MAX_EXP_RELATIVE_ERROR}"
    );
}

#[test]
fn expf_handles_the_edges_softmax_can_reach() {
    assert_eq!(attention::expf(0.0).to_bits(), 1.0_f32.to_bits());
    assert_eq!(attention::expf(-200.0), 0.0);
    assert!(attention::expf(f32::NEG_INFINITY) == 0.0);
    assert!(attention::expf(f32::INFINITY).is_infinite());
    assert!(attention::expf(f32::NAN).is_nan());
}

#[test]
fn the_device_decomposition_is_bit_exact_with_the_crate_reference() {
    for shape in [gemma4_sliding(97), gemma4_full(97)] {
        let (query, keys, values) = fixture(&shape, 101);
        for head in 0..shape.num_heads {
            let mut reference = vec![0.0_f32; shape.value_length];
            let mut scratch = vec![0.0_f32; shape.positions];
            attention::attend_head(
                &shape,
                head,
                &query,
                &keys,
                &values,
                &mut scratch,
                &mut reference,
            );
            for lanes in [1_usize, 3, 8, 32] {
                let mut actual = vec![0.0_f32; shape.value_length];
                attend_head_as_device(&shape, head, &query, &keys, &values, &mut actual, lanes);
                for index in 0..shape.value_length {
                    assert_eq!(
                        reference[index].to_bits(),
                        actual[index].to_bits(),
                        "head {head}, lanes {lanes}, index {index} diverged"
                    );
                }
            }
        }
    }
}

#[test]
fn one_head_meets_the_declared_contract_against_the_cpu_kernel() {
    for (label, shape) in [
        ("sliding", gemma4_sliding(129)),
        ("full", gemma4_full(129)),
        ("short", gemma4_sliding(1)),
        ("window", gemma4_sliding(1_024)),
    ] {
        let (query, keys, values) = fixture(&shape, 103);
        let mut worst = 0.0_f32;
        for head in 0..shape.num_heads {
            let mut expected = vec![0.0_f32; shape.value_length];
            attend_head_oracle(&shape, head, &query, &keys, &values, &mut expected);
            let mut actual = vec![0.0_f32; shape.value_length];
            let mut scratch = vec![0.0_f32; shape.positions];
            attention::attend_head(
                &shape,
                head,
                &query,
                &keys,
                &values,
                &mut scratch,
                &mut actual,
            );
            worst = worst.max(relative_l2(&actual, &expected));
        }
        assert!(
            worst <= MAX_HEAD_RELATIVE_L2,
            "{label}: relative L2 {worst} exceeds the declared contract {MAX_HEAD_RELATIVE_L2}"
        );
    }
}

/// Guards that the contract is a real bound and not an unreachable ceiling: if the kernel were
/// wrong, this test must fail. Perturbing one query element by a part in a thousand has to push
/// the error past the contract, or the contract is too loose to detect anything.
#[test]
fn the_declared_contract_is_tight_enough_to_catch_a_wrong_kernel() {
    let shape = gemma4_sliding(129);
    let (query, keys, values) = fixture(&shape, 107);
    let mut expected = vec![0.0_f32; shape.value_length];
    attend_head_oracle(&shape, 0, &query, &keys, &values, &mut expected);

    let mut perturbed = query.clone();
    perturbed[0] *= 1.001;
    let mut actual = vec![0.0_f32; shape.value_length];
    let mut scratch = vec![0.0_f32; shape.positions];
    attention::attend_head(
        &shape,
        0,
        &perturbed,
        &keys,
        &values,
        &mut scratch,
        &mut actual,
    );
    assert!(
        relative_l2(&actual, &expected) > MAX_HEAD_RELATIVE_L2,
        "a 0.1% perturbation stayed inside the contract, so the contract cannot detect a \
         wrong kernel; tighten it"
    );
}

#[test]
fn grouped_query_heads_read_the_kv_head_they_share() {
    let shape = gemma4_sliding(64);
    assert_eq!(shape.kv_head(0), 0);
    assert_eq!(shape.kv_head(1), 0);
    assert_eq!(shape.kv_head(2), 1);
    assert_eq!(shape.kv_head(15), 7);

    // Two query heads in one group must read identical keys, so their scores agree when their
    // queries do.
    let (mut query, keys, values) = fixture(&shape, 109);
    let (head_a, head_b) = (0_usize, 1_usize);
    for index in 0..shape.key_length {
        let value = query[head_a * shape.key_length + index];
        query[head_b * shape.key_length + index] = value;
    }
    let mut first = vec![0.0_f32; shape.value_length];
    let mut second = vec![0.0_f32; shape.value_length];
    let mut scratch = vec![0.0_f32; shape.positions];
    attention::attend_head(
        &shape,
        head_a,
        &query,
        &keys,
        &values,
        &mut scratch,
        &mut first,
    );
    attention::attend_head(
        &shape,
        head_b,
        &query,
        &keys,
        &values,
        &mut scratch,
        &mut second,
    );
    for index in 0..shape.value_length {
        assert_eq!(first[index].to_bits(), second[index].to_bits());
    }
}

#[test]
fn a_single_position_softmaxes_to_exactly_one() {
    let shape = gemma4_sliding(1);
    let (query, keys, values) = fixture(&shape, 113);
    let mut output = vec![0.0_f32; shape.value_length];
    let mut scratch = vec![0.0_f32; 1];
    attention::attend_head(&shape, 0, &query, &keys, &values, &mut scratch, &mut output);
    // With one position the weight is exactly 1.0, so the output is the value row verbatim.
    for index in 0..shape.value_length {
        assert_eq!(values[index].to_bits(), output[index].to_bits());
    }
}

#[test]
fn shape_validation_rejects_what_the_kernel_cannot_serve() {
    assert!(gemma4_sliding(1).is_valid());
    assert!(gemma4_full(1).is_valid());

    let mut uneven = gemma4_sliding(8);
    uneven.num_kv_heads = 5; // 16 is not a multiple of 5
    assert!(!uneven.is_valid());

    let mut empty = gemma4_sliding(0);
    empty.positions = 0;
    assert!(!empty.is_valid());

    let mut narrow = gemma4_sliding(8);
    narrow.key_dim = 8; // below num_kv_heads * key_length
    assert!(!narrow.is_valid());

    let mut infinite = gemma4_sliding(8);
    infinite.scale = f32::INFINITY;
    assert!(!infinite.is_valid());
}

#[test]
fn softmax_weights_sum_to_one_within_rounding() {
    let mut scores = vec![0.0_f32; 512];
    let mut rng = Rng::new(127);
    for score in scores.iter_mut() {
        *score = rng.next_f32() * 12.0;
    }
    attention::softmax(&mut scores);
    let total: f32 = scores.iter().sum();
    assert!((total - 1.0).abs() < 1.0e-5, "softmax summed to {total}");
    assert!(scores.iter().all(|weight| *weight >= 0.0));
}
