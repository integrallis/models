// Copyright 2026 Integrallis Software, LLC
// SLEEF EXP u10: Copyright (c) 2010-2021 Naoki Shibata and contributors.
// SPDX-License-Identifier: Apache-2.0 AND BSL-1.0
//! CLIP's float32 exponential and eight-lane row reductions. EXP follows SLEEF
//! 5a1d179df9cf652951b59010a2d2075372d67f68, pinned by PyTorch 2.9.0.
//! See LICENSE-BOOST-SLEEF.txt. The graph, buffers and normalization remain Java-owned.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

fn exp_one(x: f32) -> f32 {
    if x.is_nan() {
        return x + x;
    }
    if x < -104.0 {
        return 0.;
    }
    if x > 100.0 {
        return f32::INFINITY;
    }
    let q = (x * 1.442695040888963407359924681001892137f32).round_ties_even() as i32;
    let s = (q as f32).mul_add(-0.693145751953125, x);
    let s = (q as f32).mul_add(-1.428606765330187045e-6, s);
    let mut u = 0.000198527617612853646278381f32;
    for coefficient in [
        0.00139304355252534151077271,
        0.00833336077630519866943359,
        0.0416664853692054748535156,
        0.166666671633720397949219,
        0.5,
    ] {
        u = u.mul_add(s, coefficient);
    }
    u = 1.0 + (s * s).mul_add(u, s);
    let h = q >> 1;
    u *= f32::from_bits(((h + 127) as u32) << 23);
    u * f32::from_bits(((q - h + 127) as u32) << 23)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let a = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(-104.)),
        _mm256_set1_ps(100.),
    );
    let q = _mm256_cvtps_epi32(_mm256_mul_ps(
        a,
        _mm256_set1_ps(1.442695040888963407359924681001892137),
    ));
    let fq = _mm256_cvtepi32_ps(q);
    let s = _mm256_fmadd_ps(fq, _mm256_set1_ps(-0.693145751953125), a);
    let s = _mm256_fmadd_ps(fq, _mm256_set1_ps(-1.428606765330187045e-6), s);
    let mut u = _mm256_set1_ps(0.000198527617612853646278381);
    for coefficient in [
        0.00139304355252534151077271,
        0.00833336077630519866943359,
        0.0416664853692054748535156,
        0.166666671633720397949219,
        0.5,
    ] {
        u = _mm256_fmadd_ps(u, s, _mm256_set1_ps(coefficient));
    }
    u = _mm256_add_ps(
        _mm256_set1_ps(1.),
        _mm256_fmadd_ps(_mm256_mul_ps(s, s), u, s),
    );
    let h = _mm256_srai_epi32::<1>(q);
    u = _mm256_mul_ps(
        u,
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
            h,
            _mm256_set1_epi32(127),
        ))),
    );
    u = _mm256_mul_ps(
        u,
        _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
            _mm256_sub_epi32(q, h),
            _mm256_set1_epi32(127),
        ))),
    );
    u = _mm256_blendv_ps(
        u,
        _mm256_setzero_ps(),
        _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(-104.)),
    );
    u = _mm256_blendv_ps(
        u,
        _mm256_set1_ps(f32::INFINITY),
        _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(100.)),
    );
    _mm256_blendv_ps(u, _mm256_add_ps(x, x), _mm256_cmp_ps::<_CMP_UNORD_Q>(x, x))
}
fn exponentials(values: &mut [f32], maximum: f32) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { exp_rows_avx2(values, maximum) };
        return;
    }
    for x in values {
        *x = exp_one(*x - maximum);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_rows_avx2(values: &mut [f32], maximum: f32) {
    use std::arch::x86_64::*;
    let n = values.len() / 8 * 8;
    unsafe {
        for i in (0..n).step_by(8) {
            let p = values.as_mut_ptr().add(i);
            _mm256_storeu_ps(
                p,
                exp8(_mm256_sub_ps(_mm256_loadu_ps(p), _mm256_set1_ps(maximum))),
            );
        }
    }
    for x in &mut values[n..] {
        *x = exp_one(*x - maximum);
    }
}
pub(super) fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    exponentials(values, maximum);
    let mut sums = [0f32; 8];
    for (i, value) in values.iter().enumerate() {
        sums[i % 8] += value;
    }
    let sum = if values.len() < 8 {
        sums[..values.len()].iter().sum()
    } else {
        let h = [
            sums[0] + sums[4],
            sums[1] + sums[5],
            sums[2] + sums[6],
            sums[3] + sums[7],
        ];
        (h[0] + h[2]) + (h[1] + h[3])
    };
    let inverse = 1f32 / sum;
    for x in values {
        *x *= inverse;
    }
}
pub(super) fn quick_gelu(value: f32) -> f32 {
    value * (1f32 / (1f32 + exp_one(-1.702f32 * value)))
}

// oneDNN 3.7.1, 8d263e693366ef8db40acc569cc7d8edf644556d:
// Copyright 2019-2025 Intel Corporation. Apache-2.0.
// Source-order GELU/EXP from jit_uni_eltwise_injector.cpp, AVX2 path.
fn projector_exp(value: f32) -> f32 {
    let c = f32::from_bits;
    if value < c(0xc2aeac50) {
        return 0.;
    }
    let x = value.clamp(c(0xc2aeac50), c(0x42b17218));
    let n = (x * c(0x3fb8aa3b) + 0.5).floor();
    let r = (-n).mul_add(std::f32::consts::LN_2, x);
    let mut p = c(0x3c07cfce);
    for bits in [0x3d2b9d0d, 0x3e2aad40, 0x3efffee3, 0x3f7ffffb, 0x3f800000] {
        p = p.mul_add(r, c(bits));
    }
    (p * c(((n as i32 + 126) as u32) << 23)) * 2.
}
/// Float32 GELU preserving the reference projector's FMA and exponential boundaries.
pub(super) fn gelu_erf(value: f32) -> f32 {
    let c = f32::from_bits;
    let x = value * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1f32 / c(0x3ea7ba05).mul_add(x.abs(), 1.);
    let r = c(0x3f87dc22)
        .mul_add(t, c(0xbfba00e3))
        .mul_add(t, c(0x3fb5f0e3))
        .mul_add(t, c(0xbe91a98e))
        .mul_add(t, c(0x3e827906));
    let erf = (-projector_exp(-(x * x)) * t).mul_add(r, 1.);
    let erf = c(erf.to_bits() ^ (value.to_bits() & 0x80000000));
    let half = value * 0.5;
    erf.mul_add(half, half)
}

pub(super) fn gelu_rows(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        unsafe { gelu_rows_avx2(values) };
        return;
    }
    for x in values {
        *x = gelu_erf(*x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_rows_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let c = |bits| _mm256_set1_ps(f32::from_bits(bits));
    let one = _mm256_set1_ps(1.);
    let half = _mm256_set1_ps(0.5);
    let sign = _mm256_set1_ps(-0.);
    let end = values.len() / 8 * 8;
    unsafe {
        for i in (0..end).step_by(8) {
            let p = values.as_mut_ptr().add(i);
            let value = _mm256_loadu_ps(p);
            let x = _mm256_mul_ps(value, c(0x3f3504f3));
            let abs = _mm256_andnot_ps(sign, x);
            let t = _mm256_div_ps(one, _mm256_fmadd_ps(c(0x3ea7ba05), abs, one));
            let exponent = _mm256_xor_ps(sign, _mm256_mul_ps(x, x));
            let low = c(0xc2aeac50);
            let clamped = _mm256_max_ps(low, _mm256_min_ps(exponent, c(0x42b17218)));
            let n = _mm256_floor_ps(_mm256_add_ps(_mm256_mul_ps(clamped, c(0x3fb8aa3b)), half));
            let residual = _mm256_fnmadd_ps(n, c(0x3f317218), clamped);
            let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(
                _mm256_cvtps_epi32(n),
                _mm256_set1_epi32(126),
            )));
            let scale = _mm256_andnot_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(exponent, low), scale);
            let mut exp = c(0x3c07cfce);
            for bits in [0x3d2b9d0d, 0x3e2aad40, 0x3efffee3, 0x3f7ffffb, 0x3f800000] {
                exp = _mm256_fmadd_ps(exp, residual, c(bits));
            }
            exp = _mm256_mul_ps(_mm256_mul_ps(exp, scale), _mm256_set1_ps(2.));
            let mut polynomial = c(0x3f87dc22);
            for bits in [0xbfba00e3, 0x3fb5f0e3, 0xbe91a98e, 0x3e827906] {
                polynomial = _mm256_fmadd_ps(polynomial, t, c(bits));
            }
            let erf = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_xor_ps(sign, exp), t), polynomial, one);
            let erf = _mm256_xor_ps(erf, _mm256_and_ps(sign, value));
            let s = _mm256_mul_ps(value, half);
            _mm256_storeu_ps(p, _mm256_fmadd_ps(erf, s, s));
        }
    }
    for x in &mut values[end..] {
        *x = gelu_erf(*x);
    }
}
