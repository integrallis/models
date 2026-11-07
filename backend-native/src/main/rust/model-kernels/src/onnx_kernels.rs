//! Kernels for the pure-Java ONNX executor (`models-backend-onnx`): a single-threaded f32 matrix
//! product over `matrixmultiply`, and a single-precision sine computed in double precision with
//! exact range reduction. Each call is synchronous and stateless; no pointer escapes it.
//!
//! The sine matches `(float) Math.sin((double) x)` to within the final rounding: the argument is
//! reduced by a three-part Cody-Waite split of pi/2 (fdlibm's PIO2_1..3, each product exact for
//! |k| < 2^20, i.e. |x| below about 1.6e6) and evaluated with fdlibm's kernel polynomials. Larger
//! arguments fall back to the platform `sin`.

use std::panic::{AssertUnwindSafe, catch_unwind};

// fdlibm's published constants, kept digit for digit so they can be checked against the source.
#[allow(clippy::excessive_precision)]
mod fdlibm {
    pub const PIO2_1: f64 = 1.57079632673412561417e+00;
    pub const PIO2_2: f64 = 6.07710050650619224932e-11;
    pub const PIO2_3: f64 = 2.02226624879595063154e-21;
    pub const S1: f64 = -1.66666666666666324348e-01;
    pub const S2: f64 = 8.33333333332248946124e-03;
    pub const S3: f64 = -1.98412698298579493134e-04;
    pub const S4: f64 = 2.75573137070700676789e-06;
    pub const S5: f64 = -2.50507602534068634195e-08;
    pub const S6: f64 = 1.58969099521155010221e-10;
    pub const C1: f64 = 4.16666666666666019037e-02;
    pub const C2: f64 = -1.38888888888741095749e-03;
    pub const C3: f64 = 2.48015872894767294178e-05;
    pub const C4: f64 = -2.75573143513906633035e-07;
    pub const C5: f64 = 2.08757232129817482790e-09;
    pub const C6: f64 = -1.13596475577881948265e-11;
}
use fdlibm::*;

#[unsafe(no_mangle)]
pub extern "C" fn jmodels_onnx_abi() -> i32 {
    2
}

/// `C[i][j] += sum_p A[i][p] * B[p][j]`, row-major with leading dimensions `lda`, `ldb`, `ldc`.
/// Returns 0 on success, -1 for invalid arguments, -3 on panic.
/// # Safety
/// The pointers must be valid for the extents implied by the dimensions and leading dimensions
/// for the duration of the call; C must not overlap A or B.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_sgemm(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if a.is_null() || b.is_null() || c.is_null() || lda < k || ldb < n || ldc < n {
            return -1;
        }
        if m == 0 || n == 0 || k == 0 {
            return 0;
        }
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a,
                lda as isize,
                1,
                b,
                ldb as isize,
                1,
                1.0,
                c,
                ldc as isize,
                1,
            );
        }
        0
    }))
    .unwrap_or(-3)
}

const REDUCIBLE: f64 = 1.6e6;

/// 2^52 + 2^51: adding it to a double of magnitude below 2^51 rounds to the nearest integer and
/// leaves that integer in the low mantissa bits, without a float-to-int conversion (which AVX2
/// cannot do packed for 64-bit lanes and which would keep this loop scalar).
const ROUND_MAGIC: f64 = 6_755_399_441_055_744.0;

/// The polynomial path, valid for |x| < REDUCIBLE; branch-free so the loop vectorizes.
#[inline(always)]
fn sin_reduced(x: f32) -> f32 {
    let xd = x as f64;
    let t = xd * std::f64::consts::FRAC_2_PI + ROUND_MAGIC;
    let kf = t - ROUND_MAGIC;
    let q = t.to_bits() & 3;
    let r = ((xd - kf * PIO2_1) - kf * PIO2_2) - kf * PIO2_3;
    let z = r * r;
    let s = r + r * z * (S1 + z * (S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))));
    let c = 1.0 - 0.5 * z + z * z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    let odd = (q & 1).wrapping_neg(); // all ones when the quadrant is odd
    let v = f64::from_bits((s.to_bits() & !odd) | (c.to_bits() & odd));
    f64::from_bits(v.to_bits() ^ ((q & 2) << 62)) as f32
}

#[cfg(test)]
fn sin_one(x: f32) -> f32 {
    let xd = x as f64;
    if xd.is_nan() || xd.abs() >= REDUCIBLE {
        return xd.sin() as f32;
    }
    sin_reduced(x)
}

/// The platform sine for arguments the three-part reduction cannot take. Out of line on purpose:
/// `f64::sin` lowers to an intrinsic LLVM may speculate, and inlined into the fallback scan it was
/// if-converted into a call on every element.
#[cold]
#[inline(never)]
fn sin_out_of_range(v: f64) -> f32 {
    v.sin() as f32
}

#[inline(always)]
fn sin_slice(x: &[f32], y: &mut [f32]) {
    for (o, &v) in y.iter_mut().zip(x.iter()) {
        *o = sin_reduced(v);
    }
    // Arguments the three-part reduction cannot take exactly (rare: |x| >= 1.6e6, NaN, inf).
    for (o, &v) in y.iter_mut().zip(x.iter()) {
        let vd = v as f64;
        if vd.is_nan() || vd.abs() >= REDUCIBLE {
            *o = sin_out_of_range(vd);
        }
    }
}

/// Four lanes of [`sin_reduced`] at a time with explicit AVX2/FMA intrinsics; the tail and the
/// out-of-range fallback reuse the scalar code.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sin_slice_avx2(x: &[f32], y: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len().min(y.len());
    let lanes = n - n % 4;
    unsafe {
        let two_over_pi = _mm256_set1_pd(std::f64::consts::FRAC_2_PI);
        let magic = _mm256_set1_pd(ROUND_MAGIC);
        let (p1, p2, p3) = (
            _mm256_set1_pd(PIO2_1),
            _mm256_set1_pd(PIO2_2),
            _mm256_set1_pd(PIO2_3),
        );
        let (s1, s2, s3) = (_mm256_set1_pd(S1), _mm256_set1_pd(S2), _mm256_set1_pd(S3));
        let (s4, s5, s6) = (_mm256_set1_pd(S4), _mm256_set1_pd(S5), _mm256_set1_pd(S6));
        let (c1, c2, c3) = (_mm256_set1_pd(C1), _mm256_set1_pd(C2), _mm256_set1_pd(C3));
        let (c4, c5, c6) = (_mm256_set1_pd(C4), _mm256_set1_pd(C5), _mm256_set1_pd(C6));
        let one = _mm256_set1_pd(1.0);
        let half = _mm256_set1_pd(0.5);
        let bit0 = _mm256_set1_epi64x(1);
        let bit1 = _mm256_set1_epi64x(2);
        let mut i = 0;
        while i < lanes {
            let xv = _mm256_cvtps_pd(_mm_loadu_ps(x.as_ptr().add(i)));
            let t = _mm256_fmadd_pd(xv, two_over_pi, magic);
            let kf = _mm256_sub_pd(t, magic);
            let q = _mm256_castpd_si256(t);
            let mut r = _mm256_fnmadd_pd(kf, p1, xv);
            r = _mm256_fnmadd_pd(kf, p2, r);
            r = _mm256_fnmadd_pd(kf, p3, r);
            let z = _mm256_mul_pd(r, r);
            let mut ps = _mm256_fmadd_pd(s6, z, s5);
            ps = _mm256_fmadd_pd(ps, z, s4);
            ps = _mm256_fmadd_pd(ps, z, s3);
            ps = _mm256_fmadd_pd(ps, z, s2);
            ps = _mm256_fmadd_pd(ps, z, s1);
            let sv = _mm256_fmadd_pd(_mm256_mul_pd(r, z), ps, r);
            let mut pc = _mm256_fmadd_pd(c6, z, c5);
            pc = _mm256_fmadd_pd(pc, z, c4);
            pc = _mm256_fmadd_pd(pc, z, c3);
            pc = _mm256_fmadd_pd(pc, z, c2);
            pc = _mm256_fmadd_pd(pc, z, c1);
            let cv = _mm256_fmadd_pd(_mm256_mul_pd(z, z), pc, _mm256_fnmadd_pd(half, z, one));
            let odd = _mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_and_si256(q, bit0), bit0));
            let v = _mm256_blendv_pd(sv, cv, odd);
            let sign = _mm256_castsi256_pd(_mm256_slli_epi64::<62>(_mm256_and_si256(q, bit1)));
            let out = _mm256_cvtpd_ps(_mm256_xor_pd(v, sign));
            _mm_storeu_ps(y.as_mut_ptr().add(i), out);
            i += 4;
        }
    }
    for j in lanes..n {
        y[j] = sin_reduced(x[j]);
    }
    for (o, &v) in y.iter_mut().zip(x.iter()) {
        let vd = v as f64;
        if vd.is_nan() || vd.abs() >= REDUCIBLE {
            *o = sin_out_of_range(vd);
        }
    }
}

/// `y[i] = sin(x[i])` for `n` floats. Returns 0 on success, -1 for invalid arguments, -3 on panic.
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_sin_f32(x: *const f32, y: *mut f32, n: usize) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if n == 0 {
            return 0;
        }
        if x.is_null() || y.is_null() {
            return -1;
        }
        let xs = unsafe { std::slice::from_raw_parts(x, n) };
        let ys = unsafe { std::slice::from_raw_parts_mut(y, n) };
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                unsafe { sin_slice_avx2(xs, ys) };
                return 0;
            }
        }
        sin_slice(xs, ys);
        0
    }))
    .unwrap_or(-3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_matches_double_precision_sine_rounded_to_float() {
        let mut worst = 0u32;
        let mut x = -200_000.0f32;
        while x < 200_000.0 {
            let ours = sin_one(x);
            let reference = (x as f64).sin() as f32;
            let ulps = (ours.to_bits() as i64 - reference.to_bits() as i64).unsigned_abs() as u32;
            worst = worst.max(ulps);
            x += 0.37;
        }
        assert!(worst <= 1, "worst disagreement {worst} ulp");
        // The dispatched path (AVX2 on x86_64 when present) over the same arguments.
        let xs: Vec<f32> = (0..1_081_082)
            .map(|i| -200_000.0f32 + i as f32 * 0.37)
            .collect();
        let mut ys = vec![0f32; xs.len()];
        assert_eq!(
            unsafe { jmodels_onnx_sin_f32(xs.as_ptr(), ys.as_mut_ptr(), xs.len()) },
            0
        );
        let mut worst_simd = 0u32;
        for (&v, &o) in xs.iter().zip(ys.iter()) {
            let reference = (v as f64).sin() as f32;
            worst_simd = worst_simd
                .max((o.to_bits() as i64 - reference.to_bits() as i64).unsigned_abs() as u32);
        }
        assert!(
            worst_simd <= 1,
            "dispatched path: worst disagreement {worst_simd} ulp"
        );
        assert_eq!(sin_one(0.0), 0.0);
        assert!(sin_one(f32::NAN).is_nan());
        assert_eq!(sin_one(3.0e7), (3.0e7f64).sin() as f32);
    }

    #[test]
    fn sgemm_accumulates_into_c() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let mut c = [10.0f32, 10.0, 10.0, 10.0];
        let rc =
            unsafe { jmodels_onnx_sgemm(2, 3, 2, a.as_ptr(), 3, b.as_ptr(), 2, c.as_mut_ptr(), 2) };
        assert_eq!(rc, 0);
        assert_eq!(c, [14.0, 15.0, 20.0, 21.0]);
    }
}

// ---- exp, tanh, logistic, erf ----
//
// onnxruntime's MLAS approximations (onnxruntime/core/mlas/lib/compute.cpp, tanh.cpp, logistic.cpp,
// erf.cpp at v1.23.2; Copyright (c) Microsoft Corporation, MIT License), translated with their
// constants digit for digit so the executor computes what the reference runtime computes. Each is
// written branch-free on one f32 and applied over a slice inside an AVX2/FMA target-feature
// function, where LLVM vectorizes the loop and lowers `mul_add` to a fused multiply-add. Clamps use
// comparisons rather than `f32::min`/`max` so that a NaN input stays NaN, as in MLAS. The Java
// executor's `VectorMath` carries the same formulas for hosts without this library.

// MLAS's published constants, kept as written upstream so they can be checked against the source;
// `LOG2_E` there is 1.44269504088896341 rounded to f32, not the std constant by name.
#[allow(clippy::excessive_precision, clippy::approx_constant)]
mod mlas {
    pub const EXP_LOWER: f32 = -103.972_084;
    pub const EXP_UPPER: f32 = 88.776_26;
    pub const EXP_BIAS: f32 = 12_582_912.0;
    pub const EXP_LOG2_RECIPROCAL: f32 = 1.442_695;
    pub const EXP_LOG2_HIGH: f32 = -6.931_457_52e-1;
    pub const EXP_LOG2_LOW: f32 = -1.428_606_77e-6;
    pub const EXP_P0: f32 = f32::from_bits(0x3AB4_A000); // 0x1.694000p-10
    pub const EXP_P1: f32 = f32::from_bits(0x3C09_2F6E); // 0x1.125edcp-7
    pub const EXP_P2: f32 = f32::from_bits(0x3D2A_ADAD); // 0x1.555b5ap-5
    pub const EXP_P3: f32 = f32::from_bits(0x3E2A_AA28); // 0x1.555450p-3
    pub const EXP_P4: f32 = f32::from_bits(0x3EFF_FFFB); // 0x1.fffff6p-2
    pub const EXP_P56: f32 = 1.0;
    pub const EXP_MIN_EXPONENT: i32 = 0xC100_0000u32 as i32;
    pub const EXP_MAX_EXPONENT: i32 = 0x3F80_0000;

    pub const TANH_LOWER: f32 = -9.0;
    pub const TANH_UPPER: f32 = 9.0;
    pub const TANH_A13: f32 = -2.760_768_477_423_55e-16;
    pub const TANH_A11: f32 = 2.000_187_904_824_77e-13;
    pub const TANH_A9: f32 = -8.604_671_522_137_35e-11;
    pub const TANH_A7: f32 = 5.122_297_090_371_14e-08;
    pub const TANH_A5: f32 = 1.485_722_357_179_79e-05;
    pub const TANH_A3: f32 = 6.372_619_288_754_36e-04;
    pub const TANH_A1: f32 = 4.893_524_558_917_86e-03;
    pub const TANH_B6: f32 = 1.198_258_394_667_02e-06;
    pub const TANH_B4: f32 = 1.185_347_056_866_54e-04;
    pub const TANH_B2: f32 = 2.268_434_632_439_00e-03;
    pub const TANH_B0: f32 = 4.893_525_185_543_85e-03;

    pub const LOGISTIC_LOWER: f32 = -18.0;
    pub const LOGISTIC_UPPER: f32 = 18.0;
    pub const LOGISTIC_A9: f32 = 4.370_310_125_798_01e-11;
    pub const LOGISTIC_A7: f32 = 1.156_273_244_599_42e-07;
    pub const LOGISTIC_A5: f32 = 6.085_748_646_001_43e-05;
    pub const LOGISTIC_A3: f32 = 8.513_771_333_047_01e-03;
    pub const LOGISTIC_A1: f32 = 2.482_879_470_615_29e-01;
    pub const LOGISTIC_B10: f32 = 6.102_473_897_556_81e-13;
    pub const LOGISTIC_B8: f32 = 5.761_021_369_934_27e-09;
    pub const LOGISTIC_B6: f32 = 6.291_067_850_170_40e-06;
    pub const LOGISTIC_B4: f32 = 1.701_988_173_740_94e-03;
    pub const LOGISTIC_B2: f32 = 1.168_176_569_044_53e-01;
    pub const LOGISTIC_B0: f32 = 9.931_519_210_231_80e-01;

    pub const ERF_UPPER_ABS: f32 = 3.925;
    pub const ERF_SPLIT: f32 = 0.921_875;
    pub const ERF_SMALL_P0: f32 = -5.991_049_69e-4;
    pub const ERF_SMALL_P1: f32 = 4.993_395_88e-3;
    pub const ERF_SMALL_P2: f32 = -2.676_673_42e-2;
    pub const ERF_SMALL_P3: f32 = 1.128_180_25e-1;
    pub const ERF_SMALL_P4: f32 = -3.761_248_59e-1;
    pub const ERF_SMALL_P5_MINUS_ONE: f32 = 1.283_791_51e-1;
    pub const ERF_BIG_P0: f32 = 1.729_489_30e-5;
    pub const ERF_BIG_P1: f32 = -3.832_086_80e-4;
    pub const ERF_BIG_P2: f32 = 3.883_934_35e-3;
    pub const ERF_BIG_P3: f32 = -2.425_456_05e-2;
    pub const ERF_BIG_P4: f32 = 1.067_778_47e-1;
    pub const ERF_BIG_P5: f32 = 6.348_466_87e-1;
    pub const ERF_BIG_P6_MINUS_ONE: f32 = 1.287_175_12e-1;
    pub const ERF_EXP_LOWER: f32 = -88.376_26;
    pub const ERF_EXP_LOG2_RECIPROCAL: f32 = 1.442_695;
    pub const ERF_EXP_LOG2_HI: f32 = -6.931_457_52e-1;
    pub const ERF_EXP_LOG2_LO: f32 = -1.428_606_77e-6;
    pub const ERF_EXP_P0: f32 = 1.383_198_08e-3;
    pub const ERF_EXP_P1: f32 = 8.375_500_33e-3;
    pub const ERF_EXP_P2: f32 = 4.166_895_15e-2;
    pub const ERF_EXP_P3: f32 = 1.666_644_66e-1;
    pub const ERF_EXP_P4: f32 = 4.999_998_51e-1;
    pub const ERF_EXP_P5: f32 = 1.0;
    pub const ERF_EXP_P6: f32 = 1.0;
    pub const ERF_EXP_C: f32 = 1.258_291_20e+7;
}

#[inline(always)]
fn clamp_keep_nan(v: f32, lo: f32, hi: f32) -> f32 {
    let v = if v < lo { lo } else { v };
    if v > hi { hi } else { v }
}

/// MLAS `MlasComputeExpVector` on one lane.
#[inline(always)]
fn exp_one(x: f32) -> f32 {
    use mlas::*;
    let mut v = clamp_keep_nan(x, EXP_LOWER, EXP_UPPER);
    let biased = v.mul_add(EXP_LOG2_RECIPROCAL, EXP_BIAS);
    let m = biased - EXP_BIAS;
    v = m.mul_add(EXP_LOG2_HIGH, v);
    v = m.mul_add(EXP_LOG2_LOW, v);
    let overflow = (biased.to_bits() as i32).wrapping_shl(23);
    let normal = overflow.clamp(EXP_MIN_EXPONENT, EXP_MAX_EXPONENT);
    let overflow = overflow.wrapping_sub(normal).wrapping_add(EXP_MAX_EXPONENT);
    let normal = normal.wrapping_add(EXP_MAX_EXPONENT);
    let mut p = EXP_P0;
    p = p.mul_add(v, EXP_P1);
    p = p.mul_add(v, EXP_P2);
    p = p.mul_add(v, EXP_P3);
    p = p.mul_add(v, EXP_P4);
    p = p.mul_add(v, EXP_P56);
    let overflow_scale = f32::from_bits(overflow as u32);
    v *= overflow_scale;
    p = p.mul_add(v, overflow_scale);
    p * f32::from_bits(normal as u32)
}

/// Eight lanes of the existing MLAS polynomial, shared by dense attention softmax.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn exp8(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use mlas::*;
    use std::arch::x86_64::*;
    let mut v = _mm256_min_ps(
        _mm256_max_ps(x, _mm256_set1_ps(EXP_LOWER)),
        _mm256_set1_ps(EXP_UPPER),
    );
    let biased = _mm256_fmadd_ps(
        v,
        _mm256_set1_ps(EXP_LOG2_RECIPROCAL),
        _mm256_set1_ps(EXP_BIAS),
    );
    let m = _mm256_sub_ps(biased, _mm256_set1_ps(EXP_BIAS));
    v = _mm256_fmadd_ps(m, _mm256_set1_ps(EXP_LOG2_HIGH), v);
    v = _mm256_fmadd_ps(m, _mm256_set1_ps(EXP_LOG2_LOW), v);
    let overflow = _mm256_slli_epi32::<23>(_mm256_castps_si256(biased));
    let normal = _mm256_min_epi32(
        _mm256_max_epi32(overflow, _mm256_set1_epi32(EXP_MIN_EXPONENT)),
        _mm256_set1_epi32(EXP_MAX_EXPONENT),
    );
    let overflow = _mm256_add_epi32(
        _mm256_sub_epi32(overflow, normal),
        _mm256_set1_epi32(EXP_MAX_EXPONENT),
    );
    let normal = _mm256_add_epi32(normal, _mm256_set1_epi32(EXP_MAX_EXPONENT));
    let mut p = _mm256_set1_ps(EXP_P0);
    for coefficient in [EXP_P1, EXP_P2, EXP_P3, EXP_P4, EXP_P56] {
        p = _mm256_fmadd_ps(p, v, _mm256_set1_ps(coefficient));
    }
    let scale = _mm256_castsi256_ps(overflow);
    v = _mm256_mul_ps(v, scale);
    p = _mm256_fmadd_ps(p, v, scale);
    _mm256_mul_ps(p, _mm256_castsi256_ps(normal))
}

/// MLAS `MlasTanhKernel` on one lane.
#[inline(always)]
fn tanh_one(x: f32) -> f32 {
    use mlas::*;
    let v = clamp_keep_nan(x, TANH_LOWER, TANH_UPPER);
    let s = v * v;
    let mut p = s.mul_add(TANH_A13, TANH_A11);
    p = p.mul_add(s, TANH_A9);
    p = p.mul_add(s, TANH_A7);
    p = p.mul_add(s, TANH_A5);
    p = p.mul_add(s, TANH_A3);
    p = p.mul_add(s, TANH_A1);
    p *= v;
    let mut q = s.mul_add(TANH_B6, TANH_B4);
    q = q.mul_add(s, TANH_B2);
    q = q.mul_add(s, TANH_B0);
    p / q
}

/// MLAS `MlasLogisticKernel` on one lane.
#[inline(always)]
fn logistic_one(x: f32) -> f32 {
    use mlas::*;
    let v = clamp_keep_nan(x, LOGISTIC_LOWER, LOGISTIC_UPPER);
    let s = v * v;
    let mut p = s.mul_add(LOGISTIC_A9, LOGISTIC_A7);
    p = p.mul_add(s, LOGISTIC_A5);
    p = p.mul_add(s, LOGISTIC_A3);
    p = p.mul_add(s, LOGISTIC_A1);
    p *= v;
    let mut q = s.mul_add(LOGISTIC_B10, LOGISTIC_B8);
    q = q.mul_add(s, LOGISTIC_B6);
    q = q.mul_add(s, LOGISTIC_B4);
    q = q.mul_add(s, LOGISTIC_B2);
    q = q.mul_add(s, LOGISTIC_B0);
    clamp_keep_nan(p / q + 0.5, 0.0, 1.0)
}

/// MLAS `MlasErfKernel` on one lane: both halves computed, the split chosen by select.
#[inline(always)]
fn erf_one(value: f32) -> f32 {
    use mlas::*;
    let a = value.abs();
    let abs = if a > ERF_UPPER_ABS { ERF_UPPER_ABS } else { a };
    let square = abs * abs;
    let mut small = ERF_SMALL_P0;
    small = small.mul_add(square, ERF_SMALL_P1);
    small = small.mul_add(square, ERF_SMALL_P2);
    small = small.mul_add(square, ERF_SMALL_P3);
    small = small.mul_add(square, ERF_SMALL_P4);
    small = small.mul_add(square, ERF_SMALL_P5_MINUS_ONE);
    small = small.mul_add(abs, abs);
    let mut r = ERF_BIG_P0;
    r = r.mul_add(abs, ERF_BIG_P1);
    r = r.mul_add(abs, ERF_BIG_P2);
    r = r.mul_add(abs, ERF_BIG_P3);
    r = r.mul_add(abs, ERF_BIG_P4);
    r = r.mul_add(abs, ERF_BIG_P5);
    r = r.mul_add(abs, ERF_BIG_P6_MINUS_ONE);
    r = r.mul_add(abs, abs);
    let neg = -r;
    let r = if neg < ERF_EXP_LOWER {
        ERF_EXP_LOWER
    } else {
        neg
    };
    let k = r.mul_add(ERF_EXP_LOG2_RECIPROCAL, ERF_EXP_C) - ERF_EXP_C;
    let mut fx = k.mul_add(ERF_EXP_LOG2_HI, r);
    fx = k.mul_add(ERF_EXP_LOG2_LO, fx);
    let mut y = ERF_EXP_P0;
    y = y.mul_add(fx, ERF_EXP_P1);
    y = y.mul_add(fx, ERF_EXP_P2);
    y = y.mul_add(fx, ERF_EXP_P3);
    y = y.mul_add(fx, ERF_EXP_P4);
    y = y.mul_add(fx, ERF_EXP_P5);
    y = y.mul_add(fx, ERF_EXP_P6);
    y *= f32::from_bits(((k as i32).wrapping_add(127) << 23) as u32);
    let big = 1.0 - y;
    let magnitude = if abs > ERF_SPLIT { big } else { small };
    // Copy the input's sign bit, as MLAS does, so erf(-0) = -0.
    f32::from_bits(magnitude.to_bits() | (value.to_bits() & 0x8000_0000))
}

#[inline(always)]
fn map_slice(x: &[f32], y: &mut [f32], f: impl Fn(f32) -> f32) {
    for (o, &v) in y.iter_mut().zip(x.iter()) {
        *o = f(v);
    }
}

macro_rules! avx2_slice {
    ($avx2:ident, $one:ident) => {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx2(x: &[f32], y: &mut [f32]) {
            map_slice(x, y, $one);
        }
    };
}

avx2_slice!(exp_slice_avx2, exp_one);
avx2_slice!(tanh_slice_avx2, tanh_one);
avx2_slice!(logistic_slice_avx2, logistic_one);
avx2_slice!(erf_slice_avx2, erf_one);

/// Validates the buffers and applies one activation: the AVX2/FMA build of the loop where the CPU
/// has it, the same scalar formula otherwise. Returns 0, -1 for invalid arguments, -3 on panic.
///
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
unsafe fn apply_unary(
    x: *const f32,
    y: *mut f32,
    n: usize,
    #[allow(unused_variables)] avx2: unsafe fn(&[f32], &mut [f32]),
    one: fn(f32) -> f32,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if n == 0 {
            return 0;
        }
        if x.is_null() || y.is_null() {
            return -1;
        }
        let xs = unsafe { std::slice::from_raw_parts(x, n) };
        let ys = unsafe { std::slice::from_raw_parts_mut(y, n) };
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                unsafe { avx2(xs, ys) };
                return 0;
            }
        }
        map_slice(xs, ys, one);
        0
    }))
    .unwrap_or(-3)
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn no_avx2(_: &[f32], _: &mut [f32]) {}

macro_rules! avx2_or_none {
    ($avx2:ident) => {{
        #[cfg(target_arch = "x86_64")]
        let f: unsafe fn(&[f32], &mut [f32]) = $avx2;
        #[cfg(not(target_arch = "x86_64"))]
        let f: unsafe fn(&[f32], &mut [f32]) = no_avx2;
        f
    }};
}

/// `y[i] = exp(x[i])` for `n` floats with onnxruntime's MLAS approximation.
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_exp_f32(x: *const f32, y: *mut f32, n: usize) -> i32 {
    unsafe { apply_unary(x, y, n, avx2_or_none!(exp_slice_avx2), exp_one) }
}

/// `y[i] = tanh(x[i])` for `n` floats with onnxruntime's MLAS approximation.
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_tanh_f32(x: *const f32, y: *mut f32, n: usize) -> i32 {
    unsafe { apply_unary(x, y, n, avx2_or_none!(tanh_slice_avx2), tanh_one) }
}

/// `y[i] = 1 / (1 + exp(-x[i]))` for `n` floats with onnxruntime's MLAS approximation.
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_logistic_f32(x: *const f32, y: *mut f32, n: usize) -> i32 {
    unsafe { apply_unary(x, y, n, avx2_or_none!(logistic_slice_avx2), logistic_one) }
}

/// `y[i] = erf(x[i])` for `n` floats with onnxruntime's MLAS approximation.
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_erf_f32(x: *const f32, y: *mut f32, n: usize) -> i32 {
    unsafe { apply_unary(x, y, n, avx2_or_none!(erf_slice_avx2), erf_one) }
}

#[cfg(test)]
mod erf_tests {
    use super::*;

    #[test]
    fn erf_matches_reference_values() {
        // Reference values from Python's math.erf, rounded to float.
        let cases = [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (-1.5, -0.966_105_16),
            (3.0, 0.999_977_9),
        ];
        let x: Vec<f32> = cases.iter().map(|c| c.0).collect();
        let mut y = vec![0f32; x.len()];
        assert_eq!(
            unsafe { jmodels_onnx_erf_f32(x.as_ptr(), y.as_mut_ptr(), x.len()) },
            0
        );
        assert_eq!(erf_one(-0.0).to_bits(), (-0.0f32).to_bits());
        for (got, (_, want)) in y.iter().zip(cases.iter()) {
            assert!((got - want).abs() <= 1e-6, "erf: got {got}, want {want}");
        }
    }
}

#[cfg(test)]
mod activation_tests {
    use super::*;

    fn sweep(lo: f32, hi: f32) -> Vec<f32> {
        let n = 100_003;
        (0..n)
            .map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
            .collect()
    }

    fn worst(
        xs: &[f32],
        f: unsafe extern "C" fn(*const f32, *mut f32, usize) -> i32,
        r: impl Fn(f64) -> f64,
        relative: bool,
    ) -> f64 {
        let mut y = vec![0f32; xs.len()];
        assert_eq!(unsafe { f(xs.as_ptr(), y.as_mut_ptr(), xs.len()) }, 0);
        xs.iter()
            .zip(y.iter())
            .map(|(&x, &got)| {
                let want = r(x as f64);
                let err = (got as f64 - want).abs();
                if relative {
                    err / want.abs().max(f64::MIN_POSITIVE)
                } else {
                    err
                }
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn exp_tanh_logistic_track_double_precision() {
        assert!(worst(&sweep(-87.0, 88.0), jmodels_onnx_exp_f32, f64::exp, true) < 5e-7);
        assert!(worst(&sweep(-12.0, 12.0), jmodels_onnx_tanh_f32, f64::tanh, false) < 1e-6);
        assert!(
            worst(
                &sweep(-25.0, 25.0),
                jmodels_onnx_logistic_f32,
                |x| 1.0 / (1.0 + (-x).exp()),
                false
            ) < 1e-6
        );
    }

    #[test]
    fn vector_path_equals_scalar_path_bit_for_bit() {
        let xs = sweep(-30.0, 30.0);
        for (f, one) in [
            (
                jmodels_onnx_exp_f32 as unsafe extern "C" fn(*const f32, *mut f32, usize) -> i32,
                exp_one as fn(f32) -> f32,
            ),
            (jmodels_onnx_tanh_f32, tanh_one),
            (jmodels_onnx_logistic_f32, logistic_one),
            (jmodels_onnx_erf_f32, erf_one),
        ] {
            let mut y = vec![0f32; xs.len()];
            assert_eq!(unsafe { f(xs.as_ptr(), y.as_mut_ptr(), xs.len()) }, 0);
            for (&x, &got) in xs.iter().zip(y.iter()) {
                assert_eq!(got.to_bits(), one(x).to_bits(), "x={x}");
            }
        }
    }

    #[test]
    fn nan_stays_nan() {
        for one in [exp_one as fn(f32) -> f32, tanh_one, logistic_one, erf_one] {
            assert!(one(f32::NAN).is_nan());
        }
    }
}

#[cfg(test)]
mod sgemm_tests {
    use super::*;

    #[test]
    fn sgemm_matches_naive_reference_on_ragged_shapes_and_leading_dimensions() {
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for &(m, k, n) in &[
            (1, 1, 1),
            (7, 3, 5),
            (6, 256, 16),
            (13, 300, 33),
            (602, 600, 433),
            (5, 513, 17),
            (39, 256, 256),
        ] {
            let (lda, ldb, ldc) = (k + 3, n + 2, n + 5);
            let a: Vec<f32> = (0..m * lda).map(|_| next()).collect();
            let b: Vec<f32> = (0..k * ldb).map(|_| next()).collect();
            let c0: Vec<f32> = (0..m * ldc).map(|_| next()).collect();
            let mut c = c0.clone();
            assert_eq!(
                unsafe {
                    jmodels_onnx_sgemm(
                        m,
                        k,
                        n,
                        a.as_ptr(),
                        lda,
                        b.as_ptr(),
                        ldb,
                        c.as_mut_ptr(),
                        ldc,
                    )
                },
                0
            );
            for i in 0..m {
                for j in 0..ldc {
                    let want = if j < n {
                        c0[i * ldc + j] as f64
                            + (0..k)
                                .map(|p| a[i * lda + p] as f64 * b[p * ldb + j] as f64)
                                .sum::<f64>()
                    } else {
                        c0[i * ldc + j] as f64 // columns past n must be untouched
                    };
                    let tol = 1e-4 * (1.0 + (k as f64).sqrt());
                    assert!(
                        (c[i * ldc + j] as f64 - want).abs() <= tol,
                        "m={m} k={k} n={n} at ({i},{j}): {} vs {want}",
                        c[i * ldc + j]
                    );
                }
            }
        }
    }
}
