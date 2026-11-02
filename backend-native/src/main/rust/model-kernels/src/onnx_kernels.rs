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
    1
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

/// `y[i] = erf(x[i])` for `n` floats, each evaluated in double precision by `libm` (MIT, a port of
/// musl's, itself FreeBSD's fdlibm `s_erf.c`) and rounded once. Returns 0, -1 for invalid
/// arguments, -3 on panic.
/// # Safety
/// `x` and `y` must be valid for `n` floats; they may be the same buffer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_onnx_erf_f32(x: *const f32, y: *mut f32, n: usize) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if n == 0 {
            return 0;
        }
        if x.is_null() || y.is_null() {
            return -1;
        }
        let xs = unsafe { std::slice::from_raw_parts(x, n) };
        let ys = unsafe { std::slice::from_raw_parts_mut(y, n) };
        for (o, &v) in ys.iter_mut().zip(xs.iter()) {
            *o = libm::erf(v as f64) as f32;
        }
        0
    }))
    .unwrap_or(-3)
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
        for (got, (_, want)) in y.iter().zip(cases.iter()) {
            assert!((got - want).abs() <= 1e-6, "erf: got {got}, want {want}");
        }
    }
}
