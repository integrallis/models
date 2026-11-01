//! Float SwiGLU arithmetic adapted from Arm AdvSIMD expf.
//! Copyright(c)2019-2026 Arm Limited.
//! SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception.
//! Explicit fused operations match the corresponding Java float polynomial.
pub(super) fn supported() -> bool {
    super::supports_f32_fma_simd()
}

/// Caller validates shape, features, numeric domain and live buffers without forward output overlap.
pub(super) unsafe fn apply(out: *mut f32, gate: *const f32, up: *const f32, len: usize) {
    // SAFETY: validated by the sole exported entry point.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        x86(out, gate, up, len);
        #[cfg(target_arch = "aarch64")]
        super::sleef_activations::swi(out, gate, up, len);
    }
}
#[cfg(target_arch = "x86_64")]
const INV: f32 = f32::from_bits(0x3fb8aa3b);
#[cfg(target_arch = "x86_64")]
const HI: f32 = f32::from_bits(0x3f317200);
#[cfg(target_arch = "x86_64")]
const LO: f32 = f32::from_bits(0x35bfbe8e);
#[cfg(target_arch = "x86_64")]
const C0: f32 = f32::from_bits(0x3c072010);
#[cfg(target_arch = "x86_64")]
const C1: f32 = f32::from_bits(0x3d2b9f17);
#[cfg(target_arch = "x86_64")]
const C2: f32 = f32::from_bits(0x3e2aaf33);
#[cfg(target_arch = "x86_64")]
const C3: f32 = f32::from_bits(0x3efffedb);
#[cfg(target_arch = "x86_64")]
const C4: f32 = f32::from_bits(0x3f7ffff6);
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86(out: *mut f32, gate: *const f32, up: *const f32, len: usize) {
    // SAFETY: caller validates live buffers, lengths and CPU support.
    unsafe {
        use std::arch::x86_64::*;
        let c = |x| _mm256_set1_ps(x);
        let mut i = 0;
        while i + 8 <= len {
            let value = _mm256_loadu_ps(gate.add(i));
            let u = _mm256_loadu_ps(up.add(i));
            let x = _mm256_xor_ps(value, c(-0.0));
            let n = _mm256_sub_ps(
                _mm256_add_ps(_mm256_mul_ps(x, c(INV)), c(12582912.0)),
                c(12582912.0),
            );
            let neg = _mm256_xor_ps(n, c(-0.0));
            let r = _mm256_fmadd_ps(neg, c(HI), x);
            let r = _mm256_fmadd_ps(neg, c(LO), r);
            let k = _mm256_cvttps_epi32(n);
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_slli_epi32::<23>(k),
                _mm256_set1_epi32(0x3f800000),
            ));
            let r2 = _mm256_mul_ps(r, r);
            let p = _mm256_fmadd_ps(r, c(C0), c(C1));
            let q = _mm256_fmadd_ps(r, c(C2), c(C3));
            let q = _mm256_fmadd_ps(p, r2, q);
            let p = _mm256_mul_ps(c(C4), r);
            let poly = _mm256_fmadd_ps(q, r2, p);
            let e = _mm256_fmadd_ps(scale, poly, scale);
            let y = _mm256_mul_ps(_mm256_div_ps(value, _mm256_add_ps(c(1.0), e)), u);
            _mm256_storeu_ps(out.add(i), y);
            i += 8;
        }
        if i < len {
            let mask = _mm256_cmpgt_epi32(
                _mm256_set1_epi32((len - i) as i32),
                _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            );
            let value = _mm256_maskload_ps(gate.add(i), mask);
            let u = _mm256_maskload_ps(up.add(i), mask);
            let x = _mm256_xor_ps(value, c(-0.0));
            let n = _mm256_sub_ps(
                _mm256_add_ps(_mm256_mul_ps(x, c(INV)), c(12582912.0)),
                c(12582912.0),
            );
            let neg = _mm256_xor_ps(n, c(-0.0));
            let r = _mm256_fmadd_ps(neg, c(HI), x);
            let r = _mm256_fmadd_ps(neg, c(LO), r);
            let k = _mm256_cvttps_epi32(n);
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_slli_epi32::<23>(k),
                _mm256_set1_epi32(0x3f800000),
            ));
            let r2 = _mm256_mul_ps(r, r);
            let p = _mm256_fmadd_ps(r, c(C0), c(C1));
            let q = _mm256_fmadd_ps(r, c(C2), c(C3));
            let q = _mm256_fmadd_ps(p, r2, q);
            let p = _mm256_mul_ps(c(C4), r);
            let poly = _mm256_fmadd_ps(q, r2, p);
            let e = _mm256_fmadd_ps(scale, poly, scale);
            let y = _mm256_mul_ps(_mm256_div_ps(value, _mm256_add_ps(c(1.0), e)), u);
            _mm256_maskstore_ps(out.add(i), mask, y);
        }
    }
}
