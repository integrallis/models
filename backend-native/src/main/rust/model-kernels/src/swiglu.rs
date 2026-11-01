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
        arm(out, gate, up, len);
    }
}
const INV: f32 = f32::from_bits(0x3fb8aa3b);
const HI: f32 = f32::from_bits(0x3f317200);
const LO: f32 = f32::from_bits(0x35bfbe8e);
const C0: f32 = f32::from_bits(0x3c072010);
const C1: f32 = f32::from_bits(0x3d2b9f17);
const C2: f32 = f32::from_bits(0x3e2aaf33);
const C3: f32 = f32::from_bits(0x3efffedb);
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
#[cfg(target_arch = "aarch64")]
unsafe fn arm(out: *mut f32, gate: *const f32, up: *const f32, len: usize) {
    // SAFETY: caller validates live buffers, lengths and CPU support.
    unsafe {
        use std::arch::aarch64::*;
        let c = |x| vdupq_n_f32(x);
        for i in (0..len).step_by(4) {
            let value = vld1q_f32(gate.add(i));
            let u = vld1q_f32(up.add(i));
            let x = vnegq_f32(value);
            let n = vsubq_f32(
                vaddq_f32(vmulq_f32(x, c(INV)), c(12582912.0)),
                c(12582912.0),
            );
            let neg = vnegq_f32(n);
            let r = vfmaq_f32(x, neg, c(HI));
            let r = vfmaq_f32(r, neg, c(LO));
            let scale = vreinterpretq_f32_s32(vaddq_s32(
                vshlq_n_s32::<23>(vcvtq_s32_f32(n)),
                vdupq_n_s32(0x3f800000),
            ));
            let r2 = vmulq_f32(r, r);
            let p = vfmaq_f32(c(C1), r, c(C0));
            let q = vfmaq_f32(c(C3), r, c(C2));
            let q = vfmaq_f32(q, p, r2);
            let p = vmulq_f32(c(C4), r);
            let poly = vfmaq_f32(p, q, r2);
            let e = vfmaq_f32(scale, scale, poly);
            vst1q_f32(
                out.add(i),
                vmulq_f32(vdivq_f32(value, vaddq_f32(c(1.0), e)), u),
            );
        }
    }
}

#[cfg(test)]
mod contract_tests {
    use crate::{
        STATUS_BUFFER_TOO_SMALL, STATUS_INVALID_SHAPE, STATUS_NULL_POINTER, STATUS_OK,
        jmodels_swiglu_f32,
    };

    #[test]
    fn checked_abi_rejects_invalid_inputs_without_writes() {
        let mut output = [42.0f32; 12];
        let mut gate = [0.5f32; 12];
        let up = [0.25f32; 12];
        // SAFETY: all non-null arguments are live and aligned; null is rejected before access.
        assert_eq!(
            unsafe {
                jmodels_swiglu_f32(
                    output.as_mut_ptr(),
                    12,
                    0,
                    std::ptr::null(),
                    12,
                    0,
                    up.as_ptr(),
                    12,
                    0,
                    8,
                )
            },
            STATUS_NULL_POINTER
        );
        if !super::supported() {
            return;
        }
        for length in [0, 3] {
            // SAFETY: live arrays; invalid length must return before any access or write.
            assert_eq!(
                unsafe {
                    jmodels_swiglu_f32(
                        output.as_mut_ptr(),
                        12,
                        0,
                        gate.as_ptr(),
                        12,
                        0,
                        up.as_ptr(),
                        12,
                        0,
                        length,
                    )
                },
                STATUS_INVALID_SHAPE
            );
        }
        // SAFETY: advertised capacity is shorter than the requested valid array region.
        assert_eq!(
            unsafe {
                jmodels_swiglu_f32(
                    output.as_mut_ptr(),
                    7,
                    0,
                    gate.as_ptr(),
                    12,
                    0,
                    up.as_ptr(),
                    12,
                    0,
                    8,
                )
            },
            STATUS_BUFFER_TOO_SMALL
        );
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            80.00001,
            -80.00001,
        ] {
            gate[9] = value;
            // SAFETY: live arrays; exceptional final gate must be found before output writes.
            assert_eq!(
                unsafe {
                    jmodels_swiglu_f32(
                        output.as_mut_ptr(),
                        12,
                        2,
                        gate.as_ptr(),
                        12,
                        2,
                        up.as_ptr(),
                        12,
                        2,
                        8,
                    )
                },
                STATUS_INVALID_SHAPE
            );
            assert_eq!(output, [42.0; 12]);
        }
    }

    #[test]
    fn checked_abi_accepts_in_place_and_rejects_forward_overlap() {
        if !super::supported() {
            return;
        }
        let mut shared = [0.0f32; 16];
        shared[2] = -0.0;
        let up = [1.0f32; 16];
        let original = shared.map(f32::to_bits);
        let ptr = shared.as_mut_ptr();
        // SAFETY: the exact alias is supported, and all arrays have advertised capacities.
        assert_eq!(
            unsafe { jmodels_swiglu_f32(ptr, 16, 0, ptr, 16, 0, up.as_ptr(), 16, 0, 12) },
            STATUS_OK
        );
        assert_eq!(shared.map(f32::to_bits), original);
        // SAFETY: forward overlap is rejected before dereferencing writable elements.
        assert_eq!(
            unsafe { jmodels_swiglu_f32(ptr, 16, 2, ptr, 16, 0, up.as_ptr(), 16, 0, 12) },
            STATUS_INVALID_SHAPE
        );
        assert_eq!(shared.map(f32::to_bits), original);
        // SAFETY: backward overlap is safe under the forward native block traversal.
        assert_eq!(
            unsafe { jmodels_swiglu_f32(ptr, 16, 0, ptr, 16, 2, up.as_ptr(), 16, 0, 12) },
            STATUS_OK
        );
        assert_eq!(shared[0].to_bits(), (-0.0f32).to_bits());
    }
}
