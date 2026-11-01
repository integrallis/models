// SPDX-License-Identifier: Apache-2.0
//! Exact Q5_1 SIMD row reuse. Scheduling and immutable partition counts stay unchanged.
use super::Q5_1_BLOCK_BYTES;
#[cfg(target_arch = "aarch64")]
use super::f16_to_f32;
use std::cell::RefCell;
thread_local! { static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) }; }

#[inline]
fn fold(lanes: &[f32; 8]) -> f32 {
    ((lanes[4] + lanes[0]) + (lanes[6] + lanes[2]))
        + ((lanes[5] + lanes[1]) + (lanes[7] + lanes[3]))
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn decode_avx2(weights: &[u8], row: usize, cols: usize, dst: &mut [f32]) {
    for block in 0..cols / 32 {
        let base = (row * (cols / 32) + block) * Q5_1_BLOCK_BYTES;
        let d = _mm256_set1_ps(super::f16_to_f32_f16c(u16::from_le_bytes([
            weights[base],
            weights[base + 1],
        ])));
        let m = _mm256_set1_ps(super::f16_to_f32_f16c(u16::from_le_bytes([
            weights[base + 2],
            weights[base + 3],
        ])));
        // SAFETY: complete Q51 block and destination row were validated by the caller.
        unsafe {
            let q = super::unpack_q5_1_avx2(
                weights.as_ptr().add(base + 4),
                weights.as_ptr().add(base + 8),
            );
            let low = _mm256_extracti128_si256(q, 0);
            let high = _mm256_extracti128_si256(q, 1);
            let groups = [
                _mm256_cvtepu8_epi32(low),
                _mm256_cvtepu8_epi32(_mm_srli_si128(low, 8)),
                _mm256_cvtepu8_epi32(high),
                _mm256_cvtepu8_epi32(_mm_srli_si128(high, 8)),
            ];
            for (g, q) in groups.iter().enumerate() {
                // Unfused dequantization preserves the Java q*d+m contract.
                let v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(*q), d), m);
                _mm256_storeu_ps(dst.as_mut_ptr().add(block * 32 + g * 8), v);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn compute_avx2(
    weights: &[u8],
    input: &[f32],
    output: *mut f32,
    batch: usize,
    rows: usize,
    cols: usize,
    start: usize,
    end: usize,
) {
    ROW.with(|cell| {
        let mut buffer = cell.borrow_mut();
        buffer.resize(cols, 0.0);
        for row in start..end {
            // SAFETY: validated full row and detected features.
            unsafe {
                decode_avx2(weights, row, cols, &mut buffer);
            }
            let mut b = 0;
            while b + 1 < batch {
                let mut first = [_mm256_setzero_ps(); 4];
                let mut second = [_mm256_setzero_ps(); 4];
                for c in (0..cols).step_by(32) {
                    for g in 0..4 {
                        // SAFETY: both queries and all full blocks are within validated slices.
                        unsafe {
                            let w = _mm256_loadu_ps(buffer.as_ptr().add(c + g * 8));
                            first[g] = _mm256_fmadd_ps(
                                _mm256_loadu_ps(input.as_ptr().add(b * cols + c + g * 8)),
                                w,
                                first[g],
                            );
                            second[g] = _mm256_fmadd_ps(
                                _mm256_loadu_ps(input.as_ptr().add((b + 1) * cols + c + g * 8)),
                                w,
                                second[g],
                            );
                        }
                    }
                }
                for (query, acc) in [(b, first), (b + 1, second)] {
                    let sum =
                        _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
                    let mut lanes = [0.0; 8];
                    // SAFETY: this worker owns the output row and the local array has eight lanes.
                    unsafe {
                        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
                        *output.add(query * rows + row) = fold(&lanes);
                    }
                }
                b += 2;
            }
            if b < batch {
                let mut acc = [_mm256_setzero_ps(); 4];
                for c in (0..cols).step_by(32) {
                    for (g, a) in acc.iter_mut().enumerate() {
                        // SAFETY: the final query and full blocks fit validated slices.
                        unsafe {
                            *a = _mm256_fmadd_ps(
                                _mm256_loadu_ps(input.as_ptr().add(b * cols + c + g * 8)),
                                _mm256_loadu_ps(buffer.as_ptr().add(c + g * 8)),
                                *a,
                            );
                        }
                    }
                }
                let sum =
                    _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
                let mut lanes = [0.0; 8];
                // SAFETY: local array and disjoint output row are valid.
                unsafe {
                    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
                    *output.add(b * rows + row) = fold(&lanes);
                }
            }
        }
    });
}

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_neon(weights: &[u8], row: usize, cols: usize, dst: &mut [f32]) {
    for block in 0..cols / 32 {
        let base = (row * (cols / 32) + block) * Q5_1_BLOCK_BYTES;
        let d = vdupq_n_f32(f16_to_f32(u16::from_le_bytes([
            weights[base],
            weights[base + 1],
        ])));
        let m = vdupq_n_f32(f16_to_f32(u16::from_le_bytes([
            weights[base + 2],
            weights[base + 3],
        ])));
        let high = u32::from_le_bytes(weights[base + 4..base + 8].try_into().unwrap());
        // SAFETY: constant arrays contain four lanes; complete block/output validated.
        unsafe {
            let byteshift = vld1q_s32([0, -8, -16, -24].as_ptr());
            let bitshift = vld1q_s32([0, -1, -2, -3].as_ptr());
            for n in (0..16).step_by(4) {
                let packed =
                    u32::from_le_bytes(weights[base + 8 + n..base + 12 + n].try_into().unwrap());
                let p = vshlq_u32(vdupq_n_u32(packed), byteshift);
                let bits = vshlq_u32(vdupq_n_u32(high >> n), bitshift);
                let low = vorrq_u32(
                    vandq_u32(p, vdupq_n_u32(15)),
                    vshlq_n_u32::<4>(vandq_u32(bits, vdupq_n_u32(1))),
                );
                let upper = vorrq_u32(
                    vandq_u32(vshrq_n_u32::<4>(p), vdupq_n_u32(15)),
                    vshlq_n_u32::<4>(vandq_u32(vshrq_n_u32::<16>(bits), vdupq_n_u32(1))),
                );
                vst1q_f32(
                    dst.as_mut_ptr().add(block * 32 + n),
                    vaddq_f32(vmulq_f32(vcvtq_f32_u32(low), d), m),
                );
                vst1q_f32(
                    dst.as_mut_ptr().add(block * 32 + 16 + n),
                    vaddq_f32(vmulq_f32(vcvtq_f32_u32(upper), d), m),
                );
            }
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn compute_neon(
    weights: &[u8],
    input: &[f32],
    output: *mut f32,
    batch: usize,
    rows: usize,
    cols: usize,
    start: usize,
    end: usize,
) {
    ROW.with(|cell| {
        let mut buffer = cell.borrow_mut();
        buffer.resize(cols, 0.0);
        for row in start..end {
            // SAFETY: detected NEON and validated complete row.
            unsafe {
                decode_neon(weights, row, cols, &mut buffer);
            }
            for b in 0..batch {
                let mut acc = [vdupq_n_f32(0.0); 8];
                for c in (0..cols).step_by(32) {
                    for (g, a) in acc.iter_mut().enumerate() {
                        // SAFETY: full 32-value blocks within validated inputs and row buffer.
                        unsafe {
                            *a = vfmaq_f32(
                                *a,
                                vld1q_f32(input.as_ptr().add(b * cols + c + g * 4)),
                                vld1q_f32(buffer.as_ptr().add(c + g * 4)),
                            );
                        }
                    }
                }
                let low = vaddq_f32(vaddq_f32(acc[0], acc[2]), vaddq_f32(acc[4], acc[6]));
                let high = vaddq_f32(vaddq_f32(acc[1], acc[3]), vaddq_f32(acc[5], acc[7]));
                let mut lanes = [0.0; 8];
                // SAFETY: two four-lane stores fit; each worker owns disjoint output rows.
                unsafe {
                    vst1q_f32(lanes.as_mut_ptr(), low);
                    vst1q_f32(lanes.as_mut_ptr().add(4), high);
                    *output.add(b * rows + row) = fold(&lanes);
                }
            }
        }
    });
}
