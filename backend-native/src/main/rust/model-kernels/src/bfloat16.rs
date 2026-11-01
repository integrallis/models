// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Fixed-tree BF16 projection for the measured 48 by 5120 mapped matrix shape.
//! Even/odd accumulation and half-tree reduction preserve the selected Java lane contract.

pub(super) fn supported() -> bool {
    super::supports_f32_fma_simd()
}

/// Caller validates CPU features, disjoint live buffers and the fixed shape before entry.
pub(super) unsafe fn multiply(
    weight: *const u16,
    input: *const f32,
    output: *mut f32,
    batch: usize,
    lanes: u32,
    fused: bool,
) {
    // SAFETY: the sole exported caller enforces this module's shape and feature contract.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        match (lanes, fused) {
            (4, true) => x86_4::<true>(weight, input, output, batch),
            (4, false) => x86_4::<false>(weight, input, output, batch),
            (8, true) => x86_8::<true>(weight, input, output, batch),
            (8, false) => x86_8::<false>(weight, input, output, batch),
            _ => unreachable!("validated lane count"),
        }
        #[cfg(target_arch = "aarch64")]
        match (lanes, fused) {
            (4, true) => arm::<1, true>(weight, input, output, batch),
            (4, false) => arm::<1, false>(weight, input, output, batch),
            (8, true) => arm::<2, true>(weight, input, output, batch),
            (8, false) => arm::<2, false>(weight, input, output, batch),
            _ => unreachable!("validated lane count"),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86_4<const FUSED: bool>(
    weight: *const u16,
    input: *const f32,
    output: *mut f32,
    batch: usize,
) {
    // SAFETY: the exported entry validates shape, buffers, lane count and CPU support.
    unsafe {
        use std::arch::x86_64::*;
        for b in 0..batch {
            for row in (0..48).step_by(4) {
                let mut sum0 = _mm_setzero_ps();
                let mut sum1 = _mm_setzero_ps();
                let mut sum2 = _mm_setzero_ps();
                let mut sum3 = _mm_setzero_ps();
                for col in (0..5120).step_by(8) {
                    let first = _mm_loadu_ps(input.add(b * 5120 + col));
                    let second = _mm_loadu_ps(input.add(b * 5120 + col + 4));
                    let even = _mm_shuffle_ps::<0x88>(first, second);
                    let odd = _mm_shuffle_ps::<0xdd>(first, second);
                    let packed = _mm_loadu_si128(weight.add(row * 5120 + col) as *const __m128i);
                    let ew = _mm_castsi128_ps(_mm_slli_epi32::<16>(packed));
                    let ow = _mm_castsi128_ps(_mm_and_si128(
                        packed,
                        _mm_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum0 = _mm_fmadd_ps(ew, even, sum0);
                        sum0 = _mm_fmadd_ps(ow, odd, sum0);
                    } else {
                        sum0 = _mm_add_ps(_mm_mul_ps(ew, even), sum0);
                        sum0 = _mm_add_ps(_mm_mul_ps(ow, odd), sum0);
                    }
                    let packed =
                        _mm_loadu_si128(weight.add((row + 1) * 5120 + col) as *const __m128i);
                    let ew = _mm_castsi128_ps(_mm_slli_epi32::<16>(packed));
                    let ow = _mm_castsi128_ps(_mm_and_si128(
                        packed,
                        _mm_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum1 = _mm_fmadd_ps(ew, even, sum1);
                        sum1 = _mm_fmadd_ps(ow, odd, sum1);
                    } else {
                        sum1 = _mm_add_ps(_mm_mul_ps(ew, even), sum1);
                        sum1 = _mm_add_ps(_mm_mul_ps(ow, odd), sum1);
                    }
                    let packed =
                        _mm_loadu_si128(weight.add((row + 2) * 5120 + col) as *const __m128i);
                    let ew = _mm_castsi128_ps(_mm_slli_epi32::<16>(packed));
                    let ow = _mm_castsi128_ps(_mm_and_si128(
                        packed,
                        _mm_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum2 = _mm_fmadd_ps(ew, even, sum2);
                        sum2 = _mm_fmadd_ps(ow, odd, sum2);
                    } else {
                        sum2 = _mm_add_ps(_mm_mul_ps(ew, even), sum2);
                        sum2 = _mm_add_ps(_mm_mul_ps(ow, odd), sum2);
                    }
                    let packed =
                        _mm_loadu_si128(weight.add((row + 3) * 5120 + col) as *const __m128i);
                    let ew = _mm_castsi128_ps(_mm_slli_epi32::<16>(packed));
                    let ow = _mm_castsi128_ps(_mm_and_si128(
                        packed,
                        _mm_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum3 = _mm_fmadd_ps(ew, even, sum3);
                        sum3 = _mm_fmadd_ps(ow, odd, sum3);
                    } else {
                        sum3 = _mm_add_ps(_mm_mul_ps(ew, even), sum3);
                        sum3 = _mm_add_ps(_mm_mul_ps(ow, odd), sum3);
                    }
                }
                let mut v = [0.0f32; 4];
                _mm_storeu_ps(v.as_mut_ptr(), sum0);
                *output.add(b * 48 + row) = (v[0] + v[2]) + (v[1] + v[3]);
                let mut v = [0.0f32; 4];
                _mm_storeu_ps(v.as_mut_ptr(), sum1);
                *output.add(b * 48 + row + 1) = (v[0] + v[2]) + (v[1] + v[3]);
                let mut v = [0.0f32; 4];
                _mm_storeu_ps(v.as_mut_ptr(), sum2);
                *output.add(b * 48 + row + 2) = (v[0] + v[2]) + (v[1] + v[3]);
                let mut v = [0.0f32; 4];
                _mm_storeu_ps(v.as_mut_ptr(), sum3);
                *output.add(b * 48 + row + 3) = (v[0] + v[2]) + (v[1] + v[3]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86_8<const FUSED: bool>(
    weight: *const u16,
    input: *const f32,
    output: *mut f32,
    batch: usize,
) {
    // SAFETY: the exported entry validates shape, buffers, lane count and CPU support.
    unsafe {
        use std::arch::x86_64::*;
        for b in 0..batch {
            for row in (0..48).step_by(4) {
                let mut sum0 = _mm256_setzero_ps();
                let mut sum1 = _mm256_setzero_ps();
                let mut sum2 = _mm256_setzero_ps();
                let mut sum3 = _mm256_setzero_ps();
                for col in (0..5120).step_by(16) {
                    let first = _mm256_loadu_ps(input.add(b * 5120 + col));
                    let second = _mm256_loadu_ps(input.add(b * 5120 + col + 8));
                    let even = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xd8>(_mm256_castps_pd(
                        _mm256_shuffle_ps::<0x88>(first, second),
                    )));
                    let odd = _mm256_castpd_ps(_mm256_permute4x64_pd::<0xd8>(_mm256_castps_pd(
                        _mm256_shuffle_ps::<0xdd>(first, second),
                    )));
                    let packed = _mm256_loadu_si256(weight.add(row * 5120 + col) as *const __m256i);
                    let ew = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(packed));
                    let ow = _mm256_castsi256_ps(_mm256_and_si256(
                        packed,
                        _mm256_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum0 = _mm256_fmadd_ps(ew, even, sum0);
                        sum0 = _mm256_fmadd_ps(ow, odd, sum0);
                    } else {
                        sum0 = _mm256_add_ps(_mm256_mul_ps(ew, even), sum0);
                        sum0 = _mm256_add_ps(_mm256_mul_ps(ow, odd), sum0);
                    }
                    let packed =
                        _mm256_loadu_si256(weight.add((row + 1) * 5120 + col) as *const __m256i);
                    let ew = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(packed));
                    let ow = _mm256_castsi256_ps(_mm256_and_si256(
                        packed,
                        _mm256_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum1 = _mm256_fmadd_ps(ew, even, sum1);
                        sum1 = _mm256_fmadd_ps(ow, odd, sum1);
                    } else {
                        sum1 = _mm256_add_ps(_mm256_mul_ps(ew, even), sum1);
                        sum1 = _mm256_add_ps(_mm256_mul_ps(ow, odd), sum1);
                    }
                    let packed =
                        _mm256_loadu_si256(weight.add((row + 2) * 5120 + col) as *const __m256i);
                    let ew = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(packed));
                    let ow = _mm256_castsi256_ps(_mm256_and_si256(
                        packed,
                        _mm256_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum2 = _mm256_fmadd_ps(ew, even, sum2);
                        sum2 = _mm256_fmadd_ps(ow, odd, sum2);
                    } else {
                        sum2 = _mm256_add_ps(_mm256_mul_ps(ew, even), sum2);
                        sum2 = _mm256_add_ps(_mm256_mul_ps(ow, odd), sum2);
                    }
                    let packed =
                        _mm256_loadu_si256(weight.add((row + 3) * 5120 + col) as *const __m256i);
                    let ew = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(packed));
                    let ow = _mm256_castsi256_ps(_mm256_and_si256(
                        packed,
                        _mm256_set1_epi32(0xffff0000u32 as i32),
                    ));
                    if FUSED {
                        sum3 = _mm256_fmadd_ps(ew, even, sum3);
                        sum3 = _mm256_fmadd_ps(ow, odd, sum3);
                    } else {
                        sum3 = _mm256_add_ps(_mm256_mul_ps(ew, even), sum3);
                        sum3 = _mm256_add_ps(_mm256_mul_ps(ow, odd), sum3);
                    }
                }
                let mut v = [0.0f32; 8];
                _mm256_storeu_ps(v.as_mut_ptr(), sum0);
                *output.add(b * 48 + row) =
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
                let mut v = [0.0f32; 8];
                _mm256_storeu_ps(v.as_mut_ptr(), sum1);
                *output.add(b * 48 + row + 1) =
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
                let mut v = [0.0f32; 8];
                _mm256_storeu_ps(v.as_mut_ptr(), sum2);
                *output.add(b * 48 + row + 2) =
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
                let mut v = [0.0f32; 8];
                _mm256_storeu_ps(v.as_mut_ptr(), sum3);
                *output.add(b * 48 + row + 3) =
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn arm<const PARTS: usize, const FUSED: bool>(
    weight: *const u16,
    input: *const f32,
    output: *mut f32,
    batch: usize,
) {
    // SAFETY: the exported entry validates shape, buffers, lane count and CPU support.
    unsafe {
        use std::arch::aarch64::*;
        for b in 0..batch {
            for row in (0..48).step_by(4) {
                let mut sum0 = [vdupq_n_f32(0.0); PARTS];
                let mut sum1 = [vdupq_n_f32(0.0); PARTS];
                let mut sum2 = [vdupq_n_f32(0.0); PARTS];
                let mut sum3 = [vdupq_n_f32(0.0); PARTS];
                for col in (0..5120).step_by(PARTS * 8) {
                    for part in 0..PARTS {
                        let c = col + part * 8;
                        let first = vld1q_f32(input.add(b * 5120 + c));
                        let second = vld1q_f32(input.add(b * 5120 + c + 4));
                        let even = vuzp1q_f32(first, second);
                        let odd = vuzp2q_f32(first, second);
                        let packed = vreinterpretq_u32_u16(vld1q_u16(weight.add(row * 5120 + c)));
                        let ew = vreinterpretq_f32_u32(vshlq_n_u32::<16>(packed));
                        let ow = vreinterpretq_f32_u32(vandq_u32(packed, vdupq_n_u32(0xffff0000)));
                        if FUSED {
                            sum0[part] = vfmaq_f32(sum0[part], ew, even);
                            sum0[part] = vfmaq_f32(sum0[part], ow, odd);
                        } else {
                            sum0[part] = vaddq_f32(vmulq_f32(ew, even), sum0[part]);
                            sum0[part] = vaddq_f32(vmulq_f32(ow, odd), sum0[part]);
                        }
                        let packed =
                            vreinterpretq_u32_u16(vld1q_u16(weight.add((row + 1) * 5120 + c)));
                        let ew = vreinterpretq_f32_u32(vshlq_n_u32::<16>(packed));
                        let ow = vreinterpretq_f32_u32(vandq_u32(packed, vdupq_n_u32(0xffff0000)));
                        if FUSED {
                            sum1[part] = vfmaq_f32(sum1[part], ew, even);
                            sum1[part] = vfmaq_f32(sum1[part], ow, odd);
                        } else {
                            sum1[part] = vaddq_f32(vmulq_f32(ew, even), sum1[part]);
                            sum1[part] = vaddq_f32(vmulq_f32(ow, odd), sum1[part]);
                        }
                        let packed =
                            vreinterpretq_u32_u16(vld1q_u16(weight.add((row + 2) * 5120 + c)));
                        let ew = vreinterpretq_f32_u32(vshlq_n_u32::<16>(packed));
                        let ow = vreinterpretq_f32_u32(vandq_u32(packed, vdupq_n_u32(0xffff0000)));
                        if FUSED {
                            sum2[part] = vfmaq_f32(sum2[part], ew, even);
                            sum2[part] = vfmaq_f32(sum2[part], ow, odd);
                        } else {
                            sum2[part] = vaddq_f32(vmulq_f32(ew, even), sum2[part]);
                            sum2[part] = vaddq_f32(vmulq_f32(ow, odd), sum2[part]);
                        }
                        let packed =
                            vreinterpretq_u32_u16(vld1q_u16(weight.add((row + 3) * 5120 + c)));
                        let ew = vreinterpretq_f32_u32(vshlq_n_u32::<16>(packed));
                        let ow = vreinterpretq_f32_u32(vandq_u32(packed, vdupq_n_u32(0xffff0000)));
                        if FUSED {
                            sum3[part] = vfmaq_f32(sum3[part], ew, even);
                            sum3[part] = vfmaq_f32(sum3[part], ow, odd);
                        } else {
                            sum3[part] = vaddq_f32(vmulq_f32(ew, even), sum3[part]);
                            sum3[part] = vaddq_f32(vmulq_f32(ow, odd), sum3[part]);
                        }
                    }
                }
                let mut v = [0.0f32; 8];
                for (part, &sum) in sum0.iter().enumerate() {
                    vst1q_f32(v.as_mut_ptr().add(part * 4), sum);
                }
                *output.add(b * 48 + row) = if PARTS == 1 {
                    (v[0] + v[2]) + (v[1] + v[3])
                } else {
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]))
                };
                let mut v = [0.0f32; 8];
                for (part, &sum) in sum1.iter().enumerate() {
                    vst1q_f32(v.as_mut_ptr().add(part * 4), sum);
                }
                *output.add(b * 48 + row + 1) = if PARTS == 1 {
                    (v[0] + v[2]) + (v[1] + v[3])
                } else {
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]))
                };
                let mut v = [0.0f32; 8];
                for (part, &sum) in sum2.iter().enumerate() {
                    vst1q_f32(v.as_mut_ptr().add(part * 4), sum);
                }
                *output.add(b * 48 + row + 2) = if PARTS == 1 {
                    (v[0] + v[2]) + (v[1] + v[3])
                } else {
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]))
                };
                let mut v = [0.0f32; 8];
                for (part, &sum) in sum3.iter().enumerate() {
                    vst1q_f32(v.as_mut_ptr().add(part * 4), sum);
                }
                *output.add(b * 48 + row + 3) = if PARTS == 1 {
                    (v[0] + v[2]) + (v[1] + v[3])
                } else {
                    ((v[0] + v[4]) + (v[2] + v[6])) + ((v[1] + v[5]) + (v[3] + v[7]))
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        STATUS_BUFFER_TOO_SMALL, STATUS_INVALID_SHAPE, STATUS_NULL_POINTER, STATUS_OK,
        jmodels_bf16_f32_batched_matmul,
    };

    // Independent scalar implementation of the externally selected accumulation contract.
    fn reference(weight: &[u16], input: &[f32], row: usize, lanes: usize, fused: bool) -> f32 {
        let mut sums = [0.0f32; 8];
        for col in (0..5120).step_by(lanes * 2) {
            for (lane, sum) in sums.iter_mut().enumerate().take(lanes) {
                for parity in 0..2 {
                    let index = col + lane * 2 + parity;
                    let w = f32::from_bits(u32::from(weight[row * 5120 + index]) << 16);
                    *sum = if fused {
                        w.mul_add(input[index], *sum)
                    } else {
                        w * input[index] + *sum
                    };
                }
            }
        }
        let mut width = lanes / 2;
        while width != 0 {
            for lane in 0..width {
                sums[lane] += sums[lane + width];
            }
            width /= 2;
        }
        sums[0]
    }

    #[test]
    #[cfg_attr(miri, ignore = "hardware SIMD is checked by native tests")]
    fn exported_bf16_matches_selected_lane_and_fma_contract() {
        if !supported() {
            return;
        }
        let weights: Vec<u16> = (0..48 * 5120)
            .map(|i| {
                (((i * 17 % 127) as f32 - 63.0) * 0.0037)
                    .to_bits()
                    .wrapping_shr(16) as u16
            })
            .collect();
        for batch in [1, 3, 32] {
            let input: Vec<f32> = (0..batch * 5120)
                .map(|i| ((i * 31 % 113) as f32 - 56.0) * 0.0123)
                .collect();
            for lanes in [4, 8] {
                for fused in [false, true] {
                    let mut output = vec![42.0; batch * 48 + 2];
                    // SAFETY: all buffers live and disjoint, exact fixed shape plus output canaries.
                    let status = unsafe {
                        jmodels_bf16_f32_batched_matmul(
                            weights.as_ptr(),
                            (weights.len() * 2) as u64,
                            input.as_ptr(),
                            input.len() as u64,
                            output.as_mut_ptr().add(1),
                            (batch * 48) as u64,
                            batch as u32,
                            48,
                            5120,
                            lanes,
                            u32::from(fused),
                        )
                    };
                    assert_eq!(status, STATUS_OK);
                    for b in 0..batch {
                        for row in 0..48 {
                            assert_eq!(
                                output[1 + b * 48 + row].to_bits(),
                                reference(&weights, &input[b * 5120..], row, lanes as usize, fused)
                                    .to_bits(),
                                "batch={batch} row={row} lanes={lanes} fused={fused}"
                            );
                        }
                    }
                    assert_eq!(output[0], 42.0);
                    assert_eq!(output[batch * 48 + 1], 42.0);
                }
            }
        }
    }

    #[test]
    fn bf16_invalid_abi_arguments_do_not_write_output() {
        let weights = vec![0u16; 48 * 5120];
        let mut input = vec![1.0; 5120];
        let mut output = [42.0f32; 48];
        // SAFETY: each invalid call must return before dereferencing unsupported buffers.
        unsafe {
            assert_eq!(
                jmodels_bf16_f32_batched_matmul(
                    std::ptr::null(),
                    0,
                    input.as_ptr(),
                    5120,
                    output.as_mut_ptr(),
                    48,
                    1,
                    48,
                    5120,
                    4,
                    1
                ),
                STATUS_NULL_POINTER
            );
            assert_eq!(
                jmodels_bf16_f32_batched_matmul(
                    weights.as_ptr(),
                    491520,
                    input.as_ptr(),
                    5120,
                    output.as_mut_ptr(),
                    48,
                    1,
                    48,
                    5120,
                    16,
                    1
                ),
                STATUS_INVALID_SHAPE
            );
            if supported() {
                assert_eq!(
                    jmodels_bf16_f32_batched_matmul(
                        weights.as_ptr(),
                        491519,
                        input.as_ptr(),
                        5120,
                        output.as_mut_ptr(),
                        48,
                        1,
                        48,
                        5120,
                        4,
                        1
                    ),
                    STATUS_BUFFER_TOO_SMALL
                );
                assert_eq!(
                    jmodels_bf16_f32_batched_matmul(
                        weights.as_ptr(),
                        491520,
                        input.as_ptr(),
                        5120,
                        input.as_mut_ptr(),
                        48,
                        1,
                        48,
                        5120,
                        4,
                        1
                    ),
                    STATUS_INVALID_SHAPE
                );
                assert_eq!(
                    jmodels_bf16_f32_batched_matmul(
                        weights.as_ptr(),
                        491520,
                        input.as_ptr(),
                        5120,
                        output.as_mut_ptr(),
                        47,
                        1,
                        48,
                        5120,
                        4,
                        1
                    ),
                    STATUS_BUFFER_TOO_SMALL
                );
            }
        }
        assert_eq!(output, [42.0; 48]);
        assert!(input.iter().all(|&v| v == 1.0));
    }
}
