// SPDX-License-Identifier: Apache-2.0
//! Exact Q8_0×F32-activation SIMD row reuse. Scheduling and immutable partition counts stay unchanged.
use super::Q8_0_BLOCK_BYTES;
use std::arch::x86_64::*;
use std::cell::RefCell;
thread_local! { static ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) }; }

#[inline]
fn fold(lanes: &[f32; 8]) -> f32 {
    ((lanes[4] + lanes[0]) + (lanes[6] + lanes[2]))
        + ((lanes[5] + lanes[1]) + (lanes[7] + lanes[3]))
}

#[target_feature(enable = "avx2")]
unsafe fn decode_avx2(weights: &[u8], row: usize, cols: usize, dst: &mut [f32]) {
    for block in 0..cols / 32 {
        let base = (row * (cols / 32) + block) * Q8_0_BLOCK_BYTES;
        let scale = _mm256_set1_ps(super::f16_to_f32(u16::from_le_bytes([
            weights[base],
            weights[base + 1],
        ])));
        // SAFETY: complete Q8_0 blocks and the destination row were validated by the caller.
        unsafe {
            let quants = [
                _mm_loadu_si128(weights.as_ptr().add(base + 2).cast()),
                _mm_loadu_si128(weights.as_ptr().add(base + 18).cast()),
            ];
            let groups = [
                _mm256_cvtepi8_epi32(quants[0]),
                _mm256_cvtepi8_epi32(_mm_srli_si128(quants[0], 8)),
                _mm256_cvtepi8_epi32(quants[1]),
                _mm256_cvtepi8_epi32(_mm_srli_si128(quants[1], 8)),
            ];
            for (g, q) in groups.iter().enumerate() {
                // Q8_0's dequantization is one multiply; there is no addend to keep unfused.
                let v = _mm256_mul_ps(_mm256_cvtepi32_ps(*q), scale);
                _mm256_storeu_ps(dst.as_mut_ptr().add(block * 32 + g * 8), v);
            }
        }
    }
}

#[target_feature(enable = "avx2,fma")]
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
