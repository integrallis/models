//! The CPU_REPACK Q4_K numerical contract used by the image decoder.
//! GEMM folds 64-element subblocks; GEMV folds whole 256-element blocks.
//! The choice depends on the original batch, never on worker partition sizes.
use super::*;

pub(super) fn supported() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
        && is_x86_feature_detected!("f16c")
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn rows(
    weights: &[u8],
    q: &[i8],
    scales: &[f32],
    sums: &[i16],
    out: *mut f32,
    batch: usize,
    rows: usize,
    cols: usize,
    start: usize,
    end: usize,
) {
    if batch > 1 {
        // SAFETY: this worker owns exactly the supplied output-row interval.
        unsafe {
            compute_q4_k_packed_rows(
                weights,
                q,
                scales,
                sums,
                out,
                batch,
                rows,
                cols,
                start,
                end,
                Q4Reduction::Repacked,
            );
        }
    } else {
        for row in start..end {
            // SAFETY: a single output per validated row belongs to this worker.
            unsafe {
                out.add(row)
                    .write(dot(weights, q, scales, sums, 0, row, cols, false));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn dot(
    weights: &[u8],
    q: &[i8],
    scales: &[f32],
    sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
    subblocks: bool,
) -> f32 {
    let blocks = cols / 256;
    let mut total = 0_f32;
    let mut correction = 0_f32;
    let mask = _mm256_set1_epi8(15);
    for block in 0..blocks {
        let off = (row * blocks + block) * 144;
        let d = f16_to_f32_f16c(u16::from_le_bytes([weights[off], weights[off + 1]]))
            * scales[batch * blocks + block];
        let dm = f16_to_f32_f16c(u16::from_le_bytes([weights[off + 2], weights[off + 3]]))
            * scales[batch * blocks + block];
        let packed_scales = &weights[off + 4..off + 16];
        let mut block_dot = 0_i32;
        let mut block_min = 0_i32;
        for pair in 0..4 {
            // SAFETY: every Q4_K block contains four complete 32-byte quant groups.
            let w =
                unsafe { _mm256_loadu_si256(weights.as_ptr().add(off + 16 + pair * 32).cast()) };
            let mut lanes = _mm256_setzero_si256();
            for g in 0..2 {
                let quants = if g == 0 {
                    _mm256_and_si256(w, mask)
                } else {
                    _mm256_and_si256(_mm256_srli_epi16(w, 4), mask)
                };
                // SAFETY: the corresponding Q8_K activation group has 32 bytes.
                let a = unsafe {
                    _mm256_loadu_si256(
                        q.as_ptr()
                            .add(batch * cols + block * 256 + pair * 64 + g * 32)
                            .cast(),
                    )
                };
                lanes = _mm256_add_epi32(
                    lanes,
                    _mm256_madd_epi16(
                        _mm256_maddubs_epi16(quants, a),
                        _mm256_set1_epi16(qk_scale(packed_scales, pair * 2 + g) as i16),
                    ),
                );
            }
            let integer = horizontal_sum_i32_avx2(lanes);
            let so = batch * cols / 16 + block * 16 + pair * 4;
            let minimum = qk_min(packed_scales, pair * 2) * (sums[so] as i32 + sums[so + 1] as i32)
                + qk_min(packed_scales, pair * 2 + 1) * (sums[so + 2] as i32 + sums[so + 3] as i32);
            if subblocks {
                total = d.mul_add(integer as f32, total);
                correction = dm.mul_add(minimum as f32, correction);
            } else {
                block_dot += integer;
                block_min += minimum;
            }
        }
        if !subblocks {
            total = d.mul_add(block_dot as f32, total);
            correction = dm.mul_add(block_min as f32, correction);
        }
    }
    total - correction
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn tile<const B: usize>(
    packed: &[PackedQ4Block],
    q: &[i8],
    scales: &[f32],
    sums: &[i16],
    out: *mut f32,
    batch: usize,
    row: usize,
    rows: usize,
    cols: usize,
) {
    let mut result = [_mm256_setzero_ps(); B];
    let mut minimum = [_mm256_setzero_ps(); B];
    for (block, p) in packed.iter().enumerate() {
        let d = unsafe { _mm256_loadu_ps(p.d.as_ptr()) };
        let dm = unsafe { _mm256_loadu_ps(p.dm.as_ptr()) };
        let mut ds = [_mm256_setzero_ps(); B];
        let mut dms = [_mm256_setzero_ps(); B];
        for b in 0..B {
            let scale = _mm256_set1_ps(scales[(batch + b) * packed.len() + block]);
            ds[b] = _mm256_mul_ps(d, scale);
            dms[b] = _mm256_mul_ps(dm, scale);
        }
        let mut block_dot = [_mm256_setzero_si256(); B];
        let mut block_min = [_mm256_setzero_si256(); B];
        for pair in 0..4 {
            let mut dot = [_mm256_setzero_si256(); B];
            for g in pair * 2..pair * 2 + 2 {
                let scale = unsafe { _mm256_loadu_si256(p.scales16[g].as_ptr().cast()) };
                for k in 0..8 {
                    for b in 0..B {
                        let a = unsafe {
                            q.as_ptr()
                                .add((batch + b) * cols + block * 256 + g * 32 + k * 4)
                                .cast::<i32>()
                                .read_unaligned()
                        };
                        let products = _mm256_maddubs_epi16(p.quants[g][k], _mm256_set1_epi32(a));
                        dot[b] = _mm256_add_epi32(dot[b], _mm256_madd_epi16(products, scale));
                    }
                }
            }
            let m = unsafe { _mm256_loadu_si256(p.minima16[pair].as_ptr().cast()) };
            for b in 0..B {
                let off = (batch + b) * cols / 16 + block * 16 + pair * 4;
                let a = sums[off] + sums[off + 1];
                let c = sums[off + 2] + sums[off + 3];
                let pair_sum =
                    _mm256_set1_epi32(((a as u16 as u32) | ((c as u16 as u32) << 16)) as i32);
                let product = _mm256_madd_epi16(m, pair_sum);
                if B == 1 {
                    block_dot[b] = _mm256_add_epi32(block_dot[b], dot[b]);
                    block_min[b] = _mm256_add_epi32(block_min[b], product);
                } else {
                    result[b] = _mm256_fmadd_ps(ds[b], _mm256_cvtepi32_ps(dot[b]), result[b]);
                    minimum[b] = _mm256_fmadd_ps(dms[b], _mm256_cvtepi32_ps(product), minimum[b]);
                }
            }
        }
        if B == 1 {
            result[0] = _mm256_fmadd_ps(ds[0], _mm256_cvtepi32_ps(block_dot[0]), result[0]);
            minimum[0] = _mm256_fmadd_ps(dms[0], _mm256_cvtepi32_ps(block_min[0]), minimum[0]);
        }
    }
    for b in 0..B {
        unsafe {
            _mm256_storeu_ps(
                out.add((batch + b) * rows + row),
                _mm256_sub_ps(result[b], minimum[b]),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    }
    #[test]
    fn reference_graph_bits_survive_worker_partitions_and_batch_tails() {
        if !supported() {
            return;
        }
        {
            let weights = include_bytes!("../testdata/q4-repacked/768/weights.q4k");
            let input = floats(include_bytes!("../testdata/q4-repacked/768/input.f32"));
            let cases: &[(usize, &[u8])] = &[
                (
                    1,
                    include_bytes!("../testdata/q4-repacked/768/1-expected.f32"),
                ),
                (
                    3,
                    include_bytes!("../testdata/q4-repacked/768/3-expected.f32"),
                ),
                (
                    4,
                    include_bytes!("../testdata/q4-repacked/768/4-expected.f32"),
                ),
                (
                    8,
                    include_bytes!("../testdata/q4-repacked/768/8-expected.f32"),
                ),
                (
                    17,
                    include_bytes!("../testdata/q4-repacked/768/17-expected.f32"),
                ),
            ];
            for &(batch, expected) in cases {
                let expected = floats(expected);
                for threads in [1, 3, 8] {
                    let context = jmodels_kernels_context_create(threads);
                    assert!(!context.is_null());
                    let mut output = vec![f32::NAN; batch * 8];
                    // SAFETY: complete fixture buffers remain live through synchronous execution.
                    let status = unsafe {
                        jmodels_quantized_f32_batched_matmul_with_context(
                            context,
                            11,
                            weights.as_ptr(),
                            weights.len() as u64,
                            input.as_ptr(),
                            (batch * 768) as u64,
                            output.as_mut_ptr(),
                            output.len() as u64,
                            batch as u32,
                            8,
                            768,
                        )
                    };
                    // SAFETY: this test owns the context and the call has completed.
                    assert_eq!(
                        unsafe { jmodels_kernels_context_destroy(context) },
                        STATUS_OK
                    );
                    assert_eq!(status, STATUS_OK);
                    for (index, (&a, &b)) in output.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "cols=768 batch={batch} threads={threads} output={index}: {a} versus {b}"
                        );
                    }
                }
            }
        }
        {
            let weights = include_bytes!("../testdata/q4-repacked/4096/weights.q4k");
            let input = floats(include_bytes!("../testdata/q4-repacked/4096/input.f32"));
            let cases: &[(usize, &[u8])] = &[
                (
                    1,
                    include_bytes!("../testdata/q4-repacked/4096/1-expected.f32"),
                ),
                (
                    3,
                    include_bytes!("../testdata/q4-repacked/4096/3-expected.f32"),
                ),
                (
                    4,
                    include_bytes!("../testdata/q4-repacked/4096/4-expected.f32"),
                ),
                (
                    8,
                    include_bytes!("../testdata/q4-repacked/4096/8-expected.f32"),
                ),
                (
                    17,
                    include_bytes!("../testdata/q4-repacked/4096/17-expected.f32"),
                ),
            ];
            for &(batch, expected) in cases {
                let expected = floats(expected);
                for threads in [1, 3, 8] {
                    let context = jmodels_kernels_context_create(threads);
                    assert!(!context.is_null());
                    let mut output = vec![f32::NAN; batch * 8];
                    // SAFETY: complete fixture buffers remain live through synchronous execution.
                    let status = unsafe {
                        jmodels_quantized_f32_batched_matmul_with_context(
                            context,
                            11,
                            weights.as_ptr(),
                            weights.len() as u64,
                            input.as_ptr(),
                            (batch * 4096) as u64,
                            output.as_mut_ptr(),
                            output.len() as u64,
                            batch as u32,
                            8,
                            4096,
                        )
                    };
                    // SAFETY: this test owns the context and the call has completed.
                    assert_eq!(
                        unsafe { jmodels_kernels_context_destroy(context) },
                        STATUS_OK
                    );
                    assert_eq!(status, STATUS_OK);
                    for (index, (&a, &b)) in output.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "cols=4096 batch={batch} threads={threads} output={index}: {a} versus {b}"
                        );
                    }
                }
            }
        }
    }
}
