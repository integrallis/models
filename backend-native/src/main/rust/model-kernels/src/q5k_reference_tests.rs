//! Independent pinned-ggml fixtures: protect numerical fidelity, not just agreement
//! between the single-row and batched implementations of the same kernel.
use super::*;

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

type Fixture = (usize, &'static [u8], &'static [u8], &'static [u8]);

#[test]
fn q5k_matches_independent_reference_across_batch_shapes() {
    if !(std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c"))
    {
        return;
    }
    let fixtures: &[Fixture] = &[
        (
            768,
            include_bytes!("../testdata/q5k-row-accumulation/768/weights.q5k"),
            include_bytes!("../testdata/q5k-row-accumulation/768/input.f32"),
            include_bytes!("../testdata/q5k-row-accumulation/768/expected.f32"),
        ),
        (
            4096,
            include_bytes!("../testdata/q5k-row-accumulation/4096/weights.q5k"),
            include_bytes!("../testdata/q5k-row-accumulation/4096/input.f32"),
            include_bytes!("../testdata/q5k-row-accumulation/4096/expected.f32"),
        ),
    ];
    for &(cols, weights, input, expected) in fixtures {
        let input = floats(input);
        let expected = floats(expected);
        for batch in [1, 3, 8, 17] {
            let mut output = vec![f32::NAN; batch * 9];
            let context = jmodels_kernels_context_create(3);
            assert!(!context.is_null());
            // SAFETY: the context and complete fixture buffers outlive this synchronous call.
            let status = unsafe {
                jmodels_quantized_f32_batched_matmul_with_context(
                    context,
                    4,
                    weights.as_ptr(),
                    weights.len() as u64,
                    input.as_ptr(),
                    (batch * cols) as u64,
                    output.as_mut_ptr(),
                    output.len() as u64,
                    batch as u32,
                    9,
                    cols as u32,
                )
            };
            // SAFETY: all synchronous work has completed and this test owns the context.
            assert_eq!(
                unsafe { jmodels_kernels_context_destroy(context) },
                STATUS_OK
            );
            assert_eq!(status, STATUS_OK);
            for (index, (&actual, &reference)) in output.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    reference.to_bits(),
                    "cols={cols}, batch={batch}, output={index}: {actual} versus {reference}"
                );
            }
        }
    }
}
