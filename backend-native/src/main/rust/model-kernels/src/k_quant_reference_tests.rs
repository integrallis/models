//! Independent CPU-graph fixtures cover tiled dispatch and its batch-size boundary.
use super::*;
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
#[test]
fn reference_k_quant_graph_matches_at_dispatch_boundary() {
    if !q4_repack::supported() {
        return;
    }
    macro_rules! check {
        ($kind:literal, $format:literal, $cols:literal, $batch:literal) => {{
            let w = include_bytes!(concat!(
                "../testdata/k-quant-graph/",
                $kind,
                "/",
                stringify!($cols),
                "/weights.bin"
            ));
            let x = floats(include_bytes!(concat!(
                "../testdata/k-quant-graph/",
                $kind,
                "/",
                stringify!($cols),
                "/input.f32"
            )));
            let expected = floats(include_bytes!(concat!(
                "../testdata/k-quant-graph/",
                $kind,
                "/",
                stringify!($cols),
                "/expected-",
                stringify!($batch),
                ".f32"
            )));
            for threads in [1, 3, 8] {
                let ctx = jmodels_kernels_context_create(threads);
                assert!(!ctx.is_null());
                let mut out = vec![f32::NAN; 8 * $batch];
                // SAFETY: complete buffers and exclusive output outlive the synchronous call.
                let status = unsafe {
                    jmodels_quantized_f32_batched_matmul_with_context(
                        ctx,
                        $format,
                        w.as_ptr(),
                        w.len() as u64,
                        x.as_ptr(),
                        ($cols * $batch) as u64,
                        out.as_mut_ptr(),
                        out.len() as u64,
                        $batch,
                        8,
                        $cols,
                    )
                };
                // SAFETY: this test owns the idle context.
                assert_eq!(unsafe { jmodels_kernels_context_destroy(ctx) }, STATUS_OK);
                assert_eq!(status, STATUS_OK);
                for (i, (&a, &b)) in out.iter().zip(&expected).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "kind={}, cols={}, batch={}, threads={threads}, index={i}",
                        $kind,
                        $cols,
                        $batch
                    );
                }
            }
        }};
    }
    check!("q5", 12, 768, 1);
    check!("q5", 12, 768, 3);
    check!("q5", 12, 768, 7);
    check!("q5", 12, 768, 8);
    check!("q5", 12, 768, 17);
    check!("q5", 12, 4096, 1);
    check!("q5", 12, 4096, 3);
    check!("q5", 12, 4096, 7);
    check!("q5", 12, 4096, 8);
    check!("q5", 12, 4096, 17);
    check!("q6", 13, 768, 1);
    check!("q6", 13, 768, 3);
    check!("q6", 13, 768, 7);
    check!("q6", 13, 768, 8);
    check!("q6", 13, 768, 17);
    check!("q6", 13, 4096, 1);
    check!("q6", 13, 4096, 3);
    check!("q6", 13, 4096, 7);
    check!("q6", 13, 4096, 8);
    check!("q6", 13, 4096, 17);
}
