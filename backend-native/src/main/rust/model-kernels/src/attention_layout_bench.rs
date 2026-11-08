// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0
// Remote diagnostic only; this does not establish model qualification.
use super::{AttentionJob, WorkerPool};
use std::hint::black_box;
use std::time::Instant;

#[test]
#[ignore = "remote-only layout diagnostic, outside admission"]
fn compare_attention_cache_layouts() {
    assert_eq!(std::env::consts::OS, "linux");
    let pool = WorkerPool::new(8).unwrap();
    let (heads, width) = (32, 128);
    let dimension = heads * width;
    let mut seed = 511_u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    };
    let query: Vec<f32> = (0..dimension).map(|_| next()).collect();
    let eviction = vec![1.0_f32; 16 * 1024 * 1024];
    for positions in [64, 606, 1024] {
        let original_keys: Vec<f32> = (0..positions * dimension).map(|_| next()).collect();
        let original_values: Vec<f32> = (0..positions * dimension).map(|_| next()).collect();
        let mut expected = None;
        for layout in ["token-major", "padded", "head-major", "token-major-repeat"] {
            let (row_stride, head_stride) = match layout {
                "head-major" => (width, positions * width),
                "padded" => (dimension + 16, width),
                _ => (dimension, width),
            };
            let elements = (positions - 1) * row_stride + (heads - 1) * head_stride + width;
            let mut keys = vec![0.0; elements];
            let mut values = vec![0.0; elements];
            for row in 0..positions {
                for head in 0..heads {
                    let source = row * dimension + head * width;
                    let target = row * row_stride + head * head_stride;
                    keys[target..target + width]
                        .copy_from_slice(&original_keys[source..source + width]);
                    values[target..target + width]
                        .copy_from_slice(&original_values[source..source + width]);
                }
            }
            let mut output = vec![0.0; dimension];
            let mut scores = vec![0.0; heads * positions];
            let job = AttentionJob {
                query: query.as_ptr() as usize,
                query_offset: 0,
                keys: [keys.as_ptr() as usize, 0],
                key_offsets: [0, 0],
                values: [values.as_ptr() as usize, 0],
                value_offsets: [0, 0],
                positions: [positions, 0],
                output: output.as_mut_ptr() as usize,
                output_offset: 0,
                scores: scores.as_mut_ptr() as usize,
                key_dim: row_stride,
                value_dim: row_stride,
                key_length: width,
                value_length: width,
                key_head_stride: head_stride,
                value_head_stride: head_stride,
                num_heads: heads,
                num_kv_heads: heads,
                scale: (1.0 / (width as f64).sqrt()) as f32,
            };
            assert!(pool.execute_attention(job));
            if let Some(ref original) = expected {
                assert_eq!(&output, original, "layout changed output: {layout}");
            } else {
                expected = Some(output.clone());
            }
            for _ in 0..20 {
                assert!(pool.execute_attention(job));
            }
            for evict in [false, true] {
                let mut samples = Vec::new();
                for _ in 0..7 {
                    let mut seconds = 0.0;
                    for _ in 0..30 {
                        if evict {
                            // Controller-only eviction is reported explicitly; it does not
                            // guarantee eviction from every worker's private cache/CCX.
                            black_box(eviction.iter().step_by(16).fold(0.0_f32, |a, b| a + b));
                        }
                        let start = Instant::now();
                        assert!(pool.execute_attention(job));
                        seconds += start.elapsed().as_secs_f64();
                    }
                    samples.push(seconds * 1000.0 / 30.0);
                    assert_eq!(Some(&output), expected.as_ref());
                }
                println!(
                    "{}",
                    serde_json::json!({"schema":"attention-layout-diagnostic-v1", "layout":layout,
                    "positions":positions,"heads":heads,"width":width,"threads":8,
                    "controllerEviction":evict,"warmups":20,"callsPerSample":30,
                    "latencyMs":samples,"exactOutputParity":true})
                );
            }
        }
    }
}
