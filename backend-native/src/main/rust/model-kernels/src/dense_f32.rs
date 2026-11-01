// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0
//! Synchronous row-partitioned F32 projection. Caller owns every buffer.
use rayon::prelude::*;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
static NEXT: AtomicU64 = AtomicU64::new(1);
static POOLS: LazyLock<Mutex<HashMap<u64, Arc<rayon::ThreadPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[unsafe(no_mangle)]
pub extern "C" fn jmodels_dense_f32_abi() -> i32 {
    1
}
#[unsafe(no_mangle)]
pub extern "C" fn jmodels_dense_f32_create(threads: u32) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        if threads == 0 || threads > 256 {
            return 0;
        }
        let Ok(pool) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .build()
        else {
            return 0;
        };
        let handle = NEXT.fetch_add(1, Ordering::Relaxed);
        if handle == 0 {
            return 0;
        }
        let Ok(mut registry) = POOLS.lock() else {
            return 0;
        };
        registry.insert(handle, Arc::new(pool));
        handle
    }))
    .unwrap_or(0)
}
#[unsafe(no_mangle)]
pub extern "C" fn jmodels_dense_f32_destroy(handle: u64) {
    let _ = catch_unwind(|| {
        if let Ok(mut registry) = POOLS.lock() {
            registry.remove(&handle);
        }
    });
}
/// Returns 0 on success, -1 for invalid dimensions/buffers, -2 for a stale handle, -3 on panic.
/// # Safety
/// All pointers must be aligned f32 buffers valid for the advertised element counts throughout
/// this synchronous call. Output must not overlap either input. No pointer escapes the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_dense_f32_project(
    handle: u64,
    x: *const f32,
    x_len: usize,
    w: *const f32,
    w_len: usize,
    y: *mut f32,
    y_len: usize,
    batch: usize,
    inputs: usize,
    outputs: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if batch == 0
            || inputs == 0
            || outputs == 0
            || x.is_null()
            || w.is_null()
            || y.is_null()
            || !(x as usize).is_multiple_of(4)
            || !(w as usize).is_multiple_of(4)
            || !(y as usize).is_multiple_of(4)
        {
            return -1;
        }
        let Some(x_need) = batch.checked_mul(inputs) else {
            return -1;
        };
        let Some(w_need) = outputs.checked_mul(inputs) else {
            return -1;
        };
        let Some(y_need) = batch.checked_mul(outputs) else {
            return -1;
        };
        if x_need > x_len
            || w_need > w_len
            || y_need > y_len
            || [x_need, w_need, y_need]
                .iter()
                .any(|&n| n > isize::MAX as usize / 4)
        {
            return -1;
        }
        let pool = match POOLS.lock() {
            Ok(r) => r.get(&handle).cloned(),
            Err(_) => None,
        };
        let Some(pool) = pool else {
            return -2;
        };
        let count = pool.current_num_threads().min(outputs);
        let (xp, wp, yp) = (x as usize, w as usize, y as usize);
        pool.install(|| {
            (0..count).into_par_iter().for_each(|worker| {
                let width = outputs / count;
                let remainder = outputs % count;
                let begin = width * worker + remainder.min(worker);
                let end = begin + width + usize::from(worker < remainder);
                // Each worker owns disjoint output columns. Shared inputs/weights are read-only.
                unsafe {
                    matrixmultiply::sgemm(
                        batch,
                        inputs,
                        end - begin,
                        1.0,
                        xp as *const f32,
                        inputs as isize,
                        1,
                        (wp as *const f32).add(begin * inputs),
                        1,
                        inputs as isize,
                        0.0,
                        (yp as *mut f32).add(begin),
                        outputs as isize,
                        1,
                    );
                }
            })
        });
        0
    }))
    .unwrap_or(-3)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uneven_rows_match_scalar_and_invalid_calls_preserve_output() {
        assert_eq!(jmodels_dense_f32_create(0), 0);
        for threads in [1, 3, 8] {
            let handle = jmodels_dense_f32_create(threads);
            assert_ne!(handle, 0);
            let (batch, inputs, outputs) = (7, 19, 13);
            let x: Vec<f32> = (0..batch * inputs)
                .map(|i| ((i * 17 % 31) as f32 - 15.0) / 16.0)
                .collect();
            let w: Vec<f32> = (0..outputs * inputs)
                .map(|i| ((i * 7 % 23) as f32 - 11.0) / 16.0)
                .collect();
            let mut y = vec![99f32; batch * outputs + 1];
            assert_eq!(
                unsafe {
                    jmodels_dense_f32_project(
                        handle,
                        x.as_ptr(),
                        x.len(),
                        w.as_ptr(),
                        w.len(),
                        y.as_mut_ptr(),
                        batch * outputs,
                        batch,
                        inputs,
                        outputs,
                    )
                },
                0
            );
            for m in 0..batch {
                for n in 0..outputs {
                    let expected: f32 = (0..inputs)
                        .map(|k| x[m * inputs + k] * w[n * inputs + k])
                        .sum();
                    assert!((y[m * outputs + n] - expected).abs() < 1e-5);
                }
            }
            assert_eq!(y[batch * outputs], 99.0);
            let previous = y.clone();
            assert_eq!(
                unsafe {
                    jmodels_dense_f32_project(
                        handle,
                        x.as_ptr(),
                        x.len() - 1,
                        w.as_ptr(),
                        w.len(),
                        y.as_mut_ptr(),
                        batch * outputs,
                        batch,
                        inputs,
                        outputs,
                    )
                },
                -1
            );
            assert_eq!(y, previous);
            jmodels_dense_f32_destroy(handle);
            jmodels_dense_f32_destroy(handle);
            assert_eq!(
                unsafe {
                    jmodels_dense_f32_project(
                        handle,
                        x.as_ptr(),
                        x.len(),
                        w.as_ptr(),
                        w.len(),
                        y.as_mut_ptr(),
                        batch * outputs,
                        batch,
                        inputs,
                        outputs,
                    )
                },
                -2
            );
        }
    }
}
