// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0
//! Synchronous F32 matrix and disentangled-attention kernels. Caller owns every buffer.
use rayon::prelude::*;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread::JoinHandle;
static NEXT: AtomicU64 = AtomicU64::new(1);
static POOLS: LazyLock<Mutex<HashMap<u64, Arc<OwnedPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
// Rayon dropping a pool requests shutdown but does not join its OS threads. The FFM owner
// may unload this library immediately after destroy, so all threads must have returned first.
struct OwnedPool {
    pool: Option<rayon::ThreadPool>,
    workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
}
impl OwnedPool {
    fn new(threads: usize) -> Result<Self, rayon::ThreadPoolBuildError> {
        let mut owned = Self {
            pool: None,
            workers: Arc::new(Mutex::new(Vec::new())),
        };
        let workers = Arc::clone(&owned.workers);
        owned.pool = Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .spawn_handler(move |thread| {
                    let mut builder = std::thread::Builder::new();
                    if let Some(name) = thread.name() {
                        builder = builder.name(name.to_owned());
                    }
                    if let Some(size) = thread.stack_size() {
                        builder = builder.stack_size(size);
                    }
                    let handle = builder.spawn(move || thread.run())?;
                    workers
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(handle);
                    Ok(())
                })
                .build()?,
        );
        Ok(owned)
    }
}
impl std::ops::Deref for OwnedPool {
    type Target = rayon::ThreadPool;
    fn deref(&self) -> &Self::Target {
        self.pool.as_ref().expect("live DeBERTa pool")
    }
}
impl Drop for OwnedPool {
    fn drop(&mut self) {
        drop(self.pool.take());
        for worker in self
            .workers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
        {
            let _ = worker.join();
        }
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn jmodels_deberta_abi() -> i32 {
    1
}
#[unsafe(no_mangle)]
pub extern "C" fn jmodels_deberta_create(threads: u32) -> u64 {
    catch_unwind(AssertUnwindSafe(|| {
        if threads == 0 || threads > 256 {
            return 0;
        }
        let Ok(pool) = OwnedPool::new(threads as usize) else {
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
pub extern "C" fn jmodels_deberta_destroy(handle: u64) {
    let _ = catch_unwind(|| {
        let removed = POOLS
            .lock()
            .ok()
            .and_then(|mut registry| registry.remove(&handle));
        // Joining outside the registry lock lets independent model calls continue.
        drop(removed);
    });
}
/// Returns 0 on success, -1 for invalid dimensions/buffers, -2 for a stale handle, -3 on panic.
/// # Safety
/// All pointers must be aligned f32 buffers valid for the advertised element counts throughout
/// this synchronous call. Output must not overlap either input. No pointer escapes the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_deberta_project(
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

#[derive(Default)]
struct AttentionScratch {
    scores: Vec<f32>,
    query_relative: Vec<f32>,
    key_relative: Vec<f32>,
}
thread_local! {
    static ATTENTION_SCRATCH: std::cell::RefCell<AttentionScratch> = std::cell::RefCell::new(AttentionScratch::default());
}

/// DeBERTa content/content, content/position, position/content attention, then softmax * V.
/// Packed tensors are [Q,K,V], relative tensors are [relative Q,relative K], each row-major.
/// # Safety
/// All pointers must be aligned and valid for the advertised counts during the synchronous call.
/// Output must not alias inputs. No buffer pointer is retained after return.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_deberta_attend(
    handle: u64,
    packed: *const f32,
    packed_len: usize,
    relative: *const f32,
    relative_len: usize,
    indices: *const i32,
    indices_len: usize,
    output: *mut f32,
    output_len: usize,
    sequence: usize,
    dimension: usize,
    heads: usize,
    relative_rows: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if sequence == 0
            || dimension == 0
            || heads == 0
            || relative_rows == 0
            || !dimension.is_multiple_of(heads)
            || sequence > 4096
            || dimension > 16384
            || relative_rows > 4096
            || heads > 256
            || [
                packed as usize,
                relative as usize,
                indices as usize,
                output as usize,
            ]
            .iter()
            .any(|&p| p == 0 || !p.is_multiple_of(4))
        {
            return -1;
        }
        let Some(count) = sequence.checked_mul(dimension) else {
            return -1;
        };
        let Some(rel_count) = relative_rows.checked_mul(dimension) else {
            return -1;
        };
        let Some(square) = sequence.checked_mul(sequence) else {
            return -1;
        };
        let Some(packed_need) = count.checked_mul(3) else {
            return -1;
        };
        let Some(relative_need) = rel_count.checked_mul(2) else {
            return -1;
        };
        if packed_len < packed_need
            || relative_len < relative_need
            || indices_len < square
            || output_len < count
            || [packed_need, relative_need, square, count]
                .iter()
                .any(|&n| n > isize::MAX as usize / 4)
        {
            return -1;
        }
        let buckets = unsafe { std::slice::from_raw_parts(indices, square) };
        if buckets
            .iter()
            .any(|&b| b < 0 || b as usize >= relative_rows)
        {
            return -1;
        }
        let begin = *buckets.iter().min().unwrap() as usize;
        let span = *buckets.iter().max().unwrap() as usize + 1 - begin;
        let pool = match POOLS.lock() {
            Ok(r) => r.get(&handle).cloned(),
            Err(_) => None,
        };
        let Some(pool) = pool else {
            return -2;
        };
        let width = dimension / heads;
        let scale = (width as f32 * 3.0).sqrt();
        let mut head_outputs = vec![0.0f32; count];
        let (packed_address, relative_address) = (packed as usize, relative as usize);
        let result: Result<(), ()> = pool.install(|| {
            head_outputs
                .par_chunks_mut(sequence * width)
                .enumerate()
                .try_for_each(|(head, target)| {
                    ATTENTION_SCRATCH.with(|storage| {
                        let mut storage = storage.borrow_mut();
                        let AttentionScratch {
                            scores,
                            query_relative,
                            key_relative,
                        } = &mut *storage;
                        scores.resize(square, 0.0);
                        query_relative.resize(sequence * span, 0.0);
                        key_relative.resize(sequence * span, 0.0);
                        let offset = head * width;
                        unsafe {
                            let q = (packed_address as *const f32).add(offset);
                            let k = (packed_address as *const f32).add(count + offset);
                            let v = (packed_address as *const f32).add(2 * count + offset);
                            let rq =
                                (relative_address as *const f32).add(begin * dimension + offset);
                            let rk = (relative_address as *const f32)
                                .add(rel_count + begin * dimension + offset);
                            matrixmultiply::sgemm(
                                sequence,
                                width,
                                sequence,
                                1.0,
                                q,
                                dimension as isize,
                                1,
                                k,
                                1,
                                dimension as isize,
                                0.0,
                                scores.as_mut_ptr(),
                                sequence as isize,
                                1,
                            );
                            matrixmultiply::sgemm(
                                sequence,
                                width,
                                span,
                                1.0,
                                q,
                                dimension as isize,
                                1,
                                rk,
                                1,
                                dimension as isize,
                                0.0,
                                query_relative.as_mut_ptr(),
                                span as isize,
                                1,
                            );
                            matrixmultiply::sgemm(
                                sequence,
                                width,
                                span,
                                1.0,
                                k,
                                dimension as isize,
                                1,
                                rq,
                                1,
                                dimension as isize,
                                0.0,
                                key_relative.as_mut_ptr(),
                                span as isize,
                                1,
                            );
                            for query in 0..sequence {
                                let row = &mut scores[query * sequence..(query + 1) * sequence];
                                let mut maximum = f32::NEG_INFINITY;
                                for key in 0..sequence {
                                    let bucket = buckets[query * sequence + key] as usize - begin;
                                    let score = (row[key]
                                        + query_relative[query * span + bucket]
                                        + key_relative[key * span + bucket])
                                        / scale;
                                    if !score.is_finite() {
                                        return Err(());
                                    }
                                    row[key] = score;
                                    maximum = maximum.max(score);
                                }
                                let mut sum = 0.0f32;
                                for score in row.iter_mut() {
                                    *score = (f64::from(*score - maximum)).exp() as f32;
                                    sum += *score;
                                }
                                let inverse = 1.0 / sum;
                                for score in row {
                                    *score *= inverse;
                                }
                            }
                            matrixmultiply::sgemm(
                                sequence,
                                sequence,
                                width,
                                1.0,
                                scores.as_ptr(),
                                sequence as isize,
                                1,
                                v,
                                dimension as isize,
                                1,
                                0.0,
                                target.as_mut_ptr(),
                                width as isize,
                                1,
                            );
                        }
                        if target.iter().any(|v| !v.is_finite()) {
                            return Err(());
                        }
                        Ok(())
                    })
                })
        });
        if result.is_err() {
            return -1;
        }
        // Commit only after every head succeeds; invalid input leaves caller output untouched.
        for head in 0..heads {
            for row in 0..sequence {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        head_outputs.as_ptr().add((head * sequence + row) * width),
                        output.add(row * dimension + head * width),
                        width,
                    );
                }
            }
        }
        0
    }))
    .unwrap_or(-3)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn destroying_pool_waits_for_worker_thread_local_cleanup() {
        use std::cell::RefCell;
        use std::sync::atomic::AtomicUsize;
        struct Exit(Arc<AtomicUsize>);
        impl Drop for Exit {
            fn drop(&mut self) {
                // Make an asynchronous Rayon shutdown observably different from a joined one.
                std::thread::sleep(std::time::Duration::from_millis(20));
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        thread_local! { static EXIT: RefCell<Option<Exit>> = const { RefCell::new(None) }; }
        for _ in 0..5 {
            let exited = Arc::new(AtomicUsize::new(0));
            let pool = OwnedPool::new(3).unwrap();
            pool.broadcast(|_| {
                EXIT.with(|slot| *slot.borrow_mut() = Some(Exit(Arc::clone(&exited))))
            });
            drop(pool);
            assert_eq!(
                exited.load(Ordering::SeqCst),
                3,
                "all TLS destructors must finish before the FFM library can unload"
            );
        }
    }

    #[test]
    fn uneven_rows_match_scalar_and_invalid_calls_preserve_output() {
        assert_eq!(jmodels_deberta_create(0), 0);
        for threads in [1, 3, 8] {
            let handle = jmodels_deberta_create(threads);
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
                    jmodels_deberta_project(
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
                    jmodels_deberta_project(
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
            jmodels_deberta_destroy(handle);
            jmodels_deberta_destroy(handle);
            assert_eq!(
                unsafe {
                    jmodels_deberta_project(
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
