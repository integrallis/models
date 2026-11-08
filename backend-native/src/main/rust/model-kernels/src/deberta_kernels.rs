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

fn dense_softmax(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: runtime checks both target features and all loads stay within the slice.
        unsafe {
            dense_softmax_avx2(values);
        }
        return;
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in values.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inverse = 1.0 / sum;
    for x in values {
        *x *= inverse;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dense_softmax_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let vm = _mm256_set1_ps(max);
    let end = values.len() / 8 * 8;
    let mut sums = _mm256_setzero_ps();
    for offset in (0..end).step_by(8) {
        unsafe {
            let p = values.as_mut_ptr().add(offset);
            let exp = super::onnx_kernels::exp8(_mm256_sub_ps(_mm256_loadu_ps(p), vm));
            _mm256_storeu_ps(p, exp);
            sums = _mm256_add_ps(sums, exp);
        }
    }
    let halves = _mm_add_ps(_mm256_castps256_ps128(sums), _mm256_extractf128_ps(sums, 1));
    let pairs = _mm_hadd_ps(halves, halves);
    let mut sum = _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
    for x in &mut values[end..] {
        *x = (*x - max).exp();
        sum += *x;
    }
    let inverse = 1.0 / sum;
    let vi = _mm256_set1_ps(inverse);
    for offset in (0..end).step_by(8) {
        unsafe {
            let p = values.as_mut_ptr().add(offset);
            _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), vi));
        }
    }
    for x in &mut values[end..] {
        *x *= inverse;
    }
}

/// Ordinary multi-head attention over row-major Q, K, V packed consecutively.
/// # Safety
/// Buffers must be aligned, valid for their lengths and non-overlapping throughout the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_dense_attention(
    handle: u64,
    qkv: *const f32,
    qkv_len: usize,
    out: *mut f32,
    out_len: usize,
    tokens: usize,
    dim: usize,
    heads: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if tokens == 0
            || dim == 0
            || heads == 0
            || !dim.is_multiple_of(heads)
            || qkv.is_null()
            || out.is_null()
            || !(qkv as usize).is_multiple_of(4)
            || !(out as usize).is_multiple_of(4)
        {
            return -1;
        }
        let Some(count) = tokens.checked_mul(dim) else {
            return -1;
        };
        let Some(packed) = count.checked_mul(3) else {
            return -1;
        };
        let Some(scores_len) = tokens.checked_mul(tokens) else {
            return -1;
        };
        if packed > qkv_len
            || count > out_len
            || packed > isize::MAX as usize / 4
            || scores_len > isize::MAX as usize / 4
        {
            return -1;
        }
        let Some(pool) = POOLS.lock().ok().and_then(|r| r.get(&handle).cloned()) else {
            return -2;
        };
        let (src, dst) = (qkv as usize, out as usize);
        let head_dim = dim / heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        pool.install(|| {
            (0..heads).into_par_iter().for_each(|head| {
                ATTENTION_SCRATCH.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    scratch.scores.resize(scores_len, 0.0);
                    let scores = &mut scratch.scores;
                    unsafe {
                        let q = (src as *const f32).add(head * head_dim);
                        let k = q.add(count);
                        let v = k.add(count);
                        matrixmultiply::sgemm(
                            tokens,
                            head_dim,
                            tokens,
                            scale,
                            q,
                            dim as isize,
                            1,
                            k,
                            1,
                            dim as isize,
                            0.0,
                            scores.as_mut_ptr(),
                            tokens as isize,
                            1,
                        );
                        for row in scores.chunks_exact_mut(tokens) {
                            dense_softmax(row);
                        }
                        matrixmultiply::sgemm(
                            tokens,
                            tokens,
                            head_dim,
                            1.0,
                            scores.as_ptr(),
                            tokens as isize,
                            1,
                            v,
                            dim as isize,
                            1,
                            0.0,
                            (dst as *mut f32).add(head * head_dim),
                            dim as isize,
                            1,
                        );
                    }
                });
            })
        });
        0
    }))
    .unwrap_or(-3)
}

/// Batched causal attention with a retained row-major K/V prefix (ordinary MHA).
/// # Safety
/// Each aligned pointer must be valid for its advertised length; output cannot alias inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_dense_causal_attention(
    handle: u64,
    query: *const f32,
    query_len: usize,
    key: *const f32,
    key_len: usize,
    value: *const f32,
    value_len: usize,
    out: *mut f32,
    out_len: usize,
    batch: usize,
    positions: usize,
    dim: usize,
    heads: usize,
) -> i32 {
    unsafe {
        dense_causal_attention_impl(
            handle, query, query_len, key, key_len, value, value_len, out, out_len, batch,
            positions, dim, heads, None,
        )
    }
}

/// Batched causal attention with an explicit head-major cache capacity.
/// # Safety
/// Pointers must cover their advertised lengths and output cannot alias inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_dense_causal_attention_head_major(
    handle: u64,
    query: *const f32,
    query_len: usize,
    key: *const f32,
    key_len: usize,
    value: *const f32,
    value_len: usize,
    out: *mut f32,
    out_len: usize,
    batch: usize,
    positions: usize,
    dim: usize,
    heads: usize,
    capacity: usize,
) -> i32 {
    unsafe {
        dense_causal_attention_impl(
            handle,
            query,
            query_len,
            key,
            key_len,
            value,
            value_len,
            out,
            out_len,
            batch,
            positions,
            dim,
            heads,
            Some(capacity),
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dense_causal_attention_impl(
    handle: u64,
    query: *const f32,
    query_len: usize,
    key: *const f32,
    key_len: usize,
    value: *const f32,
    value_len: usize,
    out: *mut f32,
    out_len: usize,
    batch: usize,
    positions: usize,
    dim: usize,
    heads: usize,
    capacity: Option<usize>,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if batch == 0
            || positions < batch
            || dim == 0
            || heads == 0
            || !dim.is_multiple_of(heads)
            || [query, key, value, out.cast_const()]
                .iter()
                .any(|p| p.is_null() || !(*p as usize).is_multiple_of(4))
        {
            return -1;
        }
        let Some(qn) = batch.checked_mul(dim) else {
            return -1;
        };
        let cache_positions = capacity.unwrap_or(positions);
        if cache_positions < positions {
            return -1;
        }
        let Some(kn) = cache_positions.checked_mul(dim) else {
            return -1;
        };
        let Some(sn) = batch.checked_mul(positions) else {
            return -1;
        };
        if qn > query_len
            || kn > key_len
            || kn > value_len
            || qn > out_len
            || [qn, kn, sn].iter().any(|n| *n > isize::MAX as usize / 4)
        {
            return -1;
        }
        let Some(pool) = POOLS.lock().ok().and_then(|r| r.get(&handle).cloned()) else {
            return -2;
        };
        let (qp, kp, vp, yp) = (query as usize, key as usize, value as usize, out as usize);
        let width = dim / heads;
        let scale = 1.0 / (width as f32).sqrt();
        let row_stride = if capacity.is_some() { width } else { dim };
        let head_stride = if capacity.is_some() {
            cache_positions * width
        } else {
            width
        };
        pool.install(|| {
            (0..heads).into_par_iter().for_each(|head| {
                ATTENTION_SCRATCH.with(|cell| {
                    let mut scratch = cell.borrow_mut();
                    scratch.scores.resize(sn, 0.0);
                    let scores = &mut scratch.scores;
                    unsafe {
                        matrixmultiply::sgemm(
                            batch,
                            width,
                            positions,
                            scale,
                            (qp as *const f32).add(head * width),
                            dim as isize,
                            1,
                            (kp as *const f32).add(head * head_stride),
                            1,
                            row_stride as isize,
                            0.0,
                            scores.as_mut_ptr(),
                            positions as isize,
                            1,
                        );
                        for (row, values) in scores.chunks_exact_mut(positions).enumerate() {
                            let visible = positions - batch + row + 1;
                            let (valid, masked) = values.split_at_mut(visible);
                            dense_softmax(valid);
                            masked.fill(0.0);
                        }
                        matrixmultiply::sgemm(
                            batch,
                            positions,
                            width,
                            1.0,
                            scores.as_ptr(),
                            positions as isize,
                            1,
                            (vp as *const f32).add(head * head_stride),
                            row_stride as isize,
                            1,
                            0.0,
                            (yp as *mut f32).add(head * width),
                            dim as isize,
                            1,
                        );
                    }
                });
            })
        });
        0
    }))
    .unwrap_or(-3)
}

/// In-place QuickGELU, using the same owned dense pool as adjacent projections.
/// # Safety
/// `values` must address `length` aligned writable floats, with no concurrent accesses.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_dense_quick_gelu(
    handle: u64,
    values: *mut f32,
    length: usize,
) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        if values.is_null()
            || !(values as usize).is_multiple_of(4)
            || length > isize::MAX as usize / 4
        {
            return -1;
        }
        let Some(pool) = POOLS.lock().ok().and_then(|r| r.get(&handle).cloned()) else {
            return -2;
        };
        let ptr = values as usize;
        let chunks = length.div_ceil(4096);
        pool.install(|| {
            (0..chunks).into_par_iter().for_each(|chunk| {
                let start = chunk * 4096;
                let count = (length - start).min(4096);
                let values =
                    unsafe { std::slice::from_raw_parts_mut((ptr as *mut f32).add(start), count) };
                #[cfg(target_arch = "x86_64")]
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    unsafe {
                        quick_gelu_avx2(values);
                    }
                    return;
                }
                for x in values {
                    *x = (*x as f64 / (1.0 + (-1.702 * *x as f64).exp())) as f32;
                }
            })
        });
        0
    }))
    .unwrap_or(-3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quick_gelu_avx2(values: &mut [f32]) {
    use std::arch::x86_64::*;
    let scale = _mm256_set1_ps(-1.702);
    let one = _mm256_set1_ps(1.0);
    let end = values.len() / 8 * 8;
    for offset in (0..end).step_by(8) {
        unsafe {
            let ptr = values.as_mut_ptr().add(offset);
            let x = _mm256_loadu_ps(ptr);
            let denominator =
                _mm256_add_ps(one, super::onnx_kernels::exp8(_mm256_mul_ps(scale, x)));
            _mm256_storeu_ps(ptr, _mm256_div_ps(x, denominator));
        }
    }
    for x in &mut values[end..] {
        *x = (*x as f64 / (1.0 + (-1.702 * *x as f64).exp())) as f32;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn head_major_prefill_preserves_gemm_results_with_unused_capacity() {
        let handle = jmodels_deberta_create(3);
        assert_ne!(handle, 0);
        let (batch, positions, heads, width, capacity) = (8, 13, 3, 16, 17);
        let dim = heads * width;
        let query: Vec<f32> = (0..batch * dim)
            .map(|i| (i % 19) as f32 * 0.1 - 1.0)
            .collect();
        let keys: Vec<f32> = (0..capacity * dim)
            .map(|i| (i % 23) as f32 * 0.09 - 1.0)
            .collect();
        let values: Vec<f32> = (0..keys.len())
            .map(|i| (i % 17) as f32 * 0.11 - 1.0)
            .collect();
        let mut head_keys = vec![0.0; keys.len()];
        let mut head_values = head_keys.clone();
        for row in 0..capacity {
            for head in 0..heads {
                let a = row * dim + head * width;
                let b = (head * capacity + row) * width;
                head_keys[b..b + width].copy_from_slice(&keys[a..a + width]);
                head_values[b..b + width].copy_from_slice(&values[a..a + width]);
            }
        }
        let mut expected = vec![0.0; query.len()];
        let mut actual = expected.clone();
        let old = unsafe {
            jmodels_dense_causal_attention(
                handle,
                query.as_ptr(),
                query.len(),
                keys.as_ptr(),
                keys.len(),
                values.as_ptr(),
                values.len(),
                expected.as_mut_ptr(),
                expected.len(),
                batch,
                positions,
                dim,
                heads,
            )
        };
        let new = unsafe {
            jmodels_dense_causal_attention_head_major(
                handle,
                query.as_ptr(),
                query.len(),
                head_keys.as_ptr(),
                head_keys.len(),
                head_values.as_ptr(),
                head_values.len(),
                actual.as_mut_ptr(),
                actual.len(),
                batch,
                positions,
                dim,
                heads,
                capacity,
            )
        };
        assert_eq!(old, 0);
        assert_eq!(new, 0);
        assert_eq!(actual, expected);
        jmodels_deberta_destroy(handle);
    }

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
