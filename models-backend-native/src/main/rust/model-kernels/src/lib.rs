// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

#[cfg(test)]
use std::cell::Cell;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(all(test, target_arch = "x86_64"))]
std::thread_local! {
    static Q5_0_HORIZONTAL_REDUCTIONS: Cell<usize> = const { Cell::new(0) };
    static Q4_K_BATCH_HORIZONTAL_REDUCTIONS: Cell<usize> = const { Cell::new(0) };
    static F16C_CONVERSIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
std::thread_local! {
    static K_QUANT_WEIGHT_BLOCK_DECODES: Cell<usize> = const { Cell::new(0) };
}

#[inline(always)]
fn record_k_quant_weight_block_decode() {
    #[cfg(test)]
    K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(count.get() + 1));
}

#[inline(always)]
fn record_q4_k_batch_horizontal_reduction() {
    #[cfg(all(test, target_arch = "x86_64"))]
    Q4_K_BATCH_HORIZONTAL_REDUCTIONS.with(|count| count.set(count.get() + 1));
}

const ABI_VERSION: u32 = 2;
const CAPABILITY_Q4_0_F32_BATCHED_MATMUL: u64 = 1;
const CAPABILITY_Q4_0_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 1;
const CAPABILITY_PERSISTENT_WORKER_CONTEXT: u64 = 1 << 2;
const CAPABILITY_Q8_0_F32_BATCHED_MATMUL: u64 = 1 << 3;
const CAPABILITY_Q8_0_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 4;
const CAPABILITY_Q4_K_F32_BATCHED_MATMUL: u64 = 1 << 5;
const CAPABILITY_Q4_K_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 6;
const CAPABILITY_Q6_K_F32_BATCHED_MATMUL: u64 = 1 << 7;
const CAPABILITY_Q6_K_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 8;
const CAPABILITY_MIXED_K_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 9;
const CAPABILITY_Q5_K_F32_BATCHED_MATMUL: u64 = 1 << 10;
const CAPABILITY_Q5_K_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 11;
const CAPABILITY_Q5_0_F32_BATCHED_MATMUL: u64 = 1 << 12;
const CAPABILITY_Q5_0_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 13;
const CAPABILITY_K_QUANT_BATCH_WEIGHT_REUSE: u64 = 1 << 14;
const CAPABILITY_Q4_K_BATCH_VECTOR_ACCUMULATION: u64 = 1 << 15;

const STATUS_OK: i32 = 0;
const STATUS_NULL_POINTER: i32 = 1;
const STATUS_INVALID_SHAPE: i32 = 2;
const STATUS_BUFFER_TOO_SMALL: i32 = 3;
const STATUS_PANIC: i32 = 4;

const QK_0: usize = 32;
const QK_K: usize = 256;
const Q8_K_SUM_BLOCK: usize = 16;
const Q4_0_BLOCK_BYTES: usize = 18;
const Q5_0_BLOCK_BYTES: usize = 22;
const Q8_0_BLOCK_BYTES: usize = 34;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q5_K_BLOCK_BYTES: usize = 176;
const Q6_K_BLOCK_BYTES: usize = 210;
const PARALLEL_OUTPUT_THRESHOLD: usize = 64;
const WORKER_SPIN_ITERS: usize = 4_000;
const COMPLETION_SPIN_ITERS: usize = 4_000;

#[derive(Clone, Copy)]
enum DotKernel {
    Q4,
    #[cfg(target_arch = "x86_64")]
    Q4Avx2,
    Q5,
    #[cfg(target_arch = "x86_64")]
    Q5Avx2,
    Q8,
    #[cfg(target_arch = "x86_64")]
    Q8Avx2,
    Q4K,
    #[cfg(target_arch = "x86_64")]
    Q4KAvx2,
    Q5K,
    #[cfg(target_arch = "x86_64")]
    Q5KAvx2,
    Q6K,
    #[cfg(target_arch = "x86_64")]
    Q6KAvx2,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WeightFormat {
    Q4_0,
    Q5_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActivationFormat {
    Q8_0,
    Q8K,
}

impl WeightFormat {
    fn from_code(code: u32) -> Option<Self> {
        match code {
            0 => Some(Self::Q4_0),
            1 => Some(Self::Q8_0),
            2 => Some(Self::Q4K),
            3 => Some(Self::Q6K),
            4 => Some(Self::Q5K),
            5 => Some(Self::Q5_0),
            _ => None,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => Q4_0_BLOCK_BYTES,
            Self::Q5_0 => Q5_0_BLOCK_BYTES,
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q5K => Q5_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
        }
    }

    fn block_elements(self) -> usize {
        match self {
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 => QK_0,
            Self::Q4K | Self::Q5K | Self::Q6K => QK_K,
        }
    }

    fn activation_format(self) -> ActivationFormat {
        match self {
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 => ActivationFormat::Q8_0,
            Self::Q4K | Self::Q5K | Self::Q6K => ActivationFormat::Q8K,
        }
    }

    fn selected_kernel(self) -> DotKernel {
        match self {
            Self::Q4_0 => selected_q4_kernel(),
            Self::Q5_0 => selected_q5_kernel(),
            Self::Q8_0 => selected_q8_kernel(),
            Self::Q4K => selected_q4_k_kernel(),
            Self::Q5K => selected_q5_k_kernel(),
            Self::Q6K => selected_q6_k_kernel(),
        }
    }
}

pub struct KernelContext {
    workers: WorkerPool,
    scratch: Mutex<ActivationScratch>,
}

#[derive(Default)]
struct ActivationScratch {
    quantized: Vec<i8>,
    scales: Vec<f32>,
    sums: Vec<i16>,
}

impl ActivationScratch {
    fn prepare(&mut self, quantized_elements: usize, scale_elements: usize, sum_elements: usize) {
        self.quantized.resize(quantized_elements, 0);
        self.scales.resize(scale_elements, 0.0);
        self.sums.resize(sum_elements, 0);
    }
}

struct WorkerPool {
    shared: Arc<WorkerShared>,
    workers: Vec<JoinHandle<()>>,
    total_threads: usize,
    execution: Mutex<()>,
}

#[repr(align(128))]
struct CachePadded<T>(T);

struct WorkerShared {
    generation: CachePadded<AtomicU64>,
    remaining: CachePadded<AtomicUsize>,
    state: Mutex<WorkerState>,
    work_available: Condvar,
    work_complete: Condvar,
}

struct WorkerState {
    shutdown: bool,
    job: Option<ParallelJob>,
    failed: bool,
}

#[derive(Clone, Copy)]
struct MatrixJob {
    weights: usize,
    weight_bytes: usize,
    output: usize,
    rows: usize,
    kernel: DotKernel,
}

#[derive(Clone, Copy)]
struct ParallelJob {
    matrices: [Option<MatrixJob>; 3],
    matrix_count: usize,
    quantized: usize,
    quantized_elements: usize,
    activation_scales: usize,
    scale_elements: usize,
    activation_sums: usize,
    sum_elements: usize,
    output_elements: usize,
    batch_size: usize,
    cols: usize,
}

impl WorkerPool {
    fn new(requested_threads: usize) -> Result<Self, ()> {
        let total_threads = requested_threads.max(1);
        let shared = Arc::new(WorkerShared {
            generation: CachePadded(AtomicU64::new(0)),
            remaining: CachePadded(AtomicUsize::new(0)),
            state: Mutex::new(WorkerState {
                shutdown: false,
                job: None,
                failed: false,
            }),
            work_available: Condvar::new(),
            work_complete: Condvar::new(),
        });
        let mut pool = Self {
            shared,
            workers: Vec::with_capacity(total_threads.saturating_sub(1)),
            total_threads,
            execution: Mutex::new(()),
        };
        for worker_index in 1..total_threads {
            let shared = Arc::clone(&pool.shared);
            let worker = thread::Builder::new()
                .name(format!("jmodels-kernel-{worker_index}"))
                .spawn(move || worker_loop(shared, worker_index, total_threads))
                .map_err(|_| ())?;
            pool.workers.push(worker);
        }
        Ok(pool)
    }

    fn execute(&self, job: ParallelJob) -> bool {
        if self.workers.is_empty() || job.output_elements < PARALLEL_OUTPUT_THRESHOLD {
            return catch_unwind(AssertUnwindSafe(|| {
                // SAFETY: the caller owns all job buffers for this synchronous execution.
                unsafe { execute_job_partition(job, 0, 1) }
            }))
            .is_ok();
        }

        let _execution = lock(&self.execution);
        {
            let mut state = lock(&self.shared.state);
            state.job = Some(job);
            state.failed = false;
            self.shared
                .remaining
                .0
                .store(self.workers.len(), Ordering::Relaxed);
            self.shared.generation.0.fetch_add(1, Ordering::Release);
            self.shared.work_available.notify_all();
        }

        let caller_succeeded = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: worker zero receives a range disjoint from every persistent worker.
            unsafe { execute_job_partition(job, 0, self.total_threads) }
        }))
        .is_ok();

        let completed = poll_completion(&self.shared.remaining.0, COMPLETION_SPIN_ITERS);
        let mut state = lock(&self.shared.state);
        while !completed && self.shared.remaining.0.load(Ordering::Acquire) != 0 {
            state = wait(&self.shared.work_complete, state);
        }
        caller_succeeded && !state.failed
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        {
            let mut state = lock(&self.shared.state);
            state.shutdown = true;
            self.shared.generation.0.fetch_add(1, Ordering::Release);
            self.shared.work_available.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(shared: Arc<WorkerShared>, worker_index: usize, total_threads: usize) {
    let mut observed_generation = 0;
    loop {
        let job = {
            let mut next_generation =
                poll_generation(&shared.generation.0, observed_generation, WORKER_SPIN_ITERS);
            let mut state = lock(&shared.state);
            while !state.shutdown && next_generation.is_none() {
                let published_generation = shared.generation.0.load(Ordering::Acquire);
                if published_generation != observed_generation {
                    next_generation = Some(published_generation);
                    break;
                }
                state = wait(&shared.work_available, state);
            }
            if state.shutdown {
                return;
            }
            observed_generation =
                next_generation.unwrap_or_else(|| shared.generation.0.load(Ordering::Acquire));
            state.job.expect("worker generation must provide a job")
        };
        let succeeded = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: every worker receives a distinct output range and read-only shared inputs.
            unsafe { execute_job_partition(job, worker_index, total_threads) }
        }))
        .is_ok();
        if !succeeded {
            lock(&shared.state).failed = true;
        }
        let previous_remaining = shared.remaining.0.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous_remaining > 0);
        if previous_remaining == 1 {
            let _state = lock(&shared.state);
            shared.work_complete.notify_one();
        }
    }
}

fn poll_generation(
    generation: &AtomicU64,
    observed_generation: u64,
    iterations: usize,
) -> Option<u64> {
    for _ in 0..iterations {
        let published_generation = generation.load(Ordering::Acquire);
        if published_generation != observed_generation {
            return Some(published_generation);
        }
        std::hint::spin_loop();
    }
    None
}

fn poll_completion(remaining: &AtomicUsize, iterations: usize) -> bool {
    for _ in 0..iterations {
        if remaining.load(Ordering::Acquire) == 0 {
            return true;
        }
        std::hint::spin_loop();
    }
    false
}

unsafe fn execute_job_partition(job: ParallelJob, worker_index: usize, total_threads: usize) {
    let quantized =
        unsafe { slice::from_raw_parts(job.quantized as *const i8, job.quantized_elements) };
    let activation_scales =
        unsafe { slice::from_raw_parts(job.activation_scales as *const f32, job.scale_elements) };
    let activation_sums =
        unsafe { slice::from_raw_parts(job.activation_sums as *const i16, job.sum_elements) };
    for matrix in job.matrices[..job.matrix_count].iter().flatten() {
        let start_row = matrix.rows * worker_index / total_threads;
        let end_row = matrix.rows * (worker_index + 1) / total_threads;
        if start_row == end_row {
            continue;
        }
        // SAFETY: KernelContext::execute is synchronous and partitions every output matrix by row.
        let weights =
            unsafe { slice::from_raw_parts(matrix.weights as *const u8, matrix.weight_bytes) };
        unsafe {
            compute_batched_row_range(
                weights,
                quantized,
                activation_scales,
                activation_sums,
                matrix.output as *mut f32,
                job.batch_size,
                matrix.rows,
                job.cols,
                start_row,
                end_row,
                matrix.kernel,
            );
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn wait<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    condvar
        .wait(guard)
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[unsafe(no_mangle)]
pub extern "C" fn jmodels_kernels_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn jmodels_kernels_capabilities() -> u64 {
    CAPABILITY_Q4_0_F32_BATCHED_MATMUL
        | CAPABILITY_Q4_0_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_PERSISTENT_WORKER_CONTEXT
        | CAPABILITY_Q8_0_F32_BATCHED_MATMUL
        | CAPABILITY_Q8_0_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_Q4_K_F32_BATCHED_MATMUL
        | CAPABILITY_Q4_K_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_Q6_K_F32_BATCHED_MATMUL
        | CAPABILITY_Q6_K_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_MIXED_K_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_Q5_K_F32_BATCHED_MATMUL
        | CAPABILITY_Q5_K_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_Q5_0_F32_BATCHED_MATMUL
        | CAPABILITY_Q5_0_F32_GROUPED_BATCHED_MATMUL
        | CAPABILITY_K_QUANT_BATCH_WEIGHT_REUSE
        | CAPABILITY_Q4_K_BATCH_VECTOR_ACCUMULATION
}

#[unsafe(no_mangle)]
pub extern "C" fn jmodels_kernels_context_create(thread_count: u32) -> *mut KernelContext {
    match catch_unwind(AssertUnwindSafe(|| {
        let requested_threads = if thread_count == 0 {
            thread::available_parallelism().map_or(1, usize::from)
        } else {
            thread_count as usize
        };
        WorkerPool::new(requested_threads.min(256))
            .map(|workers| {
                Box::into_raw(Box::new(KernelContext {
                    workers,
                    scratch: Mutex::new(ActivationScratch::default()),
                }))
            })
            .unwrap_or(std::ptr::null_mut())
    })) {
        Ok(context) => context,
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be the unique live pointer returned by `jmodels_kernels_context_create`.
pub unsafe extern "C" fn jmodels_kernels_context_destroy(context: *mut KernelContext) -> i32 {
    if context.is_null() {
        return STATUS_NULL_POINTER;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: this function consumes the unique context pointer returned by context_create.
        drop(unsafe { Box::from_raw(context) });
    })) {
        Ok(()) => STATUS_OK,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be live. Every data pointer must remain valid for its advertised length for the
/// synchronous call. Output must be writable and must not alias either input.
pub unsafe extern "C" fn jmodels_quantized_f32_batched_matmul_with_context(
    context: *const KernelContext,
    format: u32,
    weights: *const u8,
    weight_bytes: u64,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    rows: u32,
    cols: u32,
) -> i32 {
    if context.is_null() {
        return STATUS_NULL_POINTER;
    }
    let Some(format) = WeightFormat::from_code(format) else {
        return STATUS_INVALID_SHAPE;
    };
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Java owns the context and all data buffers for this synchronous call.
        let context = unsafe { &*context };
        quantized_f32_batched_matmul(
            Some(context),
            weights,
            weight_bytes,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            rows,
            cols,
            format,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be live and every metadata array must contain `matrix_count` entries. Every data
/// pointer must remain valid for its advertised length, and output must not alias any input.
pub unsafe extern "C" fn jmodels_quantized_f32_grouped_batched_matmul_with_context(
    context: *const KernelContext,
    formats: *const u32,
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    matrix_count: u32,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
) -> i32 {
    if context.is_null() || formats.is_null() {
        return STATUS_NULL_POINTER;
    }
    if !(2..=3).contains(&matrix_count) {
        return STATUS_INVALID_SHAPE;
    }
    // SAFETY: the caller advertises matrix_count entries and the count is bounded above.
    let format_codes = unsafe { slice::from_raw_parts(formats, matrix_count as usize) };
    let Some(formats) = format_codes
        .iter()
        .copied()
        .map(WeightFormat::from_code)
        .collect::<Option<Vec<_>>>()
    else {
        return STATUS_INVALID_SHAPE;
    };
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Java owns the context and all data buffers for this synchronous call.
        let context = unsafe { &*context };
        quantized_f32_mixed_grouped_batched_matmul(
            Some(context),
            weight_pointers,
            weight_bytes,
            rows,
            &formats,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            cols,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// Every pointer must remain valid for its advertised length for the synchronous call. Output must
/// be writable and must not alias either input.
pub unsafe extern "C" fn jmodels_q4_0_f32_batched_matmul(
    weights: *const u8,
    weight_bytes: u64,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    rows: u32,
    cols: u32,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        quantized_f32_batched_matmul(
            None,
            weights,
            weight_bytes,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            rows,
            cols,
            WeightFormat::Q4_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be a live kernel context. Every data pointer must remain valid for its advertised
/// length for the synchronous call. Output must be writable and must not alias either input.
pub unsafe extern "C" fn jmodels_q4_0_f32_batched_matmul_with_context(
    context: *const KernelContext,
    weights: *const u8,
    weight_bytes: u64,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    rows: u32,
    cols: u32,
) -> i32 {
    if context.is_null() {
        return STATUS_NULL_POINTER;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Java owns the context for the duration of this synchronous call.
        let context = unsafe { &*context };
        quantized_f32_batched_matmul(
            Some(context),
            weights,
            weight_bytes,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            rows,
            cols,
            WeightFormat::Q4_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// Every pointer must remain valid for its advertised length for the synchronous call. Output must
/// be writable and must not alias either input.
pub unsafe extern "C" fn jmodels_q8_0_f32_batched_matmul(
    weights: *const u8,
    weight_bytes: u64,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    rows: u32,
    cols: u32,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        quantized_f32_batched_matmul(
            None,
            weights,
            weight_bytes,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            rows,
            cols,
            WeightFormat::Q8_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be a live kernel context. Every data pointer must remain valid for its advertised
/// length for the synchronous call. Output must be writable and must not alias either input.
pub unsafe extern "C" fn jmodels_q8_0_f32_batched_matmul_with_context(
    context: *const KernelContext,
    weights: *const u8,
    weight_bytes: u64,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    rows: u32,
    cols: u32,
) -> i32 {
    if context.is_null() {
        return STATUS_NULL_POINTER;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Java owns the context for the duration of this synchronous call.
        let context = unsafe { &*context };
        quantized_f32_batched_matmul(
            Some(context),
            weights,
            weight_bytes,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            rows,
            cols,
            WeightFormat::Q8_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// Metadata arrays must contain `matrix_count` entries. Every data pointer must remain valid for
/// its advertised length for the synchronous call, and output must not alias any input.
pub unsafe extern "C" fn jmodels_q4_0_f32_grouped_batched_matmul(
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    matrix_count: u32,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        quantized_f32_grouped_batched_matmul(
            None,
            weight_pointers,
            weight_bytes,
            rows,
            matrix_count,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            cols,
            WeightFormat::Q4_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be live, metadata arrays must contain `matrix_count` entries, and every data
/// pointer must remain valid for its advertised length. Output must not alias any input.
pub unsafe extern "C" fn jmodels_q4_0_f32_grouped_batched_matmul_with_context(
    context: *const KernelContext,
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    matrix_count: u32,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
) -> i32 {
    if context.is_null() {
        return STATUS_NULL_POINTER;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Java owns the context for the duration of this synchronous call.
        let context = unsafe { &*context };
        quantized_f32_grouped_batched_matmul(
            Some(context),
            weight_pointers,
            weight_bytes,
            rows,
            matrix_count,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            cols,
            WeightFormat::Q4_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// Metadata arrays must contain `matrix_count` entries. Every data pointer must remain valid for
/// its advertised length for the synchronous call, and output must not alias any input.
pub unsafe extern "C" fn jmodels_q8_0_f32_grouped_batched_matmul(
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    matrix_count: u32,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        quantized_f32_grouped_batched_matmul(
            None,
            weight_pointers,
            weight_bytes,
            rows,
            matrix_count,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            cols,
            WeightFormat::Q8_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[unsafe(no_mangle)]
/// # Safety
///
/// `context` must be live, metadata arrays must contain `matrix_count` entries, and every data
/// pointer must remain valid for its advertised length. Output must not alias any input.
pub unsafe extern "C" fn jmodels_q8_0_f32_grouped_batched_matmul_with_context(
    context: *const KernelContext,
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    matrix_count: u32,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
) -> i32 {
    if context.is_null() {
        return STATUS_NULL_POINTER;
    }
    match catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Java owns the context for the duration of this synchronous call.
        let context = unsafe { &*context };
        quantized_f32_grouped_batched_matmul(
            Some(context),
            weight_pointers,
            weight_bytes,
            rows,
            matrix_count,
            input,
            input_elements,
            output,
            output_elements,
            batch_size,
            cols,
            WeightFormat::Q8_0,
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[allow(clippy::too_many_arguments)]
fn quantized_f32_batched_matmul(
    context: Option<&KernelContext>,
    weights: *const u8,
    weight_bytes: u64,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    rows: u32,
    cols: u32,
    format: WeightFormat,
) -> i32 {
    if weights.is_null() || input.is_null() || output.is_null() {
        return STATUS_NULL_POINTER;
    }
    let (batch_size, rows, cols) = (batch_size as usize, rows as usize, cols as usize);
    let block_elements = format.block_elements();
    if batch_size == 0 || rows == 0 || cols == 0 || cols % block_elements != 0 {
        return STATUS_INVALID_SHAPE;
    }

    let blocks_per_row = cols / block_elements;
    let Some(required_weight_bytes) = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(format.block_bytes()))
    else {
        return STATUS_INVALID_SHAPE;
    };
    let Some(required_input_elements) = batch_size.checked_mul(cols) else {
        return STATUS_INVALID_SHAPE;
    };
    let Some(required_output_elements) = batch_size.checked_mul(rows) else {
        return STATUS_INVALID_SHAPE;
    };
    if weight_bytes < required_weight_bytes as u64
        || input_elements < required_input_elements as u64
        || output_elements < required_output_elements as u64
    {
        return STATUS_BUFFER_TOO_SMALL;
    }

    // SAFETY: pointers are non-null and their advertised lengths were checked above. Java retains
    // every segment for the synchronous duration of this call, and output does not alias inputs.
    let weights = unsafe { slice::from_raw_parts(weights, required_weight_bytes) };
    let input = unsafe { slice::from_raw_parts(input, required_input_elements) };
    let output = unsafe { slice::from_raw_parts_mut(output, required_output_elements) };

    let scale_elements = batch_size * blocks_per_row;
    let sum_elements = activation_sum_elements(format.activation_format(), batch_size, cols);
    let succeeded = if let Some(context) = context {
        let mut scratch = lock(&context.scratch);
        scratch.prepare(required_input_elements, scale_elements, sum_elements);
        compute_with_scratch(
            Some(context),
            weights,
            input,
            output,
            batch_size,
            rows,
            cols,
            format,
            &mut scratch,
        )
    } else {
        let mut scratch = ActivationScratch::default();
        scratch.prepare(required_input_elements, scale_elements, sum_elements);
        compute_with_scratch(
            None,
            weights,
            input,
            output,
            batch_size,
            rows,
            cols,
            format,
            &mut scratch,
        )
    };
    if !succeeded {
        return STATUS_PANIC;
    }
    STATUS_OK
}

#[allow(clippy::too_many_arguments)]
fn compute_with_scratch(
    context: Option<&KernelContext>,
    weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    rows: usize,
    cols: usize,
    format: WeightFormat,
    scratch: &mut ActivationScratch,
) -> bool {
    quantize_activation_batch(
        format.activation_format(),
        input,
        batch_size,
        cols,
        &mut scratch.quantized,
        &mut scratch.scales,
        &mut scratch.sums,
    );
    compute_outputs(
        context,
        weights,
        &scratch.quantized,
        &scratch.scales,
        &scratch.sums,
        batch_size,
        rows,
        cols,
        output,
        format.selected_kernel(),
    )
}

#[allow(clippy::too_many_arguments)]
fn quantized_f32_grouped_batched_matmul(
    context: Option<&KernelContext>,
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    matrix_count: u32,
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
    format: WeightFormat,
) -> i32 {
    let formats = [format; 3];
    let matrix_count_usize = matrix_count as usize;
    if !(2..=3).contains(&matrix_count_usize) {
        return STATUS_INVALID_SHAPE;
    }
    quantized_f32_mixed_grouped_batched_matmul(
        context,
        weight_pointers,
        weight_bytes,
        rows,
        &formats[..matrix_count_usize],
        input,
        input_elements,
        output,
        output_elements,
        batch_size,
        cols,
    )
}

#[allow(clippy::too_many_arguments)]
fn quantized_f32_mixed_grouped_batched_matmul(
    context: Option<&KernelContext>,
    weight_pointers: *const *const u8,
    weight_bytes: *const u64,
    rows: *const u32,
    formats: &[WeightFormat],
    input: *const f32,
    input_elements: u64,
    output: *mut f32,
    output_elements: u64,
    batch_size: u32,
    cols: u32,
) -> i32 {
    if weight_pointers.is_null()
        || weight_bytes.is_null()
        || rows.is_null()
        || input.is_null()
        || output.is_null()
    {
        return STATUS_NULL_POINTER;
    }
    let (matrix_count, batch_size, cols) = (formats.len(), batch_size as usize, cols as usize);
    if !(2..=3).contains(&matrix_count) || batch_size == 0 || cols == 0 {
        return STATUS_INVALID_SHAPE;
    }
    let activation_format = formats[0].activation_format();
    if formats
        .iter()
        .any(|format| format.activation_format() != activation_format)
        || formats
            .iter()
            .any(|format| cols % format.block_elements() != 0)
    {
        return STATUS_INVALID_SHAPE;
    }

    // SAFETY: each metadata array is non-null and the caller advertises matrix_count entries.
    let weight_pointers = unsafe { slice::from_raw_parts(weight_pointers, matrix_count) };
    let weight_bytes = unsafe { slice::from_raw_parts(weight_bytes, matrix_count) };
    let rows = unsafe { slice::from_raw_parts(rows, matrix_count) };
    let Some(required_input_elements) = batch_size.checked_mul(cols) else {
        return STATUS_INVALID_SHAPE;
    };
    let mut required_output_elements = 0_usize;
    for matrix in 0..matrix_count {
        let matrix_rows = rows[matrix] as usize;
        if weight_pointers[matrix].is_null() || matrix_rows == 0 {
            return STATUS_INVALID_SHAPE;
        }
        let blocks_per_row = cols / formats[matrix].block_elements();
        let Some(required_weight_bytes) = matrix_rows
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(formats[matrix].block_bytes()))
        else {
            return STATUS_INVALID_SHAPE;
        };
        if weight_bytes[matrix] < required_weight_bytes as u64 {
            return STATUS_BUFFER_TOO_SMALL;
        }
        let Some(matrix_output_elements) = batch_size.checked_mul(matrix_rows) else {
            return STATUS_INVALID_SHAPE;
        };
        let Some(total_output_elements) =
            required_output_elements.checked_add(matrix_output_elements)
        else {
            return STATUS_INVALID_SHAPE;
        };
        required_output_elements = total_output_elements;
    }
    if input_elements < required_input_elements as u64
        || output_elements < required_output_elements as u64
    {
        return STATUS_BUFFER_TOO_SMALL;
    }

    // SAFETY: data pointers are non-null and their checked lengths fit the caller-advertised
    // buffers. Java retains every segment for this synchronous call and output does not alias.
    let input = unsafe { slice::from_raw_parts(input, required_input_elements) };
    let output = unsafe { slice::from_raw_parts_mut(output, required_output_elements) };
    let activation_blocks = cols / formats[0].block_elements();
    let scale_elements = batch_size * activation_blocks;
    let sum_elements = activation_sum_elements(activation_format, batch_size, cols);
    let succeeded = if let Some(context) = context {
        let mut scratch = lock(&context.scratch);
        scratch.prepare(required_input_elements, scale_elements, sum_elements);
        compute_grouped_with_scratch(
            Some(context),
            weight_pointers,
            rows,
            formats,
            input,
            output,
            batch_size,
            cols,
            activation_format,
            &mut scratch,
        )
    } else {
        let mut scratch = ActivationScratch::default();
        scratch.prepare(required_input_elements, scale_elements, sum_elements);
        compute_grouped_with_scratch(
            None,
            weight_pointers,
            rows,
            formats,
            input,
            output,
            batch_size,
            cols,
            activation_format,
            &mut scratch,
        )
    };
    if !succeeded {
        return STATUS_PANIC;
    }
    STATUS_OK
}

#[allow(clippy::too_many_arguments)]
fn compute_grouped_with_scratch(
    context: Option<&KernelContext>,
    weight_pointers: &[*const u8],
    rows: &[u32],
    formats: &[WeightFormat],
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    cols: usize,
    activation_format: ActivationFormat,
    scratch: &mut ActivationScratch,
) -> bool {
    quantize_activation_batch(
        activation_format,
        input,
        batch_size,
        cols,
        &mut scratch.quantized,
        &mut scratch.scales,
        &mut scratch.sums,
    );

    if let Some(context) = context {
        let mut matrices = [None; 3];
        let mut output_offset = 0;
        for matrix in 0..formats.len() {
            let format = formats[matrix];
            let matrix_rows = rows[matrix] as usize;
            let blocks_per_row = cols / format.block_elements();
            let required_weight_bytes = matrix_rows * blocks_per_row * format.block_bytes();
            let matrix_output_elements = batch_size * matrix_rows;
            matrices[matrix] = Some(MatrixJob {
                weights: weight_pointers[matrix] as usize,
                weight_bytes: required_weight_bytes,
                // SAFETY: output_offset was validated from the sum of every matrix output above.
                output: unsafe { output.as_mut_ptr().add(output_offset) } as usize,
                rows: matrix_rows,
                kernel: format.selected_kernel(),
            });
            output_offset += matrix_output_elements;
        }
        return context.workers.execute(ParallelJob {
            matrices,
            matrix_count: formats.len(),
            quantized: scratch.quantized.as_ptr() as usize,
            quantized_elements: scratch.quantized.len(),
            activation_scales: scratch.scales.as_ptr() as usize,
            scale_elements: scratch.scales.len(),
            activation_sums: scratch.sums.as_ptr() as usize,
            sum_elements: scratch.sums.len(),
            output_elements: output.len(),
            batch_size,
            cols,
        });
    }

    let mut output_offset = 0;
    for matrix in 0..formats.len() {
        let format = formats[matrix];
        let matrix_rows = rows[matrix] as usize;
        let blocks_per_row = cols / format.block_elements();
        let required_weight_bytes = matrix_rows * blocks_per_row * format.block_bytes();
        let matrix_output_elements = batch_size * matrix_rows;
        // SAFETY: each pointer and byte length was validated above.
        let weights =
            unsafe { slice::from_raw_parts(weight_pointers[matrix], required_weight_bytes) };
        if !compute_outputs(
            None,
            weights,
            &scratch.quantized,
            &scratch.scales,
            &scratch.sums,
            batch_size,
            matrix_rows,
            cols,
            &mut output[output_offset..output_offset + matrix_output_elements],
            format.selected_kernel(),
        ) {
            return false;
        }
        output_offset += matrix_output_elements;
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn compute_outputs(
    context: Option<&KernelContext>,
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch_size: usize,
    rows: usize,
    cols: usize,
    output: &mut [f32],
    kernel: DotKernel,
) -> bool {
    if let Some(context) = context {
        return context.workers.execute(ParallelJob {
            matrices: [
                Some(MatrixJob {
                    weights: weights.as_ptr() as usize,
                    weight_bytes: weights.len(),
                    output: output.as_mut_ptr() as usize,
                    rows,
                    kernel,
                }),
                None,
                None,
            ],
            matrix_count: 1,
            quantized: quantized.as_ptr() as usize,
            quantized_elements: quantized.len(),
            activation_scales: activation_scales.as_ptr() as usize,
            scale_elements: activation_scales.len(),
            activation_sums: activation_sums.as_ptr() as usize,
            sum_elements: activation_sums.len(),
            output_elements: output.len(),
            batch_size,
            cols,
        });
    }

    let parallelism = thread::available_parallelism().map_or(1, usize::from);
    let worker_count = parallelism.min(output.len());
    if worker_count < 2 || output.len() < PARALLEL_OUTPUT_THRESHOLD {
        compute_output_range(
            weights,
            quantized,
            activation_scales,
            activation_sums,
            rows,
            cols,
            0,
            output,
            kernel,
        );
        return true;
    }

    let chunk_size = output.len().div_ceil(worker_count);
    thread::scope(|scope| {
        for (chunk_index, output_chunk) in output.chunks_mut(chunk_size).enumerate() {
            let start_index = chunk_index * chunk_size;
            scope.spawn(move || {
                compute_output_range(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    rows,
                    cols,
                    start_index,
                    output_chunk,
                    kernel,
                );
            });
        }
    });
    debug_assert_eq!(output.len(), batch_size * rows);
    true
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_batched_row_range(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
    kernel: DotKernel,
) {
    match kernel {
        DotKernel::Q4 => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_q4_batched_row_range_scalar(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Q4Avx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_q4_batched_row_range_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        DotKernel::Q5 => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_q5_batched_row_range_scalar(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Q5Avx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_q5_batched_row_range_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        DotKernel::Q8 => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_q8_batched_row_range_scalar(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Q8Avx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_q8_batched_row_range_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        DotKernel::Q4K => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_q4_k_batched_row_range_scalar(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Q4KAvx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_q4_k_batched_row_range_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        DotKernel::Q5K => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_q5_k_batched_row_range_scalar(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Q5KAvx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_q5_k_batched_row_range_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        DotKernel::Q6K => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_q6_k_batched_row_range_scalar(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
        #[cfg(target_arch = "x86_64")]
        DotKernel::Q6KAvx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_q6_k_batched_row_range_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    output,
                    batch_size,
                    rows,
                    cols,
                    start_row,
                    end_row,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_q4_batched_row_range_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    let blocks_per_row = cols / QK_0;
    let mut sums = vec![0_f32; batch_size];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            for batch in 0..batch_size {
                let input_offset = batch * cols + block * QK_0;
                let integer_sum = q4_0_q8_0_block_sum_scalar(
                    &weights[weight_offset + 2..],
                    &quantized[input_offset..],
                );
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_q4_batched_row_range_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    let blocks_per_row = cols / QK_0;
    let mut sums = vec![0_f32; batch_size];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
            // SAFETY: every validated Q4_0 block contains 16 packed bytes.
            let signed_weights =
                unsafe { unpack_q4_0_avx2(weights.as_ptr().add(weight_offset + 2)) };
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            for batch in 0..batch_size {
                let input_offset = batch * cols + block * QK_0;
                // SAFETY: every validated Q8_0 activation block contains 32 bytes.
                let integer_sum = unsafe {
                    q4_0_q8_0_signed_block_sum_avx2(
                        signed_weights,
                        quantized.as_ptr().add(input_offset),
                    )
                };
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_q5_batched_row_range_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    let blocks_per_row = cols / QK_0;
    let mut sums = vec![0_f32; batch_size];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q5_0_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            let high_bits = u32::from_le_bytes([
                weights[weight_offset + 2],
                weights[weight_offset + 3],
                weights[weight_offset + 4],
                weights[weight_offset + 5],
            ]);
            for batch in 0..batch_size {
                let input_offset = batch * cols + block * QK_0;
                let integer_sum = q5_0_q8_0_block_sum_scalar(
                    high_bits,
                    &weights[weight_offset + 6..],
                    &quantized[input_offset..],
                );
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn compute_q5_batched_row_range_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            // SAFETY: AVX2, FMA, and F16C are enabled and all buffers were validated.
            let sum = unsafe {
                dot_q5_0_q8_0_row_avx2(weights, quantized, activation_scales, 0, row, cols)
            };
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_0;
    let mut sums = vec![0_f32; batch_size];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q5_0_BLOCK_BYTES;
            // SAFETY: every validated Q5_0 block contains a four-byte high-bit plane followed by
            // 16 packed low-nibble bytes.
            let signed_weights = unsafe {
                unpack_q5_0_avx2(
                    weights.as_ptr().add(weight_offset + 2),
                    weights.as_ptr().add(weight_offset + 6),
                )
            };
            let weight_scale = f16_to_f32_f16c(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            for batch in 0..batch_size {
                let input_offset = batch * cols + block * QK_0;
                // SAFETY: every validated Q8_0 activation block contains 32 bytes.
                let integer_sum = unsafe {
                    q4_0_q8_0_signed_block_sum_avx2(
                        signed_weights,
                        quantized.as_ptr().add(input_offset),
                    )
                };
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_q8_batched_row_range_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    let blocks_per_row = cols / QK_0;
    let mut sums = vec![0_f32; batch_size];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            for batch in 0..batch_size {
                let input_offset = batch * cols + block * QK_0;
                let integer_sum = q8_0_q8_0_block_sum_scalar(
                    &weights[weight_offset + 2..],
                    &quantized[input_offset..],
                );
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
unsafe fn compute_q8_batched_row_range_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    let blocks_per_row = cols / QK_0;
    let mut sums = vec![0_f32; batch_size];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            let weight_offset = (row * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            for batch in 0..batch_size {
                let input_offset = batch * cols + block * QK_0;
                // SAFETY: every validated weight and activation block contains 32 signed bytes.
                let integer_sum = unsafe {
                    q8_0_q8_0_block_sum_avx2(
                        weights.as_ptr().add(weight_offset + 2).cast(),
                        quantized.as_ptr().add(input_offset),
                    )
                };
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = scale.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_q4_k_batched_row_range_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            let sum = dot_q4_k_q8_k_row_scalar(
                weights,
                quantized,
                activation_scales,
                activation_sums,
                0,
                row,
                cols,
            );
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let sums_per_batch = cols / Q8_K_SUM_BLOCK;
    let mut sums = vec![0_f32; batch_size];
    let mut decoded = [0_u8; QK_K];
    let mut group_scales = [0_i32; 8];
    let mut group_mins = [0_i32; 8];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            let weight_min_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset + 2],
                weights[weight_offset + 3],
            ]));
            let scales = &weights[weight_offset + 4..weight_offset + 16];
            let quants = &weights[weight_offset + 16..weight_offset + Q4_K_BLOCK_BYTES];
            for group in 0..8 {
                group_scales[group] = qk_scale(scales, group);
                group_mins[group] = qk_min(scales, group);
                let packed_offset = (group >> 1) * 32;
                let shift = (group & 1) * 4;
                for index in 0..32 {
                    decoded[group * 32 + index] = (quants[packed_offset + index] >> shift) & 0x0f;
                }
            }

            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let sum_offset = batch * sums_per_batch + block * QK_K / Q8_K_SUM_BLOCK;
                let mut quantized_sum = 0_i32;
                let mut minimum_sum = 0_i32;
                for group in 0..8 {
                    let group_activation_offset = activation_offset + group * 32;
                    let mut group_dot = 0_i32;
                    for index in 0..32 {
                        group_dot += decoded[group * 32 + index] as i32
                            * quantized[group_activation_offset + index] as i32;
                    }
                    quantized_sum += group_scales[group] * group_dot;
                    minimum_sum += group_mins[group]
                        * (activation_sums[sum_offset + group * 2] as i32
                            + activation_sums[sum_offset + group * 2 + 1] as i32);
                }
                let activation_scale = activation_scales[batch * blocks_per_row + block];
                sums[batch] =
                    (weight_scale * activation_scale).mul_add(quantized_sum as f32, sums[batch]);
                sums[batch] =
                    (-weight_min_scale * activation_scale).mul_add(minimum_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_q5_k_batched_row_range_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            let sum = dot_q5_k_q8_k_row_scalar(
                weights,
                quantized,
                activation_scales,
                activation_sums,
                0,
                row,
                cols,
            );
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let sums_per_batch = cols / Q8_K_SUM_BLOCK;
    let mut sums = vec![0_f32; batch_size];
    let mut decoded = [0_u8; QK_K];
    let mut group_scales = [0_i32; 8];
    let mut group_mins = [0_i32; 8];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q5_K_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            let weight_min_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset + 2],
                weights[weight_offset + 3],
            ]));
            let scales = &weights[weight_offset + 4..weight_offset + 16];
            let high_bits = &weights[weight_offset + 16..weight_offset + 48];
            let quants = &weights[weight_offset + 48..weight_offset + Q5_K_BLOCK_BYTES];
            for group in 0..8 {
                group_scales[group] = qk_scale(scales, group);
                group_mins[group] = qk_min(scales, group);
                let packed_offset = (group >> 1) * 32;
                let shift = (group & 1) * 4;
                let high_bit = 1_u8 << group;
                for index in 0..32 {
                    decoded[group * 32 + index] = ((quants[packed_offset + index] >> shift) & 0x0f)
                        | if high_bits[index] & high_bit == 0 {
                            0
                        } else {
                            16
                        };
                }
            }

            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let sum_offset = batch * sums_per_batch + block * QK_K / Q8_K_SUM_BLOCK;
                let mut quantized_sum = 0_i32;
                let mut minimum_sum = 0_i32;
                for group in 0..8 {
                    let group_activation_offset = activation_offset + group * 32;
                    let mut group_dot = 0_i32;
                    for index in 0..32 {
                        group_dot += decoded[group * 32 + index] as i32
                            * quantized[group_activation_offset + index] as i32;
                    }
                    quantized_sum += group_scales[group] * group_dot;
                    minimum_sum += group_mins[group]
                        * (activation_sums[sum_offset + group * 2] as i32
                            + activation_sums[sum_offset + group * 2 + 1] as i32);
                }
                let activation_scale = activation_scales[batch * blocks_per_row + block];
                sums[batch] =
                    (weight_scale * activation_scale).mul_add(quantized_sum as f32, sums[batch]);
                sums[batch] =
                    (-weight_min_scale * activation_scale).mul_add(minimum_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compute_q6_k_batched_row_range_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            let sum = dot_q6_k_q8_k_row_scalar(weights, quantized, activation_scales, 0, row, cols);
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let mut lane_sums = vec![[0_f32; 8]; batch_size];
    let mut decoded = [0_i8; QK_K];
    let mut group_scales = [0_i32; 16];
    for row in start_row..end_row {
        lane_sums.fill([0.0; 8]);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
            let weight_scale = f16_to_f32(u16::from_le_bytes([
                weights[weight_offset + 208],
                weights[weight_offset + 209],
            ]));
            let ql = &weights[weight_offset..weight_offset + 128];
            let qh = &weights[weight_offset + 128..weight_offset + 192];
            let scales = &weights[weight_offset + 192..weight_offset + 208];
            for (index, &scale) in scales.iter().enumerate() {
                group_scales[index] = scale as i8 as i32;
            }
            for super_block in 0..2 {
                let ql_base = super_block * 64;
                let qh_base = super_block * 32;
                let quant_base = super_block * 128;
                for index in 0..32 {
                    let ql1 = ql[ql_base + index];
                    let ql2 = ql[ql_base + 32 + index];
                    let high = qh[qh_base + index];
                    decoded[quant_base + index] =
                        (((ql1 & 0x0f) | ((high & 0x03) << 4)) as i32 - 32) as i8;
                    decoded[quant_base + index + 32] =
                        (((ql2 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i32 - 32) as i8;
                    decoded[quant_base + index + 64] =
                        (((ql1 >> 4) | (((high >> 4) & 0x03) << 4)) as i32 - 32) as i8;
                    decoded[quant_base + index + 96] =
                        (((ql2 >> 4) | (((high >> 6) & 0x03) << 4)) as i32 - 32) as i8;
                }
            }

            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let mut integer_sums = [0_i32; 8];
                for super_block in 0..2 {
                    let group_base = super_block * 8;
                    let quant_base = super_block * 128;
                    for index in 0..32 {
                        let scale_index = index / 16;
                        let lane = index & 7;
                        integer_sums[lane] += group_scales[group_base + scale_index]
                            * decoded[quant_base + index] as i32
                            * quantized[activation_offset + quant_base + index] as i32;
                        integer_sums[lane] += group_scales[group_base + scale_index + 2]
                            * decoded[quant_base + index + 32] as i32
                            * quantized[activation_offset + quant_base + index + 32] as i32;
                        integer_sums[lane] += group_scales[group_base + scale_index + 4]
                            * decoded[quant_base + index + 64] as i32
                            * quantized[activation_offset + quant_base + index + 64] as i32;
                        integer_sums[lane] += group_scales[group_base + scale_index + 6]
                            * decoded[quant_base + index + 96] as i32
                            * quantized[activation_offset + quant_base + index + 96] as i32;
                    }
                }
                let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
                for (lane_sum, &integer_sum) in lane_sums[batch].iter_mut().zip(integer_sums.iter())
                {
                    *lane_sum = scale.mul_add(integer_sum as f32, *lane_sum);
                }
            }
        }
        for (batch, batch_lane_sums) in lane_sums.iter().enumerate() {
            let sum = batch_lane_sums.iter().copied().sum();
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn compute_q4_k_batched_row_range_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            // SAFETY: AVX2, FMA, and F16C are enabled and all buffers were validated.
            let sum = unsafe {
                dot_q4_k_q8_k_row_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    0,
                    row,
                    cols,
                )
            };
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let sums_per_batch = cols / Q8_K_SUM_BLOCK;
    let nibble_mask = _mm256_set1_epi8(0x0f);
    let mut sums = vec![0_f32; batch_size];
    let mut decoded = [_mm256_setzero_si256(); 8];
    let mut group_scales = [0_i32; 8];
    let mut group_mins = [0_i32; 8];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
            let weight_scale = f16_to_f32_f16c(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            let weight_min_scale = f16_to_f32_f16c(u16::from_le_bytes([
                weights[weight_offset + 2],
                weights[weight_offset + 3],
            ]));
            let scales = &weights[weight_offset + 4..weight_offset + 16];
            let quants = weights.as_ptr().wrapping_add(weight_offset + 16);
            for pair in 0..4 {
                // SAFETY: every Q4_K block contains four complete 32-byte packed groups.
                let packed = unsafe { _mm256_loadu_si256(quants.add(pair * 32).cast()) };
                decoded[pair * 2] = _mm256_and_si256(packed, nibble_mask);
                decoded[pair * 2 + 1] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
            }
            for group in 0..8 {
                group_scales[group] = qk_scale(scales, group);
                group_mins[group] = qk_min(scales, group);
            }
            let minimum_values = _mm256_setr_epi32(
                group_mins[0],
                group_mins[1],
                group_mins[2],
                group_mins[3],
                group_mins[4],
                group_mins[5],
                group_mins[6],
                group_mins[7],
            );

            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let sum_offset = batch * sums_per_batch + block * QK_K / Q8_K_SUM_BLOCK;
                let mut quantized_lanes = _mm256_setzero_si256();
                for group in 0..8 {
                    // SAFETY: every decoded group and activation group contains 32 bytes.
                    let activation = unsafe {
                        _mm256_loadu_si256(
                            quantized
                                .as_ptr()
                                .add(activation_offset + group * 32)
                                .cast(),
                        )
                    };
                    let products = _mm256_maddubs_epi16(decoded[group], activation);
                    let scale = _mm256_set1_epi16(group_scales[group] as i16);
                    quantized_lanes =
                        _mm256_add_epi32(quantized_lanes, _mm256_madd_epi16(products, scale));
                }
                record_q4_k_batch_horizontal_reduction();
                let quantized_sum = horizontal_sum_i32_avx2(quantized_lanes);
                // SAFETY: each Q8_K activation block has sixteen signed group sums.
                let activation_sum_values =
                    unsafe { _mm256_loadu_si256(activation_sums.as_ptr().add(sum_offset).cast()) };
                let paired_sums = _mm256_madd_epi16(activation_sum_values, _mm256_set1_epi16(1));
                let minimum_sum =
                    horizontal_sum_i32_avx2(_mm256_mullo_epi32(minimum_values, paired_sums));
                let activation_scale = activation_scales[batch * blocks_per_row + block];
                let d = weight_scale * activation_scale;
                let d_min = weight_min_scale * activation_scale;
                sums[batch] = d.mul_add(quantized_sum as f32, sums[batch]);
                sums[batch] = (-d_min).mul_add(minimum_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn compute_q5_k_batched_row_range_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            // SAFETY: AVX2, FMA, and F16C are enabled and all buffers were validated.
            let sum = unsafe {
                dot_q5_k_q8_k_row_avx2(
                    weights,
                    quantized,
                    activation_scales,
                    activation_sums,
                    0,
                    row,
                    cols,
                )
            };
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let sums_per_batch = cols / Q8_K_SUM_BLOCK;
    let nibble_mask = _mm256_set1_epi8(0x0f);
    let high_value = _mm256_set1_epi8(16);
    let mut sums = vec![0_f32; batch_size];
    let mut decoded = [_mm256_setzero_si256(); 8];
    let mut group_scales = [0_i32; 8];
    let mut group_mins = [0_i32; 8];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q5_K_BLOCK_BYTES;
            let weight_scale = f16_to_f32_f16c(u16::from_le_bytes([
                weights[weight_offset],
                weights[weight_offset + 1],
            ]));
            let weight_min_scale = f16_to_f32_f16c(u16::from_le_bytes([
                weights[weight_offset + 2],
                weights[weight_offset + 3],
            ]));
            let scales = &weights[weight_offset + 4..weight_offset + 16];
            // SAFETY: every Q5_K block contains one complete 32-byte high-bit plane.
            let high_bits =
                unsafe { _mm256_loadu_si256(weights.as_ptr().add(weight_offset + 16).cast()) };
            let quants = weights.as_ptr().wrapping_add(weight_offset + 48);
            for pair in 0..4 {
                // SAFETY: every Q5_K block contains four complete 32-byte packed groups.
                let packed = unsafe { _mm256_loadu_si256(quants.add(pair * 32).cast()) };
                let low_nibbles = _mm256_and_si256(packed, nibble_mask);
                let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
                let low_group = pair * 2;
                let high_group = low_group + 1;
                let low_mask = _mm256_set1_epi8((1_u8 << low_group) as i8);
                let high_mask = _mm256_set1_epi8((1_u8 << high_group) as i8);
                let low_extra = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, low_mask), low_mask),
                    high_value,
                );
                let high_extra = _mm256_and_si256(
                    _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, high_mask), high_mask),
                    high_value,
                );
                decoded[low_group] = _mm256_or_si256(low_nibbles, low_extra);
                decoded[high_group] = _mm256_or_si256(high_nibbles, high_extra);
            }
            for group in 0..8 {
                group_scales[group] = qk_scale(scales, group);
                group_mins[group] = qk_min(scales, group);
            }
            let minimum_values = _mm256_setr_epi32(
                group_mins[0],
                group_mins[1],
                group_mins[2],
                group_mins[3],
                group_mins[4],
                group_mins[5],
                group_mins[6],
                group_mins[7],
            );

            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let sum_offset = batch * sums_per_batch + block * QK_K / Q8_K_SUM_BLOCK;
                let mut quantized_lanes = _mm256_setzero_si256();
                for pair in 0..4 {
                    let low_group = pair * 2;
                    let high_group = low_group + 1;
                    let low_activation = quantized
                        .as_ptr()
                        .wrapping_add(activation_offset + pair * 64);
                    let high_activation = low_activation.wrapping_add(32);
                    // SAFETY: every activation group contains 32 signed bytes.
                    let low_values = unsafe { _mm256_loadu_si256(low_activation.cast()) };
                    // SAFETY: every activation group contains 32 signed bytes.
                    let high_values = unsafe { _mm256_loadu_si256(high_activation.cast()) };
                    let low_pairs = _mm256_maddubs_epi16(decoded[low_group], low_values);
                    let high_pairs = _mm256_maddubs_epi16(decoded[high_group], high_values);
                    let low_scale = _mm256_set1_epi16(group_scales[low_group] as i16);
                    let high_scale = _mm256_set1_epi16(group_scales[high_group] as i16);
                    let low_scaled = _mm256_madd_epi16(low_pairs, low_scale);
                    let high_scaled = _mm256_madd_epi16(high_pairs, high_scale);
                    quantized_lanes = _mm256_add_epi32(
                        quantized_lanes,
                        _mm256_add_epi32(low_scaled, high_scaled),
                    );
                }
                let quantized_sum = horizontal_sum_i32_avx2(quantized_lanes);
                // SAFETY: each Q8_K activation block has sixteen signed group sums.
                let activation_sum_values =
                    unsafe { _mm256_loadu_si256(activation_sums.as_ptr().add(sum_offset).cast()) };
                let paired_sums = _mm256_madd_epi16(activation_sum_values, _mm256_set1_epi16(1));
                let minimum_sum =
                    horizontal_sum_i32_avx2(_mm256_mullo_epi32(minimum_values, paired_sums));
                let activation_scale = activation_scales[batch * blocks_per_row + block];
                let d = weight_scale * activation_scale;
                let d_min = weight_min_scale * activation_scale;
                sums[batch] = d.mul_add(quantized_sum as f32, sums[batch]);
                sums[batch] = (-d_min).mul_add(minimum_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn compute_q6_k_batched_row_range_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
) {
    if batch_size == 1 {
        for row in start_row..end_row {
            // SAFETY: AVX2, FMA, and F16C are enabled and all buffers were validated.
            let sum = unsafe {
                dot_q6_k_q8_k_row_avx2(weights, quantized, activation_scales, 0, row, cols)
            };
            // SAFETY: each worker owns this output row.
            unsafe {
                output.add(row).write(sum);
            }
        }
        return;
    }

    let blocks_per_row = cols / QK_K;
    let low_mask = _mm256_set1_epi8(0x0f);
    let high_two_mask = _mm256_set1_epi8(0x03);
    let offset = _mm256_set1_epi8(32);
    let mut sums = vec![0_f32; batch_size];
    let mut decoded = [_mm256_setzero_si256(); 8];
    let mut group_scales = [0_i32; 16];
    for row in start_row..end_row {
        sums.fill(0.0);
        for block in 0..blocks_per_row {
            record_k_quant_weight_block_decode();
            let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
            let weight_scale = f16_to_f32_f16c(u16::from_le_bytes([
                weights[weight_offset + 208],
                weights[weight_offset + 209],
            ]));
            let ql = weights.as_ptr().wrapping_add(weight_offset);
            let qh = ql.wrapping_add(128);
            let scales = &weights[weight_offset + 192..weight_offset + 208];
            for (index, &scale) in scales.iter().enumerate() {
                group_scales[index] = scale as i8 as i32;
            }
            for super_block in 0..2 {
                // SAFETY: every Q6_K super-block contains two 32-byte low-bit groups and one
                // 32-byte high-bit group.
                let low_first = unsafe { _mm256_loadu_si256(ql.add(super_block * 64).cast()) };
                let low_second =
                    unsafe { _mm256_loadu_si256(ql.add(super_block * 64 + 32).cast()) };
                let high = unsafe { _mm256_loadu_si256(qh.add(super_block * 32).cast()) };
                let decoded_offset = super_block * 4;
                decoded[decoded_offset] = _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(low_first, low_mask),
                        _mm256_slli_epi16(_mm256_and_si256(high, high_two_mask), 4),
                    ),
                    offset,
                );
                decoded[decoded_offset + 1] = _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(low_second, low_mask),
                        _mm256_slli_epi16(
                            _mm256_and_si256(_mm256_srli_epi16(high, 2), high_two_mask),
                            4,
                        ),
                    ),
                    offset,
                );
                decoded[decoded_offset + 2] = _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_first, 4), low_mask),
                        _mm256_slli_epi16(
                            _mm256_and_si256(_mm256_srli_epi16(high, 4), high_two_mask),
                            4,
                        ),
                    ),
                    offset,
                );
                decoded[decoded_offset + 3] = _mm256_sub_epi8(
                    _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(low_second, 4), low_mask),
                        _mm256_slli_epi16(
                            _mm256_and_si256(_mm256_srli_epi16(high, 6), high_two_mask),
                            4,
                        ),
                    ),
                    offset,
                );
            }

            for batch in 0..batch_size {
                let activation_offset = batch * cols + block * QK_K;
                let mut integer_sum = 0_i32;
                for super_block in 0..2 {
                    let activation = quantized
                        .as_ptr()
                        .wrapping_add(activation_offset + super_block * 128);
                    let decoded_offset = super_block * 4;
                    // SAFETY: each decoded group and corresponding activation span 32 bytes.
                    let q1_dot =
                        unsafe { signed_dot_halves_avx2(decoded[decoded_offset], activation) };
                    let q2_dot = unsafe {
                        signed_dot_halves_avx2(decoded[decoded_offset + 1], activation.add(32))
                    };
                    let q3_dot = unsafe {
                        signed_dot_halves_avx2(decoded[decoded_offset + 2], activation.add(64))
                    };
                    let q4_dot = unsafe {
                        signed_dot_halves_avx2(decoded[decoded_offset + 3], activation.add(96))
                    };
                    let scale_base = super_block * 8;
                    integer_sum += group_scales[scale_base] * q1_dot.0;
                    integer_sum += group_scales[scale_base + 1] * q1_dot.1;
                    integer_sum += group_scales[scale_base + 2] * q2_dot.0;
                    integer_sum += group_scales[scale_base + 3] * q2_dot.1;
                    integer_sum += group_scales[scale_base + 4] * q3_dot.0;
                    integer_sum += group_scales[scale_base + 5] * q3_dot.1;
                    integer_sum += group_scales[scale_base + 6] * q4_dot.0;
                    integer_sum += group_scales[scale_base + 7] * q4_dot.1;
                }
                let d = weight_scale * activation_scales[batch * blocks_per_row + block];
                sums[batch] = d.mul_add(integer_sum as f32, sums[batch]);
            }
        }
        for (batch, &sum) in sums.iter().enumerate() {
            // SAFETY: each worker owns this row across all batch-major output planes.
            unsafe {
                output.add(batch * rows + row).write(sum);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_output_range(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    rows: usize,
    cols: usize,
    start_index: usize,
    output: &mut [f32],
    kernel: DotKernel,
) {
    for (local_index, slot) in output.iter_mut().enumerate() {
        let output_index = start_index + local_index;
        let batch = output_index / rows;
        let row = output_index % rows;
        *slot = match kernel {
            DotKernel::Q4 => {
                dot_q4_0_q8_0_row_scalar(weights, quantized, activation_scales, batch, row, cols)
            }
            #[cfg(target_arch = "x86_64")]
            DotKernel::Q4Avx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q4_0_q8_0_row_avx2(weights, quantized, activation_scales, batch, row, cols)
                }
            }
            DotKernel::Q5 => {
                dot_q5_0_q8_0_row_scalar(weights, quantized, activation_scales, batch, row, cols)
            }
            #[cfg(target_arch = "x86_64")]
            DotKernel::Q5Avx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q5_0_q8_0_row_avx2(weights, quantized, activation_scales, batch, row, cols)
                }
            }
            DotKernel::Q8 => {
                dot_q8_0_q8_0_row_scalar(weights, quantized, activation_scales, batch, row, cols)
            }
            #[cfg(target_arch = "x86_64")]
            DotKernel::Q8Avx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q8_0_q8_0_row_avx2(weights, quantized, activation_scales, batch, row, cols)
                }
            }
            DotKernel::Q4K => dot_q4_k_q8_k_row_scalar(
                weights,
                quantized,
                activation_scales,
                activation_sums,
                batch,
                row,
                cols,
            ),
            #[cfg(target_arch = "x86_64")]
            DotKernel::Q4KAvx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q4_k_q8_k_row_avx2(
                        weights,
                        quantized,
                        activation_scales,
                        activation_sums,
                        batch,
                        row,
                        cols,
                    )
                }
            }
            DotKernel::Q5K => dot_q5_k_q8_k_row_scalar(
                weights,
                quantized,
                activation_scales,
                activation_sums,
                batch,
                row,
                cols,
            ),
            #[cfg(target_arch = "x86_64")]
            DotKernel::Q5KAvx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q5_k_q8_k_row_avx2(
                        weights,
                        quantized,
                        activation_scales,
                        activation_sums,
                        batch,
                        row,
                        cols,
                    )
                }
            }
            DotKernel::Q6K => {
                dot_q6_k_q8_k_row_scalar(weights, quantized, activation_scales, batch, row, cols)
            }
            #[cfg(target_arch = "x86_64")]
            DotKernel::Q6KAvx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q6_k_q8_k_row_avx2(weights, quantized, activation_scales, batch, row, cols)
                }
            }
        };
    }
}

fn selected_q4_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return DotKernel::Q4Avx2;
    }
    DotKernel::Q4
}

fn selected_q8_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return DotKernel::Q8Avx2;
    }
    DotKernel::Q8
}

fn selected_q5_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        return DotKernel::Q5Avx2;
    }
    DotKernel::Q5
}

fn selected_q4_k_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        return DotKernel::Q4KAvx2;
    }
    DotKernel::Q4K
}

fn selected_q5_k_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        return DotKernel::Q5KAvx2;
    }
    DotKernel::Q5K
}

fn selected_q6_k_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("f16c")
    {
        return DotKernel::Q6KAvx2;
    }
    DotKernel::Q6K
}

fn dot_q4_0_q8_0_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_0;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK_0;
        let integer_sum =
            q4_0_q8_0_block_sum_scalar(&weights[weight_offset + 2..], &quantized[input_offset..]);
        let weight_scale = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ]));
        let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
        sum = scale.mul_add(integer_sum as f32, sum);
    }
    sum
}

#[inline(always)]
fn q4_0_q8_0_block_sum_scalar(packed_weights: &[u8], quantized: &[i8]) -> i32 {
    let mut integer_sum = 0_i32;
    for lane in 0..16 {
        let packed = packed_weights[lane];
        integer_sum += ((packed & 0x0f) as i32 - 8) * quantized[lane] as i32;
        integer_sum += ((packed >> 4) as i32 - 8) * quantized[lane + 16] as i32;
    }
    integer_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_q8_0_row_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_0;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK_0;
        // SAFETY: each validated Q4_0 block provides 16 packed bytes and each Q8 activation block
        // provides 32 bytes. AVX2 and FMA were detected before entering this function.
        let integer_sum = unsafe {
            q4_0_q8_0_block_sum_avx2(
                weights.as_ptr().add(weight_offset + 2),
                quantized.as_ptr().add(input_offset),
            )
        };
        let weight_scale = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ]));
        let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
        sum = scale.mul_add(integer_sum as f32, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4_0_q8_0_block_sum_avx2(packed_weights: *const u8, quantized: *const i8) -> i32 {
    // SAFETY: callers provide one complete Q4_0 block.
    let signed_weights = unsafe { unpack_q4_0_avx2(packed_weights) };
    // SAFETY: callers provide one complete Q8_0 block.
    unsafe { q4_0_q8_0_signed_block_sum_avx2(signed_weights, quantized) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_q4_0_avx2(packed_weights: *const u8) -> __m256i {
    // SAFETY: callers provide the 16 packed bytes in one complete Q4_0 block.
    let packed = unsafe { _mm_loadu_si128(packed_weights.cast()) };
    let nibbles =
        _mm256_inserti128_si256(_mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1);
    _mm256_sub_epi8(
        _mm256_and_si256(nibbles, _mm256_set1_epi8(0x0f)),
        _mm256_set1_epi8(8),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4_0_q8_0_signed_block_sum_avx2(signed_weights: __m256i, quantized: *const i8) -> i32 {
    // SAFETY: callers provide one complete Q8_0 block.
    let pair_sums = unsafe { q4_0_q8_0_signed_pair_sums_avx2(signed_weights, quantized) };
    horizontal_sum_i32_avx2(pair_sums)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q4_0_q8_0_signed_pair_sums_avx2(
    signed_weights: __m256i,
    quantized: *const i8,
) -> __m256i {
    // SAFETY: callers provide one complete Q8_0 block.
    let activations = unsafe { _mm256_loadu_si256(quantized.cast()) };
    let absolute_weights = _mm256_sign_epi8(signed_weights, signed_weights);
    let signed_activations = _mm256_sign_epi8(activations, signed_weights);
    let pair_products = _mm256_maddubs_epi16(absolute_weights, signed_activations);
    _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1))
}

fn dot_q5_0_q8_0_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_0;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q5_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK_0;
        let high_bits = u32::from_le_bytes([
            weights[weight_offset + 2],
            weights[weight_offset + 3],
            weights[weight_offset + 4],
            weights[weight_offset + 5],
        ]);
        let integer_sum = q5_0_q8_0_block_sum_scalar(
            high_bits,
            &weights[weight_offset + 6..],
            &quantized[input_offset..],
        );
        let weight_scale = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ]));
        let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
        sum = scale.mul_add(integer_sum as f32, sum);
    }
    sum
}

#[inline(always)]
fn q5_0_q8_0_block_sum_scalar(high_bits: u32, packed_weights: &[u8], quantized: &[i8]) -> i32 {
    let mut integer_sum = 0_i32;
    for lane in 0..16 {
        let packed = packed_weights[lane];
        let low = ((packed & 0x0f) | (((high_bits >> lane) as u8 & 1) << 4)) as i32 - 16;
        let high = ((packed >> 4) | (((high_bits >> (lane + 16)) as u8 & 1) << 4)) as i32 - 16;
        integer_sum += low * quantized[lane] as i32;
        integer_sum += high * quantized[lane + 16] as i32;
    }
    integer_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q5_0_q8_0_row_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_0;
    let mut sums = _mm256_setzero_ps();
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q5_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK_0;
        // SAFETY: each validated Q5_0 block provides its four-byte high-bit plane and 16 packed
        // low-nibble bytes, and each Q8 activation block provides 32 bytes.
        let signed_weights = unsafe {
            unpack_q5_0_avx2(
                weights.as_ptr().add(weight_offset + 2),
                weights.as_ptr().add(weight_offset + 6),
            )
        };
        let pair_sums = unsafe {
            q4_0_q8_0_signed_pair_sums_avx2(signed_weights, quantized.as_ptr().add(input_offset))
        };
        let weight_scale = f16_to_f32_f16c(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ]));
        let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
        sums = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(pair_sums), sums);
    }
    #[cfg(test)]
    Q5_0_HORIZONTAL_REDUCTIONS.with(|count| count.set(count.get() + 1));
    horizontal_sum_f32_avx2(sums)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
fn f16_to_f32_f16c(value: u16) -> f32 {
    #[cfg(test)]
    F16C_CONVERSIONS.with(|count| count.set(count.get() + 1));
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(value as i32)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_q5_0_avx2(high_bits: *const u8, packed_weights: *const u8) -> __m256i {
    // SAFETY: callers provide the 16 packed bytes and four-byte bit plane from one Q5_0 block.
    let packed = unsafe { _mm_loadu_si128(packed_weights.cast()) };
    let nibbles = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(packed), _mm_srli_epi16(packed, 4), 1),
        _mm256_set1_epi8(0x0f),
    );
    let bit_plane = u32::from_le(unsafe { high_bits.cast::<u32>().read_unaligned() });
    let shuffled_bits = _mm256_shuffle_epi8(
        _mm256_set1_epi32(bit_plane as i32),
        _mm256_set_epi64x(
            0x0303_0303_0303_0303,
            0x0202_0202_0202_0202,
            0x0101_0101_0101_0101,
            0,
        ),
    );
    let clear_high_nibble = _mm256_cmpeq_epi8(
        _mm256_set1_epi64x(-1),
        _mm256_or_si256(
            _mm256_set1_epi64x(0x7fbf_dfef_f7fb_fdfe_u64 as i64),
            shuffled_bits,
        ),
    );
    let signed_high_nibble =
        _mm256_andnot_si256(clear_high_nibble, _mm256_set1_epi8(0xf0_u8 as i8));
    _mm256_or_si256(nibbles, signed_high_nibble)
}

fn dot_q8_0_q8_0_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_0;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK_0;
        let integer_sum =
            q8_0_q8_0_block_sum_scalar(&weights[weight_offset + 2..], &quantized[input_offset..]);
        let weight_scale = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ]));
        let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
        sum = scale.mul_add(integer_sum as f32, sum);
    }
    sum
}

#[inline(always)]
fn q8_0_q8_0_block_sum_scalar(weights: &[u8], quantized: &[i8]) -> i32 {
    let mut integer_sum = 0_i32;
    for lane in 0..QK_0 {
        integer_sum += weights[lane] as i8 as i32 * quantized[lane] as i32;
    }
    integer_sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_q8_0_row_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_0;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK_0;
        // SAFETY: each validated Q8_0 block provides 32 signed bytes.
        let integer_sum = unsafe {
            q8_0_q8_0_block_sum_avx2(
                weights.as_ptr().add(weight_offset + 2).cast(),
                quantized.as_ptr().add(input_offset),
            )
        };
        let weight_scale = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ]));
        let scale = weight_scale * activation_scales[batch * blocks_per_row + block];
        sum = scale.mul_add(integer_sum as f32, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q8_0_q8_0_block_sum_avx2(weights: *const i8, quantized: *const i8) -> i32 {
    // Widen signed bytes before multiplication so every Q8 value, including -128, is exact and
    // adjacent products cannot saturate in 16-bit lanes.
    let weight_low = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(weights.cast()) });
    let weight_high = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(weights.add(16).cast()) });
    let input_low = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(quantized.cast()) });
    let input_high = _mm256_cvtepi8_epi16(unsafe { _mm_loadu_si128(quantized.add(16).cast()) });
    let low_pairs = _mm256_madd_epi16(weight_low, input_low);
    let high_pairs = _mm256_madd_epi16(weight_high, input_high);
    let pair_sums = _mm256_add_epi32(low_pairs, high_pairs);
    let mut lanes = [0_i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), pair_sums) };
    lanes.into_iter().sum()
}

fn dot_q4_k_q8_k_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let scale_offset = batch * blocks_per_row + block;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let d = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ])) * activation_scales[scale_offset];
        let d_min = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset + 2],
            weights[weight_offset + 3],
        ])) * activation_scales[scale_offset];
        let scales = &weights[weight_offset + 4..weight_offset + 16];
        let quants = &weights[weight_offset + 16..weight_offset + Q4_K_BLOCK_BYTES];
        let mut quantized_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for group in 0..8 {
            let group_scale = qk_scale(scales, group);
            let group_min = qk_min(scales, group);
            let packed_offset = (group >> 1) * 32;
            let shift = (group & 1) * 4;
            let group_activation_offset = activation_offset + group * 32;
            let mut group_dot = 0_i32;
            for index in 0..32 {
                let quant = (quants[packed_offset + index] >> shift) & 0x0f;
                group_dot += quant as i32 * quantized[group_activation_offset + index] as i32;
            }
            quantized_sum += group_scale * group_dot;
            minimum_sum += group_min
                * (activation_sums[sum_offset + group * 2] as i32
                    + activation_sums[sum_offset + group * 2 + 1] as i32);
        }
        sum = d.mul_add(quantized_sum as f32, sum);
        sum = (-d_min).mul_add(minimum_sum as f32, sum);
    }
    sum
}

fn dot_q5_k_q8_k_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q5_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let scale_offset = batch * blocks_per_row + block;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let d = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ])) * activation_scales[scale_offset];
        let d_min = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset + 2],
            weights[weight_offset + 3],
        ])) * activation_scales[scale_offset];
        let scales = &weights[weight_offset + 4..weight_offset + 16];
        let high_bits = &weights[weight_offset + 16..weight_offset + 48];
        let quants = &weights[weight_offset + 48..weight_offset + Q5_K_BLOCK_BYTES];
        let mut quantized_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for group in 0..8 {
            let group_scale = qk_scale(scales, group);
            let group_min = qk_min(scales, group);
            let packed_offset = (group >> 1) * 32;
            let shift = (group & 1) * 4;
            let high_bit = 1_u8 << group;
            let group_activation_offset = activation_offset + group * 32;
            let mut group_dot = 0_i32;
            for index in 0..32 {
                let quant = ((quants[packed_offset + index] >> shift) & 0x0f)
                    | if high_bits[index] & high_bit == 0 {
                        0
                    } else {
                        16
                    };
                group_dot += quant as i32 * quantized[group_activation_offset + index] as i32;
            }
            quantized_sum += group_scale * group_dot;
            minimum_sum += group_min
                * (activation_sums[sum_offset + group * 2] as i32
                    + activation_sums[sum_offset + group * 2 + 1] as i32);
        }
        sum = d.mul_add(quantized_sum as f32, sum);
        sum = (-d_min).mul_add(minimum_sum as f32, sum);
    }
    sum
}

fn dot_q6_k_q8_k_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let mut lane_sums = [0_f32; 8];
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let d = f16_to_f32(u16::from_le_bytes([
            weights[weight_offset + 208],
            weights[weight_offset + 209],
        ])) * activation_scales[batch * blocks_per_row + block];
        let ql = &weights[weight_offset..weight_offset + 128];
        let qh = &weights[weight_offset + 128..weight_offset + 192];
        let scales = &weights[weight_offset + 192..weight_offset + 208];
        let mut integer_sums = [0_i32; 8];
        for super_block in 0..2 {
            let ql_base = super_block * 64;
            let qh_base = super_block * 32;
            let scale_base = super_block * 8;
            let quant_base = activation_offset + super_block * 128;
            for index in 0..32 {
                let scale_index = index / 16;
                let ql1 = ql[ql_base + index];
                let ql2 = ql[ql_base + 32 + index];
                let high = qh[qh_base + index];
                let q1 = ((ql1 & 0x0f) | ((high & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql2 & 0x0f) | (((high >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql1 >> 4) | (((high >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql2 >> 4) | (((high >> 6) & 0x03) << 4)) as i32 - 32;
                let s1 = scales[scale_base + scale_index] as i8 as i32;
                let s2 = scales[scale_base + scale_index + 2] as i8 as i32;
                let s3 = scales[scale_base + scale_index + 4] as i8 as i32;
                let s4 = scales[scale_base + scale_index + 6] as i8 as i32;
                let lane = index & 7;
                integer_sums[lane] += s1 * q1 * quantized[quant_base + index] as i32;
                integer_sums[lane] += s2 * q2 * quantized[quant_base + index + 32] as i32;
                integer_sums[lane] += s3 * q3 * quantized[quant_base + index + 64] as i32;
                integer_sums[lane] += s4 * q4 * quantized[quant_base + index + 96] as i32;
            }
        }
        for lane in 0..lane_sums.len() {
            lane_sums[lane] = d.mul_add(integer_sums[lane] as f32, lane_sums[lane]);
        }
    }
    let mut sum = 0_f32;
    for lane_sum in lane_sums {
        sum += lane_sum;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q4_k_q8_k_row_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let nibble_mask = _mm256_set1_epi8(0x0f);
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q4_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let scale_offset = batch * blocks_per_row + block;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let d = f16_to_f32_f16c(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ])) * activation_scales[scale_offset];
        let d_min = f16_to_f32_f16c(u16::from_le_bytes([
            weights[weight_offset + 2],
            weights[weight_offset + 3],
        ])) * activation_scales[scale_offset];
        let scales = &weights[weight_offset + 4..weight_offset + 16];
        let quants = weights.as_ptr().wrapping_add(weight_offset + 16);
        let mut quantized_sum = 0_i32;
        let mut minimum_sum = 0_i32;
        for pair in 0..4 {
            // SAFETY: each Q4_K block contains four complete 32-byte packed groups.
            let packed = unsafe { _mm256_loadu_si256(quants.add(pair * 32).cast()) };
            let low = _mm256_and_si256(packed, nibble_mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
            let low_activation = quantized
                .as_ptr()
                .wrapping_add(activation_offset + pair * 64);
            let high_activation = low_activation.wrapping_add(32);
            // SAFETY: every activation group contains 32 signed bytes.
            let low_dot = unsafe { unsigned_signed_dot_avx2(low, low_activation) };
            // SAFETY: every activation group contains 32 signed bytes.
            let high_dot = unsafe { unsigned_signed_dot_avx2(high, high_activation) };
            quantized_sum += qk_scale(scales, pair * 2) * low_dot;
            quantized_sum += qk_scale(scales, pair * 2 + 1) * high_dot;
        }
        for group in 0..8 {
            minimum_sum += qk_min(scales, group)
                * (activation_sums[sum_offset + group * 2] as i32
                    + activation_sums[sum_offset + group * 2 + 1] as i32);
        }
        sum = d.mul_add(quantized_sum as f32, sum);
        sum = (-d_min).mul_add(minimum_sum as f32, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q5_k_q8_k_row_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    activation_sums: &[i16],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let nibble_mask = _mm256_set1_epi8(0x0f);
    let high_value = _mm256_set1_epi8(16);
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q5_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let scale_offset = batch * blocks_per_row + block;
        let sum_offset = batch * cols / Q8_K_SUM_BLOCK + block * QK_K / Q8_K_SUM_BLOCK;
        let d = f16_to_f32_f16c(u16::from_le_bytes([
            weights[weight_offset],
            weights[weight_offset + 1],
        ])) * activation_scales[scale_offset];
        let d_min = f16_to_f32_f16c(u16::from_le_bytes([
            weights[weight_offset + 2],
            weights[weight_offset + 3],
        ])) * activation_scales[scale_offset];
        let scales = &weights[weight_offset + 4..weight_offset + 16];
        // SAFETY: every Q5_K block contains one complete 32-byte high-bit plane.
        let high_bits =
            unsafe { _mm256_loadu_si256(weights.as_ptr().add(weight_offset + 16).cast()) };
        let quants = weights.as_ptr().wrapping_add(weight_offset + 48);
        let mut quantized_lanes = _mm256_setzero_si256();
        for pair in 0..4 {
            // SAFETY: every Q5_K block contains four complete 32-byte packed groups.
            let packed = unsafe { _mm256_loadu_si256(quants.add(pair * 32).cast()) };
            let low_nibbles = _mm256_and_si256(packed, nibble_mask);
            let high_nibbles = _mm256_and_si256(_mm256_srli_epi16(packed, 4), nibble_mask);
            let low_group = pair * 2;
            let high_group = low_group + 1;
            let low_mask = _mm256_set1_epi8((1_u8 << low_group) as i8);
            let high_mask = _mm256_set1_epi8((1_u8 << high_group) as i8);
            let low_extra = _mm256_and_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, low_mask), low_mask),
                high_value,
            );
            let high_extra = _mm256_and_si256(
                _mm256_cmpeq_epi8(_mm256_and_si256(high_bits, high_mask), high_mask),
                high_value,
            );
            let low = _mm256_or_si256(low_nibbles, low_extra);
            let high = _mm256_or_si256(high_nibbles, high_extra);
            let low_activation = quantized
                .as_ptr()
                .wrapping_add(activation_offset + pair * 64);
            let high_activation = low_activation.wrapping_add(32);
            // SAFETY: every activation group contains 32 signed bytes.
            let low_values = unsafe { _mm256_loadu_si256(low_activation.cast()) };
            // SAFETY: every activation group contains 32 signed bytes.
            let high_values = unsafe { _mm256_loadu_si256(high_activation.cast()) };
            let low_pairs = _mm256_maddubs_epi16(low, low_values);
            let high_pairs = _mm256_maddubs_epi16(high, high_values);
            let low_scale = _mm256_set1_epi16(qk_scale(scales, low_group) as i16);
            let high_scale = _mm256_set1_epi16(qk_scale(scales, high_group) as i16);
            let low_scaled = _mm256_madd_epi16(low_pairs, low_scale);
            let high_scaled = _mm256_madd_epi16(high_pairs, high_scale);
            quantized_lanes =
                _mm256_add_epi32(quantized_lanes, _mm256_add_epi32(low_scaled, high_scaled));
        }
        let quantized_sum = horizontal_sum_i32_avx2(quantized_lanes);
        // SAFETY: each Q8_K activation block has sixteen signed group sums.
        let activation_sum_values =
            unsafe { _mm256_loadu_si256(activation_sums.as_ptr().add(sum_offset).cast()) };
        let paired_sums = _mm256_madd_epi16(activation_sum_values, _mm256_set1_epi16(1));
        let minimum_values = _mm256_setr_epi32(
            qk_min(scales, 0),
            qk_min(scales, 1),
            qk_min(scales, 2),
            qk_min(scales, 3),
            qk_min(scales, 4),
            qk_min(scales, 5),
            qk_min(scales, 6),
            qk_min(scales, 7),
        );
        let minimum_sum = horizontal_sum_i32_avx2(_mm256_mullo_epi32(minimum_values, paired_sums));
        sum = d.mul_add(quantized_sum as f32, sum);
        sum = (-d_min).mul_add(minimum_sum as f32, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q6_k_q8_k_row_avx2(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK_K;
    let low_mask = _mm256_set1_epi8(0x0f);
    let high_two_mask = _mm256_set1_epi8(0x03);
    let offset = _mm256_set1_epi8(32);
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        record_k_quant_weight_block_decode();
        let weight_offset = (row * blocks_per_row + block) * Q6_K_BLOCK_BYTES;
        let activation_offset = batch * cols + block * QK_K;
        let d = f16_to_f32_f16c(u16::from_le_bytes([
            weights[weight_offset + 208],
            weights[weight_offset + 209],
        ])) * activation_scales[batch * blocks_per_row + block];
        let ql = weights.as_ptr().wrapping_add(weight_offset);
        let qh = ql.wrapping_add(128);
        let scales = &weights[weight_offset + 192..weight_offset + 208];
        let mut integer_sum = 0_i32;
        for super_block in 0..2 {
            // SAFETY: every Q6_K super-block contains two 32-byte low-bit groups and one
            // 32-byte high-bit group.
            let low_first = unsafe { _mm256_loadu_si256(ql.add(super_block * 64).cast()) };
            let low_second = unsafe { _mm256_loadu_si256(ql.add(super_block * 64 + 32).cast()) };
            let high = unsafe { _mm256_loadu_si256(qh.add(super_block * 32).cast()) };
            let q1 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(low_first, low_mask),
                    _mm256_slli_epi16(_mm256_and_si256(high, high_two_mask), 4),
                ),
                offset,
            );
            let q2 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(low_second, low_mask),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(high, 2), high_two_mask),
                        4,
                    ),
                ),
                offset,
            );
            let q3 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(low_first, 4), low_mask),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(high, 4), high_two_mask),
                        4,
                    ),
                ),
                offset,
            );
            let q4 = _mm256_sub_epi8(
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16(low_second, 4), low_mask),
                    _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srli_epi16(high, 6), high_two_mask),
                        4,
                    ),
                ),
                offset,
            );
            let activation = quantized
                .as_ptr()
                .wrapping_add(activation_offset + super_block * 128);
            // SAFETY: each Q6 group and corresponding activation span 32 signed bytes.
            let q1_dot = unsafe { signed_dot_halves_avx2(q1, activation) };
            let q2_dot = unsafe { signed_dot_halves_avx2(q2, activation.add(32)) };
            let q3_dot = unsafe { signed_dot_halves_avx2(q3, activation.add(64)) };
            let q4_dot = unsafe { signed_dot_halves_avx2(q4, activation.add(96)) };
            let scale_base = super_block * 8;
            integer_sum += scales[scale_base] as i8 as i32 * q1_dot.0;
            integer_sum += scales[scale_base + 1] as i8 as i32 * q1_dot.1;
            integer_sum += scales[scale_base + 2] as i8 as i32 * q2_dot.0;
            integer_sum += scales[scale_base + 3] as i8 as i32 * q2_dot.1;
            integer_sum += scales[scale_base + 4] as i8 as i32 * q3_dot.0;
            integer_sum += scales[scale_base + 5] as i8 as i32 * q3_dot.1;
            integer_sum += scales[scale_base + 6] as i8 as i32 * q4_dot.0;
            integer_sum += scales[scale_base + 7] as i8 as i32 * q4_dot.1;
        }
        sum = d.mul_add(integer_sum as f32, sum);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unsigned_signed_dot_avx2(unsigned: __m256i, signed: *const i8) -> i32 {
    // SAFETY: callers provide 32 signed activation bytes.
    let signed = unsafe { _mm256_loadu_si256(signed.cast()) };
    let products = _mm256_maddubs_epi16(unsigned, signed);
    let sums = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    horizontal_sum_i32_avx2(sums)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn signed_dot_halves_avx2(weights: __m256i, activations: *const i8) -> (i32, i32) {
    // SAFETY: callers provide 32 signed activation bytes.
    let activations = unsafe { _mm256_loadu_si256(activations.cast()) };
    let absolute_weights = _mm256_sign_epi8(weights, weights);
    let signed_activations = _mm256_sign_epi8(activations, weights);
    let products = _mm256_maddubs_epi16(absolute_weights, signed_activations);
    let sums = _mm256_madd_epi16(products, _mm256_set1_epi16(1));
    let mut lanes = [0_i32; 8];
    // SAFETY: lanes provides one complete 256-bit destination.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), sums) };
    (lanes[..4].iter().sum(), lanes[4..].iter().sum())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn horizontal_sum_i32_avx2(values: __m256i) -> i32 {
    let mut lanes = [0_i32; 8];
    // SAFETY: lanes provides one complete 256-bit destination.
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), values) };
    lanes.into_iter().sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn horizontal_sum_f32_avx2(values: __m256) -> f32 {
    let mut sum = _mm256_extractf128_ps(values, 1);
    sum = _mm_add_ps(sum, _mm256_castps256_ps128(values));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

#[inline(always)]
fn qk_scale(scales: &[u8], group: usize) -> i32 {
    if group < 4 {
        return (scales[group] & 0x3f) as i32;
    }
    let low = scales[group + 4] & 0x0f;
    let high = scales[group - 4] >> 6;
    (low | (high << 4)) as i32
}

#[inline(always)]
fn qk_min(scales: &[u8], group: usize) -> i32 {
    if group < 4 {
        return (scales[group + 4] & 0x3f) as i32;
    }
    let low = scales[group + 4] >> 4;
    let high = scales[group] >> 6;
    (low | (high << 4)) as i32
}

fn activation_sum_elements(format: ActivationFormat, batch_size: usize, cols: usize) -> usize {
    match format {
        ActivationFormat::Q8_0 => 0,
        ActivationFormat::Q8K => batch_size * cols / Q8_K_SUM_BLOCK,
    }
}

fn quantize_activation_batch(
    format: ActivationFormat,
    input: &[f32],
    batch_size: usize,
    cols: usize,
    quantized: &mut [i8],
    scales: &mut [f32],
    sums: &mut [i16],
) {
    match format {
        ActivationFormat::Q8_0 => {
            quantize_q8_0_batch(input, batch_size, cols, quantized, scales);
        }
        ActivationFormat::Q8K => {
            quantize_q8_k_batch(input, batch_size, cols, quantized, scales, sums);
        }
    }
}

fn quantize_q8_k_batch(
    input: &[f32],
    batch_size: usize,
    cols: usize,
    quantized: &mut [i8],
    scales: &mut [f32],
    sums: &mut [i16],
) {
    let blocks_per_row = cols / QK_K;
    let sums_per_row = cols / Q8_K_SUM_BLOCK;
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let input_offset = batch * cols + block * QK_K;
            let values = &input[input_offset..input_offset + QK_K];
            let mut max = 0_f32;
            let mut absolute_max = 0_f32;
            for &value in values {
                let absolute = value.abs();
                if absolute > absolute_max {
                    absolute_max = absolute;
                    max = value;
                }
            }
            if absolute_max == 0.0 {
                quantized[input_offset..input_offset + QK_K].fill(0);
                let sum_offset = batch * sums_per_row + block * QK_K / Q8_K_SUM_BLOCK;
                sums[sum_offset..sum_offset + QK_K / Q8_K_SUM_BLOCK].fill(0);
                scales[batch * blocks_per_row + block] = 0.0;
                continue;
            }
            let inverse_scale = -127.0 / max;
            let mut block_sum = 0_i32;
            for (index, &value) in values.iter().enumerate() {
                let quant = ggml_nearest_int(inverse_scale * value).min(127) as i8;
                quantized[input_offset + index] = quant;
                block_sum += quant as i32;
                if (index + 1) % Q8_K_SUM_BLOCK == 0 {
                    let sum_index = batch * sums_per_row
                        + block * QK_K / Q8_K_SUM_BLOCK
                        + index / Q8_K_SUM_BLOCK;
                    sums[sum_index] = block_sum as i16;
                    block_sum = 0;
                }
            }
            scales[batch * blocks_per_row + block] = 1.0 / inverse_scale;
        }
    }
}

fn quantize_q8_0_batch(
    input: &[f32],
    batch_size: usize,
    cols: usize,
    quantized: &mut [i8],
    scales: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected and all slices cover batch_size * cols elements.
        unsafe {
            quantize_q8_0_batch_avx2(input, batch_size, cols, quantized, scales);
        }
        return;
    }

    quantize_q8_0_batch_scalar(input, batch_size, cols, quantized, scales);
}

fn quantize_q8_0_batch_scalar(
    input: &[f32],
    batch_size: usize,
    cols: usize,
    quantized: &mut [i8],
    scales: &mut [f32],
) {
    let blocks_per_row = cols / QK_0;
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let input_offset = batch * cols + block * QK_0;
            let values = &input[input_offset..input_offset + QK_0];
            let mut absolute_max = 0_f32;
            for &value in values {
                absolute_max = absolute_max.max(value.abs());
            }
            let inverse_scale = if absolute_max == 0.0 {
                0.0
            } else {
                127.0 / absolute_max
            };
            scales[batch * blocks_per_row + block] = f16_to_f32(f32_to_f16(absolute_max / 127.0));
            for (lane, &value) in values.iter().enumerate() {
                quantized[input_offset + lane] = ggml_nearest_int(value * inverse_scale) as i8;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_q8_0_batch_avx2(
    input: &[f32],
    batch_size: usize,
    cols: usize,
    quantized: &mut [i8],
    scales: &mut [f32],
) {
    let blocks_per_row = cols / QK_0;
    let sign_bit = _mm256_set1_ps(-0.0);
    let pack_permutation = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let input_offset = batch * cols + block * QK_0;
            // SAFETY: every validated block contains exactly 32 contiguous floats.
            let values = unsafe { input.as_ptr().add(input_offset) };
            let v0 = unsafe { _mm256_loadu_ps(values) };
            let v1 = unsafe { _mm256_loadu_ps(values.add(8)) };
            let v2 = unsafe { _mm256_loadu_ps(values.add(16)) };
            let v3 = unsafe { _mm256_loadu_ps(values.add(24)) };
            let max01 = _mm256_max_ps(
                _mm256_andnot_ps(sign_bit, v0),
                _mm256_andnot_ps(sign_bit, v1),
            );
            let max23 = _mm256_max_ps(
                _mm256_andnot_ps(sign_bit, v2),
                _mm256_andnot_ps(sign_bit, v3),
            );
            let max_values = _mm256_max_ps(max01, max23);
            let max128 = _mm_max_ps(
                _mm256_castps256_ps128(max_values),
                _mm256_extractf128_ps(max_values, 1),
            );
            let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
            let max32 = _mm_max_ss(max64, _mm_movehdup_ps(max64));
            let absolute_max = _mm_cvtss_f32(max32);
            let inverse_scale = if absolute_max == 0.0 {
                0.0
            } else {
                127.0 / absolute_max
            };
            scales[batch * blocks_per_row + block] = f16_to_f32(f32_to_f16(absolute_max / 127.0));

            let multiplier = _mm256_set1_ps(inverse_scale);
            let q0 = _mm256_cvtps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(_mm256_mul_ps(v0, multiplier)));
            let q1 = _mm256_cvtps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(_mm256_mul_ps(v1, multiplier)));
            let q2 = _mm256_cvtps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(_mm256_mul_ps(v2, multiplier)));
            let q3 = _mm256_cvtps_epi32(_mm256_round_ps::<
                { _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC },
            >(_mm256_mul_ps(v3, multiplier)));
            let q01 = _mm256_packs_epi32(q0, q1);
            let q23 = _mm256_packs_epi32(q2, q3);
            let packed =
                _mm256_permutevar8x32_epi32(_mm256_packs_epi16(q01, q23), pack_permutation);
            // SAFETY: the destination block contains 32 bytes.
            unsafe {
                _mm256_storeu_si256(quantized.as_mut_ptr().add(input_offset).cast(), packed);
            }
        }
    }
}

fn ggml_nearest_int(value: f32) -> i32 {
    let bits = (value + 12_582_912.0).to_bits();
    ((bits & 0x007f_ffff) as i32) - 0x0040_0000
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let significand = (value & 0x03ff) as u32;
    if exponent == 0 {
        if significand == 0 {
            return f32::from_bits(sign);
        }
        let magnitude = significand as f32 * f32::from_bits(0x3380_0000);
        return if sign == 0 { magnitude } else { -magnitude };
    }
    let bits = match exponent {
        31 => sign | 0x7f80_0000 | (significand << 13),
        _ => sign | (((exponent - 15 + 127) as u32) << 23) | (significand << 13),
    };
    f32::from_bits(bits)
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let significand = bits & 0x007f_ffff;

    if exponent == 0xff {
        let payload = (significand >> 13) as u16;
        return sign
            | 0x7c00
            | if payload == 0 && significand != 0 {
                1
            } else {
                payload
            };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let normalized = significand | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let mut rounded = normalized >> shift;
        let remainder_mask = (1_u32 << shift) - 1;
        let remainder = normalized & remainder_mask;
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
            rounded += 1;
        }
        return sign | rounded as u16;
    }

    let mut rounded = significand >> 13;
    let remainder = significand & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
        rounded += 1;
    }
    let mut encoded_exponent = half_exponent as u16;
    if rounded == 0x0400 {
        rounded = 0;
        encoded_exponent += 1;
        if encoded_exponent >= 31 {
            return sign | 0x7c00;
        }
    }
    sign | (encoded_exponent << 10) | rounded as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_stable_abi_and_capabilities() {
        assert_eq!(jmodels_kernels_abi_version(), 2);
        assert_eq!(
            jmodels_kernels_capabilities(),
            CAPABILITY_Q4_0_F32_BATCHED_MATMUL
                | CAPABILITY_Q4_0_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_PERSISTENT_WORKER_CONTEXT
                | CAPABILITY_Q8_0_F32_BATCHED_MATMUL
                | CAPABILITY_Q8_0_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_Q4_K_F32_BATCHED_MATMUL
                | CAPABILITY_Q4_K_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_Q6_K_F32_BATCHED_MATMUL
                | CAPABILITY_Q6_K_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_MIXED_K_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_Q5_K_F32_BATCHED_MATMUL
                | CAPABILITY_Q5_K_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_Q5_0_F32_BATCHED_MATMUL
                | CAPABILITY_Q5_0_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_K_QUANT_BATCH_WEIGHT_REUSE
                | CAPABILITY_Q4_K_BATCH_VECTOR_ACCUMULATION
        );
    }

    #[test]
    fn creates_and_joins_persistent_worker_context() {
        let context = jmodels_kernels_context_create(4);
        assert!(!context.is_null());
        // SAFETY: the test consumes the unique context pointer exactly once.
        assert_eq!(
            unsafe { jmodels_kernels_context_destroy(context) },
            STATUS_OK
        );
    }

    #[test]
    fn worker_poll_observes_a_published_generation_without_parking() {
        let generation = std::sync::atomic::AtomicU64::new(7);

        assert_eq!(poll_generation(&generation, 6, 1), Some(7));
        assert_eq!(poll_generation(&generation, 7, 1), None);
    }

    #[test]
    fn completion_poll_observes_finished_workers_without_parking() {
        let remaining = std::sync::atomic::AtomicUsize::new(1);

        assert!(!poll_completion(&remaining, 1));
        remaining.store(0, Ordering::Release);
        assert!(poll_completion(&remaining, 1));
    }

    #[test]
    fn persistent_context_reuses_activation_workspace() {
        let context = jmodels_kernels_context_create(2);
        assert!(!context.is_null());
        let weights = [0_u8; Q4_K_BLOCK_BYTES];
        let input = [0.25_f32; QK_K];
        let mut output = [0_f32; 1];

        // SAFETY: the context and all buffers remain live and non-aliasing for each call.
        assert_eq!(
            unsafe {
                jmodels_quantized_f32_batched_matmul_with_context(
                    context,
                    2,
                    weights.as_ptr(),
                    weights.len() as u64,
                    input.as_ptr(),
                    input.len() as u64,
                    output.as_mut_ptr(),
                    output.len() as u64,
                    1,
                    1,
                    QK_K as u32,
                )
            },
            STATUS_OK
        );
        // SAFETY: the test owns the live context until the final destroy call.
        let first_pointer = unsafe { lock(&(*context).scratch).quantized.as_ptr() };

        // SAFETY: the context and all buffers remain live and non-aliasing for each call.
        assert_eq!(
            unsafe {
                jmodels_quantized_f32_batched_matmul_with_context(
                    context,
                    2,
                    weights.as_ptr(),
                    weights.len() as u64,
                    input.as_ptr(),
                    input.len() as u64,
                    output.as_mut_ptr(),
                    output.len() as u64,
                    1,
                    1,
                    QK_K as u32,
                )
            },
            STATUS_OK
        );
        // SAFETY: the test owns the live context until the final destroy call.
        let second_pointer = unsafe { lock(&(*context).scratch).quantized.as_ptr() };
        assert_eq!(first_pointer, second_pointer);

        // SAFETY: the test consumes the unique context pointer exactly once.
        assert_eq!(
            unsafe { jmodels_kernels_context_destroy(context) },
            STATUS_OK
        );
    }

    #[test]
    fn grouped_projection_uses_one_worker_generation() {
        let context = jmodels_kernels_context_create(2);
        assert!(!context.is_null());
        let weights: [Vec<u8>; 3] = std::array::from_fn(|_| vec![0_u8; 64 * Q5_0_BLOCK_BYTES]);
        let weight_pointers = [
            weights[0].as_ptr(),
            weights[1].as_ptr(),
            weights[2].as_ptr(),
        ];
        let weight_bytes = [(64 * Q5_0_BLOCK_BYTES) as u64; 3];
        let formats = [5_u32; 3];
        let rows = [64_u32; 3];
        let input = [0.25_f32; QK_0];
        let mut output = [0_f32; 64 * 3];
        // SAFETY: the test owns the live context until the final destroy call.
        let context_ref = unsafe { &*context };
        let before = context_ref
            .workers
            .shared
            .generation
            .0
            .load(Ordering::Acquire);

        // SAFETY: the context and every test buffer remain live and non-aliasing for the call.
        assert_eq!(
            unsafe {
                jmodels_quantized_f32_grouped_batched_matmul_with_context(
                    context,
                    formats.as_ptr(),
                    weight_pointers.as_ptr(),
                    weight_bytes.as_ptr(),
                    rows.as_ptr(),
                    3,
                    input.as_ptr(),
                    input.len() as u64,
                    output.as_mut_ptr(),
                    output.len() as u64,
                    1,
                    QK_0 as u32,
                )
            },
            STATUS_OK
        );
        // SAFETY: the call completed synchronously and the context remains live.
        let after = context_ref
            .workers
            .shared
            .generation
            .0
            .load(Ordering::Acquire);

        assert_eq!(after.wrapping_sub(before), 1);
        // SAFETY: the test consumes the unique context pointer exactly once.
        assert_eq!(
            unsafe { jmodels_kernels_context_destroy(context) },
            STATUS_OK
        );
    }

    #[test]
    fn persistent_workers_complete_repeated_generations() {
        const GENERATIONS: u64 = 256;

        let context = jmodels_kernels_context_create(4);
        assert!(!context.is_null());
        let weights = vec![0_u8; 64 * Q5_0_BLOCK_BYTES];
        let input = [0.25_f32; QK_0];
        let mut output = [0_f32; 64];
        // SAFETY: the test owns the live context until the final destroy call.
        let context_ref = unsafe { &*context };
        let before = context_ref
            .workers
            .shared
            .generation
            .0
            .load(Ordering::Acquire);

        for _ in 0..GENERATIONS {
            output.fill(f32::NAN);
            // SAFETY: the context and every test buffer remain live and non-aliasing for the call.
            assert_eq!(
                unsafe {
                    jmodels_quantized_f32_batched_matmul_with_context(
                        context,
                        5,
                        weights.as_ptr(),
                        weights.len() as u64,
                        input.as_ptr(),
                        input.len() as u64,
                        output.as_mut_ptr(),
                        output.len() as u64,
                        1,
                        64,
                        QK_0 as u32,
                    )
                },
                STATUS_OK
            );
            assert!(output.iter().all(|value| *value == 0.0));
        }

        let after = context_ref
            .workers
            .shared
            .generation
            .0
            .load(Ordering::Acquire);
        assert_eq!(after.wrapping_sub(before), GENERATIONS);
        // SAFETY: the test consumes the unique context pointer exactly once.
        assert_eq!(
            unsafe { jmodels_kernels_context_destroy(context) },
            STATUS_OK
        );
    }

    #[test]
    fn batched_k_quantized_rows_decode_each_weight_block_once() {
        const ROWS: usize = 2;
        const BLOCKS: usize = 2;
        const BATCH_SIZE: usize = 3;
        const COLS: usize = BLOCKS * QK_K;

        let mut quantized = vec![0_i8; BATCH_SIZE * COLS];
        for (index, value) in quantized.iter_mut().enumerate() {
            *value = (((index * 17 + 11) % 255) as i32 - 127) as i8;
        }
        let activation_scales: Vec<f32> = (0..BATCH_SIZE * BLOCKS)
            .map(|index| 0.015625 * (index + 1) as f32)
            .collect();
        let activation_sums: Vec<i16> = quantized
            .chunks_exact(Q8_K_SUM_BLOCK)
            .map(|values| values.iter().map(|&value| value as i16).sum())
            .collect();

        let mut q4_weights = vec![0_u8; ROWS * BLOCKS * Q4_K_BLOCK_BYTES];
        for block in 0..ROWS * BLOCKS {
            let offset = block * Q4_K_BLOCK_BYTES;
            q4_weights[offset..offset + 2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
            q4_weights[offset + 2..offset + 4].copy_from_slice(&f32_to_f16(0.0625).to_le_bytes());
            for (index, value) in q4_weights[offset + 4..offset + Q4_K_BLOCK_BYTES]
                .iter_mut()
                .enumerate()
            {
                *value = (index * 19 + block * 7 + 3) as u8;
            }
        }
        let mut expected_q4 = Vec::with_capacity(BATCH_SIZE * ROWS);
        for batch in 0..BATCH_SIZE {
            for row in 0..ROWS {
                expected_q4.push(dot_q4_k_q8_k_row_scalar(
                    &q4_weights,
                    &quantized,
                    &activation_scales,
                    &activation_sums,
                    batch,
                    row,
                    COLS,
                ));
            }
        }
        let mut actual_q4 = vec![f32::NAN; BATCH_SIZE * ROWS];
        K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(0));
        // SAFETY: every fixture buffer contains the complete validated matrix shape.
        unsafe {
            compute_q4_k_batched_row_range_scalar(
                &q4_weights,
                &quantized,
                &activation_scales,
                &activation_sums,
                actual_q4.as_mut_ptr(),
                BATCH_SIZE,
                ROWS,
                COLS,
                0,
                ROWS,
            );
        }
        assert_eq!(actual_q4, expected_q4);
        K_QUANT_WEIGHT_BLOCK_DECODES
            .with(|count| assert_eq!(count.get(), ROWS * BLOCKS, "Q4_K weight block decodes"));

        let mut q5_weights = vec![0_u8; ROWS * BLOCKS * Q5_K_BLOCK_BYTES];
        for block in 0..ROWS * BLOCKS {
            let offset = block * Q5_K_BLOCK_BYTES;
            q5_weights[offset..offset + 2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
            q5_weights[offset + 2..offset + 4].copy_from_slice(&f32_to_f16(0.0625).to_le_bytes());
            for (index, value) in q5_weights[offset + 4..offset + Q5_K_BLOCK_BYTES]
                .iter_mut()
                .enumerate()
            {
                *value = (index * 23 + block * 11 + 5) as u8;
            }
        }
        let mut expected_q5 = Vec::with_capacity(BATCH_SIZE * ROWS);
        for batch in 0..BATCH_SIZE {
            for row in 0..ROWS {
                expected_q5.push(dot_q5_k_q8_k_row_scalar(
                    &q5_weights,
                    &quantized,
                    &activation_scales,
                    &activation_sums,
                    batch,
                    row,
                    COLS,
                ));
            }
        }
        let mut actual_q5 = vec![f32::NAN; BATCH_SIZE * ROWS];
        K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(0));
        // SAFETY: every fixture buffer contains the complete validated matrix shape.
        unsafe {
            compute_q5_k_batched_row_range_scalar(
                &q5_weights,
                &quantized,
                &activation_scales,
                &activation_sums,
                actual_q5.as_mut_ptr(),
                BATCH_SIZE,
                ROWS,
                COLS,
                0,
                ROWS,
            );
        }
        assert_eq!(actual_q5, expected_q5);
        K_QUANT_WEIGHT_BLOCK_DECODES
            .with(|count| assert_eq!(count.get(), ROWS * BLOCKS, "Q5_K weight block decodes"));

        let mut q6_weights = vec![0_u8; ROWS * BLOCKS * Q6_K_BLOCK_BYTES];
        for block in 0..ROWS * BLOCKS {
            let offset = block * Q6_K_BLOCK_BYTES;
            for (index, value) in q6_weights[offset..offset + 208].iter_mut().enumerate() {
                *value = (index * 29 + block * 13 + 7) as u8;
            }
            q6_weights[offset + 208..offset + Q6_K_BLOCK_BYTES]
                .copy_from_slice(&f32_to_f16(0.03125).to_le_bytes());
        }
        let mut expected_q6 = Vec::with_capacity(BATCH_SIZE * ROWS);
        for batch in 0..BATCH_SIZE {
            for row in 0..ROWS {
                expected_q6.push(dot_q6_k_q8_k_row_scalar(
                    &q6_weights,
                    &quantized,
                    &activation_scales,
                    batch,
                    row,
                    COLS,
                ));
            }
        }
        let mut actual_q6 = vec![f32::NAN; BATCH_SIZE * ROWS];
        K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(0));
        // SAFETY: every fixture buffer contains the complete validated matrix shape.
        unsafe {
            compute_q6_k_batched_row_range_scalar(
                &q6_weights,
                &quantized,
                &activation_scales,
                actual_q6.as_mut_ptr(),
                BATCH_SIZE,
                ROWS,
                COLS,
                0,
                ROWS,
            );
        }
        assert_eq!(actual_q6, expected_q6);
        K_QUANT_WEIGHT_BLOCK_DECODES
            .with(|count| assert_eq!(count.get(), ROWS * BLOCKS, "Q6_K weight block decodes"));

        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            let mut expected_q4_avx2 = Vec::with_capacity(BATCH_SIZE * ROWS);
            let mut expected_q5_avx2 = Vec::with_capacity(BATCH_SIZE * ROWS);
            let mut expected_q6_avx2 = Vec::with_capacity(BATCH_SIZE * ROWS);
            for batch in 0..BATCH_SIZE {
                for row in 0..ROWS {
                    // SAFETY: runtime detection established every required x86 feature and each
                    // fixture buffer contains complete K-quant blocks.
                    unsafe {
                        expected_q4_avx2.push(dot_q4_k_q8_k_row_avx2(
                            &q4_weights,
                            &quantized,
                            &activation_scales,
                            &activation_sums,
                            batch,
                            row,
                            COLS,
                        ));
                        expected_q5_avx2.push(dot_q5_k_q8_k_row_avx2(
                            &q5_weights,
                            &quantized,
                            &activation_scales,
                            &activation_sums,
                            batch,
                            row,
                            COLS,
                        ));
                        expected_q6_avx2.push(dot_q6_k_q8_k_row_avx2(
                            &q6_weights,
                            &quantized,
                            &activation_scales,
                            batch,
                            row,
                            COLS,
                        ));
                    }
                }
            }

            let mut actual_q4_avx2 = vec![f32::NAN; BATCH_SIZE * ROWS];
            K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(0));
            Q4_K_BATCH_HORIZONTAL_REDUCTIONS.with(|count| count.set(0));
            // SAFETY: runtime detection established every required x86 feature and each fixture
            // buffer contains the complete validated matrix shape.
            unsafe {
                compute_q4_k_batched_row_range_avx2(
                    &q4_weights,
                    &quantized,
                    &activation_scales,
                    &activation_sums,
                    actual_q4_avx2.as_mut_ptr(),
                    BATCH_SIZE,
                    ROWS,
                    COLS,
                    0,
                    ROWS,
                );
            }
            assert_eq!(actual_q4_avx2, expected_q4_avx2);
            K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| {
                assert_eq!(count.get(), ROWS * BLOCKS, "Q4_K AVX2 weight block decodes");
            });
            Q4_K_BATCH_HORIZONTAL_REDUCTIONS.with(|count| {
                assert_eq!(
                    count.get(),
                    BATCH_SIZE * ROWS * BLOCKS,
                    "Q4_K AVX2 horizontal reductions"
                );
            });

            let mut actual_q5_avx2 = vec![f32::NAN; BATCH_SIZE * ROWS];
            K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(0));
            // SAFETY: runtime detection established every required x86 feature and each fixture
            // buffer contains the complete validated matrix shape.
            unsafe {
                compute_q5_k_batched_row_range_avx2(
                    &q5_weights,
                    &quantized,
                    &activation_scales,
                    &activation_sums,
                    actual_q5_avx2.as_mut_ptr(),
                    BATCH_SIZE,
                    ROWS,
                    COLS,
                    0,
                    ROWS,
                );
            }
            assert_eq!(actual_q5_avx2, expected_q5_avx2);
            K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| {
                assert_eq!(count.get(), ROWS * BLOCKS, "Q5_K AVX2 weight block decodes");
            });

            let mut actual_q6_avx2 = vec![f32::NAN; BATCH_SIZE * ROWS];
            K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| count.set(0));
            // SAFETY: runtime detection established every required x86 feature and each fixture
            // buffer contains the complete validated matrix shape.
            unsafe {
                compute_q6_k_batched_row_range_avx2(
                    &q6_weights,
                    &quantized,
                    &activation_scales,
                    actual_q6_avx2.as_mut_ptr(),
                    BATCH_SIZE,
                    ROWS,
                    COLS,
                    0,
                    ROWS,
                );
            }
            assert_eq!(actual_q6_avx2, expected_q6_avx2);
            K_QUANT_WEIGHT_BLOCK_DECODES.with(|count| {
                assert_eq!(count.get(), ROWS * BLOCKS, "Q6_K AVX2 weight block decodes");
            });
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_k_quantized_rows_match_scalar_kernels() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
            || !std::arch::is_x86_feature_detected!("f16c")
        {
            return;
        }
        F16C_CONVERSIONS.with(|count| count.set(0));
        let mut quantized = [0_i8; QK_K];
        let mut activation_sums = [0_i16; QK_K / Q8_K_SUM_BLOCK];
        for (index, quant) in quantized.iter_mut().enumerate() {
            *quant = (((index * 17 + 11) % 255) as i32 - 127) as i8;
        }
        for (index, values) in quantized.chunks_exact(Q8_K_SUM_BLOCK).enumerate() {
            activation_sums[index] = values.iter().map(|&value| value as i16).sum();
        }
        let activation_scales = [0.03125_f32];

        let mut q4_weights = [0_u8; Q4_K_BLOCK_BYTES];
        q4_weights[..2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
        q4_weights[2..4].copy_from_slice(&f32_to_f16(0.0625).to_le_bytes());
        for (index, value) in q4_weights[4..16].iter_mut().enumerate() {
            *value = (index * 19 + 7) as u8;
        }
        for (index, value) in q4_weights[16..].iter_mut().enumerate() {
            *value = (index * 23 + 5) as u8;
        }
        let q4_scalar = dot_q4_k_q8_k_row_scalar(
            &q4_weights,
            &quantized,
            &activation_scales,
            &activation_sums,
            0,
            0,
            QK_K,
        );
        // SAFETY: AVX2 was detected and each slice contains one complete validated block.
        let q4_avx2 = unsafe {
            dot_q4_k_q8_k_row_avx2(
                &q4_weights,
                &quantized,
                &activation_scales,
                &activation_sums,
                0,
                0,
                QK_K,
            )
        };
        assert!((q4_avx2 - q4_scalar).abs() <= 1e-4);

        let mut q5_weights = [0_u8; Q5_K_BLOCK_BYTES];
        q5_weights[..2].copy_from_slice(&f32_to_f16(0.125).to_le_bytes());
        q5_weights[2..4].copy_from_slice(&f32_to_f16(0.0625).to_le_bytes());
        for (index, value) in q5_weights[4..].iter_mut().enumerate() {
            *value = (index * 31 + 13) as u8;
        }
        let q5_scalar = dot_q5_k_q8_k_row_scalar(
            &q5_weights,
            &quantized,
            &activation_scales,
            &activation_sums,
            0,
            0,
            QK_K,
        );
        // SAFETY: AVX2 was detected and each slice contains one complete validated block.
        let q5_avx2 = unsafe {
            dot_q5_k_q8_k_row_avx2(
                &q5_weights,
                &quantized,
                &activation_scales,
                &activation_sums,
                0,
                0,
                QK_K,
            )
        };
        assert!((q5_avx2 - q5_scalar).abs() <= 1e-4);

        let mut q6_weights = [0_u8; Q6_K_BLOCK_BYTES];
        for (index, value) in q6_weights[..208].iter_mut().enumerate() {
            *value = (index * 29 + 3) as u8;
        }
        q6_weights[208..].copy_from_slice(&f32_to_f16(0.03125).to_le_bytes());
        let q6_scalar =
            dot_q6_k_q8_k_row_scalar(&q6_weights, &quantized, &activation_scales, 0, 0, QK_K);
        // SAFETY: AVX2 was detected and each slice contains one complete validated block.
        let q6_avx2 = unsafe {
            dot_q6_k_q8_k_row_avx2(&q6_weights, &quantized, &activation_scales, 0, 0, QK_K)
        };
        assert!((q6_avx2 - q6_scalar).abs() <= 1e-4);
        F16C_CONVERSIONS.with(|count| assert_eq!(count.get(), 5));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q5_0_bit_plane_matches_scalar_kernel() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut weights = [0_u8; Q5_0_BLOCK_BYTES];
        let high_bits = 0xa5c3_5a7f_u32;
        weights[2..6].copy_from_slice(&high_bits.to_le_bytes());
        for (index, packed) in weights[6..].iter_mut().enumerate() {
            *packed = (index * 37 + 11) as u8;
        }
        let mut quantized = [0_i8; QK_0];
        for (index, quant) in quantized.iter_mut().enumerate() {
            *quant = (((index * 17 + 5) % 255) as i32 - 127) as i8;
        }

        let scalar = q5_0_q8_0_block_sum_scalar(high_bits, &weights[6..], &quantized);
        // SAFETY: AVX2 was detected and the arrays contain complete Q5_0 and Q8_0 blocks.
        let avx2 = unsafe {
            q4_0_q8_0_signed_block_sum_avx2(
                unpack_q5_0_avx2(weights.as_ptr().add(2), weights.as_ptr().add(6)),
                quantized.as_ptr(),
            )
        };

        assert_eq!(avx2, scalar);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_q5_0_row_reduces_simd_lanes_once() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
            || !std::arch::is_x86_feature_detected!("f16c")
        {
            return;
        }
        const BLOCKS: usize = 8;
        let cols = BLOCKS * QK_0;
        let mut weights = vec![0_u8; BLOCKS * Q5_0_BLOCK_BYTES];
        for block in 0..BLOCKS {
            let offset = block * Q5_0_BLOCK_BYTES;
            weights[offset..offset + 2]
                .copy_from_slice(&f32_to_f16(0.015625 * (block + 1) as f32).to_le_bytes());
            weights[offset + 2..offset + 6]
                .copy_from_slice(&(0xa5c3_5a7f_u32.rotate_left(block as u32)).to_le_bytes());
            for (index, packed) in weights[offset + 6..offset + Q5_0_BLOCK_BYTES]
                .iter_mut()
                .enumerate()
            {
                *packed = (index * 37 + block * 11 + 5) as u8;
            }
        }
        let mut quantized = vec![0_i8; cols];
        for (index, quant) in quantized.iter_mut().enumerate() {
            *quant = (((index * 17 + 5) % 255) as i32 - 127) as i8;
        }
        let activation_scales: Vec<f32> = (0..BLOCKS)
            .map(|block| 0.0078125 * (block + 1) as f32)
            .collect();

        let scalar = dot_q5_0_q8_0_row_scalar(&weights, &quantized, &activation_scales, 0, 0, cols);
        Q5_0_HORIZONTAL_REDUCTIONS.with(|count| count.set(0));
        F16C_CONVERSIONS.with(|count| count.set(0));
        // SAFETY: AVX2 was detected and each buffer contains eight complete validated blocks.
        let avx2 =
            unsafe { dot_q5_0_q8_0_row_avx2(&weights, &quantized, &activation_scales, 0, 0, cols) };

        assert!((avx2 - scalar).abs() <= 1e-3);
        Q5_0_HORIZONTAL_REDUCTIONS.with(|count| assert_eq!(count.get(), 1));
        F16C_CONVERSIONS.with(|count| assert_eq!(count.get(), BLOCKS));
    }

    #[test]
    fn rejects_invalid_dimensions_without_dereferencing_buffers() {
        let weights = [0_u8; Q4_0_BLOCK_BYTES];
        let input = [0_f32; QK_0];
        let mut output = [0_f32; 1];
        // SAFETY: all test arrays remain live and non-aliasing for the synchronous call.
        let status = unsafe {
            jmodels_q4_0_f32_batched_matmul(
                weights.as_ptr(),
                weights.len() as u64,
                input.as_ptr(),
                input.len() as u64,
                output.as_mut_ptr(),
                output.len() as u64,
                1,
                1,
                31,
            )
        };
        assert_eq!(status, STATUS_INVALID_SHAPE);
    }

    #[test]
    fn half_round_trip_matches_known_values() {
        for (value, encoded) in [
            (0.0_f32, 0x0000_u16),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (0.125, 0x3000),
            (65_504.0, 0x7bff),
        ] {
            assert_eq!(f32_to_f16(value), encoded);
            assert_eq!(f16_to_f32(encoded), value);
        }
    }

    #[test]
    fn half_conversion_preserves_subnormal_boundaries() {
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
        assert_eq!(f16_to_f32(0x03ff).to_bits(), 0x387f_c000);
        assert_eq!(f16_to_f32(0x8001).to_bits(), 0xb380_0000);
        assert_eq!(f16_to_f32(0x83ff).to_bits(), 0xb87f_c000);
    }
}
