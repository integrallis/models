// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::slice;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const ABI_VERSION: u32 = 1;
const CAPABILITY_Q4_0_F32_BATCHED_MATMUL: u64 = 1;
const CAPABILITY_Q4_0_F32_GROUPED_BATCHED_MATMUL: u64 = 1 << 1;
const CAPABILITY_PERSISTENT_WORKER_CONTEXT: u64 = 1 << 2;

const STATUS_OK: i32 = 0;
const STATUS_NULL_POINTER: i32 = 1;
const STATUS_INVALID_SHAPE: i32 = 2;
const STATUS_BUFFER_TOO_SMALL: i32 = 3;
const STATUS_PANIC: i32 = 4;

const QK: usize = 32;
const Q4_0_BLOCK_BYTES: usize = 18;
const PARALLEL_OUTPUT_THRESHOLD: usize = 64;

#[derive(Clone, Copy)]
enum Q4Kernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
}

pub struct KernelContext {
    workers: WorkerPool,
}

struct WorkerPool {
    shared: Arc<WorkerShared>,
    workers: Vec<JoinHandle<()>>,
    total_threads: usize,
    execution: Mutex<()>,
}

struct WorkerShared {
    state: Mutex<WorkerState>,
    work_available: Condvar,
    work_complete: Condvar,
}

struct WorkerState {
    generation: u64,
    shutdown: bool,
    job: Option<ParallelJob>,
    remaining: usize,
    failed: bool,
}

#[derive(Clone, Copy)]
struct ParallelJob {
    weights: usize,
    weight_bytes: usize,
    quantized: usize,
    quantized_elements: usize,
    activation_scales: usize,
    scale_elements: usize,
    output: usize,
    output_elements: usize,
    batch_size: usize,
    rows: usize,
    cols: usize,
    kernel: Q4Kernel,
}

impl WorkerPool {
    fn new(requested_threads: usize) -> Result<Self, ()> {
        let total_threads = requested_threads.max(1);
        let shared = Arc::new(WorkerShared {
            state: Mutex::new(WorkerState {
                generation: 0,
                shutdown: false,
                job: None,
                remaining: 0,
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
            state.remaining = self.workers.len();
            state.failed = false;
            state.generation = state.generation.wrapping_add(1);
            self.shared.work_available.notify_all();
        }

        let caller_succeeded = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: worker zero receives a range disjoint from every persistent worker.
            unsafe { execute_job_partition(job, 0, self.total_threads) }
        }))
        .is_ok();

        let mut state = lock(&self.shared.state);
        while state.remaining != 0 {
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
            state.generation = state.generation.wrapping_add(1);
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
            let mut state = lock(&shared.state);
            while !state.shutdown && state.generation == observed_generation {
                state = wait(&shared.work_available, state);
            }
            if state.shutdown {
                return;
            }
            observed_generation = state.generation;
            state.job.expect("worker generation must provide a job")
        };
        let succeeded = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: every worker receives a distinct output range and read-only shared inputs.
            unsafe { execute_job_partition(job, worker_index, total_threads) }
        }))
        .is_ok();
        let mut state = lock(&shared.state);
        state.failed |= !succeeded;
        state.remaining -= 1;
        if state.remaining == 0 {
            shared.work_complete.notify_one();
        }
    }
}

unsafe fn execute_job_partition(job: ParallelJob, worker_index: usize, total_threads: usize) {
    let start_row = job.rows * worker_index / total_threads;
    let end_row = job.rows * (worker_index + 1) / total_threads;
    if start_row == end_row {
        return;
    }
    // SAFETY: KernelContext::execute is synchronous and partitions output by matrix row.
    let weights = unsafe { slice::from_raw_parts(job.weights as *const u8, job.weight_bytes) };
    let quantized =
        unsafe { slice::from_raw_parts(job.quantized as *const i8, job.quantized_elements) };
    let activation_scales =
        unsafe { slice::from_raw_parts(job.activation_scales as *const f32, job.scale_elements) };
    unsafe {
        compute_batched_row_range(
            weights,
            quantized,
            activation_scales,
            job.output as *mut f32,
            job.batch_size,
            job.rows,
            job.cols,
            start_row,
            end_row,
            job.kernel,
        );
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
            .map(|workers| Box::into_raw(Box::new(KernelContext { workers })))
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
        q4_0_f32_batched_matmul(
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
        q4_0_f32_batched_matmul(
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
        q4_0_f32_grouped_batched_matmul(
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
        q4_0_f32_grouped_batched_matmul(
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
        )
    })) {
        Ok(status) => status,
        Err(_) => STATUS_PANIC,
    }
}

#[allow(clippy::too_many_arguments)]
fn q4_0_f32_batched_matmul(
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
) -> i32 {
    if weights.is_null() || input.is_null() || output.is_null() {
        return STATUS_NULL_POINTER;
    }
    let (batch_size, rows, cols) = (batch_size as usize, rows as usize, cols as usize);
    if batch_size == 0 || rows == 0 || cols == 0 || cols % QK != 0 {
        return STATUS_INVALID_SHAPE;
    }

    let blocks_per_row = cols / QK;
    let Some(required_weight_bytes) = rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(Q4_0_BLOCK_BYTES))
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

    let mut quantized = vec![0_i8; required_input_elements];
    let mut activation_scales = vec![0_f32; batch_size * blocks_per_row];
    quantize_q8_0_batch(
        input,
        batch_size,
        cols,
        &mut quantized,
        &mut activation_scales,
    );

    if !compute_outputs(
        context,
        weights,
        &quantized,
        &activation_scales,
        batch_size,
        rows,
        cols,
        output,
        selected_q4_kernel(),
    ) {
        return STATUS_PANIC;
    }
    STATUS_OK
}

#[allow(clippy::too_many_arguments)]
fn q4_0_f32_grouped_batched_matmul(
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
) -> i32 {
    if weight_pointers.is_null()
        || weight_bytes.is_null()
        || rows.is_null()
        || input.is_null()
        || output.is_null()
    {
        return STATUS_NULL_POINTER;
    }
    let (matrix_count, batch_size, cols) =
        (matrix_count as usize, batch_size as usize, cols as usize);
    if !(2..=3).contains(&matrix_count) || batch_size == 0 || cols == 0 || cols % QK != 0 {
        return STATUS_INVALID_SHAPE;
    }

    // SAFETY: each metadata array is non-null and the caller advertises matrix_count entries.
    let weight_pointers = unsafe { slice::from_raw_parts(weight_pointers, matrix_count) };
    let weight_bytes = unsafe { slice::from_raw_parts(weight_bytes, matrix_count) };
    let rows = unsafe { slice::from_raw_parts(rows, matrix_count) };
    let blocks_per_row = cols / QK;
    let Some(required_input_elements) = batch_size.checked_mul(cols) else {
        return STATUS_INVALID_SHAPE;
    };
    let mut required_output_elements = 0_usize;
    for matrix in 0..matrix_count {
        let matrix_rows = rows[matrix] as usize;
        if weight_pointers[matrix].is_null() || matrix_rows == 0 {
            return STATUS_INVALID_SHAPE;
        }
        let Some(required_weight_bytes) = matrix_rows
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(Q4_0_BLOCK_BYTES))
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
    let mut quantized = vec![0_i8; required_input_elements];
    let mut activation_scales = vec![0_f32; batch_size * blocks_per_row];
    quantize_q8_0_batch(
        input,
        batch_size,
        cols,
        &mut quantized,
        &mut activation_scales,
    );

    let kernel = selected_q4_kernel();
    let mut output_offset = 0;
    for matrix in 0..matrix_count {
        let matrix_rows = rows[matrix] as usize;
        let required_weight_bytes = matrix_rows * blocks_per_row * Q4_0_BLOCK_BYTES;
        let matrix_output_elements = batch_size * matrix_rows;
        // SAFETY: each pointer and byte length was validated above.
        let weights =
            unsafe { slice::from_raw_parts(weight_pointers[matrix], required_weight_bytes) };
        if !compute_outputs(
            context,
            weights,
            &quantized,
            &activation_scales,
            batch_size,
            matrix_rows,
            cols,
            &mut output[output_offset..output_offset + matrix_output_elements],
            kernel,
        ) {
            return STATUS_PANIC;
        }
        output_offset += matrix_output_elements;
    }
    STATUS_OK
}

#[allow(clippy::too_many_arguments)]
fn compute_outputs(
    context: Option<&KernelContext>,
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch_size: usize,
    rows: usize,
    cols: usize,
    output: &mut [f32],
    kernel: Q4Kernel,
) -> bool {
    if let Some(context) = context {
        return context.workers.execute(ParallelJob {
            weights: weights.as_ptr() as usize,
            weight_bytes: weights.len(),
            quantized: quantized.as_ptr() as usize,
            quantized_elements: quantized.len(),
            activation_scales: activation_scales.as_ptr() as usize,
            scale_elements: activation_scales.len(),
            output: output.as_mut_ptr() as usize,
            output_elements: output.len(),
            batch_size,
            rows,
            cols,
            kernel,
        });
    }

    let parallelism = thread::available_parallelism().map_or(1, usize::from);
    let worker_count = parallelism.min(output.len());
    if worker_count < 2 || output.len() < PARALLEL_OUTPUT_THRESHOLD {
        compute_output_range(
            weights,
            quantized,
            activation_scales,
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
    output: *mut f32,
    batch_size: usize,
    rows: usize,
    cols: usize,
    start_row: usize,
    end_row: usize,
    kernel: Q4Kernel,
) {
    match kernel {
        Q4Kernel::Scalar => {
            // SAFETY: the caller assigns this worker an exclusive matrix-row range.
            unsafe {
                compute_batched_row_range_scalar(
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
        Q4Kernel::Avx2 => {
            // SAFETY: runtime dispatch selected this variant only when AVX2 and FMA are available.
            unsafe {
                compute_batched_row_range_avx2(
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
unsafe fn compute_batched_row_range_scalar(
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
    let blocks_per_row = cols / QK;
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
                let input_offset = batch * cols + block * QK;
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
unsafe fn compute_batched_row_range_avx2(
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
    let blocks_per_row = cols / QK;
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
                let input_offset = batch * cols + block * QK;
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
fn compute_output_range(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    rows: usize,
    cols: usize,
    start_index: usize,
    output: &mut [f32],
    kernel: Q4Kernel,
) {
    for (local_index, slot) in output.iter_mut().enumerate() {
        let output_index = start_index + local_index;
        let batch = output_index / rows;
        let row = output_index % rows;
        *slot = match kernel {
            Q4Kernel::Scalar => {
                dot_q4_0_q8_0_row_scalar(weights, quantized, activation_scales, batch, row, cols)
            }
            #[cfg(target_arch = "x86_64")]
            Q4Kernel::Avx2 => {
                // SAFETY: this variant is selected only after runtime AVX2 and FMA detection.
                unsafe {
                    dot_q4_0_q8_0_row_avx2(weights, quantized, activation_scales, batch, row, cols)
                }
            }
        };
    }
}

fn selected_q4_kernel() -> Q4Kernel {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        return Q4Kernel::Avx2;
    }
    Q4Kernel::Scalar
}

fn dot_q4_0_q8_0_row_scalar(
    weights: &[u8],
    quantized: &[i8],
    activation_scales: &[f32],
    batch: usize,
    row: usize,
    cols: usize,
) -> f32 {
    let blocks_per_row = cols / QK;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK;
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
    let blocks_per_row = cols / QK;
    let mut sum = 0_f32;
    for block in 0..blocks_per_row {
        let weight_offset = (row * blocks_per_row + block) * Q4_0_BLOCK_BYTES;
        let input_offset = batch * cols + block * QK;
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
    let activations = unsafe { _mm256_loadu_si256(quantized.cast()) };
    let absolute_weights = _mm256_sign_epi8(signed_weights, signed_weights);
    let signed_activations = _mm256_sign_epi8(activations, signed_weights);
    let pair_products = _mm256_maddubs_epi16(absolute_weights, signed_activations);
    let pair_sums = _mm256_madd_epi16(pair_products, _mm256_set1_epi16(1));
    let mut lanes = [0_i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), pair_sums) };
    lanes.into_iter().sum()
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
    let blocks_per_row = cols / QK;
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let input_offset = batch * cols + block * QK;
            let values = &input[input_offset..input_offset + QK];
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
    let blocks_per_row = cols / QK;
    let sign_bit = _mm256_set1_ps(-0.0);
    let pack_permutation = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for batch in 0..batch_size {
        for block in 0..blocks_per_row {
            let input_offset = batch * cols + block * QK;
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
    let bits = match exponent {
        0 if significand == 0 => sign,
        0 => {
            let leading = significand.leading_zeros() - 22;
            let normalized = (significand << leading) & 0x03ff;
            let adjusted_exponent = (127 - 14 - leading as i32) as u32;
            sign | (adjusted_exponent << 23) | (normalized << 13)
        }
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
        assert_eq!(jmodels_kernels_abi_version(), 1);
        assert_eq!(
            jmodels_kernels_capabilities(),
            CAPABILITY_Q4_0_F32_BATCHED_MATMUL
                | CAPABILITY_Q4_0_F32_GROUPED_BATCHED_MATMUL
                | CAPABILITY_PERSISTENT_WORKER_CONTEXT
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
    fn rejects_invalid_dimensions_without_dereferencing_buffers() {
        let weights = [0_u8; Q4_0_BLOCK_BYTES];
        let input = [0_f32; QK];
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
}
