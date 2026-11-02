//! Costs the inner batched-GEMM tile at 256-bit against 512-bit, in isolation.
//!
//! The profile puts 96.6% of mellum's prefill in the batched quantised matmul, and the op-count
//! says the only remaining lever there is instruction width: `maddubs` + `madd_epi16` is already
//! two multiply-class ops per 32 elements, and `madd_epi16` earns its keep twice over by applying
//! the group scale *and* reducing pairs. VNNI's `vpdpbusd` cannot beat that -- it wins the dot in
//! one op but then needs a separate multiply for the scale, so the count is unchanged. AVX-512
//! width can: the same two ops cover 64 elements instead of 32.
//!
//! This benchmark exists so that claim is measured before any dispatch path is written. It runs
//! the two tiles over the same synthetic block and reports nanoseconds per tile, and it checks
//! they agree, because a faster tile that computes something else is not a result.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
/// 256-bit tile: eight 32-element groups, two multiply-class ops each.
unsafe fn tile_avx2(decoded: &[__m256i; 8], scales: &[i32; 8], activation: *const i8) -> i32 {
    let mut lanes = _mm256_setzero_si256();
    for group in 0..8 {
        // SAFETY: the caller provides 256 activation bytes.
        let acts = unsafe { _mm256_loadu_si256(activation.add(group * 32).cast()) };
        let products = _mm256_maddubs_epi16(decoded[group], acts);
        let scale = _mm256_set1_epi16(scales[group] as i16);
        lanes = _mm256_add_epi32(lanes, _mm256_madd_epi16(products, scale));
    }
    let halves = _mm_add_epi32(
        _mm256_castsi256_si128(lanes),
        _mm256_extracti128_si256(lanes, 1),
    );
    let folded = _mm_add_epi32(halves, _mm_shuffle_epi32(halves, 0b0100_1110));
    let last = _mm_add_epi32(folded, _mm_shuffle_epi32(folded, 0b1011_0001));
    _mm_cvtsi128_si32(last)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
/// 512-bit tile: four 64-element steps covering two Q4_K groups each.
///
/// Two groups share one instruction, so the scale vector carries `scales[2k]` in its low sixteen
/// i16 lanes and `scales[2k+1]` in the high sixteen. Same arithmetic, half the instructions.
unsafe fn tile_avx512(decoded: &[__m512i; 4], scale_pairs: &[__m512i; 4], activation: *const i8) -> i32 {
    let mut lanes = _mm512_setzero_si512();
    for step in 0..4 {
        // SAFETY: the caller provides 256 activation bytes.
        let acts = unsafe { _mm512_loadu_si512(activation.add(step * 64).cast()) };
        let products = _mm512_maddubs_epi16(decoded[step], acts);
        lanes = _mm512_add_epi32(lanes, _mm512_madd_epi16(products, scale_pairs[step]));
    }
    _mm512_reduce_add_epi32(lanes)
}

/// Runs both tiles `iters` times over one synthetic block.
///
/// Returns 0 on success and writes `[ns_avx2, ns_avx512, sum_avx2, sum_avx512]` into `out`, so the
/// caller can see both the cost and whether the two agree.
///
/// # Safety
///
/// `out` must address four f64.
#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jmodels_bench_q4k_tile(iters: u64, out: *mut f64) -> i32 {
    if out.is_null() {
        return -1;
    }
    if !std::arch::is_x86_feature_detected!("avx2") {
        return -2;
    }
    let has512 = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw");

    let mut seed = 0x2026_1102_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let mut nibbles = [0_u8; 256];
    for byte in nibbles.iter_mut() {
        *byte = (next() & 0x0F) as u8;
    }
    let mut acts = [0_i8; 256];
    for byte in acts.iter_mut() {
        *byte = ((next() % 255) as i32 - 127) as i8;
    }
    let mut scales = [0_i32; 8];
    for slot in scales.iter_mut() {
        *slot = (next() % 64) as i32;
    }

    // SAFETY: avx2 was detected above and every buffer is in scope for the call.
    let (ns2, sum2) = unsafe {
        let decoded: [__m256i; 8] =
            std::array::from_fn(|g| _mm256_loadu_si256(nibbles.as_ptr().add(g * 32).cast()));
        let start = Instant::now();
        let mut acc = 0_i64;
        for _ in 0..iters {
            // black_box on both the inputs and the result: without it LLVM hoists the whole tile
            // out of the loop, since nothing in it varies, and the benchmark reports ~0 ns.
            let d = std::hint::black_box(&decoded);
            let sc = std::hint::black_box(&scales);
            let a = std::hint::black_box(acts.as_ptr());
            acc += std::hint::black_box(tile_avx2(d, sc, a)) as i64;
        }
        let elapsed = start.elapsed().as_nanos() as f64;
        (elapsed / iters as f64, acc / iters.max(1) as i64)
    };

    let (ns512, sum512) = if has512 {
        // SAFETY: avx512f and avx512bw were detected above.
        unsafe {
            let decoded: [__m512i; 4] =
                std::array::from_fn(|s| _mm512_loadu_si512(nibbles.as_ptr().add(s * 64).cast()));
            let scale_pairs: [__m512i; 4] = std::array::from_fn(|s| {
                let mut lanes = [0_i16; 32];
                for (i, slot) in lanes.iter_mut().enumerate() {
                    *slot = scales[s * 2 + usize::from(i >= 16)] as i16;
                }
                _mm512_loadu_si512(lanes.as_ptr().cast())
            });
            let start = Instant::now();
            let mut acc = 0_i64;
            for _ in 0..iters {
                let d = std::hint::black_box(&decoded);
                let sp = std::hint::black_box(&scale_pairs);
                let a = std::hint::black_box(acts.as_ptr());
                acc += std::hint::black_box(tile_avx512(d, sp, a)) as i64;
            }
            let elapsed = start.elapsed().as_nanos() as f64;
            (elapsed / iters as f64, acc / iters.max(1) as i64)
        }
    } else {
        (f64::NAN, 0)
    };

    // SAFETY: the caller guarantees four f64.
    unsafe {
        *out = ns2;
        *out.add(1) = ns512;
        *out.add(2) = sum2 as f64;
        *out.add(3) = sum512 as f64;
    }
    0
}
