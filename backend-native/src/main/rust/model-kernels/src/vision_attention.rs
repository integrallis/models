// Copyright 2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0 AND MIT
//! LLaVA attention arithmetic. The online, split-KV, tiled GEMM and exponential
//! calculations are adapted from ggml 2cbca389f515df455412dafa07e2b1359cc72a6d.
//! Upstream copyright and MIT terms are retained in LICENSE-GGML.txt.
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn round_slice_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let n = x.len() / 8 * 8;
        for i in (0..n).step_by(8) {
            _mm256_storeu_ps(
                x.as_mut_ptr().add(i),
                _mm256_cvtph_ps(_mm256_cvtps_ph::<0>(_mm256_loadu_ps(x.as_ptr().add(i)))),
            );
        }
        for v in &mut x[n..] {
            *v = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtps_ph::<0>(_mm_set_ss(*v))));
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_avx2(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut a = [_mm256_setzero_ps(); 4];
        let n = x.len() / 32 * 32;
        for i in (0..n).step_by(32) {
            for (j, accumulator) in a.iter_mut().enumerate() {
                *accumulator = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * j)),
                    _mm256_loadu_ps(y.as_ptr().add(i + 8 * j)),
                    *accumulator,
                );
            }
        }
        a[0] = _mm256_add_ps(a[0], a[2]);
        a[1] = _mm256_add_ps(a[1], a[3]);
        a[0] = _mm256_add_ps(a[0], a[1]);
        let mut v = _mm_add_ps(
            _mm256_extractf128_ps::<1>(a[0]),
            _mm256_castps256_ps128(a[0]),
        );
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        let mut sum = _mm_cvtss_f32(v) as f64;
        for i in n..x.len() {
            sum += (x[i] * y[i]) as f64;
        }
        sum as f32
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn half_update_avx2(dst: &mut [f32], src: &[f32], scale: f32, add: bool) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len() / 8 * 8;
        let s = _mm256_set1_ps(scale);
        for i in (0..n).step_by(8) {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let v = if add {
                _mm256_fmadd_ps(_mm256_loadu_ps(src.as_ptr().add(i)), s, d)
            } else {
                _mm256_mul_ps(d, s)
            };
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i),
                _mm256_cvtph_ps(_mm256_cvtps_ph::<0>(v)),
            );
        }
        for i in n..dst.len() {
            dst[i] = if add {
                src[i].mul_add(scale, dst[i])
            } else {
                dst[i] * scale
            };
        }
        round_slice(&mut dst[n..]);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_half_avx2(x: &[f32], y: &[u16]) -> f32 {
    use std::arch::x86_64::*;
    // Lane order, accumulator count, reduction tree and f64 tail are those of dot_avx2. The only
    // difference is where each operand comes from: cvtph_ps widens exactly, so it reproduces the
    // f32 a widened cache held bit for bit.
    unsafe {
        let mut a = [_mm256_setzero_ps(); 4];
        let n = x.len() / 32 * 32;
        for i in (0..n).step_by(32) {
            for (j, accumulator) in a.iter_mut().enumerate() {
                *accumulator = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * j)),
                    _mm256_cvtph_ps(_mm_loadu_si128(y.as_ptr().add(i + 8 * j).cast())),
                    *accumulator,
                );
            }
        }
        a[0] = _mm256_add_ps(a[0], a[2]);
        a[1] = _mm256_add_ps(a[1], a[3]);
        a[0] = _mm256_add_ps(a[0], a[1]);
        let mut v = _mm_add_ps(
            _mm256_extractf128_ps::<1>(a[0]),
            _mm256_castps256_ps128(a[0]),
        );
        v = _mm_hadd_ps(v, v);
        v = _mm_hadd_ps(v, v);
        let mut sum = _mm_cvtss_f32(v) as f64;
        for i in n..x.len() {
            sum += (x[i] * super::f16_to_f32(y[i])) as f64;
        }
        sum as f32
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn half_update_half_avx2(dst: &mut [f32], src: &[u16], scale: f32, add: bool) {
    use std::arch::x86_64::*;
    unsafe {
        let n = dst.len() / 8 * 8;
        let s = _mm256_set1_ps(scale);
        for i in (0..n).step_by(8) {
            let d = _mm256_loadu_ps(dst.as_ptr().add(i));
            let v = if add {
                _mm256_fmadd_ps(
                    _mm256_cvtph_ps(_mm_loadu_si128(src.as_ptr().add(i).cast())),
                    s,
                    d,
                )
            } else {
                _mm256_mul_ps(d, s)
            };
            _mm256_storeu_ps(
                dst.as_mut_ptr().add(i),
                _mm256_cvtph_ps(_mm256_cvtps_ph::<0>(v)),
            );
        }
        for i in n..dst.len() {
            dst[i] = if add {
                super::f16_to_f32(src[i]).mul_add(scale, dst[i])
            } else {
                dst[i] * scale
            };
        }
        round_slice(&mut dst[n..]);
    }
}
/// How the arithmetic reads one retained K/V element.
///
/// The Java cache rounds every appended row to F16 exactly once, so `u16` storage holds those F16
/// bits and `f32` storage holds the same bits already widened. Widening binary16 to binary32 is
/// exact, so both implementations feed identical operands to identical operations in identical
/// order, and the storage width is a memory-traffic choice rather than a numerical one. The decode
/// path reads the whole retained prefix once per token, so halving its width halves that read.
///
/// `f32` remains instantiated to serve as the parity oracle the tests compare against; the shipped
/// decoder stores `u16`.
pub(super) trait CacheElement: Copy + Send + Sync {
    /// Dot product of an already-rounded query row with one cache row.
    ///
    /// # Safety
    /// `row` must be at least as long as `query`.
    unsafe fn dot_row(query: &[f32], row: &[Self]) -> f32;
    /// `dst = half(dst * scale)`, or `dst = half(row * scale + dst)` when `add`.
    ///
    /// # Safety
    /// When `add`, `row` must be at least as long as `dst`; otherwise `row` is not read.
    unsafe fn fused_update(dst: &mut [f32], row: &[Self], scale: f32, add: bool);
    /// One element, widened to the f32 the arithmetic operates on.
    fn widen(self) -> f32;
}
impl CacheElement for f32 {
    unsafe fn dot_row(query: &[f32], row: &[f32]) -> f32 {
        unsafe { dot(query, row) }
    }
    unsafe fn fused_update(dst: &mut [f32], row: &[f32], scale: f32, add: bool) {
        unsafe { half_update(dst, row, scale, add) };
    }
    fn widen(self) -> f32 {
        self
    }
}
impl CacheElement for u16 {
    unsafe fn dot_row(query: &[f32], row: &[u16]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            return unsafe { dot_half_avx2(query, row) };
        }
        let widened: Vec<f32> = row[..query.len()].iter().map(|h| h.widen()).collect();
        unsafe { dot(query, &widened) }
    }
    unsafe fn fused_update(dst: &mut [f32], row: &[u16], scale: f32, add: bool) {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && std::is_x86_feature_detected!("f16c")
        {
            unsafe { half_update_half_avx2(dst, row, scale, add) };
            return;
        }
        if !add {
            unsafe { half_update(dst, &[], scale, false) };
            return;
        }
        let widened: Vec<f32> = row[..dst.len()].iter().map(|h| h.widen()).collect();
        unsafe { half_update(dst, &widened, scale, true) };
    }
    fn widen(self) -> f32 {
        super::f16_to_f32(self)
    }
}
fn chunk<E: CacheElement>(
    q: &[f32],
    k: &[E],
    v: &[E],
    begin: usize,
    end: usize,
    width: usize,
    scale: f32,
) -> (f32, f32, Vec<f32>) {
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0f32;
    let mut output = vec![0f32; width];
    for p in begin..end {
        let score = unsafe { E::dot_row(q, &k[p * width..(p + 1) * width]) } * scale;
        let mut ms = 1f32;
        let mut vs = 1f32;
        if score > maximum {
            ms = (maximum - score).exp();
            maximum = score;
            unsafe { E::fused_update(&mut output, &[], ms, false) };
        } else {
            vs = (score - maximum).exp();
        }
        unsafe { E::fused_update(&mut output, &v[p * width..(p + 1) * width], vs, true) };
        sum = sum.mul_add(ms, vs);
    }
    (maximum, sum, output)
}
#[allow(clippy::too_many_arguments)]
fn small<E: CacheElement>(
    q: &[f32],
    k: &[E],
    v: &[E],
    visible: usize,
    padded: usize,
    width: usize,
    threads: usize,
    decode: bool,
) -> Vec<f32> {
    let mut query = q.to_vec();
    unsafe { round_slice(&mut query) };
    let scale = 1f32 / (width as f32).sqrt();
    if !decode || padded < 512 {
        let (_, sum, mut o) = chunk(&query, k, v, 0, visible, width, scale);
        let inv = 1f32 / sum;
        for x in &mut o {
            *x *= inv;
        }
        return o;
    }
    let size = padded.div_ceil(threads);
    let mut out = vec![0f32; width];
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0f32;
    for i in 0..threads {
        let a = i * size;
        let b = ((i + 1) * size).min(visible);
        if a >= b {
            continue;
        }
        let (m, s, o) = chunk(&query, k, v, a, b, width, scale);
        let new = max.max(m);
        let old_scale = (max - new).exp();
        let new_scale = (m - new).exp();
        for d in 0..width {
            out[d] = out[d].mul_add(old_scale, o[d] * new_scale);
        }
        sum = sum.mul_add(old_scale, s * new_scale);
        max = new;
    }
    let inv = 1f32 / sum;
    for x in &mut out {
        *x *= inv;
    }
    out
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_avx2(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::*;
    // Six rows share each pair of loaded B vectors. Every output retains the same
    // increasing-K fused accumulation order as the scalar and ragged kernels.
    if !n.is_multiple_of(16) {
        unsafe { gemm_rows_avx2(c, a, b, m, k, n) };
        return;
    }
    let end = m / 6 * 6;
    unsafe {
        for row in (0..end).step_by(6) {
            for col in (0..n).step_by(16) {
                let mut acc = [[_mm256_setzero_ps(); 2]; 6];
                for (r, pair) in acc.iter_mut().enumerate() {
                    pair[0] = _mm256_loadu_ps(c.as_ptr().add((row + r) * n + col));
                    pair[1] = _mm256_loadu_ps(c.as_ptr().add((row + r) * n + col + 8));
                }
                for x in 0..k {
                    let b0 = _mm256_loadu_ps(b.as_ptr().add(x * n + col));
                    let b1 = _mm256_loadu_ps(b.as_ptr().add(x * n + col + 8));
                    for (r, pair) in acc.iter_mut().enumerate() {
                        let av = _mm256_set1_ps(a[(row + r) * k + x]);
                        pair[0] = _mm256_fmadd_ps(av, b0, pair[0]);
                        pair[1] = _mm256_fmadd_ps(av, b1, pair[1]);
                    }
                }
                for (r, pair) in acc.iter().enumerate() {
                    _mm256_storeu_ps(c.as_mut_ptr().add((row + r) * n + col), pair[0]);
                    _mm256_storeu_ps(c.as_mut_ptr().add((row + r) * n + col + 8), pair[1]);
                }
            }
        }
        gemm_rows_avx2(&mut c[end * n..], &a[end * k..], b, m - end, k, n);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_rows_avx2(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    use std::arch::x86_64::*;
    unsafe {
        for row in 0..m {
            for col in (0..n).step_by(8) {
                let mut acc = _mm256_loadu_ps(c.as_ptr().add(row * n + col));
                for x in 0..k {
                    acc = _mm256_fmadd_ps(
                        _mm256_set1_ps(a[row * k + x]),
                        _mm256_loadu_ps(b.as_ptr().add(x * n + col)),
                        acc,
                    );
                }
                _mm256_storeu_ps(c.as_mut_ptr().add(row * n + col), acc);
            }
        }
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_negative(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let c = |bits| _mm256_set1_ps(f32::from_bits(bits));
    let r = c(0x4b400000);
    let z = _mm256_fmadd_ps(x, c(0x3fb8aa3b), r);
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(n, c(0x35bfbe8e), _mm256_fnmadd_ps(n, c(0x3f317200), x));
    let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
    let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_set1_epi32(0x3f800000)));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(c(0x3c072010), b, c(0x3d2b9f17)),
            u,
            _mm256_fmadd_ps(c(0x3e2aaf33), b, c(0x3efffedb)),
        ),
        u,
        _mm256_mul_ps(c(0x3f7ffff6), b),
    );
    let mut result = _mm256_fmadd_ps(j, k, k);
    if _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(-87.0))) != 0 {
        let mut xs = [0f32; 8];
        let mut ys = [0f32; 8];
        unsafe {
            _mm256_storeu_ps(xs.as_mut_ptr(), x);
            _mm256_storeu_ps(ys.as_mut_ptr(), result);
        }
        for i in 0..8 {
            if xs[i] < -87.0 {
                ys[i] = xs[i].exp();
            }
        }
        result = unsafe { _mm256_loadu_ps(ys.as_ptr()) };
    }
    result
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_exp_avx2(row: &mut [f32], max: f32) -> f64 {
    use std::arch::x86_64::*;
    let mut sum = 0f64;
    unsafe {
        for j in (0..64).step_by(8) {
            let y = exp_negative(_mm256_sub_ps(
                _mm256_loadu_ps(row.as_ptr().add(j)),
                _mm256_set1_ps(max),
            ));
            _mm256_storeu_ps(row.as_mut_ptr().add(j), y);
            let mut a = _mm_add_ps(_mm256_extractf128_ps::<1>(y), _mm256_castps256_ps128(y));
            a = _mm_add_ps(a, _mm_movehl_ps(a, a));
            a = _mm_add_ss(a, _mm_movehdup_ps(a));
            sum += _mm_cvtss_f32(a) as f64;
        }
    }
    sum
}
#[allow(clippy::too_many_arguments)]
fn tiled<E: CacheElement>(
    q: &[f32],
    k: &[E],
    v: &[E],
    batch: usize,
    start: usize,
    dim: usize,
    head: usize,
    width: usize,
) -> Vec<f32> {
    let positions = start + batch;
    let mut output = vec![0f32; batch * width];
    let scale = 1f32 / (width as f32).sqrt();
    for qr in (0..batch).step_by(64) {
        let rows = (batch - qr).min(64);
        let mut queries = vec![0f32; rows * width];
        for row in 0..rows {
            queries[row * width..(row + 1) * width].copy_from_slice(
                &q[(qr + row) * dim + head * width..(qr + row) * dim + (head + 1) * width],
            );
        }
        let mut result = vec![0f32; rows * width];
        let mut max = vec![f32::NEG_INFINITY; rows];
        let mut sum = vec![0f32; rows];
        let mut kt = vec![0f32; width * 64];
        let mut vt = vec![0f32; 64 * width];
        let mut scores = vec![0f32; rows * 64];
        for begin in (0..positions).step_by(64) {
            if begin >= start + qr + rows {
                break;
            }
            let n = (positions - begin).min(64);
            kt.fill(0.);
            vt.fill(0.);
            scores.fill(0.);
            for j in 0..n {
                for d in 0..width {
                    kt[d * 64 + j] = k[(begin + j) * width + d].widen();
                    vt[j * width + d] = v[(begin + j) * width + d].widen();
                }
            }
            unsafe {
                gemm(&mut scores, &queries, &kt, rows, width, 64);
            }
            for row in 0..rows {
                let count = n.min((start + qr + row + 1).saturating_sub(begin));
                let s = &mut scores[row * 64..(row + 1) * 64];
                if count == 0 {
                    s.fill(0.);
                    continue;
                }
                let mut m = f32::NEG_INFINITY;
                for x in &mut s[..count] {
                    *x *= scale;
                    m = m.max(*x);
                }
                s[count..].fill(f32::NEG_INFINITY);
                let new = max[row].max(m);
                if new > max[row] {
                    let factor = (max[row] - new).exp();
                    for x in &mut result[row * width..(row + 1) * width] {
                        *x *= factor;
                    }
                    sum[row] *= factor;
                }
                max[row] = new;
                sum[row] = (sum[row] as f64 + unsafe { tile_exp(s, new) }) as f32;
            }
            unsafe {
                gemm(&mut result, &scores, &vt, rows, 64, width);
            }
        }
        for row in 0..rows {
            let inv = 1f32 / sum[row];
            for d in 0..width {
                output[(qr + row) * width + d] = result[row * width + d] * inv;
            }
        }
    }
    output
}

#[inline]
fn half(value: f32) -> f32 {
    super::f16_to_f32(super::f32_to_f16(value))
}
unsafe fn round_slice(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("f16c")
        && std::is_x86_feature_detected!("fma")
    {
        unsafe { round_slice_avx2(values) };
        return;
    }
    for value in values {
        *value = half(*value);
    }
}
unsafe fn dot(x: &[f32], y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
    {
        return unsafe { dot_avx2(x, y) };
    }
    let n = x.len() / 32 * 32;
    let mut sums = [[0f32; 8]; 4];
    for i in (0..n).step_by(32) {
        for (j, group) in sums.iter_mut().enumerate() {
            for (lane, sum) in group.iter_mut().enumerate() {
                let d = i + 8 * j + lane;
                *sum = x[d].mul_add(y[d], *sum);
            }
        }
    }
    let mut lanes = [0f32; 8];
    for i in 0..8 {
        lanes[i] = (sums[0][i] + sums[2][i]) + (sums[1][i] + sums[3][i]);
    }
    let h = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    let mut sum = ((h[0] + h[1]) + (h[2] + h[3])) as f64;
    for i in n..x.len() {
        sum += (x[i] * y[i]) as f64;
    }
    sum as f32
}
unsafe fn half_update(dst: &mut [f32], src: &[f32], scale: f32, add: bool) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("f16c")
    {
        unsafe { half_update_avx2(dst, src, scale, add) };
        return;
    }
    for i in 0..dst.len() {
        dst[i] = half(if add {
            src[i].mul_add(scale, dst[i])
        } else {
            dst[i] * scale
        });
    }
}
/// Accumulates C += A B in increasing reduction-index order.
pub(super) unsafe fn gemm(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
    #[cfg(target_arch = "x86_64")]
    if n.is_multiple_of(8)
        && std::is_x86_feature_detected!("avx2")
        && std::is_x86_feature_detected!("fma")
    {
        unsafe { gemm_avx2(c, a, b, m, k, n) };
        return;
    }
    for row in 0..m {
        for col in 0..n {
            let mut sum = c[row * n + col];
            for d in 0..k {
                sum = a[row * k + d].mul_add(b[d * n + col], sum);
            }
            c[row * n + col] = sum;
        }
    }
}
fn exp_negative_one(x: f32) -> f32 {
    if x < -87.0 {
        return x.exp();
    }
    let c = f32::from_bits;
    let r = c(0x4b400000);
    let z = x.mul_add(c(0x3fb8aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(c(0x35bfbe8e), (-n).mul_add(c(0x3f317200), x));
    let k = c(z.to_bits().wrapping_shl(23).wrapping_add(0x3f800000));
    let u = b * b;
    let j = c(0x3c072010)
        .mul_add(b, c(0x3d2b9f17))
        .mul_add(u, c(0x3e2aaf33).mul_add(b, c(0x3efffedb)))
        .mul_add(u, c(0x3f7ffff6) * b);
    j.mul_add(k, k)
}
unsafe fn tile_exp(row: &mut [f32], max: f32) -> f64 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        return unsafe { tile_exp_avx2(row, max) };
    }
    let mut sum = 0f64;
    for chunk in row.chunks_exact_mut(8) {
        for x in chunk.iter_mut() {
            *x = exp_negative_one(*x - max);
        }
        let h = [
            chunk[0] + chunk[4],
            chunk[1] + chunk[5],
            chunk[2] + chunk[6],
            chunk[3] + chunk[7],
        ];
        sum += ((h[0] + h[2]) + (h[1] + h[3])) as f64;
    }
    sum
}
/// The FFI owner validates all shapes and lifetimes before entering this synchronous call.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn run<E: CacheElement>(
    pool: &rayon::ThreadPool,
    q: *const f32,
    k: *const E,
    v: *const E,
    out: *mut f32,
    batch: usize,
    positions: usize,
    heads: usize,
    width: usize,
    capacity: usize,
    original_batch: usize,
) {
    let start = positions - batch;
    let dim = heads * width;
    let padded = positions.div_ceil(256) * 256;
    let threads = pool.current_num_threads();
    let q = unsafe { std::slice::from_raw_parts(q, batch * dim) };
    let k = unsafe { std::slice::from_raw_parts(k, capacity * dim) };
    let v = unsafe { std::slice::from_raw_parts(v, capacity * dim) };
    let output = out as usize;
    pool.install(|| {
        (0..heads).into_par_iter().for_each(|head| {
            // The Java-owned cache rounds newly appended rows to F16 exactly once. Borrow that
            // immutable prefix for this synchronous call; never retain pointers after returning.
            let keys = &k[head * capacity * width..head * capacity * width + positions * width];
            let values = &v[head * capacity * width..head * capacity * width + positions * width];
            if original_batch < 64 {
                for row in 0..batch {
                    let r = small(
                        &q[row * dim + head * width..row * dim + (head + 1) * width],
                        keys,
                        values,
                        start + row + 1,
                        padded,
                        width,
                        threads,
                        original_batch == 1,
                    );
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            r.as_ptr(),
                            (output as *mut f32).add(row * dim + head * width),
                            width,
                        )
                    };
                }
                return;
            }
            let result = tiled(q, keys, values, batch, start, dim, head, width);
            for row in 0..batch {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        result.as_ptr().add(row * width),
                        (output as *mut f32).add(row * dim + head * width),
                        width,
                    );
                }
            }
        })
    });
}

#[cfg(test)]
mod tests {
    use crate::deberta_kernels::{
        jmodels_deberta_create, jmodels_deberta_destroy, jmodels_vision_causal_attention_f16_cache,
        jmodels_vision_causal_attention_f16_store,
    };
    fn data(count: usize, state: &mut u32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((*state >> 8) as f32 / 16777216f32) * 4f32 - 2f32
            })
            .collect()
    }
    /// The nine shapes of the golden-fixture case, as (batch, start, pruned).
    const SHAPES: [(usize, usize, bool); 9] = [
        (1, 0, false),
        (8, 13, false),
        (31, 609, false),
        (64, 12, false),
        (128, 35, false),
        (1, 639, false),
        (1, 1023, false),
        (31, 609, true),
        (128, 35, true),
    ];
    #[test]
    fn f16_storage_is_bit_identical_to_the_widened_cache() {
        let handle = jmodels_deberta_create(8);
        assert_ne!(handle, 0);
        for (batch, start, pruned) in SHAPES {
            let mut state = 0x12345678;
            let q = data(batch * 256, &mut state);
            let mut k = data(1024 * 256, &mut state);
            let mut v = data(k.len(), &mut state);
            // Both entry points receive the same cache: one widened, one as the F16 bits it was
            // rounded to. Rounding here is what the Java cache owner does on append.
            unsafe {
                super::round_slice(&mut k);
                super::round_slice(&mut v);
            }
            let kh: Vec<u16> = k.iter().copied().map(crate::f32_to_f16).collect();
            let vh: Vec<u16> = v.iter().copied().map(crate::f32_to_f16).collect();
            // The round trip must be closed, or the two arms would not hold the same values.
            for (widened, half) in k.iter().zip(&kh).chain(v.iter().zip(&vh)) {
                assert_eq!(widened.to_bits(), crate::f16_to_f32(*half).to_bits());
            }
            let physical = if pruned { 1 } else { batch };
            let query = if pruned {
                &q[(batch - 1) * 256..]
            } else {
                &q[..]
            };
            let mut widened_output = vec![0f32; physical * 256];
            let mut half_output = vec![0f32; physical * 256];
            let widened = unsafe {
                jmodels_vision_causal_attention_f16_cache(
                    handle,
                    query.as_ptr(),
                    query.len(),
                    k.as_ptr(),
                    k.len(),
                    v.as_ptr(),
                    v.len(),
                    widened_output.as_mut_ptr(),
                    widened_output.len(),
                    physical,
                    start + batch,
                    256,
                    2,
                    1024,
                    batch,
                )
            };
            let half = unsafe {
                jmodels_vision_causal_attention_f16_store(
                    handle,
                    query.as_ptr(),
                    query.len(),
                    kh.as_ptr(),
                    kh.len(),
                    vh.as_ptr(),
                    vh.len(),
                    half_output.as_mut_ptr(),
                    half_output.len(),
                    physical,
                    start + batch,
                    256,
                    2,
                    1024,
                    batch,
                )
            };
            assert_eq!((widened, half), (0, 0));
            for (i, (a, b)) in widened_output.iter().zip(&half_output).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "batch={batch} start={start} pruned={pruned} index={i} widened={a} half={b}"
                );
            }
        }
        jmodels_deberta_destroy(handle);
    }
    #[test]
    fn f16_storage_rejects_exactly_what_the_widened_cache_rejects() {
        let handle = jmodels_deberta_create(2);
        let q = [1f32; 8];
        let k = [1f32; 8];
        let kh = [0x3c00u16; 8];
        let mut output = [99f32; 8];
        // (batch, positions, dim, heads, capacity, original_batch), valid last.
        let shapes: [(usize, usize, usize, usize, usize, usize); 9] = [
            (1, 2, 4, 1, 2, 0),
            (1, 2, 4, 1, 2, 3),
            (0, 2, 4, 1, 2, 1),
            (3, 2, 4, 1, 2, 3),
            (1, 2, 4, 0, 2, 1),
            (1, 2, 3, 2, 2, 1),
            (1, 2, 4, 1, 1, 1),
            (1, 2, 8, 1, 2, 1),
            (1, 2, 4, 1, 2, 1),
        ];
        for shape in shapes {
            let (batch, positions, dim, heads, capacity, original) = shape;
            let widened = unsafe {
                jmodels_vision_causal_attention_f16_cache(
                    handle,
                    q.as_ptr(),
                    8,
                    k.as_ptr(),
                    8,
                    k.as_ptr(),
                    8,
                    output.as_mut_ptr(),
                    8,
                    batch,
                    positions,
                    dim,
                    heads,
                    capacity,
                    original,
                )
            };
            let half = unsafe {
                jmodels_vision_causal_attention_f16_store(
                    handle,
                    q.as_ptr(),
                    8,
                    kh.as_ptr(),
                    8,
                    kh.as_ptr(),
                    8,
                    output.as_mut_ptr(),
                    8,
                    batch,
                    positions,
                    dim,
                    heads,
                    capacity,
                    original,
                )
            };
            assert_eq!(widened, half, "shape={shape:?}");
        }
        // Output aliasing an input, which the F16 entry point measures in two-byte spans.
        let aliased = unsafe {
            jmodels_vision_causal_attention_f16_store(
                handle,
                output.as_ptr(),
                8,
                kh.as_ptr(),
                8,
                kh.as_ptr(),
                8,
                output.as_mut_ptr(),
                8,
                1,
                2,
                4,
                1,
                2,
                1,
            )
        };
        assert_eq!(aliased, -1);
        let overlapping = unsafe {
            jmodels_vision_causal_attention_f16_store(
                handle,
                q.as_ptr(),
                8,
                output.as_ptr().cast(),
                8,
                kh.as_ptr(),
                8,
                output.as_mut_ptr(),
                8,
                1,
                2,
                4,
                1,
                2,
                1,
            )
        };
        assert_eq!(overlapping, -1);
        let unknown = unsafe {
            jmodels_vision_causal_attention_f16_store(
                0,
                q.as_ptr(),
                8,
                kh.as_ptr(),
                8,
                kh.as_ptr(),
                8,
                output.as_mut_ptr(),
                8,
                1,
                2,
                4,
                1,
                2,
                1,
            )
        };
        assert_eq!(unknown, -2);
        jmodels_deberta_destroy(handle);
    }
    #[test]
    fn matches_independent_flash_attention_including_pruned_queries() {
        let cases: [(&[u8], usize, usize, bool); 9] = [
            (
                include_bytes!("../testdata/vision-attention/origin.f32"),
                1,
                0,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/small.f32"),
                8,
                13,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/suffix.f32"),
                31,
                609,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/tile64.f32"),
                64,
                12,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/tile128.f32"),
                128,
                35,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/decode640.f32"),
                1,
                639,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/decode1024.f32"),
                1,
                1023,
                false,
            ),
            (
                include_bytes!("../testdata/vision-attention/pruned-suffix.f32"),
                31,
                609,
                true,
            ),
            (
                include_bytes!("../testdata/vision-attention/pruned-tile.f32"),
                128,
                35,
                true,
            ),
        ];
        let handle = jmodels_deberta_create(8);
        assert_ne!(handle, 0);
        for (bytes, batch, start, pruned) in cases {
            let mut state = 0x12345678;
            let q = data(batch * 256, &mut state);
            let mut k = data(1024 * 256, &mut state);
            let mut v = data(k.len(), &mut state);
            // This ABI receives the F16-rounded, Java-owned retained cache.
            unsafe {
                super::round_slice(&mut k);
                super::round_slice(&mut v);
            }
            let physical = if pruned { 1 } else { batch };
            let query = if pruned {
                &q[(batch - 1) * 256..]
            } else {
                &q[..]
            };
            let mut output = vec![0f32; physical * 256];
            let rc = unsafe {
                jmodels_vision_causal_attention_f16_cache(
                    handle,
                    query.as_ptr(),
                    query.len(),
                    k.as_ptr(),
                    k.len(),
                    v.as_ptr(),
                    v.len(),
                    output.as_mut_ptr(),
                    output.len(),
                    physical,
                    start + batch,
                    256,
                    2,
                    1024,
                    batch,
                )
            };
            assert_eq!(rc, 0);
            let expected: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(expected.len(), output.len());
            for (i, (actual, reference)) in output.iter().zip(expected).enumerate() {
                assert!(
                    (actual - reference).abs() <= 1e-6,
                    "batch={batch} start={start} pruned={pruned} index={i} actual={actual} reference={reference}"
                );
            }
        }
        jmodels_deberta_destroy(handle);
    }
    #[test]
    fn rejects_invalid_original_batch_and_alias_before_writing() {
        let handle = jmodels_deberta_create(2);
        let q = [1f32; 8];
        let k = [1f32; 8];
        let v = [1f32; 8];
        let mut output = [99f32; 8];
        for original in [0, 3] {
            let rc = unsafe {
                jmodels_vision_causal_attention_f16_cache(
                    handle,
                    q.as_ptr(),
                    8,
                    k.as_ptr(),
                    8,
                    v.as_ptr(),
                    8,
                    output.as_mut_ptr(),
                    8,
                    1,
                    2,
                    4,
                    1,
                    2,
                    original,
                )
            };
            assert_eq!(rc, -1);
            assert_eq!(output, [99.; 8]);
        }
        let rc = unsafe {
            jmodels_vision_causal_attention_f16_cache(
                handle,
                output.as_ptr(),
                8,
                k.as_ptr(),
                8,
                v.as_ptr(),
                8,
                output.as_mut_ptr(),
                8,
                1,
                2,
                4,
                1,
                2,
                1,
            )
        };
        assert_eq!(rc, -1);
        assert_eq!(output, [99.; 8]);
        jmodels_deberta_destroy(handle);
    }
}
