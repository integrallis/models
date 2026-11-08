//! External arithmetic experiment. Not included in Models runtime artifacts.
//! Online/split-KV attention follows ggml, MIT, pinned 2cbca389f515df455412dafa07e2b1359cc72a6d.
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
static POOLS: LazyLock<Mutex<HashMap<usize, rayon::ThreadPool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn round_slice(x: &mut [f32]) {
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
unsafe fn dot(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut a = [_mm256_setzero_ps(); 4];
        let n = x.len() / 32 * 32;
        for i in (0..n).step_by(32) {
            for j in 0..4 {
                a[j] = _mm256_fmadd_ps(
                    _mm256_loadu_ps(x.as_ptr().add(i + 8 * j)),
                    _mm256_loadu_ps(y.as_ptr().add(i + 8 * j)),
                    a[j],
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
unsafe fn half_update(dst: &mut [f32], src: &[f32], scale: f32, add: bool) {
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
fn chunk(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    begin: usize,
    end: usize,
    width: usize,
    scale: f32,
) -> (f32, f32, Vec<f32>) {
    let mut maximum = f32::NEG_INFINITY;
    let mut sum = 0f32;
    let mut output = vec![0f32; width];
    for p in begin..end {
        let score = unsafe { dot(q, &k[p * width..(p + 1) * width]) } * scale;
        let mut ms = 1f32;
        let mut vs = 1f32;
        if score > maximum {
            ms = (maximum - score).exp();
            maximum = score;
            unsafe { half_update(&mut output, &[], ms, false) };
        } else {
            vs = (score - maximum).exp();
        }
        unsafe { half_update(&mut output, &v[p * width..(p + 1) * width], vs, true) };
        sum = sum.mul_add(ms, vs);
    }
    (maximum, sum, output)
}
fn small(
    q: &[f32],
    k: &[f32],
    v: &[f32],
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
// All pointer lengths are supplied by the Java diagnostic bridge; runtime FFI requires explicit lengths.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oracle_attention(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    batch: i32,
    start: i32,
    heads: i32,
    width: i32,
    capacity: i32,
    threads: i32,
) -> i32 {
    std::panic::catch_unwind(|| {
        if q.is_null()
            || k.is_null()
            || v.is_null()
            || out.is_null()
            || batch < 1
            || start < 0
            || heads < 1
            || heads > 128
            || width < 1
            || width > 512
            || capacity < batch
            || start > capacity - batch
            || capacity > 4096
            || threads < 1
            || threads > 256
        {
            return -1;
        }
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("f16c")
        {
            return -4;
        }
        let (batch, start, heads, width, capacity, threads) = (
            batch as usize,
            start as usize,
            heads as usize,
            width as usize,
            capacity as usize,
            threads as usize,
        );
        let dim = heads * width;
        let positions = start + batch;
        let padded = positions.div_ceil(256) * 256;
        let q = unsafe { std::slice::from_raw_parts(q, batch * dim) };
        let k = unsafe { std::slice::from_raw_parts(k, capacity * dim) };
        let v = unsafe { std::slice::from_raw_parts(v, capacity * dim) };
        let output = out as usize;
        let mut pools = POOLS.lock().unwrap();
        let pool = pools.entry(threads).or_insert_with(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        });
        pool.install(|| {
            (0..heads).into_par_iter().for_each(|head| {
                let mut keys = k
                    [head * capacity * width..head * capacity * width + positions * width]
                    .to_vec();
                let mut values = v
                    [head * capacity * width..head * capacity * width + positions * width]
                    .to_vec();
                unsafe {
                    round_slice(&mut keys);
                    round_slice(&mut values);
                }
                if batch < 64 {
                    for row in 0..batch {
                        let r = small(
                            &q[row * dim + head * width..row * dim + (head + 1) * width],
                            &keys,
                            &values,
                            start + row + 1,
                            padded,
                            width,
                            threads,
                            batch == 1,
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
                let result = tiled(q, &keys, &values, batch, start, dim, head, width);
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
        0
    })
    .unwrap_or(-3)
}
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm(c: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
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
#[target_feature(enable = "avx2,fma")]
unsafe fn tile_exp(row: &mut [f32], max: f32) -> f64 {
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
fn tiled(
    q: &[f32],
    k: &[f32],
    v: &[f32],
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
                    kt[d * 64 + j] = k[(begin + j) * width + d];
                    vt[j * width + d] = v[(begin + j) * width + d];
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
