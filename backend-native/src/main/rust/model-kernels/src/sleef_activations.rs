// SLEEF 3.6.1 algorithms: Copyright (c) 2010-2021 Naoki Shibata and contributors.
// Distributed under the Boost Software License, Version 1.0; see ../LICENSE-BOOST-SLEEF.txt.
// Source-order port of pinned sleefinline_advsimd.h EXP u10 and LOG u10.
// No fast-math, no implicit contraction, no Rust slice/reference alias assumptions.
// Safety: callers validate nonnull Java-sized ranges and n*k extents; convolution k>=2.
#![cfg(target_arch = "aarch64")]
#![allow(clippy::excessive_precision, clippy::approx_constant)]
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn c(x: f32) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vdupq_n_f32(x) }
}
#[inline(always)]
unsafe fn add(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vaddq_f32(a, b) }
}
#[inline(always)]
unsafe fn sub(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vsubq_f32(a, b) }
}
#[inline(always)]
unsafe fn mul(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vmulq_f32(a, b) }
}
#[inline(always)]
unsafe fn div(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vdivq_f32(a, b) }
}
#[inline(always)]
unsafe fn neg(a: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vnegq_f32(a) }
}
#[inline(always)]
unsafe fn fma(a: float32x4_t, b: float32x4_t, d: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vfmaq_f32(d, a, b) }
}
#[inline(always)]
unsafe fn sel(mask: uint32x4_t, yes: float32x4_t, no: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vbslq_f32(mask, yes, no) }
}

#[inline(always)]
unsafe fn exp4_full(d: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let nan = vmvnq_u32(vceqq_f32(d, d));
        let invalid = vorrq_u32(vcgtq_f32(vabsq_f32(d), c(104.0)), nan);
        let a = sel(invalid, c(0.0), d);
        let q = vcvtnq_s32_f32(mul(
            a,
            c(1.442695040888963407359924681001892137426645954152985934135449406931),
        ));
        let fq = vcvtq_f32_s32(q);
        let s = fma(fq, c(-0.693145751953125), a);
        let s = fma(fq, c(-1.428606765330187045e-06), s);
        let u = fma(
            c(0.000198527617612853646278381),
            s,
            c(0.00139304355252534151077271),
        );
        let u = fma(u, s, c(0.00833336077630519866943359));
        let u = fma(u, s, c(0.0416664853692054748535156));
        let u = fma(u, s, c(0.166666671633720397949219));
        let u = fma(u, s, c(0.5));
        let u = add(fma(mul(s, s), u, s), c(1.0));
        let h = vshrq_n_s32::<1>(q);
        let u = mul(
            u,
            vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(h, vdupq_n_s32(127)))),
        );
        let u = mul(
            u,
            vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(
                vsubq_s32(q, h),
                vdupq_n_s32(127),
            ))),
        );
        let u = sel(vcltq_f32(d, c(-104.0)), c(0.0), u);
        let u = sel(vcgtq_f32(d, c(100.0)), c(f32::INFINITY), u);
        sel(nan, add(d, d), u)
    }
}
#[inline(always)]
unsafe fn log4_full(input: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let subnormal = vcltq_f32(input, c(f32::MIN_POSITIVE));
        let d = sel(subnormal, mul(input, c(18446744073709551616.0)), input);
        let e = vsubq_s32(
            vreinterpretq_s32_u32(vandq_u32(
                vshrq_n_u32::<23>(vreinterpretq_u32_f32(mul(d, c(1.0 / 0.75)))),
                vdupq_n_u32(255),
            )),
            vdupq_n_s32(127),
        );
        let m = vreinterpretq_f32_s32(vaddq_s32(
            vreinterpretq_s32_f32(d),
            vshlq_n_s32::<23>(vnegq_s32(e)),
        ));
        let e = vbslq_s32(subnormal, vsubq_s32(e, vdupq_n_s32(64)), e);
        let ef = vcvtq_f32_s32(e);
        let sh = mul(ef, c(0.69314718246459960938));
        let sl = fma(
            c(-1.904654323148236017e-09),
            ef,
            fma(c(0.69314718246459960938), ef, neg(sh)),
        );
        let nh = add(m, c(-1.0));
        let nv = add(nh, c(1.0));
        let nl = add(sub(c(-1.0), sub(nh, nv)), sub(m, nv));
        let dh = add(m, c(1.0));
        let dv = sub(dh, c(1.0));
        let dl = add(sub(c(1.0), sub(dh, dv)), sub(m, dv));
        let t = div(c(1.0), dh);
        let xh = mul(nh, t);
        let u = fma(t, nh, neg(xh));
        let v = fma(neg(dl), t, fma(neg(dh), t, c(1.0)));
        let xl = fma(xh, v, fma(nl, t, u));
        let x2 = mul(xh, xh);
        let t = fma(c(0.3027294874), x2, c(0.3996108174));
        let t = fma(t, x2, c(0.6666694880));
        let yh = mul(xh, c(2.0));
        let yl = mul(xl, c(2.0));
        let sum = add(sh, yh);
        let sl = add(add(add(sub(sh, sum), yh), sl), yl);
        let sh = sum;
        let y = mul(mul(x2, xh), t);
        let sum = add(sh, y);
        let sl = add(add(sub(sh, sum), y), sl);
        let r = add(sum, sl);
        let r = sel(vceqq_f32(input, c(f32::INFINITY)), c(f32::INFINITY), r);
        let r = sel(
            vorrq_u32(vcltq_f32(input, c(0.0)), vmvnq_u32(vceqq_f32(input, input))),
            c(f32::NAN),
            r,
        );
        sel(vceqq_f32(input, c(0.0)), c(f32::NEG_INFINITY), r)
    }
}
#[inline(always)]
unsafe fn exp4_ordinary(d: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let q = vcvtnq_s32_f32(mul(
            d,
            c(1.442695040888963407359924681001892137426645954152985934135449406931),
        ));
        let fq = vcvtq_f32_s32(q);
        let s = fma(fq, c(-0.693145751953125), d);
        let s = fma(fq, c(-1.428606765330187045e-06), s);
        let u = fma(
            c(0.000198527617612853646278381),
            s,
            c(0.00139304355252534151077271),
        );
        let u = fma(u, s, c(0.00833336077630519866943359));
        let u = fma(u, s, c(0.0416664853692054748535156));
        let u = fma(u, s, c(0.166666671633720397949219));
        let u = fma(u, s, c(0.5));
        let u = add(fma(mul(s, s), u, s), c(1.0));
        let h = vshrq_n_s32::<1>(q);
        let u = mul(
            u,
            vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(h, vdupq_n_s32(127)))),
        );
        mul(
            u,
            vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(
                vsubq_s32(q, h),
                vdupq_n_s32(127),
            ))),
        )
    }
}
#[inline(always)]
unsafe fn exp4(d: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        // This branch changes only exceptional handling work, never the ordinary arithmetic.
        let valid = vandq_u32(vcgeq_f32(d, c(-104.0)), vcleq_f32(d, c(100.0)));
        if vminvq_u32(valid) == u32::MAX {
            exp4_ordinary(d)
        } else {
            exp4_full(d)
        }
    }
}
#[inline(always)]
unsafe fn log4_ordinary(d: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let e = vsubq_s32(
            vreinterpretq_s32_u32(vandq_u32(
                vshrq_n_u32::<23>(vreinterpretq_u32_f32(mul(d, c(1.0 / 0.75)))),
                vdupq_n_u32(255),
            )),
            vdupq_n_s32(127),
        );
        let m = vreinterpretq_f32_s32(vaddq_s32(
            vreinterpretq_s32_f32(d),
            vshlq_n_s32::<23>(vnegq_s32(e)),
        ));
        let ef = vcvtq_f32_s32(e);
        let sh = mul(ef, c(0.69314718246459960938));
        let sl = fma(
            c(-1.904654323148236017e-09),
            ef,
            fma(c(0.69314718246459960938), ef, neg(sh)),
        );
        let nh = add(m, c(-1.0));
        let nv = add(nh, c(1.0));
        let nl = add(sub(c(-1.0), sub(nh, nv)), sub(m, nv));
        let dh = add(m, c(1.0));
        let dv = sub(dh, c(1.0));
        let dl = add(sub(c(1.0), sub(dh, dv)), sub(m, dv));
        let t = div(c(1.0), dh);
        let xh = mul(nh, t);
        let u = fma(t, nh, neg(xh));
        let v = fma(neg(dl), t, fma(neg(dh), t, c(1.0)));
        let xl = fma(xh, v, fma(nl, t, u));
        let x2 = mul(xh, xh);
        let t = fma(c(0.3027294874), x2, c(0.3996108174));
        let t = fma(t, x2, c(0.6666694880));
        let yh = mul(xh, c(2.0));
        let yl = mul(xl, c(2.0));
        let sum = add(sh, yh);
        let sl = add(add(add(sub(sh, sum), yh), sl), yl);
        let sh = sum;
        let y = mul(mul(x2, xh), t);
        let sum = add(sh, y);
        let sl = add(add(sub(sh, sum), y), sl);
        add(sum, sl)
    }
}
#[inline(always)]
unsafe fn log4(input: float32x4_t) -> float32x4_t {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let valid = vandq_u32(
            vcgeq_f32(input, c(f32::MIN_POSITIVE)),
            vcltq_f32(input, c(f32::INFINITY)),
        );
        if vminvq_u32(valid) == u32::MAX {
            log4_ordinary(input)
        } else {
            log4_full(input)
        }
    }
}
#[inline(always)]
unsafe fn exps(x: f32) -> f32 {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vgetq_lane_f32::<0>(exp4(c(x))) }
}
#[inline(always)]
unsafe fn logs(x: f32) -> f32 {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe { vgetq_lane_f32::<0>(log4(c(x))) }
}

pub(super) unsafe fn swi(out: *mut f32, gate: *const f32, up: *const f32, n: usize) {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let mut i = 0;
        while i + 4 <= n {
            let g = vld1q_f32(gate.add(i));
            let u = vld1q_f32(up.add(i));
            vst1q_f32(out.add(i), mul(div(g, add(c(1.0), exp4(neg(g)))), u));
            i += 4;
        }
        while i < n {
            let g = *gate.add(i);
            *out.add(i) = g / (1.0 + exps(-g)) * *up.add(i);
            i += 1;
        }
    }
}
pub(super) unsafe fn gates(
    sigmoid: *mut f32,
    softplus: *const f32,
    bias: *const f32,
    scale: *const f32,
    out: *mut f32,
    n: usize,
) {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let mut i = 0;
        while i + 4 <= n {
            let sig = vld1q_f32(sigmoid.add(i));
            vst1q_f32(sigmoid.add(i), div(c(1.0), add(c(1.0), exp4(neg(sig)))));
            let x = add(vld1q_f32(softplus.add(i)), vld1q_f32(bias.add(i)));
            let y = log4(add(c(1.0), exp4(x)));
            let y = sel(vcgtq_f32(x, c(20.0)), x, y);
            vst1q_f32(out.add(i), mul(y, vld1q_f32(scale.add(i))));
            i += 4;
        }
        while i < n {
            let s = *sigmoid.add(i);
            *sigmoid.add(i) = 1.0 / (1.0 + exps(-s));
            let x = *softplus.add(i) + *bias.add(i);
            let y = if x > 20.0 { x } else { logs(1.0 + exps(x)) };
            *out.add(i) = *scale.add(i) * y;
            i += 1;
        }
    }
}
pub(super) unsafe fn conv(
    values: *mut f32,
    history: *mut f32,
    weights: *const f32,
    n: usize,
    k: usize,
    fused: i32,
) {
    // SAFETY: validated live buffers and ARM NEON are required by the public entry point.
    unsafe {
        let h = k - 1;
        let mut i = 0;
        while i + 4 <= n {
            let input = vld1q_f32(values.add(i));
            let mut sum = mul(input, vld1q_f32(weights.add(h * n + i)));
            let previous = vld1q_f32(history.add(i));
            let w = vld1q_f32(weights.add(i));
            sum = if fused != 0 {
                fma(previous, w, sum)
            } else {
                add(mul(previous, w), sum)
            };
            for tap in 1..h {
                let at = tap * n + i;
                let current = vld1q_f32(history.add(at));
                let w = vld1q_f32(weights.add(at));
                sum = if fused != 0 {
                    fma(current, w, sum)
                } else {
                    add(mul(current, w), sum)
                };
                vst1q_f32(history.add(at - n), current);
            }
            vst1q_f32(history.add((h - 1) * n + i), input);
            vst1q_f32(values.add(i), div(sum, add(c(1.0), exp4(neg(sum)))));
            i += 4;
        }
        while i < n {
            let input = *values.add(i);
            let mut sum = input * *weights.add(h * n + i);
            let previous = *history.add(i);
            sum += previous * *weights.add(i);
            for tap in 1..h {
                let at = tap * n + i;
                let current = *history.add(at);
                sum += current * *weights.add(at);
                *history.add(at - n) = current;
            }
            *history.add((h - 1) * n + i) = input;
            *values.add(i) = sum / (1.0 + exps(-sum));
            i += 1;
        }
    }
}
