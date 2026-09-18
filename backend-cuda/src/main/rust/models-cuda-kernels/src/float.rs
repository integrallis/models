// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! The one floating-point operation `core` does not provide.
//!
//! `f32::mul_add` is an inherent method on `std`'s `f32`, not `core`'s, so it disappears under
//! `no_std` — which is every `nvptx64` build. There is no `core` equivalent on stable.
//!
//! This matters more than an ergonomic wrinkle would: a fused multiply-add and a separate
//! multiply-then-add give **different** results, and the whole bit-exactness argument in
//! [`crate::kquant`] rests on the device performing the same fused operation the CPU kernel
//! does. Rewriting `a.mul_add(b, c)` as `a * b + c` would silently change the answer.
//!
//! Both arms below lower to `llvm.fma.f32`: the host through `f32::mul_add`, the device through
//! `core::intrinsics::fmaf32`, which the NVPTX backend emits as `fma.rn.f32` — the IEEE-754
//! fused operation with round-to-nearest-even. Verified by reading the generated PTX.
//!
//! Recorded as `CU-002` in `backend-cuda/UPSTREAM.md`.

/// Fused multiply-add: `a * b + c` with a single rounding.
///
/// Identical on host and device, which is what lets the host tests stand in for the device.
#[inline(always)]
pub fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        core::intrinsics::fmaf32(a, b, c)
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        a.mul_add(b, c)
    }
}
