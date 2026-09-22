// Copyright 2025-2026 Integrallis Software, LLC
// SPDX-License-Identifier: Apache-2.0

//! Models-owned GPU compute kernels, written in Rust and compiled to PTX.
//!
//! This crate is deliberately tiny. It holds the two hot paths that the CPU Vector API cannot
//! carry for a 27B-class Q4_K_M model and nothing else:
//!
//! * a fused K-quant dequantise-and-multiply projection (Q4_K, and Q6_K because Q4_K_M promotes
//!   the value and output projections to it);
//! * grouped-query attention for the single-token decode step.
//!
//! Model parsing, the tokenizer, the graph, KV cache ownership, sampling and the generation loop
//! all stay in Java, exactly as `backend-native` states for the CPU shim. There is no engine
//! here, no vendor math library, and no inference runtime.
//!
//! # Layout
//!
//! * [`kquant`] and [`attention`] hold the arithmetic. They are `no_std`-compatible and compile
//!   unchanged for the host and for `nvptx64-nvidia-cuda`.
//! * [`device`] holds the `extern "ptx-kernel"` entry points and exists only when targeting
//!   `nvptx64`. It contains scheduling, not arithmetic.
//!
//! The split is what makes the off-device tests meaningful: `cargo test` on any host — no GPU,
//! no CUDA toolkit — exercises the same functions the PTX calls, against the CPU reference.
//! What a GPU host still has to prove is listed in `backend-cuda/README.md`.
//!
//! # Build
//!
//! ```text
//! cargo +nightly-2026-09-17 build --release --target nvptx64-nvidia-cuda \
//!     -Zbuild-std=core,compiler_builtins -Zbuild-std-features=compiler-builtins-mem
//! ```
//!
//! The Gradle module drives this; see `backend-cuda/build.gradle.kts`. Toolchain limitations
//! encountered are recorded, with reproductions, in `backend-cuda/UPSTREAM.md`.

#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(
    target_arch = "nvptx64",
    feature(abi_ptx, stdarch_nvptx, asm_experimental_arch, core_intrinsics)
)]
#![cfg_attr(target_arch = "nvptx64", allow(internal_features))]
#![deny(missing_docs)]

pub mod attention;
pub mod float;
pub mod kquant;

#[cfg(target_arch = "nvptx64")]
pub mod device;

/// The device has no unwinder and nowhere to report a panic, so it stops the thread.
///
/// Every kernel is launched only after the host has validated its shapes, so reaching this is a
/// bug in the host checks rather than a runtime condition.
#[cfg(target_arch = "nvptx64")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // SAFETY: `trap` is the documented PTX abort and does not return.
    unsafe { core::arch::nvptx::trap() }
}

/// ABI version of the PTX module, checked by the Java loader before any launch.
///
/// Bump whenever a kernel's name, parameter list or launch contract changes. The Java side
/// refuses a module whose recorded ABI differs, exactly as `backend-native` does for its
/// shared library.
pub const PTX_ABI_VERSION: u32 = 2;

/// Kernel entry-point names the Java loader resolves in the compiled module.
///
/// Kept here so the Java constant and the `#[unsafe(no_mangle)]` names cannot drift apart
/// without this list changing too; `tests/abi.rs` asserts they are the names the PTX actually
/// exports.
pub const KERNEL_NAMES: [&str; 3] = [
    "models_q4k_decode_projection",
    "models_q6k_decode_projection",
    "models_gqa_decode_attention",
];
