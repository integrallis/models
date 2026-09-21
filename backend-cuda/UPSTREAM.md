# Rust-to-PTX toolchain ledger

Updated: 2026-09-18

Limitations hit while building `backend-cuda`. Follows the spirit of
`references/JVM_VECTOR_API_UPSTREAM_LEDGER.md`: **a reproducible case per entry, not a
complaint.** Each row separates what we needed, what the toolchain does today, the minimal
reproduction, and whether a workaround exists.

Every entry below was reproduced on this machine on 2026-09-18 unless the row says otherwise:
macOS 25.6.0 x86_64, `rustc 1.100.0-nightly (330d31712 2026-09-17)`, target
`nvptx64-nvidia-cuda`. **No GPU and no CUDA toolkit were involved** — everything here is a
compile-time finding, which is itself the useful part: the whole toolchain is exercisable without
a device.

Upstream issues, comments and pull requests require explicit maintainer approval from the project
owner before submission. Until then reproducers and proposed text stay here.

## Reporting gate

Before opening anything upstream, a candidate needs:

1. a standalone reproducer that does not depend on Integrallis code;
2. the exact toolchain date and target triple;
3. the emitted PTX (or the error), quoted;
4. a check that the behaviour still holds on the newest nightly;
5. a tracker search showing it is not already covered.

Rows below are marked with which of these they already have.

## Summary

| ID | Finding | Severity | Workaround | Upstream status |
| --- | --- | --- | --- | --- |
| `CU-001` | `-C target-cpu=sm_XX` is rejected against a prebuilt `core`/`compiler_builtins` | Friction | `-Zbuild-std` | Not searched |
| `CU-002` | No `fma` for `f32` in `core`; `mul_add` is `std`-only | **Correctness trap** | `core::intrinsics::fmaf32` (nightly) | Not searched |
| `CU-003` | `#[link_section = ".shared"]` silently emits into **global** address space | **Silent miscompile** | `global_asm!` + inline PTX | Not searched — best candidate |
| `CU-004` | `core::arch::nvptx` exposes no shared memory, warp shuffle, or atomics | Gap | Inline PTX `asm!` | Not searched |
| `CU-005` | No floating-point transcendentals in `core`; no libdevice linkage for `nvptx64` | Gap | Own `expf`, contract stated | Not searched |
| `CU-006` | PTX kernels require three nightly features | Expected | Pinned nightly | Tracking issue exists upstream |
| `CU-007` | PTX cannot be assembled without `ptxas` (CUDA toolkit, Linux) | Environmental | GPU-host gate | Not a bug |

---

## CU-001 — `-C target-cpu` is rejected against a prebuilt `core`

**What we needed.** PTX targeting compute capability 8.0, the campaign's device floor.

**What the toolchain does.** `rustup target add nvptx64-nvidia-cuda` installs a `core` and
`compiler_builtins` built with no `-C target-cpu`. Setting it on our own crate then trips the ABI
consistency check, and the emitted module stays at the target default `sm_70`.

**Reproduction.**

```console
$ RUSTFLAGS="-C target-cpu=sm_80" cargo +nightly build --release --target nvptx64-nvidia-cuda
error: `-Ctarget-cpu=sm_80` in this crate is incompatible with `-Ctarget-cpu` being unset in
       dependency `compiler_builtins`
  = help: unset `-Ctarget-cpu` in this crate or set `-Ctarget-cpu=sm_80` in `compiler_builtins`
  = help: if you are sure this will not cause problems, you may use
          `-Cunsafe-allow-abi-mismatch=target-cpu` to silence this error
```

**Workaround, and the one we rejected.** `-Zbuild-std=core,compiler_builtins` rebuilds both for
`sm_80` and the build succeeds with `.target sm_80` in the output. That is what
`backend-cuda/build.gradle.kts` does.

`-Cunsafe-allow-abi-mismatch=target-cpu` also works and is much faster (no `core` rebuild), but it
silences a check rather than satisfying it, and a target-cpu mismatch on a device backend is
exactly the class of thing that shows up as a wrong answer rather than a link error. Rejected.

**Note.** The diagnostic is good and the workaround is documented in it. This may be working as
intended for a target whose `rust-std` is necessarily generic; the useful upstream ask would be a
prebuilt `nvptx64` std per common `sm_` level, or a note in the target docs that `-Zbuild-std` is
the expected route. Has (1), (2), (3), (4). Needs (5).

---

## CU-002 — no `fma` for `f32` in `core`

**What we needed.** A fused multiply-add on the device that rounds exactly as the CPU kernel's
does. This is load-bearing: the bit-exactness argument for the K-quant projections rests on the
device performing the *same fused* operation, and `a * b + c` gives a different answer.

**What the toolchain does.** `f32::mul_add` is an inherent method on `std`'s `f32`, not `core`'s,
so it does not exist under `no_std` — which is every `nvptx64` build. There is no stable `core`
equivalent.

**Reproduction.**

```rust
#![no_std]
pub fn f(a: f32, b: f32, c: f32) -> f32 { a.mul_add(b, c) }
```

```console
$ cargo +nightly build --release --target nvptx64-nvidia-cuda
error[E0599]: no method named `mul_add` found for type `f32` in the current scope
```

**Workaround.** `core::intrinsics::fmaf32` behind `feature(core_intrinsics)`. It lowers to
`llvm.fma.f32` and the NVPTX backend emits `fma.rn.f32`, which is what we want. Confined to
`src/float.rs` so there is one place to change if a stable spelling appears.

**Why this is a trap and not just friction.** The compiler error names a method, and the obvious
"fix" a reader reaches for is `a * b + c`. That compiles, runs, produces plausible numbers, and
silently breaks bit-exactness with the CPU path — a failure only a real-model token comparison
would catch. `PtxModuleContractTest.theModuleUsesFusedMultiplyAdd` and the `preparePtxResources`
build check both assert `fma.rn.f32` appears in the emitted PTX, so the degradation cannot land
quietly here. Upstream, the ask is a stable `core::f32::fma` (or making the inherent method
available in `core`), which would help every `no_std` numeric crate, not only GPU ones.

Has (1), (2), (3), (4). Needs (5) — `core_float_math` and similar tracking issues may already
cover this; search before filing.

---

## CU-003 — `link_section = ".shared"` silently lands in global memory

**Best upstream candidate in this ledger.** It is a silent miscompile, not a diagnostic gap.

**What we needed.** Shared memory: the projection kernels stage exact integer per-super-block
partials there, then one lane folds the float scales in block order.

**What the toolchain does.** The idiom a reader would reach for — and that circulates as the way
to get shared memory out of rustc's NVPTX backend — is a `static mut` with
`#[unsafe(link_section = ".shared")]`. It compiles without a warning and emits the static into
the **global** address space.

**Reproduction.**

```rust
#![no_std]
#![feature(abi_ptx, stdarch_nvptx)]
use core::arch::nvptx::*;

#[unsafe(link_section = ".shared")]
static mut SCRATCH: [i32; 64] = [0; 64];

#[unsafe(no_mangle)]
pub unsafe extern "ptx-kernel" fn probe(out: *mut i32) {
    let t = unsafe { _thread_idx_x() } as usize;
    unsafe { *(&raw mut SCRATCH).cast::<i32>().add(t) = t as i32 * 3 };
    unsafe { _syncthreads() };
    if t == 0 {
        let mut s = 0;
        for i in 0..64 { s += unsafe { *(&raw const SCRATCH).cast::<i32>().add(i) }; }
        unsafe { *out = s };
    }
}
```

Emitted PTX (abridged, symbol name demangled for readability):

```ptx
.global .align 4 .b8 SCRATCH[256];          // <- expected .shared
...
st.global.b32   [%rd5], %r2;                 // <- expected st.shared.b32
bar.sync        0;
ld.global.b32   %r3, [SCRATCH];              // <- expected ld.shared.b32
```

**Consequences.** Every CUDA block in the grid writes the same global array, so the kernel races
across blocks and produces wrong results non-deterministically. It is also far slower than
intended. Nothing warns.

**Workaround.** Declare the array with `global_asm!` and access it through inline PTX:

```rust
global_asm!(".shared .align 4 .b8 models_cuda_scratch[4096];");

unsafe fn scratch_store(index: u32, value: i32) {
    asm!("{{",
         ".reg .u64 %scratch;",
         "mov.u64 %scratch, models_cuda_scratch;",
         "add.u64 %scratch, %scratch, {offset};",
         "st.shared.b32 [%scratch], {value};",
         "}}",
         offset = in(reg64) (index as u64) * 4, value = in(reg32) value);
}
```

This emits `.shared .align 4 .b8 models_cuda_scratch[4096];` and real `st.shared.b32` /
`ld.shared.b32`. Verified in the packaged module; `PtxModuleContractTest` and the build both
assert the global-memory form does **not** reappear.

**Proposed upstream ask.** Either honour `.shared` (and `.const`) as address-space placement on
the NVPTX target, or reject the attribute with an error naming the supported spelling. Emitting
a silently different address space is the worst of the three. Has (1), (2), (3), (4). Needs (5).

---

## CU-004 — `core::arch::nvptx` is missing the cooperative primitives

**What the toolchain provides.** The whole public surface of
`library/stdarch/crates/core_arch/src/nvptx/mod.rs` is: `_syncthreads`, the nine
block/grid/thread index accessors, `trap`, `vprintf`, `malloc`, `free`, `__assert_fail`.

**What is absent.** Warp shuffle (`shfl.sync.*`), any shared-memory declaration or accessor,
device atomics, `ballot`/`vote`, `dp4a`, and the tensor-core `mma` family.

**Reproduction.** Compile-time: no such items exist to import.

**Workaround.** Inline PTX through `core::arch::asm!` with `feature(asm_experimental_arch)`.
Confirmed working for all three of the constructs we probed:

```ptx
shfl.sync.idx.b32 %r1, %r2, 0, 31, 0xffffffff;
dp4a.u32.s32      %r1, %r2, %r3, 0;
st.shared.b32     [kq_scratch+...], %r2;
```

**Impact on this branch.** None on correctness — the kernels need only `_syncthreads` plus the
shared-memory workaround from CU-003. It is relevant to the *next* optimisation: `dp4a` would
collapse the Q4_K inner loop's four 8-bit products into one instruction, and the integer domain
is exact so it cannot perturb the bit-exactness argument. Left for after the first device
measurement decides whether the kernels are instruction-bound at all.

Has (1), (2), (3). Needs (4), (5).

---

## CU-005 — no floating-point transcendentals for `nvptx64`

**What we needed.** `expf`, for the attention softmax.

**What the toolchain does.** `core` has no float transcendentals at all (they live in `std`,
backed by the platform libm), and rustc emits no linkage to NVIDIA's `libdevice` bitcode for
`nvptx64`. `ex2.approx.f32` is available through inline PTX but is documented at about two ulp.

**Reproduction.** As CU-002, with `x.exp()`:

```console
error[E0599]: no method named `exp` found for type `f32` in the current scope
```

**Workaround.** `attention::expf` in this crate: Cody-Waite reduction, degree-5 minimax
polynomial, exponent construction by bit manipulation. Used on **both** host and device, which is
what makes the off-device parity test meaningful. Measured against the platform `expf` over
`[-120, 40]` at 1e-6 relative error, asserted in `tests/attention_parity.rs`.

**This is the one place the kernels are not bit-exact with the CPU path**, and it is why attention
carries a stated relative-L2 contract (2.0e-5, the same one the sibling TornadoVM attention branch
adopted) while the projections carry none.

**Proposed upstream ask.** A documented way to link `libdevice` from a rustc `nvptx64` build would
remove a whole class of hand-rolled math from every Rust GPU crate. Lower priority than CU-003.
Has (1), (2), (3). Needs (4), (5).

---

## CU-006 — PTX kernels require nightly

`extern "ptx-kernel"` needs `feature(abi_ptx)`, the index intrinsics need
`feature(stdarch_nvptx)`, and inline PTX needs `feature(asm_experimental_arch)`. Plus
`feature(core_intrinsics)` for CU-002. Expected for a tier-2 target; recorded so the nightly pin
is explained rather than mysterious.

The crate pins `nightly-2026-09-17` exactly, in
`src/main/rust/models-cuda-kernels/rust-toolchain.toml`, and `build.gradle.kts` fails the build if
the two drift. The pin is also written into the artifact descriptor and every report JSON, so a
toolchain regression is distinguishable from one of our changes.

---

## CU-007 — PTX cannot be assembled off a CUDA host

`ptxas` ships with the CUDA toolkit, which is Linux (or Windows) only. So on this machine we can
**generate** PTX and inspect it, but not assemble it. A malformed inline-PTX fragment would be
caught by `ptxas`, not by rustc.

Not a toolchain bug; recorded because it bounds what the off-device tests can claim.
`backend-cuda/README.md` lists "assemble the module with `ptxas -arch=sm_80`" as the first step of
the GPU-host checklist, before any timing, precisely so this gap is closed explicitly rather than
assumed away.

---

## Note on the two tracks named in the campaign amendment

The amendment names `cuda-oxide` as a Rust-to-PTX compiler released 2026-09-08 and `cutile-rs` as
the tile track. Checking the registry rather than assuming:

- **`cuda-oxide`** on crates.io is `0.4.0`, last updated **2021-06-16**, described as "a
  high-level, rusty wrapper over CUDA" (`github.com/Protryon/cuda-oxide`). It is a *driver API
  binding*, not a compiler, and it is five years stale. The amendment's description does not match
  this crate; it is most likely a name collision.
- **`cutile-rs`** does not exist on crates.io. **`cutile`** does: `0.3.1`, published
  2026-09-04, "cuTile Rust lets programmers safely author and execute tile kernels directly in
  Rust". That is real and current, and is presumably what the amendment meant.

Neither was adopted, and the reasoning is in `README.md` under "Which track, and why". The short
version: rustc's own in-tree `nvptx64-nvidia-cuda` target needs no third-party compiler at all,
which fits the amendment's own policy boundary ("no vendor math library, no third-party inference
runtime") better than either. `cutile` additionally pulls `cuda-core`, `cuda-async`,
`cutile-compiler`, `cutile-ir`, `tokio` and `dashmap`, and *executes* kernels as well as authoring
them — it would own part of the host side that the amendment assigns to Java.

This is a correction to the amendment's premise, not a deviation from its policy. Flagged here
because "we picked a third thing" should be visible rather than buried.
