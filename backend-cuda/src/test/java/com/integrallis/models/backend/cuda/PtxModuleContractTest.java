/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * The packaged PTX module must be the one this Java binding expects.
 *
 * <p>Runs on any host: it reads the artifact out of the classpath and inspects the text. No GPU, no
 * CUDA driver. What it cannot do is assemble the PTX — that needs {@code ptxas} from the CUDA
 * toolkit, which in turn needs a Linux host. The GPU-host checklist in {@code README.md} names that
 * step explicitly rather than leaving it implied.
 */
class PtxModuleContractTest {

  @Test
  @DisplayName("the packaged module resolves and its recorded digest matches its bytes")
  void thePackagedModuleResolvesAndVerifies() {
    BundledPtxModule module = BundledPtxModule.resolve();
    assertTrue(module.sha256().matches("[0-9a-f]{64}"));
    assertTrue(module.ptx().length > 0);
    // resolve() recomputes and compares the digest itself, so reaching here is the assertion.
  }

  @Test
  @DisplayName("the module exports exactly the kernels the Java binding resolves")
  void theModuleExportsTheExpectedKernels() {
    BundledPtxModule module = BundledPtxModule.resolve();
    String ptx = new String(module.ptx(), StandardCharsets.UTF_8);
    for (String kernel : CudaKernelAbi.KERNEL_NAMES) {
      assertTrue(
          module.kernelNames().contains(kernel),
          "descriptor omits " + kernel + ": " + module.kernelNames());
      assertTrue(ptx.contains(".visible .entry " + kernel + "("), "PTX does not export " + kernel);
    }
  }

  @Test
  @DisplayName("the module targets the compute capability the driver check enforces")
  void theModuleTargetsTheEnforcedComputeCapability() {
    BundledPtxModule module = BundledPtxModule.resolve();
    assertEquals("sm_80", module.target());
    assertEquals(80, module.minimumComputeCapability());
    assertEquals(
        CudaDriver.MINIMUM_COMPUTE_CAPABILITY,
        module.minimumComputeCapability(),
        "the driver's capability floor and the PTX target must agree, or a device is admitted "
            + "that cannot run the module");
  }

  @Test
  @DisplayName("the module uses fused multiply-add, which is what makes it bit-exact")
  void theModuleUsesFusedMultiplyAdd() {
    // f32::mul_add does not exist in core, so the kernels route through a shim. If that shim
    // ever degrades to a * b + c the kernels still compile, still run, and stop being bit-exact
    // with the CPU K-quant path -- a G1 failure that only a real model would reveal.
    String ptx = new String(BundledPtxModule.resolve().ptx(), StandardCharsets.UTF_8);
    assertTrue(ptx.contains("fma.rn.f32"), "no fused multiply-add in the compiled kernels");
  }

  @Test
  @DisplayName("the module keeps its scratch in shared memory, not global")
  void theModuleKeepsItsScratchInSharedMemory() {
    // link_section = ".shared" silently places the static in GLOBAL memory. That compiles, runs,
    // and races across CUDA blocks. The kernels declare the scratch through global_asm! instead;
    // this pins the outcome rather than the mechanism.
    String ptx = new String(BundledPtxModule.resolve().ptx(), StandardCharsets.UTF_8);
    assertTrue(
        ptx.contains(".shared .align 4 .b8 models_cuda_scratch"),
        "the scratch is not declared in shared memory");
    assertTrue(ptx.contains("st.shared.b32"), "the scratch is never written through shared memory");
    assertTrue(ptx.contains("ld.shared.b32"), "the scratch is never read through shared memory");
    assertFalse(
        ptx.contains(".global .align 4 .b8 models_cuda_scratch"),
        "the scratch landed in global memory; every CUDA block would race on it");
  }

  @Test
  @DisplayName("the descriptor names the exact toolchain that produced the module")
  void theDescriptorNamesTheToolchain() {
    // A performance result has to be attributable to a toolchain, or a rustc regression is
    // indistinguishable from one of our changes.
    BundledPtxModule module = BundledPtxModule.resolve();
    assertTrue(
        module.toolchain().startsWith("nightly-"),
        "expected a pinned nightly, got: " + module.toolchain());
  }
}
