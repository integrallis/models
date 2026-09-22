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

import java.util.List;

/**
 * The contract between this Java binding and the PTX module it loads.
 *
 * <p>Kept in one place so the launch code, the packaging descriptor and the Rust {@code
 * #[unsafe(no_mangle)]} names cannot drift apart silently. {@code PtxModuleContractTest} asserts
 * these values against the built PTX itself, so a rename on either side fails the build rather than
 * producing a kernel that is never found at runtime.
 */
final class CudaKernelAbi {

  /**
   * Version of the kernel entry-point contract.
   *
   * <p>Bump whenever a kernel's name, parameter list or launch contract changes. Mirrors {@code
   * PTX_ABI_VERSION} in the Rust crate.
   */
  static final int VERSION = 2;

  /** Fused Q4_K dequantise-and-multiply projection. */
  static final String Q4_K_PROJECTION = "models_q4k_decode_projection";

  /** Fused Q6_K dequantise-and-multiply projection. */
  static final String Q6_K_PROJECTION = "models_q6k_decode_projection";

  /** Grouped-query attention for the single-token decode step. */
  static final String GQA_ATTENTION = "models_gqa_decode_attention";

  /** Every entry point the module must export, in declaration order. */
  static final List<String> KERNEL_NAMES = List.of(Q4_K_PROJECTION, Q6_K_PROJECTION, GQA_ATTENTION);

  /**
   * Threads per CUDA block for every kernel: one warp.
   *
   * <p>Not a tuning knob. The projection kernels stride super-blocks by this value and fold the
   * float scales on lane zero after a barrier; the attention kernel strides positions and output
   * dimensions by it. Changing it without changing the kernels changes the answer.
   */
  static final int BLOCK_THREADS = 32;

  /**
   * Super-blocks one row may hold, bounding the device's shared scratch.
   *
   * <p>Mirrors {@code MAX_BLOCKS_PER_ROW} in the Rust crate: {@code cols <= 128 * 256 = 32,768}. A
   * wider tensor is refused on the host rather than overrunning the scratch on the device.
   */
  static final int MAX_BLOCKS_PER_ROW = 128;

  /** Weights per K-quant super-block. */
  static final int SUPER_BLOCK_VALUES = 256;

  /** Activation values covered by one Q8_K partial sum. */
  static final int SUM_BLOCK_VALUES = 16;

  private CudaKernelAbi() {}
}
