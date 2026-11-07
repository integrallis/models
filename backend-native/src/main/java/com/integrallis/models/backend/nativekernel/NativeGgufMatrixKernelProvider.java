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
package com.integrallis.models.backend.nativekernel;

import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import java.nio.file.Path;
import java.util.Optional;

/** Makes the shipping Rust GGUF kernels available to multimodal decoders. */
public final class NativeGgufMatrixKernelProvider implements GgufMatrixKernelProvider {
  @Override
  public Optional<GgufBatchedMatrixKernel> open(int threads) {
    String configured = System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY);
    if (configured == null || configured.isBlank())
      configured = System.getenv(RustFfmBackend.LIBRARY_PATH_ENV);
    if (configured != null && !configured.isBlank())
      return Optional.of(RustGgufBatchedMatrixKernel.open(Path.of(configured), threads));
    try {
      return Optional.of(RustGgufBatchedMatrixKernel.openBundled(threads));
    } catch (IllegalStateException unavailable) {
      System.getLogger(getClass().getName())
          .log(
              System.Logger.Level.DEBUG,
              "Native GGUF kernels unavailable: {0}",
              unavailable.getMessage());
      return Optional.empty();
    }
  }

  @Override
  public com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel openAttention(
      int threads) {
    String configured = System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY);
    if (configured == null || configured.isBlank())
      configured = System.getenv(RustFfmBackend.LIBRARY_PATH_ENV);
    try {
      NativeDebertaKernels kernels =
          configured == null || configured.isBlank()
              ? NativeDebertaKernels.load(threads)
              : NativeDebertaKernels.load(Path.of(configured), threads);
      if (kernels.supportsCausalAttention()) return new NativeCausalAttention(kernels);
      kernels.close();
      return com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel.none();
    } catch (java.io.IOException failure) {
      throw new IllegalStateException("Cannot load causal attention kernels", failure);
    }
  }
}
