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

import com.integrallis.models.backend.purejava.spi.DenseProjectionKernels;
import com.integrallis.models.backend.purejava.spi.DenseProjectionProvider;
import java.nio.file.Path;
import java.util.Optional;

/** Reuses the owned Rust GEMM implementation used by the DeBERTa runtime. */
public final class NativeDenseProjectionProvider implements DenseProjectionProvider {
  @Override
  public Optional<DenseProjectionKernels> open(int threads) {
    String configured = System.getProperty("models.native.kernels.library");
    if (configured == null || configured.isBlank())
      configured = System.getenv("MODELS_NATIVE_KERNELS_LIBRARY");
    if (configured != null && !configured.isBlank()) {
      return Optional.of(coordinate(NativeDebertaKernels.load(Path.of(configured), threads)));
    }
    try {
      return Optional.of(coordinate(NativeDebertaKernels.load(threads)));
    } catch (java.io.IOException | IllegalStateException unavailable) {
      System.getLogger(getClass().getName())
          .log(
              System.Logger.Level.DEBUG,
              "Dense native kernels unavailable: {0}",
              unavailable.getMessage());
      return Optional.empty();
    }
  }

  private static DenseProjectionKernels coordinate(DenseProjectionKernels kernels) {
    // Match RustGgufBatchedMatrixKernel: idle Java workers must not compete with native GEMM.
    RustGgufBatchedMatrixKernel.parkJavaExecutor();
    return kernels;
  }
}
