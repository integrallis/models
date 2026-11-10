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
package com.integrallis.models.bench;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;

/** Exact filesystem source used by an in-process benchmark. */
record PureJavaModelSource(String identity, Path artifact) {

  PureJavaModelSource {
    if (identity == null || identity.isBlank()) {
      throw new IllegalArgumentException("identity must not be blank");
    }
    String resolvedIdentity = identity.trim();
    identity = resolvedIdentity;
    Path resolvedArtifact = requireArtifact(Objects.requireNonNull(artifact, "artifact"));
    artifact = resolvedArtifact;
  }

  static PureJavaModelSource resolve(String pathValue) {
    if (pathValue == null || pathValue.isBlank()) {
      throw new IllegalArgumentException("in-process backends require --model");
    }
    String identity = pathValue.trim();
    return new PureJavaModelSource(identity, Path.of(identity));
  }

  PureJavaBackend load() {
    return PureJavaBackend.load(artifact);
  }

  /**
   * Loads with an injected native kernel when one is installed, which is the route the product
   * takes.
   *
   * <p>{@link #load()} passes no kernel, so every profile taken through it measures the pure-Java
   * Panama path. That is a legitimate thing to profile and remains the default, but it is not what
   * a qualified model runs: a profile of an 8-thread LLaVA decode taken through {@link #load()}
   * spends 9,780 of 11,204 samples in {@code ggufQ4_KQ8_KLongOffsetRowDot} and only 43 in native
   * frames, because the Rust kernel was never opened. Reading that as production behaviour is the
   * mistake this method exists to prevent.
   *
   * <p>Mirrors {@code LlavaBackend.decoder()}: the first {@link GgufMatrixKernelProvider} that
   * answers supplies both the matrix and attention kernels, so there is no compile-time dependency
   * on the native module and the selection rule cannot drift from production.
   *
   * @param threads the pool size to request, as the qualification harness pins it
   * @throws IllegalStateException if no provider answers, rather than silently measuring Java
   */
  PureJavaBackend loadNative(int threads) {
    for (GgufMatrixKernelProvider provider : ServiceLoader.load(GgufMatrixKernelProvider.class)) {
      Optional<GgufBatchedMatrixKernel> available = provider.open(threads);
      if (available.isEmpty()) {
        continue;
      }
      GgufBatchedMatrixKernel kernel = available.orElseThrow();
      BatchedCausalAttentionKernel attention;
      try {
        attention = provider.openAttention(threads, kernel);
      } catch (RuntimeException | Error failure) {
        try (kernel) {
          throw failure;
        }
      }
      return PureJavaBackend.load(artifact, kernel, attention);
    }
    throw new IllegalStateException(
        "No GgufMatrixKernelProvider is installed, so --native cannot measure the production "
            + "route. Build the distribution with -PmodelsBenchNative=true.");
  }

  private static Path requireArtifact(Path artifact) {
    if (!Files.isRegularFile(artifact)) {
      throw new IllegalArgumentException("artifact does not exist: " + artifact);
    }
    return artifact;
  }
}
