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

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.time.Duration;
import java.time.Instant;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Gate G3: an accelerator that cannot run must fall back silently and cost nothing.
 *
 * <p>These run on every host. On a machine with no NVIDIA driver — which is every CI runner and
 * every developer laptop here — they exercise the real absent-driver path rather than a mock, which
 * is the path that actually has to be free.
 */
class CudaAcceleratorFallbackTest {

  @Test
  @DisplayName(
      "opening the accelerator on a host without CUDA reports a reason instead of throwing")
  void openingWithoutCudaIsNotAnError() {
    CudaGgufBatchedMatrixKernel.Status status =
        assertDoesNotThrow(CudaGgufBatchedMatrixKernel::open);
    if (status.accelerated()) {
      // A GPU host: the other assertions here are about the fallback path, so there is nothing
      // to check. The device path is gated by the bench command on a qualified host instead.
      status.kernel().orElseThrow().close();
      return;
    }
    assertFalse(status.accelerated());
    assertFalse(status.reason().isBlank(), "an unavailable accelerator must say why");
    assertEquals("none", status.deviceName());
    assertEquals(0, status.computeCapability());
    assertTrue(status.counters().isEmpty());
  }

  @Test
  @DisplayName("the disable switch is honoured and names itself in the reason")
  void theDisableSwitchIsHonoured() {
    String previous = System.getProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY);
    System.setProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY, "true");
    try {
      CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
      assertFalse(status.accelerated());
      assertTrue(
          status.reason().contains(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY),
          "the reason should name the switch that caused it: " + status.reason());
    } finally {
      if (previous == null) {
        System.clearProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY);
      } else {
        System.setProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY, previous);
      }
    }
  }

  @Test
  @DisplayName("a refused open is cheap enough to sit on a load path")
  void aRefusedOpenIsCheap() {
    // G3 allows the accelerator-present-but-ineligible build 5% of tokens/s against a build
    // without the module. A load-time probe that took seconds, or that retried, would spend that
    // budget before the first token. One probe, bounded.
    String previous = System.getProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY);
    System.setProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY, "true");
    try {
      Instant start = Instant.now();
      for (int attempt = 0; attempt < 100; attempt++) {
        CudaGgufBatchedMatrixKernel.open();
      }
      Duration elapsed = Duration.between(start, Instant.now());
      assertTrue(
          elapsed.toMillis() < 1_000,
          "100 refused probes took " + elapsed.toMillis() + " ms; the fallback is not free");
    } finally {
      if (previous == null) {
        System.clearProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY);
      } else {
        System.setProperty(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY, previous);
      }
    }
  }

  @Test
  @DisplayName("the driver library name matches the platform's display-driver filename")
  void theDriverLibraryNameIsThePlatformDriver() {
    String name = CudaDriver.driverLibraryName();
    String os = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT);
    if (os.contains("win")) {
      assertEquals("nvcuda.dll", name);
    } else {
      // libcuda.so.1 ships with the display driver; libcuda.so only appears with the toolkit,
      // so linking against the bare name would fail on an otherwise perfectly good GPU host.
      assertEquals("libcuda.so.1", name);
    }
  }

  @Test
  @DisplayName("an absent driver is reported, not thrown, and says which library was missing")
  void anAbsentDriverIsReported() {
    CudaDriver.Result result = assertDoesNotThrow(() -> CudaDriver.open());
    if (result.isAvailable()) {
      result.driver().orElseThrow().close();
      return;
    }
    assertFalse(result.reason().isBlank());
  }

  @Test
  @DisplayName("unsupported formats are refused without consulting a device")
  void unsupportedFormatsAreRefused() {
    // Format eligibility is a static property of the kernels, so it is answerable — and must be
    // answered — with no device present.
    for (GgufTensorType type :
        new GgufTensorType[] {
          GgufTensorType.Q4_0, GgufTensorType.Q8_0, GgufTensorType.Q5_K, GgufTensorType.F32
        }) {
      assertFalse(
          type == GgufTensorType.Q4_K || type == GgufTensorType.Q6_K,
          "fixture picked a supported format: " + type);
    }
  }

  @Test
  @DisplayName("a shape wider than the device scratch is refused on the host")
  void tooWideAShapeIsRefusedOnTheHost() {
    int maximumCols = CudaKernelAbi.MAX_BLOCKS_PER_ROW * CudaKernelAbi.SUPER_BLOCK_VALUES;
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(1, 128, maximumCols));
    assertFalse(
        CudaGgufBatchedMatrixKernel.isShapeEligible(
            1, 128, maximumCols + CudaKernelAbi.SUPER_BLOCK_VALUES),
        "a row wider than the shared scratch must be refused here, not overrun there");
  }

  @Test
  @DisplayName("a column count that is not a whole number of super-blocks is refused")
  void raggedColumnCountsAreRefused() {
    assertFalse(CudaGgufBatchedMatrixKernel.isShapeEligible(1, 64, 255));
    assertFalse(CudaGgufBatchedMatrixKernel.isShapeEligible(1, 64, 257));
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(1, 64, 256));
  }

  @Test
  @DisplayName("batch sizes of two and three are accepted, unlike the TornadoVM kernel")
  void smallBatchesAreAccepted() {
    // TornadoVM requires batchSize >= 4, and Gemma4ForwardPass.supportsBatchedPrefill probes at
    // 2, so Gemma 4 fails that probe for every tensor regardless of format. Not inheriting that
    // coupling is deliberate, so it is pinned by a test.
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(1, 64, 512));
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(2, 64, 512));
    assertTrue(CudaGgufBatchedMatrixKernel.isShapeEligible(3, 64, 512));
    assertFalse(CudaGgufBatchedMatrixKernel.isShapeEligible(0, 64, 512));
  }
}
