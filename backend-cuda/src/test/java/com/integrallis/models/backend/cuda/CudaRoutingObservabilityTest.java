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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.util.Map;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Gate G2: a run must report which work ran on device, per format and per stage, and an inert
 * accelerator must be visible as inert.
 */
class CudaRoutingObservabilityTest {

  @Test
  @DisplayName("a fresh counter set reports itself as inert")
  void aFreshCounterSetIsInert() {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    assertTrue(counters.inert());
    assertEquals(0L, counters.totalAcceleratedOperations());
    assertEquals(Map.of(), counters.acceleratedOperations());
  }

  @Test
  @DisplayName("accelerated work is counted separately per format and per stage")
  void acceleratedWorkIsCountedPerFormatAndStage() {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    counters.accelerated(GgufTensorType.Q4_K, CudaStage.DECODE_PROJECTION, 3);
    counters.accelerated(GgufTensorType.Q4_K, CudaStage.PREFILL_PROJECTION, 1);
    counters.accelerated(GgufTensorType.Q6_K, CudaStage.DECODE_PROJECTION, 2);

    Map<String, Long> operations = counters.acceleratedOperations();
    assertEquals(3L, operations.get("Q4_K/DECODE_PROJECTION"));
    assertEquals(1L, operations.get("Q4_K/PREFILL_PROJECTION"));
    assertEquals(2L, operations.get("Q6_K/DECODE_PROJECTION"));
    assertEquals(6L, counters.totalAcceleratedOperations());
    assertFalse(counters.inert());
    // A Q4_K_M model must show both formats: the value projection is Q6_K. A report showing
    // only Q4_K means the mixed triple never routed.
    assertTrue(operations.keySet().stream().anyMatch(key -> key.startsWith("Q6_K/")));
  }

  @Test
  @DisplayName("refusals record why, so nothing-accelerated is distinguishable from nothing-helped")
  void refusalsRecordWhy() {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    counters.refused(CudaStage.DECODE_ATTENTION, "shared-kv-prefix-span");
    counters.refused(CudaStage.DECODE_ATTENTION, "shared-kv-prefix-span");
    counters.refused(CudaStage.DECODE_PROJECTION, "unsupported-format");

    Map<String, Long> refusals = counters.refusals();
    assertEquals(2L, refusals.get("DECODE_ATTENTION/shared-kv-prefix-span"));
    assertEquals(1L, refusals.get("DECODE_PROJECTION/unsupported-format"));
    // Still inert: refusals are not acceleration. This is the combination G2 fails.
    assertTrue(counters.inert());
  }

  @Test
  @DisplayName("launch and transfer counts yield the per-token overhead term")
  void launchAndTransferCountsYieldThePerTokenOverheadTerm() {
    // The large-model analysis names per-token launch and transfer latency as the term that
    // decides G4 and the term nobody has measured. These counters are what turn the first GPU
    // run into a measurement of it.
    CudaRoutingCounters counters = new CudaRoutingCounters();
    for (int token = 0; token < 4; token++) {
      for (int launch = 0; launch < 30; launch++) {
        counters.launched();
      }
      counters.copiedToDevice(11_264);
      counters.copiedToHost(11_264);
      counters.decodeStep();
    }
    assertEquals(4L, counters.decodeSteps());
    assertEquals(120L, counters.kernelLaunches());
    assertEquals(30.0, counters.launchesPerDecodeStep(), 1.0e-9);
    assertEquals(2.0, counters.transfersPerDecodeStep(), 1.0e-9);
    assertEquals(22_528.0, counters.activationBytesPerDecodeStep(), 1.0e-9);
  }

  @Test
  @DisplayName("weight uploads are counted apart from activations so streaming is detectable")
  void weightUploadsAreCountedApartFromActivations() {
    // Weights must be uploaded once per tensor for the life of the process. A weight-upload
    // count that grows with tokens means the design regressed into per-token streaming, which
    // the memory arithmetic puts at 0.75x the CPU path -- slower than shipping no GPU at all.
    CudaRoutingCounters counters = new CudaRoutingCounters();
    counters.uploadedWeights(1_048_576);
    counters.uploadedWeights(2_097_152);
    for (int token = 0; token < 10; token++) {
      counters.copiedToDevice(4_096);
      counters.decodeStep();
    }
    assertEquals(2L, counters.weightUploads());
    assertEquals(3_145_728L, counters.weightUploadBytes());
    // Activation traffic must not include the weight uploads, or the per-token figure is wrong.
    assertEquals(40_960L, counters.hostToDeviceBytes());
    assertTrue(
        counters.weightUploads() < counters.decodeSteps(),
        "weight uploads should not grow with tokens");
  }

  @Test
  @DisplayName("counters are safe to update from several threads")
  void countersAreThreadSafe() throws InterruptedException {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    int threads = 8;
    int perThread = 1_000;
    Thread[] workers = new Thread[threads];
    for (int index = 0; index < threads; index++) {
      workers[index] =
          new Thread(
              () -> {
                for (int iteration = 0; iteration < perThread; iteration++) {
                  counters.accelerated(GgufTensorType.Q4_K, CudaStage.DECODE_PROJECTION, 1);
                  counters.launched();
                }
              });
      workers[index].start();
    }
    for (Thread worker : workers) {
      worker.join();
    }
    assertEquals((long) threads * perThread, counters.totalAcceleratedOperations());
    assertEquals((long) threads * perThread, counters.kernelLaunches());
  }
}
