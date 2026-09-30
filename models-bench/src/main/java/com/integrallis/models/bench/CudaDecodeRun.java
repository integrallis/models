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

import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;

/**
 * Gate G4: decode throughput of the accelerated arm against the CPU control on the same host.
 *
 * <p>Throughput is pooled — total tokens over total time — not the mean of the per-prompt rates. A
 * mean of ratios weights a short prompt the same as a long one and is not the quantity the gate is
 * about.
 */
final class CudaDecodeRun {

  private CudaDecodeRun() {}

  /** One arm's measurement. */
  record ArmMeasurement(
      String arm,
      boolean accelerated,
      int promptCount,
      int promptTokens,
      int generatedTokens,
      int decodeSteps,
      int sequencesHittingEndOfGeneration,
      long prefillNanos,
      long decodeNanos,
      long wallClockNanos,
      double prefillTokensPerSecond,
      double decodeTokensPerSecond,
      long residentBytesAtEnd,
      long heapUsedBytesAtEnd,
      long peakDeviceBytes,
      long deviceTotalMemoryBytes) {}

  /** Aggregates one arm's sequences into the figures the gate report carries. */
  static ArmMeasurement summarise(
      String arm,
      boolean accelerated,
      List<GreedyDecode.Sequence> sequences,
      long wallClockNanos,
      long peakDeviceBytes,
      long deviceTotalMemoryBytes) {
    Objects.requireNonNull(arm, "arm");
    Objects.requireNonNull(sequences, "sequences");
    if (sequences.isEmpty()) {
      throw new IllegalArgumentException("arm " + arm + " produced no sequences");
    }
    int promptTokens = 0;
    int generatedTokens = 0;
    int decodeSteps = 0;
    int endOfGeneration = 0;
    long prefillNanos = 0;
    long decodeNanos = 0;
    for (GreedyDecode.Sequence sequence : sequences) {
      promptTokens += sequence.promptTokenCount();
      generatedTokens += sequence.tokenIds().size();
      decodeSteps += sequence.decodeSteps();
      endOfGeneration += sequence.hitEndOfGeneration() ? 1 : 0;
      prefillNanos += sequence.prefillNanos();
      decodeNanos += sequence.decodeNanos();
    }
    return new ArmMeasurement(
        arm,
        accelerated,
        sequences.size(),
        promptTokens,
        generatedTokens,
        decodeSteps,
        endOfGeneration,
        prefillNanos,
        decodeNanos,
        wallClockNanos,
        prefillNanos <= 0 ? 0.0 : promptTokens * 1_000_000_000.0 / prefillNanos,
        decodeNanos <= 0 ? 0.0 : decodeSteps * 1_000_000_000.0 / decodeNanos,
        ProcessMemory.snapshot(ProcessHandle.current().pid()).residentBytes(),
        Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory(),
        peakDeviceBytes,
        deviceTotalMemoryBytes);
  }

  /**
   * Decode speedup of the accelerated arm over the control, when both were measured.
   *
   * <p>Empty rather than a sentinel when only one arm ran. G4 is a ratio; with one arm there is no
   * ratio, and a report that printed 0.0 would be read as a measured failure rather than as a
   * measurement that was not taken.
   */
  static OptionalDouble speedup(ArmMeasurement accelerated, ArmMeasurement control) {
    if (accelerated == null || control == null || control.decodeTokensPerSecond() <= 0.0) {
      return OptionalDouble.empty();
    }
    return OptionalDouble.of(accelerated.decodeTokensPerSecond() / control.decodeTokensPerSecond());
  }
}
