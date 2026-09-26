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

import com.integrallis.models.backend.cuda.CudaRoutingCounters;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Marks token boundaries on the routing counters and keeps the per-token delta.
 *
 * <p>Three jobs, all of which only the harness can do.
 *
 * <p><strong>Marking steps.</strong> The kernel is handed one projection at a time and cannot see
 * where one generated token ends and the next begins, so it counts projections and this class
 * counts steps. Without that split, {@code launchesPerDecodeStep} would report launches per
 * <em>projection</em> — a constant, dressed as the overhead term the pre-registration says decides
 * G4.
 *
 * <p><strong>Excluding what is not a decode step.</strong> The counters are cumulative from process
 * start, so dividing their totals by the step count would charge warmup, weight uploads and every
 * prefill launch to the per-token figure. The per-step terms here are summed from the deltas of
 * measured decode steps only: not the warmup sequence, and not the prefill-produced first token of
 * each prompt, which costs no forward pass.
 *
 * <p><strong>Attribution.</strong> When G1 fails, "the first divergent token is number 37" is not
 * actionable; "at token 37 the accelerated arm ran one F32 decode attention and no projection" is,
 * because the pre-registration says a divergence confined to attention points at {@code expf} and a
 * divergence in the projections does not.
 */
final class CudaRoutingRecorder implements GreedyDecode.StepListener {

  private final CudaRoutingCounters counters;
  private final List<List<StepRouting>> byPrompt = new ArrayList<>();
  private Map<String, Long> previousAccelerated = Map.of();
  private Map<String, Long> previousRefusals = Map.of();
  private long previousLaunches;
  private long previousTransfers;
  private long previousBytes;

  private long measuredSteps;
  private long measuredLaunches;
  private long measuredTransfers;
  private long measuredActivationBytes;

  CudaRoutingRecorder(CudaRoutingCounters counters) {
    this.counters = counters;
    if (counters != null) {
      rebase();
    }
  }

  /** A recorder for a run with no device, which records nothing and marks no steps. */
  static CudaRoutingRecorder absent() {
    return new CudaRoutingRecorder(null);
  }

  @Override
  public void generated(int promptIndex, int tokenIndex, int tokenId, boolean decodeStep) {
    if (counters == null) {
      return;
    }
    if (promptIndex < 0) {
      // The warmup sequence. It runs, and its launches and uploads stay in the cumulative
      // counters, but it is not a measured step and must not move the baseline forward silently.
      rebase();
      return;
    }
    if (decodeStep) {
      counters.decodeStep();
    }

    Map<String, Long> accelerated = counters.acceleratedOperations();
    Map<String, Long> refusals = counters.refusals();
    long launches = counters.kernelLaunches();
    long transfers = counters.hostToDeviceTransfers() + counters.deviceToHostTransfers();
    long bytes = counters.hostToDeviceBytes() + counters.deviceToHostBytes();

    StepRouting step =
        new StepRouting(
            delta(previousAccelerated, accelerated),
            delta(previousRefusals, refusals),
            launches - previousLaunches,
            transfers - previousTransfers,
            bytes - previousBytes);

    if (decodeStep) {
      measuredSteps++;
      measuredLaunches += step.kernelLaunches();
      measuredTransfers += step.hostTransfers();
      measuredActivationBytes += step.activationBytes();
    }

    previousAccelerated = accelerated;
    previousRefusals = refusals;
    previousLaunches = launches;
    previousTransfers = transfers;
    previousBytes = bytes;

    while (byPrompt.size() <= promptIndex) {
      byPrompt.add(new ArrayList<>());
    }
    byPrompt.get(promptIndex).add(step);
  }

  /** The routing delta attributable to one generated token, if it was recorded. */
  Optional<StepRouting> at(int promptIndex, int tokenIndex) {
    if (promptIndex < 0 || promptIndex >= byPrompt.size()) {
      return Optional.empty();
    }
    List<StepRouting> steps = byPrompt.get(promptIndex);
    if (tokenIndex < 0 || tokenIndex >= steps.size()) {
      return Optional.empty();
    }
    return Optional.of(steps.get(tokenIndex));
  }

  /** The overhead term, summed over measured decode steps only. */
  Measured measured() {
    return new Measured(
        measuredSteps, measuredLaunches, measuredTransfers, measuredActivationBytes);
  }

  private void rebase() {
    previousAccelerated = counters.acceleratedOperations();
    previousRefusals = counters.refusals();
    previousLaunches = counters.kernelLaunches();
    previousTransfers = counters.hostToDeviceTransfers() + counters.deviceToHostTransfers();
    previousBytes = counters.hostToDeviceBytes() + counters.deviceToHostBytes();
  }

  private static Map<String, Long> delta(Map<String, Long> before, Map<String, Long> after) {
    Map<String, Long> result = new LinkedHashMap<>();
    for (Map.Entry<String, Long> entry : after.entrySet()) {
      long difference = entry.getValue() - before.getOrDefault(entry.getKey(), 0L);
      if (difference != 0) {
        result.put(entry.getKey(), difference);
      }
    }
    return Map.copyOf(result);
  }

  /**
   * The per-token overhead term the large-model analysis could only guess at.
   *
   * <p>Zero steps means the term was not measured, not that it is zero. The report says which.
   */
  record Measured(long steps, long kernelLaunches, long hostTransfers, long activationBytes) {

    boolean measured() {
      return steps > 0;
    }

    double launchesPerDecodeStep() {
      return steps == 0 ? 0.0 : (double) kernelLaunches / steps;
    }

    double transfersPerDecodeStep() {
      return steps == 0 ? 0.0 : (double) hostTransfers / steps;
    }

    double activationBytesPerDecodeStep() {
      return steps == 0 ? 0.0 : (double) activationBytes / steps;
    }
  }

  /**
   * What the accelerator did while producing one token.
   *
   * <p>Keys are the counters' own {@code FORMAT/STAGE} and {@code STAGE/reason} spellings, so a
   * report is readable against {@link CudaRoutingCounters} without a translation table.
   */
  record StepRouting(
      Map<String, Long> acceleratedOperations,
      Map<String, Long> refusals,
      long kernelLaunches,
      long hostTransfers,
      long activationBytes) {

    StepRouting {
      acceleratedOperations = Map.copyOf(acceleratedOperations);
      refusals = Map.copyOf(refusals);
    }

    /**
     * Whether decode attention ran on the device for this token.
     *
     * <p>The question G1's failure branch turns on. Attention is the one stage with a stated
     * tolerance ({@code UPSTREAM.md} CU-005, 2.0e-5 relative L2, because of {@code expf}); the
     * projections are bit-exact by construction. A divergence at a token where attention routed and
     * a divergence at a token where it did not have different causes and different fixes.
     */
    boolean attentionRouted() {
      return acceleratedOperations.keySet().stream()
          .anyMatch(key -> key.endsWith("/DECODE_ATTENTION"));
    }

    /** Whether any projection ran on the device for this token. */
    boolean projectionRouted() {
      return acceleratedOperations.keySet().stream().anyMatch(key -> key.endsWith("_PROJECTION"));
    }
  }
}
