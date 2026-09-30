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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Per-format, per-stage record of what actually executed on the device, and what did not.
 *
 * <p>This exists for gate G2: "every run records which work ran on device, per format and per
 * stage. A run that reports 0 accelerated operations is an inert accelerator and fails, whatever
 * its speed." Two design choices follow from that wording and are worth stating.
 *
 * <p><strong>Refusals are counted, not just dispatches.</strong> Knowing that zero Q4_K projections
 * ran is only half an answer; the useful half is why. Every rejection path calls {@link #refused}
 * with a reason, so a run that accelerates nothing says whether the format was unsupported, the
 * shape ineligible, or the device absent — the difference between "this stage does not matter" and
 * "this stage never ran".
 *
 * <p><strong>Launches and host transfers are counted separately from operations.</strong> The
 * large-model analysis on {@code feat/tornado-large-model-plan} identifies per-token launch and
 * transfer latency as the term that decides G4, and the term nobody has measured: at 70% of an
 * L40S's bandwidth, zero overhead implies about 233 tok/s while 4 ms/token implies about 121 and 8
 * ms about 81. Those rows are labelled guesses in that document. Counting launches, transfers and
 * transferred bytes here turns the first GPU run into a measurement of exactly that term rather
 * than another estimate of it.
 *
 * <p>Instances are thread-safe. Counters are monotonic within a run and are snapshotted into the
 * report JSON at the end.
 */
public final class CudaRoutingCounters {

  private final Map<String, AtomicLong> accelerated = new ConcurrentHashMap<>();
  private final Map<String, AtomicLong> refusedByReason = new ConcurrentHashMap<>();
  private final AtomicLong kernelLaunches = new AtomicLong();
  private final AtomicLong hostToDeviceTransfers = new AtomicLong();
  private final AtomicLong deviceToHostTransfers = new AtomicLong();
  private final AtomicLong hostToDeviceBytes = new AtomicLong();
  private final AtomicLong deviceToHostBytes = new AtomicLong();
  private final AtomicLong weightUploads = new AtomicLong();
  private final AtomicLong weightUploadBytes = new AtomicLong();
  private final AtomicLong decodeSteps = new AtomicLong();
  private final AtomicLong decodeProjections = new AtomicLong();
  private final AtomicLong deviceBytesInUse = new AtomicLong();
  private final AtomicLong peakDeviceBytes = new AtomicLong();

  /** Records one operation that executed on the device. */
  public void accelerated(GgufTensorType type, CudaStage stage, int operations) {
    Objects.requireNonNull(type, "type");
    Objects.requireNonNull(stage, "stage");
    accelerated.computeIfAbsent(key(type, stage), unused -> new AtomicLong()).addAndGet(operations);
  }

  /** Records one operation that fell back to the Vector API, and why. */
  public void refused(CudaStage stage, String reason) {
    Objects.requireNonNull(stage, "stage");
    Objects.requireNonNull(reason, "reason");
    refusedByReason
        .computeIfAbsent(stage.name() + "/" + reason, unused -> new AtomicLong())
        .incrementAndGet();
  }

  /** Records one kernel launch. */
  public void launched() {
    kernelLaunches.incrementAndGet();
  }

  /** Records one host-to-device copy of {@code bytes}. */
  public void copiedToDevice(long bytes) {
    hostToDeviceTransfers.incrementAndGet();
    hostToDeviceBytes.addAndGet(bytes);
  }

  /** Records one device-to-host copy of {@code bytes}. */
  public void copiedToHost(long bytes) {
    deviceToHostTransfers.incrementAndGet();
    deviceToHostBytes.addAndGet(bytes);
  }

  /**
   * Records one weight tensor uploaded to the device.
   *
   * <p>Separate from {@link #copiedToDevice} because the two answer different questions: weight
   * uploads should happen once per tensor for the life of the process, and a count that grows with
   * tokens means the design regressed into per-token streaming, which the memory analysis shows is
   * slower than the CPU path.
   */
  public void uploadedWeights(long bytes) {
    weightUploads.incrementAndGet();
    weightUploadBytes.addAndGet(bytes);
  }

  /**
   * Marks one generated token, so per-token averages are derivable.
   *
   * <p>Called by the measurement harness, never by the kernel. The kernel is handed one projection
   * at a time and cannot tell which projections belong to the same generated token; if it marked
   * steps itself, {@link #launchesPerDecodeStep()} would report launches per <em>projection</em>,
   * which is 1 by construction for every model. That would look like a measurement of the term the
   * pre-registration says decides G4 while being a restatement of the dispatch loop. Per-projection
   * dispatch is counted by {@link #decodeProjection()} instead.
   */
  public void decodeStep() {
    decodeSteps.incrementAndGet();
  }

  /** Records one single-token projection dispatched to the device. */
  public void decodeProjection() {
    decodeProjections.incrementAndGet();
  }

  /** Records {@code bytes} of device memory allocated by this kernel. */
  public void deviceAllocated(long bytes) {
    long inUse = deviceBytesInUse.addAndGet(bytes);
    peakDeviceBytes.accumulateAndGet(inUse, Math::max);
  }

  /** Records {@code bytes} of device memory released by this kernel. */
  public void deviceFreed(long bytes) {
    deviceBytesInUse.addAndGet(-bytes);
  }

  /** Accelerated operation counts keyed {@code FORMAT/STAGE}. */
  public Map<String, Long> acceleratedOperations() {
    return snapshot(accelerated);
  }

  /** Fallback counts keyed {@code STAGE/reason}. */
  public Map<String, Long> refusals() {
    return snapshot(refusedByReason);
  }

  /** Total operations that ran on device across every format and stage. */
  public long totalAcceleratedOperations() {
    return accelerated.values().stream().mapToLong(AtomicLong::get).sum();
  }

  /**
   * Whether this run accelerated anything at all.
   *
   * <p>Gate G2 fails a run for which this is false, whatever its tokens/s.
   */
  public boolean inert() {
    return totalAcceleratedOperations() == 0;
  }

  /** Kernel launches issued. */
  public long kernelLaunches() {
    return kernelLaunches.get();
  }

  /** Host-to-device copies issued, excluding one-time weight uploads. */
  public long hostToDeviceTransfers() {
    return hostToDeviceTransfers.get();
  }

  /** Device-to-host copies issued. */
  public long deviceToHostTransfers() {
    return deviceToHostTransfers.get();
  }

  /** Bytes copied host to device, excluding one-time weight uploads. */
  public long hostToDeviceBytes() {
    return hostToDeviceBytes.get();
  }

  /** Bytes copied device to host. */
  public long deviceToHostBytes() {
    return deviceToHostBytes.get();
  }

  /** Weight tensors uploaded. */
  public long weightUploads() {
    return weightUploads.get();
  }

  /** Bytes of weights uploaded. */
  public long weightUploadBytes() {
    return weightUploadBytes.get();
  }

  /** Generated tokens marked by the measurement harness. */
  public long decodeSteps() {
    return decodeSteps.get();
  }

  /** Single-token projections dispatched to the device. */
  public long decodeProjections() {
    return decodeProjections.get();
  }

  /**
   * Whether any caller marked a token boundary.
   *
   * <p>False means the per-step terms below are unmeasured, not zero. A report must say which.
   */
  public boolean decodeStepsMarked() {
    return decodeSteps.get() > 0;
  }

  /** Device memory this kernel currently holds. */
  public long deviceBytesInUse() {
    return deviceBytesInUse.get();
  }

  /**
   * High-water mark of device memory held by this kernel.
   *
   * <p>Kernel-owned allocations only: weights, the staging scratch, and the per-call attention
   * buffers. The CUDA context, the loaded module and the driver's own reservations are not
   * included, so this is a lower bound on the process's device footprint rather than a reading of
   * it.
   */
  public long peakDeviceBytes() {
    return peakDeviceBytes.get();
  }

  /**
   * Kernel launches per decode step, or zero when no step completed.
   *
   * <p>With the per-launch latency a GPU host can measure, this is the overhead term the
   * large-model analysis could only guess at.
   */
  public double launchesPerDecodeStep() {
    long steps = decodeSteps.get();
    return steps == 0 ? 0.0 : (double) kernelLaunches.get() / steps;
  }

  /** Host round trips per decode step, or zero when no step completed. */
  public double transfersPerDecodeStep() {
    long steps = decodeSteps.get();
    return steps == 0
        ? 0.0
        : (double) (hostToDeviceTransfers.get() + deviceToHostTransfers.get()) / steps;
  }

  /** Activation bytes crossing the link per decode step, excluding weight uploads. */
  public double activationBytesPerDecodeStep() {
    long steps = decodeSteps.get();
    return steps == 0 ? 0.0 : (double) (hostToDeviceBytes.get() + deviceToHostBytes.get()) / steps;
  }

  private static String key(GgufTensorType type, CudaStage stage) {
    return type.name() + "/" + stage.name();
  }

  private static Map<String, Long> snapshot(Map<String, AtomicLong> counters) {
    Map<String, Long> result = new LinkedHashMap<>();
    counters.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .forEach(entry -> result.put(entry.getKey(), entry.getValue().get()));
    return Map.copyOf(result);
  }
}
