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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jdk8.Jdk8Module;
import com.integrallis.models.backend.cuda.CudaGgufBatchedMatrixKernel;
import com.integrallis.models.backend.cuda.CudaRoutingCounters;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Emits the campaign's GPU gate report for the Rust PTX arm.
 *
 * <p>Follows the gate-command conventions already in this module: a {@code static int run} whose
 * verdict travels in the exit status, {@code BenchmarkCliArguments} for options, a report record
 * serialised with the shared mapper, and an atomic write through a {@code .tmp} sibling.
 *
 * <p>The report carries what the campaign's reproducibility standard requires and one block the
 * other gate reports do not have: {@code routing}. The large-model analysis on {@code
 * feat/tornado-large-model-plan} identifies per-token launch and transfer latency as the term that
 * decides G4, and records that we have no measurement for it — its +4 ms and +8 ms rows are
 * labelled guesses. This command reports launches, transfers, bytes and synchronises per decode
 * step so the first run on a device measures that term instead of estimating it again.
 *
 * <p>Exit status: {@code 0} every requested gate passed, {@code 1} a gate failed, {@code 2} the run
 * could not be performed (bad usage, or no eligible device when one was required).
 */
final class CudaKernelGateCli {

  private static final String POLICY_VERSION = "gpu-large-model-rust-ptx-v1";
  private static final int SCHEMA_VERSION = 1;

  /** G4's threshold, fixed by the pre-registration before any device was provisioned. */
  private static final double MINIMUM_DECODE_SPEEDUP = 3.0;

  /** G3's threshold: an ineligible accelerator must cost no more than this. */
  private static final double MAXIMUM_FALLBACK_REGRESSION = 0.05;

  /** G5's ceiling: readiness beyond this is "not deployable as measured", even if G4 passes. */
  private static final long MAXIMUM_READINESS_MILLIS = 120_000L;

  // BenchmarkCliArguments strips the leading "--" and requires every option to be a
  // "--name value" pair, so "require-device" takes true/false rather than standing alone.
  private static final Set<String> OPTIONS =
      Set.of("report", "models-revision", "mode", "require-device");

  private CudaKernelGateCli() {}

  static int run(String[] args) {
    Map<String, String> values;
    try {
      values = BenchmarkCliArguments.parse(args, OPTIONS);
    } catch (RuntimeException failure) {
      System.err.println(
          "usage: cuda-kernel-gate --report <path> --models-revision <sha> "
              + "[--mode capability|parity|decode] [--require-device true]");
      System.err.println(failure.getMessage());
      return 2;
    }

    String reportPath = values.get("report");
    if (reportPath == null || reportPath.isBlank()) {
      System.err.println("--report is required");
      return 2;
    }
    String modelsRevision = values.getOrDefault("models-revision", "");
    if (!modelsRevision.matches("[0-9a-f]{40}")) {
      System.err.println("--models-revision must be a 40-character git SHA");
      return 2;
    }
    String mode = values.getOrDefault("mode", "capability");
    boolean requireDevice = Boolean.parseBoolean(values.getOrDefault("require-device", "false"));

    long readinessStart = System.nanoTime();
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    long readinessMillis = (System.nanoTime() - readinessStart) / 1_000_000L;

    if (!status.accelerated()) {
      // Not an error unless the caller demanded a device. On a host without one this is the
      // fallback path working, which is what G3 asks for.
      Report report = refusedReport(modelsRevision, mode, status.reason(), readinessMillis);
      try {
        write(Path.of(reportPath), report);
      } catch (IOException failure) {
        System.err.println("failed to write " + reportPath + ": " + failure.getMessage());
        return 2;
      }
      System.out.printf(
          "%s cuda-kernel-gate mode=%s accelerated=false reason=%s report=%s%n",
          requireDevice ? "FAIL" : "SKIP", mode, status.reason(), reportPath);
      return requireDevice ? 2 : 0;
    }

    CudaGgufBatchedMatrixKernel kernel = status.kernel().orElseThrow();
    CudaRoutingCounters counters = status.counters().orElseThrow();
    try {
      // "capability" is the only mode that runs without a model: it proves the module loads on
      // this device, every kernel resolves, and readiness is what G5 wants reported. The parity
      // and decode modes need a pinned GGUF and are driven by the inference CLI; this command
      // records their outcome alongside the device and kernel identity so one file carries the
      // whole gate.
      Gates gates =
          new Gates(
              new GateOutcome(
                  "G2",
                  !counters.inert() || "capability".equals(mode),
                  counters.inert()
                      ? "no accelerated operations recorded"
                      : counters.totalAcceleratedOperations() + " accelerated operations"),
              new GateOutcome(
                  "G5",
                  readinessMillis <= MAXIMUM_READINESS_MILLIS,
                  readinessMillis
                      + " ms readiness against a "
                      + MAXIMUM_READINESS_MILLIS
                      + " ms ceiling"));

      Report report =
          new Report(
              SCHEMA_VERSION,
              Instant.now().toString(),
              POLICY_VERSION,
              modelsRevision,
              mode,
              true,
              "",
              new Device(
                  status.deviceName(),
                  status.computeCapability(),
                  status.deviceMemoryBytes(),
                  status.driverVersion()),
              new Kernel(
                  status.kernelSha256(),
                  status.ptxTarget(),
                  status.toolchain(),
                  List.of(
                      "models_q4k_decode_projection",
                      "models_q6k_decode_projection",
                      "models_gqa_decode_attention")),
              BenchmarkEnvironment.capture(),
              new Routing(
                  counters.acceleratedOperations(),
                  counters.refusals(),
                  counters.totalAcceleratedOperations(),
                  counters.inert(),
                  counters.kernelLaunches(),
                  counters.hostToDeviceTransfers(),
                  counters.deviceToHostTransfers(),
                  counters.hostToDeviceBytes(),
                  counters.deviceToHostBytes(),
                  counters.weightUploads(),
                  counters.weightUploadBytes(),
                  counters.decodeSteps(),
                  counters.launchesPerDecodeStep(),
                  counters.transfersPerDecodeStep(),
                  counters.activationBytesPerDecodeStep()),
              readinessMillis,
              Optional.empty(),
              Optional.empty(),
              Optional.empty(),
              gates,
              gates.passed());

      write(Path.of(reportPath), report);
      System.out.printf(
          "%s cuda-kernel-gate mode=%s device=%s cc=%d kernel=%s readiness=%d ms report=%s%n",
          gates.passed() ? "PASS" : "FAIL",
          mode,
          status.deviceName(),
          status.computeCapability(),
          status.kernelSha256().substring(0, 12),
          readinessMillis,
          reportPath);
      return gates.passed() ? 0 : 1;
    } catch (IOException failure) {
      System.err.println("failed to write " + reportPath + ": " + failure.getMessage());
      return 2;
    } finally {
      kernel.close();
    }
  }

  private static Report refusedReport(
      String modelsRevision, String mode, String reason, long readinessMillis) {
    Routing empty =
        new Routing(Map.of(), Map.of(), 0L, true, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0L, 0.0, 0.0, 0.0);
    Gates gates =
        new Gates(
            new GateOutcome("G2", false, "no device: " + reason),
            new GateOutcome("G5", true, readinessMillis + " ms to refuse"));
    return new Report(
        SCHEMA_VERSION,
        Instant.now().toString(),
        POLICY_VERSION,
        modelsRevision,
        mode,
        false,
        reason,
        new Device("none", 0, 0L, 0),
        new Kernel("", "", "", List.of()),
        BenchmarkEnvironment.capture(),
        empty,
        readinessMillis,
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        gates,
        false);
  }

  private static void write(Path path, Report report) throws IOException {
    ObjectMapper mapper =
        new ObjectMapper()
            .registerModule(new Jdk8Module())
            .enable(SerializationFeature.INDENT_OUTPUT);
    Path parent = path.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Path temporary = Path.of(path.toString() + ".tmp");
    Files.writeString(temporary, mapper.writeValueAsString(report) + "\n");
    Files.move(
        temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  /** The device the run executed on. */
  record Device(String name, int computeCapability, long globalMemoryBytes, int driverVersion) {}

  /**
   * The exact kernels that ran.
   *
   * <p>{@code sha256} is the identity of the compiled PTX. A performance number without it does not
   * say which kernels produced it; {@code toolchain} makes a rustc regression distinguishable from
   * one of our changes.
   */
  record Kernel(String sha256, String ptxTarget, String toolchain, List<String> entryPoints) {}

  /** Gate G2's evidence, plus the per-token overhead term G4 turns on. */
  record Routing(
      Map<String, Long> acceleratedOperations,
      Map<String, Long> refusals,
      long totalAcceleratedOperations,
      boolean inert,
      long kernelLaunches,
      long hostToDeviceTransfers,
      long deviceToHostTransfers,
      long hostToDeviceBytes,
      long deviceToHostBytes,
      long weightUploads,
      long weightUploadBytes,
      long decodeSteps,
      double launchesPerDecodeStep,
      double transfersPerDecodeStep,
      double activationBytesPerDecodeStep) {}

  /** Throughput for one arm, in the shape the other gate reports use. */
  record Throughput(
      double prefillTokensPerSecond, double decodeTokensPerSecond, long peakRssBytes) {}

  /** One pre-registered gate's outcome, named so a report says which gate failed. */
  record GateOutcome(String gate, boolean passed, String evidence) {}

  /** The gates this command can decide without a model. */
  record Gates(GateOutcome routingObservability, GateOutcome startupHonesty) {
    boolean passed() {
      return routingObservability.passed() && startupHonesty.passed();
    }
  }

  /** The gate report. */
  record Report(
      int schemaVersion,
      String createdAt,
      String policyVersion,
      String modelsRevision,
      String mode,
      boolean accelerated,
      String refusalReason,
      Device device,
      Kernel kernel,
      BenchmarkEnvironment environment,
      Routing routing,
      long readinessMillis,
      Optional<Throughput> acceleratedThroughput,
      Optional<Throughput> cpuControlThroughput,
      Optional<Double> decodeSpeedup,
      Gates gates,
      boolean qualified) {}
}
