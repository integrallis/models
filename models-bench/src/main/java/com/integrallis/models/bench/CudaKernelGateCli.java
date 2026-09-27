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
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.cuda.CudaGgufBatchedMatrixKernel;
import com.integrallis.models.backend.cuda.CudaRoutingCounters;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.Set;

/**
 * Emits the campaign's GPU gate reports for the Rust PTX arm.
 *
 * <p>Three modes, one report shape, one exit-status convention.
 *
 * <ul>
 *   <li>{@code capability} — the module loads on this device, every kernel resolves, readiness is
 *       under G5's ceiling. Needs no model. This is the mode that has run on an L40S.
 *   <li>{@code parity} — G1. One GGUF, loaded twice: once with the CUDA kernels injected and once
 *       with {@link GgufBatchedMatrixKernel#none()}. Both arms greedily decode the same prompts and
 *       the full token id sequences must be identical. Exit 0 only if every sequence matches.
 *   <li>{@code decode} — G4. The same two arms, measured: prefill and decode tokens/s, wall clock,
 *       the routing block, peak device and host memory, and the configuration and environment the
 *       reproducibility standard requires.
 * </ul>
 *
 * <p><strong>Off-device, parity mode still runs.</strong> With no eligible device both arms are the
 * Vector API path, which makes the run a self-test of the harness rather than evidence for G1. The
 * report says so — {@code parity.selfTest} is true, {@code accelerated} is false, and {@code
 * qualified} is false whatever the sequences did — and the console line reads {@code SELF-TEST}
 * rather than {@code PASS}. A CPU-vs-CPU pass is not a gate result and must never be filed as one.
 * {@code --require-device true} refuses to run at all without a device, which is what the campaign
 * uses on the pod.
 *
 * <p>The report carries one block the other gate reports do not have: {@code routing}. The
 * large-model analysis identifies per-token launch and transfer latency as the term that decides
 * G4, and records that we have no measurement for it — its +4 ms and +8 ms rows are labelled
 * guesses. {@code launchesPerDecodeStep}, {@code transfersPerDecodeStep} and {@code
 * activationBytesPerDecodeStep} are reported whether or not G4 passes, because a failing run that
 * pins the overhead term is more useful than a passing one that does not.
 *
 * <p>Exit status: {@code 0} every requested gate passed, {@code 1} a gate failed, {@code 2} the run
 * could not be performed (bad usage, or no eligible device when one was required).
 */
final class CudaKernelGateCli {

  private static final String POLICY_VERSION = "gpu-large-model-rust-ptx-v2";

  /**
   * Bumped from 1 when parity and decode modes were added.
   *
   * <p>A version-1 report is a capability report from before those modes existed; it has no {@code
   * parity}, {@code decode}, {@code memory} or {@code configuration} block and its {@code gates}
   * object has two members rather than four.
   */
  private static final int SCHEMA_VERSION = 2;

  /** G4's threshold, fixed by the pre-registration before any device was provisioned. */
  private static final double MINIMUM_DECODE_SPEEDUP = 3.0;

  /** G5's ceiling: readiness beyond this is "not deployable as measured", even if G4 passes. */
  private static final long MAXIMUM_READINESS_MILLIS = 120_000L;

  private static final String CAPABILITY = "capability";
  private static final String PARITY = "parity";
  private static final String DECODE = "decode";
  private static final Set<String> MODES = Set.of(CAPABILITY, PARITY, DECODE);

  private static final String ARM_BOTH = "both";
  private static final String ARM_ACCELERATED = "accelerated";
  private static final String ARM_CONTROL = "control";
  private static final Set<String> ARMS = Set.of(ARM_BOTH, ARM_ACCELERATED, ARM_CONTROL);

  // BenchmarkCliArguments strips the leading "--" and requires every option to be a
  // "--name value" pair, so "require-device" takes true/false rather than standing alone.
  private static final Set<String> OPTIONS =
      Set.of(
          "report",
          "models-revision",
          "mode",
          "require-device",
          "model",
          "prompts",
          "max-tokens",
          "warmup-tokens",
          "context",
          "prompt-limit",
          "arm");

  private static final String USAGE =
      """
      usage: cuda-kernel-gate --report <path> --models-revision <sha> --mode capability|parity|decode
        capability          no model needed
        parity   (G1)       --model <gguf> --prompts <file> [--max-tokens 64] [--context 4096]
        decode   (G4)       --model <gguf> --prompts <file> [--max-tokens 64] [--warmup-tokens 16]
                            [--context 4096] [--arm both|accelerated|control]
        all modes           [--require-device true] [--prompt-limit N]\
      """;

  private CudaKernelGateCli() {}

  /** Opens one arm's backend. Separated so the gate logic is testable with no model and no GPU. */
  interface ArmLoader {

    /**
     * @param accelerated whether to inject the CUDA kernel; false is the Vector API control
     */
    InferenceBackend open(boolean accelerated);
  }

  /** Device and kernel identity, plus the readiness G5 measures. */
  record Accelerator(
      boolean accelerated,
      String reason,
      long readinessMillis,
      Device device,
      Kernel kernel,
      Optional<CudaRoutingCounters> counters) {

    static Accelerator from(CudaGgufBatchedMatrixKernel.Status status, long readinessMillis) {
      if (!status.accelerated()) {
        return new Accelerator(
            false,
            status.reason(),
            readinessMillis,
            new Device("none", 0, 0L, 0),
            new Kernel("", "", "", List.of()),
            Optional.empty());
      }
      return new Accelerator(
          true,
          "",
          readinessMillis,
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
          status.counters());
    }

    /** An accelerator that is deliberately absent, for the off-device self-test. */
    static Accelerator absent(String reason, long readinessMillis) {
      return new Accelerator(
          false,
          reason,
          readinessMillis,
          new Device("none", 0, 0L, 0),
          new Kernel("", "", "", List.of()),
          Optional.empty());
    }
  }

  static int run(String[] args) {
    Configuration configuration;
    try {
      configuration = parse(args);
    } catch (RuntimeException | IOException failure) {
      System.err.println(USAGE);
      System.err.println(failure.getMessage());
      return 2;
    }

    long readinessStart = System.nanoTime();
    CudaGgufBatchedMatrixKernel.Status status = CudaGgufBatchedMatrixKernel.open();
    long readinessMillis = (System.nanoTime() - readinessStart) / 1_000_000L;
    Accelerator accelerator = Accelerator.from(status, readinessMillis);

    Path model = configuration.modelPath().orElse(null);
    ArmLoader loader =
        accelerated -> {
          Path path = java.util.Objects.requireNonNull(model, "model");
          return accelerated && status.accelerated()
              ? PureJavaBackend.load(path, status.kernel().orElseThrow())
              : PureJavaBackend.load(path, GgufBatchedMatrixKernel.none());
        };

    try {
      return run(configuration, accelerator, loader);
    } finally {
      status.kernel().ifPresent(CudaGgufBatchedMatrixKernel::close);
    }
  }

  /** The gate, with the accelerator and the backends supplied. Exercised directly by the tests. */
  static int run(Configuration configuration, Accelerator accelerator, ArmLoader loader) {
    if (!accelerator.accelerated() && configuration.requireDevice()) {
      Report report = report(configuration, accelerator, null, null, null, null);
      if (!writeQuietly(configuration.report(), report)) {
        return 2;
      }
      System.out.printf(
          "FAIL cuda-kernel-gate mode=%s accelerated=false reason=%s report=%s%n",
          configuration.mode(), accelerator.reason(), configuration.report());
      return 2;
    }

    try {
      return switch (configuration.mode()) {
        case CAPABILITY -> capability(configuration, accelerator);
        case PARITY -> parity(configuration, accelerator, loader);
        case DECODE -> decode(configuration, accelerator, loader);
        default -> throw new IllegalArgumentException("unknown mode: " + configuration.mode());
      };
    } catch (RuntimeException failure) {
      System.err.println("cuda-kernel-gate " + configuration.mode() + " failed: " + failure);
      return 2;
    }
  }

  // --- modes -------------------------------------------------------------------------------

  private static int capability(Configuration configuration, Accelerator accelerator) {
    Report report = report(configuration, accelerator, null, null, null, null);
    if (!writeQuietly(configuration.report(), report)) {
      return 2;
    }
    if (!accelerator.accelerated()) {
      System.out.printf(
          "SKIP cuda-kernel-gate mode=capability accelerated=false reason=%s report=%s%n",
          accelerator.reason(), configuration.report());
      return 0;
    }
    System.out.printf(
        "%s cuda-kernel-gate mode=capability device=%s cc=%d kernel=%s readiness=%d ms report=%s%n",
        report.qualified() ? "PASS" : "FAIL",
        accelerator.device().name(),
        accelerator.device().computeCapability(),
        abbreviate(accelerator.kernel().sha256()),
        accelerator.readinessMillis(),
        configuration.report());
    return report.qualified() ? 0 : 1;
  }

  private static int parity(
      Configuration configuration, Accelerator accelerator, ArmLoader loader) {
    List<String> prompts = configuration.prompts();
    CudaRoutingRecorder recorder =
        accelerator.counters().map(CudaRoutingRecorder::new).orElseGet(CudaRoutingRecorder::absent);

    // The arms run one at a time. Both hold the same GGUF, and a 26B Q4_K_M is 16.8 GB; loading
    // them together would double the host residency for no gain, since the comparison is over
    // recorded sequences rather than interleaved steps.
    List<GreedyDecode.Sequence> acceleratedSequences =
        runArm(loader, true, configuration, prompts, recorder);
    List<GreedyDecode.Sequence> controlSequences =
        runArm(loader, false, configuration, prompts, GreedyDecode.StepListener.none());

    CudaParityRun.Result result =
        CudaParityRun.compare(acceleratedSequences, controlSequences, recorder);
    boolean selfTest = !accelerator.accelerated();
    Parity parity =
        new Parity(
            selfTest,
            result.promptCount(),
            result.tokensPerPrompt(),
            result.comparedTokens(),
            result.endOfGenerationSequences(),
            result.identical(),
            result.firstDivergence(),
            acceleratedSequences.stream().map(GreedyDecode.Sequence::promptDigest).toList());

    Report report = report(configuration, accelerator, parity, null, recorder, memory(accelerator));
    if (!writeQuietly(configuration.report(), report)) {
      return 2;
    }

    if (selfTest) {
      System.out.printf(
          "SELF-TEST cuda-kernel-gate mode=parity accelerated=false reason=%s identical=%s "
              + "tokens=%d report=%s%n"
              + "  CPU vs CPU only. This is not G1 evidence; re-run on a device.%n",
          accelerator.reason(),
          result.identical(),
          result.comparedTokens(),
          configuration.report());
      return result.identical() ? 0 : 1;
    }

    if (result.identical()) {
      System.out.printf(
          "PASS cuda-kernel-gate mode=parity device=%s prompts=%d tokens=%d identical report=%s%n",
          accelerator.device().name(),
          result.promptCount(),
          result.comparedTokens(),
          configuration.report());
      return 0;
    }
    CudaParityRun.Divergence divergence = result.firstDivergence().orElseThrow();
    System.out.printf(
        "FAIL cuda-kernel-gate mode=parity device=%s first divergence prompt=%d token=%d "
            + "accelerated=%d control=%d attentionRouted=%s projectionRouted=%s report=%s%n  %s%n",
        accelerator.device().name(),
        divergence.promptIndex(),
        divergence.tokenIndex(),
        divergence.acceleratedTokenId(),
        divergence.controlTokenId(),
        divergence.attentionRouted(),
        divergence.projectionRouted(),
        configuration.report(),
        divergence.reading());
    return 1;
  }

  private static int decode(
      Configuration configuration, Accelerator accelerator, ArmLoader loader) {
    List<String> prompts = configuration.prompts();
    CudaRoutingRecorder recorder =
        accelerator.counters().map(CudaRoutingRecorder::new).orElseGet(CudaRoutingRecorder::absent);

    CudaDecodeRun.ArmMeasurement acceleratedArm = null;
    CudaDecodeRun.ArmMeasurement controlArm = null;
    long wallClockStart = System.nanoTime();

    if (!ARM_CONTROL.equals(configuration.arm())) {
      long start = System.nanoTime();
      List<GreedyDecode.Sequence> sequences =
          runArm(loader, true, configuration, prompts, recorder);
      acceleratedArm =
          CudaDecodeRun.summarise(
              ARM_ACCELERATED,
              accelerator.accelerated(),
              sequences,
              System.nanoTime() - start,
              accelerator.counters().map(CudaRoutingCounters::peakDeviceBytes).orElse(0L),
              accelerator.device().globalMemoryBytes());
    }
    if (!ARM_ACCELERATED.equals(configuration.arm())) {
      long start = System.nanoTime();
      List<GreedyDecode.Sequence> sequences =
          runArm(loader, false, configuration, prompts, GreedyDecode.StepListener.none());
      controlArm =
          CudaDecodeRun.summarise(
              ARM_CONTROL,
              false,
              sequences,
              System.nanoTime() - start,
              0L,
              accelerator.device().globalMemoryBytes());
    }

    OptionalDouble speedup = CudaDecodeRun.speedup(acceleratedArm, controlArm);
    Decode decodeBlock =
        new Decode(
            Optional.ofNullable(acceleratedArm),
            Optional.ofNullable(controlArm),
            speedup,
            MINIMUM_DECODE_SPEEDUP,
            (System.nanoTime() - wallClockStart) / 1_000_000L);

    Report report =
        report(configuration, accelerator, null, decodeBlock, recorder, memory(accelerator));
    if (!writeQuietly(configuration.report(), report)) {
      return 2;
    }

    String verdict =
        accelerator.accelerated() ? (report.qualified() ? "PASS" : "FAIL") : "SELF-TEST";
    System.out.printf(
        "%s cuda-kernel-gate mode=decode device=%s arm=%s accelerated=%.2f tok/s "
            + "control=%.2f tok/s speedup=%s threshold=%.2f report=%s%n",
        verdict,
        accelerator.device().name(),
        configuration.arm(),
        acceleratedArm == null ? 0.0 : acceleratedArm.decodeTokensPerSecond(),
        controlArm == null ? 0.0 : controlArm.decodeTokensPerSecond(),
        speedup.isPresent()
            ? String.format(Locale.ROOT, "%.3f", speedup.getAsDouble())
            : "not-measured",
        MINIMUM_DECODE_SPEEDUP,
        configuration.report());
    // The overhead term is printed whether or not G4 passed. It is the reason the run exists.
    CudaRoutingRecorder.Measured measured = recorder.measured();
    accelerator
        .counters()
        .ifPresent(
            counters ->
                System.out.printf(
                    "  routing: launches/step=%s transfers/step=%s activationBytes/step=%s "
                        + "(%d measured steps, %d decode projections, %d weight uploads)%n",
                    perStep(measured.measured(), measured.launchesPerDecodeStep()),
                    perStep(measured.measured(), measured.transfersPerDecodeStep()),
                    perStep(measured.measured(), measured.activationBytesPerDecodeStep()),
                    measured.steps(),
                    counters.decodeProjections(),
                    counters.weightUploads()));
    if (!accelerator.accelerated()) {
      System.out.println(
          "  CPU only. This is not G4 evidence; the accelerated arm did not run on a device.");
      return 0;
    }
    return report.qualified() ? 0 : 1;
  }

  private static String perStep(boolean marked, double value) {
    return marked ? String.format(Locale.ROOT, "%.3f", value) : "unmeasured";
  }

  private static List<GreedyDecode.Sequence> runArm(
      ArmLoader loader,
      boolean accelerated,
      Configuration configuration,
      List<String> prompts,
      GreedyDecode.StepListener listener) {
    try (InferenceBackend backend = loader.open(accelerated)) {
      GreedyDecode.warmUp(
          backend, prompts.get(0), configuration.warmupTokens(), configuration.contextLength());
      return GreedyDecode.generateAll(
          backend, prompts, configuration.maxTokens(), configuration.contextLength(), listener);
    }
  }

  // --- report ------------------------------------------------------------------------------

  private static Memory memory(Accelerator accelerator) {
    ProcessMemory.Snapshot snapshot = ProcessMemory.snapshot(ProcessHandle.current().pid());
    long peakDevice = accelerator.counters().map(CudaRoutingCounters::peakDeviceBytes).orElse(0L);
    return new Memory(
        snapshot.highWaterBytes(),
        snapshot.residentBytes(),
        Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory(),
        peakDevice,
        accelerator.device().globalMemoryBytes(),
        "peakDeviceBytes covers kernel-owned allocations only (weights, staging scratch, "
            + "per-call attention buffers); the CUDA context and loaded module are not included, "
            + "so it is a lower bound. peakProcessRssBytes is process-wide and therefore not "
            + "separable per arm in a two-arm run: use --arm accelerated and --arm control in "
            + "separate processes for per-arm host peaks.");
  }

  private static Report report(
      Configuration configuration,
      Accelerator accelerator,
      Parity parity,
      Decode decode,
      CudaRoutingRecorder recorder,
      Memory memory) {
    CudaRoutingRecorder.Measured measured =
        recorder == null ? new CudaRoutingRecorder.Measured(0, 0, 0, 0) : recorder.measured();
    Routing routing =
        accelerator
            .counters()
            .map(counters -> routing(counters, measured))
            .orElseGet(() -> emptyRouting(measured));

    boolean inertIsExpected = CAPABILITY.equals(configuration.mode());
    GateOutcome g2 =
        new GateOutcome(
            "G2",
            accelerator.accelerated() && (!routing.inert() || inertIsExpected),
            !accelerator.accelerated()
                ? "no device: " + accelerator.reason()
                : routing.inert()
                    ? "no accelerated operations recorded"
                    : routing.totalAcceleratedOperations() + " accelerated operations");
    GateOutcome g5 =
        new GateOutcome(
            "G5",
            accelerator.readinessMillis() <= MAXIMUM_READINESS_MILLIS,
            accelerator.readinessMillis()
                + " ms readiness against a "
                + MAXIMUM_READINESS_MILLIS
                + " ms ceiling");

    Optional<GateOutcome> g1 =
        parity == null
            ? Optional.empty()
            : Optional.of(
                new GateOutcome(
                    "G1",
                    parity.identical() && !parity.selfTest(),
                    parity.selfTest()
                        ? "self-test only: no device, so both arms were the Vector API path. "
                            + parity.comparedTokens()
                            + " tokens compared, identical="
                            + parity.identical()
                            + ". Not G1 evidence."
                        : parity.identical()
                            ? parity.comparedTokens()
                                + " token ids identical across "
                                + parity.promptCount()
                                + " prompts"
                            : "first divergence at prompt "
                                + parity
                                    .firstDivergence()
                                    .map(CudaParityRun.Divergence::promptIndex)
                                    .orElse(-1)
                                + " token "
                                + parity
                                    .firstDivergence()
                                    .map(CudaParityRun.Divergence::tokenIndex)
                                    .orElse(-1)));
    Optional<GateOutcome> g4 =
        decode == null
            ? Optional.empty()
            : Optional.of(
                new GateOutcome(
                    "G4",
                    accelerator.accelerated()
                        && decode.speedup().isPresent()
                        && decode.speedup().getAsDouble() >= MINIMUM_DECODE_SPEEDUP,
                    !accelerator.accelerated()
                        ? "no device: the accelerated arm was the Vector API path, so there is no "
                            + "speedup to measure"
                        : decode.speedup().isEmpty()
                            ? "only one arm was measured (--arm "
                                + configuration.arm()
                                + "); G4 is a ratio and was not measured"
                            : String.format(
                                Locale.ROOT,
                                "%.3fx decode against a %.2fx threshold",
                                decode.speedup().getAsDouble(),
                                MINIMUM_DECODE_SPEEDUP)));

    Gates gates = new Gates(g2, g5, g1, g4);
    return new Report(
        SCHEMA_VERSION,
        Instant.now().toString(),
        POLICY_VERSION,
        configuration.modelsRevision(),
        configuration.mode(),
        accelerator.accelerated(),
        accelerator.reason(),
        accelerator.device(),
        accelerator.kernel(),
        BenchmarkEnvironment.capture(),
        configuration.describe(),
        routing,
        accelerator.readinessMillis(),
        Optional.ofNullable(parity),
        Optional.ofNullable(decode),
        Optional.ofNullable(memory),
        gates,
        gates.passed());
  }

  private static Routing routing(
      CudaRoutingCounters counters, CudaRoutingRecorder.Measured measured) {
    return new Routing(
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
        counters.decodeProjections(),
        measured.steps(),
        measured.measured(),
        measured.launchesPerDecodeStep(),
        measured.transfersPerDecodeStep(),
        measured.activationBytesPerDecodeStep());
  }

  private static Routing emptyRouting(CudaRoutingRecorder.Measured measured) {
    return new Routing(
        Map.of(),
        Map.of(),
        0L,
        true,
        0L,
        0L,
        0L,
        0L,
        0L,
        0L,
        0L,
        0L,
        0L,
        measured.steps(),
        measured.measured(),
        measured.launchesPerDecodeStep(),
        measured.transfersPerDecodeStep(),
        measured.activationBytesPerDecodeStep());
  }

  private static String abbreviate(String sha256) {
    return sha256 == null || sha256.length() < 12
        ? String.valueOf(sha256)
        : sha256.substring(0, 12);
  }

  private static boolean writeQuietly(Path path, Report report) {
    try {
      write(path, report);
      return true;
    } catch (IOException failure) {
      System.err.println("failed to write " + path + ": " + failure.getMessage());
      return false;
    }
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
    Path temporary = Path.of(path + ".tmp");
    Files.writeString(temporary, mapper.writeValueAsString(report) + "\n");
    Files.move(
        temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  // --- configuration -----------------------------------------------------------------------

  static Configuration parse(String[] args) throws IOException {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);

    String reportPath = values.get("report");
    if (reportPath == null || reportPath.isBlank()) {
      throw new IllegalArgumentException("--report is required");
    }
    String modelsRevision = values.getOrDefault("models-revision", "");
    if (!modelsRevision.matches("[0-9a-f]{40}")) {
      throw new IllegalArgumentException("--models-revision must be a 40-character git SHA");
    }
    String mode = values.getOrDefault("mode", CAPABILITY);
    if (!MODES.contains(mode)) {
      throw new IllegalArgumentException("--mode must be one of " + MODES);
    }
    String arm = values.getOrDefault("arm", ARM_BOTH);
    if (!ARMS.contains(arm)) {
      throw new IllegalArgumentException("--arm must be one of " + ARMS);
    }

    int maxTokens = BenchmarkCliArguments.integer(values, "max-tokens", 64);
    int warmupTokens = BenchmarkCliArguments.integer(values, "warmup-tokens", 16);
    int contextLength = BenchmarkCliArguments.integer(values, "context", 4_096);
    int promptLimit = BenchmarkCliArguments.integer(values, "prompt-limit", 0);
    if (maxTokens < 2) {
      // One token is the prefill argmax and costs no decode step; a parity or decode run of one
      // token would measure the prefill path and report it as decode.
      throw new IllegalArgumentException("--max-tokens must be at least 2");
    }
    if (warmupTokens < 0 || contextLength < 1 || promptLimit < 0) {
      throw new IllegalArgumentException(
          "--warmup-tokens, --context and --prompt-limit must be >= 0");
    }

    Optional<Path> modelPath = Optional.empty();
    List<String> prompts = List.of();
    String promptsPath = "";
    String promptsDigest = "";
    String modelDigest = "";
    long modelBytes = 0L;

    if (!CAPABILITY.equals(mode)) {
      String model = values.get("model");
      String promptsOption = values.get("prompts");
      if (model == null || model.isBlank()) {
        throw new IllegalArgumentException("--model is required in " + mode + " mode");
      }
      if (promptsOption == null || promptsOption.isBlank()) {
        throw new IllegalArgumentException("--prompts is required in " + mode + " mode");
      }
      Path resolved = Path.of(model);
      if (!Files.isRegularFile(resolved)) {
        throw new IllegalArgumentException("model does not exist: " + resolved);
      }
      Path promptsFile = Path.of(promptsOption);
      if (!Files.isRegularFile(promptsFile)) {
        throw new IllegalArgumentException("prompt file does not exist: " + promptsFile);
      }
      List<String> all = GreedyDecode.readPrompts(promptsFile);
      prompts =
          promptLimit > 0 && promptLimit < all.size()
              ? List.copyOf(all.subList(0, promptLimit))
              : all;
      modelPath = Optional.of(resolved);
      promptsPath = promptsFile.toAbsolutePath().toString();
      promptsDigest = Hashing.sha256(promptsFile);
      modelDigest = Hashing.sha256(resolved);
      modelBytes = Files.size(resolved);
    }

    return new Configuration(
        mode,
        Path.of(reportPath),
        modelsRevision,
        Boolean.parseBoolean(values.getOrDefault("require-device", "false")),
        modelPath,
        modelDigest,
        modelBytes,
        promptsPath,
        promptsDigest,
        prompts,
        maxTokens,
        warmupTokens,
        contextLength,
        arm);
  }

  /** Everything the run was told to do, which the report repeats so a rerun is possible. */
  record Configuration(
      String mode,
      Path report,
      String modelsRevision,
      boolean requireDevice,
      Optional<Path> modelPath,
      String modelSha256,
      long modelBytes,
      String promptsPath,
      String promptsSha256,
      List<String> prompts,
      int maxTokens,
      int warmupTokens,
      int contextLength,
      String arm) {

    Configuration {
      prompts = List.copyOf(prompts);
    }

    Optional<RunConfiguration> describe() {
      if (CAPABILITY.equals(mode)) {
        return Optional.empty();
      }
      return Optional.of(
          new RunConfiguration(
              mode,
              modelPath.map(Path::toAbsolutePath).map(Path::toString).orElse(""),
              modelSha256,
              modelBytes,
              promptsPath,
              promptsSha256,
              prompts.size(),
              maxTokens,
              warmupTokens,
              contextLength,
              arm,
              GreedyDecode.SAMPLING_RULE,
              Boolean.getBoolean(CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY),
              Integer.getInteger("models.cuda.device", 0),
              List.copyOf(ManagementFactory.getRuntimeMXBean().getInputArguments()),
              System.getProperty("user.dir", "")));
    }
  }

  // --- report records ----------------------------------------------------------------------

  /** The device the run executed on. */
  record Device(String name, int computeCapability, long globalMemoryBytes, int driverVersion) {}

  /**
   * The exact kernels that ran.
   *
   * <p>{@code sha256} is the identity of the compiled PTX. A performance number without it does not
   * say which kernels produced it; {@code toolchain} makes a rustc regression distinguishable from
   * one of our changes.
   */
  record Kernel(String sha256, String ptxTarget, String toolchain, List<String> entryPoints) {

    Kernel {
      entryPoints = List.copyOf(entryPoints);
    }
  }

  /** What the run was asked to do, in full, so the command can be reconstructed from the report. */
  record RunConfiguration(
      String mode,
      String modelPath,
      String modelSha256,
      long modelBytes,
      String promptsPath,
      String promptsSha256,
      int promptCount,
      int maxTokens,
      int warmupTokens,
      int contextLength,
      String arm,
      String sampling,
      boolean cudaDisabledProperty,
      int cudaDeviceOrdinal,
      List<String> jvmArguments,
      String workingDirectory) {

    RunConfiguration {
      jvmArguments = List.copyOf(jvmArguments);
    }
  }

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
      long decodeProjections,
      long measuredDecodeSteps,
      boolean decodeStepsMarked,
      double launchesPerDecodeStep,
      double transfersPerDecodeStep,
      double activationBytesPerDecodeStep) {

    Routing {
      acceleratedOperations = Map.copyOf(acceleratedOperations);
      refusals = Map.copyOf(refusals);
    }
  }

  /** Host and device memory, with what each figure does and does not cover. */
  record Memory(
      long peakProcessRssBytes,
      long residentBytes,
      long heapUsedBytes,
      long peakDeviceBytes,
      long deviceTotalMemoryBytes,
      String reading) {}

  /** G1's evidence. */
  record Parity(
      boolean selfTest,
      int promptCount,
      int tokensPerPrompt,
      int comparedTokens,
      int sequencesHittingEndOfGeneration,
      boolean identical,
      Optional<CudaParityRun.Divergence> firstDivergence,
      List<String> promptDigests) {

    Parity {
      promptDigests = List.copyOf(promptDigests);
    }
  }

  /** G4's evidence. */
  record Decode(
      Optional<CudaDecodeRun.ArmMeasurement> accelerated,
      Optional<CudaDecodeRun.ArmMeasurement> control,
      OptionalDouble speedup,
      double threshold,
      long wallClockMillis) {}

  /** One pre-registered gate's outcome, named so a report says which gate failed. */
  record GateOutcome(String gate, boolean passed, String evidence) {}

  /** The gates this run decided. Absent gates were not exercised by this mode. */
  record Gates(
      GateOutcome routingObservability,
      GateOutcome startupHonesty,
      Optional<GateOutcome> tokenParity,
      Optional<GateOutcome> decodeSpeedup) {

    boolean passed() {
      List<GateOutcome> decided = new ArrayList<>(List.of(routingObservability, startupHonesty));
      tokenParity.ifPresent(decided::add);
      decodeSpeedup.ifPresent(decided::add);
      return decided.stream().allMatch(GateOutcome::passed);
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
      Optional<RunConfiguration> configuration,
      Routing routing,
      long readinessMillis,
      Optional<Parity> parity,
      Optional<Decode> decode,
      Optional<Memory> memory,
      Gates gates,
      boolean qualified) {}
}
