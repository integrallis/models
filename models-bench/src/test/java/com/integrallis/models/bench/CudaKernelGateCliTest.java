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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.backend.cuda.CudaRoutingCounters;
import com.integrallis.models.backend.cuda.CudaStage;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The gate command, exercised off-device.
 *
 * <p>Everything here runs with no GPU and no model. That is the point: the campaign's failure mode
 * was a guide full of commands that had never been run, and a command whose only proving ground is
 * a paid GPU host is a command that gets proven at $1.09/hr.
 */
class CudaKernelGateCliTest {

  private static final String REVISION = "0123456789abcdef0123456789abcdef01234567";
  private static final ObjectMapper MAPPER = new ObjectMapper();

  @TempDir Path directory;

  // --- argument parsing --------------------------------------------------------------------

  @Test
  @DisplayName("capability mode needs no model, the model-driven modes do")
  void capabilityModeNeedsNoModelTheOthersDo() throws Exception {
    Path report = directory.resolve("report.json");

    CudaKernelGateCli.Configuration capability =
        CudaKernelGateCli.parse(
            new String[] {
              "--report", report.toString(), "--models-revision", REVISION, "--mode", "capability"
            });
    assertThat(capability.modelPath()).isEmpty();
    assertThat(capability.describe()).isEmpty();

    assertThatThrownBy(
            () ->
                CudaKernelGateCli.parse(
                    new String[] {
                      "--report",
                      report.toString(),
                      "--models-revision",
                      REVISION,
                      "--mode",
                      "parity"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--model is required");
  }

  @Test
  @DisplayName("a mode or arm the command cannot run is refused before anything is loaded")
  void unknownModesAndArmsAreRefused() {
    Path report = directory.resolve("report.json");
    assertThatThrownBy(
            () ->
                CudaKernelGateCli.parse(
                    new String[] {
                      "--report",
                      report.toString(),
                      "--models-revision",
                      REVISION,
                      "--mode",
                      "compare"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--mode must be one of");
  }

  @Test
  @DisplayName("the models revision must be a real git SHA")
  void theModelsRevisionMustBeARealGitSha() {
    Path report = directory.resolve("report.json");
    assertThatThrownBy(
            () ->
                CudaKernelGateCli.parse(
                    new String[] {"--report", report.toString(), "--models-revision", "HEAD"}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("40-character git SHA");
  }

  @Test
  @DisplayName("a one-token run is refused because it would measure prefill and call it decode")
  void aOneTokenRunIsRefused() throws Exception {
    Path model = Files.write(directory.resolve("model.gguf"), new byte[] {1, 2, 3});
    Path prompts = Files.writeString(directory.resolve("prompts.txt"), "a\nb\n");
    assertThatThrownBy(
            () ->
                CudaKernelGateCli.parse(
                    new String[] {
                      "--report",
                      directory.resolve("r.json").toString(),
                      "--models-revision",
                      REVISION,
                      "--mode",
                      "parity",
                      "--model",
                      model.toString(),
                      "--prompts",
                      prompts.toString(),
                      "--max-tokens",
                      "1"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--max-tokens");
  }

  @Test
  @DisplayName("the configuration block pins the model, the prompts and the sampling rule")
  void theConfigurationBlockPinsWhatWasRun() throws Exception {
    Path model = Files.write(directory.resolve("model.gguf"), new byte[] {9, 9, 9});
    Path prompts =
        Files.writeString(directory.resolve("prompts.txt"), "# header\nalpha\nbeta\ngamma\n");

    CudaKernelGateCli.Configuration configuration =
        CudaKernelGateCli.parse(
            new String[] {
              "--report",
              directory.resolve("r.json").toString(),
              "--models-revision",
              REVISION,
              "--mode",
              "decode",
              "--model",
              model.toString(),
              "--prompts",
              prompts.toString(),
              "--prompt-limit",
              "2"
            });

    assertThat(configuration.prompts()).containsExactly("alpha", "beta");
    CudaKernelGateCli.RunConfiguration described = configuration.describe().orElseThrow();
    assertThat(described.modelSha256()).isEqualTo(Hashing.sha256(model));
    assertThat(described.promptsSha256()).isEqualTo(Hashing.sha256(prompts));
    assertThat(described.promptCount()).isEqualTo(2);
    assertThat(described.sampling()).isEqualTo(GreedyDecode.SAMPLING_RULE);
    assertThat(described.arm()).isEqualTo("both");
  }

  // --- parity, G1 --------------------------------------------------------------------------

  @Test
  @DisplayName("with no device parity still runs, and says it is a self-test and not G1 evidence")
  void withNoDeviceParityRunsAsASelfTest() throws Exception {
    Path report = directory.resolve("parity.json");
    int status =
        CudaKernelGateCli.run(
            configuration("parity", report, false, "both"),
            CudaKernelGateCli.Accelerator.absent("no CUDA driver on this host", 3L),
            accelerated -> new ScriptedBackend(List.of(11, 12, 13, 14, 15), 0));

    assertThat(status).isZero();
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("accelerated").asBoolean()).isFalse();
    assertThat(json.get("parity").get("selfTest").asBoolean()).isTrue();
    assertThat(json.get("parity").get("identical").asBoolean()).isTrue();
    // A CPU-vs-CPU pass is not a gate result, and the report must not let it be filed as one.
    assertThat(json.get("qualified").asBoolean()).isFalse();
    assertThat(json.get("gates").get("tokenParity").get("passed").asBoolean()).isFalse();
    assertThat(json.get("gates").get("tokenParity").get("evidence").asText())
        .contains("self-test")
        .contains("Not G1 evidence");
  }

  @Test
  @DisplayName("--require-device refuses to produce a self-test where a gate result was asked for")
  void requireDeviceRefusesASelfTest() throws Exception {
    Path report = directory.resolve("parity.json");
    int status =
        CudaKernelGateCli.run(
            configuration("parity", report, true, "both"),
            CudaKernelGateCli.Accelerator.absent("no CUDA driver on this host", 3L),
            accelerated -> {
              throw new AssertionError("no arm may load when the device was required and absent");
            });

    assertThat(status).isEqualTo(2);
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("accelerated").asBoolean()).isFalse();
    assertThat(json.get("refusalReason").asText()).isEqualTo("no CUDA driver on this host");
  }

  @Test
  @DisplayName("identical sequences on a device pass G1")
  void identicalSequencesOnADevicePassG1() throws Exception {
    Path report = directory.resolve("parity.json");
    CudaRoutingCounters counters = routedCounters();

    int status =
        CudaKernelGateCli.run(
            configuration("parity", report, false, "both"),
            accelerator(counters),
            accelerated -> new ScriptedBackend(List.of(11, 12, 13, 14, 15), 0));

    assertThat(status).isZero();
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("qualified").asBoolean()).isTrue();
    assertThat(json.get("parity").get("selfTest").asBoolean()).isFalse();
    assertThat(json.get("parity").get("identical").asBoolean()).isTrue();
    assertThat(json.get("parity").get("comparedTokens").asInt()).isEqualTo(3 * 5);
    assertThat(json.get("parity").get("promptDigests")).hasSize(3);
    assertThat(json.get("gates").get("tokenParity").get("passed").asBoolean()).isTrue();
  }

  @Test
  @DisplayName("a divergence fails G1 and reports the index, both token ids, and what routed")
  void aDivergenceReportsTheIndexBothTokensAndWhatRouted() throws Exception {
    Path report = directory.resolve("parity.json");
    CudaRoutingCounters counters = new CudaRoutingCounters();

    int status =
        CudaKernelGateCli.run(
            configuration("parity", report, false, "both"),
            accelerator(counters),
            accelerated -> {
              if (accelerated) {
                // The accelerated arm routes an attention step for every token, then diverges at
                // token 2. That is the shape the pre-registration says points at expf.
                return new RoutingScriptedBackend(List.of(11, 12, 77, 14, 15), counters);
              }
              return new ScriptedBackend(List.of(11, 12, 13, 14, 15), 0);
            });

    assertThat(status).isEqualTo(1);
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("qualified").asBoolean()).isFalse();
    JsonNode divergence = json.get("parity").get("firstDivergence");
    assertThat(divergence.get("promptIndex").asInt()).isZero();
    assertThat(divergence.get("tokenIndex").asInt()).isEqualTo(2);
    assertThat(divergence.get("acceleratedTokenId").asInt()).isEqualTo(77);
    assertThat(divergence.get("controlTokenId").asInt()).isEqualTo(13);
    assertThat(divergence.get("attentionRouted").asBoolean()).isTrue();
    assertThat(divergence.get("projectionRouted").asBoolean()).isFalse();
    assertThat(divergence.get("reading").asText())
        .contains("expf")
        .contains("G1 is not to be loosened");
    assertThat(
            divergence
                .get("routing")
                .get("acceleratedOperations")
                .get("F32/DECODE_ATTENTION")
                .asLong())
        .isEqualTo(1L);
  }

  // --- decode, G4 --------------------------------------------------------------------------

  @Test
  @DisplayName("decode reports the overhead term, the identity and the configuration")
  void decodeReportsTheOverheadTermAndTheIdentity() throws Exception {
    Path report = directory.resolve("decode.json");
    CudaRoutingCounters counters = new CudaRoutingCounters();

    int status =
        CudaKernelGateCli.run(
            configuration("decode", report, false, "both"),
            accelerator(counters),
            accelerated ->
                accelerated
                    ? new RoutingScriptedBackend(List.of(1, 2, 3, 4, 5), counters)
                    : new ScriptedBackend(List.of(1, 2, 3, 4, 5), 0));

    JsonNode json = MAPPER.readTree(Files.readString(report));
    JsonNode routing = json.get("routing");
    // The pre-registration calls this the term that decides G4 and says to report it whether or
    // not G4 passes. It is present on a failing run by construction: the block is unconditional.
    assertThat(routing.get("decodeStepsMarked").asBoolean()).isTrue();
    assertThat(routing.get("launchesPerDecodeStep").asDouble()).isEqualTo(2.0);
    assertThat(routing.get("transfersPerDecodeStep").asDouble()).isEqualTo(1.0);
    assertThat(routing.get("activationBytesPerDecodeStep").asDouble()).isEqualTo(4096.0);
    // Steps are generated tokens; projections are dispatches. Conflating them would make the
    // launches-per-step figure 1.0 for every model, forever.
    assertThat(routing.get("decodeProjections").asLong())
        .isGreaterThan(routing.get("decodeSteps").asLong());
    // The measured step count excludes the warmup sequence and the prefill-produced first token
    // of each prompt: 3 prompts x (5 tokens - 1) = 12.
    assertThat(routing.get("measuredDecodeSteps").asLong()).isEqualTo(12L);
    assertThat(routing.get("kernelLaunches").asLong())
        .as("cumulative launches include warmup and prefill, which the per-step term excludes")
        .isGreaterThan(2 * routing.get("measuredDecodeSteps").asLong());

    assertThat(json.get("device").get("name").asText()).isEqualTo("NVIDIA L40S");
    assertThat(json.get("kernel").get("sha256").asText()).isNotEmpty();
    assertThat(json.get("kernel").get("entryPoints")).hasSize(3);
    assertThat(json.get("environment").get("processors").asInt()).isPositive();
    assertThat(json.get("configuration").get("sampling").asText())
        .isEqualTo(GreedyDecode.SAMPLING_RULE);
    assertThat(json.get("memory").get("reading").asText()).contains("lower bound");
    assertThat(json.get("decode").get("accelerated").get("decodeTokensPerSecond").asDouble())
        .isPositive();
    assertThat(json.get("decode").get("control").get("decodeTokensPerSecond").asDouble())
        .isPositive();
    assertThat(json.get("decode").get("threshold").asDouble()).isEqualTo(3.0);
    assertThat(status).isIn(0, 1);
  }

  @Test
  @DisplayName("G4 passes only when the accelerated arm is at least three times the control")
  void g4PassesOnlyAtThreeTimesTheControl() throws Exception {
    Path fast = directory.resolve("fast.json");
    CudaRoutingCounters counters = routedCounters();

    int passing =
        CudaKernelGateCli.run(
            configuration("decode", fast, false, "both"),
            accelerator(counters),
            accelerated ->
                new ScriptedBackend(
                    List.of(1, 2, 3, 4, 5), 0, accelerated ? 200_000L : 20_000_000L));

    assertThat(passing).isZero();
    JsonNode json = MAPPER.readTree(Files.readString(fast));
    assertThat(json.get("decode").get("speedup").asDouble()).isGreaterThan(3.0);
    assertThat(json.get("gates").get("decodeSpeedup").get("passed").asBoolean()).isTrue();
    assertThat(json.get("qualified").asBoolean()).isTrue();

    Path slow = directory.resolve("slow.json");
    int failing =
        CudaKernelGateCli.run(
            configuration("decode", slow, false, "both"),
            accelerator(routedCounters()),
            accelerated ->
                new ScriptedBackend(
                    List.of(1, 2, 3, 4, 5), 0, accelerated ? 10_000_000L : 12_000_000L));

    assertThat(failing).isEqualTo(1);
    JsonNode slowJson = MAPPER.readTree(Files.readString(slow));
    assertThat(slowJson.get("decode").get("speedup").asDouble()).isLessThan(3.0);
    assertThat(slowJson.get("gates").get("decodeSpeedup").get("passed").asBoolean()).isFalse();
  }

  @Test
  @DisplayName("one arm alone leaves G4 unmeasured rather than reporting a zero speedup")
  void oneArmAloneLeavesG4Unmeasured() throws Exception {
    Path report = directory.resolve("one-arm.json");
    int status =
        CudaKernelGateCli.run(
            configuration("decode", report, false, "accelerated"),
            accelerator(routedCounters()),
            accelerated -> new ScriptedBackend(List.of(1, 2, 3, 4, 5), 0));

    assertThat(status).isEqualTo(1);
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("decode").get("control").isNull()).isTrue();
    assertThat(json.get("decode").get("speedup").isNull()).isTrue();
    assertThat(json.get("gates").get("decodeSpeedup").get("evidence").asText())
        .contains("G4 is a ratio and was not measured");
  }

  @Test
  @DisplayName("an inert accelerator fails G2 even when it is fast")
  void anInertAcceleratorFailsG2() throws Exception {
    Path report = directory.resolve("inert.json");
    int status =
        CudaKernelGateCli.run(
            configuration("decode", report, false, "both"),
            accelerator(new CudaRoutingCounters()),
            accelerated ->
                new ScriptedBackend(
                    List.of(1, 2, 3, 4, 5), 0, accelerated ? 200_000L : 20_000_000L));

    assertThat(status).isEqualTo(1);
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("decode").get("speedup").asDouble()).isGreaterThan(3.0);
    assertThat(json.get("gates").get("routingObservability").get("passed").asBoolean()).isFalse();
    assertThat(json.get("qualified").asBoolean()).isFalse();
  }

  // --- capability, unchanged behaviour -----------------------------------------------------

  @Test
  @DisplayName("capability mode without a device still writes a report and exits 0")
  void capabilityWithoutADeviceStillWritesAReport() throws Exception {
    Path report = directory.resolve("capability.json");
    int status =
        CudaKernelGateCli.run(
            configuration("capability", report, false, "both"),
            CudaKernelGateCli.Accelerator.absent("no CUDA driver on this host", 5L),
            accelerated -> {
              throw new AssertionError("capability mode must not load a model");
            });

    assertThat(status).isZero();
    JsonNode json = MAPPER.readTree(Files.readString(report));
    assertThat(json.get("schemaVersion").asInt()).isEqualTo(2);
    assertThat(json.get("accelerated").asBoolean()).isFalse();
    assertThat(json.get("parity").isNull()).isTrue();
    assertThat(json.get("decode").isNull()).isTrue();
    assertThat(json.get("gates").get("tokenParity").isNull()).isTrue();
  }

  // --- helpers -----------------------------------------------------------------------------

  private CudaKernelGateCli.Configuration configuration(
      String mode, Path report, boolean requireDevice, String arm) {
    return new CudaKernelGateCli.Configuration(
        mode,
        report,
        REVISION,
        requireDevice,
        Optional.of(directory.resolve("model.gguf")),
        "modeldigest",
        123L,
        directory.resolve("prompts.txt").toString(),
        "promptsdigest",
        List.of("alpha", "beta", "gamma"),
        5,
        2,
        4_096,
        arm);
  }

  private static CudaKernelGateCli.Accelerator accelerator(CudaRoutingCounters counters) {
    return new CudaKernelGateCli.Accelerator(
        true,
        "",
        525L,
        new CudaKernelGateCli.Device("NVIDIA L40S", 89, 48_305_799_168L, 12_080),
        new CudaKernelGateCli.Kernel(
            "8f6fcdb39837aa11223344556677889900aabbccddeeff00112233445566778899",
            "sm_80",
            "nightly-2026-09-17",
            List.of(
                "models_q4k_decode_projection",
                "models_q6k_decode_projection",
                "models_gqa_decode_attention")),
        Optional.of(counters));
  }

  private static CudaRoutingCounters routedCounters() {
    CudaRoutingCounters counters = new CudaRoutingCounters();
    counters.accelerated(GgufTensorType.Q4_K, CudaStage.DECODE_PROJECTION, 1);
    return counters;
  }

  /** A scripted backend that also drives the routing counters, as the real kernel would. */
  private static final class RoutingScriptedBackend implements InferenceBackend {

    private final ScriptedBackend delegate;
    private final CudaRoutingCounters counters;

    private RoutingScriptedBackend(List<Integer> script, CudaRoutingCounters counters) {
      this.delegate = new ScriptedBackend(script, 0);
      this.counters = counters;
    }

    @Override
    public String name() {
      return delegate.name();
    }

    @Override
    public com.integrallis.models.api.ModelMetadata metadata() {
      return delegate.metadata();
    }

    @Override
    public com.integrallis.models.api.Tokenizer tokenizer() {
      return delegate.tokenizer();
    }

    @Override
    public float[] prefill(int[] tokens, int startPosition) {
      counters.accelerated(GgufTensorType.Q4_K, CudaStage.PREFILL_PROJECTION, 1);
      counters.launched();
      return delegate.prefill(tokens, startPosition);
    }

    @Override
    public float[] forward(int token, int position) {
      // Two launches, one round trip and 4 KiB of activations per generated token: what the real
      // kernel's counters would show for a single-layer model with attention routed.
      counters.accelerated(GgufTensorType.F32, CudaStage.DECODE_ATTENTION, 1);
      counters.decodeProjection();
      counters.decodeProjection();
      counters.launched();
      counters.launched();
      counters.copiedToDevice(4_096);
      return delegate.forward(token, position);
    }

    @Override
    public void reset() {
      delegate.reset();
    }

    @Override
    public void close() {
      delegate.close();
    }
  }
}
