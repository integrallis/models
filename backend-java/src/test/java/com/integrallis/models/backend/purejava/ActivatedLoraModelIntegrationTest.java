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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter;
import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter.Architecture;
import com.integrallis.models.backend.purejava.lora.ActivatedLoraAdapter.Projection;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.SharedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.Properties;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Real-weight gate for adapter loading, physical KV sharing, and exact base continuation. */
@Tag("integration")
@EnabledIfSystemProperty(named = ActivatedLoraModelIntegrationTest.ADAPTER_PROPERTY, matches = ".+")
class ActivatedLoraModelIntegrationTest {

  static final String ADAPTER_PROPERTY = "models.fixtures.activatedLoraDirectory";
  static final String MODEL_PROPERTY = "models.fixtures.activatedLoraModel";
  static final String ORACLE_PROPERTY = "models.fixtures.activatedLoraOracle";
  static final String CANDIDATE_PROPERTY = "models.fixtures.activatedLoraCandidate";

  private static final ModelFixtureRequirement QWEN3_0_6B_Q4_0 =
      ModelFixtureRequirement.of("hf://ggml-org/Qwen3-0.6B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation");
  private static final ModelFixtureRequirement QWEN3_1_7B_Q8_0 =
      ModelFixtureRequirement.of("hf://Qwen/Qwen3-1.7B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("text-generation");
  private static final ToolSpec WEATHER =
      new ToolSpec(
          "get-weather-for-zipcode",
          "Gets weather for a given zipcode",
          "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}},"
              + "\"required\":[\"zipcode\"]}");
  private static final SamplingOptions TOOL_OPTIONS =
      SamplingOptions.builder().temperature(0.0f).maxTokens(48).build();
  private static final SamplingOptions RESPONSE_OPTIONS =
      SamplingOptions.builder().temperature(0.0f).maxTokens(16).build();
  private static final String TRAINING_DATASET = "edbuildingstuff/bfcl-ft-data";
  private static final String TRAINING_DATASET_REVISION =
      "a7ceb3b1e1605f609fda6f2befec704f575290de";
  private static final String TRAINING_SOURCE_SHA256 =
      "915679b445c64676d7212e8953a0c3379c1024235e3c8168d1af2d2ef2ae7973";
  private static final String NO_CALL_DATASET = "MadeAgents/xlam-irrelevance-7.5k";
  private static final String NO_CALL_DATASET_REVISION = "34323bf09efc7e4a394998a0fa91ff997617c369";
  private static final String NO_CALL_SOURCE_SHA256 =
      "2e6f3d0adbd40248a592ea001e3f3a4f1624a5d50f434fa1e7d019e93e922327";
  private static final String PREPARATION_MANIFEST_SHA256 =
      "c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9";
  private static final String PREPARED_TRAIN_SHA256 =
      "3fe555c1e4a68b65b6715341cd1d1cbf9995e549cbdb267f15896b0a9b7edf77";
  private static final String PREPARED_VALIDATION_SHA256 =
      "f12c4c34c875d929b252d075749468f2c53d919744eca711e907927d9ecd83ca";

  private record ActivatedFixture(
      ModelFixtureRequirement requirement,
      String baseModel,
      String baseRevision,
      String formatterSha256,
      Architecture architecture,
      boolean requiresExactSelection) {}

  @Test
  void runsTheActivatedAndExactBaseBranchesOverOnePhysicalPrefix() {
    Path modelPath = modelPath();
    Path adapterPath = Path.of(System.getProperty(ADAPTER_PROPERTY));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "1024");

    try (ActivatedToolCallingModel model =
        new ActivatedToolCallingModel(
            PureJavaBackend.loadActivatedAdapter(modelPath, adapterPath), 1)) {
      assertPinnedTrainingProvenance(model.adapter(), modelPath);
      ModelPrompt selectionPrompt =
          ChatTemplate.CHATML_NO_THINK.render(
              List.of(ChatMessage.user("What is the weather for 88252?")), List.of(WEATHER));
      ToolCall selectedCall = ToolCall.of(0, WEATHER.name(), "{\"zipcode\":\"88252\"}");
      ModelPrompt resultPrompt =
          ChatTemplate.CHATML_NO_THINK.render(
              List.of(
                  ChatMessage.user("What is the weather for 88252?"),
                  ChatMessage.assistantToolCalls("", List.of(selectedCall)),
                  ChatMessage.tool(
                      WEATHER.name(),
                      "{\"zipcode\":\"88252\",\"conditions\":\"Rain\","
                          + "\"temperatureInFahrenheit\":78}")),
              List.of(WEATHER));

      String ordinaryBase = model.generate(resultPrompt, RESPONSE_OPTIONS);
      String recomputedToolOutput;
      try (ActivatedToolTurn recomputed =
          model.openToolTurn(
              selectionPrompt, ActivatedToolCallingModel.PrefixStrategy.RECOMPUTED)) {
        recomputedToolOutput =
            recomputed.generateToolCall(TOOL_OPTIONS, TokenConstraint.unrestricted());
        assertThat(recomputed.physicallySharesPrefix()).isFalse();
      }
      try (ActivatedToolTurn turn = model.openToolTurn(selectionPrompt)) {
        assertThat(turn.physicallySharesPrefix()).isTrue();
        assertThat(turn.sharedPrefixTokens()).isPositive();
        assertThat(turn.sharedPrefixBytes()).isPositive();

        String toolOutput = turn.generateToolCall(TOOL_OPTIONS, TokenConstraint.unrestricted());
        assertThat(toolOutput).isEqualTo(recomputedToolOutput).isNotBlank();
        assertThat(turn.toolMetrics().promptCache().cacheReadInputTokens())
            .isEqualTo(turn.sharedPrefixTokens());

        String sharedBase = turn.generateBaseResponse(resultPrompt, RESPONSE_OPTIONS);
        assertThat(sharedBase).isEqualTo(ordinaryBase);
        assertThat(turn.responseMetrics().promptCache().cacheReadInputTokens())
            .isEqualTo(turn.sharedPrefixTokens());
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  void extendsPhysicalKvSharingAcrossConsecutiveToolSelectionTurns() {
    Path modelPath = modelPath();
    Path adapterPath = Path.of(System.getProperty(ADAPTER_PROPERTY));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "1024");
    SamplingOptions oneToken = SamplingOptions.builder().temperature(0.0f).maxTokens(1).build();

    try (ActivatedToolCallingModel model =
        new ActivatedToolCallingModel(
            PureJavaBackend.loadActivatedAdapter(modelPath, adapterPath), 1)) {
      assertPinnedTrainingProvenance(model.adapter(), modelPath);
      ModelPrompt initialPrompt =
          ChatTemplate.CHATML_NO_THINK.render(
              List.of(ChatMessage.user("What is the weather for 88252?")), List.of(WEATHER));
      ToolCall selectedCall = ToolCall.of(0, WEATHER.name(), "{\"zipcode\":\"88252\"}");
      ModelPrompt resultPrompt =
          ChatTemplate.CHATML_NO_THINK.render(
              List.of(
                  ChatMessage.user("What is the weather for 88252?"),
                  ChatMessage.assistantToolCalls("", List.of(selectedCall)),
                  ChatMessage.tool(
                      WEATHER.name(),
                      "{\"zipcode\":\"88252\",\"conditions\":\"Rain\","
                          + "\"temperatureInFahrenheit\":78}")),
              List.of(WEATHER));
      String ordinaryBase = model.generate(resultPrompt, RESPONSE_OPTIONS);

      try (ActivatedToolTurn first = model.openToolTurn(initialPrompt)) {
        first.generateToolCall(oneToken, TokenConstraint.unrestricted());
        long initialBytes = first.sharedPrefixBytes();
        int initialTokens = first.sharedPrefixTokens();

        try (SharedToolTurn second = first.continueToolSelection(resultPrompt)) {
          assertThat(second.physicallySharesPrefix()).isTrue();
          assertThat(second.sharedPrefixTokens()).isGreaterThan(initialTokens);
          assertThat(second.sharedPrefixBytes()).isGreaterThan(initialBytes);

          second.generateToolCall(oneToken, TokenConstraint.unrestricted());
          assertThat(second.toolMetrics().promptCache().cacheReadInputTokens())
              .isEqualTo(second.sharedPrefixTokens());
          String sharedBase = second.generateBaseResponse(resultPrompt, RESPONSE_OPTIONS);

          assertThat(sharedBase).isEqualTo(ordinaryBase);
          assertThat(second.responseMetrics().promptCache().cacheReadInputTokens())
              .isEqualTo(second.sharedPrefixTokens());
        }
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  void extendsPhysicalKvSharingAfterTheBaseBranchCompletesAConversationalTurn() {
    Path modelPath = modelPath();
    Path adapterPath = Path.of(System.getProperty(ADAPTER_PROPERTY));
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "1024");
    SamplingOptions oneToken = SamplingOptions.builder().temperature(0.0f).maxTokens(1).build();
    ToolCall firstCall = ToolCall.of(0, WEATHER.name(), "{\"zipcode\":\"88252\"}");
    List<ChatMessage> resultHistory =
        List.of(
            ChatMessage.user("What is the weather for 88252?"),
            ChatMessage.assistantToolCalls("", List.of(firstCall)),
            ChatMessage.tool(
                WEATHER.name(),
                "{\"zipcode\":\"88252\",\"conditions\":\"Rain\","
                    + "\"temperatureInFahrenheit\":78}"));

    try (ActivatedToolCallingModel model =
        new ActivatedToolCallingModel(
            PureJavaBackend.loadActivatedAdapter(modelPath, adapterPath), 1)) {
      assertPinnedTrainingProvenance(model.adapter(), modelPath);
      ModelPrompt selectionPrompt =
          ChatTemplate.CHATML_NO_THINK.render(
              List.of(ChatMessage.user("What is the weather for 88252?")), List.of(WEATHER));
      ModelPrompt resultPrompt =
          ChatTemplate.CHATML_NO_THINK.render(resultHistory, List.of(WEATHER));
      String ordinaryBase = model.generate(resultPrompt, RESPONSE_OPTIONS);
      ModelPrompt laterSelectionPrompt =
          ChatTemplate.CHATML_NO_THINK.render(
              java.util.stream.Stream.concat(
                      resultHistory.stream(),
                      java.util.stream.Stream.of(
                          ChatMessage.assistant(ordinaryBase),
                          ChatMessage.user("What is the weather for 10001?")))
                  .toList(),
              List.of(WEATHER));

      try (ActivatedToolTurn first = model.openToolTurn(selectionPrompt)) {
        first.generateToolCall(oneToken, TokenConstraint.unrestricted());
        first.generateBaseResponse(resultPrompt, RESPONSE_OPTIONS);
        int firstTokens = first.sharedPrefixTokens();

        try (SharedToolTurn later = first.continueToolSelection(laterSelectionPrompt)) {
          assertThat(later.physicallySharesPrefix()).isTrue();
          assertThat(later.sharedPrefixTokens()).isGreaterThan(firstTokens);
          later.generateToolCall(oneToken, TokenConstraint.unrestricted());
          assertThat(later.toolMetrics().promptCache().cacheReadInputTokens())
              .isEqualTo(later.sharedPrefixTokens());
        }
      }
    } finally {
      restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
    }
  }

  @Test
  @EnabledIfSystemProperty(named = ORACLE_PROPERTY, matches = ".+")
  void matchesThePinnedTrainingTokenizerAndToolTemplateExactly() throws Exception {
    Path modelPath = modelPath();
    Path oraclePath = Path.of(System.getProperty(ORACLE_PROPERTY));
    Properties oracle = new Properties();
    try (InputStream input =
        Files.newInputStream(oraclePath.resolve("projection-oracle.properties"))) {
      oracle.load(input);
    }
    ActivatedFixture fixture = fixture();
    assertThat(required(oracle, "tokenizer.base.model")).isEqualTo(fixture.baseModel());
    assertThat(required(oracle, "tokenizer.base.revision")).isEqualTo(fixture.baseRevision());

    Path promptPath = oraclePath.resolve(required(oracle, "tokenizer.prompt.file"));
    assertThat(Files.size(promptPath)).isEqualTo(integer(oracle, "tokenizer.prompt.bytes"));
    assertThat(sha256(promptPath)).isEqualTo(required(oracle, "tokenizer.prompt.sha256"));
    ModelPrompt prompt =
        ChatTemplate.CHATML_NO_THINK.render(
            List.of(ChatMessage.user("What is the weather for 88252?")), List.of(WEATHER));
    assertThat(prompt.text()).isEqualTo(Files.readString(promptPath));

    Path tokensPath = oraclePath.resolve(required(oracle, "tokenizer.tokens.file"));
    assertThat(sha256(tokensPath)).isEqualTo(required(oracle, "tokenizer.tokens.sha256"));
    int[] expected = integers(tokensPath, integer(oracle, "tokenizer.tokens.count"));
    try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
      assertThat(backend.tokenizer().encode(prompt)).containsExactly(expected);
    }
  }

  @Test
  @EnabledIfSystemProperty(named = ORACLE_PROPERTY, matches = ".+")
  void matchesEveryActualAdapterProjectionAgainstTheIndependentNumpyOracle() throws Exception {
    Path modelPath = modelPath();
    Path adapterPath = Path.of(System.getProperty(ADAPTER_PROPERTY));
    Path oraclePath = Path.of(System.getProperty(ORACLE_PROPERTY));
    Properties oracle = new Properties();
    try (InputStream input =
        Files.newInputStream(oraclePath.resolve("projection-oracle.properties"))) {
      oracle.load(input);
    }
    assertThat(integer(oracle, "schema.version")).isEqualTo(1);
    Path binaryPath = oraclePath.resolve(required(oracle, "binary.file"));
    assertThat(sha256(binaryPath)).isEqualTo(required(oracle, "binary.sha256"));
    byte[] bytes = Files.readAllBytes(binaryPath);
    assertThat(bytes.length).isEqualTo(integer(oracle, "binary.floats") * Float.BYTES);
    ByteBuffer values = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    float maxAbsoluteError = Float.parseFloat(required(oracle, "max.absolute.error"));
    double minimumCosine = Double.parseDouble(required(oracle, "minimum.cosine"));
    Architecture architecture = fixture().architecture();

    try (Arena arena = Arena.ofConfined()) {
      ActivatedLoraAdapter adapter =
          ActivatedLoraAdapter.open(adapterPath, arena, modelPath, architecture);
      assertThat(adapter.adapterSha256()).isEqualTo(required(oracle, "adapter.sha256"));
      assertThat(adapter.rank()).isEqualTo(integer(oracle, "rank"));
      assertThat(adapter.alpha()).isEqualTo(integer(oracle, "alpha"));

      int probeCount = integer(oracle, "probe.count");
      assertThat(probeCount).isEqualTo(architecture.layers() * Projection.values().length);
      for (int probe = 0; probe < probeCount; probe++) {
        String prefix = "probe." + probe + ".";
        int layer = integer(oracle, prefix + "layer");
        Projection projection = Projection.valueOf(required(oracle, prefix + "projection"));
        float[] input =
            floats(
                values,
                integer(oracle, prefix + "input.offset"),
                integer(oracle, prefix + "input.length"));
        float[] expected =
            floats(
                values,
                integer(oracle, prefix + "output.offset"),
                integer(oracle, prefix + "output.length"));
        assertThat(input).hasSize(adapter.inputDimension(projection));
        assertThat(expected).hasSize(adapter.outputDimension(projection));

        float[] actual = new float[expected.length];
        adapter.addTo(layer, projection, actual, 0, input, 0);
        assertThat(maxAbsoluteDifference(actual, expected))
            .as("layer %d %s max absolute error", layer, projection)
            .isLessThanOrEqualTo(maxAbsoluteError);
        assertThat(cosine(actual, expected))
            .as("layer %d %s cosine", layer, projection)
            .isGreaterThanOrEqualTo(minimumCosine);
      }
    }
  }

  private static String required(Properties properties, String name) {
    String value = properties.getProperty(name);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException("missing oracle property: " + name);
    }
    return value;
  }

  private static void assertPinnedTrainingProvenance(
      ActivatedAdapterMetadata adapter, Path modelPath) {
    ActivatedFixture fixture = fixture();
    assertThat(adapter.baseModel()).isEqualTo(fixture.baseModel());
    assertThat(adapter.baseRevision()).isEqualTo(fixture.baseRevision());
    try {
      assertThat(adapter.baseArtifactSha256()).isEqualTo(sha256(modelPath));
    } catch (Exception failure) {
      throw new IllegalStateException("cannot hash real base fixture", failure);
    }
    ActivatedAdapterMetadata.TrainingProvenance training = adapter.trainingProvenance();
    assertThat(training.dataset()).isEqualTo(TRAINING_DATASET);
    assertThat(training.datasetRevision()).isEqualTo(TRAINING_DATASET_REVISION);
    assertThat(training.sourceFile()).isEqualTo("train.javajs.messages.jsonl");
    assertThat(training.sourceSha256()).isEqualTo(TRAINING_SOURCE_SHA256);
    if (fixture.requiresExactSelection()) {
      assertThat(training.sources())
          .containsExactly(
              new ActivatedAdapterMetadata.TrainingSource(
                  "tool-calls",
                  TRAINING_DATASET,
                  TRAINING_DATASET_REVISION,
                  "train.javajs.messages.jsonl",
                  TRAINING_SOURCE_SHA256),
              new ActivatedAdapterMetadata.TrainingSource(
                  "no-call",
                  NO_CALL_DATASET,
                  NO_CALL_DATASET_REVISION,
                  "xlam-7.5k-irrelevancek.json",
                  NO_CALL_SOURCE_SHA256));
      assertThat(training.preparedSchemaVersion()).isEqualTo(2);
      assertThat(training.preparedManifestSha256()).isEqualTo(PREPARATION_MANIFEST_SHA256);
      assertThat(training.trainSha256()).isEqualTo(PREPARED_TRAIN_SHA256);
      assertThat(training.validationSha256()).isEqualTo(PREPARED_VALIDATION_SHA256);
    }
    assertThat(training.formatter()).isEqualTo("train_alora.py");
    assertThat(training.formatterSha256()).isEqualTo(fixture.formatterSha256());
    if (fixture.requiresExactSelection()) {
      assertThat(training.trainSelection()).isPresent();
      assertThat(training.validationSelection()).isPresent();
    }
  }

  private static ActivatedFixture fixture() {
    String candidate = System.getProperty(CANDIDATE_PROPERTY, "qwen3-0.6b");
    return switch (candidate) {
      case "qwen3-0.6b" ->
          new ActivatedFixture(
              QWEN3_0_6B_Q4_0,
              "Qwen/Qwen3-0.6B",
              "c1899de289a04d12100db370d81485cdf75e47ca",
              "43e329817c6e7d7c16790316249c0c6c57d97c5ee7e27dcc132cd657e8ff9d08",
              new Architecture(28, 1024, 2048, 1024, 1024, 2048, 3072),
              false);
      case "qwen3-1.7b" ->
          new ActivatedFixture(
              QWEN3_1_7B_Q8_0,
              "Qwen/Qwen3-1.7B",
              "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
              "bcf5942e8689aa92c02998ce858b56580c728c1fce7e23451578b5c75e05ad8c",
              new Architecture(28, 2048, 2048, 1024, 1024, 2048, 6144),
              true);
      default -> throw new IllegalArgumentException("unsupported activated adapter: " + candidate);
    };
  }

  private static Path modelPath() {
    String configured = System.getProperty(MODEL_PROPERTY);
    if (configured != null && !configured.isBlank()) {
      Path path = Path.of(configured).toAbsolutePath().normalize();
      if (!Files.isRegularFile(path)) {
        throw new IllegalArgumentException("activated adapter model does not exist: " + path);
      }
      return path;
    }
    return ModelFixtureRegistry.fromClasspath()
        .resolve(fixture().requirement())
        .orElseThrow()
        .localPath()
        .orElseThrow();
  }

  private static int integer(Properties properties, String name) {
    return Integer.parseInt(required(properties, name));
  }

  private static float[] floats(ByteBuffer values, int floatOffset, int length) {
    float[] result = new float[length];
    int byteOffset = Math.multiplyExact(floatOffset, Float.BYTES);
    for (int index = 0; index < length; index++) {
      result[index] = values.getFloat(byteOffset + index * Float.BYTES);
    }
    return result;
  }

  private static int[] integers(Path path, int length) throws Exception {
    byte[] bytes = Files.readAllBytes(path);
    assertThat(bytes).hasSize(length * Integer.BYTES);
    ByteBuffer values = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    int[] result = new int[length];
    for (int index = 0; index < length; index++) {
      result[index] = values.getInt(index * Integer.BYTES);
    }
    return result;
  }

  private static float maxAbsoluteDifference(float[] actual, float[] expected) {
    float maximum = 0;
    for (int index = 0; index < actual.length; index++) {
      maximum = Math.max(maximum, Math.abs(actual[index] - expected[index]));
    }
    return maximum;
  }

  private static double cosine(float[] actual, float[] expected) {
    double dot = 0;
    double actualNorm = 0;
    double expectedNorm = 0;
    for (int index = 0; index < actual.length; index++) {
      dot += (double) actual[index] * expected[index];
      actualNorm += (double) actual[index] * actual[index];
      expectedNorm += (double) expected[index] * expected[index];
    }
    if (actualNorm == 0 || expectedNorm == 0) {
      throw new IllegalArgumentException("projection oracle vectors must be nonzero");
    }
    return dot / Math.sqrt(actualNorm * expectedNorm);
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (InputStream input = Files.newInputStream(path)) {
      byte[] buffer = new byte[8192];
      for (int read; (read = input.read(buffer)) >= 0; ) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
