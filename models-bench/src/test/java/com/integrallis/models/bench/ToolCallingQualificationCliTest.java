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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.runtime.GenerationMetrics;
import com.integrallis.models.runtime.SharedToolTurn;
import com.integrallis.models.runtime.TokenConstraint;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.ToolSyntax;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ToolCallingQualificationCliTest {

  @Test
  void acceptsThePinnedQwen3EightBArtifact() throws Exception {
    Path artifact = temporary.resolve("qwen3-8b.gguf");
    Files.writeString(artifact, "fixture");

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "qwen3-8b",
              "--model",
              artifact.toString(),
              "--models-revision",
              "2c702d4b0801b6b16de6c93c04c373efb4ee1853"
            });

    assertThat(configuration.candidate()).isEqualTo(ToolCallingCandidate.QWEN3_8B);
    assertThat(configuration.candidate().modelId()).isEqualTo("qwen3_8b_q4_k_m");
  }

  private static final String REVISION = "a".repeat(40);

  @TempDir Path temporary;

  @Test
  void parsesAPinnedCandidateAndOutputControls() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "qwen3-0.6b",
              "--model",
              artifact.toString(),
              "--models-revision",
              REVISION,
              "--max-tokens",
              "192",
              "--case",
              "currency",
              "--report",
              temporary.resolve("report.json").toString()
            });

    assertThat(configuration.candidate()).isEqualTo(ToolCallingCandidate.QWEN3_06B);
    assertThat(configuration.model()).isEqualTo(artifact);
    assertThat(configuration.maxTokens()).isEqualTo(192);
    assertThat(configuration.caseId()).isEqualTo("currency");
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
    assertThat(configuration.adapter()).isNull();
  }

  @Test
  void parsesAnActivatedAdapterDirectoryForTheQwenCandidate() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "qwen3-0.6b",
              "--model",
              artifact.toString(),
              "--adapter",
              adapter.toString(),
              "--models-revision",
              REVISION
            });

    assertThat(configuration.adapter()).isEqualTo(adapter);
  }

  @Test
  void parsesAnActivatedAdapterDirectoryForQwen3OnePointSevenBillion() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "qwen3-1.7b",
              "--model",
              artifact.toString(),
              "--adapter",
              adapter.toString(),
              "--models-revision",
              REVISION
            });

    assertThat(configuration.candidate()).isEqualTo(ToolCallingCandidate.QWEN3_17B);
    assertThat(configuration.adapter()).isEqualTo(adapter);
  }

  @Test
  void rejectsActivatedAdaptersForCandidatesWithoutTheMatchingRuntimeGraph() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ToolCallingQualificationCli.parse(
                    new String[] {
                      "--candidate",
                      "hammer2.1-0.5b",
                      "--model",
                      artifact.toString(),
                      "--adapter",
                      adapter.toString(),
                      "--models-revision",
                      REVISION
                    }))
        .withMessageContaining("Qwen3");
  }

  @Test
  void parsesThePinnedHammerCandidate() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("hammer.gguf"), "fixture");

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "hammer2.1-0.5b",
              "--model",
              artifact.toString(),
              "--models-revision",
              REVISION
            });

    assertThat(configuration.candidate()).isEqualTo(ToolCallingCandidate.HAMMER21_05B);
    assertThat(configuration.candidate().template()).isEqualTo(ChatTemplate.HAMMER);
    assertThat(configuration.candidate().synthesizesToolResults()).isFalse();
    assertThat(ToolCallingCandidate.QWEN3_06B.synthesizesToolResults()).isTrue();
  }

  @Test
  void parsesThePinnedHammerOnePointFiveBillionCandidate() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("hammer-1.5b.gguf"), "fixture");

    ToolCallingQualificationCli.Configuration configuration =
        ToolCallingQualificationCli.parse(
            new String[] {
              "--candidate",
              "hammer2.1-1.5b",
              "--model",
              artifact.toString(),
              "--models-revision",
              REVISION
            });

    assertThat(configuration.candidate()).isEqualTo(ToolCallingCandidate.HAMMER21_15B);
    assertThat(configuration.candidate().template()).isEqualTo(ChatTemplate.HAMMER);
    assertThat(configuration.candidate().synthesizesToolResults()).isFalse();
  }

  @Test
  void requiresCandidateModelAndImmutableRevision() throws Exception {
    Path artifact = Files.writeString(temporary.resolve("model.gguf"), "fixture");

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ToolCallingQualificationCli.parse(
                    new String[] {"--model", artifact.toString(), "--models-revision", REVISION}))
        .withMessageContaining("--candidate");
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ToolCallingQualificationCli.parse(
                    new String[] {"--candidate", "qwen3-0.6b", "--model", artifact.toString()}))
        .withMessageContaining("--models-revision");
  }

  @Test
  void followUpRequiresAConversationalAnswerInsteadOfRawJsonOrAnotherCall() {
    ToolSpec weather =
        new ToolSpec(
            "get-weather-for-zipcode",
            "Gets weather",
            "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}}}");
    ObjectMapper mapper = new ObjectMapper();

    assertThat(
            ToolCallingQualificationCli.evaluateFollowUp(
                    mapper,
                    ToolSyntax.QWEN,
                    List.of(weather),
                    "The weather for 88252 is raining cats and dogs at 78 F.",
                    20)
                .passed())
        .isTrue();
    assertThat(
            ToolCallingQualificationCli.evaluateFollowUp(
                    mapper,
                    ToolSyntax.QWEN,
                    List.of(weather),
                    "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\",\"temperatureInFahrenheit\":78}",
                    20)
                .passed())
        .isFalse();
    assertThat(
            ToolCallingQualificationCli.evaluateFollowUp(
                    mapper,
                    ToolSyntax.QWEN,
                    List.of(weather),
                    "<tool_call>{\"name\":\"get-weather-for-zipcode\",\"arguments\":{\"zipcode\":\"88252\"}}</tool_call>",
                    20)
                .passed())
        .isFalse();
  }

  @Test
  void activatedFollowUpAsksTheSpecialistWhetherAnotherToolIsNeededBeforeBaseNarration() {
    ScriptedTurn afterResult =
        new ScriptedTurn("<tool_call>[]</tool_call>", "It is raining at 78 F.", null);
    ScriptedTurn initial =
        new ScriptedTurn(
            "<tool_call>{\"name\":\"get-weather-for-zipcode\",\"arguments\":{\"zipcode\":\"88252\"}}</tool_call>",
            null,
            afterResult);

    ToolCallingQualificationCli.ActivatedFollowUp result =
        ToolCallingQualificationCli.generateActivatedFollowUp(
            initial,
            ModelPrompt.builder().text("result prompt").build(),
            SamplingOptions.builder().temperature(0).maxTokens(32).build(),
            ToolSyntax.QWEN,
            List.of(
                new ToolSpec(
                    "get-weather-for-zipcode",
                    "Gets weather",
                    "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}}}")),
            TokenConstraint::unrestricted);

    assertThat(result.initialToolOutput()).contains("88252");
    assertThat(result.postResultToolOutput()).isEqualTo("<tool_call>[]</tool_call>");
    assertThat(result.answer()).isEqualTo("It is raining at 78 F.");
    assertThat(initial.toolGenerations).isEqualTo(1);
    assertThat(initial.continuations).isEqualTo(1);
    assertThat(initial.baseGenerations).isZero();
    assertThat(afterResult.toolGenerations).isEqualTo(1);
    assertThat(afterResult.baseGenerations).isEqualTo(1);
  }

  @Test
  void activatedFollowUpSurfacesARepeatedCallInsteadOfFabricatingNarration() {
    String repeated =
        "<tool_call>{\"name\":\"get-weather-for-zipcode\",\"arguments\":{\"zipcode\":\"88252\"}}</tool_call>";
    ScriptedTurn afterResult = new ScriptedTurn(repeated, null, null);
    ScriptedTurn initial = new ScriptedTurn(repeated, null, afterResult);
    ToolSpec weather =
        new ToolSpec(
            "get-weather-for-zipcode",
            "Gets weather",
            "{\"type\":\"object\",\"properties\":{\"zipcode\":{\"type\":\"string\"}}}");

    ToolCallingQualificationCli.ActivatedFollowUp result =
        ToolCallingQualificationCli.generateActivatedFollowUp(
            initial,
            ModelPrompt.builder().text("result prompt").build(),
            SamplingOptions.builder().temperature(0).maxTokens(32).build(),
            ToolSyntax.QWEN,
            List.of(weather),
            TokenConstraint::unrestricted);

    assertThat(result.answer()).isEqualTo(repeated);
    assertThat(afterResult.baseGenerations).isZero();
  }

  private static final class ScriptedTurn implements SharedToolTurn {
    private final String toolOutput;
    private final String baseOutput;
    private final SharedToolTurn continuation;
    private int toolGenerations;
    private int baseGenerations;
    private int continuations;

    private ScriptedTurn(String toolOutput, String baseOutput, SharedToolTurn continuation) {
      this.toolOutput = toolOutput;
      this.baseOutput = baseOutput;
      this.continuation = continuation;
    }

    @Override
    public String generateToolCall(SamplingOptions options, TokenConstraint constraint) {
      toolGenerations++;
      return toolOutput;
    }

    @Override
    public void generateToolCall(
        SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
      throw new UnsupportedOperationException();
    }

    @Override
    public SharedToolTurn continueToolSelection(ModelPrompt renderedToolPrompt) {
      continuations++;
      return continuation;
    }

    @Override
    public String generateBaseResponse(ModelPrompt prompt, SamplingOptions options) {
      baseGenerations++;
      return baseOutput;
    }

    @Override
    public void generateBaseResponse(
        ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      throw new UnsupportedOperationException();
    }

    @Override
    public int sharedPrefixTokens() {
      return 1;
    }

    @Override
    public long sharedPrefixBytes() {
      return 1;
    }

    @Override
    public boolean physicallySharesPrefix() {
      return true;
    }

    @Override
    public GenerationMetrics toolMetrics() {
      return GenerationMetrics.unavailable();
    }

    @Override
    public GenerationMetrics responseMetrics() {
      return GenerationMetrics.unavailable();
    }

    @Override
    public void close() {}
  }
}
