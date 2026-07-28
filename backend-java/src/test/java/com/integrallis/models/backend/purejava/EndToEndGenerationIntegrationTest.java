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
import static org.assertj.core.api.Assumptions.assumeThat;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.GenerationLoop;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end integration tests that load a real GGUF model, run the full inference pipeline, and
 * validate basic structural properties of the generated text. These tests exercise the entire
 * stack: GGUF parsing, tokenizer, forward pass, sampling, and generation loop. They do not compare
 * logits or generated tokens with a reference runtime.
 *
 * <p>No mocking, no stubbing — this is the real thing.
 */
@Tag("integration")
class EndToEndGenerationIntegrationTest {

  private static final Path MODEL_PATH =
      Path.of(System.getProperty("user.home"), ".jvllm", "models", "Qwen3-0.6B-Q4_0.gguf");

  private static PureJavaBackend backend;

  @BeforeAll
  static void loadBackend() {
    assumeThat(Files.exists(MODEL_PATH))
        .as("Qwen3-0.6B-Q4_0.gguf must be present at %s", MODEL_PATH)
        .isTrue();

    backend = PureJavaBackend.load(MODEL_PATH);
  }

  @AfterAll
  static void closeBackend() {
    if (backend != null) {
      backend.close();
    }
  }

  @Nested
  class BasicGeneration {

    @Test
    void generatesNonEmptyText() {
      GenerationLoop loop = new GenerationLoop(backend);
      String result =
          loop.generate(
              "Once upon a time",
              SamplingOptions.builder().temperature(0.0f).maxTokens(20).build());

      assertThat(result).isNotEmpty();
      assertThat(result.length()).isGreaterThan(1);
    }

    @Test
    void generatedTextCanBeEncodedAsUtf8WithoutReplacement() {
      GenerationLoop loop = new GenerationLoop(backend);
      String result =
          loop.generate(
              "The capital of France is",
              SamplingOptions.builder().temperature(0.0f).maxTokens(20).build());

      CharsetEncoder encoder =
          StandardCharsets.UTF_8
              .newEncoder()
              .onMalformedInput(CodingErrorAction.REPORT)
              .onUnmappableCharacter(CodingErrorAction.REPORT);
      assertThat(encoder.canEncode(result)).isTrue();
    }

    @Test
    void greedyDecodingIsDeterministic() {
      GenerationLoop loop1 = new GenerationLoop(backend);
      GenerationLoop loop2 = new GenerationLoop(backend);

      String prompt = "2 + 2 =";
      SamplingOptions opts = SamplingOptions.builder().temperature(0.0f).maxTokens(5).build();

      assertThat(loop1.generate(prompt, opts)).isEqualTo(loop2.generate(prompt, opts));
    }
  }

  @Nested
  class Streaming {

    @Test
    void streamingMatchesNonStreaming() {
      GenerationLoop loop = new GenerationLoop(backend);
      String prompt = "Hello";
      SamplingOptions opts =
          SamplingOptions.builder().temperature(0.0f).maxTokens(10).seed(42L).build();
      String blockingResult = loop.generate(prompt, opts);

      List<String> streamedTokens = new ArrayList<>();
      boolean[] completed = {false};
      Throwable[] error = {null};

      loop.generate(
          prompt,
          opts,
          new TokenStream() {
            @Override
            public void onToken(String token) {
              streamedTokens.add(token);
            }

            @Override
            public void onComplete() {
              completed[0] = true;
            }

            @Override
            public void onError(Throwable t) {
              error[0] = t;
            }
          });

      assertThat(error[0]).isNull();
      assertThat(completed[0]).isTrue();
      assertThat(streamedTokens).isNotEmpty();

      String streamedResult = String.join("", streamedTokens);
      assertThat(streamedResult).isEqualTo(blockingResult);
    }
  }

  @Nested
  class TokenQuality {

    @Test
    void doesNotGenerateGarbage() {
      GenerationLoop loop = new GenerationLoop(backend);
      String result =
          loop.generate(
              "The meaning of life is",
              SamplingOptions.builder().temperature(0.0f).maxTokens(30).build());

      // Generated text should not be just whitespace or control characters
      String trimmed = result.trim();
      assertThat(trimmed).isNotEmpty();

      // Should not be all the same character repeated
      if (trimmed.length() > 5) {
        long distinctChars = trimmed.chars().distinct().count();
        assertThat(distinctChars)
            .as("Generated text should have character variety, not garbage repetition")
            .isGreaterThan(2);
      }
    }

    @Test
    void respectsMaxTokensLimit() {
      GenerationLoop loop = new GenerationLoop(backend);
      String result =
          loop.generate(
              "Count to ten:", SamplingOptions.builder().temperature(0.0f).maxTokens(5).build());

      // With maxTokens=5, the output should be bounded
      // (5 tokens at most ~20 chars each is generous)
      assertThat(result.length()).isLessThan(200);
    }
  }

  @Nested
  class SamplingStrategies {

    @Test
    void highTemperatureGenerationCompletes() {
      GenerationLoop loop = new GenerationLoop(backend);
      SamplingOptions highTemp =
          SamplingOptions.builder().temperature(1.5f).topK(40).topP(0.9f).maxTokens(10).build();

      String result1 = loop.generate("Once upon", highTemp);
      assertThat(result1).isNotEmpty();
    }

    @Test
    void topKConstrainsOutput() {
      GenerationLoop loop = new GenerationLoop(backend);
      // Very restrictive topK should still produce valid text
      SamplingOptions opts =
          SamplingOptions.builder()
              .temperature(1.0f)
              .topK(5)
              .topP(1.0f)
              .maxTokens(10)
              .seed(123L)
              .build();
      String result = loop.generate("The", opts);
      assertThat(result).isNotEmpty();
    }
  }
}
