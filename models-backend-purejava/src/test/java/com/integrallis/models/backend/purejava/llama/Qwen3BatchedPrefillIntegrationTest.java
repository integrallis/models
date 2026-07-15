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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

@Tag("integration")
class Qwen3BatchedPrefillIntegrationTest {

  private static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";
  private static final ModelJarRequirement QWEN3_0_6B_Q4_0 =
      ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
          .versionRange("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  @Test
  void batchedPrefillMatchesSequentialStateAtEveryLayer() throws Exception {
    Path model =
        ModelJarRegistry.fromClasspath()
            .resolve(QWEN3_0_6B_Q4_0)
            .orElseThrow()
            .localPath()
            .orElseThrow();
    String previous = System.getProperty(PREFILL_BATCH_SIZE_PROPERTY);

    try (Arena arena = Arena.ofShared()) {
      GgufFile file = GgufParser.parse(model, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = GgufTokenizer.fromMetadata(file.metadata()).encode("The quick brown fox");
      Map<LayerPosition, float[]> sequentialStates = new HashMap<>();
      Map<LayerPosition, float[]> batchedStates = new HashMap<>();

      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "1");
      LlamaForwardPass sequential = forwardPass(config, weights, sequentialStates);
      float[] sequentialLogits = null;
      for (int position = 0; position < tokens.length; position++) {
        sequentialLogits = sequential.forwardTransient(tokens[position], position);
      }

      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "32");
      LlamaForwardPass batched = forwardPass(config, weights, batchedStates);
      assertThat(batched.usesBatchedPrefill()).isTrue();
      float[] batchedLogits = batched.prefill(tokens, 0);

      assertStatesMatch(config, tokens.length, sequentialStates, batchedStates);
      assertSameBits("prompt logits", sequentialLogits, batchedLogits);

      for (int generated = 0; generated < 4; generated++) {
        int sequentialToken = argmax(sequentialLogits);
        int batchedToken = argmax(batchedLogits);
        assertThat(batchedToken)
            .as("generated token %d must match sequential argmax", generated)
            .isEqualTo(sequentialToken);
        int position = tokens.length + generated;
        sequentialLogits = sequential.forwardTransient(sequentialToken, position);
        batchedLogits = batched.forwardTransient(sequentialToken, position);
        for (int layer = 0; layer < config.numLayers(); layer++) {
          LayerPosition key = new LayerPosition(layer, position);
          assertSameBits(key.toString(), sequentialStates.get(key), batchedStates.get(key));
        }
        assertSameBits("decode logits at position " + position, sequentialLogits, batchedLogits);
      }
    } finally {
      restoreProperty(previous);
    }
  }

  private static LlamaForwardPass forwardPass(
      LlamaConfig config, LlamaWeights weights, Map<LayerPosition, float[]> states) {
    KvCache cache = new KvCache(config.numLayers(), 128, config.keyDim(), config.valueDim());
    return new LlamaForwardPass(
        config,
        weights,
        cache,
        (layer, position, state, offset, length) ->
            states.put(
                new LayerPosition(layer, position),
                Arrays.copyOfRange(state, offset, offset + length)));
  }

  private static void assertStatesMatch(
      LlamaConfig config,
      int positions,
      Map<LayerPosition, float[]> expected,
      Map<LayerPosition, float[]> actual) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      for (int position = 0; position < positions; position++) {
        LayerPosition key = new LayerPosition(layer, position);
        assertSameBits(key.toString(), expected.get(key), actual.get(key));
      }
    }
  }

  private static void assertSameBits(String label, float[] expected, float[] actual) {
    assertThat(actual).as("missing batched state for %s", label).isNotNull();
    assertThat(expected).as("missing sequential state for %s", label).isNotNull();
    for (int index = 0; index < expected.length; index++) {
      if (Float.floatToRawIntBits(actual[index]) != Float.floatToRawIntBits(expected[index])) {
        throw new AssertionError(
            label
                + " diverged at index "
                + index
                + ": sequential="
                + expected[index]
                + ", batched="
                + actual[index]
                + ", delta="
                + (actual[index] - expected[index]));
      }
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static void restoreProperty(String previous) {
    if (previous == null) {
      System.clearProperty(PREFILL_BATCH_SIZE_PROPERTY);
    } else {
      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, previous);
    }
  }

  private record LayerPosition(int layer, int position) {}
}
