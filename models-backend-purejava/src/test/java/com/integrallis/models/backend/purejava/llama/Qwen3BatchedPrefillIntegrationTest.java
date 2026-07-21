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
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import com.integrallis.vectors.core.GgufQ4Kernel;
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
  private static final int[] LLAMA_CPP_GREEDY_TOKENS = {34208, 916, 279, 15678};
  private static final ModelJarRequirement QWEN3_0_6B_Q4_0 =
      ModelJarRequirement.forSource("hf://ggml-org/Qwen3-0.6B-GGUF")
          .versionRange("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation")
          .build();

  @Test
  void batchedLayerZeroKeyProjectionMatchesIndependentGemv() throws Exception {
    Path model = modelPath();

    try (Arena arena = Arena.ofShared()) {
      GgufFile file = GgufParser.parse(model, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      LlamaWeights.LayerWeights layer = weights.layer(0);
      int[] tokens = GgufTokenizer.fromMetadata(file.metadata()).encode("The quick brown fox");
      int batchSize = tokens.length;
      int cols = config.embeddingDim();
      int rows = config.keyDim();
      int headDim = config.keyLength();
      int blocks = cols / 32;
      float[] inputs = new float[batchSize * cols];
      float[] embedding = new float[cols];
      for (int batch = 0; batch < batchSize; batch++) {
        weights.embedToken(tokens[batch], embedding);
        TensorOps.rmsNorm(
            inputs, batch * cols, embedding, 0, layer.attentionNorm(), cols, config.rmsNormEps());
      }

      float[] expectedProjection = new float[batchSize * rows];
      float[] expectedBiased = new float[batchSize * rows];
      float[] expectedNormalized = new float[batchSize * rows];
      float[] expectedRope = new float[batchSize * rows];
      float[] actual = new float[batchSize * rows];
      float[] query = new float[cols];
      float[] gemvOut = new float[rows];
      byte[] gemvQuants = new byte[cols];
      float[] gemvScales = new float[blocks];
      byte[] expectedQuants = new byte[batchSize * cols];
      float[] expectedScales = new float[batchSize * blocks];
      byte[] batchQuants = new byte[batchSize * cols];
      float[] batchScales = new float[batchSize * blocks];
      float[] batchLanes = new float[batchSize * rows * 8];
      RopeTable sequentialRope =
          new RopeTable(headDim, config.ropeTheta(), config.ropeFrequencyScale());
      RopeTable batchedRope =
          new RopeTable(headDim, config.ropeTheta(), config.ropeFrequencyScale());

      for (int iteration = 0; iteration < 16; iteration++) {
        for (int batch = 0; batch < batchSize; batch++) {
          System.arraycopy(inputs, batch * cols, query, 0, cols);
          TensorOps.ggufMatmul(
              gemvOut,
              query,
              layer.wk(),
              layer.wkType(),
              rows,
              cols,
              gemvQuants,
              gemvScales,
              new short[(cols + 15) / 16],
              GgufQ4Kernel.WIDENED);
          int outputOffset = batch * rows;
          System.arraycopy(gemvQuants, 0, expectedQuants, batch * cols, cols);
          System.arraycopy(gemvScales, 0, expectedScales, batch * blocks, blocks);
          System.arraycopy(gemvOut, 0, expectedProjection, outputOffset, rows);
          addBias(gemvOut, layer.kBias());
          System.arraycopy(gemvOut, 0, expectedBiased, outputOffset, rows);
          for (int head = 0; head < config.numKvHeads(); head++) {
            int headOffset = head * headDim;
            TensorOps.rmsNorm(
                gemvOut,
                headOffset,
                gemvOut,
                headOffset,
                layer.kNorm(),
                headDim,
                config.rmsNormEps());
          }
          System.arraycopy(gemvOut, 0, expectedNormalized, outputOffset, rows);
          sequentialRope.prepare(batch);
          if (config.usesRope(0)) {
            for (int head = 0; head < config.numKvHeads(); head++) {
              sequentialRope.apply(gemvOut, head * headDim, config.usesNeoxRope());
            }
          }
          System.arraycopy(gemvOut, 0, expectedRope, outputOffset, rows);
        }

        TensorOps.ggufBatchedMatmul(
            actual,
            inputs,
            layer.wk(),
            layer.wkType(),
            batchSize,
            rows,
            cols,
            batchQuants,
            batchScales,
            new short[batchSize * ((cols + 15) / 16)],
            batchLanes,
            GgufQ4Kernel.WIDENED);
        assertSameBytes("Q8 activation iteration " + iteration, expectedQuants, batchQuants);
        assertSameBits("Q8 scale iteration " + iteration, expectedScales, batchScales);
        assertSameBits("layer 0 key projection iteration " + iteration, expectedProjection, actual);

        for (int batch = 0; batch < batchSize; batch++) {
          int outputOffset = batch * rows;
          addBias(actual, outputOffset, layer.kBias());
        }
        assertSameBits("layer 0 biased key iteration " + iteration, expectedBiased, actual);

        for (int batch = 0; batch < batchSize; batch++) {
          int outputOffset = batch * rows;
          for (int head = 0; head < config.numKvHeads(); head++) {
            int headOffset = outputOffset + head * headDim;
            TensorOps.rmsNorm(
                actual,
                headOffset,
                actual,
                headOffset,
                layer.kNorm(),
                headDim,
                config.rmsNormEps());
          }
        }
        assertSameBits("layer 0 normalized key iteration " + iteration, expectedNormalized, actual);

        batchedRope.prepareBatch(0, batchSize);
        if (config.usesRope(0)) {
          for (int batch = 0; batch < batchSize; batch++) {
            int outputOffset = batch * rows;
            for (int head = 0; head < config.numKvHeads(); head++) {
              batchedRope.applyBatch(
                  actual, outputOffset + head * headDim, batch, config.usesNeoxRope());
            }
          }
        }
        assertSameBits("layer 0 rotary key iteration " + iteration, expectedRope, actual);
      }
    }
  }

  @Test
  void batchedPrefillMatchesSequentialStateAtEveryLayer() throws Exception {
    Path model = modelPath();
    String previous = System.getProperty(PREFILL_BATCH_SIZE_PROPERTY);

    try (Arena arena = Arena.ofShared()) {
      GgufFile file = GgufParser.parse(model, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      int[] tokens = GgufTokenizer.fromMetadata(file.metadata()).encode("The quick brown fox");
      Map<LayerPosition, float[]> sequentialStates = new HashMap<>();
      Map<LayerPosition, float[]> batchedStates = new HashMap<>();

      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "1");
      ForwardFixture sequentialFixture = forwardPass(config, weights, sequentialStates);
      LlamaForwardPass sequential = sequentialFixture.forwardPass();
      float[] sequentialLogits = null;
      for (int position = 0; position < tokens.length; position++) {
        sequentialLogits = sequential.forwardTransient(tokens[position], position);
      }

      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, "32");
      ForwardFixture batchedFixture = forwardPass(config, weights, batchedStates);
      LlamaForwardPass batched = batchedFixture.forwardPass();
      assertThat(batched.usesBatchedPrefill()).isTrue();
      float[] batchedLogits = batched.prefill(tokens, 0);

      assertCachesMatch(config, tokens.length, sequentialFixture.cache(), batchedFixture.cache());
      assertStatesMatch(config, tokens.length, sequentialStates, batchedStates);
      assertSameBits("prompt logits", sequentialLogits, batchedLogits);

      for (int generated = 0; generated < 4; generated++) {
        int sequentialToken = argmax(sequentialLogits);
        int batchedToken = argmax(batchedLogits);
        assertThat(sequentialToken)
            .as("sequential generated token %d must match llama.cpp", generated)
            .isEqualTo(LLAMA_CPP_GREEDY_TOKENS[generated]);
        assertThat(batchedToken)
            .as("generated token %d must match sequential argmax", generated)
            .isEqualTo(sequentialToken);
        int position = tokens.length + generated;
        sequentialLogits = sequential.forwardTransient(sequentialToken, position);
        batchedLogits = batched.forwardTransient(sequentialToken, position);
        assertCachesMatch(config, position + 1, sequentialFixture.cache(), batchedFixture.cache());
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

  private static ForwardFixture forwardPass(
      LlamaConfig config, LlamaWeights weights, Map<LayerPosition, float[]> states) {
    KvCache cache = new KvCache(config.numLayers(), 128, config.keyDim(), config.valueDim());
    LlamaForwardPass forwardPass =
        new LlamaForwardPass(
            config,
            weights,
            cache,
            (layer, position, state, offset, length) ->
                states.put(
                    new LayerPosition(layer, position),
                    Arrays.copyOfRange(state, offset, offset + length)));
    return new ForwardFixture(forwardPass, cache);
  }

  private static void assertCachesMatch(
      LlamaConfig config, int positions, KvCache expected, KvCache actual) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      for (int position = 0; position < positions; position++) {
        assertSameBits(
            "key cache at " + new LayerPosition(layer, position),
            expected.key(layer, position),
            actual.key(layer, position));
        assertSameBits(
            "value cache at " + new LayerPosition(layer, position),
            expected.value(layer, position),
            actual.value(layer, position));
      }
    }
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

  private static void assertSameBytes(String label, byte[] expected, byte[] actual) {
    for (int index = 0; index < expected.length; index++) {
      if (actual[index] != expected[index]) {
        throw new AssertionError(
            label
                + " diverged at index "
                + index
                + ": sequential="
                + expected[index]
                + ", batched="
                + actual[index]);
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

  private static void addBias(float[] vector, float[] bias) {
    addBias(vector, 0, bias);
  }

  private static void addBias(float[] vector, int offset, float[] bias) {
    for (int index = 0; index < bias.length; index++) {
      vector[offset + index] += bias[index];
    }
  }

  private static void restoreProperty(String previous) {
    if (previous == null) {
      System.clearProperty(PREFILL_BATCH_SIZE_PROPERTY);
    } else {
      System.setProperty(PREFILL_BATCH_SIZE_PROPERTY, previous);
    }
  }

  private static Path modelPath() {
    return ModelJarRegistry.fromClasspath()
        .resolve(QWEN3_0_6B_Q4_0)
        .orElseThrow()
        .localPath()
        .orElseThrow();
  }

  private record LayerPosition(int layer, int position) {}

  private record ForwardFixture(LlamaForwardPass forwardPass, KvCache cache) {}
}
