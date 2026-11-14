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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Batched encoding over documents of unequal length, on the real forward pass.
 *
 * <p>The batched path (#440) masked padded positions as keys and as queries with one condition and
 * then ran softmax on every row, so a padded query row had no finite input and the first ragged
 * batch threw (#455). The tests that existed for the batch path drove in-test stubs of {@code
 * RerankingModel} and never reached this class, and the one that did was integration-tagged and not
 * run by CI. This one is unit-tagged and builds its own weights, so the path has a test that runs.
 */
@Tag("unit")
class BertForwardPassRaggedBatchTest {

  private static final int DIM = 16;
  private static final int HEADS = 2;
  private static final int LAYERS = 2;
  private static final int VOCAB = 32;
  private static final int CONTEXT = 16;
  private static final int HIDDEN = 32;

  private static BertConfig config() {
    return new BertConfig(
        "bert",
        BertConfig.EncoderStyle.BERT,
        DIM,
        LAYERS,
        HEADS,
        VOCAB,
        CONTEXT,
        HIDDEN,
        1.0e-5f,
        0.0f,
        BertConfig.Pooling.RANK);
  }

  private static BertWeights.Matrix matrix(Random random, int rows, int columns, float scale) {
    float[] values = new float[rows * columns];
    for (int i = 0; i < values.length; i++) {
      values[i] = (random.nextFloat() * 2 - 1) * scale;
    }
    return new BertWeights.Matrix(MemorySegment.ofArray(values), GgufTensorType.F32, rows, columns);
  }

  private static float[] vector(Random random, int size, float scale, float offset) {
    float[] values = new float[size];
    for (int i = 0; i < size; i++) {
      values[i] = offset + (random.nextFloat() * 2 - 1) * scale;
    }
    return values;
  }

  /** Small random weights of the shapes BertWeights.load produces for a bert-style model. */
  private static BertWeights weights(long seed) {
    Random random = new Random(seed);
    BertWeights.Layer[] layers = new BertWeights.Layer[LAYERS];
    for (int layer = 0; layer < LAYERS; layer++) {
      layers[layer] =
          new BertWeights.Layer(
              matrix(random, DIM, DIM, 0.2f),
              vector(random, DIM, 0.05f, 0f),
              matrix(random, DIM, DIM, 0.2f),
              vector(random, DIM, 0.05f, 0f),
              matrix(random, DIM, DIM, 0.2f),
              vector(random, DIM, 0.05f, 0f),
              matrix(random, DIM, DIM, 0.2f),
              vector(random, DIM, 0.05f, 0f),
              vector(random, DIM, 0.05f, 1f),
              vector(random, DIM, 0.05f, 0f),
              matrix(random, HIDDEN, DIM, 0.2f),
              vector(random, HIDDEN, 0.05f, 0f),
              matrix(random, DIM, HIDDEN, 0.2f),
              vector(random, DIM, 0.05f, 0f),
              vector(random, DIM, 0.05f, 1f),
              vector(random, DIM, 0.05f, 0f),
              null,
              null);
    }
    return new BertWeights(
        matrix(random, VOCAB, DIM, 0.5f),
        matrix(random, CONTEXT, DIM, 0.5f),
        matrix(random, 2, DIM, 0.5f),
        vector(random, DIM, 0.05f, 1f),
        vector(random, DIM, 0.05f, 0f),
        layers);
  }

  private static int[] tokens(Random random, int length) {
    int[] tokens = new int[length];
    for (int i = 0; i < length; i++) {
      tokens[i] = random.nextInt(VOCAB);
    }
    return tokens;
  }

  private static int[] types(int length) {
    int[] types = new int[length];
    for (int i = length / 2; i < length; i++) {
      types[i] = 1;
    }
    return types;
  }

  @Test
  void raggedBatchDoesNotThrowAndMatchesSequentialEncoding() {
    BertForwardPass pass = new BertForwardPass(config(), weights(7L));
    Random random = new Random(11L);
    // Unequal lengths are the whole point: 3 and 5 are padded against the 7, so each has padded
    // query rows, which is exactly the shape that reached softmax with no finite input.
    int[] lengths = {3, 7, 5};
    int batch = lengths.length;
    int widest = 7;
    int[][] tokens = new int[batch][widest];
    int[][] types = new int[batch][widest];
    int[][] exactTokens = new int[batch][];
    int[][] exactTypes = new int[batch][];
    for (int doc = 0; doc < batch; doc++) {
      exactTokens[doc] = tokens(random, lengths[doc]);
      exactTypes[doc] = types(lengths[doc]);
      System.arraycopy(exactTokens[doc], 0, tokens[doc], 0, lengths[doc]);
      System.arraycopy(exactTypes[doc], 0, types[doc], 0, lengths[doc]);
    }

    assertThatCode(() -> pass.encodeBatchCls(tokens, types, lengths)).doesNotThrowAnyException();
    float[][] batched = pass.encodeBatchCls(tokens, types, lengths);

    // The property the batched path was introduced on: each document's CLS equals the one it
    // gets when encoded alone. Padding must change nothing about a document's own result.
    for (int doc = 0; doc < batch; doc++) {
      float[] alone = pass.encodeCls(exactTokens[doc], exactTypes[doc]);
      assertThat(batched[doc]).hasSize(alone.length);
      for (int i = 0; i < alone.length; i++) {
        // Exact, not close: the packed layout runs each row through the same operations as
        // encoding it alone, so any difference is a layout bug, not rounding.
        assertThat(Float.floatToIntBits(batched[doc][i]))
            .describedAs(
                "document %s (length %s) element %s: %s vs %s",
                doc, lengths[doc], i, batched[doc][i], alone[i])
            .isEqualTo(Float.floatToIntBits(alone[i]));
      }
    }
  }

  @Test
  void aBatchOfEqualLengthsStillAgreesWithItself() {
    // The case the old mask handled: no padded rows at all. Kept so the fix cannot regress it.
    BertForwardPass pass = new BertForwardPass(config(), weights(3L));
    Random random = new Random(5L);
    int[][] tokens = {tokens(random, 6), tokens(random, 6)};
    int[][] types = {types(6), types(6)};
    float[][] withLengths = pass.encodeBatchCls(tokens, types, new int[] {6, 6});
    float[][] withoutLengths = pass.encodeBatchCls(tokens, types);
    for (int doc = 0; doc < 2; doc++) {
      assertThat(withLengths[doc]).containsExactly(withoutLengths[doc]);
    }
  }
}
