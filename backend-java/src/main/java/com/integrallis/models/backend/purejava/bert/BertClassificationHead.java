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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorData;
import com.integrallis.models.backend.purejava.gguf.GgufTensorValues;
import java.util.Arrays;
import java.util.Objects;

/** Dense-tanh-output classification pooler used by corrected BERT cross-encoder GGUFs. */
public final class BertClassificationHead {

  private final int inputWidth;
  private final int hiddenWidth;
  private final float[] denseWeight;
  private final float[] denseBias;
  private final float[] outputWeight;
  private final float outputBias;
  private final float[] hidden;

  private BertClassificationHead(
      int inputWidth,
      int hiddenWidth,
      float[] denseWeight,
      float[] denseBias,
      float[] outputWeight,
      float outputBias) {
    this.inputWidth = inputWidth;
    this.hiddenWidth = hiddenWidth;
    this.denseWeight = denseWeight;
    this.denseBias = denseBias;
    this.outputWeight = outputWeight;
    this.outputBias = outputBias;
    this.hidden = new float[hiddenWidth];
  }

  /** Loads and validates the corrected MiniLM classifier tensor contract. */
  public static BertClassificationHead fromGgufFile(GgufFile file, int inputWidth) {
    Objects.requireNonNull(file, "file");
    if (inputWidth <= 0) {
      throw new IllegalArgumentException("inputWidth must be > 0: " + inputWidth);
    }
    GgufTensorData dense = required(file, "classifier.dense.weight");
    long[] denseShape = dense.shape();
    if (denseShape.length != 2 || denseShape[0] != inputWidth) {
      throw new IllegalArgumentException(
          "classifier.dense.weight shape must be ["
              + inputWidth
              + ", hidden]: "
              + Arrays.toString(denseShape));
    }
    int hiddenWidth = Math.toIntExact(denseShape[1]);
    GgufTensorData output = required(file, "classifier.out_proj.weight");
    long[] outputShape = output.shape();
    boolean scalarProjection =
        Arrays.equals(outputShape, new long[] {hiddenWidth})
            || Arrays.equals(outputShape, new long[] {hiddenWidth, 1});
    if (!scalarProjection) {
      throw new IllegalArgumentException(
          "classifier.out_proj.weight shape must be ["
              + hiddenWidth
              + "] or ["
              + hiddenWidth
              + ", 1]: "
              + Arrays.toString(outputShape));
    }
    float[] outputBias = vector(file, "classifier.out_proj.bias", 1);
    return new BertClassificationHead(
        inputWidth,
        hiddenWidth,
        GgufTensorValues.toFloatArray(dense),
        vector(file, "classifier.dense.bias", hiddenWidth),
        GgufTensorValues.toFloatArray(output),
        outputBias[0]);
  }

  /** Applies dense + tanh + scalar output projection to the CLS hidden state. */
  public synchronized double score(float[] cls) {
    Objects.requireNonNull(cls, "cls");
    if (cls.length != inputWidth) {
      throw new IllegalArgumentException("CLS width must be " + inputWidth + ": " + cls.length);
    }
    for (int row = 0; row < hiddenWidth; row++) {
      int weightOffset = row * inputWidth;
      float sum = denseBias[row];
      for (int column = 0; column < inputWidth; column++) {
        sum += denseWeight[weightOffset + column] * cls[column];
      }
      hidden[row] = (float) Math.tanh(sum);
    }
    float score = outputBias;
    for (int index = 0; index < hiddenWidth; index++) {
      score += outputWeight[index] * hidden[index];
    }
    return score;
  }

  private static GgufTensorData required(GgufFile file, String name) {
    try {
      return file.getTensor(name);
    } catch (IllegalArgumentException missing) {
      throw new IllegalArgumentException(
          "GGUF reranker is missing "
              + name
              + "; use a corrected conversion that retains the trained classifier pooler",
          missing);
    }
  }

  private static float[] vector(GgufFile file, String name, int size) {
    GgufTensorData tensor = required(file, name);
    if (!Arrays.equals(tensor.shape(), new long[] {size})) {
      throw new IllegalArgumentException(
          name + " shape must be [" + size + "]: " + Arrays.toString(tensor.shape()));
    }
    return GgufTensorValues.toFloatArray(tensor);
  }
}
