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
package com.integrallis.models.backend.onnx;

import java.util.Map;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Where a {@code RandomNormalLike} node gets its standard-normal draw.
 *
 * <p>The default is a seeded generator, so a given seed reproduces a given waveform. {@link
 * #injected(Map)} replaces the draw of a named node output with a tensor supplied by the caller,
 * which is how parity against the publisher's runtime is made a defined question: the oracle run
 * exposes the noise it drew, and this executor consumes that same tensor.
 */
@FunctionalInterface
public interface NoiseSource {

  /**
   * A tensor of standard-normal samples with the given shape for the node producing {@code
   * outputName}, scaled by the node's {@code mean} and {@code scale} attributes by the caller.
   */
  Tensor sample(String outputName, long[] shape);

  /** A seeded source; the same seed yields the same sequence of draws in the same graph. */
  static NoiseSource seeded(long seed) {
    RandomGenerator generator = RandomGeneratorFactory.of("L64X128MixRandom").create(seed);
    return (outputName, shape) -> {
      int count = (int) Tensor.elementCount(shape);
      float[] values = new float[count];
      for (int i = 0; i < count; i++) {
        values[i] = (float) generator.nextGaussian();
      }
      return Tensor.ofFloats(shape, values);
    };
  }

  /**
   * A source that answers named node outputs from {@code tensors} and falls back to {@code
   * fallback} for anything else; a shape mismatch on an injected tensor is an error.
   */
  static NoiseSource injected(Map<String, Tensor> tensors, NoiseSource fallback) {
    Objects.requireNonNull(tensors, "tensors");
    Objects.requireNonNull(fallback, "fallback");
    return (outputName, shape) -> {
      Tensor provided = tensors.get(outputName);
      if (provided == null) {
        return fallback.sample(outputName, shape);
      }
      if (!java.util.Arrays.equals(provided.shape(), shape)) {
        throw new IllegalStateException(
            "injected noise for "
                + outputName
                + " has shape "
                + java.util.Arrays.toString(provided.shape())
                + ", the graph needs "
                + java.util.Arrays.toString(shape));
      }
      return provided;
    };
  }

  /** Injected tensors with no fallback: an unexpected draw fails loudly. */
  static NoiseSource injected(Map<String, Tensor> tensors) {
    return injected(
        tensors,
        (outputName, shape) -> {
          throw new IllegalStateException("no injected noise for " + outputName);
        });
  }
}
