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

import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.Pooling;
import java.util.List;
import java.util.Objects;

/**
 * Produces sentence embeddings from a GGUF model using the pure-Java forward pass.
 *
 * <p>An embedding is the generative pass with its head removed: the transformer stack runs
 * unchanged, and the vocabulary projection — the widest matmul in the pass — is skipped in favour
 * of the activation it would have consumed. Embedding a token therefore costs strictly less than
 * generating one on the same weights.
 *
 * <p>Pooling and normalization travel with the instance because they are properties of the
 * embedding model. Getting them wrong does not fail; it quietly degrades retrieval, which is a far
 * worse failure mode than an exception.
 *
 * <p>Not thread-safe: each call drives one sequence through shared backend state and resets it
 * between texts. Use one instance per thread, or guard it.
 */
public final class GgufEmbeddingBackend implements EmbeddingBackend {

  private final PureJavaBackend backend;
  private final Pooling pooling;
  private final boolean normalize;
  private final int dimension;
  private boolean closed;

  private GgufEmbeddingBackend(Builder builder) {
    this.backend = builder.backend;
    this.pooling = builder.pooling;
    this.normalize = builder.normalize;
    this.dimension = backend.metadata().embeddingDim();
    if (!backend.supportsHiddenState()) {
      throw new IllegalArgumentException(
          "model architecture "
              + backend.metadata().modelFamily()
              + " does not expose hidden states for embedding");
    }
  }

  /** Starts configuring an embedding backend over an already-loaded model. */
  public static Builder builder(PureJavaBackend backend) {
    return new Builder(backend);
  }

  @Override
  public int dimension() {
    return dimension;
  }

  @Override
  public float[] embed(String text) {
    Objects.requireNonNull(text, "text");
    checkOpen();
    int[] tokens = tokenize(text);
    // Every text is an independent sequence; without this the previous one stays in the KV cache
    // and results depend on call order.
    backend.reset();
    float[] pooled = pooling == Pooling.MEAN ? meanPooled(tokens) : lastTokenPooled(tokens);
    if (normalize) {
      l2Normalize(pooled);
    }
    return pooled;
  }

  @Override
  public float[][] embedAll(List<String> texts) {
    Objects.requireNonNull(texts, "texts");
    float[][] rows = new float[texts.size()][];
    for (int index = 0; index < texts.size(); index++) {
      rows[index] = embed(Objects.requireNonNull(texts.get(index), "texts must not contain null"));
    }
    return rows;
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      backend.close();
    }
  }

  /**
   * Encodes text, substituting the beginning-of-sequence token when it yields nothing.
   *
   * <p>A blank row in a corpus should embed to something stable rather than abort an ingest run.
   */
  private int[] tokenize(String text) {
    int[] tokens = backend.tokenizer().encode(text);
    return tokens.length == 0 ? new int[] {backend.tokenizer().bosToken()} : tokens;
  }

  /** Takes the final position's state: the only one that has attended to the whole input. */
  private float[] lastTokenPooled(int[] tokens) {
    return backend.prefillHiddenState(tokens, 0).clone();
  }

  /** Averages every position's state, which costs one hidden state per token. */
  private float[] meanPooled(int[] tokens) {
    double[] sum = new double[dimension];
    for (int index = 0; index < tokens.length; index++) {
      float[] hidden = backend.hiddenState(tokens[index], index);
      for (int component = 0; component < dimension; component++) {
        sum[component] += hidden[component];
      }
    }
    float[] pooled = new float[dimension];
    for (int component = 0; component < dimension; component++) {
      pooled[component] = (float) (sum[component] / tokens.length);
    }
    return pooled;
  }

  /** Scales to unit length so downstream cosine similarity reduces to a dot product. */
  private static void l2Normalize(float[] vector) {
    double sumOfSquares = 0;
    for (float value : vector) {
      sumOfSquares += (double) value * value;
    }
    double magnitude = Math.sqrt(sumOfSquares);
    if (magnitude == 0.0) {
      // A zero vector has no direction; scaling would divide by zero.
      return;
    }
    for (int index = 0; index < vector.length; index++) {
      vector[index] = (float) (vector[index] / magnitude);
    }
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("embedding backend is closed");
    }
  }

  /** Configures pooling and normalization, which must match how the model was trained. */
  public static final class Builder {

    private final PureJavaBackend backend;
    private Pooling pooling = Pooling.LAST_TOKEN;
    private boolean normalize = true;

    private Builder(PureJavaBackend backend) {
      this.backend = Objects.requireNonNull(backend, "backend");
    }

    /** Defaults to {@link Pooling#LAST_TOKEN}, correct for causal decoder-only embedders. */
    public Builder pooling(Pooling pooling) {
      this.pooling = Objects.requireNonNull(pooling, "pooling");
      return this;
    }

    /** Defaults to true; unit vectors are what retrieval and semantic caching expect. */
    public Builder normalize(boolean normalize) {
      this.normalize = normalize;
      return this;
    }

    public GgufEmbeddingBackend build() {
      return new GgufEmbeddingBackend(this);
    }
  }
}
