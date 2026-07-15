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

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;

/** Separates prompt forward-pass time from autoregressive decode time without changing runtime. */
final class TimingBackend implements InferenceBackend {

  private final InferenceBackend delegate;
  private int inputTokens;
  private long prefillNanos;

  TimingBackend(InferenceBackend delegate) {
    this.delegate = Objects.requireNonNull(delegate, "delegate");
  }

  void beginTrial(int inputTokens) {
    this.inputTokens = inputTokens;
    prefillNanos = 0;
  }

  long prefillNanos() {
    return prefillNanos;
  }

  @Override
  public String name() {
    return delegate.name();
  }

  @Override
  public ModelMetadata metadata() {
    return delegate.metadata();
  }

  @Override
  public Tokenizer tokenizer() {
    return delegate.tokenizer();
  }

  @Override
  public float[] forward(int token, int position) {
    long start = System.nanoTime();
    float[] logits = delegate.forward(token, position);
    recordForward(position, System.nanoTime() - start);
    return logits;
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    long start = System.nanoTime();
    float[] logits = delegate.forwardTransient(token, position);
    recordForward(position, System.nanoTime() - start);
    return logits;
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    long start = System.nanoTime();
    float[] logits = delegate.prefill(tokens, startPosition);
    prefillNanos += System.nanoTime() - start;
    return logits;
  }

  private void recordForward(int position, long elapsed) {
    if (position < inputTokens) {
      prefillNanos += elapsed;
    }
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
