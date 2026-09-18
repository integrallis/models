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
package com.integrallis.models.bench.fusion;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.runtime.InferencePipeline;
import java.util.Objects;

/**
 * A member over the public {@link InferencePipeline} ({@code prefill}, {@code forward}, {@code
 * checkpoint}, {@code rewind}). Multi-position scoring uses the backend's speculative-verification
 * capability when present, because the pipeline does not expose per-position prefill logits.
 */
final class PipelineMember implements FusionMember {
  private final String name;
  private final InferencePipeline pipeline;
  private final InferenceBackend backend;
  private final Tokenizer tokenizer;

  PipelineMember(String name, InferenceBackend backend) {
    this.name = Objects.requireNonNull(name, "name");
    this.backend = Objects.requireNonNull(backend, "backend");
    this.pipeline = new InferencePipeline(backend);
    this.tokenizer = pipeline.tokenizer();
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public Tokenizer tokenizer() {
    return tokenizer;
  }

  InferencePipeline pipeline() {
    return pipeline;
  }

  @Override
  public void reset() {
    pipeline.resetContext();
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return pipeline.prefill(tokens, startPosition);
  }

  @Override
  public float[] forward(int token, int position) {
    return pipeline.forward(token, position);
  }

  @Override
  public int checkpoint() {
    return pipeline.checkpoint();
  }

  @Override
  public void rewind(int checkpoint) {
    pipeline.rewind(checkpoint);
  }

  @Override
  public float[][] verify(int[] tokens, int startPosition) {
    if (!(backend instanceof SpeculativeInferenceBackend speculative)) {
      return null;
    }
    synchronized (backend) {
      LogitBatch batch = speculative.verify(tokens, startPosition);
      float[][] rows = new float[batch.tokenCount()][];
      for (int i = 0; i < rows.length; i++) {
        rows[i] = batch.copyRow(i);
      }
      return rows;
    }
  }

  @Override
  public void close() {
    pipeline.close();
  }
}
