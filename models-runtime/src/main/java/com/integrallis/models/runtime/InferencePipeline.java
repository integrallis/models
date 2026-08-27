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
package com.integrallis.models.runtime;

import com.integrallis.models.api.AuxiliaryInferenceBackend;
import com.integrallis.models.api.AuxiliaryTextGenerationModel;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.InferenceContextWindow;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owning, state-coordinated access to a loaded model's complete inference pipeline.
 *
 * <p>The pipeline supports the high-level {@link TextGenerationModel} contract and controlled
 * low-level access to tokenization, context state, prefill, forward-pass logits, checkpointing, and
 * rewind. Calls that mutate inference state are serialized on the backend. A direct context
 * operation invalidates the high-level prompt-prefix cache before changing backend state.
 *
 * <p>The pipeline owns the backend supplied to {@link #InferencePipeline(InferenceBackend)} and
 * closes it exactly once. It must not be used after {@link #close()}.
 */
public final class InferencePipeline
    implements ConstrainedTextGenerationModel, AuxiliaryTextGenerationModel {
  private final InferenceBackend backend;
  private final GenerationLoop generationLoop;
  private final AtomicBoolean closed = new AtomicBoolean();

  /** Creates a pipeline that owns the supplied loaded backend. */
  public InferencePipeline(InferenceBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
    generationLoop = new GenerationLoop(backend);
  }

  /** Returns immutable architecture metadata for the loaded model. */
  public ModelMetadata metadata() {
    synchronized (backend) {
      requireOpen();
      return backend.metadata();
    }
  }

  /** Returns the loaded model's read-only, thread-safe tokenizer. */
  public Tokenizer tokenizer() {
    synchronized (backend) {
      requireOpen();
      return backend.tokenizer();
    }
  }

  /** Returns the active context capacity and, when supported, its next token position. */
  public InferenceContextWindow contextWindow() {
    synchronized (backend) {
      requireOpen();
      OptionalInt position =
          backend instanceof RewindableInferenceBackend rewindable
              ? OptionalInt.of(rewindable.checkpoint())
              : OptionalInt.empty();
      return new InferenceContextWindow(backend.contextCapacity(), position);
    }
  }

  /** Encodes ordinary text without recognizing text that merely spells a special token. */
  public int[] tokenize(String text) {
    return tokenizer().encode(Objects.requireNonNull(text, "text"));
  }

  /** Encodes a segmented prompt while recognizing trusted template-control segments. */
  public int[] tokenize(ModelPrompt prompt) {
    return tokenizer().encode(Objects.requireNonNull(prompt, "prompt"));
  }

  /** Tokenizes and prefills a structured prompt at the requested context position. */
  public float[] prefill(ModelPrompt prompt, int startPosition) {
    Objects.requireNonNull(prompt, "prompt");
    return prefill(tokenize(prompt), startPosition);
  }

  /** Prefills a contiguous token sequence and returns stable logits for its final token. */
  public float[] prefill(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    synchronized (backend) {
      requireOpen();
      generationLoop.invalidatePromptCache();
      return backend.prefill(tokens, startPosition).clone();
    }
  }

  /** Runs one token at an explicit position and returns stable logits. */
  public float[] forward(int token, int position) {
    synchronized (backend) {
      requireOpen();
      generationLoop.invalidatePromptCache();
      return backend.forward(token, position);
    }
  }

  /** Clears the active context so the next token position is zero. */
  public void resetContext() {
    synchronized (backend) {
      requireOpen();
      generationLoop.invalidatePromptCache();
      backend.reset();
    }
  }

  /** Returns whether this loaded backend supports checkpoint and rewind. */
  public boolean rewindable() {
    synchronized (backend) {
      requireOpen();
      return backend instanceof RewindableInferenceBackend;
    }
  }

  /** Returns the next token position for a rewindable context. */
  public int checkpoint() {
    synchronized (backend) {
      requireOpen();
      return rewindableBackend().checkpoint();
    }
  }

  /** Discards context state at and after a prior checkpoint. */
  public void rewind(int checkpoint) {
    synchronized (backend) {
      requireOpen();
      generationLoop.invalidatePromptCache();
      rewindableBackend().rewind(checkpoint);
    }
  }

  @Override
  public String modelName() {
    return metadata().modelName();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    synchronized (backend) {
      requireOpen();
      return backend.diagnostics();
    }
  }

  @Override
  public String generate(String prompt, SamplingOptions options) {
    synchronized (backend) {
      requireOpen();
      return generationLoop.generate(prompt, options);
    }
  }

  @Override
  public String generate(ModelPrompt prompt, SamplingOptions options) {
    synchronized (backend) {
      requireOpen();
      return generationLoop.generate(prompt, options);
    }
  }

  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    synchronized (backend) {
      requireOpen();
      generationLoop.generate(prompt, options, stream);
    }
  }

  @Override
  public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    synchronized (backend) {
      requireOpen();
      generationLoop.generate(prompt, options, stream);
    }
  }

  @Override
  public void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    synchronized (backend) {
      requireOpen();
      generationLoop.generate(prompt, options, stream, constraint);
    }
  }

  @Override
  public boolean supportsContrastiveEncoding() {
    synchronized (backend) {
      requireOpen();
      return backend instanceof AuxiliaryInferenceBackend auxiliary
          && auxiliary.supportsContrastiveEncoding();
    }
  }

  @Override
  public int contrastiveDimension() {
    synchronized (backend) {
      requireOpen();
      return auxiliaryWithContrastiveHead().contrastiveDimension();
    }
  }

  @Override
  public float[] encodeContrastive(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    synchronized (backend) {
      requireOpen();
      return auxiliaryWithContrastiveHead().encodeContrastive(backend.tokenizer().encode(prompt));
    }
  }

  @Override
  public boolean supportsConfidenceScoring() {
    synchronized (backend) {
      requireOpen();
      return backend instanceof AuxiliaryInferenceBackend auxiliary
          && auxiliary.supportsConfidenceScoring();
    }
  }

  @Override
  public float scoreConfidence(ModelPrompt sequence) {
    Objects.requireNonNull(sequence, "sequence");
    synchronized (backend) {
      requireOpen();
      return auxiliaryWithConfidenceHead().scoreConfidence(backend.tokenizer().encode(sequence));
    }
  }

  /** Closes the owned backend exactly once. */
  @Override
  public void close() {
    synchronized (backend) {
      if (closed.compareAndSet(false, true)) {
        generationLoop.invalidatePromptCache();
        backend.close();
      }
    }
  }

  private RewindableInferenceBackend rewindableBackend() {
    if (backend instanceof RewindableInferenceBackend rewindable) {
      return rewindable;
    }
    throw new UnsupportedOperationException(
        "Backend " + backend.name() + " does not support checkpoint and rewind");
  }

  private AuxiliaryInferenceBackend auxiliaryWithContrastiveHead() {
    if (backend instanceof AuxiliaryInferenceBackend auxiliary
        && auxiliary.supportsContrastiveEncoding()) {
      return auxiliary;
    }
    throw new UnsupportedOperationException("model has no contrastive encoding head");
  }

  private AuxiliaryInferenceBackend auxiliaryWithConfidenceHead() {
    if (backend instanceof AuxiliaryInferenceBackend auxiliary
        && auxiliary.supportsConfidenceScoring()) {
      return auxiliary;
    }
    throw new UnsupportedOperationException("model has no confidence head");
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("inference pipeline is closed");
    }
  }
}
