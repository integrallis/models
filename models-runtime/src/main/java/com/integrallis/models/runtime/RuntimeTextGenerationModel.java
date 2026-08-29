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
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;

/** High-level generation adapter for a pure-Java {@link InferenceBackend}. */
public final class RuntimeTextGenerationModel
    implements ConstrainedTextGenerationModel, AuxiliaryTextGenerationModel {
  private final InferenceBackend backend;
  private final GenerationLoop generationLoop;

  public RuntimeTextGenerationModel(InferenceBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
    this.generationLoop = new GenerationLoop(backend);
  }

  @Override
  public String modelName() {
    return backend.metadata().modelName();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    return backend.diagnostics();
  }

  @Override
  public Tokenizer tokenizer() {
    return backend.tokenizer();
  }

  /** Returns phase timings and usage for the most recently completed generation. */
  public GenerationMetrics lastGenerationMetrics() {
    return generationLoop.lastGenerationMetrics();
  }

  @Override
  public String generate(String prompt, SamplingOptions options) {
    return generationLoop.generate(prompt, options);
  }

  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    generationLoop.generate(prompt, options, stream);
  }

  @Override
  public String generate(ModelPrompt prompt, SamplingOptions options) {
    return generationLoop.generate(prompt, options);
  }

  @Override
  public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    generationLoop.generate(prompt, options, stream);
  }

  @Override
  public void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    generationLoop.generate(prompt, options, stream, constraint);
  }

  @Override
  public boolean supportsContrastiveEncoding() {
    return backend instanceof AuxiliaryInferenceBackend auxiliary
        && auxiliary.supportsContrastiveEncoding();
  }

  @Override
  public int contrastiveDimension() {
    return auxiliaryWithContrastiveHead().contrastiveDimension();
  }

  @Override
  public float[] encodeContrastive(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    synchronized (backend) {
      return auxiliaryWithContrastiveHead().encodeContrastive(backend.tokenizer().encode(prompt));
    }
  }

  @Override
  public boolean supportsConfidenceScoring() {
    return backend instanceof AuxiliaryInferenceBackend auxiliary
        && auxiliary.supportsConfidenceScoring();
  }

  @Override
  public float scoreConfidence(ModelPrompt sequence) {
    Objects.requireNonNull(sequence, "sequence");
    synchronized (backend) {
      return auxiliaryWithConfidenceHead().scoreConfidence(backend.tokenizer().encode(sequence));
    }
  }

  private AuxiliaryInferenceBackend auxiliaryWithContrastiveHead() {
    if (!(backend instanceof AuxiliaryInferenceBackend auxiliary)
        || !auxiliary.supportsContrastiveEncoding()) {
      throw new UnsupportedOperationException("model has no contrastive encoding head");
    }
    return auxiliary;
  }

  private AuxiliaryInferenceBackend auxiliaryWithConfidenceHead() {
    if (!(backend instanceof AuxiliaryInferenceBackend auxiliary)
        || !auxiliary.supportsConfidenceScoring()) {
      throw new UnsupportedOperationException("model has no confidence head");
    }
    return auxiliary;
  }
}
