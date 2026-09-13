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

import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * One base model plus a pinned Activated-LoRA tool specialist sharing the same loaded weights.
 *
 * <p>Ordinary generation is exact base-model generation. {@link #openToolTurn} splits a rendered
 * tool prompt immediately before the adapter's pinned invocation sequence, evaluates the long
 * prefix once, and opens an activated tool-selection branch and an exact base response branch over
 * the same physical KV storage. Template-control tokens after the invocation remain on the
 * activated branch.
 */
public final class ActivatedToolCallingModel implements ActivatedToolModel {
  private static final String NO_TOOL_OUTPUT = "<tool_call>\n[]\n</tool_call>";

  /** How the base and activated branches obtain the context before the activation boundary. */
  public enum PrefixStrategy {
    /** Evaluate the prefix once and fork both branches over the same physical KV storage. */
    SHARED,
    /** Evaluate the same base prefix independently in both branches without shared KV storage. */
    RECOMPUTED
  }

  private final InferencePipeline pipeline;
  private final ActivatedAdapterMetadata adapter;
  private final AtomicBoolean closed = new AtomicBoolean();

  /** Creates a model that owns the supplied loaded backend. */
  public ActivatedToolCallingModel(SharedPrefixInferenceBackend backend) {
    Objects.requireNonNull(backend, "backend");
    if (!backend.supportsActivatedBranch()) {
      throw new IllegalArgumentException("backend has no activated-adapter branch");
    }
    this.adapter =
        backend
            .activatedAdapter()
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "backend did not expose its activated-adapter metadata"));
    this.pipeline = new InferencePipeline(backend);
  }

  /** Returns the exact adapter provenance and activation sequence. */
  public ActivatedAdapterMetadata adapter() {
    requireOpen();
    return adapter;
  }

  @Override
  public java.util.List<String> toolAbstentionOutputs() {
    requireOpen();
    return java.util.List.of(NO_TOOL_OUTPUT);
  }

  /** Opens one request-scoped tool turn from a rendered prompt containing the pinned invocation. */
  public ActivatedToolTurn openToolTurn(ModelPrompt renderedToolPrompt) {
    return openSharedToolTurn(renderedToolPrompt, null);
  }

  /**
   * Opens one request-scoped turn with an explicit prefix strategy.
   *
   * <p>{@link PrefixStrategy#RECOMPUTED} exists so qualification can measure the real sharing
   * crossover against the same loaded model and adapter. Normal callers should use the default
   * shared path.
   */
  public ActivatedToolTurn openToolTurn(
      ModelPrompt renderedToolPrompt, PrefixStrategy prefixStrategy) {
    requireOpen();
    Objects.requireNonNull(prefixStrategy, "prefixStrategy");
    if (prefixStrategy == PrefixStrategy.SHARED) {
      return openSharedToolTurn(renderedToolPrompt, null);
    }
    return openRecomputedToolTurn(renderedToolPrompt);
  }

  /** Opens one stateful base/tool conversation that retains one physical cache lineage. */
  public ActivatedToolConversation openConversation() {
    requireOpen();
    return new ActivatedToolConversation(this, pipeline);
  }

  ActivatedToolTurn openSharedToolTurn(
      ModelPrompt renderedToolPrompt, TextGenerationSession retainedBaseSession) {
    requireOpen();
    Objects.requireNonNull(renderedToolPrompt, "renderedToolPrompt");
    int[] promptTokens = pipeline.tokenize(renderedToolPrompt);
    int[] invocation = adapter.invocationTokens().stream().mapToInt(Integer::intValue).toArray();
    int prefixLength = lastIndexOf(promptTokens, invocation);
    if (prefixLength <= 0) {
      throw new IllegalArgumentException(
          "rendered tool prompt must contain the activated adapter invocation sequence");
    }

    int[] sharedTokens = java.util.Arrays.copyOf(promptTokens, prefixLength);
    SharedPromptPrefix prefix =
        retainedBaseSession == null
            ? pipeline.prepareSharedTokenPrefix(sharedTokens)
            : pipeline.extendSharedTokenPrefix(retainedBaseSession, sharedTokens);
    TextGenerationSession base = null;
    TextGenerationSession tool = null;
    try {
      base = pipeline.openGenerationSession(prefix, SharedPrefixInferenceBackend.Branch.BASE);
      tool =
          pipeline.openGenerationSession(
              prefix, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER);
      boolean physicallyShared = base.sharesPrefixStorageWith(tool);
      if (!physicallyShared) {
        throw new IllegalStateException(
            "backend violated the shared-prefix contract: branches do not share KV storage");
      }
      return new ActivatedToolTurn(
          pipeline,
          invocation,
          renderedToolPrompt,
          prefix.tokenCount(),
          prefix.tokenCount(),
          prefix.sharedBytes(),
          base,
          tool,
          physicallyShared);
    } catch (RuntimeException | Error failure) {
      if (tool != null) {
        tool.close();
      }
      if (base != null) {
        base.close();
      }
      throw failure;
    }
  }

  private ActivatedToolTurn openRecomputedToolTurn(ModelPrompt renderedToolPrompt) {
    Objects.requireNonNull(renderedToolPrompt, "renderedToolPrompt");
    int[] promptTokens = pipeline.tokenize(renderedToolPrompt);
    int[] invocation = adapter.invocationTokens().stream().mapToInt(Integer::intValue).toArray();
    int prefixLength = lastIndexOf(promptTokens, invocation);
    if (prefixLength <= 0) {
      throw new IllegalArgumentException(
          "rendered tool prompt must contain the activated adapter invocation sequence");
    }

    int[] prefixTokens = java.util.Arrays.copyOf(promptTokens, prefixLength);
    TextGenerationSession base = null;
    TextGenerationSession tool = null;
    try {
      base =
          pipeline.openGenerationSessionAfterBasePrefix(
              prefixTokens, SharedPrefixInferenceBackend.Branch.BASE);
      tool =
          pipeline.openGenerationSessionAfterBasePrefix(
              prefixTokens, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER);
      if (base.sharesPrefixStorageWith(tool)) {
        throw new IllegalStateException(
            "recomputed branches unexpectedly share physical KV storage");
      }
      return new ActivatedToolTurn(
          pipeline, invocation, renderedToolPrompt, prefixLength, 0, 0, base, tool, false);
    } catch (RuntimeException | Error failure) {
      if (tool != null) {
        tool.close();
      }
      if (base != null) {
        base.close();
      }
      throw failure;
    }
  }

  @Override
  public String modelName() {
    requireOpen();
    return pipeline.modelName();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    requireOpen();
    return pipeline.diagnostics();
  }

  @Override
  public Tokenizer tokenizer() {
    requireOpen();
    return pipeline.tokenizer();
  }

  @Override
  public String generate(String prompt, SamplingOptions options) {
    requireOpen();
    return pipeline.generate(prompt, options);
  }

  @Override
  public String generate(ModelPrompt prompt, SamplingOptions options) {
    requireOpen();
    return pipeline.generate(prompt, options);
  }

  @Override
  public void generate(String prompt, SamplingOptions options, TokenStream stream) {
    requireOpen();
    pipeline.generate(prompt, options, stream);
  }

  @Override
  public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
    requireOpen();
    pipeline.generate(prompt, options, stream);
  }

  @Override
  public void generate(
      ModelPrompt prompt, SamplingOptions options, TokenStream stream, TokenConstraint constraint) {
    requireOpen();
    pipeline.generate(prompt, options, stream, constraint);
  }

  @Override
  public void close() {
    if (closed.compareAndSet(false, true)) {
      pipeline.close();
    }
  }

  static int lastIndexOf(int[] tokens, int[] sequence) {
    for (int offset = tokens.length - sequence.length; offset >= 0; offset--) {
      boolean match = true;
      for (int index = 0; index < sequence.length; index++) {
        if (tokens[offset + index] != sequence[index]) {
          match = false;
          break;
        }
      }
      if (match) {
        return offset;
      }
    }
    return -1;
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("activated tool-calling model is closed");
    }
  }
}
