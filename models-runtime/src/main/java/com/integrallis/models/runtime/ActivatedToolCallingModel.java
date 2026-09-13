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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
  static final String TOOL_DECISION_PREFIX = "<tool_call>\n";
  private static final String NO_TOOL_OUTPUT = "<tool_call>\n[]\n</tool_call>";

  /** How the base and activated branches obtain the context before the activation boundary. */
  public enum PrefixStrategy {
    /** Select sharing only when the prompt reaches the measured crossover. */
    AUTO,
    /** Evaluate the prefix once and fork both branches over the same physical KV storage. */
    SHARED,
    /** Evaluate the same base prefix independently in both branches without shared KV storage. */
    RECOMPUTED
  }

  private final InferencePipeline pipeline;
  private final ActivatedAdapterMetadata adapter;
  private final int minimumSharedPrefixTokens;
  private final AtomicBoolean closed = new AtomicBoolean();

  /**
   * Creates a model that owns the supplied loaded backend.
   *
   * @param minimumSharedPrefixTokens measured token crossover at which physical sharing is no
   *     slower than independent prefix evaluation
   */
  public ActivatedToolCallingModel(
      SharedPrefixInferenceBackend backend, int minimumSharedPrefixTokens) {
    Objects.requireNonNull(backend, "backend");
    if (minimumSharedPrefixTokens <= 0) {
      throw new IllegalArgumentException("minimumSharedPrefixTokens must be > 0");
    }
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
    this.minimumSharedPrefixTokens = minimumSharedPrefixTokens;
    this.pipeline = new InferencePipeline(backend);
  }

  /** Returns the exact adapter provenance and activation sequence. */
  public ActivatedAdapterMetadata adapter() {
    requireOpen();
    return adapter;
  }

  /** Returns the qualified prefix length at which automatic turns begin physical sharing. */
  public int minimumSharedPrefixTokens() {
    requireOpen();
    return minimumSharedPrefixTokens;
  }

  /** Returns whether this loaded graph performs true ragged multi-session decision prefill. */
  public boolean supportsRaggedDecisionPrefill() {
    requireOpen();
    return pipeline.supportsRaggedSharedPrefill();
  }

  /** Returns the largest decision batch accepted by the loaded graph. */
  public int maximumDecisionBatchSize() {
    requireOpen();
    return pipeline.maximumSharedPrefillBatchSize();
  }

  @Override
  public java.util.List<String> toolAbstentionOutputs() {
    requireOpen();
    return java.util.List.of(NO_TOOL_OUTPUT);
  }

  /** Opens one request-scoped tool turn from a rendered prompt containing the pinned invocation. */
  public ActivatedToolTurn openToolTurn(ModelPrompt renderedToolPrompt) {
    return openToolTurn(renderedToolPrompt, null, PrefixStrategy.AUTO);
  }

  /**
   * Opens one request-scoped turn with an explicit prefix strategy.
   *
   * <p>The forced strategies exist so qualification can measure the real sharing crossover against
   * the same loaded model and adapter. Normal callers should use the automatic path.
   */
  public ActivatedToolTurn openToolTurn(
      ModelPrompt renderedToolPrompt, PrefixStrategy prefixStrategy) {
    requireOpen();
    Objects.requireNonNull(prefixStrategy, "prefixStrategy");
    return openToolTurn(renderedToolPrompt, null, prefixStrategy);
  }

  /**
   * Scores several independent tool prompts using bounded ragged prefill batches.
   *
   * <p>Every prompt is validated before backend state is opened. Each item evaluates its base
   * prefix once, physically forks exact-base and activated-adapter branches from that immutable KV
   * storage, and scores the configured call/no-call tokens after {@value #TOOL_DECISION_PREFIX}.
   * The request batch is split at both {@code maximumBatchSize} and the loaded backend's capacity.
   */
  public List<ActivatedToolDecision> scoreToolDecisions(
      List<ModelPrompt> renderedToolPrompts,
      int callTokenId,
      int noCallTokenId,
      int maximumBatchSize) {
    requireOpen();
    Objects.requireNonNull(renderedToolPrompts, "renderedToolPrompts");
    if (renderedToolPrompts.isEmpty()) {
      throw new IllegalArgumentException("renderedToolPrompts must not be empty");
    }
    if (maximumBatchSize <= 0) {
      throw new IllegalArgumentException("maximumBatchSize must be > 0");
    }
    int vocabularySize = pipeline.tokenizer().vocabSize();
    requireVocabularyToken("callTokenId", callTokenId, vocabularySize);
    requireVocabularyToken("noCallTokenId", noCallTokenId, vocabularySize);
    if (callTokenId == noCallTokenId) {
      throw new IllegalArgumentException("call and no-call token IDs must differ");
    }

    int[] invocation = adapter.invocationTokens().stream().mapToInt(Integer::intValue).toArray();
    List<InferencePipeline.ActivatedDecisionInput> inputs =
        new ArrayList<>(renderedToolPrompts.size());
    for (int index = 0; index < renderedToolPrompts.size(); index++) {
      ModelPrompt prompt =
          Objects.requireNonNull(
              renderedToolPrompts.get(index), "renderedToolPrompts[" + index + "]");
      int[] promptTokens = pipeline.tokenize(prompt);
      int prefixLength = lastIndexOf(promptTokens, invocation);
      if (prefixLength <= 0) {
        throw new IllegalArgumentException(
            "rendered tool prompt at index "
                + index
                + " must contain the activated adapter invocation sequence");
      }
      int[] decisionTokens = pipeline.tokenize(withDecisionPrefix(prompt));
      requireTokenPrefix(promptTokens, decisionTokens, index);
      inputs.add(
          new InferencePipeline.ActivatedDecisionInput(
              promptTokens.length,
              Arrays.copyOf(promptTokens, prefixLength),
              Arrays.copyOfRange(decisionTokens, prefixLength, decisionTokens.length)));
    }
    return pipeline.scoreSharedActivatedDecisions(
        inputs, callTokenId, noCallTokenId, maximumBatchSize);
  }

  /** Opens one stateful base/tool conversation that retains one physical cache lineage. */
  public ActivatedToolConversation openConversation() {
    requireOpen();
    return new ActivatedToolConversation(this, pipeline);
  }

  ActivatedToolTurn openAutomaticToolTurn(
      ModelPrompt renderedToolPrompt, TextGenerationSession retainedBaseSession) {
    return openToolTurn(renderedToolPrompt, retainedBaseSession, PrefixStrategy.AUTO);
  }

  private ActivatedToolTurn openToolTurn(
      ModelPrompt renderedToolPrompt,
      TextGenerationSession retainedBaseSession,
      PrefixStrategy prefixStrategy) {
    requireOpen();
    Objects.requireNonNull(renderedToolPrompt, "renderedToolPrompt");
    Objects.requireNonNull(prefixStrategy, "prefixStrategy");
    int[] promptTokens = pipeline.tokenize(renderedToolPrompt);
    int[] invocation = adapter.invocationTokens().stream().mapToInt(Integer::intValue).toArray();
    int prefixLength = lastIndexOf(promptTokens, invocation);
    if (prefixLength <= 0) {
      throw new IllegalArgumentException(
          "rendered tool prompt must contain the activated adapter invocation sequence");
    }

    int[] prefixTokens = java.util.Arrays.copyOf(promptTokens, prefixLength);
    if (shouldShare(prefixLength, prefixStrategy)) {
      return openSharedToolTurn(
          renderedToolPrompt, retainedBaseSession, prefixStrategy, invocation, prefixTokens);
    }
    return openRecomputedToolTurn(
        renderedToolPrompt, retainedBaseSession, prefixStrategy, invocation, prefixTokens);
  }

  private ActivatedToolTurn openSharedToolTurn(
      ModelPrompt renderedToolPrompt,
      TextGenerationSession retainedBaseSession,
      PrefixStrategy prefixStrategy,
      int[] invocation,
      int[] prefixTokens) {
    SharedPromptPrefix prefix =
        retainedBaseSession == null
            ? pipeline.prepareSharedTokenPrefix(prefixTokens)
            : pipeline.extendSharedTokenPrefix(retainedBaseSession, prefixTokens);
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
          physicallyShared,
          prefixStrategy,
          minimumSharedPrefixTokens);
    } catch (RuntimeException | Error failure) {
      closeAfterFailure(failure, tool, base);
      throw failure;
    }
  }

  private ActivatedToolTurn openRecomputedToolTurn(
      ModelPrompt renderedToolPrompt,
      TextGenerationSession retainedBaseSession,
      PrefixStrategy prefixStrategy,
      int[] invocation,
      int[] prefixTokens) {
    TextGenerationSession base = null;
    TextGenerationSession tool = null;
    try {
      if (retainedBaseSession == null) {
        base =
            pipeline.openGenerationSessionAfterBasePrefix(
                prefixTokens, SharedPrefixInferenceBackend.Branch.BASE);
      } else {
        retainedBaseSession.prefillTokenPrefix(prefixTokens);
        base = retainedBaseSession;
      }
      tool =
          pipeline.openGenerationSessionAfterBasePrefix(
              prefixTokens, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER);
      if (base.sharesPrefixStorageWith(tool)) {
        throw new IllegalStateException(
            "recomputed branches unexpectedly share physical KV storage");
      }
      return new ActivatedToolTurn(
          pipeline,
          invocation,
          renderedToolPrompt,
          prefixTokens.length,
          0,
          0,
          base,
          tool,
          false,
          prefixStrategy,
          minimumSharedPrefixTokens);
    } catch (RuntimeException | Error failure) {
      closeAfterFailure(failure, tool, base);
      throw failure;
    }
  }

  static void closeAfterFailure(Throwable primary, TextGenerationSession... sessions) {
    for (TextGenerationSession session : sessions) {
      if (session == null) {
        continue;
      }
      try {
        session.close();
      } catch (RuntimeException | Error closeFailure) {
        primary.addSuppressed(closeFailure);
      }
    }
  }

  static boolean shouldShare(
      int prefixTokens, PrefixStrategy prefixStrategy, int minimumSharedPrefixTokens) {
    return prefixStrategy == PrefixStrategy.SHARED
        || (prefixStrategy == PrefixStrategy.AUTO && prefixTokens >= minimumSharedPrefixTokens);
  }

  private boolean shouldShare(int prefixTokens, PrefixStrategy prefixStrategy) {
    return shouldShare(prefixTokens, prefixStrategy, minimumSharedPrefixTokens);
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

  private static ModelPrompt withDecisionPrefix(ModelPrompt renderedToolPrompt) {
    ModelPrompt.Builder prompt = ModelPrompt.builder();
    for (ModelPrompt.Segment segment : renderedToolPrompt.segments()) {
      if (segment.kind() == ModelPrompt.SegmentKind.CONTROL) {
        prompt.control(segment.text());
      } else {
        prompt.text(segment.text());
      }
    }
    return prompt.control(TOOL_DECISION_PREFIX).build();
  }

  private static void requireTokenPrefix(int[] expected, int[] actual, int promptIndex) {
    if (actual.length <= expected.length) {
      throw new IllegalStateException(
          "tool decision prefix did not add tokens for prompt at index " + promptIndex);
    }
    for (int index = 0; index < expected.length; index++) {
      if (expected[index] != actual[index]) {
        throw new IllegalStateException(
            "tool decision prefix changed the rendered prompt at token "
                + index
                + " for item "
                + promptIndex);
      }
    }
  }

  private static void requireVocabularyToken(String name, int tokenId, int vocabularySize) {
    if (tokenId < 0 || tokenId >= vocabularySize) {
      throw new IllegalArgumentException(
          name + " must be within the loaded vocabulary: " + tokenId + " of " + vocabularySize);
    }
  }

  private void requireOpen() {
    if (closed.get()) {
      throw new IllegalStateException("activated tool-calling model is closed");
    }
  }
}
