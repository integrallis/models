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
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.InferenceContextWindow;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.RewindableInferenceBackend;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.Tokenizer;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
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
  private final ContinuousBatchingScheduler continuousBatching;
  private final Set<TextGenerationSession> generationSessions =
      Collections.newSetFromMap(new IdentityHashMap<>());
  private final AtomicBoolean closed = new AtomicBoolean();

  /** Creates a pipeline that owns the supplied loaded backend. */
  public InferencePipeline(InferenceBackend backend) {
    this.backend = Objects.requireNonNull(backend, "backend");
    generationLoop = new GenerationLoop(backend);
    continuousBatching = null;
  }

  /**
   * Creates a pipeline whose explicit generation sessions share an opt-in continuous scheduler.
   *
   * <p>The backend's default generation lineage remains available through this pipeline. Calls on
   * that lineage and scheduled session batches serialize on the same loaded backend.
   */
  public InferencePipeline(InferenceBackend backend, ContinuousBatchingOptions batchingOptions) {
    this.backend = Objects.requireNonNull(backend, "backend");
    generationLoop = new GenerationLoop(backend);
    if (!(backend instanceof BatchInferenceBackend batchBackend)) {
      throw new IllegalArgumentException(
          "Backend " + backend.name() + " does not support independent inference sessions");
    }
    continuousBatching =
        new ContinuousBatchingScheduler(
            batchBackend, backend, Objects.requireNonNull(batchingOptions, "batchingOptions"));
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

  /** Returns phase timings and usage for the most recently completed generation. */
  public GenerationMetrics lastGenerationMetrics() {
    synchronized (backend) {
      requireOpen();
      return generationLoop.lastGenerationMetrics();
    }
  }

  /** Prefills this pipeline's prompt/KV lineage without decoding output tokens. */
  public PromptPrefillMetrics prefillPrompt(ModelPrompt prompt) {
    synchronized (backend) {
      requireOpen();
      return generationLoop.prefillPrompt(prompt);
    }
  }

  /** Returns whether this backend can hold more than one independent generation lineage. */
  public boolean supportsGenerationSessions() {
    synchronized (backend) {
      requireOpen();
      return backend instanceof BatchInferenceBackend;
    }
  }

  /** Returns scheduler measurements when this pipeline was created with continuous batching. */
  public Optional<ContinuousBatchingMetrics> continuousBatchingMetrics() {
    requireOpen();
    return continuousBatching == null
        ? Optional.empty()
        : Optional.of(continuousBatching.metrics());
  }

  /**
   * Opens independent prompt/KV state while sharing this pipeline's loaded model weights.
   *
   * <p>Use one session per conversation or other cache lineage. The ordinary pipeline serializes
   * calls because the loaded backend may reuse transient inference scratch. A pipeline constructed
   * with {@link ContinuousBatchingOptions} instead schedules concurrent session calls together.
   */
  public TextGenerationSession openGenerationSession() {
    synchronized (backend) {
      requireOpen();
      if (!(backend instanceof BatchInferenceBackend batchBackend)) {
        throw new UnsupportedOperationException(
            "Backend " + backend.name() + " does not support independent inference sessions");
      }
      InferenceSession state = batchBackend.openSession();
      TextGenerationSession session =
          new TextGenerationSession(
              batchBackend,
              state,
              backend,
              () -> generationSessions.removeIf(TextGenerationSession::isClosed),
              continuousBatching);
      generationSessions.add(session);
      return session;
    }
  }

  /**
   * Evaluates a base-model prefix into independent session storage and then selects the requested
   * branch for subsequent tokens.
   *
   * <p>Unlike {@link #prepareSharedPromptPrefix(ModelPrompt)}, this method deliberately does not
   * share physical KV storage. It is the production recomputation path used below an empirically
   * measured sharing crossover. For an activated branch, the supplied prompt must stop immediately
   * before the adapter's pinned invocation sequence.
   */
  public TextGenerationSession openGenerationSessionAfterBasePrefix(
      ModelPrompt prompt, SharedPrefixInferenceBackend.Branch branch) {
    Objects.requireNonNull(prompt, "prompt");
    return openGenerationSessionAfterBasePrefix(tokenize(prompt), branch);
  }

  /** Opens an independent branch after evaluating trusted base-prefix token IDs. */
  public TextGenerationSession openGenerationSessionAfterBasePrefix(
      int[] tokens, SharedPrefixInferenceBackend.Branch branch) {
    Objects.requireNonNull(tokens, "tokens");
    Objects.requireNonNull(branch, "branch");
    synchronized (backend) {
      requireOpen();
      SharedPrefixInferenceBackend sharing = sharedPrefixBackend();
      if (tokens.length == 0) {
        throw new IllegalArgumentException("prompt produced no tokens");
      }
      int[] trustedTokens = tokens.clone();
      InferenceSession state = sharing.openSession();
      try {
        sharing.prefill(state, trustedTokens, 0);
        if (branch == SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER) {
          if (!sharing.supportsActivatedBranch()) {
            throw new IllegalStateException("loaded model has no activated adapter");
          }
          sharing.activateAdapter(state);
        }
        TextGenerationSession session =
            new TextGenerationSession(
                sharing,
                state,
                backend,
                () -> generationSessions.removeIf(TextGenerationSession::isClosed),
                continuousBatching,
                trustedTokens);
        generationSessions.add(session);
        return session;
      } catch (RuntimeException | Error failure) {
        state.close();
        throw failure;
      }
    }
  }

  /** Returns whether this loaded model can physically share immutable KV-cache prefixes. */
  public boolean supportsSharedPromptPrefixes() {
    synchronized (backend) {
      requireOpen();
      return backend instanceof SharedPrefixInferenceBackend sharing
          && sharing.supportsSharedPrefixes();
    }
  }

  /**
   * Tokenizes and evaluates one nonempty prompt into immutable, physically shared KV storage.
   *
   * <p>The returned handle can open independent base or activated-adapter branches without copying
   * or recomputing these tokens.
   */
  public SharedPromptPrefix prepareSharedPromptPrefix(ModelPrompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    synchronized (backend) {
      requireOpen();
      SharedPrefixInferenceBackend sharing = sharedPrefixBackend();
      return prepareSharedTokenPrefix(sharing, backend.tokenizer().encode(prompt));
    }
  }

  /**
   * Evaluates trusted token IDs into immutable, physically shared KV storage.
   *
   * <p>This lower-level form is useful when a structured prompt must be split immediately before an
   * activated-adapter invocation sequence.
   */
  public SharedPromptPrefix prepareSharedTokenPrefix(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    synchronized (backend) {
      requireOpen();
      return prepareSharedTokenPrefix(sharedPrefixBackend(), tokens.clone());
    }
  }

  /**
   * Extends one owned base branch and consumes it into a new immutable physical prefix.
   *
   * <p>Only tokens after the branch's retained prefix are evaluated. Existing immutable KV blocks
   * remain referenced in place, so repeated tool turns can grow a persistent prefix chain without
   * copying or recomputing earlier context.
   */
  public SharedPromptPrefix extendSharedPromptPrefix(
      TextGenerationSession source, ModelPrompt prompt) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(prompt, "prompt");
    return extendSharedTokenPrefix(source, tokenize(prompt));
  }

  /** Extends and freezes an owned base branch from trusted token IDs. */
  public SharedPromptPrefix extendSharedTokenPrefix(TextGenerationSession source, int[] tokens) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(tokens, "tokens");
    SharedPrefixInferenceBackend sharing;
    int[] trustedTokens;
    synchronized (backend) {
      requireOpen();
      if (!generationSessions.contains(source)) {
        throw new IllegalArgumentException(
            "generation session is closed or belongs to a different pipeline");
      }
      if (tokens.length == 0) {
        throw new IllegalArgumentException("prompt produced no tokens");
      }
      trustedTokens = tokens.clone();
      sharing = sharedPrefixBackend();
    }
    SharedInferencePrefix prefix = source.extendAndFreezeSharedPrefix(sharing, trustedTokens);
    synchronized (backend) {
      requireOpen();
      generationSessions.remove(source);
      return new SharedPromptPrefix(this, prefix, trustedTokens, prefix.sharedBytes());
    }
  }

  /**
   * Reconciles a canonically rerendered conversation and consumes the base branch into a new
   * immutable physical prefix.
   *
   * <p>Chat templates may omit generation-only controls when a completed assistant response is
   * rendered back into history. This operation rewinds only the mutable divergent suffix, evaluates
   * the canonical replacement, and retains every block in the branch's immutable prefix. It fails
   * rather than replacing any immutable shared token.
   */
  public SharedPromptPrefix reconcileSharedPromptPrefix(
      TextGenerationSession source, ModelPrompt prompt) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(prompt, "prompt");
    return reconcileSharedTokenPrefix(source, tokenize(prompt));
  }

  /** Reconciles and freezes an owned base branch from trusted canonical token IDs. */
  public SharedPromptPrefix reconcileSharedTokenPrefix(TextGenerationSession source, int[] tokens) {
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(tokens, "tokens");
    SharedPrefixInferenceBackend sharing;
    int[] trustedTokens;
    synchronized (backend) {
      requireOpen();
      if (!generationSessions.contains(source)) {
        throw new IllegalArgumentException(
            "generation session is closed or belongs to a different pipeline");
      }
      if (tokens.length == 0) {
        throw new IllegalArgumentException("prompt produced no tokens");
      }
      trustedTokens = tokens.clone();
      sharing = sharedPrefixBackend();
    }
    SharedInferencePrefix prefix = source.reconcileAndFreezeSharedPrefix(sharing, trustedTokens);
    synchronized (backend) {
      requireOpen();
      generationSessions.remove(source);
      return new SharedPromptPrefix(this, prefix, trustedTokens, prefix.sharedBytes());
    }
  }

  /** Opens a base-model generation branch from a prepared physical prompt prefix. */
  public TextGenerationSession openGenerationSession(SharedPromptPrefix prefix) {
    return openGenerationSession(prefix, SharedPrefixInferenceBackend.Branch.BASE);
  }

  /** Opens the requested generation branch from a prepared physical prompt prefix. */
  public TextGenerationSession openGenerationSession(
      SharedPromptPrefix prefix, SharedPrefixInferenceBackend.Branch branch) {
    Objects.requireNonNull(prefix, "prefix");
    Objects.requireNonNull(branch, "branch");
    synchronized (backend) {
      requireOpen();
      if (prefix.owner != this) {
        throw new IllegalArgumentException("shared prompt prefix belongs to a different pipeline");
      }
      SharedPrefixInferenceBackend sharing = sharedPrefixBackend();
      InferenceSession state = sharing.fork(prefix.backendPrefix, branch);
      try {
        TextGenerationSession session =
            new TextGenerationSession(
                sharing,
                state,
                backend,
                () -> generationSessions.removeIf(TextGenerationSession::isClosed),
                continuousBatching,
                prefix.promptTokens);
        generationSessions.add(session);
        return session;
      } catch (RuntimeException | Error failure) {
        state.close();
        throw failure;
      }
    }
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
    Set<TextGenerationSession> sessionsToClose;
    synchronized (backend) {
      if (!closed.compareAndSet(false, true)) {
        return;
      }
      sessionsToClose = Set.copyOf(generationSessions);
      generationSessions.clear();
    }

    Throwable closeFailure = null;
    if (continuousBatching != null) {
      closeFailure = closeResource(closeFailure, continuousBatching::close);
    }
    for (TextGenerationSession session : sessionsToClose) {
      closeFailure = closeResource(closeFailure, session::close);
    }
    synchronized (backend) {
      try {
        generationLoop.invalidatePromptCache();
      } catch (RuntimeException | Error failure) {
        closeFailure = combineCloseFailures(closeFailure, failure);
      }
      closeFailure = closeResource(closeFailure, backend::close);
    }
    if (closeFailure instanceof RuntimeException runtimeFailure) {
      throw runtimeFailure;
    }
    if (closeFailure instanceof Error error) {
      throw error;
    }
    if (closeFailure != null) {
      throw new IllegalStateException("failed to close inference pipeline", closeFailure);
    }
  }

  private static Throwable closeResource(Throwable current, Runnable close) {
    try {
      close.run();
      return current;
    } catch (RuntimeException | Error failure) {
      return combineCloseFailures(current, failure);
    }
  }

  private static Throwable combineCloseFailures(Throwable current, Throwable next) {
    if (current == null) {
      return next;
    }
    current.addSuppressed(next);
    return current;
  }

  private RewindableInferenceBackend rewindableBackend() {
    if (backend instanceof RewindableInferenceBackend rewindable) {
      return rewindable;
    }
    throw new UnsupportedOperationException(
        "Backend " + backend.name() + " does not support checkpoint and rewind");
  }

  private SharedPrefixInferenceBackend sharedPrefixBackend() {
    if (backend instanceof SharedPrefixInferenceBackend sharing
        && sharing.supportsSharedPrefixes()) {
      return sharing;
    }
    throw new UnsupportedOperationException(
        "Backend "
            + backend.name()
            + " cannot physically share prompt prefixes for the loaded model graph");
  }

  private SharedPromptPrefix prepareSharedTokenPrefix(
      SharedPrefixInferenceBackend sharing, int[] tokens) {
    if (tokens.length == 0) {
      throw new IllegalArgumentException("prompt produced no tokens");
    }
    InferenceSession source = sharing.openSession();
    try {
      sharing.prefill(source, tokens, 0);
      var prefix = sharing.freezePrefix(source);
      return new SharedPromptPrefix(this, prefix, tokens, prefix.sharedBytes());
    } catch (RuntimeException | Error failure) {
      source.close();
      throw failure;
    }
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
