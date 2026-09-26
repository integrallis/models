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
import com.integrallis.models.api.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Deterministic greedy decoding, shared by the CUDA parity gate (G1) and decode gate (G4).
 *
 * <p>Sampling is argmax with ties broken by the lowest token id. There is no temperature, no top-k
 * and no seed, because there is nothing to seed: G1 compares full token id sequences between two
 * arms, and any sampling that consults a random source would make a divergence ambiguous between
 * "the kernels disagree" and "the streams diverged". The guide's {@code --temperature 0 --top-k 1
 * --seed 42} described this same decision rule through options that would have had no effect; the
 * rule is stated here instead of being configurable.
 *
 * <p>The first token of a sequence is the argmax of the prefill logits, so it costs no decode step.
 * The remaining {@code maxTokens - 1} tokens each cost exactly one single-token forward pass, and
 * those are the steps {@link Sequence#decodeNanos()} times and that the routing counters are asked
 * to attribute. Decode throughput is therefore {@code (maxTokens - 1) / decodeSeconds}, never
 * {@code maxTokens / decodeSeconds}: charging the prefill-produced token to the decode clock would
 * inflate short runs.
 *
 * <p>End-of-generation is recorded but not obeyed. Stopping early would shorten the compared
 * sequence, and a parity gate that compares fewer tokens when the arms agree early is a gate that
 * gets weaker exactly where it should get stronger.
 */
final class GreedyDecode {

  /** The sampling rule, recorded in every report so the comparison is reproducible. */
  static final String SAMPLING_RULE = "greedy-argmax-lowest-index-v1";

  private GreedyDecode() {}

  /** Notified after each generated token so a caller can attribute per-step routing counts. */
  interface StepListener {

    /**
     * Called after token {@code tokenIndex} of prompt {@code promptIndex} has been produced.
     *
     * @param decodeStep whether this token cost a single-token forward pass; false for the first
     *     token of a sequence, which comes from the prefill logits
     */
    void generated(int promptIndex, int tokenIndex, int tokenId, boolean decodeStep);

    /** A listener that records nothing. */
    static StepListener none() {
      return (promptIndex, tokenIndex, tokenId, decodeStep) -> {};
    }
  }

  /** One prompt's generated token id sequence and its timings. */
  record Sequence(
      int promptIndex,
      String promptDigest,
      int promptTokenCount,
      List<Integer> tokenIds,
      long prefillNanos,
      long decodeNanos,
      int endOfGenerationIndex) {

    Sequence {
      tokenIds = List.copyOf(Objects.requireNonNull(tokenIds, "tokenIds"));
      Objects.requireNonNull(promptDigest, "promptDigest");
    }

    /** Single-token forward passes performed, which is one fewer than the tokens generated. */
    int decodeSteps() {
      return Math.max(0, tokenIds.size() - 1);
    }

    /** Prompt tokens ingested per second of prefill. */
    double prefillTokensPerSecond() {
      return prefillNanos <= 0 ? 0.0 : promptTokenCount * 1_000_000_000.0 / prefillNanos;
    }

    /** Tokens produced per second of decode, counting only tokens that cost a forward pass. */
    double decodeTokensPerSecond() {
      return decodeNanos <= 0 ? 0.0 : decodeSteps() * 1_000_000_000.0 / decodeNanos;
    }

    /** Whether the model emitted an end-of-generation token that decoding deliberately ignored. */
    boolean hitEndOfGeneration() {
      return endOfGenerationIndex >= 0;
    }
  }

  /**
   * Reads the campaign prompt file.
   *
   * <p>Blank lines and lines starting with {@code #} are ignored, so the file can carry its own
   * provenance without a second file to keep in step with it. Order is the file's order, which is
   * what makes a rerun comparable.
   */
  static List<String> readPrompts(Path path) throws IOException {
    List<String> prompts = new ArrayList<>();
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      String trimmed = line.strip();
      if (trimmed.isEmpty() || trimmed.startsWith("#")) {
        continue;
      }
      prompts.add(trimmed);
    }
    if (prompts.isEmpty()) {
      throw new IllegalArgumentException("no prompts in " + path);
    }
    return List.copyOf(prompts);
  }

  /** Index of the largest logit, ties broken by the lowest index. */
  static int argmax(float[] logits) {
    Objects.requireNonNull(logits, "logits");
    if (logits.length == 0) {
      throw new IllegalArgumentException("logits must not be empty");
    }
    int best = 0;
    float bestValue = logits[0];
    for (int index = 1; index < logits.length; index++) {
      if (logits[index] > bestValue) {
        bestValue = logits[index];
        best = index;
      }
    }
    return best;
  }

  /**
   * Greedily generates {@code maxTokens} token ids for one prompt.
   *
   * <p>The backend is reset first, so a sequence never depends on what ran before it. That matters
   * beyond hygiene: the kernel uploads weights on first use, and a run whose first prompt paid for
   * every upload while the rest did not would report a prefill figure that is mostly one-time cost.
   * Warm the arm with {@link #warmUp} before measuring.
   */
  static Sequence generate(
      InferenceBackend backend,
      int promptIndex,
      String prompt,
      int maxTokens,
      int contextLength,
      StepListener listener) {
    Objects.requireNonNull(backend, "backend");
    Objects.requireNonNull(prompt, "prompt");
    Objects.requireNonNull(listener, "listener");
    if (maxTokens < 1) {
      throw new IllegalArgumentException("--max-tokens must be at least 1");
    }

    Tokenizer tokenizer = backend.tokenizer();
    int[] promptTokens = tokenizer.encode(prompt);
    if (promptTokens.length == 0) {
      throw new IllegalArgumentException("prompt " + promptIndex + " produced no tokens");
    }
    int required = Math.addExact(promptTokens.length, maxTokens);
    if (required > contextLength) {
      // Reported rather than silently truncated: a capped prompt caps the metric for reasons
      // unrelated to the kernels.
      throw new IllegalArgumentException(
          "prompt "
              + promptIndex
              + " needs "
              + required
              + " context positions but --context is "
              + contextLength);
    }

    backend.reset();
    long prefillStart = System.nanoTime();
    float[] logits = backend.prefill(promptTokens, 0);
    long prefillNanos = System.nanoTime() - prefillStart;

    List<Integer> tokenIds = new ArrayList<>(maxTokens);
    int token = argmax(logits);
    tokenIds.add(token);
    int endOfGenerationIndex = tokenizer.isEndOfGeneration(token) ? 0 : -1;
    listener.generated(promptIndex, 0, token, false);

    int position = promptTokens.length;
    long decodeStart = System.nanoTime();
    for (int index = 1; index < maxTokens; index++) {
      logits = backend.forwardTransient(token, position++);
      token = argmax(logits);
      tokenIds.add(token);
      if (endOfGenerationIndex < 0 && tokenizer.isEndOfGeneration(token)) {
        endOfGenerationIndex = index;
      }
      listener.generated(promptIndex, index, token, true);
    }
    long decodeNanos = System.nanoTime() - decodeStart;

    return new Sequence(
        promptIndex,
        Hashing.sha256(prompt),
        promptTokens.length,
        tokenIds,
        prefillNanos,
        decodeNanos,
        endOfGenerationIndex);
  }

  /**
   * Runs one throwaway sequence so first-use costs land outside the measurement.
   *
   * <p>Uses the first prompt, which is therefore the only prompt whose weight uploads are paid
   * twice. The counters record the uploads either way, so the report shows whether the warmup did
   * its job rather than the reader having to assume it.
   */
  static void warmUp(InferenceBackend backend, String prompt, int warmupTokens, int contextLength) {
    if (warmupTokens <= 0) {
      return;
    }
    generate(backend, -1, prompt, warmupTokens, contextLength, StepListener.none());
    backend.reset();
  }

  /** Generates every prompt in order. */
  static List<Sequence> generateAll(
      InferenceBackend backend,
      List<String> prompts,
      int maxTokens,
      int contextLength,
      StepListener listener) {
    List<Sequence> sequences = new ArrayList<>(prompts.size());
    for (int index = 0; index < prompts.size(); index++) {
      sequences.add(
          generate(backend, index, prompts.get(index), maxTokens, contextLength, listener));
    }
    return List.copyOf(sequences);
  }
}
