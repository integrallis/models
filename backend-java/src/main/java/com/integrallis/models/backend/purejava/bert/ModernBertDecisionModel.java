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
package com.integrallis.models.backend.purejava.bert;

import com.integrallis.models.backend.purejava.bert.ModernBertConfig.QuestionType;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * A typed-decision model: a state and a typed question in, a chosen option and a calibrated
 * probability out, in one forward pass.
 *
 * <p>The public face of the {@code modern-bert} decision work. Everything behind it — the layout
 * arithmetic, the weights contract, the forward pass, the temperature table — is package-private on
 * purpose, because none of it is a decision a caller should have to make. What a caller needs is
 * this: open an artifact, ask typed questions, read answers.
 *
 * <p><b>It reports when it does not know how confident it is.</b> {@link
 * ModernBertDecisionAnswer#uncalibrated()} is true when the artifact declares no usable temperature
 * for the question's bucket, and this checkpoint has one: it ships {@code
 * decision.temperature.choice.11 = 0.1006}, which multiplies logits roughly tenfold and would
 * publish a 0.24 top probability as 0.99. The reference clamps it and warns; so does this. A caller
 * gating on confidence is told the number is not one anybody fitted, rather than being handed a
 * coin flip dressed as a certainty.
 *
 * <p>Not a {@code TextGenerationModel} and deliberately not adapted to one. There is no generation
 * here and never will be — the model is non-autoregressive, it answers every option in a single
 * pass, and wrapping that in a token-streaming interface would misdescribe both the cost and the
 * contract.
 */
public final class ModernBertDecisionModel implements AutoCloseable {

  private final Arena arena;
  private final ModernBertConfig config;
  private final ModernBertDecisionPrompt prompt;
  private final ModernBertDecisionForwardPass forward;

  private ModernBertDecisionModel(
      Arena arena,
      ModernBertConfig config,
      ModernBertDecisionPrompt prompt,
      ModernBertDecisionForwardPass forward) {
    this.arena = arena;
    this.config = config;
    this.prompt = prompt;
    this.forward = forward;
  }

  /**
   * Opens a {@code modern-bert} decision artifact.
   *
   * @param artifact the GGUF file
   * @return the model, which the caller closes
   * @throws IllegalArgumentException when the artifact is not a decision model
   * @throws java.io.UncheckedIOException when the file cannot be read
   */
  public static ModernBertDecisionModel open(Path artifact) {
    return open(artifact, GgufBatchedMatrixKernel.none());
  }

  /**
   * Opens the model with an injected projection kernel.
   *
   * <p>The no-argument overload stays pure Java deliberately: this module does not depend on
   * backend-native, so it cannot reach the Rust shim without inverting that layering. A caller that
   * has the native kernel on its classpath passes it here, which is the same seam {@code
   * SopranoTextToSpeechModel.loadNative} uses.
   *
   * <p>The kernel is asked whether it wants each shape rather than told to take it, so an
   * implementation that would be slower on a projection can decline and leave the exact Java path
   * in place for that one call.
   *
   * @param artifact the GGUF file
   * @param matrixKernel the projection kernel, or {@link GgufBatchedMatrixKernel#none()}
   * @return the model, which the caller closes
   */
  public static ModernBertDecisionModel open(Path artifact, GgufBatchedMatrixKernel matrixKernel) {
    Objects.requireNonNull(artifact, "artifact");
    Objects.requireNonNull(matrixKernel, "matrixKernel");
    Arena arena = Arena.ofShared();
    try {
      GgufFile file = GgufParser.parse(artifact, arena);
      ModernBertConfig config = ModernBertConfig.fromMetadata(file.metadata());
      if (config.decision().isEmpty()) {
        throw new IllegalArgumentException(
            "this modern-bert artifact declares no decision head; it cannot decide: " + artifact);
      }
      ModernBertWeights weights = ModernBertWeights.load(file, config);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      return new ModernBertDecisionModel(
          arena,
          config,
          new ModernBertDecisionPrompt(tokenizer, config),
          new ModernBertDecisionForwardPass(config, weights, matrixKernel));
    } catch (java.io.IOException failure) {
      arena.close();
      throw new java.io.UncheckedIOException("cannot read " + artifact, failure);
    } catch (RuntimeException failure) {
      arena.close();
      throw failure;
    }
  }

  /**
   * Answers one typed question about one state.
   *
   * @param type the question type, which selects both the option rendering and the calibration
   *     bucket
   * @param instructions the question
   * @param options the options, in the order the answer's probabilities will be in
   * @param state the text being decided about
   * @return the chosen option, the calibrated probabilities, and whether they are calibrated at all
   */
  public ModernBertDecisionAnswer decide(
      QuestionType type,
      String instructions,
      List<ModernBertDecisionPrompt.Option> options,
      String state) {
    ModernBertDecisionPrompt.Encoded encoded = prompt.encode(type, instructions, options, state);
    ModernBertDecisionLayout layout =
        ModernBertDecisionLayout.of(encoded.markers(), encoded.headLength(), encoded.ids().length);
    float[] logits = forward.logits(encoded.ids(), layout, type);
    return ModernBertDecisionAnswer.of(type, logits, config.decision());
  }

  /** The artifact's configuration, for callers that want to report what they ran. */
  public ModernBertConfig configuration() {
    return config;
  }

  @Override
  public void close() {
    arena.close();
  }
}
