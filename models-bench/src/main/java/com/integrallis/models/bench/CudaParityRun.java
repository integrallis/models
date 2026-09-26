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

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Gate G1: the accelerated arm must reproduce the control arm's token id sequence exactly.
 *
 * <p>No token-level tolerance is permitted, and this class offers none. The comparison is over full
 * sequences rather than a first-token check or a text diff: a kernel can agree on the first dozen
 * tokens and drift later, and two different token sequences can decode to the same string.
 */
final class CudaParityRun {

  private CudaParityRun() {}

  /** The first place the two arms disagreed, and what the accelerator was doing there. */
  record Divergence(
      int promptIndex,
      String promptDigest,
      int tokenIndex,
      int acceleratedTokenId,
      int controlTokenId,
      boolean attentionRouted,
      boolean projectionRouted,
      Optional<CudaRoutingRecorder.StepRouting> routing,
      String reading) {}

  /** G1's outcome. */
  record Result(
      int promptCount,
      int tokensPerPrompt,
      int comparedTokens,
      int endOfGenerationSequences,
      boolean identical,
      Optional<Divergence> firstDivergence) {

    /** Whether every compared sequence matched. */
    boolean passed() {
      return identical;
    }
  }

  /**
   * Compares two arms' sequences prompt by prompt, token by token, stopping at the first mismatch.
   *
   * @param accelerated the arm under test, in prompt order
   * @param control the Vector API arm, in the same prompt order
   * @param recorder per-token routing for the accelerated arm, used only to explain a failure
   */
  static Result compare(
      List<GreedyDecode.Sequence> accelerated,
      List<GreedyDecode.Sequence> control,
      CudaRoutingRecorder recorder) {
    Objects.requireNonNull(accelerated, "accelerated");
    Objects.requireNonNull(control, "control");
    Objects.requireNonNull(recorder, "recorder");
    if (accelerated.size() != control.size()) {
      throw new IllegalArgumentException(
          "arms ran different prompt counts: " + accelerated.size() + " and " + control.size());
    }
    if (accelerated.isEmpty()) {
      throw new IllegalArgumentException("no sequences to compare");
    }

    int compared = 0;
    int endOfGeneration = 0;
    int tokensPerPrompt = accelerated.get(0).tokenIds().size();
    for (int promptIndex = 0; promptIndex < accelerated.size(); promptIndex++) {
      GreedyDecode.Sequence left = accelerated.get(promptIndex);
      GreedyDecode.Sequence right = control.get(promptIndex);
      if (!left.promptDigest().equals(right.promptDigest())) {
        throw new IllegalArgumentException(
            "arms ran different prompts at index " + promptIndex + "; the comparison is void");
      }
      if (left.hitEndOfGeneration() || right.hitEndOfGeneration()) {
        endOfGeneration++;
      }
      int tokens = Math.min(left.tokenIds().size(), right.tokenIds().size());
      for (int tokenIndex = 0; tokenIndex < tokens; tokenIndex++) {
        int acceleratedToken = left.tokenIds().get(tokenIndex);
        int controlToken = right.tokenIds().get(tokenIndex);
        compared++;
        if (acceleratedToken != controlToken) {
          Optional<CudaRoutingRecorder.StepRouting> routing = recorder.at(promptIndex, tokenIndex);
          boolean attention =
              routing.map(CudaRoutingRecorder.StepRouting::attentionRouted).orElse(false);
          boolean projection =
              routing.map(CudaRoutingRecorder.StepRouting::projectionRouted).orElse(false);
          return new Result(
              accelerated.size(),
              tokensPerPrompt,
              compared,
              endOfGeneration,
              false,
              Optional.of(
                  new Divergence(
                      promptIndex,
                      left.promptDigest(),
                      tokenIndex,
                      acceleratedToken,
                      controlToken,
                      attention,
                      projection,
                      routing,
                      reading(routing.isPresent(), attention, projection))));
        }
      }
      if (left.tokenIds().size() != right.tokenIds().size()) {
        throw new IllegalArgumentException(
            "arms generated different token counts for prompt " + promptIndex);
      }
    }
    return new Result(
        accelerated.size(), tokensPerPrompt, compared, endOfGeneration, true, Optional.empty());
  }

  /**
   * States what the divergence points at, in the pre-registration's own terms.
   *
   * <p>Deliberately not a verdict. It names the stage the evidence implicates and the fix the
   * pre-registration already committed to; loosening G1 is not among the options it offers.
   */
  private static String reading(boolean recorded, boolean attention, boolean projection) {
    if (!recorded) {
      return "no routing was recorded for this token: the accelerator was absent or inert, so this "
          + "divergence is between two Vector API runs and indicates nondeterminism in the "
          + "harness or the backend, not in the kernels";
    }
    if (attention && !projection) {
      return "attention routed and no projection did at this token: consistent with device expf "
          + "disagreeing with the host libm (UPSTREAM.md CU-005). The fix is to make device expf "
          + "agree exactly. G1 is not to be loosened";
    }
    if (attention) {
      return "both attention and projections routed at this token, so expf and the K-quant path "
          + "are both live suspects. Re-run with attention refused to separate them before "
          + "concluding anything";
    }
    if (projection) {
      return "projections routed and attention did not: the K-quant path is meant to be bit-exact "
          + "by construction, so this is a kernel defect rather than a tolerance question";
    }
    return "nothing routed on the device at this token, so the two arms ran the same code and "
        + "still disagreed: look for nondeterminism in the harness before the kernels";
  }
}
