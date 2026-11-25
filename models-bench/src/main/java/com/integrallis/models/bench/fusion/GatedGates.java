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
import com.integrallis.models.runtime.fusion.EnsembleGate;
import com.integrallis.models.runtime.fusion.FusedInferenceBackend;
import com.integrallis.models.runtime.fusion.FusionMath;
import com.integrallis.models.runtime.fusion.FusionRecipe;
import com.integrallis.models.runtime.fusion.FusionRule;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.function.Function;

/**
 * The checks behind gates G1-gated and G8 of the gated-fusion protocol, as pure functions over
 * members so that toy members can prove them and the gate commands run them on real weights.
 *
 * <p>G1-gated: the runtime's gated backend, driven through the bench's single-member loop, must
 * equal the lead alone when the gate never fires, and equal the bench's every-token {@link
 * FusionDecoder#fuse} reference when it always fires, token for token at T = 0.
 *
 * <p>G8: an auxiliary that catches up over {@code d} tokens in one {@code prefill(delta, start)}
 * call must produce the same logits as the same member having stepped those tokens one {@code
 * forward} at a time. This is the exactness of the efficiency mechanism; the toy proves the
 * harness, real weights prove the kernels.
 */
final class GatedGates {
  private GatedGates() {}

  /** Two token sequences and where they first differ ({@code -1} when identical). */
  record Identity(int[] reference, int[] gated, int firstDivergence, boolean identical) {
    static Identity of(DecodeResult reference, DecodeResult gated) {
      int[] a = reference.tokens();
      int[] b = gated.tokens();
      int divergence = Arrays.mismatch(a, b);
      return new Identity(
          a, b, divergence, divergence < 0 && reference.truncated() == gated.truncated());
    }
  }

  /** The gated backend over {@code members} (lead first) as a bench member. */
  static PipelineMember gated(
      List<FusionMember> members,
      Function<FusionMember, InferenceBackend> backendOf,
      double[] weights,
      EnsembleGate gate) {
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            members.stream().map(backendOf).toList(),
            new FusionRecipe(FusionRule.POE, weights, gate));
    return new PipelineMember("gated", fused);
  }

  /** G1-gated, part one: a never-firing gate is the lead alone. */
  static Identity leadIdentity(
      List<FusionMember> members,
      Function<FusionMember, InferenceBackend> backendOf,
      int[] prompt,
      DecodeSettings settings) {
    double[] weights = new double[members.size()];
    weights[0] = 1.0;
    DecodeResult alone = FusionDecoder.single(members.getFirst(), prompt, settings);
    DecodeResult gated =
        FusionDecoder.single(
            gated(members, backendOf, weights, EnsembleGate.never()), prompt, settings);
    return Identity.of(alone, gated);
  }

  /** G1-gated, part two: an always-firing gate with uniform weights is every-token fusion. */
  static Identity everyTokenIdentity(
      List<FusionMember> members,
      Function<FusionMember, InferenceBackend> backendOf,
      int[] prompt,
      DecodeSettings settings,
      ExecutorService executor) {
    double[] uniform = new double[members.size()];
    Arrays.fill(uniform, 1.0 / members.size());
    DecodeResult reference =
        FusionDecoder.fuse(members, prompt, uniform, FusionRule.POE, settings, executor);
    DecodeResult gated =
        FusionDecoder.single(
            gated(members, backendOf, uniform, EnsembleGate.always()), prompt, settings);
    return Identity.of(reference, gated);
  }

  /** One G8 comparison. */
  record CatchUp(
      String member,
      int delta,
      double maxAbsDiff,
      int steppedArgmax,
      int caughtUpArgmax,
      boolean argmaxEqual,
      boolean within) {}

  /**
   * G8: logits after stepping {@code delta} tokens of {@code continuation} one at a time, against
   * logits after one prefill of the same tokens, on the same member from the same prompt.
   */
  static CatchUp catchUpEqualsStepping(
      FusionMember member, int[] prompt, int[] continuation, int delta, double tolerance) {
    if (delta < 1 || delta > continuation.length) {
      throw new IllegalArgumentException(
          "delta must be in [1, " + continuation.length + "]: " + delta);
    }
    int[] tokens = Arrays.copyOf(continuation, delta);
    member.reset();
    member.prefill(prompt, 0);
    float[] stepped = null;
    for (int k = 0; k < delta; k++) {
      stepped = member.forward(tokens[k], prompt.length + k).clone();
    }
    member.reset();
    member.prefill(prompt, 0);
    float[] caughtUp = member.prefill(tokens, prompt.length).clone();
    if (stepped.length != caughtUp.length) {
      throw new IllegalStateException(
          "vocabulary differs between paths: " + stepped.length + " vs " + caughtUp.length);
    }
    double maxAbsDiff = 0;
    for (int t = 0; t < stepped.length; t++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs((double) stepped[t] - caughtUp[t]));
    }
    int a = FusionMath.argmax(stepped);
    int b = FusionMath.argmax(caughtUp);
    return new CatchUp(member.name(), delta, maxAbsDiff, a, b, a == b, maxAbsDiff <= tolerance);
  }
}
