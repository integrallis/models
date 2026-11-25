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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.runtime.fusion.EnsembleGate;
import com.integrallis.models.runtime.fusion.EntropyGate;
import com.integrallis.models.runtime.fusion.FusedInferenceBackend;
import com.integrallis.models.runtime.fusion.FusionRecipe;
import com.integrallis.models.runtime.fusion.FusionRule;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/**
 * Gates G1 and G1b on toy members: the runtime's gated backend, driven through the bench's
 * single-member loop, equals the lead alone when the gate never fires and equals the bench's
 * every-token {@link FusionDecoder#fuse} reference when it always fires.
 */
class GatedArmTest {
  private final ExecutorService executor = Executors.newFixedThreadPool(3);
  private final DecodeSettings settings = new DecodeSettings(0f, 0, 1f, 7L, 24, 1);

  @AfterEach
  void shutdown() {
    executor.shutdownNow();
  }

  private static List<FusionMember> toys() {
    return List.of(
        new ToyMember("L", 1.3, 0.0), new ToyMember("M", 2.1, 0.4), new ToyMember("I", 0.7, -0.2));
  }

  private static PipelineMember gated(
      List<FusionMember> members, double[] weights, EnsembleGate gate) {
    FusedInferenceBackend fused =
        new FusedInferenceBackend(
            members.stream().map(member -> new ToyMemberBackend(member)).toList(),
            new FusionRecipe(FusionRule.POE, weights, gate));
    return new PipelineMember("gated", fused);
  }

  @Test
  void neverFiringGateIsTheLeadAloneTokenForToken() {
    int[] prompt = {3, 1, 4, 1, 5};
    DecodeResult lead = FusionDecoder.single(new ToyMember("L", 1.3, 0.0), prompt, settings);
    DecodeResult gated =
        FusionDecoder.single(
            gated(toys(), new double[] {0.55, 0.25, 0.20}, EnsembleGate.never()), prompt, settings);
    assertThat(gated.tokens()).isEqualTo(lead.tokens());
    assertThat(gated.text()).isEqualTo(lead.text());
  }

  @Test
  void alwaysFiringGateEqualsTheEveryTokenFusionReference() {
    int[] prompt = {3, 1, 4, 1, 5};
    double[] uniform = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    DecodeResult reference =
        FusionDecoder.fuse(toys(), prompt, uniform, FusionRule.POE, settings, executor);
    DecodeResult gated =
        FusionDecoder.single(gated(toys(), uniform, new EntropyGate(0.0)), prompt, settings);
    assertThat(gated.tokens()).isEqualTo(reference.tokens());
    assertThat(gated.text()).isEqualTo(reference.text());
  }
}
