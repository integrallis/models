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
package com.integrallis.models.decisions;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Establishes, on the real qualified artifact rather than from reading the code, that the base this
 * tier is built on genuinely exposes a hidden state and genuinely forks a shared physical prefix.
 *
 * <p>This is a capability probe, not a measurement. It produces no number anyone should quote; it
 * exists so the architecture is known to be viable before any harvest is paid for.
 *
 * <p>Skips cleanly when the artifact is absent, so it never fails a workstation build.
 */
@Tag("integration")
class GranitePrefixCapabilityIT {

  private static final String MODEL_PROPERTY = "decisions.granite.model";

  @Test
  void theQualifiedGraniteBaseExposesHiddenStatesAndSharesAPhysicalPrefix() {
    Path model = modelPath();
    assumeTrue(model != null && Files.isRegularFile(model), "granite artifact not present");

    try (PureJavaBackend backend = PureJavaBackend.load(model)) {
      assertThat(backend).isInstanceOf(SharedPrefixInferenceBackend.class);
      SharedPrefixInferenceBackend sharing = backend;

      assertThat(sharing.supportsHiddenState())
          .describedAs("granite must expose final normalized hidden states")
          .isTrue();
      assertThat(sharing.supportsSharedPrefixes())
          .describedAs("granite must support an immutable physical prefix")
          .isTrue();

      int[] state = backend.tokenizer().encode("The launch window closes on Friday.");
      int[] question = backend.tokenizer().encode(" Answerable?");

      InferenceSession stateSession = sharing.openSession();
      sharing.prefillHiddenState(stateSession, state, stateSession.checkpoint());
      SharedInferencePrefix prefix = sharing.freezePrefix(stateSession);

      try (InferenceSession first = sharing.fork(prefix);
          InferenceSession second = sharing.fork(prefix)) {

        assertThat(sharing.sharesPrefixStorage(first, second))
            .describedAs("two forks of one prefix must reference the same physical storage")
            .isTrue();

        float[] hiddenFirst =
            sharing.prefillHiddenState(first, question, first.checkpoint()).clone();
        float[] hiddenSecond =
            sharing.prefillHiddenState(second, question, second.checkpoint()).clone();

        assertThat(hiddenFirst).hasSize(2560);
        assertThat(hiddenFirst)
            .describedAs("identical suffixes over one shared prefix must agree exactly")
            .isEqualTo(hiddenSecond);
        assertThat(isFinite(hiddenFirst)).isTrue();
        assertThat(prefix.sharedBytes()).isPositive();
      }

      // The evaluator drives the same backend end to end, over a bounded space.
      Noul answerable = new Noul("the state answers the question");
      double[][] weights = new double[2][2560];
      weights[1][0] = 1.0;
      DecisionHead head = new LinearDecisionHead(answerable, weights, new double[] {0.0, 0.0});
      DecisionBatch batch =
          new SharedPrefixDecisionEvaluator(sharing)
              .evaluate(state, List.of(new Question(answerable, question, head, 1.0)));

      assertThat(batch.verdicts()).hasSize(1);
      assertThat(batch.prefixFreezes()).isEqualTo(1);
      assertThat(answerable.labels()).contains(batch.verdicts().get(0).winner());
    }
  }

  private static boolean isFinite(float[] values) {
    for (float value : values) {
      if (!Float.isFinite(value)) {
        return false;
      }
    }
    return true;
  }

  private static Path modelPath() {
    String configured =
        System.getProperty(MODEL_PROPERTY, System.getenv("DECISIONS_GRANITE_MODEL"));
    return configured == null || configured.isBlank() ? null : Path.of(configured);
  }
}
