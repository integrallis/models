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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.SharedInferencePrefix;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Batching several questions against one frozen prefix must not change any answer.
 *
 * <p>A batched path that is merely close would be worse than no batched path at all: it would make
 * a decision head's output depend on how many questions happened to be asked together, and that
 * difference would surface as an unreproducible benchmark rather than as a failure.
 */
@Tag("integration")
class BatchedHiddenStatePrefillIntegrationTest {

  private static final ModelFixtureRequirement QWEN3_0_6B_Q4_0 =
      ModelFixtureRequirement.of("hf://ggml-org/Qwen3-0.6B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation");

  private static final List<String> QUESTIONS =
      List.of(
          " What is the monthly fee?\nAnswer:",
          " Which state's law governs this agreement and any disputes arising under it?\nAnswer:",
          " Is there a cap?\nAnswer:",
          " What happens on termination for convenience after month eighteen?\nAnswer:");

  @Test
  void batchedHiddenStatesMatchTheSequentialOnesExactly() throws Exception {
    var fixture = ModelFixtureRegistry.fromClasspath().resolve(QWEN3_0_6B_Q4_0).orElseThrow();
    var path = fixture.localPath().orElseThrow();

    try (PureJavaBackend backend = PureJavaBackend.load(path)) {
      assertThat(backend.supportsBatchedHiddenStates()).isTrue();

      int[] document =
          backend
              .tokenizer()
              .encode("Fees are 42,000 per month. Delaware law governs.\n\nQuestion:");
      int[][] tails = new int[QUESTIONS.size()][];
      int[] starts = new int[QUESTIONS.size()];
      for (int index = 0; index < QUESTIONS.size(); index++) {
        tails[index] = backend.tokenizer().encode(QUESTIONS.get(index));
        starts[index] = document.length;
      }

      SharedInferencePrefix prefix = freeze(backend, document);

      // Sequential: the path whose numbers are already published.
      List<float[]> sequential = new ArrayList<>();
      for (int[] tail : tails) {
        try (InferenceSession branch = backend.fork(prefix)) {
          sequential.add(backend.prefillHiddenState(branch, tail, document.length));
        }
      }

      // Batched: forks opened together, prefilled in one ragged pass.
      InferenceSession[] branches = new InferenceSession[tails.length];
      float[][] batched;
      try {
        for (int index = 0; index < tails.length; index++) {
          branches[index] = backend.fork(prefix);
        }
        batched = backend.prefillBatchHiddenStates(branches, tails, starts);
      } finally {
        for (InferenceSession branch : branches) {
          if (branch != null) {
            branch.close();
          }
        }
      }

      assertThat(batched.length).isEqualTo(sequential.size());
      for (int index = 0; index < batched.length; index++) {
        assertThat(batched[index])
            .as("question %d of %d", index + 1, batched.length)
            .isEqualTo(sequential.get(index));
      }
    }
  }

  @Test
  void aSingleQuestionTakesTheSamePathAsBefore() throws Exception {
    var fixture = ModelFixtureRegistry.fromClasspath().resolve(QWEN3_0_6B_Q4_0).orElseThrow();
    var path = fixture.localPath().orElseThrow();

    try (PureJavaBackend backend = PureJavaBackend.load(path)) {
      int[] document = backend.tokenizer().encode("A short document.\n\nQuestion:");
      int[] tail = backend.tokenizer().encode(" Anything?\nAnswer:");
      SharedInferencePrefix prefix = freeze(backend, document);

      float[] sequential;
      try (InferenceSession branch = backend.fork(prefix)) {
        sequential = backend.prefillHiddenState(branch, tail, document.length);
      }
      float[][] batched;
      try (InferenceSession branch = backend.fork(prefix)) {
        batched =
            backend.prefillBatchHiddenStates(
                new InferenceSession[] {branch}, new int[][] {tail}, new int[] {document.length});
      }
      // A batch of one must not quietly become a different computation.
      assertThat(batched[0]).isEqualTo(sequential);
    }
  }

  private static SharedInferencePrefix freeze(PureJavaBackend backend, int[] document) {
    try (InferenceSession source = backend.openSession()) {
      backend.prefill(source, document, 0);
      return backend.freezePrefix(source);
    }
  }
}
