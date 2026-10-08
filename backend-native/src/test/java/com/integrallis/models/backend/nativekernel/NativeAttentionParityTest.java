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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The same default-off parity question as {@code DefaultOffOptimizationParityTest}, asked on the
 * path that actually ships.
 *
 * <p>That test measures {@code PureJavaBackend}. {@code batchedAttentionScores} and {@code
 * batchedAttentionValues} are not among the plan recommendations the Rust kernel forces, so turning
 * them on by default would change the {@code rust-ffm} path too — and every certified record was
 * produced with them reported DISABLED. A qualification record is a claim about the exact tokens a
 * model produced, so the default cannot move on pure-Java evidence alone.
 *
 * <p>{@code stagedQuantizedFfn} and {@code stagedQuantizedLayer} are deliberately forced to false
 * by {@code RustGgufBatchedMatrixKernel}, because the shim already owns those projections. They are
 * therefore not in scope here: their default cannot reach this path.
 *
 * <p>Needs a GGUF and the built shim: {@code -Dmodels.parity.model=<artifact.gguf>} and {@code
 * -Dmodels.native.kernels.library=<libjmodels_kernels.dylib>}.
 */
@Tag("slow")
class NativeAttentionParityTest {

  private static final String MODEL_PROPERTY = "models.parity.model";
  private static final int GENERATED_TOKENS = 24;

  private static final List<String[]> IN_SCOPE =
      List.of(
          new String[] {
            PureJavaPlanConfiguration.BATCHED_ATTENTION_SCORES_PROPERTY, "batched-attention-scores"
          },
          new String[] {
            PureJavaPlanConfiguration.BATCHED_ATTENTION_VALUES_PROPERTY, "batched-attention-values"
          });

  @Test
  void recordsWhetherBatchedAttentionChangesTokensOnTheShippedPath() {
    String configuredModel = System.getProperty(MODEL_PROPERTY);
    String configuredLibrary = System.getProperty(RustFfmBackend.LIBRARY_PATH_PROPERTY);
    assumeTrue(configuredModel != null, MODEL_PROPERTY + " is not set");
    assumeTrue(configuredLibrary != null, RustFfmBackend.LIBRARY_PATH_PROPERTY + " is not set");
    Path model = Path.of(configuredModel);
    Path library = Path.of(configuredLibrary);
    assumeTrue(Files.isRegularFile(model), model + " is not a file");
    assumeTrue(Files.isRegularFile(library), library + " is not a file");

    int[] prompt;
    int[] baseline;
    try (RustFfmBackend backend =
        RustFfmBackend.load(model, library, BackendConfiguration.empty())) {
      assertThat(backend.diagnostics().environment())
          .as("the shim must actually be serving this run")
          .containsEntry("kernel-runtime", "rust-ffm");
      prompt = backend.tokenizer().encode("The capital of France is");
      baseline = greedy(backend, prompt);
    }

    List<String> identical = new ArrayList<>();
    List<String> divergent = new ArrayList<>();
    List<String> notRun = new ArrayList<>();

    for (String[] entry : IN_SCOPE) {
      try (RustFfmBackend backend =
          RustFfmBackend.load(
              model,
              library,
              new BackendConfiguration(Map.of(), Map.of(entry[0], "true"), List.of()))) {
        Optional<OptimizationDecision> decision = backend.diagnostics().optimization(entry[1]);
        if (decision.isEmpty() || decision.get().status() != OptimizationStatus.ENABLED) {
          notRun.add(entry[1] + "=" + decision.map(d -> d.status().name()).orElse("ABSENT"));
          continue;
        }
        int[] arm = greedy(backend, prompt);
        int first = firstDifference(baseline, arm);
        if (first < 0) {
          identical.add(entry[1]);
        } else {
          divergent.add(entry[1] + " (first differs at token " + first + ")");
        }
      }
    }

    System.out.println("== rust-ffm attention parity: " + model.getFileName());
    System.out.println("   byte-identical on the shipped path : " + identical);
    System.out.println("   changes generated tokens           : " + divergent);
    System.out.println("   never ran, NOT a measurement       : " + notRun);

    assertThat(identical.size() + divergent.size() + notRun.size()).isEqualTo(IN_SCOPE.size());
  }

  private static int[] greedy(RustFfmBackend backend, int[] prompt) {
    backend.reset();
    float[] logits = null;
    int position = 0;
    for (int token : prompt) {
      logits = backend.forward(token, position++);
    }
    int[] generated = new int[GENERATED_TOKENS];
    for (int index = 0; index < GENERATED_TOKENS; index++) {
      int token = argmax(logits);
      generated[index] = token;
      if (index + 1 < GENERATED_TOKENS) {
        logits = backend.forward(token, position++);
      }
    }
    return generated;
  }

  private static int firstDifference(int[] left, int[] right) {
    for (int index = 0; index < Math.min(left.length, right.length); index++) {
      if (left[index] != right[index]) {
        return index;
      }
    }
    return left.length == right.length ? -1 : Math.min(left.length, right.length);
  }

  private static int argmax(float[] logits) {
    int best = 0;
    for (int index = 1; index < logits.length; index++) {
      if (logits[index] > logits[best]) {
        best = index;
      }
    }
    return best;
  }
}
