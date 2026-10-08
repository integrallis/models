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
package com.integrallis.models.backend.purejava.plan;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Measures whether each optimization that defaults to <em>off</em> changes generated tokens.
 *
 * <p>Seven of the eleven pure-Java optimizations default off. Only one, {@code
 * fusedGroupedAttention}, documents why: it agrees with the head-by-head loop to 1.4e-6 and that
 * still changed 2 of 9 generated answers, because greedy decoding is a discrete argmax. For the
 * other six there was no evidence either way — no test compared output across the setting — so "it
 * changes bytes" was an assumption, and so was "it is safe to turn on".
 *
 * <p>This test produces the evidence. For each setting it greedy-decodes the same prompt with the
 * setting off and on and compares token ids exactly. A setting whose tokens are identical can
 * become the only route; one whose tokens differ is an epoch change, because a published
 * qualification record is a claim about the exact tokens a model produced.
 *
 * <p><b>It also asserts the setting was observable.</b> A toggle that silently does nothing
 * produces identical tokens and looks like "no effect" when the truth is "it never ran". Each arm
 * therefore requires the planner to have reported the optimization {@code ENABLED}; an arm the
 * planner reports {@code UNSUPPORTED} for this model is skipped by name rather than counted as
 * parity.
 *
 * <p>Needs a real GGUF, since synthetic fixtures hold constant the very thing being varied. Point
 * {@code -Dmodels.parity.model} at one. Q4_0 and Q8_0 artifacts exercise the staged and block-major
 * settings that a Q4_K artifact reports UNSUPPORTED.
 */
@Tag("slow")
class DefaultOffOptimizationParityTest {

  private static final String MODEL_PROPERTY = "models.parity.model";
  private static final int GENERATED_TOKENS = 24;

  /**
   * Settings that are now enabled by default, paired with the optimization id the planner reports.
   *
   * <p>Each arm turns the setting <em>off</em> and checks the tokens still match. That is the same
   * measurement that justified adopting them — the route choice does not change output, so the
   * faster one is free — kept running as a regression guard. If one of these ever starts diverging,
   * the default that was adopted on this evidence has to be revisited.
   */
  private static final List<String[]> DEFAULT_ON =
      List.of(
          new String[] {
            PureJavaPlanConfiguration.BATCHED_ATTENTION_SCORES_PROPERTY, "batched-attention-scores"
          },
          new String[] {
            PureJavaPlanConfiguration.BATCHED_ATTENTION_VALUES_PROPERTY, "batched-attention-values"
          },
          new String[] {
            PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY, "staged-quantized-ffn"
          },
          new String[] {
            PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "staged-quantized-layer"
          },
          new String[] {
            PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY,
            "block-major-q8-activations"
          },
          new String[] {
            PureJavaPlanConfiguration.PARALLEL_Q8_FFN_PREPARATION_PROPERTY,
            "parallel-q8-ffn-preparation"
          });

  /**
   * Settings that cannot be enabled on their own, with the prerequisites the planner demands.
   *
   * <p>{@code blockMajorQ8Activations} needs a retained staged plan and batched prefill; {@code
   * parallelQ8FfnPreparation} needs the staged layer plan and block-major activations. Set alone,
   * each is reported DISABLED with that reason, which would read as "never ran" and is not a
   * measurement of anything. Each arm here therefore carries its prerequisites, and still asserts
   * the planner reported it ENABLED before its tokens are compared.
   */
  private static final List<String[]> DEPENDENT =
      List.of(
          new String[] {
            "block-major-q8-activations",
            PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY,
            PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
            PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY
          },
          new String[] {
            "parallel-q8-ffn-preparation",
            PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY,
            PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
            PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY,
            PureJavaPlanConfiguration.PARALLEL_Q8_FFN_PREPARATION_PROPERTY
          });

  /** Still off by default, because it measured divergent; the arm therefore turns it on. */
  private static final List<String[]> STILL_OFF =
      List.<String[]>of(
          new String[] {
            PureJavaPlanConfiguration.FUSED_GROUPED_ATTENTION_PROPERTY, "fused-grouped-attention"
          });

  @Test
  void recordsWhichSettingsChangeGeneratedTokens() {
    String configured = System.getProperty(MODEL_PROPERTY);
    assumeTrue(configured != null, MODEL_PROPERTY + " is not set");
    Path model = Path.of(configured);
    assumeTrue(Files.isRegularFile(model), model + " is not a file");

    int[] prompt;
    int[] baseline;
    try (PureJavaBackend backend = PureJavaBackend.load(model, BackendConfiguration.empty())) {
      prompt = backend.tokenizer().encode("The capital of France is");
      baseline = greedy(backend, prompt);
    }
    assertThat(baseline).as("baseline must generate something to compare").isNotEmpty();

    List<String> identical = new ArrayList<>();
    List<String> divergent = new ArrayList<>();
    List<String> unsupported = new ArrayList<>();

    // One arm per setting, flipped away from whatever its default now is: the six adopted ones get
    // "false" so divergence would show the adoption was unsafe, and the one still off gets "true".
    List<String[]> arms = new ArrayList<>();
    for (String[] entry : DEFAULT_ON) {
      arms.add(new String[] {entry[0], entry[1], "false"});
    }
    for (String[] entry : STILL_OFF) {
      arms.add(new String[] {entry[0], entry[1], "true"});
    }

    for (String[] entry : arms) {
      String property = entry[0];
      String optimizationId = entry[1];
      String value = entry[2];
      // "The arm took effect" means the planner reports the status that was asked for, which is
      // DISABLED for an arm that turned a setting off. Requiring ENABLED here would silently
      // discard every one of those arms.
      OptimizationStatus expected =
          "true".equals(value) ? OptimizationStatus.ENABLED : OptimizationStatus.DISABLED;
      try (PureJavaBackend backend =
          PureJavaBackend.load(
              model, new BackendConfiguration(Map.of(), Map.of(property, value), List.of()))) {
        Optional<OptimizationDecision> decision =
            backend.diagnostics().optimization(optimizationId);
        if (decision.isEmpty() || decision.get().status() != expected) {
          // Not parity: the arm never took effect. Named so it cannot be mistaken for a measured
          // null.
          unsupported.add(
              optimizationId
                  + "="
                  + decision.map(status -> status.status().name()).orElse("ABSENT"));
          continue;
        }
        int[] arm = greedy(backend, prompt);
        int firstDifference = firstDifference(baseline, arm);
        if (firstDifference < 0) {
          identical.add(optimizationId);
        } else {
          divergent.add(optimizationId + " (first differs at token " + firstDifference + ")");
        }
      }
    }

    for (String[] entry : DEPENDENT) {
      String optimizationId = entry[0];
      Map<String, String> recommendations = new java.util.LinkedHashMap<>();
      for (int index = 1; index < entry.length; index++) {
        recommendations.put(entry[index], "true");
      }
      // These are enabled by default now, so an explicit "true" is what the default already does;
      // the arm is kept because it is the only way to see them ENABLED on an artifact whose
      // topology supports them, and the token comparison is still the evidence.
      try (PureJavaBackend backend =
          PureJavaBackend.load(
              model, new BackendConfiguration(Map.of(), Map.copyOf(recommendations), List.of()))) {
        Optional<OptimizationDecision> decision =
            backend.diagnostics().optimization(optimizationId);
        if (decision.isEmpty() || decision.get().status() != OptimizationStatus.ENABLED) {
          unsupported.add(
              optimizationId
                  + "+prereqs="
                  + decision.map(value -> value.status().name()).orElse("ABSENT")
                  + " ("
                  + decision.map(OptimizationDecision::reason).orElse("absent")
                  + ")");
          continue;
        }
        int[] arm = greedy(backend, prompt);
        int firstDifference = firstDifference(baseline, arm);
        if (firstDifference < 0) {
          identical.add(optimizationId + "+prereqs");
        } else {
          divergent.add(
              optimizationId + "+prereqs (first differs at token " + firstDifference + ")");
        }
      }
    }

    System.out.println("== default-off optimization parity: " + model.getFileName());
    System.out.println("   byte-identical, so adoptable as the only route : " + identical);
    System.out.println("   changes generated tokens, so an epoch change   : " + divergent);
    System.out.println("   never ran for this artifact, NOT a measurement  : " + unsupported);

    assertThat(identical.size() + divergent.size() + unsupported.size())
        .as("every arm must be classified, never silently skipped")
        .isEqualTo(DEFAULT_ON.size() + STILL_OFF.size() + DEPENDENT.size());
  }

  private static int[] greedy(PureJavaBackend backend, int[] prompt) {
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
