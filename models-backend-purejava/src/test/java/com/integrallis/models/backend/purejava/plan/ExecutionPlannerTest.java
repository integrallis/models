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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ExecutionPlannerTest {

  @Test
  void plansQ8GroupingAndBatchedPrefillFromTopology() {
    RuntimeFingerprint runtime = runtime("hotspot-c2");
    ModelTopology topology = uniformTopology(GgufTensorType.Q8_0);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(runtime, topology, PureJavaPlanConfiguration.defaults());

    assertThat(plan.diagnostics().planVersion()).isEqualTo("pure-java-v14");
    assertThat(plan.groupedProjections()).isTrue();
    assertThat(plan.q4Kernel()).isEqualTo(GgufQ4Kernel.WIDENED);
    assertThat(plan.prefillBatchSize()).isEqualTo(32);
    assertThat(plan.finalLayerPrefillPruning()).isTrue();
    assertThat(plan.finalLayerKvOnlyPrefill()).isTrue();
    assertThat(plan.batchedAttentionValues()).isFalse();
    assertThat(plan.batchedAttentionScores()).isFalse();
    assertThat(plan.diagnostics().environment()).containsEntry("compiler", "hotspot-c2");
    assertThat(plan.diagnostics().environment())
        .containsEntry("vector-provider", "test-vector")
        .containsEntry("active-vector-bits", "256")
        .containsEntry("q4-unsigned-pairwise-supported", "true")
        .containsEntry("gguf-executor", "persistent");
    assertThat(plan.diagnostics().optimization("grouped-projections"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("qkv", "independent")
                  .containsEntry("gate-up", "grouped");
            });
    assertThat(plan.diagnostics().optimization("batched-prefill"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings()).containsEntry("batch-size", "32");
            });
    assertThat(plan.diagnostics().optimization("batched-attention-values"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
    assertThat(plan.diagnostics().optimization("batched-attention-scores"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
    assertThat(plan.diagnostics().optimization("final-layer-prefill-pruning"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(plan.diagnostics().optimization("persistent-row-executor"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("threads", "8")
                  .containsEntry("chunks-per-thread", "2");
            });
  }

  @Test
  void plansQ5_0BatchedPrefillFromTopology() {
    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"),
            uniformTopology(GgufTensorType.Q5_0),
            PureJavaPlanConfiguration.defaults());

    assertThat(plan.prefillBatchSize()).isEqualTo(32);
    assertThat(plan.finalLayerPrefillPruning()).isFalse();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("batched-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(plan.diagnostics().optimization("final-layer-prefill-pruning"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
  }

  @Test
  void selectsShortPairwiseOnlyForEligibleQ4ModelsAndRuntimes() {
    PureJavaPlanConfiguration configuration =
        new PureJavaPlanConfiguration(
            true,
            true,
            GgufQ4Kernel.SHORT_PAIRWISE,
            32,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            false);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q4_0), configuration);

    assertThat(plan.q4Kernel()).isEqualTo(GgufQ4Kernel.SHORT_PAIRWISE);
    assertThat(plan.diagnostics().optimization("q4-kernel"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
  }

  @Test
  void selectsUnsignedPairwiseForCapableGraalRuntime() {
    PureJavaPlanConfiguration configuration =
        new PureJavaPlanConfiguration(
            true,
            true,
            GgufQ4Kernel.UNSIGNED_PAIRWISE,
            32,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            false);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q4_0), configuration);

    assertThat(plan.q4Kernel()).isEqualTo(GgufQ4Kernel.UNSIGNED_PAIRWISE);
    assertThat(plan.diagnostics().optimization("q4-kernel"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("requested", "unsigned-pairwise")
                  .containsEntry("compiler", "graal-jvmci")
                  .containsEntry("java-feature", "25");
            });
  }

  @Test
  void selectsUnsignedPairwiseForAnyRuntimeWithTheRequiredVectorCapability() {
    PureJavaPlanConfiguration configuration =
        new PureJavaPlanConfiguration(
            true,
            true,
            GgufQ4Kernel.UNSIGNED_PAIRWISE,
            32,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            false);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"), uniformTopology(GgufTensorType.Q4_0), configuration);

    assertThat(plan.q4Kernel()).isEqualTo(GgufQ4Kernel.UNSIGNED_PAIRWISE);
    assertThat(plan.diagnostics().optimization("q4-kernel"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
  }

  @Test
  void rejectsManuallyConstructedUnsignedPairwisePlanWithoutVectorCapability() {
    RuntimeFingerprint runtime =
        new RuntimeFingerprint(
            "25.0.3",
            "OpenJDK 64-Bit Server VM",
            "Eclipse Adoptium",
            "25.0.3+9",
            "hotspot-c2",
            "Linux",
            "amd64",
            128,
            8);
    ModelTopology topology = uniformTopology(GgufTensorType.Q4_0);
    PureJavaExecutionPlan valid =
        ExecutionPlanner.plan(runtime, topology, PureJavaPlanConfiguration.defaults());

    assertThatThrownBy(
            () ->
                new PureJavaExecutionPlan(
                    runtime,
                    topology,
                    valid.groupedProjections(),
                    valid.mixedKProjections(),
                    GgufQ4Kernel.UNSIGNED_PAIRWISE,
                    valid.prefillBatchSize(),
                    valid.finalLayerPrefillPruning(),
                    valid.finalLayerKvOnlyPrefill(),
                    valid.batchedAttentionScores(),
                    valid.batchedAttentionValues(),
                    valid.stagedQuantizedFfn(),
                    valid.stagedQuantizedLayer(),
                    valid.blockMajorQ8Activations(),
                    valid.parallelQ8FfnPreparation(),
                    valid.diagnostics()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("execution plan runtime");
  }

  @Test
  void fallsBackToWidenedWhenTheRuntimeCannotExecuteShortPairwise() {
    RuntimeFingerprint runtime =
        new RuntimeFingerprint(
            "25.0.3",
            "OpenJDK 64-Bit Server VM",
            "Eclipse Adoptium",
            "25.0.3+9",
            "hotspot-c2",
            "Linux",
            "amd64",
            128,
            8);
    PureJavaPlanConfiguration configuration =
        new PureJavaPlanConfiguration(
            true,
            true,
            GgufQ4Kernel.SHORT_PAIRWISE,
            32,
            true,
            true,
            false,
            false,
            false,
            false,
            false,
            false);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(runtime, uniformTopology(GgufTensorType.Q4_0), configuration);

    assertThat(plan.q4Kernel()).isEqualTo(GgufQ4Kernel.WIDENED);
    assertThat(plan.diagnostics().optimization("q4-kernel"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
  }

  @Test
  void plansTheMixedKQuantProjectionKernelOnlyForEligibleLayers() {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(
            GgufTensorType.Q4_K,
            GgufTensorType.Q4_K,
            GgufTensorType.Q6_K,
            GgufTensorType.Q6_K,
            GgufTensorType.Q4_K,
            GgufTensorType.Q4_K,
            GgufTensorType.Q6_K);
    ModelTopology topology = new ModelTopology("minicpm", 1024, 256, 256, List.of(layer), true);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"), topology, PureJavaPlanConfiguration.defaults());

    assertThat(plan.groupedProjections()).isTrue();
    assertThat(plan.mixedKProjections()).isTrue();
    assertThat(plan.finalLayerPrefillPruning()).isFalse();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("grouped-projections"))
        .hasValueSatisfying(
            decision ->
                assertThat(decision.settings())
                    .containsEntry("qkv", "grouped")
                    .containsEntry("gate-up", "grouped"));
    assertThat(plan.diagnostics().optimization("mixed-k-projections"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("eligible-layers", "1")
                  .containsEntry("formats", "Q4_K,Q4_K,Q6_K");
            });
    assertThat(plan.diagnostics().optimization("final-layer-prefill-pruning"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
  }

  @Test
  void explicitOverrideDisablesOnlyTheMixedKQuantProjectionKernel() {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(
            GgufTensorType.Q4_K,
            GgufTensorType.Q4_K,
            GgufTensorType.Q6_K,
            GgufTensorType.Q6_K,
            GgufTensorType.Q4_K,
            GgufTensorType.Q4_K,
            GgufTensorType.Q6_K);
    ModelTopology topology = new ModelTopology("minicpm", 1024, 256, 256, List.of(layer), true);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"),
            topology,
            new PureJavaPlanConfiguration(
                true,
                false,
                GgufQ4Kernel.WIDENED,
                32,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                false));

    assertThat(plan.groupedProjections()).isTrue();
    assertThat(plan.mixedKProjections()).isFalse();
    assertThat(plan.diagnostics().optimization("mixed-k-projections"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
  }

  @Test
  void explicitOverridesDisableOtherwiseEligibleOptimizations() {
    PureJavaPlanConfiguration configuration =
        new PureJavaPlanConfiguration(
            false,
            true,
            GgufQ4Kernel.WIDENED,
            1,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"), uniformTopology(GgufTensorType.Q4_0), configuration);

    assertThat(plan.groupedProjections()).isFalse();
    assertThat(plan.prefillBatchSize()).isEqualTo(1);
    assertThat(plan.finalLayerPrefillPruning()).isFalse();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("grouped-projections"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
    assertThat(plan.diagnostics().optimization("batched-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
    assertThat(plan.diagnostics().optimization("final-layer-prefill-pruning"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
  }

  @Test
  void explicitOverrideKeepsFinalFfnPruningWhileDisablingKvOnlyPrefill() {
    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"),
            uniformTopology(GgufTensorType.Q4_0),
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                32,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                false));

    assertThat(plan.finalLayerPrefillPruning()).isTrue();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("final-layer-prefill-pruning"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
  }

  @Test
  void kvOnlyPrefillRequiresFinalFfnPruning() {
    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"),
            uniformTopology(GgufTensorType.Q4_0),
            new PureJavaPlanConfiguration(
                true,
                true,
                GgufQ4Kernel.WIDENED,
                32,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false));

    assertThat(plan.finalLayerPrefillPruning()).isFalse();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
  }

  @Test
  void kvOnlyPrefillRejectsAnUngatedFinalAttentionLayout() {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0);
    ModelTopology topology = new ModelTopology("llama", 1024, 128, 128, List.of(layer), true);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"), topology, PureJavaPlanConfiguration.defaults());

    assertThat(plan.finalLayerPrefillPruning()).isTrue();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED);
              assertThat(decision.settings()).containsEntry("formats", "Q4_0,Q4_0,Q4_0,Q8_0");
            });
  }

  @Test
  void kvOnlyPrefillRejectsAnUngatedCrossFormatFinalLayer() {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q8_0);
    ModelTopology topology = new ModelTopology("llama", 1024, 128, 128, List.of(layer), true);

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"), topology, PureJavaPlanConfiguration.defaults());

    assertThat(plan.finalLayerPrefillPruning()).isTrue();
    assertThat(plan.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(plan.diagnostics().optimization("final-layer-kv-only-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
  }

  @Test
  void unsupportedTensorTopologyUsesConservativeFallbacks() {
    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"),
            uniformTopology(GgufTensorType.F32),
            PureJavaPlanConfiguration.defaults());

    assertThat(plan.groupedProjections()).isFalse();
    assertThat(plan.prefillBatchSize()).isEqualTo(1);
    assertThat(plan.diagnostics().optimization("grouped-projections"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
    assertThat(plan.diagnostics().optimization("batched-prefill"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
  }

  @Test
  void enablesStagedQuantizedFfnOnlyForEligiblePersistentBatchedPlans() {
    PureJavaPlanConfiguration recommended =
        PureJavaPlanConfiguration.from(
            Map.of(), Map.of(PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY, "true"));

    PureJavaExecutionPlan enabled =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q4_0), recommended);

    assertThat(enabled.stagedQuantizedFfn()).isTrue();
    assertThat(enabled.diagnostics().optimization("staged-quantized-ffn"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("eligible-layers", "1")
                  .containsEntry("stages", "2");
            });

    PureJavaExecutionPlan q8 =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q8_0), recommended);
    assertThat(q8.stagedQuantizedFfn()).isTrue();

    PureJavaExecutionPlan unsupported =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q5_0), recommended);
    assertThat(unsupported.stagedQuantizedFfn()).isFalse();
    assertThat(unsupported.diagnostics().optimization("staged-quantized-ffn"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
    assertThat(PureJavaPlanConfiguration.defaults().stagedQuantizedFfn()).isFalse();
  }

  @Test
  void enablesStagedQuantizedLayerOnlyForEligiblePersistentBatchedPlans() {
    PureJavaPlanConfiguration recommended =
        PureJavaPlanConfiguration.from(
            Map.of(), Map.of(PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "true"));

    PureJavaExecutionPlan enabled =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q4_0), recommended);

    assertThat(enabled.stagedQuantizedLayer()).isTrue();
    assertThat(enabled.diagnostics().optimization("staged-quantized-layer"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("eligible-layers", "1")
                  .containsEntry("stages", "7");
            });

    PureJavaExecutionPlan q8 =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q8_0), recommended);
    assertThat(q8.stagedQuantizedLayer()).isTrue();

    PureJavaExecutionPlan unsupported =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q5_0), recommended);
    assertThat(unsupported.stagedQuantizedLayer()).isFalse();
    assertThat(unsupported.diagnostics().optimization("staged-quantized-layer"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));

    ModelTopology.LayerTopology q8Output =
        new ModelTopology.LayerTopology(
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q4_0);
    PureJavaExecutionPlan mixed =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"),
            new ModelTopology("llama", 1024, 128, 128, List.of(q8Output), true),
            recommended);
    assertThat(mixed.stagedQuantizedLayer()).isTrue();
    assertThat(mixed.topology().stagedQuantizedFfnLayers()).isEqualTo(1);
    assertThat(PureJavaPlanConfiguration.defaults().stagedQuantizedLayer()).isFalse();
  }

  @Test
  void enablesBlockMajorActivationsOnlyForRequestedStagedQ8Prefill() {
    PureJavaPlanConfiguration recommended =
        PureJavaPlanConfiguration.from(
            Map.of(),
            Map.of(
                PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
                "true",
                PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY,
                "true"));

    PureJavaExecutionPlan q8 =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q8_0), recommended);

    assertThat(q8.blockMajorQ8Activations()).isTrue();
    assertThat(q8.diagnostics().optimization("block-major-q8-activations"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("layout", "block-major-bytes")
                  .containsEntry("prefill-batch-size", "32");
            });

    PureJavaExecutionPlan q4 =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q4_0), recommended);
    assertThat(q4.blockMajorQ8Activations()).isFalse();
    assertThat(PureJavaPlanConfiguration.defaults().blockMajorQ8Activations()).isFalse();
  }

  @Test
  void enablesParallelQ8FfnPreparationOnlyForRequestedAllQ8StagedBlockMajorLayers() {
    PureJavaPlanConfiguration recommended =
        PureJavaPlanConfiguration.from(
            Map.of(),
            Map.of(
                PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
                "true",
                PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY,
                "true",
                PureJavaPlanConfiguration.PARALLEL_Q8_FFN_PREPARATION_PROPERTY,
                "true"));

    PureJavaExecutionPlan q8 =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q8_0), recommended);

    assertThat(q8.parallelQ8FfnPreparation()).isTrue();
    assertThat(q8.diagnostics().optimization("parallel-q8-ffn-preparation"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(decision.settings())
                  .containsEntry("partition", "batch-rows")
                  .containsEntry("partitions", "8")
                  .containsEntry("eligible-layers", "1");
            });

    PureJavaExecutionPlan singleParticipant =
        ExecutionPlanner.plan(
            runtime("graal-jvmci", true, 1), uniformTopology(GgufTensorType.Q8_0), recommended);
    assertThat(singleParticipant.parallelQ8FfnPreparation()).isFalse();
    assertThat(singleParticipant.diagnostics().optimization("parallel-q8-ffn-preparation"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason()).contains("at least two GGUF participants");
              assertThat(decision.settings()).containsEntry("partitions", "1");
            });

    PureJavaExecutionPlan q4 =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q4_0), recommended);
    assertThat(q4.parallelQ8FfnPreparation()).isFalse();
    assertThat(q4.diagnostics().optimization("parallel-q8-ffn-preparation"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));

    ModelTopology.LayerTopology mixedFfnInput =
        new ModelTopology.LayerTopology(
            GgufTensorType.Q8_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q8_0,
            GgufTensorType.Q4_0,
            GgufTensorType.Q8_0);
    PureJavaExecutionPlan mixed =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"),
            new ModelTopology("llama", 1024, 128, 128, List.of(mixedFfnInput), true),
            recommended);
    assertThat(mixed.stagedQuantizedLayer()).isTrue();
    assertThat(mixed.blockMajorQ8Activations()).isTrue();
    assertThat(mixed.parallelQ8FfnPreparation()).isFalse();

    PureJavaPlanConfiguration withoutBlockMajor =
        PureJavaPlanConfiguration.from(
            Map.of(),
            Map.of(
                PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
                "true",
                PureJavaPlanConfiguration.PARALLEL_Q8_FFN_PREPARATION_PROPERTY,
                "true"));
    PureJavaExecutionPlan packed =
        ExecutionPlanner.plan(
            runtime("graal-jvmci"), uniformTopology(GgufTensorType.Q8_0), withoutBlockMajor);
    assertThat(packed.parallelQ8FfnPreparation()).isFalse();
    assertThat(packed.diagnostics().optimization("parallel-q8-ffn-preparation"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason()).contains("block-major Q8 activations");
            });
    assertThat(PureJavaPlanConfiguration.defaults().parallelQ8FfnPreparation()).isFalse();
  }

  @Test
  void rejectsStagedQuantizedFfnWhenParallelExecutionOrWeightSharingIsUnavailable() {
    PureJavaPlanConfiguration recommended =
        PureJavaPlanConfiguration.from(
            Map.of(), Map.of(PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY, "true"));

    PureJavaExecutionPlan parallelDisabled =
        ExecutionPlanner.plan(
            runtime("graal-jvmci", false), uniformTopology(GgufTensorType.Q4_0), recommended);
    assertThat(parallelDisabled.stagedQuantizedFfn()).isFalse();
    assertThat(parallelDisabled.diagnostics().optimization("staged-quantized-ffn"))
        .hasValueSatisfying(
            decision -> {
              assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(decision.reason()).contains("parallel GGUF execution");
            });

    ModelTopology confined =
        new ModelTopology(
            "llama", 1024, 128, 128, uniformTopology(GgufTensorType.Q4_0).layers(), false);
    PureJavaExecutionPlan weightsNotShareable =
        ExecutionPlanner.plan(runtime("graal-jvmci"), confined, recommended);
    assertThat(weightsNotShareable.stagedQuantizedFfn()).isFalse();
    assertThat(weightsNotShareable.diagnostics().optimization("staged-quantized-ffn"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.UNSUPPORTED));
  }

  @Test
  void runtimeClassificationIsPropertyBasedAndDoesNotBenchmarkAtStartup() {
    RuntimeFingerprint graal =
        RuntimeFingerprint.from(
            Map.of(
                "java.version", "25.0.3",
                "java.vm.name", "OpenJDK 64-Bit Server VM",
                "java.vm.vendor", "GraalVM Community",
                "java.vm.version", "25.0.3+9-jvmci-25.1-b19",
                "java.runtime.name", "OpenJDK Runtime Environment GraalVM CE",
                "os.name", "Linux",
                "os.arch", "amd64",
                "models.runtime.cpuModel", "AMD EPYC-Milan Processor"),
            256,
            8);

    assertThat(graal.compiler()).isEqualTo("graal-jvmci");
    assertThat(graal.vectorBits()).isEqualTo(256);
    assertThat(graal.processors()).isEqualTo(8);
    assertThat(graal.asEnvironment())
        .containsEntry("architecture", "amd64")
        .containsEntry("java-feature", "25")
        .containsEntry("compiler", "graal-jvmci")
        .containsEntry("cpu-model", "AMD EPYC-Milan Processor")
        .containsEntry("vm-version", "25.0.3+9-jvmci-25.1-b19");
  }

  @Test
  void invalidConfigurationIsRejectedWithoutAStartupProbe() {
    assertThatThrownBy(
            () ->
                new PureJavaPlanConfiguration(
                    true,
                    true,
                    GgufQ4Kernel.WIDENED,
                    0,
                    true,
                    true,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.prefillBatchSize");
    assertThatThrownBy(() -> PureJavaPlanConfiguration.groupedProjections("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.groupedProjections");
    assertThatThrownBy(() -> PureJavaPlanConfiguration.mixedKProjections("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.mixedKProjections");
    assertThat(PureJavaPlanConfiguration.mixedKProjections(null)).isTrue();
    assertThat(PureJavaPlanConfiguration.mixedKProjections("false")).isFalse();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.prefillBatchSize("many"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.prefillBatchSize");
    assertThatThrownBy(() -> PureJavaPlanConfiguration.prefillBatchSize("0"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.prefillBatchSize");
    assertThat(PureJavaPlanConfiguration.prefillBatchSize(null)).isEqualTo(32);
    assertThat(PureJavaPlanConfiguration.prefillBatchSize("16")).isEqualTo(16);
    assertThatThrownBy(() -> PureJavaPlanConfiguration.finalLayerPrefillPruning("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.finalLayerPrefillPruning");
    assertThat(PureJavaPlanConfiguration.finalLayerPrefillPruning(null)).isTrue();
    assertThat(PureJavaPlanConfiguration.finalLayerPrefillPruning("false")).isFalse();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.finalLayerKvOnlyPrefill("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.finalLayerKvOnlyPrefill");
    assertThat(PureJavaPlanConfiguration.finalLayerKvOnlyPrefill(null)).isTrue();
    assertThat(PureJavaPlanConfiguration.finalLayerKvOnlyPrefill("false")).isFalse();
    assertThat(PureJavaPlanConfiguration.batchedAttentionValues(null)).isFalse();
    assertThat(PureJavaPlanConfiguration.batchedAttentionValues("true")).isTrue();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.batchedAttentionValues("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.batchedAttentionValues");
    assertThat(PureJavaPlanConfiguration.batchedAttentionScores(null)).isFalse();
    assertThat(PureJavaPlanConfiguration.batchedAttentionScores("true")).isTrue();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.batchedAttentionScores("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.batchedAttentionScores");
    assertThat(PureJavaPlanConfiguration.stagedQuantizedFfn(null)).isFalse();
    assertThat(PureJavaPlanConfiguration.stagedQuantizedFfn("true")).isTrue();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.stagedQuantizedFfn("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.stagedQuantizedFfn");
    assertThat(PureJavaPlanConfiguration.stagedQuantizedLayer(null)).isFalse();
    assertThat(PureJavaPlanConfiguration.stagedQuantizedLayer("true")).isTrue();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.stagedQuantizedLayer("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.stagedQuantizedLayer");
    assertThat(PureJavaPlanConfiguration.blockMajorQ8Activations(null)).isFalse();
    assertThat(PureJavaPlanConfiguration.blockMajorQ8Activations("true")).isTrue();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.blockMajorQ8Activations("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.blockMajorQ8Activations");
    assertThat(PureJavaPlanConfiguration.parallelQ8FfnPreparation(null)).isFalse();
    assertThat(PureJavaPlanConfiguration.parallelQ8FfnPreparation("true")).isTrue();
    assertThatThrownBy(() -> PureJavaPlanConfiguration.parallelQ8FfnPreparation("sometimes"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.parallelQ8FfnPreparation");
    assertThat(PureJavaPlanConfiguration.q4Kernel(null)).isEqualTo(GgufQ4Kernel.WIDENED);
    assertThat(PureJavaPlanConfiguration.q4Kernel("short-pairwise"))
        .isEqualTo(GgufQ4Kernel.SHORT_PAIRWISE);
    assertThat(PureJavaPlanConfiguration.q4Kernel("unsigned-pairwise"))
        .isEqualTo(GgufQ4Kernel.UNSIGNED_PAIRWISE);
    assertThat(PureJavaPlanConfiguration.q4Kernel("widened")).isEqualTo(GgufQ4Kernel.WIDENED);
    assertThatThrownBy(() -> PureJavaPlanConfiguration.q4Kernel("fastest"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.q4Kernel");
    assertThatThrownBy(() -> PureJavaPlanConfiguration.q4Kernel(" "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("models.purejava.q4Kernel");
  }

  @Test
  void modelRecommendationsConfigureThePlanAndDeploymentSettingsTakePrecedence() {
    Map<String, String> recommendations =
        Map.ofEntries(
            Map.entry(PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY, "false"),
            Map.entry(PureJavaPlanConfiguration.MIXED_K_PROJECTIONS_PROPERTY, "false"),
            Map.entry(PureJavaPlanConfiguration.Q4_KERNEL_PROPERTY, "short-pairwise"),
            Map.entry(PureJavaPlanConfiguration.PREFILL_BATCH_SIZE_PROPERTY, "16"),
            Map.entry(PureJavaPlanConfiguration.FINAL_LAYER_PREFILL_PRUNING_PROPERTY, "false"),
            Map.entry(PureJavaPlanConfiguration.FINAL_LAYER_KV_ONLY_PREFILL_PROPERTY, "false"),
            Map.entry(PureJavaPlanConfiguration.BATCHED_ATTENTION_VALUES_PROPERTY, "true"),
            Map.entry(PureJavaPlanConfiguration.BATCHED_ATTENTION_SCORES_PROPERTY, "true"),
            Map.entry(PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY, "true"),
            Map.entry(PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "true"),
            Map.entry(PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY, "true"),
            Map.entry(PureJavaPlanConfiguration.PARALLEL_Q8_FFN_PREPARATION_PROPERTY, "true"));

    PureJavaPlanConfiguration recommended =
        PureJavaPlanConfiguration.from(Map.of(), recommendations);

    assertThat(recommended.groupedProjections()).isFalse();
    assertThat(recommended.mixedKProjections()).isFalse();
    assertThat(recommended.q4Kernel()).isEqualTo(GgufQ4Kernel.SHORT_PAIRWISE);
    assertThat(recommended.prefillBatchSize()).isEqualTo(16);
    assertThat(recommended.finalLayerPrefillPruning()).isFalse();
    assertThat(recommended.finalLayerKvOnlyPrefill()).isFalse();
    assertThat(recommended.batchedAttentionValues()).isTrue();
    assertThat(recommended.batchedAttentionScores()).isTrue();
    assertThat(recommended.stagedQuantizedFfn()).isTrue();
    assertThat(recommended.stagedQuantizedLayer()).isTrue();
    assertThat(recommended.blockMajorQ8Activations()).isTrue();
    assertThat(recommended.parallelQ8FfnPreparation()).isTrue();

    PureJavaPlanConfiguration overridden =
        PureJavaPlanConfiguration.from(
            Map.of(
                PureJavaPlanConfiguration.MIXED_K_PROJECTIONS_PROPERTY,
                "true",
                PureJavaPlanConfiguration.Q4_KERNEL_PROPERTY,
                "widened",
                PureJavaPlanConfiguration.BATCHED_ATTENTION_VALUES_PROPERTY,
                "false",
                PureJavaPlanConfiguration.BATCHED_ATTENTION_SCORES_PROPERTY,
                "false",
                PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY,
                "false",
                PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY,
                "false",
                PureJavaPlanConfiguration.BLOCK_MAJOR_Q8_ACTIVATIONS_PROPERTY,
                "false",
                PureJavaPlanConfiguration.PARALLEL_Q8_FFN_PREPARATION_PROPERTY,
                "false"),
            recommendations);

    assertThat(overridden.mixedKProjections()).isTrue();
    assertThat(overridden.q4Kernel()).isEqualTo(GgufQ4Kernel.WIDENED);
    assertThat(overridden.batchedAttentionValues()).isFalse();
    assertThat(overridden.batchedAttentionScores()).isFalse();
    assertThat(overridden.stagedQuantizedFfn()).isFalse();
    assertThat(overridden.stagedQuantizedLayer()).isFalse();
    assertThat(overridden.blockMajorQ8Activations()).isFalse();
    assertThat(overridden.parallelQ8FfnPreparation()).isFalse();
    assertThatThrownBy(
            () ->
                PureJavaPlanConfiguration.from(
                    Map.of(), Map.of("models.purejava.unrecognized", "true")))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unsupported pure-Java recommendation");
    assertThatThrownBy(
            () ->
                PureJavaPlanConfiguration.from(
                    Map.of("models.purejava.misspelled", "true"), Map.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Unsupported pure-Java deployment setting");
  }

  @Test
  void equivalentInputsAlwaysProduceEqualPlans() {
    RuntimeFingerprint runtime = runtime("hotspot-c2");
    ModelTopology topology = uniformTopology(GgufTensorType.Q4_K);
    PureJavaPlanConfiguration configuration = PureJavaPlanConfiguration.defaults();

    assertThat(ExecutionPlanner.plan(runtime, topology, configuration))
        .isEqualTo(ExecutionPlanner.plan(runtime, topology, configuration));
  }

  @Test
  void rejectsAPlanThatContradictsItsTensorTopology() {
    RuntimeFingerprint runtime = runtime("hotspot-c2");
    ModelTopology unsupported = uniformTopology(GgufTensorType.F32);
    PureJavaExecutionPlan valid =
        ExecutionPlanner.plan(runtime, unsupported, PureJavaPlanConfiguration.defaults());

    assertThatThrownBy(
            () ->
                new PureJavaExecutionPlan(
                    runtime,
                    unsupported,
                    true,
                    false,
                    GgufQ4Kernel.WIDENED,
                    32,
                    true,
                    true,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    valid.diagnostics()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("topology");
  }

  @Test
  void rejectsKvOnlyPrefillWithoutFinalFfnPruning() {
    RuntimeFingerprint runtime = runtime("hotspot-c2");
    ModelTopology topology = uniformTopology(GgufTensorType.Q4_0);
    PureJavaExecutionPlan valid =
        ExecutionPlanner.plan(runtime, topology, PureJavaPlanConfiguration.defaults());

    assertThatThrownBy(
            () ->
                new PureJavaExecutionPlan(
                    runtime,
                    topology,
                    true,
                    false,
                    GgufQ4Kernel.WIDENED,
                    32,
                    false,
                    true,
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    valid.diagnostics()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("K/V-only");
  }

  private static RuntimeFingerprint runtime(String compiler) {
    return runtime(compiler, true);
  }

  private static RuntimeFingerprint runtime(String compiler, boolean ggufParallel) {
    return runtime(compiler, ggufParallel, 8);
  }

  private static RuntimeFingerprint runtime(
      String compiler, boolean ggufParallel, int ggufThreads) {
    return new RuntimeFingerprint(
        "25.0.3",
        "OpenJDK 64-Bit Server VM",
        "Eclipse Adoptium",
        "25.0.3+9",
        compiler,
        "Linux",
        "amd64",
        "AMD EPYC-Milan Processor",
        "test-vector",
        true,
        256,
        256,
        false,
        false,
        false,
        true,
        true,
        ggufParallel,
        "persistent",
        ggufThreads,
        2,
        8);
  }

  private static ModelTopology uniformTopology(GgufTensorType type) {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(type, type, type, type, type, type, type);
    return new ModelTopology("llama", 1024, 128, 128, List.of(layer), true);
  }
}
