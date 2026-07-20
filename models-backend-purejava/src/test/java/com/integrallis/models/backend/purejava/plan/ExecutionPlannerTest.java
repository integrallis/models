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

    assertThat(plan.groupedProjections()).isTrue();
    assertThat(plan.prefillBatchSize()).isEqualTo(32);
    assertThat(plan.finalLayerPrefillPruning()).isTrue();
    assertThat(plan.finalLayerKvOnlyPrefill()).isTrue();
    assertThat(plan.diagnostics().environment()).containsEntry("compiler", "hotspot-c2");
    assertThat(plan.diagnostics().environment())
        .containsEntry("vector-provider", "test-vector")
        .containsEntry("active-vector-bits", "256")
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
    ModelTopology topology = new ModelTopology("minicpm", 1024, 256, 256, List.of(layer));

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
    ModelTopology topology = new ModelTopology("minicpm", 1024, 256, 256, List.of(layer));

    PureJavaExecutionPlan plan =
        ExecutionPlanner.plan(
            runtime("hotspot-c2"),
            topology,
            new PureJavaPlanConfiguration(true, false, 32, true, true));

    assertThat(plan.groupedProjections()).isTrue();
    assertThat(plan.mixedKProjections()).isFalse();
    assertThat(plan.diagnostics().optimization("mixed-k-projections"))
        .hasValueSatisfying(
            decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.DISABLED));
  }

  @Test
  void explicitOverridesDisableOtherwiseEligibleOptimizations() {
    PureJavaPlanConfiguration configuration =
        new PureJavaPlanConfiguration(false, true, 1, false, false);

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
            new PureJavaPlanConfiguration(true, true, 32, true, false));

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
            new PureJavaPlanConfiguration(true, true, 32, false, true));

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
    ModelTopology topology = new ModelTopology("llama", 1024, 128, 128, List.of(layer));

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
    ModelTopology topology = new ModelTopology("llama", 1024, 128, 128, List.of(layer));

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
    assertThatThrownBy(() -> new PureJavaPlanConfiguration(true, true, 0, true, true))
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
                    runtime, unsupported, true, false, 32, true, true, valid.diagnostics()))
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
                    runtime, topology, true, false, 32, false, true, valid.diagnostics()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("K/V-only");
  }

  private static RuntimeFingerprint runtime(String compiler) {
    return new RuntimeFingerprint(
        "25.0.3",
        "OpenJDK 64-Bit Server VM",
        "Eclipse Adoptium",
        "25.0.3+9",
        compiler,
        "Linux",
        "amd64",
        256,
        8);
  }

  private static ModelTopology uniformTopology(GgufTensorType type) {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(type, type, type, type, type, type, type);
    return new ModelTopology("llama", 1024, 128, 128, List.of(layer));
  }
}
