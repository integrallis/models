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

import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliff;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffRecording;
import com.integrallis.models.backend.purejava.diagnostics.PerformanceCliffs;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.GgufQ8BlockMajorKernel;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Each planner fallback reports its named cliff exactly once; the fast plan reports nothing. */
@Tag("unit")
class ExecutionPlannerPerformanceCliffTest {

  @Test
  void fastPlanReportsNoCliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int plan = 0; plan < 3; plan++) {
        ExecutionPlanner.plan(
            runtime(true, 256, 256, "persistent"),
            uniformTopology(GgufTensorType.Q4_0),
            configuration(GgufQ4Kernel.SHORT_PAIRWISE, 32));
      }

      assertThat(recording.reasons()).isEmpty();
      assertThat(PerformanceCliffs.reported()).isEmpty();
    }
  }

  @Test
  void scalarVectorProviderReportsVectorApiUnavailableOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int plan = 0; plan < 3; plan++) {
        ExecutionPlanner.plan(
            runtime(false, 256, 0, "persistent"),
            uniformTopology(GgufTensorType.Q4_0),
            PureJavaPlanConfiguration.defaults());
      }

      assertThat(recording.count(PerformanceCliff.VECTOR_API_UNAVAILABLE)).isEqualTo(1);
      assertThat(recording.count(PerformanceCliff.VECTOR_WIDTH_CAPPED)).isZero();
      assertThat(PerformanceCliffs.environment())
          .containsKey("performance-cliff.vector-api-unavailable");
    }
  }

  @Test
  void activeWidthBelowPreferredReportsVectorWidthCappedOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int plan = 0; plan < 3; plan++) {
        ExecutionPlanner.plan(
            runtime(true, 512, 256, "persistent"),
            uniformTopology(GgufTensorType.Q4_0),
            PureJavaPlanConfiguration.defaults());
      }

      assertThat(recording.count(PerformanceCliff.VECTOR_WIDTH_CAPPED)).isEqualTo(1);
      assertThat(PerformanceCliffs.reported().get(PerformanceCliff.VECTOR_WIDTH_CAPPED))
          .contains("active-vector-bits=256")
          .contains("preferred-vector-bits=512");
    }
  }

  @Test
  void nonPersistentExecutorReportsOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int plan = 0; plan < 3; plan++) {
        ExecutionPlanner.plan(
            runtime(true, 256, 256, "fork-join"),
            uniformTopology(GgufTensorType.Q4_0),
            PureJavaPlanConfiguration.defaults());
      }

      assertThat(recording.count(PerformanceCliff.PERSISTENT_EXECUTOR_NOT_USED)).isEqualTo(1);
      assertThat(PerformanceCliffs.reported().get(PerformanceCliff.PERSISTENT_EXECUTOR_NOT_USED))
          .contains("executor=fork-join");
    }
  }

  @Test
  void serialGgufRowsOnAMultiProcessorHostReportOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int plan = 0; plan < 3; plan++) {
        ExecutionPlanner.plan(
            runtime(true, 256, 256, "persistent", false, 8),
            uniformTopology(GgufTensorType.Q4_0),
            PureJavaPlanConfiguration.defaults());
      }

      assertThat(recording.count(PerformanceCliff.GGUF_PARALLEL_DISABLED)).isEqualTo(1);
      assertThat(recording.count(PerformanceCliff.PERSISTENT_EXECUTOR_NOT_USED)).isZero();
      assertThat(PerformanceCliffs.reported().get(PerformanceCliff.GGUF_PARALLEL_DISABLED))
          .contains("gguf-parallel=false")
          .contains("processors=8");
    }
  }

  @Test
  void serialGgufRowsOnOneProcessorAreNotACliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      ExecutionPlanner.plan(
          runtime(true, 256, 256, "persistent", false, 1),
          uniformTopology(GgufTensorType.Q4_0),
          PureJavaPlanConfiguration.defaults());

      assertThat(recording.count(PerformanceCliff.GGUF_PARALLEL_DISABLED)).isZero();
    }
  }

  @Test
  void unsupportedPairwiseQ4KernelReportsFallbackOnce() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      for (int plan = 0; plan < 3; plan++) {
        PureJavaExecutionPlan selected =
            ExecutionPlanner.plan(
                runtime(true, 128, 128, "persistent"),
                uniformTopology(GgufTensorType.Q4_0),
                configuration(GgufQ4Kernel.SHORT_PAIRWISE, 32));
        assertThat(selected.q4Kernel()).isEqualTo(GgufQ4Kernel.WIDENED);
      }

      assertThat(recording.count(PerformanceCliff.Q4_PAIRWISE_KERNEL_UNSUPPORTED)).isEqualTo(1);
    }
  }

  @Test
  void explicitlyWidenedQ4KernelIsAChoiceNotACliff() {
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      ExecutionPlanner.plan(
          runtime(true, 128, 128, "persistent"),
          uniformTopology(GgufTensorType.Q4_0),
          configuration(GgufQ4Kernel.WIDENED, 32));

      assertThat(recording.count(PerformanceCliff.Q4_PAIRWISE_KERNEL_UNSUPPORTED)).isZero();
    }
  }

  @Test
  void plannerLeavesBatchedPrefillCliffToTheForwardPassThatOwnsTheActualTopology() {
    // Mapped architectures (bert, gpt-oss, mobilemoe, needle2) plan over a neutral all-F32
    // topology that is not their real tensor layout, so the planner must not report this cliff.
    try (PerformanceCliffRecording recording = PerformanceCliffRecording.start()) {
      ExecutionPlanner.plan(
          runtime(true, 256, 256, "persistent"),
          ModelTopology.mappedArchitecture("mobilemoe", 64, 16, 16, 2),
          PureJavaPlanConfiguration.defaults());

      assertThat(recording.reasons()).isEmpty();
    }
  }

  private static RuntimeFingerprint runtime(
      boolean vectorApi, int preferredBits, int activeBits, String executor) {
    return runtime(vectorApi, preferredBits, activeBits, executor, true, 8);
  }

  private static RuntimeFingerprint runtime(
      boolean vectorApi,
      int preferredBits,
      int activeBits,
      String executor,
      boolean ggufParallel,
      int processors) {
    return new RuntimeFingerprint(
        "25.0.3",
        "OpenJDK 64-Bit Server VM",
        "Eclipse Adoptium",
        "25.0.3+9",
        "hotspot-c2",
        "Linux",
        "amd64",
        "AMD EPYC-Milan Processor",
        vectorApi ? "panama" : "scalar",
        vectorApi,
        preferredBits,
        activeBits,
        false,
        false,
        false,
        activeBits >= 256,
        activeBits >= 256,
        ggufParallel,
        executor,
        8,
        2,
        processors);
  }

  private static PureJavaPlanConfiguration configuration(GgufQ4Kernel q4Kernel, int batchSize) {
    return new PureJavaPlanConfiguration(
        true,
        true,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
        batchSize,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        GgufQ8BlockMajorKernel.SCATTERED,
        false);
  }

  private static ModelTopology uniformTopology(GgufTensorType type) {
    ModelTopology.LayerTopology layer =
        new ModelTopology.LayerTopology(type, type, type, type, type, type, type);
    return new ModelTopology("llama", 1024, 128, 128, List.of(layer), true);
  }
}
