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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Device-independent capacity arithmetic for 27B-class Q4_K_M models.
 *
 * <p>Every model constant below is read from this repository, not assumed:
 *
 * <ul>
 *   <li>weights {@code 1_650_027_640 + 15_130_165_248} are the resident and routed-expert byte
 *       counts asserted in {@code Gemma4LargeModelFixtureSlowTest} for the pinned {@code
 *       gemma-4-26B-A4B-it-Q4_K_M.gguf};
 *   <li>the KV geometry (30 layers, 25 sliding with keyDim 2048, 5 full with keyDim 1024, ring
 *       capacity {@code slidingWindow 1024 + prefillCapacity 1}) is asserted in {@code
 *       Gemma4ConfigTest.parsesThePinnedGemma426BA4BMetadata} and built in {@code
 *       Gemma4KvCache.create};
 *   <li>the retained plan count is the number of distinct {@code (weight, batch shape)} keys {@code
 *       TornadoGgufBatchedMatrixKernel} would create for this graph.
 * </ul>
 *
 * <p>Device global-memory figures are driver-reported values, which sit below the nameplate
 * capacity: the A40-4Q gate reported 3,917.7 MiB on a 4,096 MiB profile.
 */
class LargeModelEligibilityTest {

  private static final long MIB = 1024L * 1024L;
  private static final long GIB = 1024L * MIB;

  /** Gemma 4 26B-A4B IT Q4_K_M: resident tensors plus routed experts. */
  private static final long GEMMA4_WEIGHT_BYTES = 1_650_027_640L + 15_130_165_248L;

  /**
   * Distinct retained plans: 30 layers x (128 experts x 2 tensors + 4 resident projections) x 2
   * retained batch shapes.
   */
  private static final int GEMMA4_PLAN_COUNT = 30 * (128 * 2 + 4) * 2;

  /** Device activation, scale, and output buffers summed over every retained plan. */
  private static final long GEMMA4_PLAN_SCRATCH_BYTES = 2_733_284_400L;

  /** Largest single tensor: the 262,144 x 2,816 tied embedding at Q8_0. */
  private static final long GEMMA4_LARGEST_ALLOCATION_BYTES = 262_144L * 2_816L / 32L * 34L;

  private static final long QWEN3_06B_Q4_0_BYTES = 428_970_080L;

  /** KV bytes the {@code LayeredKvCache} would hold at a runtime context length. */
  private static long gemma4KvBytes(int contextLength) {
    long slidingRing = 25L * 2L * 2_048L * 4L * (1_024L + 1L);
    long full = 5L * 2L * 1_024L * 4L * Math.min(contextLength, 262_144L);
    return slidingRing + full;
  }

  private static DeviceMemoryRequest gemma4(int contextLength, int planCount, long planScratch) {
    return DeviceMemoryRequest.detailed("gemma-4-26B-A4B-it-Q4_K_M.gguf")
        .weightBytes(GEMMA4_WEIGHT_BYTES)
        .retainedShapes(2)
        .retainedPlanCount(planCount)
        .planScratchBytes(planScratch)
        .largestAllocationBytes(GEMMA4_LARGEST_ALLOCATION_BYTES)
        .deviceKvCacheBytes(gemma4KvBytes(contextLength))
        .build();
  }

  private static AcceleratorEligibility.DeviceCapabilities gpu(String name, long globalBytes) {
    return new AcceleratorEligibility.DeviceCapabilities(name, "PTX", "GPU", globalBytes, 8L * GIB);
  }

  static Stream<Arguments> deviceCapacityTable() {
    // device, driver-reported global memory, context length, expected outcome marker
    return Stream.of(
        Arguments.of("A40 24 GB", 23L * GIB, 4_096, "short by"),
        Arguments.of("A40 24 GB", 23L * GIB, 32_768, "short by"),
        Arguments.of("A40 24 GB", 23L * GIB, 262_144, "short by"),
        Arguments.of("L40S 48 GB", 44L * GIB, 4_096, "SHARED_WEIGHT_UPLOAD"),
        Arguments.of("L40S 48 GB", 44L * GIB, 32_768, "SHARED_WEIGHT_UPLOAD"),
        Arguments.of("L40S 48 GB", 44L * GIB, 262_144, "SHARED_WEIGHT_UPLOAD"),
        Arguments.of("H100 80 GB", 79L * GIB, 4_096, "eager readiness"),
        Arguments.of("H100 80 GB", 79L * GIB, 262_144, "eager readiness"));
  }

  @ParameterizedTest(name = "{0} at context {2} is refused because of \"{3}\"")
  @MethodSource("deviceCapacityTable")
  void refusesTheShippedPlanShapeForA27bClassModelOnEveryDeviceSize(
      String deviceName, long globalBytes, int contextLength, String expectedReason) {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(gpu(deviceName, globalBytes)),
            gemma4(contextLength, GEMMA4_PLAN_COUNT, GEMMA4_PLAN_SCRATCH_BYTES));

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains(expectedReason);
    assertThat(decision.reason()).contains(deviceName);
  }

  @Test
  void itemisesEveryTermOfTheRefusedBudget() {
    DeviceMemoryRequest request = gemma4(4_096, GEMMA4_PLAN_COUNT, GEMMA4_PLAN_SCRATCH_BYTES);
    DeviceBudget budget =
        AcceleratorEligibility.budget(request, PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL);

    assertThat(budget.deviceWeightBytes()).isEqualTo(2 * GEMMA4_WEIGHT_BYTES);
    assertThat(budget.kvCacheBytes()).isEqualTo(587_612_160L);
    assertThat(budget.planScratchBytes()).isEqualTo(GEMMA4_PLAN_SCRATCH_BYTES);
    assertThat(budget.baseOverheadBytes()).isEqualTo(256 * MIB);
    assertThat(budget.totalBytes()).isEqualTo(37_149_717_792L);
    assertThat(budget.hostWeightCopyBytes()).isEqualTo(2 * GEMMA4_WEIGHT_BYTES);
    assertThat(budget.estimatedPlanCompileTime().toSeconds()).isEqualTo(948L);
  }

  @Test
  void sharedWeightUploadHalvesTheWeightTermButIsNotImplemented() {
    DeviceMemoryRequest request = gemma4(4_096, GEMMA4_PLAN_COUNT, GEMMA4_PLAN_SCRATCH_BYTES);
    DeviceBudget shared =
        AcceleratorEligibility.budget(request, PlanShapeStrategy.SHARED_WEIGHT_UPLOAD);

    assertThat(shared.deviceWeightBytes()).isEqualTo(GEMMA4_WEIGHT_BYTES);
    assertThat(shared.totalBytes()).isEqualTo(20_369_524_904L);
    assertThat(PlanShapeStrategy.SHARED_WEIGHT_UPLOAD.implemented()).isFalse();
    assertThat(PlanShapeStrategy.SHARED_WEIGHT_UPLOAD.limitation()).contains("ByteArray");
    assertThat(PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL.implemented()).isTrue();
    assertThat(PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL.limitation()).isNull();
    assertThat(PlanShapeStrategy.BOUNDED_RESIDENT_WORKING_SET.capacityComputable()).isFalse();
  }

  @Test
  void admitsAnEightyGigabyteDeviceOnceThePerExpertPlansAreGone() {
    int residentOnlyPlans = 30 * 4 * 2;
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(gpu("H100 80 GB", 79L * GIB)), gemma4(4_096, residentOnlyPlans, 90_302_490L));

    assertThat(decision.eligible()).isTrue();
    assertThat(decision.strategy()).isEqualTo(PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL);
    assertThat(decision.budget().estimatedPlanCompileTime().toSeconds()).isEqualTo(14L);
  }

  @Test
  void refusesAFileSizeOnlyBudgetForALargeModel() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(gpu("H100 80 GB", 79L * GIB)),
            DeviceMemoryRequest.ofModelFile(
                "gemma-4-26B-A4B-it-Q4_K_M.gguf", 16_796_015_136L, true));

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("file-size-only device budget");
    assertThat(decision.reason()).contains("15.64 GiB of weights");
    assertThat(decision.reason()).contains("8.00 GiB");
    assertThat(decision.reason()).contains("DeviceMemoryRequest.detailed");
  }

  @Test
  void refusesATensorThatCannotFitInATornadoByteArray() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(gpu("H100 80 GB", 79L * GIB)),
            DeviceMemoryRequest.detailed("oversized.gguf")
                .weightBytes(10L * GIB)
                .retainedShapes(1)
                .retainedPlanCount(8)
                .planScratchBytes(MIB)
                .largestAllocationBytes(3L * GIB)
                .build());

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("largest device allocation is 3.00 GiB");
    assertThat(decision.reason()).contains("2.00 GiB");
    assertThat(decision.reason()).contains("must be split");
  }

  @Test
  void refusesADeviceWhoseSingleAllocationLimitIsTooSmall() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(
                new AcceleratorEligibility.DeviceCapabilities(
                    "A40-4Q", "PTX", "GPU", 79L * GIB, 512L * MIB)),
            gemma4(4_096, 240, 90_302_490L));

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("single allocation of at most 0.50 GiB");
    assertThat(decision.reason()).contains("needs one of 0.73 GiB");
  }

  @Test
  void namesEveryDiscoveredDeviceWhenNoneQualifies() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(
            List.of(
                new AcceleratorEligibility.DeviceCapabilities(
                    "AMD Radeon", "OPENCL", "GPU", 24L * GIB, 4L * GIB),
                new AcceleratorEligibility.DeviceCapabilities(
                    "host CPU", "JAVA", "CPU", 64L * GIB, 16L * GIB)),
            gemma4(4_096, 240, 90_302_490L));

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("PTX or CUDA backend on a GPU device");
    assertThat(decision.reason()).contains("AMD Radeon [OPENCL/GPU]");
    assertThat(decision.reason()).contains("host CPU [JAVA/CPU]");
  }

  @Test
  void reportsNoDevicesWhenTheRuntimeFoundNone() {
    AcceleratorEligibility.Decision decision =
        AcceleratorEligibility.select(List.of(), gemma4(4_096, 240, 90_302_490L));

    assertThat(decision.eligible()).isFalse();
    assertThat(decision.reason()).contains("no devices");
  }

  @Test
  void keepsThePublishedCoarseBudgetForTheQualifiedSmallModel() {
    DeviceMemoryRequest request =
        DeviceMemoryRequest.ofModelFile("Qwen3-0.6B-Q4_0.gguf", QWEN3_06B_Q4_0_BYTES, true);
    DeviceBudget budget =
        AcceleratorEligibility.budget(request, PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL);

    assertThat(request.detailed()).isFalse();
    assertThat(budget.totalBytes()).isEqualTo(1_126_375_616L);
    assertThat(budget.kvCacheBytes()).isZero();
    assertThat(budget.estimatedPlanCompileTime()).isZero();
  }

  @Test
  void carriesADeviceResidentKvCacheOntoAnExistingRequest() {
    DeviceMemoryRequest base =
        DeviceMemoryRequest.ofModelFile("Qwen3-0.6B-Q4_0.gguf", QWEN3_06B_Q4_0_BYTES, false);

    DeviceMemoryRequest withKv = base.withDeviceKvCacheBytes(64 * MIB);

    assertThat(base.deviceKvCacheBytes()).isZero();
    assertThat(withKv.deviceKvCacheBytes()).isEqualTo(64 * MIB);
    assertThat(withKv.retainedShapes()).isEqualTo(1);
    assertThat(
            AcceleratorEligibility.budget(withKv, PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL)
                .totalBytes())
        .isEqualTo(QWEN3_06B_Q4_0_BYTES + 64 * MIB + 256 * MIB);
  }

  @Test
  void rejectsMalformedRequests() {
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () -> DeviceMemoryRequest.ofModelFile(" ", 1, true)))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () -> DeviceMemoryRequest.detailed("m").weightBytes(0).build()))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () -> DeviceMemoryRequest.detailed("m").weightBytes(1).retainedShapes(0).build()))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () ->
                    DeviceMemoryRequest.detailed("m")
                        .weightBytes(1)
                        .deviceKvCacheBytes(-1)
                        .build()))
        .isInstanceOf(IllegalArgumentException.class);
    assertThat(
            org.assertj.core.api.Assertions.catchThrowable(
                () ->
                    PlanShapeStrategy.BOUNDED_RESIDENT_WORKING_SET.deviceWeightBytes(
                        DeviceMemoryRequest.ofModelFile("m", 1, false))))
        .isInstanceOf(IllegalStateException.class);
  }
}
