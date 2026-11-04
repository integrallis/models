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
import static org.mockito.Mockito.mockStatic;

import com.integrallis.vectors.core.VectorUtil;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * Where the native kernel pool's default thread count comes from, and what happens when the answer
 * is missing.
 *
 * <p>The pool sizes itself to the same number the vectors GGUF executor uses, so the two do not
 * each claim the whole machine. That makes this class depend on a static on another library's
 * class, and a static can be replaced: a test that calls {@code mockStatic(VectorUtil.class)} to
 * stub one method leaves every other static on it returning a default, which for an {@code int} is
 * zero. That is how this first broke -- two unrelated tests failed with "threads must be between 1
 * and 256: 0", a long way from the cause.
 *
 * <p>So the contract asserted here is that an absent recommendation degrades to this JVM's
 * processor count rather than propagating a number the settings then reject.
 */
class NativeKernelThreadCountTest {

  @Test
  void anAbsentRecommendationFallsBackToTheProcessorCount() {
    try (var vectors = mockStatic(VectorUtil.class)) {
      // Exactly the shape that broke it: the class is mocked, nothing stubs computeThreads, so it
      // answers 0.
      vectors.when(VectorUtil::computeThreads).thenReturn(0);

      NativeKernelSettings settings = NativeKernelSettings.fromSystemProperties(Map.of());

      assertThat(settings.threadCount())
          .as("a provider that cannot answer must not become a thread count")
          .isEqualTo(Math.max(1, Runtime.getRuntime().availableProcessors()));
    }
  }

  @Test
  void aNegativeRecommendationIsTreatedAsAbsentRatherThanPassedThrough() {
    try (var vectors = mockStatic(VectorUtil.class)) {
      vectors.when(VectorUtil::computeThreads).thenReturn(-4);

      assertThat(NativeKernelSettings.fromSystemProperties(Map.of()).threadCount())
          .isEqualTo(Math.max(1, Runtime.getRuntime().availableProcessors()));
    }
  }

  @Test
  void aUsableRecommendationIsWhatThePoolSizesItselfTo() {
    try (var vectors = mockStatic(VectorUtil.class)) {
      vectors.when(VectorUtil::computeThreads).thenReturn(6);

      assertThat(NativeKernelSettings.fromSystemProperties(Map.of()).threadCount()).isEqualTo(6);
    }
  }

  @Test
  void anExplicitRecommendationStillWinsOverTheSharedDefault() {
    try (var vectors = mockStatic(VectorUtil.class)) {
      vectors.when(VectorUtil::computeThreads).thenReturn(6);

      NativeKernelSettings settings =
          NativeKernelSettings.fromSystemProperties(
              Map.of(NativeKernelLibrary.THREAD_COUNT_PROPERTY, "3"));

      assertThat(settings.threadCount()).isEqualTo(3);
    }
  }
}
