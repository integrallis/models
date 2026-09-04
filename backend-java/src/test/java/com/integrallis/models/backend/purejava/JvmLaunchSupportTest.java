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

import static org.assertj.core.api.Assertions.assertThatCode;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class JvmLaunchSupportTest {

  @Test
  void rejectsSpringBootsC1OnlyDevelopmentLaunch() {
    assertThatThrownBy(
            () -> JvmLaunchSupport.requireOptimizingCompiler(List.of("-XX:TieredStopAtLevel=1")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("-XX:TieredStopAtLevel=1")
        .hasMessageContaining("optimizedLaunch = false")
        .hasMessageContaining("Disable launch optimization");
  }

  @Test
  void rejectsAnyTieredLevelThatCannotReachC2() {
    for (int level = 1; level < 4; level++) {
      int stopLevel = level;
      assertThatThrownBy(
              () ->
                  JvmLaunchSupport.requireOptimizingCompiler(
                      List.of("-XX:TieredStopAtLevel=" + stopLevel)))
          .isInstanceOf(IllegalStateException.class);
    }
  }

  @Test
  void acceptsTheNormalOptimizingCompiler() {
    assertThatCode(() -> JvmLaunchSupport.requireOptimizingCompiler(List.of()))
        .doesNotThrowAnyException();
    assertThatCode(
            () -> JvmLaunchSupport.requireOptimizingCompiler(List.of("-XX:TieredStopAtLevel=4")))
        .doesNotThrowAnyException();
    assertThatCode(
            () -> JvmLaunchSupport.requireOptimizingCompiler(List.of("-XX:-TieredCompilation")))
        .doesNotThrowAnyException();
  }

  @Test
  void rejectsInterpretedOnlyExecution() {
    assertThatThrownBy(() -> JvmLaunchSupport.requireOptimizingCompiler(List.of("-Xint")))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("-Xint");
  }
}
