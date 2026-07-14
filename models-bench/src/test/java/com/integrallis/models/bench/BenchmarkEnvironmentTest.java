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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class BenchmarkEnvironmentTest {

  @Test
  void capturesTheJVMAndEffectiveVectorRuntime() {
    BenchmarkEnvironment environment = BenchmarkEnvironment.capture();

    assertThat(environment.jvmArguments()).isNotNull();
    assertThat(environment.vectorProvider()).isNotBlank();
    assertThat(environment.preferredVectorBits()).isPositive();
    assertThat(environment.effectiveVectorBits()).isNotNegative();
    assertThat(environment.effectiveVectorBits())
        .isLessThanOrEqualTo(environment.preferredVectorBits());
    if (environment.vectorApiEnabled()) {
      assertThat(environment.effectiveVectorBits()).isPositive();
    } else {
      assertThat(environment.effectiveVectorBits()).isZero();
    }
    assertThat(environment.runtimeProperties()).containsKey("models.purejava.maxContextLength");
  }

  @Test
  void honorsExplicitSourceIdentityFromAReproducibleBuild() {
    String previousRevision = System.getProperty("models.benchmark.sourceRevision");
    String previousDirty = System.getProperty("models.benchmark.sourceDirty");
    try {
      System.setProperty("models.benchmark.sourceRevision", "abc123");
      System.setProperty("models.benchmark.sourceDirty", "false");

      BenchmarkEnvironment environment = BenchmarkEnvironment.capture();

      assertThat(environment.sourceRevision()).isEqualTo("abc123");
      assertThat(environment.sourceDirty()).isFalse();
    } finally {
      restore("models.benchmark.sourceRevision", previousRevision);
      restore("models.benchmark.sourceDirty", previousDirty);
    }
  }

  private static void restore(String property, String value) {
    if (value == null) {
      System.clearProperty(property);
    } else {
      System.setProperty(property, value);
    }
  }
}
