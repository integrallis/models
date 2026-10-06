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
package com.integrallis.models.bench.fusion;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/** Gate G6 as a unit test: every hand-constructed trace must extract exactly as expected. */
class G6FixtureTest {

  @ParameterizedTest
  @EnumSource(
      value = DatasetKind.class,
      names = {"GSM8K", "ARC", "MATH500"})
  void everyFixtureExtractsExactly(DatasetKind dataset) throws Exception {
    G6Fixtures.Result result = G6Fixtures.evaluate(dataset, G6Fixtures.loadBundled(dataset));
    assertThat(result.failures()).isEmpty();
    assertThat(result.total()).isEqualTo(30);
    assertThat(result.passed()).isEqualTo(30);
    assertThat(result.perCategory())
        .containsKeys("agree", "disagree", "missing-think", "truncated-think");
  }
}
