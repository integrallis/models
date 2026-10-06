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

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Gate G4 as a unit test over every gold answer of the pinned datasets. */
class G4GoldAnswerTest {

  @ParameterizedTest
  @CsvSource({
    "GSM8K, gsm8k-test.jsonl, 1319",
    "GSM8K, gsm8k-dev.jsonl, 500",
    "ARC, arc-challenge-test.jsonl, 1172",
    "MATH500, math500-test.jsonl, 500"
  })
  void everyGoldAnswerRenderedInTheAnswerFormatScoresCorrect(
      DatasetKind dataset, String file, int expectedItems) throws Exception {
    Path path =
        Path.of(
            G4GoldAnswerTest.class
                .getResource("/com/integrallis/models/bench/fusion/gold/" + file)
                .toURI());
    List<DatasetItem> items = DatasetItem.loadJsonl(path, dataset, FieldMapping.standard());
    G4GoldCheck.Result result = G4GoldCheck.evaluate(dataset, items);
    assertThat(result.total()).isEqualTo(expectedItems);
    assertThat(result.failures()).isEmpty();
    assertThat(result.pass()).isTrue();
  }
}
