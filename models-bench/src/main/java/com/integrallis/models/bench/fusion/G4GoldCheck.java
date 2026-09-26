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

import java.util.ArrayList;
import java.util.List;

/** Gate G4: gold answers rendered in the prompt's answer format must extract and score correct. */
final class G4GoldCheck {

  private G4GoldCheck() {}

  record Failure(String id, String gold, String rendered, String extracted) {}

  record Result(
      String dataset, String scorer, int total, int passed, List<Failure> failures, boolean pass) {
    Result {
      failures = List.copyOf(failures);
    }
  }

  static Result evaluate(DatasetKind dataset, List<DatasetItem> items) {
    return evaluate(dataset.id(), AnswerExtractor.forDataset(dataset), items);
  }

  static Result evaluate(String dataset, AnswerExtractor extractor, List<DatasetItem> items) {
    List<Failure> failures = new ArrayList<>();
    int passed = 0;
    for (DatasetItem item : items) {
      String rendered = extractor.renderGold(item.answer());
      TraceAnalysis analysis = TraceAnalysis.analyze(extractor, rendered, item.labels());
      boolean ok =
          analysis.statedAnswer() != null
              && extractor.equivalent(analysis.statedAnswer(), item.answer());
      if (ok) {
        passed++;
      } else {
        failures.add(new Failure(item.id(), item.answer(), rendered, analysis.statedAnswer()));
      }
    }
    return new Result(
        dataset,
        extractor.id(),
        items.size(),
        passed,
        failures,
        !items.isEmpty() && failures.isEmpty());
  }
}
