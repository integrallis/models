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
package com.integrallis.models.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.RerankingModel;
import dev.langchain4j.data.segment.TextSegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelsScoringModelTest {

  @Test
  void scoresSegmentsInCallerOrderWithTheQueryFirst() {
    ScriptedModel backend = new ScriptedModel();
    var model = new ModelsScoringModel(backend);

    List<Double> scores =
        model
            .scoreAll(List.of(TextSegment.from("alpha"), TextSegment.from("beta")), "the query")
            .content();

    assertThat(scores).containsExactly(5.0, 4.0);
    assertThat(backend.seen).containsExactly("the query|alpha", "the query|beta");
  }

  @Test
  void frameworkConvenienceMethodsUseTheSameBackend() {
    ScriptedModel backend = new ScriptedModel();
    var model = new ModelsScoringModel(backend);

    assertThat(model.score("document", "query").content()).isEqualTo(8.0);
    assertThat(model.score(TextSegment.from("segment"), "query").content()).isEqualTo(7.0);
    assertThat(backend.seen).containsExactly("query|document", "query|segment");
  }

  @Test
  void closesTheOwnedModelAndRejectsNulls() {
    ScriptedModel backend = new ScriptedModel();
    var model = new ModelsScoringModel(backend);

    model.close();

    assertThat(backend.closed).isTrue();
    assertThatThrownBy(() -> new ModelsScoringModel(null)).isInstanceOf(NullPointerException.class);
  }

  private static final class ScriptedModel implements RerankingModel {
    private final List<String> seen = new ArrayList<>();
    private boolean closed;

    @Override
    public double score(String query, String document) {
      seen.add(query + "|" + document);
      return document.length();
    }

    @Override
    public void close() {
      closed = true;
    }
  }
}
