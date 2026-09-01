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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Cross-encoder scores retained by the corrected GGUF converter against its ONNX oracle. */
@Tag("integration")
class MsMarcoMiniLmRerankerIntegrationTest {

  private static final String QUERY = "How many people live in Berlin?";
  private static final List<String> DOCUMENTS =
      List.of(
          "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
          "Paris is the capital and most populous city of France.",
          "Berlin is well known for its museums and its metropolitan area of about six million people.",
          "Domestic cats sleep for a large part of the day.",
          "New York City had an estimated population of 8,804,190 in 2020.",
          "The Berlin Wall divided the city from 1961 until 1989.");
  private static final double[] ONNX_REFERENCE = {8.846, -10.886, 7.401, -11.225, -5.200, -4.944};
  private static final double[] QUANTIZED_REFERENCE = {
    8.821, -10.985, 7.313, -11.271, -5.149, -4.892
  };
  private static final ModelFixtureRequirement RERANKER =
      ModelFixtureRequirement.of("hf://cstr/ms-marco-MiniLM-L-6-v2-GGUF")
          .version("[2.0.0,3.0.0)")
          .variant("q4_k_imatrix_g7c_f7")
          .backend("pure-java")
          .capability("reranking");

  private static ModelFixtureDescriptor fixture() {
    return ModelFixtureRegistry.fromClasspath().resolve(RERANKER).orElseThrow();
  }

  @Test
  void sentencePairTokensAndTypesMatchThePinnedReference() throws IOException {
    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(fixture().localPath().orElseThrow(), arena);
      var tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      var pair = tokenizer.encodePair(QUERY, DOCUMENTS.get(0), 512);

      assertThat(pair.tokens())
          .containsExactly(
              101, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102, 4068, 2038, 1037, 2313, 1997,
              1017, 1010, 19611, 1010, 6021, 2487, 5068, 4864, 1999, 2019, 2181, 1997, 6486, 2487,
              1012, 6445, 2675, 7338, 1012, 102);
      assertThat(pair.tokenTypes())
          .containsExactly(
              0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
              1, 1, 1, 1, 1);
    }
  }

  @Test
  void correctedQuantizedScoresMatchTheOnnxScaleAndOrdering() {
    try (GgufRerankingModel model = GgufRerankingModel.load(fixture().localPath().orElseThrow())) {
      List<Double> scores = model.scoreAll(QUERY, DOCUMENTS);

      assertThat(scores).hasSize(ONNX_REFERENCE.length);
      for (int index = 0; index < scores.size(); index++) {
        assertThat(scores.get(index)).isCloseTo(ONNX_REFERENCE[index], within(0.15));
        assertThat(scores.get(index)).isCloseTo(QUANTIZED_REFERENCE[index], within(0.05));
      }
      assertThat(model.rerank(QUERY, DOCUMENTS, 2))
          .extracting(result -> result.originalIndex())
          .containsExactly(0, 2);
    }
  }

  @Test
  void repeatedScoresAreDeterministicAndPairOrderMatters() {
    try (GgufRerankingModel model = GgufRerankingModel.load(fixture().localPath().orElseThrow())) {
      double first = model.score(QUERY, DOCUMENTS.get(0));

      assertThat(model.score(QUERY, DOCUMENTS.get(0))).isEqualTo(first);
      assertThat(model.score(DOCUMENTS.get(0), QUERY)).isNotEqualTo(first);
    }
  }
}
