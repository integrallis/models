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
package com.integrallis.models.backend.purejava.deberta;

import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import com.integrallis.models.backend.purejava.tokenizer.DebertaV2Tokenizer;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Objects;

/** Package boundary for the pure-Java DeBERTa-v2 reranking implementation. */
public final class DebertaV2Engine {

  private final DebertaV2Tokenizer tokenizer;
  private final DebertaV2ForwardPass forwardPass;

  private DebertaV2Engine(DebertaV2Tokenizer tokenizer, DebertaV2ForwardPass forwardPass) {
    this.tokenizer = tokenizer;
    this.forwardPass = forwardPass;
  }

  /** Loads a supported Hugging Face DeBERTa-v2 sequence classifier. */
  public static DebertaV2Engine load(Path directory) throws IOException {
    Objects.requireNonNull(directory, "directory");
    DebertaV2Config config = DebertaV2Config.parse(directory.resolve("config.json"));
    DebertaV2Tokenizer tokenizer =
        DebertaV2Tokenizer.fromJson(directory.resolve("tokenizer.json"), config.maxPositions());
    DebertaV2Weights weights;
    try (Arena arena = Arena.ofConfined()) {
      weights =
          DebertaV2Weights.load(
              new SafetensorsTensorSource(SafetensorsBundle.open(directory, arena)), config);
    }
    return new DebertaV2Engine(tokenizer, new DebertaV2ForwardPass(config, weights));
  }

  /** Scores one query-document pair. */
  public double score(String query, String document) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(document, "document");
    return forwardPass.score(tokenizer.encodePair(query, document).tokens());
  }
}
