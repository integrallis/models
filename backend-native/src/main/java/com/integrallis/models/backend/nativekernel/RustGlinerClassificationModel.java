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

import com.integrallis.models.api.SchemaClassificationModel;
import com.integrallis.models.backend.purejava.gliner.GlinerClassificationModel;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/** Lifecycle-owning GLiNER runtime: shared Java encoder and native HF tokenizer. */
public final class RustGlinerClassificationModel implements SchemaClassificationModel {
  private final NativeHuggingFaceTokenizer tokenizer;
  private final GlinerClassificationModel model;

  private RustGlinerClassificationModel(
      NativeHuggingFaceTokenizer tokenizer, GlinerClassificationModel model) {
    this.tokenizer = tokenizer;
    this.model = model;
  }

  /** Opens the pinned snapshot with the integrity-checked native platform artifact. */
  public static RustGlinerClassificationModel load(Path snapshot) throws IOException {
    var tokenizer = NativeHuggingFaceTokenizer.load(snapshot.resolve("tokenizer.json"));
    try {
      return new RustGlinerClassificationModel(
          tokenizer, GlinerClassificationModel.load(snapshot, tokenizer::encodePiece));
    } catch (IOException | RuntimeException | Error failure) {
      try {
        tokenizer.close();
      } catch (RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  @Override
  public synchronized List<Result> classify(Request request) {
    return model.classify(request);
  }

  @Override
  public synchronized void close() {
    try {
      model.close();
    } finally {
      tokenizer.close();
    }
  }
}
