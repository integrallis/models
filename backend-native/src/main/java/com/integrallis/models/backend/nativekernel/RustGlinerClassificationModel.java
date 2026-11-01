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
import com.integrallis.models.backend.purejava.gliner.GlinerSchemaCompiler;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Lifecycle-owning GLiNER runtime: shared Java orchestration, native matrices/attention and HF
 * tokenizer.
 */
public final class RustGlinerClassificationModel implements SchemaClassificationModel {
  private final NativeHuggingFaceTokenizer tokenizer;
  private final GlinerClassificationModel model;
  private final NativeDebertaKernels kernels;

  private RustGlinerClassificationModel(
      NativeHuggingFaceTokenizer tokenizer,
      NativeDebertaKernels kernels,
      GlinerClassificationModel model) {
    this.tokenizer = tokenizer;
    this.model = model;
    this.kernels = kernels;
  }

  /** Opens the pinned snapshot with the integrity-checked native platform artifact. */
  public static RustGlinerClassificationModel load(Path snapshot) throws IOException {
    return load(snapshot, BundledNativeKernelLibrary.resolve());
  }

  /**
   * Opens the same runtime with an explicitly pinned native library for controlled qualification.
   */
  public static RustGlinerClassificationModel load(Path snapshot, Path nativeLibrary)
      throws IOException {
    int processors = Math.max(1, Runtime.getRuntime().availableProcessors());
    int threads =
        Integer.getInteger("models.deberta.threads", processors > 4 ? processors - 2 : processors);
    var tokenizer =
        NativeHuggingFaceTokenizer.load(nativeLibrary, snapshot.resolve("tokenizer.json"));
    NativeDebertaKernels kernels = null;
    try {
      kernels = NativeDebertaKernels.load(nativeLibrary, threads);
      return new RustGlinerClassificationModel(
          tokenizer,
          kernels,
          GlinerClassificationModel.load(snapshot, tokenizer::encodePiece, kernels));
    } catch (IOException | RuntimeException | Error failure) {
      if (kernels != null) {
        try {
          kernels.close();
        } catch (RuntimeException closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      try {
        tokenizer.close();
      } catch (RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** Production preprocessing exposed for exact-token qualification checks. */
  public synchronized GlinerSchemaCompiler.Encoded compile(Request request) {
    return model.compile(request);
  }

  /** Production head logits exposed for numerical qualification checks. */
  public synchronized float[] logits(Request request) {
    return model.logits(request);
  }

  /** Completed native projection calls; includes relative projections prepared at load time. */
  public long projectionCalls() {
    return kernels.projectCalls();
  }

  /** Completed native attention calls. */
  public long attentionCalls() {
    return kernels.attentionCalls();
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
      try {
        kernels.close();
      } finally {
        tokenizer.close();
      }
    }
  }
}
