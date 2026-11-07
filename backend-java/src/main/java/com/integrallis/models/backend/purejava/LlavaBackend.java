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

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ImageInferenceBackend;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import com.integrallis.models.backend.purejava.vision.LlavaVisionProjector;
import java.nio.file.Path;
import java.util.ServiceLoader;

/** Two-file LLaVA 1.5 inference, sharing the shipping Llama decoder and its KV cache. */
public final class LlavaBackend implements ImageInferenceBackend {
  private final PureJavaBackend text;
  private final LlavaVisionProjector vision;
  private boolean closed;
  private int position;

  private LlavaBackend(PureJavaBackend text, LlavaVisionProjector vision) {
    this.text = text;
    this.vision = vision;
  }

  /** Loads a matched LLaVA text GGUF and converted 23-block CLIP MLP mmproj. */
  public static LlavaBackend load(Path textModel, Path mmproj) {
    LlavaVisionProjector vision = LlavaVisionProjector.load(mmproj);
    PureJavaBackend text = null;
    try {
      text = decoder(textModel);
      if (!text.supportsEmbeddingInput()
          || text.embeddingInputDimension() != vision.embeddingDimension()) {
        throw new IllegalArgumentException(
            "Text model and projector architectures/dimensions do not match");
      }
      if (text.contextCapacity() <= vision.tokenCount())
        throw new IllegalArgumentException("Context must fit image patches and text");
      return new LlavaBackend(text, vision);
    } catch (RuntimeException | Error e) {
      var failedText = text;
      try (vision;
          failedText) {
        throw e;
      }
    }
  }

  private static PureJavaBackend decoder(Path model) {
    int threads = RuntimeFingerprint.capture().ggufThreads();
    for (GgufMatrixKernelProvider provider : ServiceLoader.load(GgufMatrixKernelProvider.class)) {
      var available = provider.open(threads);
      if (available.isPresent()) {
        var kernel = available.orElseThrow();
        com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel attention;
        try {
          attention = provider.openAttention(threads, kernel);
        } catch (RuntimeException | Error failure) {
          try (kernel) {
            throw failure;
          }
        }
        return PureJavaBackend.load(model, kernel, attention);
      }
    }
    return PureJavaBackend.load(model);
  }

  @Override
  public String name() {
    checkOpen();
    return "pure-java-llava";
  }

  @Override
  public ModelMetadata metadata() {
    checkOpen();
    return text.metadata();
  }

  @Override
  public BackendDiagnostics diagnostics() {
    checkOpen();
    var decoder = text.diagnostics();
    var environment = new java.util.LinkedHashMap<>(decoder.environment());
    environment.put("vision-projections", vision.implementation());
    environment.put("decoder-attention", text.attentionKernelImplementation());
    return new BackendDiagnostics(
        name(), "llava-v1/" + decoder.planVersion(), environment, decoder.optimizations());
  }

  @Override
  public Tokenizer tokenizer() {
    checkOpen();
    return text.tokenizer();
  }

  @Override
  public int contextCapacity() {
    checkOpen();
    return text.contextCapacity();
  }

  @Override
  public int imageTokenCount() {
    checkOpen();
    return vision.tokenCount();
  }

  @Override
  public float[] prefillImage(byte[] encodedImage, int startPosition) {
    requirePosition(startPosition, imageTokenCount());
    float[] result = text.prefillEmbeddings(vision.encode(encodedImage), startPosition);
    position += imageTokenCount();
    return result;
  }

  @Override
  public float[] forward(int token, int startPosition) {
    requirePosition(startPosition, 1);
    float[] result = text.forward(token, startPosition);
    position++;
    return result;
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    requirePosition(startPosition, tokens.length);
    for (int token : tokens) {
      if (token < 0 || token >= metadata().vocabSize())
        throw new IllegalArgumentException("Invalid token");
    }
    float[] result = text.prefill(tokens, startPosition);
    position += tokens.length;
    return result;
  }

  @Override
  public void reset() {
    checkOpen();
    text.reset();
    position = 0;
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      try {
        text.close();
      } finally {
        vision.close();
      }
    }
  }

  private void checkOpen() {
    if (closed) throw new IllegalStateException("Backend is closed");
  }

  private void requirePosition(int start, int count) {
    checkOpen();
    if (start != position || count <= 0 || count > contextCapacity() - start) {
      throw new IllegalArgumentException("Input must be contiguous and fit the context");
    }
  }
}
