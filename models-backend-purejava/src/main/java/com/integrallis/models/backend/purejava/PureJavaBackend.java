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

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.api.ModelMetadata;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.llama.LlamaConfig;
import com.integrallis.models.backend.purejava.llama.LlamaForwardPass;
import com.integrallis.models.backend.purejava.llama.LlamaWeights;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Objects;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.ModelJarRegistry;
import org.modeljars.ModelJarRequirement;

/**
 * Pure Java inference backend that loads a GGUF model and runs Llama-family forward passes without
 * any native dependencies.
 */
public final class PureJavaBackend implements SpeculativeInferenceBackend {

  static final String MAX_CONTEXT_LENGTH_PROPERTY = "models.purejava.maxContextLength";

  private final Arena arena;
  private final LlamaConfig config;
  private final GgufTokenizer tokenizer;
  private final LlamaForwardPass forwardPass;
  private final ModelMetadata modelMetadata;

  private PureJavaBackend(
      Arena arena,
      LlamaConfig config,
      GgufTokenizer tokenizer,
      LlamaForwardPass forwardPass,
      ModelMetadata modelMetadata) {
    this.arena = arena;
    this.config = config;
    this.tokenizer = tokenizer;
    this.forwardPass = forwardPass;
    this.modelMetadata = modelMetadata;
  }

  /** Loads a GGUF model file and returns a ready-to-use backend. */
  public static PureJavaBackend load(Path modelPath) {
    return load(modelPath, Arena.ofShared());
  }

  static PureJavaBackend load(Path modelPath, Arena arena) {
    Objects.requireNonNull(arena, "arena");
    try {
      Objects.requireNonNull(modelPath, "modelPath");
      GgufFile file = GgufParser.parse(modelPath, arena);
      LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
      LlamaWeights weights = LlamaWeights.fromGgufFile(file, config);
      GgufTokenizer tokenizer = GgufTokenizer.fromMetadata(file.metadata());
      KvCache cache =
          new KvCache(
              config.numLayers(), runtimeContextLength(config), config.keyDim(), config.valueDim());
      LlamaForwardPass forwardPass = new LlamaForwardPass(config, weights, cache);

      String modelName =
          file.metadata().getString("general.name").orElse(modelPath.getFileName().toString());
      String modelFamily = file.metadata().getString("general.architecture").orElse("llama");

      ModelMetadata metadata =
          new ModelMetadata(
              modelFamily,
              modelName,
              config.contextLength(),
              config.vocabSize(),
              config.embeddingDim(),
              config.numLayers(),
              config.numHeads(),
              config.numKvHeads());

      return new PureJavaBackend(arena, config, tokenizer, forwardPass, metadata);
    } catch (IOException e) {
      closeAfterFailure(arena, e);
      throw new UncheckedIOException("Failed to load model: " + modelPath, e);
    } catch (RuntimeException | Error e) {
      closeAfterFailure(arena, e);
      throw e;
    }
  }

  /** Resolves a model from classpath ModelJars metadata and loads it. */
  public static PureJavaBackend load(ModelJarRequirement requirement) {
    Objects.requireNonNull(requirement, "requirement");
    ModelJarDescriptor descriptor =
        ModelJarRegistry.fromClasspath()
            .resolve(requirement)
            .orElseThrow(
                () -> new ModelJarException("No ModelJars descriptor matched " + requirement));
    return load(descriptor);
  }

  /** Loads a GGUF model described by a ModelJars marker descriptor. */
  public static PureJavaBackend load(ModelJarDescriptor descriptor) {
    Objects.requireNonNull(descriptor, "descriptor");
    if (!descriptor.supportsBackend("pure-java")) {
      throw new ModelJarException(
          "ModelJars descriptor does not support pure-java backend: " + descriptor.alias());
    }
    if (!"gguf".equals(descriptor.format())) {
      throw new ModelJarException(
          "PureJavaBackend only supports GGUF ModelJars descriptors: " + descriptor.alias());
    }
    Path modelPath =
        descriptor
            .localPath()
            .orElseThrow(
                () ->
                    new ModelJarException(
                        "ModelJars descriptor has no local path: " + descriptor.alias()));
    return load(modelPath);
  }

  @Override
  public String name() {
    return "pure-java";
  }

  @Override
  public ModelMetadata metadata() {
    return modelMetadata;
  }

  @Override
  public Tokenizer tokenizer() {
    return tokenizer;
  }

  @Override
  public float[] forward(int token, int position) {
    return forwardPass.forward(token, position);
  }

  @Override
  public float[] forwardTransient(int token, int position) {
    return forwardPass.forwardTransient(token, position);
  }

  @Override
  public float[] prefill(int[] tokens, int startPosition) {
    return forwardPass.prefill(tokens, startPosition);
  }

  @Override
  public int checkpoint() {
    return forwardPass.checkpoint();
  }

  @Override
  public LogitBatch verify(int[] tokens, int startPosition) {
    return forwardPass.verify(tokens, startPosition);
  }

  @Override
  public void rewind(int checkpoint) {
    forwardPass.rewind(checkpoint);
  }

  @Override
  public void reset() {
    forwardPass.reset();
  }

  @Override
  public void close() {
    arena.close();
  }

  private static int runtimeContextLength(LlamaConfig config) {
    String value = System.getProperty(MAX_CONTEXT_LENGTH_PROPERTY);
    if (value == null || value.isBlank()) {
      return config.contextLength();
    }
    int maxContextLength;
    try {
      maxContextLength = Integer.parseInt(value);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(
          MAX_CONTEXT_LENGTH_PROPERTY + " must be a positive integer: " + value, e);
    }
    if (maxContextLength <= 0) {
      throw new IllegalArgumentException(
          MAX_CONTEXT_LENGTH_PROPERTY + " must be a positive integer: " + value);
    }
    return Math.min(config.contextLength(), maxContextLength);
  }

  private static void closeAfterFailure(Arena arena, Throwable failure) {
    try {
      arena.close();
    } catch (RuntimeException closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }
}
