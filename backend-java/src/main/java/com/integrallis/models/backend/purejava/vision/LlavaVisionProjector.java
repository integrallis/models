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
package com.integrallis.models.backend.purejava.vision;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Owns a converted LLaVA 1.5 CLIP encoder and its trained two-layer MLP projector. Instances reuse
 * image scratch buffers and must not be called concurrently.
 */
public final class LlavaVisionProjector implements AutoCloseable {
  private final com.integrallis.models.backend.purejava.spi.DenseProjectionKernels denseKernels;
  private final com.integrallis.models.backend.purejava.spi.DenseProjectionKernels.Projection
      upProjection;
  private final com.integrallis.models.backend.purejava.spi.DenseProjectionKernels.Projection
      downProjection;
  private final Arena arena;
  private final ClipVisionConfig config;
  private final ClipVisionEncoder encoder;
  private final ClipVisionWeights.Matrix up;
  private final ClipVisionWeights.Matrix down;
  private final float[] upBias;
  private final float[] downBias;
  private final float[] mean;
  private final float[] std;
  private boolean closed;

  private LlavaVisionProjector(
      Arena arena,
      GgufFile file,
      com.integrallis.models.backend.purejava.spi.DenseProjectionKernels denseKernels) {
    this.arena = arena;
    this.denseKernels = denseKernels;
    var meta = file.metadata();
    if (!meta.getString("general.architecture").orElse("").equals("clip")
        || !meta.getString("clip.projector_type").orElse("").equals("mlp")
        || !meta.getBool("clip.has_llava_projector").orElse(false)
        || meta.getArraySize("clip.vision.image_grid_pinpoints").orElse(0) != 0
        || meta.getArraySize("clip.vision.feature_layer").orElse(0) != 0
        || meta.getBool("clip.use_silu").orElse(false)) {
      throw new IllegalArgumentException(
          "Requires a fixed-resolution LLaVA 1.5 CLIP MLP projector; Qwen-VL, SigLIP and any-resolution projectors are unsupported");
    }
    file.getTensor("v.class_embd");
    file.getTensor("v.pre_ln.weight");
    file.getTensor("v.pre_ln.bias");
    meta.getBool("clip.use_gelu")
        .orElseThrow(() -> new IllegalArgumentException("Missing CLIP activation metadata"));
    config = ClipVisionConfig.fromGguf(file);
    // This GGUF profile has already removed the final CLIP layer at conversion time.
    // Never subtract another layer: upstream CLIP has 24, LLaVA uses hidden_states[-2].
    if (config.numLayers() != 23
        || file.hasTensor("v.blk.23.ln1.weight")
        || file.hasTensor("v.post_ln.weight")) {
      throw new IllegalArgumentException(
          "Expected the converted 23-block LLaVA 1.5 feature extractor without post-LN");
    }
    var first = file.getTensor("mm.0.weight");
    var second = file.getTensor("mm.2.weight");
    if (first.shape().length != 2 || second.shape().length != 2)
      throw new IllegalArgumentException("Projector weights must be matrices");
    int intermediate = Math.toIntExact(first.shape()[1]);
    int output = Math.toIntExact(second.shape()[1]);
    if (intermediate <= 0 || output <= 0) throw new IllegalArgumentException("Empty projector");
    up = ClipVisionWeights.matrix(file, "mm.0.weight", intermediate, config.hiddenSize());
    down = ClipVisionWeights.matrix(file, "mm.2.weight", output, intermediate);
    for (var matrix : new ClipVisionWeights.Matrix[] {up, down}) {
      if (matrix.type() != com.integrallis.models.backend.purejava.gguf.GgufTensorType.F32
          && matrix.type() != com.integrallis.models.backend.purejava.gguf.GgufTensorType.F16
          && matrix.type() != com.integrallis.models.backend.purejava.gguf.GgufTensorType.BF16) {
        throw new IllegalArgumentException("MLP projector requires F32, F16 or BF16 weights");
      }
    }
    upBias = ClipVisionWeights.vector(file, "mm.0.bias", intermediate);
    downBias = ClipVisionWeights.vector(file, "mm.2.bias", output);
    mean = channels(meta.getFloat32Array("clip.vision.image_mean").orElseThrow(), "mean");
    std = channels(meta.getFloat32Array("clip.vision.image_std").orElseThrow(), "std");
    upProjection = VisionProjection.prepare(up, denseKernels);
    downProjection = VisionProjection.prepare(down, denseKernels);
    encoder =
        new ClipVisionEncoder(config, ClipVisionWeights.fromGgufFile(file, config), denseKernels);
  }

  /** Loads the supported converted LLaVA projector; the caller owns and closes the result. */
  public static LlavaVisionProjector load(Path path) {
    Arena arena = Arena.ofShared();
    com.integrallis.models.backend.purejava.spi.DenseProjectionKernels dense = null;
    try {
      for (var provider :
          java.util.ServiceLoader.load(
              com.integrallis.models.backend.purejava.spi.DenseProjectionProvider.class)) {
        var available =
            provider.open(
                com.integrallis.models.backend.purejava.plan.RuntimeFingerprint.capture()
                    .ggufThreads());
        if (available.isPresent()) {
          dense = available.orElseThrow();
          break;
        }
      }
      return new LlavaVisionProjector(arena, GgufParser.parse(path, arena), dense);
    } catch (Exception | Error e) {
      // Close both resources even if provider cleanup fails; retain the load failure as primary.
      var failedDense = dense;
      try (arena;
          failedDense) {
        throw new IllegalArgumentException("Cannot load LLaVA projector: " + path, e);
      }
    }
  }

  /** Identifies the projection kernel selected for this loaded projector. */
  public String implementation() {
    checkOpen();
    return denseKernels == null ? "vector-api" : denseKernels.implementation();
  }

  /** Number of projected patch rows, excluding the CLIP class token. */
  public int tokenCount() {
    checkOpen();
    return config.numTokens() - 1;
  }

  /** Width of each projected row in the matched text decoder's input space. */
  public int embeddingDimension() {
    checkOpen();
    return down.rows();
  }

  /** Target square image width and height, in pixels, before patch extraction. */
  public int imageSize() {
    checkOpen();
    return config.imageSize();
  }

  /** Returns normalized RGB pixels in channel-first order, using this model's image metadata. */
  public float[] preprocess(byte[] image) {
    checkOpen();
    return ClipImagePreprocessor.preprocess(image, config.imageSize(), mean, std);
  }

  /** Returns fresh row-major decoder embeddings for one encoded PNG/JPEG image. */
  public float[] encode(byte[] image) {
    return encodePreprocessed(preprocess(image));
  }

  /** Exact same encoder/projector path, exposed for stage-by-stage numerical qualification. */
  public float[] encodePreprocessed(float[] chw) {
    checkOpen();
    float[] features = encoder.encode(chw);
    // LLaVA uses patch features, excluding CLS, with no post-layer normalization.
    float[] patches = Arrays.copyOfRange(features, config.hiddenSize(), features.length);
    float[] intermediate = project(patches, up, upBias, upProjection);
    com.integrallis.vectors.core.GgufStagePlan.of(
            com.integrallis.vectors.core.GgufStagePlan.stage(
                tokenCount(),
                (from, to) ->
                    TensorOps.geluErf(
                        intermediate,
                        from * up.rows(),
                        intermediate,
                        from * up.rows(),
                        (to - from) * up.rows())))
        .execute();
    return project(intermediate, down, downBias, downProjection);
  }

  private float[] project(
      float[] input,
      ClipVisionWeights.Matrix matrix,
      float[] bias,
      com.integrallis.models.backend.purejava.spi.DenseProjectionKernels.Projection projection) {
    float[] output = new float[Math.multiplyExact(tokenCount(), matrix.rows())];
    projection.project(output, input, tokenCount());
    for (int i = 0; i < output.length; i++) output[i] += bias[i % bias.length];
    return output;
  }

  private static float[] channels(List<Float> values, String name) {
    if (values.size() != 3)
      throw new IllegalArgumentException("Expected three image " + name + " channels");
    float[] result = new float[3];
    for (int i = 0; i < 3; i++) {
      result[i] = values.get(i);
      if (!Float.isFinite(result[i]) || (name.equals("std") && result[i] <= 0))
        throw new IllegalArgumentException("Invalid image " + name);
    }
    return result;
  }

  private void checkOpen() {
    if (closed) throw new IllegalStateException("Projector is closed");
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      try {
        if (denseKernels != null) denseKernels.close();
      } finally {
        arena.close();
      }
    }
  }
}
