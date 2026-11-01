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
package com.integrallis.models.backend.purejava.gliner;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonToken;
import com.integrallis.models.api.SchemaClassificationModel;
import com.integrallis.models.backend.purejava.deberta.DebertaV2Config;
import com.integrallis.models.backend.purejava.deberta.DebertaV2Encoder;
import com.integrallis.models.backend.purejava.deberta.DebertaV2Kernels;
import com.integrallis.models.backend.purejava.safetensors.SafetensorsBundle;
import com.integrallis.models.backend.purejava.tensor.SafetensorsTensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.models.backend.purejava.tensor.TensorStorage;
import com.integrallis.vectors.core.VectorUtil;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/** GLiNER span-checkpoint classification head over the shared DeBERTa encoder. */
public final class GlinerClassificationModel implements SchemaClassificationModel {
  private DebertaV2Encoder encoder;
  private final GlinerSchemaCompiler compiler;
  private final int hiddenSize;
  private final float[] up;
  private final float[] bias;
  private final float[] down;
  private final float outputBias;

  private GlinerClassificationModel(
      DebertaV2Encoder encoder,
      GlinerSchemaCompiler compiler,
      int hiddenSize,
      float[] up,
      float[] bias,
      float[] down,
      float outputBias) {
    this.encoder = encoder;
    this.compiler = compiler;
    this.hiddenSize = hiddenSize;
    this.up = up;
    this.bias = bias;
    this.down = down;
    this.outputBias = outputBias;
  }

  /** Loads F32 GLiNER weights; the caller owns the supplied piece tokenizer's lifecycle. */
  public static GlinerClassificationModel load(Path snapshot, Function<String, int[]> tokenizer)
      throws IOException {
    return loadInternal(snapshot, tokenizer, null);
  }

  /** Uses a caller-owned dense projector; its lifecycle must enclose this model's lifecycle. */
  public static GlinerClassificationModel loadWithKernels(
      Path snapshot, Function<String, int[]> tokenizer, DebertaV2Kernels projection)
      throws IOException {
    return loadInternal(snapshot, tokenizer, Objects.requireNonNull(projection, "projection"));
  }

  private static GlinerClassificationModel loadInternal(
      Path snapshot, Function<String, int[]> tokenizer, DebertaV2Kernels projection)
      throws IOException {
    validateConfig(snapshot.resolve("config.json"));
    DebertaV2Config config =
        DebertaV2Config.parseEncoder(snapshot.resolve("encoder_config/config.json"));
    try (Arena arena = Arena.ofConfined()) {
      TensorSource source = new SafetensorsTensorSource(SafetensorsBundle.open(snapshot, arena));
      int dim = config.hiddenSize();
      return new GlinerClassificationModel(
          projection == null
              ? DebertaV2Encoder.load(source, config, "encoder.")
              : DebertaV2Encoder.loadWithKernels(source, config, "encoder.", projection),
          new GlinerSchemaCompiler(tokenizer, config.maxPositions()),
          dim,
          f32(source, "classifier.0.weight", dim * 2, dim),
          f32(source, "classifier.0.bias", dim * 2),
          f32(source, "classifier.2.weight", 1, dim * 2),
          f32(source, "classifier.2.bias", 1)[0]);
    }
  }

  /** Exposes exactly the production preprocessing for parity evidence. */
  public synchronized GlinerSchemaCompiler.Encoded compile(Request request) {
    requireOpen();
    return compiler.compile(request);
  }

  /** Raw head logits in task/label order, using the production inference path. */
  public synchronized float[] logits(Request request) {
    return logits(compile(request));
  }

  private float[] logits(GlinerSchemaCompiler.Encoded input) {
    requireOpen();
    float[][] states = encoder.encode(input.tokenIds(), input.markerPositions());
    float[] logits = new float[states.length];
    float[] intermediate = new float[bias.length];
    for (int label = 0; label < states.length; label++) {
      VectorUtil.batchDotProductExact(
          states[label], 0, up, 0, hiddenSize, bias.length, hiddenSize, intermediate, 0);
      for (int index = 0; index < intermediate.length; index++) {
        intermediate[index] = Math.max(0.0f, intermediate[index] + bias[index]);
      }
      logits[label] = VectorUtil.dotProduct(intermediate, down) + outputBias;
      if (!Float.isFinite(logits[label])) {
        throw new IllegalStateException("non-finite GLiNER logit");
      }
    }
    return logits;
  }

  @Override
  public synchronized List<Result> classify(Request request) {
    return decode(request, logits(request));
  }

  /** Publisher's softmax/sigmoid decoding, including multi-label top-label fallback. */
  public static List<Result> decode(Request request, float[] logits) {
    int expected = request.tasks().stream().mapToInt(t -> t.labels().size()).sum();
    if (logits.length != expected) {
      throw new IllegalArgumentException("logit count differs from label count");
    }
    List<Result> results = new ArrayList<>();
    int offset = 0;
    for (Task task : request.tasks()) {
      int count = task.labels().size();
      int best = 0;
      for (int i = 0; i < count; i++) {
        if (!Float.isFinite(logits[offset + i])) {
          throw new IllegalArgumentException("non-finite logit");
        }
        if (logits[offset + i] > logits[offset + best]) {
          best = i;
        }
      }
      double[] scores = new double[count];
      double sum = 0;
      for (int i = 0; i < count; i++) {
        scores[i] =
            task.multiLabel()
                ? 1.0 / (1.0 + Math.exp(-logits[offset + i]))
                : Math.exp(logits[offset + i] - logits[offset + best]);
        sum += scores[i];
      }
      boolean any = false;
      for (int i = 0; i < count; i++) {
        if (!task.multiLabel()) {
          scores[i] /= sum;
        }
        any |= scores[i] >= task.threshold();
      }
      boolean fallback = task.multiLabel() && !any;
      List<LabelScore> labels = new ArrayList<>();
      for (int i = 0; i < count; i++) {
        boolean selected =
            task.multiLabel()
                ? scores[i] >= task.threshold() || (fallback && i == best)
                : i == best;
        labels.add(new LabelScore(task.labels().get(i).name(), scores[i], selected));
      }
      results.add(new Result(task.name(), labels, fallback));
      offset += count;
    }
    return List.copyOf(results);
  }

  private void requireOpen() {
    if (encoder == null) {
      throw new IllegalStateException("GLiNER model is closed");
    }
  }

  @Override
  public synchronized void close() {
    encoder = null;
  }

  private static float[] f32(TensorSource source, String name, long... shape) {
    var tensor = source.tensor(name);
    if (!Arrays.equals(tensor.shape(), shape)
        || !tensor.storage().equals(new TensorStorage("safetensors", "F32", 1, Float.BYTES))) {
      throw new IllegalArgumentException("unexpected GLiNER tensor shape/storage: " + name);
    }
    return tensor
        .data()
        .toArray(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN));
  }

  private static void validateConfig(Path file) throws IOException {
    String modelType = null;
    String architecture = null;
    String pooling = null;
    boolean moe = true;
    try (var parser = new JsonFactory().createParser(file.toFile())) {
      if (parser.nextToken() != JsonToken.START_OBJECT) {
        throw new IllegalArgumentException("invalid GLiNER config");
      }
      while (parser.nextToken() != JsonToken.END_OBJECT) {
        if (parser.currentToken() != JsonToken.FIELD_NAME) {
          throw new IllegalArgumentException("invalid GLiNER config field");
        }
        String key = parser.currentName();
        parser.nextToken();
        switch (key) {
          case "model_type" -> modelType = parser.getValueAsString();
          case "architecture" -> architecture = parser.getValueAsString();
          case "token_pooling" -> pooling = parser.getValueAsString();
          case "use_moe" -> moe = parser.getBooleanValue();
          default -> parser.skipChildren();
        }
      }
      if (parser.nextToken() != null) {
        throw new IllegalArgumentException("trailing GLiNER config content");
      }
    }
    if (!"gliner2".equals(modelType)
        || !"span".equals(architecture)
        || !"first".equals(pooling)
        || moe) {
      throw new IllegalArgumentException("unsupported GLiNER classification graph");
    }
  }
}
