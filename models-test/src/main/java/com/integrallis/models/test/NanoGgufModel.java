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
package com.integrallis.models.test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/** Writes a deterministic, tiny GGUF model for examples and downstream integration tests. */
public final class NanoGgufModel {
  private static final int GGUF_MAGIC = 0x46554747;
  private static final int GGUF_VERSION = 3;
  private static final int GGUF_ALIGNMENT = 32;
  private static final int GGUF_TYPE_UINT32 = 4;
  private static final int GGUF_TYPE_FLOAT32 = 6;
  private static final int GGUF_TYPE_STRING = 8;
  private static final int GGUF_TYPE_ARRAY = 9;
  private static final int GGUF_TENSOR_F32 = 0;

  private static final int DIMENSION = 16;
  private static final int ATTENTION_HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HIDDEN_DIMENSION = 32;
  private static final int VOCABULARY_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT_LENGTH = 64;

  private NanoGgufModel() {}

  /**
   * Writes the fixture to {@code target}, replacing an existing file.
   *
   * @return the supplied target path
   */
  public static Path write(Path target) throws IOException {
    Objects.requireNonNull(target, "target");
    Path parent = target.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Files.write(target, build());
    return target;
  }

  private static byte[] build() {
    List<String> tokens = new ArrayList<>(VOCABULARY_SIZE);
    List<Float> scores = new ArrayList<>(VOCABULARY_SIZE);
    for (int index = 0; index < VOCABULARY_SIZE; index++) {
      tokens.add("t" + index);
      scores.add(0.0f);
    }

    GgufBuilder builder =
        new GgufBuilder()
            .string("general.name", "ModelsNano")
            .string("general.architecture", "llama")
            .uint32("llama.embedding_length", DIMENSION)
            .uint32("llama.block_count", LAYERS)
            .uint32("llama.attention.head_count", ATTENTION_HEADS)
            .uint32("llama.attention.head_count_kv", KV_HEADS)
            .uint32("llama.vocab_size", VOCABULARY_SIZE)
            .uint32("llama.context_length", CONTEXT_LENGTH)
            .uint32("llama.feed_forward_length", HIDDEN_DIMENSION)
            .strings("tokenizer.ggml.tokens", tokens)
            .floats("tokenizer.ggml.scores", scores)
            .uint32("tokenizer.ggml.bos_token_id", 0)
            .uint32("tokenizer.ggml.eos_token_id", 1);

    Random random = new Random(42);
    builder.tensor(
        "token_embd.weight",
        new long[] {DIMENSION, VOCABULARY_SIZE},
        randomFloats(random, VOCABULARY_SIZE * DIMENSION));
    builder.tensor("output_norm.weight", new long[] {DIMENSION}, constantFloats(DIMENSION, 1.0f));
    builder.tensor(
        "output.weight",
        new long[] {DIMENSION, VOCABULARY_SIZE},
        randomFloats(random, VOCABULARY_SIZE * DIMENSION));

    int kvDimension = KV_HEADS * (DIMENSION / ATTENTION_HEADS);
    for (int layer = 0; layer < LAYERS; layer++) {
      String prefix = "blk." + layer + ".";
      builder.tensor(
          prefix + "attn_norm.weight", new long[] {DIMENSION}, constantFloats(DIMENSION, 1.0f));
      builder.tensor(
          prefix + "attn_q.weight",
          new long[] {DIMENSION, DIMENSION},
          randomFloats(random, DIMENSION * DIMENSION));
      builder.tensor(
          prefix + "attn_k.weight",
          new long[] {DIMENSION, kvDimension},
          randomFloats(random, kvDimension * DIMENSION));
      builder.tensor(
          prefix + "attn_v.weight",
          new long[] {DIMENSION, kvDimension},
          randomFloats(random, kvDimension * DIMENSION));
      builder.tensor(
          prefix + "attn_output.weight",
          new long[] {DIMENSION, DIMENSION},
          randomFloats(random, DIMENSION * DIMENSION));
      builder.tensor(
          prefix + "ffn_norm.weight", new long[] {DIMENSION}, constantFloats(DIMENSION, 1.0f));
      builder.tensor(
          prefix + "ffn_gate.weight",
          new long[] {DIMENSION, HIDDEN_DIMENSION},
          randomFloats(random, HIDDEN_DIMENSION * DIMENSION));
      builder.tensor(
          prefix + "ffn_up.weight",
          new long[] {DIMENSION, HIDDEN_DIMENSION},
          randomFloats(random, HIDDEN_DIMENSION * DIMENSION));
      builder.tensor(
          prefix + "ffn_down.weight",
          new long[] {HIDDEN_DIMENSION, DIMENSION},
          randomFloats(random, DIMENSION * HIDDEN_DIMENSION));
    }
    return builder.build();
  }

  private static byte[] randomFloats(Random random, int count) {
    ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(count, Float.BYTES));
    bytes.order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      bytes.putFloat((random.nextFloat() - 0.5f) * 0.1f);
    }
    return bytes.array();
  }

  private static byte[] constantFloats(int count, float value) {
    ByteBuffer bytes = ByteBuffer.allocate(Math.multiplyExact(count, Float.BYTES));
    bytes.order(ByteOrder.LITTLE_ENDIAN);
    for (int index = 0; index < count; index++) {
      bytes.putFloat(value);
    }
    return bytes.array();
  }

  private static final class GgufBuilder {
    private final List<Metadata> metadata = new ArrayList<>();
    private final List<Tensor> tensors = new ArrayList<>();

    private GgufBuilder string(String key, String value) {
      metadata.add(new Metadata(key, GGUF_TYPE_STRING, value));
      return this;
    }

    private GgufBuilder uint32(String key, int value) {
      metadata.add(new Metadata(key, GGUF_TYPE_UINT32, value));
      return this;
    }

    private GgufBuilder strings(String key, List<String> values) {
      metadata.add(new Metadata(key, GGUF_TYPE_ARRAY, new StringValues(List.copyOf(values))));
      return this;
    }

    private GgufBuilder floats(String key, List<Float> values) {
      metadata.add(new Metadata(key, GGUF_TYPE_ARRAY, new FloatValues(List.copyOf(values))));
      return this;
    }

    private void tensor(String name, long[] shape, byte[] data) {
      tensors.add(new Tensor(name, shape.clone(), data.clone()));
    }

    private byte[] build() {
      ByteArrayOutputStream output = new ByteArrayOutputStream();
      writeInt(output, GGUF_MAGIC);
      writeInt(output, GGUF_VERSION);
      writeLong(output, tensors.size());
      writeLong(output, metadata.size());

      metadata.forEach(entry -> writeMetadata(output, entry));

      long tensorOffset = 0;
      for (Tensor tensor : tensors) {
        writeString(output, tensor.name());
        writeInt(output, tensor.shape().length);
        for (long dimension : tensor.shape()) {
          writeLong(output, dimension);
        }
        writeInt(output, GGUF_TENSOR_F32);
        writeLong(output, tensorOffset);
        tensorOffset = Math.addExact(tensorOffset, tensor.data().length);
      }

      int padding = align(output.size(), GGUF_ALIGNMENT) - output.size();
      output.writeBytes(new byte[padding]);
      tensors.forEach(tensor -> output.writeBytes(tensor.data()));
      return output.toByteArray();
    }

    private static void writeMetadata(ByteArrayOutputStream output, Metadata metadata) {
      writeString(output, metadata.key());
      writeInt(output, metadata.type());
      switch (metadata.type()) {
        case GGUF_TYPE_STRING -> writeString(output, (String) metadata.value());
        case GGUF_TYPE_UINT32 -> writeInt(output, (Integer) metadata.value());
        case GGUF_TYPE_ARRAY -> writeArray(output, metadata.value());
        default ->
            throw new IllegalStateException(
                "Unsupported fixture metadata type: " + metadata.type());
      }
    }

    private static void writeArray(ByteArrayOutputStream output, Object value) {
      if (value instanceof StringValues strings) {
        writeInt(output, GGUF_TYPE_STRING);
        writeLong(output, strings.values().size());
        strings.values().forEach(item -> writeString(output, item));
        return;
      }
      if (value instanceof FloatValues floats) {
        writeInt(output, GGUF_TYPE_FLOAT32);
        writeLong(output, floats.values().size());
        floats.values().forEach(item -> writeFloat(output, item));
        return;
      }
      throw new IllegalStateException("Unsupported fixture array: " + value.getClass().getName());
    }
  }

  private static void writeInt(ByteArrayOutputStream output, int value) {
    ByteBuffer bytes = ByteBuffer.allocate(Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putInt(value);
    output.writeBytes(bytes.array());
  }

  private static void writeLong(ByteArrayOutputStream output, long value) {
    ByteBuffer bytes = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putLong(value);
    output.writeBytes(bytes.array());
  }

  private static void writeFloat(ByteArrayOutputStream output, float value) {
    ByteBuffer bytes = ByteBuffer.allocate(Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    bytes.putFloat(value);
    output.writeBytes(bytes.array());
  }

  private static void writeString(ByteArrayOutputStream output, String value) {
    byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
    writeLong(output, bytes.length);
    output.writeBytes(bytes);
  }

  private static int align(int value, int alignment) {
    return Math.addExact(value, alignment - 1) & -alignment;
  }

  private record Metadata(String key, int type, Object value) {}

  private record StringValues(List<String> values) {}

  private record FloatValues(List<Float> values) {}

  private record Tensor(String name, long[] shape, byte[] data) {}
}
