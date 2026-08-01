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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class PureJavaBackendTest {

  private static final int DIM = 32;
  private static final int HEADS = 2;
  private static final int KV_HEADS = 1;
  private static final int HIDDEN_DIM = 32;
  private static final int VOCAB_SIZE = 32;
  private static final int LAYERS = 2;
  private static final int CONTEXT = 64;

  static Path buildNanoModelFile(Path dir, Random rng) throws IOException {
    return buildNanoModelFile(dir, rng, GgufTensorType.F32);
  }

  static Path buildNanoModelFile(Path dir, Random rng, GgufTensorType projectionType)
      throws IOException {
    List<String> tokens = new ArrayList<>();
    for (int i = 0; i < VOCAB_SIZE; i++) {
      tokens.add("t" + i);
    }

    List<Float> scores = new ArrayList<>();
    for (int i = 0; i < VOCAB_SIZE; i++) {
      scores.add(0.0f);
    }

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.name", "NanoTest")
            .addString("general.architecture", "llama")
            .addUint32("llama.embedding_length", DIM)
            .addUint32("llama.block_count", LAYERS)
            .addUint32("llama.attention.head_count", HEADS)
            .addUint32("llama.attention.head_count_kv", KV_HEADS)
            .addUint32("llama.vocab_size", VOCAB_SIZE)
            .addUint32("llama.context_length", CONTEXT)
            .addUint32("llama.feed_forward_length", HIDDEN_DIM)
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);

    builder.addTensor(
        "token_embd.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    builder.addTensor(
        "output.weight",
        GgufTensorType.F32,
        new long[] {DIM, VOCAB_SIZE},
        randomF32(rng, VOCAB_SIZE * DIM));

    int kvDim = KV_HEADS * (DIM / HEADS);
    for (int l = 0; l < LAYERS; l++) {
      String prefix = "blk." + l + ".";
      builder.addTensor(
          prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "attn_q.weight",
          projectionType,
          new long[] {DIM, DIM},
          projectionData(rng, projectionType, DIM * DIM));
      builder.addTensor(
          prefix + "attn_k.weight",
          projectionType,
          new long[] {DIM, kvDim},
          projectionData(rng, projectionType, kvDim * DIM));
      builder.addTensor(
          prefix + "attn_v.weight",
          projectionType,
          new long[] {DIM, kvDim},
          projectionData(rng, projectionType, kvDim * DIM));
      builder.addTensor(
          prefix + "attn_output.weight",
          projectionType,
          new long[] {DIM, DIM},
          projectionData(rng, projectionType, DIM * DIM));
      builder.addTensor(
          prefix + "ffn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
      builder.addTensor(
          prefix + "ffn_gate.weight",
          projectionType,
          new long[] {DIM, HIDDEN_DIM},
          projectionData(rng, projectionType, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_up.weight",
          projectionType,
          new long[] {DIM, HIDDEN_DIM},
          projectionData(rng, projectionType, HIDDEN_DIM * DIM));
      builder.addTensor(
          prefix + "ffn_down.weight",
          projectionType,
          new long[] {HIDDEN_DIM, DIM},
          projectionData(rng, projectionType, DIM * HIDDEN_DIM));
    }

    byte[] data = builder.build();
    Path modelPath = dir.resolve("nano.gguf");
    Files.write(modelPath, data);
    return modelPath;
  }

  @Nested
  class LoadAndInfer {

    @Test
    void loadsModelAndRunsForward(@TempDir Path dir) throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42));

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend.name()).isEqualTo("pure-java");
        assertThat(backend.metadata().modelName()).isEqualTo("NanoTest");
        assertThat(backend.metadata().modelFamily()).isEqualTo("llama");
        assertThat(backend.tokenizer().vocabSize()).isEqualTo(VOCAB_SIZE);
        assertThat(backend.diagnostics().backend()).isEqualTo("pure-java");
        assertThat(backend.diagnostics().optimization("mapped-model-weights"))
            .hasValueSatisfying(
                decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));

        float[] logits = backend.forward(5, 0);
        assertThat(logits).hasSize(VOCAB_SIZE);
      }
    }

    @Test
    void appliesAnExternalBackendProfileWithoutDependingOnItsRegistry(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42));
      BackendConfiguration profile =
          new BackendConfiguration(
              Map.of("profile-source", "test-registry"),
              Map.of(),
              List.of(
                  new OptimizationDecision(
                      "profile.nano",
                      OptimizationStatus.ENABLED,
                      "the exact artifact and runtime matched",
                      Map.of("artifact", "nano"))));

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath, profile)) {
        assertThat(backend.diagnostics().environment())
            .containsEntry("profile-source", "test-registry");
        assertThat(backend.diagnostics().optimization("profile.nano"))
            .hasValueSatisfying(
                decision -> assertThat(decision.status()).isEqualTo(OptimizationStatus.ENABLED));
      }
    }

    @Test
    void exposesSpeculativeVerificationAndRollback(@TempDir Path dir) throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42));

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend).isInstanceOf(SpeculativeInferenceBackend.class);
        backend.prefill(new int[] {5, 7}, 0);
        int checkpoint = backend.checkpoint();

        var verification = backend.verify(new int[] {11, 13}, checkpoint);

        assertThat(verification.tokenCount()).isEqualTo(2);
        assertThat(verification.vocabularySize()).isEqualTo(VOCAB_SIZE);
        assertThat(backend.checkpoint()).isEqualTo(checkpoint + 2);

        backend.rewind(checkpoint + 1);
        assertThat(backend.checkpoint()).isEqualTo(checkpoint + 1);
        assertThat(backend.forward(17, checkpoint + 1)).hasSize(VOCAB_SIZE);
      }
    }

    @Test
    void exposesIndependentSessionBatchingThroughTheBackendSpi(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42), GgufTensorType.Q4_0);

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend).isInstanceOf(BatchInferenceBackend.class);
        BatchInferenceBackend batching = backend;
        assertThat(batching.maxBatchSize()).isGreaterThanOrEqualTo(2);
        try (InferenceSession first = batching.openSession();
            InferenceSession second = batching.openSession()) {
          batching.prefill(first, new int[] {5, 7}, 0);
          batching.prefill(second, new int[] {11, 13, 17}, 0);

          var logits =
              batching.forwardBatch(new InferenceSession[] {first, second}, new int[] {19, 23});

          assertThat(logits.tokenCount()).isEqualTo(2);
          assertThat(logits.vocabularySize()).isEqualTo(VOCAB_SIZE);
          assertThat(first.checkpoint()).isEqualTo(3);
          assertThat(second.checkpoint()).isEqualTo(4);
        }
      }
    }

    @Test
    void capsRuntimeContextLengthWithoutChangingModelMetadata(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(43));
      String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "4");

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend.metadata().contextLength()).isEqualTo(CONTEXT);
        for (int position = 0; position <= 3; position++) {
          assertThat(backend.forward(5, position)).hasSize(VOCAB_SIZE);
        }
        assertThatThrownBy(() -> backend.forward(5, 4))
            .isInstanceOf(IllegalArgumentException.class)
            .hasMessageContaining("position out of range");
      } finally {
        restoreSystemProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }

    @Test
    void nonExistentFileThrows(@TempDir Path dir) {
      Path noFile = dir.resolve("missing.gguf");
      assertThatThrownBy(() -> PureJavaBackend.load(noFile))
          .isInstanceOf(UncheckedIOException.class);
    }

    @Test
    void closesOwnedArenaWhenRuntimeFailureInterruptsLoading(@TempDir Path dir) throws IOException {
      Path invalidModel = dir.resolve("invalid.gguf");
      Files.write(invalidModel, new byte[32]);
      Arena arena = Arena.ofShared();

      assertThatThrownBy(() -> PureJavaBackend.load(invalidModel, arena))
          .isInstanceOf(RuntimeException.class);
      assertThat(arena.scope().isAlive()).isFalse();
    }
  }

  private static byte[] randomF32(Random rng, int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, (rng.nextFloat() - 0.5f) * 0.1f);
    }
    return data;
  }

  private static byte[] projectionData(Random rng, GgufTensorType type, int count) {
    return switch (type) {
      case F32 -> randomF32(rng, count);
      case Q4_0 -> randomQ40(rng, count);
      default -> throw new IllegalArgumentException("unsupported test projection type: " + type);
    };
  }

  private static byte[] randomQ40(Random rng, int count) {
    if (count % 32 != 0) {
      throw new IllegalArgumentException("Q4_0 value count must be a multiple of 32");
    }
    byte[] data = new byte[(count / 32) * 18];
    rng.nextBytes(data);
    ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int block = 0; block < count / 32; block++) {
      buffer.putShort(block * 18, Float.floatToFloat16(0.01f));
    }
    return data;
  }

  private static byte[] onesF32(int count) {
    byte[] data = new byte[count * 4];
    ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < count; i++) {
      buf.putFloat(i * 4, 1.0f);
    }
    return data;
  }

  private static void restoreSystemProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
