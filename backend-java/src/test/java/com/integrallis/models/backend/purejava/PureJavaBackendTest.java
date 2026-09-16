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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.api.OptimizationStatus;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.SharedPrefixInferenceBackend;
import com.integrallis.models.api.SpeculativeInferenceBackend;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.gguf.SyntheticGgufBuilder;
import com.integrallis.models.backend.purejava.safetensors.SyntheticSafetensorsBuilder;
import com.integrallis.models.runtime.InferencePipeline;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
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
    void automaticLoadingFallsBackToTheVectorApiWithoutAnInstalledProvider(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(43));

      try (PureJavaBackend backend = PureJavaBackend.loadAutomatic(modelPath)) {
        assertThat(backend.name()).isEqualTo("pure-java");
        assertThat(backend.forward(5, 0)).hasSize(VOCAB_SIZE);
      }
    }

    @Test
    void loadsGemma4ThroughTheCompleteBackendContract(@TempDir Path dir) throws IOException {
      Path modelPath = buildNanoGemma4ModelFile(dir, new Random(84));

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend.metadata().modelFamily()).isEqualTo("gemma4");
        assertThat(backend.metadata().modelName()).isEqualTo("NanoGemma4");
        assertThat(backend.tokenizer().vocabSize()).isEqualTo(VOCAB_SIZE);
        assertThat(backend.executionPlan().topology().architecture()).isEqualTo("gemma4");
        assertThat(backend.executionPlan().prefillBatchSize()).isEqualTo(1);

        float[] first = backend.forward(5, 0);
        assertFiniteLogits(first);
        backend.reset();
        assertFiniteLogits(backend.prefill(new int[] {5, 7}, 0));

        try (InferenceSession session = backend.openSession()) {
          assertFiniteLogits(backend.prefill(session, new int[] {11, 13}, 0));
          assertThat(session.checkpoint()).isEqualTo(2);
        }
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
    void batchedSessionPrefillMatchesTokenAtATimeSessionForward(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42), GgufTensorType.Q4_0);
      int[] tokens = {5, 7, 11, 13, 17};

      try (PureJavaBackend sequentialBackend = PureJavaBackend.load(modelPath);
          PureJavaBackend batchedBackend = PureJavaBackend.load(modelPath);
          InferenceSession sequential = sequentialBackend.openSession();
          InferenceSession batched = batchedBackend.openSession()) {
        assertThat(((BatchInferenceBackend) batchedBackend).maxBatchSize())
            .as("the nano plan must batch, or this test proves nothing")
            .isGreaterThanOrEqualTo(2);
        float[] expected = null;
        for (int index = 0; index < tokens.length; index++) {
          expected = sequentialBackend.forward(sequential, tokens[index], index);
        }
        float[] expectedNext = sequentialBackend.forward(sequential, 19, tokens.length).clone();

        float[] actual = batchedBackend.prefill(batched, tokens, 0).clone();

        assertThat(batched.checkpoint()).isEqualTo(tokens.length);
        assertThat(actual).containsExactly(expected, within(1e-4f));
        assertThat(batchedBackend.forward(batched, 19, tokens.length))
            .as("the session's KV lineage after a batched prefill must continue identically")
            .containsExactly(expectedNext, within(1e-4f));
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
        assertThat(batching.supportsRaggedPrefillBatch()).isTrue();
        try (InferenceSession first = batching.openSession();
            InferenceSession second = batching.openSession()) {
          var promptLogits =
              batching.prefillBatch(
                  new InferenceSession[] {first, second}, new int[][] {{5, 7}, {11, 13, 17}});

          assertThat(promptLogits.tokenCount()).isEqualTo(2);
          assertThat(promptLogits.vocabularySize()).isEqualTo(VOCAB_SIZE);

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
    void exposesPhysicallySharedPrefixForksThroughThePublicBackendSpi(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(44), GgufTensorType.Q4_0);
      int[] prefixTokens = {5, 7, 11, 13};

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath);
          PureJavaBackend baseline = PureJavaBackend.load(modelPath)) {
        assertThat(backend).isInstanceOf(SharedPrefixInferenceBackend.class);
        SharedPrefixInferenceBackend sharing = backend;
        InferenceSession source = sharing.openSession();
        sharing.prefill(source, prefixTokens, 0);

        var prefix = sharing.freezePrefix(source);

        assertThat(prefix.checkpoint()).isEqualTo(prefixTokens.length);
        assertThat(prefix.sharedBytes()).isPositive();
        assertThat(source.isClosed()).isTrue();
        try (InferenceSession first = sharing.fork(prefix);
            InferenceSession second =
                sharing.fork(prefix, SharedPrefixInferenceBackend.Branch.BASE);
            InferenceSession expected = baseline.openSession()) {
          baseline.prefill(expected, prefixTokens, 0);

          assertThat(sharing.sharesPrefixStorage(first, second)).isTrue();
          assertThat(first.allocatedStateBytes()).isPresent();
          assertThat(first.allocatedStateBytes().orElseThrow()).isPositive();
          assertThat(second.allocatedStateBytes()).isPresent();
          assertThat(second.allocatedStateBytes().orElseThrow()).isPositive();
          assertThat(
                  first.allocatedStateBytes().orElseThrow()
                      + second.allocatedStateBytes().orElseThrow()
                      - prefix.sharedBytes())
              .isPositive();
          assertThat(sharing.forward(first, 17, prefixTokens.length))
              .containsExactly(baseline.forward(expected, 17, prefixTokens.length));
          assertThat(sharing.forward(second, 19, prefixTokens.length)).hasSize(VOCAB_SIZE);
        }

        assertThat(sharing.supportsActivatedBranch()).isFalse();
        assertThatThrownBy(
                () -> sharing.fork(prefix, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER))
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("activated adapter");
      }
    }

    @Test
    void loadsAPinnedActivatedAdapterAndKeepsTheBaseBranchExact(@TempDir Path dir)
        throws Exception {
      Path modelPath = buildNanoModelFile(dir, new Random(45), GgufTensorType.Q4_0);
      Path adapterDirectory = Files.createDirectory(dir.resolve("tool-adapter"));
      writeNanoAdapter(adapterDirectory, modelPath);

      try (PureJavaBackend activated =
              PureJavaBackend.loadActivatedAdapter(modelPath, adapterDirectory);
          PureJavaBackend baseline = PureJavaBackend.load(modelPath)) {
        assertThat(activated.supportsActivatedBranch()).isTrue();
        assertThat(activated.activatedAdapter())
            .hasValueSatisfying(
                adapter -> {
                  assertThat(adapter.baseModel()).isEqualTo("test/nano-llama");
                  assertThat(adapter.invocationTokens()).containsExactly(11, 13);
                });
        InferenceSession source = activated.openSession();
        activated.prefill(source, new int[] {5, 7}, 0);
        var prefix = activated.freezePrefix(source);

        try (InferenceSession base = activated.fork(prefix);
            InferenceSession specialist =
                activated.fork(prefix, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER);
            InferenceSession expected = baseline.openSession()) {
          baseline.prefill(expected, new int[] {5, 7}, 0);
          float[] baseLogits = activated.forward(base, 11, 2);
          float[] specialistLogits = activated.forward(specialist, 11, 2);

          assertThat(activated.sharesPrefixStorage(base, specialist)).isTrue();
          assertThat(baseLogits).containsExactly(baseline.forward(expected, 11, 2));
          assertThat(specialistLogits).isNotEqualTo(baseLogits);
        }

        try (InferenceSession wrong =
            activated.fork(prefix, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER)) {
          assertThatThrownBy(() -> activated.forward(wrong, 12, 2))
              .isInstanceOf(IllegalArgumentException.class)
              .hasMessageContaining("invocation token 0")
              .hasMessageContaining("must be 11");
        }

        try (InferenceSession specialist = activated.openSession();
            InferenceSession expected = baseline.openSession()) {
          activated.prefill(specialist, new int[] {5, 7}, 0);
          baseline.prefill(expected, new int[] {5, 7, 11}, 0);

          activated.activateAdapter(specialist);
          activated.forward(specialist, 11, 2);
          float[] specialistLogits = activated.forward(specialist, 13, 3);

          assertThat(specialistLogits).isNotEqualTo(baseline.forward(expected, 13, 3));
          assertThatThrownBy(() -> activated.activateAdapter(specialist))
              .isInstanceOf(IllegalStateException.class)
              .hasMessageContaining("already activated");
          assertThatThrownBy(() -> activated.freezePrefix(specialist))
              .isInstanceOf(IllegalStateException.class)
              .hasMessageContaining("adapter-tainted");
        }
      }
    }

    @Test
    void exposesActivatedSessionHiddenStateWithoutBreakingPhysicalPrefixSharing(@TempDir Path dir)
        throws Exception {
      Path modelPath = buildNanoModelFile(dir, new Random(45), GgufTensorType.Q4_0);
      Path adapterDirectory = Files.createDirectory(dir.resolve("tool-hidden-adapter"));
      writeNanoAdapter(adapterDirectory, modelPath);

      try (PureJavaBackend backend =
          PureJavaBackend.loadActivatedAdapter(modelPath, adapterDirectory)) {
        InferenceSession source = backend.openSession();
        backend.prefill(source, new int[] {5, 7}, 0);
        var prefix = backend.freezePrefix(source);
        try (InferenceSession base = backend.fork(prefix);
            InferenceSession specialist =
                backend.fork(prefix, SharedPrefixInferenceBackend.Branch.ACTIVATED_ADAPTER)) {
          float[] baseHidden = backend.forwardHiddenState(base, 11, 2);
          float[] specialistHidden = backend.forwardHiddenState(specialist, 11, 2);

          assertThat(backend.supportsHiddenState()).isTrue();
          assertThat(baseHidden).hasSize(DIM);
          assertThat(specialistHidden).hasSize(DIM).isNotEqualTo(baseHidden);
          assertThat(backend.sharesPrefixStorage(base, specialist)).isTrue();
          assertThat(base.checkpoint()).isEqualTo(3);
          assertThat(specialist.checkpoint()).isEqualTo(3);
        }
      }
    }

    @Test
    void exposesIndependentHighLevelGenerationSessions(@TempDir Path dir) throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(42), GgufTensorType.Q4_0);
      var options = SamplingOptions.builder().temperature(0).maxTokens(1).build();

      try (InferencePipeline pipeline = new InferencePipeline(PureJavaBackend.load(modelPath));
          var first = pipeline.openGenerationSession();
          var second = pipeline.openGenerationSession()) {
        first.generate("t5", options);

        assertThat(first.contextWindow().position().orElseThrow()).isPositive();
        assertThat(second.contextWindow().position()).hasValue(0);
        assertThat(first.lastGenerationMetrics().available()).isTrue();
        assertThat(second.lastGenerationMetrics().available()).isFalse();

        second.generate("t7", options);

        assertThat(second.contextWindow().position().orElseThrow()).isPositive();
        assertThat(second.lastGenerationMetrics().available()).isTrue();
      }
    }

    @Test
    void closedRawSessionDetachesItsOwnedKvState(@TempDir Path dir) throws Exception {
      Path modelPath = buildNanoModelFile(dir, new Random(146), GgufTensorType.Q4_0);

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        InferenceSession session = backend.openSession();
        backend.prefill(session, new int[] {5, 7}, 0);
        assertThat(sessionDelegate(session)).isNotNull();

        session.close();
        session.close();

        assertThat(session.isClosed()).isTrue();
        assertThat(sessionDelegate(session)).isNull();
      }
    }

    @Test
    void freezingTransfersPrefixOwnershipAndDetachesTheSourceSession(@TempDir Path dir)
        throws Exception {
      Path modelPath = buildNanoModelFile(dir, new Random(147), GgufTensorType.Q4_0);

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        InferenceSession source = backend.openSession();
        backend.prefill(source, new int[] {5, 7}, 0);
        var prefix = backend.freezePrefix(source);

        assertThat(source.isClosed()).isTrue();
        assertThat(sessionDelegate(source)).isNull();
        InferenceSession fork = backend.fork(prefix);
        assertThat(backend.forward(fork, 11, 2)).hasSize(VOCAB_SIZE);
        fork.close();
        assertThat(sessionDelegate(fork)).isNull();
      }
    }

    @Test
    void backendCloseDetachesEveryStillOpenRawSession(@TempDir Path dir) throws Exception {
      Path modelPath = buildNanoModelFile(dir, new Random(148), GgufTensorType.Q4_0);
      PureJavaBackend backend = PureJavaBackend.load(modelPath);
      InferenceSession first = backend.openSession();
      InferenceSession second = backend.openSession();
      backend.prefill(first, new int[] {5, 7}, 0);
      backend.prefill(second, new int[] {11, 13}, 0);

      backend.close();

      assertThat(first.isClosed()).isTrue();
      assertThat(second.isClosed()).isTrue();
      assertThat(sessionDelegate(first)).isNull();
      assertThat(sessionDelegate(second)).isNull();
    }

    @Test
    void capsRuntimeContextLengthWithoutChangingModelMetadata(@TempDir Path dir)
        throws IOException {
      Path modelPath = buildNanoModelFile(dir, new Random(43));
      String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "4");

      try (PureJavaBackend backend = PureJavaBackend.load(modelPath)) {
        assertThat(backend.metadata().contextLength()).isEqualTo(CONTEXT);
        assertThat(backend.contextCapacity()).isEqualTo(4);
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

  private static void writeNanoAdapter(Path directory, Path modelPath) throws Exception {
    SyntheticSafetensorsBuilder tensors = new SyntheticSafetensorsBuilder();
    int keyValueDimension = KV_HEADS * (DIM / HEADS);
    for (int layer = 0; layer < LAYERS; layer++) {
      addNanoLoraProjection(tensors, layer, "self_attn.q_proj", DIM, DIM);
      addNanoLoraProjection(tensors, layer, "self_attn.k_proj", DIM, keyValueDimension);
      addNanoLoraProjection(tensors, layer, "self_attn.v_proj", DIM, keyValueDimension);
      addNanoLoraProjection(tensors, layer, "self_attn.o_proj", DIM, DIM);
      addNanoLoraProjection(tensors, layer, "mlp.gate_proj", DIM, HIDDEN_DIM);
      addNanoLoraProjection(tensors, layer, "mlp.up_proj", DIM, HIDDEN_DIM);
      addNanoLoraProjection(tensors, layer, "mlp.down_proj", HIDDEN_DIM, DIM);
    }
    Path weights = directory.resolve("adapter_model.safetensors");
    Files.write(weights, tensors.build());
    Files.writeString(
        directory.resolve("models-activated-lora.json"),
        """
        {
          "schemaVersion": 2,
          "kind": "activated-lora-tool-specialist",
          "base": {
            "model": "test/nano-llama",
            "revision": "%s",
            "artifactSha256": "%s"
          },
          "tokenizer": {
            "files": [{"name": "tokenizer.json", "sha256": "%s"}]
          },
          "adapter": {
            "file": "adapter_model.safetensors",
            "sha256": "%s",
            "rank": 1,
            "alpha": 1,
            "targetModules": [
              "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ]
          },
          "invocation": {"tokens": [11, 13]},
          "training": {
            "dataset": "test/dataset",
            "datasetRevision": "%s",
            "sourceFile": "train.jsonl",
            "sourceSha256": "%s",
            "preparedManifestSha256": "%s",
            "trainSha256": "%s",
            "validationSha256": "%s",
            "formatter": "formatter.java",
            "formatterSha256": "%s"
          }
        }
        """
            .formatted(
                "b".repeat(40),
                sha256(modelPath),
                "d".repeat(64),
                sha256(weights),
                "e".repeat(40),
                "f".repeat(64),
                "1".repeat(64),
                "2".repeat(64),
                "3".repeat(64),
                "4".repeat(64)));
  }

  private static void addNanoLoraProjection(
      SyntheticSafetensorsBuilder tensors,
      int layer,
      String module,
      int inputDimension,
      int outputDimension) {
    String prefix = "base_model.model.model.layers." + layer + "." + module;
    float[] a = new float[inputDimension];
    float[] b = new float[outputDimension];
    java.util.Arrays.fill(a, 0.01f);
    java.util.Arrays.fill(b, 0.02f);
    tensors.addF32(prefix + ".lora_A.weight", new long[] {1, inputDimension}, a);
    tensors.addF32(prefix + ".lora_B.weight", new long[] {outputDimension, 1}, b);
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var input = Files.newInputStream(path)) {
      byte[] buffer = new byte[8192];
      for (int read; (read = input.read(buffer)) >= 0; ) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static void assertFiniteLogits(float[] logits) {
    assertThat(logits).hasSize(VOCAB_SIZE);
    for (float logit : logits) {
      assertThat(Float.isFinite(logit)).isTrue();
    }
  }

  private static Path buildNanoGemma4ModelFile(Path dir, Random rng) throws IOException {
    int headLength = DIM / HEADS;
    int keyValueDim = KV_HEADS * headLength;
    int expertCount = 2;
    int expertHiddenDim = 4;
    List<String> tokens = new ArrayList<>();
    List<Float> scores = new ArrayList<>();
    for (int token = 0; token < VOCAB_SIZE; token++) {
      tokens.add("t" + token);
      scores.add(0.0f);
    }

    SyntheticGgufBuilder builder =
        new SyntheticGgufBuilder()
            .addString("general.name", "NanoGemma4")
            .addString("general.architecture", "gemma4")
            .addUint32("gemma4.embedding_length", DIM)
            .addUint32("gemma4.block_count", LAYERS)
            .addUint32("gemma4.attention.head_count", HEADS)
            .addInt32Array("gemma4.attention.head_count_kv", List.of(KV_HEADS, KV_HEADS))
            .addUint32("gemma4.vocab_size", VOCAB_SIZE)
            .addUint32("gemma4.context_length", CONTEXT)
            .addUint32("gemma4.feed_forward_length", HIDDEN_DIM)
            .addUint32("gemma4.expert_feed_forward_length", expertHiddenDim)
            .addUint32("gemma4.expert_count", expertCount)
            .addUint32("gemma4.expert_used_count", 1)
            .addUint32("gemma4.attention.key_length", headLength)
            .addUint32("gemma4.attention.key_length_swa", headLength)
            .addUint32("gemma4.attention.value_length", headLength)
            .addUint32("gemma4.attention.value_length_swa", headLength)
            .addFloat32("gemma4.rope.freq_base", 1_000_000.0f)
            .addFloat32("gemma4.rope.freq_base_swa", 10_000.0f)
            .addUint32("gemma4.rope.dimension_count", headLength)
            .addUint32("gemma4.rope.dimension_count_swa", headLength)
            .addFloat32("gemma4.attention.layer_norm_rms_epsilon", 1.0e-6f)
            .addUint32("gemma4.attention.sliding_window", 4)
            .addBoolArray("gemma4.attention.sliding_window_pattern", List.of(true, false))
            .addFloat32("gemma4.final_logit_softcapping", 30.0f)
            .addString("tokenizer.ggml.model", "gemma4")
            .addStringArray("tokenizer.ggml.tokens", tokens)
            .addFloat32Array("tokenizer.ggml.scores", scores)
            .addUint32("tokenizer.ggml.bos_token_id", 0)
            .addUint32("tokenizer.ggml.eos_token_id", 1);

    addF32(builder, "token_embd.weight", new long[] {DIM, VOCAB_SIZE}, rng);
    builder.addTensor("output_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    builder.addTensor(
        "rope_freqs.weight",
        GgufTensorType.F32,
        new long[] {headLength / 2},
        onesF32(headLength / 2));
    for (int layer = 0; layer < LAYERS; layer++) {
      addGemma4Layer(
          builder, rng, layer, layer == 0, headLength, keyValueDim, expertHiddenDim, expertCount);
    }

    Path modelPath = dir.resolve("nano-gemma4.gguf");
    Files.write(modelPath, builder.build());
    return modelPath;
  }

  private static void addGemma4Layer(
      SyntheticGgufBuilder builder,
      Random rng,
      int layer,
      boolean sliding,
      int headLength,
      int keyValueDim,
      int expertHiddenDim,
      int expertCount) {
    String prefix = "blk." + layer + ".";
    builder.addTensor(
        prefix + "attn_norm.weight", GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
    addF32(builder, prefix + "attn_q.weight", new long[] {DIM, HEADS * headLength}, rng);
    addF32(builder, prefix + "attn_k.weight", new long[] {DIM, keyValueDim}, rng);
    if (sliding) {
      addF32(builder, prefix + "attn_v.weight", new long[] {DIM, keyValueDim}, rng);
    }
    addF32(builder, prefix + "attn_output.weight", new long[] {HEADS * headLength, DIM}, rng);
    builder.addTensor(
        prefix + "attn_q_norm.weight",
        GgufTensorType.F32,
        new long[] {headLength},
        onesF32(headLength));
    builder.addTensor(
        prefix + "attn_k_norm.weight",
        GgufTensorType.F32,
        new long[] {headLength},
        onesF32(headLength));
    addGemma4Norm(builder, prefix + "post_attention_norm.weight");

    addGemma4Norm(builder, prefix + "ffn_norm.weight");
    addF32(builder, prefix + "ffn_gate.weight", new long[] {DIM, HIDDEN_DIM}, rng);
    addF32(builder, prefix + "ffn_up.weight", new long[] {DIM, HIDDEN_DIM}, rng);
    addF32(builder, prefix + "ffn_down.weight", new long[] {HIDDEN_DIM, DIM}, rng);
    addGemma4Norm(builder, prefix + "pre_ffw_norm_2.weight");
    addGemma4Norm(builder, prefix + "post_ffw_norm_1.weight");
    addGemma4Norm(builder, prefix + "post_ffw_norm_2.weight");
    addGemma4Norm(builder, prefix + "post_ffw_norm.weight");

    addGemma4Norm(builder, prefix + "ffn_gate_inp.scale");
    addF32(builder, prefix + "ffn_gate_inp.weight", new long[] {DIM, expertCount}, rng);
    builder.addTensor(
        prefix + "ffn_down_exps.scale",
        GgufTensorType.F32,
        new long[] {expertCount},
        onesF32(expertCount));
    builder.addTensor(
        prefix + "layer_output_scale.weight", GgufTensorType.F32, new long[] {1}, onesF32(1));
    addF32(
        builder,
        prefix + "ffn_gate_up_exps.weight",
        new long[] {DIM, 2L * expertHiddenDim, expertCount},
        rng);
    addF32(
        builder,
        prefix + "ffn_down_exps.weight",
        new long[] {expertHiddenDim, DIM, expertCount},
        rng);
  }

  private static void addGemma4Norm(SyntheticGgufBuilder builder, String name) {
    builder.addTensor(name, GgufTensorType.F32, new long[] {DIM}, onesF32(DIM));
  }

  private static void addF32(SyntheticGgufBuilder builder, String name, long[] shape, Random rng) {
    long elements = 1;
    for (long dimension : shape) {
      elements = Math.multiplyExact(elements, dimension);
    }
    builder.addTensor(name, GgufTensorType.F32, shape, randomF32(rng, Math.toIntExact(elements)));
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

  private static Object sessionDelegate(InferenceSession session) throws Exception {
    var delegate = session.getClass().getDeclaredField("delegate");
    delegate.setAccessible(true);
    return delegate.get(session);
  }
}
