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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Steady-state decode rate on the {@code rust-ffm} route, for comparison against the identical
 * measurement on the pure-Java route.
 *
 * <p>The sibling probe in {@code backend-java} runs {@code PureJavaBackend.load(Path)}, which
 * passes {@code GgufBatchedMatrixKernel.none()}. This one passes the bundled Rust kernel instead.
 * Nothing else differs: same prompt, same warmup, same token count, same context. The pair measures
 * what the native kernels are worth on a given model, which is the question that decides whether a
 * model can clear the relative decode gate -- 65 of 68 catalog entries qualify on this route and
 * the three that do not are grandfathered.
 *
 * <p>Prints rather than asserts a rate: a throughput threshold would be measuring the box.
 */
@Tag("slow")
@EnabledIfSystemProperty(named = NativeDecodeRateProbe.MODEL_PROPERTY, matches = ".+")
class NativeDecodeRateProbe {

  static final String MODEL_PROPERTY = "models.fixtures.decodeRateModel";

  @Test
  void reportsSteadyStateDecodeRateOnTheNativeRoute() {
    Path model = Path.of(System.getProperty(MODEL_PROPERTY));
    int warmup = Integer.getInteger("models.fixtures.decodeRateWarmup", 12);
    int measured = Integer.getInteger("models.fixtures.decodeRateTokens", 120);
    int threads =
        Integer.getInteger(
            "models.fixtures.decodeRateThreads", Runtime.getRuntime().availableProcessors());
    // EdgeRock declares a context of 262144; loading it there asks for roughly 17 GB of key and
    // value cache. Both arms must also agree on this for their rates to be comparable.
    String context = System.getProperty("models.fixtures.decodeRateContext", "16384");
    String previous = System.getProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
    System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, context);

    // The locally cargo-built library, which every test in this module is given by the build:
    // tasks.withType<Test> sets models.native.kernels.library to it. openBundled() instead wants a
    // packaged classpath artifact for the host platform, and there is none for macos-x86_64.
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    try (PureJavaBackend backend =
        PureJavaBackend.load(model, RustGgufBatchedMatrixKernel.open(library, threads))) {
      int[] prompt = backend.tokenizer().encode("The capital of France is");
      float[] logits = null;
      int position = 0;
      for (int token : prompt) {
        logits = backend.forward(token, position++);
      }
      for (int step = 0; step < warmup; step++) {
        logits = backend.forward(argmax(logits), position++);
      }

      long start = System.nanoTime();
      for (int step = 0; step < measured; step++) {
        logits = backend.forward(argmax(logits), position++);
      }
      double seconds = (System.nanoTime() - start) / 1e9;

      assertThat(logits).isNotNull();
      System.out.printf(
          "[decode-rate:native] %s: %d tokens in %.1fs = %.2f tok/s "
              + "(vocab %d, warmup %d, threads %d, context %s)%n",
          model.getFileName(),
          measured,
          seconds,
          measured / seconds,
          logits.length,
          warmup,
          threads,
          context);
    } finally {
      if (previous == null) {
        System.clearProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY);
      } else {
        System.setProperty(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, previous);
      }
    }
  }

  @Test
  void theMemoryLayerAgreesWithTheJavaRouteOnRealWeights() throws java.io.IOException {
    // The speedup above comes from sending the memory layer's projections through the Rust kernel
    // instead of the calling thread. That is only a speedup if it computes the same thing, and a
    // projection that disagrees here would show up as fluent, slightly wrong text rather than as a
    // failure, so the two routes are compared directly on real weights.
    //
    // Pure-Java reference for this fixture, measured in backend-java: L1 = 5.94345993946024.
    Path model = Path.of(System.getProperty(MODEL_PROPERTY));
    Path library = Path.of(System.getProperty("models.native.kernels.library"));

    double javaRoute;
    double nativeRoute;
    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared()) {
      com.integrallis.models.backend.purejava.gguf.GgufFile file =
          com.integrallis.models.backend.purejava.gguf.GgufParser.parse(model, arena);
      javaRoute =
          memoryL1(
              com.integrallis.models.backend.purejava.edgerock.EdgeRockMemory.load(file), "java");
      try (RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(library, 8)) {
        nativeRoute =
            memoryL1(
                com.integrallis.models.backend.purejava.edgerock.EdgeRockMemory.load(file, kernel),
                "native");
      }
    }

    System.out.printf("[memory-parity] java=%.9f native=%.9f%n", javaRoute, nativeRoute);
    assertThat(javaRoute).isGreaterThan(0.0);
    // Not bit-for-bit: the kernels reduce in a different order, which is expected and is why this
    // asserts a relative tolerance rather than equality.
    assertThat(nativeRoute)
        .isCloseTo(javaRoute, org.assertj.core.api.Assertions.withinPercentage(1.0));
  }

  @Test
  void theFusedProjectionRouteIsActuallyTakenOnRealWeights() throws java.io.IOException {
    // Without this the fusion is unobservable. If isTripleEligible says no at batch size 1, the
    // layer falls back to individual projections, nothing changes, and an A/B would report "no
    // effect" when the truth is "the fast path never ran".
    Path model = Path.of(System.getProperty(MODEL_PROPERTY));
    Path library = Path.of(System.getProperty("models.native.kernels.library"));
    try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared();
        RustGgufBatchedMatrixKernel kernel = RustGgufBatchedMatrixKernel.open(library, 8)) {
      var file = com.integrallis.models.backend.purejava.gguf.GgufParser.parse(model, arena);
      var memory =
          com.integrallis.models.backend.purejava.edgerock.EdgeRockMemory.load(file, kernel);
      var weights =
          com.integrallis.models.backend.purejava.edgerock.EdgeRockWeights.load(
                  file, memory.config(), kernel)
              .memoryLayer(memory.config().fullAttentionLayers()[0]);

      int embedding = memory.config().embeddingLength();
      int heads = memory.config().headCount();
      int groups = memory.config().headCountKv();
      int dim = memory.config().headDim();
      int writer = memory.config().memoryWriterIntermediate();
      float[] input = new float[embedding];
      java.util.Arrays.fill(input, 0.01f);

      boolean triple =
          com.integrallis.models.backend.purejava.edgerock.QuantizedProjection.fusedTriple(
              weights.queryProjection(),
              new float[heads * dim],
              weights.writeGate(),
              new float[writer],
              weights.writeUp(),
              new float[writer],
              input);
      boolean dual =
          com.integrallis.models.backend.purejava.edgerock.QuantizedProjection.fusedDual(
              weights.keyProjection(),
              new float[groups * dim],
              weights.valueProjection(),
              new float[groups * dim],
              input);

      System.out.printf("[fusion] triple=%s dual=%s%n", triple, dual);
      assertThat(triple)
          .as("q_proj + write_gate + write_up must fuse on the real kernel, or the port is inert")
          .isTrue();
      assertThat(dual).as("k_proj + v_proj must fuse").isTrue();
    }
  }

  private static double memoryL1(
      com.integrallis.models.backend.purejava.edgerock.EdgeRockMemory memory, String label) {
    var banks = memory.newBanks();
    int embedding = memory.config().embeddingLength();
    int layer = memory.config().fullAttentionLayers()[0];
    int pairs = memory.config().ropeDimensionCount() / 2;
    float[] cos = new float[pairs];
    float[] sin = new float[pairs];
    java.util.Arrays.fill(cos, 1.0f);
    var rotation =
        com.integrallis.models.backend.purejava.edgerock.EdgeRockMemoryLayer.rotation(cos, sin);
    float[] hidden = new float[embedding];
    float[] out = new float[embedding];
    java.util.Random random = new java.util.Random(20_261_030L);
    int chunk = memory.config().memoryCommitChunk();
    for (int position = 0; position <= chunk; position++) {
      for (int i = 0; i < embedding; i++) {
        hidden[i] = (float) random.nextGaussian() * 0.1f;
      }
      memory.layer(layer).forward(hidden, position, rotation, banks[layer], out);
    }
    double total = 0;
    for (float value : out) {
      total += Math.abs(value);
    }
    return total;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}
