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

import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * What a second and third resident native model cost the first on one host.
 *
 * <p>Each {@code RustFfmBackend.load} used to build its own {@link NativeKernelLibrary}, and each
 * of those created a worker pool holding one thread per logical CPU, so three resident models ran
 * three pools' worth of workers against one machine's cores. {@link
 * NativeKernelLibrary#SHARED_POOL_PROPERTY} selects between that and one pool per process, so the
 * two are measured against each other here rather than argued about.
 *
 * <p>Run one JVM per arm, because the shared pool is created once per process:
 *
 * <pre>
 *   ./gradlew :backend-native:test --tests '*SharedWorkerPoolOversubscriptionSlowTest' \
 *     -Dmodels.native.kernels.sharedPool=true  -PincludeTags=oversubscription
 *   ./gradlew :backend-native:test --tests '*SharedWorkerPoolOversubscriptionSlowTest' \
 *     -Dmodels.native.kernels.sharedPool=false -PincludeTags=oversubscription
 * </pre>
 */
@Tag("slow")
class SharedWorkerPoolOversubscriptionSlowTest {

  private static final int DECODE_STEPS = 24;
  private static final int REPETITIONS = 5;
  private static final String PROMPT = "I was charged twice for my September order.";

  /** The three catalog models that qualify on rust-ffm, by artifact SHA-256. */
  private static final String TIMED = "48ab3034d0dd401f";

  private static final List<String> RESIDENT = List.of("7b064f5842bf9532", "00fe7986ff5f6b46");

  private static Path artifact(String shaPrefix) throws IOException {
    Path root = Path.of(System.getProperty("user.home"), ".modeljars", "cache", "sha256");
    try (var buckets = Files.list(root.resolve(shaPrefix.substring(0, 2)))) {
      return buckets
          .filter(dir -> dir.getFileName().toString().startsWith(shaPrefix))
          .map(dir -> dir.resolve("model.gguf"))
          .findFirst()
          .orElseThrow(() -> new IllegalStateException("not cached: " + shaPrefix));
    }
  }

  private static BackendConfiguration shortContext() {
    return new BackendConfiguration(
        Map.of(), Map.of(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "256"), List.of());
  }

  /** Milliseconds for a prefill plus {@value #DECODE_STEPS} greedy steps. */
  private static long decodeMillis(RustFfmBackend backend) {
    int[] tokens = backend.tokenizer().encode(PROMPT);
    backend.reset();
    long began = System.nanoTime();
    float[] logits = backend.prefill(tokens, 0);
    for (int step = 0; step < DECODE_STEPS; step++) {
      logits = backend.forward(argmax(logits), tokens.length + step);
    }
    return (System.nanoTime() - began) / 1_000_000;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }

  private static long nativeThreads() throws IOException, InterruptedException {
    var process =
        new ProcessBuilder("ps", "-M", Long.toString(ProcessHandle.current().pid()))
            .redirectErrorStream(true)
            .start();
    try (var lines =
        new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))) {
      long count = lines.lines().count() - 2; // header plus the process line
      process.waitFor();
      return count;
    }
  }

  @Test
  void aResidentFleetCostsTheFirstModelLessWhenItSharesThePool() throws Exception {
    boolean shared =
        !"false".equals(System.getProperty(NativeKernelLibrary.SHARED_POOL_PROPERTY, "true"));
    int processors = Runtime.getRuntime().availableProcessors();

    try (var timed = RustFfmBackend.load(artifact(TIMED), shortContext())) {
      long threadsOneModel = nativeThreads();
      long alone = fastest(timed);

      // The discriminator: what one extra resident native model adds. A pool per model adds a
      // pool's worth of workers; sharing adds none, leaving only that model's non-worker threads.
      long perModelDelta;
      var others = new ArrayList<RustFfmBackend>();
      try {
        others.add(RustFfmBackend.load(artifact(RESIDENT.get(0)), shortContext()));
        perModelDelta = nativeThreads() - threadsOneModel;

        for (int index = 1; index < RESIDENT.size(); index++) {
          others.add(RustFfmBackend.load(artifact(RESIDENT.get(index)), shortContext()));
        }
        long threadsResident = nativeThreads();
        long contended = fastest(timed);

        System.out.printf(
            "%narm=%s processors=%d loadavg=%s%n"
                + "threads: %d with 1 model, %d with %d, +%d for the first extra model%n"
                + "timed model: %d ms alone, %d ms with %d resident, ratio %.2fx%n",
            shared ? "shared-pool" : "pool-per-model",
            processors,
            loadAverage(),
            threadsOneModel,
            threadsResident,
            1 + RESIDENT.size(),
            perModelDelta,
            alone,
            contended,
            RESIDENT.size(),
            (double) contended / alone);

        assertThat(contended).isPositive();
        if (shared) {
          assertThat(perModelDelta)
              .as("a shared pool adds no worker threads for an extra resident model")
              .isLessThan(processors);
        } else {
          assertThat(perModelDelta)
              .as("a pool per model adds a pool's worth of workers")
              .isGreaterThanOrEqualTo(processors - 1L);
        }
      } finally {
        others.forEach(RustFfmBackend::close);
      }
    }
  }

  /** The fastest of several runs, which is the figure least shaped by whatever else is running. */
  private static long fastest(RustFfmBackend backend) {
    long best = Long.MAX_VALUE;
    for (int repetition = 0; repetition < REPETITIONS; repetition++) {
      best = Math.min(best, decodeMillis(backend));
    }
    return best;
  }

  private static String loadAverage() throws IOException, InterruptedException {
    var process =
        new ProcessBuilder("sysctl", "-n", "vm.loadavg").redirectErrorStream(true).start();
    try (var reader =
        new java.io.BufferedReader(new java.io.InputStreamReader(process.getInputStream()))) {
      String line = reader.readLine();
      process.waitFor();
      return line == null ? "unknown" : line.trim();
    }
  }
}
