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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.models.backend.purejava.fixture.ModelFixtureDescriptor;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Locale;

/** Focused replay-versus-copy experiment for Qwen3.5 Gated DeltaNet prefix state. */
public final class Qwen35LinearStateSnapshotExperiment {
  private static final String FIXTURE_ID = "qwen3_5_0_8b_q4_k_m";
  private static final int[] TOKEN_PATTERN = {760, 3_841, 13_477, 37_550};
  private static final int TAIL_TOKEN = 321;

  private Qwen35LinearStateSnapshotExperiment() {}

  public static void main(String[] args) throws Exception {
    int prefixTokens = positiveArgument(args, 0, 32, "prefix tokens");
    int warmups = nonNegativeArgument(args, 1, 1, "warmups");
    int iterations = positiveArgument(args, 2, 5, "iterations");
    ModelFixtureDescriptor fixture =
        ModelFixtureRegistry.fromClasspath().descriptors().stream()
            .filter(candidate -> FIXTURE_ID.equals(candidate.id()))
            .findFirst()
            .orElseThrow();

    try (Arena arena = Arena.ofConfined()) {
      var file = GgufParser.parse(fixture.localPath().orElseThrow(), arena);
      Qwen35ForwardPass graph = Qwen35ForwardPass.fromGgufFile(file);
      Qwen35ForwardPass.Session session = graph.openSession(prefixTokens + 1);
      int[] prefix = repeatedTokens(prefixTokens);
      graph.prefill(session, prefix, 0);
      Qwen35ForwardPass.LinearStateSnapshot snapshot = graph.captureLinearState(session);
      float[] reference = graph.forward(session, TAIL_TOKEN, prefixTokens);

      for (int warmup = 0; warmup < warmups; warmup++) {
        runPair(graph, session, snapshot, reference, prefixTokens);
      }
      long[] replayNanos = new long[iterations];
      long[] restoreNanos = new long[iterations];
      for (int iteration = 0; iteration < iterations; iteration++) {
        Sample sample = runPair(graph, session, snapshot, reference, prefixTokens);
        replayNanos[iteration] = sample.replayNanos();
        restoreNanos[iteration] = sample.restoreNanos();
      }

      long medianReplay = median(replayNanos);
      long medianRestore = median(restoreNanos);
      System.out.printf(
          Locale.ROOT,
          """
          {
            "model": "%s",
            "prefixTokens": %d,
            "snapshotBytes": %d,
            "warmups": %d,
            "iterations": %d,
            "replayNanos": %s,
            "restoreNanos": %s,
            "medianReplayMillis": %.3f,
            "medianRestoreMillis": %.3f,
            "medianSpeedup": %.2f,
            "continuationExact": true
          }
          """,
          fixture.displayName(),
          prefixTokens,
          snapshot.bytes(),
          warmups,
          iterations,
          Arrays.toString(replayNanos),
          Arrays.toString(restoreNanos),
          nanosToMillis(medianReplay),
          nanosToMillis(medianRestore),
          (double) medianReplay / medianRestore);
    }
  }

  private static Sample runPair(
      Qwen35ForwardPass graph,
      Qwen35ForwardPass.Session session,
      Qwen35ForwardPass.LinearStateSnapshot snapshot,
      float[] reference,
      int prefixTokens) {
    long start = System.nanoTime();
    graph.rewind(session, prefixTokens);
    long replayNanos = System.nanoTime() - start;
    float[] replayed = graph.forward(session, TAIL_TOKEN, prefixTokens);

    start = System.nanoTime();
    graph.restoreLinearState(session, snapshot);
    long restoreNanos = System.nanoTime() - start;
    float[] restored = graph.forward(session, TAIL_TOKEN, prefixTokens);
    if (!Arrays.equals(reference, replayed) || !Arrays.equals(reference, restored)) {
      throw new IllegalStateException("replay and snapshot continuation logits must be exact");
    }
    return new Sample(replayNanos, restoreNanos);
  }

  private static int[] repeatedTokens(int length) {
    int[] tokens = new int[length];
    for (int index = 0; index < length; index++) {
      tokens[index] = TOKEN_PATTERN[index % TOKEN_PATTERN.length];
    }
    return tokens;
  }

  private static long median(long[] measurements) {
    long[] sorted = measurements.clone();
    Arrays.sort(sorted);
    return sorted[sorted.length / 2];
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private static int positiveArgument(String[] args, int index, int fallback, String name) {
    int value = nonNegativeArgument(args, index, fallback, name);
    if (value == 0) {
      throw new IllegalArgumentException(name + " must be greater than zero");
    }
    return value;
  }

  private static int nonNegativeArgument(String[] args, int index, int fallback, String name) {
    int value = args.length > index ? Integer.parseInt(args[index]) : fallback;
    if (value < 0) {
      throw new IllegalArgumentException(name + " must not be negative");
    }
    return value;
  }

  private record Sample(long replayNanos, long restoreNanos) {}
}
