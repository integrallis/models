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
package com.integrallis.models.bench;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordedFrame;
import jdk.jfr.consumer.RecordedStackTrace;
import jdk.jfr.consumer.RecordingFile;

/**
 * Where a recording's samples actually landed, counting Java and native frames together.
 *
 * <p>Built because counting one of them alone is actively misleading on this stack. {@code
 * jdk.ExecutionSample} excludes a thread that is in native state, and on the native-kernel backend
 * that is precisely where the matmul time goes -- so a profile read from execution samples alone
 * shows the Rust kernels costing nothing and points at whatever Java frame happened to be third.
 * {@code jdk.NativeMethodSample} covers those threads, and the two have to be added up before any
 * frame can be called a hotspot.
 *
 * <p>Reports self time by leaf frame and total time by frame appearing anywhere in a stack, since
 * the two answer different questions: a leaf is where the cycles went, and a total is what to stop
 * calling.
 *
 * <pre>{@code
 * ... TriageDecisionBenchCli --model m.gguf ... # any run that writes a .jfr
 * ... ProfileHotspotCli --recording run.jfr --top 25
 * }</pre>
 */
final class ProfileHotspotCli {

  private static final Set<String> OPTIONS = Set.of("recording", "top", "package");

  /** The sampling events that together cover a thread in any state. */
  private static final Set<String> SAMPLE_EVENTS =
      Set.of("jdk.ExecutionSample", "jdk.NativeMethodSample");

  private ProfileHotspotCli() {}

  static void run(String[] args) throws Exception {
    Map<String, String> values = BenchmarkCliArguments.parse(args, OPTIONS);
    String recording = values.get("recording");
    if (recording == null || recording.isBlank()) {
      throw new IllegalArgumentException("--recording <file.jfr> is required");
    }
    Path file = Path.of(recording);
    if (!Files.isRegularFile(file)) {
      throw new IllegalArgumentException("not a file: " + file);
    }
    int top = BenchmarkCliArguments.integer(values, "top", 25);
    String prefix = values.get("package");

    Map<String, long[]> perEvent = new TreeMap<>();
    Map<String, Long> self = new LinkedHashMap<>();
    Map<String, Long> total = new LinkedHashMap<>();
    long samples = 0;

    try (RecordingFile events = new RecordingFile(file)) {
      while (events.hasMoreEvents()) {
        RecordedEvent event = events.readEvent();
        String kind = event.getEventType().getName();
        if (!SAMPLE_EVENTS.contains(kind)) {
          continue;
        }
        perEvent.computeIfAbsent(kind, key -> new long[1])[0]++;
        samples++;
        RecordedStackTrace stack = event.getStackTrace();
        if (stack == null) {
          continue;
        }
        List<RecordedFrame> frames = stack.getFrames();
        if (frames.isEmpty()) {
          continue;
        }
        String leaf = describe(frames.getFirst());
        if (matches(leaf, prefix)) {
          self.merge(leaf, 1L, Long::sum);
        }
        // One count per frame per stack, not per occurrence, so a recursive frame is not inflated.
        List<String> seen = new ArrayList<>();
        for (RecordedFrame frame : frames) {
          String name = describe(frame);
          if (matches(name, prefix) && !seen.contains(name)) {
            seen.add(name);
            total.merge(name, 1L, Long::sum);
          }
        }
      }
    }

    if (samples == 0) {
      System.out.println(
          "no "
              + String.join(" or ", SAMPLE_EVENTS)
              + " events in "
              + file
              + " -- was the recording made with a sampling configuration?");
      return;
    }

    System.out.printf("recording: %s%n", file);
    System.out.printf("samples:   %d%n", samples);
    long totalSamples = samples;
    perEvent.forEach(
        (kind, count) ->
            System.out.printf(
                "  %-26s %7d  (%5.1f%%)%n", kind, count[0], 100.0 * count[0] / totalSamples));
    // Named explicitly: a reader who sees only execution samples here should know the native ones
    // were looked for and were absent, rather than assume they were counted.
    if (!perEvent.containsKey("jdk.NativeMethodSample")) {
      System.out.println(
          "  note: no native-method samples present, so native-state time is NOT represented");
    }
    print("self time (leaf frame)", self, samples, top);
    print("total time (frame anywhere in stack)", total, samples, top);
  }

  private static void print(String title, Map<String, Long> counts, long samples, int top) {
    System.out.printf("%n== %s ==%n", title);
    counts.entrySet().stream()
        .sorted(
            Map.Entry.<String, Long>comparingByValue().reversed().thenComparing(Map.Entry::getKey))
        .limit(top)
        .forEach(
            entry ->
                System.out.printf(
                    Locale.ROOT,
                    "%6.2f%%  %7d  %s%n",
                    100.0 * entry.getValue() / samples,
                    entry.getValue(),
                    entry.getKey()));
  }

  private static boolean matches(String frame, String prefix) {
    return prefix == null || prefix.isBlank() || frame.startsWith(prefix);
  }

  private static String describe(RecordedFrame frame) {
    if (frame.getMethod() == null) {
      return "<unknown>";
    }
    String type = frame.getMethod().getType() == null ? "?" : frame.getMethod().getType().getName();
    return type + '.' + frame.getMethod().getName();
  }

  /** Orders frames by descending count, for callers that want the raw ranking. */
  static Comparator<Map.Entry<String, Long>> byDescendingCount() {
    return Map.Entry.<String, Long>comparingByValue().reversed();
  }
}
