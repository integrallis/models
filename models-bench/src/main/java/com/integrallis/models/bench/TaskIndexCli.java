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

import com.integrallis.models.api.Pooling;
import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.router.PretrainedTaskClassifier;
import com.integrallis.models.router.TaskExemplars;
import com.integrallis.models.router.TaskIndex;
import com.integrallis.models.router.TaskIndexBuilder;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

/**
 * Builds the router's task index from a corpus, and measures it against the held-out split.
 *
 * <p>Both halves live in one command because they must use the same embedding model: an index and
 * an accuracy figure produced by different models say nothing about each other.
 *
 * <pre>
 * build     --model M.gguf --model-id ID --corpus C.tsv --out DIR [--pooling last]
 * evaluate  --model M.gguf --corpus C.tsv --index DIR [--threshold T] [--min-accuracy A]
 * </pre>
 *
 * <p>{@code evaluate} exits non-zero when accuracy falls below {@code --min-accuracy}, which is how
 * CI gates a corpus or model change.
 */
public final class TaskIndexCli {

  private TaskIndexCli() {}

  private static TaskExemplars corpus(Path path) throws IOException {
    try (Reader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
      return TaskExemplars.parse(reader);
    }
  }

  private static GgufEmbeddingBackend embedding(Path model, String pooling) {
    return GgufEmbeddingBackend.builder(PureJavaBackend.load(model))
        .pooling(Pooling.valueOf(pooling.toUpperCase(Locale.ROOT)))
        .normalize(true)
        .build();
  }

  private static int build(Map<String, String> options) throws IOException {
    Path model = Path.of(required(options, "model"));
    Path out = Path.of(required(options, "out"));
    String modelId = required(options, "model-id");
    TaskExemplars exemplars = corpus(Path.of(required(options, "corpus")));

    try (GgufEmbeddingBackend backend =
        embedding(model, options.getOrDefault("pooling", "last_token"))) {
      long started = System.nanoTime();
      int indexed = TaskIndexBuilder.build(exemplars, backend::embed, modelId, out);
      long millis = (System.nanoTime() - started) / 1_000_000;
      System.out.printf(
          "indexed %d prompts over %d tasks from %s in %d ms -> %s%n",
          indexed, exemplars.taskNames().size(), modelId, millis, out);
    }
    return 0;
  }

  private static int evaluate(Map<String, String> options) throws IOException {
    Path model = Path.of(required(options, "model"));
    Path indexDirectory = Path.of(required(options, "index"));
    TaskExemplars exemplars = corpus(Path.of(required(options, "corpus")));
    double threshold = Double.parseDouble(options.getOrDefault("threshold", "0.0"));
    double floor = Double.parseDouble(options.getOrDefault("min-accuracy", "0.0"));

    List<TaskExemplars.Labelled> held = exemplars.evaluationPrompts();
    if (held.isEmpty()) {
      System.err.println("corpus has no held-out prompts to evaluate against");
      return 2;
    }

    try (GgufEmbeddingBackend backend =
            embedding(model, options.getOrDefault("pooling", "last_token"));
        TaskIndex index = TaskIndex.open(indexDirectory)) {
      PretrainedTaskClassifier classifier =
          PretrainedTaskClassifier.using(index, backend::embed, threshold);

      Map<String, int[]> perTask = new TreeMap<>();
      List<String> misses = new ArrayList<>();
      int correct = 0;
      int unclassified = 0;
      for (TaskExemplars.Labelled labelled : held) {
        String predicted = classifier.classify(labelled.prompt());
        boolean hit = labelled.task().equals(predicted);
        correct += hit ? 1 : 0;
        unclassified += predicted == null ? 1 : 0;
        int[] counts = perTask.computeIfAbsent(labelled.task(), key -> new int[2]);
        counts[1]++;
        counts[0] += hit ? 1 : 0;
        if (!hit && misses.size() < 20) {
          misses.add(
              String.format(
                  "  want=%-14s got=%-14s %s",
                  labelled.task(),
                  String.valueOf(predicted),
                  labelled.prompt().substring(0, Math.min(90, labelled.prompt().length()))));
        }
      }

      double accuracy = correct / (double) held.size();
      System.out.printf(
          "index=%s model=%s threshold=%.3f%n",
          indexDirectory, index.embeddingModelId(), threshold);
      perTask.forEach(
          (task, counts) ->
              System.out.printf(
                  "  %-14s %3d/%-3d  %.3f%n",
                  task, counts[0], counts[1], counts[0] / (double) counts[1]));
      System.out.printf(
          "accuracy %.4f (%d/%d), unclassified %d%n", accuracy, correct, held.size(), unclassified);
      if (!misses.isEmpty()) {
        System.out.println("first misses:");
        misses.forEach(System.out::println);
      }

      if (accuracy < floor) {
        System.err.printf("FAIL accuracy %.4f is below the floor of %.4f%n", accuracy, floor);
        return 1;
      }
      return 0;
    }
  }

  private static String required(Map<String, String> options, String name) {
    String value = options.get(name);
    if (value == null) {
      throw new IllegalArgumentException("missing --" + name);
    }
    return value;
  }

  private static Map<String, String> parse(String[] args, int from) {
    Map<String, String> options = new TreeMap<>();
    for (int index = from; index < args.length; index++) {
      String argument = args[index];
      if (!argument.startsWith("--")) {
        throw new IllegalArgumentException("unexpected argument: " + argument);
      }
      if (index + 1 >= args.length) {
        throw new IllegalArgumentException("missing value for " + argument);
      }
      options.put(argument.substring(2), args[++index]);
    }
    return options;
  }

  /**
   * Runs one subcommand.
   *
   * @param args {@code build} or {@code evaluate} followed by options
   * @return process exit status; non-zero when a gate fails or usage is wrong
   * @throws IOException if the corpus or index cannot be read
   */
  public static int run(String[] args) throws IOException {
    if (args.length == 0) {
      System.err.println(
          """
          usage:
            build     --model M.gguf --model-id ID --corpus C.tsv --out DIR [--pooling last_token]
            evaluate  --model M.gguf --corpus C.tsv --index DIR [--threshold T] [--min-accuracy A]
          """);
      return 2;
    }
    int status;
    try {
      status =
          switch (args[0]) {
            case "build" -> build(parse(args, 1));
            case "evaluate" -> evaluate(parse(args, 1));
            default -> {
              System.err.println("unknown command: " + args[0]);
              yield 2;
            }
          };
    } catch (IllegalArgumentException e) {
      System.err.println(e.getMessage());
      status = 2;
    }
    return status;
  }
}
