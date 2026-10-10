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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.backend.purejava.GgufEmbeddingBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.router.PretrainedTaskClassifier;
import com.integrallis.models.router.TaskIndexResource;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReference;

/** Collects paired, deterministic observations using only local Models artifacts. */
final class RouterObservationCollectorCli {
  private static final ObjectMapper JSON =
      new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
  private static final String WARMUP = "Answer with one word only: yes.";
  private static final String SYSTEM =
      "You are answering benchmark questions. Follow the requested answer format exactly. "
          + "Return only the final answer, without explanation.";

  private RouterObservationCollectorCli() {}

  static void run(String[] args) throws Exception {
    Path casesPath = null, outputPath = null, classifiedPath = null, classifierPath = null;
    List<ModelSpec> specs = new ArrayList<>();
    int maxTokens = 64;
    long seed = 20260926L;
    for (int i = 0; i < args.length; i++) {
      switch (args[i]) {
        case "--cases" -> casesPath = Path.of(value(args, ++i, "--cases")).toAbsolutePath();
        case "--output" -> outputPath = Path.of(value(args, ++i, "--output")).toAbsolutePath();
        case "--classifier" ->
            classifierPath = Path.of(value(args, ++i, "--classifier")).toAbsolutePath();
        case "--classified-cases" ->
            classifiedPath = Path.of(value(args, ++i, "--classified-cases")).toAbsolutePath();
        case "--model" -> {
          String[] pair = value(args, ++i, "--model").split("=", 2);
          if (pair.length != 2 || pair[0].isBlank()) {
            throw new IllegalArgumentException("--model must be id=/absolute/path/model.gguf");
          }
          specs.add(new ModelSpec(pair[0], Path.of(pair[1]).toAbsolutePath()));
        }
        case "--max-tokens" -> maxTokens = Integer.parseInt(value(args, ++i, "--max-tokens"));
        case "--seed" -> seed = Long.parseLong(value(args, ++i, "--seed"));
        default -> throw new IllegalArgumentException("unknown option: " + args[i]);
      }
    }
    if (casesPath == null
        || outputPath == null
        || classifierPath == null
        || classifiedPath == null
        || specs.size() < 2
        || maxTokens < 1) {
      throw new IllegalArgumentException(
          "router-collect --cases CASES.json --classified-cases CLASSIFIED.json "
              + "--output OBSERVATIONS.json --classifier MODEL.gguf --model id=path "
              + "[--model id=path ...] [--max-tokens N] [--seed N]");
    }
    Path partialPath = outputPath.resolveSibling(outputPath.getFileName() + ".partial");
    if (Files.exists(outputPath) || Files.exists(classifiedPath) || Files.exists(partialPath)) {
      throw new IllegalArgumentException("refusing to overwrite collector evidence");
    }
    JsonNode cases = JSON.readTree(casesPath.toFile());
    if (!cases.isArray() || cases.isEmpty()) {
      throw new IllegalArgumentException("cases must be a non-empty JSON array");
    }
    List<Pair> schedule = new ArrayList<>();
    for (JsonNode item : cases) {
      String id = required(item, "id");
      String prompt = required(item, "prompt");
      for (ModelSpec spec : specs) schedule.add(new Pair(id, prompt, spec));
    }
    Collections.shuffle(schedule, new Random(seed));

    Map<String, LoadedModel> loaded = new LinkedHashMap<>();
    System.setProperty("models.purejava.maxContextLength", "4096");
    try {
      if (!Files.isRegularFile(classifierPath)) {
        throw new IllegalArgumentException("classifier artifact does not exist: " + classifierPath);
      }
      PureJavaBackend classifierBackend = PureJavaBackend.load(classifierPath);
      GgufEmbeddingBackend embedding =
          GgufEmbeddingBackend.builder(classifierBackend).normalize(true).build();
      String classifierSha = sha256(classifierPath);
      ArrayNode classified = JSON.createArrayNode();
      try (var index = TaskIndexResource.openBundled()) {
        PretrainedTaskClassifier classifier =
            PretrainedTaskClassifier.using(index, embedding::embed, 0.0);
        for (JsonNode item : cases) {
          ObjectNode mutable = item.deepCopy();
          long started = System.nanoTime();
          String prediction = classifier.classify(required(item, "prompt"));
          double elapsedMillis = (System.nanoTime() - started) / 1_000_000.0;
          mutable.put("predictedTask", prediction);
          mutable.put("classifierMillis", elapsedMillis);
          classified.add(mutable);
        }
        JSON.writeValue(classifiedPath.toFile(), classified);
      }
      embedding.close();
      classifierBackend.close();
      for (ModelSpec spec : specs) {
        if (!Files.isRegularFile(spec.path())) {
          throw new IllegalArgumentException("model artifact does not exist: " + spec.path());
        }
        PureJavaBackend backend = PureJavaBackend.load(spec.path());
        loaded.put(
            spec.id(), new LoadedModel(backend, new GenerationLoop(backend), sha256(spec.path())));
      }
      for (LoadedModel model : loaded.values()) {
        generate(model, WARMUP, 4);
      }
      List<Map<String, Object>> observations = new ArrayList<>();
      for (Pair pair : schedule) {
        LoadedModel model = loaded.get(pair.model().id());
        observations.add(observe(pair, model, maxTokens));
        Map<String, Object> latest = observations.get(observations.size() - 1);
        System.out.printf(
            "%s %s success=%s ttftMs=%s latencyMs=%s%n",
            pair.caseId(),
            pair.model().id(),
            latest.get("success"),
            latest.get("ttftMillis"),
            latest.get("latencyMillis"));
        JSON.writeValue(partialPath.toFile(), observations);
      }
      Files.move(partialPath, outputPath);
      System.out.printf("wrote %d paired observations to %s%n", observations.size(), outputPath);
      System.out.printf(
          "classifier sha256=%s; classified cases=%s%n", classifierSha, classifiedPath);
    } finally {
      for (LoadedModel model : loaded.values()) model.backend().close();
    }
  }

  private static Map<String, Object> observe(Pair pair, LoadedModel model, int maxTokens) {
    String output = "";
    long start = System.nanoTime();
    long[] first = {0};
    int[] count = {0};
    AtomicReference<Throwable> failure = new AtomicReference<>();
    var prompt =
        ChatTemplate.CHATML_NO_THINK.render(
            List.of(ChatMessage.system(SYSTEM), ChatMessage.user(pair.prompt())));
    int inputTokens = model.backend().tokenizer().encode(prompt).length;
    SamplingOptions options =
        SamplingOptions.builder()
            .temperature(0)
            .topP(1)
            .topK(1)
            .seed(42)
            .repetitionPenalty(1)
            .maxTokens(maxTokens)
            .build();
    StringBuilder response = new StringBuilder();
    try {
      model
          .loop()
          .generate(
              prompt,
              options,
              new TokenStream() {
                @Override
                public void onToken(String token) {
                  if (first[0] == 0) first[0] = System.nanoTime();
                  count[0]++;
                  response.append(token);
                }

                @Override
                public void onComplete() {}

                @Override
                public void onError(Throwable error) {
                  failure.set(error);
                }
              });
    } catch (Throwable error) {
      failure.compareAndSet(null, error);
    }
    long end = System.nanoTime();
    output = response.toString();
    boolean success = failure.get() == null && count[0] > 0;
    Map<String, Object> item = new LinkedHashMap<>();
    item.put("caseId", pair.caseId());
    item.put("modelId", pair.model().id());
    item.put("success", success);
    item.put("response", output);
    item.put("latencyMillis", (end - start) / 1_000_000.0);
    item.put("ttftMillis", first[0] == 0 ? 0 : (first[0] - start) / 1_000_000.0);
    item.put("inputTokens", inputTokens);
    item.put("outputTokens", count[0]);
    item.put(
        "receipt", "sha256:" + sha256(output.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
    if (failure.get() != null) item.put("error", failure.get().toString());
    return item;
  }

  private static void generate(LoadedModel model, String prompt, int maxTokens) {
    SamplingOptions options = SamplingOptions.builder().temperature(0).maxTokens(maxTokens).build();
    var rendered =
        ChatTemplate.CHATML_NO_THINK.render(
            List.of(ChatMessage.system(SYSTEM), ChatMessage.user(prompt)));
    model
        .loop()
        .generate(
            rendered,
            options,
            new TokenStream() {
              @Override
              public void onToken(String token) {}

              @Override
              public void onComplete() {}

              @Override
              public void onError(Throwable failure) {
                throw new IllegalStateException(failure);
              }
            });
  }

  private static String required(JsonNode node, String field) {
    String value = node.path(field).asText();
    if (value.isBlank()) throw new IllegalArgumentException("missing " + field);
    return value;
  }

  private static String value(String[] args, int index, String option) {
    if (index >= args.length || args[index].startsWith("--")) {
      throw new IllegalArgumentException(option + " requires a value");
    }
    return args[index];
  }

  private static String sha256(Path path) throws Exception {
    return sha256(Files.readAllBytes(path));
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (java.security.NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }

  private record ModelSpec(String id, Path path) {}

  private record Pair(String caseId, String prompt, ModelSpec model) {}

  private record LoadedModel(PureJavaBackend backend, GenerationLoop loop, String sha256) {}
}
