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
import com.integrallis.models.api.ImageInferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.LlavaBackend;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Runs a frozen image-question corpus through the public image backend, retaining every output. */
public final class VisionTaskQualificationCli {
  private static final ObjectMapper JSON = new ObjectMapper();
  private static final String PREFIX =
      "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";

  private VisionTaskQualificationCli() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 4) {
      throw new IllegalArgumentException(
          "Usage: text.gguf mmproj.gguf cases.json output-directory");
    }
    if (!System.getProperty("os.name").equals("Linux")) {
      throw new IllegalStateException("Real-model qualification requires a remote Linux worker");
    }
    Path corpus = Path.of(args[2]).toAbsolutePath();
    Path output = Path.of(args[3]);
    if (Files.exists(output)) throw new IllegalArgumentException("Output directory already exists");
    Files.createDirectories(output);
    long started = System.nanoTime();
    try (var backend = LlavaBackend.load(Path.of(args[0]), Path.of(args[1]))) {
      var configuration = new LinkedHashMap<String, Object>();
      configuration.put("loadMs", elapsed(started));
      configuration.put("host", java.net.InetAddress.getLocalHost().getHostName());
      configuration.put("context", backend.contextCapacity());
      configuration.put("maximumOutputTokens", 32);
      configuration.put("threads", Integer.parseInt(System.getProperty("vectors.gguf.threads")));
      configuration.put("revision", System.getProperty("models.qualification.revision"));
      configuration.put("diagnostics", backend.diagnostics());
      JSON.writerWithDefaultPrettyPrinter()
          .writeValue(output.resolve("configuration.json").toFile(), configuration);
      runCases(backend, corpus, output);
    }
  }

  private static void runCases(ImageInferenceBackend backend, Path corpus, Path output)
      throws Exception {
    var tokenizer = backend.tokenizer();
    for (JsonNode item : JSON.readTree(corpus.toFile()).required("cases")) {
      String id = item.required("id").asText();
      if (!id.matches("[a-zA-Z0-9_-]+")) throw new IllegalArgumentException("Unsafe case ID");
      Path destination = output.resolve(id);
      Files.createDirectory(destination);
      byte[] image =
          Files.readAllBytes(corpus.getParent().resolve(item.required("image").asText()));
      String suffix =
          "\n"
              + item.required("question").asText()
              + "\nAnswer the question using a single word or phrase. ASSISTANT:";
      int[] prefixTokens = prefix(tokenizer);
      int[] suffixTokens = tokenizer.encode(suffix);
      if (suffixTokens.length > 0 && suffixTokens[0] == tokenizer.bosToken()) {
        suffixTokens = Arrays.copyOfRange(suffixTokens, 1, suffixTokens.length);
      }
      if (prefixTokens.length + backend.imageTokenCount() + suffixTokens.length + 32
          > backend.contextCapacity())
        throw new IllegalArgumentException("Case exceeds context: " + id);
      writeTokens(destination.resolve("prefix.i32"), prefixTokens);
      writeTokens(destination.resolve("suffix.i32"), suffixTokens);
      var runs = new ArrayList<Map<String, Object>>();
      int warmups = item.required("warmups").asInt();
      int repetitions = item.required("repetitions").asInt();
      if (warmups < 0 || repetitions < 1) throw new IllegalArgumentException("Invalid run count");
      for (int run = -warmups; run < repetitions; run++) {
        var result = measure(backend, image, prefixTokens, suffixTokens);
        result.put("run", run);
        runs.add(result);
        System.out.printf(
            "case=%s run=%d answer=%s totalMs=%s%n",
            id, run, result.get("answer"), result.get("totalMs"));
      }
      JSON.writerWithDefaultPrettyPrinter()
          .writeValue(destination.resolve("runs.json").toFile(), runs);
    }
  }

  private static Map<String, Object> measure(
      ImageInferenceBackend backend, byte[] image, int[] prefix, int[] suffix) {
    backend.reset();
    long started = System.nanoTime();
    backend.prefill(prefix, 0);
    int position = prefix.length;
    backend.prefillImage(image, position);
    position += backend.imageTokenCount();
    float[] logits = backend.prefill(suffix, position);
    position += suffix.length;
    double prefill = elapsed(started);
    long decodeStarted = System.nanoTime();
    List<Integer> tokens = new ArrayList<>();
    boolean ended = false;
    for (int i = 0; i < 32; i++) {
      int best = VisionQualificationCli.bestToken(logits);
      if (backend.tokenizer().isEndOfGeneration(best)) {
        ended = true;
        break;
      }
      tokens.add(best);
      if (i < 31) logits = backend.forward(best, position++);
    }
    double decode = elapsed(decodeStarted);
    var result = new LinkedHashMap<String, Object>();
    result.put("tokens", tokens);
    result.put(
        "answer",
        backend.tokenizer().decode(tokens.stream().mapToInt(Integer::intValue).toArray()));
    result.put("prefillMs", prefill);
    result.put("decodeMs", decode);
    result.put("totalMs", prefill + decode);
    result.put("truncated", !ended);
    return result;
  }

  private static int[] prefix(Tokenizer tokenizer) {
    int[] tokens = tokenizer.encode(PREFIX);
    if (tokens.length > 0 && tokens[0] == tokenizer.bosToken()) return tokens;
    int[] withBos = new int[tokens.length + 1];
    withBos[0] = tokenizer.bosToken();
    System.arraycopy(tokens, 0, withBos, 1, tokens.length);
    return withBos;
  }

  private static void writeTokens(Path path, int[] tokens) throws Exception {
    ByteBuffer bytes =
        ByteBuffer.allocate(tokens.length * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int token : tokens) bytes.putInt(token);
    Files.write(path, bytes.array());
  }

  private static double elapsed(long started) {
    return (System.nanoTime() - started) / 1_000_000.0;
  }
}
