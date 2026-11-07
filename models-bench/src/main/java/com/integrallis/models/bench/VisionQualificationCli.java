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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.backend.purejava.LlavaBackend;
import com.integrallis.models.backend.purejava.vision.LlavaVisionProjector;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;

/** Produces real vision-stage vectors and timings, with an optional two-file greedy smoke. */
public final class VisionQualificationCli {
  private VisionQualificationCli() {}

  public static void main(String[] args) throws Exception {
    if (args.length < 5 || args.length > 6) {
      throw new IllegalArgumentException(
          "Usage: VisionQualificationCli mmproj image output-dir warmups repetitions [text-gguf]");
    }
    Path mmproj = Path.of(args[0]);
    Path image = Path.of(args[1]);
    Path output = Path.of(args[2]);
    int warmups = Integer.parseInt(args[3]);
    int repetitions = Integer.parseInt(args[4]);
    if (warmups < 1 || repetitions < 3)
      throw new IllegalArgumentException("Require at least one warmup and three measured runs");
    Files.createDirectories(output);
    byte[] encoded = Files.readAllBytes(image);
    Map<String, Object> report = new LinkedHashMap<>();
    report.put("schema", "vision-integration-v1");
    report.put("arm", "models-java");
    report.put("java", System.getProperty("java.runtime.version"));
    report.put("os", System.getProperty("os.name") + " " + System.getProperty("os.arch"));
    report.put("host", java.net.InetAddress.getLocalHost().getHostName());
    report.put("threads", Integer.parseInt(System.getProperty("vectors.gguf.threads", "1")));
    report.put("mmprojSha256", sha256(mmproj));
    report.put("imageSha256", sha256(image));
    report.put("warmups", warmups);
    report.put("repetitions", repetitions);
    report.put(
        "candidateRevision", System.getProperty("models.qualification.revision", "unrecorded"));
    var options =
        new MeasurementOptions(
            mmproj, output, warmups, repetitions, args.length == 6 ? Path.of(args[5]) : null);
    measureVision(options, encoded, report);
    if (options.textModel() != null) measureDecoder(options, encoded, report);
    new ObjectMapper()
        .writerWithDefaultPrettyPrinter()
        .writeValue(output.resolve("report.json").toFile(), report);
  }

  private record MeasurementOptions(
      Path mmproj, Path output, int warmups, int repetitions, Path textModel) {}

  private static void measureVision(
      MeasurementOptions options, byte[] encoded, Map<String, Object> report) throws Exception {
    Path mmproj = options.mmproj();
    Path output = options.output();
    int warmups = options.warmups();
    int repetitions = options.repetitions();
    long start = System.nanoTime();
    try (var vision = LlavaVisionProjector.load(mmproj)) {
      report.put("loadMs", elapsed(start));
      report.put("visionKernel", vision.implementation());
      start = System.nanoTime();
      float[] pixels = vision.preprocess(encoded);
      report.put("preprocessMs", elapsed(start));
      writeFloats(output.resolve("pixels.f32"), pixels);
      report.put("imageSize", vision.imageSize());
      report.put("tokens", vision.tokenCount());
      report.put("dimension", vision.embeddingDimension());
      for (int i = 0; i < warmups; i++) vision.encodePreprocessed(pixels);
      ArrayList<Double> times = new ArrayList<>();
      float[] embeddings = null;
      for (int i = 0; i < repetitions; i++) {
        start = System.nanoTime();
        embeddings = vision.encodePreprocessed(pixels);
        times.add(elapsed(start));
      }
      report.put("encoderProjectorMs", times);
      writeFloats(output.resolve("embeddings.f32"), embeddings);
    }
  }

  private static void measureDecoder(
      MeasurementOptions options, byte[] encoded, Map<String, Object> report) throws Exception {
    Path mmproj = options.mmproj();
    Path textModel = options.textModel();
    report.put("textSha256", sha256(textModel));
    long start = System.nanoTime();
    try (var backend = LlavaBackend.load(textModel, mmproj)) {
      report.put("twoFileLoadMs", elapsed(start));
      report.put("decoderDiagnostics", backend.diagnostics());
      report.put("context", backend.contextCapacity());
      report.put("kvType", "f32");
      report.put("maximumOutputTokens", 32);
      var prompt = preparePrompt(backend.tokenizer());
      measureDecoderRuns(options, encoded, report, backend, prompt);
    }
  }

  private record PromptTokens(int[] prefix, int[] suffix) {}

  private static PromptTokens preparePrompt(Tokenizer tokenizer) {
    // Explicit LLaVA 1.5 Vicuna template; image is inserted where <image> belongs.
    String prefix =
        "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";
    String suffix = "\nDescribe this image briefly. ASSISTANT:";
    int[] prefixTokens = tokenizer.encode(prefix);
    if (prefixTokens.length == 0 || prefixTokens[0] != tokenizer.bosToken()) {
      int[] withBos = new int[prefixTokens.length + 1];
      withBos[0] = tokenizer.bosToken();
      System.arraycopy(prefixTokens, 0, withBos, 1, prefixTokens.length);
      prefixTokens = withBos;
    }
    int[] suffixTokens = tokenizer.encode(suffix);
    // Encoding a fragment must not add another BOS inside the sequence.
    if (suffixTokens.length > 0 && suffixTokens[0] == tokenizer.bosToken())
      suffixTokens = java.util.Arrays.copyOfRange(suffixTokens, 1, suffixTokens.length);
    return new PromptTokens(prefixTokens, suffixTokens);
  }

  private static void measureDecoderRuns(
      MeasurementOptions options,
      byte[] encoded,
      Map<String, Object> report,
      LlavaBackend backend,
      PromptTokens prompt)
      throws Exception {
    int[] prefixTokens = prompt.prefix();
    int[] suffixTokens = prompt.suffix();
    var tokenizer = backend.tokenizer();
    int warmups = options.warmups();
    int repetitions = options.repetitions();
    Path output = options.output();
    long start;
    ArrayList<Double> prefillTimes = new ArrayList<>();
    ArrayList<Double> decodeTimes = new ArrayList<>();
    ArrayList<java.util.List<Integer>> generatedRuns = new ArrayList<>();
    for (int run = -warmups; run < repetitions; run++) {
      backend.reset();
      start = System.nanoTime();
      backend.prefill(prefixTokens, 0);
      int position = prefixTokens.length;
      backend.prefillImage(encoded, position);
      position += backend.imageTokenCount();
      float[] logits = backend.prefill(suffixTokens, position);
      position += suffixTokens.length;
      double prefillMs = elapsed(start);
      if (run == 0) writeFloats(output.resolve("first-logits.f32"), logits);
      ArrayList<Integer> generated = new ArrayList<>();
      start = System.nanoTime();
      decodeGreedy(backend, logits, position, generated);
      double decodeMs = elapsed(start);
      System.out.printf(
          "decoder run=%d prefillMs=%.3f decodeMs=%.3f emittedTokens=%d%n",
          run, prefillMs, decodeMs, generated.size());
      if (run >= 0) {
        prefillTimes.add(prefillMs);
        decodeTimes.add(decodeMs);
        generatedRuns.add(generated);
      }
      report.put("generatedTokens", generated);
      report.put(
          "answer", tokenizer.decode(generated.stream().mapToInt(Integer::intValue).toArray()));
    }
    report.put("imageTextPrefillMs", prefillTimes);
    report.put("decodeMs", decodeTimes);
    report.put("generatedTokenRuns", generatedRuns);
    report.put("prefixTokens", prefixTokens);
    report.put("suffixTokens", suffixTokens);
  }

  private static void decodeGreedy(
      LlavaBackend backend, float[] logits, int position, ArrayList<Integer> generated) {
    var tokenizer = backend.tokenizer();
    for (int i = 0; i < 32; i++) {
      int best = bestToken(logits);
      if (tokenizer.isEndOfGeneration(best)) break;
      generated.add(best);
      if (i < 31) logits = backend.forward(best, position++);
    }
  }

  static int bestToken(float[] logits) {
    int best = 0;
    for (int j = 0; j < logits.length; j++) {
      if (!Float.isFinite(logits[j])) throw new IllegalStateException("Nonfinite logits");
      if (logits[j] > logits[best]) best = j;
    }
    return best;
  }

  private static double elapsed(long start) {
    return (System.nanoTime() - start) / 1_000_000.0;
  }

  private static void writeFloats(Path path, float[] values) throws Exception {
    ByteBuffer buffer =
        ByteBuffer.allocate(Math.multiplyExact(values.length, Float.BYTES))
            .order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite qualification vector");
      buffer.putFloat(value);
    }
    Files.write(path, buffer.array());
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (var input = Files.newInputStream(path)) {
      byte[] buffer = new byte[1024 * 1024];
      int count;
      while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
