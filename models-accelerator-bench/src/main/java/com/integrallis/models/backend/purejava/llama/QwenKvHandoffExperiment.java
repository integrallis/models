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
package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/** Real-model control showing whether raw same-geometry Qwen KV can replace native target KV. */
public final class QwenKvHandoffExperiment {
  private static final String DEFAULT_PROMPT =
      "A Java assistant is comparing two local language models. Explain why a cache produced by "
          + "one independently trained model must be validated before another model consumes it.";

  private QwenKvHandoffExperiment() {}

  public static void main(String[] args) throws Exception {
    if (args.length < 2 || args.length > 3) {
      throw new IllegalArgumentException(
          "usage: QwenKvHandoffExperiment SOURCE.gguf TARGET.gguf [PROMPT]");
    }
    String prompt = args.length == 3 ? args[2] : DEFAULT_PROMPT;
    try (Arena sourceArena = Arena.ofShared();
        Arena targetArena = Arena.ofShared()) {
      LoadedModel source = load(Path.of(args[0]), sourceArena);
      LoadedModel target = load(Path.of(args[1]), targetArena);
      requireMatchedLayout(source.config(), target.config());
      int[] sourceTokens = source.tokenizer().encode(prompt);
      int[] targetTokens = target.tokenizer().encode(prompt);
      if (!Arrays.equals(sourceTokens, targetTokens)) {
        throw new IllegalArgumentException("source and target token IDs differ");
      }
      if (sourceTokens.length < 2) {
        throw new IllegalArgumentException("prompt must contain at least two tokens");
      }

      Capture sourceCapture = capture(source, sourceTokens);
      Capture nativeTarget = capture(target, targetTokens);
      int prefixTokens = targetTokens.length - 1;
      KvCache rawCache = cache(target.config(), targetTokens.length + 8);
      copyRawPrefix(sourceCapture.cache(), rawCache, prefixTokens);
      LlamaForwardPass rawGraph = new LlamaForwardPass(target.config(), target.weights(), rawCache);
      rawGraph.resumeFromImportedPrefix(prefixTokens);
      long rawStarted = System.nanoTime();
      float[] rawLogits = rawGraph.forward(targetTokens[prefixTokens], prefixTokens);
      double rawTailMillis = elapsedMillis(rawStarted);

      float[] nativeLogits = nativeTarget.logits();
      int nativeTop = argmax(nativeLogits);
      int rawTop = argmax(rawLogits);
      int agreements = nativeTop == rawTop ? 1 : 0;
      int steps = 8;
      int teacherToken = nativeTop;
      LlamaForwardPass nativeGraph = nativeTarget.graph();
      for (int step = 1; step < steps; step++) {
        int position = prefixTokens + step;
        nativeLogits = nativeGraph.forward(teacherToken, position);
        rawLogits = rawGraph.forward(teacherToken, position);
        int nextNative = argmax(nativeLogits);
        int nextRaw = argmax(rawLogits);
        if (nextNative == nextRaw) {
          agreements++;
        }
        teacherToken = nextNative;
      }

      System.out.printf(
          Locale.ROOT,
          "tokens=%d sourcePrefillMillis=%.3f targetPrefillMillis=%.3f rawTailMillis=%.3f%n",
          targetTokens.length,
          sourceCapture.prefillMillis(),
          nativeTarget.prefillMillis(),
          rawTailMillis);
      System.out.printf(
          Locale.ROOT,
          "firstNativeToken=%d firstRawToken=%d firstTop1Equal=%s teacherForcedTop1=%d/%d "
              + "finalLogitCosine=%.8f finalKl=%.8f%n",
          nativeTop,
          rawTop,
          nativeTop == rawTop,
          agreements,
          steps,
          cosine(nativeLogits, rawLogits),
          kl(nativeLogits, rawLogits));
    }
  }

  static void copyRawPrefix(KvCache source, KvCache target, int prefixTokens) {
    if (source.numLayers() != target.numLayers()
        || source.keyDim() != target.keyDim()
        || source.valueDim() != target.valueDim()) {
      throw new IllegalArgumentException("source and target KV cache geometry differs");
    }
    if (prefixTokens < 0 || prefixTokens > source.maxSeqLen() || prefixTokens > target.maxSeqLen()) {
      throw new IllegalArgumentException("prefix token count is out of range: " + prefixTokens);
    }
    for (int layer = 0; layer < source.numLayers(); layer++) {
      for (int position = 0; position < prefixTokens; position++) {
        float[] key = source.key(layer, position);
        float[] value = source.value(layer, position);
        if (key == null || value == null) {
          throw new IllegalArgumentException(
              "source cache is incomplete at layer " + layer + ", position " + position);
        }
        target.store(layer, position, key, value);
      }
    }
  }

  private static Capture capture(LoadedModel model, int[] tokens) {
    KvCache cache = cache(model.config(), tokens.length + 8);
    LlamaForwardPass graph = new LlamaForwardPass(model.config(), model.weights(), cache);
    long started = System.nanoTime();
    float[] logits = graph.prefill(tokens, 0).clone();
    return new Capture(cache, graph, logits, elapsedMillis(started));
  }

  private static KvCache cache(LlamaConfig config, int tokens) {
    return new KvCache(config.numLayers(), tokens, config.keyDim(), config.valueDim());
  }

  private static LoadedModel load(Path path, Arena arena) throws Exception {
    GgufFile file = GgufParser.parse(path, arena);
    LlamaConfig config = LlamaConfig.fromMetadata(file.metadata());
    return new LoadedModel(
        config,
        LlamaWeights.fromGgufFile(file, config),
        GgufTokenizer.fromMetadata(file.metadata()));
  }

  private static void requireMatchedLayout(LlamaConfig source, LlamaConfig target) {
    if (source.numLayers() != target.numLayers()
        || source.numKvHeads() != target.numKvHeads()
        || source.keyLength() != target.keyLength()
        || source.valueLength() != target.valueLength()) {
      throw new IllegalArgumentException("source and target do not have matched KV geometry");
    }
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

  private static double cosine(float[] first, float[] second) {
    double dot = 0.0;
    double firstNorm = 0.0;
    double secondNorm = 0.0;
    for (int index = 0; index < first.length; index++) {
      dot += first[index] * second[index];
      firstNorm += first[index] * first[index];
      secondNorm += second[index] * second[index];
    }
    return dot / Math.sqrt(firstNorm * secondNorm);
  }

  private static double kl(float[] expectedLogits, float[] actualLogits) {
    double expectedMax = max(expectedLogits);
    double actualMax = max(actualLogits);
    double expectedNormalizer = normalizer(expectedLogits, expectedMax);
    double actualNormalizer = normalizer(actualLogits, actualMax);
    double result = 0.0;
    for (int index = 0; index < expectedLogits.length; index++) {
      double expectedLogProbability =
          expectedLogits[index] - expectedMax - Math.log(expectedNormalizer);
      double actualLogProbability = actualLogits[index] - actualMax - Math.log(actualNormalizer);
      double probability = Math.exp(expectedLogProbability);
      result += probability * (expectedLogProbability - actualLogProbability);
    }
    return result;
  }

  private static double max(float[] values) {
    double result = Double.NEGATIVE_INFINITY;
    for (float value : values) {
      result = Math.max(result, value);
    }
    return result;
  }

  private static double normalizer(float[] values, double maximum) {
    double result = 0.0;
    for (float value : values) {
      result += Math.exp(value - maximum);
    }
    return result;
  }

  private static double elapsedMillis(long started) {
    return (System.nanoTime() - started) / 1_000_000.0;
  }

  private record LoadedModel(LlamaConfig config, LlamaWeights weights, GgufTokenizer tokenizer) {}

  private record Capture(
      KvCache cache, LlamaForwardPass graph, float[] logits, double prefillMillis) {}
}
