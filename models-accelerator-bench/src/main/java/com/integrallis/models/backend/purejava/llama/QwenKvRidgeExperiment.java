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

import com.integrallis.models.accelerator.PerHeadRidgeMapper;
import com.integrallis.models.accelerator.RidgeMapper;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufParser;
import com.integrallis.models.backend.purejava.tokenizer.GgufTokenizer;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/** Preliminary same-token, single-source-layer ridge screen for two Qwen3 KV representations. */
public final class QwenKvRidgeExperiment {
  private static final int CALIBRATION_TOKENS = 256;
  private static final int HOLDOUT_TOKENS = 80;
  private static final int[] TARGET_LAYERS = {0, 6, 13, 20, 27};
  private static final double REGULARIZATION = 1.0e-2;

  private static final String CALIBRATION_TEXT =
      "A public transit assistant answers questions from verified schedules and station records. "
          + "When a traveler asks for the next train, the assistant selects the schedule tool, "
          + "passes the station and direction as structured arguments, and explains the returned "
          + "time in one short sentence. If the request is conversational, it answers directly. "
          + "Each route change remains visible to the user, and every claim names its source. "
          + "The application keeps conversation messages in a semantic form so that different "
          + "local specialists can render the same history with their own chat templates. "
          + "A switch must preserve correctness before it is allowed to reduce latency. "
          + "A code assistant receives a repository, a failing test, and a narrowly scoped change. "
          + "It reads the existing implementation before editing, writes a regression test that "
          + "fails for the reported defect, and makes the smallest production change that passes. "
          + "The build records formatting, static analysis, unit tests, and end to end behavior. "
          + "A benchmark result names the exact artifact, machine, Java runtime, warmup, repetitions, "
          + "and correctness oracle so that a faster wrong answer cannot be promoted. "
          + "A memory assistant stores facts, preferences, and prior decisions with provenance. "
          + "Retrieval combines exact filters, semantic similarity, recency, and reranking, but every "
          + "added stage must prove that it improves held out questions rather than a hand written demo. "
          + "Repeated prompts can reuse an exact model prefix, while paraphrases belong to a semantic "
          + "cache with an explicit similarity policy. These mechanisms solve different problems. "
          + "A scientific assistant separates observations, hypotheses, and conclusions. It may use "
          + "one small model for ordinary prose and another specialist for typed tools, but the runtime "
          + "keeps their physical state attributable. Matching tensor dimensions do not prove matching "
          + "activations. An approximate handoff is admitted only after independent held out evaluation.";

  private static final String HOLDOUT_TEXT =
      "A weather assistant receives a postal code and a set of typed Java tools. It should choose "
          + "the matching function, emit valid arguments, wait for the application result, and then "
          + "turn that result into a concise answer. A general greeting should not invoke a tool. "
          + "The runtime records which physical model served the turn, how many prompt tokens were "
          + "reused, time to first token, and completion throughput. Faster execution does not excuse "
          + "a malformed call, a leaked reasoning trace, or an answer based on the wrong context.";

  private QwenKvRidgeExperiment() {}

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException("usage: QwenKvRidgeExperiment SOURCE.gguf TARGET.gguf");
    }
    try (Arena sourceArena = Arena.ofShared();
        Arena targetArena = Arena.ofShared()) {
      LoadedModel source = load(Path.of(args[0]), sourceArena);
      LoadedModel target = load(Path.of(args[1]), targetArena);
      requireMatchedLayout(source.config(), target.config());

      int[] sourceCalibration =
          limited(source.tokenizer().encode(CALIBRATION_TEXT), CALIBRATION_TOKENS);
      int[] targetCalibration =
          limited(target.tokenizer().encode(CALIBRATION_TEXT), CALIBRATION_TOKENS);
      int[] sourceHoldout = limited(source.tokenizer().encode(HOLDOUT_TEXT), HOLDOUT_TOKENS);
      int[] targetHoldout = limited(target.tokenizer().encode(HOLDOUT_TEXT), HOLDOUT_TOKENS);
      requireSameTokens("calibration", sourceCalibration, targetCalibration);
      requireSameTokens("holdout", sourceHoldout, targetHoldout);

      System.out.printf(
          Locale.ROOT,
          "capturing calibration=%d tokens and holdout=%d tokens%n",
          sourceCalibration.length,
          sourceHoldout.length);
      Capture sourceTraining = capture(source, sourceCalibration, "source calibration");
      Capture targetTraining = capture(target, targetCalibration, "target calibration");
      Capture sourceEvaluation = capture(source, sourceHoldout, "source holdout");
      Capture targetEvaluation = capture(target, targetHoldout, "target holdout");

      System.out.printf(
          Locale.ROOT,
          "prefillMillis sourceCalibration=%.3f targetCalibration=%.3f sourceHoldout=%.3f targetHoldout=%.3f%n",
          sourceTraining.prefillMillis(),
          targetTraining.prefillMillis(),
          sourceEvaluation.prefillMillis(),
          targetEvaluation.prefillMillis());
      System.out.println(
          "targetLayer,kind,sourceLayer,top2SourceLayers,trainRawCosine,holdoutSameLayerCosine,"
              + "holdoutSelectedLayerCosine,pooledMappedCosine,pooledRelativeL2,"
              + "perHeadMappedCosine,perHeadRelativeL2,top2MappedCosine,top2RelativeL2,"
              + "pooledFitMillis,perHeadFitMillis,top2FitMillis");

      float[][][] sourceTrainingKeys = samplesByLayer(sourceTraining, true);
      float[][][] sourceTrainingValues = samplesByLayer(sourceTraining, false);
      float[][][] sourceEvaluationKeys = samplesByLayer(sourceEvaluation, true);
      float[][][] sourceEvaluationValues = samplesByLayer(sourceEvaluation, false);
      for (int targetLayer : TARGET_LAYERS) {
        evaluate(
            targetLayer,
            "key",
            sourceTrainingKeys,
            sourceEvaluationKeys,
            sourceTraining,
            targetTraining,
            targetEvaluation,
            true);
        evaluate(
            targetLayer,
            "value",
            sourceTrainingValues,
            sourceEvaluationValues,
            sourceTraining,
            targetTraining,
            targetEvaluation,
            false);
      }
    }
  }

  private static void evaluate(
      int targetLayer,
      String kind,
      float[][][] sourceTrainingByLayer,
      float[][][] sourceEvaluationByLayer,
      Capture sourceTraining,
      Capture targetTraining,
      Capture targetEvaluation,
      boolean keys) {
    float[][] targetTrainingSamples = samples(targetTraining, targetLayer, keys);
    int[] sourceLayers = bestSourceLayers(sourceTrainingByLayer, targetTrainingSamples, 2);
    int sourceLayer = sourceLayers[0];
    float[][] sourceTrainingSamples = sourceTrainingByLayer[sourceLayer];
    double trainingCosine = cosine(sourceTrainingSamples, targetTrainingSamples);
    long fitStarted = System.nanoTime();
    RidgeMapper mapper =
        RidgeMapper.fit(sourceTrainingSamples, targetTrainingSamples, REGULARIZATION);
    double pooledFitMillis = elapsedMillis(fitStarted);
    long perHeadFitStarted = System.nanoTime();
    PerHeadRidgeMapper perHeadMapper =
        PerHeadRidgeMapper.fit(
            sourceTrainingSamples,
            targetTrainingSamples,
            sourceTraining.config().numKvHeads(),
            REGULARIZATION);
    double perHeadFitMillis = elapsedMillis(perHeadFitStarted);
    float[][] topTwoTraining = concatenateLayers(sourceTrainingByLayer, sourceLayers);
    long topTwoFitStarted = System.nanoTime();
    RidgeMapper topTwoMapper =
        RidgeMapper.fit(topTwoTraining, targetTrainingSamples, REGULARIZATION);
    double topTwoFitMillis = elapsedMillis(topTwoFitStarted);

    float[][] targetHeldOut = samples(targetEvaluation, targetLayer, keys);
    float[][] sameLayerHeldOut = sourceEvaluationByLayer[targetLayer];
    float[][] selectedHeldOut = sourceEvaluationByLayer[sourceLayer];
    float[][] pooledMapped = mapper.predict(selectedHeldOut);
    float[][] perHeadMapped = perHeadMapper.predict(selectedHeldOut);
    float[][] topTwoMapped =
        topTwoMapper.predict(concatenateLayers(sourceEvaluationByLayer, sourceLayers));
    System.out.printf(
        Locale.ROOT,
        "%d,%s,%d,%d+%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.3f,%.3f,%.3f%n",
        targetLayer,
        kind,
        sourceLayer,
        sourceLayers[0],
        sourceLayers[1],
        trainingCosine,
        cosine(sameLayerHeldOut, targetHeldOut),
        cosine(selectedHeldOut, targetHeldOut),
        cosine(pooledMapped, targetHeldOut),
        RidgeMapper.relativeL2(targetHeldOut, pooledMapped),
        cosine(perHeadMapped, targetHeldOut),
        RidgeMapper.relativeL2(targetHeldOut, perHeadMapped),
        cosine(topTwoMapped, targetHeldOut),
        RidgeMapper.relativeL2(targetHeldOut, topTwoMapped),
        pooledFitMillis,
        perHeadFitMillis,
        topTwoFitMillis);
  }

  private static int[] bestSourceLayers(
      float[][][] sourceByLayer, float[][] target, int requested) {
    int[] best = new int[requested];
    double[] bestMagnitude = new double[requested];
    Arrays.fill(best, -1);
    Arrays.fill(bestMagnitude, -1.0);
    for (int layer = 0; layer < sourceByLayer.length; layer++) {
      double magnitude = Math.abs(cosine(sourceByLayer[layer], target));
      for (int rank = 0; rank < requested; rank++) {
        if (magnitude > bestMagnitude[rank]) {
          for (int shift = requested - 1; shift > rank; shift--) {
            best[shift] = best[shift - 1];
            bestMagnitude[shift] = bestMagnitude[shift - 1];
          }
          best[rank] = layer;
          bestMagnitude[rank] = magnitude;
          break;
        }
      }
    }
    return best;
  }

  static float[][] concatenateLayers(float[][][] layers, int[] selectedLayers) {
    if (selectedLayers.length == 0) {
      throw new IllegalArgumentException("selectedLayers must not be empty");
    }
    int samples = layers[selectedLayers[0]].length;
    int layerWidth = layers[selectedLayers[0]][0].length;
    float[][] result = new float[samples][Math.multiplyExact(layerWidth, selectedLayers.length)];
    for (int sample = 0; sample < samples; sample++) {
      for (int selected = 0; selected < selectedLayers.length; selected++) {
        float[] source = layers[selectedLayers[selected]][sample];
        if (source.length != layerWidth) {
          throw new IllegalArgumentException("selected layer widths differ");
        }
        System.arraycopy(source, 0, result[sample], selected * layerWidth, layerWidth);
      }
    }
    return result;
  }

  private static float[][][] samplesByLayer(Capture capture, boolean keys) {
    float[][][] result = new float[capture.config().numLayers()][][];
    for (int layer = 0; layer < result.length; layer++) {
      result[layer] = samples(capture, layer, keys);
    }
    return result;
  }

  private static float[][] samples(Capture capture, int layer, boolean keys) {
    LlamaConfig config = capture.config();
    int dimensions = keys ? config.keyLength() : config.valueLength();
    float[][] result = new float[capture.tokens() * config.numKvHeads()][dimensions];
    for (int position = 0; position < capture.tokens(); position++) {
      float[] vectors =
          keys ? capture.cache().key(layer, position) : capture.cache().value(layer, position);
      for (int head = 0; head < config.numKvHeads(); head++) {
        int sourceOffset = head * dimensions;
        int sample = position * config.numKvHeads() + head;
        System.arraycopy(vectors, sourceOffset, result[sample], 0, dimensions);
        if (keys && config.usesRope(layer)) {
          removeRopeNeox(
              result[sample],
              position,
              dimensions,
              config.ropeTheta(layer),
              config.ropeFrequencyScale(layer));
        }
      }
    }
    return result;
  }

  private static Capture capture(LoadedModel model, int[] tokens, String label) {
    LlamaConfig config = model.config();
    KvCache cache =
        new KvCache(config.numLayers(), tokens.length, config.keyDim(), config.valueDim());
    LlamaForwardPass graph = new LlamaForwardPass(config, model.weights(), cache);
    long started = System.nanoTime();
    graph.prefill(tokens, 0);
    double millis = elapsedMillis(started);
    System.out.printf(Locale.ROOT, "%s complete in %.3f ms%n", label, millis);
    return new Capture(config, cache, tokens.length, millis);
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
        || source.numHeads() != target.numHeads()
        || source.numKvHeads() != target.numKvHeads()
        || source.keyLength() != target.keyLength()
        || source.valueLength() != target.valueLength()) {
      throw new IllegalArgumentException("source and target do not have matched KV geometry");
    }
    if (!source.usesNeoxRope() || !target.usesNeoxRope()) {
      throw new IllegalArgumentException("this experiment currently requires NeoX RoPE");
    }
  }

  private static int[] limited(int[] tokens, int maximum) {
    if (tokens.length < maximum) {
      throw new IllegalArgumentException(
          "experiment text encoded to only " + tokens.length + " tokens; need " + maximum);
    }
    return Arrays.copyOf(tokens, maximum);
  }

  private static void requireSameTokens(String label, int[] source, int[] target) {
    if (!Arrays.equals(source, target)) {
      throw new IllegalArgumentException(label + " token IDs differ between source and target");
    }
  }

  static void removeRopeNeox(
      float[] vector, int position, int headDim, float ropeTheta, float frequencyScale) {
    int half = headDim / 2;
    float scaledPosition = position * frequencyScale;
    for (int pair = 0; pair < half; pair++) {
      double frequency = 1.0 / Math.pow(ropeTheta, (double) (2 * pair) / headDim);
      double angle = scaledPosition * frequency;
      float cosine = (float) Math.cos(angle);
      float sine = (float) Math.sin(angle);
      float rotatedFirst = vector[pair];
      float rotatedSecond = vector[half + pair];
      vector[pair] = rotatedFirst * cosine + rotatedSecond * sine;
      vector[half + pair] = -rotatedFirst * sine + rotatedSecond * cosine;
    }
  }

  static double cosine(float[][] first, float[][] second) {
    if (first.length != second.length) {
      throw new IllegalArgumentException("sample counts differ");
    }
    double dot = 0.0;
    double firstNorm = 0.0;
    double secondNorm = 0.0;
    for (int sample = 0; sample < first.length; sample++) {
      if (first[sample].length != second[sample].length) {
        throw new IllegalArgumentException("sample widths differ at " + sample);
      }
      for (int dimension = 0; dimension < first[sample].length; dimension++) {
        double left = first[sample][dimension];
        double right = second[sample][dimension];
        dot += left * right;
        firstNorm += left * left;
        secondNorm += right * right;
      }
    }
    return dot / Math.sqrt(firstNorm * secondNorm);
  }

  private static double elapsedMillis(long started) {
    return (System.nanoTime() - started) / 1_000_000.0;
  }

  private record LoadedModel(LlamaConfig config, LlamaWeights weights, GgufTokenizer tokenizer) {}

  private record Capture(LlamaConfig config, KvCache cache, int tokens, double prefillMillis) {}
}
