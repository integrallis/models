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
package com.integrallis.models.bench.fusion;

import com.integrallis.models.runtime.Sampler;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

/**
 * The fusion loop and the single-member controls it is compared against.
 *
 * <p>Fusion: prefill every member, then per step {@code ℓᵢ = log_softmax(logitsᵢ)}, {@code s =
 * combine(rule, ℓ, w)}, pick the token (argmax of {@code s} at T = 0, otherwise the public {@link
 * Sampler} over {@code s}, which applies {@code softmax(s / T)}), append it and {@code forward} it
 * on every member. Members step on the supplied executor, one task per member per step.
 */
public final class FusionDecoder {

  private FusionDecoder() {}

  private record StepOutput(float[] logits, double[] logProbabilities, long nanos) {}

  /** Receives each step's raw member logits and the fused scores the decoder actually used. */
  @FunctionalInterface
  public interface StepObserver {
    void onStep(int step, float[][] memberLogits, double[] fusedScores);
  }

  /** Fused decode over {@code members} with simplex {@code weights}. */
  public static DecodeResult fuse(
      List<FusionMember> members,
      int[] prompt,
      double[] weights,
      FusionRule rule,
      DecodeSettings settings,
      ExecutorService executor) {
    return fuse(members, prompt, weights, rule, settings, executor, null);
  }

  /** Fused decode with an optional per-step observer (used by the G2 logit dump). */
  public static DecodeResult fuse(
      List<FusionMember> members,
      int[] prompt,
      double[] weights,
      FusionRule rule,
      DecodeSettings settings,
      ExecutorService executor,
      StepObserver observer) {
    FusionMath.validateWeights(weights);
    if (weights.length != members.size()) {
      throw new IllegalArgumentException("one weight per member is required");
    }
    int count = members.size();
    List<String> names = members.stream().map(FusionMember::name).toList();
    long[] memberNanos = new long[count];
    long started = System.nanoTime();

    List<Callable<StepOutput>> prefills = new ArrayList<>();
    for (FusionMember member : members) {
      prefills.add(
          () -> {
            long t0 = System.nanoTime();
            member.reset();
            float[] logits = member.prefill(prompt, 0);
            return new StepOutput(logits, FusionMath.logSoftmax(logits), System.nanoTime() - t0);
          });
    }
    StepOutput[] outputs = runAll(executor, prefills);
    long prefillNanos = System.nanoTime() - started;
    accumulate(memberNanos, outputs);

    int vocabulary = outputs[0].logits().length;
    Sampler sampler = settings.greedy() ? null : new Sampler(settings.samplingOptions(vocabulary));
    List<Integer> generated = new ArrayList<>();
    List<Double> tokenLogProbabilities = new ArrayList<>();
    List<TokenLogEntry> log = new ArrayList<>();
    int allAgree = 0;
    int noMember = 0;
    int[] fusedEqualsMember = new int[count];
    double sumEntropy = 0;
    double[] sumKl = new double[count];
    boolean truncated = false;
    boolean stopped = false;
    int position = prompt.length;

    for (int step = 0; ; step++) {
      float[][] logits = new float[count][];
      double[][] logProbabilities = new double[count][];
      for (int i = 0; i < count; i++) {
        logits[i] = outputs[i].logits();
        logProbabilities[i] = outputs[i].logProbabilities();
      }
      double[] scores = FusionMath.combine(rule, logits, logProbabilities, weights);
      if (observer != null) {
        observer.onStep(step, logits, scores);
      }
      int token =
          settings.greedy() ? FusionMath.argmax(scores) : sampler.sample(toFloat(scores), null);
      double[] fusedLogProbabilities = FusionMath.logSoftmax(scores);

      int[] argmax = new int[count];
      boolean agree = true;
      boolean matchedAny = false;
      double entropy = FusionMath.entropy(fusedLogProbabilities);
      double[] kl = new double[count];
      for (int i = 0; i < count; i++) {
        argmax[i] = FusionMath.argmax(logProbabilities[i]);
        agree &= argmax[i] == argmax[0];
        if (argmax[i] == token) {
          fusedEqualsMember[i]++;
          matchedAny = true;
        }
        kl[i] = FusionMath.klDivergence(fusedLogProbabilities, logProbabilities[i]);
        sumKl[i] += kl[i];
      }
      allAgree += agree ? 1 : 0;
      noMember += matchedAny ? 0 : 1;
      sumEntropy += entropy;
      if (settings.tokenLogEvery() > 0 && step % settings.tokenLogEvery() == 0) {
        Map<String, Integer> argmaxByMember = new LinkedHashMap<>();
        Map<String, Double> klByMember = new LinkedHashMap<>();
        List<String> matches = new ArrayList<>();
        for (int i = 0; i < count; i++) {
          argmaxByMember.put(names.get(i), argmax[i]);
          klByMember.put(names.get(i), kl[i]);
          if (argmax[i] == token) {
            matches.add(names.get(i));
          }
        }
        log.add(
            new TokenLogEntry(
                step,
                token,
                argmaxByMember,
                matches,
                entropy,
                klByMember,
                fusedLogProbabilities[token]));
      }

      if (members.getFirst().tokenizer().isEndOfGeneration(token)) {
        stopped = true;
        break;
      }
      generated.add(token);
      tokenLogProbabilities.add(fusedLogProbabilities[token]);
      if (generated.size() >= settings.maxTokens()) {
        truncated = true;
        break;
      }
      int at = position++;
      List<Callable<StepOutput>> forwards = new ArrayList<>();
      for (FusionMember member : members) {
        forwards.add(
            () -> {
              long t0 = System.nanoTime();
              float[] next = member.forward(token, at);
              return new StepOutput(next, FusionMath.logSoftmax(next), System.nanoTime() - t0);
            });
      }
      outputs = runAll(executor, forwards);
      accumulate(memberNanos, outputs);
    }

    Map<String, Integer> equals = new LinkedHashMap<>();
    Map<String, Double> kls = new LinkedHashMap<>();
    Map<String, Long> timing = new LinkedHashMap<>();
    for (int i = 0; i < count; i++) {
      equals.put(names.get(i), fusedEqualsMember[i]);
      kls.put(names.get(i), sumKl[i]);
      timing.put(names.get(i), memberNanos[i]);
    }
    int steps = generated.size() + (stopped ? 1 : 0);
    int[] tokens = generated.stream().mapToInt(Integer::intValue).toArray();
    return new DecodeResult(
        tokens,
        members.getFirst().tokenizer().decode(tokens),
        truncated,
        stopped,
        tokenLogProbabilities.stream().mapToDouble(Double::doubleValue).toArray(),
        prefillNanos,
        System.nanoTime() - started,
        timing,
        new AgreementStats(steps, allAgree, equals, noMember, sumEntropy, kls),
        log);
  }

  /** One member alone: raw logits, argmax at T = 0, otherwise the public {@link Sampler}. */
  public static DecodeResult single(FusionMember member, int[] prompt, DecodeSettings settings) {
    long started = System.nanoTime();
    member.reset();
    float[] logits = member.prefill(prompt, 0);
    long prefillNanos = System.nanoTime() - started;
    return continueSingle(member, logits, prompt.length, settings, started, prefillNanos);
  }

  /**
   * {@code k} samples from one member that share one prompt prefill: the KV lineage is rewound to
   * the end of the prompt before each sample, and sample {@code j} uses seed {@code seed + j}.
   */
  public static List<DecodeResult> samples(
      FusionMember member, int[] prompt, DecodeSettings settings, int k) {
    long started = System.nanoTime();
    member.reset();
    float[] promptLogits = member.prefill(prompt, 0);
    long prefillNanos = System.nanoTime() - started;
    int checkpoint = member.checkpoint();
    List<DecodeResult> results = new ArrayList<>();
    for (int j = 0; j < k; j++) {
      long sampleStart = System.nanoTime();
      member.rewind(checkpoint);
      results.add(
          continueSingle(
              member,
              promptLogits.clone(),
              prompt.length,
              settings.withSeed(settings.seed() + j),
              sampleStart,
              j == 0 ? prefillNanos : 0));
    }
    return results;
  }

  private static DecodeResult continueSingle(
      FusionMember member,
      float[] logits,
      int position,
      DecodeSettings settings,
      long started,
      long prefillNanos) {
    Sampler sampler =
        settings.greedy() ? null : new Sampler(settings.samplingOptions(logits.length));
    List<Integer> generated = new ArrayList<>();
    List<Double> logProbabilities = new ArrayList<>();
    boolean truncated = false;
    boolean stopped = false;
    long forwardNanos = prefillNanos;
    while (true) {
      int token = settings.greedy() ? FusionMath.argmax(logits) : sampler.sample(logits, null);
      if (member.tokenizer().isEndOfGeneration(token)) {
        stopped = true;
        break;
      }
      generated.add(token);
      logProbabilities.add(FusionMath.logSoftmax(logits)[token]);
      if (generated.size() >= settings.maxTokens()) {
        truncated = true;
        break;
      }
      long t0 = System.nanoTime();
      logits = member.forward(token, position++);
      forwardNanos += System.nanoTime() - t0;
    }
    int[] tokens = generated.stream().mapToInt(Integer::intValue).toArray();
    return new DecodeResult(
        tokens,
        member.tokenizer().decode(tokens),
        truncated,
        stopped,
        logProbabilities.stream().mapToDouble(Double::doubleValue).toArray(),
        prefillNanos,
        System.nanoTime() - started,
        Map.of(member.name(), forwardNanos),
        null,
        List.of());
  }

  /**
   * Teacher-forced log-probabilities of {@code continuation} after a prompt whose final logits are
   * {@code promptLogits} and whose KV ends at {@code checkpoint}. The lineage is rewound to {@code
   * checkpoint} afterwards.
   */
  public static ContinuationScore scoreContinuation(
      FusionMember member,
      float[] promptLogits,
      int checkpoint,
      int[] continuation,
      boolean preferVerify) {
    if (continuation.length == 0) {
      throw new IllegalArgumentException("continuation must not be empty");
    }
    float[][] rows = new float[continuation.length][];
    rows[0] = promptLogits;
    boolean usedVerify = false;
    if (continuation.length > 1) {
      int[] prefix = new int[continuation.length - 1];
      System.arraycopy(continuation, 0, prefix, 0, prefix.length);
      float[][] verified = preferVerify ? member.verify(prefix, checkpoint) : null;
      if (verified != null) {
        usedVerify = true;
        System.arraycopy(verified, 0, rows, 1, verified.length);
      } else {
        for (int i = 0; i < prefix.length; i++) {
          rows[i + 1] = member.forward(prefix[i], checkpoint + i);
        }
      }
    }
    member.rewind(checkpoint);
    double[] logProbabilities = new double[continuation.length];
    double[] raw = new double[continuation.length];
    for (int i = 0; i < continuation.length; i++) {
      logProbabilities[i] = FusionMath.logSoftmax(rows[i])[continuation[i]];
      raw[i] = rows[i][continuation[i]];
    }
    return new ContinuationScore(logProbabilities, raw, usedVerify);
  }

  private static float[] toFloat(double[] scores) {
    float[] result = new float[scores.length];
    for (int i = 0; i < scores.length; i++) {
      result[i] = (float) scores[i];
    }
    return result;
  }

  private static void accumulate(long[] totals, StepOutput[] outputs) {
    for (int i = 0; i < outputs.length; i++) {
      totals[i] += outputs[i].nanos();
    }
  }

  private static StepOutput[] runAll(ExecutorService executor, List<Callable<StepOutput>> tasks) {
    try {
      List<Future<StepOutput>> futures = new ArrayList<>();
      for (Callable<StepOutput> task : tasks) {
        futures.add(executor.submit(task));
      }
      StepOutput[] outputs = new StepOutput[tasks.size()];
      for (int i = 0; i < outputs.length; i++) {
        outputs[i] = futures.get(i).get();
      }
      return outputs;
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("interrupted while stepping members", interrupted);
    } catch (ExecutionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IllegalStateException("member step failed", cause);
    }
  }
}
