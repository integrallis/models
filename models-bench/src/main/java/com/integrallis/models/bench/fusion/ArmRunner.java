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

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.bench.fusion.ReportModel.CandidateScore;
import com.integrallis.models.bench.fusion.ReportModel.Item;
import com.integrallis.models.bench.fusion.ReportModel.Output;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.function.Consumer;

/** Executes one arm item by item. Model loading and report assembly live in the CLI. */
final class ArmRunner {

  /** Everything an arm needs besides its items. */
  record Context(
      ArmSpec arm,
      DatasetKind dataset,
      AnswerExtractor extractor,
      PromptBuilder prompts,
      DecodeSettings settings,
      boolean thinking,
      boolean loglik,
      boolean rerankMean,
      FrozenTuning frozen,
      boolean pilot,
      ExecutorService executor,
      Map<String, FusionMember> members,
      Map<String, Map<String, ReusedOutputs.Entry>> reused) {
    Context {
      members = Map.copyOf(members);
      reused = Map.copyOf(reused);
    }
  }

  /** How weights, tie-break and temperatures were resolved, including pilot substitutions. */
  record Resolution(
      double[] weights,
      String weightSource,
      String tieBreak,
      Map<String, Double> temperatures,
      String pilotSubstitution) {}

  private final Context context;
  private final Resolution resolution;

  ArmRunner(Context context) {
    this.context = Objects.requireNonNull(context, "context");
    this.resolution = resolve(context);
  }

  Resolution resolution() {
    return resolution;
  }

  static Resolution resolve(Context context) {
    ArmSpec arm = context.arm();
    FrozenTuning frozen = context.frozen();
    List<String> substitutions = new ArrayList<>();
    if (arm.requiresFrozen() && frozen == null && !context.pilot()) {
      throw new IllegalStateException(
          "arm " + arm.canonical() + " needs --frozen and --frozen-sha256 (or --pilot true)");
    }
    double[] weights;
    String weightSource;
    if (arm.weightMode() == ArmSpec.WeightMode.TUNED && frozen == null) {
      weights = ArmSpec.parse(arm.canonical().replace(":tuned", ":uniform")).resolveWeights(null);
      weightSource = "uniform (pilot substitution for tuned)";
      substitutions.add("tuned weights -> uniform");
    } else {
      weights = arm.resolveWeights(frozen);
      weightSource =
          switch (arm.weightMode()) {
            case TUNED -> "frozen:" + frozen.sha256();
            case UNIFORM -> "uniform";
            case EXPLICIT -> "explicit";
            case NONE -> "none";
          };
    }
    String tieBreak = null;
    if (arm.kind().isVote()) {
      if (ArmSpec.BEST.equals(arm.tieBreakMember()) && frozen == null) {
        tieBreak = arm.members().contains("B") ? "B" : arm.members().getFirst();
        substitutions.add("tie-break best -> " + tieBreak);
      } else {
        tieBreak = arm.resolveTieBreak(frozen);
      }
    }
    Map<String, Double> temperatures = null;
    if (arm.kind().isVote()) {
      if (frozen != null) {
        try {
          temperatures = frozen.temperatures();
        } catch (IllegalStateException uncalibrated) {
          if (arm.kind() == ArmSpec.Kind.VOTE_CONF) {
            throw uncalibrated;
          }
        }
      } else {
        temperatures = new LinkedHashMap<>();
        for (String member : arm.members()) {
          temperatures.put(member, 1.0);
        }
        if (arm.kind() == ArmSpec.Kind.VOTE_CONF) {
          substitutions.add("uncalibrated confidence temperature 1.0");
        }
      }
    }
    return new Resolution(
        weights,
        weightSource,
        tieBreak,
        temperatures,
        substitutions.isEmpty() ? null : String.join("; ", substitutions));
  }

  Item run(DatasetItem item) {
    long wallStart = System.nanoTime();
    long cpuStart = FusionEnvironment.processCpuNanos();
    ModelPrompt prompt =
        context
            .prompts()
            .render(context.dataset(), context.extractor().id(), item, context.thinking());
    String promptSha = Digests.sha256(prompt.text());
    List<Output> outputs = new ArrayList<>();
    List<CandidateScore> candidates = new ArrayList<>();
    Map<String, String> aggregations = new LinkedHashMap<>();
    double reusedCore = 0;
    String prediction;
    int promptTokens = -1;
    ArmSpec arm = context.arm();
    List<String> labels = item.labels();

    if (context.loglik()) {
      if (arm.kind() != ArmSpec.Kind.MEMBER && arm.kind() != ArmSpec.Kind.FUSE) {
        throw new IllegalArgumentException("loglik scoring applies to member and fuse arms only");
      }
      ModelPrompt loglikPrompt = context.prompts().renderLoglik(item);
      Tokenizer tokenizer = member(arm.members().getFirst()).tokenizer();
      int[] tokens =
          concat(
              tokenizer.encode(loglikPrompt),
              tokenizer.encode(context.prompts().loglikAssistantPrefix()));
      promptSha = Digests.sha256(loglikPrompt.text() + context.prompts().loglikAssistantPrefix());
      promptTokens = tokens.length;
      prediction = loglikChoice(item, tokens, candidates);
    } else {
      switch (arm.kind()) {
        case MEMBER -> {
          FusionMember member = member(arm.members().getFirst());
          int[] tokens = member.tokenizer().encode(prompt);
          promptTokens = tokens.length;
          outputs.add(
              toOutput(
                  member.name(),
                  null,
                  FusionDecoder.single(member, tokens, context.settings()),
                  member.tokenizer(),
                  labels));
          prediction = outputs.getFirst().statedAnswer();
        }
        case FUSE -> {
          List<FusionMember> members = arm.members().stream().map(this::member).toList();
          int[] tokens = members.getFirst().tokenizer().encode(prompt);
          promptTokens = tokens.length;
          DecodeResult result =
              FusionDecoder.fuse(
                  members,
                  tokens,
                  resolution.weights(),
                  arm.rule(),
                  context.settings(),
                  context.executor());
          outputs.add(toOutput("fused", null, result, members.getFirst().tokenizer(), labels));
          prediction = outputs.getFirst().statedAnswer();
        }
        case SELF_CONSISTENCY -> {
          FusionMember member = member(arm.members().getFirst());
          int[] tokens = member.tokenizer().encode(prompt);
          promptTokens = tokens.length;
          List<DecodeResult> samples =
              FusionDecoder.samples(member, tokens, context.settings(), arm.samples());
          List<String> answers = new ArrayList<>();
          for (int j = 0; j < samples.size(); j++) {
            Output output =
                toOutput(
                    member.name(),
                    context.settings().seed() + j,
                    samples.get(j),
                    member.tokenizer(),
                    labels);
            outputs.add(output);
            answers.add(output.statedAnswer());
          }
          prediction = Voting.selfConsistency(answers, context.extractor());
        }
        case VOTE, VOTE_CONSIST, VOTE_CONF, RERANK -> {
          for (String name : arm.members()) {
            ReusedOutputs.Entry reusedEntry = reusedEntry(name, item.id());
            if (reusedEntry != null) {
              outputs.add(reusedEntry.output());
              reusedCore += reusedEntry.coreSeconds();
              if (!reusedEntry.renderedPromptSha256().equals(promptSha)) {
                throw new IllegalStateException(
                    "reused output for "
                        + name
                        + "/"
                        + item.id()
                        + " was produced from a different prompt");
              }
            } else {
              FusionMember member = member(name);
              int[] tokens = member.tokenizer().encode(prompt);
              promptTokens = tokens.length;
              outputs.add(
                  toOutput(
                      name,
                      null,
                      FusionDecoder.single(member, tokens, context.settings()),
                      member.tokenizer(),
                      labels));
            }
          }
          List<Vote> votes =
              outputs.stream()
                  .map(
                      o ->
                          new Vote(
                              o.producer(),
                              o.statedAnswer(),
                              o.consistent(),
                              o.confidence() == null ? Double.NaN : o.confidence()))
                  .toList();
          List<String> priority =
              arm.priority(
                  resolution.tieBreak() == null ? arm.members().getFirst() : resolution.tieBreak());
          aggregations.put("majority", Voting.majority(votes, context.extractor(), priority));
          aggregations.put(
              "consist", Voting.consistencyFiltered(votes, context.extractor(), priority));
          aggregations.put(
              "conf",
              resolution.temperatures() == null
                  ? null
                  : Voting.confidenceWeighted(
                      votes, context.extractor(), priority, resolution.temperatures()));
          if (arm.kind() == ArmSpec.Kind.RERANK) {
            prediction = rerank(item, votes, candidates);
          } else {
            prediction =
                switch (arm.kind()) {
                  case VOTE_CONSIST -> aggregations.get("consist");
                  case VOTE_CONF -> aggregations.get("conf");
                  default -> aggregations.get("majority");
                };
          }
        }
        default -> throw new IllegalStateException("unhandled arm " + arm.kind());
      }
    }

    Map<String, Boolean> aggregationCorrect = new LinkedHashMap<>();
    aggregations.forEach(
        (key, value) -> aggregationCorrect.put(key, correct(value, item.answer())));
    if (!outputs.isEmpty()) {
      boolean oracle = outputs.stream().anyMatch(o -> correct(o.statedAnswer(), item.answer()));
      aggregationCorrect.put("oracle", oracle);
    }
    boolean truncated = outputs.stream().anyMatch(Output::truncated);
    int generated = outputs.stream().filter(o -> !o.reused()).mapToInt(Output::tokens).sum();
    long cpuEnd = FusionEnvironment.processCpuNanos();
    return new Item(
        item.id(),
        item.answer(),
        promptSha,
        promptTokens,
        prediction,
        correct(prediction, item.answer()),
        truncated,
        prediction == null,
        generated,
        (System.nanoTime() - wallStart) / 1_000_000L,
        cpuStart < 0 ? -1 : (cpuEnd - cpuStart) / 1e9,
        reusedCore,
        outputs,
        aggregations,
        aggregationCorrect,
        candidates);
  }

  private boolean correct(String answer, String gold) {
    return answer != null && context.extractor().equivalent(answer, gold);
  }

  private FusionMember member(String name) {
    FusionMember member = context.members().get(name);
    if (member == null) {
      throw new IllegalStateException("member " + name + " is not loaded and has no reused report");
    }
    return member;
  }

  private ReusedOutputs.Entry reusedEntry(String member, String id) {
    Map<String, ReusedOutputs.Entry> byId = context.reused().get(member);
    if (byId == null) {
      return null;
    }
    ReusedOutputs.Entry entry = byId.get(id);
    if (entry == null) {
      throw new IllegalStateException("reused report for " + member + " has no item " + id);
    }
    return entry;
  }

  Output toOutput(
      String producer, Long seed, DecodeResult result, Tokenizer tokenizer, List<String> labels) {
    TraceAnalysis analysis = TraceAnalysis.analyze(context.extractor(), result.text(), labels);
    Double confidence = null;
    if (analysis.statedStart() >= 0) {
      int[] span =
          AnswerTokens.span(
              tokenizer, result.tokens(), analysis.statedStart(), analysis.statedEnd());
      double mean = AnswerTokens.mean(result.tokenLogProbabilities(), span);
      confidence = Double.isNaN(mean) ? null : mean;
    }
    Map<String, Long> forwardMillis = new LinkedHashMap<>();
    result
        .memberForwardNanos()
        .forEach((name, nanos) -> forwardMillis.put(name, nanos / 1_000_000L));
    return new Output(
        producer,
        seed,
        result.text(),
        result.tokens(),
        result.tokenLogProbabilities(),
        analysis.reasoningAnswer(),
        analysis.statedAnswer(),
        confidence,
        analysis.consistent(),
        analysis.thinkPresent(),
        analysis.thinkTruncated(),
        result.truncated(),
        result.stoppedOnEndOfGeneration(),
        result.tokens().length,
        result.wallNanos() / 1_000_000L,
        result.prefillNanos() / 1_000_000L,
        forwardMillis,
        result.agreement(),
        result.tokenLog(),
        false);
  }

  private String loglikChoice(DatasetItem item, int[] tokens, List<CandidateScore> candidates) {
    ArmSpec arm = context.arm();
    List<FusionMember> members = arm.members().stream().map(this::member).toList();
    Map<String, float[]> promptLogits = new LinkedHashMap<>();
    Map<String, Integer> checkpoints = new LinkedHashMap<>();
    for (FusionMember member : members) {
      member.reset();
      promptLogits.put(member.name(), member.prefill(tokens, 0));
      checkpoints.put(member.name(), member.checkpoint());
    }
    FusionRule rule = arm.kind() == ArmSpec.Kind.FUSE ? arm.rule() : FusionRule.POE;
    double[] weights = arm.kind() == ArmSpec.Kind.FUSE ? resolution.weights() : new double[] {1.0};
    String best = null;
    double bestScore = Double.NEGATIVE_INFINITY;
    for (DatasetItem.Choice choice : item.choices()) {
      String continuationText = context.prompts().loglikContinuation(choice.label());
      Map<String, Double> memberScores = new LinkedHashMap<>();
      double[] sums = new double[members.size()];
      double[] raw = new double[members.size()];
      boolean verify = false;
      for (int i = 0; i < members.size(); i++) {
        FusionMember member = members.get(i);
        int[] continuation = member.tokenizer().encode(continuationText);
        ContinuationScore score =
            FusionDecoder.scoreContinuation(
                member,
                promptLogits.get(member.name()),
                checkpoints.get(member.name()),
                continuation,
                true);
        sums[i] = score.sum();
        raw[i] = score.rawSum();
        verify |= score.usedVerify();
        memberScores.put(member.name(), sums[i]);
      }
      double fused = combineSequenceScores(rule, weights, sums, raw);
      candidates.add(
          new CandidateScore(choice.label(), continuationText, memberScores, fused, verify));
      if (fused > bestScore) {
        bestScore = fused;
        best = choice.label();
      }
    }
    return best;
  }

  static double combineSequenceScores(
      FusionRule rule, double[] weights, double[] logLikelihoods, double[] rawLogits) {
    double fused = 0;
    switch (rule) {
      case POE -> {
        for (int i = 0; i < weights.length; i++) {
          fused += weights[i] == 0 ? 0 : weights[i] * logLikelihoods[i];
        }
      }
      case ARTICLE -> {
        for (int i = 0; i < weights.length; i++) {
          fused += weights[i] == 0 ? 0 : weights[i] * rawLogits[i];
        }
      }
      case MIXTURE -> {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < weights.length; i++) {
          if (weights[i] > 0) {
            max = Math.max(max, Math.log(weights[i]) + logLikelihoods[i]);
          }
        }
        double sum = 0;
        for (int i = 0; i < weights.length; i++) {
          if (weights[i] > 0) {
            sum += Math.exp(Math.log(weights[i]) + logLikelihoods[i] - max);
          }
        }
        fused = max + Math.log(sum);
      }
    }
    return fused;
  }

  private String rerank(DatasetItem item, List<Vote> votes, List<CandidateScore> candidates) {
    List<String> distinct = new ArrayList<>();
    for (Vote vote : votes) {
      if (vote.answer() != null
          && distinct.stream().noneMatch(a -> context.extractor().equivalent(a, vote.answer()))) {
        distinct.add(vote.answer());
      }
    }
    if (distinct.isEmpty()) {
      return null;
    }
    ArmSpec arm = context.arm();
    List<FusionMember> members = arm.members().stream().map(this::member).toList();
    ModelPrompt scoringPrompt =
        context.prompts().render(context.dataset(), context.extractor().id(), item, false);
    Map<String, float[]> promptLogits = new LinkedHashMap<>();
    Map<String, Integer> checkpoints = new LinkedHashMap<>();
    for (FusionMember member : members) {
      int[] tokens = member.tokenizer().encode(scoringPrompt);
      member.reset();
      promptLogits.put(member.name(), member.prefill(tokens, 0));
      checkpoints.put(member.name(), member.checkpoint());
    }
    String best = null;
    double bestScore = Double.NEGATIVE_INFINITY;
    for (String answer : distinct) {
      String continuationText = context.extractor().renderGold(answer);
      Map<String, Double> memberScores = new LinkedHashMap<>();
      double fused = 0;
      boolean verify = false;
      for (int i = 0; i < members.size(); i++) {
        FusionMember member = members.get(i);
        int[] continuation = member.tokenizer().encode(continuationText);
        ContinuationScore score =
            FusionDecoder.scoreContinuation(
                member,
                promptLogits.get(member.name()),
                checkpoints.get(member.name()),
                continuation,
                true);
        double value = context.rerankMean() ? score.mean() : score.sum();
        verify |= score.usedVerify();
        memberScores.put(member.name(), value);
        fused += resolution.weights()[i] * value;
      }
      candidates.add(new CandidateScore(answer, continuationText, memberScores, fused, verify));
      if (fused > bestScore) {
        bestScore = fused;
        best = answer;
      }
    }
    return best;
  }

  private static int[] concat(int[] left, int[] right) {
    int[] joined = Arrays.copyOf(left, left.length + right.length);
    System.arraycopy(right, 0, joined, left.length, right.length);
    return joined;
  }

  /** Runs every item, reporting progress. */
  List<Item> runAll(List<DatasetItem> items, Consumer<String> progress) {
    List<Item> results = new ArrayList<>();
    int ordinal = 0;
    for (DatasetItem item : items) {
      Item result = run(item);
      results.add(result);
      ordinal++;
      progress.accept(
          String.format(
              java.util.Locale.ROOT,
              "%4d/%d %-28s pred=%-12s gold=%-12s %-7s tokens=%d %d ms",
              ordinal,
              items.size(),
              result.id(),
              result.prediction(),
              result.gold(),
              result.correct() ? "correct" : "wrong",
              result.generatedTokens(),
              result.wallMillis()));
    }
    return results;
  }
}
