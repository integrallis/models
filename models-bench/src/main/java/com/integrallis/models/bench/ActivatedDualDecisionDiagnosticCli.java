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
import com.integrallis.models.api.ActivatedAdapterMetadata;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.ActivatedToolCallingModel;
import com.integrallis.models.runtime.ActivatedToolTurn;
import com.integrallis.models.runtime.ToolDecisionScore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/** Profiles complementary base and activated decision margins over exposed development inputs. */
final class ActivatedDualDecisionDiagnosticCli {
  private static final List<Double> BASE_WEIGHTS = List.of(-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0);

  private ActivatedDualDecisionDiagnosticCli() {}

  record Observation(
      String id,
      String kind,
      String partition,
      boolean callExpected,
      int promptTokens,
      int sharedPrefixTokens,
      boolean physicallyShared,
      double baseMargin,
      double specialistMargin,
      long elapsedMillis) {}

  record Score(
      int calls,
      int noCalls,
      int correctCalls,
      int correctNoCalls,
      double callAccuracy,
      double noCallAccuracy,
      double balancedAccuracy) {}

  record Selection(double baseWeight, double threshold, Score calibration, Score screen) {}

  record Report(
      int schemaVersion,
      String experiment,
      String createdAt,
      String modelsRevision,
      String sourceSha256,
      String modelSha256,
      ActivatedAdapterMetadata adapter,
      BenchmarkEnvironment environment,
      ActivatedDecisionProfileCli.SplitContract split,
      Selection selection,
      List<Observation> observations) {}

  static int run(String[] args) throws IOException {
    ActivatedDecisionProfileCli.Configuration configuration =
        ActivatedDecisionProfileCli.parse(args);
    String sourceSha256 = Hashing.sha256(configuration.records());
    if (!ActivatedDecisionProfileCli.SOURCE_SHA256.equals(sourceSha256)) {
      throw new IllegalArgumentException("dual diagnostic requires the exposed V15 records");
    }
    ObjectMapper mapper = ActivatedDecisionProfileCli.mapper();
    ActivatedDecisionProfileCli.LoadedCases loaded =
        ActivatedDecisionProfileCli.loadCases(mapper, configuration.records());
    List<Observation> observations = new ArrayList<>();
    ActivatedAdapterMetadata adapter;
    PureJavaBackend backend =
        PureJavaBackend.loadActivatedAdapter(configuration.model(), configuration.adapter());
    ActivatedToolCallingModel activatedModel;
    try {
      activatedModel = new ActivatedToolCallingModel(backend, 1);
    } catch (RuntimeException | Error failure) {
      try {
        backend.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
    try (ActivatedToolCallingModel model = activatedModel) {
      adapter = model.adapter();
      int ordinal = 0;
      for (ActivatedDecisionProfileCli.SourceCase item : loaded.cases()) {
        ordinal++;
        long started = System.nanoTime();
        try (ActivatedToolTurn turn =
            model.openToolTurn(item.prompt(), ActivatedToolCallingModel.PrefixStrategy.SHARED)) {
          ToolDecisionScore base =
              turn.scoreBaseToolDecision(
                  ActivatedDecisionProfileCli.CALL_TOKEN_ID,
                  ActivatedDecisionProfileCli.NO_CALL_TOKEN_ID);
          ToolDecisionScore specialist =
              turn.scoreToolDecision(
                  ActivatedDecisionProfileCli.CALL_TOKEN_ID,
                  ActivatedDecisionProfileCli.NO_CALL_TOKEN_ID);
          long elapsedMillis = (System.nanoTime() - started) / 1_000_000L;
          Observation observation =
              new Observation(
                  item.id(),
                  item.kind(),
                  item.partition(),
                  item.callExpected(),
                  backend.tokenizer().encode(item.prompt()).length,
                  turn.sharedPrefixTokens(),
                  turn.physicallySharesPrefix(),
                  base.callMargin(),
                  specialist.callMargin(),
                  elapsedMillis);
          observations.add(observation);
          System.out.printf(
              Locale.ROOT,
              "%2d/%d %-18s %-11s expected=%-5s base=%10.4f specialist=%10.4f shared=%s %d ms%n",
              ordinal,
              loaded.cases().size(),
              item.id(),
              item.partition(),
              item.callExpected(),
              base.callMargin(),
              specialist.callMargin(),
              observation.physicallyShared(),
              elapsedMillis);
        }
      }
    }
    if (!allPhysicallyShared(observations)) {
      throw new IllegalStateException(
          "dual decision diagnostic did not physically share every prefix");
    }
    Selection selection = select(observations);
    Report report =
        new Report(
            1,
            "qwen3-17b-dual-decision-diagnostic",
            Instant.now().toString(),
            configuration.modelsRevision(),
            sourceSha256,
            Hashing.sha256(configuration.model()),
            adapter,
            BenchmarkEnvironment.capture(),
            loaded.split(),
            selection,
            List.copyOf(observations));
    Path parent = configuration.output().toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper.writeValue(configuration.output().toFile(), report);
    System.out.printf(
        Locale.ROOT,
        "score = specialist + %.2f * base; call when score > %.6f%n",
        selection.baseWeight(),
        selection.threshold());
    print("calibration", selection.calibration());
    print("screen", selection.screen());
    System.out.println("diagnostic report: " + configuration.output().toAbsolutePath());
    return 0;
  }

  static Selection select(List<Observation> observations) {
    List<Observation> calibration = partition(observations, "calibration");
    List<Observation> screen = partition(observations, "screen");
    Selection selected = null;
    for (double baseWeight : BASE_WEIGHTS) {
      List<Double> scores = calibration.stream().map(item -> combined(item, baseWeight)).toList();
      for (double threshold : thresholds(scores)) {
        Selection candidate =
            new Selection(
                baseWeight,
                threshold,
                score(calibration, baseWeight, threshold),
                score(screen, baseWeight, threshold));
        if (selected == null || better(candidate, selected)) {
          selected = candidate;
        }
      }
    }
    if (selected == null) {
      throw new IllegalArgumentException("no dual decision candidate was available");
    }
    return selected;
  }

  static boolean allPhysicallyShared(List<Observation> observations) {
    return !observations.isEmpty() && observations.stream().allMatch(Observation::physicallyShared);
  }

  private static boolean predictsCall(Observation item, double baseWeight, double threshold) {
    return combined(item, baseWeight) > threshold;
  }

  private static double combined(Observation item, double baseWeight) {
    return item.specialistMargin() + baseWeight * item.baseMargin();
  }

  private static boolean better(Selection candidate, Selection current) {
    return Comparator.comparingDouble((Selection item) -> item.calibration().balancedAccuracy())
            .thenComparingInt(item -> item.calibration().correctNoCalls())
            .thenComparingInt(item -> item.calibration().correctCalls())
            .thenComparingDouble(item -> -Math.abs(item.baseWeight()))
            .thenComparingDouble(Selection::baseWeight)
            .thenComparingDouble(Selection::threshold)
            .compare(candidate, current)
        > 0;
  }

  private static Score score(List<Observation> observations, double baseWeight, double threshold) {
    int calls = 0;
    int noCalls = 0;
    int correctCalls = 0;
    int correctNoCalls = 0;
    for (Observation item : observations) {
      boolean predicted = predictsCall(item, baseWeight, threshold);
      if (item.callExpected()) {
        calls++;
        correctCalls += predicted ? 1 : 0;
      } else {
        noCalls++;
        correctNoCalls += predicted ? 0 : 1;
      }
    }
    if (calls == 0 || noCalls == 0) {
      throw new IllegalArgumentException("both applicability classes are required");
    }
    double callAccuracy = (double) correctCalls / calls;
    double noCallAccuracy = (double) correctNoCalls / noCalls;
    return new Score(
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        callAccuracy,
        noCallAccuracy,
        (callAccuracy + noCallAccuracy) / 2.0);
  }

  private static List<Double> thresholds(List<Double> values) {
    List<Double> distinct = values.stream().distinct().sorted().toList();
    if (distinct.isEmpty()) {
      throw new IllegalArgumentException("threshold input is empty");
    }
    List<Double> result = new ArrayList<>();
    result.add(Math.nextDown(distinct.getFirst()));
    for (int index = 1; index < distinct.size(); index++) {
      double lower = distinct.get(index - 1);
      double upper = distinct.get(index);
      result.add(lower + (upper - lower) / 2.0);
    }
    result.add(distinct.getLast());
    return result;
  }

  private static List<Observation> partition(List<Observation> observations, String name) {
    List<Observation> result =
        observations.stream().filter(item -> name.equals(item.partition())).toList();
    if (result.isEmpty()) {
      throw new IllegalArgumentException("missing " + name + " observations");
    }
    return result;
  }

  private static void print(String label, Score score) {
    System.out.printf(
        Locale.ROOT,
        "%s calls=%d/%d no-calls=%d/%d balanced=%.5f%n",
        label,
        score.correctCalls(),
        score.calls(),
        score.correctNoCalls(),
        score.noCalls(),
        score.balancedAccuracy());
  }
}
