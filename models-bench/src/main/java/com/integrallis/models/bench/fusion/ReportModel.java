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

import java.util.List;
import java.util.Map;

/** Versioned report schema shared by every logit-fusion command (see {@link ReportSchema}). */
final class ReportModel {

  static final int SCHEMA_VERSION = 1;

  private ReportModel() {}

  record Report(
      int schemaVersion,
      String kind,
      String createdAt,
      Config config,
      FusionEnvironment environment,
      Timings timings,
      Object summary,
      List<?> items) {}

  record Config(
      ArmConfig arm,
      DecodingConfig decoding,
      List<MemberConfig> members,
      DatasetConfig dataset,
      BackendConfig backend,
      Map<String, Object> extra) {}

  record ArmConfig(
      String label,
      String spec,
      String canonical,
      String kind,
      List<String> members,
      String rule,
      List<Double> weights,
      String weightSource,
      String tieBreakMember,
      int samples,
      String frozenPath,
      String frozenSha256,
      boolean pilot,
      String pilotSubstitution,
      String scorer,
      List<String> reusedMemberReports,
      List<String> reusedMemberReportSha256s) {}

  record DecodingConfig(
      float temperature,
      int topK,
      float topP,
      List<Long> seeds,
      int maxTokens,
      String thinking,
      String chatTemplate,
      String chatTemplateSha256,
      String promptsSource,
      String promptsSha256,
      int memberThreads,
      int contextLength,
      int tokenLogEvery,
      String arcScoring,
      String rerankScore,
      String rerankPromptThinking,
      String greedyRule) {}

  record MemberConfig(
      String name,
      String path,
      String sha256,
      long sizeBytes,
      String tokenizerSha256,
      Map<String, String> tokenizerFields,
      String chatTemplateSha256,
      String architecture,
      int ggufFileType,
      int vocabularySize,
      com.fasterxml.jackson.databind.JsonNode diagnostics,
      long loadMillis) {}

  record DatasetConfig(
      String name,
      String path,
      String sha256,
      String hfRepo,
      String hfRevision,
      String manifestPath,
      String manifestSha256,
      Boolean sha256MatchesManifest,
      String devIdsSha256,
      int offset,
      int limit,
      int itemCount,
      String selectedIdsSha256) {}

  record BackendConfig(String name, String nativeLibraryPath, String nativeLibrarySha256) {}

  record Timings(
      long wallMillis,
      double processCpuSeconds,
      long peakRssBytes,
      String peakRssMethod,
      long loadMillis) {}

  record Output(
      String producer,
      Long seed,
      String text,
      int[] tokenIds,
      double[] tokenLogProbabilities,
      String reasoningAnswer,
      String statedAnswer,
      Double confidence,
      boolean consistent,
      boolean thinkPresent,
      boolean thinkTruncated,
      boolean truncated,
      boolean stoppedOnEndOfGeneration,
      int tokens,
      long wallMillis,
      long prefillMillis,
      Map<String, Long> memberForwardMillis,
      AgreementStats agreement,
      List<TokenLogEntry> tokenLog,
      boolean reused) {}

  record CandidateScore(
      String answer,
      String continuation,
      Map<String, Double> memberScores,
      double fusedScore,
      boolean anyVerify) {}

  record Item(
      String id,
      String gold,
      String renderedPromptSha256,
      int promptTokens,
      String prediction,
      boolean correct,
      boolean truncated,
      boolean extractionFailure,
      int generatedTokens,
      long wallMillis,
      double coreSeconds,
      double reusedCoreSeconds,
      List<Output> outputs,
      Map<String, String> aggregations,
      Map<String, Boolean> aggregationCorrect,
      List<CandidateScore> candidates) {}

  record Summary(
      int items,
      int correct,
      double accuracy,
      int outputs,
      int truncatedOutputs,
      double truncationRate,
      int extractionFailures,
      double extractionFailureRate,
      long generatedTokens,
      double generationSeconds,
      double tokensPerSecond,
      double coreSeconds,
      double coreSecondsPerItem,
      double reusedCoreSeconds,
      double p95TraceTokens,
      Map<String, Double> aggregationAccuracy,
      Map<String, Double> reasoningAnswerMismatchRate,
      G3 g3,
      boolean g5TruncationFlag) {}

  record G3(
      long steps,
      long allMembersAgree,
      double agreementRate,
      Map<String, Double> fusedEqualsMemberRate,
      double fusedEqualsNoMemberRate,
      double meanFusedEntropy,
      Map<String, Double> meanKlFusedToMember,
      boolean flagged) {}
}
