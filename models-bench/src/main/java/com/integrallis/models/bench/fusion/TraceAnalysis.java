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
import java.util.Optional;

/**
 * Reasoning-aware extraction of one response: the answer reached in the reasoning (a_r), the answer
 * stated to the user (a_u), and whether they agree.
 *
 * @param reasoningAnswer a_r, or {@code null}
 * @param statedAnswer a_u, or {@code null}; this is the vote the response casts
 * @param statedStart absolute start offset of a_u in the response, or -1
 * @param statedEnd absolute end offset of a_u in the response, or -1
 * @param consistent false only when both answers exist and are not equivalent
 * @param thinkPresent whether a think block was opened
 * @param thinkTruncated whether it was never closed
 */
public record TraceAnalysis(
    String reasoningAnswer,
    String statedAnswer,
    int statedStart,
    int statedEnd,
    boolean consistent,
    boolean thinkPresent,
    boolean thinkTruncated) {

  /** Analyses a full response with the dataset's extractor. */
  public static TraceAnalysis analyze(AnswerExtractor extractor, String text, List<String> labels) {
    ReasoningTrace trace = ReasoningTrace.split(text);
    Optional<AnswerExtraction> reasoning =
        trace.reasoning() == null
            ? Optional.empty()
            : extractor
                .reasoning(trace.reasoning(), labels)
                .map(a -> a.shift(trace.reasoningStart()));
    Optional<AnswerExtraction> stated =
        trace.thinkTruncated()
            ? Optional.empty()
            : extractor.stated(trace.finalText(), labels).map(a -> a.shift(trace.finalStart()));
    String reasoningAnswer = reasoning.map(AnswerExtraction::value).orElse(null);
    String statedAnswer = stated.map(AnswerExtraction::value).orElse(null);
    boolean consistent =
        reasoningAnswer == null
            || statedAnswer == null
            || extractor.equivalent(reasoningAnswer, statedAnswer);
    return new TraceAnalysis(
        reasoningAnswer,
        statedAnswer,
        stated.map(AnswerExtraction::start).orElse(-1),
        stated.map(AnswerExtraction::end).orElse(-1),
        consistent,
        trace.thinkPresent(),
        trace.thinkTruncated());
  }
}
