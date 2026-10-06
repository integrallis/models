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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class AnswerExtractionTest {

  @Test
  void splitsThinkBlockFromFinalResponse() {
    ReasoningTrace closed = ReasoningTrace.split("<think>\nabc\n</think>\n\nfinal");
    assertThat(closed.thinkPresent()).isTrue();
    assertThat(closed.thinkTruncated()).isFalse();
    assertThat(closed.reasoning()).isEqualTo("\nabc\n");
    assertThat(closed.finalText()).isEqualTo("\n\nfinal");
    assertThat(closed.finalStart()).isEqualTo("<think>\nabc\n</think>".length());

    ReasoningTrace truncated = ReasoningTrace.split("<think>\nhalf way");
    assertThat(truncated.thinkTruncated()).isTrue();
    assertThat(truncated.finalText()).isEmpty();

    ReasoningTrace missing = ReasoningTrace.split("just an answer 5");
    assertThat(missing.thinkPresent()).isFalse();
    assertThat(missing.reasoning()).isNull();
    assertThat(missing.finalText()).isEqualTo("just an answer 5");
  }

  @Test
  void gsm8kPrefersBoxedThenAnswerPhraseThenHashesThenLastNumber() {
    AnswerExtractor gsm = AnswerExtractor.forDataset(DatasetKind.GSM8K);
    assertThat(gsm.stated("a 3 then \\boxed{1,250} and 7", null).orElseThrow().value())
        .isEqualTo("1250");
    assertThat(gsm.stated("The answer is $42.50.", null).orElseThrow().value()).isEqualTo("42.5");
    assertThat(gsm.stated("blah 3\n#### 3500", null).orElseThrow().value()).isEqualTo("3500");
    assertThat(gsm.stated("There are 54 chairs.", null).orElseThrow().value()).isEqualTo("54");
    assertThat(gsm.reasoning("12 and 18\n\nNow I need to check", null)).isEmpty();
    assertThat(gsm.equivalent("18.50", "18.5")).isTrue();
    assertThat(gsm.equivalent("-7", "7")).isFalse();
  }

  @Test
  void spanPointsAtTheExtractedAnswerText() {
    AnswerExtractor gsm = AnswerExtractor.forDataset(DatasetKind.GSM8K);
    String text = "so \\boxed{42} done";
    AnswerExtraction extraction = gsm.stated(text, null).orElseThrow();
    assertThat(text.substring(extraction.start(), extraction.end())).isEqualTo("42");
  }

  @Test
  void arcAcceptsOnlyDeclaredLabelsAndNeverSentenceInitialArticles() {
    AnswerExtractor arc = AnswerExtractor.forDataset(DatasetKind.ARC);
    List<String> letters = List.of("A", "B", "C", "D");
    assertThat(arc.stated("A lens bends light.", letters)).isEmpty();
    assertThat(arc.stated("\"answer\": \"C\"", letters).orElseThrow().value()).isEqualTo("C");
    assertThat(arc.stated("Therefore the answer is (C).", letters).orElseThrow().value())
        .isEqualTo("C");
    assertThat(arc.stated("text\n\n(B)", letters).orElseThrow().value()).isEqualTo("B");
    assertThat(arc.stated("The answer is E.", letters)).isEmpty();
    assertThat(arc.stated("answer: 4", List.of("1", "2", "3", "4")).orElseThrow().value())
        .isEqualTo("4");
  }

  @Test
  void mathNormalizerHandlesTheUsualLatexVariants() {
    AnswerExtractor math = AnswerExtractor.forDataset(DatasetKind.MATH500);
    assertThat(math.equivalent("\\dfrac{3}{4}", "\\frac{3}{4}")).isTrue();
    assertThat(math.equivalent("\\frac12", "\\frac{1}{2}")).isTrue();
    assertThat(math.equivalent("0.5", "\\frac{1}{2}")).isTrue();
    assertThat(math.equivalent("\\left( 3, -1 \\right)", "(3,-1)")).isTrue();
    assertThat(math.equivalent("(0, 4]", "[0, 4]")).isFalse();
    assertThat(math.equivalent("x^2+1", "x^2-1")).isFalse();
    assertThat(math.equivalent("90^\\circ", "90")).isTrue();
    assertThat(math.equivalent("\\text{(C)}", "(C)")).isTrue();
    assertThat(math.equivalent("x = 5", "5")).isTrue();
    assertThat(math.equivalent("1,000", "1000")).isTrue();
    assertThat(math.equivalent("\\sqrt3", "\\sqrt{3}")).isTrue();
    assertThat(math.equivalent("10\\%", "10")).isTrue();
    assertThat(math.stated("so \\boxed{\\frac{1}{\\sqrt{2}}}.", null).orElseThrow().value())
        .isEqualTo("\\frac{1}{\\sqrt{2}}");
    assertThat(math.stated("\\boxed{\\{1,2,3\\}}", null).orElseThrow().value())
        .isEqualTo("\\{1,2,3\\}");
  }

  @Test
  void consistencyOnlyFailsWhenBothAnswersExistAndDiffer() {
    AnswerExtractor gsm = AnswerExtractor.forDataset(DatasetKind.GSM8K);
    TraceAnalysis disagree =
        TraceAnalysis.analyze(gsm, "<think>\n\\boxed{3}\n</think>\n\\boxed{4}", null);
    assertThat(disagree.consistent()).isFalse();
    TraceAnalysis missing = TraceAnalysis.analyze(gsm, "\\boxed{4}", null);
    assertThat(missing.consistent()).isTrue();
    assertThat(missing.reasoningAnswer()).isNull();
    TraceAnalysis truncated = TraceAnalysis.analyze(gsm, "<think>\nso \\boxed{3}", null);
    assertThat(truncated.statedAnswer()).isNull();
    assertThat(truncated.consistent()).isTrue();
  }
}
