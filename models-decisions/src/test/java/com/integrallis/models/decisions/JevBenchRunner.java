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
package com.integrallis.models.decisions;

import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs a prepared JevBench cohort through the shared-prefix candidate evaluator.
 *
 * <p>Reads the flat task file, scores every candidate against a state evaluated once, and writes
 * one line of label probabilities per task. Nothing is generated: the model is asked for
 * distributions over candidate tokens, so the probabilities are native in the sense the harness
 * means.
 *
 * <p>Every answer space is built as a {@code Choice} over the task's own labels, whatever the
 * question type. The label set is what the harness scores against, and a Choice carries it exactly;
 * the ordinal reading of a score question is recovered by the harness from the label values.
 */
public final class JevBenchRunner {

  /** SemIf's system turn, verbatim from src/semif_phase1/core.py. */
  private static final String SEMIF_SYSTEM =
      "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
          + "Respond with only its uppercase letter, with no explanation or reasoning.";

  private JevBenchRunner() {}

  /**
   * Appends a JSON string literal the way Python's {@code json.dumps(ensure_ascii=False)} does.
   *
   * <p>The comparison is only worth anything if the bytes match, and the escaping is part of the
   * bytes: quote, backslash and the named control escapes, everything else below 0x20 as a unicode
   * escape, and no escaping of non-ASCII.
   */
  private static void appendJsonString(StringBuilder out, String value) {
    out.append('"');
    for (int index = 0; index < value.length(); index++) {
      char character = value.charAt(index);
      switch (character) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        case '\b' -> out.append("\\b");
        case '\f' -> out.append("\\f");
        default -> {
          if (character < 0x20) {
            out.append(String.format("\\u%04x", (int) character));
          } else {
            out.append(character);
          }
        }
      }
    }
    out.append('"');
  }

  /**
   * Scores one task with the labels in the order given, returning probabilities in that order.
   *
   * <p>Everything else is the shipped composition: evidence, rubric, optional instruction, then the
   * criterion, the lettered options and the answer cue.
   */
  private static double[] scoreOrder(
      PureJavaBackend backend,
      LetterLogitScorer scorer,
      String[] row,
      String id,
      List<String> order,
      String instruct) {
    Map<String, String> parsed = parseRubric(row[6].replace("\\n", "\n"), order);
    String block = LetterLogitScorer.renderCriteria(order, parsed);
    StringBuilder shared = new StringBuilder(row[4].replace("\\n", "\n"));
    if (!block.isEmpty()) {
      shared.append('\n').append(block);
    }
    if (!instruct.isEmpty()) {
      shared.append('\n').append(instruct);
    }
    int[] prompt =
        backend
            .tokenizer()
            .encode(
                shared
                    + "\n"
                    + row[5].replace("\\n", "\n")
                    + "\n"
                    + LetterLogitScorer.renderOptions(order));
    int[] letters = new int[order.size()];
    for (int slot = 0; slot < order.size(); slot++) {
      int[] encoded = backend.tokenizer().encode(" " + (char) ('A' + slot));
      letters[slot] = encoded[encoded.length - 1];
    }
    try (InferenceSession session = backend.openSession()) {
      float[] logits = backend.prefill(session, prompt, 0);
      return scorer.score(new Choice(id, order), logits, letters).probabilities();
    }
  }

  /**
   * Reads the prepared rubric block back into a per-label map.
   *
   * <p>The block is emitted by {@code prepare_tasks.py} as {@code - label: text} lines under an
   * {@code Options:} header, straight from the benchmark's own {@code question.criteria}. A label
   * whose text is just the label again carries nothing and is dropped, so an arm is not credited
   * with a rubric the benchmark did not actually provide.
   */
  private static Map<String, String> parseRubric(String rubric, List<String> labels) {
    Map<String, String> criteria = new LinkedHashMap<>();
    for (String line : rubric.split("\n")) {
      String trimmed = line.strip();
      if (!trimmed.startsWith("- ")) {
        continue;
      }
      int separator = trimmed.indexOf(": ");
      if (separator < 0) {
        continue;
      }
      String label = trimmed.substring(2, separator).strip();
      String text = trimmed.substring(separator + 2).strip();
      if (labels.contains(label) && !text.isBlank() && !text.equals(label)) {
        criteria.put(label, text);
      }
    }
    return criteria;
  }

  /**
   * The prompt the letter arm reads, in one of two orders.
   *
   * <p>Shipped order is state, question, rubric, then the lettered options. Options-first moves the
   * lettered options ahead of the question so that everything before the question is shared across
   * every question with the same answer space, leaving only the question's own words to be read per
   * decision. The answer cue stays last in both, because it is what the read-out position means.
   */
  private static String letterPrompt(
      String[] row,
      List<String> labels,
      boolean optionsFirst,
      boolean runtimePrompt,
      boolean runtimeCriteria,
      boolean sharedFirst,
      boolean rubricFirst,
      boolean shipped) {
    String prompt = row[3].replace("\\n", "\n");
    String rendered = LetterLogitScorer.renderOptions(labels);
    if (!optionsFirst
        && !runtimePrompt
        && !runtimeCriteria
        && !sharedFirst
        && !rubricFirst
        && !shipped) {
      return prompt + "\n" + rendered;
    }
    if (row.length < 7) {
      throw new IllegalArgumentException(
          "options-first needs a task file carrying the prompt's parts; re-run prepare_tasks.py");
    }
    String state = row[4].replace("\\n", "\n");
    String instructions = row[5].replace("\\n", "\n");
    String rubric = row[6].replace("\\n", "\n");
    String cue = "Answer:";
    if (runtimePrompt) {
      // Byte for byte what ModelJarDecisionRuntime.decide sent before an answer space could carry
      // a rubric: evidence, criterion, lettered options, answer cue.
      return state + "\n" + instructions + "\n" + rendered;
    }
    if (shipped) {
      // Composed exactly as ModelJarDecisionRuntime composes it: a shared prefix of evidence and
      // rubric, then the criterion, the letters and the cue.
      Map<String, String> parsed = parseRubric(rubric, labels);
      String block = LetterLogitScorer.renderCriteria(labels, parsed);
      String shared = block.isEmpty() ? state : state + "\n" + block;
      return shared + "\n" + instructions + "\n" + LetterLogitScorer.renderOptions(labels);
    }
    if (rubricFirst) {
      // Evidence, rubric, criterion, lettered options, cue. Shared through the rubric.
      String block = LetterLogitScorer.renderCriteria(labels, parseRubric(rubric, labels));
      StringBuilder text = new StringBuilder(state);
      if (!block.isEmpty()) {
        text.append('\n').append(block);
      }
      return text.append('\n').append(instructions).append('\n').append(rendered).toString();
    }
    if (sharedFirst) {
      // Evidence, rubric and lettered options, then the criterion, then the cue. Everything before
      // the criterion is identical for every question over this evidence and answer space.
      Map<String, String> parsed = parseRubric(rubric, labels);
      String block = LetterLogitScorer.renderCriteria(labels, parsed);
      String letters = LetterLogitScorer.renderOptions(labels);
      letters = letters.substring(0, letters.length() - cue.length()).stripTrailing();
      StringBuilder text = new StringBuilder(state);
      if (!block.isEmpty()) {
        text.append('\n').append(block);
      }
      return text.append('\n')
          .append(letters)
          .append('\n')
          .append(instructions)
          .append('\n')
          .append(cue)
          .toString();
    }
    if (runtimeCriteria) {
      // What it sends now. The rubric block is parsed back into a per-label map and handed to the
      // same renderer the runtime uses, so this arm measures the shipped code path and not a
      // hand-built lookalike.
      //
      // Two layouts, because which one a model reads better is not a thing to have an opinion
      // about. Inline puts each rule beside its letter; block keeps the benchmark's own shape, a
      // rubric list above a bare lettered list. MEASURED below rather than assumed.
      Map<String, String> parsed = parseRubric(rubric, labels);
      if ("block".equals(System.getProperty("decisions.rubricStyle", "inline"))) {
        StringBuilder text = new StringBuilder(state).append('\n').append(instructions);
        if (!parsed.isEmpty()) {
          text.append("\nOptions:");
          for (String label : labels) {
            String criterion = parsed.get(label);
            text.append("\n- ")
                .append(label)
                .append(": ")
                .append(criterion == null ? label : criterion);
          }
        }
        return text.append('\n').append(rendered).toString();
      }
      String block = LetterLogitScorer.renderCriteria(labels, parsed);
      StringBuilder text = new StringBuilder(state).append('\n').append(instructions);
      if (!block.isEmpty()) {
        text.append('\n').append(block);
      }
      return text.append('\n').append(LetterLogitScorer.renderOptions(labels)).toString();
    }
    String letteredOnly = rendered.substring(0, rendered.length() - cue.length()).stripTrailing();
    return state + "\n\n" + rubric + "\n" + letteredOnly + "\n" + instructions + "\n" + cue;
  }

  /** Arguments: model path, prepared TSV, output TSV, temperature. */
  public static void main(String[] args) throws IOException {
    Path model = Path.of(args[0]);
    Path tasks = Path.of(args[1]);
    Path out = Path.of(args[2]);
    double temperature = args.length > 3 ? Double.parseDouble(args[3]) : 1.0;

    List<String[]> rows = new ArrayList<>();
    try (BufferedReader reader = Files.newBufferedReader(tasks, StandardCharsets.UTF_8)) {
      String line;
      while ((line = reader.readLine()) != null) {
        if (!line.isBlank()) {
          rows.add(line.split("\t", -1));
        }
      }
    }
    System.out.printf("tasks=%d temperature=%.3f%n", rows.size(), temperature);

    long wall0 = System.nanoTime();
    int done = 0;
    int singleTokenLabels = 0;
    int multiTokenLabels = 0;
    int sharedProven = 0;
    long candidateTokens = 0;

    // Selectable so the two kernels can be run as arms of one comparison. Their answers are not
    // the same -- MEASURED 2026-09-24, 6e-2 mean absolute logit apart on this model -- so any claim
    // resting on a small accuracy difference has to be checked against what merely changing the
    // kernel does. That control is not possible if the kernel is hardcoded.
    boolean nativeKernel = !"java".equals(System.getProperty("decisions.kernel", "native"));
    try (PureJavaBackend backend =
            PureJavaBackend.load(
                model,
                nativeKernel
                    ? RustGgufBatchedMatrixKernel.openBundled()
                    : com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel.none());
        BufferedWriter writer = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {

      SharedPrefixCandidateEvaluator evaluator =
          new SharedPrefixCandidateEvaluator(backend, temperature);
      // The letter arm asks the model which letter comes next instead of scoring each option, so
      // one forward answers the whole question and the option text is read from the prompt.
      boolean letters = Boolean.parseBoolean(System.getProperty("decisions.letterLogits", "false"));
      // A decision's cost is the arithmetic of the tokens that follow the shared state, MEASURED
      // 2026-09-24 at 18.5 ms each and compute bound. Today those tokens are the question AND the
      // options, and the options are most of them and identical across every question with the
      // same answer space. Ahead of the question they are part of the shared prefix instead, which
      // measured 1.75x per decision -- and disagreed with the shipped order on 4 of 15 cases, which
      // is why it is an arm here and not a change to the prompt.
      boolean optionsFirst =
          Boolean.parseBoolean(System.getProperty("decisions.optionsFirst", "false"));
      if (optionsFirst && !letters) {
        throw new IllegalArgumentException("decisions.optionsFirst only applies to the letter arm");
      }
      // The prompt the shipped runtime can actually build. The prepared task prompt carries a
      // per-label rubric out of the benchmark's `question.criteria`, and AnswerSpace has exactly
      // two accessors -- question() and labels() -- so no caller of the published API can supply
      // one. Measuring with the rubric and shipping without it prices a product nobody can buy, so
      // this arm drops it and reports what the product does today.
      boolean runtimePrompt =
          Boolean.parseBoolean(System.getProperty("decisions.runtimePrompt", "false"));
      if (runtimePrompt && !letters) {
        throw new IllegalArgumentException(
            "decisions.runtimePrompt only applies to the letter arm");
      }
      if (runtimePrompt && optionsFirst) {
        throw new IllegalArgumentException("pick one prompt arm");
      }
      // The prompt the runtime builds once an answer space can carry a rubric. Same shape as the
      // runtime arm, with each lettered option followed by what it covers, which is the whole of
      // the difference between a score the product can reproduce and one it cannot.
      boolean runtimeCriteria =
          Boolean.parseBoolean(System.getProperty("decisions.runtimeCriteria", "false"));
      if (runtimeCriteria && (!letters || optionsFirst)) {
        throw new IllegalArgumentException("decisions.runtimeCriteria is a letter-arm prompt arm");
      }
      // Everything that does not vary between questions about one piece of evidence, placed before
      // the criterion so it can be prefilled once and resumed. For a batch of questions over one
      // answer space that is the evidence, the rubric and the lettered options; what is left per
      // question is the criterion and the answer cue.
      //
      // This is the performance question, because a decision's cost is the arithmetic of the tokens
      // after the shared prefix and MEASURED 2026-09-24 that is 18.5 ms each. Carrying a rubric is
      // worth 14 points of Intelligence and costs 40-odd tokens on every question; in the prefix it
      // costs them once. Whether the criterion still works at the end is what this measures.
      boolean sharedFirst =
          Boolean.parseBoolean(System.getProperty("decisions.sharedFirst", "false"));
      if (sharedFirst && (!letters || optionsFirst || runtimePrompt)) {
        throw new IllegalArgumentException("decisions.sharedFirst is its own prompt arm");
      }
      // Half of sharedFirst. Moving the rubric AND the lettered options ahead of the criterion cost
      // 6.5 points, but not evenly: fact fell from 1.0000 to 0.3333 while routing and ordinal rose
      // to 1.0000. Two things moved at once, so this moves only the rubric and leaves the letters
      // after the criterion, on the guess that what hurt was declaring the letter-to-label mapping
      // before the question it answers. The rubric is the larger share of the tokens anyway.
      boolean rubricFirst =
          Boolean.parseBoolean(System.getProperty("decisions.rubricFirst", "false"));
      if (rubricFirst && (!letters || optionsFirst || runtimePrompt || sharedFirst)) {
        throw new IllegalArgumentException("decisions.rubricFirst is its own prompt arm");
      }
      // The shipped composition, built by the runtime's own two halves rather than by this file, so
      // that the arm which decided the layout and the code that ships it are checked against each
      // other rather than merely believed to agree.
      // The runtime reads its answer off the final position of one prefill of the whole suffix,
      // rather than prefilling all but the last token and stepping that one alone. The step removed
      // is a single token read through all 2.55 GiB of weights, 67 ms and bandwidth bound; as one
      // more row of an already compute-bound batch it is 18.5 ms. It is also a different kernel
      // path, so it moves probabilities and has to be scored rather than assumed harmless.
      boolean foldedReadout =
          Boolean.parseBoolean(System.getProperty("decisions.foldedReadout", "false"));
      // A decision is 98.5% the prefill of its own suffix -- MEASURED 2026-09-24, resumption 4.6 ms
      // and scoring 0.2 ms out of 391 ms -- so the only lever is tokens, and for a short question
      // 11 of the 18 are the lettered list rather than the question.
      //
      // The list exists so a label of any length can be read from one forward pass. When every
      // label is already a single token the indirection looks like pure overhead, so this arm drops
      // it and reads the labels' own logits at the answer position. It is kept, and off, because it
      // measured 10 noise floors worse: Intelligence 88.9 to 78.6 over 120 items, losing on exactly
      // the families that qualified. A lettered multiple choice constrains a model in a way that an
      // answer cue followed by a free token does not, and those 11 tokens are load-bearing.
      boolean labelTokens =
          Boolean.parseBoolean(System.getProperty("decisions.labelTokens", "false"));
      // Three ways to spend, or not spend, latency on accuracy. Measured, not assumed.
      //
      // reverseLabels: a lettered multiple choice has a position bias, and the label order comes
      // from the task rather than from anything anyone chose. Free to change.
      //
      // instruct: one sentence saying what shape the answer takes. It goes in the shared prefix, so
      // it is free per question however long it is.
      //
      // averageOrders: score both label orders and average, which removes the position bias instead
      // of guessing which way it points. Costs a second pass over the question's own tokens, so it
      // roughly doubles the per-question cost.
      boolean reverseLabels =
          Boolean.parseBoolean(System.getProperty("decisions.reverseLabels", "false"));
      boolean averageOrders =
          Boolean.parseBoolean(System.getProperty("decisions.averageOrders", "false"));
      String instruct = System.getProperty("decisions.instruct", "");
      // The prompt SemIf sends, transcribed from its source rather than described from its README.
      // SemIf is the second-placed system on this benchmark and runs the same frozen 4B model, so
      // the interesting comparison is its prompt against ours on one host with one readout.
      //
      // Read from src/semif_phase1/core.py and direct.py: a chat template with a system turn, a
      // user turn carrying one JSON object of evidence, criterion and options, each option a letter
      // and a description with the label name dropped, and the answer read from the single-token
      // letters at the final position.
      boolean semifPrompt =
          Boolean.parseBoolean(System.getProperty("decisions.semifPrompt", "false"));
      // SemIf's chat template and system turn, our compact body and layout.
      //
      // MEASURED: their whole prompt beats ours on our own model, Intelligence 88.0 to 90.0 and
      // Calibration 73.0 to 82.1, and costs 2.2x the latency because a JSON payload inside a chat
      // template is a lot more tokens. But the template markers and the system turn are the same
      // for
      // every question about one piece of evidence, so they belong in the shared prefix and cost
      // nothing per question. This arm keeps them and drops the JSON.
      boolean chatTemplate =
          Boolean.parseBoolean(System.getProperty("decisions.chatTemplate", "false"));
      boolean shipped = Boolean.parseBoolean(System.getProperty("decisions.shipped", "false"));
      if (shipped && (!letters || optionsFirst || runtimePrompt || sharedFirst || rubricFirst)) {
        throw new IllegalArgumentException("decisions.shipped is its own prompt arm");
      }
      LetterLogitScorer letterScorer = new LetterLogitScorer(temperature);
      System.out.printf("mode=%s%n", letters ? "letter-logits" : "candidate-scoring");

      // Warmup, and it is not a nicety. Measured on an A40 host: the same item scored repeatedly
      // in one JVM gives 0.9110, 0.9399, then 0.9316622781200229 and that value thereafter,
      // bit-identical and reproducible in a second JVM. Without this the first two items of every
      // run are scored on partly-interpreted code, are not reproducible between JVMs, and are also
      // the slowest, which moves the p50 and p95 a speed metric is built from.
      int warmupItems = Integer.getInteger("decisions.warmupItems", 3);
      List<String[]> schedule = new ArrayList<>();
      for (int i = 0; i < warmupItems && !rows.isEmpty(); i++) {
        schedule.add(rows.get(0));
      }
      int warmupRemaining = schedule.size();
      schedule.addAll(rows);
      if (warmupRemaining > 0) {
        System.out.printf("warmup=%d items (discarded)%n", warmupRemaining);
      }

      for (String[] row : schedule) {
        String id = row[0];
        List<String> labels = List.of(row[2].split("\\|", -1));
        String prompt = row[3].replace("\\n", "\n");

        int[] state = backend.tokenizer().encode(prompt);
        List<int[]> candidates = new ArrayList<>(labels.size());
        for (String label : labels) {
          candidates.add(backend.tokenizer().encode(" " + label));
        }

        double[] p;
        double latency;
        if (chatTemplate) {
          Map<String, String> parsed = parseRubric(row[6].replace("\\n", "\n"), labels);
          String block = LetterLogitScorer.renderCriteria(labels, parsed);
          StringBuilder body = new StringBuilder(row[4].replace("\\n", "\n"));
          if (!block.isEmpty()) {
            body.append('\n').append(block);
          }
          body.append('\n')
              .append(row[5].replace("\\n", "\n"))
              .append('\n')
              .append(LetterLogitScorer.renderOptions(labels));
          String rendered =
              ChatTemplate.CHATML_NO_THINK
                  .render(
                      List.of(ChatMessage.system(SEMIF_SYSTEM), ChatMessage.user(body.toString())))
                  .text();
          int[] templateTokens = backend.tokenizer().encode(rendered);
          int[] templateLetters = new int[labels.size()];
          for (int slot = 0; slot < labels.size(); slot++) {
            // Bare letter: the assistant turn has just opened, so there is no leading space.
            int[] encoded = backend.tokenizer().encode(String.valueOf((char) ('A' + slot)));
            if (encoded.length != 1) {
              throw new IllegalStateException("answer slot is not one token");
            }
            templateLetters[slot] = encoded[0];
          }
          long startedTemplate = System.nanoTime();
          try (InferenceSession session = backend.openSession()) {
            float[] logits = backend.prefill(session, templateTokens, 0);
            p = letterScorer.score(new Choice(id, labels), logits, templateLetters).probabilities();
          }
          latency = (System.nanoTime() - startedTemplate) / 1e9;
          candidateTokens += 1;
        } else if (semifPrompt) {
          Map<String, String> parsed = parseRubric(row[6].replace("\\n", "\n"), labels);
          StringBuilder payload = new StringBuilder("{\"evidence\": ");
          appendJsonString(payload, row[4].replace("\\n", "\n"));
          payload.append(", \"criterion\": ");
          appendJsonString(payload, row[5].replace("\\n", "\n"));
          payload.append(", \"options\": [");
          for (int slot = 0; slot < labels.size(); slot++) {
            if (slot > 0) {
              payload.append(", ");
            }
            payload
                .append("{\"letter\": \"")
                .append((char) ('A' + slot))
                .append("\", \"description\": ");
            // SemIf sends only the description; the label name never reaches the model.
            appendJsonString(payload, parsed.getOrDefault(labels.get(slot), labels.get(slot)));
            payload.append('}');
          }
          payload.append("]}");
          String rendered =
              ChatTemplate.CHATML_NO_THINK
                  .render(
                      List.of(
                          ChatMessage.system(SEMIF_SYSTEM), ChatMessage.user(payload.toString())))
                  .text();
          int[] semifTokens = backend.tokenizer().encode(rendered);
          int[] semifLetters = new int[labels.size()];
          for (int slot = 0; slot < labels.size(); slot++) {
            // SemIf requires each bare uppercase letter to be one exact round-trip token.
            int[] encoded = backend.tokenizer().encode(String.valueOf((char) ('A' + slot)));
            if (encoded.length != 1) {
              throw new IllegalStateException(
                  "answer slot is not one token: " + (char) ('A' + slot));
            }
            semifLetters[slot] = encoded[0];
          }
          long startedSemif = System.nanoTime();
          try (InferenceSession session = backend.openSession()) {
            float[] logits = backend.prefill(session, semifTokens, 0);
            p = letterScorer.score(new Choice(id, labels), logits, semifLetters).probabilities();
          }
          latency = (System.nanoTime() - startedSemif) / 1e9;
          candidateTokens += 1;
        } else if (reverseLabels || averageOrders || !instruct.isEmpty()) {
          List<String> forward = labels;
          List<String> reversed = new ArrayList<>(labels);
          java.util.Collections.reverse(reversed);
          List<String> primary = reverseLabels ? reversed : forward;
          long startedOrder = System.nanoTime();
          double[] firstScores = scoreOrder(backend, letterScorer, row, id, primary, instruct);
          double[] secondScores = null;
          List<String> secondary = reverseLabels ? forward : reversed;
          if (averageOrders) {
            secondScores = scoreOrder(backend, letterScorer, row, id, secondary, instruct);
          }
          latency = (System.nanoTime() - startedOrder) / 1e9;
          // Realign onto the task's own label order, whatever order each pass used.
          p = new double[labels.size()];
          for (int slot = 0; slot < labels.size(); slot++) {
            String label = labels.get(slot);
            double value = firstScores[primary.indexOf(label)];
            if (secondScores != null) {
              value = 0.5 * (value + secondScores[secondary.indexOf(label)]);
            }
            p[slot] = value;
          }
          candidateTokens += averageOrders ? 2 : 1;
        } else if (labelTokens) {
          // Evidence, rubric, criterion, answer cue. No lettered list at all. A space whose labels
          // are not single tokens cannot be read this way and takes the lettered path, and both are
          // counted, so the result says how much of a cohort the shortcut even applies to.
          Map<String, String> parsed = parseRubric(row[6].replace("\\n", "\n"), labels);
          String block = LetterLogitScorer.renderCriteria(labels, parsed);
          String evidenceText = row[4].replace("\\n", "\n");
          String criterionText = row[5].replace("\\n", "\n");
          int[] readout = new int[labels.size()];
          boolean single = true;
          for (int slot = 0; slot < labels.size(); slot++) {
            int[] encoded = backend.tokenizer().encode(" " + labels.get(slot));
            single &= encoded.length == 1;
            readout[slot] = encoded[encoded.length - 1];
          }
          int[] readoutPrompt;
          if (single) {
            singleTokenLabels++;
            readoutPrompt =
                backend
                    .tokenizer()
                    .encode(
                        (block.isEmpty() ? evidenceText : evidenceText + "\n" + block)
                            + "\n"
                            + criterionText
                            + "\nAnswer:");
          } else {
            multiTokenLabels++;
            readoutPrompt =
                backend
                    .tokenizer()
                    .encode(letterPrompt(row, labels, false, false, false, false, false, true));
            for (int slot = 0; slot < labels.size(); slot++) {
              int[] encoded = backend.tokenizer().encode(" " + (char) ('A' + slot));
              readout[slot] = encoded[encoded.length - 1];
            }
          }
          long started = System.nanoTime();
          try (InferenceSession session = backend.openSession()) {
            float[] logits = backend.prefill(session, readoutPrompt, 0);
            p = letterScorer.score(new Choice(id, labels), logits, readout).probabilities();
          }
          latency = (System.nanoTime() - started) / 1e9;
          candidateTokens += 1;
        } else if (letters) {
          Choice space = new Choice(id, labels);
          int[] lettered =
              backend
                  .tokenizer()
                  .encode(
                      letterPrompt(
                          row,
                          labels,
                          optionsFirst,
                          runtimePrompt,
                          runtimeCriteria,
                          sharedFirst,
                          rubricFirst,
                          shipped));
          int[] letterTokens = new int[labels.size()];
          for (int i = 0; i < labels.size(); i++) {
            int[] encoded = backend.tokenizer().encode(" " + (char) ('A' + i));
            letterTokens[i] = encoded[encoded.length - 1];
          }
          long t0 = System.nanoTime();
          try (InferenceSession session = backend.openSession()) {
            float[] logits;
            if (foldedReadout) {
              logits = backend.prefill(session, lettered, 0);
            } else {
              int last = lettered.length - 1;
              if (last > 0) {
                int[] head = new int[last];
                System.arraycopy(lettered, 0, head, 0, last);
                backend.prefill(session, head, 0);
              }
              logits = backend.forward(session, lettered[last], last);
            }
            p = letterScorer.score(space, logits, letterTokens).probabilities();
          }
          latency = (System.nanoTime() - t0) / 1e9;
          candidateTokens += 1; // one forward, whatever the option count
        } else {
          long t0 = System.nanoTime();
          CandidateBatch batch = evaluator.evaluateBatch(state, new Choice(id, labels), candidates);
          latency = (System.nanoTime() - t0) / 1e9;
          candidateTokens += batch.candidateTokensEvaluated();
          sharedProven += batch.sharedPrefixProven() ? 1 : 0;
          p = batch.verdict().probabilities();
        }

        if (warmupRemaining > 0) {
          // Discarded: no row written, and the tallies restart so warmup cannot leak into them.
          warmupRemaining--;
          candidateTokens = 0;
          sharedProven = 0;
          if (warmupRemaining == 0) {
            wall0 = System.nanoTime();
          }
          continue;
        }

        StringBuilder sb = new StringBuilder(id).append('\t').append(latency);
        for (int i = 0; i < labels.size(); i++) {
          sb.append('\t').append(labels.get(i)).append('=').append(p[i]);
        }
        writer.write(sb.toString());
        writer.newLine();

        if (++done % 10 == 0) {
          writer.flush();
          System.out.printf(
              "  %d/%d  %.1f s%n", done, rows.size(), (System.nanoTime() - wall0) / 1e9);
        }
      }
    }
    System.out.printf(
        "done: %d tasks, wall %.1f s, candidate-token forwards %d, prefix sharing proven on %d/%d,"
            + " label-token readout on %d, lettered fallback on %d%n",
        done,
        (System.nanoTime() - wall0) / 1e9,
        candidateTokens,
        sharedProven,
        done,
        singleTokenLabels,
        multiTokenLabels);
  }
}
