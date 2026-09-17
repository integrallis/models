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
package com.integrallis.models.backend.purejava;

import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.RepetitionLoopDetection;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.GenerationLoop;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Runs the repetition-loop detector on real greedy generations from locally available fixtures.
 *
 * <p>Every (model, prompt) pair is generated once with detection off, which is the baseline, and
 * once per detector configuration. The report records each configuration beside its numbers, the
 * stop reason, where generation stopped, whether the detector-on text is an exact prefix of the
 * baseline (detection must not perturb what was generated before it fired), and the tail of the
 * text at the stop so a reader can judge whether a trigger was a true loop.
 *
 * <p>Arguments: {@code maxTokens}, {@code contextLength}, report path, then one or more GGUF paths.
 */
public final class GenerationSafetyLoopExperiment {

  /** Prompts written for this experiment; they are not a published dataset. */
  private static final List<String> PROMPTS =
      List.of(
          "The quick brown fox",
          "Continue_only_the_exact_sequence_without_explanation:_1_2_3_4_1_2_3_4_1_2_3_4_1_2_3_4_",
          "List the numbers from 1 to 50, one per line:\n1\n2\n",
          "Write a short poem about the sea.\n",
          "Repeat after me: I am a robot. I am a robot. I am a robot.",
          "def fibonacci(n):");

  private static final Map<String, RepetitionLoopDetection> CONFIGURATIONS = configurations();

  private GenerationSafetyLoopExperiment() {}

  public static void main(String[] args) throws Exception {
    if (args.length < 4) {
      throw new IllegalArgumentException(
          "usage: maxTokens contextLength reportPath model.gguf [model.gguf ...]");
    }
    int maxTokens = Integer.parseInt(args[0]);
    int contextLength = Integer.parseInt(args[1]);
    Path report = Path.of(args[2]);
    List<Path> models = Arrays.stream(args).skip(3).map(Path::of).toList();
    System.setProperty(
        PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(contextLength));

    StringBuilder out = new StringBuilder();
    out.append("# Repetition-loop detector on greedy fixture outputs\n\n");
    out.append("- java: ").append(System.getProperty("java.version")).append('\n');
    out.append("- os.arch: ").append(System.getProperty("os.arch")).append('\n');
    out.append("- maxTokens: ").append(maxTokens).append('\n');
    out.append("- contextLength: ").append(contextLength).append('\n');
    out.append(
        "- sampling: temperature=0 (greedy), repetitionPenalty=1.0, minP=0, no stop sequences\n");
    out.append("- detector configurations (maxSpan, minRepeats, minLoopTokens):\n");
    CONFIGURATIONS.forEach(
        (name, config) ->
            out.append("  - ")
                .append(name)
                .append(": ")
                .append(
                    config.enabled()
                        ? "("
                            + config.maxSpan()
                            + ", "
                            + config.minRepeats()
                            + ", "
                            + config.minLoopTokens()
                            + ")"
                        : "disabled (baseline)")
                .append('\n'));
    out.append('\n');

    List<String> rows = new ArrayList<>();
    for (Path model : models) {
      String sha256 = sha256(model);
      out.append("- model `")
          .append(model.getFileName())
          .append("` sha256 `")
          .append(sha256)
          .append("`\n");
      try (PureJavaBackend backend = PureJavaBackend.load(model)) {
        out.append("  - resolved end-of-generation ids: ")
            .append(Arrays.toString(backend.tokenizer().endOfGenerationTokenIds()))
            .append('\n');
        for (int promptIndex = 0; promptIndex < PROMPTS.size(); promptIndex++) {
          String prompt = PROMPTS.get(promptIndex);
          Run baseline = null;
          for (Map.Entry<String, RepetitionLoopDetection> entry : CONFIGURATIONS.entrySet()) {
            SamplingOptions options =
                SamplingOptions.builder()
                    .temperature(0.0f)
                    .repetitionPenalty(1.0f)
                    .maxTokens(maxTokens)
                    .repetitionLoopDetection(entry.getValue())
                    .build();
            Run run = generate(backend, prompt, options);
            if (baseline == null) {
              baseline = run;
            }
            boolean prefix = baseline.text().startsWith(run.text());
            rows.add(
                String.join(
                    " | ",
                    "",
                    model.getFileName().toString(),
                    "P" + promptIndex,
                    entry.getKey(),
                    run.reason().name(),
                    Integer.toString(run.completionTokens()),
                    Long.toString(run.loopStops()),
                    Boolean.toString(prefix),
                    "`" + tail(run.text()) + "`",
                    ""));
            System.out.println(rows.getLast());
          }
        }
      }
    }
    out.append("\nPrompts (written for this experiment, not a published dataset):\n\n");
    for (int index = 0; index < PROMPTS.size(); index++) {
      out.append("- P")
          .append(index)
          .append(": `")
          .append(escape(PROMPTS.get(index)))
          .append("`\n");
    }
    out.append(
        "\n| model | prompt | detector | stop reason | completion tokens | loop stops (loop"
            + " counter) | prefix of baseline | last 60 chars at stop |\n");
    out.append("|---|---|---|---|---|---|---|---|\n");
    rows.forEach(row -> out.append(row.strip()).append('\n'));
    Files.createDirectories(report.toAbsolutePath().getParent());
    Files.writeString(report, out.toString());
    System.out.println("wrote " + report.toAbsolutePath());
  }

  private static Map<String, RepetitionLoopDetection> configurations() {
    Map<String, RepetitionLoopDetection> configurations = new LinkedHashMap<>();
    configurations.put("off", RepetitionLoopDetection.disabled());
    configurations.put("A", new RepetitionLoopDetection(32, 4, 16));
    configurations.put("B", new RepetitionLoopDetection(64, 3, 32));
    configurations.put("C", new RepetitionLoopDetection(16, 10, 64));
    return configurations;
  }

  private static Run generate(PureJavaBackend backend, String prompt, SamplingOptions options) {
    GenerationLoop loop = new GenerationLoop(backend);
    StringBuilder text = new StringBuilder();
    StopReason[] reason = new StopReason[1];
    GenerationUsage[] usage = new GenerationUsage[1];
    Throwable[] failure = new Throwable[1];
    loop.generate(
        prompt,
        options,
        new TokenStream() {
          @Override
          public void onToken(String token) {
            text.append(token);
          }

          @Override
          public void onComplete() {}

          @Override
          public void onComplete(GenerationUsage completedUsage, StopReason stopReason) {
            usage[0] = completedUsage;
            reason[0] = stopReason;
          }

          @Override
          public void onError(Throwable error) {
            failure[0] = error;
          }
        });
    if (failure[0] != null) {
      throw new IllegalStateException("generation failed", failure[0]);
    }
    return new Run(
        text.toString(), reason[0], usage[0].completionTokens(), loop.repetitionLoopStops());
  }

  private static String tail(String text) {
    String tail = text.length() <= 60 ? text : text.substring(text.length() - 60);
    return escape(tail);
  }

  private static String escape(String text) {
    return text.replace("\n", "\\n").replace("|", "\\|").replace("`", "'");
  }

  private static String sha256(Path path) throws IOException, NoSuchAlgorithmException {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (InputStream input = new DigestInputStream(Files.newInputStream(path), digest)) {
      input.transferTo(java.io.OutputStream.nullOutputStream());
    }
    return HexFormat.of().formatHex(digest.digest()).toLowerCase(Locale.ROOT);
  }

  private record Run(String text, StopReason reason, int completionTokens, long loopStops) {}
}
