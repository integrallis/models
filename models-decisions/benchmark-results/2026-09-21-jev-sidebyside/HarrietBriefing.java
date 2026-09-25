package demo;

import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Verdict;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;

/**
 * The Integrallis arm of the side-by-side, rebuilt for what Harriet actually is.
 *
 * <p>The 2026-09-21 recording ran {@code BriefingDemo}: Granite 4.1 3B plus an 82 KB squad2-fitted
 * Noul head. Neither part ships any more. Harriet is frozen Qwen3.5-4B read through letter logits,
 * with no trained head at all, so the old arm cannot be re-run and the old video advertises a
 * product that does not exist. The three misses it explains -- negation read as absence, because a
 * span-presence head was asked an entailment question -- are a property of a component that is gone.
 *
 * <p>The task is unchanged so the comparison survives: same 404-token master services agreement,
 * same ten questions, same answerability framing Jev is asked ({@code "Is the question answerable
 * from the text above?"}), same gold.
 *
 * <p><b>Prefill is measured, not subtracted.</b> The old arm reported a one-off prefill next to a
 * marginal-per-question, and the honest way to get both is to establish the shared prefix with its
 * own timed call rather than to difference two averages. Harriet prefills the evidence on the first
 * decision against a given state and resumes it afterwards, so the warmup decision below IS the
 * prefill, and the ten timed calls are all marginal. A number produced by subtracting one
 * measurement from another has already caused a retraction on this project.
 */
public final class HarrietBriefing {
  public static void main(String... args) throws Exception {
    Path documentPath = Path.of(args[0]);
    Path questionPath = Path.of(args[1]);
    Path out = Path.of(args[2]);

    String document = Files.readString(documentPath, StandardCharsets.UTF_8).strip();
    List<String> questions = new ArrayList<>();
    for (String line : Files.readAllLines(questionPath, StandardCharsets.UTF_8)) {
      if (!line.isBlank()) {
        questions.add(line.strip());
      }
    }

    System.out.println();
    System.out.println("  Harriet — one document, prefilled once, many questions");
    System.out.printf(
        "  frozen Qwen3.5-4B, letter-logit readout, %d questions, document never leaves the box%n",
        questions.size());
    System.out.println();

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      // Warm the JIT on a DIFFERENT document first. Without this the timed prefill below is partly
      // the compiler: unwarmed Panama vector code takes another path with another accumulation
      // order, and the first measured run of this demo reported a 13.106 s prefill against about
      // 9.7 s of arithmetic -- roughly 3.4 s of JIT billed to the model. This project has a whole
      // section of NOTES.md about walking into exactly that, and this is the second time.
      //
      // A different document matters: warming on the real one would establish its prefix and the
      // "prefill" that follows would be a resume.
      String warmupDocument = document.substring(0, Math.min(document.length(), 200)) + "\n(warmup)";
      for (int round = 0; round < 3; round++) {
        runtime.decide(answerability(questions.get(round % questions.size())), warmupDocument);
      }

      // The prefill. First decision against THIS state pays for the document; everything after
      // resumes it. Timed on its own so the reported figure is a measurement and not a subtraction.
      AnswerSpace warm = answerability(questions.get(0));
      long started = System.nanoTime();
      runtime.decide(warm, document);
      double prefillSeconds = (System.nanoTime() - started) / 1e9;

      List<Double> marginal = new ArrayList<>();
      List<Double> probabilities = new ArrayList<>();
      long allStarted = System.nanoTime();
      for (int index = 0; index < questions.size(); index++) {
        String question = questions.get(index);
        long questionStarted = System.nanoTime();
        Verdict verdict = runtime.decide(answerability(question), document);
        double seconds = (System.nanoTime() - questionStarted) / 1e9;
        // probabilityOfTrue rather than probabilityOf("yes"): Noul owns its label names.
        double yes = verdict.probabilityOfTrue();
        marginal.add(seconds);
        probabilities.add(yes);
        String shortened = question.length() <= 46 ? question : question.substring(0, 43) + "...";
        System.out.printf(
            "  Q%-2d %-46s %s  p=%.3f  %6.3f s%n",
            index + 1, shortened, yes >= 0.5 ? "YES" : "NO ", yes, seconds);
      }
      double total = (System.nanoTime() - allStarted) / 1e9;

      // Ask the backend what it is. The first version of this file guessed from
      // runtime.toString() and reported "pure-java" for a run that was demonstrably on rust-ffm --
      // the stack trace from a failed load names RustFfmBackend. This demo's README already records
      // a published error caused by a kernel misreport, which is why it prints the kernel at all;
      // guessing it twice in the same file is not acceptable.
      String kernel = runtime.backend().name();
      String abi = runtime.backend().diagnostics().environment()
          .getOrDefault("native-kernel-abi", "n/a");
      int threads = Integer.getInteger("models.native.kernels.threads", Runtime.getRuntime().availableProcessors());

      double mean = marginal.stream().mapToDouble(Double::doubleValue).average().orElse(0);
      System.out.println();
      System.out.printf("  document sent to a server    %d%n", 0);
      System.out.printf("  input tokens billed          %d%n", 0);
      System.out.printf("  one-off prefill              %7.3f s%n", prefillSeconds);
      System.out.printf("  mean per question            %7.3f s%n", mean);
      System.out.printf("  total for %-2d questions       %7.3f s%n", questions.size(), total);
      System.out.println();
      System.out.printf(
          "  kernel %s (native abi %s), %d worker threads, %s%n",
          runtime.backend().name(), abi, threads, System.getProperty("os.arch"));
      System.out.println();

      StringBuilder json = new StringBuilder();
      json.append("{\"system\":\"integrallis-harriet\",\"kernel\":\"").append(kernel).append('"');
      json.append(",\"model\":\"qwen3.5-4b-q4_k_m\",\"readout\":\"letter-logits\",\"head\":null");
      json.append(",\"questions\":").append(questions.size());
      json.append(",\"prefill_s\":").append(round(prefillSeconds));
      json.append(",\"marginal_s\":[");
      for (int index = 0; index < marginal.size(); index++) {
        json.append(index == 0 ? "" : ",").append(round(marginal.get(index)));
      }
      json.append("],\"probabilities\":[");
      for (int index = 0; index < probabilities.size(); index++) {
        json.append(index == 0 ? "" : ",").append(round(probabilities.get(index)));
      }
      json.append("],\"total_s\":").append(round(total));
      json.append(",\"prefix_shared\":true");
      json.append(",\"native_kernel_abi\":\"").append(abi).append('"');
      json.append(",\"worker_threads\":").append(threads);
      json.append(",\"host\":\"").append(System.getProperty("harriet.demo.host", "unstated")).append("\"}");
      Files.writeString(out, json.toString(), StandardCharsets.UTF_8);
      System.out.printf("  timings written to %s%n%n", out);
    }
  }

  /**
   * The same question Jev is asked, so the two arms answer one task.
   *
   * <p>Harriet reads the labels from the prompt, so the rubric is what tells it that this is about
   * the text rather than about the world -- the difference the old head could not learn.
   */
  private static AnswerSpace answerability(String question) {
    return new Noul(
        "Is the question answerable from the text above? Question: " + question,
        "The text above states the answer to the question",
        "The text above does not state the answer to the question");
  }

  private static double round(double value) {
    return Math.round(value * 10000.0) / 10000.0;
  }
}
