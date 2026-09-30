package demo;

import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.Tokenizer;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.LetterLogitScorer;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Verdict;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * What a warm decision costs, with and without a rubric, once the rubric is shared.
 *
 * <p>A decision's cost is the arithmetic of the tokens after the shared prefix -- MEASURED
 * 2026-09-24 at 18.5 ms each, compute bound and already at this machine's ceiling -- plus one
 * bandwidth-bound step to read the answer out. So a rubric is worth whichever side of the split it
 * lands on. Carrying one is worth 14 points of Intelligence; carrying it per question would cost its
 * tokens on every question, and carrying it in the shared prefix costs them once.
 *
 * <p>Reported per question, warm, over one piece of evidence: the token split, the measured cost,
 * and what the same rubric would have cost had it stayed after the criterion.
 */
public final class Release {
  private static final List<String> CRITERIA =
      List.of(
          "Is a monthly fee stated?",
          "Is an uptime guarantee stated?",
          "May Customer Data be used to train models?",
          "Is a liability cap stated?",
          "Does the liability cap apply to data breaches?",
          "Can the agreement be terminated for convenience?",
          "Is an early termination fee stated?",
          "Is the governing law stated?",
          "Is arbitration required?",
          "Is a renewal term stated?",
          "Is a confidentiality clause present?",
          "Is a service credit stated?",
          "Is an audit right granted?",
          "Is subprocessing permitted?",
          "Is a notice period stated?",
          "Is indemnification addressed?",
          "Is force majeure addressed?",
          "Is assignment restricted?",
          "Is a warranty disclaimed?",
          "Is a data retention period stated?");

  private static final String WHEN_TRUE =
      "The contract states it explicitly, in terms a reader could quote, and nothing elsewhere"
          + " withdraws or qualifies it";
  private static final String WHEN_FALSE =
      "The contract is silent on it, or states it only as an intention, or a later clause removes"
          + " it";

  public static void main(String... args) throws Exception {
    String evidence = Files.readString(Path.of(args[0])).strip();

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      InferenceBackend backend = runtime.backend();
      Tokenizer tokenizer = backend.tokenizer();

      List<AnswerSpace> bare = new ArrayList<>();
      List<AnswerSpace> withRubric = new ArrayList<>();
      for (String criterion : CRITERIA) {
        bare.add(new Noul(criterion));
        withRubric.add(new Noul(criterion, WHEN_TRUE, WHEN_FALSE));
      }

      // Warm both shapes. An unwarmed timing measures the compiler.
      for (int round = 0; round < 3; round++) {
        for (int index = 0; index < 4; index++) {
          runtime.decide(bare.get(index), evidence);
          runtime.decide(withRubric.get(index), evidence);
        }
      }

      System.out.println();
      System.out.printf(
          "  %-14s %-8s %-8s %-11s %-11s %-11s%n",
          "rubric", "shared", "suffix", "first s", "later s", "per q s");
      report(runtime, tokenizer, "none", bare, evidence);
      report(runtime, tokenizer, "declared", withRubric, evidence);

      int rubricTokens =
          tokenizer.encode(shared(withRubric.get(0), evidence)).length
              - tokenizer.encode(evidence).length;
      System.out.printf(
          "%n  the rubric is %d tokens. Shared, it is read once. After the criterion it would be"
              + " read on every question, at 18.5 ms each: %.3f s per question.%n%n",
          rubricTokens, rubricTokens * 0.0185);
    }
  }

  /**
   * The shared prefix, composed the way the runtime composes it from the same public renderers.
   *
   * <p>Duplicated here rather than reaching into the runtime's package-private helpers, because a
   * harness is not a reason to widen an API. The composition is asserted against the runtime's own
   * by ModelJarDecisionRuntimePromptTest, and the arm that decided this layout is checked against
   * the shipped code by JevBenchRunner's shipped arm.
   */
  private static String shared(AnswerSpace space, String evidence) {
    String criteria = LetterLogitScorer.renderCriteria(space.labels(), space.criteria());
    return criteria.isEmpty() ? evidence : evidence + "\n" + criteria;
  }

  private static String suffix(AnswerSpace space) {
    return "\n" + space.question() + "\n" + LetterLogitScorer.renderOptions(space.labels());
  }

  private static void report(
      ModelJarDecisionRuntime runtime,
      Tokenizer tokenizer,
      String label,
      List<AnswerSpace> spaces,
      String evidence)
      throws Exception {
    AnswerSpace first = spaces.get(0);
    int shared = tokenizer.encode(shared(first, evidence)).length;
    int whole = tokenizer.encode(shared(first, evidence) + suffix(first)).length;

    // A cold first question pays for the shared prefix; every one after it resumes.
    runtime.decide(new Noul("Unrelated, to displace the cached prefix"), evidence + " ");
    long started = System.nanoTime();
    runtime.decide(first, evidence);
    double firstSeconds = (System.nanoTime() - started) / 1e9;

    started = System.nanoTime();
    List<Verdict> verdicts = new ArrayList<>();
    for (int index = 1; index < spaces.size(); index++) {
      verdicts.add(runtime.decide(spaces.get(index), evidence));
    }
    double later = (System.nanoTime() - started) / 1e9;

    System.out.printf(
        "  %-14s %-8d %-8d %-11.3f %-11.3f %-11.3f%n",
        label, shared, whole - shared, firstSeconds, later, later / verdicts.size());
  }
}
