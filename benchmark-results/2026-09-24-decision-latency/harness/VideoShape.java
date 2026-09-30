package demo;

import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Score;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * What the demo videos would actually show, in the two shapes they could be scripted in.
 *
 * <p>The release's wins are all on the shared prefix: the evidence and the rubric are read once and
 * every question after that costs only its own tokens, MEASURED at 18.5 ms each. A demo that gives
 * every case its own document has no shared prefix, so it shows none of that -- each case pays for
 * its own evidence from cold, and the only gain left is the readout step folded into the prefill.
 *
 * <p>So both shapes are timed here. One case per document is what the recorded videos do. One
 * document, many questions is what a batch of decisions actually looks like and what the release was
 * built for.
 */
public final class VideoShape {
  public static void main(String... args) throws Exception {
    List<String[]> cases = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of(args[0]))) {
      String[] fields = line.split("\t", -1);
      if (fields.length == 5 && !fields[0].equals("id")) {
        cases.add(fields);
      }
    }
    String contract = Files.readString(Path.of(args[1])).strip();

    try (ModelJarDecisionRuntime runtime = Harriet.open()) {
      for (int round = 0; round < 2; round++) {
        for (String[] one : cases) {
          runtime.decide(space(one), one[4]);
        }
      }

      long started = System.nanoTime();
      for (String[] one : cases) {
        runtime.decide(space(one), one[4]);
      }
      double perCase = (System.nanoTime() - started) / 1e9;

      // The same number of decisions, one document, questions sharing its prefix.
      List<AnswerSpace> shared = new ArrayList<>();
      for (String[] one : cases) {
        shared.add(new Noul(one[3]));
      }
      for (int round = 0; round < 2; round++) {
        for (AnswerSpace space : shared) {
          runtime.decide(space, contract);
        }
      }
      started = System.nanoTime();
      for (AnswerSpace space : shared) {
        runtime.decide(space, contract);
      }
      double sharedEvidence = (System.nanoTime() - started) / 1e9;

      // And with a rubric, which is what the release added. Free when shared, not when not.
      List<AnswerSpace> withRubric = new ArrayList<>();
      for (String[] one : cases) {
        withRubric.add(
            new Noul(
                one[3],
                "The document states it explicitly, in terms a reader could quote",
                "The document is silent on it, or a later clause removes it"));
      }
      for (int round = 0; round < 2; round++) {
        for (AnswerSpace space : withRubric) {
          runtime.decide(space, contract);
        }
      }
      started = System.nanoTime();
      for (AnswerSpace space : withRubric) {
        runtime.decide(space, contract);
      }
      double rubricShared = (System.nanoTime() - started) / 1e9;

      started = System.nanoTime();
      for (int index = 0; index < withRubric.size(); index++) {
        // A different document each time, so the rubric cannot be shared either.
        runtime.decide(withRubric.get(index), cases.get(index)[4]);
      }
      double rubricPerCase = (System.nanoTime() - started) / 1e9;

      int n = cases.size();
      System.out.printf("%n  %d decisions in each shape%n%n", n);
      System.out.printf("  %-44s %-10s %-10s%n", "shape", "total s", "per s");
      row("one document per case, no rubric  (the videos)", perCase, n);
      row("one document per case, with rubric", rubricPerCase, n);
      row("one shared document, no rubric", sharedEvidence, n);
      row("one shared document, with rubric", rubricShared, n);
      System.out.println();
    }
  }

  private static void row(String label, double total, int n) {
    System.out.printf("  %-44s %-10.3f %-10.3f%n", label, total, total / n);
  }

  private static AnswerSpace space(String[] one) {
    List<String> options = List.of(one[2].split("\\|", -1));
    return switch (one[1]) {
      case "noul" -> new Noul(one[3]);
      case "choice" -> new Choice(one[3], options);
      default -> new Score(one[3], options);
    };
  }
}
