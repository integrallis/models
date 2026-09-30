package demo;

import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Noul;
import com.integrallis.models.decisions.Verdict;
import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Grouped against the same questions asked one at a time, both measured, not extrapolated. */
public final class Batch2 {
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
          "Is a police report required?",
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
          "Is a warranty disclaimed?");

  public static void main(String... args) throws Exception {
    String evidence = Files.readString(Path.of(args[0])).strip();
    try (var harriet = Harriet.open()) {
      // Warm both paths, not just one. The grouped path is separate code from the one-at-a-time
      // path, so warming only the latter leaves the former interpreted and the first grouped timing
      // measures compilation. Worse, unwarmed Panama vector code rounds differently, so the answer
      // drift printed below was measuring the JIT too. Recorded in this project's working agreement;
      // the first version of this harness ignored it.
      List<AnswerSpace> warmup = new ArrayList<>();
      for (int index = 0; index < 4; index++) {
        warmup.add(new Noul(CRITERIA.get(index)));
      }
      for (int round = 0; round < 4; round++) {
        for (AnswerSpace space : warmup) {
          harriet.decide(space, evidence);
        }
        Harriet.decideAll(harriet, warmup, evidence);
      }

      System.out.println();
      System.out.printf("  %-4s %-14s %-14s %-9s%n", "n", "one at a time", "grouped", "speedup");
      for (int n : new int[] {2, 5, 10, 20}) {
        List<AnswerSpace> spaces = new ArrayList<>();
        for (int index = 0; index < n; index++) {
          spaces.add(new Noul(CRITERIA.get(index % CRITERIA.size())));
        }

        long t = System.nanoTime();
        List<Verdict> one = new ArrayList<>();
        for (AnswerSpace space : spaces) {
          one.add(harriet.decide(space, evidence));
        }
        double sequential = (System.nanoTime() - t) / 1e9;

        t = System.nanoTime();
        List<Verdict> all = Harriet.decideAll(harriet, spaces, evidence);
        double grouped = (System.nanoTime() - t) / 1e9;

        double drift = 0;
        for (int index = 0; index < n; index++) {
          drift = Math.max(drift, Math.abs(one.get(index).confidence() - all.get(index).confidence()));
        }
        System.out.printf(
            "  %-4d %-14.3f %-14.3f %-9s  max answer drift %.2e%n",
            n, sequential, grouped, String.format("%.2fx", sequential / grouped), drift);
      }
      System.out.println();
    }
  }
}
