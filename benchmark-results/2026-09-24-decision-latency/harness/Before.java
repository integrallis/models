package demo;

import org.modeljars.composite.harriet.Harriet;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * One number, measured on both builds: what a decision costs in the shape the demo videos use.
 *
 * <p>Uses only API that exists in both the published 0.1.52 and the current build, so the same class
 * runs against each without recompiling. Every case carries its own document, which is what the
 * recorded videos do and the shape with no shared prefix to exploit.
 */
public final class Before {
  public static void main(String... args) throws Exception {
    List<String[]> cases = new ArrayList<>();
    for (String line : Files.readAllLines(Path.of(args[0]))) {
      String[] fields = line.split("\t", -1);
      if (fields.length == 5 && !fields[0].equals("id")) {
        cases.add(fields);
      }
    }

    try (var runtime = Harriet.open()) {
      for (int round = 0; round < 2; round++) {
        for (String[] one : cases) {
          Harriet.noul(runtime, one[3], one[4]);
        }
      }
      double best = Double.MAX_VALUE;
      for (int round = 0; round < 3; round++) {
        long started = System.nanoTime();
        for (String[] one : cases) {
          Harriet.noul(runtime, one[3], one[4]);
        }
        best = Math.min(best, (System.nanoTime() - started) / 1e9);
      }
      System.out.printf(
          "%n  %d decisions, each with its own document: %.3f s total, %.3f s each%n%n",
          cases.size(), best, best / cases.size());
    }
  }
}
