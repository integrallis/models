import com.integrallis.models.backend.purejava.bert.ModernBertConfig.QuestionType;
import com.integrallis.models.backend.purejava.bert.ModernBertDecisionModel;
import com.integrallis.models.backend.purejava.bert.ModernBertDecisionPrompt.Option;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Per-decision latency for Laya, timed the way the Harriet timing records are: a one-off cost
 * reported separately from the per-question marginal cost.
 *
 * Laya has no shared prefill to amortise -- it answers in one non-autoregressive pass -- so its
 * one-off cost is the model load and its marginal cost is the whole decision. That asymmetry is
 * the finding, not a gap in the measurement.
 */
public final class LayaLatency {
  public static void main(String[] args) throws Exception {
    Path artifact = Path.of(args[0]);
    Path cohort = Path.of(args[1]);
    List<String> lines = Files.readAllLines(cohort);

    long loadStart = System.nanoTime();
    try (var model = ModernBertDecisionModel.open(artifact)) {
      double loadSeconds = (System.nanoTime() - loadStart) / 1e9;

      List<Case> cases = new ArrayList<>();
      for (String line : lines) {
        Case parsed = parse(line);
        if (parsed != null) cases.add(parsed);
      }

      // Warm up on cases that are not then timed, so JIT and allocator state are not counted
      // against the first measured decision -- the same reason the Harriet record warms on a
      // different document before timing its prefill.
      for (int i = 0; i < Math.min(10, cases.size()); i++) decide(model, cases.get(i));

      double[] millis = new double[cases.size()];
      StringBuilder tsv = new StringBuilder();
      for (int i = 0; i < cases.size(); i++) {
        Case one = cases.get(i);
        long start = System.nanoTime();
        var answer = decide(model, one);
        millis[i] = (System.nanoTime() - start) / 1e6;
        // id<TAB>latency_seconds<TAB>label=p... -- the format the Java runner writes, so the
        // benchmark's own score_arm.py scores this arm with the same code as every other.
        tsv.append(one.id()).append('\t').append(String.format(Locale.ROOT, "%.9f", millis[i] / 1000.0));
        float[] probabilities = answer.probabilities();
        for (int option = 0; option < one.labels().size(); option++) {
          tsv.append('\t').append(one.labels().get(option)).append('=')
              .append(String.format(Locale.ROOT, "%.17g", (double) probabilities[option]));
        }
        tsv.append('\n');
      }
      if (args.length > 2) Files.writeString(Path.of(args[2]), tsv.toString());
      double[] sorted = millis.clone();
      Arrays.sort(sorted);
      double sum = 0;
      for (double m : millis) sum += m;

      System.out.printf("{%n");
      System.out.printf("  \"model\": \"laya-q8_0\",%n");
      System.out.printf("  \"readout\": \"typed-decision single pass\",%n");
      System.out.printf("  \"prefix_shared\": false,%n");
      System.out.printf("  \"decisions\": %d,%n", millis.length);
      System.out.printf("  \"load_s\": %.4f,%n", loadSeconds);
      System.out.printf("  \"marginal_ms_mean\": %.2f,%n", sum / millis.length);
      System.out.printf("  \"marginal_ms_p50\": %.2f,%n", sorted[sorted.length / 2]);
      System.out.printf("  \"marginal_ms_p95\": %.2f,%n", sorted[(int) (sorted.length * 0.95)]);
      System.out.printf("  \"marginal_ms_min\": %.2f,%n", sorted[0]);
      System.out.printf("  \"marginal_ms_max\": %.2f,%n", sorted[sorted.length - 1]);
      System.out.printf("  \"processors\": %d%n", Runtime.getRuntime().availableProcessors());
      System.out.printf("}%n");
    }
  }

  private static com.integrallis.models.backend.purejava.bert.ModernBertDecisionAnswer decide(
      ModernBertDecisionModel model, Case one) {
    List<Option> options = new ArrayList<>();
    for (String label : one.labels) options.add(new Option(label, one.criteria.get(label)));
    return model.decide(one.type, one.instructions, options, one.state);
  }

  private record Case(
      String id, QuestionType type, String instructions, List<String> labels,
      Map<String, String> criteria, String state) {}

  private static Case parse(String line) {
    String typeName = string(line, "type");
    String instructions = string(line, "instructions");
    String state = string(line, "state");
    if (typeName == null || instructions == null || state == null) return null;
    List<String> labels = new ArrayList<>();
    Matcher array = Pattern.compile("\"labels\":\\s*\\[(.*?)]", Pattern.DOTALL).matcher(line);
    if (array.find()) {
      Matcher each = Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(array.group(1));
      while (each.find()) labels.add(unescape(each.group(1)));
    }
    Map<String, String> criteria = new LinkedHashMap<>();
    Matcher object = Pattern.compile("\"criteria\":\\s*\\{(.*?)}", Pattern.DOTALL).matcher(line);
    if (object.find()) {
      Matcher pair =
          Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
              .matcher(object.group(1));
      while (pair.find()) criteria.put(unescape(pair.group(1)), unescape(pair.group(2)));
    } else {
      Matcher positional =
          Pattern.compile("\"criteria\":\\s*\\[(.*?)]", Pattern.DOTALL).matcher(line);
      if (positional.find()) {
        Matcher each = Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\"").matcher(positional.group(1));
        int index = 0;
        while (each.find() && index < labels.size()) {
          criteria.put(labels.get(index++), unescape(each.group(1)));
        }
      }
    }
    String id = string(line, "id");
    return new Case(
        id, QuestionType.valueOf(typeName.toUpperCase()), instructions, labels, criteria, state);
  }

  private static String string(String line, String key) {
    Matcher m =
        Pattern.compile("\"" + key + "\":\\s*\"((?:[^\"\\\\]|\\\\.)*)\"", Pattern.DOTALL)
            .matcher(line);
    return m.find() ? unescape(m.group(1)) : null;
  }

  private static String unescape(String raw) {
    return raw.replace("\\\"", "\"").replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\");
  }
}
