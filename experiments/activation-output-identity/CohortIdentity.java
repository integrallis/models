import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Generates over the JevBench cohort and records exactly what the model chose, per case.
 *
 * <p>This answers the quality question without the scoring harness that produced the original
 * 76/120 figure, whose source is no longer in the tree or the archive. Task quality can only change
 * if the chosen tokens change, so running two vectors-core arms under one fixed template and
 * diffing the token sequences bounds the quality impact exactly: cases whose tokens match cannot
 * score differently, and the count that differ is the maximum possible swing. The template does not
 * have to match the lost harness, because both arms run under the same one.
 *
 * <p>Greedy decoding, fixed token budget, one model load for the whole cohort. Writes one JSON line
 * per case: id, expected label, prompt token count, and the generated token ids.
 */
public final class CohortIdentity {

  public static void main(String[] args) throws Exception {
    Path model = Path.of(args[0]);
    Path cohort = Path.of(args[1]);
    Path output = Path.of(args[2]);
    int tokens = args.length > 3 ? Integer.parseInt(args[3]) : 8;

    List<String> lines = Files.readAllLines(cohort, StandardCharsets.UTF_8);
    var configuration =
        new BackendConfiguration(
            Map.of(), Map.of(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, "1024"), List.of());

    var out = new StringBuilder();
    try (PureJavaBackend backend = PureJavaBackend.load(model, configuration)) {
      int index = 0;
      for (String line : lines) {
        if (line.isBlank()) {
          continue;
        }
        String id = field(line, "id");
        String expected = field(line, "expected");
        String state = field(line, "state");
        String labels = String.join(", ", fieldArray(line, "labels"));

        // One fixed template for both arms. Its wording is irrelevant to an identity comparison.
        String prompt =
            "Classify the message into exactly one category.\nCategories: "
                + labels
                + "\nMessage: "
                + state
                + "\nCategory:";

        backend.reset();
        int[] promptTokens = backend.tokenizer().encode(prompt);
        float[] logits = backend.prefill(promptTokens, 0);
        var generated = new ArrayList<Integer>(tokens);
        for (int step = 0; step < tokens; step++) {
          int next = argmax(logits);
          generated.add(next);
          logits = backend.forward(next, promptTokens.length + step);
        }
        out.append("{\"id\":\"").append(id).append("\",\"expected\":\"").append(expected)
            .append("\",\"promptTokens\":").append(promptTokens.length)
            .append(",\"generated\":").append(generated).append("}\n");
        if (++index % 20 == 0) {
          System.out.printf("  %d/%d cases%n", index, lines.size());
        }
      }
      System.out.printf("cases=%d%n", index);
    }
    Files.writeString(output, out.toString(), StandardCharsets.UTF_8);
    System.out.println("wrote " + output.toAbsolutePath());
  }

  /** Minimal field read; the cohort is one flat JSON object per line with string scalars. */
  private static String field(String json, String name) {
    String key = "\"" + name + "\":";
    int at = json.indexOf(key);
    if (at < 0) {
      return "";
    }
    int start = json.indexOf('"', at + key.length()) + 1;
    var value = new StringBuilder();
    for (int i = start; i < json.length(); i++) {
      char c = json.charAt(i);
      if (c == '\\') {
        char next = json.charAt(++i);
        value.append(next == 'n' ? '\n' : next);
        continue;
      }
      if (c == '"') {
        break;
      }
      value.append(c);
    }
    return value.toString();
  }

  private static List<String> fieldArray(String json, String name) {
    String key = "\"" + name + "\":";
    int at = json.indexOf(key);
    int open = json.indexOf('[', at);
    int close = json.indexOf(']', open);
    var values = new ArrayList<String>();
    for (String part : json.substring(open + 1, close).split(",")) {
      String trimmed = part.trim();
      if (trimmed.length() > 1) {
        values.add(trimmed.substring(1, trimmed.length() - 1));
      }
    }
    return values;
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }
}
