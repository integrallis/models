///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25
//RUNTIME_OPTIONS --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED
//
// Per-decision latency for a typed decision, on YOUR hardware, same model both ways.
//
// Two arms answer one identical triage task over identical messages with an identical rubric:
//
//   in-process   ModelJars opens Qwen2.5-1.5B-Instruct Q4_K_M and reads the option letters'
//                logits at the final position. No server, no HTTP, no generation.
//   over-server  dukehoff-decide's OpenAiBackend against a /v1/chat/completions endpoint that
//                returns logprobs (llama.cpp's llama-server, or Ollama).
//
// The model is held constant on purpose. Point the server at the same
// Qwen2.5-1.5B-Instruct-Q4_K_M GGUF and the only variable left is the runtime, which is the
// comparison worth having. Serve a different file and the script will still run, but it is then
// comparing two models and says so in the output.
//
// Coordinates were resolved from Maven Central on 2026-11-03 and pinned, rather than floated, so
// two people running this script are running the same thing.
//
//DEPS org.modeljars:modeljars:0.1.76
//DEPS org.modeljars.huggingface:qwen.qwen2.5-1.5b-instruct-gguf.q4_k_m:2.5.0-q4_k_m.1
//DEPS com.dukehoff.decide:dukehoff-decide-openai:0.2.0

import com.dukehoff.decide.Decider;
import com.dukehoff.decide.Decision;
import com.dukehoff.decide.DecisionTask;
import com.dukehoff.decide.Option;
import com.dukehoff.decide.Question;
import com.dukehoff.decide.State;
import com.dukehoff.decide.openai.OpenAiBackend;
import com.dukehoff.decide.spi.ScoringBackend;
import com.integrallis.models.decisions.AnswerSpace;
import com.integrallis.models.decisions.Choice;
import com.integrallis.models.decisions.Verdict;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.modeljars.ModelJarDecisionRuntime;
import org.modeljars.ModelJars;

public class DecideLatency {

  /** The marker whose GGUF both arms should be running. */
  private static final String MARKER =
      "org.modeljars.huggingface:qwen.qwen2.5-1.5b-instruct-gguf.q4_k_m:2.5.0-q4_k_m.1";

  private static final String QUESTION = "Which team should handle this customer message?";

  /**
   * One written rule per outcome, not just a label.
   *
   * <p>This is not decoration. On our own 120-case set the same model scored 0.75 with a bare
   * question and 0.89 with a rule per outcome, and the published numbers this script exists to
   * check were produced with option descriptions too. A rubric-free version of this task would
   * measure the prompt, not the runtime.
   */
  private static final Map<String, String> CRITERIA =
      new LinkedHashMap<>() {
        {
          put("BILLING", "charges, invoices, refunds, payment methods, pricing disputes");
          put("ACCOUNT", "sign-in, passwords, profile details, closing or merging an account");
          put("ORDER", "placing, changing, cancelling or tracking an order or delivery");
          put("TECHNICAL", "something is broken, erroring, slow, or not working as documented");
          put("FEEDBACK", "praise, complaint or suggestion with no request to action");
        }
      };

  private static final List<String> OPTIONS = List.copyOf(CRITERIA.keySet());

  /** The same five outcomes and the same rules, in the form the over-server arm requires. */
  @Question(QUESTION)
  enum Triage {
    @Option("charges, invoices, refunds, payment methods, pricing disputes")
    BILLING,
    @Option("sign-in, passwords, profile details, closing or merging an account")
    ACCOUNT,
    @Option("placing, changing, cancelling or tracking an order or delivery")
    ORDER,
    @Option("something is broken, erroring, slow, or not working as documented")
    TECHNICAL,
    @Option("praise, complaint or suggestion with no request to action")
    FEEDBACK
  }

  /** Message plus the team a human assigned it to. */
  record Message(String text, String expected) {}

  /**
   * Fifteen messages, three per outcome.
   *
   * <p>Deliberately few: this script measures <b>latency</b>, and fifteen messages is ample for
   * that while keeping a run short enough that people actually do it. Fifteen cases cannot support
   * an accuracy claim and the output says so — agreement is printed only to catch an arm that is
   * answering nonsense, which a latency number alone would hide.
   */
  private static final List<Message> MESSAGES =
      List.of(
          new Message("I was charged twice for the same subscription this month.", "BILLING"),
          new Message("Can you explain the extra 40 euros on my last invoice?", "BILLING"),
          new Message("I want a refund for the plan I never used.", "BILLING"),
          new Message("I can't sign in, the password reset email never arrives.", "ACCOUNT"),
          new Message("Please close my account and delete my profile.", "ACCOUNT"),
          new Message("How do I change the email address on my profile?", "ACCOUNT"),
          new Message("My parcel says delivered but nothing arrived.", "ORDER"),
          new Message("I need to cancel order 55812 before it ships.", "ORDER"),
          new Message("Can I change the delivery address on my open order?", "ORDER"),
          new Message("The export button throws a 500 error every time.", "TECHNICAL"),
          new Message("The dashboard takes thirty seconds to load since the update.", "TECHNICAL"),
          new Message("Your API returns 401 even with a fresh token.", "TECHNICAL"),
          new Message("Just wanted to say your support team was excellent.", "FEEDBACK"),
          new Message("The new layout is much worse than the old one.", "FEEDBACK"),
          new Message("It would be nice if reports could be scheduled weekly.", "FEEDBACK"));

  public static void main(String[] args) throws Exception {
    String serverUrl = argument(args, "--server", null);
    String serverModel = argument(args, "--server-model", "qwen2.5:1.5b");
    Path json = argument(args, "--json", null) == null ? null : Path.of(argument(args, "--json", null));
    boolean skipLocal = Arrays.asList(args).contains("--no-local");

    System.out.println("decide-latency — typed decision, per-decision wall time");
    System.out.println(machine());
    System.out.printf(
        "task: %d options with a rule each, %d messages%n", OPTIONS.size(), MESSAGES.size());
    System.out.println();

    List<Result> results = new ArrayList<>();
    if (!skipLocal) {
      results.add(inProcess());
    }
    if (serverUrl != null) {
      results.add(overServer(serverUrl, serverModel));
    } else {
      System.out.println("over-server arm skipped: pass --server http://localhost:8080/v1");
      System.out.println("  llama.cpp:  llama-server -m Qwen2.5-1.5B-Instruct-Q4_K_M.gguf --port 8080");
      System.out.println("  Ollama:     ollama serve   (then --server http://localhost:11434/v1)");
      System.out.println();
    }

    report(results);
    if (json != null) {
      Files.writeString(json, toJson(results));
      System.out.println("wrote " + json.toAbsolutePath());
    }
  }

  /** Our arm: the GGUF opened in this JVM, option logits read in one forward pass. */
  private static Result inProcess() {
    System.out.println("== in-process (ModelJars) ==");
    long openedAt = System.nanoTime();
    try (ModelJarDecisionRuntime runtime = ModelJars.openDecisionRuntime(MARKER)) {
      double openMs = millisSince(openedAt);
      // Report the route rather than assuming it: the same marker can resolve to different
      // kernels, and a latency number whose backend is unknown is not comparable to anything.
      System.out.printf("  backend: %s%n", runtime.backend().name());
      System.out.printf("  open: %.0f ms (resolve, verify and map the weights)%n", openMs);

      AnswerSpace space = new Choice(QUESTION, OPTIONS, CRITERIA);

      long firstAt = System.nanoTime();
      Verdict first = runtime.decide(space, MESSAGES.get(0).text());
      double firstMs = millisSince(firstAt);
      System.out.printf(
          "  first decision: %.0f ms (includes JIT warm-up) -> %s%n", firstMs, first.winner());

      List<Double> warm = new ArrayList<>();
      int agreed = 0;
      for (Message message : MESSAGES) {
        long at = System.nanoTime();
        Verdict verdict = runtime.decide(space, message.text());
        warm.add(millisSince(at));
        if (verdict.winner().equalsIgnoreCase(message.expected())) {
          agreed++;
        }
      }
      return finish("in-process", runtime.backend().name(), openMs, firstMs, warm, agreed);
    }
  }

  /** His arm: the same task over HTTP to a logprobs endpoint, which is how it is measured today. */
  private static Result overServer(String baseUrl, String model) {
    System.out.printf("== over-server (dukehoff-decide -> %s, model %s) ==%n", baseUrl, model);
    long openedAt = System.nanoTime();
    ScoringBackend backend = OpenAiBackend.builder().baseUrl(baseUrl).model(model).build();
    DecisionTask<Triage> task = Decider.builder().backend(backend).build().task(Triage.class);
    double openMs = millisSince(openedAt);
    System.out.printf("  open: %.0f ms (no weights to load; the server already holds them)%n", openMs);

    long firstAt = System.nanoTime();
    Decision<Triage> first = task.decide(State.of(MESSAGES.get(0).text()));
    double firstMs = millisSince(firstAt);
    System.out.printf("  first decision: %.0f ms -> %s%n", firstMs, first.value());

    List<Double> warm = new ArrayList<>();
    int agreed = 0;
    for (Message message : MESSAGES) {
      long at = System.nanoTime();
      Decision<Triage> decision = task.decide(State.of(message.text()));
      warm.add(millisSince(at));
      if (decision.value().name().equalsIgnoreCase(message.expected())) {
        agreed++;
      }
    }
    return finish("over-server", model + " @ " + baseUrl, openMs, firstMs, warm, agreed);
  }

  record Result(
      String arm,
      String route,
      double openMs,
      double firstMs,
      double p50,
      double p95,
      double mean,
      int agreed,
      int total) {}

  private static Result finish(
      String arm, String route, double openMs, double firstMs, List<Double> warm, int agreed) {
    double[] sorted = warm.stream().mapToDouble(Double::doubleValue).sorted().toArray();
    double mean = Arrays.stream(sorted).average().orElse(Double.NaN);
    Result result =
        new Result(
            arm, route, openMs, firstMs, percentile(sorted, 0.50), percentile(sorted, 0.95),
            mean, agreed, warm.size());
    System.out.printf(
        "  warm: p50 %.1f ms, p95 %.1f ms, mean %.1f ms%n", result.p50(), result.p95(), mean);
    System.out.printf("  agreement with the human label: %d/%d%n%n", agreed, warm.size());
    return result;
  }

  private static void report(List<Result> results) {
    if (results.isEmpty()) {
      System.out.println("nothing ran.");
      return;
    }
    System.out.println("== summary ==");
    System.out.printf(
        "%-13s %-34s %9s %9s %9s %9s%n", "arm", "route", "open", "first", "p50", "p95");
    for (Result r : results) {
      System.out.printf(
          "%-13s %-34s %9.0f %9.0f %9.1f %9.1f%n",
          r.arm(), truncate(r.route(), 34), r.openMs(), r.firstMs(), r.p50(), r.p95());
    }
    System.out.println();
    System.out.println("All times are milliseconds of wall clock for ONE decision.");
    System.out.println(
        "Agreement over "
            + results.get(0).total()
            + " messages is a sanity check, not an accuracy measurement — far too few cases.");
    if (results.size() == 2) {
      double ratio = results.get(1).p50() / results.get(0).p50();
      System.out.printf(
          "p50 ratio over-server / in-process: %.2fx%n", ratio);
      System.out.println(
          "Only read that as a runtime comparison if the server is serving the same"
              + " Qwen2.5-1.5B-Instruct-Q4_K_M file this script downloaded.");
    }
  }

  private static String machine() {
    Runtime runtime = Runtime.getRuntime();
    return String.format(
        Locale.ROOT,
        "machine: %s %s, %d available processors, JVM %s, max heap %d MiB",
        System.getProperty("os.name"),
        System.getProperty("os.arch"),
        runtime.availableProcessors(),
        Runtime.version(),
        runtime.maxMemory() >> 20);
  }

  private static String toJson(List<Result> results) {
    StringBuilder out = new StringBuilder();
    out.append("{\n  \"machine\": \"").append(machine().replace("\"", "'")).append("\",\n");
    out.append("  \"marker\": \"").append(MARKER).append("\",\n");
    out.append("  \"arms\": [\n");
    for (int index = 0; index < results.size(); index++) {
      Result r = results.get(index);
      out.append(
          String.format(
              Locale.ROOT,
              "    {\"arm\": \"%s\", \"route\": \"%s\", \"openMs\": %.1f, \"firstMs\": %.1f,"
                  + " \"p50Ms\": %.2f, \"p95Ms\": %.2f, \"meanMs\": %.2f, \"agreed\": %d,"
                  + " \"messages\": %d}%s%n",
              r.arm(), r.route().replace("\"", "'"), r.openMs(), r.firstMs(), r.p50(), r.p95(),
              r.mean(), r.agreed(), r.total(), index + 1 < results.size() ? "," : ""));
    }
    out.append("  ]\n}\n");
    return out.toString();
  }

  private static double percentile(double[] sorted, double quantile) {
    if (sorted.length == 0) {
      return Double.NaN;
    }
    int index = (int) Math.ceil(quantile * sorted.length) - 1;
    return sorted[Math.max(0, Math.min(sorted.length - 1, index))];
  }

  private static double millisSince(long startNanos) {
    return (System.nanoTime() - startNanos) / 1_000_000.0;
  }

  private static String truncate(String value, int width) {
    return value.length() <= width ? value : value.substring(0, width - 1) + "~";
  }

  private static String argument(String[] args, String name, String fallback) {
    for (int index = 0; index + 1 < args.length; index++) {
      if (args[index].equals(name)) {
        return args[index + 1];
      }
    }
    return fallback;
  }
}
