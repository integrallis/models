/* Copyright 2025-2026 Integrallis Software, LLC. SPDX-License-Identifier: Apache-2.0 */
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.*;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.rag.*;
import com.integrallis.models.runtime.GenerationLoop;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;

/** Remote public-configuration check; released JAR identity is verified by the enclosing worker. */
public final class FlintReleasedProfileProbe {
  public static void main(String[] args) throws Exception {
    if (args.length != 3 || !System.getProperty("os.name").equals("Linux"))
      throw new IllegalArgumentException("Remote Linux only: MODEL EXPECTED_RAW_SHA256 NEW_REPORT");
    if (!args[1].matches("[0-9a-f]{64}")) throw new IllegalArgumentException("Invalid expected digest");
    for (String name : System.getProperties().stringPropertyNames())
      if (name.startsWith("models.native.")) throw new IllegalArgumentException("Deployment override: " + name);
    Path report = Path.of(args[2]);
    if (Files.exists(report)) throw new IllegalArgumentException("Report exists");
    System.setProperty("models.purejava.maxContextLength", "2048");
    var corpus = RagCorpus.loadDefault();
    var test = corpus.cases().stream().filter(c -> c.id().equals("auto-glass-deadline")).findFirst().orElseThrow();
    var document = corpus.documents().stream().filter(d -> d.id().equals("claims-auto-glass")).findFirst().orElseThrow();
    var prompt = RagPromptRenderer.renderPrompt(test.question(), List.of(new RetrievedDocument(document, 10f, 1)), RagPromptTemplate.FLINT);
    var arms = new ArrayList<Map<String, Object>>();
    boolean passed = true;
    for (int threads : new int[] {16, 8}) {
      var configuration = threads == 16 ? BackendConfiguration.empty()
          : new BackendConfiguration(Map.of(), Map.of("models.native.kernels.threads", "8"), List.of());
      try (var backend = RustFfmBackend.load(Path.of(args[0]), configuration)) {
        var environment = backend.diagnostics().environment();
        boolean applied = Integer.toString(threads).equals(environment.get("native-kernel-threads"));
        boolean routesPresent = backend.diagnostics().optimizations().stream().anyMatch(decision ->
            decision.id().equals("flint-injected-projection-groups")
            && decision.status() == OptimizationStatus.ENABLED
            && "47".equals(decision.settings().get("routed-expert-eligible-layers"))
            && "49".equals(decision.settings().get("qkv-eligible-layers"))
            && "49".equals(decision.settings().get("dense-shared-gate-up-eligible-layers"))
            && "sequential-block-order".equals(decision.settings().get("q4-k-reduction")));
        passed &= applied && routesPresent;
        var loop = new GenerationLoop(backend);
        var options = SamplingOptions.builder().temperature(0).maxTokens(512).build();
        String warmup = loop.generate(prompt, options);
        boolean warmupMatches = sha(warmup).equals(args[1]);
        passed &= warmupMatches;
        var runs = new ArrayList<Map<String, Object>>();
        for (int iteration = 0; iteration < 3; iteration++) {
          var sink = new MeasuredStream();
          long start = System.nanoTime();
          loop.generate(prompt, options, sink);
          long end = System.nanoTime();
          boolean matched = sha(sink.text.toString()).equals(args[1]);
          boolean complete = sink.reason == StopReason.EOS && sink.usage != null;
          passed &= matched && complete;
          if (!complete) throw new IllegalStateException("No complete generation usage");
          runs.add(Map.of("rawText", sink.text.toString(), "rawSha256", sha(sink.text.toString()),
              "matchesSourceQualification", matched, "stopReason", sink.reason,
              "usage", sink.usage, "endToEndMillis", (end - start) / 1e6,
              "ttftMillis", (sink.firstToken - start) / 1e6,
              "decodeTokensPerSecond", (sink.usage.completionTokens() - 1) * 1e9 / (end - sink.firstToken)));
        }
        arms.add(Map.of("nativeThreads", threads, "recommendations", configuration.recommendations(),
            "configurationApplied", applied, "requiredProjectionRoutesAvailable", routesPresent, "warmupMatchesSourceQualification", warmupMatches,
            "backendDiagnostics", backend.diagnostics(), "runs", runs));
      }
    }
    var result = new LinkedHashMap<String, Object>();
    result.put("kind", "flint-public-profile-round-trip-v1");
    result.put("passed", passed);
    result.put("promptSha256", sha(prompt.text()));
    result.put("expectedRawSha256", args[1]);
    result.put("limitation", "One-case released-library/profile compatibility and paired timing; full qualification remains separate. Expected output is used only after generation.");
    result.put("runtimeCodeSource", RustFfmBackend.class.getProtectionDomain().getCodeSource().getLocation().toString());
    result.put("arms", arms);
    Files.writeString(report, new ObjectMapper().writerWithDefaultPrettyPrinter().writeValueAsString(result), StandardOpenOption.CREATE_NEW);
    if (!passed) throw new IllegalStateException("Released public profile check failed: " + report);
  }
  private static String sha(String text) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(text.getBytes(StandardCharsets.UTF_8)));
  }
  private static final class MeasuredStream implements TokenStream {
    final StringBuilder text = new StringBuilder();
    long firstToken;
    GenerationUsage usage;
    StopReason reason;
    public void onToken(String token) { if (firstToken == 0) firstToken = System.nanoTime(); text.append(token); }
    public void onComplete() { throw new IllegalStateException("Usage required"); }
    public void onComplete(GenerationUsage usage, StopReason reason) { this.usage = usage; this.reason = reason; }
    public void onError(Throwable failure) { throw new IllegalStateException("Generation failed", failure); }
  }
}
