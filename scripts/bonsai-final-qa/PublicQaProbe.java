import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.*;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.GenerationLoop;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;

/** Frozen external QA cases through the ordinary generation loop, with raw outputs retained. */
public class PublicQaProbe {
  public static void main(String[] args) throws Exception {
    // model, cases, output, backend (or validate), context, first case, count, runtime identity
    if (args.length != 8) throw new IllegalArgumentException("model cases output backend context first count runtimeIdentity");
    var mapper = new ObjectMapper();
    Path model = Path.of(args[0]), cases = Path.of(args[1]), output = Path.of(args[2]);
    byte[] caseBytes = Files.readAllBytes(cases);
    JsonNode manifest = mapper.readTree(caseBytes);
    int context = Integer.parseInt(args[4]), first = Integer.parseInt(args[5]), count = Integer.parseInt(args[6]);
    if (first < 0 || count < 1 || first + count > manifest.get("cases").size()) throw new IllegalArgumentException("case range");
    var configuration = new BackendConfiguration(Map.of(), Map.of(PureJavaBackend.MAX_CONTEXT_LENGTH_PROPERTY, Integer.toString(context)), List.of());
    long loadedAt = System.nanoTime();
    try (InferenceBackend backend = args[3].equals("pure-java") ? PureJavaBackend.load(model, configuration) : RustFfmBackend.load(model, configuration);
         var writer = Files.newBufferedWriter(output, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      var header = new LinkedHashMap<String,Object>();
      header.put("type", "manifest"); header.put("createdAt", Instant.now().toString());
      header.put("caseSha256", HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256").digest(caseBytes)));
      header.put("modelPath", model.toString()); header.put("artifactSizeBytes", Files.size(model));
      header.put("backend", args[3]); header.put("runtimeIdentity", args[7]); header.put("context", context);
      header.put("first", first); header.put("count", count); header.put("processors", Runtime.getRuntime().availableProcessors());
      header.put("classOrigins", Map.of(
          "core", Class.forName("com.integrallis.vectors.core.BFloat16Matrix").getProtectionDomain().getCodeSource().getLocation().toString(),
          "models", PureJavaBackend.class.getProtectionDomain().getCodeSource().getLocation().toString(),
          "native", RustFfmBackend.class.getProtectionDomain().getCodeSource().getLocation().toString(),
          "runtime", GenerationLoop.class.getProtectionDomain().getCodeSource().getLocation().toString()));
      header.put("javaVersion", System.getProperty("java.version")); header.put("loadMillis", (System.nanoTime()-loadedAt)/1e6);
      header.put("generation", Map.of("temperature", 0, "topK", 1, "topP", 1, "seed", 42, "repetitionPenalty", 1, "rawPrompt", true, "promptCache", false));
      header.put("tokenizationMode", "trusted-frozen-chatml-v2");
      header.put("diagnostics", backend.getClass().getMethod("diagnostics").invoke(backend));
      writer.write(mapper.writeValueAsString(header)); writer.newLine(); writer.flush();
      for (int index = first; index < first + count; index++) {
        JsonNode c = manifest.get("cases").get(index);
        // This immutable benchmark owns the complete rendered template. Match the publisher's
        // raw completion API, which parses its controls; String generation intentionally does not.
        ModelPrompt prompt = ModelPrompt.control(c.get("prompt").asText());
        int max = c.get("maxTokens").asInt();
        var row = new LinkedHashMap<String,Object>();
        row.put("index", index); row.put("dataset", c.get("dataset").asText()); row.put("id", c.get("id").asText());
        row.put("promptSha256", c.get("promptSha256").asText()); row.put("maxTokens", max);
        StringBuilder text = new StringBuilder();
        long start = System.nanoTime();
        try {
          int[] tokenIds = backend.tokenizer().encode(prompt);
          int tokens = tokenIds.length;
          row.put("inputTokens", tokens);
          row.put("inputTokenIds", tokenIds);
          if (tokens + max > context) throw new IllegalArgumentException("context overflow: " + tokens + " + " + max + " > " + context);
          if (!args[3].equals("validate")) {
            backend.reset();
            var loop = new GenerationLoop(backend); // fresh cache for each measured request
            loop.generate(prompt, SamplingOptions.builder().temperature(0).topK(1).topP(1).seed(42).repetitionPenalty(1).maxTokens(max).build(), new TokenStream() {
              public void onToken(String fragment) { text.append(fragment); }
              public void onComplete() {}
              public void onComplete(GenerationUsage usage, StopReason reason) {
                row.put("usage", usage); row.put("stopReason", reason.toString());
              }
              public void onError(Throwable error) { row.put("error", error.toString()); }
            });
            var metrics = loop.lastGenerationMetrics();
            row.put("prefillMillis", metrics.prefill().toNanos()/1e6);
            row.put("decodeMillis", metrics.decode().toNanos()/1e6);
            row.put("ttftMillis", metrics.timeToFirstToken().map(d -> d.toNanos()/1e6).orElse(null));
          }
        } catch (RuntimeException error) { row.put("error", error.toString()); }
        row.put("output", text.toString()); row.put("totalMillis", (System.nanoTime()-start)/1e6);
        writer.write(mapper.writeValueAsString(row)); writer.newLine(); writer.flush();
        System.out.println("case=" + index + " dataset=" + c.get("dataset").asText() + " error=" + row.get("error") + " stop=" + row.get("stopReason") + " millis=" + row.get("totalMillis"));
      }
    }
  }
}
