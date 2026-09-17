import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.InferenceBackend;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

/** Loads a model, runs a short greedy generation, prints diagnostics relevant to cliffs. */
public class CliffProbe {
  public static void main(String[] args) throws Exception {
    String mode = args[0];
    Path model = Path.of(args[1]);
    int decode = Integer.parseInt(args[2]);
    long start = System.nanoTime();
    InferenceBackend backend =
        mode.equals("native")
            ? RustFfmBackend.load(model, Path.of(System.getProperty("probe.library")),
                com.integrallis.models.api.BackendConfiguration.empty())
            : PureJavaBackend.load(model);
    long loaded = System.nanoTime();
    try (backend) {
      int[] prompt =
          backend.tokenizer().encode(ModelPrompt.text("The quick brown fox jumps over the lazy dog because"));
      float[] logits = backend.prefill(prompt, 0);
      int position = prompt.length;
      StringBuilder tokens = new StringBuilder();
      for (int step = 0; step < decode; step++) {
        int next = argmax(logits);
        tokens.append(next).append(' ');
        logits = backend.forward(next, position++);
      }
      long done = System.nanoTime();
      BackendDiagnostics diagnostics =
          (BackendDiagnostics) backend.getClass().getMethod("diagnostics").invoke(backend);
      Map<String, String> env = new TreeMap<>(diagnostics.environment());
      System.out.println("PROBE backend=" + diagnostics.backend() + " plan=" + diagnostics.planVersion());
      for (String key : new String[] {"model-architecture", "vector-provider", "vector-api",
          "preferred-vector-bits", "active-vector-bits", "gguf-executor", "gguf-parallel",
          "native-grouped-attention", "native-gated-delta-net", "native-kernel-poll-millis"}) {
        if (env.containsKey(key)) System.out.println("PROBE env " + key + "=" + env.get(key));
      }
      env.forEach((k, v) -> { if (k.startsWith("performance-cliff")) System.out.println("PROBE cliff " + k + "=" + v); });
      diagnostics.optimization("q4-kernel").ifPresent(d -> System.out.println("PROBE q4-kernel " + d.status() + " " + d.reason()));
      System.out.println("PROBE prompt-tokens=" + prompt.length + " decode=" + decode
          + " load-ms=" + (loaded - start) / 1_000_000 + " run-ms=" + (done - loaded) / 1_000_000);
      System.out.println("PROBE greedy-tokens=" + tokens.toString().trim());
    }
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
    return best;
  }
}
