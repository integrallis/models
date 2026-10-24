import com.integrallis.models.api.OptimizationDecision;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Greedy continuation of a prompt LONGER than maple's 512-token sliding window.
 *
 * A short prompt cannot see this defect: below the window, bounded and full attention are the
 * same computation, which is why the four-token load probe reported finite logits and proved
 * nothing at all. Above it the two diverge on the 21 of 24 layers the pattern marks local, so
 * the continuation below is a real comparison against llama.cpp's.
 */
public final class MapleLongPrompt {
  public static void main(String[] args) throws Exception {
    String prompt = Files.readString(Path.of(args[1]));
    int generate = Integer.parseInt(args[2]);
    // "native" is the shipped route: the Rust batched-matmul shim. PureJavaBackend.load(Path)
    // passes GgufBatchedMatrixKernel.none(), so the unbatched arm below is a reference path and
    // not what ships -- a prefill time measured there says nothing about the product.
    boolean useNative = args.length > 3 && "native".equals(args[3]);
    try (var backend =
        useNative
            ? PureJavaBackend.load(
                Path.of(args[0]), RustGgufBatchedMatrixKernel.openBundled())
            : PureJavaBackend.load(Path.of(args[0]))) {
      System.out.println("route        = " + (useNative ? "rust-batched-matmul" : "unbatched-reference"));
      for (OptimizationDecision decision : backend.diagnostics().optimizations()) {
        if (decision.id().equals("sliding-window-attention")
            || decision.id().equals("expert-swiglu-clamp")) {
          System.out.println(
              "  " + decision.id() + " = " + decision.status() + " " + decision.settings());
        }
      }
      int[] tokens = backend.tokenizer().encode(prompt);
      System.out.println("promptTokens = " + tokens.length);
      System.out.println("window       = 512  exceeded=" + (tokens.length > 512));

      long t0 = System.nanoTime();
      float[] logits = backend.prefill(tokens, 0);
      long prefillNanos = System.nanoTime() - t0;

      System.out.printf("prefill      = %.2f s%n", prefillNanos / 1e9);
      StringBuilder out = new StringBuilder();
      int position = tokens.length;
      for (int step = 0; step < generate; step++) {
        int best = argmax(logits);
        if (!Float.isFinite(logits[best])) {
          System.out.println("NON-FINITE logit at step " + step);
          return;
        }
        String piece = backend.tokenizer().decode(new int[] {best});
        out.append(piece);
        System.out.println("  step " + step + " token=" + best + " " + piece.replace("\n", "\\n"));
        logits = backend.forward(best, position++);
      }
      System.out.println("continuation = " + out.toString().replace("\n", "\\n"));
    }
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
