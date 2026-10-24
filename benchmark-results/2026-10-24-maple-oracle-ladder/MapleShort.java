import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;

/** Top-k next-token distribution for a SHORT prompt, to localise the divergence. */
public final class MapleShort {
  public static void main(String[] args) {
    String prompt = args[1];
    try (var backend =
        PureJavaBackend.load(Path.of(args[0]), RustGgufBatchedMatrixKernel.openBundled())) {
      System.out.println("contextCapacity = " + backend.contextCapacity());
      System.out.println("metadataContext = " + backend.metadata().contextLength());
      int[] tokens = backend.tokenizer().encode(prompt);
      System.out.println("tokens = " + java.util.Arrays.toString(tokens));
      for (int t : tokens) {
        System.out.println("   " + t + " -> " + backend.tokenizer().decode(new int[] {t}));
      }
      if (backend.supportsHiddenState()) {
        float[] hidden = backend.prefillHiddenState(tokens, 0);
        double hs = 0.0;
        for (float v : hidden) {
          hs += v;
        }
        System.out.printf(
            "result_norm: [0]=%.4f [1]=%.4f [2]=%.4f ... [n-3]=%.4f [n-2]=%.4f [n-1]=%.4f sum=%.6f n=%d%n",
            hidden[0], hidden[1], hidden[2], hidden[hidden.length - 3],
            hidden[hidden.length - 2], hidden[hidden.length - 1], hs, hidden.length);
      } else {
        System.out.println("result_norm: hidden state unsupported on this route");
      }
      float[] logits = backend.prefill(tokens, 0);
      Integer[] idx = new Integer[logits.length];
      for (int i = 0; i < idx.length; i++) {
        idx[i] = i;
      }
      java.util.Arrays.sort(idx, (a, b) -> Float.compare(logits[b], logits[a]));
      int n = logits.length;
      double sum = 0.0;
      for (float v : logits) {
        sum += v;
      }
      System.out.printf(
          "oracle-compare: [0]=%.4f [1]=%.4f [2]=%.4f ... [n-3]=%.4f [n-2]=%.4f [n-1]=%.4f sum=%.1f%n",
          logits[0], logits[1], logits[2], logits[n - 3], logits[n - 2], logits[n - 1], sum);
      System.out.println("top-8:");
      for (int i = 0; i < 8; i++) {
        System.out.printf(
            "  %d  %.4f  %s%n",
            idx[i], logits[idx[i]], backend.tokenizer().decode(new int[] {idx[i]}).replace("\n", "\\n"));
      }
    }
  }
}
