///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 25+
//DEPS com.integrallis:models:0.3.42
//DEPS com.integrallis:backend-java:0.3.42
//RUNTIME_OPTIONS --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED

import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.nio.file.Path;
import java.util.List;

/** Greedy product-of-experts fusion of two Qwen3 GGUFs that share one tokenizer. */
public class LogitFusion {
  public static void main(String[] args) {
    if (args.length < 3) {
      System.err.println("usage: LogitFusion <small.gguf> <large.gguf> <question> [maxTokens] [weightSmall]");
      System.exit(2);
    }
    int maxTokens = args.length > 3 ? Integer.parseInt(args[3]) : 48;
    double w = args.length > 4 ? Double.parseDouble(args[4]) : 0.5;
    try (var a = new InferencePipeline(PureJavaBackend.load(Path.of(args[0])));
        var b = new InferencePipeline(PureJavaBackend.load(Path.of(args[1])))) {
      var prompt = ChatTemplate.CHATML_NO_THINK.render(List.of(ChatMessage.user(args[2])));
      int[] tokens = a.tokenize(prompt); // G0 must have shown both tokenizers are identical
      float[] la = a.prefill(tokens, 0), lb = b.prefill(tokens, 0);
      int agree = 0, steps = 0;
      StringBuilder text = new StringBuilder();
      for (int pos = tokens.length; steps < maxTokens; pos++) {
        double[] pa = logSoftmax(la), pb = logSoftmax(lb);
        int best = 0, argA = 0, argB = 0;
        for (int t = 1; t < pa.length; t++) {
          if (w * pa[t] + (1 - w) * pb[t] > w * pa[best] + (1 - w) * pb[best]) best = t;
          if (pa[t] > pa[argA]) argA = t;
          if (pb[t] > pb[argB]) argB = t;
        }
        agree += argA == argB ? 1 : 0;
        steps++;
        System.out.printf("%3d fused=%-7d A=%-7d B=%-7d %s%n", steps, best, argA, argB,
            argA == argB ? "agree" : best == argA ? "took A" : best == argB ? "took B" : "neither");
        if (a.tokenizer().isEndOfGeneration(best)) break;
        text.append(a.tokenizer().decode(best));
        la = a.forward(best, pos);
        lb = b.forward(best, pos);
      }
      System.out.printf("%nfused text: %s%nmember argmax agreement: %d/%d%n", text, agree, steps);
    }
  }

  static double[] logSoftmax(float[] logits) {
    double max = Double.NEGATIVE_INFINITY, sum = 0;
    for (float x : logits) max = Math.max(max, x);
    for (float x : logits) sum += Math.exp(x - max);
    double norm = max + Math.log(sum);
    double[] out = new double[logits.length];
    for (int i = 0; i < logits.length; i++) out[i] = logits[i] - norm;
    return out;
  }
}
