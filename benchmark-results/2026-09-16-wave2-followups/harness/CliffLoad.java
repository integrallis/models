import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Path;
public class CliffLoad {
  public static void main(String[] a) throws Exception {
    try (PureJavaBackend b = PureJavaBackend.load(Path.of(a[0]))) {
      b.forward(1, 0);
      b.diagnostics().environment().entrySet().stream()
        .filter(e -> e.getKey().startsWith("performance-cliff") || e.getKey().startsWith("gguf-") || e.getKey().startsWith("end-of-generation") || e.getKey().equals("processors"))
        .sorted(java.util.Map.Entry.comparingByKey())
        .forEach(e -> System.out.println("  " + e.getKey() + "=" + e.getValue()));
    }
  }
}
