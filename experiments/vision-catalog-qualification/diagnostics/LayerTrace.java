import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ServiceLoader;

/** Remote-only diagnostic. Uses the existing layer observer, never a production inference mode. */
class LayerTrace {
  public static void main(String[] args) throws Exception {
    if (!System.getProperty("os.name").equals("Linux") || args.length != 5)
      throw new IllegalArgumentException("MODEL PREFIX.i32 SUFFIX.i32 EMBEDDINGS.f32 NEW_OUTPUT_DIR");
    Path output = Path.of(args[4]);
    Files.createDirectory(output);
    int[] prefix = ints(Path.of(args[1]));
    int[] suffix = ints(Path.of(args[2]));
    var input = ByteBuffer.wrap(Files.readAllBytes(Path.of(args[3]))).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    float[] embeddings = new float[input.remaining()];
    input.get(embeddings);
    var provider = ServiceLoader.load(GgufMatrixKernelProvider.class).findFirst().orElseThrow();
    var kernel = provider.open(8).orElseThrow();
    var attention = provider.openAttention(8, kernel);
    try (var backend = PureJavaBackend.load(Path.of(args[0]), kernel, attention)) {
      int target = prefix.length + embeddings.length / backend.embeddingInputDimension() + suffix.length - 1;
      installObserver(backend, output, target);
      backend.prefill(prefix, 0);
      backend.prefillEmbeddings(embeddings, prefix.length);
      float[] logits = backend.prefill(suffix, target + 1 - suffix.length);
      // First prediction only: both arms have identical complete inputs at this boundary.
      floats(output.resolve("logits.f32"), logits, 0, logits.length);
      int best = 0;
      for (int i = 0; i < logits.length; i++) {
        if (!Float.isFinite(logits[i])) throw new IllegalStateException("Nonfinite logits");
        if (logits[i] > logits[best]) best = i;
      }
      Files.writeString(output.resolve("prediction.json"), "{\"position\":" + target
          + ",\"topToken\":" + best + ",\"logit\":" + logits[best] + "}\n");
    }
  }

  private static void installObserver(PureJavaBackend backend, Path output, int target) throws Exception {
    var decoderField = PureJavaBackend.class.getDeclaredField("decoder");
    decoderField.setAccessible(true);
    Object decoder = decoderField.get(backend);
    Object pass = null;
    for (var field : decoder.getClass().getDeclaredFields()) {
      field.setAccessible(true);
      Object value = field.get(decoder);
      if (value != null && value.getClass().getName().endsWith("llama.LlamaForwardPass")) pass = value;
    }
    if (pass == null) throw new IllegalStateException("No Llama forward pass");
    var observerField = pass.getClass().getDeclaredField("layerObserver");
    observerField.setAccessible(true);
    Class<?> observerType = observerField.getType();
    observerField.set(pass, Proxy.newProxyInstance(observerType.getClassLoader(), new Class<?>[] {observerType},
        (proxy, method, values) -> {
          if (method.getName().equals("onLayerComplete") && (int) values[1] == target)
            floats(output.resolve("layer-" + values[0] + ".f32"), (float[]) values[2],
                (int) values[3], (int) values[4]);
          return method.getName().equals("toString") ? "LayerTrace" : null;
        }));
  }

  private static void floats(Path path, float[] values, int offset, int length) throws Exception {
    var bytes = ByteBuffer.allocate(length * 4).order(ByteOrder.LITTLE_ENDIAN);
    bytes.asFloatBuffer().put(values, offset, length);
    Files.write(path, bytes.array());
  }

  private static int[] ints(Path path) throws Exception {
    var values = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
    int[] result = new int[values.remaining()];
    values.get(result);
    return result;
  }
}
