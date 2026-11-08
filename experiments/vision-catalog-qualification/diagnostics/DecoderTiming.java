import com.fasterxml.jackson.core.JsonEncoding;
import com.fasterxml.jackson.core.JsonFactory;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.ServiceLoader;

/** Controlled decoder-only A/B; qualification still requires the full public vision workload. */
class DecoderTiming {
  public static void main(String[] args) throws Exception {
    if (!System.getProperty("os.name").equals("Linux") || args.length != 5)
      throw new IllegalArgumentException("MODEL PREFIX.i32 SUFFIX.i32 EMBEDDINGS.f32 NEW_OUTPUT.json");
    Path output = Path.of(args[4]);
    if (Files.exists(output)) throw new IllegalArgumentException("Preserve existing results");
    int[] prefix = ints(Path.of(args[1])), suffix = ints(Path.of(args[2]));
    var input = ByteBuffer.wrap(Files.readAllBytes(Path.of(args[3]))).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    float[] embeddings = new float[input.remaining()]; input.get(embeddings);
    var provider = ServiceLoader.load(GgufMatrixKernelProvider.class).findFirst().orElseThrow();
    var kernel = provider.open(8).orElseThrow();
    var attention = provider.openAttention(8, kernel);
    try (var backend = PureJavaBackend.load(Path.of(args[0]), kernel, attention);
         var json = new JsonFactory().createGenerator(output.toFile(), JsonEncoding.UTF8)) {
      json.useDefaultPrettyPrinter();
      json.writeStartObject();
      json.writeStringField("revision", System.getProperty("models.qualification.revision"));
      json.writeStringField("nativeLibrary", System.getProperty("models.native.kernels.library"));
      json.writeNumberField("threads", 8); json.writeNumberField("warmups", 3); json.writeNumberField("repetitions", 7);
      json.writeArrayFieldStart("runs");
      for (int run = -3; run < 7; run++) {
        backend.reset();
        long start = System.nanoTime();
        backend.prefill(prefix, 0);
        backend.prefillEmbeddings(embeddings, prefix.length);
        int position = prefix.length + embeddings.length / backend.embeddingInputDimension();
        float[] logits = backend.prefill(suffix, position); position += suffix.length;
        double prefill = (System.nanoTime()-start)/1e6;
        ArrayList<Integer> tokens = new ArrayList<>();
        start = System.nanoTime();
        for (int step = 0; step < 32; step++) {
          int best = 0;
          for (int i = 0; i < logits.length; i++) {
            if (!Float.isFinite(logits[i])) throw new IllegalStateException("Nonfinite logits");
            if (logits[i] > logits[best]) best = i;
          }
          if (backend.tokenizer().isEndOfGeneration(best)) break;
          tokens.add(best);
          if (step < 31) logits = backend.forward(best, position++);
        }
        double decode = (System.nanoTime()-start)/1e6;
        json.writeStartObject(); json.writeNumberField("run", run);
        json.writeNumberField("prefillMs", prefill); json.writeNumberField("decodeMs", decode);
        json.writeArrayFieldStart("tokens"); for (int token : tokens) json.writeNumber(token); json.writeEndArray();
        json.writeEndObject(); json.flush();
        System.out.printf("run=%d prefill=%.3f decode=%.3f tokens=%d%n", run, prefill, decode, tokens.size());
      }
      json.writeEndArray(); json.writeEndObject();
    }
  }

  private static int[] ints(Path path) throws Exception {
    var values = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
    int[] result = new int[values.remaining()]; values.get(result); return result;
  }
}
