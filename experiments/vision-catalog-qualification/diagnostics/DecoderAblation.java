import com.fasterxml.jackson.core.JsonFactory;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufMatrixKernelProvider;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.ServiceLoader;

/** External diagnostic: the shipping native decoder with independently supplied image tokens. */
class DecoderAblation {
  public static void main(String[] args) throws Exception {
    if (!System.getProperty("os.name").equals("Linux") || args.length != 5)
      throw new IllegalArgumentException("Remote Linux: MODEL PREFIX.i32 SUFFIX.i32 EMBEDDINGS.f32 OUTPUT.json");
    Path output = Path.of(args[4]);
    if (Files.exists(output)) throw new IllegalArgumentException("Preserve prior diagnostics");
    var provider = ServiceLoader.load(GgufMatrixKernelProvider.class).findFirst().orElseThrow();
    var kernel = provider.open(8).orElseThrow();
    var attention = provider.openAttention(8, kernel);
    int[] prefix = ints(Path.of(args[1]));
    int[] suffix = ints(Path.of(args[2]));
    var floats = ByteBuffer.wrap(Files.readAllBytes(Path.of(args[3]))).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
    float[] embeddings = new float[floats.remaining()];
    floats.get(embeddings);
    try (var backend = PureJavaBackend.load(Path.of(args[0]), kernel, attention);
         var json = new JsonFactory().createGenerator(output.toFile(), com.fasterxml.jackson.core.JsonEncoding.UTF8)) {
      json.useDefaultPrettyPrinter();
      backend.prefill(prefix, 0);
      backend.prefillEmbeddings(embeddings, prefix.length);
      int position = prefix.length + embeddings.length / backend.embeddingInputDimension();
      float[] logits = backend.prefill(suffix, position);
      position += suffix.length;
      ArrayList<Integer> tokens = new ArrayList<>();
      json.writeStartObject();
      json.writeStringField("embeddingPath", args[3]);
      json.writeStringField("candidateRevision", System.getProperty("models.qualification.revision"));
      json.writeArrayFieldStart("trace");
      for (int step = 0; step < 32; step++) {
        int best = 0, second = -1;
        for (int i = 0; i < logits.length; i++) {
          if (!Float.isFinite(logits[i])) throw new IllegalStateException("Nonfinite logits");
          if (logits[i] > logits[best]) { second = best; best = i; }
          else if (i != best && (second < 0 || logits[i] > logits[second])) second = i;
        }
        json.writeStartObject();
        json.writeNumberField("step", step);
        json.writeNumberField("topToken", best);
        json.writeNumberField("secondToken", second);
        json.writeNumberField("margin", logits[best] - logits[second]);
        json.writeEndObject();
        if (backend.tokenizer().isEndOfGeneration(best)) break;
        tokens.add(best);
        if (step < 31) logits = backend.forward(best, position++);
      }
      json.writeEndArray();
      json.writeArrayFieldStart("tokens");
      for (int token : tokens) json.writeNumber(token);
      json.writeEndArray();
      json.writeStringField("answer", backend.tokenizer().decode(tokens.stream().mapToInt(Integer::intValue).toArray()));
      json.writeEndObject();
    }
  }

  private static int[] ints(Path path) throws Exception {
    var values = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
    int[] result = new int[values.remaining()];
    values.get(result);
    return result;
  }
}
