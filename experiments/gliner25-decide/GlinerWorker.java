import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.SchemaClassificationModel;
import com.integrallis.models.backend.purejava.gliner.GlinerClassificationModel;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Qualification worker; stdout is reserved for JSONL. No model is downloaded by this class. */
public class GlinerWorker {
  static final ObjectMapper JSON = new ObjectMapper();

  public static void main(String[] args) throws Exception {
    Path snapshot = Path.of(args[0]);
    Path nativeLibrary = Path.of(args[1]);
    JsonNode manifest = JSON.readTree(Path.of(args[2]).toFile());
    for (JsonNode file : manifest.get("files")) {
      Path path = snapshot.resolve(file.get("path").asText());
      if (Files.size(path) != file.get("sizeBytes").asLong()
          || !digest(path).equals(file.get("sha256").asText())) {
        throw new IllegalStateException("snapshot identity mismatch: " + path);
      }
    }
    long start = System.nanoTime();
    try (var model = com.integrallis.models.backend.nativekernel.RustGlinerClassificationModel.load(snapshot, nativeLibrary);
         var input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
      emit(Map.of("type", "ready", "backend", "rust-ffm-gliner", "snapshot", manifest,
          "device", List.of("cpu"), "dtype", List.of("float32"),
          "threads", Integer.parseInt(System.getProperty("models.deberta.threads")),
          "nativeSha256", digest(nativeLibrary), "javaVersion", System.getProperty("java.version"),
          "loadMs", (System.nanoTime() - start) / 1e6));
      String line;
      while ((line = input.readLine()) != null) {
        JsonNode envelope = JSON.readTree(line);
        String id = envelope.get("id").asText();
        try {
          var request = JSON.treeToValue(envelope.get("request"), SchemaClassificationModel.Request.class);
          boolean diagnostic = envelope.path("mode").asText().equals("diagnostic");
          var output = new LinkedHashMap<String, Object>();
          output.put("id", id);
          List<SchemaClassificationModel.Result> result;
          start = System.nanoTime();
          if (diagnostic) {
            var compiled = model.compile(request);
            float[] logits = model.logits(request);
            result = GlinerClassificationModel.decode(request, logits);
            output.put("tokenIds", compiled.tokenIds());
            output.put("markerPositions", compiled.markerPositions());
            output.put("logits", logits);
          } else { result = model.classify(request); }
          output.put("elapsedMs", (System.nanoTime() - start) / 1e6);
          var selected = new LinkedHashMap<String, List<String>>();
          for (var task : result) {
            selected.put(task.task(), task.labels().stream().filter(SchemaClassificationModel.LabelScore::selected)
                .map(SchemaClassificationModel.LabelScore::label).toList());
          }
          output.put("selected", selected);
          output.put("raw", result);
          output.put("denseProjectionCalls", model.projectionCalls());
          output.put("attentionCalls", model.attentionCalls());
          emit(output);
        } catch (Exception failure) {
          failure.printStackTrace(System.err);
          emit(Map.of("id", id, "error", failure.toString()));
        }
      }
    }
  }

  static void emit(Object value) throws Exception { System.out.println(JSON.writeValueAsString(value)); }

  static String digest(Path path) throws Exception {
    MessageDigest sha = MessageDigest.getInstance("SHA-256");
    try (var in = new DigestInputStream(Files.newInputStream(path), sha)) {
      in.transferTo(java.io.OutputStream.nullOutputStream());
    }
    return HexFormat.of().formatHex(sha.digest());
  }
}
