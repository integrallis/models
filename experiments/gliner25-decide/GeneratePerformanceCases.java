import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.SchemaClassificationModel;
import com.integrallis.models.backend.nativekernel.NativeHuggingFaceTokenizer;
import com.integrallis.models.backend.purejava.gliner.GlinerSchemaCompiler;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;

/** Metadata-only deterministic shape matrix; never opens model weights. */
public class GeneratePerformanceCases {
  static final List<String> NAMES = List.of(
      "finance", "travel", "food", "sports", "music", "news", "science", "health",
      "education", "technology", "weather", "politics", "business", "art", "books", "games",
      "history", "law", "nature", "fashion", "family", "pets", "cars", "housing",
      "employment", "shopping", "movies", "fitness", "banking", "insurance", "security", "transport");

  public static void main(String[] args) throws Exception {
    var json = new ObjectMapper();
    try (var tokenizer = NativeHuggingFaceTokenizer.load(Path.of(args[0]), Path.of(args[1]))) {
      var compiler = new GlinerSchemaCompiler(tokenizer::encodePiece, 4096);
      for (int tokens : new int[]{32, 128, 256, 512}) {
        for (int labels : new int[]{2, 8, 32}) {
          emitBucket(json, compiler, tokens, labels, 1);
        }
      }
      emitBucket(json, compiler, 128, 8, 2);
      emitBucket(json, compiler, 512, 8, 2);
    }
  }

  static void emitBucket(ObjectMapper json, GlinerSchemaCompiler compiler,
                         int tokens, int labels, int taskCount) throws Exception {
    String bucket = "tokens"+tokens+"-labels"+labels+"-tasks"+taskCount;
    for (int sample = 0; sample < 33; sample++) {
      var tasks = new ArrayList<SchemaClassificationModel.Task>();
      var names = new ArrayList<>(NAMES.subList(0, labels));
      Collections.rotate(names, sample % labels);
      var choices = names.stream().map(SchemaClassificationModel.Label::new).toList();
      for (int task = 0; task < taskCount; task++) {
        tasks.add(new SchemaClassificationModel.Task(task == 0 ? "topic" : "category", choices, false, 0.5));
      }
      String text = "The report discusses the latest news.";
      int base = compiler.compile(new SchemaClassificationModel.Request(text, tasks)).tokenIds().length;
      if (base > tokens) {
        if (sample != 0) throw new IllegalStateException("shape changed with permutation");
        System.err.println("unsupported shape (schema exceeds budget): "+bucket);
        return;
      }
      // Each standalone 'the' is one token in the pinned tokenizer. Assert, never assume.
      text = "the ".repeat(tokens-base) + text;
      var request = new SchemaClassificationModel.Request(text, tasks);
      var encoded = compiler.compile(request);
      if (encoded.tokenIds().length != tokens) throw new IllegalStateException("wrong encoded length");
      var row = new LinkedHashMap<String, Object>();
      row.put("id", bucket+"-"+sample); row.put("bucket", bucket);
      row.put("phase", sample < 3 ? "warmup" : "timed");
      row.put("encodedTokens", tokens); row.put("labelsPerTask", labels);
      row.put("taskCount", taskCount); row.put("batchSize", 1); row.put("concurrency", 1);
      row.put("request", request);
      row.put("tokenIds", encoded.tokenIds());
      System.out.println(json.writeValueAsString(row));
    }
  }
}
