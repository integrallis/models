import com.integrallis.models.api.SchemaClassificationModel;
import com.integrallis.models.backend.nativekernel.RustGlinerClassificationModel;
import java.nio.file.Path;
import java.util.List;

/** Exercises default native discovery from the released backend JAR. */
public class ReleasedGlinerSmoke {
  public static void main(String[] args) throws Exception {
    try (var model = RustGlinerClassificationModel.load(Path.of(args[0]))) {
      var task = new SchemaClassificationModel.Task("sentiment", List.of(
          new SchemaClassificationModel.Label("positive"),
          new SchemaClassificationModel.Label("negative"),
          new SchemaClassificationModel.Label("neutral")), false, 0.5);
      var results = model.classify(new SchemaClassificationModel.Request("I love this product!", List.of(task)));
      var selected = results.getFirst().labels().stream().filter(SchemaClassificationModel.LabelScore::selected)
          .map(SchemaClassificationModel.LabelScore::label).toList();
      if (!selected.equals(List.of("positive"))) throw new AssertionError(selected);
      System.out.println("{\"defaultNativeDiscovery\":true,\"selected\":[\"positive\"]}");
    }
  }
}
