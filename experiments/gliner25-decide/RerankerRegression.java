import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.backend.purejava.SafetensorsRerankingModel;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Remote-only repeat of the existing F16 DeBERTa regression oracle, with raw scores retained. */
public class RerankerRegression {
  public static void main(String[] args) throws Exception {
    // Identical fixtures and tolerance to MxbaiDebertaRerankerIntegrationTest on the base branch.
    var documents = List.of(
        "Berlin has a population of about 3.7 million people.",
        "The Eiffel Tower is located in Paris.",
        "Berlin is well known for its museums and its metropolitan area of about six million people.",
        "Domestic cats sleep for a large part of the day.",
        "New York City had an estimated population of 8,804,190 in 2020.",
        "The Berlin Wall divided the city from 1961 until 1989.");
    double[] expected = {2.610485553741455, -3.1909432411193848, 1.1550936698913574,
                        -3.9371538162231445, -3.6461875438690186, -1.0840436220169067};
    try (var model = SafetensorsRerankingModel.load(Path.of(args[0]))) {
      var scores = model.scoreAll("What is the population of Berlin?", documents);
      boolean passed = scores.size() == expected.length;
      double maxError = 0;
      for (int i=0; i<scores.size(); i++) {
        double error = Math.abs(scores.get(i)-expected[i]);
        maxError = Math.max(maxError, error);
        passed &= Double.isFinite(scores.get(i)) && error <= .001;
      }
      System.out.println(new ObjectMapper().writeValueAsString(Map.of(
          "stage", "existing-f16-reranker-regression", "passed", passed,
          "expectedLogits", expected, "actualLogits", scores, "maxAbsoluteError", maxError,
          "reference", "MxbaiDebertaRerankerIntegrationTest at 5d4de4262045afe825f295e022c9a64ce1e1e6e7")));
      if (!passed) throw new IllegalStateException("existing DeBERTa reranker regression");
    }
  }
}
