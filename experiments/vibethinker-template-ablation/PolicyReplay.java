// Replays every generation in a committed candidate report through the real grounding policy, and
// reports which ones the policy accepts -- and which ones it would accept with the model's shown
// working removed.
//
// Why this exists. The grid says VibeThinker 1.5B produces almost no accepted model answer on
// `math`, and the cap explains only part of it: under `chatml-no-think` at cap 1792, 7 of 9
// generations finished on their own and none were accepted. That leaves the question of *why* a
// finished, correct answer is rejected, and the answer must come from the policy itself rather
// than from reading its source and inferring.
//
// The strings are read from the report, never typed here: the question, the retrieved documents
// and the raw generation all come out of the committed artifact, so this cannot drift from what
// was measured.
//
// Run (from the models checkout, after `./gradlew :models-rag-bench:installDist`):
//   CP=$(ls models-rag-bench/build/install/models-rag-bench/lib/*.jar | tr '\n' ':')
//   javac -cp "$CP" -d /tmp/replay experiments/vibethinker-template-ablation/PolicyReplay.java
//   java -cp "$CP:/tmp/replay" PolicyReplay experiments/vibethinker-template-ablation/evidence/*.json
import com.fasterxml.jackson.databind.*;
import java.io.InputStream;
import com.integrallis.models.rag.*;
import java.nio.file.*;
import java.util.*;

public class PolicyReplay {

  // The model's shown working is what the policy screens out: a derivation step is a claim, and the
  // retrieved document states only the result. This strips the lines that carry working or
  // commentary and keeps the ones that state an answer, so the two can be scored separately.
  private static String answerOnly(String raw) {
    String tail = raw;
    int marker = Math.max(raw.lastIndexOf("</think>"), raw.lastIndexOf("ANSWER"));
    if (marker >= 0) {
      int from = raw.startsWith("ANSWER", marker) ? marker + "ANSWER".length() : marker + "</think>".length();
      tail = raw.substring(from);
    }
    List<String> kept = new ArrayList<>();
    for (String line : tail.split("\\R")) {
      String t = line.strip();
      if (t.isEmpty()) continue;
      if (t.contains("=") || t.contains("*") || t.matches(".*\\bcalculated\\b.*")) continue;
      if (t.startsWith("This ") || t.startsWith("The answer should") || t.contains("without adding")) continue;
      kept.add(t);
    }
    return String.join(" ", kept).strip();
  }

  // The question text is not stored on a run, and the policy reads it -- `hasOnlySupportedClaims`
  // and `hasEvidenceAnchorsForEveryQuestionClause` both take it. Passing the case id instead
  // changed three decisions and made this replay disagree with the report it was replaying, so the
  // questions are loaded from the corpus the run used and the disagreement is now a hard failure.
  private static Map<String, String> questions(ObjectMapper mapper, String workload)
      throws Exception {
    try (InputStream in = PolicyReplay.class.getResourceAsStream(
        "/rag/" + workload + "/cases.json")) {
      if (in == null) throw new IllegalStateException("no cases.json for workload " + workload);
      Map<String, String> byId = new LinkedHashMap<>();
      for (JsonNode c : mapper.readTree(in)) byId.put(c.get("id").asText(), c.get("question").asText());
      return byId;
    }
  }

  public static void main(String[] args) throws Exception {
    var mapper = new ObjectMapper();
    var policy = GroundedAnswerPolicy.productionDefault();
    int totalFinished = 0, acceptedAsIs = 0, acceptedStripped = 0;

    for (String path : args) {
      JsonNode report = mapper.readTree(Files.readString(Path.of(path)));
      int cap = report.at("/settings/maxOutputTokens").asInt();
      String tpl = report.at("/settings/promptTemplate").asText();
      var questionById = questions(mapper, report.at("/settings/workload").asText());
      System.out.printf("=== %s @ cap %d  (%s)%n", tpl, cap, Path.of(path).getFileName());

      for (JsonNode run : report.get("runs")) {
        String caseId = run.get("caseId").asText();
        int outputTokens = run.at("/generation/outputTokens").asInt();
        boolean capped = outputTokens >= cap;
        String raw = run.at("/grounding/rawText").asText();

        List<GroundingDocument> docs = new ArrayList<>();
        for (JsonNode hit : run.get("retrieved")) {
          docs.add(new GroundingDocument(
              hit.at("/document/id").asText(), hit.at("/document/title").asText(),
              hit.at("/document/text").asText(),
              (float) hit.get("score").asDouble(), hit.get("rank").asInt()));
        }
        String question = Objects.requireNonNull(
            questionById.get(caseId), "no question for case " + caseId);

        var asIs = policy.apply(question, docs, raw);
        String recorded = run.at("/grounding/decision").asText();
        if (!asIs.decision().name().equals(recorded)) {
          throw new IllegalStateException("replay is not faithful for " + caseId + ": policy says "
              + asIs.decision() + " but the report recorded " + recorded
              + ". The replay inputs no longer match the measured run.");
        }
        String stripped = answerOnly(raw);
        var withoutWorking = stripped.isBlank() ? null : policy.apply(question, docs, stripped);

        if (!capped) {
          totalFinished++;
          if (asIs.decision().modelContributed()) acceptedAsIs++;
          if (withoutWorking != null && withoutWorking.decision().modelContributed()) acceptedStripped++;
        }
        System.out.printf("  %-28s %5d %-8s as-is=%-34s stripped=%s%n",
            caseId, outputTokens, capped ? "CAPPED" : "finished",
            asIs.decision(), withoutWorking == null ? "(empty)" : withoutWorking.decision());
      }
      System.out.println();
    }
    System.out.printf("Of %d generations that finished before the cap: %d accepted as produced, "
        + "%d accepted with the shown working removed.%n",
        totalFinished, acceptedAsIs, acceptedStripped);
  }
}
