/* Copyright 2026 Integrallis Software, LLC. SPDX-License-Identifier: Apache-2.0 */
import com.integrallis.models.api.*;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.runtime.GenerationLoop;
import com.integrallis.models.runtime.chat.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.util.*;

/** Diagnostic only: retains raw answers, prompt token IDs and termination evidence. */
public class Lfm3Generate {
  public static void main(String[] args) throws Exception {
    if (args.length != 4) throw new IllegalArgumentException("MODEL MODE OUTPUT MAX_TOKENS");
    System.setProperty("models.purejava.maxContextLength", "2048");
    var mapper = new ObjectMapper();
    var results = new ArrayList<Map<String,Object>>();
    try (InferenceBackend backend = args[1].equals("rust")
        ? RustFfmBackend.load(Path.of(args[0])) : PureJavaBackend.load(Path.of(args[0]))) {
      for (String question : List.of("What is 2 + 2? Reply with just the number.",
          "Context: The museum closes at 6 PM on Fridays. Question: When does the museum close on Friday? Answer using only the context.",
          "Context: The museum closes at 6 PM on Fridays. Question: How much does admission cost? If the context does not say, reply UNKNOWN.")) {
        backend.reset();
        var prompt = ChatTemplate.LFM3_NO_THINK.render(List.of(ChatMessage.user(question)));
        var result = new LinkedHashMap<String,Object>();
        result.put("question",question);result.put("prompt",prompt.text());
        result.put("promptTokens",backend.tokenizer().encode(prompt));
        result.put("backend",backend.name());result.put("template","lfm3-no-think");
        var answer = new StringBuilder();long start=System.nanoTime();
        new GenerationLoop(backend).generate(prompt,
            SamplingOptions.builder().temperature(0).maxTokens(Integer.parseInt(args[3])).build(),
            new TokenStream() {
              public void onToken(String token) { answer.append(token); }
              public void onComplete() {}
              public void onComplete(GenerationUsage usage,StopReason reason) {
                result.put("usage",usage);result.put("stopReason",reason.name());
              }
              public void onError(Throwable failure) {throw new RuntimeException(failure);}
            });
        result.put("elapsedMs",(System.nanoTime()-start)/1e6);result.put("output",answer.toString());
        results.add(result);mapper.writerWithDefaultPrettyPrinter().writeValue(Path.of(args[2]).toFile(),results);
        System.out.println(mapper.writeValueAsString(result));
      }
    }
  }
}
