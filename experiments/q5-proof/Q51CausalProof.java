package experiments.q5;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.integrallis.models.api.BatchInferenceBackend;
import com.integrallis.models.api.BackendConfiguration;
import com.integrallis.models.api.InferenceSession;
import com.integrallis.models.backend.nativekernel.RustFfmBackend;
import com.integrallis.models.backend.nativekernel.RustGgufBatchedMatrixKernel;
import com.integrallis.models.backend.purejava.PureJavaBackend;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;

/** Separate processes select immutable library artifacts, never an inference-route flag. */
public final class Q51CausalProof {
  static final ObjectMapper JSON = new ObjectMapper();
  static final com.sun.management.ThreadMXBean ALLOCATION =
      (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();
  static {
    if (!ALLOCATION.isThreadAllocatedMemorySupported())
      throw new AssertionError("Java allocation accounting unavailable");
    ALLOCATION.setThreadAllocatedMemoryEnabled(true);
  }
  
  static final BackendConfiguration WORKLOAD = new BackendConfiguration(
      Map.of("gdn-qualification-context-budget", "2048"),
      Map.of("models.purejava.maxContextLength", "2048"), List.of());

  static void emit(Map<String, ?> row) throws Exception {
    System.out.println(JSON.writeValueAsString(row));
  }

  static String digest(float[] values) throws Exception {
    ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      if (!Float.isFinite(value)) throw new AssertionError("non-finite logit");
      bytes.putInt(Float.floatToRawIntBits(value));
    }
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes.array()));
  }

  static int argmax(float[] values) {
    int best = 0;
    for (int n = 1; n < values.length; n++) if (values[n] > values[best]) best = n;
    return best;
  }

  static Map<String, Object> request(BatchInferenceBackend backend, JsonNode task, int repeat, int steps)
      throws Exception {
    int[] input = backend.tokenizer().encode(task.path("prompt").asText());
    List<String> hashes = new ArrayList<>();
    List<float[]> snapshots = new ArrayList<>();
    List<Integer> tokens = new ArrayList<>();
    long prefill, decode = 0;
    float[] initial;
    long allocatedBefore = ALLOCATION.getTotalThreadAllocatedBytes();
    long requestStart = System.nanoTime();
    try (InferenceSession session = backend.openSession()) {
      long start = System.nanoTime();
      float[] logits = backend.prefill(session, input, 0);
      prefill = System.nanoTime() - start;
      initial = logits.clone();
      snapshots.add(logits.clone());
      for (int n = 0; n < steps; n++) {
        int token = argmax(logits);
        tokens.add(token);
        start = System.nanoTime();
        logits = backend.forward(session, token, input.length + n);
        decode += System.nanoTime() - start;
        snapshots.add(logits.clone());
      }
    }
    long requestNanos = System.nanoTime() - requestStart;
    long javaAllocatedBytes = ALLOCATION.getTotalThreadAllocatedBytes() - allocatedBefore;
    if (allocatedBefore < 0 || javaAllocatedBytes < 0) throw new AssertionError("invalid allocation counter");
    for (float[] snapshot : snapshots) hashes.add(digest(snapshot));
    String prediction = "";
    float best = -Float.MAX_VALUE;
    for (int n = 0; n < task.path("labels").size(); n++) {
      int[] label = backend.tokenizer().encode(" " + (char) ('A' + n));
      float score = initial[label[label.length - 1]];
      if (score > best) { best = score; prediction = task.path("labels").get(n).asText(); }
    }
    Map<String, Object> row = new LinkedHashMap<>();
    row.put("kind", "request"); row.put("id", task.path("id").asText());
    row.put("dataset", task.path("dataset").asText()); row.put("repeat", repeat);
    row.put("promptTokens", input.length); row.put("inputTokens", input);
    row.put("prefillNanos", prefill); row.put("decodeNanos", decode);
    row.put("javaAllocatedBytes", javaAllocatedBytes);
    row.put("modelNanos", prefill + decode); row.put("requestNanos", requestNanos); row.put("greedyTokens", tokens);
    row.put("fullLogitSha256", hashes); row.put("prediction", prediction);
    row.put("expected", task.path("expected").asText());
    emit(row);
    return row;
  }

  static void routing(Path model, Path nativePath, JsonNode tasks, boolean expectedGrouped) throws Exception {
    long[] calls=new long[2]; Map<Integer,Long> batches=new TreeMap<>();
    try(var provider=RustGgufBatchedMatrixKernel.open(nativePath)) {
      GgufBatchedMatrixKernel observer=(GgufBatchedMatrixKernel) Proxy.newProxyInstance(Q51CausalProof.class.getClassLoader(),new Class<?>[]{GgufBatchedMatrixKernel.class},(proxy,method,args)->{
        if(method.getName().equals("multiply") && args[3] == com.integrallis.models.backend.purejava.gguf.GgufTensorType.Q5_1) {calls[0]++;batches.merge((Integer)args[4],1L,Long::sum);}
        
        try{return method.invoke(provider,args);}catch(InvocationTargetException failure){throw failure.getCause();}
      });
      try(var backend=PureJavaBackend.load(model,WORKLOAD,observer)){
        request(backend,tasks.get(0),0,8);
        emit(Map.of("kind","routing","q51Calls",calls[0],"batches",batches));
        if (batches.keySet().stream().anyMatch(batch -> batch != 1)) throw new AssertionError("native prefill still active");
        if((calls[0]+calls[1]>0)!=expectedGrouped)throw new AssertionError("actual Q5_1 single projection route was not exercised");
      }
    }
  }

  public static void main(String[] args) throws Exception {
    String phase = args[0];
    Path model = Path.of(args[1]), nativePath = Path.of(args[2]);
    JsonNode tasks = JSON.readTree(Path.of(args[3]).toFile());
    if (phase.equals("routing")) {
      routing(model, nativePath, tasks, Boolean.parseBoolean(args[4]));
      return;
    }
    long start = System.nanoTime();
    String arm = args[5];
    try (BatchInferenceBackend backend = arm.equals("java") ? PureJavaBackend.load(model, WORKLOAD) : RustFfmBackend.load(model, nativePath, WORKLOAD)) {
      emit(Map.of("kind", "load", "nanos", System.nanoTime() - start,
          "arm", arm, "phase", phase));
      if (phase.equals("cohort")) {
        for (int n = 0; n < tasks.size(); n++) request(backend, tasks.get(n), 0, 8);
      } else if (phase.equals("timing")) {
        int pair = Integer.parseInt(args[4]);
        List<JsonNode> jevbench = new ArrayList<>(), snips = new ArrayList<>();
        for (var task : tasks) {
          if (task.path("dataset").asText().equals("jevbench")) jevbench.add(task);
          else if (task.path("dataset").asText().equals("snips")) snips.add(task);
          else throw new AssertionError("undeclared timing dataset");
        }
        List<JsonNode> timed = new ArrayList<>();
        for (int n = 0; n < 15; n++) { timed.add(jevbench.get(n)); timed.add(snips.get(n)); }
        for (int n = 0; n < 32; n++) request(backend, timed.get(n % timed.size()), n-32, 8);
        for (int n = 0; n < 3; n++) request(backend, timed.get((pair * 3 + n) % timed.size()), n, 8);
      } else throw new IllegalArgumentException(phase);
    }
  }
}
