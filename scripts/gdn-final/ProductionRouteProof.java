package com.integrallis.models.backend.purejava.qwen35;

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
public final class ProductionRouteProof {
  static final ObjectMapper JSON = new ObjectMapper();
  static final com.sun.management.ThreadMXBean ALLOCATION =
      (com.sun.management.ThreadMXBean) java.lang.management.ManagementFactory.getThreadMXBean();
  static final java.lang.management.CompilationMXBean COMPILATION =
      java.lang.management.ManagementFactory.getCompilationMXBean();
  static final java.lang.management.ClassLoadingMXBean CLASS_LOADING =
      java.lang.management.ManagementFactory.getClassLoadingMXBean();
  static {
    if (!ALLOCATION.isThreadAllocatedMemorySupported())
      throw new AssertionError("Java allocation accounting unavailable");
    ALLOCATION.setThreadAllocatedMemoryEnabled(true);
    if (COMPILATION == null || !COMPILATION.isCompilationTimeMonitoringSupported())
      throw new AssertionError("JIT compilation accounting unavailable");
  }
  static long recurrenceCalls, groupedCalls;
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
    long compilationBefore = COMPILATION.getTotalCompilationTime();
    long classesBefore = CLASS_LOADING.getTotalLoadedClassCount();
    long allocatedBefore = ALLOCATION.getTotalThreadAllocatedBytes();
    long requestStart = System.nanoTime();
    try (InferenceSession session = backend.openSession()) {
      long start = System.nanoTime();
      float[] logits = backend.prefill(session, input, 0);
      prefill = System.nanoTime() - start;
      initial = logits.clone();
      snapshots.add(initial);
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
    long compilationAfter = COMPILATION.getTotalCompilationTime();
    long classesAfter = CLASS_LOADING.getTotalLoadedClassCount();
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
    row.put("compilationMillisBefore", compilationBefore);
    row.put("compilationMillisAfter", compilationAfter);
    row.put("compilationMillisDuringRequest", compilationAfter - compilationBefore);
    row.put("totalLoadedClassesBefore", classesBefore);
    row.put("totalLoadedClassesAfter", classesAfter);
    row.put("classesLoadedDuringRequest", classesAfter - classesBefore);
    row.put("modelNanos", prefill + decode); row.put("requestNanos", requestNanos); row.put("greedyTokens", tokens);
    row.put("fullLogitSha256", hashes); row.put("prediction", prediction);
    row.put("expected", task.path("expected").asText());
    emit(row);
    return row;
  }

  static GgufBatchedMatrixKernel observe(GgufBatchedMatrixKernel provider) {
    return (GgufBatchedMatrixKernel) Proxy.newProxyInstance(
        ProductionRouteProof.class.getClassLoader(), new Class<?>[] {GgufBatchedMatrixKernel.class},
        (proxy, method, args) -> {
          if (method.getName().equals("gatedDeltaNet")) recurrenceCalls++;
          if (method.getName().equals("groupedGatedDeltaNet")) groupedCalls++;
          try { return method.invoke(provider, args); }
          catch (InvocationTargetException failure) { throw failure.getCause(); }
        });
  }

  static boolean supportsRecurrence(GgufBatchedMatrixKernel provider) throws Exception {
    try { return (boolean) provider.getClass().getMethod("supportsGatedDeltaNet").invoke(provider); }
    catch (NoSuchMethodException removed) { return false; }
  }

  static void routing(Path model, Path nativePath, JsonNode tasks, boolean expectedNative) throws Exception {
    try (var provider = RustGgufBatchedMatrixKernel.open(nativePath);
         var backend = PureJavaBackend.load(model, WORKLOAD, observe(provider))) {
      if (supportsRecurrence(provider) != expectedNative) throw new AssertionError("wrong default capability");
      request(backend, tasks.get(0), 0, 8);
      if ((recurrenceCalls > 0) != expectedNative) throw new AssertionError("default recurrence route not exercised");
      emit(Map.of("kind", "initialRoute", "nativeExpected", expectedNative, "recurrenceCalls", recurrenceCalls));
      // Public prompts with different suffix lengths force a shrinking, noncontiguous set of slots.
      for (int size : new int[] {1, 3, 8}) {
        Map<Integer, List<int[]>> groups = new TreeMap<>();
        for (var task : tasks) {
          int[] tokens = backend.tokenizer().encode(task.path("prompt").asText());
          groups.computeIfAbsent(tokens[0], unused -> new ArrayList<>()).add(tokens);
        }
        List<int[]> selected = groups.values().stream()
            .max(Comparator.comparingInt(List::size)).orElseThrow();
        if (selected.size() < size) throw new AssertionError("insufficient public prompts sharing a prefix");
        int[][] input = new int[size][];
        for (int n = 0; n < size; n++) input[n] = size == 1
            ? backend.tokenizer().encode(tasks.get(0).path("prompt").asText()) : selected.get(n);
        int prefix = input[0].length;
        for (int n = 1; n < size; n++) {
          prefix = Math.min(prefix, input[n].length);
          for (int t = 0; t < prefix; t++) if (input[n][t] != input[0][t]) { prefix = t; break; }
        }
        prefix = Math.min(prefix, input[0].length - 1);
        if (prefix < 1) throw new AssertionError("public tasks have no shared prefix");
        List<String> expected = new ArrayList<>();
        for (int[] tokens : input) {
          backend.reset();
          expected.add(digest(backend.prefill(tokens, 0)));
        }
        backend.reset();
        backend.prefill(Arrays.copyOf(input[0], prefix), 0);
        int[][] suffixes = new int[size][];
        for (int n = 0; n < size; n++) suffixes[n] = Arrays.copyOfRange(input[n], prefix, input[n].length);
        float[][] actual = backend.decideGrouped(suffixes);
        List<String> observed = new ArrayList<>();
        for (float[] logits : actual) observed.add(digest(logits));
        emit(Map.of("kind", "group", "size", size, "sharedTokens", prefix,
            "suffixLengths", Arrays.stream(suffixes).mapToInt(a -> a.length).toArray(),
            "expected", expected, "actual", observed, "groupedCalls", groupedCalls));
        if (!expected.equals(observed)) throw new AssertionError("grouped execution changed full logits");
      }
      if ((groupedCalls > 0) != expectedNative) throw new AssertionError("default grouped route not exercised");
      emit(Map.of("kind", "route", "nativeExpected", expectedNative,
          "recurrenceCalls", recurrenceCalls, "groupedCalls", groupedCalls));
    }
  }

  static void eligibility(Path nativePath) throws Exception {
    try (var provider = RustGgufBatchedMatrixKernel.open(nativePath)) {
      boolean expected = provider.supportsGatedDeltaNet();
      for (int n = 0; n < 10000; n++) {
        provider.supportsGatedDeltaNet();
        com.integrallis.vectors.core.VectorUtil.runtimeCapabilities();
      }
      long thread = Thread.currentThread().threadId();
      for (int sample = 0; sample < 5; sample++) {
        long before = ALLOCATION.getThreadAllocatedBytes(thread);
        for (int n = 0; n < 100000; n++) {
          if (provider.supportsGatedDeltaNet() != expected)
            throw new AssertionError("cached eligibility changed");
        }
        long cachedBytes = ALLOCATION.getThreadAllocatedBytes(thread) - before;
        before = ALLOCATION.getThreadAllocatedBytes(thread);
        for (int n = 0; n < 100000; n++) {
          var capabilities = com.integrallis.vectors.core.VectorUtil.runtimeCapabilities();
          boolean fresh = capabilities.fastScalarFma()
              && (!capabilities.vectorApi() || capabilities.fastVectorFma());
          if (fresh != expected) throw new AssertionError("eligibility arithmetic decision changed");
        }
        long snapshotBytes = ALLOCATION.getThreadAllocatedBytes(thread) - before;
        emit(Map.of("kind", "eligibilityAllocation", "sample", sample, "calls", 100000,
            "nativeExpected", expected, "cachedBytes", cachedBytes, "snapshotBytes", snapshotBytes));
      }
    }
  }

  public static void main(String[] args) throws Exception {
    String phase = args[0];
    Path model = Path.of(args[1]), nativePath = Path.of(args[2]);
    if (phase.equals("eligibility")) { eligibility(nativePath); return; }
    JsonNode tasks = JSON.readTree(Path.of(args[3]).toFile());
    if (phase.equals("routing")) {
      routing(model, nativePath, tasks, Boolean.parseBoolean(args[4]));
      return;
    }
    long start = System.nanoTime();
    try (var backend = RustFfmBackend.load(model, nativePath, WORKLOAD)) {
      emit(Map.of("kind", "load", "nanos", System.nanoTime() - start,
          "diagnostics", backend.diagnostics(), "phase", phase));
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
        emit(Map.of("kind", "timingProtocol", "version", "fixed-warmup-32-v1",
            "warmupRequests", 32, "measuredRequests", 3,
            "compiler", COMPILATION.getName(), "anchorId", timed.get(0).path("id").asText()));
        // A fixed sweep exercises every declared timing prompt before two identical anchors.
        // Never change the warmup count after inspecting timing or allocation observations.
        for (int n = 0; n < timed.size(); n++) request(backend, timed.get(n), -32 + n, 8);
        var firstAnchor = request(backend, timed.get(0), -2, 8);
        var secondAnchor = request(backend, timed.get(0), -1, 8);
        emit(Map.of("kind", "warmupAnchors", "id", timed.get(0).path("id").asText(),
            "firstRepeat", -2, "secondRepeat", -1,
            "firstAllocatedBytes", firstAnchor.get("javaAllocatedBytes"),
            "secondAllocatedBytes", secondAnchor.get("javaAllocatedBytes"),
            "allocationPlateauExact", firstAnchor.get("javaAllocatedBytes").equals(secondAnchor.get("javaAllocatedBytes")),
            "firstCompilationMillis", firstAnchor.get("compilationMillisDuringRequest"),
            "secondCompilationMillis", secondAnchor.get("compilationMillisDuringRequest")));
        for (int n = 0; n < 3; n++) request(backend, timed.get((pair * 3 + n) % timed.size()), n, 8);
      } else throw new IllegalArgumentException(phase);
    }
  }
}
