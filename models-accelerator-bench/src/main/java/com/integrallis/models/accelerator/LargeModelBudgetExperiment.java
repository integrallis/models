/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.accelerator;

import com.integrallis.models.backend.tornado.AcceleratorEligibility;
import com.integrallis.models.backend.tornado.DeviceBudget;
import com.integrallis.models.backend.tornado.DeviceMemoryRequest;
import com.integrallis.models.backend.tornado.PlanShapeStrategy;
import com.integrallis.models.backend.tornado.TornadoRuntimeDevices;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Prints the accelerator device budget for a model shape against a table of devices.
 *
 * <p>Runs without a GPU: the device table is data, so a new question about a new device or context
 * length never needs a recompile. When a TornadoVM runtime is present its discovered devices are
 * appended to the table.
 *
 * <p>Example:
 *
 * <pre>{@code
 * LargeModelBudgetExperiment --preset gemma4-26b-a4b --context 4096,32768,262144 \
 *     --device "A40 24 GB:23:8" --device "L40S 48 GB:44:8" --device "H100 80 GB:79:32"
 * }</pre>
 */
public final class LargeModelBudgetExperiment {

  private static final long MIB = 1024L * 1024L;
  private static final long GIB = 1024L * MIB;

  private LargeModelBudgetExperiment() {}

  public static void main(String[] args) {
    Map<String, String> options = parse(args);
    Shape shape = shape(options);
    List<Integer> contexts = contexts(options);
    List<AcceleratorEligibility.DeviceCapabilities> devices = devices(options);

    System.out.printf(
        Locale.ROOT,
        "model=%s weights=%s shapes=%d plans=%d planScratch=%s largestAllocation=%s%n",
        shape.label(),
        DeviceBudget.gib(shape.weightBytes()),
        shape.retainedShapes(),
        shape.retainedPlanCount(),
        DeviceBudget.gib(shape.planScratchBytes()),
        DeviceBudget.gib(shape.largestAllocationBytes()));
    System.out.println();

    for (int context : contexts) {
      DeviceMemoryRequest request = shape.request(context);
      DeviceBudget shipped =
          AcceleratorEligibility.budget(request, PlanShapeStrategy.PER_SHAPE_WHOLE_MODEL);
      DeviceBudget shared =
          AcceleratorEligibility.budget(request, PlanShapeStrategy.SHARED_WEIGHT_UPLOAD);
      System.out.printf(
          Locale.ROOT,
          "context=%d kv=%s shipped=%s shared=%s hostCopies=%s readiness=%d s%n",
          context,
          DeviceBudget.gib(request.deviceKvCacheBytes()),
          DeviceBudget.gib(shipped.totalBytes()),
          DeviceBudget.gib(shared.totalBytes()),
          DeviceBudget.gib(shipped.hostWeightCopyBytes()),
          shipped.estimatedPlanCompileTime().toSeconds());
      for (AcceleratorEligibility.DeviceCapabilities device : devices) {
        AcceleratorEligibility.Decision decision =
            AcceleratorEligibility.select(List.of(device), request);
        System.out.printf(
            Locale.ROOT,
            "    %-16s global=%s eligible=%s %s%n",
            device.name(),
            DeviceBudget.gib(device.globalMemoryBytes()),
            decision.eligible(),
            decision.eligible() ? "" : decision.reason());
      }
      System.out.println();
    }
  }

  private record Shape(
      String label,
      long weightBytes,
      int retainedShapes,
      int retainedPlanCount,
      long planScratchBytes,
      long largestAllocationBytes,
      KvGeometry kv) {

    private DeviceMemoryRequest request(int contextLength) {
      return DeviceMemoryRequest.detailed(label)
          .weightBytes(weightBytes)
          .retainedShapes(retainedShapes)
          .retainedPlanCount(retainedPlanCount)
          .planScratchBytes(planScratchBytes)
          .largestAllocationBytes(largestAllocationBytes)
          .deviceKvCacheBytes(kv.bytes(contextLength))
          .build();
    }
  }

  /** Ring-bounded plus context-linear KV, matching {@code LayeredKvCache}. */
  private record KvGeometry(long ringBytes, long bytesPerToken, int maxContext) {
    private long bytes(int contextLength) {
      return ringBytes + bytesPerToken * Math.min(contextLength, maxContext);
    }
  }

  private static Shape shape(Map<String, String> options) {
    String preset = options.getOrDefault("preset", "gemma4-26b-a4b");
    Shape base =
        switch (preset) {
          // Gemma 4 26B-A4B IT Q4_K_M, every constant read from the Models repository:
          // resident and routed-expert bytes from Gemma4LargeModelFixtureSlowTest; 30 layers with
          // 25 sliding (keyDim 2048, ring 1024+1) and 5 full (keyDim 1024) from Gemma4ConfigTest
          // and Gemma4KvCache; plans = 30 x (128 experts x 2 + 4 resident) x 2 retained shapes.
          case "gemma4-26b-a4b" ->
              new Shape(
                  "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                  1_650_027_640L + 15_130_165_248L,
                  2,
                  30 * (128 * 2 + 4) * 2,
                  2_733_284_400L,
                  262_144L * 2_816L / 32L * 34L,
                  new KvGeometry(25L * 2 * 2_048 * 4 * 1_025, 5L * 2 * 1_024 * 4, 262_144));
          // Qwen3 0.6B Q4_0: the published A16-2Q / A40-4Q continuity control.
          case "qwen3-0.6b" ->
              new Shape(
                  "Qwen3-0.6B-Q4_0.gguf",
                  428_970_080L,
                  2,
                  223,
                  0L,
                  0L,
                  new KvGeometry(0L, 0L, 40_960));
          default -> throw new IllegalArgumentException("unknown preset: " + preset);
        };
    return new Shape(
        options.getOrDefault("label", base.label()),
        longOption(options, "weight-bytes", base.weightBytes()),
        (int) longOption(options, "shapes", base.retainedShapes()),
        (int) longOption(options, "plans", base.retainedPlanCount()),
        longOption(options, "plan-scratch", base.planScratchBytes()),
        longOption(options, "largest-allocation", base.largestAllocationBytes()),
        base.kv());
  }

  private static List<Integer> contexts(Map<String, String> options) {
    String raw = options.getOrDefault("context", "4096,32768,131072,262144");
    List<Integer> contexts = new ArrayList<>();
    for (String token : raw.split(",", -1)) {
      contexts.add(Integer.parseInt(token.strip()));
    }
    return contexts;
  }

  private static List<AcceleratorEligibility.DeviceCapabilities> devices(
      Map<String, String> options) {
    List<AcceleratorEligibility.DeviceCapabilities> devices = new ArrayList<>();
    String raw = options.getOrDefault("device", "A40 24 GB:23:8|L40S 48 GB:44:8|H100 80 GB:79:32");
    for (String entry : raw.split("\\|", -1)) {
      String[] parts = entry.split(":", -1);
      if (parts.length != 3) {
        throw new IllegalArgumentException("device must be name:globalGiB:maxAllocGiB: " + entry);
      }
      devices.add(
          new AcceleratorEligibility.DeviceCapabilities(
              parts[0].strip(),
              "PTX",
              "GPU",
              Long.parseLong(parts[1].strip()) * GIB,
              Long.parseLong(parts[2].strip()) * GIB));
    }
    devices.addAll(TornadoRuntimeDevices.discover());
    return devices;
  }

  private static long longOption(Map<String, String> options, String name, long fallback) {
    String value = options.get(name);
    return value == null ? fallback : Long.parseLong(value.strip());
  }

  private static Map<String, String> parse(String[] args) {
    Map<String, String> options = new LinkedHashMap<>();
    for (int index = 0; index < args.length; index++) {
      String argument = args[index];
      if (!argument.startsWith("--") || index + 1 >= args.length) {
        throw new IllegalArgumentException("expected --option value pairs, found: " + argument);
      }
      String key = argument.substring(2);
      String value = args[++index];
      options.merge(key, value, (existing, added) -> existing + "|" + added);
    }
    return options;
  }
}
