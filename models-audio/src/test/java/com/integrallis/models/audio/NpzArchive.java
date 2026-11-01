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
package com.integrallis.models.audio;

import com.integrallis.models.backend.onnx.OnnxDataType;
import com.integrallis.models.backend.onnx.Tensor;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Enumeration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

/** Test-side reader for NumPy {@code .npz} archives: a zip of {@code .npy} members. */
final class NpzArchive {

  private static final Pattern DESCR = Pattern.compile("'descr':\\s*'([^']+)'");
  private static final Pattern ORDER = Pattern.compile("'fortran_order':\\s*(True|False)");
  private static final Pattern SHAPE = Pattern.compile("'shape':\\s*\\(([^)]*)\\)");

  private NpzArchive() {}

  /** Reads every member into a tensor keyed by its name without the {@code .npy} suffix. */
  static Map<String, Tensor> read(Path path) throws IOException {
    Map<String, Tensor> tensors = new LinkedHashMap<>();
    try (ZipFile zip = new ZipFile(path.toFile())) {
      Enumeration<? extends ZipEntry> entries = zip.entries();
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        String name = entry.getName();
        if (!name.endsWith(".npy")) {
          continue;
        }
        try (InputStream in = zip.getInputStream(entry)) {
          tensors.put(name.substring(0, name.length() - 4), readNpy(in.readAllBytes(), name));
        }
      }
    }
    return tensors;
  }

  static Tensor readNpy(byte[] bytes, String name) {
    ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
    if ((buffer.get() & 0xFF) != 0x93 || buffer.get() != 'N') {
      throw new IllegalArgumentException(name + " is not an .npy file");
    }
    buffer.position(6);
    int major = buffer.get();
    buffer.get();
    int headerLength = major == 1 ? buffer.getShort() & 0xFFFF : buffer.getInt();
    byte[] headerBytes = new byte[headerLength];
    buffer.get(headerBytes);
    String header = new String(headerBytes, StandardCharsets.ISO_8859_1);
    Matcher descr = DESCR.matcher(header);
    Matcher order = ORDER.matcher(header);
    Matcher shapeMatch = SHAPE.matcher(header);
    if (!descr.find() || !order.find() || !shapeMatch.find()) {
      throw new IllegalArgumentException(name + " has an unreadable .npy header: " + header);
    }
    if (order.group(1).equals("True")) {
      throw new IllegalArgumentException(name + " is Fortran-ordered");
    }
    String[] dims =
        shapeMatch.group(1).trim().isEmpty() ? new String[0] : shapeMatch.group(1).split(",");
    long[] shape = new long[0];
    int count = 1;
    java.util.List<Long> parsed = new java.util.ArrayList<>();
    for (String dim : dims) {
      if (!dim.isBlank()) {
        parsed.add(Long.parseLong(dim.trim()));
      }
    }
    shape = parsed.stream().mapToLong(Long::longValue).toArray();
    for (long dim : shape) {
      count *= (int) dim;
    }
    String type = descr.group(1);
    switch (type) {
      case "<f4" -> {
        float[] values = new float[count];
        buffer.asFloatBuffer().get(values);
        return Tensor.ofFloats(shape, values);
      }
      case "<f8" -> {
        double[] values = new double[count];
        buffer.asDoubleBuffer().get(values);
        return Tensor.ofDoubles(shape, values);
      }
      case "<i8" -> {
        long[] values = new long[count];
        buffer.asLongBuffer().get(values);
        return Tensor.ofLongs(shape, values);
      }
      case "<i4" -> {
        long[] values = new long[count];
        for (int i = 0; i < count; i++) {
          values[i] = buffer.getInt();
        }
        return Tensor.ofInts(shape, values);
      }
      case "|b1", "|i1" -> {
        byte[] values = new byte[count];
        buffer.get(values);
        return Tensor.ofBytes(
            type.equals("|b1") ? OnnxDataType.BOOL : OnnxDataType.INT8, shape, values);
      }
      case "|u1" -> {
        byte[] values = new byte[count];
        buffer.get(values);
        return Tensor.ofBytes(OnnxDataType.UINT8, shape, values);
      }
      default -> throw new IllegalArgumentException(name + " has unsupported dtype " + type);
    }
  }
}
