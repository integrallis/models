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
package com.integrallis.models.runtime;

import com.integrallis.models.api.PortablePrefixInferenceBackend;
import com.integrallis.models.api.SharedInferencePrefix;
import java.io.*;
import java.security.*;

/** Versioned, lossless, bounded prompt/KV envelope. Not Java object serialization. */
final class PortablePromptPrefixes {
  private static final int MAGIC = 0x4d504b31;

  private PortablePromptPrefixes() {}

  static void write(
      PortablePrefixInferenceBackend backend,
      SharedPromptPrefix prefix,
      int count,
      OutputStream output)
      throws IOException {
    if (count <= 0 || count > prefix.tokenCount())
      throw new IllegalArgumentException("invalid prefix length");
    var digest = sha256();
    var stream = new DigestOutputStream(output, digest);
    var data = new DataOutputStream(stream);
    data.writeInt(MAGIC);
    data.writeInt(count);
    for (int i = 0; i < count; i++) data.writeInt(prefix.promptTokens[i]);
    backend.writePrefix(prefix.backendPrefix, count, data);
    data.flush();
    output.write(digest.digest());
    output.flush();
  }

  record Imported(SharedInferencePrefix prefix, int[] tokens) {}

  static Imported read(PortablePrefixInferenceBackend backend, InputStream input, long maxBytes)
      throws IOException {
    if (maxBytes < 96) throw new IOException("prefix exceeds import budget");
    var bounded = new BoundedInput(input, maxBytes);
    var digest = sha256();
    var hashing = new DigestInputStream(bounded, digest);
    var data = new DataInputStream(hashing);
    if (data.readInt() != MAGIC) throw new IOException("unsupported prompt prefix version");
    int count = data.readInt();
    long overhead = 8L + 4L * count + 32L;
    if (count <= 0 || count > backend.contextCapacity() || overhead > maxBytes - 52) {
      throw new IOException("invalid prompt prefix size");
    }
    int[] tokens = new int[count];
    for (int i = 0; i < count; i++) {
      tokens[i] = data.readInt();
      if (tokens[i] < 0 || tokens[i] >= backend.tokenizer().vocabSize())
        throw new IOException("invalid prefix token");
    }
    var prefix = backend.readPrefix(data, count, maxBytes - overhead);
    byte[] checksum = new byte[32];
    new DataInputStream(bounded).readFully(checksum);
    if (!MessageDigest.isEqual(digest.digest(), checksum))
      throw new IOException("prefix checksum mismatch");
    if (input.read() != -1) throw new IOException("trailing prefix data");
    return new Imported(prefix, tokens);
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError(impossible);
    }
  }

  private static final class BoundedInput extends FilterInputStream {
    private long remaining;

    BoundedInput(InputStream input, long limit) {
      super(input);
      remaining = limit;
    }

    @Override
    public int read() throws IOException {
      if (remaining == 0) throw new IOException("prefix exceeds import budget");
      int value = in.read();
      if (value >= 0) remaining--;
      return value;
    }

    @Override
    public int read(byte[] bytes, int offset, int length) throws IOException {
      if (length == 0) return 0;
      if (remaining == 0) throw new IOException("prefix exceeds import budget");
      int count = in.read(bytes, offset, (int) Math.min(length, remaining));
      if (count > 0) remaining -= count;
      return count;
    }
  }
}
