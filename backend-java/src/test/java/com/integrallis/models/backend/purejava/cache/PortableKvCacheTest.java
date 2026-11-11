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
package com.integrallis.models.backend.purejava.cache;

import static org.assertj.core.api.Assertions.*;

import java.io.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PortableKvCacheTest {
  @Test
  void transfersChainedUnequalWidthPrefixAndForksWithoutAliasingMutableSuffix() throws Exception {
    var source = new KvCache(2, 16, 2, 3);
    fill(source, 0, 3);
    var branch = source.freezePrefix(3).fork();
    fill(branch, 3, 5);
    var prefix = branch.freezePrefix(5);
    var bytes = new ByteArrayOutputStream();
    prefix.writeTo(new DataOutputStream(bytes), 4);
    var imported =
        KvCache.readPrefix(
            new DataInputStream(new ByteArrayInputStream(bytes.toByteArray())),
            2,
            16,
            2,
            3,
            4,
            1024);
    var a = imported.fork();
    var b = imported.fork();
    assertThat(a.sharesPrefixStorageWith(b)).isTrue();
    assertThat(a.sharesPrefixStorageWith(prefix.fork())).isFalse();
    for (int layer = 0; layer < 2; layer++)
      for (int pos = 0; pos < 4; pos++) {
        assertThat(a.key(layer, pos)).containsExactly(branch.key(layer, pos));
        assertThat(a.value(layer, pos)).containsExactly(branch.value(layer, pos));
      }
    a.store(0, 4, new float[] {99, 99}, new float[] {98, 98, 98});
    assertThat(b.key(0, 4)).isNull();
    assertThat(imported.truncate(2).length()).isEqualTo(2);
    assertThat(imported.truncate(2).fork().attentionView(0, 0, 2).span(0).keyBuffer())
        .isSameAs(a.attentionView(0, 0, 2).span(0).keyBuffer());
  }

  @Test
  void rejectsShortInputInvalidShapeAndBudgetBeforeAllocating() throws Exception {
    var source = new KvCache(2, 16, 2, 3);
    fill(source, 0, 3);
    var bytes = new ByteArrayOutputStream();
    source.freezePrefix(3).writeTo(new DataOutputStream(bytes), 3);
    byte[] payload = bytes.toByteArray();
    assertThatThrownBy(() -> KvCache.readPrefix(input(payload), 2, 16, 2, 3, 3, 10))
        .isInstanceOf(IOException.class);
    assertThatThrownBy(() -> KvCache.readPrefix(input(payload), 3, 16, 2, 3, 3, 4096))
        .isInstanceOf(IOException.class);
    assertThatThrownBy(
            () ->
                KvCache.readPrefix(
                    input(java.util.Arrays.copyOf(payload, payload.length - 1)),
                    2,
                    16,
                    2,
                    3,
                    3,
                    4096))
        .isInstanceOf(EOFException.class);
    assertThatThrownBy(
            () ->
                KvCache.readPrefix(input(payload), 2, 16, 2, 3, Integer.MAX_VALUE, Long.MAX_VALUE))
        .isInstanceOf(IOException.class);
  }

  private static DataInputStream input(byte[] bytes) {
    return new DataInputStream(new ByteArrayInputStream(bytes));
  }

  private static void fill(KvCache cache, int from, int to) {
    for (int layer = 0; layer < 2; layer++)
      for (int pos = from; pos < to; pos++)
        cache.store(
            layer, pos, new float[] {layer + pos * .25f, -0.0f}, new float[] {pos, layer, 1.25f});
  }
}
