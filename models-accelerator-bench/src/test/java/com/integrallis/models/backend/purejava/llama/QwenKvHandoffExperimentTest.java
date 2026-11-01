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
package com.integrallis.models.backend.purejava.llama;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.cache.KvCache;
import org.junit.jupiter.api.Test;

class QwenKvHandoffExperimentTest {

  @Test
  void copiesOnlyTheDeclaredPrefixIntoAnEqualGeometryCache() {
    KvCache source = new KvCache(2, 4, 2, 2);
    KvCache target = new KvCache(2, 4, 2, 2);
    for (int layer = 0; layer < 2; layer++) {
      for (int position = 0; position < 3; position++) {
        float value = layer * 10.0f + position;
        source.store(layer, position, new float[] {value, value + 0.25f}, new float[] {-value, 1});
      }
    }

    QwenKvHandoffExperiment.copyRawPrefix(source, target, 2);

    assertThat(target.key(1, 1)).containsExactly(11.0f, 11.25f);
    assertThat(target.value(1, 1)).containsExactly(-11.0f, 1.0f);
    assertThat(target.key(1, 2)).isNull();
  }
}
