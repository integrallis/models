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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.api.PortablePrefixCapabilities.Status;
import com.integrallis.models.api.PortablePrefixCapabilities.Truncation;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Proxy;
import org.junit.jupiter.api.Test;

class PortablePrefixCapabilitiesTest {
  @Test
  void legacyProvidersKeepTheirBooleanWithoutInventingTruncationOrCallingOtherMethods() {
    for (boolean supported : new boolean[] {true, false}) {
      var type = PortablePrefixInferenceBackend.class;
      var backend =
          (PortablePrefixInferenceBackend)
              Proxy.newProxyInstance(
                  type.getClassLoader(),
                  new Class<?>[] {type},
                  (proxy, method, args) -> {
                    if (method.getName().equals("supportsPortablePrefixes")) return supported;
                    if (method.isDefault())
                      return InvocationHandler.invokeDefault(proxy, method, args);
                    throw new AssertionError("unexpected call: " + method.getName());
                  });
      var capabilities = backend.portablePrefixCapabilities();
      assertThat(capabilities.supported()).isEqualTo(backend.supportsPortablePrefixes());
      assertThat(capabilities.status())
          .isEqualTo(supported ? Status.SUPPORTED : Status.BACKEND_UNSUPPORTED);
      assertThat(capabilities.truncation())
          .isEqualTo(supported ? Truncation.UNSPECIFIED : Truncation.UNSUPPORTED);
    }
  }

  @Test
  void completeCheckpointBoundariesAndArbitraryTruncationRemainDistinct() {
    var retained = PortablePrefixCapabilities.supported(Truncation.RETAINED_CHECKPOINTS);
    assertThat(retained.supported()).isTrue();
    assertThat(retained).isNotEqualTo(PortablePrefixCapabilities.supported(Truncation.ANY_PREFIX));
  }

  @Test
  void contradictoryOrMissingCapabilitiesAreRejected() {
    assertThatIllegalArgumentException()
        .isThrownBy(() -> PortablePrefixCapabilities.supported(Truncation.UNSUPPORTED));
    assertThatIllegalArgumentException()
        .isThrownBy(() -> PortablePrefixCapabilities.unsupported(Status.SUPPORTED));
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                new PortablePrefixCapabilities(Status.BACKEND_UNSUPPORTED, Truncation.ANY_PREFIX));
    assertThatNullPointerException().isThrownBy(() -> PortablePrefixCapabilities.supported(null));
    assertThatNullPointerException().isThrownBy(() -> PortablePrefixCapabilities.unsupported(null));
  }
}
