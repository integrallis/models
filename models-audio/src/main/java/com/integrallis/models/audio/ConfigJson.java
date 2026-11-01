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

import java.util.Map;

/** Bridges the audio package to the phonemizer package's JSON reader for small config files. */
final class ConfigJson {

  private ConfigJson() {}

  static Map<String, Object> object(String json) {
    return com.integrallis.models.audio.phonemizer.JsonAccess.object(json);
  }
}
