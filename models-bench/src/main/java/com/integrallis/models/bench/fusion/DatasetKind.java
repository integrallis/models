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
package com.integrallis.models.bench.fusion;

import java.util.Locale;

/** Datasets the runner knows how to prompt and score. */
public enum DatasetKind {
  GSM8K("gsm8k"),
  ARC("arc"),
  MATH500("math500"),
  /** A post-cutoff or other JSONL set; its scorer is chosen with {@code --scorer}. */
  GENERIC("generic");

  private final String id;

  DatasetKind(String id) {
    this.id = id;
  }

  public String id() {
    return id;
  }

  public static DatasetKind parse(String value) {
    for (DatasetKind kind : values()) {
      if (kind.id.equals(value == null ? null : value.toLowerCase(Locale.ROOT))) {
        return kind;
      }
    }
    throw new IllegalArgumentException(
        "--dataset must be gsm8k, arc, math500 or generic: " + value);
  }
}
