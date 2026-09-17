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

/**
 * JSONL field names for a dataset file; the prepared study files use {@link #standard()}.
 *
 * @param id identifier field
 * @param question question text field
 * @param answer gold answer field
 * @param choices optional array of {label, text} objects
 */
public record FieldMapping(String id, String question, String answer, String choices) {

  public static FieldMapping standard() {
    return new FieldMapping("id", "question", "answer", "choices");
  }
}
