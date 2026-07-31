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

#include "jmodels_apple_foundation.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static jmodels_afm_result *make_result(
    int32_t status,
    int32_t value,
    const uint8_t *data,
    size_t length) {
  jmodels_afm_result *result = calloc(1, sizeof(jmodels_afm_result));
  if (result == NULL) {
    return NULL;
  }
  result->status = status;
  result->value = value;
  if (length == 0) {
    return result;
  }
  result->data = malloc(length);
  if (result->data == NULL) {
    result->status = 1;
    return result;
  }
  memcpy(result->data, data, length);
  result->length = length;
  return result;
}

static jmodels_afm_result *make_text_result(
    int32_t status,
    int32_t value,
    const char *text) {
  return make_result(status, value, (const uint8_t *) text, strlen(text));
}

static char *copy_as_c_string(const uint8_t *bytes, size_t length) {
  if (length > 0 && bytes == NULL) {
    return NULL;
  }
  char *text = malloc(length + 1);
  if (text == NULL) {
    return NULL;
  }
  if (length > 0) {
    memcpy(text, bytes, length);
  }
  text[length] = '\0';
  return text;
}

static int contains_case_insensitive(const char *haystack, const char *needle) {
  if (haystack == NULL || needle == NULL || needle[0] == '\0') {
    return 0;
  }
  size_t needle_length = strlen(needle);
  for (const char *cursor = haystack; *cursor != '\0'; cursor++) {
    size_t index = 0;
    while (index < needle_length
        && cursor[index] != '\0'
        && tolower((unsigned char) cursor[index]) == tolower((unsigned char) needle[index])) {
      index++;
    }
    if (index == needle_length) {
      return 1;
    }
  }
  return 0;
}

static const char *subject(const char *prompt) {
  const char *colon = strchr(prompt, ':');
  if (colon == NULL) {
    return prompt;
  }
  colon++;
  while (*colon != '\0' && isspace((unsigned char) *colon)) {
    colon++;
  }
  return colon;
}

jmodels_afm_result *jmodels_afm_available(void) {
  return make_text_result(
      0, 1, "Apple Foundation Models native stub mode is available");
}

jmodels_afm_result *jmodels_afm_generate(
    const uint8_t *prompt_bytes,
    size_t prompt_length,
    const uint8_t *instructions,
    size_t instructions_length,
    int32_t max_output_tokens) {
  (void) instructions;
  (void) instructions_length;
  (void) max_output_tokens;
  char *prompt = copy_as_c_string(prompt_bytes, prompt_length);
  if (prompt == NULL) {
    return make_text_result(1, 0, "prompt payload was invalid or out of memory");
  }

  jmodels_afm_result *result;
  if (contains_case_insensitive(prompt, "force native error")) {
    const char *prefix = "forced native failure: ";
    size_t length = strlen(prefix) + strlen(prompt) + 1;
    char *message = malloc(length);
    if (message == NULL) {
      result = make_text_result(1, 0, "out of memory");
    } else {
      snprintf(message, length, "%s%s", prefix, prompt);
      result = make_text_result(1, 0, message);
      free(message);
    }
  } else if (contains_case_insensitive(prompt, "embedded-null")) {
    static const uint8_t embedded_null[] = {
      'p', 'r', 'e', 'f', 'i', 'x', '\0', 's', 'u', 'f', 'f', 'i', 'x'
    };
    result = make_result(0, 0, embedded_null, sizeof(embedded_null));
  } else if (contains_case_insensitive(prompt, "single word hello")) {
    result = make_text_result(0, 0, "hello");
  } else {
    const char *prefix =
        contains_case_insensitive(prompt, "summarize") ? "Stub summary: " : "Stub response: ";
    const char *body = contains_case_insensitive(prompt, "summarize") ? subject(prompt) : prompt;
    size_t length = strlen(prefix) + strlen(body) + 1;
    char *response = malloc(length);
    if (response == NULL) {
      result = make_text_result(1, 0, "out of memory");
    } else {
      snprintf(response, length, "%s%s", prefix, body);
      result = make_text_result(0, 0, response);
      free(response);
    }
  }
  free(prompt);
  return result;
}

void jmodels_afm_result_free(jmodels_afm_result *result) {
  if (result != NULL) {
    free(result->data);
    free(result);
  }
}
