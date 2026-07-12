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

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *last_error = NULL;

static char *duplicate_string(const char *value) {
  if (value == NULL) {
    return NULL;
  }
  size_t length = strlen(value) + 1;
  char *copy = malloc(length);
  if (copy != NULL) {
    memcpy(copy, value, length);
  }
  return copy;
}

static void set_last_error(const char *message) {
  free(last_error);
  last_error = duplicate_string(message);
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

int jmodels_afm_available(void) {
  set_last_error("Apple Foundation Models native stub mode is available");
  return 1;
}

char *jmodels_afm_generate(
    const char *prompt,
    const char *instructions,
    int max_output_tokens) {
  (void) instructions;
  (void) max_output_tokens;
  if (prompt == NULL) {
    set_last_error("prompt pointer was null");
    return NULL;
  }
  if (contains_case_insensitive(prompt, "single word hello")) {
    set_last_error("ok");
    return duplicate_string("hello");
  }

  const char *prefix =
      contains_case_insensitive(prompt, "summarize") ? "Stub summary: " : "Stub response: ";
  const char *body = contains_case_insensitive(prompt, "summarize") ? subject(prompt) : prompt;
  size_t length = strlen(prefix) + strlen(body) + 1;
  char *response = malloc(length);
  if (response == NULL) {
    set_last_error("out of memory");
    return NULL;
  }
  snprintf(response, length, "%s%s", prefix, body);
  set_last_error("ok");
  return response;
}

char *jmodels_afm_last_error(void) {
  if (last_error == NULL) {
    return duplicate_string("Apple Foundation Models native stub mode has not been initialized");
  }
  return duplicate_string(last_error);
}

void jmodels_afm_free(char *pointer) {
  free(pointer);
}
