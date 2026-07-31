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

#ifndef JMODELS_APPLE_FOUNDATION_H
#define JMODELS_APPLE_FOUNDATION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jmodels_afm_result {
  int32_t status;
  int32_t value;
  size_t length;
  uint8_t *data;
} jmodels_afm_result;

jmodels_afm_result *jmodels_afm_available(void);

jmodels_afm_result *jmodels_afm_generate(
    const uint8_t *prompt,
    size_t prompt_length,
    const uint8_t *instructions,
    size_t instructions_length,
    int32_t max_output_tokens);

void jmodels_afm_result_free(jmodels_afm_result *result);

#ifdef __cplusplus
}
#endif

#endif
