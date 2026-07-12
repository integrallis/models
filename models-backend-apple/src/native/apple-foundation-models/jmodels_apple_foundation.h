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

#ifdef __cplusplus
extern "C" {
#endif

int jmodels_afm_available(void);

char *jmodels_afm_generate(
    const char *prompt,
    const char *instructions,
    int max_output_tokens);

char *jmodels_afm_last_error(void);

void jmodels_afm_free(char *pointer);

#ifdef __cplusplus
}
#endif

#endif
