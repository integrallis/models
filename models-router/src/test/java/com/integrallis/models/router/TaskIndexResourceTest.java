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
package com.integrallis.models.router;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TaskIndexResourceTest {

  @Test
  void saysWhatToDoWhenNoIndexIsPackaged(@TempDir Path cache) {
    // The index is produced by a build step. Until it runs, the jar has no resource, and the
    // message has to point at the step rather than read as a missing-file error.
    assertThatThrownBy(() -> TaskIndexResource.extractTo(cache))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("no task index packaged")
        .hasMessageContaining("TaskIndexBuilder");
  }

  @Test
  void expandsAnArchiveAndReusesTheExpansion(@TempDir Path cache) throws Exception {
    Path archive = ArchiveFixture.write(cache.resolve("src"));
    Path first = ArchiveFixture.extract(archive, cache.resolve("cache"));
    assertThat(Files.isReadable(first.resolve(TaskIndexBuilder.MANIFEST))).isTrue();

    Path second = ArchiveFixture.extract(archive, cache.resolve("cache"));
    assertThat(second).isEqualTo(first);
  }

  @Test
  void refusesAnEntryThatEscapesTheDestination(@TempDir Path cache) throws Exception {
    // A crafted archive could otherwise write anywhere the process can reach.
    Path archive = ArchiveFixture.writeEscaping(cache.resolve("evil"));

    assertThatThrownBy(() -> ArchiveFixture.extract(archive, cache.resolve("cache")))
        .rootCause()
        .hasMessageContaining("escapes the destination");
  }
}
