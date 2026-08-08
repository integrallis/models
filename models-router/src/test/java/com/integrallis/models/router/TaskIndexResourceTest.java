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
  void expandsTheIndexPackagedInThisJar(@TempDir Path cache) {
    // This asserted the opposite until an index was actually packaged: that extractTo failed with
    // a message naming the build step that produces one. models-router now ships an index, so the
    // absent case is unreachable from the classpath these tests run on. BundledTaskIndexTest
    // covers what the artifact contains.
    assertThat(TaskIndexResource.extractTo(cache).resolve(TaskIndexBuilder.MANIFEST))
        .satisfies(manifest -> assertThat(Files.isReadable(manifest)).isTrue());
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
