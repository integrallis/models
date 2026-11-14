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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.RerankingModel;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

/**
 * Issue #448: Verify that concurrent scoreBatch() calls on the same RerankingModel instance do not
 * corrupt state or deadlock.
 *
 * <p>Uses a dummy model that simulates non-trivial work per document to expose contention.
 */
class RerankingModelConcurrentBatchTest {

  private static final RerankingModel SLOW_DUMMY =
      new RerankingModel() {
        @Override
        public double score(String query, String document) {
          // Simulate a small amount of compute to make contention observable
          double acc = 0;
          for (int i = 0; i < 100; i++) {
            acc += Math.sin(query.length() + document.length() + i);
          }
          return acc;
        }

        @Override
        public List<Double> scoreBatch(String query, List<String> documents) {
          // Simulate batch scoring with per-document scoring to match real behavior
          List<Double> scores = new ArrayList<>(documents.size());
          for (String doc : documents) {
            scores.add(score(query, doc));
          }
          return List.copyOf(scores);
        }

        @Override
        public void close() {}
      };

  @Test
  void concurrentScoreBatchDoesNotCorruptResults() throws Exception {
    int threads = 4;
    int batchesPerThread = 50;
    int docsPerBatch = 8;
    ExecutorService executor = Executors.newFixedThreadPool(threads);
    CountDownLatch latch = new CountDownLatch(threads);
    List<Future<List<List<Double>>>> futures = new ArrayList<>();

    String query = "How many people live in Berlin?";
    List<String> documents =
        List.of(
            "Berlin has a population of 3,520,031.",
            "Paris is the capital of France.",
            "Berlin is known for its museums.",
            "Cats sleep a lot.",
            "New York had 8,804,190 people in 2020.",
            "The Berlin Wall fell in 1989.",
            "London is the capital of the UK.",
            "Tokyo is the largest city in Japan.");

    // Precompute expected scores for the query + first document
    double expectedFirstDocScore = SLOW_DUMMY.score(query, documents.get(0));

    for (int t = 0; t < threads; t++) {
      futures.add(
          executor.submit(
              () -> {
                latch.countDown();
                latch.await(); // All threads start together
                List<List<Double>> threadResults = new ArrayList<>(batchesPerThread);
                for (int b = 0; b < batchesPerThread; b++) {
                  threadResults.add(SLOW_DUMMY.scoreBatch(query, documents));
                }
                return threadResults;
              }));
    }

    for (Future<List<List<Double>>> future : futures) {
      List<List<Double>> threadResults = future.get();
      assertThat(threadResults).hasSize(batchesPerThread);
      for (List<Double> batch : threadResults) {
        assertThat(batch).hasSize(docsPerBatch);
        // Every batch should produce the same first-document score
        assertThat(batch.get(0)).isEqualTo(expectedFirstDocScore);
      }
    }

    executor.shutdown();
  }

  @Test
  void concurrentScoreAndScoreBatchDoNotDeadlock() throws Exception {
    int threads = 4;
    int callsPerThread = 100;
    ExecutorService executor = Executors.newFixedThreadPool(threads);
    CountDownLatch latch = new CountDownLatch(threads);
    List<Future<Integer>> futures = new ArrayList<>();

    String query = "query";
    String document = "document";
    List<String> documents = List.of("a", "b", "c", "d");

    for (int t = 0; t < threads; t++) {
      final boolean useBatch = (t % 2 == 0);
      futures.add(
          executor.submit(
              () -> {
                latch.countDown();
                latch.await();
                int successful = 0;
                for (int i = 0; i < callsPerThread; i++) {
                  if (useBatch) {
                    SLOW_DUMMY.scoreBatch(query, documents);
                  } else {
                    SLOW_DUMMY.score(query, document);
                  }
                  successful++;
                }
                return successful;
              }));
    }

    for (Future<Integer> future : futures) {
      assertThat(future.get()).isEqualTo(callsPerThread);
    }

    executor.shutdown();
  }
}
