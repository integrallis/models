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
package com.integrallis.models.api;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/** A non-generative model that scores caller-declared labels in one or more tasks. */
public interface SchemaClassificationModel extends AutoCloseable {
  /** One label and its optional explanation to the model. */
  record Label(String name, String description) {
    public Label {
      if (Objects.requireNonNull(name, "name").isBlank()) {
        throw new IllegalArgumentException("label name must not be blank");
      }
      description = Objects.requireNonNull(description, "description");
    }

    public Label(String name) {
      this(name, "");
    }
  }

  /** Ordered labels; multi-label scores are independent, not a probability distribution. */
  record Task(String name, List<Label> labels, boolean multiLabel, double threshold) {
    public Task {
      if (Objects.requireNonNull(name, "name").isBlank()) {
        throw new IllegalArgumentException("task name must not be blank");
      }
      labels = List.copyOf(labels);
      if (labels.isEmpty() || labels.size() > 256) {
        throw new IllegalArgumentException("task must contain 1..256 labels");
      }
      HashSet<String> names = new HashSet<>();
      for (Label label : labels) {
        if (!names.add(label.name())) {
          throw new IllegalArgumentException("duplicate label: " + label.name());
        }
      }
      if (!Double.isFinite(threshold) || threshold < 0 || threshold > 1) {
        throw new IllegalArgumentException("threshold must be in [0,1]");
      }
    }
  }

  /** Ordered input tasks, each scored against the same document. */
  record Request(String text, List<Task> tasks) {
    public Request {
      text = Objects.requireNonNull(text, "text");
      tasks = List.copyOf(tasks);
      if (tasks.isEmpty() || tasks.size() > 32) {
        throw new IllegalArgumentException("request must contain 1..32 tasks");
      }
      HashSet<String> names = new HashSet<>();
      for (Task task : tasks) {
        if (!names.add(task.name())) {
          throw new IllegalArgumentException("duplicate task: " + task.name());
        }
      }
    }
  }

  /** A model score; this record makes no calibrated-confidence claim. */
  record LabelScore(String label, double score, boolean selected) {
    public LabelScore {
      label = Objects.requireNonNull(label, "label");
      if (!Double.isFinite(score) || score < 0 || score > 1) {
        throw new IllegalArgumentException("classification score must be finite in [0,1]");
      }
    }
  }

  /** One result per input task; thresholdFallback marks the publisher's forced top-label choice. */
  record Result(String task, List<LabelScore> labels, boolean thresholdFallback) {
    public Result {
      task = Objects.requireNonNull(task, "task");
      labels = List.copyOf(labels);
    }
  }

  List<Result> classify(Request request);

  @Override
  void close();
}
