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
package com.integrallis.models.rag;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.QueryBuilder;

/** In-memory Lucene BM25 retriever used by the controlled workload. */
public final class LuceneRagRetriever implements RagRetriever, AutoCloseable {
  private static final String ID_FIELD = "id";
  private static final String CONTENT_FIELD = "content";

  private final Analyzer analyzer;
  private final Directory directory;
  private final DirectoryReader reader;
  private final IndexSearcher searcher;
  private final Map<String, RagDocument> documentsById;

  public LuceneRagRetriever(List<RagDocument> documents) throws IOException {
    Objects.requireNonNull(documents, "documents");
    this.analyzer = new StandardAnalyzer();
    this.directory = new ByteBuffersDirectory();
    this.documentsById = new HashMap<>();

    IndexWriterConfig config = new IndexWriterConfig(analyzer);
    config.setSimilarity(new BM25Similarity());
    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (RagDocument source : documents) {
        if (documentsById.put(source.id(), source) != null) {
          throw new IllegalArgumentException("duplicate document id: " + source.id());
        }
        Document indexed = new Document();
        indexed.add(new StringField(ID_FIELD, source.id(), StringField.Store.YES));
        indexed.add(
            new TextField(CONTENT_FIELD, source.title() + " " + source.text(), TextField.Store.NO));
        writer.addDocument(indexed);
      }
    }
    this.reader = DirectoryReader.open(directory);
    this.searcher = new IndexSearcher(reader);
    this.searcher.setSimilarity(new BM25Similarity());
  }

  @Override
  public List<RetrievedDocument> retrieve(String question, int topK) throws IOException {
    Objects.requireNonNull(question, "question");
    if (question.isBlank()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    if (topK < 1) {
      throw new IllegalArgumentException("topK must be positive");
    }

    Query query =
        new QueryBuilder(analyzer)
            .createBooleanQuery(CONTENT_FIELD, question, BooleanClause.Occur.SHOULD);
    if (query == null) {
      return List.of();
    }

    ScoreDoc[] scoreDocs = searcher.search(query, topK).scoreDocs;
    List<RetrievedDocument> hits = new ArrayList<>(scoreDocs.length);
    for (int index = 0; index < scoreDocs.length; index++) {
      ScoreDoc scoreDoc = scoreDocs[index];
      String id = searcher.storedFields().document(scoreDoc.doc).get(ID_FIELD);
      RagDocument source = documentsById.get(id);
      if (source == null) {
        throw new IllegalStateException("Lucene returned an unknown document: " + id);
      }
      hits.add(new RetrievedDocument(source, scoreDoc.score, index + 1));
    }
    return List.copyOf(hits);
  }

  @Override
  public void close() throws IOException {
    reader.close();
    directory.close();
    analyzer.close();
  }
}
