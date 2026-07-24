# models-rag

`models-rag` provides framework-neutral response grounding for production RAG
applications. It is intended for local models that can answer narrow questions
but should not be trusted to enforce source attribution or abstention alone.

```kotlin
implementation("com.integrallis:models-rag:0.1.0")
```

```java
var policy = new GroundedAnswerPolicy(2.0f);
var evidence = retrieved.stream()
    .map(hit -> new GroundingDocument(
        hit.id(), hit.text(), hit.score(), hit.rank()))
    .toList();

GroundedAnswer answer = policy.apply(question, evidence, rawModelText);
System.out.println(answer.text());
System.out.println(answer.decision());
```

The policy has three explicit outcomes:

- `MODEL_ANSWER`: the response contains only citations from retrieved sources.
- `RETRIEVAL_ABSTENTION`: no retrieved source clears the configured score.
- `EXTRACTIVE_FALLBACK`: retrieval is strong, but the model refuses, omits a
  citation, or emits an unsupported citation; exact source text is returned.

The score threshold must be calibrated for the chosen retriever, corpus, and
chunking strategy. `2.0` is the committed benchmark corpus threshold, not a
portable BM25 default.
