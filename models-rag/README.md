# models-rag

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/models/main/models-rag/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

`models-rag` provides framework-neutral retrieval preflight and response
grounding for RAG applications. It bounds and screens retrieved evidence before
generation, then enforces source attribution, abstention, and extractive fallback
on the generated answer.

```kotlin
implementation("com.integrallis:models-rag:0.3.23")
```

```java
var policy = new GroundedAnswerPolicy(2.0f);
var evidence = retrieved.stream()
    .map(hit -> new GroundingDocument(
        hit.id(), hit.title(), hit.text(), hit.score(), hit.rank()))
    .toList();
var prompt = GroundedRagPrompt.prepare(policy, question, evidence);
if (!prompt.generationAllowed()) {
    return GroundedAnswerPolicy.ABSTENTION;
}

String rawModelText = model.generate(prompt.text());
GroundedAnswer answer = policy.apply(question, evidence, rawModelText);
System.out.println(answer.text());
System.out.println(answer.decision());
```

The policy has five answer outcomes:

- `MODEL_ANSWER`: the response contains only citations from retrieved sources.
- `MODEL_ANSWER_WITH_DERIVED_CITATIONS`: the response text is supported by the
  retrieved evidence but omitted citations, so the library retained the model
  text and attached the retrieved source IDs.
- `MODEL_ABSTENTION`: the model explicitly reports insufficient context.
- `RETRIEVAL_ABSTENTION`: no retrieved source clears the configured score.
- `EXTRACTIVE_FALLBACK`: retrieval is strong, but the model refuses or emits
  unsupported text or citations; exact source text is returned.

The score threshold must be calibrated for the chosen retriever, corpus, and
chunking strategy. `2.0` is the committed benchmark corpus threshold, not a
portable BM25 default.
