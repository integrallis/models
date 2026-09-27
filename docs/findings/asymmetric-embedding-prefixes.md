# Asymmetric embedding prefixes are not applied anywhere

Recorded 2026-09-27. **Provenance: read from the source.** Nothing here is measured — no retrieval
quality number has been produced before or after, and the size of the effect is unknown.

## The finding

`models` ships and qualifies embedding models whose input is **asymmetric**: the same text must be
embedded differently depending on whether it is a document being indexed or a query being searched.

| model | document side | query side |
|---|---|---|
| Nomic-embed-text-v1.5 | `search_document: ` | `search_query: ` |
| E5-Mistral-7B-Instruct | `passage: ` | `query: ` |

**No prefix is applied anywhere.** `search_document`, `search_query`, `passage: ` and `query: ` do not
appear in any source file, main or test. (The `prefix` occurrences in `GgufEmbeddingBackend` are
Matryoshka dimension truncation and unrelated.)

The production path embeds raw text directly:

```java
// VectorCollectionEmbeddingSink
float[] vector = validateVector(backend.embed(text), 0);
collection.add(Document.of(id, vector, text));
```

`Document.of(id, vector, text)` also offers nowhere to record which model or prefix produced the
vector, so nothing detects the omission afterwards.

## What this does and does not affect

This is the distinction that matters, and it is easy to overstate:

| claim | affected |
|---|---|
| "our embeddings match the pinned reference implementation" | **No. Still true.** |
| retrieval quality for a user embedding with Nomic or E5 through the sink | **Yes. Silently degraded.** |

`EmbeddingEquivalence` compares our vectors against a reference implementation's vectors **of the same
raw probe text**. Both sides omit the prefix identically, so the comparison is unaffected and the
equivalence claim stands. It asserts that our kernel reproduces the reference, not that the input was
constructed correctly for retrieval.

Retrieval still *works*: documents and queries are both embedded without prefixes, so they remain
mutually consistent. It is simply worse than the model is capable of, with nothing to indicate it.
MiniLM is symmetric, so any qualification exercised mainly on MiniLM would not reveal this.

## Why it went unnoticed, and what fixes it

An instruction prefix is a property of the **model**, but it was left to each **call site** — and no
call site knew it was responsible. Nothing in the API made the obligation visible or its absence
detectable.

`vectors` now has the mechanism: `EmbeddingRecipe` carries `documentPrefix` and `queryPrefix` as
separate fields, with `documentInput(text)` and `queryInput(text)` applying them, so the prefix lives
with the model definition rather than at the call site. The Spring AI adapter in `vectors` already
applies both sides, and its content hash covers the prefixed text so the stored hash answers "would
re-embedding change this vector".

**Blocked on a release.** The recipe API is unreleased — it is on `feat/embedding-provenance` and not in
vectors 0.1.23, which is what `models` pins. Adoption needs vectors 0.1.24 first.

## What to do, in order

1. Release vectors 0.1.24 with the recipe API.
2. Bump `vectorsVersion` in `models` and give `VectorCollectionEmbeddingSink` an optional recipe,
   applying `documentInput` on ingest.
3. **Measure before claiming anything.** Run a retrieval benchmark with Nomic on a public dataset,
   prefixed and unprefixed, as an A/B on one host. Until that exists the effect size is unknown, and
   "we fixed retrieval quality" is not a statement this project is entitled to make.
4. Check the qualification corpus. If published Nomic or E5 numbers were produced through the sink,
   they measured the unprefixed path — which is still a valid measurement of *that* path, and should be
   relabelled rather than deleted.

## What is explicitly not claimed

That any published number is wrong. That retrieval is broken. That the effect is large. The defect is
that a model's documented input contract is not honoured and nothing detects it; the consequence is
unmeasured.
