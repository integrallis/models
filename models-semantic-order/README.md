# models-semantic-order

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/models/main/models-semantic-order/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Pure-Java runtime for compact one-dimensional semantic-order models. The first
supported format is `wordtour-v1`: one unique UTF-8 term per line, arranged as a
cycle.

## ModelJars use

Add `models-modeljars` and the WordTour marker JAR to the application classpath,
then resolve it by source and semantic version:

```java
var modelJar = ModelJar.of("github://joisino/wordtour")
    .version("[1.0.0,2.0.0)")
    .variant("optimal")
    .backend("semantic-order")
    .capability("semantic-neighbors");

var descriptor = ModelJarRegistry.fromClasspath().resolve(modelJar).orElseThrow();
WordTour tour = ModelJarBackends.loadWordTour(descriptor);
tour.neighbors("cat", 5);
```

The canonical marker coordinate is:

```text
org.modeljars.github:joisino.wordtour-glove-6b-300d.optimal:1.0.0-optimal.1
```

Its payload is pinned to upstream revision
`de6c20e3e6c26f61a5b7cb0a5317cff582e53637` and verified with SHA-256
`56f880329c5ffa73fe549a04603ebe64fc745b4c8492392102bb935ae1c9a0b6`.

## Document representation

`BlurredBagOfWords` reproduces the reference WordTour blur:

```java
var left = BlurredBagOfWords.encode(tour, List.of("cat", "pet"));
var right = BlurredBagOfWords.encode(tour, List.of("dog", "animal"));
double distance = left.l1Distance(right);
```

The default radius is 10 and the default Gaussian denominator is 10. Unknown
terms are ignored. Callers own tokenization and text normalization.

## Semantics

WordTour is sound but incomplete: nearby ranks tend to be semantically related,
while distant ranks are not guaranteed to be unrelated. It is appropriate for
local neighbor enumeration, lexical expansion, and blurred bag-of-words. It is
not a drop-in dense embedding, and it does not implement `EmbeddingBackend`.

The semantic-order runtime itself has no dependency on ModelJars, Vectors, or
the Java Vector API. Registry integration is isolated in `models-modeljars`.
