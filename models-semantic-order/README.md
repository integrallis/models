# models-semantic-order

[![MFCQI](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/integrallis/models/main/models-semantic-order/.github/badges/mfcqi.json)](https://github.com/integrallis/mfcqi-java)

Pure-Java runtime for compact one-dimensional semantic-order models. The first
supported format is `wordtour-v1`: one unique UTF-8 term per line, arranged as a
cycle.

## Loading

Load a `wordtour-v1` payload from a file or any other input stream:

```java
try (var input = Files.newInputStream(Path.of("wordtour.txt"))) {
    WordTour tour = WordTour.load(input);
    tour.neighbors("cat", 5);
}
```

The same canonical payload is independently packaged as:

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

The default radius is 10 and the default Gaussian denominator is 10. A radius
larger than half the vocabulary is capped so each rank is weighted once at its
shortest cyclic distance. Unknown terms are ignored. Callers own tokenization
and text normalization.

## Semantics

WordTour is sound but incomplete: nearby ranks tend to be semantically related,
while distant ranks are not guaranteed to be unrelated. It is appropriate for
local neighbor enumeration, lexical expansion, and blurred bag-of-words. It is
not a drop-in dense embedding, and it does not implement `EmbeddingBackend`.

The semantic-order runtime has no dependency on a model catalog, Vectors, or
the Java Vector API. Downstream catalog adapters can supply its input stream.
