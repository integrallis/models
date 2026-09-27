# Training corpus

The prompts the pretrained classifier was trained on live in a separate repository:

**https://github.com/integrallis/model-router-corpus**

They are third-party text under a mix of source licences. Keeping them out of this repository means
a licence question about one benchmark cannot block a release of the library. What ships here is the
derived index — embeddings, not prompts — at
`models-router/src/main/resources/com/integrallis/models/router/task-index.zip`.

The index records the corpus digest it was built from, so the two cannot drift apart unnoticed:

    corpusSha256 = 639d305a2c4fee52779cb16832f365d91bea621d0908f97d501df74872121299

To rebuild the index, clone that repository and follow its README, or point the build at a checkout:

    ./gradlew :models-bench:run --args="task-index build \
      --model ~/.jvllm/models/embeddinggemma-300M-Q8_0.gguf \
      --model-id google_embeddinggemma_300m_gguf_q8_0 \
      --corpus /path/to/model-router-corpus/benchmark-prompts.tsv \
      --out /tmp/task-index --quantizer SQ4"

That command writes a collection *directory*. What ships is a zip of its contents, with the
generation directories at the archive root rather than nested under a parent:

    cd /tmp/task-index && zip -rX \
      .../models-router/src/main/resources/com/integrallis/models/router/task-index.zip \
      CURRENT task-index.properties gen-*

The archive is expanded into a cache keyed on its own SHA-256, so a replaced index never collides
with a previously expanded one.

`BundledTaskIndexTest` pins the corpus digest and the prompt count against the archive in the jar,
so a rebuild that changed either has to update that test deliberately.

## Held-out accuracy

The corpus splits into training and held-out prompts. Only the training split is indexed and only it
feeds `corpusSha256`, which is why the index holds fewer prompts than the corpus has lines. The
held-out split is what `task-index evaluate` scores:

    ./gradlew :models-bench:run --args="task-index evaluate \
      --model ~/.jvllm/models/embeddinggemma-300M-Q8_0.gguf \
      --corpus /path/to/model-router-corpus/benchmark-prompts.tsv \
      --index /tmp/task-index --threshold 0.0"

Measured here on the SQ4 index that ships, EmbeddingGemma-300M Q8_0, threshold 0.0:

    accuracy 0.9044 (435/481), unclassified 0

The corpus repository's README also quotes an SQ4 figure of 0.9019. That number is not comparable to
this one: no whole number of correct answers out of 481 held-out prompts yields 0.9019, so it was
measured against a differently sized split. Re-measure rather than diffing against it.

`TaskExemplarsTest` reads the corpus when it can find one, and skips otherwise. Point it at a
checkout with `-Dmodels.router.corpus=/path/to/model-router-corpus/benchmark-prompts.tsv`.
