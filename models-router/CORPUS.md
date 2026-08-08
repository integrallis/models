# Training corpus

The prompts the pretrained classifier was trained on live in a separate repository:

**https://github.com/integrallis/model-router-corpus**

They are third-party text under a mix of source licences. Keeping them out of this repository means
a licence question about one benchmark cannot block a release of the library. What ships here is the
derived index — embeddings, not prompts — at
`models-router/src/main/resources/com/integrallis/models/router/task-index.zip`.

The index records the corpus digest it was built from, so the two cannot drift apart unnoticed:

    corpusSha256 = aeb0fb6ec86d8a6d4dfed8b407bc6c36c0d5147aaaaf88789430a33afa62bfae

To rebuild the index, clone that repository and follow its README, or point the build at a checkout:

    ./gradlew :models-bench:run --args="task-index build \
      --model ~/.jvllm/models/embeddinggemma-300M-Q8_0.gguf \
      --model-id google_embeddinggemma_300m_gguf_q8_0 \
      --corpus /path/to/model-router-corpus/benchmark-prompts.tsv \
      --out /tmp/task-index --quantizer SQ4"

`TaskExemplarsTest` reads the corpus when it can find one, and skips otherwise. Point it at a
checkout with `-Dmodels.router.corpus=/path/to/model-router-corpus/benchmark-prompts.tsv`.
