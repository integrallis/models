// models-langchain4j — LangChain4j chat model adapter

dependencies {
    api(project(":models-api"))
    api("dev.langchain4j:langchain4j:1.17.2")

    implementation(project(":models-runtime"))
    implementation("org.modeljars:modeljars-core:0.1.0-SNAPSHOT")

    testRuntimeOnly("org.modeljars.huggingface:ggml-org.qwen3-0.6b-gguf.q4_0:3.0.0-q4_0.1")
}
