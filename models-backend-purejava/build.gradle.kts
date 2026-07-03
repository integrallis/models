// models-backend-purejava — GGUF parser, scalar inference kernels, KV cache

dependencies {
    api(project(":models-api"))

    // Integration tests use GenerationLoop from models-runtime
    testImplementation(project(":models-runtime"))
}
