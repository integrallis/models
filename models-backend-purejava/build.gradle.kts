// models-backend-purejava — GGUF parser, scalar inference kernels, KV cache

dependencies {
    api(project(":models-api"))

    implementation("com.integrallis:vectors-core:0.1.0-SNAPSHOT")

    // Integration tests use GenerationLoop from models-runtime
    testImplementation(project(":models-runtime"))
}
