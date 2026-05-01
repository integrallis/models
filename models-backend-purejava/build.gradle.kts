// models-backend-purejava — GGUF parser, Vector API kernels, KV cache
// Depends on vectors-core for SIMD distance kernels (VectorUtil, PanamaVectorUtilSupport)

dependencies {
    api(project(":models-api"))

    // SIMD kernels from java-vectors (zero transitive deps)
    implementation("com.integrallis:vectors-core:0.1.0-SNAPSHOT")
}
