// models-spring-boot-starter — Auto-configuration for inference + optional vectors

dependencies {
    api("org.modeljars:modeljars-core:0.1.0-SNAPSHOT")

    implementation("org.springframework.boot:spring-boot-autoconfigure:4.1.0")
    annotationProcessor("org.springframework.boot:spring-boot-configuration-processor:4.1.0")

    testImplementation("org.springframework.boot:spring-boot-starter-test:4.1.0")
    testRuntimeOnly("org.modeljars.huggingface:ggml-org.qwen3-0.6b-gguf.q4_0:3.0.0-q4_0.1")
}
