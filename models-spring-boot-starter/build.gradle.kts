// models-spring-boot-starter — Auto-configuration for inference + optional vectors

dependencies {
    api(project(":models-runtime"))
    api(project(":models-spring-ai"))

    // Spring Boot auto-configuration
    compileOnly("org.springframework.boot:spring-boot-autoconfigure:3.4.5")
    annotationProcessor("org.springframework.boot:spring-boot-configuration-processor:3.4.5")

    // Spring AI model interfaces
    compileOnly("org.springframework.ai:spring-ai-model:1.1.4")

    // Optional bridge to embedding/vector storage
    compileOnly(project(":models-embedding"))

    testImplementation("org.springframework.boot:spring-boot-autoconfigure:3.4.5")
    testImplementation("org.springframework.boot:spring-boot-test:3.4.5")
    testImplementation("org.springframework.ai:spring-ai-model:1.1.4")
}
