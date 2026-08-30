// models-spring-ai — Spring AI ChatModel + StreamingChatModel adapter

val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("2.0.0")

dependencies {
    api(project(":models-runtime"))

    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")

    compileOnly("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-client-chat:$springAiVersion")
    testImplementation(project(":backend-java"))
}

val configuredGptOssHuggingFaceDirectory =
    providers.systemProperty("models.fixtures.gptOssHuggingFaceDirectory")

tasks.withType<Test>().configureEach {
    configuredGptOssHuggingFaceDirectory.orNull?.let {
        systemProperty("models.fixtures.gptOssHuggingFaceDirectory", it)
    }
}

tasks.register<Test>("gptOssSpringAiIntegrationTest") {
    description = "Run the Spring AI tool loop against the pinned official GPT-OSS 20B checkpoint"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("integration")
    }
    filter {
        includeTestsMatching(
            "com.integrallis.models.spring.ai.GptOssOfficialToolCallingIntegrationTest",
        )
    }
    outputs.upToDateWhen { false }
    maxParallelForks = 1
    maxHeapSize = "4g"
}
