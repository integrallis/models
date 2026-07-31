// models-spring-ai — Spring AI ChatModel + StreamingChatModel adapter

val springAiVersion = providers.gradleProperty("springAiVersion").getOrElse("2.0.0")

dependencies {
    api(project(":models-runtime"))

    compileOnly("org.springframework.ai:spring-ai-model:$springAiVersion")
    testImplementation("org.springframework.ai:spring-ai-model:$springAiVersion")
}
