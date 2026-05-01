// models-spring-ai — Spring AI ChatModel + StreamingChatModel adapter

dependencies {
    api(project(":models-api"))

    // Spring AI provided by the consuming application
    compileOnly("org.springframework.ai:spring-ai-model:1.1.4")

    testImplementation("org.springframework.ai:spring-ai-model:1.1.4")
}
