// models-spring-ai — Spring AI ChatModel + StreamingChatModel adapter

dependencies {
    api(project(":models-api"))
    api("org.springframework.ai:spring-ai-model:2.0.0")

    implementation(project(":models-runtime"))
}
