// models-langchain4j — LangChain4j chat model adapter

dependencies {
    api(project(":models-api"))
    api("dev.langchain4j:langchain4j:1.17.2")

    implementation(project(":models-runtime"))
}
