// models-langchain4j — LangChain4j ChatLanguageModel adapter

dependencies {
    api(project(":models-api"))

    // LangChain4j provided by the consuming application
    compileOnly("dev.langchain4j:langchain4j-core:1.13.1")

    testImplementation("dev.langchain4j:langchain4j-core:1.13.1")
}
