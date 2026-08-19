// models-langchain4j — LangChain4j chat model adapter

val langchain4jVersion = providers.gradleProperty("langchain4jVersion").getOrElse("1.17.2")

dependencies {
    api(project(":models-runtime"))

    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")

    compileOnly("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j-core:$langchain4jVersion")
    testImplementation("dev.langchain4j:langchain4j:$langchain4jVersion")

    testImplementation(
        "com.integrallis:vectors-langchain4j:${providers.gradleProperty("vectorsVersion").get()}"
    )
}
