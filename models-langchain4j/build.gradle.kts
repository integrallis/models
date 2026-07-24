// models-langchain4j — LangChain4j chat model adapter

dependencies {
    api(project(":models-api"))
    api("dev.langchain4j:langchain4j:1.17.2")

    implementation(project(":models-runtime"))

    testImplementation("org.modeljars:modeljars-core:0.1.0-SNAPSHOT")
    testRuntimeOnly("org.modeljars:modeljars-catalog:0.1.0-SNAPSHOT")
}
