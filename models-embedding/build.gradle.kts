// models-embedding — Bridge to the released Vectors VectorCollection and SemanticCache APIs

dependencies {
    api(project(":models-api"))

    // java-vectors embedding storage and semantic cache
    implementation("com.integrallis:vectors-db:${providers.gradleProperty("vectorsVersion").get()}")
    implementation(
        "com.integrallis:vectors-cache-semantic-db:${providers.gradleProperty("vectorsVersion").get()}"
    )
}
