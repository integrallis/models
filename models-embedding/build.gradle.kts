// models-embedding — Bridge to java-vectors: VectorCollection, SemanticCache

dependencies {
    api(project(":models-api"))

    // java-vectors embedding storage and semantic cache
    implementation("com.integrallis:vectors-db:0.1.0-SNAPSHOT")
    implementation("com.integrallis:vectors-cache-semantic-db:0.1.0-SNAPSHOT")
}
