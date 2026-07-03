// models-embedding — Bridge to java-vectors: VectorCollection, SemanticCache
//
// Not in the published allowlist: it depends on unpublished vectors SNAPSHOTs, resolved in-place
// through the includeBuild("../vectors") composite build declared in settings.gradle.kts.

dependencies {
    api(project(":models-api"))

    // java-vectors embedding storage and semantic cache
    implementation("com.integrallis:vectors-db:0.1.0-SNAPSHOT")
    implementation("com.integrallis:vectors-cache-semantic-db:0.1.0-SNAPSHOT")
}
