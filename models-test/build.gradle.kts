// models-test — VCR record/replay, model output assertions

dependencies {
    api(project(":models-api"))

    // VCR test harness from java-vectors
    implementation("com.integrallis:vectors-vcr-core:0.1.0-SNAPSHOT")
    implementation("com.integrallis:vectors-vcr-junit5:0.1.0-SNAPSHOT")

    // JUnit 5 (provided transitively, but explicit for clarity)
    implementation("org.junit.jupiter:junit-jupiter:5.11.4")
}
