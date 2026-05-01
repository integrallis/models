// models-runtime — Generation lifecycle, prompt/chat templates, sampling, Micrometer, JFR

dependencies {
    api(project(":models-api"))

    // Observability
    implementation("io.micrometer:micrometer-core:1.14.4")
}
