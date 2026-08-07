// models-router — Multidimensional model selection across local and hosted models

dependencies {
    // Persistent index the pretrained task classifier searches. Vectors is a tier below models,
    // so this direction is allowed; models-router stays free of any backend dependency and the
    // build wires in whichever embedding ModelJar the index is pinned to.
    api("com.integrallis:vectors-db:${providers.gradleProperty("vectorsVersion").get()}")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}
