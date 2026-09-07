// models-router — Multidimensional model selection across local and hosted models

dependencies {
    // Persistent index the pretrained task classifier searches. Vectors is a tier below models,
    // so this direction is allowed; models-router stays free of any backend dependency and the
    // build wires in whichever embedding ModelJar the index is pinned to.
    api("com.integrallis:vectors-db:${providers.gradleProperty("vectorsVersion").get()}")

    // Runtime chat types let VirtualChatRouter adapt the provider-neutral decision engine to the
    // in-process virtual model. This still pulls no backend or ModelJars dependency, so ModelJars
    // remains free to implement the catalog SPI without inverting the tiers.
    api(project(":models-runtime"))

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}
