plugins {
    java
    application
    id("com.github.spotbugs")
}

java {
    toolchain {
        languageVersion = JavaLanguageVersion.of(25)
    }
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
    options.compilerArgs.addAll(listOf(
        "-parameters",
        "-Xlint:all",
        "-Xlint:-processing",
        "-Xlint:-incubating",
        "-Xlint:-classfile",
        "-Werror"
    ))
}

tasks.withType<Test> {
    useJUnitPlatform()
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

tasks.named("spotbugsTest") {
    enabled = false
}

application {
    mainClass = "com.integrallis.models.rag.RagBenchmarkCli"
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

dependencies {
    implementation(project(":models-rag"))
    implementation(project(":models-runtime"))
    implementation(project(":models-backend-purejava"))
    implementation(project(":models-langchain4j"))
    implementation(project(":models-spring-ai"))
    implementation("org.apache.lucene:lucene-core:10.4.0")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")
    implementation("org.springframework.ai:spring-ai-rag:2.0.0")
    runtimeOnly("org.modeljars:modeljars-catalog:0.1.0-SNAPSHOT")

    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}
