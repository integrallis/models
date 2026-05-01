// models-bench — JMH benchmarks (not published as a library)

plugins {
    java
    id("me.champeau.jmh") version "0.7.2"
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
        "-Werror",
        "--add-modules", "jdk.incubator.vector"
    ))
}

tasks.withType<Test> {
    jvmArgs("--add-modules", "jdk.incubator.vector")
    useJUnitPlatform()
}

dependencies {
    implementation(project(":models-runtime"))
    implementation(project(":models-backend-purejava"))
    implementation("org.slf4j:slf4j-api:2.0.16")
    runtimeOnly("ch.qos.logback:logback-classic:1.5.15")
}

jmh {
    jvmArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}
