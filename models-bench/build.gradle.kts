// models-bench — JMH benchmarks (not published as a library)

plugins {
    java
    application
    id("com.github.spotbugs")
    id("com.diffplug.spotless")
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
        "-Werror"
    ))
}

tasks.withType<Test> {
    useJUnitPlatform()
}

spotless {
    java {
        googleJavaFormat("1.35.0")
        removeUnusedImports()
        trimTrailingWhitespace()
        endWithNewline()
    }
}

tasks.named("spotbugsTest") {
    enabled = false
}

application {
    mainClass = "com.integrallis.models.bench.InferenceBenchmarkCli"
    applicationDefaultJvmArgs = listOf("--add-modules", "jdk.incubator.vector")
}

tasks.withType<JavaExec>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}

dependencies {
    implementation(project(":models-runtime"))
    implementation(project(":models-backend-purejava"))
    implementation("com.integrallis:vectors-core:0.1.0-SNAPSHOT")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.21.4")
    testImplementation("org.junit.jupiter:junit-jupiter:5.11.4")
    testImplementation("org.assertj:assertj-core:3.27.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher:1.11.4")
}

jmh {
    jvmArgs.addAll(listOf(
        "--add-modules", "jdk.incubator.vector",
        "-Xms2g", "-Xmx8g",
        "-XX:+UseG1GC"
    ))

    (project.findProperty("bench.model") as String?)?.let {
        jvmArgs.add("-Dmodels.bench.model=$it")
    }
    (project.findProperty("jmh.includes") as String?)?.let {
        includes.set(listOf(it))
    }
    (project.findProperty("jmh.fork") as String?)?.let {
        fork.set(it.toInt())
    }
    (project.findProperty("jmh.warmup") as String?)?.let {
        warmupIterations.set(it.toInt())
    }
    (project.findProperty("jmh.iterations") as String?)?.let {
        iterations.set(it.toInt())
    }
    (project.findProperty("jmh.timeOnIteration") as String?)?.let {
        timeOnIteration.set(it)
    }
    resultFormat.set(project.findProperty("jmh.resultFormat") as String? ?: "JSON")
    val resultExtension = (resultFormat.get() as String).lowercase()
    resultsFile.set(project.file("build/results/jmh/results.$resultExtension"))
}
