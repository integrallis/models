// models-backend-apple — Apple Foundation Models bridge via Java FFM

dependencies {
    api(project(":models-api"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    System.getProperty("models.apple.foundation.library")?.let {
        systemProperty("models.apple.foundation.library", it)
    }
}
