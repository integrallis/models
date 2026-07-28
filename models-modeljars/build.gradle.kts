// models-modeljars - Optional adapter from ModelJars metadata to Models backend configuration

dependencies {
    api(project(":backend-java"))
    api(project(":backend-native"))
    api(project(":models-semantic-order"))
    api("org.modeljars:modeljars-core:0.1.0-SNAPSHOT")

    testRuntimeOnly("org.modeljars:modeljars-catalog:0.1.0-SNAPSHOT")
}
