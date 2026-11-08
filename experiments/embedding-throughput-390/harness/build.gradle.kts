plugins { application }

repositories { mavenCentral() }

java { toolchain { languageVersion.set(JavaLanguageVersion.of(25)) } }

dependencies {
    implementation("com.integrallis:backend-native:0.5.16")
    implementation("com.integrallis:backend-java:0.5.16")
    implementation("com.fasterxml.jackson.core:jackson-databind:2.22.3")
}

application { mainClass.set("EmbeddingProfile") }

dependencyLocking { lockAllConfigurations() }
