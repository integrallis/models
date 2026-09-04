# Spring Boot development-launch inference reproduction

Date: 2026-09-04

## Question

Why does the same Spring AI and ModelJars application complete when started with `java -jar`, but
appear to hang inside `ChatClient` when started by Gradle `bootRun` or an IntelliJ Spring Boot run
configuration?

## Reproducer

- Application: `habuma/spring-ai-recipes`, commit
  `79e6852b3f2435d6cdd0f79691d26889a6fc7dbd`, subproject `local-llms/modeljars`
- Models baseline: `3f0adc46ee735f2db7faa43cf5d75b30be2ab517` (0.3.27)
- Model: Qwen3 1.7B Q8_0, SHA-256
  `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`
- Prompt: `What is the weather in 88252?`
- Host: macOS 26.6.2 x86-64, Intel Core i7-9750H, 12 logical processors
- JVM: Eclipse Temurin 25.0.3+9-LTS
- Gradle: 9.4.1
- Spring Boot: 4.1.0
- Spring AI: 2.0.0

The reproduction retained Craig's application wiring: a Spring context, `ChatClient.Builder`,
ordered `ChatClientBuilderCustomizer` beans, `MessageChatMemoryAdvisor`, a typed Java weather tool,
and Qwen's tool-result continuation.

## Observation

The default `bootRun` child command contained:

```text
--add-modules=jdk.incubator.vector -XX:TieredStopAtLevel=1
```

After accepting the prompt, a thread dump showed the main thread and all persistent GGUF workers
`RUNNABLE` in `PanamaVectorUtilSupport.q8_0Q8_0AccumulateBatchedBlock`. There was no lock cycle and
the prompt had already entered `ModelsSpringAiChatModel`. HotSpot reported only C1 compiler threads.
The absence of output was therefore unoptimized inference, not blocked console input or a Spring
tool-calling deadlock.

Spring Boot and IntelliJ enable the flag to reduce development startup time. It prevents HotSpot
from reaching tier 4/C2, which the Vector API kernels require for practical model inference.

With this Gradle configuration:

```groovy
tasks.named('bootRun') {
    standardInput = System.in
    optimizedLaunch = false
    jvmArgs '--add-modules', 'jdk.incubator.vector'
}
```

the unchanged application started, Qwen selected
`get-weather-for-zipcode("88252")`, Spring invoked the Java method, and Qwen returned:

```text
The weather in 88252 is currently raining cats and dogs, and the temperature is 78 degrees Fahrenheit.
```

## Product decision

- Reject `-Xint` and `-XX:TieredStopAtLevel=1..3` before a pure-Java model is parsed or inference
  starts. The exception names the Gradle, Maven, and IntelliJ remedies.
- Retain a subprocess regression that starts a real C1-only JVM and proves it fails promptly before
  touching model bytes.
- Retain a small Spring-context regression with the exact builder customizers, memory advisor,
  typed tool, tool invocation, and second model turn.
- Document launch optimization alongside the Spring AI and Spring Boot setup rather than treating
  `java -jar` success as sufficient development-tool coverage.

This finding does not require a native shim or a different inference engine. The normal HotSpot C2
path completes the existing pure-Java implementation.
