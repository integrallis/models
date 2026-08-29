# Models Jupyter notebooks

Three executable Java notebooks exercise Models with the
[DFLib JJava](https://dflib.org/jjava/docs/1.x/) kernel on JDK 25. The
checked-in notebooks include the outputs from their last validated execution,
so they also render as complete examples without starting Jupyter.

| Notebook | Topic | Dependency modes |
|---|---|---|
| [01_getting_started.ipynb](01_getting_started.ipynb) | Generate a deterministic nano GGUF, load it with `backend-java`, and inspect real inference output and diagnostics | Source, release |
| [02_framework_adapters.ipynb](02_framework_adapters.ipynb) | Plain Java, LangChain4j, and Spring AI over one `TextGenerationModel` | Source, release |
| [03_guarded_rag.ipynb](03_guarded_rag.ipynb) | Trusted citations, derived citations, weak-retrieval abstention, and extractive fallback | Source, release |

## Dependency modes

Notebook cells contain no JAR paths or dependency versions. Gradle populates
`build/notebooks/classpath/`, and the JJava kernel loads that stable directory
through `JJAVA_CLASSPATH`.

- `source` resolves the current project outputs and their runtime dependencies.
- `release` resolves an exact set of `com.integrallis` artifacts from Maven
  Central or a supplied staging repository.

Prepare either classpath without Docker:

```bash
./gradlew prepareNotebookClasspath -PnotebookMode=source
./gradlew prepareNotebookClasspath \
  -PnotebookMode=release \
  -PnotebookVersion=0.3.14
```

Test a staged release before it reaches Maven Central:

```bash
./gradlew verifyStagedPublications
./gradlew prepareNotebookClasspath \
  -PnotebookMode=release \
  -PnotebookVersion=0.3.14 \
  -PnotebookRepository=build/staging-deploy
```

## Launch

Docker and Docker Compose are the host prerequisites:

```bash
cd notebooks
./docker-compose.sh up --build
```

Jupyter Lab is available at <http://localhost:8888> without a token or
password. The launcher uses anonymous access for the public container images
and does not read the host Docker credential store. To use published artifacts:

```bash
MODELS_NOTEBOOK_MODE=release \
MODELS_VERSION=0.3.14 \
./docker-compose.sh up --build
```

## Verify

The test script executes notebooks into `build/notebooks/executed/`, checks
every output cell, and leaves the checked-in files unchanged:

```bash
cd notebooks
./docker-compose.sh build
./docker-compose.sh run --rm --no-deps jupyter \
  bash /home/jovyan/work/models/notebooks/scripts/test-notebooks.sh
```

Release mode executes the same notebooks from the selected artifacts:

```bash
MODELS_NOTEBOOK_MODE=release \
MODELS_VERSION=0.3.14 \
./docker-compose.sh run --rm --no-deps jupyter \
  bash /home/jovyan/work/models/notebooks/scripts/test-notebooks.sh
```

`./gradlew verifyNotebooks` is the static release gate. It rejects hardcoded
JAR paths, stale artifact names, unexecuted cells, stale source digests,
checked-in execution errors, stderr, and incorrect semantic results.

After a successful source-mode execution, maintainers promote only the
validated execution counts and outputs:

```bash
python3 notebooks/scripts/validate_notebook_outputs.py \
  build/notebooks/executed
python3 notebooks/scripts/update_notebook_outputs.py \
  notebooks build/notebooks/executed
./gradlew verifyNotebooks
```

## Runtime

- **Kernel:** DFLib JJava `1.0-a7`, backed by JShell
- **JDK:** Eclipse Temurin 25
- **Vector API:** `jdk.incubator.vector` enabled for the kernel JVM and compiler
- **Native access:** enabled for Models FFM backends
