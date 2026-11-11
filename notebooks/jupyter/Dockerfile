# JDK 25, Jupyter Lab, and the DFLib JJava kernel for Models notebooks.
# Match the notebook base's Ubuntu 24.04 ABI and copy the complete JDK distribution.
FROM eclipse-temurin:25-jdk-noble AS java
FROM quay.io/jupyter/base-notebook:python-3.12

USER root

COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="$JAVA_HOME/bin:$PATH"

ARG JJAVA_VERSION=1.0-a7
ARG JJAVA_SHA256=74d23fe8d559d111206d6e4d920bd0f4a6ccb787bb5136f9578589b657712acc
# BuildKit fetches and verifies the archive outside RUN's network namespace. Image construction
# needs no apt repositories or container DNS; Python's standard library extracts the pinned ZIP.
ADD --checksum=sha256:${JJAVA_SHA256} https://github.com/dflib/jjava/releases/download/${JJAVA_VERSION}/jjava-${JJAVA_VERSION}-kernelspec.zip /tmp/jjava-kernelspec.zip
RUN java --version && javac --version && \
    python -m zipfile -e /tmp/jjava-kernelspec.zip /tmp/jjava-kernel && \
    jupyter kernelspec install /tmp/jjava-kernel --name=java --prefix=/opt/conda && \
    rm -rf /tmp/jjava-kernel /tmp/jjava-kernelspec.zip

COPY jupyter/kernel.json /opt/conda/share/jupyter/kernels/java/kernel.json
COPY jupyter/prepare-classpath.sh /usr/local/bin/before-notebook.d/20-models-classpath.sh
RUN chmod +x /usr/local/bin/before-notebook.d/20-models-classpath.sh

RUN mkdir -p /home/jovyan/.gradle && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan

USER ${NB_UID}

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.allow_root=True"]
