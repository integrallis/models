# JDK 25, Jupyter Lab, and the DFLib JJava kernel for Models notebooks.
FROM quay.io/jupyter/base-notebook:python-3.12

USER root

RUN apt-get update && \
    apt-get install --no-install-recommends -y wget gpg ca-certificates unzip && \
    wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | \
        gpg --dearmor -o /usr/share/keyrings/adoptium.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/adoptium.gpg] https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print $2}' /etc/os-release) main" \
        > /etc/apt/sources.list.d/adoptium.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y temurin-25-jdk && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s "$(ls -d /usr/lib/jvm/temurin-25-jdk-* | head -n1)" /usr/lib/jvm/temurin-25-jdk

ENV JAVA_HOME=/usr/lib/jvm/temurin-25-jdk
ENV PATH="$JAVA_HOME/bin:$PATH"

ARG JJAVA_VERSION=1.0-a7
RUN cd /tmp && \
    wget -q "https://github.com/dflib/jjava/releases/download/${JJAVA_VERSION}/jjava-${JJAVA_VERSION}-kernelspec.zip" && \
    unzip -q "jjava-${JJAVA_VERSION}-kernelspec.zip" -d jjava-kernel && \
    jupyter kernelspec install jjava-kernel --name=java --prefix=/opt/conda && \
    rm -rf /tmp/jjava-kernel "/tmp/jjava-${JJAVA_VERSION}-kernelspec.zip"

COPY jupyter/kernel.json /opt/conda/share/jupyter/kernels/java/kernel.json
COPY jupyter/prepare-classpath.sh /usr/local/bin/before-notebook.d/20-models-classpath.sh
RUN chmod +x /usr/local/bin/before-notebook.d/20-models-classpath.sh

RUN mkdir -p /home/jovyan/.gradle && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan

USER ${NB_UID}

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.allow_root=True"]
