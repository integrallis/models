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
# Fetched in an outer retry loop, not only with wget's own --tries. wget retries the transient
# HTTP responses it is told to, but aborts immediately on a network-layer failure (exit 4), which
# is what a brief GitHub-releases outage looks like. On 2026-11-09 one such minute failed this
# step on main and on an unrelated PR at the same time, reddening a gate that had passed its
# previous 23 runs, for a reason no change of ours could cause.
RUN cd /tmp && \
    url="https://github.com/dflib/jjava/releases/download/${JJAVA_VERSION}/jjava-${JJAVA_VERSION}-kernelspec.zip" && \
    archive="jjava-${JJAVA_VERSION}-kernelspec.zip" && \
    for attempt in 1 2 3 4 5; do \
        if wget --no-verbose --timeout=30 --tries=3 --waitretry=5 \
                --retry-connrefused --retry-on-host-error \
                --retry-on-http-error=429,500,502,503,504 \
                -O "${archive}" "${url}"; then \
            break; \
        fi; \
        rm -f "${archive}"; \
        if [ "${attempt}" = 5 ]; then \
            echo "Could not fetch ${url} after 5 attempts." >&2; \
            exit 1; \
        fi; \
        echo "Fetch attempt ${attempt} failed; retrying in $((attempt * 10))s." >&2; \
        sleep $((attempt * 10)); \
    done && \
    unzip -q "${archive}" -d jjava-kernel && \
    jupyter kernelspec install jjava-kernel --name=java --prefix=/opt/conda && \
    rm -rf /tmp/jjava-kernel "/tmp/${archive}"

COPY jupyter/kernel.json /opt/conda/share/jupyter/kernels/java/kernel.json
COPY jupyter/prepare-classpath.sh /usr/local/bin/before-notebook.d/20-models-classpath.sh
RUN chmod +x /usr/local/bin/before-notebook.d/20-models-classpath.sh

RUN mkdir -p /home/jovyan/.gradle && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan

USER ${NB_UID}

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.allow_root=True"]
