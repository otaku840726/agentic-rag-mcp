# ── Stage 1: Build scip CLI and scip-go from source ─────────────────────────
FROM golang:1.24 AS go-builder

RUN go install github.com/sourcegraph/scip/cmd/scip@latest
RUN go install github.com/sourcegraph/scip-go/cmd/scip-go@latest

# ── Stage 2: Runtime image ───────────────────────────────────────────────────
FROM node:20-slim

# Copy Go binaries (scip CLI + scip-go)
COPY --from=go-builder /go/bin/scip     /usr/local/bin/scip
COPY --from=go-builder /go/bin/scip-go  /usr/local/bin/scip-go

# Install scip-typescript
RUN npm install -g @sourcegraph/scip-typescript

# ── Java 17 + scip-java (supports Java and Kotlin) ───────────────────────────
# Install via coursier bootstrap --standalone (fat jar from Maven Central, no external channel needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless curl gzip \
    && rm -rf /var/lib/apt/lists/* \
    && ARCH=$(dpkg --print-architecture) \
    && case "$ARCH" in \
         amd64) CS_ARCH="x86_64-pc-linux" ;; \
         arm64) CS_ARCH="aarch64-pc-linux" ;; \
         *) echo "Unsupported arch: $ARCH" && exit 1 ;; \
       esac \
    && curl -fL "https://github.com/coursier/launchers/raw/master/cs-${CS_ARCH}.gz" \
       | gzip -d > /tmp/cs \
    && chmod +x /tmp/cs \
    && /tmp/cs bootstrap --standalone \
         -o /usr/local/bin/scip-java \
         com.sourcegraph:scip-java_2.13:0.11.2 \
         --main com.sourcegraph.scip_java.ScipJava \
    && rm /tmp/cs

# ── Android SDK (for Android Kotlin/Java projects) ───────────────────────────
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools

RUN apt-get update && apt-get install -y --no-install-recommends unzip \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p $ANDROID_HOME/cmdline-tools \
    && curl -fLo /tmp/cmdline-tools.zip \
         "https://dl.google.com/android/repository/commandlinetools-linux-14742923_latest.zip" \
    && unzip -q /tmp/cmdline-tools.zip -d $ANDROID_HOME/cmdline-tools \
    && mv $ANDROID_HOME/cmdline-tools/cmdline-tools $ANDROID_HOME/cmdline-tools/latest \
    && rm /tmp/cmdline-tools.zip \
    && yes | sdkmanager --licenses > /dev/null \
    && sdkmanager "platform-tools" "build-tools;35.0.0"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Injected at build time by DockerAnalyzer auto-rebuild logic.
ARG SOURCE_HASH=unknown
LABEL source_hash=$SOURCE_HASH
