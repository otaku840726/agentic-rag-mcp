# Build Stage
FROM maven:3.9-eclipse-temurin-17 AS build
WORKDIR /build
COPY pom.xml .
RUN mvn dependency:go-offline -q
COPY src ./src
RUN mvn package -q -DskipTests

# Runtime Stage — use Maven image so entrypoint.sh can run mvn dependency:resolve
# Mount host ~/.m2/repository to /root/.m2/repository for package cache reuse
FROM maven:3.9-eclipse-temurin-17
WORKDIR /app
COPY --from=build /build/target/spoon-analyzer.jar .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Usage: docker run --rm \
#   -v /path/to/java-module:/src \
#   -v ~/.m2/repository:/root/.m2/repository \
#   agentic-rag-spoon-analyzer:latest /src/pom.xml
ENTRYPOINT ["/entrypoint.sh"]

# Injected at build time by DockerAnalyzer auto-rebuild logic.
# Stores a hash of all source files so the next run can detect changes.
ARG SOURCE_HASH=unknown
LABEL source_hash=$SOURCE_HASH
