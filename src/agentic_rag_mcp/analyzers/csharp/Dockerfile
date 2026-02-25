# Build Stage
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY RoslynAnalyzer.csproj .
RUN dotnet restore

COPY . .
RUN dotnet publish -c Release -o /app

# Runtime Stage - sdk:9.0 supports net6/7/8/9 (backward compatible)
FROM mcr.microsoft.com/dotnet/sdk:9.0
WORKDIR /app
COPY --from=build /app .

# Entrypoint script: runs dotnet restore then Roslyn analysis
# ~/.nuget/packages on host should be mounted to /root/.nuget/packages
# Usage: docker run --rm \
#   -v /path/to/solution:/src \
#   -v ~/.nuget/packages:/root/.nuget/packages \
#   agentic-rag-roslyn-analyzer:latest /src/MyProject.sln
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Injected at build time by DockerAnalyzer auto-rebuild logic.
ARG SOURCE_HASH=unknown
LABEL source_hash=$SOURCE_HASH
