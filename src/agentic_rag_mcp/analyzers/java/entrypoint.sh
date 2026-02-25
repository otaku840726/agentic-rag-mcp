#!/bin/bash
set -e

echo "Resolving Maven dependencies for: $1" >&2
mvn -f "$1" dependency:resolve -q 2>/dev/null || true

echo "Running Spoon analysis..." >&2
exec java -jar /app/spoon-analyzer.jar "$@"