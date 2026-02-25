#!/bin/bash
set -e

echo "Restoring NuGet packages for: $1" >&2
dotnet restore "$1" -p:EnableWindowsTargeting=true >&2

echo "Running Roslyn analysis..." >&2
exec dotnet RoslynAnalyzer.dll "$@"
