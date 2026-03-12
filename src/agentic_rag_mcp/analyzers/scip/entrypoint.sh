#!/bin/bash
# SCIP universal entrypoint — TypeScript/JavaScript, Go, Java, Kotlin, Android.
# Usage: /entrypoint.sh <project_dir> [language]
#   language: typescript | javascript | go | java  (auto-detected if omitted)
#
# Outputs SCIP index as JSON to stdout. Logs to stderr.
set -e

PROJECT_DIR="${1:-.}"
OVERRIDE_LANG="${2:-}"

cd "$PROJECT_DIR"

# ── Language auto-detection ──────────────────────────────────────────────────
if [ -n "$OVERRIDE_LANG" ]; then
    LANG="$OVERRIDE_LANG"
elif [ -f "go.mod" ]; then
    LANG="go"
elif [ -f "pom.xml" ] || [ -f "build.gradle" ] || [ -f "build.gradle.kts" ] || \
     [ -f "settings.gradle" ] || [ -f "settings.gradle.kts" ]; then
    LANG="java"
elif [ -f "tsconfig.json" ] || find . -maxdepth 3 -name "*.ts" ! -path "*/node_modules/*" | grep -q .; then
    LANG="typescript"
elif find . -maxdepth 3 -name "*.js" ! -path "*/node_modules/*" | grep -q .; then
    LANG="javascript"
else
    echo "ERROR: Could not detect language in $PROJECT_DIR" >&2
    exit 1
fi

echo "[scip-entrypoint] Language: $LANG, Dir: $PROJECT_DIR" >&2

INDEX_OUT="/tmp/index_$$.scip"
trap "rm -f $INDEX_OUT" EXIT

# ── Android project detection ─────────────────────────────────────────────────
# Checks for AndroidManifest.xml or com.android.application/library plugin usage
is_android_project() {
    find . -maxdepth 5 -name "AndroidManifest.xml" ! -path "*/build/*" | grep -q . && return 0
    grep -rl "com\.android\.application\|com\.android\.library\|com\.android\.kotlin\.multiplatform\.library" \
         --include="*.gradle" --include="*.gradle.kts" . 2>/dev/null | grep -q . && return 0
    return 1
}

case "$LANG" in
    typescript|javascript)
        # Install deps for type resolution (best-effort; failure is non-fatal)
        if [ -f "package.json" ] && [ ! -d "node_modules" ]; then
            echo "[scip-entrypoint] Installing npm deps..." >&2
            npm install --prefer-offline --ignore-scripts --no-audit --no-fund 1>&2 2>&1 || true
        fi
        echo "[scip-entrypoint] Running scip-typescript..." >&2
        scip-typescript index \
            --cwd "$PROJECT_DIR" \
            --output "$INDEX_OUT" \
            1>&2
        ;;
    go)
        echo "[scip-entrypoint] Running scip-go..." >&2
        scip-go index \
            --module "$(go list -m 2>/dev/null || echo module)" \
            --output "$INDEX_OUT" \
            1>&2
        ;;
    java)
        if is_android_project; then
            echo "[scip-entrypoint] Android project detected." >&2

            # ── Dynamically resolve and install required Android SDK platform ──
            COMPILE_SDK=$(grep -rh "compileSdk" \
                --include="*.gradle" --include="*.gradle.kts" . 2>/dev/null \
                | grep -o '[0-9]\{2,\}' | sort -n | tail -1)

            if [ -z "$COMPILE_SDK" ]; then
                echo "[scip-entrypoint] WARNING: Could not detect compileSdkVersion, defaulting to 35" >&2
                COMPILE_SDK=35
            fi

            PLATFORM_DIR="$ANDROID_HOME/platforms/android-${COMPILE_SDK}"
            BUILD_TOOLS_VERSION="${COMPILE_SDK}.0.0"
            BUILD_TOOLS_DIR="$ANDROID_HOME/build-tools/${BUILD_TOOLS_VERSION}"

            if [ ! -d "$PLATFORM_DIR" ]; then
                echo "[scip-entrypoint] Installing platforms;android-${COMPILE_SDK}..." >&2
                sdkmanager "platforms;android-${COMPILE_SDK}" 1>&2 || {
                    echo "[scip-entrypoint] ERROR: Failed to install android-${COMPILE_SDK}" >&2
                    exit 1
                }
            else
                echo "[scip-entrypoint] platforms;android-${COMPILE_SDK} already installed." >&2
            fi

            if [ ! -d "$BUILD_TOOLS_DIR" ]; then
                echo "[scip-entrypoint] Installing build-tools;${BUILD_TOOLS_VERSION}..." >&2
                sdkmanager "build-tools;${BUILD_TOOLS_VERSION}" 1>&2 || \
                    echo "[scip-entrypoint] WARNING: build-tools;${BUILD_TOOLS_VERSION} not available, using 35.0.0" >&2
            fi

            # Force all tasks to re-run via Gradle init script (bypasses UP-TO-DATE checks)
            mkdir -p "$HOME/.gradle/init.d"
            cat > "$HOME/.gradle/init.d/scip-force-rerun.gradle" << 'INITEOF'
gradle.taskGraph.whenReady { taskGraph ->
    taskGraph.allTasks.each { task ->
        task.outputs.upToDateWhen { false }
    }
}
INITEOF
            echo "[scip-entrypoint] Running scip-java (Android, compileSdk=${COMPILE_SDK})..." >&2
            scip-java index \
                --output "$INDEX_OUT" \
                1>&2
        else
            # Force all tasks to re-run via Gradle init script (bypasses UP-TO-DATE checks)
            mkdir -p "$HOME/.gradle/init.d"
            cat > "$HOME/.gradle/init.d/scip-force-rerun.gradle" << 'INITEOF'
gradle.taskGraph.whenReady { taskGraph ->
    taskGraph.allTasks.each { task ->
        task.outputs.upToDateWhen { false }
    }
}
INITEOF
            echo "[scip-entrypoint] Running scip-java (JVM)..." >&2
            scip-java index \
                --output "$INDEX_OUT" \
                1>&2
        fi
        ;;
    *)
        echo "ERROR: Unsupported language: $LANG" >&2
        exit 1
        ;;
esac

echo "[scip-entrypoint] Converting to JSON..." >&2
scip print --json "$INDEX_OUT"
