"""
Docker-based Analyzer Driver.
Runs code analysis tools inside Docker containers for isolation.

Auto-rebuild: if source_dir is provided, the image is automatically rebuilt
whenever the source files change (detected via MD5 hash stored in the image label).
"""

import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, List

from .analyzer import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)


class DockerAnalyzer(BaseAnalyzer):
    def __init__(self, image: str, command_template: str = "{file_path}",
                 extra_volumes: Optional[List[str]] = None,
                 source_dir: Optional[str] = None):
        """
        Args:
            image: Docker image to run (e.g., 'agentic-rag-spoon-analyzer:latest').
            command_template: Command arguments passed to the container.
                              Use {file_path} as placeholder for the in-container path.
            extra_volumes: Additional volume mounts in "host_path:container_path" format.
            source_dir: Path to the directory containing the Dockerfile and source files.
                        When provided, the image is auto-rebuilt if source files change.
        """
        self.image = image
        self.command_template = command_template
        self.extra_volumes = extra_volumes or []
        self.source_dir = source_dir
        # Cache: only check/rebuild once per DockerAnalyzer instance (per index run)
        self._image_verified = False

    # ── Auto-rebuild helpers ──────────────────────────────────────────────────

    def _compute_source_hash(self) -> Optional[str]:
        """MD5 of every file under source_dir, sorted for determinism."""
        src = Path(self.source_dir)
        if not src.exists():
            logger.warning(f"source_dir not found: {src}")
            return None
        h = hashlib.md5()
        for f in sorted(src.rglob("*")):
            if f.is_file():
                try:
                    h.update(f.read_bytes())
                except OSError:
                    pass
        return h.hexdigest()

    def _get_image_source_hash(self) -> Optional[str]:
        """Read the source_hash label stored in the Docker image."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format",
                 "{{index .Config.Labels \"source_hash\"}}", self.image],
                capture_output=True, text=True, check=True,
            )
            label = result.stdout.strip()
            return label if label and label != "<no value>" else None
        except subprocess.CalledProcessError:
            return None  # image doesn't exist yet

    def _rebuild_image(self, source_hash: str):
        """Rebuild the Docker image, embedding source_hash as a label."""
        logger.info(f"[auto-rebuild] Source changed — rebuilding {self.image} ...")
        cmd = [
            "docker", "build",
            "--build-arg", f"SOURCE_HASH={source_hash}",
            "-t", self.image,
            str(self.source_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Docker build failed for {self.image}:\n{result.stderr[-2000:]}"
            )
        logger.info(f"[auto-rebuild] {self.image} rebuilt successfully.")

    def _ensure_image_current(self):
        """Check if the image is up-to-date; rebuild automatically if not."""
        if self._image_verified or not self.source_dir:
            return
        self._image_verified = True  # mark before rebuild to avoid re-entry

        current_hash = self._compute_source_hash()
        if current_hash is None:
            return

        image_hash = self._get_image_source_hash()
        if current_hash == image_hash:
            logger.debug(f"[auto-rebuild] {self.image} is up-to-date (hash={current_hash[:8]})")
            return

        if image_hash is None:
            logger.info(f"[auto-rebuild] {self.image} not found or has no hash label — building.")
        else:
            logger.info(
                f"[auto-rebuild] {self.image} source changed "
                f"({image_hash[:8]} → {current_hash[:8]}) — rebuilding."
            )
        self._rebuild_image(current_hash)

    # ── Core analysis ─────────────────────────────────────────────────────────

    def analyze(self, file_path: str, content: Optional[str] = None) -> AnalysisResult:
        # Auto-rebuild image if source has changed (no-op after first call)
        self._ensure_image_current()

        abs_path = os.path.abspath(file_path)
        dir_name = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)

        mount_point = "/src"
        container_file_path = f"{mount_point}/{file_name}"

        import shlex
        cmd_args = shlex.split(self.command_template.format(file_path=container_file_path))

        extra_vol_flags = []
        for vol in self.extra_volumes:
            host_part, _, container_part = vol.partition(":")
            host_part = os.path.expanduser(host_part)
            extra_vol_flags += ["-v", f"{host_part}:{container_part}"]

        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{dir_name}:{mount_point}",
        ] + extra_vol_flags + [self.image] + cmd_args

        try:
            logger.info(f"Running docker analysis: {' '.join(docker_cmd)}")
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            output = result.stdout.strip()
            if not output:
                raise RuntimeError("Empty output from Docker analyzer")

            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1:
                json_str = output[start:end + 1]
                data = json.loads(json_str)
            else:
                data = json.loads(output)

            return AnalysisResult(
                file_path=file_path,
                language=data.get("language", "unknown"),
                symbols=data.get("symbols", []),
                relationships=data.get("relationships", []),
                raw_ast=data.get("raw_ast"),
            )

        except subprocess.CalledProcessError as e:
            err_msg = e.stderr or e.stdout or "Unknown error"
            logger.error(f"Docker analysis failed for {file_path}: {err_msg}")
            raise RuntimeError(f"Docker analysis failed: {err_msg}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {output[:200]}...")
            raise RuntimeError(f"Invalid JSON from Docker analyzer: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error in DockerAnalyzer: {e}")
            raise
