"""
Docker-based Analyzer Driver.
Runs code analysis tools inside Docker containers for isolation.
"""

import subprocess
import json
import logging
import os
from typing import Optional, Dict, Any, List
from .analyzer import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

class DockerAnalyzer(BaseAnalyzer):
    def __init__(self, image: str, command_template: str = "{file_path}",
                 extra_volumes: Optional[List[str]] = None):
        """
        Args:
            image: Docker image to run (e.g., 'mcr.microsoft.com/dotnet/sdk').
            command_template: Command arguments to pass to the container.
                              Use {file_path} as placeholder for the file path inside container.
                              The directory of the file will be mounted to /src.
            extra_volumes: Additional volume mounts in "host_path:container_path" format.
                           e.g. ["~/.nuget/packages:/root/.nuget/packages"]
        """
        self.image = image
        self.command_template = command_template
        self.extra_volumes = extra_volumes or []

    def analyze(self, file_path: str, content: Optional[str] = None) -> AnalysisResult:
        abs_path = os.path.abspath(file_path)
        dir_name = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)

        # We mount the file's directory to /src in the container
        mount_point = "/src"
        container_file_path = f"{mount_point}/{file_name}"

        # Prepare command arguments
        import shlex
        cmd_args = shlex.split(self.command_template.format(file_path=container_file_path))

        # Build extra volume flags, expanding ~ in host paths
        extra_vol_flags = []
        for vol in self.extra_volumes:
            host_part, _, container_part = vol.partition(":")
            host_part = os.path.expanduser(host_part)
            extra_vol_flags += ["-v", f"{host_part}:{container_part}"]

        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{dir_name}:{mount_point}",
        ] + extra_vol_flags + [
            self.image
        ] + cmd_args
        
        try:
            logger.info(f"Running docker analysis: {' '.join(docker_cmd)}")
            result = subprocess.run(
                docker_cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            output = result.stdout.strip()
            if not output:
                 raise RuntimeError("Empty output from Docker analyzer")
                 
            # Find JSON in output (in case of logs or extra text)
            # Simple heuristic: find first '{' and last '}'
            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1:
                json_str = output[start:end+1]
                data = json.loads(json_str)
            else:
                data = json.loads(output)
            
            # Validate/Transform to AnalysisResult
            return AnalysisResult(
                file_path=file_path, # Use original host path
                language=data.get("language", "unknown"),
                symbols=data.get("symbols", []),
                relationships=data.get("relationships", []),
                raw_ast=data.get("raw_ast")
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
