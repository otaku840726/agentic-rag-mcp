"""
Project Type Detector

Automatically detects project types and recommends appropriate analyzers.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Set, Optional

logger = logging.getLogger(__name__)


class ProjectType(Enum):
    """Supported project types."""
    JAVA_SPRING = "java_spring"
    JAVA_GENERIC = "java_generic"
    DOTNET = "dotnet"
    NODEJS = "nodejs"
    PYTHON = "python"
    GO = "go"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AnalyzerType(Enum):
    """Available analyzer types."""
    SPOON = "spoon"           # Java via Docker
    ROSLYN = "roslyn"         # C# via Docker
    TREE_SITTER = "tree-sitter"  # Fallback for all languages


# Project type detection markers
PROJECT_MARKERS = {
    ProjectType.JAVA_SPRING: {
        "files": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "dirs": ["src/main/java"],
        "patterns": ["**/application.properties", "**/application.yml"]
    },
    ProjectType.JAVA_GENERIC: {
        "files": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "dirs": ["src/main/java", "src"],
        "patterns": []
    },
    ProjectType.DOTNET: {
        "files": ["*.csproj", "*.sln", "*.fsproj", "*.vbproj"],
        "dirs": ["Properties", "obj", "bin"],
        "patterns": []
    },
    ProjectType.NODEJS: {
        "files": ["package.json", "yarn.lock", "pnpm-lock.yaml"],
        "dirs": ["node_modules"],
        "patterns": []
    },
    ProjectType.PYTHON: {
        "files": ["setup.py", "pyproject.toml", "requirements.txt", "Pipfile"],
        "dirs": [],
        "patterns": []
    },
    ProjectType.GO: {
        "files": ["go.mod", "go.sum"],
        "dirs": [],
        "patterns": []
    }
}


# File extension to analyzer mapping
EXTENSION_ANALYZER_MAP = {
    # Java
    ".java": {
        ProjectType.JAVA_SPRING: AnalyzerType.SPOON,
        ProjectType.JAVA_GENERIC: AnalyzerType.SPOON,
        "default": AnalyzerType.TREE_SITTER
    },
    # .NET
    ".cs": {
        ProjectType.DOTNET: AnalyzerType.ROSLYN,
        "default": AnalyzerType.TREE_SITTER
    },
    ".fs": {
        ProjectType.DOTNET: AnalyzerType.ROSLYN,
        "default": AnalyzerType.TREE_SITTER
    },
    ".vb": {
        ProjectType.DOTNET: AnalyzerType.ROSLYN,
        "default": AnalyzerType.TREE_SITTER
    },
    # All others default to tree-sitter
    ".js": {"default": AnalyzerType.TREE_SITTER},
    ".jsx": {"default": AnalyzerType.TREE_SITTER},
    ".ts": {"default": AnalyzerType.TREE_SITTER},
    ".tsx": {"default": AnalyzerType.TREE_SITTER},
    ".py": {"default": AnalyzerType.TREE_SITTER},
    ".go": {"default": AnalyzerType.TREE_SITTER},
    ".rs": {"default": AnalyzerType.TREE_SITTER},
    ".c": {"default": AnalyzerType.TREE_SITTER},
    ".cpp": {"default": AnalyzerType.TREE_SITTER},
    ".h": {"default": AnalyzerType.TREE_SITTER},
}


class ProjectDetector:
    """Detects project type and recommends analyzers."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def detect_project_type(self, directory: Path) -> ProjectType:
        """
        Detect the primary project type in a directory.
        
        Args:
            directory: Directory to analyze (absolute or relative to base_dir)
            
        Returns:
            Detected ProjectType
        """
        if not directory.is_absolute():
            directory = self.base_dir / directory
            
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return ProjectType.UNKNOWN

        detected_types = []
        
        # Check for each project type
        for project_type, markers in PROJECT_MARKERS.items():
            if self._has_markers(directory, markers):
                detected_types.append(project_type)
                logger.info(f"Detected {project_type.value} in {directory}")

        # Return result based on detections
        if not detected_types:
            return ProjectType.UNKNOWN
        elif len(detected_types) == 1:
            return detected_types[0]
        else:
            # Multiple project types detected
            logger.info(f"Multiple project types detected: {[t.value for t in detected_types]}")
            return ProjectType.MIXED

    def _has_markers(self, directory: Path, markers: Dict) -> bool:
        """Check if directory contains project type markers."""
        # Check for marker files
        for file_pattern in markers.get("files", []):
            if "*" in file_pattern:
                # Glob pattern - use rglob for recursive search
                # Convert *.ext to **/*.ext for recursive search
                if not file_pattern.startswith("**/"):
                    file_pattern = "**/" + file_pattern
                if list(directory.glob(file_pattern)):
                    return True
            else:
                # Exact filename
                if (directory / file_pattern).exists():
                    return True
        
        # Check for marker directories
        for dir_name in markers.get("dirs", []):
            if (directory / dir_name).exists():
                return True
        
        # Check for file patterns (recursive)
        for pattern in markers.get("patterns", []):
            if list(directory.glob(pattern)):
                return True
                
        return False

    def get_analyzer_for_file(
        self, 
        file_path: Path, 
        project_type: ProjectType
    ) -> AnalyzerType:
        """
        Determine the best analyzer for a specific file.
        
        Args:
            file_path: Path to the file
            project_type: Detected project type
            
        Returns:
            Recommended AnalyzerType
        """
        ext = file_path.suffix.lower()
        
        # Get analyzer mapping for this extension
        analyzer_map = EXTENSION_ANALYZER_MAP.get(ext)
        if not analyzer_map:
            # Unknown extension, use tree-sitter
            return AnalyzerType.TREE_SITTER
        
        # Check if there's a project-specific analyzer
        if project_type in analyzer_map:
            return analyzer_map[project_type]
        
        # Fall back to default
        return analyzer_map.get("default", AnalyzerType.TREE_SITTER)

    def get_analyzer_strategy(
        self, 
        directory: Path, 
        file_paths: list[Path]
    ) -> Dict[AnalyzerType, list[Path]]:
        """
        Group files by the analyzer that should process them.
        
        Args:
            directory: Directory being analyzed
            file_paths: List of files to analyze
            
        Returns:
            Dictionary mapping AnalyzerType to list of file paths
        """
        # Detect project type
        project_type = self.detect_project_type(directory)
        logger.info(f"Project type: {project_type.value}")
        
        # Group files by analyzer
        strategy: Dict[AnalyzerType, list[Path]] = {
            AnalyzerType.SPOON: [],
            AnalyzerType.ROSLYN: [],
            AnalyzerType.TREE_SITTER: []
        }
        
        for file_path in file_paths:
            analyzer = self.get_analyzer_for_file(file_path, project_type)
            strategy[analyzer].append(file_path)
        
        # Remove empty groups
        strategy = {k: v for k, v in strategy.items() if v}
        
        # Log strategy
        for analyzer, files in strategy.items():
            logger.info(f"{analyzer.value}: {len(files)} files")
        
        return strategy

    def is_analyzer_available(self, analyzer_type: AnalyzerType, image_name: str = None) -> bool:
        """
        Check if a specific analyzer is available.
        
        For Docker-based analyzers, this checks if Docker is running.
        If the image doesn't exist, it will attempt to build it automatically.
        
        Args:
            analyzer_type: Analyzer to check
            image_name: Optional Docker image name to check for Docker analyzers
            
        Returns:
            True if analyzer is available (or successfully built)
        """
        if analyzer_type == AnalyzerType.TREE_SITTER:
            # Tree-sitter is always available (built-in)
            return True
        
        # For Docker analyzers, check if Docker is available
        try:
            import subprocess
            import os
            
            # Check if Docker is running
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                logger.warning(f"Docker is not running")
                return False
            
            # If image name provided, check if it exists
            if image_name:
                result = subprocess.run(
                    ["docker", "images", "-q", image_name],
                    capture_output=True,
                    timeout=5,
                    text=True
                )
                if not result.stdout.strip():
                    logger.info(f"Docker image '{image_name}' not found locally, attempting to build...")
                    # Try to build the image
                    if self._build_docker_image(analyzer_type, image_name):
                        logger.info(f"Successfully built Docker image '{image_name}'")
                        return True
                    else:
                        logger.warning(f"Failed to build Docker image '{image_name}'")
                        return False
            
            return True
            
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"{analyzer_type.value} requires Docker, but Docker is not available: {e}")
            return False

    def _build_docker_image(self, analyzer_type: AnalyzerType, image_name: str) -> bool:
        """
        Build a Docker image for the specified analyzer type.
        
        Args:
            analyzer_type: Type of analyzer
            image_name: Name to tag the built image
            
        Returns:
            True if build succeeded
        """
        import subprocess
        from pathlib import Path
        
        # Determine Dockerfile location based on analyzer type
        dockerfile_dir = None
        
        if analyzer_type == AnalyzerType.ROSLYN:
            # Roslyn analyzer Dockerfile is in analyzers/csharp/
            dockerfile_dir = Path(__file__).parent.parent / "analyzers" / "csharp"
        elif analyzer_type == AnalyzerType.SPOON:
            # Spoon analyzer Dockerfile (if it exists)
            dockerfile_dir = Path(__file__).parent.parent / "analyzers" / "java"
        
        if not dockerfile_dir or not dockerfile_dir.exists():
            logger.error(f"Dockerfile directory not found for {analyzer_type.value}: {dockerfile_dir}")
            return False
        
        dockerfile_path = dockerfile_dir / "Dockerfile"
        if not dockerfile_path.exists():
            logger.error(f"Dockerfile not found at {dockerfile_path}")
            return False
        
        logger.info(f"Building Docker image '{image_name}' from {dockerfile_dir}")
        
        try:
            # Build the Docker image
            result = subprocess.run(
                ["docker", "build", "-t", image_name, "."],
                cwd=str(dockerfile_dir),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for build
            )
            
            if result.returncode == 0:
                logger.info(f"Docker build succeeded for {image_name}")
                return True
            else:
                logger.error(f"Docker build failed for {image_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Docker build timed out for {image_name}")
            return False
        except Exception as e:
            logger.error(f"Error building Docker image {image_name}: {e}")
            return False
