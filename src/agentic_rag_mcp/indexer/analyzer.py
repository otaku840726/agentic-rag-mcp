"""
Analyzer Interface and Factory.
Defines the contract for code analysis drivers (TreeSitter, Roslyn/Docker, etc.).
"""

import abc
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Standardized result from an analysis driver."""
    file_path: str
    language: str
    symbols: List[Dict[str, Any]]  # Flat list of symbols (classes, methods) with metadata
    relationships: List[Dict[str, Any]]  # Call graph, inheritance, etc.
    raw_ast: Optional[Dict[str, Any]] = None  # Optional full AST dump

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

class BaseAnalyzer(abc.ABC):
    """Abstract base class for analysis drivers."""

    @abc.abstractmethod
    def analyze(self, file_path: str, content: Optional[str] = None) -> AnalysisResult:
        """
        Analyze a file and return structured results.
        
        Args:
            file_path: Absolute path to the file.
            content: Optional content override (if None, read from file).
        """
        pass

class AnalyzerFactory:
    """Factory to create the appropriate analyzer driver."""

    @staticmethod
    def create(driver_name: str = "tree-sitter", config: Optional[Dict[str, Any]] = None) -> BaseAnalyzer:
        """Create analyzer by explicit driver name (backward compatibility)."""
        if driver_name == "tree-sitter":
            # Lazy import to avoid circular deps
            from .ast_chunker import TreeSitterAnalyzer
            return TreeSitterAnalyzer()
        elif driver_name == "docker":
            from .docker_analyzer import DockerAnalyzer
            if config is None:
                config = {}
            image = config.get("image", "agentic-rag-analyzer:latest")
            # Default command uses only the file path
            command = config.get("command", "{file_path}")
            return DockerAnalyzer(image=image, command_template=command)
        else:
            raise ValueError(f"Unknown analyzer driver: {driver_name}")

    @staticmethod
    def create_auto(file_path: str, analyzer_type: str) -> BaseAnalyzer:
        """
        Create analyzer based on analyzer type from ProjectDetector.
        
        Args:
            file_path: Path to file being analyzed (for context)
            analyzer_type: AnalyzerType value (e.g., "spoon", "roslyn", "tree-sitter")
            
        Returns:
            Appropriate BaseAnalyzer instance
        """
        from .project_detector import AnalyzerType
        
        # Convert string to AnalyzerType if needed
        if isinstance(analyzer_type, str):
            analyzer_type = AnalyzerType(analyzer_type)
        
        if analyzer_type == AnalyzerType.TREE_SITTER:
            from .ast_chunker import TreeSitterAnalyzer
            return TreeSitterAnalyzer()
        
        elif analyzer_type == AnalyzerType.SPOON:
            # Use Docker analyzer with Spoon image
            from .docker_analyzer import DockerAnalyzer
            image = os.getenv("ANALYZER_SPOON_IMAGE", "agentic-rag-spoon-analyzer:latest")
            command = os.getenv("ANALYZER_SPOON_COMMAND", "{file_path}")
            # Mount host ~/.m2/repository so mvn dependency:resolve uses cache
            m2_cache = os.getenv(
                "ANALYZER_SPOON_M2_CACHE",
                "~/.m2/repository:/root/.m2/repository"
            )
            extra_volumes = [m2_cache] if m2_cache else []
            return DockerAnalyzer(image=image, command_template=command, extra_volumes=extra_volumes)
        
        elif analyzer_type == AnalyzerType.ROSLYN:
            # Use Docker analyzer with Roslyn image
            from .docker_analyzer import DockerAnalyzer
            image = os.getenv("ANALYZER_ROSLYN_IMAGE", "agentic-rag-roslyn-analyzer:latest")
            command = os.getenv("ANALYZER_ROSLYN_COMMAND", "{file_path}")
            # Mount host NuGet cache so dotnet restore doesn't re-download packages
            nuget_cache = os.getenv(
                "ANALYZER_ROSLYN_NUGET_CACHE",
                "~/.nuget/packages:/root/.nuget/packages"
            )
            extra_volumes = [nuget_cache] if nuget_cache else []
            return DockerAnalyzer(image=image, command_template=command, extra_volumes=extra_volumes)
        
        else:
            # Fallback to tree-sitter
            logger.warning(f"Unknown analyzer type {analyzer_type}, falling back to tree-sitter")
            from .ast_chunker import TreeSitterAnalyzer
            return TreeSitterAnalyzer()


def save_analysis_artifact(result: AnalysisResult, output_dir: str):
    """Save analysis result to a JSON file in the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the path
    # e.g. /path/to/src/main.py -> src_main_py_HASH.json
    import hashlib
    path_hash = hashlib.md5(result.file_path.encode()).hexdigest()[:8]
    basename = os.path.basename(result.file_path)
    safe_name = f"{basename}_{path_hash}.json"
    
    output_path = os.path.join(output_dir, safe_name)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.to_json())
    logger.info(f"Saved analysis to {output_path}")
