"""
Helper method for two-phase analysis strategy.

Phase 1: Analyze all .NET/Java projects with specialized analyzers
Phase 2: Analyze remaining files with tree-sitter
"""

def run_analysis_two_phase(
    self, 
    directory: str, 
    output_dir: Optional[str] = None,
    file_extensions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Two-phase analysis strategy:
    1. Find and analyze all .csproj/.sln (Roslyn) and Java projects (Spoon)
    2. Analyze remaining files with tree-sitter
    
    This ensures complete coverage without duplication.
    """
    from .project_detector import ProjectDetector, AnalyzerType, ProjectType
    from pathlib import Path
    import os
    
    # Resolve directory path
    dir_path = self.base_dir / directory
    if not dir_path.exists():
        return {"error": f"Directory not found: {directory}"}
    
    # Default output directory
    if output_dir is None:
        folder_name = Path(directory).name
        cache_dir = self.base_dir / ".agentic-rag-cache" / "analysis" / folder_name
        output_dir = str(cache_dir)
    else:
        cache_dir = Path(output_dir)
    
    # Clean up previous results
    if cache_dir.exists():
        import shutil
        logger.info(f"Cleaning up previous analysis results in {cache_dir}")
        shutil.rmtree(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    detector = ProjectDetector(self.base_dir)
    
    # Track all results
    total_analyzed = 0
    total_skipped = 0
    errors = []
    results_by_analyzer = {}
    analyzed_file_paths = set()  # Track which files have been analyzed
    
    # ===== PHASE 1: Project-level analysis =====
    logger.info("Phase 1: Analyzing .NET and Java projects")
    
    # Find all .csproj files
    csproj_files = list(dir_path.rglob("*.csproj"))
    logger.info(f"Found {len(csproj_files)} .csproj files")
    
    if csproj_files:
        image_name = os.getenv("ANALYZER_ROSLYN_IMAGE", "agentic-rag-roslyn-analyzer:latest")
        
        if detector.is_analyzer_available(AnalyzerType.ROSLYN, image_name):
            try:
                analyzer = AnalyzerFactory.create_auto(str(csproj_files[0]), "roslyn")
                analyzed_projects = []
                
                for csproj_path in csproj_files:
                    rel_path = str(csproj_path.relative_to(self.base_dir))
                    
                    try:
                        logger.info(f"Analyzing .NET project: {rel_path}")
                        result = analyzer.analyze(str(csproj_path))
                        save_analysis_artifact(result, output_dir)
                        analyzed_projects.append(rel_path)
                        total_analyzed += 1
                        
                        # Mark all .cs files in this project as analyzed
                        project_dir = csproj_path.parent
                        for cs_file in project_dir.rglob("*.cs"):
                            analyzed_file_paths.add(cs_file)
                        
                    except Exception as e:
                        errors.append({"project": rel_path, "error": str(e)})
                        logger.error(f"Error analyzing {rel_path}: {e}")
                        total_skipped += 1
                
                results_by_analyzer["roslyn"] = {
                    "projects_analyzed": len(analyzed_projects),
                    "projects": analyzed_projects
                }
                logger.info(f"Roslyn: analyzed {len(analyzed_projects)} projects")
                
            except Exception as e:
                logger.error(f"Failed to create Roslyn analyzer: {e}")
                errors.append({"analyzer": "roslyn", "error": str(e)})
        else:
            logger.warning("Roslyn analyzer not available, skipping .NET projects")
    
    # TODO: Add Java/Spoon project analysis here (similar pattern)
    
    # ===== PHASE 2: File-level analysis for remaining files =====
    logger.info("Phase 2: Analyzing remaining files with tree-sitter")
    
    # Collect all files not yet analyzed
    remaining_files = []
    for file_path in dir_path.rglob("*"):
        if not file_path.is_file():
            continue
        
        # Skip if already analyzed in Phase 1
        if file_path in analyzed_file_paths:
            continue
        
        # Apply exclusion logic
        rel_path = file_path.relative_to(self.base_dir)
        if self._is_excluded(Path(rel_path)):
            continue
        
        # Filter by extension if specified
        if file_extensions:
            if file_path.suffix.lower() not in file_extensions:
                continue
        
        # Skip project files themselves
        if file_path.suffix.lower() in ['.csproj', '.sln', '.pom', '.gradle']:
            continue
        
        remaining_files.append(file_path)
    
    logger.info(f"Found {len(remaining_files)} remaining files for tree-sitter analysis")
    
    if remaining_files:
        try:
            # Create tree-sitter analyzer
            analyzer = AnalyzerFactory.create_auto(str(remaining_files[0]), "tree-sitter")
            analyzed_files = []
            
            for file_path in remaining_files:
                rel_path = str(file_path.relative_to(self.base_dir))
                
                try:
                    result = analyzer.analyze(str(file_path))
                    save_analysis_artifact(result, output_dir)
                    analyzed_files.append(rel_path)
                    total_analyzed += 1
                    logger.debug(f"Analyzed {rel_path} with tree-sitter")
                    
                except Exception as e:
                    errors.append({"file": rel_path, "error": str(e)})
                    logger.error(f"Error analyzing {rel_path}: {e}")
                    total_skipped += 1
            
            results_by_analyzer["tree-sitter"] = {
                "files_analyzed": len(analyzed_files),
                "files": analyzed_files
            }
            logger.info(f"Tree-sitter: analyzed {len(analyzed_files)} files")
            
        except Exception as e:
            logger.error(f"Failed to create tree-sitter analyzer: {e}")
            errors.append({"analyzer": "tree-sitter", "error": str(e)})
    
    logger.info(
        f"Analysis completed: {total_analyzed} items analyzed, "
        f"{total_skipped} skipped/failed"
    )
    
    return {
        "directory": directory,
        "output_dir": output_dir,
        "total_items_analyzed": total_analyzed,
        "total_items_skipped": total_skipped,
        "errors": errors,
        "results_by_analyzer": results_by_analyzer
    }
