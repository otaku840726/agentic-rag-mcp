using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.MSBuild;
using Microsoft.Build.Locator;
using Newtonsoft.Json;

namespace RoslynAnalyzer
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Usage: RoslynAnalyzer <csproj_or_sln_path> [output_file]");
                return 1;
            }

            string inputPath = args[0];
            string? outputFile = args.Length > 1 ? args[1] : null;

            if (!File.Exists(inputPath))
            {
                Console.Error.WriteLine($"File not found: {inputPath}");
                return 1;
            }

            try
            {
                // Register MSBuild defaults
                if (!MSBuildLocator.IsRegistered)
                {
                    var instances = MSBuildLocator.QueryVisualStudioInstances().ToArray();
                    var instance = instances.Length == 1
                        ? instances[0]
                        : instances.FirstOrDefault(i => i.Version.Major >= 17) ?? instances.First();
                    
                    MSBuildLocator.RegisterInstance(instance);
                    Console.Error.WriteLine($"Using MSBuild from: {instance.MSBuildPath}");
                }

                // Create workspace
                using var workspace = MSBuildWorkspace.Create();
                
                // Subscribe to workspace failures
                workspace.WorkspaceFailed += (sender, e) =>
                {
                    Console.Error.WriteLine($"Workspace diagnostic: {e.Diagnostic.Message}");
                };

                // Determine if input is .sln or .csproj
                bool isSolution = inputPath.EndsWith(".sln", StringComparison.OrdinalIgnoreCase);
                
                IEnumerable<Compilation> compilations;
                
                if (isSolution)
                {
                    Console.Error.WriteLine($"Loading solution: {inputPath}");
                    var solution = await workspace.OpenSolutionAsync(inputPath);
                    
                    Console.Error.WriteLine($"Compiling {solution.Projects.Count()} projects in solution: {solution.FilePath}");
                    
                    // Get compilations for all projects
                    var compilationTasks = solution.Projects
                        .Select(p => p.GetCompilationAsync())
                        .ToList();
                    
                    var compilationResults = await Task.WhenAll(compilationTasks);
                    compilations = compilationResults.Where(c => c != null)!;
                }
                else
                {
                    Console.Error.WriteLine($"Loading project: {inputPath}");
                    var project = await workspace.OpenProjectAsync(inputPath);
                    
                    Console.Error.WriteLine($"Compiling project: {project.Name}");
                    var compilation = await project.GetCompilationAsync();
                    
                    if (compilation == null)
                    {
                        Console.Error.WriteLine("Failed to get compilation");
                        return 1;
                    }
                    
                    compilations = new[] { compilation };
                }

                // Analyze all compilations
                var allSymbols = new List<Symbol>();
                var allRelationships = new List<Relationship>();
                int totalFiles = 0;

                foreach (var compilation in compilations)
                {
                    // Check for compilation errors
                    var diagnostics = compilation.GetDiagnostics()
                        .Where(d => d.Severity == DiagnosticSeverity.Error)
                        .ToList();

                    if (diagnostics.Any())
                    {
                        Console.Error.WriteLine($"Warning: Compilation has {diagnostics.Count} errors");
                        foreach (var diag in diagnostics.Take(3))
                        {
                            Console.Error.WriteLine($"  {diag.GetMessage()}");
                        }
                    }

                    // Analyze all syntax trees in this compilation
                    totalFiles += compilation.SyntaxTrees.Count();
                    
                    foreach (var syntaxTree in compilation.SyntaxTrees)
                    {
                        var semanticModel = compilation.GetSemanticModel(syntaxTree);
                        var extractor = new SymbolExtractor(semanticModel, syntaxTree.FilePath);
                        
                        extractor.Visit(syntaxTree.GetRoot());
                        
                        allSymbols.AddRange(extractor.GetSymbols());
                        allRelationships.AddRange(extractor.GetRelationships());
                    }
                }

                Console.Error.WriteLine($"Analyzed {totalFiles} files");
                Console.Error.WriteLine($"Extracted {allSymbols.Count} symbols and {allRelationships.Count} relationships");

                // Create result
                var result = new AnalysisResult
                {
                    file_path = inputPath,
                    language = "csharp",
                    symbols = allSymbols,
                    relationships = allRelationships
                };

                string json = JsonConvert.SerializeObject(result, Formatting.Indented);

                if (outputFile != null)
                {
                    File.WriteAllText(outputFile, json);
                    Console.Error.WriteLine($"Output written to: {outputFile}");
                }
                else
                {
                    Console.WriteLine(json);
                }

                return 0;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"Error analyzing {inputPath}: {ex.Message}");
                Console.Error.WriteLine(ex.StackTrace);
                return 1;
            }
        }
    }
}
