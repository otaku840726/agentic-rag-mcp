using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RoslynAnalyzer
{
    public class SymbolExtractor : CSharpSyntaxWalker
    {
        private readonly SemanticModel _semanticModel;
        private readonly string _filePath;
        private List<Symbol> _symbols = new List<Symbol>();
        private List<Relationship> _relationships = new List<Relationship>();
        
        // Context tracking
        private string? _currentNamespace;
        private string? _currentClass;
        private string? _currentMethod;

        public SymbolExtractor(SemanticModel semanticModel, string filePath) : base(SyntaxWalkerDepth.Node)
        {
            _semanticModel = semanticModel;
            _filePath = filePath;
        }

        public List<Symbol> GetSymbols() => _symbols;
        public List<Relationship> GetRelationships() => _relationships;

        private void AddSymbol(SyntaxNode node, string name, string type, Dictionary<string, object>? metadata = null)
        {
            var span = node.Span;
            var lineSpan = node.SyntaxTree.GetLineSpan(span);
            
            var symbol = new Symbol
            {
                name = name,
                node_type = type,
                content = node.ToString(),
                start_byte = span.Start,
                end_byte = span.End,
                start_line = lineSpan.StartLinePosition.Line,
                end_line = lineSpan.EndLinePosition.Line,
                metadata = metadata ?? new Dictionary<string, object>()
            };
            
            // Add file path and namespace
            symbol.metadata["file_path"] = _filePath;
            if (_currentNamespace != null)
            {
                symbol.metadata["namespace"] = _currentNamespace;
            }
            
            // Extract doc comments
            var trivia = node.GetLeadingTrivia();
            var docs = trivia.Where(t => t.IsKind(SyntaxKind.SingleLineDocumentationCommentTrivia) || 
                                         t.IsKind(SyntaxKind.MultiLineDocumentationCommentTrivia));
                                         
            if (docs.Any())
            {
                symbol.metadata["doc_comment"] = string.Join("\n", docs.Select(d => d.ToString()));
            }

            _symbols.Add(symbol);
        }

        private void AddRelationship(string source, string target, string type, Dictionary<string, object>? metadata = null)
        {
            _relationships.Add(new Relationship
            {
                source = source,
                target = target,
                type = type,
                metadata = metadata ?? new Dictionary<string, object>()
            });
        }

        private string GetFullyQualifiedName(ISymbol symbol)
        {
            if (symbol.ContainingType != null)
            {
                return $"{GetFullyQualifiedName(symbol.ContainingType)}.{symbol.Name}";
            }
            if (symbol.ContainingNamespace != null && !symbol.ContainingNamespace.IsGlobalNamespace)
            {
                return $"{symbol.ContainingNamespace.ToDisplayString()}.{symbol.Name}";
            }
            return symbol.Name;
        }

        public override void VisitNamespaceDeclaration(NamespaceDeclarationSyntax node)
        {
            var previousNamespace = _currentNamespace;
            _currentNamespace = node.Name.ToString();
            
            base.VisitNamespaceDeclaration(node);
            
            _currentNamespace = previousNamespace;
        }

        public override void VisitFileScopedNamespaceDeclaration(FileScopedNamespaceDeclarationSyntax node)
        {
            _currentNamespace = node.Name.ToString();
            base.VisitFileScopedNamespaceDeclaration(node);
        }

        public override void VisitClassDeclaration(ClassDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol == null)
            {
                base.VisitClassDeclaration(node);
                return;
            }

            var meta = new Dictionary<string, object>();
            var fullyQualifiedName = GetFullyQualifiedName(symbol);
            
            // Base types
            if (symbol.BaseType != null && symbol.BaseType.SpecialType != SpecialType.System_Object)
            {
                var baseTypeName = GetFullyQualifiedName(symbol.BaseType);
                meta["base_type"] = baseTypeName;
                
                // Add inheritance relationship
                AddRelationship(fullyQualifiedName, baseTypeName, "inherits");
            }
            
            // Interfaces
            if (symbol.Interfaces.Any())
            {
                var interfaces = symbol.Interfaces.Select(i => GetFullyQualifiedName(i)).ToList();
                meta["interfaces"] = interfaces;
                
                // Add implementation relationships
                foreach (var iface in interfaces)
                {
                    AddRelationship(fullyQualifiedName, iface, "implements");
                }
            }
            
            // Modifiers
            if (node.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)))
            {
                meta["is_partial"] = true;
            }
            if (node.Modifiers.Any(m => m.IsKind(SyntaxKind.AbstractKeyword)))
            {
                meta["is_abstract"] = true;
            }
            if (node.Modifiers.Any(m => m.IsKind(SyntaxKind.SealedKeyword)))
            {
                meta["is_sealed"] = true;
            }
            
            AddSymbol(node, fullyQualifiedName, "class", meta);
            
            var previousClass = _currentClass;
            _currentClass = fullyQualifiedName;
            
            base.VisitClassDeclaration(node);
            
            _currentClass = previousClass;
        }

        public override void VisitInterfaceDeclaration(InterfaceDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                var meta = new Dictionary<string, object>();
                
                // Base interfaces
                if (symbol.Interfaces.Any())
                {
                    var interfaces = symbol.Interfaces.Select(i => GetFullyQualifiedName(i)).ToList();
                    meta["base_interfaces"] = interfaces;
                    
                    foreach (var iface in interfaces)
                    {
                        AddRelationship(fullyQualifiedName, iface, "inherits");
                    }
                }
                
                AddSymbol(node, fullyQualifiedName, "interface", meta);
            }
            
            base.VisitInterfaceDeclaration(node);
        }
        
        public override void VisitRecordDeclaration(RecordDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                AddSymbol(node, fullyQualifiedName, "record");
            }
            
            base.VisitRecordDeclaration(node);
        }

        public override void VisitMethodDeclaration(MethodDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol == null)
            {
                base.VisitMethodDeclaration(node);
                return;
            }

            var fullyQualifiedName = GetFullyQualifiedName(symbol);
            var meta = new Dictionary<string, object>
            {
                { "return_type", symbol.ReturnType.ToDisplayString() },
                { "parameters", symbol.Parameters.Select(p => $"{p.Type.ToDisplayString()} {p.Name}").ToList() }
            };

            // Override info
            if (symbol.IsOverride)
            {
                meta["is_override"] = true;
                if (symbol.OverriddenMethod != null)
                {
                    var overriddenName = GetFullyQualifiedName(symbol.OverriddenMethod);
                    meta["overrides"] = overriddenName;
                    AddRelationship(fullyQualifiedName, overriddenName, "overrides");
                }
            }

            if (symbol.IsVirtual)
            {
                meta["is_virtual"] = true;
            }

            AddSymbol(node, fullyQualifiedName, "method", meta);

            // Gap 2: Return type → USES_TYPE
            foreach (var typeName in ExtractUserDefinedTypeNames(symbol.ReturnType))
                AddRelationship(fullyQualifiedName, typeName, "uses_type");

            // Gap 1: Parameter types → USES_TYPE
            foreach (var param in symbol.Parameters)
                foreach (var typeName in ExtractUserDefinedTypeNames(param.Type))
                    AddRelationship(fullyQualifiedName, typeName, "uses_type");

            var previousMethod = _currentMethod;
            _currentMethod = fullyQualifiedName;

            base.VisitMethodDeclaration(node);

            _currentMethod = previousMethod;
        }

        public override void VisitConstructorDeclaration(ConstructorDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                AddSymbol(node, fullyQualifiedName, "constructor");

                // Gap 4: Constructor parameter types → USES_TYPE
                foreach (var param in symbol.Parameters)
                    foreach (var typeName in ExtractUserDefinedTypeNames(param.Type))
                        AddRelationship(fullyQualifiedName, typeName, "uses_type");

                var previousMethod = _currentMethod;
                _currentMethod = fullyQualifiedName;

                base.VisitConstructorDeclaration(node);

                _currentMethod = previousMethod;
            }
        }

        public override void VisitPropertyDeclaration(PropertyDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                var meta = new Dictionary<string, object>
                {
                    { "type", symbol.Type.ToDisplayString() }
                };
                AddSymbol(node, fullyQualifiedName, "property", meta);
                
                // Add type reference relationship
                AddRelationship(fullyQualifiedName, symbol.Type.ToDisplayString(), "references");
            }
            
            base.VisitPropertyDeclaration(node);
        }
        
        public override void VisitFieldDeclaration(FieldDeclarationSyntax node)
        {
            foreach (var variable in node.Declaration.Variables)
            {
                var symbol = _semanticModel.GetDeclaredSymbol(variable);
                if (symbol is IFieldSymbol fieldSymbol)
                {
                    var fullyQualifiedName = GetFullyQualifiedName(fieldSymbol);
                    var meta = new Dictionary<string, object>
                    {
                        { "type", fieldSymbol.Type.ToDisplayString() }
                    };
                    AddSymbol(variable, fullyQualifiedName, "field", meta);
                    
                    // Add type reference relationship
                    AddRelationship(fullyQualifiedName, fieldSymbol.Type.ToDisplayString(), "references");
                }
            }
            
            base.VisitFieldDeclaration(node);
        }
        
        public override void VisitEnumDeclaration(EnumDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                AddSymbol(node, fullyQualifiedName, "enum");
            }
            
            base.VisitEnumDeclaration(node);
        }
        
        public override void VisitEnumMemberDeclaration(EnumMemberDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                AddSymbol(node, fullyQualifiedName, "enum_member");
            }
        }

        // Relationship extraction from method bodies

        public override void VisitInvocationExpression(InvocationExpressionSyntax node)
        {
            if (_currentMethod != null)
            {
                var symbolInfo = _semanticModel.GetSymbolInfo(node);
                if (symbolInfo.Symbol is IMethodSymbol methodSymbol)
                {
                    // Skip external/system calls (same filter as ExtractUserDefinedTypeNames)
                    var ns = methodSymbol.ContainingNamespace?.ToDisplayString() ?? "";
                    var isExternal = ns.StartsWith("System") || ns.StartsWith("Microsoft") ||
                                     ns.StartsWith("Newtonsoft") || ns.StartsWith("NLog") ||
                                     ns.StartsWith("AutoMapper") || ns.StartsWith("Flurl") ||
                                     ns.StartsWith("EasyNetQ") || ns.StartsWith("StackExchange");
                    if (isExternal)
                    {
                        // --- EasyNetQ messaging extraction (semantic path) ---
                        if (ns.StartsWith("EasyNetQ"))
                            ExtractEasyNetQRelationship(node, methodSymbol);
                        // -----------------------------------------------------
                        base.VisitInvocationExpression(node);
                        return;
                    }

                    var targetMethod = GetFullyQualifiedName(methodSymbol);
                    var lineSpan = node.SyntaxTree.GetLineSpan(node.Span);

                    AddRelationship(
                        _currentMethod,
                        targetMethod,
                        "calls",
                        new Dictionary<string, object> { { "line", lineSpan.StartLinePosition.Line } }
                    );
                }
                else
                {
                    // Fallback: syntax-level EasyNetQ detection when types are unresolved
                    // (e.g. NuGet packages unavailable during analysis)
                    TryExtractEasyNetQFromSyntax(node);
                }
            }

            base.VisitInvocationExpression(node);
        }

        public override void VisitObjectCreationExpression(ObjectCreationExpressionSyntax node)
        {
            if (_currentMethod != null)
            {
                var symbolInfo = _semanticModel.GetSymbolInfo(node);
                if (symbolInfo.Symbol is IMethodSymbol constructorSymbol)
                {
                    var typeName = constructorSymbol.ContainingType.ToDisplayString();
                    var lineSpan = node.SyntaxTree.GetLineSpan(node.Span);

                    AddRelationship(
                        _currentMethod,
                        typeName,
                        "creates",
                        new Dictionary<string, object> { { "line", lineSpan.StartLinePosition.Line } }
                    );
                }
            }

            base.VisitObjectCreationExpression(node);
        }

        /// <summary>
        /// Syntax-level EasyNetQ detection for when Roslyn cannot resolve types
        /// (e.g. NuGet packages unavailable). Looks for .PubSub.Publish/Subscribe patterns.
        /// </summary>
        private void TryExtractEasyNetQFromSyntax(InvocationExpressionSyntax node)
        {
            if (_currentMethod == null) return;

            // Extract method name from the expression
            string? methodName = null;
            if (node.Expression is MemberAccessExpressionSyntax ma)
                methodName = ma.Name.Identifier.ValueText;

            if (methodName == null) return;

            bool isPublish   = methodName == "Publish"   || methodName == "PublishAsync" || methodName == "Send";
            bool isSubscribe = methodName == "Subscribe" || methodName == "SubscribeAsync" || methodName == "Receive";

            if (!isPublish && !isSubscribe) return;

            // Guard: only match when receiver chain looks like EasyNetQ PubSub usage
            // e.g. _bus.PubSub.Publish(...) or _bus.PubSub.SubscribeAsync(...)
            var receiverText = node.Expression.ToString();
            bool likelyEasyNetQ = receiverText.Contains("PubSub") ||
                                  (receiverText.Contains("bus") && !receiverText.Contains("_redisDb"));
            if (!likelyEasyNetQ) return;

            // Delegate to shared extraction logic (passing null for methodSymbol)
            ExtractEasyNetQRelationshipFromSyntax(node, isPublish, isSubscribe);
        }

        private void ExtractEasyNetQRelationship(InvocationExpressionSyntax node, IMethodSymbol methodSymbol)
        {
            if (_currentMethod == null) return;

            var methodName = methodSymbol.Name;
            bool isPublish   = methodName == "Publish"   || methodName == "PublishAsync" || methodName == "Send";
            bool isSubscribe = methodName == "Subscribe" || methodName == "SubscribeAsync" || methodName == "Receive";

            if (!isPublish && !isSubscribe) return;

            var lineSpan = node.SyntaxTree.GetLineSpan(node.Span);
            var metadata = new Dictionary<string, object>
            {
                { "line", lineSpan.StartLinePosition.Line }
            };

            // Strategy 1: Generic type argument — bus.Publish<TMessage>() / bus.Subscribe<TMessage>()
            // Only use this when the type is a user-defined type (not string/primitive)
            if (methodSymbol.TypeArguments.Length > 0)
            {
                var msgType = methodSymbol.TypeArguments[0];
                if (msgType.SpecialType == SpecialType.None)
                {
                    var msgTypeFqn = msgType.ToDisplayString();
                    AddRelationship(_currentMethod, msgTypeFqn, isPublish ? "publishes_to" : "subscribes_to", metadata);
                    return;
                }
            }

            // Strategies 2 & 3 (topic-string and WithTopic patterns)
            ExtractEasyNetQRelationshipFromSyntax(node, isPublish, isSubscribe);
        }

        private void ExtractEasyNetQRelationshipFromSyntax(
            InvocationExpressionSyntax node, bool isPublish, bool isSubscribe)
        {
            if (_currentMethod == null) return;

            var lineSpan = node.SyntaxTree.GetLineSpan(node.Span);
            var metadata = new Dictionary<string, object>
            {
                { "line", lineSpan.StartLinePosition.Line }
            };

            // Strategy 2: Topic string as 2nd argument — bus.PubSub.Publish(message, "TopicName")
            if (isPublish && node.ArgumentList.Arguments.Count >= 2)
            {
                var topicArg = node.ArgumentList.Arguments[1].Expression;
                if (topicArg is LiteralExpressionSyntax topicLit &&
                    topicLit.IsKind(SyntaxKind.StringLiteralExpression))
                {
                    var topicFqn = $"RabbitMQ.Topic.{topicLit.Token.ValueText}";
                    AddRelationship(_currentMethod, topicFqn, "publishes_to", metadata);
                    return;
                }
            }

            // Strategy 3: WithTopic inside config lambda — SubscribeAsync<T>("id", handler, x => x.WithTopic("TopicName"))
            if (isSubscribe)
            {
                // Extract subscription ID from first string-literal arg
                if (node.ArgumentList.Arguments.Count > 0)
                {
                    var firstArg = node.ArgumentList.Arguments[0].Expression;
                    if (firstArg is LiteralExpressionSyntax idLit &&
                        idLit.IsKind(SyntaxKind.StringLiteralExpression))
                    {
                        metadata["queue_name"] = idLit.Token.ValueText;
                    }
                }

                // Search all argument subtrees for a WithTopic("TopicName") call
                foreach (var arg in node.ArgumentList.Arguments)
                {
                    var withTopicCall = arg.DescendantNodes()
                        .OfType<InvocationExpressionSyntax>()
                        .FirstOrDefault(inv =>
                            inv.Expression is MemberAccessExpressionSyntax maInner &&
                            maInner.Name.Identifier.ValueText == "WithTopic");

                    if (withTopicCall != null && withTopicCall.ArgumentList.Arguments.Count > 0)
                    {
                        var topicExpr = withTopicCall.ArgumentList.Arguments[0].Expression;
                        if (topicExpr is LiteralExpressionSyntax topicLit2 &&
                            topicLit2.IsKind(SyntaxKind.StringLiteralExpression))
                        {
                            var topicFqn = $"RabbitMQ.Topic.{topicLit2.Token.ValueText}";
                            AddRelationship(_currentMethod, topicFqn, "subscribes_to", metadata);
                            return;
                        }
                    }
                }
            }
        }

        // Gap 3: Enum member access in method bodies → USES_TYPE
        // e.g. enDepositStatus.TransferSuccess, enRobotType.Web
        public override void VisitMemberAccessExpression(MemberAccessExpressionSyntax node)
        {
            if (_currentMethod != null)
            {
                var symbolInfo = _semanticModel.GetSymbolInfo(node);
                if (symbolInfo.Symbol is IFieldSymbol fieldSymbol &&
                    fieldSymbol.ContainingType.TypeKind == TypeKind.Enum)
                {
                    var enumFqn = GetFullyQualifiedName(fieldSymbol.ContainingType);
                    AddRelationship(_currentMethod, enumFqn, "uses_type");
                }
            }

            base.VisitMemberAccessExpression(node);
        }

        // ── Helpers ───────────────────────────────────────────────

        // Returns FQNs of user-defined types embedded in a type expression.
        // Unwraps common generic wrappers (Task<T>, List<T>, IEnumerable<T>, etc.)
        // and nullable value types so that the inner user type is captured.
        private IEnumerable<string> ExtractUserDefinedTypeNames(ITypeSymbol type)
        {
            if (type == null) yield break;

            // Skip primitives (string, int, bool, etc.)
            if (type.SpecialType != SpecialType.None) yield break;

            // Skip unresolvable or generic type parameters (T, TResult)
            if (type.TypeKind == TypeKind.Error ||
                type.TypeKind == TypeKind.TypeParameter ||
                type.TypeKind == TypeKind.Dynamic) yield break;

            // Unwrap array element type
            if (type is IArrayTypeSymbol array)
            {
                foreach (var t in ExtractUserDefinedTypeNames(array.ElementType))
                    yield return t;
                yield break;
            }

            if (type is INamedTypeSymbol named)
            {
                // Unwrap Nullable<T>
                if (named.OriginalDefinition?.SpecialType == SpecialType.System_Nullable_T)
                {
                    foreach (var t in ExtractUserDefinedTypeNames(named.TypeArguments[0]))
                        yield return t;
                    yield break;
                }

                var ns = named.ContainingNamespace?.ToDisplayString() ?? "";
                bool isSystemType = ns.StartsWith("System") || ns.StartsWith("Microsoft") ||
                                    ns.StartsWith("Newtonsoft") || ns.StartsWith("NLog") ||
                                    ns.StartsWith("AutoMapper") || ns.StartsWith("Flurl") ||
                                    ns.StartsWith("EasyNetQ") || ns.StartsWith("StackExchange");

                if (isSystemType)
                {
                    // For system generic wrappers (Task<T>, List<T>, IEnumerable<T>, etc.)
                    // yield the user-defined type arguments instead
                    if (named.IsGenericType)
                        foreach (var arg in named.TypeArguments)
                            foreach (var t in ExtractUserDefinedTypeNames(arg))
                                yield return t;
                    yield break;
                }

                // User-defined generic type: yield itself + type arguments
                yield return GetFullyQualifiedName(named);
                if (named.IsGenericType)
                    foreach (var arg in named.TypeArguments)
                        foreach (var t in ExtractUserDefinedTypeNames(arg))
                            yield return t;
            }
        }
    }
}
