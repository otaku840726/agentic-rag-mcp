using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

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

        // Class-level context for method inheritance
        private string? _classRoute;
        private string? _classFileType;
        private string? _classTableName;
        private bool? _classAuthRequired;
        private string? _classAuthRoles;

        public SymbolExtractor(SemanticModel semanticModel, string filePath) : base(SyntaxWalkerDepth.Node)
        {
            _semanticModel = semanticModel;
            _filePath = filePath;
        }

        public List<Symbol> GetSymbols() => _symbols;
        public List<Relationship> GetRelationships() => _relationships;

        private Symbol AddSymbol(SyntaxNode node, string name, string type, Dictionary<string, object>? metadata = null)
        {
            var span = node.Span;
            var lineSpan = node.SyntaxTree.GetLineSpan(span);

            var symbol = new Symbol
            {
                name = name,
                fqn = name,
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
            return symbol;
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

                AddRelationship(fullyQualifiedName, baseTypeName, "inherits");
            }

            // Interfaces
            if (symbol.Interfaces.Any())
            {
                var interfaces = symbol.Interfaces.Select(i => GetFullyQualifiedName(i)).ToList();
                meta["interfaces"] = interfaces;

                foreach (var iface in interfaces)
                    AddRelationship(fullyQualifiedName, iface, "implements");
            }

            // Modifiers
            if (node.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword)))
                meta["is_partial"] = true;

            var sym = AddSymbol(node, fullyQualifiedName, "class", meta);

            // Semantic enrichment
            var classAttrs = symbol.GetAttributes();
            sym.file_type      = ClassifyFileType(symbol, classAttrs);
            sym.visibility     = GetVisibility(symbol);
            sym.is_abstract    = symbol.IsAbstract ? (bool?)true : null;
            sym.is_static      = symbol.IsStatic   ? (bool?)true : null;
            sym.is_test        = IsTestClass(symbol, classAttrs);
            sym.is_deprecated  = IsDeprecated(symbol) ? (bool?)true : null;
            sym.table_name     = ExtractTableName(classAttrs);
            sym.auth_required  = IsAuthRequired(classAttrs);
            sym.auth_roles     = ExtractAuthRoles(classAttrs);

            // Push class-level context so methods can inherit it
            var prevClassRoute        = _classRoute;
            var prevClassFileType     = _classFileType;
            var prevClassTableName    = _classTableName;
            var prevClassAuthRequired = _classAuthRequired;
            var prevClassAuthRoles    = _classAuthRoles;

            _classRoute        = ExtractRouteTemplate(classAttrs);
            _classFileType     = sym.file_type;
            _classTableName    = sym.table_name;
            _classAuthRequired = sym.auth_required;
            _classAuthRoles    = sym.auth_roles;

            var previousClass = _currentClass;
            _currentClass = fullyQualifiedName;

            base.VisitClassDeclaration(node);

            _currentClass      = previousClass;
            _classRoute        = prevClassRoute;
            _classFileType     = prevClassFileType;
            _classTableName    = prevClassTableName;
            _classAuthRequired = prevClassAuthRequired;
            _classAuthRoles    = prevClassAuthRoles;
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
                        AddRelationship(fullyQualifiedName, iface, "inherits");
                }

                var sym = AddSymbol(node, fullyQualifiedName, "interface", meta);
                sym.visibility    = GetVisibility(symbol);
                sym.is_deprecated = IsDeprecated(symbol) ? (bool?)true : null;
            }

            base.VisitInterfaceDeclaration(node);
        }

        public override void VisitRecordDeclaration(RecordDeclarationSyntax node)
        {
            var symbol = _semanticModel.GetDeclaredSymbol(node);
            if (symbol != null)
            {
                var fullyQualifiedName = GetFullyQualifiedName(symbol);
                var sym = AddSymbol(node, fullyQualifiedName, "record");
                sym.visibility    = GetVisibility(symbol);
                sym.is_deprecated = IsDeprecated(symbol) ? (bool?)true : null;
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
                meta["is_virtual"] = true;

            var sym = AddSymbol(node, fullyQualifiedName, "method", meta);

            // Semantic enrichment
            var methodAttrs   = symbol.GetAttributes();
            sym.visibility    = GetVisibility(symbol);
            sym.is_static     = symbol.IsStatic   ? (bool?)true : null;
            sym.is_abstract   = symbol.IsAbstract ? (bool?)true : null;
            sym.is_deprecated = IsDeprecated(symbol) ? (bool?)true : null;

            // HTTP endpoint fields
            sym.http_method      = ExtractHttpMethod(methodAttrs);
            sym.http_path        = ExtractMethodPath(_classRoute, methodAttrs);
            sym.entry_point_type = sym.http_method != null ? "api" : null;

            // Auth: method-level overrides class-level
            sym.auth_required = IsAuthRequired(methodAttrs) ?? _classAuthRequired;
            sym.auth_roles    = ExtractAuthRoles(methodAttrs) ?? _classAuthRoles;

            // Inherit class-level context
            sym.file_type   = _classFileType;
            sym.table_name  = _classTableName;

            // Operation and HTTP call
            sym.operation_type = DetermineOperationType(sym.http_method, symbol.Name);
            sym.makes_http_call = DetectHttpCall(node) ? (bool?)true : null;

            // Type relationships
            foreach (var typeName in ExtractUserDefinedTypeNames(symbol.ReturnType))
                AddRelationship(fullyQualifiedName, typeName, "uses_type");
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
                var sym = AddSymbol(node, fullyQualifiedName, "constructor");
                sym.visibility    = GetVisibility(symbol);
                sym.is_deprecated = IsDeprecated(symbol) ? (bool?)true : null;
                sym.file_type     = _classFileType;

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
                var sym = AddSymbol(node, fullyQualifiedName, "property", meta);
                sym.visibility    = GetVisibility(symbol);
                sym.is_static     = symbol.IsStatic ? (bool?)true : null;
                sym.is_deprecated = IsDeprecated(symbol) ? (bool?)true : null;

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
                    var sym = AddSymbol(variable, fullyQualifiedName, "field", meta);
                    sym.visibility    = GetVisibility(fieldSymbol);
                    sym.is_static     = fieldSymbol.IsStatic ? (bool?)true : null;
                    sym.is_deprecated = IsDeprecated(fieldSymbol) ? (bool?)true : null;

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
                var sym = AddSymbol(node, fullyQualifiedName, "enum");
                sym.visibility    = GetVisibility(symbol);
                sym.is_deprecated = IsDeprecated(symbol) ? (bool?)true : null;
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

        // ── Relationship extraction from method bodies ─────────────

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

            string? methodName = null;
            if (node.Expression is MemberAccessExpressionSyntax ma)
                methodName = ma.Name.Identifier.ValueText;

            if (methodName == null) return;

            bool isPublish   = methodName == "Publish"   || methodName == "PublishAsync" || methodName == "Send";
            bool isSubscribe = methodName == "Subscribe" || methodName == "SubscribeAsync" || methodName == "Receive";

            if (!isPublish && !isSubscribe) return;

            var receiverText = node.Expression.ToString();
            bool likelyEasyNetQ = receiverText.Contains("PubSub") ||
                                  (receiverText.Contains("bus") && !receiverText.Contains("_redisDb"));
            if (!likelyEasyNetQ) return;

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
                if (node.ArgumentList.Arguments.Count > 0)
                {
                    var firstArg = node.ArgumentList.Arguments[0].Expression;
                    if (firstArg is LiteralExpressionSyntax idLit &&
                        idLit.IsKind(SyntaxKind.StringLiteralExpression))
                    {
                        metadata["queue_name"] = idLit.Token.ValueText;
                    }
                }

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

        // ── Semantic enrichment helpers ─────────────────────────────

        private string? ClassifyFileType(INamedTypeSymbol symbol, IEnumerable<AttributeData> attrs)
        {
            var name = symbol.Name;
            var attrNames = attrs.Select(a => a.AttributeClass?.Name ?? "").ToHashSet();

            // Controller: attribute-based
            if (attrNames.Contains("ApiController") || attrNames.Contains("ApiControllerAttribute") ||
                attrNames.Contains("Controller")    || attrNames.Contains("ControllerAttribute"))
                return "controller";
            // Controller: name-based
            if (name.EndsWith("Controller"))
                return "controller";
            // Controller: base type
            if (symbol.BaseType != null &&
                (symbol.BaseType.Name == "ControllerBase" || symbol.BaseType.Name == "Controller" ||
                 symbol.BaseType.Name == "ApiController"))
                return "controller";

            // Service
            if (name.EndsWith("Service") || name.EndsWith("ServiceImpl") || name.EndsWith("Manager"))
                return "service";

            // Repository
            if (name.EndsWith("Repository") || name.EndsWith("Repo") || name.EndsWith("Dao") || name.EndsWith("DAL"))
                return "repository";

            // Entity: attribute-based or name-based
            if (attrNames.Contains("Table") || attrNames.Contains("TableAttribute"))
                return "entity";
            if (name.EndsWith("Entity") || name.EndsWith("Model") || name.EndsWith("Dto") || name.EndsWith("DTO"))
                return "entity";

            // Middleware
            if (name.EndsWith("Middleware") || name.EndsWith("Filter") || name.EndsWith("Handler"))
                return "middleware";

            // Config
            if (name.EndsWith("Config") || name.EndsWith("Configuration") || name.EndsWith("Settings") || name.EndsWith("Options"))
                return "config";

            return null;
        }

        private string? GetVisibility(ISymbol symbol)
        {
            return symbol.DeclaredAccessibility switch
            {
                Accessibility.Public             => "public",
                Accessibility.Private            => "private",
                Accessibility.Protected          => "protected",
                Accessibility.Internal           => "internal",
                Accessibility.ProtectedOrInternal  => "protected_internal",
                Accessibility.ProtectedAndInternal => "private_protected",
                _ => null
            };
        }

        private bool? IsTestClass(INamedTypeSymbol symbol, IEnumerable<AttributeData> attrs)
        {
            var attrNames = attrs.Select(a => a.AttributeClass?.Name ?? "").ToHashSet();
            if (attrNames.Contains("TestClass")     || attrNames.Contains("TestClassAttribute") ||
                attrNames.Contains("TestFixture")   || attrNames.Contains("TestFixtureAttribute"))
                return true;
            var lower = symbol.Name.ToLower();
            if (lower.EndsWith("test") || lower.EndsWith("tests") || lower.EndsWith("spec"))
                return true;
            var fp = _filePath.ToLower().Replace("\\", "/");
            if (fp.Contains("/test/") || fp.Contains("/tests/") || fp.Contains("test.cs") || fp.Contains(".test."))
                return true;
            return null;
        }

        private bool IsDeprecated(ISymbol symbol)
        {
            return symbol.GetAttributes().Any(a =>
                a.AttributeClass?.Name == "Obsolete" ||
                a.AttributeClass?.Name == "ObsoleteAttribute");
        }

        private string? ExtractTableName(IEnumerable<AttributeData> attrs)
        {
            var tableAttr = attrs.FirstOrDefault(a =>
                a.AttributeClass?.Name == "Table" || a.AttributeClass?.Name == "TableAttribute");
            if (tableAttr != null && tableAttr.ConstructorArguments.Length > 0)
            {
                if (tableAttr.ConstructorArguments[0].Value is string tableName && !string.IsNullOrEmpty(tableName))
                    return tableName;
            }
            return null;
        }

        private string? ExtractRouteTemplate(IEnumerable<AttributeData> attrs)
        {
            var routeAttr = attrs.FirstOrDefault(a =>
                a.AttributeClass?.Name == "Route" || a.AttributeClass?.Name == "RouteAttribute");
            if (routeAttr != null && routeAttr.ConstructorArguments.Length > 0)
            {
                if (routeAttr.ConstructorArguments[0].Value is string route)
                    return route;
            }
            return null;
        }

        private bool? IsAuthRequired(IEnumerable<AttributeData> attrs)
        {
            return attrs.Any(a =>
                a.AttributeClass?.Name == "Authorize" || a.AttributeClass?.Name == "AuthorizeAttribute")
                ? (bool?)true : null;
        }

        private string? ExtractAuthRoles(IEnumerable<AttributeData> attrs)
        {
            var authAttr = attrs.FirstOrDefault(a =>
                a.AttributeClass?.Name == "Authorize" || a.AttributeClass?.Name == "AuthorizeAttribute");
            if (authAttr == null) return null;

            // Named argument: [Authorize(Roles = "Admin,User")]
            var rolesArg = authAttr.NamedArguments.FirstOrDefault(n => n.Key == "Roles");
            if (rolesArg.Value.Value is string roles && !string.IsNullOrEmpty(roles))
                return roles;

            return null;
        }

        private string? ExtractHttpMethod(IEnumerable<AttributeData> attrs)
        {
            foreach (var attr in attrs)
            {
                var name = attr.AttributeClass?.Name ?? "";
                switch (name)
                {
                    case "HttpGet":     case "HttpGetAttribute":     return "GET";
                    case "HttpPost":    case "HttpPostAttribute":    return "POST";
                    case "HttpPut":     case "HttpPutAttribute":     return "PUT";
                    case "HttpDelete":  case "HttpDeleteAttribute":  return "DELETE";
                    case "HttpPatch":   case "HttpPatchAttribute":   return "PATCH";
                    case "HttpHead":    case "HttpHeadAttribute":    return "HEAD";
                    case "HttpOptions": case "HttpOptionsAttribute": return "OPTIONS";
                }
            }
            return null;
        }

        private string? ExtractMethodPath(string? classRoute, IEnumerable<AttributeData> attrs)
        {
            string? methodRoute = null;
            foreach (var attr in attrs)
            {
                var name = attr.AttributeClass?.Name ?? "";
                bool isHttpVerb = name.StartsWith("Http") && name != "HttpContext" && name != "HttpContextAttribute";
                bool isRoute    = name == "Route" || name == "RouteAttribute";

                if ((isHttpVerb || isRoute) && attr.ConstructorArguments.Length > 0)
                {
                    if (attr.ConstructorArguments[0].Value is string route)
                    {
                        methodRoute = route;
                        break;
                    }
                }
            }

            if (classRoute == null && methodRoute == null) return null;
            if (classRoute == null) return NormalizePath(methodRoute!);
            if (methodRoute == null) return NormalizePath(classRoute);

            var combined = classRoute.TrimEnd('/') + "/" + methodRoute.TrimStart('/');
            return NormalizePath(combined);
        }

        private static string NormalizePath(string path)
        {
            // Preserve [controller]/[action] template tokens as-is (common in ASP.NET Core)
            if (!path.StartsWith("/")) path = "/" + path;
            return path;
        }

        private string? DetermineOperationType(string? httpMethod, string methodName)
        {
            if (httpMethod != null)
            {
                return httpMethod switch
                {
                    "GET"     => "READ",
                    "HEAD"    => "READ",
                    "POST"    => "WRITE",
                    "PUT"     => "WRITE",
                    "PATCH"   => "WRITE",
                    "DELETE"  => "DELETE",
                    _ => null
                };
            }

            var lower = methodName.ToLower();
            if (lower.StartsWith("get")    || lower.StartsWith("find")   || lower.StartsWith("list")  ||
                lower.StartsWith("search") || lower.StartsWith("query")  || lower.StartsWith("fetch") ||
                lower.StartsWith("read")   || lower.StartsWith("load")   || lower.StartsWith("count") ||
                lower.StartsWith("exists") || lower.StartsWith("check"))
                return "READ";

            if (lower.StartsWith("delete") || lower.StartsWith("remove") || lower.StartsWith("destroy") ||
                lower.StartsWith("purge")  || lower.StartsWith("clear"))
                return "DELETE";

            if (lower.StartsWith("create") || lower.StartsWith("add")    || lower.StartsWith("insert") ||
                lower.StartsWith("save")   || lower.StartsWith("store")  || lower.StartsWith("persist") ||
                lower.StartsWith("update") || lower.StartsWith("modify") || lower.StartsWith("edit")   ||
                lower.StartsWith("put")    || lower.StartsWith("set")    || lower.StartsWith("patch")  ||
                lower.StartsWith("post")   || lower.StartsWith("process"))
                return "WRITE";

            return null;
        }

        private static bool DetectHttpCall(MethodDeclarationSyntax node)
        {
            var text = node.ToString();
            return Regex.IsMatch(text,
                @"\b(HttpClient|_client\.|_httpClient\.|\.GetAsync|\.PostAsync|\.PutAsync|\.DeleteAsync|\.PatchAsync|\.SendAsync)\b");
        }

        // ── Type extraction helper ──────────────────────────────────

        private IEnumerable<string> ExtractUserDefinedTypeNames(ITypeSymbol type)
        {
            if (type == null) yield break;

            if (type.SpecialType != SpecialType.None) yield break;

            if (type.TypeKind == TypeKind.Error ||
                type.TypeKind == TypeKind.TypeParameter ||
                type.TypeKind == TypeKind.Dynamic) yield break;

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
                    if (named.IsGenericType)
                        foreach (var arg in named.TypeArguments)
                            foreach (var t in ExtractUserDefinedTypeNames(arg))
                                yield return t;
                    yield break;
                }

                yield return GetFullyQualifiedName(named);
                if (named.IsGenericType)
                    foreach (var arg in named.TypeArguments)
                        foreach (var t in ExtractUserDefinedTypeNames(arg))
                            yield return t;
            }
        }
    }
}
