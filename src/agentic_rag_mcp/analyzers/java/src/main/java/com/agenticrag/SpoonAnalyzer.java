
package com.agenticrag;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.code.*;
import spoon.reflect.declaration.*;
import spoon.reflect.reference.*;
import spoon.reflect.visitor.CtScanner;

import java.io.File;
import java.nio.file.Path;
import java.util.*;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Spoon-based Java analyzer.
 * Usage: java -jar spoon-analyzer.jar /src/pom.xml
 * Outputs JSON to stdout in the same format as the Roslyn analyzer.
 */
public class SpoonAnalyzer {

    private static final ObjectMapper mapper = new ObjectMapper();
    private static Path srcRoot;

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: SpoonAnalyzer <path-to-pom.xml-or-src-dir>");
            System.exit(1);
        }

        File input = new File(args[0]);
        File srcDir;

        if (input.getName().equals("pom.xml")) {
            srcDir = input.getParentFile();
        } else {
            srcDir = input;
        }

        srcRoot = srcDir.toPath();

        // Find Java source root (src/main/java preferred, fallback to srcDir)
        File mainJava = new File(srcDir, "src/main/java");
        File sourcePath = mainJava.exists() ? mainJava : srcDir;

        Launcher launcher = new Launcher();
        launcher.addInputResource(sourcePath.getAbsolutePath());
        launcher.getEnvironment().setNoClasspath(true);  // allow analysis without full classpath
        launcher.getEnvironment().setCommentEnabled(false);
        launcher.getEnvironment().setComplianceLevel(17);

        try {
            launcher.buildModel();
        } catch (Exception e) {
            System.err.println("Warning: model build had errors (noClasspath mode): " + e.getMessage());
        }

        CtModel model = launcher.getModel();

        ArrayNode symbols = mapper.createArrayNode();
        ArrayNode relationships = mapper.createArrayNode();

        Set<String> seenSymbols = new HashSet<>();

        for (CtType<?> type : model.getAllTypes()) {
            processType(type, symbols, relationships, seenSymbols, srcDir.toPath());
        }

        ObjectNode output = mapper.createObjectNode();
        output.put("file_path", args[0]);
        output.put("language", "java");
        output.set("symbols", symbols);
        output.set("relationships", relationships);
        output.set("raw_ast", mapper.createObjectNode());

        System.out.println(mapper.writeValueAsString(output));
    }

    private static void processType(CtType<?> type, ArrayNode symbols, ArrayNode relationships,
                                     Set<String> seen, Path projectRoot) {
        if (type.isAnonymous() || type.getSimpleName().isEmpty()) return;

        String fqn = type.getQualifiedName();
        if (fqn == null || fqn.isEmpty()) return;

        String kind = getTypeKind(type);
        String filePath = getRelativeFilePath(type, projectRoot);
        int startLine = type.getPosition().isValidPosition() ? type.getPosition().getLine() : 0;
        int endLine = type.getPosition().isValidPosition() ? type.getPosition().getEndLine() : 0;

        // Compute class-level semantic fields once (used by class sym + passed to methods)
        ObjectNode classAnnotations = extractAnnotationValues(type.getAnnotations());
        String classFileType = classifyFileType(classAnnotations, type.getSimpleName(), filePath);
        // Class-level @RequestMapping base path for HTTP controller methods
        String httpBase = classAnnotations.has("RequestMapping")
            ? classAnnotations.get("RequestMapping").asText("") : "";
        // Class-level table name from @Table or Entity convention
        String classTableName = null;
        if (classAnnotations.has("Table")) {
            String tv = classAnnotations.get("Table").asText("");
            classTableName = tv.isEmpty() ? camelToSnake(type.getSimpleName()) : tv;
        } else if (classAnnotations.has("Entity")) {
            classTableName = camelToSnake(type.getSimpleName());
        }

        if (!seen.contains(fqn)) {
            seen.add(fqn);
            ObjectNode sym = mapper.createObjectNode();
            sym.put("fqn", fqn);
            sym.put("name", fqn);
            sym.put("node_type", kind);
            sym.put("content", getShortSignature(type));
            sym.put("start_line", startLine);
            sym.put("end_line", endLine);
            sym.put("start_byte", 0);
            sym.put("end_byte", 0);
            sym.put("visibility", getVisibilityString(type.getVisibility()));
            sym.put("is_static", type.getModifiers().contains(ModifierKind.STATIC));
            sym.put("is_abstract", type.getModifiers().contains(ModifierKind.ABSTRACT));
            sym.put("is_deprecated", hasDeprecated(type.getAnnotations()));
            sym.set("annotations", classAnnotations);
            if (classFileType != null) sym.put("file_type", classFileType);
            if (classTableName != null) sym.put("table_name", classTableName);

            ObjectNode meta = mapper.createObjectNode();
            meta.put("file_path", "/src/" + filePath);
            meta.put("namespace", type.getPackage() != null ? type.getPackage().getQualifiedName() : "");
            sym.set("metadata", meta);
            symbols.add(sym);
        }

        // Class-level annotations → ANNOTATED_BY
        addAnnotationEdges(type.getAnnotations(), fqn, relationships);

        // Superclass → INHERITS
        if (type instanceof CtClass<?> ctClass) {
            CtTypeReference<?> superRef = ctClass.getSuperclass();
            if (superRef != null && !superRef.getQualifiedName().equals("java.lang.Object")) {
                addRelationship(relationships, fqn, superRef.getQualifiedName(), "inherits");
            }
        }

        // Interfaces → IMPLEMENTS
        for (CtTypeReference<?> iface : type.getSuperInterfaces()) {
            addRelationship(relationships, fqn, iface.getQualifiedName(), "implements");
        }

        // Enum members → MEMBER_OF  [Bug 1 fix: enum members already had this]
        if (type instanceof CtEnum<?> ctEnum) {
            for (CtEnumValue<?> ev : ctEnum.getEnumValues()) {
                String memberFqn = fqn + "." + ev.getSimpleName();
                if (!seen.contains(memberFqn)) {
                    seen.add(memberFqn);
                    ObjectNode sym = mapper.createObjectNode();
                    sym.put("name", memberFqn);
                    sym.put("node_type", "enum_member");
                    sym.put("content", ev.getSimpleName());
                    sym.put("start_line", ev.getPosition().isValidPosition() ? ev.getPosition().getLine() : 0);
                    sym.put("end_line", ev.getPosition().isValidPosition() ? ev.getPosition().getEndLine() : 0);
                    sym.put("start_byte", 0);
                    sym.put("end_byte", 0);
                    sym.put("visibility", "public");
                    sym.put("is_static", true);
                    sym.put("is_abstract", false);
                    sym.put("is_deprecated", hasDeprecated(ev.getAnnotations()));
                    sym.set("annotations", extractAnnotationValues(ev.getAnnotations()));
                    ObjectNode meta = mapper.createObjectNode();
                    meta.put("file_path", "/src/" + filePath);
                    meta.put("namespace", type.getPackage() != null ? type.getPackage().getQualifiedName() : "");
                    sym.set("metadata", meta);
                    symbols.add(sym);
                    addRelationship(relationships, memberFqn, fqn, "member_of");
                }
            }
        }

        // Methods → symbols + MEMBER_OF + CALLS + USES_TYPE  [Bug 1 fix: MEMBER_OF now added]
        for (CtMethod<?> method : type.getMethods()) {
            processMethod(method, fqn, filePath, symbols, relationships, seen, type,
                          classAnnotations, httpBase, classFileType, classTableName);
        }

        // Bug 3 fix: Constructors → symbols + MEMBER_OF + CALLS + USES_TYPE
        // getConstructors() is only available on CtClass, not CtType base interface
        if (type instanceof CtClass<?> ctClassForCtors) {
            for (CtConstructor<?> ctor : ctClassForCtors.getConstructors()) {
                processConstructor(ctor, fqn, filePath, symbols, relationships, seen, type);
            }
        }

        // Bug 2 fix: Fields → symbols + MEMBER_OF + USES_TYPE
        for (CtField<?> field : type.getFields()) {
            processField(field, fqn, filePath, symbols, relationships, seen, type);
        }

        // Nested types
        for (CtType<?> nested : type.getNestedTypes()) {
            processType(nested, symbols, relationships, seen, projectRoot);
        }
    }

    private static void processMethod(CtMethod<?> method, String ownerFqn, String filePath,
                                       ArrayNode symbols, ArrayNode relationships,
                                       Set<String> seen, CtType<?> ownerType,
                                       ObjectNode classAnnotations, String httpBase,
                                       String classFileType, String classTableName) {
        // Bug 4 fix: include param types in FQN to support overloaded methods
        String paramSig = method.getParameters().stream()
            .map(p -> p.getType().getSimpleName())
            .collect(Collectors.joining(","));
        String methodFqn = ownerFqn + "." + method.getSimpleName() + "(" + paramSig + ")";
        int startLine = method.getPosition().isValidPosition() ? method.getPosition().getLine() : 0;
        int endLine = method.getPosition().isValidPosition() ? method.getPosition().getEndLine() : 0;

        if (!seen.contains(methodFqn)) {
            seen.add(methodFqn);
            ObjectNode methodAnnotations = extractAnnotationValues(method.getAnnotations());

            // ── Semantic enrichment (no Python post-processing needed) ──────
            // entry_point_type
            String entryPointType = null;
            String httpMethod = null;
            String httpPath = null;
            if (methodAnnotations.has("Scheduled")) {
                entryPointType = "cron";
            } else if (methodAnnotations.has("EventListener")) {
                entryPointType = "event";
            } else if (methodAnnotations.has("RabbitListener") || methodAnnotations.has("KafkaListener")) {
                entryPointType = "queue";
            } else {
                httpMethod = extractHttpMethod(methodAnnotations);
                if (httpMethod != null) {
                    entryPointType = "api";
                    String methodPath = extractHttpPath(methodAnnotations);
                    if (httpBase != null && !httpBase.isEmpty()) {
                        String base = httpBase.endsWith("/") ? httpBase.substring(0, httpBase.length() - 1) : httpBase;
                        String mp = (methodPath != null && !methodPath.isEmpty())
                            ? (methodPath.startsWith("/") ? methodPath : "/" + methodPath) : "";
                        httpPath = (base + mp).isEmpty() ? "/" : base + mp;
                        httpPath = httpPath.replaceAll("//+", "/");
                    } else {
                        httpPath = methodPath;
                    }
                }
            }

            // auth_required / auth_roles
            Boolean authRequired = null;
            String authRoles = null;
            if (methodAnnotations.has("PreAuthorize")) {
                authRequired = true;
                authRoles = methodAnnotations.get("PreAuthorize").asText("");
            } else if (methodAnnotations.has("Secured")) {
                authRequired = true;
                authRoles = methodAnnotations.get("Secured").asText("");
            } else if (methodAnnotations.has("RolesAllowed")) {
                authRequired = true;
                authRoles = methodAnnotations.get("RolesAllowed").asText("");
            }

            // operation_type (method name heuristics or @Modifying)
            String operationType = extractOperationType(methodAnnotations, method.getSimpleName());

            // table_name: inherit from class if present
            String tableName = classTableName;

            // makes_http_call: scan method body for HTTP client usage
            boolean makesHttpCall = detectHttpCall(method);

            ObjectNode sym = mapper.createObjectNode();
            sym.put("fqn", methodFqn);
            sym.put("name", methodFqn);
            sym.put("node_type", "method");
            sym.put("content", method.getSimpleName() + getParamSignature(method));
            sym.put("start_line", startLine);
            sym.put("end_line", endLine);
            sym.put("start_byte", 0);
            sym.put("end_byte", 0);
            sym.put("visibility", getVisibilityString(method.getVisibility()));
            sym.put("is_static", method.getModifiers().contains(ModifierKind.STATIC));
            sym.put("is_abstract", method.getModifiers().contains(ModifierKind.ABSTRACT));
            sym.put("is_deprecated", hasDeprecated(method.getAnnotations()));
            sym.put("return_type", method.getType() != null ? method.getType().getSimpleName() : "");
            sym.put("params", buildParamsJson(method.getParameters()));
            sym.set("annotations", methodAnnotations);
            // Semantic fields
            if (classFileType != null) sym.put("file_type", classFileType);
            if (entryPointType != null) sym.put("entry_point_type", entryPointType);
            if (httpMethod != null) sym.put("http_method", httpMethod);
            if (httpPath != null) sym.put("http_path", httpPath);
            if (authRequired != null) sym.put("auth_required", authRequired);
            if (authRoles != null) sym.put("auth_roles", authRoles);
            if (operationType != null) sym.put("operation_type", operationType);
            if (tableName != null) sym.put("table_name", tableName);
            if (makesHttpCall) sym.put("makes_http_call", true);
            ObjectNode meta = mapper.createObjectNode();
            meta.put("file_path", "/src/" + filePath);
            meta.put("namespace", ownerType.getPackage() != null ? ownerType.getPackage().getQualifiedName() : "");
            sym.set("metadata", meta);
            symbols.add(sym);
            // Bug 1 fix: method → owning class MEMBER_OF edge
            addRelationship(relationships, methodFqn, ownerFqn, "member_of");
        }

        // Method-level annotations → ANNOTATED_BY
        addAnnotationEdges(method.getAnnotations(), methodFqn, relationships);

        // Return type → USES_TYPE
        if (method.getType() != null) {
            collectUserDefinedTypes(method.getType()).forEach(t ->
                addRelationship(relationships, methodFqn, t, "uses_type"));
        }

        // Parameter types → USES_TYPE
        for (CtParameter<?> param : method.getParameters()) {
            collectUserDefinedTypes(param.getType()).forEach(t ->
                addRelationship(relationships, methodFqn, t, "uses_type"));
        }

        // @RabbitListener on method → SUBSCRIBES_TO
        extractRabbitListenerQueues(method).forEach(queue ->
            addRelationship(relationships, methodFqn, "RabbitMQ.Topic." + queue, "subscribes_to"));

        // Method body: invocations → CALLS, field accesses on enums → USES_TYPE
        if (method.getBody() != null) {
            method.getBody().accept(new CtScanner() {
                @Override
                public <T> void visitCtInvocation(CtInvocation<T> invocation) {
                    CtExecutableReference<?> exec = invocation.getExecutable();
                    String methodName = exec != null ? exec.getSimpleName() : null;

                    // Spring AMQP publisher detection
                    if (methodName != null && isAmqpPublishMethod(methodName)) {
                        String receiverText = invocation.getTarget() != null
                            ? invocation.getTarget().toString().toLowerCase() : "";
                        if (receiverText.contains("rabbit") || receiverText.contains("amqp")) {
                            extractAmqpRoutingKey(invocation).ifPresent(key ->
                                addRelationship(relationships, methodFqn, key, "publishes_to"));
                        }
                    }

                    // Bug 6 fix: fallback to target expression type when declaringType is null
                    if (exec != null) {
                        CtTypeReference<?> declType = exec.getDeclaringType();
                        if (declType == null && invocation.getTarget() != null) {
                            try { declType = invocation.getTarget().getType(); } catch (Exception ignored) {}
                        }
                        if (declType != null) {
                            String targetClass = declType.getQualifiedName();
                            if (!isExternalType(targetClass)) {
                                String callee = targetClass + "." + exec.getSimpleName();
                                addRelationship(relationships, methodFqn, callee, "calls");
                            }
                        }
                    }
                    super.visitCtInvocation(invocation);
                }

                @Override
                public <T> void visitCtFieldRead(CtFieldRead<T> fieldRead) {
                    if (fieldRead.getVariable() != null && fieldRead.getVariable().getDeclaringType() != null) {
                        CtTypeReference<?> declType = fieldRead.getVariable().getDeclaringType();
                        // Enum access (e.g. MyEnum.VALUE) → USES_TYPE
                        try {
                            CtType<?> resolved = declType.getTypeDeclaration();
                            if (resolved instanceof CtEnum<?> && !isExternalType(declType.getQualifiedName())) {
                                addRelationship(relationships, methodFqn, declType.getQualifiedName(), "uses_type");
                            }
                        } catch (Exception ignored) {
                            // noClasspath mode may throw
                        }
                    }
                    super.visitCtFieldRead(fieldRead);
                }
            });
        }
    }

    // Bug 3 fix: process constructors as first-class symbols
    private static void processConstructor(CtConstructor<?> ctor, String ownerFqn, String filePath,
                                            ArrayNode symbols, ArrayNode relationships,
                                            Set<String> seen, CtType<?> ownerType) {
        String paramSig = ctor.getParameters().stream()
            .map(p -> p.getType().getSimpleName())
            .collect(Collectors.joining(","));
        String ctorFqn = ownerFqn + ".<init>(" + paramSig + ")";
        if (seen.contains(ctorFqn)) return;
        seen.add(ctorFqn);

        int startLine = ctor.getPosition().isValidPosition() ? ctor.getPosition().getLine() : 0;
        int endLine = ctor.getPosition().isValidPosition() ? ctor.getPosition().getEndLine() : 0;

        ObjectNode sym = mapper.createObjectNode();
        sym.put("name", ctorFqn);
        sym.put("node_type", "constructor");
        sym.put("content", ownerType.getSimpleName() + "(" + paramSig + ")");
        sym.put("start_line", startLine);
        sym.put("end_line", endLine);
        sym.put("start_byte", 0);
        sym.put("end_byte", 0);
        sym.put("visibility", getVisibilityString(ctor.getVisibility()));
        sym.put("is_static", false);
        sym.put("is_abstract", false);
        sym.put("is_deprecated", hasDeprecated(ctor.getAnnotations()));
        sym.put("params", buildParamsJson(ctor.getParameters()));
        sym.set("annotations", extractAnnotationValues(ctor.getAnnotations()));
        ObjectNode meta = mapper.createObjectNode();
        meta.put("file_path", "/src/" + filePath);
        meta.put("namespace", ownerType.getPackage() != null ? ownerType.getPackage().getQualifiedName() : "");
        sym.set("metadata", meta);
        symbols.add(sym);
        addRelationship(relationships, ctorFqn, ownerFqn, "member_of");

        // Constructor-level annotations → ANNOTATED_BY
        addAnnotationEdges(ctor.getAnnotations(), ctorFqn, relationships);

        // Parameter types → USES_TYPE
        for (CtParameter<?> param : ctor.getParameters()) {
            collectUserDefinedTypes(param.getType()).forEach(t ->
                addRelationship(relationships, ctorFqn, t, "uses_type"));
        }

        // Body → CALLS (Bug 6 fix applied here too)
        if (ctor.getBody() != null) {
            ctor.getBody().accept(new CtScanner() {
                @Override
                public <T> void visitCtInvocation(CtInvocation<T> invocation) {
                    CtExecutableReference<?> exec = invocation.getExecutable();
                    if (exec != null) {
                        CtTypeReference<?> declType = exec.getDeclaringType();
                        if (declType == null && invocation.getTarget() != null) {
                            try { declType = invocation.getTarget().getType(); } catch (Exception ignored) {}
                        }
                        if (declType != null) {
                            String targetClass = declType.getQualifiedName();
                            if (!isExternalType(targetClass)) {
                                String callee = targetClass + "." + exec.getSimpleName();
                                addRelationship(relationships, ctorFqn, callee, "calls");
                            }
                        }
                    }
                    super.visitCtInvocation(invocation);
                }
            });
        }
    }

    // Bug 2 fix: process fields as first-class symbols
    private static void processField(CtField<?> field, String ownerFqn, String filePath,
                                      ArrayNode symbols, ArrayNode relationships,
                                      Set<String> seen, CtType<?> ownerType) {
        String fieldFqn = ownerFqn + "." + field.getSimpleName();
        if (seen.contains(fieldFqn)) return;
        seen.add(fieldFqn);

        int startLine = field.getPosition().isValidPosition() ? field.getPosition().getLine() : 0;

        ObjectNode sym = mapper.createObjectNode();
        sym.put("name", fieldFqn);
        sym.put("node_type", "field");
        sym.put("content", field.getSimpleName());
        sym.put("start_line", startLine);
        sym.put("end_line", startLine);
        sym.put("start_byte", 0);
        sym.put("end_byte", 0);
        sym.put("visibility", getVisibilityString(field.getVisibility()));
        sym.put("is_static", field.getModifiers().contains(ModifierKind.STATIC));
        sym.put("is_abstract", false);
        sym.put("is_deprecated", hasDeprecated(field.getAnnotations()));
        sym.put("return_type", field.getType() != null ? field.getType().getSimpleName() : "");
        sym.set("annotations", extractAnnotationValues(field.getAnnotations()));
        ObjectNode meta = mapper.createObjectNode();
        meta.put("file_path", "/src/" + filePath);
        meta.put("namespace", ownerType.getPackage() != null ? ownerType.getPackage().getQualifiedName() : "");
        sym.set("metadata", meta);
        symbols.add(sym);
        addRelationship(relationships, fieldFqn, ownerFqn, "member_of");

        // Field-level annotations → ANNOTATED_BY (@Autowired, @Column, @Id, custom, ...)
        addAnnotationEdges(field.getAnnotations(), fieldFqn, relationships);

        // Field type → USES_TYPE
        if (field.getType() != null) {
            collectUserDefinedTypes(field.getType()).forEach(t ->
                addRelationship(relationships, fieldFqn, t, "uses_type"));
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /** Convert ModifierKind visibility to lowercase string, null → "package" */
    private static String getVisibilityString(ModifierKind vis) {
        if (vis == null) return "package";
        switch (vis) {
            case PUBLIC:    return "public";
            case PROTECTED: return "protected";
            case PRIVATE:   return "private";
            default:        return "package";
        }
    }

    /** True if any annotation is @Deprecated */
    private static boolean hasDeprecated(Collection<CtAnnotation<?>> annotations) {
        return annotations.stream().anyMatch(a -> {
            try { return a.getAnnotationType().getSimpleName().equals("Deprecated"); }
            catch (Exception e) { return false; }
        });
    }

    /**
     * Extract annotation name → first string value map for ALL annotations
     * (including Spring/external ones — used for payload enrichment in Python).
     * Examples:
     *   @GetMapping("/deposit")              → {"GetMapping": "/deposit"}
     *   @Table(name = "deposits")            → {"Table": "deposits"}
     *   @Scheduled(cron = "0 * * * * ?")     → {"Scheduled": "0 * * * * ?"}
     *   @Transactional                        → {"Transactional": ""}
     */
    private static ObjectNode extractAnnotationValues(Collection<CtAnnotation<?>> annotations) {
        ObjectNode result = mapper.createObjectNode();
        for (CtAnnotation<?> ann : annotations) {
            try {
                String simpleName = ann.getAnnotationType().getSimpleName();
                String value = extractPrimaryAnnotationValue(ann);
                result.put(simpleName, value);
            } catch (Exception ignored) {}
        }
        return result;
    }

    /**
     * Get the "primary" string value of an annotation.
     * Priority: value attribute → first named attribute → empty string.
     */
    private static String extractPrimaryAnnotationValue(CtAnnotation<?> ann) {
        // Try "value" first (most common: @GetMapping("/path"), @Table(name="..."))
        for (String attr : List.of("value", "name", "cron", "path", "mapping")) {
            try {
                CtExpression<?> expr = ann.getValue(attr);
                if (expr != null) {
                    List<String> vals = extractStringLiterals(expr);
                    if (!vals.isEmpty()) return String.join(",", vals);
                }
            } catch (Exception ignored) {}
        }
        // Try any attribute — skip non-path attributes (produces, consumes, etc.)
        try {
            Set<String> nonPathAttrs = Set.of("produces", "consumes", "headers", "params", "method");
            Map<String, CtExpression> allValues = ann.getValues();
            if (allValues != null && !allValues.isEmpty()) {
                for (Map.Entry<String, CtExpression> entry : allValues.entrySet()) {
                    if (nonPathAttrs.contains(entry.getKey())) continue;
                    List<String> vals = extractStringLiterals(entry.getValue());
                    if (!vals.isEmpty()) return String.join(",", vals);
                }
            }
        } catch (Exception ignored) {}
        return "";
    }

    /**
     * Build params JSON array string: [{"name":"amount","type":"BigDecimal"},...]
     */
    private static String buildParamsJson(List<CtParameter<?>> params) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < params.size(); i++) {
            if (i > 0) sb.append(",");
            CtParameter<?> p = params.get(i);
            String name = p.getSimpleName();
            String type = p.getType() != null ? p.getType().getSimpleName() : "Object";
            sb.append("{\"name\":\"").append(name.replace("\"", "\\\""))
              .append("\",\"type\":\"").append(type.replace("\"", "\\\""))
              .append("\"}");
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Emit ANNOTATED_BY edges for all non-external annotations on a code element.
     * Skips standard Java/Spring/Lombok annotations — only tracks user-defined
     * and project-local annotations (custom @interface types).
     */
    private static void addAnnotationEdges(Collection<CtAnnotation<?>> annotations,
                                            String sourceFqn, ArrayNode relationships) {
        for (CtAnnotation<?> ann : annotations) {
            try {
                String annFqn = ann.getAnnotationType().getQualifiedName();
                if (!isExternalType(annFqn)) {
                    addRelationship(relationships, sourceFqn, annFqn, "annotated_by");
                }
            } catch (Exception ignored) {
                // noClasspath mode may fail to resolve annotation type
            }
        }
    }

    private static List<String> collectUserDefinedTypes(CtTypeReference<?> ref) {
        if (ref == null) return Collections.emptyList();
        List<String> result = new ArrayList<>();
        collectUserDefinedTypesRec(ref, result);
        return result;
    }

    private static void collectUserDefinedTypesRec(CtTypeReference<?> ref, List<String> result) {
        if (ref == null) return;
        String fqn = ref.getQualifiedName();
        if (fqn == null || fqn.startsWith("[") || fqn.equals("void")) return;

        // Unwrap arrays
        if (ref instanceof CtArrayTypeReference<?> arr) {
            collectUserDefinedTypesRec(arr.getComponentType(), result);
            return;
        }

        if (!isExternalType(fqn)) {
            result.add(fqn);
        }

        // Recurse into generic type arguments
        for (CtTypeReference<?> arg : ref.getActualTypeArguments()) {
            collectUserDefinedTypesRec(arg, result);
        }
    }

    private static final Set<String> PRIMITIVES = Set.of(
        "int", "long", "double", "float", "boolean", "byte", "short", "char", "void",
        "Integer", "Long", "Double", "Float", "Boolean", "Byte", "Short", "Character",
        "String", "Object", "Number", "Void"
    );

    private static boolean isExternalType(String fqn) {
        if (fqn == null || fqn.isEmpty()) return true;
        if (PRIMITIVES.contains(fqn)) return true;
        // Generic wildcard / unknown
        if (fqn.startsWith("?") || fqn.contains("<")) return true;
        return fqn.startsWith("java.") || fqn.startsWith("javax.") ||
               fqn.startsWith("jakarta.") || fqn.startsWith("sun.") ||
               fqn.startsWith("com.sun.") || fqn.startsWith("org.springframework.") ||
               fqn.startsWith("org.slf4j.") || fqn.startsWith("org.apache.") ||
               fqn.startsWith("com.fasterxml.") || fqn.startsWith("io.") ||
               fqn.startsWith("lombok.") || fqn.startsWith("reactor.") ||
               fqn.startsWith("kotlin.");
    }

    private static String getTypeKind(CtType<?> type) {
        if (type instanceof CtEnum<?>) return "enum";
        if (type instanceof CtInterface<?>) return "interface";
        if (type instanceof CtAnnotationType<?>) return "annotation";
        if (type instanceof CtClass<?> ctClass && ctClass.isAbstract()) return "abstract_class";
        return "class";
    }

    private static String getShortSignature(CtType<?> type) {
        StringBuilder sb = new StringBuilder();
        if (!type.getModifiers().isEmpty()) {
            type.getModifiers().forEach(m -> sb.append(m.toString().toLowerCase()).append(" "));
        }
        sb.append(getTypeKind(type)).append(" ").append(type.getSimpleName());
        return sb.toString().trim();
    }

    private static String getParamSignature(CtMethod<?> method) {
        StringBuilder sb = new StringBuilder("(");
        List<CtParameter<?>> params = method.getParameters();
        for (int i = 0; i < params.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(params.get(i).getType().getSimpleName());
        }
        sb.append(")");
        return sb.toString();
    }

    private static String getRelativeFilePath(CtType<?> type, Path projectRoot) {
        try {
            if (type.getPosition().isValidPosition() && type.getPosition().getFile() != null) {
                Path filePath = type.getPosition().getFile().toPath();
                // Try to make relative to projectRoot
                try {
                    return projectRoot.relativize(filePath).toString().replace("\\", "/");
                } catch (IllegalArgumentException e) {
                    return filePath.toString().replace("\\", "/");
                }
            }
        } catch (Exception ignored) {}
        return type.getQualifiedName().replace(".", "/") + ".java";
    }

    // ── Spring AMQP Helpers ───────────────────────────────────────────────────

    private static boolean isAmqpPublishMethod(String name) {
        return name.equals("convertAndSend") || name.equals("send") ||
               name.equals("convertAndReceive") || name.equals("convertSendAndReceive");
    }

    /** Extract routing key / queue name from rabbitTemplate.convertAndSend(...) args.
     *  Handles both string literals ("my.queue") and constant references (MyKeys.MY_QUEUE). */
    private static Optional<String> extractAmqpRoutingKey(CtInvocation<?> inv) {
        List<CtExpression<?>> args = inv.getArguments();
        if (args.isEmpty()) return Optional.empty();
        // convertAndSend(exchange, routingKey, msg) → prefer index 0 (queue/exchange name)
        // convertAndSend(routingKey, msg) → use index 0
        int keyIndex = 0;
        return extractQueueName(args.get(keyIndex)).map(k -> "RabbitMQ.Topic." + k);
    }

    private static Optional<String> extractQueueName(CtExpression<?> expr) {
        // String literal: "my.queue"
        if (expr instanceof CtLiteral<?> lit && lit.getValue() instanceof String s && !s.isEmpty()) {
            return Optional.of(s);
        }
        // Constant field reference: MessageQueueKey.MY_QUEUE → use field name
        if (expr instanceof CtFieldRead<?> fr && fr.getVariable() != null) {
            String fieldName = fr.getVariable().getSimpleName();
            if (!fieldName.isEmpty() && !fieldName.equals("this")) {
                return Optional.of(fieldName);
            }
        }
        return Optional.empty();
    }

    /** Extract queue names from @RabbitListener(queues/value = ...) annotation. */
    private static List<String> extractRabbitListenerQueues(CtMethod<?> method) {
        List<String> queues = new ArrayList<>();
        for (CtAnnotation<?> ann : method.getAnnotations()) {
            if (!ann.getAnnotationType().getSimpleName().equals("RabbitListener")) continue;
            for (String attr : List.of("queues", "value")) {
                try {
                    CtExpression<?> val = ann.getValue(attr);
                    if (val != null) queues.addAll(extractStringLiterals(val));
                } catch (Exception ignored) {}
            }
        }
        return queues;
    }

    private static List<String> extractStringLiterals(CtExpression<?> expr) {
        List<String> result = new ArrayList<>();
        if (expr instanceof CtLiteral<?> lit && lit.getValue() instanceof String s) {
            result.add(s);
        } else if (expr instanceof CtFieldRead<?> fr && fr.getVariable() != null) {
            // Constant reference: MessageQueueKey.MY_QUEUE → use field name
            String name = fr.getVariable().getSimpleName();
            if (!name.isEmpty() && !name.equals("this")) result.add(name);
        } else if (expr instanceof CtNewArray<?> arr) {
            for (CtExpression<?> elem : arr.getElements()) {
                result.addAll(extractStringLiterals(elem));
            }
        }
        return result;
    }

    private static void addRelationship(ArrayNode relationships, String source, String target, String type) {
        if (source == null || target == null || source.equals(target)) return;
        ObjectNode rel = mapper.createObjectNode();
        rel.put("source", source);
        rel.put("target", target);
        rel.put("type", type);
        rel.set("metadata", mapper.createObjectNode());
        relationships.add(rel);
    }

    // ── Semantic enrichment helpers ───────────────────────────────────────────

    /**
     * Classify file_type from class-level Spring annotations + naming conventions.
     * Returns null if cannot be determined.
     */
    private static String classifyFileType(ObjectNode annotations, String simpleName, String filePath) {
        // Annotation-based (most reliable)
        if (annotations.has("RestController") || annotations.has("Controller")) return "controller";
        if (annotations.has("Service")) return "service";
        if (annotations.has("Repository")) return "repository";
        if (annotations.has("Entity")) return "entity";
        if (annotations.has("Configuration") || annotations.has("SpringBootApplication")) return "config";
        if (annotations.has("Aspect")) return "middleware";
        if (annotations.has("ControllerAdvice") || annotations.has("RestControllerAdvice")) return "middleware";
        if (annotations.has("Component")) return "service";

        // Naming convention fallback
        String name = simpleName.toLowerCase();
        if (name.endsWith("controller")) return "controller";
        if (name.endsWith("service") || name.endsWith("serviceimpl")) return "service";
        if (name.endsWith("repository") || name.endsWith("repo")) return "repository";
        if (name.endsWith("entity")) return "entity";
        if (name.endsWith("dto") || name.endsWith("vo") || name.endsWith("request")
                || name.endsWith("response") || name.endsWith("command") || name.endsWith("event")) return "dto";
        if (name.endsWith("config") || name.endsWith("configuration")) return "config";
        if (name.endsWith("job") || name.endsWith("task") || name.endsWith("scheduler")) return "job";
        if (name.endsWith("listener") || name.endsWith("handler")) return "event_listener";

        // Path-based fallback
        String fp = filePath.replace("\\", "/").toLowerCase();
        if (fp.contains("/controller/") || fp.contains("/controllers/")) return "controller";
        if (fp.contains("/service/") || fp.contains("/services/")) return "service";
        if (fp.contains("/repository/") || fp.contains("/repositories/")) return "repository";
        if (fp.contains("/entity/") || fp.contains("/entities/") || fp.contains("/model/")) return "entity";

        return null;
    }

    /** Extract HTTP verb from Spring mapping annotations. */
    private static String extractHttpMethod(ObjectNode annotations) {
        if (annotations.has("GetMapping")) return "GET";
        if (annotations.has("PostMapping")) return "POST";
        if (annotations.has("PutMapping")) return "PUT";
        if (annotations.has("DeleteMapping")) return "DELETE";
        if (annotations.has("PatchMapping")) return "PATCH";
        if (annotations.has("RequestMapping")) return "GET";  // default; actual method= not parsed
        return null;
    }

    /** Extract HTTP path from Spring mapping annotations. */
    private static String extractHttpPath(ObjectNode annotations) {
        for (String ann : List.of("GetMapping", "PostMapping", "PutMapping",
                                   "DeleteMapping", "PatchMapping", "RequestMapping")) {
            if (annotations.has(ann)) {
                return annotations.get(ann).asText("");
            }
        }
        return null;
    }

    /** Derive operation_type from @Modifying or method name prefix heuristics. */
    private static String extractOperationType(ObjectNode annotations, String methodName) {
        if (annotations.has("Modifying")) return "WRITE";
        String name = methodName.toLowerCase();
        if (name.startsWith("save") || name.startsWith("insert") || name.startsWith("update")
                || name.startsWith("delete") || name.startsWith("remove") || name.startsWith("create")
                || name.startsWith("add") || name.startsWith("put") || name.startsWith("write")) {
            return "WRITE";
        }
        if (name.startsWith("find") || name.startsWith("get") || name.startsWith("load")
                || name.startsWith("fetch") || name.startsWith("select") || name.startsWith("read")
                || name.startsWith("list") || name.startsWith("count") || name.startsWith("exists")
                || name.startsWith("query") || name.startsWith("search")) {
            return "READ";
        }
        return null;
    }

    /** Scan method body for known HTTP client types. */
    private static boolean detectHttpCall(CtMethod<?> method) {
        if (method.getBody() == null) return false;
        String body = method.getBody().toString();
        return body.contains("RestTemplate") || body.contains("WebClient")
            || body.contains("HttpClient") || body.contains("restTemplate")
            || body.contains("webClient") || body.contains("FeignClient");
    }

    /** Convert CamelCase class name to snake_case (for default table name). */
    private static String camelToSnake(String name) {
        String s = name.replaceAll("([A-Z]+)([A-Z][a-z])", "$1_$2");
        s = s.replaceAll("([a-z\\d])([A-Z])", "$1_$2");
        return s.toLowerCase();
    }
}
