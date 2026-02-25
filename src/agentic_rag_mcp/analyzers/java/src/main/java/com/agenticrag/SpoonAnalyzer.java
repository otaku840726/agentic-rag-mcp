
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

        if (!seen.contains(fqn)) {
            seen.add(fqn);
            ObjectNode sym = mapper.createObjectNode();
            sym.put("name", fqn);
            sym.put("node_type", kind);
            sym.put("content", getShortSignature(type));
            sym.put("start_line", startLine);
            sym.put("end_line", endLine);
            sym.put("start_byte", 0);
            sym.put("end_byte", 0);

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
            processMethod(method, fqn, filePath, symbols, relationships, seen, type);
        }

        // Bug 3 fix: Constructors → symbols + MEMBER_OF + CALLS + USES_TYPE
        for (CtConstructor<?> ctor : type.getConstructors()) {
            processConstructor(ctor, fqn, filePath, symbols, relationships, seen, type);
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
                                       Set<String> seen, CtType<?> ownerType) {
        // Bug 4 fix: include param types in FQN to support overloaded methods
        String paramSig = method.getParameters().stream()
            .map(p -> p.getType().getSimpleName())
            .collect(Collectors.joining(","));
        String methodFqn = ownerFqn + "." + method.getSimpleName() + "(" + paramSig + ")";
        int startLine = method.getPosition().isValidPosition() ? method.getPosition().getLine() : 0;
        int endLine = method.getPosition().isValidPosition() ? method.getPosition().getEndLine() : 0;

        if (!seen.contains(methodFqn)) {
            seen.add(methodFqn);
            ObjectNode sym = mapper.createObjectNode();
            sym.put("name", methodFqn);
            sym.put("node_type", "method");
            sym.put("content", method.getSimpleName() + getParamSignature(method));
            sym.put("start_line", startLine);
            sym.put("end_line", endLine);
            sym.put("start_byte", 0);
            sym.put("end_byte", 0);
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
}
