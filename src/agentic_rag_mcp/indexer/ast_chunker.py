"""
AST-based code chunking using tree-sitter.

Splits files by semantic boundaries (method, class, function, etc.) using
language-specific tree-sitter grammars. Falls back to line-based chunking
for any language without a grammar.
"""

import logging
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict

from .analyzer import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

# ── tree-sitter lazy init ──────────────────────────────────────────
# Each entry: ext → Parser (or None if unavailable)
_parsers: Dict[str, Any] = {}
_ts_init_done = False

# Convenience aliases kept for backward compat
_cs_parser = None
_java_parser = None

# Map extension → (module_import_name, language_attr)
_GRAMMAR_MAP: Dict[str, tuple] = {
    # ── .NET ──
    ".cs":         ("tree_sitter_c_sharp",    "language"),
    # ── JVM ──
    ".java":       ("tree_sitter_java",        "language"),
    ".kt":         ("tree_sitter_kotlin",      "language"),
    ".kts":        ("tree_sitter_kotlin",      "language"),
    # ── Python ──
    ".py":         ("tree_sitter_python",      "language"),
    # ── JavaScript / TypeScript ──
    ".js":         ("tree_sitter_javascript",  "language"),
    ".jsx":        ("tree_sitter_javascript",  "language"),
    ".cjs":        ("tree_sitter_javascript",  "language"),
    ".ts":         ("tree_sitter_typescript",  "language_typescript"),
    ".tsx":        ("tree_sitter_typescript",  "language_tsx"),
    # ── Go ──
    ".go":         ("tree_sitter_go",          "language"),
    # ── Rust ──
    ".rs":         ("tree_sitter_rust",        "language"),
    # ── Swift / Dart (Swift has package, Dart doesn't) ──
    ".swift":      ("tree_sitter_swift",       "language"),
    # ── Web ──
    ".html":       ("tree_sitter_html",        "language"),
    ".css":        ("tree_sitter_css",         "language"),
    # ── Data / Config ──
    ".json":       ("tree_sitter_json",        "language"),
    ".xml":        ("tree_sitter_xml",         "language_xml"),
    ".sql":        ("tree_sitter_sql",         "language"),
    # ── Shell ──
    ".sh":         ("tree_sitter_bash",        "language"),
    ".bash":       ("tree_sitter_bash",        "language"),
}


def _ensure_ts():
    """Lazy-init all available tree-sitter parsers (once)."""
    global _parsers, _ts_init_done, _cs_parser, _java_parser
    if _ts_init_done:
        return
    _ts_init_done = True

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        logger.warning("tree-sitter not installed, AST chunking disabled")
        return

    seen_modules: Dict[str, Any] = {}  # avoid re-importing same module for multiple exts

    for ext, (module_name, lang_attr) in _GRAMMAR_MAP.items():
        cache_key = f"{module_name}.{lang_attr}"
        if cache_key in seen_modules:
            _parsers[ext] = seen_modules[cache_key]
            continue
        try:
            mod = __import__(module_name)
            lang_fn = getattr(mod, lang_attr)
            parser = Parser(Language(lang_fn()))
            _parsers[ext] = parser
            seen_modules[cache_key] = parser
            logger.debug(f"Loaded tree-sitter grammar for {ext} ({module_name})")
        except Exception as e:
            logger.debug(f"tree-sitter grammar unavailable for {ext}: {e}")

    # backward-compat aliases
    _cs_parser   = _parsers.get(".cs")
    _java_parser = _parsers.get(".java")

    loaded = [ext for ext in _GRAMMAR_MAP if ext in _parsers]
    logger.info(f"tree-sitter grammars loaded for: {loaded}")


def is_available(ext: str) -> bool:
    """Check if AST chunking is available for the given file extension."""
    _ensure_ts()
    return ext in _parsers


# ── Data structures ────────────────────────────────────────────────

@dataclass
class ASTChunk:
    """A semantic code chunk extracted via AST."""
    content: str
    node_type: str          # method, constructor, property, class, enum, ...
    name: str               # identifier of the node
    parent_class: Optional[str] = None
    namespace: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    metadata_header: str = ""   # pre-built header for embedding enrichment


# ── Node type sets ─────────────────────────────────────────────────

CS_MEMBER_TYPES = {
    'method_declaration',
    'constructor_declaration',
    'property_declaration',
    'indexer_declaration',
    'operator_declaration',
    'event_declaration',
    'conversion_operator_declaration',
}

CS_TYPE_DECL_TYPES = {
    'class_declaration',
    'interface_declaration',
    'enum_declaration',
    'struct_declaration',
    'record_declaration',
    'record_struct_declaration',
}

JAVA_MEMBER_TYPES = {
    'method_declaration',
    'constructor_declaration',
}

JAVA_TYPE_DECL_TYPES = {
    'class_declaration',
    'interface_declaration',
    'enum_declaration',
    'record_declaration',
}


# ── Analyzer Implementation ────────────────────────────────────────

class TreeSitterAnalyzer(BaseAnalyzer):
    """
    In-process Analyzer using Tree-sitter.
    Acts as the default driver for 'analyze-code'.
    """

    def analyze(self, file_path: str, content: Optional[str] = None) -> AnalysisResult:
        """Analyze file using Tree-sitter parsers.

        For C# and Java: full semantic chunking (methods, classes, properties).
        For all other supported languages: generic AST-based chunking using
        top-level declarations (functions, classes, definitions).
        """
        if content is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

        ext = os.path.splitext(file_path)[1].lower()
        _ensure_ts()
        chunks: List[ASTChunk] = []

        if ext == '.cs':
            chunks = parse_csharp(content, file_path)
        elif ext == '.java':
            chunks = parse_java(content, file_path)
        elif ext in _parsers:
            # Generic AST chunking for all other supported languages
            chunks = parse_generic(content, file_path, ext, _parsers[ext])

        # Parse the tree once more for enhanced relationship extraction (CS/Java only)
        tree_root = None
        if ext == '.cs' and _cs_parser is not None:
            try:
                tree = _cs_parser.parse(bytes(content, 'utf-8'))
                tree_root = tree.root_node
            except Exception:
                pass
        elif ext == '.java' and _java_parser is not None:
            try:
                tree = _java_parser.parse(bytes(content, 'utf-8'))
                tree_root = tree.root_node
            except Exception:
                pass

        # Detect namespace for FQN building
        namespace = None
        if tree_root:
            namespace = _find_namespace(tree_root) or _find_package(tree_root)

        # Convert ASTChunks to AnalysisResult symbols
        symbols = []
        relationships = []
        seen_class_fqns = set()  # Track parent classes to create skeleton Symbol nodes

        for chunk in chunks:
            # Build FQN for each symbol
            fqn = _build_fqn(chunk.namespace or namespace, chunk.parent_class, chunk.name)

            # Map chunk to symbol
            symbol_data = asdict(chunk)
            # Remove metadata_header (it's internal to chunker)
            symbol_data.pop('metadata_header', None)
            # Add FQN
            symbol_data['fqn'] = fqn

            # Note: We KEEP 'content' because index_analysis needs it to generate embeddings.

            symbols.append(symbol_data)

            # Infer relationship (Parent-Child) with FQN
            if chunk.parent_class:
                parent_fqn = _build_fqn(chunk.namespace or namespace, None, chunk.parent_class)
                relationships.append({
                    "type": "member_of",
                    "source": fqn,
                    "target": parent_fqn,
                    "kind": chunk.node_type
                })
                # Ensure parent class has a Symbol node (skeleton, no content)
                if parent_fqn not in seen_class_fqns:
                    seen_class_fqns.add(parent_fqn)

        # Add skeleton Symbol entries for parent classes that aren't already in symbols
        existing_fqns = {s.get('fqn') for s in symbols}
        for class_fqn in seen_class_fqns:
            if class_fqn not in existing_fqns:
                class_name = class_fqn.rsplit('.', 1)[-1] if '.' in class_fqn else class_fqn
                symbols.append({
                    'name': class_name,
                    'node_type': 'class',
                    'fqn': class_fqn,
                    'namespace': namespace,
                    'parent_class': None,
                    'start_line': 0,
                    'end_line': 0,
                    'content': '',  # No content for skeleton nodes
                })

        # Enhanced relationship extraction from AST
        if tree_root:
            # Extract using/import statements
            using_rels = _extract_using_relationships(tree_root, file_path)
            relationships.extend(using_rels)

            # Extract inheritance and implementation relationships
            type_decl_types = CS_TYPE_DECL_TYPES if ext == '.cs' else JAVA_TYPE_DECL_TYPES
            inherit_rels = _extract_inheritance_relationships(tree_root, namespace, type_decl_types, ext)
            relationships.extend(inherit_rels)

            # Extract type references from properties/fields
            member_types = CS_MEMBER_TYPES if ext == '.cs' else JAVA_MEMBER_TYPES
            type_ref_rels = _extract_type_references(tree_root, namespace, type_decl_types, member_types, ext)
            relationships.extend(type_ref_rels)

        return AnalysisResult(
            file_path=file_path,
            language=ext[1:] if ext.startswith('.') else ext,
            symbols=symbols,
            relationships=relationships,
            raw_ast=None  # Tree-sitter AST is too verbose to dump raw
        )


# ── Public API ─────────────────────────────────────────────────────

def parse_csharp(content: str, file_path: str) -> List[ASTChunk]:
    """Parse a C# file into semantic chunks."""
    _ensure_ts()
    if _cs_parser is None:
        return []
    return _parse_file(
        _cs_parser, content, file_path,
        CS_MEMBER_TYPES, CS_TYPE_DECL_TYPES,
    )


def parse_java(content: str, file_path: str) -> List[ASTChunk]:
    """Parse a Java file into semantic chunks."""
    _ensure_ts()
    if _java_parser is None:
        return []
    return _parse_file(
        _java_parser, content, file_path,
        JAVA_MEMBER_TYPES, JAVA_TYPE_DECL_TYPES,
    )


# ── Generic multi-language parser ─────────────────────────────────

# Node types considered "top-level declarations" per language family.
# These are the boundaries we split on for semantic chunking.
_GENERIC_SPLIT_TYPES: Dict[str, set] = {
    # Python
    ".py":    {"function_definition", "class_definition", "decorated_definition"},
    # JavaScript / TypeScript
    ".js":    {"function_declaration", "class_declaration", "arrow_function",
               "export_statement", "method_definition"},
    ".jsx":   {"function_declaration", "class_declaration", "arrow_function",
               "export_statement", "jsx_element"},
    ".cjs":   {"function_declaration", "class_declaration", "export_statement"},
    ".ts":    {"function_declaration", "class_declaration", "interface_declaration",
               "type_alias_declaration", "enum_declaration", "export_statement"},
    ".tsx":   {"function_declaration", "class_declaration", "interface_declaration",
               "export_statement", "jsx_element"},
    # Go
    ".go":    {"function_declaration", "method_declaration", "type_declaration",
               "const_declaration", "var_declaration"},
    # Kotlin
    ".kt":    {"function_declaration", "class_declaration", "object_declaration",
               "property_declaration", "companion_object"},
    ".kts":   {"function_declaration", "class_declaration", "property_declaration"},
    # Rust
    ".rs":    {"function_item", "impl_item", "struct_item", "enum_item",
               "trait_item", "mod_item"},
    # Swift
    ".swift": {"function_declaration", "class_declaration", "struct_declaration",
               "enum_declaration", "protocol_declaration", "extension_declaration"},
    # HTML — chunk by top-level elements (head, body, main sections)
    ".html":  {"element", "script_element", "style_element"},
    # CSS
    ".css":   {"rule_set", "media_statement", "keyframes_statement",
               "supports_statement", "at_rule"},
    # JSON — chunk by top-level keys
    ".json":  {"pair"},
    # XML — chunk by direct children of root
    ".xml":   {"element"},
    # SQL — chunk by statements
    ".sql":   {"statement", "create_statement", "select_statement",
               "insert_statement", "update_statement", "delete_statement",
               "alter_statement", "drop_statement", "create_function_statement",
               "create_procedure_statement"},
    # Shell
    ".sh":    {"function_definition", "compound_statement"},
    ".bash":  {"function_definition", "compound_statement"},
}

# Name extraction: for each split type, which child field/type holds the name
_NAME_FIELD_MAP: Dict[str, str] = {
    # Generic fields tried in order
    "name":       "identifier",  # fallback
}

_MAX_GENERIC_CHUNK_LINES = 150  # split oversized nodes further


def parse_generic(content: str, file_path: str, ext: str, parser) -> List[ASTChunk]:
    """Generic AST-based chunking for languages other than C#/Java.

    Splits the file at top-level declaration boundaries (function, class, etc.)
    using language-specific node types. Falls back to whole-file chunk if the
    grammar doesn't produce useful top-level nodes.
    """
    try:
        tree = parser.parse(bytes(content, 'utf-8'))
    except Exception as e:
        logger.warning(f"Generic parse failed for {file_path}: {e}")
        return _whole_file_chunk(content, file_path, ext)

    lines = content.splitlines()
    split_types = _GENERIC_SPLIT_TYPES.get(ext, set())
    chunks: List[ASTChunk] = []

    def _extract_name(node) -> str:
        """Try common child fields to get a declaration name."""
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier",
                              "type_identifier", "field_identifier"):
                try:
                    return child.text.decode("utf-8") if isinstance(child.text, bytes) else child.text
                except Exception:
                    pass
        # Use line number as fallback name
        return f"line_{node.start_point[0] + 1}"

    def _node_to_chunk(node, parent_name: Optional[str] = None) -> Optional[ASTChunk]:
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        try:
            chunk_content = "\n".join(lines[start_line:end_line + 1])
        except Exception:
            return None
        if not chunk_content.strip():
            return None
        name = _extract_name(node)
        return ASTChunk(
            name=name,
            node_type=node.type,
            content=chunk_content,
            start_line=start_line + 1,
            end_line=end_line + 1,
            parent_class=parent_name,
            namespace=None,
        )

    root = tree.root_node

    # Walk direct children of root looking for split_types
    found_any = False
    for child in root.children:
        if child.type in split_types:
            chunk = _node_to_chunk(child)
            if chunk:
                chunks.append(chunk)
                found_any = True
            # Recurse one level for nested declarations (e.g. methods inside classes)
            for grandchild in child.children:
                if grandchild.type in split_types:
                    inner = _node_to_chunk(grandchild, parent_name=chunk.name if chunk else None)
                    if inner:
                        chunks.append(inner)
                        found_any = True
        elif not split_types:
            # No split types defined — just grab each direct child as a chunk
            chunk = _node_to_chunk(child)
            if chunk:
                chunks.append(chunk)
                found_any = True

    if not found_any:
        # Grammar parsed OK but no matching split nodes → whole-file chunk
        return _whole_file_chunk(content, file_path, ext)

    return chunks


def _whole_file_chunk(content: str, file_path: str, ext: str) -> List[ASTChunk]:
    """Fallback: treat the entire file as a single chunk."""
    lines = content.splitlines()
    if not content.strip():
        return []
    return [ASTChunk(
        name=os.path.basename(file_path),
        node_type="file",
        content=content,
        start_line=1,
        end_line=len(lines),
        parent_class=None,
        namespace=None,
    )]


# ── Core parsing logic ────────────────────────────────────────────

def _parse_file(
    parser,
    content: str,
    file_path: str,
    member_types: set,
    type_decl_types: set,
) -> List[ASTChunk]:
    """Generic AST file parser for C#/Java."""
    try:
        tree = parser.parse(bytes(content, 'utf-8'))
    except Exception as e:
        logger.warning("tree-sitter parse failed for %s: %s", file_path, e)
        return []

    root = tree.root_node
    lines = content.split('\n')

    # Detect namespace (C# specific, Java uses package)
    namespace = _find_namespace(root) or _find_package(root)

    # Collect using/import preamble
    preamble = _extract_preamble(root, lines)

    chunks: List[ASTChunk] = []

    # Walk top-level nodes
    _walk_declarations(
        root, lines, file_path, namespace, preamble,
        member_types, type_decl_types, chunks,
    )

    # Build metadata headers for all chunks
    for chunk in chunks:
        chunk.metadata_header = _build_header(file_path, chunk)

    return chunks


def _walk_declarations(
    node, lines, file_path, namespace, preamble,
    member_types, type_decl_types, out_chunks,
    parent_class=None,
):
    """Recursively walk AST nodes and extract chunks."""
    for child in node.children:
        # Namespace / package: recurse into body
        if child.type in ('namespace_declaration', 'file_scoped_namespace_declaration'):
            ns = _get_qualified_name(child) or namespace
            body = _find_child_by_type(child, 'declaration_list')
            if body:
                _walk_declarations(
                    body, lines, file_path, ns, preamble,
                    member_types, type_decl_types, out_chunks,
                )
            continue

        # Type declaration (class, interface, enum, struct)
        if child.type in type_decl_types:
            class_name = _get_identifier(child)
            members = _extract_members(
                child, lines, class_name, namespace,
                member_types, type_decl_types, file_path, preamble,
            )
            if members:
                out_chunks.extend(members)
            else:
                # Small class / enum with no splittable members: keep whole
                text = _node_text(child, lines)
                out_chunks.append(ASTChunk(
                    content=text,
                    node_type=child.type.replace('_declaration', ''),
                    name=class_name or 'unknown',
                    parent_class=parent_class,
                    namespace=namespace,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                ))


def _extract_members(
    class_node, lines, class_name, namespace,
    member_types, type_decl_types, file_path, preamble,
) -> List[ASTChunk]:
    """Extract individual members from a type declaration."""
    body = _find_child_by_type(class_node, 'declaration_list')
    if body is None:
        body = _find_child_by_type(class_node, 'class_body')
    if body is None:
        body = _find_child_by_type(class_node, 'interface_body')
    if body is None:
        body = _find_child_by_type(class_node, 'enum_body')
    if body is None:
        return []

    members: List[ASTChunk] = []
    nested_types: List[ASTChunk] = []

    for child in body.children:
        if child.type in member_types:
            name = _get_identifier(child)
            text = _node_text(child, lines)
            members.append(ASTChunk(
                content=text,
                node_type=child.type.replace('_declaration', ''),
                name=name or 'unknown',
                parent_class=class_name,
                namespace=namespace,
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
            ))
        elif child.type in type_decl_types:
            # Nested type: recurse
            nested_name = _get_identifier(child)
            nested = _extract_members(
                child, lines, nested_name, namespace,
                member_types, type_decl_types, file_path, preamble,
            )
            if nested:
                nested_types.extend(nested)
            else:
                text = _node_text(child, lines)
                nested_types.append(ASTChunk(
                    content=text,
                    node_type=child.type.replace('_declaration', ''),
                    name=nested_name or 'unknown',
                    parent_class=class_name,
                    namespace=namespace,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                ))

    all_chunks = members + nested_types

    # If only 1 member (or none), don't split — let caller handle as whole class
    if len(all_chunks) <= 1:
        return []

    return all_chunks


# ── Metadata header builder ───────────────────────────────────────

def _build_header(file_path: str, chunk: ASTChunk) -> str:
    """Build a metadata header string to prepend to chunk content."""
    parts = [f"// File: {file_path}"]
    if chunk.namespace:
        parts.append(f"// Namespace: {chunk.namespace}")
    if chunk.parent_class:
        parts.append(f"// Class: {chunk.parent_class}")
    parts.append(f"// {chunk.node_type.title()}: {chunk.name}")
    parts.append(f"// Lines: {chunk.start_line}-{chunk.end_line}")
    return '\n'.join(parts)


# ── Tree-sitter helpers ───────────────────────────────────────────

def _node_text(node, lines: list) -> str:
    """Extract source text for a node."""
    start_row = node.start_point[0]
    end_row = node.end_point[0]
    if start_row == end_row:
        return lines[start_row][node.start_point[1]:node.end_point[1]]
    result = []
    for i in range(start_row, min(end_row + 1, len(lines))):
        result.append(lines[i])
    return '\n'.join(result)


def _get_identifier(node) -> Optional[str]:
    """Get the identifier (name) of a declaration node.

    Uses tree-sitter field 'name' first (reliable for method/class/etc.),
    falls back to first direct 'identifier' child.
    """
    # Prefer field-based access (handles return-type ambiguity)
    name_node = node.child_by_field_name('name')
    if name_node is not None:
        text = name_node.text
        return text.decode('utf-8') if isinstance(text, bytes) else text

    # Fallback: first direct identifier child
    for child in node.children:
        if child.type == 'identifier':
            text = child.text
            return text.decode('utf-8') if isinstance(text, bytes) else text
    return None


def _get_qualified_name(node) -> Optional[str]:
    """Get qualified_name or identifier from a namespace/package node."""
    for child in node.children:
        if child.type in ('qualified_name', 'identifier', 'name', 'scoped_identifier'):
            return child.text.decode('utf-8') if isinstance(child.text, bytes) else child.text
    return None


def _find_namespace(root) -> Optional[str]:
    """Find C# namespace declaration."""
    for child in root.children:
        if child.type in ('namespace_declaration', 'file_scoped_namespace_declaration'):
            return _get_qualified_name(child)
    return None


def _find_package(root) -> Optional[str]:
    """Find Java package declaration."""
    for child in root.children:
        if child.type == 'package_declaration':
            return _get_qualified_name(child)
    return None


def _find_child_by_type(node, type_name: str):
    """Find first direct child of given type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _extract_preamble(root, lines: list) -> str:
    """Extract using/import statements from file top."""
    preamble_end = 0
    for child in root.children:
        if child.type in (
            'using_directive', 'extern_alias_directive',
            'global_attribute_list', 'attribute_list',
            'import_declaration', 'package_declaration',
        ):
            preamble_end = max(preamble_end, child.end_point[0] + 1)
        elif child.type in (
            'namespace_declaration', 'file_scoped_namespace_declaration',
            'class_declaration', 'interface_declaration',
        ):
            break
    if preamble_end > 0:
        return '\n'.join(lines[:preamble_end])
    return ""


# ── FQN builder ──────────────────────────────────────────────────

def _build_fqn(namespace: Optional[str], parent_class: Optional[str], name: str) -> str:
    """Build a fully qualified name: namespace.class.member"""
    parts = []
    if namespace:
        parts.append(namespace)
    if parent_class:
        parts.append(parent_class)
    parts.append(name)
    return ".".join(parts)


# ── Enhanced relationship extraction ─────────────────────────────

def _extract_using_relationships(root, file_path: str) -> List[Dict[str, Any]]:
    """Extract using/import directives as 'imports' relationships."""
    rels = []
    for child in root.children:
        if child.type == 'using_directive':
            # C#: using Namespace.Sub;
            name_node = _find_child_by_type(child, 'qualified_name') or _find_child_by_type(child, 'identifier')
            if name_node:
                text = name_node.text
                ns = text.decode('utf-8') if isinstance(text, bytes) else text
                rels.append({
                    "type": "imports",
                    "source": file_path,
                    "target": ns,
                })
        elif child.type == 'import_declaration':
            # Java: import com.example.Foo;
            name_node = _find_child_by_type(child, 'scoped_identifier')
            if name_node:
                text = name_node.text
                ns = text.decode('utf-8') if isinstance(text, bytes) else text
                rels.append({
                    "type": "imports",
                    "source": file_path,
                    "target": ns,
                })
    return rels


def _extract_inheritance_relationships(
    root, namespace: Optional[str], type_decl_types: set, ext: str
) -> List[Dict[str, Any]]:
    """Extract inheritance and implementation relationships from type declarations."""
    rels = []
    _walk_for_inheritance(root, namespace, type_decl_types, ext, rels)
    return rels


def _walk_for_inheritance(node, namespace, type_decl_types, ext, rels, parent_class=None):
    """Recursively walk AST to find base types."""
    for child in node.children:
        if child.type in ('namespace_declaration', 'file_scoped_namespace_declaration'):
            ns = _get_qualified_name(child) or namespace
            body = _find_child_by_type(child, 'declaration_list')
            if body:
                _walk_for_inheritance(body, ns, type_decl_types, ext, rels)
            continue

        if child.type in type_decl_types:
            class_name = _get_identifier(child)
            if not class_name:
                continue

            class_fqn = _build_fqn(namespace, parent_class, class_name)

            # Extract base types
            base_types = _extract_base_types(child, ext)
            for base_info in base_types:
                rels.append({
                    "type": base_info["rel_type"],  # "inherits" or "implements"
                    "source": class_fqn,
                    "target": base_info["name"],  # May be unqualified
                })

            # Recurse into nested types
            body = (
                _find_child_by_type(child, 'declaration_list')
                or _find_child_by_type(child, 'class_body')
                or _find_child_by_type(child, 'interface_body')
            )
            if body:
                _walk_for_inheritance(body, namespace, type_decl_types, ext, rels, class_name)


def _extract_base_types(class_node, ext: str) -> List[Dict[str, str]]:
    """Parse base_list (C#) or superclass/super_interfaces (Java).

    Returns list of {"name": "BaseName", "rel_type": "inherits"|"implements"}.
    """
    results = []

    if ext == '.cs':
        # C#: base_list contains base types separated by commas
        base_list = _find_child_by_type(class_node, 'base_list')
        if base_list:
            is_interface = class_node.type == 'interface_declaration'
            for child in base_list.children:
                if child.type in ('identifier', 'qualified_name', 'generic_name',
                                  'simple_base_type', 'type_constraint'):
                    text = child.text
                    name = text.decode('utf-8') if isinstance(text, bytes) else text
                    # Clean generic parameters: IFoo<T> -> IFoo
                    if '<' in name:
                        name = name[:name.index('<')]
                    # Heuristic: if starts with 'I' and is PascalCase, it's likely an interface
                    if is_interface or (len(name) > 1 and name[0] == 'I' and name[1].isupper()):
                        results.append({"name": name, "rel_type": "implements"})
                    else:
                        results.append({"name": name, "rel_type": "inherits"})

    elif ext == '.java':
        # Java: superclass and super_interfaces are separate fields
        superclass = class_node.child_by_field_name('superclass')
        if superclass:
            text = superclass.text
            name = text.decode('utf-8') if isinstance(text, bytes) else text
            # Remove "extends " prefix if present
            name = name.replace('extends ', '').strip()
            if '<' in name:
                name = name[:name.index('<')]
            results.append({"name": name, "rel_type": "inherits"})

        interfaces = class_node.child_by_field_name('interfaces')
        if interfaces:
            # super_interfaces node contains type_list
            for child in interfaces.children:
                if child.type in ('type_identifier', 'scoped_type_identifier', 'generic_type'):
                    text = child.text
                    name = text.decode('utf-8') if isinstance(text, bytes) else text
                    if '<' in name:
                        name = name[:name.index('<')]
                    results.append({"name": name, "rel_type": "implements"})

    return results


def _extract_type_references(
    root, namespace: Optional[str], type_decl_types: set, member_types: set, ext: str
) -> List[Dict[str, Any]]:
    """Extract type references from property/field declarations.

    Identifies types used in properties, fields, method params and return types.
    """
    rels = []
    _walk_for_type_refs(root, namespace, type_decl_types, member_types, ext, rels)
    return rels


def _walk_for_type_refs(node, namespace, type_decl_types, member_types, ext, rels, parent_class=None):
    """Recursively walk to find type references in members."""
    for child in node.children:
        if child.type in ('namespace_declaration', 'file_scoped_namespace_declaration'):
            ns = _get_qualified_name(child) or namespace
            body = _find_child_by_type(child, 'declaration_list')
            if body:
                _walk_for_type_refs(body, ns, type_decl_types, member_types, ext, rels)
            continue

        if child.type in type_decl_types:
            class_name = _get_identifier(child)
            if not class_name:
                continue
            body = (
                _find_child_by_type(child, 'declaration_list')
                or _find_child_by_type(child, 'class_body')
                or _find_child_by_type(child, 'interface_body')
            )
            if body:
                _walk_for_type_refs(body, namespace, type_decl_types, member_types, ext, rels, class_name)
            continue

        # Property declarations (C#)
        if child.type == 'property_declaration' and parent_class:
            type_node = child.child_by_field_name('type')
            if type_node:
                type_name = _extract_type_name(type_node)
                if type_name and not _is_primitive_type(type_name):
                    source_fqn = _build_fqn(namespace, parent_class, _get_identifier(child) or "")
                    rels.append({
                        "type": "uses_type",
                        "source": source_fqn,
                        "target": type_name,
                    })

        # Field declarations (Java)
        elif child.type == 'field_declaration' and parent_class:
            type_node = child.child_by_field_name('type')
            if type_node:
                type_name = _extract_type_name(type_node)
                if type_name and not _is_primitive_type(type_name):
                    # Field may have multiple declarators
                    for decl in child.children:
                        if decl.type == 'variable_declarator':
                            name = _get_identifier(decl)
                            if name:
                                source_fqn = _build_fqn(namespace, parent_class, name)
                                rels.append({
                                    "type": "uses_type",
                                    "source": source_fqn,
                                    "target": type_name,
                                })


def _extract_type_name(type_node) -> Optional[str]:
    """Extract a clean type name from a type AST node."""
    if type_node is None:
        return None
    text = type_node.text
    name = text.decode('utf-8') if isinstance(text, bytes) else text
    # Strip nullable suffix
    name = name.rstrip('?')
    # Strip generic params for the base type name
    if '<' in name:
        name = name[:name.index('<')]
    # Strip array brackets
    name = name.rstrip('[]')
    return name.strip() if name.strip() else None


# Primitive / built-in types to ignore for type reference relationships
_PRIMITIVE_TYPES = {
    # C#
    'int', 'long', 'short', 'byte', 'float', 'double', 'decimal',
    'bool', 'char', 'string', 'object', 'void', 'var', 'dynamic',
    'Int32', 'Int64', 'Int16', 'Byte', 'Single', 'Double', 'Decimal',
    'Boolean', 'Char', 'String', 'Object', 'Void',
    # Java
    'int', 'long', 'short', 'byte', 'float', 'double', 'boolean', 'char',
    'Integer', 'Long', 'Short', 'Byte', 'Float', 'Double', 'Boolean', 'Character',
    # Common generic containers (too noisy)
    'List', 'Dictionary', 'IEnumerable', 'IList', 'ICollection',
    'Task', 'Action', 'Func', 'Nullable',
    'Map', 'Set', 'Collection', 'Optional', 'CompletableFuture',
}


def _is_primitive_type(name: str) -> bool:
    """Check if a type name is a primitive/built-in type."""
    return name in _PRIMITIVE_TYPES
