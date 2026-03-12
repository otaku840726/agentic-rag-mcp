"""
SCIP (Sourcegraph Code Intelligence Protocol) output parser.

Converts SCIP JSON (from `scip print --json`) to our internal
Symbol/Relationship format, compatible with _ingest_analysis_to_graph.

SCIP reference: https://github.com/sourcegraph/scip/blob/main/scip.proto

Key observations from real scip-typescript output:
- JSON uses snake_case field names (relative_path, symbol_roles, etc.)
- scip-typescript does NOT emit `kind` or `display_name` in the symbols array
- `enclosing_range` on definition occurrences gives the full symbol extent
- Range format: 3-element = [line, startChar, endChar]; 4-element = [startLine, startChar, endLine, endChar]
- Symbol kind must be inferred from the symbol descriptor string or documentation
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Symbol role flags ────────────────────────────────────────────────────────
_ROLE_DEFINITION = 1   # bit 0

# ── node_type inference from documentation string ────────────────────────────
# Pattern: ```ts\n<kind-keyword> <name>...
_DOC_KIND_MAP = [
    (re.compile(r'^\s*class\s+',        re.M), "class"),
    (re.compile(r'^\s*abstract\s+class\s+', re.M), "class"),
    (re.compile(r'^\s*interface\s+',    re.M), "interface"),
    (re.compile(r'^\s*enum\s+',         re.M), "enum"),
    (re.compile(r'^\s*type\s+\w+\s*=',  re.M), "class"),      # type alias
    (re.compile(r'^\s*struct\s+',       re.M), "class"),       # Go struct
    (re.compile(r'constructor\s*\(',    re.I), "constructor"),
    (re.compile(r'^\s*\(method\)\s+',   re.M), "method"),
    (re.compile(r'^\s*function\s+\w+\s*\(', re.M), "function"),
    (re.compile(r'^\s*(?:async\s+)?function\s*\*?\s*\w+\s*\(', re.M), "function"),
    (re.compile(r'^\s*(?:readonly\s+)?\w+:\s*', re.M), "property"),
    (re.compile(r'^\s*(?:const|let|var)\s+', re.M), "variable"),
]

# node_types to skip (too noisy / not useful in graph)
_SKIP_NODE_TYPES = {"variable", "module", "file", ""}

# ── Symbol descriptor → node_type inference ──────────────────────────────────

def _infer_kind_from_descriptor(descriptor: str) -> str:
    """
    Infer node_type from the SCIP symbol descriptor suffix.

    SCIP descriptor syntax:
      TypeName#           → class/interface (type descriptor)
      method().           → method/function (term descriptor with parens)
      field.              → property/field (term descriptor without parens)
      Namespace/          → namespace (skip)
      `file.ts`/          → module/file (skip)
    """
    # Strip trailing whitespace
    d = descriptor.strip()

    # If ends with `().\` sequence it's a method
    if re.search(r'\(\)\.$', d):
        return "method"

    # If ends with `#` it's a type (class/interface/enum)
    if d.endswith('#'):
        return "class"

    # If ends with `.` (without parens before) it's a property/field
    if d.endswith('.') and not d.endswith(').'):
        return "property"

    # If ends with `/` or backtick pattern → file/module (skip)
    if d.endswith('/') or re.search(r'`[^`]+`/$', d):
        return "file"

    return ""


def _infer_kind_from_doc(doc_str: str) -> str:
    """Infer node_type from a documentation string (markdown from SCIP)."""
    # Strip markdown code fences
    text = re.sub(r'```\w*\n?', '', doc_str).strip()
    for pattern, kind in _DOC_KIND_MAP:
        if pattern.search(text):
            return kind
    return ""


# ── Symbol string parsing ─────────────────────────────────────────────────────

def _parse_scip_symbol(scip_sym: str) -> Tuple[str, str, str]:
    """
    Parse SCIP symbol string into (scheme, package, descriptor).

    Format: "{scheme} {manager} {package-name} {package-version} {descriptor}"
    Example: "scip-typescript npm @myapp/pkg 1.0.0 src/`UserCtrl.ts`/UserCtrl#"
    """
    parts = scip_sym.split(" ", 4)
    scheme     = parts[0] if len(parts) > 0 else ""
    package    = parts[2] if len(parts) > 2 else ""
    descriptor = parts[4] if len(parts) > 4 else scip_sym
    return scheme, package, descriptor


def _descriptor_to_fqn(descriptor: str, file_rel_prefix: str) -> str:
    """
    Convert a SCIP descriptor to a human-readable FQN.

    Examples:
      "src/`UserController.ts`/UserController#"          → "src/UserController.UserController"
      "src/`UserController.ts`/UserController#create()." → "src/UserController.UserController.create"
    """
    # Remove backticks (SCIP quotes file paths)
    d = re.sub(r'`([^`]*)`', r'\1', descriptor)

    # Split on SCIP descriptor separators: #, ., (), /
    tokens = re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*', d)

    if not tokens:
        return descriptor

    # For "src/UserController.ts/UserController/create":
    # We want "src/UserController.UserController.create"
    # Strategy: reconstruct from tokens but replace file extension suffix
    result_parts: List[str] = []
    file_path_done = False
    for tok in tokens:
        if not file_path_done and (tok.endswith('ts') or tok.endswith('js') or
                                    tok.endswith('py') or tok.endswith('go')):
            # This is likely a file path segment — keep without extension
            tok = re.sub(r'\.(ts|tsx|js|jsx|py|go)$', '', tok)
            file_path_done = True
        result_parts.append(tok)

    # Deduplicate consecutive identical tokens (SCIP often repeats the filename)
    deduped: List[str] = []
    for p in result_parts:
        if not deduped or deduped[-1] != p:
            deduped.append(p)

    return ".".join(deduped)


def _extract_display_name(scip_sym: str, doc_str: str) -> str:
    """Extract a short display name from symbol string or documentation."""
    # Try documentation first: "class Foo" / "(method) Foo.bar()" / "interface Foo"
    if doc_str:
        text = re.sub(r'```\w*\n?', '', doc_str).strip()
        # class Foo / interface Foo / enum Foo / struct Foo
        m = re.search(r'(?:class|interface|enum|struct|type)\s+(\w+)', text)
        if m:
            return m.group(1)
        # (method) ClassName.methodName()
        m = re.search(r'\(method\)\s+\w+\.(\w+)', text)
        if m:
            return m.group(1)
        # function foo(
        m = re.search(r'function\s+(\w+)\s*\(', text)
        if m:
            return m.group(1)
        # constructor
        if 'constructor' in text.lower():
            return "__init__"
        # const/let/var name:
        m = re.search(r'(?:const|let|var)\s+(\w+)', text)
        if m:
            return m.group(1)

    # Fall back to last identifier in descriptor
    _, _, descriptor = _parse_scip_symbol(scip_sym)
    d = re.sub(r'`([^`]*)`', r'\1', descriptor)
    tokens = re.findall(r'[A-Za-z_$][A-Za-z0-9_$]*', d)
    return tokens[-1] if tokens else scip_sym


def _is_local_symbol(scip_sym: str) -> bool:
    return scip_sym.startswith("local ")


def _is_external_pkg(scip_sym: str, local_package: str) -> bool:
    """True if the symbol belongs to a different package (external dependency)."""
    _, pkg, _ = _parse_scip_symbol(scip_sym)
    return bool(local_package) and pkg != local_package


# ── Source file content reading ───────────────────────────────────────────────

def _read_lines(file_abs: str, start_line: int, end_line: int) -> str:
    try:
        with open(file_abs, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[start_line: end_line + 1])
    except OSError:
        return ""


# ── File-type classification (TypeScript/Python/Go) ──────────────────────────

_FILE_TYPE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'(Controller|Router|Resource|Handler)$', re.I), "controller"),
    (re.compile(r'(Service|ServiceImpl|Manager|Facade)$', re.I),  "service"),
    (re.compile(r'(Repository|Repo|DAO|Store|Mapper)$', re.I),    "repository"),
    (re.compile(r'(Entity|Model|Schema|Record|DTO|Dto)$', re.I),  "entity"),
    (re.compile(r'(Middleware|Interceptor|Guard|Filter)$', re.I), "middleware"),
    (re.compile(r'(Config|Configuration|Settings|Options)$', re.I), "config"),
]


def _classify_file_type(class_name: str) -> Optional[str]:
    for pattern, ft in _FILE_TYPE_PATTERNS:
        if pattern.search(class_name):
            return ft
    return None


# ── TypeScript decorator detection ───────────────────────────────────────────

_TS_HTTP_DECORATOR = re.compile(
    r'@(Get|Post|Put|Delete|Patch|Head|Options)\s*\(\s*[\'"]?([^\'")]*)[\'"]?\s*\)',
    re.I,
)
_TS_CLASS_ROUTE = re.compile(r'@(Controller|Route)\s*\(\s*[\'"]([^\'"]*)[\'"]', re.I)
_TS_AUTH = re.compile(r'@(UseGuards|Authorize|Auth)\s*\(', re.I)
_HTTP_CALL_PAT = re.compile(
    r'\b(axios|fetch\s*\(|http\.get|http\.post|HttpClient|got\s*\.|superagent)\b'
)


def _extract_ts_http_info(content: str) -> Dict[str, Any]:
    m = _TS_HTTP_DECORATOR.search(content)
    if not m:
        return {}
    verb = m.group(1).upper()
    path_frag = m.group(2) or ""
    op = {"GET": "READ", "HEAD": "READ", "POST": "WRITE",
          "PUT": "WRITE", "PATCH": "WRITE", "DELETE": "DELETE"}.get(verb)
    return {
        "http_method":       verb,
        "http_path_fragment": path_frag,
        "auth_required":     True if _TS_AUTH.search(content) else None,
        "operation_type":    op,
        "entry_point_type":  "api",
    }


# ── Main parser ───────────────────────────────────────────────────────────────

def parse_scip_json(
    scip_data: Dict[str, Any],
    project_abs_dir: str,
    project_rel_prefix: str,
    language: str,
) -> Dict[str, Any]:
    """
    Convert SCIP JSON output to our internal AnalysisResult format.

    Args:
        scip_data:           Parsed JSON from `scip print --json`.
        project_abs_dir:     Absolute path to the project directory on the host.
        project_rel_prefix:  Path prefix from root_for_rel_path to project dir
                             (e.g. "frontend" → file_paths become "frontend/src/Foo.ts").
        language:            "typescript" | "javascript" | "go"

    Returns:
        Dict with: file_path (anchor), language, symbols, relationships
    """
    all_symbols:      List[Dict] = []
    all_relationships: List[Dict] = []

    # Determine local package name to filter out external symbols
    metadata = scip_data.get("metadata", {})
    tool_info = metadata.get("tool_info", {})
    # project_root gives us the root; we can extract package from symbol strings
    local_package: str = ""  # will be inferred from first local occurrence

    documents = scip_data.get("documents", [])

    for doc in documents:
        scip_rel = doc.get("relative_path", "")
        if not scip_rel:
            continue

        file_abs = os.path.join(project_abs_dir, scip_rel)
        file_rel = f"{project_rel_prefix}/{scip_rel}" if project_rel_prefix else scip_rel

        # Build symbol-info map for this document (symbol_string → info dict)
        sym_info_by_scip: Dict[str, Dict] = {}
        for si in doc.get("symbols", []):
            sym_info_by_scip[si["symbol"]] = si

        # Detect local package from first definition (non-local) symbol
        if not local_package:
            for occ in doc.get("occurrences", []):
                if occ.get("symbol_roles", 0) & _ROLE_DEFINITION and \
                        not _is_local_symbol(occ.get("symbol", "")):
                    _, pkg, _ = _parse_scip_symbol(occ["symbol"])
                    if pkg:
                        local_package = pkg
                        break

        # First pass: collect class-level context (route, file_type, auth)
        class_route:         Optional[str]  = None
        class_file_type:     Optional[str]  = None
        class_auth_required: Optional[bool] = None

        for occ in doc.get("occurrences", []):
            if occ.get("symbol_roles", 0) & _ROLE_DEFINITION == 0:
                continue
            sym_str = occ.get("symbol", "")
            if _is_local_symbol(sym_str):
                continue

            _, _, desc = _parse_scip_symbol(sym_str)
            kind = _infer_kind_from_descriptor(desc)
            if kind not in ("class", "interface"):
                continue

            si = sym_info_by_scip.get(sym_str, {})
            doc_str = "\n".join(si.get("documentation", []))
            kind2 = _infer_kind_from_doc(doc_str)
            if kind2 not in ("class", "interface", ""):
                continue

            display = _extract_display_name(sym_str, doc_str)
            class_file_type = _classify_file_type(display)

            if language in ("typescript", "javascript"):
                # Read a small content window to detect class decorators
                rng = occ.get("range", [0])
                start_ln = rng[0]
                snippet = _read_lines(file_abs, max(0, start_ln - 3), start_ln + 5)
                class_route = None
                m = _TS_CLASS_ROUTE.search(snippet)
                if m:
                    class_route = m.group(2)
                if _TS_AUTH.search(snippet):
                    class_auth_required = True

        # Second pass: emit all definition occurrences
        for occ in doc.get("occurrences", []):
            if occ.get("symbol_roles", 0) & _ROLE_DEFINITION == 0:
                continue

            sym_str = occ.get("symbol", "")
            if _is_local_symbol(sym_str):
                continue

            si = sym_info_by_scip.get(sym_str, {})
            doc_str = "\n".join(si.get("documentation", []))

            # Infer node_type from descriptor + documentation
            _, _, desc = _parse_scip_symbol(sym_str)
            node_type = _infer_kind_from_descriptor(desc)
            if not node_type or node_type == "file":
                node_type = _infer_kind_from_doc(doc_str)
            if not node_type or node_type in _SKIP_NODE_TYPES:
                continue

            # Extract range information
            # range: 3-elem [line, startChar, endChar] or 4-elem [startLine, startChar, endLine, endChar]
            rng = occ.get("range", [0, 0, 0])
            if len(rng) == 3:
                start_line, start_char, end_char = rng
                end_line = start_line
            else:
                start_line, start_char, end_line, end_char = rng[0], rng[1], rng[2], rng[3]

            # Use enclosing_range for full symbol extent
            enc = occ.get("enclosing_range", [])
            if enc:
                if len(enc) == 3:
                    enc_end_line = enc[0]
                else:
                    enc_end_line = enc[2]
                content = _read_lines(file_abs, start_line, enc_end_line)
                end_line = enc_end_line
            else:
                content = _read_lines(file_abs, start_line, min(end_line, start_line + 50))

            # Names
            display_name = _extract_display_name(sym_str, doc_str)
            fqn = _descriptor_to_fqn(desc, project_rel_prefix)

            sym: Dict[str, Any] = {
                "name":       fqn,
                "fqn":        fqn,
                "node_type":  node_type,
                "content":    content,
                "start_line": start_line,
                "end_line":   end_line,
                "start_byte": 0,
                "end_byte":   0,
                "language":   language,
                "visibility": "public",   # SCIP doesn't emit visibility; assume public
                "metadata": {
                    "file_path":    file_rel,
                    "display_name": display_name,
                    **({"doc_comment": doc_str} if doc_str else {}),
                },
            }

            # ── Class-level enrichment ─────────────────────────────────────────
            if node_type in ("class", "interface"):
                sym["file_type"]    = _classify_file_type(display_name)
                sym["auth_required"] = class_auth_required

            # ── Method/function-level enrichment ──────────────────────────────
            elif node_type in ("method", "function", "constructor"):
                sym["file_type"] = class_file_type

                if language in ("typescript", "javascript"):
                    http_info = _extract_ts_http_info(content)
                    if http_info:
                        verb = http_info.get("http_method")
                        frag = http_info.get("http_path_fragment", "")
                        sym["http_method"] = verb
                        if class_route and frag:
                            sym["http_path"] = "/" + class_route.strip("/") + "/" + frag.strip("/")
                        elif class_route:
                            sym["http_path"] = "/" + class_route.strip("/")
                        elif frag:
                            sym["http_path"] = "/" + frag.strip("/")
                        sym["entry_point_type"] = "api"
                        sym["auth_required"]    = http_info.get("auth_required") or class_auth_required
                        sym["operation_type"]   = http_info.get("operation_type")

                if _HTTP_CALL_PAT.search(content):
                    sym["makes_http_call"] = True

            all_symbols.append(sym)

            # ── Relationships from SCIP symbol info ────────────────────────────
            for rel in si.get("relationships", []):
                target_scip = rel.get("symbol", "")
                if _is_local_symbol(target_scip):
                    continue

                _, _, target_desc = _parse_scip_symbol(target_scip)
                target_fqn = _descriptor_to_fqn(target_desc, project_rel_prefix)

                rel_type = None
                if rel.get("is_implementation"):
                    # is_implementation: source implements/extends target
                    target_kind = _infer_kind_from_descriptor(target_desc)
                    rel_type = "implements" if target_kind == "interface" else "inherits"
                elif rel.get("is_type_definition") or rel.get("isTypeDefinition"):
                    rel_type = "uses_type"
                elif rel.get("is_reference") or rel.get("isReference"):
                    rel_type = "references"

                if rel_type and target_fqn:
                    all_relationships.append({
                        "source":   fqn,
                        "target":   target_fqn,
                        "type":     rel_type,
                        "metadata": {},
                    })

    logger.info(
        f"SCIP parsed {len(documents)} documents → "
        f"{len(all_symbols)} symbols, {len(all_relationships)} relationships"
    )
    return {
        "file_path":     "",
        "language":      language,
        "symbols":       all_symbols,
        "relationships": all_relationships,
    }
