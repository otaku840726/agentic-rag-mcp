"""
PHP routes.php parser for Laravel projects.

Builds a mapping: {controller_class_short -> {method_name -> {http_method, http_path}}}
so the enricher can populate http_path/http_method for PHP controller symbols.

Handles:
  Route::group(['prefix' => 'api/x', 'middleware' => [...]], function () {
      Route::get('/foo', 'FooController@bar');
      Route::post('/foo', [FooController::class, 'bar']);
  });
"""

import re
import os
from pathlib import Path
from typing import Dict, Optional


# Maps Laravel HTTP method helpers to standard HTTP verbs
_METHOD_MAP = {
    "get": "GET",
    "post": "POST",
    "put": "PUT",
    "patch": "PATCH",
    "delete": "DELETE",
    "any": "ANY",
    "match": "GET",  # Route::match(['get','post'], ...) — simplified
}

_ROUTE_PATTERN = re.compile(
    r"Route\s*::\s*(get|post|put|patch|delete|any|match)\s*\("
    r"\s*['\"]([^'\"]+)['\"]\s*,\s*"
    r"(?:"
    r"['\"]([A-Za-z0-9_\\]+)@([A-Za-z0-9_]+)['\"]"   # 'Controller@method'
    r"|"
    r"\[\s*([A-Za-z0-9_\\]+)::class\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*\]"  # [Controller::class, 'method']
    r")",
    re.IGNORECASE,
)

_GROUP_START = re.compile(
    r"Route\s*::\s*group\s*\(",
    re.IGNORECASE,
)

_PREFIX_PATTERN = re.compile(r"['\"]prefix['\"]\s*=>\s*['\"]([^'\"]+)['\"]")
_MIDDLEWARE_PATTERN = re.compile(
    r"['\"]middleware['\"]\s*=>\s*\[([^\]]*)\]"
)
_SINGLE_MIDDLEWARE_PATTERN = re.compile(
    r"['\"]middleware['\"]\s*=>\s*['\"]([^'\"]+)['\"]"
)


def _find_group_body(content: str, start: int) -> tuple:
    """Find the brace-enclosed body of a Route::group starting at `start`.
    Returns (body_content, end_index) or ("", start) if not found."""
    body_start = content.find("{", start)
    if body_start == -1:
        return "", start
    depth = 0
    for j in range(body_start, len(content)):
        if content[j] == "{":
            depth += 1
        elif content[j] == "}":
            depth -= 1
            if depth == 0:
                return content[body_start + 1:j], j
    return "", start


def _parse_block_recursive(
    content: str,
    inherited_prefix: str,
    inherited_middleware: list,
    result: dict,
):
    """
    Recursively parse Route:: declarations and nested Route::group() blocks,
    passing down accumulated prefix and middleware.
    """
    i = 0
    while i < len(content):
        # Look for the next Route::group OR Route::{method} — whichever comes first
        group_m = _GROUP_START.search(content, i)
        route_start = group_m.start() if group_m else len(content)

        # Parse all direct routes before the next group
        direct_block = content[i:route_start]
        _parse_routes_direct(direct_block, inherited_prefix, inherited_middleware, result)

        if not group_m:
            break

        # Extract group attributes using bracket-aware parser
        # Route::group( starts right after the match; find the outer array [ ... ]
        pos = group_m.end()
        # Skip whitespace then expect '['
        while pos < len(content) and content[pos] in " \t\n\r":
            pos += 1
        attrs_str = ""
        if pos < len(content) and content[pos] == "[":
            depth = 0
            attr_start = pos
            for k in range(pos, len(content)):
                if content[k] == "[":
                    depth += 1
                elif content[k] == "]":
                    depth -= 1
                    if depth == 0:
                        attrs_str = content[attr_start + 1:k]
                        pos = k + 1
                        break

        prefix_m = _PREFIX_PATTERN.search(attrs_str)
        local_prefix = prefix_m.group(1).rstrip("/") if prefix_m else ""

        # Accumulate prefix
        if inherited_prefix and local_prefix:
            full_prefix = inherited_prefix.rstrip("/") + "/" + local_prefix.lstrip("/")
        else:
            full_prefix = inherited_prefix or local_prefix

        # Accumulate middleware
        local_mw: list = []
        mw_m = _MIDDLEWARE_PATTERN.search(attrs_str)
        if mw_m:
            local_mw = [s.strip().strip("'\"") for s in mw_m.group(1).split(",") if s.strip()]
        else:
            mw_single = _SINGLE_MIDDLEWARE_PATTERN.search(attrs_str)
            if mw_single:
                local_mw = [mw_single.group(1)]
        combined_mw = list(set(inherited_middleware + local_mw))

        # Get body: find the function() { ... } block after the attributes
        body, body_end = _find_group_body(content, pos)
        if body_end == group_m.end():
            break  # malformed

        # Recurse into group body
        _parse_block_recursive(body, full_prefix, combined_mw, result)

        i = body_end + 1


def _parse_routes_direct(block: str, prefix: str, middleware: list, result: dict):
    """Extract individual Route:: declarations (non-group) from a block."""
    for m in _ROUTE_PATTERN.finditer(block):
        http_verb = _METHOD_MAP.get(m.group(1).lower(), m.group(1).upper())
        path_part = m.group(2).lstrip("/")
        if prefix:
            full_path = ("/" + prefix.strip("/") + "/" + path_part).rstrip("/")
        else:
            full_path = "/" + path_part.rstrip("/")
        full_path = re.sub(r"/+", "/", full_path) or "/"

        # Controller@method format
        if m.group(3) and m.group(4):
            ctrl = m.group(3).split("\\")[-1]
            method = m.group(4)
        # [Controller::class, 'method'] format
        elif m.group(5) and m.group(6):
            ctrl = m.group(5).split("\\")[-1]
            method = m.group(6)
        else:
            continue

        if ctrl not in result:
            result[ctrl] = {}
        is_auth = any(mw in ("auth", "jwt", "api_auth") for mw in middleware)
        result[ctrl][method] = {
            "http_method": http_verb,
            "http_path": full_path,
            "auth_required": is_auth,
            "middleware": middleware,
        }


def parse_routes_files(project_root: str) -> dict:
    """
    Scan all routes/*.php files under project_root and return
    {ControllerShortName: {methodName: {http_method, http_path, auth_required, middleware}}}
    """
    result: Dict[str, dict] = {}
    project_path = Path(project_root)

    route_files = list(project_path.rglob("routes/*.php"))
    # Laravel nwidart modules: */Routes/*.php
    route_files += [p for p in project_path.rglob("*/Routes/*.php") if p not in route_files]
    # Laravel Module flat pattern: Http/routes.php (e.g. memberservice, paybnbservice)
    route_files += [p for p in project_path.rglob("Http/routes.php") if p not in route_files]
    # Also: Http/Routes/*.php
    route_files += [p for p in project_path.rglob("Http/Routes/*.php") if p not in route_files]

    for route_file in route_files:
        try:
            content = route_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Recursively parse all groups and direct routes
        _parse_block_recursive(content, "", [], result)

    return result


def get_route_info(routes_map: dict, controller_short: str, method_name: str) -> Optional[dict]:
    """Lookup route info for a controller/method pair."""
    if not routes_map:
        return None
    ctrl_map = routes_map.get(controller_short)
    if not ctrl_map:
        return None
    return ctrl_map.get(method_name)
