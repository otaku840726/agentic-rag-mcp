"""
Markdown Analyzer — heading-based chunker for .md files.

Splits Markdown documents by heading boundaries, preserving:
- Heading hierarchy (# to ######)
- Fenced code blocks (never splits inside)
- Tables (kept together with parent heading)
- Parent heading path as context prefix for embedding quality
"""

import logging
import re
import os
from typing import List, Optional, Dict, Any, Tuple

import tiktoken

from .analyzer import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

# Regex for ATX headings (# to ######)
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')

# Regex for fenced code block delimiters (backticks or tildes, 3+)
_FENCE_RE = re.compile(r'^(`{3,}|~{3,})')


def _iter_lines_outside_fences(
    lines: List[str],
) -> List[Tuple[int, str, bool]]:
    """Classify each line as inside or outside a fenced code block.

    Tracks fence character to avoid mismatched close (e.g. ``` opened, ~~~ ignored).
    Returns list of (index, line, is_outside_fence) tuples.
    """
    result = []
    in_fence = False
    fence_char: Optional[str] = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        fence_match = _FENCE_RE.match(stripped)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]  # '`' or '~'
                result.append((i, line, False))
                continue
            elif marker[0] == fence_char:
                in_fence = False
                fence_char = None
                result.append((i, line, False))
                continue
        result.append((i, line, not in_fence))

    return result


class MarkdownAnalyzer(BaseAnalyzer):
    """Heading-based Markdown chunker that produces AnalysisResult."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def analyze(self, file_path: str, content: Optional[str] = None) -> AnalysisResult:
        if content is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        lines = content.split('\n')
        sections = self._split_by_headings(lines)
        symbols = []
        relationships = []

        for section in sections:
            heading_path = section["heading_path"]
            prefix = f"[{heading_path}]\n\n" if heading_path else ""
            body = section["body"]

            if self._token_len(prefix + body) <= self.max_tokens:
                symbols.append({
                    "content": prefix + body,
                    "node_type": "section" if section["heading"] else "document",
                    "name": section["heading"] or os.path.basename(file_path),
                    "heading_path": heading_path,
                    "heading_level": section["level"],
                    "start_line": section["start_line"],
                    "end_line": section["end_line"],
                    "parent_section": section["parent"],
                })
            else:
                # Sub-split large sections by paragraphs, respecting code blocks
                parts = self._subsplit(body, prefix)
                for i, part_text in enumerate(parts):
                    symbols.append({
                        "content": part_text,
                        "node_type": "section_part",
                        "name": section["heading"] or os.path.basename(file_path),
                        "heading_path": heading_path,
                        "heading_level": section["level"],
                        "start_line": section["start_line"],
                        "end_line": section["end_line"],
                        "parent_section": section["parent"],
                        "part_index": i,
                        "total_parts": len(parts),
                    })

            # Build parent-child relationships
            if section["parent"]:
                relationships.append({
                    "type": "subsection_of",
                    "source": section["heading"],
                    "target": section["parent"],
                })

        # Fallback: if no symbols produced, treat whole file as one chunk
        if not symbols:
            symbols.append({
                "content": content,
                "node_type": "document",
                "name": os.path.basename(file_path),
                "heading_path": "",
                "heading_level": 0,
                "start_line": 1,
                "end_line": len(lines),
                "parent_section": None,
            })

        return AnalysisResult(
            file_path=file_path,
            language="markdown",
            symbols=symbols,
            relationships=relationships,
        )

    def _split_by_headings(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Split lines into sections by heading boundaries.

        Returns a list of section dicts with heading info and body text.
        Line numbers are 1-based inclusive (start_line and end_line both 1-based).
        """
        # Find all heading positions (skipping those inside fenced code)
        classified = _iter_lines_outside_fences(lines)
        headings = []  # (line_index, level, text)

        for i, line, outside in classified:
            if not outside:
                continue
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                text = m.group(2).strip()
                headings.append((i, level, text))

        if not headings:
            # No headings: entire file is one section
            body = '\n'.join(lines).strip()
            if body:
                return [{
                    "heading": None,
                    "level": 0,
                    "heading_path": "",
                    "parent": None,
                    "body": body,
                    "start_line": 1,
                    "end_line": len(lines),
                }]
            return []

        # Build sections from heading boundaries
        sections = []
        stack: List[tuple] = []  # Heading hierarchy: [(level, text)]

        # Content before first heading (preamble)
        if headings[0][0] > 0:
            preamble = '\n'.join(lines[:headings[0][0]]).strip()
            if preamble:
                sections.append({
                    "heading": None,
                    "level": 0,
                    "heading_path": "",
                    "parent": None,
                    "body": preamble,
                    "start_line": 1,
                    "end_line": headings[0][0],  # 0-based exclusive == 1-based inclusive
                })

        for idx, (line_idx, level, text) in enumerate(headings):
            # End boundary: next heading start or EOF
            if idx + 1 < len(headings):
                end_idx = headings[idx + 1][0]
            else:
                end_idx = len(lines)

            # Update heading stack: pop all >= current level
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent = stack[-1][1] if stack else None
            stack.append((level, text))

            heading_path = " > ".join(t for _, t in stack)
            body = '\n'.join(lines[line_idx:end_idx]).strip()

            if body:
                sections.append({
                    "heading": text,
                    "level": level,
                    "heading_path": heading_path,
                    "parent": parent,
                    "body": body,
                    "start_line": line_idx + 1,
                    "end_line": end_idx,
                })

        return sections

    def _subsplit(self, body: str, prefix: str) -> List[str]:
        """Sub-split a large section by paragraphs, respecting code block boundaries.

        Returns list of text chunks, each prefixed with the heading path context.
        """
        blocks = self._split_respecting_fences(body)

        parts = []
        current = prefix

        for block in blocks:
            candidate = current + block + "\n\n"
            if self._token_len(candidate) > self.max_tokens and current != prefix:
                # Flush current
                parts.append(current.rstrip())
                current = prefix + block + "\n\n"
            else:
                current = candidate

        if current.strip() and current.strip() != prefix.strip():
            parts.append(current.rstrip())

        # If nothing was produced (single huge block), return as-is
        if not parts:
            fallback = (prefix + body).rstrip()
            if fallback:
                parts.append(fallback)

        return parts

    def _split_respecting_fences(self, text: str) -> List[str]:
        """Split text by double-newlines but keep fenced code blocks as single units."""
        blocks = []
        current_block: List[str] = []
        in_fence = False
        fence_char: Optional[str] = None

        for line in text.split('\n'):
            stripped = line.strip()

            fence_match = _FENCE_RE.match(stripped)
            if fence_match:
                marker = fence_match.group(1)
                if not in_fence:
                    in_fence = True
                    fence_char = marker[0]
                    current_block.append(line)
                    continue
                elif marker[0] == fence_char:
                    in_fence = False
                    fence_char = None
                    current_block.append(line)
                    # End of fenced block — flush as one unit
                    blocks.append('\n'.join(current_block))
                    current_block = []
                    continue

            if in_fence:
                current_block.append(line)
                continue

            # Outside fence: split on blank lines
            if stripped == '' and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
            else:
                current_block.append(line)

        if current_block:
            blocks.append('\n'.join(current_block))

        return blocks
