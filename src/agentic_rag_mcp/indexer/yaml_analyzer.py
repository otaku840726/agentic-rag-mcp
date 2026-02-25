"""
YAML Analyzer — structure-aware chunker for .yaml/.yml files.

Dual-mode detection:
  - Type A (Pure YAML): Split by top-level keys, sub-split by second-level keys
  - Type B (Markdown-in-YAML): Delegate to MarkdownAnalyzer

Many knowledge-base files (e.g. entities.yaml, flows.yaml) use Markdown syntax
with YAML code blocks inside. This analyzer detects and routes accordingly.
"""

import logging
import os
import re
from typing import List, Optional, Dict, Any

import tiktoken

from .analyzer import BaseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)

# Top-level key pattern: word chars at column 0 followed by colon (with or without inline value)
_TOP_KEY_RE = re.compile(r'^(\w[\w_.-]*):\s*(.*)')

# Second-level key pattern: 2-space indent (best-effort heuristic for sub-splitting)
_SUB_KEY_RE = re.compile(r'^  (\w[\w_-]*):\s*')

# Markdown heading patterns (used for markdown-in-yaml detection)
_MD_SUBHEADING_RE = re.compile(r'^#{2,6}\s+\S')
_MD_FENCED_RE = re.compile(r'^```')


class YAMLAnalyzer(BaseAnalyzer):
    """Structure-aware YAML chunker with Markdown fallback."""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _token_len(self, text: str) -> int:
        return len(self._enc.encode(text))

    def analyze(self, file_path: str, content: Optional[str] = None) -> AnalysisResult:
        if content is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

        if self._is_markdown_in_yaml(content):
            return self._analyze_as_markdown(file_path, content)
        return self._analyze_as_yaml(file_path, content)

    def _is_markdown_in_yaml(self, content: str) -> bool:
        """Detect whether this file uses Markdown syntax rather than pure YAML.

        Heuristic: if early non-blank, non-YAML-comment lines start with
        Markdown headings (# Title), or the file contains ## or ```yaml,
        treat it as Markdown-in-YAML.
        """
        lines = content.split('\n')
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # YAML comment that looks like a Markdown heading: "# Some Title"
            # Pure YAML comments are typically lowercase descriptions,
            # while Markdown headings are title-cased or mixed
            if stripped.startswith('# ') and not stripped.startswith('# -'):
                rest = stripped[2:].strip()
                # If it looks like a heading (no colon at end, not all lowercase)
                if rest and not rest.endswith(':'):
                    # Check for subheadings or fenced blocks anywhere
                    if any(_MD_SUBHEADING_RE.match(l.strip()) for l in lines[:50]):
                        return True
                    if any(_MD_FENCED_RE.match(l.strip()) for l in lines[:50]):
                        return True
            # If first meaningful line is a YAML key, it's pure YAML
            if _TOP_KEY_RE.match(stripped):
                return False
            break

        # Additional check: scan for ## headings or ```yaml at column 0
        # (only non-indented lines — indented content could be YAML multiline strings)
        for line in lines:
            if line != line.lstrip():
                continue  # skip indented lines
            stripped = line.strip()
            if _MD_SUBHEADING_RE.match(stripped):
                return True
            if stripped.startswith('```yaml') or stripped.startswith('```yml'):
                return True

        return False

    def _analyze_as_markdown(self, file_path: str, content: str) -> AnalysisResult:
        """Delegate to MarkdownAnalyzer for Markdown-in-YAML files."""
        from .markdown_analyzer import MarkdownAnalyzer
        result = MarkdownAnalyzer(max_tokens=self.max_tokens).analyze(file_path, content)
        # Override language to reflect the file type
        return AnalysisResult(
            file_path=result.file_path,
            language="yaml",
            symbols=result.symbols,
            relationships=result.relationships,
        )

    def _analyze_as_yaml(self, file_path: str, content: str) -> AnalysisResult:
        """Analyze pure YAML by splitting on top-level keys."""
        lines = content.split('\n')
        sections = self._split_by_top_keys(lines)
        symbols = []

        for section in sections:
            body = section["body"]
            key_path = section["key_path"]

            if self._token_len(body) <= self.max_tokens:
                symbols.append({
                    "content": body,
                    "node_type": "yaml_section",
                    "name": section["key"],
                    "key_path": key_path,
                    "start_line": section["start_line"],
                    "end_line": section["end_line"],
                })
            else:
                # Sub-split by second-level keys
                subsections = self._split_by_sub_keys(
                    lines[section["start_line"] - 1 : section["end_line"]],
                    section["key"],
                    section["start_line"],
                )
                if len(subsections) > 1:
                    for sub in subsections:
                        symbols.append({
                            "content": sub["body"],
                            "node_type": "yaml_subsection",
                            "name": sub["key"],
                            "key_path": f"{section['key']} > {sub['key']}",
                            "start_line": sub["start_line"],
                            "end_line": sub["end_line"],
                        })
                else:
                    # Can't sub-split further; emit as-is
                    symbols.append({
                        "content": body,
                        "node_type": "yaml_section",
                        "name": section["key"],
                        "key_path": key_path,
                        "start_line": section["start_line"],
                        "end_line": section["end_line"],
                    })

        # Fallback: whole file as one chunk
        if not symbols:
            symbols.append({
                "content": content,
                "node_type": "document",
                "name": os.path.basename(file_path),
                "key_path": "",
                "start_line": 1,
                "end_line": len(lines),
            })

        return AnalysisResult(
            file_path=file_path,
            language="yaml",
            symbols=symbols,
            relationships=[],
        )

    def _split_by_top_keys(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Split YAML lines into sections by top-level keys."""
        sections = []
        current_key = None
        start_idx = 0
        preamble_lines = []

        for i, line in enumerate(lines):
            m = _TOP_KEY_RE.match(line)
            if m:
                # Flush previous section
                if current_key is not None:
                    body = '\n'.join(lines[start_idx:i]).strip()
                    if body:
                        sections.append({
                            "key": current_key,
                            "key_path": current_key,
                            "body": body,
                            "start_line": start_idx + 1,
                            "end_line": i,
                        })
                elif preamble_lines:
                    # Content before first key (comments/headers)
                    preamble = '\n'.join(preamble_lines).strip()
                    if preamble:
                        sections.append({
                            "key": "_preamble",
                            "key_path": "_preamble",
                            "body": preamble,
                            "start_line": 1,
                            "end_line": i,
                        })

                current_key = m.group(1)
                start_idx = i
            elif current_key is None:
                preamble_lines.append(line)

        # Flush last section
        if current_key is not None:
            body = '\n'.join(lines[start_idx:]).strip()
            if body:
                sections.append({
                    "key": current_key,
                    "key_path": current_key,
                    "body": body,
                    "start_line": start_idx + 1,
                    "end_line": len(lines),
                })
        elif not sections:
            # No top-level keys found at all
            body = '\n'.join(lines).strip()
            if body:
                sections.append({
                    "key": "_document",
                    "key_path": "_document",
                    "body": body,
                    "start_line": 1,
                    "end_line": len(lines),
                })

        return sections

    def _split_by_sub_keys(
        self, lines: List[str], parent_key: str, base_line: int
    ) -> List[Dict[str, Any]]:
        """Split a section's lines by second-level (2-space indent) keys."""
        subsections = []
        current_sub = None
        start_idx = 0

        for i, line in enumerate(lines):
            m = _SUB_KEY_RE.match(line)
            if m:
                if current_sub is not None:
                    body = '\n'.join(lines[start_idx:i]).strip()
                    if body:
                        subsections.append({
                            "key": current_sub,
                            "body": body,
                            "start_line": base_line + start_idx,
                            "end_line": base_line + i - 1,
                        })
                current_sub = m.group(1)
                start_idx = i

        # Flush last
        if current_sub is not None:
            body = '\n'.join(lines[start_idx:]).strip()
            if body:
                subsections.append({
                    "key": current_sub,
                    "body": body,
                    "start_line": base_line + start_idx,
                    "end_line": base_line + len(lines) - 1,
                })

        return subsections
