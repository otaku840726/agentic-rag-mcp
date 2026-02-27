"""
Specialist components for the new Coordinator architecture.

Each specialist has a single cognitive responsibility:
  AnchorDetector    — pure Python, no LLM: detect phase and extract anchors
  SubjectAnalyst    — LLM, once: identify subject and actors from query
  TechStackInferrer — LLM, once: infer tech stack from Phase1 evidence
  CoverageAnalyst   — LLM, per iteration (parallel): what is covered?
  SymmetryChecker   — LLM, per iteration (parallel): what symmetry is missing?
  GapIdentifier     — LLM, per iteration: what evidence is still needed?
  QueryGenerator    — LLM, per iteration: how to search for what is needed?
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional

from .models import EvidenceCard, InvestigationState, MissingEvidence, Phase, QueryIntent
from .provider import create_client_for
from .utils import extract_json_from_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. AnchorDetector — pure Python, no LLM
# ---------------------------------------------------------------------------

class AnchorDetector:
    """Detects phase transition and extracts named anchors from evidence cards.

    Phase logic (dual condition):
      - Any card with score_rerank > HIGH_CONF_THRESHOLD and a symbol → Phase2
      - OR ≥ ANY_SYMBOL_THRESHOLD cards with a symbol → Phase2
    """

    HIGH_CONF_THRESHOLD: float = 0.85
    ANY_SYMBOL_THRESHOLD: int = 3

    def detect(self, state: InvestigationState, cards: List[EvidenceCard]) -> None:
        """Update state.anchors and state.phase in-place."""
        symbol_cards = [c for c in cards if c.symbol]
        high_conf = [c for c in symbol_cards if c.score_rerank > self.HIGH_CONF_THRESHOLD]

        # Collect unique anchor symbols (preserve insertion order)
        seen: set = set()
        anchors: List[str] = []
        for c in symbol_cards:
            if c.symbol and c.symbol not in seen:
                seen.add(c.symbol)
                anchors.append(c.symbol)

        state.anchors = anchors

        # Phase is monotonic: once Phase2, never downgrade
        if len(high_conf) >= 1 or len(symbol_cards) >= self.ANY_SYMBOL_THRESHOLD:
            state.phase = Phase.TWO


# ---------------------------------------------------------------------------
# Shared LLM helper
# ---------------------------------------------------------------------------

def _call_llm(
    client,
    model: str,
    provider: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    component_name: str,
    iteration: int,
    usage_log: Optional[list] = None,
    logger_obj: Any = None,
    search_id: str = "",
) -> str:
    """Shared LLM call wrapper with usage logging."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if provider != "local":
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    response = client.chat.completions.create(**kwargs)
    latency = (time.time() - t0) * 1000

    content = response.choices[0].message.content

    if usage_log is not None and hasattr(response, "usage") and response.usage:
        usage_log.append({
            "component": f"{component_name}_iter_{iteration}",
            "model": model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "latency_ms": round(latency),
        })

    if logger_obj and search_id:
        logger_obj.log_llm_event(
            search_id=search_id,
            step=f"{component_name}_iter_{iteration}",
            model=model,
            messages=kwargs["messages"],
            response=content,
            latency_ms=latency,
        )

    return content


# ---------------------------------------------------------------------------
# 2. SubjectAnalyst — runs once at iteration=1, before first search
# ---------------------------------------------------------------------------

_SUBJECT_ANALYST_SYSTEM = """You are a Subject Analyst.
Your only task: given a code search query, identify the core subject and key actors.

Output JSON only, no explanation.

{
    "subject": "the core thing being acted upon (a concept, not a class name)",
    "actors": [
        {"name": "actor name", "role": "their relationship to the subject"}
    ]
}

Rules:
- subject: strip delivery mechanisms and technical layers. What is the ultimate thing being changed, queried, or displayed?
- actors: who produces, consumes, queries, or displays this subject? Include end users and downstream systems even if not explicitly mentioned.
- If the query is too sparse, define a high-level generalized subject and infer actors from domain context.
- Mark uncertain actors with (inferred) in the role field.
- Do NOT fabricate specific class names or method names.
- Do NOT list what is missing or what needs to be searched.
"""


class SubjectAnalyst:
    """Runs once at iteration=1. Establishes the investigation frame (subject + actors)."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self._client, self._cfg = create_client_for("analyst")
        if provider:
            from .provider import create_client
            self._cfg.provider = provider
            self._client = create_client(provider)
        if model:
            self._cfg.model = model

    def analyze(
        self,
        state: InvestigationState,
        usage_log: Optional[list] = None,
        logger_obj: Any = None,
        search_id: str = "",
    ) -> None:
        """Write state.subject and state.actors in-place. Sets state.subject_analyst_done."""
        user_prompt = f"Query: {state.query}\n\nAnalyze and output JSON."

        content = _call_llm(
            client=self._client,
            model=self._cfg.model,
            provider=self._cfg.provider,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system_prompt=_SUBJECT_ANALYST_SYSTEM,
            user_prompt=user_prompt,
            component_name="subject_analyst",
            iteration=state.iteration,
            usage_log=usage_log,
            logger_obj=logger_obj,
            search_id=search_id,
        )

        try:
            data = extract_json_from_response(content)
            state.subject = data.get("subject", "")
            raw_actors = data.get("actors", [])
            state.actors = [
                {"name": a.get("name", ""), "role": a.get("role", "")}
                for a in raw_actors if isinstance(a, dict)
            ]
        except Exception as e:
            logger.warning(f"SubjectAnalyst parse failed: {e}")
            state.subject = state.query  # fallback: use query as subject
            state.actors = []

        state.subject_analyst_done = True


# ---------------------------------------------------------------------------
# 3. TechStackInferrer — runs once at Phase1→Phase2 transition
# ---------------------------------------------------------------------------

_TECH_STACK_INFERRER_SYSTEM = """You are a Tech Stack Inferrer.
Your only task: given code snippets from search results, identify the programming language, framework, and key libraries used by this codebase.

Output JSON only, no explanation.

{
    "tech_stack": "a single descriptive string, e.g. '.NET 6 / ASP.NET Core / EF Core 5 / EasyNetQ / Redis'"
}

Rules:
- Infer from file extensions (.cs → .NET, .java → Java, .py → Python)
- Infer from import/using statements and annotations
- Infer from framework patterns: [HttpPost] → ASP.NET, @PostMapping → Spring, @router.post → FastAPI
- Infer from library patterns: IConsumer<T> → EasyNetQ, @KafkaListener → Spring Kafka
- Be specific: include framework and key library names, not just the language
- If uncertain about a library, omit it rather than guess wrong
"""


class TechStackInferrer:
    """Runs once at Phase1→Phase2 transition. Infers tech stack from Phase1 evidence."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self._client, self._cfg = create_client_for("analyst")
        if provider:
            from .provider import create_client
            self._cfg.provider = provider
            self._client = create_client(provider)
        if model:
            self._cfg.model = model

    def infer(
        self,
        state: InvestigationState,
        cards: List[EvidenceCard],
        usage_log: Optional[list] = None,
        logger_obj: Any = None,
        search_id: str = "",
    ) -> None:
        """Write state.tech_stack in-place. Sets state.tech_stack_inferred."""
        # Use top-5 cards by rerank score as evidence
        top = sorted(cards, key=lambda c: c.score_rerank, reverse=True)[:5]
        snippets = []
        for c in top:
            snippet = c.chunk_text[:300].replace("\n", " ")
            snippets.append(f"[{c.path}]\n{snippet}")

        user_prompt = (
            "Evidence snippets from the codebase:\n\n"
            + "\n\n".join(snippets)
            + "\n\nInfer the tech stack and output JSON."
        )

        content = _call_llm(
            client=self._client,
            model=self._cfg.model,
            provider=self._cfg.provider,
            max_tokens=500,
            temperature=0,
            system_prompt=_TECH_STACK_INFERRER_SYSTEM,
            user_prompt=user_prompt,
            component_name="tech_stack_inferrer",
            iteration=state.iteration,
            usage_log=usage_log,
            logger_obj=logger_obj,
            search_id=search_id,
        )

        try:
            data = extract_json_from_response(content)
            state.tech_stack = data.get("tech_stack", "")
        except Exception as e:
            logger.warning(f"TechStackInferrer parse failed: {e}")
            state.tech_stack = ""

        state.tech_stack_inferred = True


# ---------------------------------------------------------------------------
# 4. CoverageAnalyst — per iteration, runs parallel with SymmetryChecker
# ---------------------------------------------------------------------------

_COVERAGE_ANALYST_SYSTEM = """You are a Coverage Analyst.
Your only task: identify which interaction dimensions ARE PRESENT in the current evidence.

Output JSON only, no explanation.

{
    "covered": ["concise description of covered dimension", "..."]
}

Rules:
- For each actor and their relationship to the subject, list every interaction path PRESENT in the evidence.
- Keep each item concise (one sentence max).
- Do NOT list what is missing. Do NOT suggest what to search next.
- Do NOT explain your reasoning.
- If evidence is empty, output {"covered": []}.
- Output strictly as a JSON array of concise strings.
"""


class CoverageAnalyst:
    """Per-iteration. Identifies what is covered. Runs parallel with SymmetryChecker."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self._client, self._cfg = create_client_for("analyst")
        if provider:
            from .provider import create_client
            self._cfg.provider = provider
            self._client = create_client(provider)
        if model:
            self._cfg.model = model

    def analyze(
        self,
        state: InvestigationState,
        evidence_summary: str,
        usage_log: Optional[list] = None,
        logger_obj: Any = None,
        search_id: str = "",
    ) -> List[str]:
        """Return covered dimensions list. Does NOT write to state (caller does)."""
        actors_str = ", ".join(
            f"{a['name']} ({a['role']})" for a in state.actors
        ) if state.actors else "unknown"

        user_prompt = (
            f"Subject: {state.subject}\n"
            f"Actors: {actors_str}\n\n"
            f"Current Evidence:\n{evidence_summary or 'No evidence collected yet.'}\n\n"
            "List all covered dimensions as JSON."
        )

        content = _call_llm(
            client=self._client,
            model=self._cfg.model,
            provider=self._cfg.provider,
            max_tokens=1000,
            temperature=0,
            system_prompt=_COVERAGE_ANALYST_SYSTEM,
            user_prompt=user_prompt,
            component_name="coverage_analyst",
            iteration=state.iteration,
            usage_log=usage_log,
            logger_obj=logger_obj,
            search_id=search_id,
        )

        try:
            data = extract_json_from_response(content)
            return data.get("covered", [])
        except Exception as e:
            logger.warning(f"CoverageAnalyst parse failed: {e}")
            return []


# ---------------------------------------------------------------------------
# 5. SymmetryChecker — per iteration, runs parallel with CoverageAnalyst
# ---------------------------------------------------------------------------

_SYMMETRY_CHECKER_SYSTEM = """You are a Symmetry Checker.
Your only task: for each anchor symbol found in the evidence, check whether its symmetric counterpart is also present.

Output JSON only, no explanation.

{
    "symmetry_gaps": ["Found X but missing Y counterpart", "..."]
}

Symmetry rules to check for each anchor:
- Write (INSERT/UPDATE/DAO.Save) → Read (SELECT/GET API/stored procedure)
- Push/Callback → Pull/Query API
- MQ Publish/Produce → MQ Subscribe/Consume
- Create/POST → Display/GET/Report

Rules:
- Only report gaps for anchors that ARE present in the evidence.
- If a counterpart is already present in the evidence, do NOT report it as a gap.
- Keep each item concise: "Found [X] but missing [Y counterpart]"
- Do NOT suggest queries. Do NOT explain. Do NOT list covered items.
- If no anchors or no gaps found, output {"symmetry_gaps": []}.
- Output strictly as a JSON array of concise strings.
"""


class SymmetryChecker:
    """Per-iteration. Checks data/operation symmetry for known anchors. Runs parallel with CoverageAnalyst."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self._client, self._cfg = create_client_for("analyst")
        if provider:
            from .provider import create_client
            self._cfg.provider = provider
            self._client = create_client(provider)
        if model:
            self._cfg.model = model

    def check(
        self,
        state: InvestigationState,
        evidence_summary: str,
        usage_log: Optional[list] = None,
        logger_obj: Any = None,
        search_id: str = "",
    ) -> List[str]:
        """Return symmetry_gaps list. Does NOT write to state (caller does)."""
        if not state.anchors:
            return []  # No anchors → no symmetry to check

        anchors_str = ", ".join(state.anchors[:10])  # cap at 10

        user_prompt = (
            f"Known anchors: {anchors_str}\n\n"
            f"Current Evidence:\n{evidence_summary or 'No evidence collected yet.'}\n\n"
            "Check symmetry and output JSON."
        )

        content = _call_llm(
            client=self._client,
            model=self._cfg.model,
            provider=self._cfg.provider,
            max_tokens=1000,
            temperature=0,
            system_prompt=_SYMMETRY_CHECKER_SYSTEM,
            user_prompt=user_prompt,
            component_name="symmetry_checker",
            iteration=state.iteration,
            usage_log=usage_log,
            logger_obj=logger_obj,
            search_id=search_id,
        )

        try:
            data = extract_json_from_response(content)
            return data.get("symmetry_gaps", [])
        except Exception as e:
            logger.warning(f"SymmetryChecker parse failed: {e}")
            return []


# ---------------------------------------------------------------------------
# 6. GapIdentifier — per iteration, after Coverage + Symmetry
# ---------------------------------------------------------------------------

_GAP_IDENTIFIER_SYSTEM = """You are a Gap Identifier.
Your only task: given what is currently covered and what symmetry gaps exist, determine what evidence still needs to be found.

Output JSON only, no explanation.

{
    "missing_evidence": [
        {
            "need": "describe what needs to be found",
            "accept": ["acceptance criterion 1", "acceptance criterion 2"],
            "priority": "high|medium|low"
        }
    ]
}

Impact framework (check each dimension for RELEVANCE, then COVERAGE):
- Data layer: entity definition, DAO, stored procedure, migration
- Business layer: service method, validation logic, business rules
- Interface layer: API controller/endpoint, DTO, request/response format
- Message layer: MQ producer, consumer, queue binding, subscriber
- Cross-service: callback, webhook, downstream notification, external API

Rules:
1. For each dimension, FIRST decide: is it RELEVANT to this query? If not relevant → skip it entirely.
2. If relevant AND not covered AND not already in symmetry_gaps → add to missing_evidence.
3. Carry-over rule: High-priority items from the 'Previous round' MUST be included in 'missing_evidence' if they are not yet marked as covered. Do NOT drop them just because you have new information.
4. Convert each item in symmetry_gaps into a high-priority missing_evidence entry.
5. Output AT MOST 3 missing_evidence items. Prioritize the dimensions most likely to contain named code anchors.
6. Do NOT generate search queries. Only describe what is needed.
7. Do NOT repeat items already fully covered.
"""


class GapIdentifier:
    """Per-iteration. Combines coverage + symmetry + impact framework to identify what's missing."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self._client, self._cfg = create_client_for("planner")
        if provider:
            from .provider import create_client
            self._cfg.provider = provider
            self._client = create_client(provider)
        if model:
            self._cfg.model = model

    def identify(
        self,
        state: InvestigationState,
        usage_log: Optional[list] = None,
        logger_obj: Any = None,
        search_id: str = "",
    ) -> List[MissingEvidence]:
        """Return missing_evidence list. Does NOT write to state (caller does)."""
        actors_str = ", ".join(
            f"{a['name']} ({a['role']})" for a in state.actors
        ) if state.actors else "unknown"

        covered_str = "\n".join(f"- {c}" for c in state.covered) if state.covered else "Nothing covered yet."
        sym_str = "\n".join(f"- {g}" for g in state.symmetry_gaps) if state.symmetry_gaps else "None."

        # Previous round's missing_evidence — helps GapIdentifier maintain direction
        prev_missing_str = (
            "\n".join(f"- [{m.priority}] {m.need}" for m in state.missing_evidence)
            if state.missing_evidence else "None."
        )

        # Impact reviewer mandatory gaps — must be included if not yet covered
        impact_gaps_str = (
            "\n".join(f"- {g}" for g in state.impact_reviewer_gaps)
            if state.impact_reviewer_gaps else "None."
        )

        user_prompt = (
            f"Query: {state.query}\n"
            f"Subject: {state.subject}\n"
            f"Actors: {actors_str}\n\n"
            f"Currently covered:\n{covered_str}\n\n"
            f"Symmetry gaps:\n{sym_str}\n\n"
            f"Previous round's missing evidence (still pursuing if not covered):\n{prev_missing_str}\n\n"
            f"⚠️ Mandatory gaps from impact reviewer (must include if not covered):\n{impact_gaps_str}\n\n"
            "Identify missing evidence and output JSON."
        )

        content = _call_llm(
            client=self._client,
            model=self._cfg.model,
            provider=self._cfg.provider,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system_prompt=_GAP_IDENTIFIER_SYSTEM,
            user_prompt=user_prompt,
            component_name="gap_identifier",
            iteration=state.iteration,
            usage_log=usage_log,
            logger_obj=logger_obj,
            search_id=search_id,
        )

        try:
            data = extract_json_from_response(content)
            result = []
            for item in data.get("missing_evidence", []):
                result.append(MissingEvidence(
                    need=item.get("need", ""),
                    accept=item.get("accept", []),
                    priority=item.get("priority", "medium"),
                ))
            return result
        except Exception as e:
            logger.warning(f"GapIdentifier parse failed: {e}")
            return []


# ---------------------------------------------------------------------------
# 7. QueryGenerator — per iteration, last step
# ---------------------------------------------------------------------------

_QUERY_GENERATOR_SYSTEM = """You are a Query Generator.
Your only task: for each missing evidence need, generate specific search queries to find it.

Output JSON only, no explanation.

{
    "next_queries": [
        {
            "query": "search query string or exact symbol name",
            "purpose": "what this query is looking for",
            "query_type": "semantic|keyword|exact|graph_traverse_callers|graph_traverse_callees|graph_traverse_implementations|graph_traverse_inherits",
            "operator": "semantic|keyword|exact|symbol_ref|callsite"
        }
    ]
}

Query type reference:
- semantic: Pure semantic/conceptual search — best for vague or domain-language queries
- keyword: BM25 keyword search — best for technical terms, class names
- exact: Exact match — best for known class or method names
- graph_traverse_callers: Find who calls this symbol (set query = exact symbol name)
- graph_traverse_callees: Find what this symbol calls (set query = exact symbol name)
- graph_traverse_implementations: Find who implements this interface (set query = interface name)
- graph_traverse_inherits: Find what this class inherits from (set query = class name)

PHASE 1 strategy (no anchors yet, tech_stack unknown):
- tech_stack is not yet determined — do NOT assume any framework or use framework-specific patterns
- Use broad, conceptual, domain-language queries
- Examples: "deposit process flow", "payment callback notification", "bank statement matching"
- Goal: surface named code symbols — accept imprecision
- Do NOT use technical class names or annotations you haven't seen in evidence yet
- Use query_type: "semantic" or "keyword" only in Phase 1

PHASE 2 strategy (anchors available, tech_stack known):
- Use precise anchor-based queries with known anchor symbols
- Apply tech-stack-specific patterns from the provided tech_stack context:
  .NET: look for I{Service} IMPLEMENTS, [HttpPost] controllers, IConsumer<T>
  Spring: look for @Autowired, @KafkaListener, @PostMapping
  Python/FastAPI: look for Depends(), @router.post, Celery tasks
- Prefer graph traversal when you have a specific symbol name
- For remaining semantic needs: use query_type "semantic" or "keyword"

Rules:
- Generate at most 4 queries total.
- Do NOT analyze coverage or gaps — only generate queries.
- Do NOT explain your reasoning in the output JSON.
"""


class QueryGenerator:
    """Per-iteration. Converts missing_evidence into concrete QueryIntent objects."""

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self._client, self._cfg = create_client_for("planner")
        if provider:
            from .provider import create_client
            self._cfg.provider = provider
            self._client = create_client(provider)
        if model:
            self._cfg.model = model

    def generate(
        self,
        state: InvestigationState,
        usage_log: Optional[list] = None,
        logger_obj: Any = None,
        search_id: str = "",
    ) -> List[QueryIntent]:
        """Return list of QueryIntent objects to execute this iteration."""
        missing_str = "\n".join(
            f"- [{m.priority}] {m.need}" for m in state.missing_evidence
        ) if state.missing_evidence else "No specific gaps — do a broad exploration."

        anchors_str = ", ".join(state.anchors[:10]) if state.anchors else "none"
        tech_str = state.tech_stack or "unknown (Phase 1 — do not assume)"

        # Last 10 queries — avoid regenerating already-tried searches
        history_str = (
            "\n".join(f"- {q}" for q in state.search_history[-10:])
            if state.search_history else "None yet."
        )

        user_prompt = (
            f"Current phase: {state.phase.value}\n"
            f"Known anchors: {anchors_str}\n"
            f"Tech stack: {tech_str}\n\n"
            f"Missing evidence to find:\n{missing_str}\n\n"
            f"Already searched (do NOT repeat these):\n{history_str}\n\n"
            "Generate search queries and output JSON."
        )

        content = _call_llm(
            client=self._client,
            model=self._cfg.model,
            provider=self._cfg.provider,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            system_prompt=_QUERY_GENERATOR_SYSTEM,
            user_prompt=user_prompt,
            component_name="query_generator",
            iteration=state.iteration,
            usage_log=usage_log,
            logger_obj=logger_obj,
            search_id=search_id,
        )

        try:
            data = extract_json_from_response(content)
            result = []
            for q in data.get("next_queries", []):
                result.append(QueryIntent(
                    query=q.get("query", ""),
                    purpose=q.get("purpose", ""),
                    query_type=q.get("query_type", "semantic"),
                    operator=q.get("operator", "semantic"),
                ))
            return result
        except Exception as e:
            logger.warning(f"QueryGenerator parse failed: {e}")
            # Fallback: single broad semantic search
            return [QueryIntent(
                query=state.query,
                purpose="fallback broad search",
                query_type="semantic",
                operator="semantic",
            )]


# ---------------------------------------------------------------------------
# Parallel runner for CoverageAnalyst + SymmetryChecker
# ---------------------------------------------------------------------------

def run_coverage_and_symmetry_parallel(
    coverage_analyst: CoverageAnalyst,
    symmetry_checker: SymmetryChecker,
    state: InvestigationState,
    evidence_summary: str,
    usage_log: Optional[list] = None,
    logger_obj: Any = None,
    search_id: str = "",
) -> None:
    """Run CoverageAnalyst and SymmetryChecker in parallel, write results to state."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_coverage = executor.submit(
            coverage_analyst.analyze,
            state, evidence_summary, usage_log, logger_obj, search_id,
        )
        f_symmetry = executor.submit(
            symmetry_checker.check,
            state, evidence_summary, usage_log, logger_obj, search_id,
        )
        try:
            state.covered = f_coverage.result()
        except Exception as e:
            logger.warning(f"CoverageAnalyst failed: {e}")
            state.covered = []
        try:
            state.symmetry_gaps = f_symmetry.result()
        except Exception as e:
            logger.warning(f"SymmetryChecker failed: {e}")
            state.symmetry_gaps = []
