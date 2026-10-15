"""Replaceable research-source providers.

The curated provider is deterministic and fully offline.  The arXiv provider is
best-effort and primary-source only; callers must preserve its failure status and
fall back honestly rather than presenting a stale corpus as a live refresh.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Protocol

from vllm_evolve.knowledge.schemas import MechanismCard, ResearchSource

_UPSTREAM_FILES = (
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/sched/async_scheduler.py",
    "vllm/v1/core/sched/request_queue.py",
    "vllm/v1/core/sched/output.py",
    "vllm/v1/core/kv_cache_manager.py",
)

# This is an audit vocabulary, not a candidate catalogue.  A capability is
# classified from immutable source text at the installed commit and current
# upstream main.  Missing entries can become research questions, but cannot
# become candidates until a MechanismCard supplies a separately reviewed delta.
_UPSTREAM_CAPABILITIES = (
    {
        "mechanism_id": "continuous_batching",
        "name": "iteration-level continuous batching",
        "symbols": ["Scheduler.schedule", "Scheduler.waiting", "Scheduler.running"],
        "marker_groups": (("def schedule(",), ("self.waiting",), ("self.running",)),
        "baseline_flag": "continuous_batching",
    },
    {
        "mechanism_id": "async_scheduler",
        "name": "asynchronous scheduler/model-runner overlap",
        "symbols": ["AsyncScheduler", "AsyncScheduler.update_from_output"],
        "marker_groups": (("class AsyncScheduler(",),),
        "baseline_flag": "async_scheduling",
    },
    {
        "mechanism_id": "fcfs_priority",
        "name": "FCFS and explicit request priority queues",
        "symbols": ["SchedulingPolicy.FCFS", "SchedulingPolicy.PRIORITY"],
        "marker_groups": (("SchedulingPolicy",), ("FCFS",), ("PRIORITY",)),
        "baseline_flag": "scheduler_policy",
    },
    {
        "mechanism_id": "chunked_prefill",
        "name": "chunked prefill",
        "symbols": ["Scheduler.schedule", "long_prefill_token_threshold"],
        "marker_groups": (("long_prefill_token_threshold",),),
        "baseline_flag": "enable_chunked_prefill",
    },
    {
        "mechanism_id": "prefix_caching",
        "name": "automatic prefix caching and block reuse",
        "symbols": ["KVCacheManager.get_computed_blocks", "KVCacheManager.allocate_slots"],
        "marker_groups": (("get_computed_blocks",), ("enable_caching",)),
        "baseline_flag": "enable_prefix_caching",
    },
    {
        "mechanism_id": "preempt_resume_lifecycle",
        "name": "KV-driven request preemption and resume",
        "symbols": ["Scheduler._preempt_request", "Request.num_preemptions"],
        "marker_groups": (("_preempt_request",), ("num_preemptions",)),
        "baseline_flag": "preemption",
    },
    {
        "mechanism_id": "speculative_decoding_scheduler",
        "name": "speculative-token-aware scheduling",
        "symbols": ["Scheduler.schedule", "SchedulerOutput.spec_decode_token_ids"],
        "marker_groups": (("spec_token_ids", "scheduled_spec_decode_tokens"),),
        "baseline_flag": "speculative_decoding",
    },
    {
        "mechanism_id": "kv_admission_watermark",
        "name": "KV-cache admission watermark",
        "symbols": ["KVCacheManager.watermark_blocks", "KVCacheManager.allocate_slots"],
        "marker_groups": (("watermark_blocks",),),
        "baseline_flag": "scheduler_watermark",
    },
    {
        "mechanism_id": "full_sequence_admission_reservation",
        "name": "full input-sequence KV reservation at admission",
        "symbols": ["Scheduler.scheduler_reserve_full_isl", "KVCacheManager.allocate_slots"],
        "marker_groups": (("scheduler_reserve_full_isl",), ("full_sequence_must_fit",)),
        "baseline_flag": "scheduler_reserve_full_isl",
    },
    {
        "mechanism_id": "allocation_failure_fit_bypass",
        "name": "bounded bypass of a KV-unfit waiting head",
        "symbols": ["Scheduler.schedule:waiting allocation failure"],
        "marker_groups": (("continue_after_allocation_failure", "fit_bypass"),),
        "baseline_flag": "allocation_failure_fit_bypass",
        "gap": (
            "The stock waiting loop stops when allocate_slots returns None; it does not "
            "search for a later request that fits the live KV budget."
        ),
    },
    {
        "mechanism_id": "residual_work_victim_selection",
        "name": "residual-work-aware active preemption victim selection",
        "symbols": ["Scheduler.schedule:running preemption"],
        "marker_groups": (("remaining_output_tokens", "residual_work_victim"),),
        "baseline_flag": "residual_work_victim_selection",
        "gap": (
            "Stock preemption is allocation-failure recovery, not an explicit choice of "
            "victim from scheduler-visible residual decode work."
        ),
    },
    {
        "mechanism_id": "deadline_slo_admission",
        "name": "deadline/SLO-aware admission",
        "symbols": ["Request.deadline", "Scheduler.schedule"],
        "marker_groups": (("deadline", "slo_deadline"),),
        "baseline_flag": "deadline_slo_admission",
        "required_surface_signal": "request_deadline",
    },
    {
        "mechanism_id": "tenant_fair_scheduling",
        "name": "tenant-aware fair scheduling",
        "symbols": ["Request.tenant_id", "Scheduler.schedule"],
        "marker_groups": (("tenant_id", "virtual_finish_time"),),
        "baseline_flag": "tenant_fair_scheduling",
        "required_surface_signal": "tenant_labels",
    },
)


def _contains_marker_groups(files: dict[str, str], groups) -> bool:
    joined = "\n".join(files.values())
    return all(any(marker in joined for marker in alternatives) for alternatives in groups)


def build_upstream_capability_matrix(
    *,
    installed_ref: str,
    main_ref: str,
    installed_files: dict[str, str],
    main_files: dict[str, str],
    environment: dict,
) -> dict:
    """Classify audited mechanisms against installed vLLM and live main."""
    baseline = dict(environment.get("strong_baseline") or {})
    signals = set(environment.get("workload_signals") or [])
    capabilities = []
    for spec in _UPSTREAM_CAPABILITIES:
        installed = _contains_marker_groups(installed_files, spec["marker_groups"])
        upstream = _contains_marker_groups(main_files, spec["marker_groups"])
        required_signal = spec.get("required_surface_signal")
        if required_signal and required_signal not in signals:
            status = "incompatible_with_current_surface"
        elif installed:
            status = "integrated"
        elif upstream:
            status = "partially_integrated"
        else:
            status = "missing"
        file_permalinks = [
            (
                "https://github.com/vllm-project/vllm/blob/"
                f"{installed_ref}/{path}"
            )
            for path in _UPSTREAM_FILES
            if path in installed_files
        ]
        main_permalinks = [
            f"https://github.com/vllm-project/vllm/blob/{main_ref}/{path}"
            for path in _UPSTREAM_FILES
            if path in main_files
        ]
        capabilities.append(
            {
                "mechanism_id": spec["mechanism_id"],
                "name": spec["name"],
                "status": status,
                "installed_ref": installed_ref,
                "upstream_main_ref": main_ref,
                "installed_evidence_found": installed,
                "upstream_main_evidence_found": upstream,
                "upstream_symbols_checked": list(spec["symbols"]),
                "installed_file_permalinks": file_permalinks,
                "upstream_main_file_permalinks": main_permalinks,
                "strong_baseline_enabled": bool(
                    baseline.get(spec["baseline_flag"], False)
                ),
                "current_behavior_or_gap": spec.get("gap", ""),
                "required_surface_signal": required_signal,
            }
        )
    return {
        "schema_version": 1,
        "installed_vllm_ref": installed_ref,
        "upstream_main_ref": main_ref,
        "audited_files": list(_UPSTREAM_FILES),
        "capabilities": capabilities,
    }


@dataclass
class ResearchRequest:
    target: str
    spec: dict
    diagnosis: dict
    environment: dict
    queries: list[str]


@dataclass
class ProviderResult:
    provider: str
    sources: list[ResearchSource] = field(default_factory=list)
    mechanisms: list[MechanismCard] = field(default_factory=list)
    status: str = "ok"
    detail: str = ""
    corpus_metadata: dict = field(default_factory=dict)

    def status_dict(self) -> dict:
        return {
            "provider": self.provider,
            "status": self.status,
            "detail": self.detail,
            "source_count": len(self.sources),
            "mechanism_count": len(self.mechanisms),
        }


class ResearchProvider(Protocol):
    name: str

    def fetch(self, request: ResearchRequest) -> ProviderResult: ...


class CuratedCorpusProvider:
    """Load a reviewed, immutable target corpus shipped with the package."""

    name = "curated_corpus"

    def __init__(self, corpus_dir: str | Path | None = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else None

    def _read(self, target: str) -> dict:
        filename = f"{target}.json"
        if self.corpus_dir is not None:
            path = self.corpus_dir / filename
            if not path.is_file():
                return {}
            return json.loads(path.read_text(encoding="utf-8"))
        ref = resources.files("vllm_evolve.knowledge.corpora").joinpath(filename)
        if not ref.is_file():
            return {}
        return json.loads(ref.read_text(encoding="utf-8"))

    def fetch(self, request: ResearchRequest) -> ProviderResult:
        data = self._read(request.target)
        if not data:
            return ProviderResult(
                provider=self.name,
                status="target_corpus_not_found",
                detail=f"no curated corpus for target {request.target!r}",
            )
        return ProviderResult(
            provider=self.name,
            sources=[ResearchSource.from_dict(item) for item in data.get("sources", [])],
            mechanisms=[MechanismCard.from_dict(item) for item in data.get("mechanism_cards", [])],
            corpus_metadata=dict(data.get("metadata") or {}),
        )


class ArxivResearchProvider:
    """Small stdlib arXiv Atom client used for live recent-paper refreshes."""

    name = "arxiv_primary"
    endpoint = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        max_queries: int = 4,
        max_results_per_query: int = 3,
        timeout_s: float = 12.0,
        opener=None,
    ):
        self.max_queries = max(1, int(max_queries))
        self.max_results_per_query = max(1, int(max_results_per_query))
        self.timeout_s = float(timeout_s)
        self._opener = opener or urllib.request.urlopen

    @staticmethod
    def _clean(text: str | None) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _fetch_query(self, query: str) -> list[ResearchSource]:
        terms = list(
            dict.fromkeys(
                token
                for token in re.findall(r"[A-Za-z][A-Za-z0-9-]+", query)
                if len(token) >= 3
            )
        )
        # A quoted full sentence is interpreted as an exact phrase by arXiv
        # and routinely returns zero papers.  Three conjunctive topic terms
        # retain target conditioning while still finding current systems work.
        search_query = " AND ".join(f"all:{term}" for term in terms[:3])
        if not search_query:
            search_query = f'all:"{query}"'
        params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": self.max_results_per_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        req = urllib.request.Request(
            f"{self.endpoint}?{params}",
            headers={"User-Agent": "vllm-evolve/0.1 research-compiler"},
        )
        with self._opener(req, timeout=self.timeout_s) as response:
            payload = response.read()
        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        sources: list[ResearchSource] = []
        current_year = datetime.now(timezone.utc).year
        for entry in root.findall("atom:entry", ns):
            canonical = self._clean(entry.findtext("atom:id", default="", namespaces=ns))
            if not canonical:
                continue
            arxiv_id = canonical.rstrip("/").split("/")[-1]
            published = self._clean(entry.findtext("atom:published", default="", namespaces=ns))
            try:
                year = int(published[:4])
            except (TypeError, ValueError):
                year = current_year
            authors = [
                self._clean(author.findtext("atom:name", default="", namespaces=ns))
                for author in entry.findall("atom:author", ns)
            ]
            sources.append(
                ResearchSource(
                    source_id=f"arxiv:{arxiv_id}",
                    title=self._clean(entry.findtext("atom:title", default="", namespaces=ns)),
                    authors=[author for author in authors if author],
                    year=year,
                    date=published,
                    source_kind="paper",
                    canonical_url=canonical,
                    identifier=arxiv_id,
                    freshness="recent" if year >= current_year - 2 else "classic",
                    retrieval_time=datetime.now(timezone.utc).isoformat(),
                    primary_source=True,
                    summary=self._clean(entry.findtext("atom:summary", default="", namespaces=ns)),
                    citation_status="retrieved_primary",
                    provider=self.name,
                )
            )
        return sources

    def fetch(self, request: ResearchRequest) -> ProviderResult:
        sources: list[ResearchSource] = []
        failures: list[str] = []
        for query in request.queries[: self.max_queries]:
            try:
                sources.extend(self._fetch_query(query))
            except Exception as exc:  # noqa: BLE001 - provider failure is returned, never hidden
                failures.append(f"{query}: {type(exc).__name__}: {exc}")
        if failures and not sources:
            return ProviderResult(
                provider=self.name,
                status="online_unavailable_fallback_curated",
                detail="; ".join(failures)[:1000],
            )
        return ProviderResult(
            provider=self.name,
            sources=sources,
            status="partial" if failures else "ok",
            detail="; ".join(failures)[:1000],
        )


class GitHubVLLMUpstreamProvider:
    """Live, primary-source snapshot of vLLM main and its latest release.

    This provider deliberately has no offline fallback. A live-required round
    must fail if current upstream state cannot be retrieved.
    """

    name = "github_vllm_upstream"
    api_root = "https://api.github.com/repos/vllm-project/vllm"

    def __init__(self, *, timeout_s: float = 15.0, opener=None):
        self.timeout_s = float(timeout_s)
        self._opener = opener or urllib.request.urlopen

    def _json(self, suffix: str):
        request = urllib.request.Request(
            f"{self.api_root}/{suffix.lstrip('/')}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "vllm-evolve/0.1 live-required-research",
            },
        )
        with self._opener(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _source_file(self, path: str, ref: str) -> str:
        query = urllib.parse.urlencode({"ref": ref})
        payload = self._json(f"contents/{path}?{query}")
        if not isinstance(payload, dict) or payload.get("encoding") != "base64":
            raise ValueError(f"GitHub did not return base64 contents for {path}@{ref}")
        return base64.b64decode(str(payload["content"])).decode("utf-8")

    def _source_tree(self, ref: str) -> tuple[dict[str, str], list[str]]:
        files = {}
        failures = []
        for path in _UPSTREAM_FILES:
            try:
                files[path] = self._source_file(path, ref)
            except Exception as exc:  # one absent file makes the audit partial, not fabricated
                failures.append(f"{path}@{ref}: {type(exc).__name__}: {exc}")
        return files, failures

    @staticmethod
    def _commit_source(commit: dict, retrieved: str) -> ResearchSource:
        sha = str(commit["sha"])
        metadata = dict(commit.get("commit") or {})
        committer = dict(metadata.get("committer") or {})
        message = str(metadata.get("message") or "").splitlines()[0]
        return ResearchSource(
            source_id=f"github:vllm-commit:{sha}",
            title=message or f"vLLM commit {sha[:12]}",
            authors=[],
            year=int(str(committer.get("date") or retrieved)[:4]),
            date=str(committer.get("date") or ""),
            source_kind="code",
            canonical_url=f"https://github.com/vllm-project/vllm/commit/{sha}",
            identifier=sha,
            freshness="recent",
            retrieval_time=retrieved,
            primary_source=True,
            summary=message,
            citation_status="retrieved_primary",
            provider=GitHubVLLMUpstreamProvider.name,
        )

    def fetch(self, request: ResearchRequest) -> ProviderResult:
        retrieved = datetime.now(timezone.utc).isoformat()
        try:
            head = self._json("commits/main")
            release = self._json("releases/latest")
            head_sha = str(head["sha"])
            release_tag = str(release["tag_name"])
            installed_ref = str(
                request.environment.get("installed_vllm_commit")
                or request.environment.get("vllm_commit")
                or release_tag
            )
            recent_commits = self._json(
                "commits?"
                + urllib.parse.urlencode(
                    {
                        "path": "vllm/v1/core/sched/scheduler.py",
                        "per_page": 12,
                    }
                )
            )
            if not isinstance(recent_commits, list):
                raise ValueError("GitHub commits endpoint did not return a list")
            installed_files, installed_failures = self._source_tree(installed_ref)
            main_files, main_failures = self._source_tree(head_sha)
            matrix = build_upstream_capability_matrix(
                installed_ref=installed_ref,
                main_ref=head_sha,
                installed_files=installed_files,
                main_files=main_files,
                environment=request.environment,
            )
            sources = [
                ResearchSource(
                    source_id=f"github:vllm-main:{head_sha}",
                    title=f"vLLM upstream main at {head_sha[:12]}",
                    authors=[],
                    year=int(retrieved[:4]),
                    date=str(head.get("commit", {}).get("committer", {}).get("date") or ""),
                    source_kind="code",
                    canonical_url=f"https://github.com/vllm-project/vllm/tree/{head_sha}",
                    identifier=head_sha,
                    freshness="recent",
                    retrieval_time=retrieved,
                    primary_source=True,
                    summary=str(head.get("commit", {}).get("message") or "").splitlines()[0],
                    citation_status="retrieved_primary",
                    provider=self.name,
                ),
                ResearchSource(
                    source_id=f"github:vllm-release:{release_tag}",
                    title=f"vLLM release {release_tag}",
                    authors=[],
                    year=int(str(release.get("published_at") or retrieved)[:4]),
                    date=str(release.get("published_at") or ""),
                    source_kind="framework_doc",
                    canonical_url=str(release["html_url"]),
                    identifier=release_tag,
                    freshness="recent",
                    retrieval_time=retrieved,
                    primary_source=True,
                    summary=str(release.get("name") or release_tag),
                    citation_status="retrieved_primary",
                    provider=self.name,
                ),
                *[
                    self._commit_source(commit, retrieved)
                    for commit in recent_commits
                    if str(commit.get("sha") or "") != head_sha
                ],
            ]
        except Exception as exc:  # no stale substitution in live-required mode
            return ProviderResult(
                provider=self.name,
                status="live_upstream_unavailable",
                detail=f"{type(exc).__name__}: {exc}"[:1000],
            )
        failures = [*installed_failures, *main_failures]
        excluded = [
            item
            for item in matrix["capabilities"]
            if item["status"] in {"integrated", "partially_integrated"}
        ]
        gaps = [
            item
            for item in matrix["capabilities"]
            if item["status"] in {
                "missing",
                "incompatible_with_current_surface",
                "uncertain_requires_experiment",
            }
        ]
        return ProviderResult(
            provider=self.name,
            sources=sources,
            status="partial" if failures else "ok",
            detail=(
                f"installed={installed_ref}; main={head_sha}; "
                f"latest_release={release_tag}"
                + (f"; partial_file_failures={'; '.join(failures)[:600]}" if failures else "")
            ),
            corpus_metadata={
                "installed_vllm_ref": installed_ref,
                "upstream_main_sha": head_sha,
                "latest_release_tag": release_tag,
                "retrieved_at": retrieved,
                "upstream_capability_matrix": matrix,
                "integrated_mechanisms_excluded": excluded,
                "recent_gap_sources": {
                    "gaps": gaps,
                    "primary_sources": [
                        source.to_dict()
                        for source in sources
                        if source.freshness == "recent"
                    ],
                },
                "unexplored_mechanism_gaps": [
                    item["mechanism_id"] for item in gaps
                ],
            },
        )


__all__ = [
    "ArxivResearchProvider",
    "CuratedCorpusProvider",
    "GitHubVLLMUpstreamProvider",
    "ProviderResult",
    "ResearchProvider",
    "ResearchRequest",
    "build_upstream_capability_matrix",
]
