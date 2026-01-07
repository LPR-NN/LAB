"""Decision package builder for audit trail."""

import datetime
import hashlib
import json

from source.contracts.audit import (
    CitedFragment,
    CorpusDocumentRef,
    DecisionPackage,
    ToolCallHistory,
)
from source.contracts.decision import Decision
from source.contracts.request import Request
from source.corpus.index import CorpusIndex


class DecisionPackageBuilder:
    """
    Builder for creating decision audit packages.

    Captures all information needed to reproduce and verify a decision.
    """

    def __init__(
        self,
        corpus: CorpusIndex,
        agent_version: str = "0.1.0",
        model_id: str = "",
        provider: str = "",
        temperature: float = 0.0,
    ):
        self.corpus = corpus
        self.agent_version = agent_version
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature

    def build(
        self,
        request: Request,
        decision: Decision,
        tool_calls: ToolCallHistory | None = None,
    ) -> DecisionPackage:
        """
        Build a complete audit package for a decision.

        Args:
            request: Original request
            decision: Final decision
            tool_calls: Optional history of tool calls made

        Returns:
            Complete DecisionPackage
        """
        corpus_snapshot = self._capture_corpus_snapshot(decision)
        cited_fragments = self._extract_cited_fragments(decision)

        request_json = request.model_dump_json(indent=2)
        decision_json = decision.model_dump_json(indent=2)

        mode = "deterministic" if self.temperature == 0.0 else "creative"

        return DecisionPackage(
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            request_json=request_json,
            corpus_snapshot=corpus_snapshot,
            cited_fragments=cited_fragments,
            tool_calls=tool_calls,
            decision_json=decision_json,
            agent_version=self.agent_version,
            model_id=self.model_id,
            provider=self.provider,
            mode=mode,
            temperature=self.temperature,
        )

    def _capture_corpus_snapshot(self, decision: Decision) -> list[CorpusDocumentRef]:
        """Capture references to all documents cited in the decision."""
        refs: list[CorpusDocumentRef] = []
        seen_ids: set[str] = set()

        # Get doc_ids from applicable_norms
        for norm in decision.applicable_norms:
            if norm.doc_id not in seen_ids:
                doc = self.corpus.get_by_id(norm.doc_id)
                if doc:
                    refs.append(
                        CorpusDocumentRef(
                            doc_id=doc.doc_id,
                            version=doc.metadata.version,
                            content_hash=doc.content_hash,
                        )
                    )
                    seen_ids.add(norm.doc_id)

        # Get doc_ids from citations
        for citation in decision.citations:
            if citation.doc_id not in seen_ids:
                doc = self.corpus.get_by_id(citation.doc_id)
                if doc:
                    refs.append(
                        CorpusDocumentRef(
                            doc_id=doc.doc_id,
                            version=doc.metadata.version,
                            content_hash=doc.content_hash,
                        )
                    )
                    seen_ids.add(citation.doc_id)

        return refs

    def _extract_cited_fragments(self, decision: Decision) -> list[CitedFragment]:
        """Extract all cited fragments from the decision."""
        fragments: list[CitedFragment] = []

        for citation in decision.citations:
            fragments.append(
                CitedFragment(
                    doc_id=citation.doc_id,
                    section=citation.section,
                    quoted_text=citation.quoted_text,
                    context=None,  # Could be enhanced to include surrounding context
                )
            )

        return fragments

    def with_model_info(
        self, model_id: str, provider: str, temperature: float = 0.0
    ) -> "DecisionPackageBuilder":
        """Update model information."""
        self.model_id = model_id
        self.provider = provider
        self.temperature = temperature
        return self


def compute_package_hash(package: DecisionPackage) -> str:
    """Compute SHA-256 hash of the package for integrity verification."""
    # Exclude the hash field itself
    data = package.model_dump(exclude={"content_hash"})

    # Convert to deterministic JSON string
    json_str = json.dumps(data, sort_keys=True, ensure_ascii=True, default=str)

    return hashlib.sha256(json_str.encode()).hexdigest()


__all__ = [
    "DecisionPackageBuilder",
    "compute_package_hash",
]
