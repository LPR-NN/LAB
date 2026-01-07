"""High-level Python API for AI Committee Member."""

from pathlib import Path
from typing import Literal

from source.contracts.request import Request
from source.contracts.decision import Decision
from source.contracts.audit import DecisionPackage
from source.corpus.loader import DocumentLoader
from source.corpus.index import CorpusIndex
from source.providers.factory import ProviderFactory
from source.committee.commands import (
    CommandHandler,
    IntakeResult,
    CiteResult,
    AppealDraft,
    ComparisonResult,
    RedTeamResult,
)
from source.audit.storage import AuditStorage
from source.audit.package import DecisionPackageBuilder


class CommitteeMember:
    """
    High-level API for the AI Committee Member.

    Example usage:

        committee = CommitteeMember(
            corpus_path="data/",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        # Check if request is complete
        intake_result = await committee.intake(request)

        # Make a decision
        decision = await committee.decide(request)

        # Red team the decision
        issues = await committee.redteam(request, decision)
    """

    def __init__(
        self,
        corpus_path: Path | str = "data/",
        provider: Literal["openai", "anthropic"] = "anthropic",
        model: str | None = None,
        audit_path: Path | str = "audit_logs/",
        max_steps: int = 20,
        save_audit: bool = True,
    ):
        """
        Initialize the Committee Member.

        Args:
            corpus_path: Path to corpus directory
            provider: LLM provider ("openai" or "anthropic")
            model: Model to use (default: provider default)
            audit_path: Path for audit logs
            max_steps: Maximum agent steps
            save_audit: Whether to save audit packages
        """
        self.corpus_path = Path(corpus_path)
        self.audit_path = Path(audit_path)
        self.provider_type = provider
        self.max_steps = max_steps
        self.save_audit = save_audit

        # Set default model
        if model is None:
            model = "claude-sonnet-4-20250514" if provider == "anthropic" else "gpt-4o"
        self.model = model

        # Initialize components
        self._corpus: CorpusIndex | None = None
        self._provider = None
        self._handler: CommandHandler | None = None
        self._storage: AuditStorage | None = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize components."""
        if self._corpus is None:
            loader = DocumentLoader(self.corpus_path)
            documents = loader.load_corpus_sync()
            self._corpus = CorpusIndex(documents)

        if self._provider is None:
            self._provider = ProviderFactory.create(
                self.provider_type,
                self.model,
            )

        if self._handler is None:
            self._handler = CommandHandler(
                self._provider,
                self._corpus,
                self.max_steps,
            )

        if self._storage is None:
            self._storage = AuditStorage(self.audit_path)

    @property
    def corpus(self) -> CorpusIndex:
        """Get the corpus index."""
        self._ensure_initialized()
        return self._corpus

    async def reload_corpus(self) -> int:
        """
        Reload the corpus from disk.

        Returns:
            Number of documents loaded
        """
        loader = DocumentLoader(self.corpus_path)
        documents = loader.load_corpus_sync()
        self._corpus = CorpusIndex(documents)

        # Reinitialize handler with new corpus
        self._handler = CommandHandler(
            self._provider,
            self._corpus,
            self.max_steps,
        )

        return len(documents)

    async def intake(self, request: Request) -> IntakeResult:
        """
        Check if a request is complete and ready for decision.

        Args:
            request: The case request

        Returns:
            IntakeResult with completeness assessment
        """
        self._ensure_initialized()
        return await self._handler.intake(request)

    async def decide(
        self,
        request: Request,
        save_audit: bool | None = None,
    ) -> tuple[Decision, str | None]:
        """
        Make a decision on a request.

        Args:
            request: The case request
            save_audit: Override save_audit setting

        Returns:
            (Decision, audit_package_id or None)
        """
        self._ensure_initialized()

        decision = await self._handler.decide(request)

        # Save audit package
        audit_id = None
        should_save = save_audit if save_audit is not None else self.save_audit

        if should_save:
            builder = DecisionPackageBuilder(
                self._corpus,
                model_id=self.model,
                provider=self.provider_type,
            )
            package = builder.build(request, decision)
            self._storage.save(package)
            audit_id = package.package_id

        return decision, audit_id

    async def cite(
        self,
        request: Request,
        decision: Decision,
    ) -> CiteResult:
        """
        Extract and verify citations from a decision.

        Args:
            request: Original request
            decision: Decision to analyze

        Returns:
            CiteResult with verified citations
        """
        self._ensure_initialized()
        return await self._handler.cite(request, decision)

    async def appeal(
        self,
        request: Request,
        new_facts: list[str],
    ) -> AppealDraft:
        """
        Draft an appeal based on new facts.

        Args:
            request: Original request
            new_facts: List of new facts to consider

        Returns:
            AppealDraft with recommendation
        """
        self._ensure_initialized()
        return await self._handler.appeal(request, new_facts)

    async def compare(
        self,
        request: Request,
        precedent_ids: list[str] | None = None,
    ) -> ComparisonResult:
        """
        Compare a case with precedents.

        Args:
            request: Case request
            precedent_ids: Specific precedent IDs (or None to auto-find)

        Returns:
            ComparisonResult with analysis
        """
        self._ensure_initialized()
        return await self._handler.compare(request, precedent_ids or [])

    async def redteam(
        self,
        request: Request,
        decision: Decision,
    ) -> RedTeamResult:
        """
        Perform adversarial analysis of a decision.

        Args:
            request: Original request
            decision: Decision to analyze

        Returns:
            RedTeamResult with issues and recommendations
        """
        self._ensure_initialized()
        return await self._handler.redteam(request, decision)

    def get_audit_package(self, package_id: str) -> DecisionPackage | None:
        """
        Retrieve an audit package by ID.

        Args:
            package_id: The package UUID

        Returns:
            DecisionPackage or None if not found
        """
        self._ensure_initialized()
        return self._storage.load(package_id)

    def verify_audit(self, package_id: str) -> tuple[bool, str]:
        """
        Verify integrity of an audit package.

        Args:
            package_id: The package UUID

        Returns:
            (is_valid, message) tuple
        """
        self._ensure_initialized()
        return self._storage.verify(package_id)

    def list_audits(self) -> list[str]:
        """List all audit package IDs."""
        self._ensure_initialized()
        return self._storage.list_packages()

    def corpus_stats(self) -> dict:
        """Get corpus statistics."""
        self._ensure_initialized()
        return self._corpus.get_statistics()


__all__ = ["CommitteeMember"]
