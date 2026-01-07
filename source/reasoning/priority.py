"""Priority resolution for norm conflicts."""

from typing import Literal

from source.contracts.corpus import CorpusDocument


class PriorityResolver:
    """
    Resolves norm conflicts using legal principles.

    Implements:
    - Lex Superior: Higher authority prevails
    - Lex Specialis: More specific norm prevails
    - Lex Posterior: Later norm prevails (same level)
    """

    @staticmethod
    def lex_superior(
        norm_a: CorpusDocument,
        norm_b: CorpusDocument,
    ) -> CorpusDocument:
        """
        Higher authority prevails.

        Priority order: charter > regulations > code > decision > precedent > clarification
        """
        if norm_a.priority > norm_b.priority:
            return norm_a
        elif norm_b.priority > norm_a.priority:
            return norm_b
        else:
            raise ValueError(
                f"Same priority level ({norm_a.priority}). "
                "Use lex_specialis or lex_posterior."
            )

    @staticmethod
    def lex_posterior(
        norm_a: CorpusDocument,
        norm_b: CorpusDocument,
    ) -> CorpusDocument:
        """
        Later norm prevails (for same authority level).
        """
        if norm_a.priority != norm_b.priority:
            raise ValueError(
                f"Different priority levels ({norm_a.priority} vs {norm_b.priority}). "
                "Lex posterior only applies to same-level norms."
            )

        date_a = norm_a.metadata.effective_date
        date_b = norm_b.metadata.effective_date

        if date_a is None and date_b is None:
            raise ValueError("Both norms lack effective dates.")

        if date_a is None:
            return norm_b
        if date_b is None:
            return norm_a

        return norm_a if date_a > date_b else norm_b

    @staticmethod
    def resolve(
        norm_a: CorpusDocument,
        norm_b: CorpusDocument,
        conflict_type: Literal["superior", "posterior", "auto"] = "auto",
    ) -> tuple[CorpusDocument, str]:
        """
        Resolve conflict between two norms.

        Args:
            norm_a: First norm
            norm_b: Second norm
            conflict_type: "superior", "posterior", or "auto"

        Returns:
            (prevailing_norm, detailed_reason)
        """
        if conflict_type == "superior" or (
            conflict_type == "auto" and norm_a.priority != norm_b.priority
        ):
            try:
                winner = PriorityResolver.lex_superior(norm_a, norm_b)
                loser = norm_b if winner == norm_a else norm_a
                reason = (
                    f"{winner.doc_type} (priority {winner.priority}) has higher authority "
                    f"than {loser.doc_type} (priority {loser.priority})"
                )
                return winner, reason
            except ValueError:
                pass

        if conflict_type in ("posterior", "auto"):
            try:
                winner = PriorityResolver.lex_posterior(norm_a, norm_b)
                loser = norm_b if winner == norm_a else norm_a
                date_a = norm_a.metadata.effective_date
                date_b = norm_b.metadata.effective_date

                if date_a and date_b:
                    reason = (
                        f"{winner.citation_key} ({winner.metadata.effective_date}) "
                        f"is later than {loser.citation_key} ({loser.metadata.effective_date})"
                    )
                else:
                    reason = (
                        f"{winner.citation_key} has a known effective date "
                        f"while {loser.citation_key} does not"
                    )
                return winner, reason
            except ValueError:
                if conflict_type == "posterior":
                    raise

        raise ValueError(
            f"Cannot automatically resolve conflict between "
            f"{norm_a.citation_key} and {norm_b.citation_key}. "
            "Consider lex_specialis (requires content analysis)."
        )


__all__ = ["PriorityResolver"]
