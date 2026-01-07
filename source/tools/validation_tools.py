"""Tools for validating requests and checking decision readiness."""

import json
from funcai import tool
from funcai.agents.tool import Tool

from source.contracts.request import Request
from source.contracts.tool_results import (
    ValidationResult,
    DecisionReadinessResult,
    ReadinessChecklist,
)


@tool("Validate a request and identify missing or incomplete fields")
def validate_request(request_json: str) -> ValidationResult:
    """
    Validate a request for completeness.

    Args:
        request_json: JSON string of the request

    Returns:
        Validation result with missing fields and suggestions
    """
    try:
        data = json.loads(request_json)
    except json.JSONDecodeError as e:
        return ValidationResult(
            valid=False,
            error=f"Invalid JSON: {e}",
            missing_fields=[],
            suggestions=["Fix JSON syntax errors"],
        )

    missing_fields: list[str] = []
    suggestions: list[str] = []
    warnings: list[str] = []

    # Check required fields
    if not data.get("query"):
        missing_fields.append("query")
        suggestions.append("Provide a clear description of what needs to be decided")

    if not data.get("case_type"):
        missing_fields.append("case_type")
        suggestions.append(
            "Specify case type: ethics, discipline, arbitration, procedure_interpretation, moderation, or conflict"
        )

    if not data.get("requested_remedy"):
        missing_fields.append("requested_remedy")
        suggestions.append(
            "Specify what outcome is requested: decision, sanction, clarification, or refusal"
        )

    # Check recommended fields
    if not data.get("parties") or len(data.get("parties", [])) == 0:
        warnings.append("No parties specified - consider identifying involved parties")

    if not data.get("attachments") or len(data.get("attachments", [])) == 0:
        warnings.append(
            "No attachments - decisions should be based on documented evidence"
        )

    if not data.get("time_context"):
        warnings.append("No time context - deadlines or event dates may be relevant")

    # Validate attachments if present
    for i, attachment in enumerate(data.get("attachments", [])):
        if not attachment.get("relevance_note"):
            suggestions.append(
                f"Attachment {i + 1}: Add a relevance_note explaining why this is important"
            )

    # Try to validate as Pydantic model
    try:
        Request.model_validate(data)
        model_valid = True
    except Exception as e:
        model_valid = False
        suggestions.append(f"Model validation error: {e}")

    return ValidationResult(
        valid=len(missing_fields) == 0 and model_valid,
        missing_fields=missing_fields,
        warnings=warnings,
        suggestions=suggestions,
        completeness_score=max(0, 100 - len(missing_fields) * 20 - len(warnings) * 5),
    )


@tool("Check if there is enough information to make a decision")
def check_decision_readiness(
    facts: str,
    applicable_norms: str,
) -> DecisionReadinessResult:
    """
    Check if enough information exists to proceed with a decision.

    Args:
        facts: Comma-separated list of established facts
        applicable_norms: Comma-separated list of applicable norm IDs

    Returns:
        Readiness assessment with gaps identified
    """
    fact_list = [f.strip() for f in facts.split(",") if f.strip()]
    norm_list = [n.strip() for n in applicable_norms.split(",") if n.strip()]

    gaps: list[str] = []
    ready = True

    # Check minimum requirements
    if len(fact_list) < 1:
        gaps.append("No facts established - cannot make a decision without facts")
        ready = False

    if len(norm_list) < 1:
        gaps.append("No applicable norms identified - decision must be based on norms")
        ready = False

    # Quality checks
    if len(fact_list) == 1:
        gaps.append(
            "Only one fact established - consider if more facts should be established"
        )

    if len(norm_list) == 1:
        gaps.append("Only one norm identified - verify no other norms are applicable")

    return DecisionReadinessResult(
        ready=ready,
        facts_count=len(fact_list),
        norms_count=len(norm_list),
        gaps=gaps,
        recommendation="Proceed with decision"
        if ready
        else "Gather more information before deciding",
        checklist=ReadinessChecklist(
            facts_established=len(fact_list) > 0,
            norms_identified=len(norm_list) > 0,
            no_critical_gaps=len([g for g in gaps if "cannot" in g.lower()]) == 0,
        ),
    )


def make_validation_tools() -> list[Tool]:
    """
    Create validation tools.

    Returns:
        List of funcai Tool objects
    """
    return [
        validate_request,
        check_decision_readiness,
    ]


__all__ = [
    "validate_request",
    "check_decision_readiness",
    "make_validation_tools",
]
