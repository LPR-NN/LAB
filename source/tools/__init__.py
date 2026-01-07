"""funcai tools for the AI Committee Member agent."""

from source.tools.attachment_tools import (
    make_attachment_tools,
    read_attachment,
)
from source.tools.corpus_tools import (
    cite_fragment,
    get_document,
    make_corpus_tools,
    search_corpus,
    search_document_section,
)
from source.tools.observer import (
    ToolCallHistory,
    ToolCallRecord,
    ToolObserver,
    create_progress_observer,
)
from source.tools.precedent_tools import (
    compare_with_precedent,
    find_precedents,
    make_precedent_tools,
)
from source.tools.reasoning_tools import (
    apply_priority_rules,
    check_norm_conflicts,
    make_reasoning_tools,
    verify_reasoning_chain,
)
from source.tools.validation_tools import (
    check_decision_readiness,
    make_validation_tools,
    validate_request,
)

__all__ = [
    # Attachment tools
    "read_attachment",
    "make_attachment_tools",
    # Corpus tools
    "search_corpus",
    "get_document",
    "search_document_section",
    "cite_fragment",
    "make_corpus_tools",
    # Observer
    "ToolObserver",
    "ToolCallRecord",
    "ToolCallHistory",
    "create_progress_observer",
    # Precedent tools
    "find_precedents",
    "compare_with_precedent",
    "make_precedent_tools",
    # Reasoning tools
    "check_norm_conflicts",
    "apply_priority_rules",
    "verify_reasoning_chain",
    "make_reasoning_tools",
    # Validation tools
    "validate_request",
    "check_decision_readiness",
    "make_validation_tools",
]
