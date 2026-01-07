"""Committee Agent - core decision-making agent."""

from dataclasses import dataclass, field
from typing import Any

from kungfu import LazyCoroResult, Option, Some

from funcai import Dialogue, agent, message
from funcai.agents.abc import ABCAgent, AgentError, AgentResponse
from funcai.agents.tool import Tool
from funcai.core.provider import ABCAIProvider
from source.committee.prompts import (
    COMMITTEE_SYSTEM_PROMPT,
    format_jurisdiction_rules,
)
from source.constants import DEFAULT_TOKEN_BUDGET
from source.contracts.decision import Decision
from source.contracts.request import Request
from source.corpus.context import reset_budget, reset_search_cache, set_corpus
from source.corpus.index import CorpusIndex
from source.tools.attachment_tools import make_attachment_tools
from source.tools.corpus_tools import make_corpus_tools
from source.tools.precedent_tools import make_precedent_tools
from source.tools.reasoning_tools import make_reasoning_tools
from source.tools.validation_tools import make_validation_tools


def build_tools(
    corpus: CorpusIndex,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> list[Tool]:
    """
    Build all tools for the committee agent.

    Args:
        corpus: Document corpus index
        token_budget: Token budget for document retrieval
    """
    set_corpus(corpus)
    reset_budget(token_budget)
    reset_search_cache()  # Clear cache for new session

    tools: list[Tool] = []
    tools.extend(make_attachment_tools())
    tools.extend(make_corpus_tools())
    tools.extend(make_precedent_tools())
    tools.extend(make_reasoning_tools())
    tools.extend(make_validation_tools())

    return tools


def build_request_message(request: Request) -> str:
    """Build a message from the request."""
    parts = [
        "## Case Request",
        "",
        f"**Query**: {request.query}",
        f"**Case Type**: {request.case_type}",
        f"**Requested Remedy**: {request.requested_remedy}",
    ]

    if request.org_profile:
        parts.append(f"**Organization**: {request.org_profile}")

    if request.time_context:
        parts.append(f"**Time Context**: {request.time_context}")

    if request.parties:
        parts.append("\n### Parties Involved")
        for party in request.parties:
            parts.append(f"- **{party.name}** ({party.role})")
            if party.description:
                parts.append(f"  {party.description}")

    if request.attachments:
        parts.append("\n### Attachments")
        parts.append(
            "Use the `read_attachment` tool to read the content of these files:"
        )
        for i, att in enumerate(request.attachments, 1):
            parts.append(f"{i}. **{att.file_path}** ({att.attachment_type})")
            parts.append(f"   Relevance: {att.relevance_note}")

    if request.additional_context:
        parts.append("\n### Additional Context")
        parts.append(request.additional_context)

    return "\n".join(parts)


@dataclass
class CommitteeAgent[E](ABCAgent[E, Decision]):
    """
    AI Committee Member agent for making organizational decisions.

    Extends funcai's ABCAgent with domain-specific logic for:
    - Legal-style reasoning
    - Norm conflict resolution
    - Precedent analysis
    - Proper citation generation
    """

    provider: ABCAIProvider[E]
    corpus: CorpusIndex
    tools: list[Tool] = field(default_factory=list)
    max_steps: int = 20
    schema: Option[type[Decision]] = field(default_factory=lambda: Some(Decision))

    def __post_init__(self) -> None:
        """Initialize tools if not provided."""
        if not self.tools:
            self.tools = build_tools(self.corpus, provider=self.provider)

    def run(
        self, dialogue: Dialogue
    ) -> LazyCoroResult[AgentResponse[Decision] | AgentResponse[None], AgentError]:
        """
        Run the committee agent on the dialogue.

        Uses funcai's agent() function with committee-specific tools.
        """
        return agent(
            dialogue,
            self.provider,
            self.tools,
            max_steps=self.max_steps,
            schema=Decision,
        )

    @classmethod
    def create(
        cls,
        provider: ABCAIProvider[E],
        corpus: CorpusIndex,
        max_steps: int = 20,
    ) -> "CommitteeAgent[E]":
        """
        Factory method to create a committee agent.

        Args:
            provider: Main LLM provider for reasoning
            corpus: Document corpus
            max_steps: Maximum agent steps
        """
        return cls(
            provider=provider,
            corpus=corpus,
            tools=build_tools(corpus),
            max_steps=max_steps,
        )


def create_committee_dialogue(
    request: Request,
    instruction: str = "",
) -> Dialogue:
    """
    Create a dialogue for the committee agent.

    Args:
        request: The case request
        instruction: Additional instruction for the agent

    Returns:
        Configured Dialogue with system prompt and request
    """
    # Format jurisdiction rules
    jurisdiction_rules = format_jurisdiction_rules(request.jurisdiction)

    # Build system prompt
    system_prompt = COMMITTEE_SYSTEM_PROMPT.format(
        jurisdiction_rules=jurisdiction_rules
    )

    # Build request message
    request_message = build_request_message(request)

    if instruction:
        request_message = f"{instruction}\n\n{request_message}"

    return Dialogue(
        [
            message.system(text=system_prompt),
            message.user(text=request_message),
        ]
    )


async def run_committee_decision(
    request: Request,
    provider: ABCAIProvider[Any],
    corpus: CorpusIndex,
    instruction: str = "",
    max_steps: int = 20,
) -> AgentResponse[Decision] | AgentError:
    """
    Run a complete committee decision process.

    Args:
        request: The case request
        provider: LLM provider
        corpus: Document corpus
        instruction: Additional instruction
        max_steps: Maximum agent steps

    Returns:
        Decision response or error
    """
    dialogue = create_committee_dialogue(request, instruction)
    committee = CommitteeAgent.create(provider, corpus, max_steps)

    result = await committee.run(dialogue)

    match result:
        case AgentResponse() as response:
            return response
        case AgentError() as error:
            return error
        case _:
            return result.unwrap()


__all__ = [
    "CommitteeAgent",
    "create_committee_dialogue",
    "run_committee_decision",
    "build_tools",
    "build_request_message",
]
