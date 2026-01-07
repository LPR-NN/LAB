"""Command handlers for the AI Committee Member."""

import datetime
from dataclasses import dataclass
from typing import Any, Callable

from kungfu import Error, Ok, Result
from pydantic import BaseModel

from funcai.core import message
from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider
from source.committee.agent import (
    build_tools,
    create_committee_dialogue,
)
from source.committee.prompts import (
    APPEAL_PROMPT,
    CITE_PROMPT,
    COMMITTEE_SYSTEM_PROMPT,
    COMPARE_PROMPT,
    DECIDE_PROMPT,
    INTAKE_PROMPT,
    REDTEAM_PROMPT,
    format_jurisdiction_rules,
)
from source.contracts.audit import ToolCallHistory
from source.contracts.decision import Decision
from source.contracts.request import Request
from source.contracts.tool_results import CommandError
from source.corpus.index import CorpusIndex
from source.llm import llm_agent_call
from source.tools.observer import ToolObserver, create_progress_observer


class IntakeResult(BaseModel):
    """Result of /intake command."""

    ready: bool
    missing_items: list[str]
    suggestions: list[str]
    completeness_score: int


class CiteResult(BaseModel):
    """Result of /cite command."""

    citations: list[dict]
    total_citations: int


class AppealDraft(BaseModel):
    """Result of /appeal command."""

    recommendation: str
    affected_findings: list[str]
    reasoning: str


class ComparisonResult(BaseModel):
    """Result of /compare command."""

    comparisons: list[dict]
    overall_guidance: str


class RedTeamResult(BaseModel):
    """Result of /redteam command."""

    issues_found: list[dict]
    bias_assessment: str
    recommendations: list[str]


class DecisionResult(BaseModel):
    """Result of /decide command with tool call history."""

    decision: Decision
    tool_calls: ToolCallHistory


@dataclass
class CommandResult:
    """Generic command result."""

    command: str
    success: bool
    data: Any
    error: str | None = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now(datetime.timezone.utc)


class CommandHandler:
    """
    Handler for committee commands.

    Supports:
    - /intake: Check request completeness
    - /decide: Make a decision
    - /cite: Extract citations from decision
    - /appeal: Draft appeal with new facts
    - /compare: Compare with precedents
    - /redteam: Self-check for bias
    """

    def __init__(
        self,
        provider: ABCAIProvider[Any],
        corpus: CorpusIndex,
        max_steps: int = 20,
        on_tool_call: Callable[[str], None] | None = None,
        debug: bool = False,
    ):
        self.provider = provider
        self.corpus = corpus
        self.max_steps = max_steps
        self.on_tool_call = on_tool_call
        self.debug = debug
        self._base_tools = build_tools(corpus)

    async def handle(
        self,
        command: str,
        request: Request,
        context: dict[str, Any] | None = None,
    ) -> CommandResult:
        """
        Handle a command.

        Args:
            command: Command name (with or without /)
            request: The case request
            context: Additional context (e.g., previous decision for /cite)

        Returns:
            CommandResult with outcome
        """
        # Normalize command
        cmd = command.lower().strip().lstrip("/")

        match cmd:
            case "intake":
                result = await self.intake(request)
                match result:
                    case Ok(data):
                        return CommandResult(command=cmd, success=True, data=data)
                    case Error(e):
                        return CommandResult(
                            command=cmd, success=False, data=None, error=str(e)
                        )

            case "decide":
                result = await self.decide(request)
                match result:
                    case Ok(data):
                        return CommandResult(command=cmd, success=True, data=data)
                    case Error(e):
                        return CommandResult(
                            command=cmd, success=False, data=None, error=str(e)
                        )

            case "cite":
                decision = context.get("decision") if context else None
                result = await self.cite(request, decision)
                match result:
                    case Ok(data):
                        return CommandResult(command=cmd, success=True, data=data)
                    case Error(e):
                        return CommandResult(
                            command=cmd, success=False, data=None, error=str(e)
                        )

            case "appeal":
                new_facts = context.get("new_facts", []) if context else []
                result = await self.appeal(request, new_facts)
                match result:
                    case Ok(data):
                        return CommandResult(command=cmd, success=True, data=data)
                    case Error(e):
                        return CommandResult(
                            command=cmd, success=False, data=None, error=str(e)
                        )

            case "compare":
                precedent_ids = context.get("precedent_ids", []) if context else []
                result = await self.compare(request, precedent_ids)
                match result:
                    case Ok(data):
                        return CommandResult(command=cmd, success=True, data=data)
                    case Error(e):
                        return CommandResult(
                            command=cmd, success=False, data=None, error=str(e)
                        )

            case "redteam":
                decision = context.get("decision") if context else None
                result = await self.redteam(request, decision)
                match result:
                    case Ok(data):
                        return CommandResult(command=cmd, success=True, data=data)
                    case Error(e):
                        return CommandResult(
                            command=cmd, success=False, data=None, error=str(e)
                        )

            case _:
                return CommandResult(
                    command=cmd,
                    success=False,
                    data=None,
                    error=f"Unknown command: {cmd}. Available: intake, decide, cite, appeal, compare, redteam",
                )

    async def intake(self, request: Request) -> Result[IntakeResult, CommandError]:
        """Check if request is complete and ready for decision."""
        jurisdiction_rules = format_jurisdiction_rules(request.jurisdiction)
        system_prompt = COMMITTEE_SYSTEM_PROMPT.format(
            jurisdiction_rules=jurisdiction_rules
        )

        dialogue = Dialogue(
            [
                message.system(text=system_prompt),
                message.user(
                    text=f"{INTAKE_PROMPT}\n\nRequest:\n{request.model_dump_json(indent=2)}"
                ),
            ]
        )

        if self.on_tool_call:
            observer = create_progress_observer(self.on_tool_call)
            tools = observer.wrap_tools(self._base_tools.copy())
        else:
            tools = self._base_tools

        result = await llm_agent_call(
            dialogue,
            self.provider,
            tools,
            max_steps=10,
            schema=IntakeResult,
            retry_times=3,
        )

        match result:
            case Ok(response):
                if response.parsed:
                    return Ok(response.parsed)
                return Error(
                    CommandError(message="No intake result produced", command="intake")
                )
            case Error(e):
                return Error(CommandError(message=str(e), command="intake"))

    async def decide(self, request: Request) -> Result[DecisionResult, CommandError]:
        """Make a decision on the request."""
        dialogue = create_committee_dialogue(request, DECIDE_PROMPT)

        if self.on_tool_call:
            observer = create_progress_observer(self.on_tool_call, debug=self.debug)
        else:
            observer = ToolObserver()

        tools = observer.wrap_tools(self._base_tools.copy())

        result = await llm_agent_call(
            dialogue,
            self.provider,
            tools,
            max_steps=self.max_steps,
            schema=Decision,
            timeout_seconds=6000,  # 100 min for complex decisions
            retry_times=3,
        )

        tool_history = observer.get_history()

        match result:
            case Ok(response):
                if response.parsed:
                    return Ok(
                        DecisionResult(
                            decision=response.parsed,
                            tool_calls=ToolCallHistory(
                                calls=[c.model_dump() for c in tool_history.calls],
                                total_calls=tool_history.total_calls,
                                total_duration_ms=tool_history.total_duration_ms,
                            ),
                        )
                    )
                return Error(
                    CommandError(message="No decision produced", command="decide")
                )
            case Error(e):
                return Error(CommandError(message=str(e), command="decide"))

    async def cite(
        self,
        request: Request,
        decision: Decision | None = None,
    ) -> Result[CiteResult, CommandError]:
        """Extract and verify citations from a decision."""
        jurisdiction_rules = format_jurisdiction_rules(request.jurisdiction)
        system_prompt = COMMITTEE_SYSTEM_PROMPT.format(
            jurisdiction_rules=jurisdiction_rules
        )

        decision_text = (
            decision.model_dump_json(indent=2) if decision else "No decision provided"
        )

        dialogue = Dialogue(
            [
                message.system(text=system_prompt),
                message.user(text=f"{CITE_PROMPT}\n\nDecision:\n{decision_text}"),
            ]
        )

        if self.on_tool_call:
            observer = create_progress_observer(self.on_tool_call)
            tools = observer.wrap_tools(self._base_tools.copy())
        else:
            tools = self._base_tools

        result = await llm_agent_call(
            dialogue,
            self.provider,
            tools,
            max_steps=10,
            schema=CiteResult,
            retry_times=3,
        )

        match result:
            case Ok(response):
                if response.parsed:
                    return Ok(response.parsed)
                return Error(
                    CommandError(message="No citation result produced", command="cite")
                )
            case Error(e):
                return Error(CommandError(message=str(e), command="cite"))

    async def appeal(
        self,
        request: Request,
        new_facts: list[str],
    ) -> Result[AppealDraft, CommandError]:
        """Draft an appeal based on new facts."""
        jurisdiction_rules = format_jurisdiction_rules(request.jurisdiction)
        system_prompt = COMMITTEE_SYSTEM_PROMPT.format(
            jurisdiction_rules=jurisdiction_rules
        )

        new_facts_text = (
            "\n".join(f"- {fact}" for fact in new_facts)
            if new_facts
            else "No new facts provided"
        )

        dialogue = Dialogue(
            [
                message.system(text=system_prompt),
                message.user(
                    text=f"{APPEAL_PROMPT}\n\nOriginal Request:\n{request.model_dump_json(indent=2)}\n\nNew Facts:\n{new_facts_text}"
                ),
            ]
        )

        if self.on_tool_call:
            observer = create_progress_observer(self.on_tool_call)
            tools = observer.wrap_tools(self._base_tools.copy())
        else:
            tools = self._base_tools

        result = await llm_agent_call(
            dialogue,
            self.provider,
            tools,
            max_steps=15,
            schema=AppealDraft,
            retry_times=3,
        )

        match result:
            case Ok(response):
                if response.parsed:
                    return Ok(response.parsed)
                return Error(
                    CommandError(message="No appeal draft produced", command="appeal")
                )
            case Error(e):
                return Error(CommandError(message=str(e), command="appeal"))

    async def compare(
        self,
        request: Request,
        precedent_ids: list[str],
    ) -> Result[ComparisonResult, CommandError]:
        """Compare case with specified precedents."""
        jurisdiction_rules = format_jurisdiction_rules(request.jurisdiction)
        system_prompt = COMMITTEE_SYSTEM_PROMPT.format(
            jurisdiction_rules=jurisdiction_rules
        )

        precedents_text = (
            ", ".join(precedent_ids) if precedent_ids else "Find relevant precedents"
        )

        dialogue = Dialogue(
            [
                message.system(text=system_prompt),
                message.user(
                    text=f"{COMPARE_PROMPT}\n\nRequest:\n{request.model_dump_json(indent=2)}\n\nPrecedents to compare: {precedents_text}"
                ),
            ]
        )

        if self.on_tool_call:
            observer = create_progress_observer(self.on_tool_call)
            tools = observer.wrap_tools(self._base_tools.copy())
        else:
            tools = self._base_tools

        result = await llm_agent_call(
            dialogue,
            self.provider,
            tools,
            max_steps=15,
            schema=ComparisonResult,
            retry_times=3,
        )

        match result:
            case Ok(response):
                if response.parsed:
                    return Ok(response.parsed)
                return Error(
                    CommandError(
                        message="No comparison result produced", command="compare"
                    )
                )
            case Error(e):
                return Error(CommandError(message=str(e), command="compare"))

    async def redteam(
        self,
        request: Request,
        decision: Decision | None = None,
    ) -> Result[RedTeamResult, CommandError]:
        """Perform adversarial analysis of a decision."""
        jurisdiction_rules = format_jurisdiction_rules(request.jurisdiction)
        system_prompt = COMMITTEE_SYSTEM_PROMPT.format(
            jurisdiction_rules=jurisdiction_rules
        )

        decision_text = (
            decision.model_dump_json(indent=2)
            if decision
            else "No decision provided - analyze the request"
        )

        dialogue = Dialogue(
            [
                message.system(text=system_prompt),
                message.user(
                    text=f"{REDTEAM_PROMPT}\n\nRequest:\n{request.model_dump_json(indent=2)}\n\nDecision:\n{decision_text}"
                ),
            ]
        )

        if self.on_tool_call:
            observer = create_progress_observer(self.on_tool_call)
            tools = observer.wrap_tools(self._base_tools.copy())
        else:
            tools = self._base_tools

        result = await llm_agent_call(
            dialogue,
            self.provider,
            tools,
            max_steps=15,
            schema=RedTeamResult,
            retry_times=3,
        )

        match result:
            case Ok(response):
                if response.parsed:
                    return Ok(response.parsed)
                return Error(
                    CommandError(
                        message="No red team result produced", command="redteam"
                    )
                )
            case Error(e):
                return Error(CommandError(message=str(e), command="redteam"))


__all__ = [
    "CommandHandler",
    "CommandResult",
    "DecisionResult",
    "IntakeResult",
    "CiteResult",
    "AppealDraft",
    "ComparisonResult",
    "RedTeamResult",
]
