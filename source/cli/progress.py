"""Progress tracking and display for CLI."""

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepInfo:
    """Information about a pipeline step."""

    name: str
    description: str
    status: StepStatus = StepStatus.PENDING
    start_time: float | None = None
    end_time: float | None = None
    details: list[str] = field(default_factory=list)

    @property
    def elapsed(self) -> float | None:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def elapsed_str(self) -> str:
        """Get formatted elapsed time."""
        elapsed = self.elapsed
        if elapsed is None:
            return ""
        return f"{elapsed:.1f}s"


class ProgressTracker:
    """
    Tracks and displays progress of multi-step operations.

    Usage:
        tracker = ProgressTracker()
        tracker.add_step("intake", "Checking request completeness")
        tracker.add_step("decide", "Making decision")

        tracker.start_step("intake")
        tracker.add_detail("Found 3 relevant norms")
        tracker.complete_step("intake")
    """

    SYMBOLS = {
        StepStatus.PENDING: "○",
        StepStatus.RUNNING: "◐",
        StepStatus.COMPLETED: "✓",
        StepStatus.FAILED: "✗",
        StepStatus.SKIPPED: "⊘",
    }

    COLORS = {
        StepStatus.PENDING: "\033[90m",  # Gray
        StepStatus.RUNNING: "\033[33m",  # Yellow
        StepStatus.COMPLETED: "\033[32m",  # Green
        StepStatus.FAILED: "\033[31m",  # Red
        StepStatus.SKIPPED: "\033[90m",  # Gray
    }

    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    def __init__(
        self, use_color: bool = True, output: Callable[[str], None] | None = None
    ):
        self.steps: dict[str, StepInfo] = {}
        self.step_order: list[str] = []
        self.use_color = use_color and sys.stdout.isatty()
        self.output = output or (lambda s: print(s, flush=True))
        self._last_detail_count = 0

    def add_step(self, step_id: str, description: str) -> None:
        """Add a step to track."""
        self.steps[step_id] = StepInfo(name=step_id, description=description)
        self.step_order.append(step_id)

    def start_step(self, step_id: str) -> None:
        """Mark a step as started."""
        if step_id not in self.steps:
            return

        step = self.steps[step_id]
        step.status = StepStatus.RUNNING
        step.start_time = time.time()
        step.details = []
        self._last_detail_count = 0
        self._print_step(step_id)

    def add_detail(self, detail: str, step_id: str | None = None) -> None:
        """Add a detail to current or specified step."""
        target_id = step_id
        if target_id is None:
            # Find running step
            for sid in self.step_order:
                if self.steps[sid].status == StepStatus.RUNNING:
                    target_id = sid
                    break

        if target_id and target_id in self.steps:
            self.steps[target_id].details.append(detail)
            self._print_detail(detail)

    def complete_step(self, step_id: str, success: bool = True) -> None:
        """Mark a step as completed."""
        if step_id not in self.steps:
            return

        step = self.steps[step_id]
        step.status = StepStatus.COMPLETED if success else StepStatus.FAILED
        step.end_time = time.time()
        self._print_step_complete(step_id)

    def skip_step(self, step_id: str, reason: str | None = None) -> None:
        """Mark a step as skipped."""
        if step_id not in self.steps:
            return

        step = self.steps[step_id]
        step.status = StepStatus.SKIPPED
        if reason:
            step.details.append(f"Skipped: {reason}")
        self._print_step(step_id)

    def _color(self, text: str, status: StepStatus) -> str:
        """Apply color to text based on status."""
        if not self.use_color:
            return text
        return f"{self.COLORS[status]}{text}{self.RESET}"

    def _dim(self, text: str) -> str:
        """Apply dim style to text."""
        if not self.use_color:
            return text
        return f"{self.DIM}{text}{self.RESET}"

    def _bold(self, text: str) -> str:
        """Apply bold style to text."""
        if not self.use_color:
            return text
        return f"{self.BOLD}{text}{self.RESET}"

    def _get_step_number(self, step_id: str) -> str:
        """Get step number as string."""
        idx = self.step_order.index(step_id) + 1
        total = len(self.step_order)
        return f"[{idx}/{total}]"

    def _print_step(self, step_id: str) -> None:
        """Print step status line."""
        step = self.steps[step_id]
        symbol = self.SYMBOLS[step.status]
        num = self._get_step_number(step_id)

        line = f"{num} {self._color(symbol, step.status)} {step.description}"

        if step.status == StepStatus.RUNNING:
            line += self._dim(" ...")
        elif step.elapsed_str:
            line += self._dim(f" ({step.elapsed_str})")

        self.output(line)

    def _print_step_complete(self, step_id: str) -> None:
        """Print completion status for a step."""
        step = self.steps[step_id]
        symbol = self.SYMBOLS[step.status]
        num = self._get_step_number(step_id)

        line = f"{num} {self._color(symbol, step.status)} {step.description} {self._dim(f'({step.elapsed_str})')}"
        self.output(line)

    def _print_detail(self, detail: str) -> None:
        """Print a detail line."""
        self.output(f"    → {detail}")

    def print_summary(self) -> None:
        """Print final summary."""
        completed = sum(
            1 for s in self.steps.values() if s.status == StepStatus.COMPLETED
        )
        failed = sum(1 for s in self.steps.values() if s.status == StepStatus.FAILED)
        total = len(self.steps)

        total_time = sum(
            s.elapsed or 0 for s in self.steps.values() if s.elapsed is not None
        )

        if failed > 0:
            status = self._color(f"Completed with {failed} error(s)", StepStatus.FAILED)
        else:
            status = self._color("Completed successfully", StepStatus.COMPLETED)

        self.output("")
        self.output(
            f"{status} {self._dim(f'({completed}/{total} steps, {total_time:.1f}s total)')}"
        )


class ProgressCallback:
    """
    Callback interface for progress updates.

    Pass to CommandHandler to receive progress events.
    """

    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker

    def on_step_start(self, step_id: str, description: str) -> None:
        """Called when a step starts."""
        if step_id not in self.tracker.steps:
            self.tracker.add_step(step_id, description)
        self.tracker.start_step(step_id)

    def on_step_detail(self, detail: str) -> None:
        """Called when there's a detail to report."""
        self.tracker.add_detail(detail)

    def on_step_complete(self, step_id: str, success: bool = True) -> None:
        """Called when a step completes."""
        self.tracker.complete_step(step_id, success)

    def on_tool_call(self, tool_name: str, args_summary: str) -> None:
        """Called when a tool is invoked."""
        self.tracker.add_detail(f"Tool: {tool_name} - {args_summary}")

    def on_document_found(self, doc_id: str, relevance: str) -> None:
        """Called when a relevant document is found."""
        self.tracker.add_detail(f"Found: [{doc_id}] {relevance}")


def create_pipeline_tracker(
    steps: list[tuple[str, str]],
) -> tuple[ProgressTracker, ProgressCallback]:
    """
    Create a tracker with predefined steps.

    Args:
        steps: List of (step_id, description) tuples

    Returns:
        Tuple of (tracker, callback)
    """
    tracker = ProgressTracker()
    for step_id, description in steps:
        tracker.add_step(step_id, description)

    return tracker, ProgressCallback(tracker)


# Predefined pipeline configurations
DECIDE_PIPELINE = [
    ("search", "Searching relevant norms and precedents"),
    ("analyze", "Analyzing applicable rules"),
    ("reason", "Constructing legal reasoning"),
    ("decide", "Formulating decision"),
]

INTAKE_PIPELINE = [
    ("validate", "Validating request structure"),
    ("check", "Checking completeness"),
]

REDTEAM_PIPELINE = [
    ("analyze", "Analyzing decision for bias"),
    ("check", "Checking consistency"),
    ("recommend", "Generating recommendations"),
]

COMPARE_PIPELINE = [
    ("search", "Finding precedents"),
    ("compare", "Comparing cases"),
    ("synthesize", "Synthesizing guidance"),
]


__all__ = [
    "ProgressTracker",
    "ProgressCallback",
    "StepStatus",
    "StepInfo",
    "create_pipeline_tracker",
    "DECIDE_PIPELINE",
    "INTAKE_PIPELINE",
    "REDTEAM_PIPELINE",
    "COMPARE_PIPELINE",
]
