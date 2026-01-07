"""Tools for reading case attachments."""

from pathlib import Path

from funcai import tool
from funcai.agents.tool import Tool

from source.contracts.tool_results import AttachmentContent, BudgetExceeded
from source.corpus.context import get_budget


@tool("Read the full content of an attachment file")
def read_attachment(file_path: str) -> AttachmentContent | BudgetExceeded:
    """
    Read the content of an attachment file.

    Use this tool to read evidence, statements, and other documents
    attached to the case request.

    Args:
        file_path: Path to the attachment file (as specified in the request)

    Returns:
        AttachmentContent with the file content, or BudgetExceeded if token limit reached
    """
    path = Path(file_path)

    if not path.exists():
        return AttachmentContent(
            file_path=file_path,
            content=None,
            error=f"File not found: {file_path}",
        )

    if not path.is_file():
        return AttachmentContent(
            file_path=file_path,
            content=None,
            error=f"Not a file: {file_path}",
        )

    try:
        content = path.read_text(encoding="utf-8")

        budget = get_budget()
        ok, tokens = budget.consume(content)

        if not ok:
            return BudgetExceeded(
                requested_tokens=tokens,
                remaining_tokens=budget.remaining(),
            )

        return AttachmentContent(
            file_path=file_path,
            content=content,
            error=None,
        )
    except UnicodeDecodeError:
        return AttachmentContent(
            file_path=file_path,
            content=None,
            error=f"Cannot read file (not UTF-8 text): {file_path}",
        )
    except Exception as e:
        return AttachmentContent(
            file_path=file_path,
            content=None,
            error=f"Error reading file: {e}",
        )


def make_attachment_tools() -> list[Tool]:
    """Create attachment tools."""
    return [read_attachment]


__all__ = ["read_attachment", "make_attachment_tools"]
