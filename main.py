"""
AI Committee Member - Entry point.

Usage:
    python main.py                  # Run CLI help
    python main.py chat             # Interactive mode
    python main.py decide request.json  # Make a decision
"""

from source.cli.main import app


def main():
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
