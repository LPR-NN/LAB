"""AI Committee Member - Decision-making system for political organizations."""

__version__ = "0.1.0"


# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "CommitteeMember":
        from source.api.committee import CommitteeMember

        return CommitteeMember
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CommitteeMember", "__version__"]
