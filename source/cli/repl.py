"""Interactive REPL for AI Committee Member."""

from pathlib import Path
from typing import Literal

from source.audit.package import DecisionPackageBuilder
from source.audit.storage import AuditStorage
from source.cli.progress import ProgressTracker
from source.committee.commands import CommandHandler
from source.contracts.request import Request
from source.corpus.index import CorpusIndex
from source.corpus.loader import DocumentLoader
from source.providers.factory import ProviderFactory
from source.settings import get_settings

HELP_TEXT = """
AI Committee Member - Interactive Mode

Commands:
  /intake <json>     - Check request completeness
  /decide <json>     - Make a decision
  /cite              - Show citations from last decision
  /appeal <facts>    - Draft appeal with new facts
  /compare [ids]     - Compare with precedents
  /redteam           - Adversarial analysis of last decision

  /load <file>       - Load request from JSON file
  /corpus            - Show corpus statistics
  /search <query>    - Search corpus

  /help              - Show this help
  /quit              - Exit

JSON can be provided inline or loaded from a file with /load.
"""


class Session:
    """Session state for interactive mode."""

    def __init__(self):
        self.current_request: Request | None = None
        self.last_decision = None
        self.history: list[dict] = []


async def run_repl(
    corpus_path: Path,
    provider_type: Literal["openai", "anthropic", "openrouter"],
    api_key: str | None,
    model: str,
    audit_path: Path,
):
    """Run interactive REPL."""
    # Initial loading with progress
    tracker = ProgressTracker()
    tracker.add_step("corpus", "Loading corpus")
    tracker.add_step("provider", "Initializing provider")

    print()
    tracker.start_step("corpus")
    loader = DocumentLoader(corpus_path)
    documents = loader.load_corpus_sync()

    settings = get_settings()
    search_mode = settings.search_mode

    if search_mode == "tfidf":
        corpus = CorpusIndex(documents=documents, mode="tfidf")
        mode_info = "TF-IDF (keyword)"
    else:
        from source.corpus.embeddings import SentenceTransformerProvider

        provider_embed = SentenceTransformerProvider(name=settings.embedding_model)
        corpus = CorpusIndex(
            documents=documents,
            embedding_provider=provider_embed,
            mode=search_mode,
            alpha=settings.hybrid_alpha,
            cache_dir=settings.vector_cache_dir,
        )
        mode_info = f"{search_mode} ({settings.embedding_model})"

    tracker.add_detail(f"Loaded {len(documents)} documents")
    tracker.add_detail(f"Search mode: {mode_info}")
    stats = corpus.get_statistics()
    for doc_type, count in stats.get("documents_by_type", {}).items():
        tracker.add_detail(f"  {doc_type}: {count}")
    if "total_chunks" in stats:
        tracker.add_detail(f"  chunks: {stats['total_chunks']}")
    tracker.complete_step("corpus")

    tracker.start_step("provider")
    provider = ProviderFactory.create(provider_type, model, api_key=api_key)
    tracker.add_detail(f"Provider: {provider_type}")
    tracker.add_detail(f"Model: {model}")
    tracker.complete_step("provider")

    tracker.print_summary()

    handler = CommandHandler(provider, corpus)
    storage = AuditStorage(audit_path)

    session = Session()

    print("\nAI Committee Member ready.")
    print("Type /help for commands, /quit to exit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "/corpus":
            stats = corpus.get_statistics()
            print(f"\nTotal documents: {stats['total_documents']}")
            print(f"Active: {stats['active_documents']}")
            for doc_type, count in stats["documents_by_type"].items():
                print(f"  {doc_type}: {count}")
            continue

        if user_input.lower().startswith("/search "):
            query = user_input[8:].strip()
            results = corpus.search(query, top_k=5)
            print(f"\nFound {len(results)} results:")
            for doc, score in results:
                print(
                    f"  [{doc.citation_key}] {doc.metadata.title} (score: {score:.3f})"
                )
            continue

        if user_input.lower().startswith("/load "):
            file_path = Path(user_input[6:].strip())
            try:
                content = file_path.read_text()
                session.current_request = Request.model_validate_json(content)
                print(f"Loaded request: {session.current_request.query[:50]}...")
            except Exception as e:
                print(f"Error loading file: {e}")
            continue

        # Handle commands that need a request
        if user_input.lower().startswith("/intake"):
            if session.current_request is None:
                # Try to parse JSON from the command
                json_str = user_input[7:].strip()
                if json_str:
                    try:
                        session.current_request = Request.model_validate_json(json_str)
                    except Exception as e:
                        print(f"Invalid JSON: {e}")
                        continue
                else:
                    print("No request loaded. Use /load <file> or /intake <json>")
                    continue

            try:
                tracker = ProgressTracker()
                tracker.add_step("validate", "Validating request")
                tracker.add_step("check", "Checking completeness")

                tracker.start_step("validate")
                tracker.add_detail(f"Query: {session.current_request.query[:50]}...")
                tracker.complete_step("validate")

                tracker.start_step("check")
                result = await handler.intake(session.current_request)
                tracker.add_detail(f"Completeness: {result.completeness_score}%")
                tracker.complete_step("check")

                tracker.print_summary()

                print(f"\nReady: {'Yes' if result.ready else 'No'}")
                print(f"Completeness: {result.completeness_score}%")
                if result.missing_items:
                    print("Missing:")
                    for item in result.missing_items:
                        print(f"  - {item}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if user_input.lower().startswith("/decide"):
            if session.current_request is None:
                json_str = user_input[7:].strip()
                if json_str:
                    try:
                        session.current_request = Request.model_validate_json(json_str)
                    except Exception as e:
                        print(f"Invalid JSON: {e}")
                        continue
                else:
                    print("No request loaded. Use /load <file> first")
                    continue

            try:
                tracker = ProgressTracker()
                tracker.add_step("search", "Searching relevant norms")
                tracker.add_step("decide", "Formulating decision")
                tracker.add_step("audit", "Saving audit package")

                tracker.start_step("search")
                search_results = corpus.search(session.current_request.query, top_k=5)
                for doc, score in search_results[:3]:
                    tracker.add_detail(
                        f"Found: [{doc.citation_key}] (relevance: {score:.2f})"
                    )
                tracker.complete_step("search")

                tracker.start_step("decide")
                tracker.add_detail("Analyzing applicable rules...")
                decision = await handler.decide(session.current_request)
                session.last_decision = decision
                tracker.add_detail(f"Verdict: {decision.verdict}")
                tracker.add_detail(f"Citations: {len(decision.citations)}")
                tracker.complete_step("decide")

                # Save audit
                tracker.start_step("audit")
                builder = DecisionPackageBuilder(
                    corpus,
                    model_id=model,
                    provider=provider_type,
                )
                package = builder.build(session.current_request, decision)
                storage.save(package)
                tracker.add_detail(f"Package: {package.package_id}")
                tracker.complete_step("audit")

                tracker.print_summary()

                print(f"\nVerdict: {decision.verdict}")
                print(f"Summary: {decision.verdict_summary}")
                print(f"\nCitations: {len(decision.citations)}")
                print(f"Reasoning steps: {len(decision.reasoning)}")

            except Exception as e:
                print(f"Error: {e}")
            continue

        if user_input.lower() == "/cite":
            if session.last_decision is None:
                print("No decision available. Run /decide first.")
                continue

            print("\nCitations:")
            for c in session.last_decision.citations:
                print(f"  [{c.doc_id}, {c.section}]")
                print(f'    "{c.quoted_text[:100]}..."')
            continue

        if user_input.lower().startswith("/redteam"):
            if session.last_decision is None:
                print("No decision to analyze. Run /decide first.")
                continue

            try:
                tracker = ProgressTracker()
                tracker.add_step("analyze", "Analyzing for bias")
                tracker.add_step("check", "Checking consistency")
                tracker.add_step("recommend", "Generating recommendations")

                tracker.start_step("analyze")
                tracker.add_detail(f"Decision verdict: {session.last_decision.verdict}")
                result = await handler.redteam(
                    session.current_request, session.last_decision
                )
                tracker.complete_step("analyze")

                tracker.start_step("check")
                tracker.add_detail(f"Issues found: {len(result.issues_found)}")
                tracker.complete_step("check")

                tracker.start_step("recommend")
                tracker.add_detail(f"Recommendations: {len(result.recommendations)}")
                tracker.complete_step("recommend")

                tracker.print_summary()

                print(f"\nBias Assessment: {result.bias_assessment}")
                if result.issues_found:
                    print("Issues:")
                    for issue in result.issues_found:
                        print(f"  - {issue}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # Unknown command or chat
        print("Unknown command. Type /help for available commands.")


__all__ = ["run_repl"]
