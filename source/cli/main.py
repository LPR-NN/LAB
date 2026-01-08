"""CLI entry point for AI Committee Member."""

import asyncio
import sys
from pathlib import Path
from typing import Literal

import click
from kungfu import Error, Ok

from source.audit.package import DecisionPackageBuilder
from source.audit.storage import AuditStorage
from source.cli.progress import ProgressTracker
from source.committee.commands import CommandHandler
from source.contracts.request import Request
from source.corpus.index import CorpusIndex
from source.corpus.loader import DocumentLoader
from source.providers.factory import ProviderFactory
from source.settings import SearchMode, get_settings


def load_corpus(
    corpus_path: Path,
    tracker: ProgressTracker | None = None,
    search_mode: SearchMode = "tfidf",
    embedding_model: str | None = None,
    alpha: float = 0.7,
) -> CorpusIndex:
    """Load corpus from directory with configurable search mode."""
    if tracker:
        tracker.start_step("corpus")

    loader = DocumentLoader(corpus_path)
    documents = loader.load_corpus_sync()

    # Create index based on mode
    if search_mode == "tfidf":
        index = CorpusIndex(documents=documents, mode="tfidf")
        mode_info = "TF-IDF (keyword)"
    else:
        # Lazy import to avoid requiring dependencies
        from source.corpus.embeddings import SentenceTransformerProvider

        model_name = embedding_model or get_settings().embedding_model
        provider = SentenceTransformerProvider(name=model_name)

        index = CorpusIndex(
            documents=documents,
            embedding_provider=provider,
            mode=search_mode,
            alpha=alpha,
        )
        mode_info = f"{search_mode} ({model_name})"

    if tracker:
        tracker.add_detail(f"Loaded {len(documents)} documents")
        tracker.add_detail(f"Search mode: {mode_info}")
        stats = index.get_statistics()
        for doc_type, count in stats.get("documents_by_type", {}).items():
            tracker.add_detail(f"  {doc_type}: {count}")
        if "total_chunks" in stats:
            tracker.add_detail(f"  chunks: {stats['total_chunks']}")
        tracker.complete_step("corpus")

    return index


def load_request(request_path: Path) -> Request:
    """Load request from JSON file."""
    content = request_path.read_text(encoding="utf-8")
    return Request.model_validate_json(content)


@click.group()
@click.option(
    "--corpus",
    "-c",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="data/",
    help="Path to corpus directory",
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice(["openai", "anthropic", "openrouter", "lmstudio"]),
    default="anthropic",
    help="LLM provider to use",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Model to use (default: provider default)",
)
@click.option(
    "--audit-path",
    type=click.Path(file_okay=False, path_type=Path),
    default="audit_logs/",
    help="Path for audit logs",
)
@click.option(
    "--search-mode",
    "-s",
    type=click.Choice(["tfidf", "vector", "hybrid"]),
    default=None,
    help="Search mode: tfidf (fast), vector (semantic), hybrid (best)",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Embedding model for vector/hybrid (default: multilingual-e5-base)",
)
@click.option(
    "--alpha",
    type=float,
    default=0.7,
    help="Vector weight in hybrid mode (0-1, higher = more semantic)",
)
@click.pass_context
def app(
    ctx: click.Context,
    corpus: Path,
    provider: Literal["openai", "anthropic", "openrouter", "lmstudio"],
    model: str | None,
    audit_path: Path,
    search_mode: SearchMode | None,
    embedding_model: str | None,
    alpha: float,
):
    """AI Committee Member - Decision-making system for organizations."""
    ctx.ensure_object(dict)

    settings = get_settings()

    # Set defaults
    if model is None:
        match provider:
            case "anthropic":
                model = "claude-sonnet-4-20250514"
            case "openai":
                model = "gpt-4o"
            case "openrouter":
                model = "anthropic/claude-sonnet-4-20250514"
            case "lmstudio":
                model = "local-model"  # placeholder, user should specify

    # Search mode from CLI or settings
    if search_mode is None:
        search_mode = settings.search_mode

    ctx.obj["corpus_path"] = corpus
    ctx.obj["provider_type"] = provider
    ctx.obj["model"] = model
    ctx.obj["audit_path"] = audit_path
    ctx.obj["search_mode"] = search_mode
    ctx.obj["embedding_model"] = embedding_model or settings.embedding_model
    ctx.obj["alpha"] = alpha

    api_key = settings.get_api_key(provider)
    ctx.obj["api_key"] = api_key


@app.command()
@click.argument("request_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def intake(ctx: click.Context, request_file: Path):
    """Check if a request is complete and ready for decision."""

    async def run():
        tracker = ProgressTracker()
        tracker.add_step("corpus", "Loading corpus")
        tracker.add_step("validate", "Validating request structure")
        tracker.add_step("check", "Checking completeness")

        click.echo()
        corpus = load_corpus(
            ctx.obj["corpus_path"],
            tracker,
            search_mode=ctx.obj["search_mode"],
            embedding_model=ctx.obj["embedding_model"],
            alpha=ctx.obj["alpha"],
        )
        request = load_request(request_file)

        tracker.start_step("validate")
        tracker.add_detail(f"Request: {request.query[:50]}...")
        tracker.complete_step("validate")

        provider = ProviderFactory.create(
            ctx.obj["provider_type"],
            ctx.obj["model"],
            api_key=ctx.obj["api_key"],
        )

        handler = CommandHandler(provider, corpus)

        tracker.start_step("check")
        tracker.add_detail(f"Using model: {ctx.obj['model']}")
        intake_result = await handler.intake(request)
        tracker.complete_step("check")

        tracker.print_summary()

        match intake_result:
            case Ok(result):
                click.echo("\n=== Intake Result ===")
                click.echo(f"Ready: {'Yes' if result.ready else 'No'}")
                click.echo(f"Completeness: {result.completeness_score}%")

                if result.missing_items:
                    click.echo("\nMissing items:")
                    for item in result.missing_items:
                        click.echo(f"  - {item}")

                if result.suggestions:
                    click.echo("\nSuggestions:")
                    for suggestion in result.suggestions:
                        click.echo(f"  - {suggestion}")
            case Error(e):
                click.echo(f"\n❌ Intake failed: {e}", err=True)
                sys.exit(1)

    asyncio.run(run())


@app.command()
@click.argument("request_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), help="Save decision to file"
)
@click.option("--save-audit/--no-audit", default=True, help="Save audit package")
@click.option("--verbose", "-v", is_flag=True, help="Show tool calls in progress")
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Show full tool call results (implies --verbose)",
)
@click.pass_context
def decide(
    ctx: click.Context,
    request_file: Path,
    output: Path | None,
    save_audit: bool,
    verbose: bool,
    debug: bool,
):
    """Make a decision on a request."""

    async def run():
        tracker = ProgressTracker()
        tracker.add_step("corpus", "Loading corpus")
        tracker.add_step("request", "Loading request")
        tracker.add_step("search", "Searching relevant norms and precedents")
        tracker.add_step("decide", "Formulating decision")
        if save_audit:
            tracker.add_step("audit", "Saving audit package")

        click.echo()
        corpus = load_corpus(
            ctx.obj["corpus_path"],
            tracker,
            search_mode=ctx.obj["search_mode"],
            embedding_model=ctx.obj["embedding_model"],
            alpha=ctx.obj["alpha"],
        )

        tracker.start_step("request")
        request = load_request(request_file)
        tracker.add_detail(f"Query: {request.query[:60]}...")
        if request.jurisdiction:
            tracker.add_detail(f"Jurisdiction: {request.jurisdiction}")
        tracker.complete_step("request")

        provider = ProviderFactory.create(
            ctx.obj["provider_type"],
            ctx.obj["model"],
            api_key=ctx.obj["api_key"],
        )

        show_tools = verbose or debug
        handler = CommandHandler(
            provider,
            corpus,
            on_tool_call=tracker.add_detail if show_tools else None,
            debug=debug,
        )

        tracker.start_step("search")
        tracker.add_detail(f"Provider: {ctx.obj['provider_type']}")
        tracker.add_detail(f"Model: {ctx.obj['model']}")

        priority_types = [
            ("charter", "Устав"),
            ("code", "Кодекс"),
            ("decision", "Прецеденты"),
        ]
        seen_ids: set[str] = set()
        for doc_type, label in priority_types:
            results = corpus.search(
                request.query,
                doc_types=[doc_type],
                top_k=3,
            )
            for doc, score in results:
                if doc.doc_id not in seen_ids and score > 0.3:
                    seen_ids.add(doc.doc_id)
                    tracker.add_detail(
                        f"[{label}] [{doc.citation_key}] (relevance: {score:.2f})"
                    )
        tracker.complete_step("search")

        tracker.start_step("decide")
        tracker.add_detail("Analyzing applicable rules...")
        decide_result = await handler.decide(request)

        match decide_result:
            case Ok(result):
                decision = result.decision
                tool_calls = result.tool_calls
                tracker.add_detail(f"Verdict: {decision.verdict}")
                tracker.add_detail(f"Citations: {len(decision.citations)}")
                tracker.add_detail(f"Reasoning steps: {len(decision.reasoning)}")
                tracker.add_detail(f"Tool calls: {tool_calls.total_calls}")
                tracker.complete_step("decide")
            case Error(e):
                tracker.complete_step("decide")
                tracker.print_summary()
                click.echo(f"\n❌ Decision failed: {e}", err=True)
                sys.exit(1)

        if save_audit:
            tracker.start_step("audit")
            audit_path = ctx.obj["audit_path"]
            storage = AuditStorage(audit_path)

            builder = DecisionPackageBuilder(
                corpus,
                model_id=ctx.obj["model"],
                provider=ctx.obj["provider_type"],
            )
            package = builder.build(request, decision, tool_calls)
            storage.save(package)
            tracker.add_detail(f"Package ID: {package.package_id}")
            tracker.complete_step("audit")

        tracker.print_summary()

        click.echo("\n=== Decision ===")
        click.echo(f"Verdict: {decision.verdict}")
        click.echo(f"Summary: {decision.verdict_summary}")

        click.echo("\nFindings of Fact:")
        for i, finding in enumerate(decision.findings_of_fact, 1):
            click.echo(f"  {i}. {finding.fact}")

        click.echo("\nApplicable Norms:")
        for norm in decision.applicable_norms:
            click.echo(f"  - [{norm.citation_key}] {norm.relevance}")

        click.echo("\nReasoning:")
        for step in decision.reasoning:
            click.echo(f"  Fact: {step.fact}")
            click.echo(f"  Norm: {step.norm_or_precedent}")
            click.echo(f"  Conclusion: {step.conclusion}")
            click.echo()

        if decision.uncertainty:
            click.echo(f"Uncertainty: {decision.uncertainty}")

        if decision.minority_view:
            click.echo(f"Minority View: {decision.minority_view}")

        if output:
            output.write_text(decision.model_dump_json(indent=2))
            click.echo(f"\nDecision saved to: {output}")

    asyncio.run(run())


@app.command()
@click.argument("request_file", type=click.Path(exists=True, path_type=Path))
@click.option("--precedent", "-p", multiple=True, help="Precedent IDs to compare")
@click.pass_context
def compare(ctx: click.Context, request_file: Path, precedent: tuple[str]):
    """Compare a case with precedents."""

    async def run():
        tracker = ProgressTracker()
        tracker.add_step("corpus", "Loading corpus")
        tracker.add_step("search", "Finding precedents")
        tracker.add_step("compare", "Comparing cases")
        tracker.add_step("synthesize", "Synthesizing guidance")

        click.echo()
        corpus = load_corpus(
            ctx.obj["corpus_path"],
            tracker,
            search_mode=ctx.obj["search_mode"],
            embedding_model=ctx.obj["embedding_model"],
            alpha=ctx.obj["alpha"],
        )
        request = load_request(request_file)

        provider = ProviderFactory.create(
            ctx.obj["provider_type"],
            ctx.obj["model"],
            api_key=ctx.obj["api_key"],
        )

        handler = CommandHandler(provider, corpus)

        tracker.start_step("search")
        if precedent:
            tracker.add_detail(f"Using specified precedents: {', '.join(precedent)}")
        else:
            tracker.add_detail("Searching for relevant precedents...")
        tracker.complete_step("search")

        tracker.start_step("compare")
        tracker.add_detail(f"Model: {ctx.obj['model']}")
        compare_result = await handler.compare(request, list(precedent))
        tracker.complete_step("compare")

        match compare_result:
            case Ok(result):
                tracker.start_step("synthesize")
                tracker.add_detail(f"Found {len(result.comparisons)} comparisons")
                tracker.complete_step("synthesize")

                tracker.print_summary()

                click.echo("\n=== Comparison Result ===")
                click.echo(result.overall_guidance)

                for comp in result.comparisons:
                    click.echo(f"\n{comp}")
            case Error(e):
                tracker.print_summary()
                click.echo(f"\n❌ Comparison failed: {e}", err=True)
                sys.exit(1)

    asyncio.run(run())


@app.command()
@click.argument("request_file", type=click.Path(exists=True, path_type=Path))
@click.argument("decision_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def redteam(ctx: click.Context, request_file: Path, decision_file: Path):
    """Perform adversarial analysis of a decision."""

    async def run():
        tracker = ProgressTracker()
        tracker.add_step("corpus", "Loading corpus")
        tracker.add_step("load", "Loading decision")
        tracker.add_step("analyze", "Analyzing for bias")
        tracker.add_step("check", "Checking consistency")
        tracker.add_step("recommend", "Generating recommendations")

        click.echo()
        corpus = load_corpus(
            ctx.obj["corpus_path"],
            tracker,
            search_mode=ctx.obj["search_mode"],
            embedding_model=ctx.obj["embedding_model"],
            alpha=ctx.obj["alpha"],
        )
        request = load_request(request_file)

        from source.contracts.decision import Decision

        tracker.start_step("load")
        decision_content = decision_file.read_text()
        decision = Decision.model_validate_json(decision_content)
        tracker.add_detail(f"Verdict: {decision.verdict}")
        tracker.add_detail(f"Citations: {len(decision.citations)}")
        tracker.complete_step("load")

        provider = ProviderFactory.create(
            ctx.obj["provider_type"],
            ctx.obj["model"],
            api_key=ctx.obj["api_key"],
        )

        handler = CommandHandler(provider, corpus)

        tracker.start_step("analyze")
        tracker.add_detail(f"Model: {ctx.obj['model']}")
        redteam_result = await handler.redteam(request, decision)
        tracker.complete_step("analyze")

        match redteam_result:
            case Ok(result):
                tracker.start_step("check")
                tracker.add_detail(f"Issues found: {len(result.issues_found)}")
                tracker.complete_step("check")

                tracker.start_step("recommend")
                tracker.add_detail(f"Recommendations: {len(result.recommendations)}")
                tracker.complete_step("recommend")

                tracker.print_summary()

                click.echo("\n=== Red Team Analysis ===")
                click.echo(f"Bias Assessment: {result.bias_assessment}")

                if result.issues_found:
                    click.echo("\nIssues Found:")
                    for issue in result.issues_found:
                        click.echo(f"  - {issue}")

                if result.recommendations:
                    click.echo("\nRecommendations:")
                    for rec in result.recommendations:
                        click.echo(f"  - {rec}")
            case Error(e):
                tracker.print_summary()
                click.echo(f"\n❌ Red team analysis failed: {e}", err=True)
                sys.exit(1)

    asyncio.run(run())


@app.command()
@click.pass_context
def chat(ctx: click.Context):
    """Interactive chat mode."""
    from source.cli.repl import run_repl

    asyncio.run(
        run_repl(
            corpus_path=ctx.obj["corpus_path"],
            provider_type=ctx.obj["provider_type"],
            model=ctx.obj["model"],
            api_key=ctx.obj["api_key"],
            audit_path=ctx.obj["audit_path"],
        )
    )


@app.command("fix-encoding")
@click.option("--dry-run", is_flag=True, help="Only show what would be fixed")
@click.pass_context
def fix_encoding_cmd(ctx: click.Context, dry_run: bool):
    """Fix mojibake encoding issues in corpus files."""
    from source.corpus.annotator import CorpusAnnotator

    provider = ProviderFactory.create(
        ctx.obj["provider_type"],
        ctx.obj["model"],
        api_key=ctx.obj["api_key"],
    )
    annotator = CorpusAnnotator(provider)

    click.echo(f"\nFixing encoding in corpus: {ctx.obj['corpus_path']}")
    if dry_run:
        click.echo("DRY RUN - no files will be modified\n")
    else:
        click.echo()

    fixed, _ = annotator.fix_corpus_encoding(
        ctx.obj["corpus_path"],
        dry_run=dry_run,
    )

    if not dry_run and fixed > 0:
        click.echo(f"\nFixed {fixed} files with encoding issues")


@app.command()
@click.option("--dry-run", is_flag=True, help="Only show what would be annotated")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Re-annotate all files, including already annotated",
)
@click.option(
    "--batch-size",
    default=5,
    type=int,
    help="Number of concurrent LLM calls",
)
@click.option(
    "--agent",
    "-a",
    is_flag=True,
    help="Use multi-step agent mode (better for weak models, slower)",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Show detailed step-by-step progress",
)
@click.pass_context
def annotate(
    ctx: click.Context,
    dry_run: bool,
    force: bool,
    batch_size: int,
    agent: bool,
    debug: bool,
):
    """Annotate corpus files with LLM-generated metadata.

    Two modes available:

    SIMPLE MODE (default):
        Single LLM call per file. Fast, works well with strong models.

    AGENT MODE (--agent):
        Multi-step extraction (5 steps per file). Better quality with
        weak/local models. Steps: analyze → classify → extract date →
        generate title/tags → validate.

    Examples:

        # Simple mode with Claude
        python main.py annotate

        # Agent mode with local model (for better results)
        python main.py -p lmstudio -m "model-name" annotate --agent

        # Force re-annotate all with debug output
        python main.py annotate --agent --force --debug
    """

    async def run():
        provider = ProviderFactory.create(
            ctx.obj["provider_type"],
            ctx.obj["model"],
            api_key=ctx.obj["api_key"],
        )

        click.echo(f"\nAnnotating corpus: {ctx.obj['corpus_path']}")
        click.echo(f"Provider: {ctx.obj['provider_type']}, Model: {ctx.obj['model']}")

        if agent:
            from source.corpus.agent_annotator import AgentAnnotator

            click.echo("Mode: AGENT (multi-step, 5 calls per file)")
            # Lower batch size for agent mode - 5x more calls per file
            effective_batch = min(batch_size, 2)
            click.echo(f"Batch size: {effective_batch} (reduced for agent mode)")

            annotator = AgentAnnotator(provider, debug=debug)

            if force:
                click.echo("FORCE MODE - re-annotating all files")
            if dry_run:
                click.echo("DRY RUN - no files will be modified\n")
            else:
                click.echo()

            success, failed = await annotator.annotate_corpus(
                ctx.obj["corpus_path"],
                batch_size=effective_batch,
                dry_run=dry_run,
                force=force,
            )
        else:
            from source.corpus.annotator import CorpusAnnotator

            click.echo("Mode: SIMPLE (single call per file)")
            click.echo(f"Batch size: {batch_size}")

            annotator = CorpusAnnotator(provider)

            if force:
                click.echo("FORCE MODE - re-annotating all files")
            if dry_run:
                click.echo("DRY RUN - no files will be modified\n")
            else:
                click.echo()

            success, failed = await annotator.annotate_corpus(
                ctx.obj["corpus_path"],
                batch_size=batch_size,
                dry_run=dry_run,
                force=force,
            )

        if not dry_run:
            click.echo(f"\nResult: {success} annotated, {failed} failed")

    asyncio.run(run())


@app.command()
@click.pass_context
def corpus_stats(ctx: click.Context):
    """Show corpus statistics."""
    corpus = load_corpus(
        ctx.obj["corpus_path"],
        search_mode=ctx.obj["search_mode"],
        embedding_model=ctx.obj["embedding_model"],
        alpha=ctx.obj["alpha"],
    )
    stats = corpus.get_statistics()

    click.echo("\n=== Corpus Statistics ===")
    click.echo(f"Total documents: {stats['total_documents']}")
    click.echo(f"Search mode: {stats.get('mode', 'tfidf')}")
    if "unique_terms" in stats:
        click.echo(f"Unique terms: {stats['unique_terms']}")
    if "total_chunks" in stats:
        click.echo(f"Total chunks: {stats['total_chunks']}")
    if "embedding_model" in stats:
        click.echo(f"Embedding model: {stats['embedding_model']}")
    click.echo(f"Active documents: {stats['active_documents']}")

    click.echo("\nDocuments by type:")
    for doc_type, count in stats["documents_by_type"].items():
        click.echo(f"  {doc_type}: {count}")


@app.command()
@click.option("--delete", "do_delete", is_flag=True, help="Actually delete duplicates")
@click.option("--verbose", "-v", is_flag=True, help="Show all files in each group")
@click.pass_context
def dedup(ctx: click.Context, do_delete: bool, verbose: bool):
    """Find and remove duplicate files in corpus."""
    from source.corpus.dedup import CorpusDeduplicator

    deduplicator = CorpusDeduplicator(ctx.obj["corpus_path"])
    groups = deduplicator.find_duplicates()

    if not groups:
        click.echo("No duplicates found.")
        return

    stats = deduplicator.get_statistics()
    click.echo("\n=== Duplicate Analysis ===")
    click.echo(f"Duplicate groups: {stats['duplicate_groups']}")
    click.echo(f"Total duplicates: {stats['total_duplicates']}")
    click.echo(f"Wasted space: {stats['bytes_wasted'] / 1024:.1f} KB")

    if verbose:
        click.echo("\n=== Duplicate Groups ===")
        for i, group in enumerate(groups, 1):
            click.echo(
                f"\nGroup {i} ({len(group.files)} files, {group.size_bytes} bytes):"
            )
            click.echo(f"  Keep: {group.keeper}")
            for dup in group.duplicates:
                click.echo(f"  Remove: {dup}")

    if do_delete:
        click.echo("\n=== Removing Duplicates ===")
        removed, freed = deduplicator.remove_duplicates(dry_run=not do_delete)
        click.echo(f"\nRemoved {removed} files, freed {freed / 1024:.1f} KB")
    else:
        click.echo("\nUse --delete to actually remove duplicates.")


@app.command()
@click.argument("package_id")
@click.pass_context
def verify_audit(ctx: click.Context, package_id: str):
    """Verify integrity of an audit package."""
    storage = AuditStorage(ctx.obj["audit_path"])
    is_valid, message = storage.verify(package_id)

    if is_valid:
        click.echo(f"[OK] {message}")
    else:
        click.echo(f"[FAIL] {message}")
        sys.exit(1)


@app.command("find-corrupted")
@click.pass_context
def find_corrupted_cmd(ctx: click.Context):
    """Find files with irreversible encoding corruption.

    These files have lost bytes (replaced with ?) and cannot be fixed by ftfy.
    """
    import re

    from source.corpus.constants import CORPUS_EXTENSIONS

    CORRUPTED_PATTERNS = [
        re.compile(r"Ñ\?"),
        re.compile(r"Ð---"),
        re.compile(r"Ð[\?-]{2,}"),
        re.compile(r"[а-яА-Я]Ñ\?[а-яА-Я]"),
    ]

    def is_corrupted(text: str) -> bool:
        for pattern in CORRUPTED_PATTERNS:
            if pattern.search(text):
                return True
        return False

    corpus_path = ctx.obj["corpus_path"]
    corrupted_files: list[Path] = []

    click.echo(f"\nScanning for corrupted files in: {corpus_path}")

    for ext in CORPUS_EXTENSIONS:
        for file_path in corpus_path.rglob(ext):
            try:
                content = file_path.read_text(encoding="utf-8")
                if is_corrupted(content):
                    corrupted_files.append(file_path)
            except Exception as e:
                click.echo(f"  ✗ Error reading {file_path.name}: {e}")

    if not corrupted_files:
        click.echo("No corrupted files found.")
        return

    click.echo(f"\nFound {len(corrupted_files)} corrupted file(s):\n")

    for file_path in corrupted_files:
        rel_path = file_path.relative_to(corpus_path)
        click.echo(f"  ✗ {rel_path}")


@app.command()
@click.option(
    "--host",
    "-h",
    type=str,
    default="127.0.0.1",
    help="Host to bind to",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to bind to",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload on code changes",
)
@click.option(
    "--org-name",
    type=str,
    default="LLM инициатива",
    help="Organization name for the header",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    reload: bool,
    org_name: str,
):
    """Start web server for viewing decisions.

    Uses HTTP Basic Auth. Configure credentials via environment variables:

        AUTH_USERNAME - required, username for Basic Auth
        AUTH_PASSWORD - required, password for Basic Auth

    ⚠️ SECURITY: Do not use weak passwords! The server has rate limiting
    but strong credentials are essential.

    Examples:

        # Set credentials (required)
        AUTH_USERNAME=myuser AUTH_PASSWORD=strong-secret-here python main.py serve

        # Custom port
        AUTH_USERNAME=myuser AUTH_PASSWORD=secret python main.py serve -p 8080

        # Development mode with auto-reload
        AUTH_USERNAME=dev AUTH_PASSWORD=dev python main.py serve --reload

    For production:
        - Use HTTPS (via reverse proxy like nginx/caddy)
        - Set strong passwords (16+ chars, mixed case, numbers, symbols)
        - Consider IP whitelisting at firewall level
    """
    import uvicorn

    from source.web.app import create_app

    static_path = Path("static")
    assets_path = Path("assets")
    data_path = Path("data")

    click.echo("\n🏛️  Committee Decisions Server")
    click.echo(f"   Static path: {static_path}")
    click.echo(f"   Assets path: {assets_path}")
    click.echo(f"   Data path: {data_path}")
    click.echo(f"   URL: http://{host}:{port}")
    click.echo("   Auth: Basic (check env vars COMMITTEE_AUTH_*)")
    click.echo()

    app_instance = create_app(
        public_path=static_path,
        assets_path=assets_path,
        data_path=data_path,
    )

    uvicorn.run(
        app_instance,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
