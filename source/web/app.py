"""FastAPI application for serving static decision files with auth."""

import json
import re
from pathlib import Path
from typing import Annotated
from urllib.parse import unquote

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from source.web.auth import verify_credentials
from source.web.viewer import render_document_viewer


def _sanitize_back_url(referer: str | None, default: str = "/") -> str:
    """
    Sanitize back URL to prevent open redirect attacks.

    Only allows relative paths starting with /.
    """
    if not referer:
        return default

    # Parse and validate - only allow relative paths
    # Remove any protocol/host part
    if referer.startswith(("http://", "https://", "//")):
        # Extract path from URL
        from urllib.parse import urlparse

        parsed = urlparse(referer)
        path = parsed.path
        if parsed.query:
            path += "?" + parsed.query
        referer = path

    # Must start with / and not contain dangerous patterns
    if not referer.startswith("/"):
        return default

    # Block javascript: and data: URLs that could be injected
    if "javascript:" in referer.lower() or "data:" in referer.lower():
        return default

    return referer


# Valid UUID4 pattern for package IDs
_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _is_valid_package_id(package_id: str) -> bool:
    """Validate that package_id is a valid UUID4."""
    return bool(_UUID4_PATTERN.match(package_id))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # XSS protection (legacy, but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "frame-ancestors 'none';"
        )
        return response


def _load_disabled_models(public_path: Path) -> set[str]:
    """Load disabled models from cases.json."""
    config_path = public_path / "cases.json"
    if not config_path.exists():
        return set()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return set(data.get("disabled_models", []))
    except Exception:
        return set()


def _is_model_disabled(model_id: str, disabled_models: set[str]) -> bool:
    """Check if a model is disabled."""
    return model_id in disabled_models


def create_app(
    public_path: Path | None = None,
    assets_path: Path | None = None,
    data_path: Path | None = None,
) -> FastAPI:
    """Create FastAPI app that serves static files with Basic Auth."""
    if public_path is None:
        public_path = Path("public")
    if assets_path is None:
        assets_path = Path("assets")
    if data_path is None:
        data_path = Path("data")

    app = FastAPI(
        title="Committee Decisions",
        description="Static file server with Basic Auth",
        version="0.1.0",
        # Hide docs in production
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Add security headers to all responses
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint (no auth)."""
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index(
        username: Annotated[str, Depends(verify_credentials)],
    ) -> FileResponse:
        """Serve index.html."""
        index_file = public_path / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(index_file, media_type="text/html")

    @app.get("/attachments/{path:path}", response_class=HTMLResponse)
    async def view_attachment(
        path: str,
        request: Request,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> HTMLResponse:
        """View attachment files (assets/) with formatted display."""
        path = unquote(path)

        if ".." in path or path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid path")

        file_path = assets_path / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Attachment not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")

        content = file_path.read_text(encoding="utf-8")

        # Build back URL from referer (sanitized)
        back_url = _sanitize_back_url(request.headers.get("referer"))

        html = render_document_viewer(
            content=content,
            file_path=f"assets/{path}",
            back_url=back_url,
        )
        return HTMLResponse(content=html)

    def _find_document_by_key(key: str) -> Path | None:
        """
        Find a corpus document by citation_key or doc_id.

        Searches through data/ for markdown files with matching frontmatter.
        """
        # Pattern to extract citation_key or doc_id from frontmatter
        # Handle both quoted and unquoted values
        citation_pattern = re.compile(
            r"^citation_key:\s*['\"]?([^'\"\n]+)['\"]?\s*$", re.MULTILINE
        )
        doc_id_pattern = re.compile(
            r"^doc_id:\s*['\"]?([^'\"\n]+)['\"]?\s*$", re.MULTILINE
        )

        for md_file in data_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                # Only check files with frontmatter
                if not content.startswith("---"):
                    continue

                # Extract frontmatter (between first two ---)
                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue
                frontmatter = parts[1]

                # Check citation_key
                match = citation_pattern.search(frontmatter)
                if match and match.group(1).strip() == key:
                    return md_file

                # Check doc_id
                match = doc_id_pattern.search(frontmatter)
                if match and match.group(1).strip() == key:
                    return md_file
            except Exception:
                continue

        return None

    # IMPORTANT: This route must be BEFORE /corpus/{path:path} to avoid being shadowed
    @app.get("/corpus/by-key/{key}")
    async def view_corpus_by_key(
        key: str,
        request: Request,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> HTMLResponse:
        """
        View corpus document by citation_key or doc_id.

        Looks up the document and renders it.
        """
        key = unquote(key)

        file_path = _find_document_by_key(key)
        if file_path is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document with key '{key}' not found",
            )

        content = file_path.read_text(encoding="utf-8")
        relative_path = file_path.relative_to(data_path)

        # Build back URL from referer (sanitized)
        back_url = _sanitize_back_url(request.headers.get("referer"))

        html = render_document_viewer(
            content=content,
            file_path=f"data/{relative_path}",
            back_url=back_url,
        )
        return HTMLResponse(content=html)

    @app.get("/corpus/{path:path}", response_class=HTMLResponse)
    async def view_corpus_document(
        path: str,
        request: Request,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> HTMLResponse:
        """View corpus documents (data/) with formatted display."""
        path = unquote(path)

        if ".." in path or path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid path")

        file_path = data_path / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail="Not a file")

        content = file_path.read_text(encoding="utf-8")

        # Build back URL from referer (sanitized)
        back_url = _sanitize_back_url(request.headers.get("referer"))

        html = render_document_viewer(
            content=content,
            file_path=f"data/{path}",
            back_url=back_url,
        )
        return HTMLResponse(content=html)

    @app.get("/attachments-raw/{path:path}")
    async def download_attachment(
        path: str,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> FileResponse:
        """Download raw attachment file."""
        path = unquote(path)

        if ".." in path or path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid path")

        file_path = assets_path / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Attachment not found")

        suffix = file_path.suffix.lower()
        media_types = {
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".txt": "text/plain",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".pdf": "application/pdf",
        }
        media_type = media_types.get(suffix, "application/octet-stream")

        return FileResponse(file_path, media_type=media_type)

    @app.get("/corpus-raw/{path:path}")
    async def download_corpus_document(
        path: str,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> FileResponse:
        """Download raw corpus document."""
        path = unquote(path)

        if ".." in path or path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid path")

        file_path = data_path / path
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Document not found")

        suffix = file_path.suffix.lower()
        media_types = {
            ".md": "text/markdown",
            ".markdown": "text/markdown",
            ".txt": "text/plain",
            ".json": "application/json",
        }
        media_type = media_types.get(suffix, "application/octet-stream")

        return FileResponse(file_path, media_type=media_type)

    @app.get("/api/decision/{package_id}")
    async def get_decision(
        package_id: str,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> JSONResponse:
        """
        Get decision JSON by package ID.

        Returns 403 if the model is disabled.
        """
        # Validate package_id format (UUID4 only)
        if not _is_valid_package_id(package_id):
            raise HTTPException(status_code=400, detail="Invalid package ID format")

        # Find the package in audit_logs
        audit_logs = Path("audit_logs")
        if not audit_logs.exists():
            raise HTTPException(status_code=404, detail="No audit logs found")

        package_file = audit_logs / f"{package_id}.json"
        if not package_file.exists():
            raise HTTPException(status_code=404, detail="Decision not found")

        try:
            data = json.loads(package_file.read_text(encoding="utf-8"))
            model_id = data.get("model_id", "")

            # Check if model is disabled
            disabled_models = _load_disabled_models(public_path)
            if _is_model_disabled(model_id, disabled_models):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access to decisions from model '{model_id}' is restricted",
                )

            return JSONResponse(content=data)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/decisions")
    async def list_decisions(
        username: Annotated[str, Depends(verify_credentials)],
    ) -> JSONResponse:
        """
        List all available decisions.

        Excludes disabled models.
        """
        audit_logs = Path("audit_logs")
        if not audit_logs.exists():
            return JSONResponse(content={"decisions": []})

        disabled_models = _load_disabled_models(public_path)
        decisions: list[dict] = []

        for json_file in audit_logs.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                model_id = data.get("model_id", "")

                # Skip disabled models
                if _is_model_disabled(model_id, disabled_models):
                    continue

                decisions.append(
                    {
                        "package_id": data.get("package_id"),
                        "model_id": model_id,
                        "timestamp": data.get("timestamp"),
                    }
                )
            except Exception:
                continue

        return JSONResponse(content={"decisions": decisions})

    @app.get("/{path:path}")
    async def serve_static(
        path: str,
        username: Annotated[str, Depends(verify_credentials)],
    ) -> FileResponse:
        """Serve static files from public directory."""
        if ".." in path or path.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid path")

        file_path = public_path / path
        if file_path.is_dir():
            file_path = file_path / "index.html"

        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        suffix = file_path.suffix.lower()
        media_types = {
            ".html": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }
        media_type = media_types.get(suffix, "application/octet-stream")

        return FileResponse(file_path, media_type=media_type)

    return app


def get_app() -> FastAPI:
    """Get the default app instance for uvicorn."""
    return create_app()
