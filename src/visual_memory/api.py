from __future__ import annotations

import hashlib
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from . import __version__
from .config import Settings, load_settings
from .imaging import Region
from .packs import PackError
from .service import VisualMemoryService

PACKAGE_DIR = Path(__file__).parent
STATIC_DIR = PACKAGE_DIR / "static"


def _static_version() -> str:
    digest = hashlib.sha256()
    for path in (STATIC_DIR / "app.css", STATIC_DIR / "app.js"):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


STATIC_VERSION = _static_version()


def _normalize_time(value: str | None) -> str | None:
    """フロントの'Z'サフィックスとDB保存形式('+00:00')の食い違いを吸収する。

    DB側の時刻文字列比較と揃うよう、UTCのisoformat('+00:00'サフィックス)に統一する。
    """
    if not value:
        return value
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        token = request.headers.get("X-Visual-Memory-Token") or request.cookies.get("vm_session")
        # クエリの?token=はCookie交換用の初回アクセス(/のみ)に限定する。
        # 他のAPIパスまで許すと、URLを共有した時点で認証トークンごと渡ってしまう
        query_token = request.query_params.get("token") if request.url.path == "/" else None
        valid_query = query_token and secrets.compare_digest(query_token, self.settings.auth_token)
        if not ((token and secrets.compare_digest(token, self.settings.auth_token)) or valid_query):
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
            return HTMLResponse(
                "<h1>Visual Memory is locked</h1><p>Open it from the visual-memory command.</p>",
                status_code=401,
            )
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            csrf = request.headers.get("X-CSRF-Token")
            if not csrf or not secrets.compare_digest(csrf, self.settings.csrf_token):
                return JSONResponse({"detail": "Invalid CSRF token"}, status_code=403)
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        if valid_query:
            response.set_cookie("vm_session", self.settings.auth_token, httponly=True, samesite="strict")
            response.set_cookie("vm_csrf", self.settings.csrf_token, httponly=False, samesite="strict")
        return response


class CaptureRequest(BaseModel):
    source_name: str = Field(min_length=1, max_length=300)


class CreatePackRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    query: str = Field(default="", max_length=1000)
    note: str = Field(default="", max_length=10000)
    event_ids: list[int]
    deduplicate_overlaps: bool = True


class RedactionShape(BaseModel):
    x: float
    y: float
    width: float
    height: float


class RedactionRequest(BaseModel):
    redactions: list[RedactionShape] = Field(max_length=100)


class CaptureRegionRequest(BaseModel):
    ignore_regions: list[RedactionShape] = Field(default_factory=list, max_length=50)
    watch_regions: list[RedactionShape] = Field(default_factory=list, max_length=50)


class ApprovalRequest(BaseModel):
    expiry_hours: int | None = Field(default=None, ge=1, le=24 * 365)


def create_app(settings: Settings | None = None, service: VisualMemoryService | None = None) -> FastAPI:
    settings = settings or load_settings()
    service = service or VisualMemoryService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        yield
        service.stop()

    app = FastAPI(title="External PC Visual Memory", version=__version__, lifespan=lifespan)
    app.state.service = service
    app.state.settings = settings
    app.add_middleware(AuthMiddleware, settings=settings)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")

    @app.exception_handler(PackError)
    async def pack_error_handler(_: Request, exc: PackError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if request.query_params.get("token"):
            response = RedirectResponse("/", status_code=303)
            response.set_cookie("vm_session", settings.auth_token, httponly=True, samesite="strict")
            response.set_cookie("vm_csrf", settings.csrf_token, httponly=False, samesite="strict")
            return response
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "max_pack_images": settings.max_pack_images,
                "pack_warning_images": settings.pack_warning_images,
                "static_version": STATIC_VERSION,
            },
        )

    @app.get("/api/status")
    def status():
        return service.status()

    @app.get("/api/devices")
    def devices():
        try:
            return {"devices": service.list_devices()}
        except Exception as exc:
            raise HTTPException(500, f"Unable to list capture devices: {exc}") from exc

    @app.post("/api/capture/start")
    def capture_start(payload: CaptureRequest):
        ready, reason = service.capture_readiness()
        if not ready:
            raise HTTPException(409, reason or "Capture is not ready")
        try:
            return asdict(service.capture.start(payload.source_name))
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/capture/stop")
    def capture_stop():
        service.capture.stop()
        return {"state": "stopped"}

    @app.get("/api/capture/regions")
    def capture_regions():
        return {
            "ignore_regions": settings.ignore_regions,
            "watch_regions": settings.watch_regions,
        }

    @app.put("/api/capture/regions")
    def update_capture_regions(payload: CaptureRegionRequest):
        if service.capture.status.state not in {"stopped", "failed"}:
            raise HTTPException(409, "Stop capture before changing detection regions")

        def normalized(items: list[RedactionShape]) -> list[dict[str, float]]:
            result: list[dict[str, float]] = []
            for value in items:
                item = Region(**value.model_dump()).normalized()
                if item.width > 0 and item.height > 0:
                    result.append({"x": item.x, "y": item.y, "width": item.width, "height": item.height})
            return result

        ignore = normalized(payload.ignore_regions)
        watch = normalized(payload.watch_regions)
        settings.save_region_config(ignore, watch)
        return {"ignore_regions": ignore, "watch_regions": watch}

    @app.get("/api/events")
    def events(
        q: str = "",
        start: str | None = None,
        end: str | None = None,
        limit: int = Query(default=60, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ):
        start = _normalize_time(start)
        end = _normalize_time(end)
        results = service.search.search(q, start, end, limit + 1, offset)
        has_more = len(results) > limit
        results = results[:limit]
        session_ids = list(dict.fromkeys(item["session_id"] for item in results))
        sessions: list[dict] = []
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            rows = service.db.fetchall(
                f"""
                SELECT s.*, COUNT(e.id) AS event_count
                FROM capture_session s
                LEFT JOIN screen_event e ON e.session_id=s.id
                WHERE s.id IN ({placeholders})
                GROUP BY s.id
                """,
                session_ids,
            )
            sessions = [dict(row) for row in rows]
        return {
            "events": results,
            "sessions": sessions,
            "has_more": has_more,
            "next_offset": offset + len(results),
            "total": service.search.timeline_count(start, end) if not q.strip() else None,
        }

    @app.get("/api/events/{event_id}")
    def event_detail(event_id: int):
        result = service.search.event_with_neighbors(event_id)
        if not result:
            raise HTTPException(404, "Event not found")
        return result

    @app.get("/api/events/{event_id}/frame")
    def event_frame(event_id: int, thumbnail: bool = False):
        row = service.db.fetchone(
            "SELECT frame_path,thumbnail_path FROM screen_event WHERE id=?", (event_id,)
        )
        if not row:
            raise HTTPException(404, "Event not found")
        path = Path(row["thumbnail_path"] if thumbnail else row["frame_path"])
        if not path.exists():
            raise HTTPException(404, "Image file is missing")
        return FileResponse(path, media_type="image/webp")

    @app.delete("/api/events")
    def delete_events(start: str | None = None, end: str | None = None, confirm: bool = False):
        # 誤爆防止のため、期間指定とconfirm=trueを必須にする(未指定での全削除を禁止)
        if not confirm or not start or not end:
            raise HTTPException(400, "start, end and confirm=true are required")
        return {"deleted": service.delete_events(_normalize_time(start), _normalize_time(end))}

    @app.get("/api/packs")
    def packs():
        return {"packs": service.packs.list(include_drafts=True)}

    @app.post("/api/packs")
    def create_pack(payload: CreatePackRequest):
        return service.packs.create(
            payload.title,
            payload.event_ids,
            payload.query,
            payload.note,
            payload.deduplicate_overlaps,
        )

    @app.get("/api/packs/{pack_id}")
    def get_pack(pack_id: str):
        return service.packs.get(pack_id, include_items=True)

    @app.put("/api/packs/{pack_id}/items/{event_id}/redactions")
    def set_redactions(pack_id: str, event_id: int, payload: RedactionRequest):
        return service.packs.set_redactions(
            pack_id, event_id, [item.model_dump() for item in payload.redactions]
        )

    @app.post("/api/packs/{pack_id}/approve")
    def approve_pack(pack_id: str, payload: ApprovalRequest | None = None):
        return service.packs.approve(pack_id, payload.expiry_hours if payload else None)

    @app.post("/api/packs/{pack_id}/revoke")
    def revoke_pack(pack_id: str):
        return service.packs.revoke(pack_id)

    @app.get("/api/packs/{pack_id}/document")
    def export_pack_document(
        pack_id: str, document_format: str = Query(default="pdf", alias="format")
    ):
        path = service.packs.export_document(pack_id, document_format)
        media_type = "application/pdf" if path.suffix == ".pdf" else "text/html; charset=utf-8"
        return FileResponse(
            path,
            media_type=media_type,
            filename=f"context-{pack_id}{path.suffix}",
        )

    return app
