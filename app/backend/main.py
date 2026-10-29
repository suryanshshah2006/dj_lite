"""DJ-Lite web backend (FastAPI).

Run from the project root:  uvicorn app.backend.main:app --port 8000
The engine (features, style vectors, FAISS index) loads once at startup.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .covers import CoverCache, oembed_fetcher
from .engine import Engine
from .evalsummary import eval_summary, figure_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = Path(__file__).with_name("settings.yaml")


def load_settings() -> dict:
    return yaml.safe_load(SETTINGS_PATH.read_text(encoding="utf-8"))


class NextRequest(BaseModel):
    seed_id: str
    n: int | None = Field(default=None, ge=1, le=50)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)


class MixRequest(BaseModel):
    seed_ids: list[str] = Field(min_length=1, max_length=10)
    arc: list[float] = Field(min_length=1, max_length=200)
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    length: int | None = Field(default=None, ge=2, le=200)
    generator: Literal["beam_v1", "greedy_v0"] | None = None

    @field_validator("arc")
    @classmethod
    def arc_in_range(cls, v: list[float]) -> list[float]:
        if any(not 0.0 <= x <= 1.0 for x in v):
            raise ValueError("arc energies must be in [0, 1]")
        return v


def create_app(engine_factory: Callable[[dict], Engine] | None = None,
               cover_cache: CoverCache | None = None,
               settings: dict | None = None,
               reports_root: Path = PROJECT_ROOT) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = engine_factory or Engine.from_project
        app.state.engine = factory(settings)
        cc = settings["covers"]
        app.state.covers = cover_cache or CoverCache(
            PROJECT_ROOT / cc["db_path"], oembed_fetcher(cc["timeout_s"]),
            rate_per_s=cc["requests_per_second"], failure_ttl_s=cc["failure_ttl_hours"] * 3600)
        yield
        app.state.covers.close()

    app = FastAPI(title="DJ-Lite API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings["cors_origins"],
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    def engine(request: Request) -> Engine:
        return request.app.state.engine

    @app.get("/api/health")
    def health(request: Request):
        return {"ok": True, "tracks": len(engine(request).t)}

    @app.get("/api/search")
    def search(request: Request, q: str = Query(min_length=1, max_length=200)):
        return engine(request).search(q)

    @app.get("/api/featured")
    def featured(request: Request):
        return engine(request).featured()

    @app.get("/api/track/{track_id}")
    def track(request: Request, track_id: str):
        try:
            return engine(request).track(track_id)
        except KeyError:
            raise HTTPException(404, f"track {track_id} not found")

    @app.post("/api/next")
    def next_tracks(request: Request, body: NextRequest):
        ns = settings["next"]
        try:
            return engine(request).next_tracks(
                body.seed_id, body.n or ns["default_n"],
                ns["default_alpha"] if body.alpha is None else body.alpha)
        except KeyError:
            raise HTTPException(404, f"track {body.seed_id} not found")

    @app.post("/api/mix")
    def mix(request: Request, body: MixRequest):
        ms = settings["mix"]
        length = body.length or ms["default_length"]
        if length > ms["max_length"]:
            raise HTTPException(422, f"length must be <= {ms['max_length']}")
        try:
            return engine(request).mix(body.seed_ids, body.arc,
                                       ms["default_alpha"] if body.alpha is None else body.alpha,
                                       length, body.generator)
        except KeyError as exc:
            raise HTTPException(404, f"track {exc.args[0]} not found")

    @app.get("/api/cover/{track_id}")
    async def cover(request: Request, track_id: str):
        try:
            return await request.app.state.covers.get(track_id)
        except ValueError:
            raise HTTPException(400, "invalid track id")

    @app.get("/api/eval")
    def evaluation():
        return eval_summary(settings, reports_root)

    @app.get("/api/figures/{name}")
    def figure(name: str):
        p = figure_path(settings, name, reports_root)
        if p is None:
            raise HTTPException(404, "figure not available")
        return FileResponse(p, media_type="image/png")

    return app


app = create_app()
