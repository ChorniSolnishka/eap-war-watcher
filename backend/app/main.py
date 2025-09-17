from __future__ import annotations

import logging
import os

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings

# Routers
from app.routers import export as export_router
from app.routers import uploads as uploads_router

logger = logging.getLogger(__name__)


def _configure_opencv_threads() -> None:
    """Configure OpenCV thread count based on settings / CPU."""
    try:
        import cv2

        n = settings.OPENCV_NUM_THREADS
        if n is None:
            n = max(1, (os.cpu_count() or 2) // 2)
        cv2.setNumThreads(int(n))
        logger.info("OpenCV configured with %d threads", n)
    except Exception as e:  # cv2 can be absent in some environments
        logger.warning("Failed to configure OpenCV threads: %s", e)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----- Static files -----
    # Serve everything under STORAGE_ROOT at /static (or whatever STATIC_URL is).
    # This includes: screens/, crops/, debug_seg_dark/,
    # debug_by_id/, debug_digits/, etc.
    static_url: str = getattr(settings, "STATIC_URL", "/static")
    app.mount(
        static_url,
        StaticFiles(directory=str(settings.STORAGE_ROOT), html=False),
        name="static",
    )

    # ----- API v1 router -----
    api_prefix: str = getattr(settings, "API_PREFIX", "/api")
    api = APIRouter(prefix=api_prefix)
    api.include_router(uploads_router.router)
    api.include_router(export_router.router)

    app.include_router(api)

    _configure_opencv_threads()
    return app


app = create_app()
