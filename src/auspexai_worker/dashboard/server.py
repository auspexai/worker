"""Daemon thread that runs uvicorn serving the dashboard FastAPI app.

Mirrors the shape of HeartbeatLoop / AssignmentPoller in the
auspexai_worker.daemon package — `start()` spawns a thread; `stop()`
signals shutdown. Bound to localhost-only by default (the §5.14
Layer B "no-external-exposure" constraint); host/port come from
config.
"""

from __future__ import annotations

import logging
import threading

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__name__)


class DashboardServer:
    """uvicorn-in-a-thread for the worker dashboard.

    Use as:
        server = DashboardServer(app=app, host="127.0.0.1", port=7799)
        server.start()
        ...
        server.stop()
    """

    def __init__(self, *, app: FastAPI, host: str, port: int) -> None:
        # uvicorn.Server.should_exit flag is what we toggle on stop;
        # the thread blocks in server.run() until that flag is set.
        self._app = app
        self._host = host
        self._port = port
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Spawn the server thread. Returns immediately."""
        if self._thread is not None:
            raise RuntimeError("DashboardServer already started")
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",  # daemon already logs lifecycle; uvicorn at WARN
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="auspexai-worker-dashboard",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "dashboard server started on http://%s:%d",
            self._host,
            self._port,
        )

    def stop(self, *, timeout: float = 5.0) -> None:
        """Signal shutdown and wait for the thread (best-effort)."""
        if self._server is None or self._thread is None:
            return
        logger.info("dashboard server stopping...")
        self._server.should_exit = True
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("dashboard server thread did not exit cleanly")
        else:
            logger.info("dashboard server stopped")
        self._server = None
        self._thread = None
