"""Background jobs so hosted POSTs return before the proxy gives up.

Cloudflare (and other tunnels) will replace a long-running POST with an HTML
error page. The playground then dies on `res.json()` with
`Unexpected token '<'`. Query/answer/research therefore return a job id
immediately; the browser polls a cheap GET until the work is done.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable


class JobFail(Exception):
    """Typed failure stored on the job, not raised through the HTTP thread."""

    def __init__(self, status: int, error: str, detail: str = "") -> None:
        super().__init__(error)
        self.status = int(status)
        self.error = error
        self.detail = detail


class JobBoard:
    def __init__(self, *, max_jobs: int = 32, ttl_s: float = 900.0) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self.max_jobs = max_jobs
        self.ttl_s = ttl_s

    def submit(self, fn: Callable[[], dict[str, Any]]) -> str:
        job_id = str(uuid.uuid4())
        rec: dict[str, Any] = {
            "status": "running",
            "result": None,
            "error": None,
            "started": time.time(),
        }
        with self._lock:
            self._gc_locked()
            self._jobs[job_id] = rec

        def run() -> None:
            try:
                rec["result"] = fn()
                rec["status"] = "done"
            except JobFail as e:
                rec["error"] = {
                    "error": e.error,
                    "detail": e.detail,
                    "http_status": e.status,
                }
                rec["status"] = "error"
            except Exception as e:
                rec["error"] = {
                    "error": "internal error",
                    "detail": f"{type(e).__name__}: {e}"[:240],
                    "http_status": 500,
                }
                rec["status"] = "error"

        threading.Thread(target=run, daemon=True, name=f"ui-job-{job_id[:8]}").start()
        return job_id

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            return {
                "job_id": job_id,
                "status": rec["status"],
                "result": rec["result"],
                "error": rec["error"],
            }

    def _gc_locked(self) -> None:
        now = time.time()
        stale = [k for k, v in self._jobs.items() if now - float(v["started"]) > self.ttl_s]
        for k in stale:
            self._jobs.pop(k, None)
        if len(self._jobs) < self.max_jobs:
            return
        oldest = sorted(self._jobs.items(), key=lambda kv: float(kv[1]["started"]))
        for k, _ in oldest[: max(0, len(self._jobs) - self.max_jobs + 1)]:
            self._jobs.pop(k, None)
