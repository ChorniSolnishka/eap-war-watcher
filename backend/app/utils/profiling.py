from __future__ import annotations

import atexit
import csv
import functools
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, ParamSpec, TypeVar, cast

P = ParamSpec("P")
R = TypeVar("R")

# psutil is optional, i decided to keep it any
psutil: Any
try:
    import psutil as _psutil  # type: ignore[import-untyped]

    psutil = _psutil
except Exception:  # pragma: no cover
    psutil = None

_logger = logging.getLogger("profiler")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _logger.addHandler(_handler)
_logger.setLevel(logging.INFO)


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


# -------- Aggregator ----------
class _Stats:
    __slots__ = (
        "count",
        "sum_ms",
        "sum_cpu_user",
        "sum_cpu_sys",
        "sum_rss_mb",
        "max_ms",
        "_samples",
        "_lock",
    )

    def __init__(self) -> None:
        self.count = 0
        self.sum_ms = 0.0
        self.sum_cpu_user = 0
        self.sum_cpu_sys = 0
        self.sum_rss_mb = 0.0
        self.max_ms = 0.0
        self._samples: List[float] = []
        self._lock = threading.Lock()

    def add(
        self,
        ms: float,
        cpu_user_ms: Optional[int],
        cpu_sys_ms: Optional[int],
        rss_mb_delta: Optional[float],
    ) -> None:
        with self._lock:
            self.count += 1
            self.sum_ms += ms
            if cpu_user_ms is not None:
                self.sum_cpu_user += cpu_user_ms
            if cpu_sys_ms is not None:
                self.sum_cpu_sys += cpu_sys_ms
            if rss_mb_delta is not None:
                self.sum_rss_mb += rss_mb_delta
            if ms > self.max_ms:
                self.max_ms = ms
            buf = self._samples
            if len(buf) < 2048:
                buf.append(ms)
            else:
                idx = self.count % 2048
                buf[idx] = ms

    def quantiles(self) -> tuple[float, float]:
        s = sorted(self._samples)
        if not s:
            return (0.0, 0.0)

        def _q(p: float) -> float:
            k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
            return s[k]

        return (_q(0.50), _q(0.95))


_AGG: Dict[str, _Stats] = {}
_AGG_LOCK = threading.Lock()
_LAST_FLUSH = 0.0


def _agg_add(
    label: str,
    ms: float,
    cpu_user_ms: Optional[int],
    cpu_sys_ms: Optional[int],
    rss_mb_delta: Optional[float],
) -> None:
    with _AGG_LOCK:
        st = _AGG.get(label)
        if st is None:
            st = _Stats()
            _AGG[label] = st
    st.add(ms, cpu_user_ms, cpu_sys_ms, rss_mb_delta)


def _flush_csv(force: bool = False) -> None:
    path = os.getenv("PROFILE_OUT_CSV")
    if not path:
        return
    global _LAST_FLUSH
    now = time.time()
    interval = float(os.getenv("PROFILE_FLUSH_SEC", "5"))
    if not force and (now - _LAST_FLUSH) < interval:
        return
    _LAST_FLUSH = now
    with _AGG_LOCK:
        items = list(_AGG.items())
    rows = []
    for name, st in items:
        p50, p95 = st.quantiles()
        avg = (st.sum_ms / st.count) if st.count else 0.0
        rows.append(
            {
                "name": name,
                "count": st.count,
                "total_ms": round(st.sum_ms, 3),
                "avg_ms": round(avg, 3),
                "p50_ms": round(p50, 3),
                "p95_ms": round(p95, 3),
                "max_ms": round(st.max_ms, 3),
                "cpu_user_ms_total": st.sum_cpu_user,
                "cpu_sys_ms_total": st.sum_cpu_sys,
                "rss_mb_delta_total": round(st.sum_rss_mb, 3),
            }
        )
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "name",
                "count",
                "total_ms",
                "avg_ms",
                "p50_ms",
                "p95_ms",
                "max_ms",
                "cpu_user_ms_total",
                "cpu_sys_ms_total",
                "rss_mb_delta_total",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception:
        pass


atexit.register(lambda: _flush_csv(force=True))


def profiled(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator: logs wall-time, CPU, and ΔRSS. When PROFILE_JSON_LOG=1 writes
    JSON logs; aggregates to CSV when PROFILE_OUT_CSV is set; optional energy
    tracking when PROFILE_ENERGY=1 (requires codecarbon).
    """

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        label = name or f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            proc = psutil.Process() if psutil else None
            rss_before = proc.memory_info().rss if proc else 0
            cpu_before = proc.cpu_times() if proc else None

            t0 = _now_ms()
            energy_mwh = None
            tracer = None

            if os.getenv("PROFILE_ENERGY", "0") == "1":
                try:
                    os.environ.setdefault("CODECARBON_LOG_LEVEL", "error")
                    from codecarbon import EmissionsTracker  # type: ignore

                    tracer = EmissionsTracker(
                        measure_power_secs=0.5,
                        save_to_file=False,
                        log_level="error",
                    )
                    tracer.start()
                except Exception:
                    tracer = None

            try:
                return fn(*args, **kwargs)
            finally:
                t1 = _now_ms()
                cpu_user_ms = cpu_sys_ms = None
                if proc and cpu_before:
                    cpu_after = proc.cpu_times()
                    cpu_user_ms = int((cpu_after.user - cpu_before.user) * 1000)
                    cpu_sys_ms = int((cpu_after.system - cpu_before.system) * 1000)
                rss_after = proc.memory_info().rss if proc else 0
                rss_delta_mb = (
                    round((rss_after - rss_before) / (1024 * 1024), 3) if proc else None
                )

                if tracer:
                    try:
                        tracer.stop()
                        energy_mwh = getattr(
                            getattr(tracer, "_emissions", None),
                            "energy_consumed",
                            None,
                        )
                    except Exception:
                        energy_mwh = None

                payload = {
                    "event": "profile",
                    "name": label,
                    "ms": round(t1 - t0, 3),
                    "cpu_user_ms": cpu_user_ms,
                    "cpu_sys_ms": cpu_sys_ms,
                    "rss_mb_delta": rss_delta_mb,
                    "thread": threading.current_thread().name,
                }
                if energy_mwh is not None:
                    payload["energy_mwh"] = energy_mwh

                try:
                    if os.getenv("PROFILE_JSON_LOG", "0") == "1":
                        _logger.info(json.dumps(payload, ensure_ascii=False))
                    else:
                        _logger.info(
                            "[PROFILE] %s: %.1f ms, ΔRSS=%s MB",
                            label,
                            payload["ms"],
                            "n/a" if rss_delta_mb is None else f"{rss_delta_mb:.2f}",
                        )
                except Exception:
                    pass

                ms_val = cast(float, payload["ms"])
                _agg_add(label, ms_val, cpu_user_ms, cpu_sys_ms, rss_delta_mb)
                _flush_csv(force=False)

        return wrapper

    return deco
