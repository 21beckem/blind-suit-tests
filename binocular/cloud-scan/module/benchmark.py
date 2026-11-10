"""Named timers with aggregate statistics, printing, and JSON export."""

from contextlib import contextmanager
import json
from pathlib import Path
import time


class Benchmark:
    def __init__(self):
        self._stats = {}
        self._running = {}

    def setup(self, *names):
        """Declare timer names and reset results, e.g. setup('capture', 'depth')."""
        if self._running:
            raise RuntimeError("Stop running timers before resetting benchmarks")
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Supply one or more nonempty timer names")
        if len(set(names)) != len(names):
            raise ValueError("Timer names must be unique")
        self._stats = {name: {"count": 0, "total_ms": 0.0, "min_ms": None, "max_ms": None}
                       for name in names}
        return self

    def start(self, name):
        if name not in self._stats:
            raise KeyError(f"Undeclared benchmark {name!r}; add it to setup()")
        if name in self._running:
            raise RuntimeError(f"Benchmark {name!r} is already running")
        self._running[name] = time.perf_counter()

    def stop(self, name):
        now = time.perf_counter()
        if name not in self._running:
            raise RuntimeError(f"Benchmark {name!r} is not running")
        elapsed = (now - self._running.pop(name)) * 1000
        stats = self._stats[name]
        stats["count"] += 1
        stats["total_ms"] += elapsed
        stats["min_ms"] = elapsed if stats["min_ms"] is None else min(stats["min_ms"], elapsed)
        stats["max_ms"] = elapsed if stats["max_ms"] is None else max(stats["max_ms"], elapsed)
        return elapsed

    @contextmanager
    def measure(self, name):
        """Time a block; stop the timer even when the block raises an exception."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def results(self):
        """Return completed samples only; memory use stays constant per timer."""
        return {name: {**stats, "mean_ms": stats["total_ms"] / stats["count"] if stats["count"] else None}
                for name, stats in self._stats.items()}

    def display(self):
        print(f"{'Benchmark':<20} {'Count':>8} {'Mean ms':>12} {'Min ms':>12} {'Max ms':>12} {'Total ms':>12}")
        for name, stats in self.results().items():
            values = ["-" if stats[key] is None else f"{stats[key]:.3f}"
                      for key in ("mean_ms", "min_ms", "max_ms", "total_ms")]
            print(f"{name:<20} {stats['count']:>8} " + " ".join(f"{value:>12}" for value in values))

    def save(self, path):
        """Export completed timer statistics as JSON (all durations in ms)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.results(), file, indent=2, allow_nan=False)
            file.write("\n")
        return path
