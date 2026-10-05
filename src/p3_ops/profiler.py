import time
from contextlib import contextmanager


class LatencyProfiler:
    def __init__(self):
        self.timings = {}

    @contextmanager
    def track(self, stage_name: str):
        t0 = time.perf_counter()
        yield
        self.timings[stage_name] = (time.perf_counter() - t0) * 1000.0
