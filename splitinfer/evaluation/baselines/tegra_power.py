#!/usr/bin/env python3
"""tegra_power.py — sample Jetson power consumption via `tegrastats`.

Why this exists:
    Spec § 7.3 lists energy consumption as a primary evaluation metric.
    Without active sampling we cannot attribute Joules to a workload.
    `tegrastats` is NVIDIA's official tool for live SoC telemetry on
    Jetson devices and exposes per-rail mW readings at configurable
    intervals (default 1 sec; we use 100 ms for finer time resolution).

Usage:
    from tegra_power import TegraStatsCollector

    with TegraStatsCollector(interval_ms=100) as col:
        run_workload()              # measured period
    summary = col.summary()
    # summary == {
    #     "samples":   <count>,
    #     "duration_s": <wall time>,
    #     "rails": {
    #         "VDD_IN":         {"mean_mw": ..., "median_mw": ..., "max_mw": ..., "min_mw": ...},
    #         "VDD_CPU_GPU_CV": {...},
    #         "VDD_SOC":        {...},
    #     },
    #     "energy_J":  <integral of VDD_IN over duration, joules>,
    # }

Output format (JetPack 6.x R36 on Orin Nano):
    04-08-2026 15:31:19 RAM 1602/7607MB ... VDD_IN 5783mW/5783mW
        VDD_CPU_GPU_CV 791mW/791mW VDD_SOC 1703mW/1703mW

The parser extracts the FIRST number of each "VDD_xxx N1mW/N2mW" pair
(N1 = instantaneous, N2 = running average computed by tegrastats itself).
We use the instantaneous reading because we compute our own statistics.

Requires:
    sudo access (tegrastats reads kernel sysfs entries).  The collector
    will spawn `sudo tegrastats` and ask for a password if not cached.
"""

import re
import statistics
import subprocess
import threading
import time
from typing import Dict, List, Optional


# Regex matches "VDD_NAME 1234mW/5678mW" or just "VDD_NAME 1234mW".
# Group 1: rail name, Group 2: instantaneous mW value.
_VDD_RE = re.compile(r"(VDD_[A-Z_0-9]+)\s+(\d+)mW")


class TegraStatsCollector:
    """Background sampler for tegrastats power readings.

    Use as a context manager:
        with TegraStatsCollector() as col:
            do_work()
        print(col.summary())

    Or manually:
        col = TegraStatsCollector()
        col.start()
        do_work()
        col.stop()
        print(col.summary())
    """

    def __init__(self, interval_ms: int = 100, sudo: bool = True):
        """
        Args:
            interval_ms: tegrastats sampling interval in milliseconds.
                100 ms is fine-grained enough to capture short bursts.
            sudo: whether to prepend `sudo` to the tegrastats invocation.
                Required on most Jetson installs because tegrastats reads
                root-only sysfs entries.
        """
        self.interval_ms = interval_ms
        self.sudo        = sudo
        self._proc:     Optional[subprocess.Popen] = None
        self._thread:   Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._samples: List[Dict[str, int]] = []   # list of {rail: mw}
        self._t_start = 0.0
        self._t_stop  = 0.0

    def start(self) -> None:
        cmd = ["tegrastats", "--interval", str(self.interval_ms)]
        if self.sudo:
            cmd = ["sudo", "-n"] + cmd  # -n: non-interactive, fail if password needed
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=1,
                universal_newlines=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "tegrastats not found.  Install JetPack BSP or run on a Jetson device.") from e

        self._stop_evt.clear()
        self._samples = []
        self._t_start = time.monotonic()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop_evt.is_set():
                break
            sample: Dict[str, int] = {}
            for m in _VDD_RE.finditer(line):
                rail, mw = m.group(1), int(m.group(2))
                sample[rail] = mw
            if sample:
                self._samples.append(sample)

    def stop(self) -> None:
        self._stop_evt.set()
        self._t_stop = time.monotonic()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._proc.kill()
                except OSError:
                    pass
            self._proc = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def summary(self) -> dict:
        """Return aggregated power statistics across all sampled rails."""
        duration = max(self._t_stop - self._t_start, 1e-6)
        rails: Dict[str, List[int]] = {}
        for s in self._samples:
            for rail, mw in s.items():
                rails.setdefault(rail, []).append(mw)

        rail_stats = {}
        for rail, vals in rails.items():
            rail_stats[rail] = {
                "mean_mw":   statistics.mean(vals)   if vals else 0.0,
                "median_mw": statistics.median(vals) if vals else 0.0,
                "max_mw":    max(vals)               if vals else 0,
                "min_mw":    min(vals)               if vals else 0,
                "samples":   len(vals),
            }

        # Energy estimate: integrate VDD_IN (total system input) over duration.
        # Trapezoidal sum at the configured interval.
        energy_j = 0.0
        if "VDD_IN" in rails and len(rails["VDD_IN"]) > 1:
            dt_s = self.interval_ms / 1000.0
            mw_vals = rails["VDD_IN"]
            for i in range(len(mw_vals) - 1):
                avg_mw = (mw_vals[i] + mw_vals[i + 1]) / 2.0
                energy_j += (avg_mw / 1000.0) * dt_s

        return {
            "samples":    len(self._samples),
            "duration_s": duration,
            "rails":      rail_stats,
            "energy_J":   energy_j,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False  # don't suppress exceptions


def main():
    """Quick smoke test: sample power for 3 seconds and print summary."""
    import json
    print("Sampling tegrastats for 3 seconds (idle)...")
    with TegraStatsCollector(interval_ms=100) as col:
        time.sleep(3)
    print(json.dumps(col.summary(), indent=2))


if __name__ == "__main__":
    main()
