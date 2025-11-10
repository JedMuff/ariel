"""Memory profiling utilities for debugging memory and performance issues.

This module provides tools to track memory usage, identify leaks, and profile
multiprocessing behavior during evolution runs.
"""

from __future__ import annotations

import gc
import os
import psutil
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


class MemoryProfiler:
    """Profile memory usage throughout evolution."""

    def __init__(self, log_dir: Path | None = None, enable_tracemalloc: bool = True):
        """Initialize memory profiler.

        Parameters
        ----------
        log_dir : Path | None
            Directory to save profiling logs, by default None.
        enable_tracemalloc : bool
            Whether to enable tracemalloc for detailed allocation tracking, by default True.
        """
        self.log_dir = log_dir
        self.enable_tracemalloc = enable_tracemalloc
        self.process = psutil.Process(os.getpid())
        self.snapshots = []
        self.memory_log = []

        if self.enable_tracemalloc:
            tracemalloc.start()

        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.memory_log_file = log_dir / "memory_profile.csv"
            # Write header
            with open(self.memory_log_file, 'w') as f:
                f.write("timestamp,event,rss_mb,vms_mb,percent,available_mb,num_threads,temp_cache_files,temp_cache_mb\n")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current memory statistics.

        Returns
        -------
        dict
            Memory statistics including RSS, VMS, percent, available, etc.
        """
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        sys_mem = psutil.virtual_memory()
        num_threads = self.process.num_threads()

        # Check temp cache size
        temp_dir = Path("/tmp/ariel_cmaes_cache")
        temp_cache_files = 0
        temp_cache_mb = 0.0
        if temp_dir.exists():
            temp_cache_files = len(list(temp_dir.glob("**/*")))
            temp_cache_mb = sum(f.stat().st_size for f in temp_dir.glob("**/*") if f.is_file()) / (1024 * 1024)

        stats = {
            "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size
            "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size
            "percent": mem_percent,  # % of system memory
            "available_mb": sys_mem.available / (1024 * 1024),
            "num_threads": num_threads,
            "temp_cache_files": temp_cache_files,
            "temp_cache_mb": temp_cache_mb,
        }

        return stats

    def log_memory(self, event: str = "") -> dict[str, Any]:
        """Log current memory usage.

        Parameters
        ----------
        event : str
            Description of the event being logged.

        Returns
        -------
        dict
            Memory statistics at this point.
        """
        stats = self.get_memory_stats()
        timestamp = time.time()

        # Store in memory
        entry = {
            "timestamp": timestamp,
            "event": event,
            **stats,
        }
        self.memory_log.append(entry)

        # Write to file if log_dir is set
        if self.log_dir:
            with open(self.memory_log_file, 'a') as f:
                f.write(f"{timestamp},{event},{stats['rss_mb']:.2f},{stats['vms_mb']:.2f},"
                       f"{stats['percent']:.2f},{stats['available_mb']:.2f},{stats['num_threads']},"
                       f"{stats['temp_cache_files']},{stats['temp_cache_mb']:.2f}\n")

        return stats

    def take_snapshot(self, label: str = "") -> None:
        """Take a tracemalloc snapshot.

        Parameters
        ----------
        label : str
            Label for this snapshot.
        """
        if not self.enable_tracemalloc:
            return

        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))

    def compare_snapshots(self, idx1: int = -2, idx2: int = -1) -> list[Any]:
        """Compare two snapshots to find memory growth.

        Parameters
        ----------
        idx1 : int
            Index of first snapshot, by default -2 (second to last).
        idx2 : int
            Index of second snapshot, by default -1 (last).

        Returns
        -------
        list
            Top 10 differences between snapshots.
        """
        if not self.enable_tracemalloc or len(self.snapshots) < 2:
            return []

        label1, snap1 = self.snapshots[idx1]
        label2, snap2 = self.snapshots[idx2]

        top_stats = snap2.compare_to(snap1, 'lineno')
        return top_stats[:10]

    def print_snapshot_comparison(self, idx1: int = -2, idx2: int = -1) -> None:
        """Print comparison between two snapshots.

        Parameters
        ----------
        idx1 : int
            Index of first snapshot.
        idx2 : int
            Index of second snapshot.
        """
        if not self.enable_tracemalloc or len(self.snapshots) < 2:
            print("Not enough snapshots to compare")
            return

        label1, _ = self.snapshots[idx1]
        label2, _ = self.snapshots[idx2]

        print(f"\n=== Memory Growth: {label1} -> {label2} ===")
        top_stats = self.compare_snapshots(idx1, idx2)

        for stat in top_stats:
            print(f"{stat}")

    def save_snapshot_report(self, output_file: Path | None = None) -> None:
        """Save detailed snapshot analysis to file.

        Parameters
        ----------
        output_file : Path | None
            File to save report to. If None, uses log_dir/snapshot_report.txt.
        """
        if not self.enable_tracemalloc or len(self.snapshots) < 2:
            return

        if output_file is None and self.log_dir:
            output_file = self.log_dir / "snapshot_report.txt"

        if output_file is None:
            return

        with open(output_file, 'w') as f:
            for i in range(len(self.snapshots) - 1):
                label1, snap1 = self.snapshots[i]
                label2, snap2 = self.snapshots[i + 1]

                f.write(f"\n{'=' * 80}\n")
                f.write(f"Memory Growth: {label1} -> {label2}\n")
                f.write(f"{'=' * 80}\n\n")

                top_stats = snap2.compare_to(snap1, 'lineno')
                for stat in top_stats[:20]:
                    f.write(f"{stat}\n")

    def print_summary(self) -> None:
        """Print summary of memory usage throughout run."""
        if not self.memory_log:
            print("No memory logs recorded")
            return

        print("\n" + "=" * 80)
        print("MEMORY PROFILE SUMMARY")
        print("=" * 80)

        # Print key events
        key_events = [entry for entry in self.memory_log if entry['event']]
        print(f"\nKey Events ({len(key_events)}):")
        for entry in key_events:
            print(f"  {entry['event']:40s} | RSS: {entry['rss_mb']:8.2f} MB | "
                  f"Temp Cache: {entry['temp_cache_files']:4d} files ({entry['temp_cache_mb']:6.2f} MB)")

        # Print statistics
        rss_values = [e['rss_mb'] for e in self.memory_log]
        print(f"\nRSS Statistics:")
        print(f"  Initial: {rss_values[0]:.2f} MB")
        print(f"  Final:   {rss_values[-1]:.2f} MB")
        print(f"  Peak:    {max(rss_values):.2f} MB")
        print(f"  Growth:  {rss_values[-1] - rss_values[0]:.2f} MB")

        # Check for temp cache accumulation
        temp_cache_values = [e['temp_cache_mb'] for e in self.memory_log]
        if max(temp_cache_values) > 100:
            print(f"\n⚠️  WARNING: Temp cache grew to {max(temp_cache_values):.2f} MB")

        print("=" * 80 + "\n")

    def force_gc(self) -> dict[str, Any]:
        """Force garbage collection and return stats.

        Returns
        -------
        dict
            Statistics about garbage collection.
        """
        stats_before = self.get_memory_stats()
        collected = gc.collect()
        stats_after = self.get_memory_stats()

        return {
            "collected": collected,
            "freed_mb": stats_before['rss_mb'] - stats_after['rss_mb'],
            "before": stats_before,
            "after": stats_after,
        }


@contextmanager
def profile_memory(profiler: MemoryProfiler, event: str):
    """Context manager for profiling a code block.

    Parameters
    ----------
    profiler : MemoryProfiler
        The profiler instance.
    event : str
        Description of the code block.

    Yields
    ------
    None
    """
    profiler.log_memory(f"START: {event}")
    profiler.take_snapshot(f"before_{event}")

    try:
        yield
    finally:
        profiler.take_snapshot(f"after_{event}")
        profiler.log_memory(f"END: {event}")


def check_temp_cache_status(cache_dir: Path | str | None = None) -> dict[str, Any]:
    """Check status of temporary CMA-ES cache.

    Parameters
    ----------
    cache_dir : Path | str | None
        Cache directory to check. If None, uses default /tmp/ariel_cmaes_cache

    Returns
    -------
    dict
        Information about temp cache (num files, total size, etc.)
    """
    if cache_dir is None:
        temp_dir = Path("/tmp/ariel_cmaes_cache")
    else:
        temp_dir = Path(cache_dir)

    if not temp_dir.exists():
        return {
            "exists": False,
            "num_files": 0,
            "total_mb": 0.0,
            "files": [],
        }

    files = list(temp_dir.glob("**/*"))
    file_info = []
    total_bytes = 0

    for f in files:
        if f.is_file():
            size = f.stat().st_size
            total_bytes += size
            file_info.append({
                "path": str(f),
                "size_mb": size / (1024 * 1024),
                "modified": f.stat().st_mtime,
            })

    return {
        "exists": True,
        "num_files": len(file_info),
        "total_mb": total_bytes / (1024 * 1024),
        "files": sorted(file_info, key=lambda x: x['size_mb'], reverse=True)[:10],  # Top 10 largest
    }


def cleanup_temp_cache(cache_dir: Path | str | None = None) -> int:
    """Clean up temporary CMA-ES cache.

    Parameters
    ----------
    cache_dir : Path | str | None
        Cache directory to clean. If None, uses default /tmp/ariel_cmaes_cache

    Returns
    -------
    int
        Number of files removed.
    """
    import shutil
    if cache_dir is None:
        temp_dir = Path("/tmp/ariel_cmaes_cache")
    else:
        temp_dir = Path(cache_dir)

    if not temp_dir.exists():
        return 0

    files_before = len(list(temp_dir.glob("**/*")))

    # Remove cache directory - if this fails, we want to know about it
    shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    return files_before
