"""Timer utility for profiling code execution"""
import time
import torch


class Timer:
    """Simple timer context manager for profiling"""
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f"  ⏱️  {self.name}: {self.elapsed:.3f}s")
