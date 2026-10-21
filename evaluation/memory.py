"""Read process working-set memory without adding optional dependencies."""
import os
from pathlib import Path
import sys


def process_memory():
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            return dict(process_rss_bytes=int(counters.WorkingSetSize),
                        process_lifetime_peak_rss_bytes=int(counters.PeakWorkingSetSize), source="Windows WorkingSetSize")
        if sys.platform.startswith("linux"):
            pages = int(Path("/proc/self/statm").read_text().split()[1])
            return dict(process_rss_bytes=pages * os.sysconf("SC_PAGE_SIZE"),
                        process_lifetime_peak_rss_bytes=None, source="Linux /proc/self/statm")
        return dict(process_rss_bytes=None, process_lifetime_peak_rss_bytes=None, source="unsupported_platform")
    except (OSError, ValueError, IndexError) as exc:
        return dict(process_rss_bytes=None, process_lifetime_peak_rss_bytes=None, source="unavailable", error=str(exc))


def gpu_memory():
    # Do not initialize CUDA or import Torch just to probe a blocked evaluation.
    torch = sys.modules.get("torch")
    if torch is None:
        return dict(gpu_bytes=None, gpu_reason="torch_not_loaded")
    if not torch.cuda.is_available():
        return dict(gpu_bytes=None, gpu_reason="cuda_unavailable")
    return dict(gpu_bytes=sum(torch.cuda.memory_allocated(device) for device in range(torch.cuda.device_count())),
                gpu_reserved_bytes=sum(torch.cuda.memory_reserved(device) for device in range(torch.cuda.device_count())),
                gpu_reason="torch_allocator_current_all_devices_not_total_driver_memory")
