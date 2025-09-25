from ._version import __version__  # light import

# Avoid importing torch-consuming modules at build time
try:
    from .core import NormalizingFlow, MultiscaleFlow
except Exception:
    NormalizingFlow = None
    MultiscaleFlow = None

from . import flows, distributions, nets, sampling, utils  # optional; guard similarly if needed

__all__ = [
    "NormalizingFlow", "MultiscaleFlow",
    "flows", "distributions", "nets", "sampling", "utils",
    "__version__",
]