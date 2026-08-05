"""Computational campaigns behind the χ(ℝⁿ) upper bounds.

`core` holds modules shared by several campaigns; `campaigns` holds the
individual runs.  Every artifact path goes through :mod:`chromatic_research.paths`
so that nothing depends on a checkout location.
"""

from . import paths

__all__ = ["paths"]
