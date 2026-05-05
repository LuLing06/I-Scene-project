"""Lightweight TRELLIS components used by IScene inference.

Subpackages are imported by their direct users. Keeping this package init small
avoids importing optional rendering dependencies for Gaussian-only inference.
"""

__all__ = ["models", "modules", "pipelines", "renderers", "representations", "utils"]
