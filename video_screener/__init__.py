"""video_screener — general-purpose AI video asset usability screening pipeline.

Frozen against taxonomy.md v0.3.1 (see ``taxonomy_schema.TAXONOMY_VERSION``).
"""

from .taxonomy_schema import TAXONOMY_VERSION

__version__ = "0.4.0"
__all__ = ["TAXONOMY_VERSION", "__version__"]
