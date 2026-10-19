from velorag.core.config import VeloRAGConfig
from velorag.core.registry import ComponentRegistry
from velorag.core.settings import Settings
from velorag.pipeline import VeloRAG

settings = Settings()
registry = ComponentRegistry()

__version__ = "0.3.0"

__all__ = ["VeloRAG", "VeloRAGConfig", "__version__", "registry", "settings"]
