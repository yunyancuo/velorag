from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    def __init__(self):
        self._components: dict[str, dict[str, Any]] = {}

    def register(self, category: str, name: str, factory: Callable[..., Any]):
        if category not in self._components:
            self._components[category] = {}
        self._components[category][name] = factory

    def get(self, category: str, name: str) -> Callable[..., Any] | None:
        return self._components.get(category, {}).get(name)

    def create(self, category: str, name: str, **kwargs) -> Any:
        factory = self.get(category, name)
        if factory is None:
            raise ValueError(f"Component not found: {category}/{name}")
        return factory(**kwargs)

    def list(self, category: str | None = None) -> dict[str, list[str]]:
        if category:
            return {category: list(self._components.get(category, {}).keys())}
        return {cat: list(components.keys()) for cat, components in self._components.items()}
