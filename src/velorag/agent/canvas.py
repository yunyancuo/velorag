from __future__ import annotations

from typing import Optional

from velorag.agent.component import BaseComponent, AgentContext, BeginComponent


class AgentCanvas:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._begin: BeginComponent = BeginComponent()
        self._current: Optional[BaseComponent] = self._begin

    def add_component(self, component: BaseComponent) -> BaseComponent:
        self._current.then(component)
        self._current = component
        return component

    async def run(self, query: str, **kwargs) -> AgentContext:
        context = AgentContext(query=query, params=kwargs)
        current = self._begin
        while current is not None:
            context = await current.execute(context)
            current = current.next
        return context

    def get_components(self) -> list[BaseComponent]:
        components = []
        current = self._begin
        while current is not None:
            components.append(current)
            current = current.next
        return components
