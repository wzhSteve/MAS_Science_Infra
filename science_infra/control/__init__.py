"""Science Control Plane: experiment YAML, process manager, REST/SSE API."""

from .app import create_app

__all__ = ["create_app"]
