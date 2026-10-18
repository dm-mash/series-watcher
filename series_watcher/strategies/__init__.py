"""Рекомендатели: один интерфейс, разные реализации."""

from .base import Recommendation, Request, Strategy
from .crew import CrewStrategy
from .registry import build, get, names, register

__all__ = [
    "CrewStrategy",
    "Recommendation",
    "Request",
    "Strategy",
    "build",
    "get",
    "names",
    "register",
]
