"""Рекомендатели: один интерфейс, разные реализации."""

from .base import Recommendation, Request, Strategy, StrategyError
from .crew import CrewStrategy
from .model import ModelStrategy
from .registry import build, get, names, register

__all__ = [
    "CrewStrategy",
    "ModelStrategy",
    "Recommendation",
    "Request",
    "Strategy",
    "StrategyError",
    "build",
    "get",
    "names",
    "register",
]
