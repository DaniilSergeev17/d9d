"""Provides communication strategies for Mixture-of-Experts routing operations."""

from .base import ExpertCommunicationHandler
from .naive import NoCommunicationHandler

__all__ = [
    "ExpertCommunicationHandler",
    "NoCommunicationHandler",
]
