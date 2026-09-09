"""Deterministic retrieval of research knowledge records."""

from .loader import load_research_records
from .retrieval import retrieve

__all__ = ["load_research_records", "retrieve"]
