"""
Schemas Pydantic para el sistema TFM.

Módulos:
- state: Estado compartido de los grafos (TFMState)
- request: Modelos de entrada (UserQuery, ToolRequest)
- outputs: Modelos de salida (InsightsReport, AggregationResult)
"""

from tfm.schemas.state import TFMState
from tfm.schemas.request import UserQuery, ToolRequest, FilterParams
from tfm.schemas.outputs import (
    InsightsReport,
    AggregationResult,
    SentimentResult,
    QAResult,
)

__all__ = [
    "TFMState",
    "UserQuery",
    "ToolRequest",
    "FilterParams",
    "InsightsReport",
    "AggregationResult",
    "SentimentResult",
    "QAResult",
]
