"""
Modelos de salida/output para el sistema TFM.

Define estructuras para:
- Reportes de insights
- Resultados de agregaciones
- Resultados de sentimiento
- Resultados de QA
"""

from typing import Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """
    Resultado de análisis de sentimiento para una review.
    
    Attributes:
        review_id: ID de la review
        sentiment_score: Score numérico (-1 a 1)
        sentiment_label: Etiqueta (positive, negative, neutral)
        confidence: Confianza del modelo (0-1)
        is_ambiguous: Flag para stars==3
        model_version: Versión del modelo usado
    """
    
    review_id: str
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: str = Field(pattern="^(positive|negative|neutral)$")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    is_ambiguous: bool = False
    model_version: str = "baseline_v1"


class AspectResult(BaseModel):
    """
    Resultado de extracción de aspectos para una review.
    
    TODO: Fase 5 - Implementar extracción de aspectos
    """
    
    review_id: str
    aspects: list[str] = Field(default_factory=list)
    aspect_sentiments: dict[str, float] = Field(default_factory=dict)
    model_version: str = "baseline_v1"


class AggregationResult(BaseModel):
    """
    Resultado de una agregación con Polars.
    
    Contiene datos tabulares serializados para consumo por el Synthesizer.
    
    Example:
        >>> result = AggregationResult(
        ...     aggregation_type="avg_sentiment_by_month",
        ...     dataset="yelp",
        ...     columns=["year", "month", "avg_sentiment", "review_count"],
        ...     data=[
        ...         {"year": 2022, "month": 1, "avg_sentiment": 0.65, "review_count": 1500},
        ...         {"year": 2022, "month": 2, "avg_sentiment": 0.68, "review_count": 1420},
        ...     ],
        ...     row_count=12
        ... )
    """
    
    aggregation_type: str = Field(description="Tipo de agregación ejecutada")
    dataset: str = Field(description="Dataset de origen")
    columns: list[str] = Field(description="Nombres de columnas")
    data: list[dict[str, Any]] = Field(description="Datos como lista de dicts")
    row_count: int = Field(description="Número de filas")
    
    # Metadata
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float = Field(default=0.0)
    source_artifact: Optional[str] = Field(default=None)
    
    # Para tablas grandes, puede incluir resumen
    summary_stats: Optional[dict[str, Any]] = Field(default=None)


class InsightBullet(BaseModel):
    """Un bullet point individual en el reporte de insights."""
    
    text: str = Field(description="Texto del insight")
    evidence: Optional[str] = Field(
        default=None,
        description="Evidencia que soporta el insight (números, porcentajes)"
    )
    confidence: str = Field(
        default="medium",
        pattern="^(high|medium|low)$",
        description="Nivel de confianza"
    )


class InsightsReport(BaseModel):
    """
    Reporte estructurado de insights generado por el Synthesizer.
    
    Este es el output principal del sistema conversacional.
    Incluye resumen, bullets, caveats y metadata para trazabilidad.
    
    Example:
        >>> report = InsightsReport(
        ...     summary="El sentimiento general es positivo con tendencia al alza.",
        ...     bullets=[
        ...         InsightBullet(text="65% de reviews son positivas", evidence="avg_sentiment=0.65"),
        ...         InsightBullet(text="Restaurantes lideran en satisfacción", evidence="category_rank=1"),
        ...     ],
        ...     caveats=["Dataset no incluye reviews de diciembre 2023"],
        ...     query_answered="¿Cuál es el sentimiento general de los restaurantes?"
        ... )
    """
    
    # Contenido principal
    summary: str = Field(
        description="Resumen ejecutivo (1-2 oraciones)"
    )
    bullets: list[InsightBullet] = Field(
        default_factory=list,
        description="Lista de insights específicos"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Advertencias y limitaciones"
    )
    
    # Contexto
    query_answered: str = Field(
        description="La pregunta que se respondió"
    )
    datasets_used: list[str] = Field(
        default_factory=list,
        description="Datasets consultados"
    )
    
    # Trazabilidad
    artifacts_used: list[str] = Field(
        default_factory=list,
        description="Paths de artefactos consultados"
    )
    aggregations_run: list[str] = Field(
        default_factory=list,
        description="Agregaciones ejecutadas"
    )
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    model_used: str = Field(default="gpt-4o-mini")
    
    def to_markdown(self) -> str:
        """
        Convierte el reporte a formato Markdown.
        
        Returns:
            String con el reporte formateado
        """
        lines = [
            f"## {self.summary}",
            "",
            "### Insights",
        ]
        
        for bullet in self.bullets:
            evidence = f" _{bullet.evidence}_" if bullet.evidence else ""
            lines.append(f"- {bullet.text}{evidence}")
        
        if self.caveats:
            lines.extend(["", "### ⚠️ Caveats"])
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
        
        lines.extend([
            "",
            "---",
            f"_Datasets: {', '.join(self.datasets_used)}_",
            f"_Generado: {self.generated_at.isoformat()}_",
        ])
        
        return "\n".join(lines)


class QACheck(BaseModel):
    """Resultado de un check individual de QA."""
    
    check_name: str = Field(description="Nombre del check")
    passed: bool = Field(description="Si el check pasó")
    message: str = Field(description="Mensaje descriptivo")
    severity: str = Field(
        default="warning",
        pattern="^(error|warning|info)$"
    )


class QAResult(BaseModel):
    """
    Resultado del proceso de QA/validación.
    
    El QA Evaluator genera este resultado después de validar
    el InsightsReport contra los datos de origen.
    
    Checks incluidos:
    - Schema validation
    - Non-empty results
    - Faithfulness (claims vs evidence)
    - Numeric consistency
    """
    
    passed: bool = Field(description="Si el QA general pasó")
    checks: list[QACheck] = Field(default_factory=list)
    
    # Métricas de faithfulness
    faithfulness_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Score de faithfulness (0-1)"
    )
    
    # Feedback para mejora
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback si hay problemas a corregir"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Sugerencias de mejora"
    )
    
    # Si necesita regenerar
    needs_regeneration: bool = Field(
        default=False,
        description="Si el Synthesizer debe regenerar el reporte"
    )
    regeneration_hints: list[str] = Field(
        default_factory=list,
        description="Hints para regeneración"
    )
    
    def summary(self) -> str:
        """Resumen del QA en una línea."""
        passed_count = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"{status} ({passed_count}/{total} checks)"


class PredictionResult(BaseModel):
    """
    Resultado de predicción de ventas (Olist).
    
    TODO: Fase 5 - Implementar modelo de predicción
    """
    
    date_range: tuple[str, str] = Field(description="Rango de fechas predicho")
    category: Optional[str] = Field(default=None)
    
    predictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Lista de predicciones por fecha"
    )
    
    # Métricas del modelo
    model_version: str = "baseline_v1"
    mae: Optional[float] = Field(default=None, description="Mean Absolute Error")
    rmse: Optional[float] = Field(default=None, description="Root Mean Square Error")
    
    # Metadata
    features_used: list[str] = Field(default_factory=list)
    training_period: Optional[tuple[str, str]] = Field(default=None)
