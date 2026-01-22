"""
Modelos de entrada/request para el sistema TFM.

Define estructuras para:
- Queries del usuario
- Requests a tools
- Parámetros de filtrado
"""

from typing import Optional, Any, Literal
from datetime import date

from pydantic import BaseModel, Field, field_validator


class FilterParams(BaseModel):
    """
    Parámetros de filtrado para consultas y tools.
    
    Soporta filtros por:
    - Rango de fechas (solo Yelp y Olist)
    - Categorías
    - IDs de entidades
    - Idioma
    - Estrellas
    - Flag de ambigüedad (stars==3)
    
    Example:
        >>> filters = FilterParams(
        ...     date_start="2022-01-01",
        ...     date_end="2022-12-31",
        ...     categories=["Restaurants", "Food"],
        ...     is_ambiguous=False
        ... )
    """
    
    # Rango de fechas (formato: YYYY-MM-DD)
    date_start: Optional[str] = Field(
        default=None,
        description="Fecha inicio (YYYY-MM-DD). Solo Yelp y Olist."
    )
    date_end: Optional[str] = Field(
        default=None,
        description="Fecha fin (YYYY-MM-DD). Solo Yelp y Olist."
    )
    
    # Filtros de categorías
    categories: Optional[list[str]] = Field(
        default=None,
        description="Lista de categorías a incluir"
    )
    exclude_categories: Optional[list[str]] = Field(
        default=None,
        description="Lista de categorías a excluir"
    )
    
    # Filtros de entidades
    entity_ids: Optional[list[str]] = Field(
        default=None,
        description="Lista de IDs específicos (business_id, product_id, etc.)"
    )
    
    # Filtros de rating
    stars: Optional[list[int]] = Field(
        default=None,
        description="Lista de valores de stars a incluir (1-5)"
    )
    min_stars: Optional[int] = Field(default=None, ge=1, le=5)
    max_stars: Optional[int] = Field(default=None, ge=1, le=5)
    
    # Flag de ambigüedad
    is_ambiguous: Optional[bool] = Field(
        default=None,
        description="True para filtrar solo stars==3, False para excluirlas"
    )
    
    # Idioma
    language: Optional[str] = Field(
        default=None,
        description="Código de idioma (en, es, pt)"
    )
    
    # Límites
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Límite de filas a retornar"
    )
    
    @field_validator("date_start", "date_end", mode="before")
    @classmethod
    def validate_date_format(cls, v: Any) -> Optional[str]:
        """Valida formato de fecha YYYY-MM-DD."""
        if v is None:
            return None
        if isinstance(v, date):
            return v.isoformat()
        if isinstance(v, str):
            # Validar formato básico
            try:
                date.fromisoformat(v)
                return v
            except ValueError:
                raise ValueError(f"Fecha inválida: {v}. Usar formato YYYY-MM-DD")
        raise ValueError(f"Tipo de fecha no soportado: {type(v)}")
    
    def to_filter_expression(self, date_col: str = "date") -> str:
        """
        Genera expresión de filtro para Polars.
        
        Args:
            date_col: Nombre de la columna de fecha
            
        Returns:
            String con condiciones de filtro
        """
        # TODO: Implementar generación de cláusula WHERE
        raise NotImplementedError("Implementar en Fase 2")
    
    def to_polars_filter(self) -> Any:
        """
        Genera expresión de filtro para Polars.
        
        Returns:
            Expresión Polars para usar en .filter()
            
        TODO: Fase 2 - Implementar filtro Polars
        """
        # TODO: Implementar filtro Polars
        raise NotImplementedError("Implementar en Fase 2")


class UserQuery(BaseModel):
    """
    Query del usuario para el sistema conversacional.
    
    El Router recibe esta estructura y genera un QueryPlan.
    
    Example:
        >>> query = UserQuery(
        ...     text="¿Cuál es el sentimiento promedio de restaurantes en 2022?",
        ...     preferred_dataset="yelp"
        ... )
    """
    
    text: str = Field(
        description="Pregunta del usuario en lenguaje natural"
    )
    preferred_dataset: Optional[Literal["yelp", "es", "olist"]] = Field(
        default=None,
        description="Dataset preferido si el usuario lo especifica"
    )
    filters: Optional[FilterParams] = Field(
        default=None,
        description="Filtros adicionales si el usuario los especifica"
    )
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Contexto adicional de la conversación"
    )


class ToolRequest(BaseModel):
    """
    Request genérico para llamar a una tool.
    
    Usado internamente por los agentes para invocar tools.
    
    Example:
        >>> request = ToolRequest(
        ...     tool_name="aggregation",
        ...     dataset="yelp",
        ...     operation="avg_sentiment_by_month",
        ...     filters=FilterParams(date_start="2022-01-01")
        ... )
    """
    
    tool_name: str = Field(description="Nombre de la tool a ejecutar")
    dataset: str = Field(description="Dataset objetivo")
    operation: str = Field(description="Operación específica de la tool")
    filters: Optional[FilterParams] = Field(default=None)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parámetros adicionales específicos de la tool"
    )


class ProfileRequest(BaseModel):
    """
    Request para profiling de un dataset.
    
    El Router usa esto para inspeccionar qué columnas y valores
    están disponibles en un dataset.
    """
    
    dataset: Literal["yelp", "es", "olist"] = Field(
        description="Dataset a perfilar"
    )
    layer: Literal["bronze", "silver", "gold"] = Field(
        default="silver",
        description="Capa de storage a inspeccionar"
    )
    include_stats: bool = Field(
        default=True,
        description="Incluir estadísticas básicas (min, max, count, nulls)"
    )
    sample_values: bool = Field(
        default=True,
        description="Incluir valores de ejemplo por columna"
    )
    sample_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Cantidad de valores de ejemplo"
    )
