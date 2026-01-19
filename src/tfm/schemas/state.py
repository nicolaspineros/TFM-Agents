"""
Estado compartido para los grafos LangGraph.

TFMState es el estado principal que fluye entre nodos de los grafos.
Contiene:
- Mensajes de la conversación
- Artefactos disponibles (paths a archivos procesados)
- Último resultado de consulta
- Metadata del routing
"""

from typing import Optional, Any
from datetime import datetime

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class ArtifactInfo(BaseModel):
    """Información sobre un artefacto persistido (Parquet, JSON, etc.)."""
    
    path: str = Field(description="Ruta relativa al artefacto")
    dataset: str = Field(description="Dataset de origen (yelp, es, olist)")
    artifact_type: str = Field(description="Tipo: silver, gold, aggregation, model")
    created_at: datetime = Field(default_factory=datetime.now)
    row_count: Optional[int] = Field(default=None, description="Número de filas si aplica")
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryPlan(BaseModel):
    """
    Plan generado por el Router para responder una query.
    
    El Router analiza la pregunta y genera este plan que indica
    qué tools ejecutar y en qué orden.
    """
    
    needs_nlp_features: bool = Field(
        default=False,
        description="Si requiere calcular features NLP (sentimiento, aspectos)"
    )
    needs_aggregation: bool = Field(
        default=False,
        description="Si requiere calcular agregaciones SQL"
    )
    needs_prediction: bool = Field(
        default=False,
        description="Si requiere modelo de predicción (Olist ventas)"
    )
    datasets_required: list[str] = Field(
        default_factory=list,
        description="Lista de datasets necesarios (yelp, es, olist)"
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtros a aplicar (date_range, categories, etc.)"
    )
    aggregation_type: Optional[str] = Field(
        default=None,
        description="Tipo de agregacion si aplica (by_month, by_category, etc.)"
    )
    aggregations_needed: list[str] = Field(
        default_factory=list,
        description="Lista de tipos de agregacion a ejecutar"
    )
    missing_artifacts: list[str] = Field(
        default_factory=list,
        description="Artefactos que deben construirse antes de responder"
    )
    guardrail_warnings: list[str] = Field(
        default_factory=list,
        description="Advertencias sobre limitaciones (ej: dataset sin fecha)"
    )
    is_valid: bool = Field(
        default=True,
        description="Si la query es válida y puede responderse"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Razón de rechazo si is_valid=False"
    )


class LastQueryResult(BaseModel):
    """
    Resultado de la última consulta/agregación ejecutada.
    
    Contiene datos serializados para que el Synthesizer pueda
    generar insights sin acceder a archivos.
    """
    
    query_type: str = Field(description="Tipo: aggregation, features, prediction")
    data_json: str = Field(description="Datos serializados como JSON (tablas pequeñas)")
    columns: list[str] = Field(default_factory=list, description="Nombres de columnas")
    row_count: int = Field(default=0, description="Número de filas")
    execution_time_ms: float = Field(default=0.0, description="Tiempo de ejecución")
    source_artifact: Optional[str] = Field(default=None, description="Artefacto de origen")


class TFMState(MessagesState):
    """
    Estado principal del sistema TFM.
    
    Hereda de MessagesState (LangGraph) para mantener historial de mensajes.
    Añade campos específicos del TFM para tracking de artefactos y resultados.
    
    Este estado fluye entre todos los nodos de los grafos:
    - Router: escribe query_plan
    - NLP Worker: actualiza artifacts, escribe en last_result
    - Aggregator: escribe en last_result
    - Synthesizer: lee last_result, genera insights
    - QA: valida y puede modificar insights
    
    Attributes:
        user_query: La pregunta original del usuario
        query_plan: Plan generado por el Router
        artifacts: Diccionario de artefactos disponibles (path -> ArtifactInfo)
        last_result: Último resultado de query/agregación
        insights_report: Reporte final generado por el Synthesizer
        qa_passed: Si el QA aprobó el resultado
        qa_feedback: Feedback del QA si hay problemas
        current_dataset: Dataset activo en el contexto
        error: Mensaje de error si algo falló
    """
    
    # Query del usuario
    user_query: str = ""
    
    # Plan de ejecución
    query_plan: Optional[QueryPlan] = None
    
    # Artefactos disponibles (lazy loaded del storage)
    artifacts: dict[str, ArtifactInfo] = {}
    
    # Ultimo resultado de consulta
    last_result: Optional[LastQueryResult] = None
    
    # Resultados de agregaciones (dict de nombre -> resultado)
    aggregation_results: dict[str, Any] = {}
    
    # Reporte de insights (output del Synthesizer)
    insights_report: Optional[dict[str, Any]] = None
    
    # QA
    qa_passed: bool = False
    qa_feedback: Optional[str] = None
    
    # Contexto
    current_dataset: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None


# === Estados específicos para sub-grafos ===

class NLPGraphState(MessagesState):
    """
    Estado para el grafo de NLP (construcción de features).
    
    Usado por nlp_graph para procesar reviews y generar silver/gold.
    """
    
    dataset: str = ""
    bronze_path: Optional[str] = None
    silver_path: Optional[str] = None
    gold_path: Optional[str] = None
    
    # Contadores de progreso
    total_reviews: int = 0
    processed_reviews: int = 0
    
    # Features calculadas
    features_computed: list[str] = []
    
    # Errores
    errors: list[str] = []


class PredictionGraphState(MessagesState):
    """
    Estado para el grafo de predicción (Olist ventas).
    
    Usado por prediction_graph para entrenar/inferir modelo de ventas.
    """
    
    # Datos de entrada
    orders_path: Optional[str] = None
    features_path: Optional[str] = None
    
    # Modelo
    model_path: Optional[str] = None
    model_version: Optional[str] = None
    
    # Predicciones
    predictions_path: Optional[str] = None
    
    # Métricas
    mae: Optional[float] = None
    rmse: Optional[float] = None
    
    # Errores
    error: Optional[str] = None


class EvaluationGraphState(MessagesState):
    """
    Estado para el grafo de evaluación offline.
    
    Usado por evaluation_graph para correr métricas ML y QA.
    """
    
    # Configuración
    eval_type: str = ""  # "ml_metrics", "qa_faithfulness", "langsmith_eval"
    dataset: Optional[str] = None
    
    # Resultados
    metrics: dict[str, float] = {}
    passed_checks: list[str] = []
    failed_checks: list[str] = []
    
    # LangSmith
    langsmith_run_id: Optional[str] = None
    
    # Errores
    error: Optional[str] = None
