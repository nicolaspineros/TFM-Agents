"""
Tools deterministas para el sistema TFM.

Estas herramientas son ejecutadas por los agentes y operan sobre
storage (Parquet + DuckDB). El LLM no procesa datos masivos; solo
invoca estas tools que devuelven resultados pequenos y estructurados.

Modulos:
- storage: Gestion de DuckDB y registro de tablas
- io_loaders: Carga de datos bronze (JSONL, CSV)
- preprocess: Limpieza y normalizacion (bronze -> silver)
- profiling: Inspeccion de schemas y estadisticas
- sentiment: Analisis de sentimiento (baseline + modelos)
- nlp_utils: Utilidades NLP (limpieza, VADER, influence score)
- aggregations: Consultas SQL y agregaciones
- features: Construccion de features gold

Principios de diseno:
- Tools NO importan agents ni graphs (evitar dependencias circulares)
- Tools son deterministas y reproducibles
- Tools persisten resultados y registran en DuckDB
- Tools soportan filtrado dinamico (FilterParams)
"""

from tfm.tools.storage import (
    get_duckdb_connection,
    register_parquet_table,
    list_registered_tables,
    table_exists,
    execute_query,
    register_all_silver_tables,
)
from tfm.tools.io_loaders import (
    load_yelp_reviews,
    load_yelp_users,
    load_yelp_business,
    load_es_reviews,
    load_olist_data,
    load_olist_all,
    get_bronze_file_info,
    check_bronze_files,
)
from tfm.tools.preprocess import (
    build_silver_yelp,
    build_silver_yelp_users,
    build_silver_yelp_business,
    build_silver_es,
    build_silver_olist,
    build_all_silver,
    check_silver_status,
)
from tfm.tools.profiling import (
    profile_dataset,
    get_column_stats,
    get_available_values,
    get_date_range,
    check_artifacts_status,
)
from tfm.tools.nlp_utils import (
    clean_text_basic,
    clean_text_spanish,
    clean_text_portuguese,
    compute_user_influence_score,
    extract_text_features,
    analyze_star_distribution,
)
# Nota: Las funciones compute_sentiment_* se importan desde sentiment.py
# no desde nlp_utils.py para evitar conflictos
from tfm.tools.sentiment import (
    compute_sentiment_baseline,
    compute_sentiment_vader,
    compute_sentiment_combined,
    compute_sentiment_batch,
    classify_ambiguous_reviews,
    extract_text_features,
    compute_user_influence_score,
)
from tfm.tools.aggregations import (
    aggregate_reviews_by_month,
    aggregate_reviews_by_stars,
    aggregate_olist_sales_by_month,
    aggregate_olist_by_category,
    aggregate_yelp_user_stats,
    aggregate_by_text_length,
    aggregate_ambiguous_reviews,
    aggregate_business_stats,
    correlate_reviews_business,
    aggregate_olist_reviews_sales,
    get_distribution,
    get_top_entities,
)
from tfm.tools.features import (
    build_gold_features,
    build_olist_sales_features,
    build_yelp_user_features,
    get_feature_stats,
    get_gold_status,
)
from tfm.tools.analysis_tools import (
    get_all_tools,
    get_tools_summary,
    get_reviews_distribution,
    get_reviews_by_month,
    get_sales_by_month,
    get_user_stats,
    get_business_stats,
    get_ambiguous_reviews_analysis,
    get_text_length_analysis,
    get_sales_by_category,
    get_reviews_sales_correlation,
    get_dataset_status,
    build_dataset_silver,
)

__all__ = [
    # Storage
    "get_duckdb_connection",
    "register_parquet_table",
    "list_registered_tables",
    "table_exists",
    "execute_query",
    "register_all_silver_tables",
    # Loaders
    "load_yelp_reviews",
    "load_yelp_users",
    "load_yelp_business",
    "load_es_reviews",
    "load_olist_data",
    "load_olist_all",
    "get_bronze_file_info",
    "check_bronze_files",
    # Preprocess
    "build_silver_yelp",
    "build_silver_yelp_users",
    "build_silver_yelp_business",
    "build_silver_es",
    "build_silver_olist",
    "build_all_silver",
    "check_silver_status",
    # Profiling
    "profile_dataset",
    "get_column_stats",
    "get_available_values",
    "get_date_range",
    "check_artifacts_status",
    # NLP Utils
    "clean_text_basic",
    "clean_text_spanish",
    "clean_text_portuguese",
    "compute_user_influence_score",
    "extract_text_features",
    "analyze_star_distribution",
    # Sentiment
    "compute_sentiment_baseline",
    "compute_sentiment_vader",
    "compute_sentiment_combined",
    "compute_sentiment_batch",
    "classify_ambiguous_reviews",
    "extract_text_features",
    "compute_user_influence_score",
    # Aggregations
    "aggregate_reviews_by_month",
    "aggregate_reviews_by_stars",
    "aggregate_olist_sales_by_month",
    "aggregate_olist_by_category",
    "aggregate_yelp_user_stats",
    "aggregate_by_text_length",
    "aggregate_ambiguous_reviews",
    "aggregate_business_stats",
    "correlate_reviews_business",
    "aggregate_olist_reviews_sales",
    "get_distribution",
    "get_top_entities",
    # Features
    "build_gold_features",
    "build_olist_sales_features",
    "build_yelp_user_features",
    "get_feature_stats",
    "get_gold_status",
    # Analysis Tools (para LLM con bind_tools)
    "get_all_tools",
    "get_tools_summary",
    "get_reviews_distribution",
    "get_reviews_by_month",
    "get_sales_by_month",
    "get_user_stats",
    "get_business_stats",
    "get_ambiguous_reviews_analysis",
    "get_text_length_analysis",
    "get_sales_by_category",
    "get_reviews_sales_correlation",
    "get_dataset_status",
    "build_dataset_silver",
]
