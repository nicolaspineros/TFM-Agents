"""
Profiling de datasets para el Router.

Este modulo permite al Router inspeccionar:
- Que columnas tiene cada dataset
- Que valores unicos hay (categorias, fechas, etc.)
- Estadisticas basicas (min, max, nulls)
- Si un artefacto existe o necesita construirse

El profiling es clave para que el Router pueda:
- Validar si una query es posible
- Sugerir filtros validos al usuario
- Detectar limitaciones (ej: dataset sin fecha)
"""

from pathlib import Path
from typing import Optional, Any, Literal, Dict, List

import polars as pl

from tfm.config.settings import get_settings, SILVER_FILES, GOLD_FILES


def profile_dataset(
    dataset: Literal["yelp", "yelp_users", "es", "olist", "olist_orders"],
    layer: Literal["bronze", "silver", "gold"] = "silver",
    include_stats: bool = True,
    sample_values: bool = True,
    sample_size: int = 5
) -> Dict[str, Any]:
    """
    Genera perfil completo de un dataset.
    
    El perfil incluye:
    - Lista de columnas con tipos
    - Estadisticas basicas por columna
    - Valores de ejemplo
    - Flags de capacidades (has_date, has_category, etc.)
    
    Args:
        dataset: Dataset a perfilar
        layer: Capa de storage (bronze, silver, gold)
        include_stats: Incluir estadisticas (min, max, nulls)
        sample_values: Incluir valores de ejemplo
        sample_size: Cantidad de valores de ejemplo
        
    Returns:
        Dict con perfil completo
        
    Example:
        >>> profile = profile_dataset("yelp", layer="silver")
        >>> print(profile["columns"])
        ['review_id', 'business_id', 'stars', 'text', 'date', ...]
    """
    settings = get_settings()
    
    # Mapeo de dataset a archivo silver
    silver_mapping = {
        "yelp": "yelp_reviews",
        "yelp_users": "yelp_users",
        "es": "es",
        "olist": "olist_reviews",
        "olist_orders": "olist_orders",
    }
    
    if layer == "silver":
        if dataset not in silver_mapping:
            raise ValueError(f"Dataset invalido: {dataset}")
        file_key = silver_mapping[dataset]
        if file_key not in SILVER_FILES:
            raise ValueError(f"Archivo silver no configurado para: {file_key}")
        file_path = settings.silver_dir / SILVER_FILES[file_key]
    else:
        raise NotImplementedError(f"Layer {layer} no implementado aun")
    
    if not file_path.exists():
        return {
            "exists": False,
            "path": str(file_path),
            "message": f"Archivo no encontrado. Ejecutar build_silver primero.",
        }
    
    # Leer parquet
    df = pl.read_parquet(file_path)
    
    profile = {
        "exists": True,
        "path": str(file_path),
        "row_count": df.height,
        "columns": df.columns,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    
    # Estadisticas
    if include_stats:
        profile["stats"] = {}
        for col in df.columns:
            col_stats = {
                "null_count": df[col].null_count(),
                "null_pct": round(df[col].null_count() / df.height * 100, 2),
            }
            
            # Estadisticas numericas
            if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Int16, pl.Int8]:
                try:
                    col_stats.update({
                        "min": df[col].min(),
                        "max": df[col].max(),
                        "mean": round(df[col].mean(), 2) if df[col].mean() else None,
                    })
                except Exception:
                    pass
            
            profile["stats"][col] = col_stats
    
    # Valores de ejemplo
    if sample_values:
        profile["sample_values"] = {}
        for col in df.columns:
            try:
                unique = df[col].unique().head(sample_size).to_list()
                profile["sample_values"][col] = unique
            except Exception:
                profile["sample_values"][col] = []
    
    # Capacidades
    profile["capabilities"] = {
        "has_date": "date" in df.columns or "purchase_date" in df.columns,
        "has_category": "category" in df.columns or "categories" in df.columns,
        "has_text": "text" in df.columns,
        "has_stars": "stars" in df.columns or "review_score" in df.columns,
        "has_entity_id": "business_id" in df.columns or "product_id" in df.columns or "order_id" in df.columns,
        "has_user_id": "user_id" in df.columns or "reviewer_id" in df.columns or "customer_id" in df.columns,
    }
    
    return profile


def get_column_stats(
    dataset: str,
    column: str,
    layer: str = "silver"
) -> Dict[str, Any]:
    """
    Obtiene estadisticas detalladas de una columna especifica.
    
    Args:
        dataset: Dataset
        column: Nombre de la columna
        layer: Capa de storage
        
    Returns:
        Dict con estadisticas detalladas
    """
    profile = profile_dataset(dataset, layer=layer, include_stats=True, sample_values=True)
    
    if not profile.get("exists"):
        return {"error": profile.get("message", "Dataset no encontrado")}
    
    if column not in profile["columns"]:
        return {"error": f"Columna {column} no existe en {dataset}"}
    
    result = {
        "column": column,
        "dtype": profile["dtypes"].get(column),
        "stats": profile["stats"].get(column, {}),
        "sample_values": profile["sample_values"].get(column, []),
    }
    
    return result


def get_available_values(
    dataset: str,
    column: str,
    layer: str = "silver",
    limit: int = 100
) -> List[Any]:
    """
    Obtiene valores unicos disponibles para una columna.
    
    Util para que el Router sugiera filtros validos.
    
    Args:
        dataset: Dataset
        column: Nombre de la columna
        layer: Capa de storage
        limit: Maximo de valores a retornar
        
    Returns:
        Lista de valores unicos
        
    Example:
        >>> categories = get_available_values("yelp", "categories")
        >>> print(categories[:5])
        ['Restaurants', 'Food', 'Shopping', 'Hotels', 'Beauty & Spas']
    """
    settings = get_settings()
    
    # Mapeo de dataset a archivo silver
    silver_mapping = {
        "yelp": "yelp_reviews",
        "yelp_users": "yelp_users",
        "es": "es",
        "olist": "olist_reviews",
        "olist_orders": "olist_orders",
    }
    
    if dataset not in silver_mapping:
        return []
    
    file_key = silver_mapping[dataset]
    if file_key not in SILVER_FILES:
        return []
    
    file_path = settings.silver_dir / SILVER_FILES[file_key]
    
    if not file_path.exists():
        return []
    
    df = pl.read_parquet(file_path)
    
    if column not in df.columns:
        return []
    
    try:
        values = df[column].unique().head(limit).to_list()
        return values
    except Exception:
        return []


def get_date_range(
    dataset: str,
    layer: str = "silver"
) -> Optional[tuple]:
    """
    Obtiene rango de fechas disponible en un dataset.
    
    Args:
        dataset: Dataset
        layer: Capa de storage
        
    Returns:
        Tuple (min_date, max_date) en formato YYYY-MM-DD o None si no hay fecha
        
    Example:
        >>> get_date_range("yelp")
        ('2018-01-01', '2022-12-31')
        >>> get_date_range("es")
        None  # Dataset sin fecha
    """
    settings = get_settings()
    
    # Dataset ES no tiene fechas
    if dataset == "es":
        return None
    
    # Mapeo de dataset a archivo silver
    silver_mapping = {
        "yelp": "yelp_reviews",
        "olist": "olist_reviews",
        "olist_orders": "olist_orders",
    }
    
    if dataset not in silver_mapping:
        return None
    
    file_key = silver_mapping[dataset]
    file_path = settings.silver_dir / SILVER_FILES[file_key]
    
    if not file_path.exists():
        return None
    
    df = pl.read_parquet(file_path)
    
    # Buscar columna de fecha
    date_col = None
    for col in ["date", "purchase_date", "review_creation_date"]:
        if col in df.columns:
            date_col = col
            break
    
    if not date_col:
        return None
    
    try:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        # Formatear como string
        if hasattr(min_date, 'strftime'):
            min_str = min_date.strftime("%Y-%m-%d")
            max_str = max_date.strftime("%Y-%m-%d")
        else:
            min_str = str(min_date)
            max_str = str(max_date)
        
        return (min_str, max_str)
    except Exception:
        return None


def check_artifacts_status(
    dataset: str
) -> dict[str, bool]:
    """
    Verifica qué artefactos existen para un dataset.
    
    Args:
        dataset: Dataset a verificar
        
    Returns:
        Dict con status de cada artefacto
        
    Example:
        >>> status = check_artifacts_status("yelp")
        >>> print(status)
        {
            'bronze_exists': True,
            'silver_exists': True,
            'gold_features_exists': False,
            'aggregations_exist': False
        }
    """
    settings = get_settings()
    
    status = {
        "bronze_exists": False,
        "silver_exists": False,
        "gold_features_exists": False,
        "aggregations_exist": False,
    }
    
    # Check bronze
    if dataset == "yelp":
        bronze_path = settings.yelp_bronze_dir / "yelp_academic_dataset_review.json"
        silver_path = settings.silver_dir / SILVER_FILES["yelp"]
        gold_path = settings.gold_dir / GOLD_FILES.get("yelp_features", "")
    elif dataset == "es":
        bronze_path = settings.es_bronze_dir / "reviews_dataframe_completo.csv"
        silver_path = settings.silver_dir / SILVER_FILES["es"]
        gold_path = settings.gold_dir / GOLD_FILES.get("es_features", "")
    elif dataset == "olist":
        bronze_path = settings.olist_bronze_dir / "olist_order_reviews_dataset.csv"
        silver_path = settings.silver_dir / SILVER_FILES.get("olist_reviews", "")
        gold_path = settings.gold_dir / GOLD_FILES.get("olist_features", "")
    else:
        return status
    
    status["bronze_exists"] = bronze_path.exists()
    status["silver_exists"] = silver_path.exists()
    status["gold_features_exists"] = gold_path.exists() if gold_path else False
    
    # Check aggregations
    agg_path = settings.gold_dir / f"agg_sentiment_by_month_{dataset}.parquet"
    status["aggregations_exist"] = agg_path.exists()
    
    return status
