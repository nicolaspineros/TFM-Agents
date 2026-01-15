"""
Cargadores de datos bronze (raw).

Este modulo maneja la lectura de archivos raw en diferentes formatos:
- JSONL (Yelp)
- CSV (ES, Olist)

Los loaders:
- Leen datos crudos sin transformar
- Soportan lectura lazy/streaming para archivos grandes
- Validan que los archivos existan
- NO transforman datos (eso es preprocess.py)
"""

from pathlib import Path
from typing import Iterator, Optional, Any, Dict, List

import polars as pl

from tfm.config.settings import get_settings, BRONZE_FILES


def load_yelp_reviews(
    limit: Optional[int] = None,
    streaming: bool = False
) -> pl.DataFrame:
    """
    Carga reviews de Yelp desde JSONL.
    
    El archivo yelp_academic_dataset_review.json es un JSONL
    (una linea JSON por review).
    
    Args:
        limit: Limite de filas a cargar (None = todas)
        streaming: Si usar lectura lazy/streaming
        
    Returns:
        DataFrame de Polars con las reviews
        
    Example:
        >>> df = load_yelp_reviews(limit=1000)
        >>> print(df.shape)
        (1000, 9)
    """
    settings = get_settings()
    file_path = settings.yelp_bronze_dir / BRONZE_FILES["yelp"]["reviews"]
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo Yelp reviews no encontrado: {file_path}. "
            "Asegurate de que el dataset este en data/bronze/yelp/"
        )
    
    if streaming:
        return pl.scan_ndjson(file_path)
    
    if limit:
        df = pl.read_ndjson(file_path, n_rows=limit)
    else:
        df = pl.read_ndjson(file_path)
    
    return df


def load_yelp_users(
    limit: Optional[int] = None
) -> pl.DataFrame:
    """
    Carga datos de usuarios de Yelp.
    
    Args:
        limit: Limite de filas a cargar
        
    Returns:
        DataFrame de Polars con los usuarios
    """
    settings = get_settings()
    file_path = settings.yelp_bronze_dir / BRONZE_FILES["yelp"]["users"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo Yelp users no encontrado: {file_path}")
    
    if limit:
        return pl.read_ndjson(file_path, n_rows=limit)
    return pl.read_ndjson(file_path)


def load_yelp_business() -> pl.DataFrame:
    """
    Carga datos de negocios de Yelp.
    
    Returns:
        DataFrame de Polars con los negocios
    """
    settings = get_settings()
    file_path = settings.yelp_bronze_dir / BRONZE_FILES["yelp"]["business"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo Yelp business no encontrado: {file_path}")
    
    return pl.read_ndjson(file_path)


def load_es_reviews(
    limit: Optional[int] = None
) -> pl.DataFrame:
    """
    Carga reviews en espanol desde CSV.
    
    Args:
        limit: Limite de filas a cargar
        
    Returns:
        DataFrame de Polars
        
    Example:
        >>> df = load_es_reviews(limit=5000)
    """
    settings = get_settings()
    file_path = settings.es_bronze_dir / BRONZE_FILES["es"]["reviews"]
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo ES reviews no encontrado: {file_path}. "
            "Asegurate de que el dataset este en data/bronze/rese_esp/"
        )
    
    # Intentar cargar con diferentes encodings
    try:
        df = pl.read_csv(file_path)
    except Exception:
        try:
            df = pl.read_csv(file_path, encoding='latin1')
        except Exception:
            df = pl.read_csv(file_path, encoding='utf-8', ignore_errors=True)
    
    if limit:
        df = df.head(limit)
    
    return df


def load_olist_data(
    table: str,
    limit: Optional[int] = None
) -> pl.DataFrame:
    """
    Carga una tabla especifica de Olist.
    
    Args:
        table: Nombre de la tabla (orders, items, reviews, products, etc.)
        limit: Limite de filas
        
    Returns:
        DataFrame de Polars
        
    Example:
        >>> orders = load_olist_data("orders", limit=10000)
        >>> reviews = load_olist_data("reviews")
    """
    if table not in BRONZE_FILES["olist"]:
        valid_tables = list(BRONZE_FILES["olist"].keys())
        raise ValueError(
            f"Tabla Olist invalida: {table}. "
            f"Tablas validas: {valid_tables}"
        )
    
    settings = get_settings()
    file_path = settings.olist_bronze_dir / BRONZE_FILES["olist"][table]
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Archivo Olist {table} no encontrado: {file_path}. "
            "Asegurate de que el dataset este en data/bronze/olist_ecommerce/"
        )
    
    df = pl.read_csv(file_path)
    if limit:
        df = df.head(limit)
    
    return df


def load_olist_all() -> Dict[str, pl.DataFrame]:
    """
    Carga todas las tablas de Olist.
    
    Returns:
        Dict con nombre de tabla -> DataFrame
    """
    tables = {}
    for table_name in BRONZE_FILES["olist"].keys():
        try:
            tables[table_name] = load_olist_data(table_name)
        except FileNotFoundError:
            tables[table_name] = None
    return tables


def iter_yelp_reviews(batch_size: int = 10000) -> Iterator[pl.DataFrame]:
    """
    Itera sobre reviews de Yelp en batches.
    
    Util para procesamiento de archivos muy grandes sin cargar
    todo en memoria.
    
    Args:
        batch_size: Tamano de cada batch
        
    Yields:
        DataFrames de Polars de tamano batch_size
        
    Example:
        >>> for batch in iter_yelp_reviews(batch_size=5000):
        ...     process_batch(batch)
    """
    settings = get_settings()
    file_path = settings.yelp_bronze_dir / BRONZE_FILES["yelp"]["reviews"]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo Yelp reviews no encontrado: {file_path}")
    
    # Usar scan para procesamiento lazy
    lazy_df = pl.scan_ndjson(file_path)
    
    # Obtener total de filas
    total_rows = lazy_df.select(pl.count()).collect().item()
    
    for offset in range(0, total_rows, batch_size):
        yield lazy_df.slice(offset, batch_size).collect()


def get_bronze_file_info(dataset: str, table: str = "reviews") -> Dict[str, Any]:
    """
    Obtiene informacion basica de un archivo bronze.
    
    Args:
        dataset: Dataset (yelp, es, olist)
        table: Tabla especifica
        
    Returns:
        Dict con path, exists, size_bytes
    """
    settings = get_settings()
    
    if dataset == "yelp":
        base_dir = settings.yelp_bronze_dir
        files = BRONZE_FILES["yelp"]
    elif dataset == "es":
        base_dir = settings.es_bronze_dir
        files = BRONZE_FILES["es"]
    elif dataset == "olist":
        base_dir = settings.olist_bronze_dir
        files = BRONZE_FILES["olist"]
    else:
        raise ValueError(f"Dataset invalido: {dataset}")
    
    if table not in files:
        raise ValueError(f"Tabla {table} no existe en dataset {dataset}")
    
    file_path = base_dir / files[table]
    
    info = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "size_bytes": 0,
        "size_mb": 0,
    }
    
    if file_path.exists():
        size = file_path.stat().st_size
        info["size_bytes"] = size
        info["size_mb"] = round(size / (1024 * 1024), 2)
    
    return info


def check_bronze_files() -> Dict[str, Dict[str, bool]]:
    """
    Verifica que todos los archivos bronze existan.
    
    Returns:
        Dict con status de cada archivo por dataset
    """
    status = {}
    
    for dataset in ["yelp", "es", "olist"]:
        status[dataset] = {}
        tables = BRONZE_FILES.get(dataset, {})
        for table in tables.keys():
            try:
                info = get_bronze_file_info(dataset, table)
                status[dataset][table] = info["exists"]
            except Exception:
                status[dataset][table] = False
    
    return status