"""
Preprocesamiento de datos (bronze -> silver).

Este modulo transforma datos crudos a formato limpio y normalizado:
- Limpieza de texto
- Normalizacion de columnas
- Calculo de campos derivados (text_length, word_count, is_ambiguous)
- Parseo de fechas
- Escritura a Parquet

Los datos silver son la base para:
- Calculo de features gold
- Registro en DuckDB
- Consultas y agregaciones
"""

from pathlib import Path
from typing import Optional, Tuple

import polars as pl

from tfm.config.settings import get_settings, SILVER_FILES
from tfm.tools.io_loaders import (
    load_yelp_reviews, load_yelp_users, load_yelp_business,
    load_es_reviews, load_olist_data
)


def build_silver_yelp(
    limit: Optional[int] = None,
    overwrite: bool = False
) -> Path:
    """
    Construye silver layer para Yelp reviews.
    
    Transformaciones:
    - Parsea 'date' a tipo Datetime
    - Extrae year, month
    - Calcula text_length, word_count
    - Anade is_ambiguous (stars == 3)
    - Anade language = 'en'
    - Limpia whitespace en text
    
    Args:
        limit: Limite de filas (para desarrollo)
        overwrite: Si sobrescribir si ya existe
        
    Returns:
        Path al archivo silver generado
        
    Example:
        >>> path = build_silver_yelp(limit=10000)
        >>> print(path)
        data/silver/yelp_reviews.parquet
    """
    settings = get_settings()
    output_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
    
    # Check si ya existe
    if output_path.exists() and not overwrite:
        print(f"Silver ya existe: {output_path}")
        return output_path
    
    settings.ensure_dirs()
    
    print(f"Cargando reviews de Yelp (limit={limit})...")
    df = load_yelp_reviews(limit=limit)
    
    print("Aplicando transformaciones...")
    # Parsear fecha
    df_silver = df.with_columns([
        pl.col("date")
          .str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S", strict=False)
          .alias("datetime_parsed")
    ]).with_columns([
        # Fecha como Date
        pl.col("datetime_parsed").cast(pl.Date).alias("date_parsed"),
        # Extraer componentes
        pl.col("datetime_parsed").dt.year().alias("year"),
        pl.col("datetime_parsed").dt.month().alias("month"),
        # Metricas de texto
        pl.col("text").str.strip_chars().alias("text_clean"),
        pl.col("text").str.len_chars().alias("text_length"),
        pl.col("text").str.split(" ").list.len().alias("word_count"),
        # Flag ambiguo
        (pl.col("stars") == 3).alias("is_ambiguous"),
        # Idioma fijo
        pl.lit("en").alias("language"),
    ]).select([
        "review_id", "user_id", "business_id", "stars",
        "text_clean", "text_length", "word_count",
        "useful", "funny", "cool",
        "date_parsed", "year", "month",
        "is_ambiguous", "language"
    ]).rename({"text_clean": "text", "date_parsed": "date"})
    
    print(f"Guardando silver: {output_path}")
    df_silver.write_parquet(output_path)
    print(f"Guardado: {df_silver.height:,} filas")
    
    return output_path


def build_silver_yelp_users(
    limit: Optional[int] = None,
    overwrite: bool = False
) -> Path:
    """
    Construye silver layer para usuarios de Yelp.
    
    Transformaciones:
    - Procesa friends (cuenta)
    - Procesa elite years
    - Calcula total_votes, total_compliments
    - Calcula influence_score
    
    Args:
        limit: Limite de filas
        overwrite: Si sobrescribir
        
    Returns:
        Path al archivo silver
    """
    settings = get_settings()
    output_path = settings.silver_dir / SILVER_FILES["yelp_users"]
    
    if output_path.exists() and not overwrite:
        print(f"Silver ya existe: {output_path}")
        return output_path
    
    settings.ensure_dirs()
    
    print(f"Cargando usuarios de Yelp (limit={limit})...")
    df = load_yelp_users(limit=limit)
    
    print("Procesando usuarios...")
    # Normalizar strings
    df = df.with_columns([
        pl.col("friends").cast(pl.Utf8).fill_null("").str.strip_chars().alias("friends_norm"),
        pl.col("elite").cast(pl.Utf8).fill_null("").str.strip_chars().alias("elite_norm"),
    ])
    
    # Calcular friends_count
    df = df.with_columns([
        pl.when(
            (pl.col("friends_norm") == "") |
            (pl.col("friends_norm").str.to_lowercase() == "none")
        )
        .then(0)
        .otherwise(
            pl.col("friends_norm")
              .str.replace_all(r"\s*,\s*", ",", literal=False)
              .str.split(",")
              .list.len()
        )
        .alias("friends_count")
    ])
    
    # Calcular elite_years_count
    df = df.with_columns([
        pl.col("elite_norm")
          .str.extract_all(r"\d{4}")
          .list.len()
          .alias("elite_years_count"),
    ]).with_columns([
        (pl.col("elite_years_count") > 0).alias("is_elite"),
    ])
    
    # Calcular metricas agregadas
    df = df.with_columns([
        (pl.col("useful") + pl.col("funny") + pl.col("cool")).alias("total_votes_given"),
        (pl.col("compliment_hot") + pl.col("compliment_more") + pl.col("compliment_profile") +
         pl.col("compliment_cute") + pl.col("compliment_list") + pl.col("compliment_note") +
         pl.col("compliment_plain") + pl.col("compliment_cool") + pl.col("compliment_funny") +
         pl.col("compliment_writer") + pl.col("compliment_photos")).alias("total_compliments"),
    ])
    
    # Seleccionar columnas para silver
    df_silver = df.select([
        "user_id", "name", "review_count", "yelping_since",
        "useful", "funny", "cool", "fans", "average_stars",
        "friends_count", "elite_years_count", "is_elite",
        "total_votes_given", "total_compliments",
    ])
    
    print(f"Guardando silver: {output_path}")
    df_silver.write_parquet(output_path)
    print(f"Guardado: {df_silver.height:,} filas")
    
    return output_path


def build_silver_es(
    limit: Optional[int] = None,
    overwrite: bool = False
) -> Path:
    """
    Construye silver layer para reviews en espanol.
    
    Transformaciones:
    - Renombra review_body -> text
    - Calcula text_length, word_count
    - Normaliza category
    - Anade is_ambiguous (stars == 3)
    - Anade language = 'es'
    
    Nota: Este dataset NO tiene campo de fecha.
    
    Args:
        limit: Limite de filas
        overwrite: Si sobrescribir
        
    Returns:
        Path al archivo silver
    """
    settings = get_settings()
    output_path = settings.silver_dir / SILVER_FILES["es"]
    
    if output_path.exists() and not overwrite:
        print(f"Silver ya existe: {output_path}")
        return output_path
    
    settings.ensure_dirs()
    
    print(f"Cargando reviews ES (limit={limit})...")
    df = load_es_reviews(limit=limit)
    
    print("Aplicando transformaciones...")
    # Renombrar columnas si es necesario
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'review_body' in col_lower or col_lower == 'body':
            column_mapping[col] = 'text'
        elif 'product_category' in col_lower:
            column_mapping[col] = 'category'
    
    if column_mapping:
        df = df.rename(column_mapping)
    
    # Determinar columna de texto
    text_col = 'text' if 'text' in df.columns else 'review_body'
    
    df_silver = df.with_columns([
        pl.col(text_col).str.strip_chars().alias("text_clean"),
        pl.col(text_col).str.len_chars().alias("text_length"),
        pl.col(text_col).str.split(" ").list.len().alias("word_count"),
        (pl.col("stars") == 3).alias("is_ambiguous"),
        pl.lit("es").alias("language"),
    ])
    
    # Seleccionar columnas
    select_cols = ["review_id", "product_id", "reviewer_id", "stars",
                   "text_clean", "text_length", "word_count",
                   "is_ambiguous", "language"]
    if "category" in df_silver.columns:
        select_cols.append("category")
    
    df_silver = df_silver.select([c for c in select_cols if c in df_silver.columns])
    df_silver = df_silver.rename({"text_clean": "text"})
    
    print(f"Guardando silver: {output_path}")
    df_silver.write_parquet(output_path)
    print(f"Guardado: {df_silver.height:,} filas")
    
    return output_path


def build_silver_olist(
    overwrite: bool = False
) -> Tuple[Path, Path]:
    """
    Construye silver layers para Olist (orders + reviews).
    
    Genera dos archivos:
    1. olist_orders.parquet: Join de orders + items + products
    2. olist_reviews.parquet: Reviews con texto procesado
    
    Transformaciones para orders:
    - Join orders con items y products
    - Extraccion de features temporales
    - Calculo de metricas de orden
    
    Transformaciones para reviews:
    - Combinacion de title + message
    - Flag has_comment
    - is_ambiguous (score == 3)
    - language = 'pt'
    
    Args:
        overwrite: Si sobrescribir
        
    Returns:
        Tuple (orders_path, reviews_path)
    """
    settings = get_settings()
    orders_path = settings.silver_dir / SILVER_FILES["olist_orders"]
    reviews_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
    
    settings.ensure_dirs()
    
    # === BUILD ORDERS ===
    if not orders_path.exists() or overwrite:
        print("Construyendo silver orders...")
        
        df_orders = load_olist_data("orders")
        df_items = load_olist_data("items")
        df_products = load_olist_data("products")
        
        # Parsear fechas
        df_orders = df_orders.with_columns([
            pl.col("order_purchase_timestamp")
              .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
              .alias("purchase_date"),
        ]).with_columns([
            pl.col("purchase_date").dt.year().alias("year"),
            pl.col("purchase_date").dt.month().alias("month"),
            pl.col("purchase_date").dt.weekday().alias("day_of_week"),
        ])
        
        # Agregar items por orden
        order_totals = df_items.group_by("order_id").agg([
            pl.col("price").sum().alias("order_total"),
            pl.col("freight_value").sum().alias("freight_total"),
            pl.count().alias("item_count"),
        ])
        
        # Join orders con totales
        df_orders_silver = df_orders.join(
            order_totals,
            on="order_id",
            how="left"
        ).select([
            "order_id", "customer_id", "order_status",
            "purchase_date", "year", "month", "day_of_week",
            "order_total", "freight_total", "item_count",
        ])
        
        print(f"Guardando: {orders_path}")
        df_orders_silver.write_parquet(orders_path)
        print(f"Guardado: {df_orders_silver.height:,} ordenes")
    else:
        print(f"Silver orders ya existe: {orders_path}")
    
    # === BUILD REVIEWS ===
    if not reviews_path.exists() or overwrite:
        print("Construyendo silver reviews...")
        
        df_reviews = load_olist_data("reviews")
        
        # Combinar titulo y mensaje
        df_reviews_silver = df_reviews.with_columns([
            # Verificar si tiene comentario
            (pl.col("review_comment_message").is_not_null() & 
             (pl.col("review_comment_message") != "")).alias("has_comment"),
            # Combinar titulo y mensaje
            pl.when(pl.col("review_comment_message").is_not_null())
            .then(pl.concat_str([
                pl.col("review_comment_title").fill_null(""),
                pl.lit(" "),
                pl.col("review_comment_message").fill_null("")
            ]))
            .otherwise(pl.col("review_comment_title"))
            .str.strip_chars()
            .alias("text"),
            # Marcar ambiguos
            (pl.col("review_score") == 3).alias("is_ambiguous"),
            # Idioma
            pl.lit("pt").alias("language"),
        ])
        
        # Agregar metricas de texto
        df_reviews_silver = df_reviews_silver.with_columns([
            pl.col("text").str.len_chars().alias("text_length"),
            pl.col("text").str.split(" ").list.len().alias("word_count"),
        ]).select([
            "review_id", "order_id", "review_score",
            "text", "text_length", "word_count",
            "has_comment", "is_ambiguous", "language",
            "review_creation_date", "review_answer_timestamp",
        ])
        
        print(f"Guardando: {reviews_path}")
        df_reviews_silver.write_parquet(reviews_path)
        print(f"Guardado: {df_reviews_silver.height:,} reviews")
    else:
        print(f"Silver reviews ya existe: {reviews_path}")
    
    return orders_path, reviews_path


def build_all_silver(overwrite: bool = False) -> dict:
    """
    Construye todos los archivos silver.
    
    Args:
        overwrite: Si sobrescribir archivos existentes
        
    Returns:
        Dict con paths de archivos generados
    """
    results = {}
    
    print("=" * 60)
    print("CONSTRUYENDO SILVER LAYER")
    print("=" * 60)
    
    # Yelp reviews
    print("\n[1/4] Yelp Reviews...")
    try:
        results["yelp_reviews"] = str(build_silver_yelp(overwrite=overwrite))
    except Exception as e:
        results["yelp_reviews"] = f"Error: {e}"
    
    # Yelp users
    print("\n[2/4] Yelp Users...")
    try:
        results["yelp_users"] = str(build_silver_yelp_users(overwrite=overwrite))
    except Exception as e:
        results["yelp_users"] = f"Error: {e}"
    
    # ES reviews
    print("\n[3/4] ES Reviews...")
    try:
        results["es"] = str(build_silver_es(overwrite=overwrite))
    except Exception as e:
        results["es"] = f"Error: {e}"
    
    # Olist
    print("\n[4/4] Olist...")
    try:
        orders_path, reviews_path = build_silver_olist(overwrite=overwrite)
        results["olist_orders"] = str(orders_path)
        results["olist_reviews"] = str(reviews_path)
    except Exception as e:
        results["olist"] = f"Error: {e}"
    
    print("\n" + "=" * 60)
    print("SILVER LAYER COMPLETADO")
    print("=" * 60)
    
    return results


def clean_text(text: str) -> str:
    """
    Limpia texto de una review.
    
    Operaciones:
    - Strip whitespace
    - Normaliza espacios multiples
    - Elimina caracteres de control
    
    Args:
        text: Texto crudo
        
    Returns:
        Texto limpio
    """
    if not text:
        return ""
    
    import re
    
    # Strip
    text = text.strip()
    
    # Normalizar espacios multiples
    text = re.sub(r"\s+", " ", text)
    
    # Eliminar caracteres de control (excepto newlines)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    return text


def compute_text_stats(text: str) -> dict:
    """
    Calcula estadisticas basicas de texto.
    
    Args:
        text: Texto limpio
        
    Returns:
        Dict con text_length y word_count
    """
    if not text:
        return {"text_length": 0, "word_count": 0}
    
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
    }


def check_silver_status() -> dict:
    """
    Verifica el estado de los archivos silver.
    
    Returns:
        Dict con estado de cada archivo
    """
    settings = get_settings()
    
    status = {}
    for name, filename in SILVER_FILES.items():
        path = settings.silver_dir / filename
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            status[name] = {
                "exists": True,
                "path": str(path),
                "size_mb": round(size_mb, 2),
            }
        else:
            status[name] = {
                "exists": False,
                "path": str(path),
            }
    
    return status
