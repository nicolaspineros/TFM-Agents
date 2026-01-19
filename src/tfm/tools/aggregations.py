"""
Agregaciones SQL sobre DuckDB.

Este modulo ejecuta consultas SQL para generar agregados:
- Sentimiento por mes/categoria/entidad
- Ventas por periodo (Olist)
- Conteos y distribuciones
- Top-N rankings

Los agregados se materializan en gold/ si no existen,
implementando lazy computation.

El LLM NO ejecuta estas agregaciones; solo las invoca
y recibe tablas pequenas como resultado.
"""

from pathlib import Path
from typing import Optional, Any, Literal, Dict, List

import polars as pl

from tfm.config.settings import get_settings, SILVER_FILES


def aggregate_reviews_by_month(
    dataset: Literal["yelp", "olist"],
    year_filter: Optional[int] = None,
    materialize: bool = False
) -> Dict[str, Any]:
    """
    Agrega metricas de reviews por ano/mes.
    
    Args:
        dataset: Dataset a agregar (yelp u olist, es no tiene fechas)
        year_filter: Filtrar por ano especifico
        materialize: Si guardar resultado en gold/
        
    Returns:
        Dict con datos agregados
        
    Example:
        >>> result = aggregate_reviews_by_month("yelp")
        >>> print(result["data"][:2])
    """
    settings = get_settings()
    
    # Determinar archivo silver
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        stars_col = "stars"
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
        stars_col = "review_score"
    else:
        return {"error": f"Dataset {dataset} no soporta agregacion temporal (sin fechas)"}
    
    if not silver_path.exists():
        return {"error": f"Silver no existe: {silver_path}. Ejecutar build_silver primero."}
    
    # Cargar datos
    df = pl.read_parquet(silver_path)
    
    # Filtrar por ano si se especifica
    if year_filter and "year" in df.columns:
        df = df.filter(pl.col("year") == year_filter)
    
    # Verificar que tenga columnas necesarias
    if "year" not in df.columns or "month" not in df.columns:
        return {"error": f"Dataset {dataset} no tiene columnas year/month"}
    
    # Agregar
    agg_df = df.group_by(["year", "month"]).agg([
        pl.count().alias("review_count"),
        pl.col(stars_col).mean().round(2).alias("avg_stars"),
        pl.col("is_ambiguous").sum().alias("ambiguous_count"),
        pl.col("text_length").mean().round(0).alias("avg_text_length"),
    ]).sort(["year", "month"])
    
    # Calcular porcentajes
    agg_df = agg_df.with_columns([
        (pl.col("ambiguous_count") / pl.col("review_count") * 100).round(1).alias("pct_ambiguous"),
    ])
    
    data = agg_df.to_dicts()
    
    result = {
        "aggregation_type": "reviews_by_month",
        "dataset": dataset,
        "row_count": len(data),
        "columns": agg_df.columns,
        "data": data,
    }
    
    # Materializar si se solicita
    if materialize:
        output_path = settings.gold_dir / f"agg_reviews_by_month_{dataset}.parquet"
        settings.ensure_dirs()
        agg_df.write_parquet(output_path)
        result["artifact_path"] = str(output_path)
    
    return result


def aggregate_reviews_by_stars(
    dataset: Literal["yelp", "es", "olist"]
) -> Dict[str, Any]:
    """
    Agrega distribucion de reviews por estrellas/score.
    
    Args:
        dataset: Dataset a agregar
        
    Returns:
        Dict con distribucion
    """
    settings = get_settings()
    
    # Determinar archivo y columna
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        stars_col = "stars"
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES["es"]
        stars_col = "stars"
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
        stars_col = "review_score"
    else:
        return {"error": f"Dataset invalido: {dataset}"}
    
    if not silver_path.exists():
        return {"error": f"Silver no existe: {silver_path}"}
    
    df = pl.read_parquet(silver_path)
    total = df.height
    
    # Agregar por stars
    agg_df = df.group_by(stars_col).agg([
        pl.count().alias("count"),
    ]).sort(stars_col)
    
    agg_df = agg_df.with_columns([
        (pl.col("count") / total * 100).round(2).alias("percentage"),
    ])
    
    data = agg_df.to_dicts()
    
    # Calcular resumen
    positive = sum(r["count"] for r in data if r[stars_col] >= 4)
    negative = sum(r["count"] for r in data if r[stars_col] <= 2)
    ambiguous = sum(r["count"] for r in data if r[stars_col] == 3)
    
    return {
        "aggregation_type": "reviews_by_stars",
        "dataset": dataset,
        "total_reviews": total,
        "distribution": data,
        "summary": {
            "positive_count": positive,
            "positive_pct": round(positive / total * 100, 1),
            "negative_count": negative,
            "negative_pct": round(negative / total * 100, 1),
            "ambiguous_count": ambiguous,
            "ambiguous_pct": round(ambiguous / total * 100, 1),
        }
    }


def aggregate_olist_sales_by_month() -> Dict[str, Any]:
    """
    Agrega ventas de Olist por mes.
    
    Returns:
        Dict con revenue, orders por mes
    """
    settings = get_settings()
    orders_path = settings.silver_dir / SILVER_FILES["olist_orders"]
    
    if not orders_path.exists():
        return {"error": f"Silver orders no existe: {orders_path}"}
    
    df = pl.read_parquet(orders_path)
    
    # Agregar por ano/mes
    agg_df = df.group_by(["year", "month"]).agg([
        pl.count().alias("order_count"),
        pl.col("order_total").sum().round(2).alias("total_revenue"),
        pl.col("order_total").mean().round(2).alias("avg_order_value"),
        pl.col("item_count").sum().alias("total_items"),
    ]).sort(["year", "month"])
    
    data = agg_df.to_dicts()
    
    # Estadisticas globales
    total_revenue = df["order_total"].sum()
    total_orders = df.height
    
    return {
        "aggregation_type": "olist_sales_by_month",
        "dataset": "olist",
        "row_count": len(data),
        "data": data,
        "summary": {
            "total_revenue": round(total_revenue, 2),
            "total_orders": total_orders,
            "avg_order_value": round(total_revenue / total_orders, 2) if total_orders > 0 else 0,
        }
    }


def aggregate_olist_by_category(top_n: int = 20) -> Dict[str, Any]:
    """
    Agrega metricas de Olist por categoria de producto.
    
    Args:
        top_n: Numero de categorias top a retornar
        
    Returns:
        Dict con metricas por categoria
    """
    settings = get_settings()
    orders_path = settings.silver_dir / SILVER_FILES["olist_orders"]
    reviews_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
    
    if not orders_path.exists():
        return {"error": f"Silver orders no existe: {orders_path}"}
    
    orders_df = pl.read_parquet(orders_path)
    
    # Verificar si tenemos categoria
    if "category_en" not in orders_df.columns:
        return {"error": "Columna category_en no disponible en orders"}
    
    # Agregar por categoria
    agg_df = orders_df.group_by("category_en").agg([
        pl.count().alias("order_count"),
        pl.col("order_total").sum().round(2).alias("total_revenue"),
        pl.col("order_total").mean().round(2).alias("avg_order_value"),
    ]).sort("total_revenue", descending=True).head(top_n)
    
    # Si hay reviews, agregar score promedio
    if reviews_path.exists():
        reviews_df = pl.read_parquet(reviews_path)
        review_agg = reviews_df.group_by("order_id").agg([
            pl.col("review_score").mean().alias("avg_score")
        ])
        # Join seria complejo, lo dejamos simple por ahora
    
    data = agg_df.to_dicts()
    
    return {
        "aggregation_type": "olist_by_category",
        "dataset": "olist",
        "row_count": len(data),
        "data": data,
    }


def get_distribution(
    dataset: Literal["yelp", "es", "olist"],
    column: str,
    table_suffix: str = "reviews"
) -> Dict[str, Any]:
    """
    Obtiene distribucion de valores para una columna.
    
    Args:
        dataset: Dataset
        column: Columna a analizar
        table_suffix: Sufijo de tabla (reviews, orders, users)
        
    Returns:
        Dict con distribucion de valores
    """
    settings = get_settings()
    
    # Determinar archivo
    if dataset == "yelp":
        if table_suffix == "users":
            silver_path = settings.silver_dir / SILVER_FILES["yelp_users"]
        else:
            silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES["es"]
    elif dataset == "olist":
        if table_suffix == "orders":
            silver_path = settings.silver_dir / SILVER_FILES["olist_orders"]
        else:
            silver_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
    else:
        return {"error": f"Dataset invalido: {dataset}"}
    
    if not silver_path.exists():
        return {"error": f"Silver no existe: {silver_path}"}
    
    df = pl.read_parquet(silver_path)
    
    if column not in df.columns:
        return {"error": f"Columna {column} no existe. Disponibles: {df.columns}"}
    
    total = df.height
    
    agg_df = df.group_by(column).agg([
        pl.count().alias("count"),
    ]).sort("count", descending=True)
    
    agg_df = agg_df.with_columns([
        (pl.col("count") / total * 100).round(2).alias("percentage"),
    ])
    
    data = agg_df.head(50).to_dicts()  # Limitar a 50 valores unicos
    
    return {
        "aggregation_type": "distribution",
        "dataset": dataset,
        "column": column,
        "unique_values": df[column].n_unique(),
        "total_rows": total,
        "data": data,
    }


def get_top_entities(
    dataset: Literal["yelp", "olist"],
    entity_column: str,
    metric: Literal["count", "avg_stars", "total_revenue"] = "count",
    n: int = 10,
    ascending: bool = False
) -> Dict[str, Any]:
    """
    Obtiene top-N entidades por una metrica.
    
    Args:
        dataset: Dataset
        entity_column: Columna de entidad (business_id, product_id, user_id, etc.)
        metric: Metrica para ordenar
        n: Cantidad de resultados
        ascending: Si ordenar ascendente
        
    Returns:
        Dict con top entidades
    """
    settings = get_settings()
    
    # Determinar archivo
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        stars_col = "stars"
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES["olist_orders"]
        stars_col = "review_score" if "review_score" in ["review_score"] else None
    else:
        return {"error": f"Dataset {dataset} no soportado"}
    
    if not silver_path.exists():
        return {"error": f"Silver no existe: {silver_path}"}
    
    df = pl.read_parquet(silver_path)
    
    if entity_column not in df.columns:
        return {"error": f"Columna {entity_column} no existe. Disponibles: {df.columns}"}
    
    # Definir agregaciones segun metrica
    if metric == "count":
        agg_df = df.group_by(entity_column).agg([
            pl.count().alias("count"),
        ])
        sort_col = "count"
    elif metric == "avg_stars" and stars_col and stars_col in df.columns:
        agg_df = df.group_by(entity_column).agg([
            pl.count().alias("count"),
            pl.col(stars_col).mean().round(2).alias("avg_stars"),
        ])
        sort_col = "avg_stars"
    elif metric == "total_revenue" and "order_total" in df.columns:
        agg_df = df.group_by(entity_column).agg([
            pl.count().alias("count"),
            pl.col("order_total").sum().round(2).alias("total_revenue"),
        ])
        sort_col = "total_revenue"
    else:
        return {"error": f"Metrica {metric} no disponible para {dataset}"}
    
    agg_df = agg_df.sort(sort_col, descending=not ascending).head(n)
    data = agg_df.to_dicts()
    
    return {
        "aggregation_type": "top_entities",
        "dataset": dataset,
        "entity_column": entity_column,
        "metric": metric,
        "ascending": ascending,
        "data": data,
    }


def aggregate_yelp_user_stats() -> Dict[str, Any]:
    """
    Agrega estadisticas de usuarios de Yelp.
    
    Returns:
        Dict con metricas de usuarios
    """
    settings = get_settings()
    users_path = settings.silver_dir / SILVER_FILES["yelp_users"]
    
    if not users_path.exists():
        return {"error": f"Silver users no existe: {users_path}"}
    
    df = pl.read_parquet(users_path)
    
    # Estadisticas basicas
    total_users = df.height
    
    stats = {
        "total_users": total_users,
        "avg_review_count": round(df["review_count"].mean(), 1),
        "avg_fans": round(df["fans"].mean(), 1),
        "avg_average_stars": round(df["average_stars"].mean(), 2),
    }
    
    # Distribucion de elite
    if "is_elite" in df.columns:
        elite_count = df.filter(pl.col("is_elite")).height
        stats["elite_users"] = elite_count
        stats["elite_pct"] = round(elite_count / total_users * 100, 1)
    
    # Top usuarios por reviews
    top_reviewers = df.sort("review_count", descending=True).head(10).select([
        "user_id", "review_count", "fans", "average_stars"
    ]).to_dicts()
    
    return {
        "aggregation_type": "yelp_user_stats",
        "dataset": "yelp",
        "stats": stats,
        "top_reviewers": top_reviewers,
    }


def aggregate_by_text_length(
    dataset: Literal["yelp", "es", "olist"],
    length_bins: List[int] = [0, 50, 200, 500, 1000, 5000]
) -> Dict[str, Any]:
    """
    Agrega metricas por longitud de texto (reviews cortas vs largas).
    
    Basado en EDA: reviews negativas tienden a ser mas largas.
    
    Args:
        dataset: Dataset a agregar
        length_bins: Limites de los bins de longitud
        
    Returns:
        Dict con metricas por bin de longitud
    """
    settings = get_settings()
    
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        stars_col = "stars"
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES["es"]
        stars_col = "stars"
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
        stars_col = "review_score"
    else:
        return {"error": f"Dataset invalido: {dataset}"}
    
    if not silver_path.exists():
        return {"error": f"Silver no existe: {silver_path}"}
    
    df = pl.read_parquet(silver_path)
    
    # Crear bins de longitud
    df = df.with_columns([
        pl.col("text_length")
          .cut(length_bins, labels=[f"{length_bins[i]}-{length_bins[i+1]}" for i in range(len(length_bins)-1)])
          .alias("length_bin")
    ])
    
    # Agregar por bin
    agg_df = df.group_by("length_bin").agg([
        pl.count().alias("count"),
        pl.col(stars_col).mean().round(2).alias("avg_stars"),
        pl.col("is_ambiguous").sum().alias("ambiguous_count"),
    ]).sort("length_bin")
    
    # Calcular porcentajes
    total = df.height
    agg_df = agg_df.with_columns([
        (pl.col("count") / total * 100).round(1).alias("pct"),
        (pl.col("ambiguous_count") / pl.col("count") * 100).round(1).alias("pct_ambiguous"),
    ])
    
    data = agg_df.to_dicts()
    
    return {
        "aggregation_type": "by_text_length",
        "dataset": dataset,
        "total_reviews": total,
        "bins": length_bins,
        "data": data,
    }


def aggregate_ambiguous_reviews(
    dataset: Literal["yelp", "es", "olist"]
) -> Dict[str, Any]:
    """
    Analisis detallado de reviews ambiguas (stars == 3).
    
    Basado en EDA: Las reviews de 3 estrellas son un caso especial
    donde el texto puede revelar sentimiento real.
    
    Args:
        dataset: Dataset a analizar
        
    Returns:
        Dict con estadisticas de reviews ambiguas
    """
    settings = get_settings()
    
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        stars_col = "stars"
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES["es"]
        stars_col = "stars"
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
        stars_col = "review_score"
    else:
        return {"error": f"Dataset invalido: {dataset}"}
    
    if not silver_path.exists():
        return {"error": f"Silver no existe: {silver_path}"}
    
    df = pl.read_parquet(silver_path)
    
    total = df.height
    ambiguous = df.filter(pl.col("is_ambiguous"))
    ambiguous_count = ambiguous.height
    
    stats = {
        "total_reviews": total,
        "ambiguous_count": ambiguous_count,
        "ambiguous_pct": round(ambiguous_count / total * 100, 2),
        "avg_text_length_ambiguous": round(ambiguous["text_length"].mean(), 0) if ambiguous_count > 0 else 0,
        "avg_text_length_non_ambiguous": round(df.filter(~pl.col("is_ambiguous"))["text_length"].mean(), 0),
    }
    
    # Comparar con positivas y negativas
    positive = df.filter(pl.col(stars_col) >= 4)
    negative = df.filter(pl.col(stars_col) <= 2)
    
    stats["positive_count"] = positive.height
    stats["negative_count"] = negative.height
    stats["positive_avg_length"] = round(positive["text_length"].mean(), 0) if positive.height > 0 else 0
    stats["negative_avg_length"] = round(negative["text_length"].mean(), 0) if negative.height > 0 else 0
    
    # Sample de reviews ambiguas (para analisis cualitativo)
    sample_texts = []
    if ambiguous_count > 0 and "text" in ambiguous.columns:
        sample_df = ambiguous.sample(min(5, ambiguous_count)).select(["text", "text_length"])
        sample_texts = sample_df.to_dicts()
    
    return {
        "aggregation_type": "ambiguous_analysis",
        "dataset": dataset,
        "stats": stats,
        "sample_ambiguous": sample_texts,
    }


def aggregate_business_stats() -> Dict[str, Any]:
    """
    Estadisticas de negocios de Yelp.
    
    Basado en EDA: Analisis de categorias, ratings, y distribucion.
    
    Returns:
        Dict con metricas de negocios
    """
    settings = get_settings()
    business_path = settings.silver_dir / SILVER_FILES["yelp_business"]
    
    if not business_path.exists():
        return {"error": f"Silver business no existe: {business_path}"}
    
    df = pl.read_parquet(business_path)
    
    total = df.height
    open_count = df.filter(pl.col("is_open") == 1).height
    
    stats = {
        "total_businesses": total,
        "open_businesses": open_count,
        "closed_businesses": total - open_count,
        "open_pct": round(open_count / total * 100, 1),
        "avg_stars": round(df["stars"].mean(), 2),
        "avg_review_count": round(df["review_count"].mean(), 1),
        "avg_category_count": round(df["category_count"].mean(), 1),
    }
    
    # Distribucion por estado
    state_dist = df.group_by("state").agg([
        pl.count().alias("count"),
        pl.col("stars").mean().round(2).alias("avg_stars"),
    ]).sort("count", descending=True).head(10).to_dicts()
    
    # Distribucion de ratings
    rating_dist = df.group_by("stars").agg([
        pl.count().alias("count"),
    ]).sort("stars").to_dicts()
    
    return {
        "aggregation_type": "business_stats",
        "dataset": "yelp",
        "stats": stats,
        "top_states": state_dist,
        "rating_distribution": rating_dist,
    }


def correlate_reviews_business(
    min_reviews: int = 10
) -> Dict[str, Any]:
    """
    Correlaciona metricas de reviews con negocios.
    
    Basado en EDA: Entender relacion entre sentimiento y rating de negocio.
    
    Args:
        min_reviews: Minimo de reviews para incluir negocio
        
    Returns:
        Dict con correlaciones
    """
    settings = get_settings()
    reviews_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
    business_path = settings.silver_dir / SILVER_FILES["yelp_business"]
    
    if not reviews_path.exists() or not business_path.exists():
        return {"error": "Silver reviews o business no existe"}
    
    df_reviews = pl.read_parquet(reviews_path)
    df_business = pl.read_parquet(business_path)
    
    # Agregar reviews por negocio
    review_agg = df_reviews.group_by("business_id").agg([
        pl.count().alias("review_count_calc"),
        pl.col("stars").mean().round(2).alias("avg_review_stars"),
        pl.col("text_length").mean().round(0).alias("avg_text_length"),
        pl.col("is_ambiguous").sum().alias("ambiguous_count"),
    ])
    
    # Join con business
    df_joined = df_business.join(
        review_agg,
        on="business_id",
        how="inner"
    ).filter(pl.col("review_count_calc") >= min_reviews)
    
    if df_joined.height == 0:
        return {"error": f"No hay negocios con >= {min_reviews} reviews"}
    
    # Calcular correlaciones
    business_stars = df_joined["stars"].to_list()
    review_stars = df_joined["avg_review_stars"].to_list()
    
    # Correlacion simple (Pearson)
    import statistics
    mean_b = statistics.mean(business_stars)
    mean_r = statistics.mean(review_stars)
    
    num = sum((b - mean_b) * (r - mean_r) for b, r in zip(business_stars, review_stars))
    den_b = sum((b - mean_b) ** 2 for b in business_stars) ** 0.5
    den_r = sum((r - mean_r) ** 2 for r in review_stars) ** 0.5
    
    correlation = round(num / (den_b * den_r), 4) if den_b * den_r > 0 else 0
    
    return {
        "aggregation_type": "reviews_business_correlation",
        "dataset": "yelp",
        "businesses_analyzed": df_joined.height,
        "min_reviews_filter": min_reviews,
        "correlation_business_vs_review_stars": correlation,
        "avg_business_stars": round(mean_b, 2),
        "avg_review_stars": round(mean_r, 2),
    }


def aggregate_olist_reviews_sales() -> Dict[str, Any]:
    """
    Correlaciona reviews con ventas en Olist.
    
    Basado en EDA: Entender si mejores reviews correlacionan con mas ventas.
    
    Returns:
        Dict con correlaciones y metricas
    """
    settings = get_settings()
    orders_path = settings.silver_dir / SILVER_FILES["olist_orders"]
    reviews_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
    
    if not orders_path.exists() or not reviews_path.exists():
        return {"error": "Silver orders o reviews no existe"}
    
    df_orders = pl.read_parquet(orders_path)
    df_reviews = pl.read_parquet(reviews_path)
    
    # Join reviews con orders
    df_joined = df_orders.join(
        df_reviews.select(["order_id", "review_score", "has_comment", "text_length"]),
        on="order_id",
        how="inner"
    )
    
    if df_joined.height == 0:
        return {"error": "No hay datos para correlacionar"}
    
    # Metricas por score de review
    score_metrics = df_joined.group_by("review_score").agg([
        pl.count().alias("order_count"),
        pl.col("order_total").mean().round(2).alias("avg_order_value"),
        pl.col("order_total").sum().round(2).alias("total_revenue"),
        pl.col("has_comment").sum().alias("with_comment"),
    ]).sort("review_score").to_dicts()
    
    # Correlacion score vs order_value
    scores = df_joined["review_score"].to_list()
    values = df_joined["order_total"].fill_null(0).to_list()
    
    import statistics
    mean_s = statistics.mean(scores)
    mean_v = statistics.mean(values)
    
    num = sum((s - mean_s) * (v - mean_v) for s, v in zip(scores, values))
    den_s = sum((s - mean_s) ** 2 for s in scores) ** 0.5
    den_v = sum((v - mean_v) ** 2 for v in values) ** 0.5
    
    correlation = round(num / (den_s * den_v), 4) if den_s * den_v > 0 else 0
    
    return {
        "aggregation_type": "olist_reviews_sales",
        "dataset": "olist",
        "orders_with_reviews": df_joined.height,
        "correlation_score_vs_value": correlation,
        "metrics_by_score": score_metrics,
        "insight": "Correlacion positiva indica que mejores reviews = mayores valores de orden"
    }
