"""
Construccion de features gold.

Este modulo transforma silver a gold anadiendo:
- Features de sentimiento (VADER, ML/SVM, o baseline)
- Features de aspectos
- Embeddings (opcional)
- Topics/clusters (opcional)

Las features gold son la base para:
- Agregaciones precomputadas
- Consultas rapidas sin recalcular
- Analisis historicos
"""

from pathlib import Path
from typing import Optional, Any, Literal, Dict, List

import polars as pl

from tfm.config.settings import get_settings, SILVER_FILES, GOLD_FILES
from tfm.tools.sentiment import (
    compute_sentiment_batch,
    compute_sentiment_baseline,
    compute_sentiment_vader,
)


def build_gold_features(
    dataset: Literal["yelp", "es", "olist"],
    overwrite: bool = False,
    sample_size: Optional[int] = None,
    batch_size: int = 50000,
    use_ml_model: bool = True,
    use_vader: bool = True,
    extract_aspects: bool = False
) -> Path:
    """
    Construye gold features para un dataset.
    
    Proceso:
    1. Lee silver layer
    2. Calcula sentimiento para cada review (ML, VADER, o baseline)
    3. Opcionalmente extrae aspectos
    4. Guarda en gold/
    
    
    Jerarquia de metodos de sentimiento:
    1. ML (SVM/TF-IDF) - si use_ml_model=True y modelo existe
    2. VADER - si use_vader=True y es ingles
    3. Baseline (estrellas) - fallback
    
    Args:
        dataset: Dataset a procesar
        overwrite: Si sobrescribir si existe
        sample_size: Si se especifica, solo procesar N filas (para pruebas)
        batch_size: Tamano de batch para procesamiento
        use_ml_model: Si usar modelo ML entrenado (SVM) cuando este disponible
        use_vader: Si usar VADER para ingles (si ML no disponible)
        extract_aspects: Si extraer aspectos (mas lento)
        
    Returns:
        Path al archivo gold generado
    """
    settings = get_settings()
    settings.ensure_dirs()
    
    # Determinar paths
    if dataset == "yelp":
        silver_path = settings.silver_dir / SILVER_FILES["yelp_reviews"]
        gold_path = settings.gold_dir / GOLD_FILES["yelp_features"]
        stars_col = "stars"
        text_col = "text"
        language = "en"
    elif dataset == "es":
        silver_path = settings.silver_dir / SILVER_FILES["es"]
        gold_path = settings.gold_dir / GOLD_FILES["es_features"]
        stars_col = "stars"
        text_col = "text"
        language = "es"
    elif dataset == "olist":
        silver_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
        gold_path = settings.gold_dir / GOLD_FILES["olist_features"]
        stars_col = "review_score"
        text_col = "text"
        language = "pt"
    else:
        raise ValueError(f"Dataset invalido: {dataset}")
    
    # Check si ya existe
    if gold_path.exists() and not overwrite:
        print(f"Gold ya existe: {gold_path}")
        return gold_path
    
    # Verificar que exista silver
    if not silver_path.exists():
        raise FileNotFoundError(
            f"Silver layer no encontrado: {silver_path}. "
            f"Ejecutar build_silver primero."
        )
    
    print(f"Cargando silver: {silver_path}")
    df = pl.read_parquet(silver_path)
    
    if sample_size and sample_size < df.height:
        df = df.head(sample_size)
        print(f"Usando muestra de {sample_size} filas")
    
    total_rows = df.height
    print(f"Procesando {total_rows} reviews...")
    
    # === DETERMINAR METODO DE SENTIMIENTO ===
    ml_pipeline = None
    model_version = "baseline_stars_v1"
    
    # Intentar cargar modelo ML si esta disponible
    if use_ml_model:
        try:
            from tfm.tools.nlp_models import load_sentiment_model, clean_text_for_ml
            ml_pipeline = load_sentiment_model("unified_svm")
            if ml_pipeline:
                model_version = "ml_unified_svm_v1"
                print(f"  Usando modelo ML (unified_svm) para sentimiento")
        except Exception as e:
            print(f"  Modelo ML no disponible: {e}")
    
    # Fallback a VADER para ingles
    use_text_analysis = False
    if ml_pipeline is None and use_vader and language == "en" and text_col in df.columns:
        use_text_analysis = True
        model_version = "vader_v1"
        print(f"  Usando VADER para analisis de texto (ingles)")
    
    if ml_pipeline is None and not use_text_analysis:
        print(f"  Usando baseline (estrellas)")
    
    # Calcular sentimiento en batches
    sentiment_scores = []
    sentiment_labels = []
    sentiment_confidence = []
    
    # Mapeo de labels ML a scores
    label_to_score = {"positive": 0.8, "neutral": 0.0, "negative": -0.8}
    
    for i in range(0, total_rows, batch_size):
        end = min(i + batch_size, total_rows)
        batch = df.slice(i, end - i)
        
        stars_list = batch[stars_col].to_list()
        texts_list = batch[text_col].fill_null("").to_list() if text_col in batch.columns else [""] * len(stars_list)
        
        if ml_pipeline is not None:
            # === USAR MODELO ML (SVM) ===
            from tfm.tools.nlp_models import clean_text_for_ml
            
            for text, stars in zip(texts_list, stars_list):
                if text and len(str(text).strip()) >= 3:
                    # Limpiar texto para ML (limpieza ligera)
                    text_clean = clean_text_for_ml(text, aggressive=False)
                    if text_clean:
                        try:
                            pred = ml_pipeline.predict([text_clean])[0]
                            sentiment_labels.append(pred)
                            sentiment_scores.append(label_to_score.get(pred, 0.0))
                            # Confidence si el modelo lo soporta
                            if hasattr(ml_pipeline, "predict_proba"):
                                proba = ml_pipeline.predict_proba([text_clean])[0]
                                sentiment_confidence.append(float(max(proba)))
                            else:
                                sentiment_confidence.append(0.7)  # Default para SVM
                            continue
                        except Exception:
                            pass
                # Fallback a baseline si texto vacio o error
                result = compute_sentiment_baseline("", stars, language)
                sentiment_scores.append(result.sentiment_score)
                sentiment_labels.append(result.sentiment_label)
                sentiment_confidence.append(result.confidence)
        
        elif use_text_analysis:
            # === USAR VADER (solo ingles) ===
            for text, stars in zip(texts_list, stars_list):
                if text and len(text.strip()) > 10:
                    result = compute_sentiment_vader(text)
                    sentiment_scores.append(result.sentiment_score)
                    sentiment_labels.append(result.sentiment_label)
                    sentiment_confidence.append(result.confidence)
                else:
                    result = compute_sentiment_baseline("", stars, language)
                    sentiment_scores.append(result.sentiment_score)
                    sentiment_labels.append(result.sentiment_label)
                    sentiment_confidence.append(result.confidence)
        
        else:
            # === BASELINE: solo estrellas ===
            for stars in stars_list:
                result = compute_sentiment_baseline("", stars, language)
                sentiment_scores.append(result.sentiment_score)
                sentiment_labels.append(result.sentiment_label)
                sentiment_confidence.append(result.confidence)
        
        if (i + batch_size) % 100000 == 0 or end == total_rows:
            print(f"  Procesado: {end}/{total_rows}")
    
    # Agregar columnas de sentimiento
    df_gold = df.with_columns([
        pl.Series("sentiment_score", sentiment_scores),
        pl.Series("sentiment_label", sentiment_labels),
        pl.Series("sentiment_confidence", sentiment_confidence),
        pl.lit(model_version).alias("model_version"),
    ])
    
    # Guardar
    print(f"Guardando gold: {gold_path}")
    df_gold.write_parquet(gold_path)
    
    # Estadisticas finales
    label_counts = df_gold.group_by("sentiment_label").agg(pl.count().alias("count")).to_dicts()
    print(f"Gold completado: {df_gold.height} filas")
    print(f"  Modelo: {model_version}")
    print(f"  Distribucion: {label_counts}")
    
    return gold_path


def build_olist_sales_features(
    overwrite: bool = False
) -> Path:
    """
    Construye features de ventas para Olist.
    
    Features por mes y categoria:
    - total_orders
    - total_revenue
    - avg_price
    - avg_review_score
    
    Args:
        overwrite: Si sobrescribir
        
    Returns:
        Path al archivo de features
    """
    settings = get_settings()
    settings.ensure_dirs()
    
    output_path = settings.gold_dir / GOLD_FILES["olist_sales"]
    
    if output_path.exists() and not overwrite:
        print(f"Gold sales ya existe: {output_path}")
        return output_path
    
    orders_path = settings.silver_dir / SILVER_FILES["olist_orders"]
    reviews_path = settings.silver_dir / SILVER_FILES["olist_reviews"]
    
    if not orders_path.exists():
        raise FileNotFoundError(f"Silver orders no existe: {orders_path}")
    
    print("Cargando orders...")
    orders_df = pl.read_parquet(orders_path)
    
    # Agregar por ano/mes/categoria
    agg_cols = [
        pl.count().alias("order_count"),
        pl.col("order_total").sum().round(2).alias("total_revenue"),
        pl.col("order_total").mean().round(2).alias("avg_order_value"),
        pl.col("item_count").sum().alias("total_items"),
    ]
    
    group_cols = ["year", "month"]
    if "category_en" in orders_df.columns:
        group_cols.append("category_en")
    
    sales_df = orders_df.group_by(group_cols).agg(agg_cols).sort(group_cols)
    
    # Si hay reviews, agregar score promedio
    if reviews_path.exists():
        print("Cargando reviews para join...")
        reviews_df = pl.read_parquet(reviews_path)
        
        # Extraer year y month de review_answer_timestamp
        # (olist_reviews no tiene columnas year/month, solo timestamps)
        reviews_df = reviews_df.with_columns([
            pl.col("review_answer_timestamp")
              .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
              .dt.year()
              .alias("year"),
            pl.col("review_answer_timestamp")
              .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
              .dt.month()
              .alias("month"),
        ])
        
        # Agregar reviews por mes
        review_agg = reviews_df.group_by(["year", "month"]).agg([
            pl.col("review_score").mean().round(2).alias("avg_review_score"),
            pl.count().alias("review_count"),
        ])
        
        # Join
        sales_df = sales_df.join(review_agg, on=["year", "month"], how="left")
    
    print(f"Guardando gold: {output_path}")
    sales_df.write_parquet(output_path)
    
    print(f"Gold sales completado: {sales_df.height} filas")
    return output_path


def build_yelp_user_features(
    overwrite: bool = False
) -> Path:
    """
    Construye features de usuarios de Yelp.
    
    Calcula influence_score y otras metricas derivadas.
    
    Args:
        overwrite: Si sobrescribir
        
    Returns:
        Path al archivo de features
    """
    settings = get_settings()
    settings.ensure_dirs()
    
    output_path = settings.gold_dir / GOLD_FILES["yelp_user_stats"]
    
    if output_path.exists() and not overwrite:
        print(f"Gold users ya existe: {output_path}")
        return output_path
    
    users_path = settings.silver_dir / SILVER_FILES["yelp_users"]
    
    if not users_path.exists():
        raise FileNotFoundError(f"Silver users no existe: {users_path}")
    
    print("Cargando users...")
    df = pl.read_parquet(users_path)
    
    # Calcular influence score
    # Formula: combina review_count, fans, useful votes, elite status
    max_reviews = df["review_count"].max()
    max_fans = df["fans"].max()
    max_useful = df["useful"].max()
    
    df_gold = df.with_columns([
        # Normalizar componentes
        (pl.col("review_count") / max_reviews).alias("review_norm"),
        (pl.col("fans") / max_fans).alias("fans_norm"),
        (pl.col("useful") / max_useful).alias("useful_norm"),
    ])
    
    # Calcular influence score (promedio ponderado)
    df_gold = df_gold.with_columns([
        (
            pl.col("review_norm") * 0.3 +
            pl.col("fans_norm") * 0.3 +
            pl.col("useful_norm") * 0.3 +
            pl.col("is_elite").cast(pl.Float64) * 0.1
        ).round(4).alias("influence_score")
    ])
    
    # Eliminar columnas intermedias
    df_gold = df_gold.drop(["review_norm", "fans_norm", "useful_norm"])
    
    print(f"Guardando gold: {output_path}")
    df_gold.write_parquet(output_path)
    
    print(f"Gold users completado: {df_gold.height} filas")
    return output_path


def get_feature_stats(dataset: str) -> Dict[str, Any]:
    """
    Obtiene estadisticas de features gold.
    
    Util para el Router y QA.
    
    Args:
        dataset: Dataset
        
    Returns:
        Dict con estadisticas de cada feature
    """
    settings = get_settings()
    
    if dataset == "yelp":
        gold_path = settings.gold_dir / GOLD_FILES["yelp_features"]
    elif dataset == "es":
        gold_path = settings.gold_dir / GOLD_FILES["es_features"]
    elif dataset == "olist":
        gold_path = settings.gold_dir / GOLD_FILES["olist_features"]
    else:
        return {"error": f"Dataset invalido: {dataset}"}
    
    if not gold_path.exists():
        return {"exists": False, "path": str(gold_path)}
    
    df = pl.read_parquet(gold_path)
    
    result = {
        "exists": True,
        "path": str(gold_path),
        "row_count": df.height,
        "columns": df.columns,
    }
    
    # Estadisticas de sentimiento si existen
    if "sentiment_label" in df.columns:
        sentiment_dist = df.group_by("sentiment_label").agg([
            pl.count().alias("count")
        ]).to_dicts()
        result["sentiment_distribution"] = sentiment_dist
    
    if "sentiment_score" in df.columns:
        result["avg_sentiment_score"] = round(df["sentiment_score"].mean(), 3)
    
    # Verificar features adicionales
    result["has_embeddings"] = "embedding_vector" in df.columns
    result["has_topics"] = "topic_id" in df.columns
    
    return result


def get_gold_status() -> Dict[str, Any]:
    """
    Obtiene el estado de todos los archivos gold.
    
    Returns:
        Dict con estado de cada archivo gold
    """
    settings = get_settings()
    
    status = {}
    for name, filename in GOLD_FILES.items():
        path = settings.gold_dir / filename
        if path.exists():
            df = pl.read_parquet(path)
            status[name] = {
                "exists": True,
                "path": str(path),
                "rows": df.height,
                "columns": len(df.columns),
            }
        else:
            status[name] = {
                "exists": False,
                "path": str(path),
            }
    
    return status
