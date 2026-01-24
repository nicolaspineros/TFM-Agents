# Data Contracts

Este documento define los contratos de datos para cada dataset del proyecto.

---

## Resumen de Datasets

| Dataset | Idioma | Formato Raw | Tiene Fecha | Entidad Principal |
|---------|--------|-------------|-------------|-------------------|
| Yelp Academic | Inglés (en) | JSONL |  Sí | `business_id` |
| Reseñas ES | Español (es) | CSV |  No | `product_id` |
| Olist | Portugués (pt) | CSV múltiples |  Sí | `order_id`, `product_id` |

---

## 1. Yelp Academic Dataset

### 1.1 Bronze: `yelp_academic_dataset_review.json`

**Formato**: JSONL (una línea por review)

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `review_id` | string | No | ID único de la reseña |
| `user_id` | string | No | ID del usuario |
| `business_id` | string | No | ID del negocio |
| `stars` | int | No | Rating 1-5 |
| `useful` | int | No | Votos "útil" |
| `funny` | int | No | Votos "gracioso" |
| `cool` | int | No | Votos "cool" |
| `text` | string | No | Texto de la reseña |
| `date` | string | No | Fecha (YYYY-MM-DD) |

### 1.2 Bronze: `yelp_academic_dataset_business.json`

**Formato**: JSONL

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `business_id` | string | No | ID único del negocio |
| `name` | string | No | Nombre del negocio |
| `address` | string | Sí | Dirección |
| `city` | string | No | Ciudad |
| `state` | string | No | Estado |
| `postal_code` | string | Sí | Código postal |
| `latitude` | float | Sí | Latitud |
| `longitude` | float | Sí | Longitud |
| `stars` | float | No | Rating promedio |
| `review_count` | int | No | Número de reseñas |
| `is_open` | int | No | 0=cerrado, 1=abierto |
| `attributes` | object | Sí | Atributos del negocio (JSON) |
| `categories` | string | Sí | Categorías separadas por coma |
| `hours` | object | Sí | Horarios (JSON) |

### 1.3 Bronze: `yelp_academic_dataset_user.json`

**Formato**: JSONL (una línea por usuario)

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `user_id` | string | No | ID único del usuario (22 chars) |
| `name` | string | No | Nombre del usuario |
| `review_count` | int | No | Número de reviews escritas |
| `yelping_since` | string | No | Fecha de registro (YYYY-MM-DD) |
| `friends` | list[string] | No | Array de user_ids de amigos |
| `useful` | int | No | Votos "útil" enviados |
| `funny` | int | No | Votos "gracioso" enviados |
| `cool` | int | No | Votos "cool" enviados |
| `fans` | int | No | Número de fans |
| `elite` | list[int] | No | Años como usuario Elite |
| `average_stars` | float | No | Rating promedio de sus reviews |
| `compliment_hot` | int | No | Compliments "hot" recibidos |
| `compliment_more` | int | No | Compliments "more" recibidos |
| `compliment_profile` | int | No | Compliments "profile" recibidos |
| `compliment_cute` | int | No | Compliments "cute" recibidos |
| `compliment_list` | int | No | Compliments "list" recibidos |
| `compliment_note` | int | No | Compliments "note" recibidos |
| `compliment_plain` | int | No | Compliments "plain" recibidos |
| `compliment_cool` | int | No | Compliments "cool" recibidos |
| `compliment_funny` | int | No | Compliments "funny" recibidos |
| `compliment_writer` | int | No | Compliments "writer" recibidos |
| `compliment_photos` | int | No | Compliments "photos" recibidos |

> **Nota sobre Users**: Este dataset permite:
> - Análisis de influencia de usuarios (fans, elite status)
> - Correlación entre experiencia del usuario y calidad de reviews
> - Análisis de red social (amigos)
> - Segmentación de reviewers (casual vs power users)

### 1.4 Silver: `yelp_reviews.parquet`

Schema esperado después de limpieza:

| Campo | Tipo | Nullable | Transformación |
|-------|------|----------|----------------|
| `review_id` | string | No | Sin cambio |
| `user_id` | string | No | Sin cambio |
| `business_id` | string | No | Sin cambio |
| `stars` | int8 | No | Cast a int8 |
| `text` | string | No | Trim whitespace |
| `text_length` | int32 | No | `len(text)` |
| `word_count` | int32 | No | Conteo de palabras |
| `date` | date | No | Parse a date |
| `year` | int16 | No | Extraído de date |
| `month` | int8 | No | Extraído de date |
| `is_ambiguous` | bool | No | `stars == 3` |
| `language` | string | No | Siempre "en" |

### 1.5 Silver: `yelp_users.parquet`

Schema de usuarios procesados:

| Campo | Tipo | Nullable | Transformación |
|-------|------|----------|----------------|
| `user_id` | string | No | Sin cambio |
| `name` | string | No | Sin cambio |
| `review_count` | int32 | No | Sin cambio |
| `yelping_since` | date | No | Parse a date |
| `years_active` | int16 | No | Calculado desde yelping_since |
| `friends_count` | int32 | No | `len(friends)` |
| `useful` | int32 | No | Sin cambio |
| `funny` | int32 | No | Sin cambio |
| `cool` | int32 | No | Sin cambio |
| `total_votes_given` | int32 | No | useful + funny + cool |
| `fans` | int32 | No | Sin cambio |
| `elite_years_count` | int16 | No | `len(elite)` |
| `is_elite` | bool | No | `len(elite) > 0` |
| `average_stars` | float32 | No | Sin cambio |
| `total_compliments` | int32 | No | Suma de todos los compliments |
| `user_influence_score` | float32 | No | Score calculado (fans + compliments) |

### 1.6 Gold: `yelp_features.parquet`

Features NLP/ML calculadas:

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `review_id` | string | No | FK a silver |
| `user_id` | string | No | FK a users |
| `business_id` | string | No | FK a business |
| `sentiment_score` | float32 | No | Score -1 a 1 |
| `sentiment_label` | string | No | positive/negative/neutral |
| `aspect_keywords` | list[string] | Sí | Aspectos extraídos |
| `topic_id` | int16 | Sí | Cluster de tópico |
| `embedding_vector` | list[float32] | Sí | Vector de embedding |
| `user_influence_score` | float32 | Sí | Score del usuario (de yelp_users) |
| `user_is_elite` | bool | Sí | Si usuario es Elite |

### 1.7 Gold: `yelp_user_stats.parquet`

Estadísticas agregadas por usuario:

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `user_id` | string | No | PK |
| `total_reviews` | int32 | No | Reviews analizadas |
| `avg_sentiment` | float32 | No | Sentimiento promedio |
| `sentiment_std` | float32 | No | Desviación estándar |
| `pct_positive` | float32 | No | % reviews positivas |
| `pct_negative` | float32 | No | % reviews negativas |
| `top_categories` | list[string] | Sí | Categorías más reseñadas |
| `avg_text_length` | float32 | No | Longitud promedio de texto |
| `review_frequency` | float32 | No | Reviews/mes desde registro |

---

## 2. Reseñas en Español

### 2.1 Bronze: `reviews_dataframe_completo.csv`

**Formato**: CSV con encabezados

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `review_id` | string | No* | ID único (puede ser índice) |
| `product_id` | string | No | ID del producto |
| `stars` | int | No | Rating 1-5 |
| `review_body` | string | No | Texto de la reseña |
| `review_title` | string | Sí | Título de la reseña |
| `product_category` | string | Sí | Categoría del producto |
| `language` | string | Sí | Código de idioma |

> **Nota**: Este dataset NO tiene campo de fecha. No es posible hacer análisis temporal.

### 2.2 Silver: `es_reviews.parquet`

| Campo | Tipo | Nullable | Transformación |
|-------|------|----------|----------------|
| `review_id` | string | No | Generar si no existe |
| `product_id` | string | No | Sin cambio |
| `stars` | int8 | No | Cast a int8 |
| `text` | string | No | `review_body` limpio |
| `title` | string | Sí | `review_title` limpio |
| `text_length` | int32 | No | `len(text)` |
| `word_count` | int32 | No | Conteo de palabras |
| `category` | string | Sí | Normalizado |
| `is_ambiguous` | bool | No | `stars == 3` |
| `language` | string | No | "es" o detectado |

### 2.3 Gold: `es_features.parquet`

Similar a Yelp pero con modelo de español:

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `review_id` | string | No | FK a silver |
| `sentiment_score` | float32 | No | Score -1 a 1 |
| `sentiment_label` | string | No | positive/negative/neutral |
| `aspect_keywords` | list[string] | Sí | Aspectos (español) |

---

## 3. Olist Brazilian E-commerce

### 3.1 Bronze Files

#### `olist_orders_dataset.csv`

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `order_id` | string | No | ID único del pedido |
| `customer_id` | string | No | FK a customers |
| `order_status` | string | No | Estado del pedido |
| `order_purchase_timestamp` | datetime | No | Fecha/hora de compra |
| `order_approved_at` | datetime | Sí | Fecha/hora aprobación |
| `order_delivered_carrier_date` | datetime | Sí | Fecha entrega a carrier |
| `order_delivered_customer_date` | datetime | Sí | Fecha entrega a cliente |
| `order_estimated_delivery_date` | datetime | No | Fecha estimada |

#### `olist_order_items_dataset.csv`

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `order_id` | string | No | FK a orders |
| `order_item_id` | int | No | Número de item en pedido |
| `product_id` | string | No | FK a products |
| `seller_id` | string | No | FK a sellers |
| `shipping_limit_date` | datetime | No | Fecha límite envío |
| `price` | float | No | Precio del item |
| `freight_value` | float | No | Costo de envío |

#### `olist_order_reviews_dataset.csv`

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `review_id` | string | No | ID único de review |
| `order_id` | string | No | FK a orders |
| `review_score` | int | No | Score 1-5 |
| `review_comment_title` | string | Sí | Título (puede estar vacío) |
| `review_comment_message` | string | Sí | Mensaje (puede estar vacío) |
| `review_creation_date` | datetime | No | Fecha creación review |
| `review_answer_timestamp` | datetime | Sí | Fecha respuesta |



#### `olist_products_dataset.csv`

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `product_id` | string | No | ID único |
| `product_category_name` | string | Sí | Categoría (portugués) |
| `product_name_length` | int | Sí | Longitud nombre |
| `product_description_length` | int | Sí | Longitud descripción |
| `product_photos_qty` | int | Sí | Cantidad de fotos |
| `product_weight_g` | int | Sí | Peso en gramos |
| `product_length_cm` | int | Sí | Largo en cm |
| `product_height_cm` | int | Sí | Alto en cm |
| `product_width_cm` | int | Sí | Ancho en cm |

#### `product_category_name_translation.csv`

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `product_category_name` | string | No | Nombre en portugués |
| `product_category_name_english` | string | No | Nombre en inglés |

### 3.2 Silver: `olist_orders.parquet`

Join de orders + items + productos con categorías traducidas:

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `order_id` | string | No | PK |
| `customer_id` | string | No | FK |
| `product_id` | string | No | FK |
| `seller_id` | string | No | FK |
| `order_status` | string | No | Estado |
| `purchase_date` | date | No | Fecha de compra |
| `purchase_year` | int16 | No | Año |
| `purchase_month` | int8 | No | Mes |
| `purchase_dow` | int8 | No | Día de semana |
| `price` | float32 | No | Precio item |
| `freight_value` | float32 | No | Costo envío |
| `total_value` | float32 | No | price + freight |
| `category_pt` | string | Sí | Categoría portugués |
| `category_en` | string | Sí | Categoría inglés |
| `delivery_days` | int16 | Sí | Días hasta entrega |

### 3.3 Silver: `olist_reviews.parquet`

Reviews con texto para NLP:

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `review_id` | string | No | PK |
| `order_id` | string | No | FK a orders |
| `review_score` | int8 | No | 1-5 |
| `has_comment` | bool | No | Si tiene texto |
| `text` | string | Sí | Título + mensaje combinados |
| `text_length` | int32 | Sí | Longitud si existe |
| `review_date` | date | No | Fecha de review |
| `is_ambiguous` | bool | No | `review_score == 3` |
| `language` | string | No | "pt" |

### 3.4 Gold: `olist_sales_features.parquet`

Features para predicción de ventas:

| Campo | Tipo | Nullable | Descripción |
|-------|------|----------|-------------|
| `date` | date | No | Fecha (granularidad día) |
| `category_en` | string | No | Categoría |
| `total_orders` | int32 | No | Cantidad de órdenes |
| `total_revenue` | float32 | No | Ingresos totales |
| `avg_price` | float32 | No | Precio promedio |
| `avg_review_score` | float32 | Sí | Score promedio reviews |
| `review_count` | int32 | No | Cantidad de reviews |
| `sentiment_avg` | float32 | Sí | Sentimiento promedio (si calculado) |

---

## 4. Agregados Precomputados (Gold)

### 4.1 `agg_sentiment_by_month.parquet`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `dataset` | string | yelp/es/olist |
| `year` | int16 | Año |
| `month` | int8 | Mes |
| `review_count` | int64 | Total reseñas |
| `avg_stars` | float32 | Promedio stars |
| `avg_sentiment` | float32 | Sentimiento promedio |
| `pct_positive` | float32 | % positivas |
| `pct_negative` | float32 | % negativas |
| `pct_ambiguous` | float32 | % stars==3 |

### 4.2 `agg_sentiment_by_category.parquet`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `dataset` | string | yelp/es/olist |
| `category` | string | Categoría normalizada |
| `review_count` | int64 | Total reseñas |
| `avg_stars` | float32 | Promedio stars |
| `avg_sentiment` | float32 | Sentimiento promedio |
| `top_aspects` | list[string] | Aspectos más mencionados |

### 4.3 `agg_sales_forecast.parquet` (Olist)

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `date` | date | Fecha |
| `category_en` | string | Categoría |
| `actual_revenue` | float32 | Ingresos reales |
| `predicted_revenue` | float32 | Predicción modelo |
| `prediction_error` | float32 | MAE |
| `model_version` | string | Versión del modelo |


---

## 5. Particionado Dinámico

Las tools deben soportar filtros por:
- `date_range`: (start_date, end_date) - Solo Yelp y Olist
- `categories`: Lista de categorías
- `entity_ids`: Lista de business_id / product_id / order_id
- `language`: Código de idioma
- `stars`: Lista de valores (ej: [1,2] para negativos)
- `is_ambiguous`: True/False para filtrar stars==3

Ejemplo de llamada a tool:
```python
await aggregation_tool(
    dataset="yelp",
    metric="avg_sentiment",
    group_by=["category", "month"],
    filters={
        "date_range": ("2022-01-01", "2022-12-31"),
        "is_ambiguous": False
    }
)
```
