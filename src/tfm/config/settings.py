"""
Configuracion centralizada del proyecto TFM.

Este modulo define todas las configuraciones y paths del sistema.
Usa pydantic-settings para cargar variables de entorno.
"""

from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_project_root() -> Path:
    """
    Detecta la raiz del proyecto buscando pyproject.toml.
    Funciona desde cualquier ubicacion (notebooks, tests, scripts).
    """
    # Empezar desde el directorio de este archivo
    current = Path(__file__).resolve().parent
    
    # Subir hasta encontrar pyproject.toml
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback: usar el directorio actual
    return Path.cwd()


# Raiz del proyecto (calculada una vez)
PROJECT_ROOT = get_project_root()


class Settings(BaseSettings):
    """
    Configuracion global del proyecto TFM.
    
    Las variables se cargan desde:
    1. Variables de entorno
    2. Archivo .env
    
    Ejemplo de uso:
        >>> settings = get_settings()
        >>> print(settings.data_dir)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # === API Keys ===
    openai_api_key: str = Field(default="", description="OpenAI API Key")
    langsmith_api_key: str = Field(default="", description="LangSmith API Key")
    
    # === LangSmith ===
    langsmith_tracing: bool = Field(default=False, description="Habilitar tracing")
    langsmith_project: str = Field(default="tfm-agents", description="Nombre del proyecto en LangSmith")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="Endpoint de LangSmith"
    )
    
    # === Paths (ahora ABSOLUTOS basados en PROJECT_ROOT) ===
    data_dir: Path = Field(default=PROJECT_ROOT / "data", description="Directorio base de datos")
    warehouse_path: Path = Field(
        default=PROJECT_ROOT / "warehouse" / "tfm.duckdb",
        description="Ruta al archivo DuckDB"
    )
    runs_dir: Path = Field(default=PROJECT_ROOT / "runs", description="Directorio de checkpoints")
    
    # === Logging ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Nivel de logging"
    )
    
    # === Modelo LLM ===
    llm_model: str = Field(default="gpt-4o-mini", description="Modelo LLM por defecto")
    llm_temperature: float = Field(default=0.0, description="Temperatura del LLM")
    
    @property
    def bronze_dir(self) -> Path:
        """Directorio de datos crudos (bronze layer)."""
        return self.data_dir / "bronze"
    
    @property
    def silver_dir(self) -> Path:
        """Directorio de datos limpios (silver layer)."""
        return self.data_dir / "silver"
    
    @property
    def gold_dir(self) -> Path:
        """Directorio de features y agregados (gold layer)."""
        return self.data_dir / "gold"
    
    @property
    def yelp_bronze_dir(self) -> Path:
        """Directorio de datos Yelp raw."""
        return self.bronze_dir / "yelp"
    
    @property
    def es_bronze_dir(self) -> Path:
        """Directorio de datos español raw."""
        return self.bronze_dir / "rese_esp"
    
    @property
    def olist_bronze_dir(self) -> Path:
        """Directorio de datos Olist raw."""
        return self.bronze_dir / "olist_ecommerce"
    
    def ensure_dirs(self) -> None:
        """Crea todos los directorios necesarios si no existen."""
        for dir_path in [
            self.bronze_dir,
            self.silver_dir,
            self.gold_dir,
            self.runs_dir,
            self.warehouse_path.parent,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Obtiene la instancia singleton de Settings.
    
    Returns:
        Settings: Configuración cargada
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.llm_model)
        'gpt-4o-mini'
    """
    return Settings()


# Constantes de datasets
DATASET_NAMES = Literal["yelp", "es", "olist"]

# Mapeo de datasets a archivos bronze
BRONZE_FILES = {
    "yelp": {
        "reviews": "yelp_academic_dataset_review.json",
        "business": "yelp_academic_dataset_business.json",
        "users": "yelp_academic_dataset_user.json",
    },
    "es": {
        "reviews": "reviews_dataframe_completo.csv",
    },
    "olist": {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "categories": "product_category_name_translation.csv",
    },
}

# Mapeo de datasets a archivos silver
SILVER_FILES = {
    "yelp_reviews": "yelp_reviews.parquet",
    "yelp_users": "yelp_users.parquet",
    "yelp_business": "yelp_business.parquet",
    "es": "es_reviews.parquet",
    "olist_orders": "olist_orders.parquet",
    "olist_reviews": "olist_reviews.parquet",
}

# Mapeo de datasets a archivos gold
GOLD_FILES = {
    "yelp_features": "yelp_features.parquet",
    "yelp_user_stats": "yelp_user_stats.parquet",
    "es_features": "es_features.parquet",
    "olist_features": "olist_review_features.parquet",
    "olist_sales": "olist_sales_features.parquet",
}

# Configuración de samples para EDA (evitar cargar datasets completos)
EDA_SAMPLE_SIZES = {
    "yelp_reviews": 500_000,  # Yelp tiene millones, sample es suficiente
    "yelp_users": 500_000,
    "yelp_business": None,  # Business es pequeño, cargar completo
    "es_reviews": None,  # Cargar completo
    "olist": None,  # Olist es manejable, cargar completo
}
