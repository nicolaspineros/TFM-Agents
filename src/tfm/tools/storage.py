"""
Gestion de storage con DuckDB.

Este modulo maneja:
- Conexion singleton a DuckDB
- Registro de tablas Parquet como vistas
- Consultas SQL
- Verificacion de existencia de artefactos

El warehouse DuckDB actua como capa de consulta sobre los archivos
Parquet en silver/ y gold/. No duplicamos datos; solo registramos
vistas que apuntan a los archivos.
"""

from pathlib import Path
from typing import Optional, Any, Dict, List
from contextlib import contextmanager

import duckdb

from tfm.config.settings import get_settings


# Singleton de conexion
_connection: Optional[duckdb.DuckDBPyConnection] = None


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Obtiene conexion singleton a DuckDB.
    
    Crea la conexion si no existe. El archivo se crea en
    warehouse/tfm.duckdb.
    
    Returns:
        Conexion DuckDB
        
    Example:
        >>> conn = get_duckdb_connection()
        >>> result = conn.execute("SELECT * FROM yelp_reviews LIMIT 5").fetchall()
    """
    global _connection
    
    if _connection is not None:
        return _connection
    
    settings = get_settings()
    settings.ensure_dirs()
    
    _connection = duckdb.connect(str(settings.warehouse_path))
    return _connection


@contextmanager
def duckdb_transaction():
    """
    Context manager para transacciones DuckDB.
    
    Example:
        >>> with duckdb_transaction() as conn:
        ...     conn.execute("INSERT INTO ...")
    """
    conn = get_duckdb_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def register_parquet_table(
    table_name: str,
    parquet_path: Path,
    replace: bool = True
) -> bool:
    """
    Registra un archivo Parquet como vista en DuckDB.
    
    Args:
        table_name: Nombre de la tabla/vista en DuckDB
        parquet_path: Ruta al archivo Parquet
        replace: Si reemplazar si ya existe
        
    Returns:
        True si se registro correctamente
        
    Example:
        >>> register_parquet_table("yelp_reviews", Path("data/silver/yelp_reviews.parquet"))
        True
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet no encontrado: {parquet_path}")
    
    conn = get_duckdb_connection()
    create_or_replace = "CREATE OR REPLACE" if replace else "CREATE"
    
    # Usar path absoluto y normalizado
    path_str = str(parquet_path.resolve()).replace("\\", "/")
    
    conn.execute(f"""
        {create_or_replace} VIEW {table_name} AS
        SELECT * FROM read_parquet('{path_str}')
    """)
    return True


def list_registered_tables() -> List[str]:
    """
    Lista todas las tablas/vistas registradas en DuckDB.
    
    Returns:
        Lista de nombres de tablas
    """
    conn = get_duckdb_connection()
    result = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
    """).fetchall()
    return [row[0] for row in result]


def table_exists(table_name: str) -> bool:
    """
    Verifica si una tabla/vista existe en DuckDB.
    
    Args:
        table_name: Nombre de la tabla a verificar
        
    Returns:
        True si existe
    """
    try:
        tables = list_registered_tables()
        return table_name in tables
    except Exception:
        return False


def execute_query(
    query: str,
    params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Ejecuta una query SQL y retorna resultados como lista de dicts.
    
    Args:
        query: Query SQL
        params: Parametros para la query (opcional)
        
    Returns:
        Lista de dicts con los resultados
        
    Example:
        >>> results = execute_query(
        ...     "SELECT * FROM yelp_reviews WHERE stars = ? LIMIT 10",
        ...     params={"stars": 5}
        ... )
    """
    conn = get_duckdb_connection()
    
    if params:
        # Convertir dict a lista de valores para DuckDB
        result = conn.execute(query, list(params.values()))
    else:
        result = conn.execute(query)
    
    if result.description is None:
        return []
    
    columns = [desc[0] for desc in result.description]
    return [dict(zip(columns, row)) for row in result.fetchall()]


def artifact_exists(artifact_path: str) -> bool:
    """
    Verifica si un artefacto (Parquet, JSON) existe.
    
    Usado por el Router para lazy computation.
    
    Args:
        artifact_path: Ruta relativa al artefacto
        
    Returns:
        True si existe
    """
    settings = get_settings()
    full_path = settings.data_dir.parent / artifact_path
    return full_path.exists()


def get_artifact_metadata(artifact_path: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene metadata de un artefacto (tamano, fecha, row_count).
    
    Args:
        artifact_path: Ruta relativa al artefacto
        
    Returns:
        Dict con metadata o None si no existe
    """
    import pyarrow.parquet as pq
    
    if not artifact_exists(artifact_path):
        return None
    
    settings = get_settings()
    full_path = settings.data_dir.parent / artifact_path
    
    try:
        pf = pq.ParquetFile(full_path)
        return {
            "row_count": pf.metadata.num_rows,
            "num_columns": pf.metadata.num_columns,
            "created_by": pf.metadata.created_by,
            "file_size_bytes": full_path.stat().st_size,
            "file_size_mb": round(full_path.stat().st_size / (1024 * 1024), 2),
        }
    except Exception as e:
        return {"error": str(e)}


def register_all_silver_tables() -> Dict[str, bool]:
    """
    Registra todas las tablas silver existentes en DuckDB.
    
    Returns:
        Dict con resultado por tabla
    """
    from tfm.config.settings import SILVER_FILES
    
    settings = get_settings()
    results = {}
    
    for table_name, filename in SILVER_FILES.items():
        path = settings.silver_dir / filename
        if path.exists():
            try:
                register_parquet_table(table_name, path)
                results[table_name] = True
            except Exception as e:
                results[table_name] = False
        else:
            results[table_name] = False
    
    return results


def close_connection():
    """
    Cierra la conexion a DuckDB.
    
    Importante llamar al final de scripts para liberar el archivo.
    """
    global _connection
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass  # Ignorar errores al cerrar
        _connection = None


def try_register_parquet_table(
    table_name: str,
    parquet_path: Path,
    replace: bool = True,
    silent: bool = False
) -> bool:
    """
    Intenta registrar un Parquet en DuckDB sin lanzar excepciones.
    
    Version robusta de register_parquet_table que maneja errores
    de conexion (archivo bloqueado, etc.) de forma graceful.
    
    Args:
        table_name: Nombre de la tabla/vista
        parquet_path: Ruta al archivo Parquet
        replace: Si reemplazar si ya existe
        silent: Si True, no imprime mensajes de error
        
    Returns:
        True si se registro correctamente, False si hubo error
    """
    try:
        return register_parquet_table(table_name, parquet_path, replace)
    except duckdb.IOException as e:
        if not silent:
            print(f"  WARN: No se pudo registrar {table_name} en DuckDB (archivo bloqueado)")
        return False
    except FileNotFoundError:
        if not silent:
            print(f"  WARN: Parquet no encontrado: {parquet_path}")
        return False
    except Exception as e:
        if not silent:
            print(f"  WARN: Error registrando {table_name}: {e}")
        return False


def is_duckdb_available() -> bool:
    """
    Verifica si DuckDB esta disponible (no bloqueado por otro proceso).
    
    Returns:
        True si se puede conectar, False si esta bloqueado
    """
    global _connection
    
    # Si ya tenemos conexion, esta disponible
    if _connection is not None:
        return True
    
    settings = get_settings()
    settings.ensure_dirs()
    
    try:
        # Intentar conectar
        test_conn = duckdb.connect(str(settings.warehouse_path))
        test_conn.close()
        return True
    except duckdb.IOException:
        return False
    except Exception:
        return False


def register_all_gold_tables() -> Dict[str, bool]:
    """
    Registra todas las tablas gold existentes en DuckDB.
    
    Returns:
        Dict con resultado por tabla
    """
    from tfm.config.settings import GOLD_FILES
    
    settings = get_settings()
    results = {}
    
    for table_name, filename in GOLD_FILES.items():
        path = settings.gold_dir / filename
        if path.exists():
            results[table_name] = try_register_parquet_table(table_name, path, silent=True)
        else:
            results[table_name] = False
    
    return results
