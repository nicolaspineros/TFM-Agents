#!/usr/bin/env python
"""
Script para consultar el estado y número de registros en cada capa de datos.

Muestra información de:
- Bronze: archivos raw (JSON, CSV)
- Silver: datos limpios (Parquet)
- Gold: features calculadas (Parquet)

Ejecutar desde la raíz del proyecto:

    uv run python scripts/data_status.py

Opciones:
    --layer LAYER  Solo mostrar una capa (bronze, silver, gold, all)
    --verbose      Mostrar más detalles (columnas, tipos)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tfm.config.settings import get_settings, BRONZE_FILES, SILVER_FILES, GOLD_FILES


def format_number(n: int) -> str:
    """Formatea número con separadores de miles."""
    return f"{n:,}"


def format_size(size_bytes: int) -> str:
    """Formatea tamaño en bytes a formato legible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_bronze_status(verbose: bool = False) -> Dict[str, Any]:
    """Obtiene estado de archivos bronze."""
    settings = get_settings()
    status = {}
    total_rows = 0
    
    for dataset, files in BRONZE_FILES.items():
        if dataset == "yelp":
            base_dir = settings.yelp_bronze_dir
        elif dataset == "es":
            base_dir = settings.es_bronze_dir
        elif dataset == "olist":
            base_dir = settings.olist_bronze_dir
        else:
            continue
        
        status[dataset] = {}
        
        for table, filename in files.items():
            path = base_dir / filename
            if path.exists():
                size = path.stat().st_size
                
                # Contar filas según tipo de archivo
                rows = None
                try:
                    if filename.endswith('.json'):
                        # JSONL: contar líneas
                        with open(path, 'r', encoding='utf-8') as f:
                            rows = sum(1 for _ in f)
                    elif filename.endswith('.csv'):
                        # CSV: contar líneas - 1 (header)
                        with open(path, 'r', encoding='utf-8') as f:
                            rows = sum(1 for _ in f) - 1
                except Exception as e:
                    rows = f"Error: {e}"
                
                status[dataset][table] = {
                    "exists": True,
                    "path": str(path),
                    "size": format_size(size),
                    "rows": rows if rows else "N/A",
                }
                if isinstance(rows, int):
                    total_rows += rows
            else:
                status[dataset][table] = {
                    "exists": False,
                    "path": str(path),
                }
    
    status["_total_rows"] = total_rows
    return status


def get_silver_status(verbose: bool = False) -> Dict[str, Any]:
    """Obtiene estado de archivos silver."""
    import polars as pl
    
    settings = get_settings()
    status = {}
    total_rows = 0
    
    for name, filename in SILVER_FILES.items():
        path = settings.silver_dir / filename
        if path.exists():
            try:
                # Leer metadata sin cargar todo el archivo
                df = pl.scan_parquet(path)
                schema = df.collect_schema()
                
                # Contar filas (más rápido con lazy)
                rows = df.select(pl.len()).collect().item()
                total_rows += rows
                
                info = {
                    "exists": True,
                    "path": str(path),
                    "size": format_size(path.stat().st_size),
                    "rows": format_number(rows),
                    "columns": len(schema),
                }
                
                if verbose:
                    info["column_names"] = list(schema.names())
                
                status[name] = info
                
            except Exception as e:
                status[name] = {
                    "exists": True,
                    "path": str(path),
                    "error": str(e),
                }
        else:
            status[name] = {
                "exists": False,
                "path": str(path),
            }
    
    status["_total_rows"] = total_rows
    return status


def get_gold_status(verbose: bool = False) -> Dict[str, Any]:
    """Obtiene estado de archivos gold."""
    import polars as pl
    
    settings = get_settings()
    status = {}
    total_rows = 0
    
    for name, filename in GOLD_FILES.items():
        path = settings.gold_dir / filename
        if path.exists():
            try:
                df = pl.scan_parquet(path)
                schema = df.collect_schema()
                rows = df.select(pl.len()).collect().item()
                total_rows += rows
                
                info = {
                    "exists": True,
                    "path": str(path),
                    "size": format_size(path.stat().st_size),
                    "rows": format_number(rows),
                    "columns": len(schema),
                }
                
                if verbose:
                    info["column_names"] = list(schema.names())
                
                status[name] = info
                
            except Exception as e:
                status[name] = {
                    "exists": True,
                    "path": str(path),
                    "error": str(e),
                }
        else:
            status[name] = {
                "exists": False,
                "path": str(path),
            }
    
    status["_total_rows"] = total_rows
    return status


def print_bronze_status(status: Dict[str, Any], verbose: bool = False):
    """Imprime estado de bronze."""
    print("\n" + "=" * 70)
    print("BRONZE LAYER (Datos Raw)")
    print("=" * 70)
    
    for dataset, tables in status.items():
        if dataset.startswith("_"):
            continue
        
        print(f"\n  [{dataset.upper()}]")
        for table, info in tables.items():
            if info["exists"]:
                rows = info.get("rows", "N/A")
                if isinstance(rows, int):
                    rows = format_number(rows)
                print(f"    [OK] {table}")
                print(f"         Filas: {rows}")
                print(f"         Tamaño: {info['size']}")
            else:
                print(f"    [--] {table}: NO EXISTE")
    
    total = status.get("_total_rows", 0)
    print(f"\n  TOTAL BRONZE: {format_number(total)} registros")


def print_silver_status(status: Dict[str, Any], verbose: bool = False):
    """Imprime estado de silver."""
    print("\n" + "=" * 70)
    print("SILVER LAYER (Datos Limpios - Parquet)")
    print("=" * 70)
    
    for name, info in status.items():
        if name.startswith("_"):
            continue
        
        if info["exists"]:
            if "error" in info:
                print(f"\n  [!!] {name}: Error - {info['error']}")
            else:
                print(f"\n  [OK] {name}")
                print(f"       Filas: {info['rows']}")
                print(f"       Columnas: {info['columns']}")
                print(f"       Tamaño: {info['size']}")
                if verbose and "column_names" in info:
                    print(f"       Cols: {', '.join(info['column_names'][:5])}...")
        else:
            print(f"\n  [--] {name}: NO EXISTE")
    
    total = status.get("_total_rows", 0)
    print(f"\n  TOTAL SILVER: {format_number(total)} registros")


def print_gold_status(status: Dict[str, Any], verbose: bool = False):
    """Imprime estado de gold."""
    print("\n" + "=" * 70)
    print("GOLD LAYER (Features NLP/ML - Parquet)")
    print("=" * 70)
    
    for name, info in status.items():
        if name.startswith("_"):
            continue
        
        if info["exists"]:
            if "error" in info:
                print(f"\n  [!!] {name}: Error - {info['error']}")
            else:
                print(f"\n  [OK] {name}")
                print(f"       Filas: {info['rows']}")
                print(f"       Columnas: {info['columns']}")
                print(f"       Tamaño: {info['size']}")
                if verbose and "column_names" in info:
                    print(f"       Cols: {', '.join(info['column_names'][:5])}...")
        else:
            print(f"\n  [--] {name}: NO EXISTE")
    
    total = status.get("_total_rows", 0)
    print(f"\n  TOTAL GOLD: {format_number(total)} registros")


def print_summary(bronze: Dict, silver: Dict, gold: Dict):
    """Imprime resumen final."""
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    
    b_total = bronze.get("_total_rows", 0)
    s_total = silver.get("_total_rows", 0)
    g_total = gold.get("_total_rows", 0)
    
    print(f"""
  ┌─────────────┬──────────────────┐
  │ Capa        │ Registros        │
  ├─────────────┼──────────────────┤
  │ Bronze      │ {format_number(b_total):>16} │
  │ Silver      │ {format_number(s_total):>16} │
  │ Gold        │ {format_number(g_total):>16} │
  └─────────────┴──────────────────┘
""")
    
    # Verificar consistencia
    if s_total > 0 and g_total > 0:
        if s_total != g_total:
            print("Silver y Gold tienen diferente número de registros.")
            print("      Esto puede ser normal si gold tiene agregaciones.")
    
    if s_total == 0:
        print("Silver vacío. Ejecuta: uv run python scripts/build_silver.py")
    
    if g_total == 0:
        print("Gold vacío. Ejecuta: uv run python scripts/build_gold.py")


def main():
    parser = argparse.ArgumentParser(
        description="Consultar estado y registros de capas de datos"
    )
    parser.add_argument(
        "--layer",
        choices=["bronze", "silver", "gold", "all"],
        default="all",
        help="Capa a consultar (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar más detalles (nombres de columnas)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("  DATA STATUS - TFM Agents")
    print("=" * 70)
    
    bronze_status = None
    silver_status = None
    gold_status = None
    
    if args.layer in ["bronze", "all"]:
        bronze_status = get_bronze_status(args.verbose)
        print_bronze_status(bronze_status, args.verbose)
    
    if args.layer in ["silver", "all"]:
        silver_status = get_silver_status(args.verbose)
        print_silver_status(silver_status, args.verbose)
    
    if args.layer in ["gold", "all"]:
        gold_status = get_gold_status(args.verbose)
        print_gold_status(gold_status, args.verbose)
    
    if args.layer == "all":
        print_summary(bronze_status or {}, silver_status or {}, gold_status or {})
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
