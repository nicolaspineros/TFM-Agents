#!/usr/bin/env python
"""
Script para construir el gold layer.

Ejecutar con:
    uv run python scripts/build_gold.py

Opciones:
    --sample N : Procesar solo N filas (para pruebas rapidas)
    --dataset X : Procesar solo dataset X (yelp, es, olist)
    --overwrite : Sobrescribir si existe
"""

import sys
import argparse
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tfm.tools.features import (
    build_gold_features,
    build_olist_sales_features,
    build_yelp_user_features,
    get_gold_status,
)
from tfm.tools.storage import (
    register_all_silver_tables,
    close_connection,
    is_duckdb_available,
)


def main():
    parser = argparse.ArgumentParser(description="Construir gold layer")
    parser.add_argument("--sample", type=int, help="Procesar solo N filas")
    parser.add_argument("--dataset", choices=["yelp", "es", "olist", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir si existe")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CONSTRUYENDO GOLD LAYER")
    print("=" * 60)
    
    # Verificar disponibilidad de DuckDB
    if not is_duckdb_available():
        print("\nADVERTENCIA: DuckDB no disponible (archivo bloqueado por otro proceso)")
        print("Los archivos Parquet se crearan pero no se registraran en DuckDB.")
        print("Cierra notebooks/procesos Python y vuelve a ejecutar si necesitas DuckDB.\n")
    else:
        # Registrar tablas silver en DuckDB
        print("\nRegistrando tablas silver en DuckDB...")
        register_all_silver_tables()
    
    datasets = ["yelp", "es", "olist"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"[{dataset.upper()}] Construyendo features...")
        print("=" * 60)
        
        try:
            path = build_gold_features(
                dataset=dataset,
                overwrite=args.overwrite,
                sample_size=args.sample,
            )
            print(f"  OK: {path}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Features adicionales para Olist
    if args.dataset in ["olist", "all"]:
        print(f"\n{'='*60}")
        print("[OLIST] Construyendo sales features...")
        print("=" * 60)
        try:
            path = build_olist_sales_features(overwrite=args.overwrite)
            print(f"  OK: {path}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Features de usuarios Yelp
    if args.dataset in ["yelp", "all"]:
        print(f"\n{'='*60}")
        print("[YELP] Construyendo user features...")
        print("=" * 60)
        try:
            path = build_yelp_user_features(overwrite=args.overwrite)
            print(f"  OK: {path}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Mostrar estado final
    print(f"\n{'='*60}")
    print("ESTADO FINAL GOLD LAYER")
    print("=" * 60)
    
    status = get_gold_status()
    for name, info in status.items():
        if info.get("exists"):
            print(f"  [OK] {name}: {info['rows']} filas")
        else:
            print(f"  [--] {name}: no existe")
    
    # Cerrar conexion DuckDB para liberar archivo
    close_connection()
    
    print("\nGOLD LAYER COMPLETADO")


if __name__ == "__main__":
    main()
