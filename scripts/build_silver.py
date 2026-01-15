#!/usr/bin/env python
"""
Script para construir la capa silver del data warehouse.

Este script transforma los datos bronze (raw) a formato silver (limpio).
Ejecutar desde la raiz del proyecto:

    uv run python scripts/build_silver.py

Opciones:
    --dataset DATASET  Solo construir un dataset (yelp, es, olist)
    --overwrite        Sobrescribir archivos existentes
    --check            Solo verificar estado sin construir
"""

import argparse
import sys
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tfm.tools.preprocess import (
    build_silver_yelp,
    build_silver_yelp_users,
    build_silver_es,
    build_silver_olist,
    build_all_silver,
    check_silver_status,
)
from tfm.tools.io_loaders import check_bronze_files


def main():
    parser = argparse.ArgumentParser(
        description="Construir silver layer del TFM"
    )
    parser.add_argument(
        "--dataset",
        choices=["yelp", "yelp_users", "es", "olist", "all"],
        default="all",
        help="Dataset a construir (default: all)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Sobrescribir archivos existentes"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Solo verificar estado sin construir"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limite de filas (para desarrollo)"
    )
    
    args = parser.parse_args()
    
    # Verificar bronze files
    print("=" * 60)
    print("VERIFICANDO ARCHIVOS BRONZE")
    print("=" * 60)
    
    bronze_status = check_bronze_files()
    all_bronze_ok = True
    
    for dataset, tables in bronze_status.items():
        print(f"\n{dataset.upper()}:")
        for table, exists in tables.items():
            status = "OK" if exists else "FALTA"
            print(f"  [{status}] {table}")
            if not exists:
                all_bronze_ok = False
    
    if not all_bronze_ok:
        print("\nALGUNOS ARCHIVOS BRONZE NO EXISTEN")
        print("Coloca los datasets en data/bronze/")
    
    # Solo check
    if args.check:
        print("\n" + "=" * 60)
        print("ESTADO SILVER LAYER")
        print("=" * 60)
        
        silver_status = check_silver_status()
        for name, info in silver_status.items():
            if info["exists"]:
                print(f"  [OK] {name}: {info['size_mb']} MB")
            else:
                print(f"  [--] {name}: No existe")
        
        return 0
    
    # Construir
    print("\n" + "=" * 60)
    print("CONSTRUYENDO SILVER LAYER")
    print("=" * 60)
    
    try:
        if args.dataset == "all":
            results = build_all_silver(overwrite=args.overwrite)
            for name, path in results.items():
                print(f"  {name}: {path}")
        
        elif args.dataset == "yelp":
            path = build_silver_yelp(limit=args.limit, overwrite=args.overwrite)
            print(f"  yelp_reviews: {path}")
        
        elif args.dataset == "yelp_users":
            path = build_silver_yelp_users(limit=args.limit, overwrite=args.overwrite)
            print(f"  yelp_users: {path}")
        
        elif args.dataset == "es":
            path = build_silver_es(limit=args.limit, overwrite=args.overwrite)
            print(f"  es: {path}")
        
        elif args.dataset == "olist":
            orders_path, reviews_path = build_silver_olist(overwrite=args.overwrite)
            print(f"  olist_orders: {orders_path}")
            print(f"  olist_reviews: {reviews_path}")
        
        print("\nSILVER LAYER COMPLETADO")
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
