"""
CLI para el sistema TFM.

Comandos disponibles:
- build-silver: Construye silver layer para un dataset
- build-gold: Construye gold features para un dataset
- ask: Hace una pregunta al sistema
- eval: Ejecuta evaluación offline
- profile: Muestra perfil de un dataset
- status: Muestra estado de artefactos

Uso:
    python -m tfm.cli build-silver --dataset yelp
    python -m tfm.cli ask "¿Cuál es el sentimiento promedio?"
"""

import argparse
import sys
import json
from typing import Optional


def main():
    """Entry point del CLI."""
    parser = argparse.ArgumentParser(
        prog="tfm",
        description="TFM: Sistema Agentic de Análisis de Reseñas"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # === build-silver ===
    silver_parser = subparsers.add_parser(
        "build-silver",
        help="Construye silver layer para un dataset"
    )
    silver_parser.add_argument(
        "--dataset", "-d",
        choices=["yelp", "es", "olist"],
        required=True,
        help="Dataset a procesar"
    )
    silver_parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Límite de filas (para desarrollo)"
    )
    silver_parser.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="Sobrescribir si existe"
    )
    
    # === build-gold ===
    gold_parser = subparsers.add_parser(
        "build-gold",
        help="Construye gold features para un dataset"
    )
    gold_parser.add_argument(
        "--dataset", "-d",
        choices=["yelp", "es", "olist"],
        required=True,
        help="Dataset a procesar"
    )
    gold_parser.add_argument(
        "--overwrite", "-f",
        action="store_true",
        help="Sobrescribir si existe"
    )
    gold_parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Incluir embeddings (costoso)"
    )
    
    # === ask ===
    ask_parser = subparsers.add_parser(
        "ask",
        help="Hace una pregunta al sistema"
    )
    ask_parser.add_argument(
        "query",
        type=str,
        help="Pregunta en lenguaje natural"
    )
    ask_parser.add_argument(
        "--dataset", "-d",
        choices=["yelp", "es", "olist"],
        default=None,
        help="Dataset preferido"
    )
    ask_parser.add_argument(
        "--json",
        action="store_true",
        help="Output en formato JSON"
    )
    
    # === eval ===
    eval_parser = subparsers.add_parser(
        "eval",
        help="Ejecuta evaluación offline"
    )
    eval_parser.add_argument(
        "--type", "-t",
        choices=["ml_metrics", "qa_faithfulness", "langsmith_eval", "all"],
        default="all",
        help="Tipo de evaluación"
    )
    eval_parser.add_argument(
        "--dataset", "-d",
        choices=["yelp", "es", "olist"],
        default=None,
        help="Dataset a evaluar"
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path para guardar resultados"
    )
    
    # === profile ===
    profile_parser = subparsers.add_parser(
        "profile",
        help="Muestra perfil de un dataset"
    )
    profile_parser.add_argument(
        "--dataset", "-d",
        choices=["yelp", "es", "olist"],
        required=True,
        help="Dataset a perfilar"
    )
    profile_parser.add_argument(
        "--layer",
        choices=["bronze", "silver", "gold"],
        default="silver",
        help="Capa a inspeccionar"
    )
    
    # === status ===
    status_parser = subparsers.add_parser(
        "status",
        help="Muestra estado de artefactos"
    )
    status_parser.add_argument(
        "--dataset", "-d",
        choices=["yelp", "es", "olist"],
        default=None,
        help="Dataset específico (opcional)"
    )
    
    # Parse args
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Ejecutar comando
    try:
        if args.command == "build-silver":
            cmd_build_silver(args)
        elif args.command == "build-gold":
            cmd_build_gold(args)
        elif args.command == "ask":
            cmd_ask(args)
        elif args.command == "eval":
            cmd_eval(args)
        elif args.command == "profile":
            cmd_profile(args)
        elif args.command == "status":
            cmd_status(args)
        else:
            parser.print_help()
            sys.exit(1)
    except NotImplementedError as e:
        print(f"  Funcionalidad no implementada: {e}")
        print("   Esta funcionalidad se implementará en fases posteriores.")
        sys.exit(1)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)


def cmd_build_silver(args):
    """Comando build-silver."""
    print(f" Construyendo silver layer para {args.dataset}...")
    
    from tfm.tools.preprocess import build_silver_yelp, build_silver_es, build_silver_olist
    
    if args.dataset == "yelp":
        path = build_silver_yelp(limit=args.limit, overwrite=args.overwrite)
    elif args.dataset == "es":
        path = build_silver_es(limit=args.limit, overwrite=args.overwrite)
    elif args.dataset == "olist":
        paths = build_silver_olist(overwrite=args.overwrite)
        path = paths[1]  # reviews path
    
    print(f" Silver layer creado: {path}")


def cmd_build_gold(args):
    """Comando build-gold."""
    print(f" Construyendo gold features para {args.dataset}...")
    
    from tfm.tools.features import build_gold_features
    
    path = build_gold_features(
        dataset=args.dataset,
        overwrite=args.overwrite,
        include_embeddings=args.embeddings
    )
    
    print(f" Gold features creados: {path}")


def cmd_ask(args):
    """Comando ask."""
    from tfm.graphs.conversation_graph import ask
    
    print(f" Procesando: {args.query}")
    print("-" * 50)
    
    result = ask(args.query, dataset=args.dataset)
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if result.get("success"):
            print(f" {result.get('summary', 'Sin resumen')}")
            print()
            
            bullets = result.get("bullets", [])
            if bullets:
                print("Insights:")
                for bullet in bullets:
                    if isinstance(bullet, dict):
                        text = bullet.get("text", str(bullet))
                        evidence = bullet.get("evidence", "")
                        print(f"  • {text}")
                        if evidence:
                            print(f"     {evidence}")
                    else:
                        print(f"  • {bullet}")
            
            caveats = result.get("caveats", [])
            if caveats:
                print()
                print("  Caveats:")
                for caveat in caveats:
                    print(f"  - {caveat}")
        else:
            print(f" Error: {result.get('error', 'Error desconocido')}")
            if result.get("qa_feedback"):
                print(f"   QA: {result['qa_feedback']}")


def cmd_eval(args):
    """Comando eval."""
    print(f" Ejecutando evaluación: {args.type}")
    
    from tfm.graphs.evaluation_graph import run_evaluation
    
    results = run_evaluation(
        eval_type=args.type,
        dataset=args.dataset,
        output_path=args.output
    )
    
    print(json.dumps(results, indent=2, default=str))
    
    if args.output:
        print(f" Resultados guardados en: {args.output}")


def cmd_profile(args):
    """Comando profile."""
    print(f" Perfil de {args.dataset} ({args.layer}):")
    print("-" * 50)
    
    from tfm.tools.profiling import profile_dataset
    
    profile = profile_dataset(
        dataset=args.dataset,
        layer=args.layer
    )
    
    if not profile.get("exists"):
        print(f" Dataset no encontrado: {profile.get('message', profile.get('path'))}")
        return
    
    print(f"Path: {profile.get('path')}")
    print(f"Filas: {profile.get('row_count', 'N/A')}")
    print(f"Columnas: {profile.get('columns', [])}")
    
    if profile.get("capabilities"):
        print()
        print("Capacidades:")
        for cap, val in profile["capabilities"].items():
            status = "OK" if val else "NO EXISTE"
            print(f"  {status} {cap}")


def cmd_status(args):
    """Comando status."""
    print("Estado de artefactos")
    print("=" * 50)
    
    from tfm.tools.profiling import check_artifacts_status
    
    datasets = [args.dataset] if args.dataset else ["yelp", "es", "olist"]
    
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        status = check_artifacts_status(dataset)
        
        for artifact, exists in status.items():
            icon = "OK" if exists else "NO EXISTE"
            print(f"  {icon} {artifact}")


if __name__ == "__main__":
    main()
