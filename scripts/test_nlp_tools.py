"""Test script para verificar tools NLP."""
import json
from tfm.tools.nlp_models import (
    get_sentiment_distribution,
    get_ambiguous_reviews_sentiment,
    get_aspect_distribution,
    get_nlp_models_status,
)

def main():
    print("=" * 60)
    print("VERIFICACION DE TOOLS NLP")
    print("=" * 60)
    
    # 1. Estado de modelos
    print("\n1. Estado de modelos NLP:")
    status = get_nlp_models_status.invoke({})
    print(f"   Ready: {status.get('ready')}")
    
    # 2. Distribucion de sentimiento
    print("\n2. Distribucion de sentimiento (Olist):")
    result = get_sentiment_distribution.invoke({"dataset": "olist"})
    if "error" not in result:
        print(f"   Total analizado: {result.get('total_analyzed'):,}")
        print(f"   Positivo: {result.get('percentages', {}).get('positive_pct')}%")
        print(f"   Neutral: {result.get('percentages', {}).get('neutral_pct')}%")
        print(f"   Negativo: {result.get('percentages', {}).get('negative_pct')}%")
        print(f"   Sentimiento promedio: {result.get('average_sentiment')}")
    else:
        print(f"   Error: {result.get('error')}")
    
    # 3. reseñass de 3 estrellas
    print("\n3. Sentimiento de reseñass 3 estrellas (ES):")
    result = get_ambiguous_reviews_sentiment.invoke({"dataset": "es"})
    if "error" not in result:
        print(f"   Total analizado: {result.get('total_analyzed'):,}")
        print(f"   Distribucion: pos={result.get('percentages', {}).get('positive_pct')}%, neg={result.get('percentages', {}).get('negative_pct')}%")
        print(f"   Sentimiento promedio: {result.get('average_sentiment')}")
        print(f"   Interpretacion: {result.get('interpretation')}")
    else:
        print(f"   Error: {result.get('error')}")
    
    # 4. Aspectos en reseñass negativas
    print("\n4. Aspectos en reseñass negativas (Yelp):")
    result = get_aspect_distribution.invoke({"dataset": "yelp", "sentiment_filter": "negative"})
    if "error" not in result:
        print(f"   Reviews analizadas: {result.get('total_reviews_analyzed'):,}")
        print(f"   Reviews con aspectos: {result.get('reviews_with_aspects'):,}")
        print(f"   Top 5 aspectos: {result.get('top_aspects')}")
    else:
        print(f"   Error: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print("VERIFICACION COMPLETADA")
    print("=" * 60)

if __name__ == "__main__":
    main()
