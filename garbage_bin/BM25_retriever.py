from rank_bm25 import BM25Okapi
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Scarica risorse solo una volta
nltk.download('stopwords', quiet=True)

def preprocess_text(text, lang='italian', use_stemming=True, remove_stop=True):
    """
    Normalizza e tokenizza un testo.
    - Rimuove punteggiatura
    - Trasforma in minuscolo
    - Applica stopwords e stemming opzionali
    """
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    
    if remove_stop:
        stop_lang = set(stopwords.words(lang))
        tokens = [t for t in tokens if t not in stop_lang]
    
    if use_stemming:
        stemmer = SnowballStemmer(lang)
        tokens = [stemmer.stem(t) for t in tokens]
    
    return tokens


def bm25_search(query, documents, top_k=5, lang='italian', k1=1.5, b=0.75):
    """
    Esegue una ricerca BM25 su una lista di documenti.
    
    Args:
        query (str): testo della query
        documents (list[str]): lista di documenti testuali
        top_k (int): numero massimo di risultati
        lang (str): lingua per stopwords/stemming (default: 'italian')
        k1, b: parametri BM25
        
    Returns:
        list[dict]: lista di dizionari con 'document', 'score', 'index'
    """
    
    # Preprocessamento
    tokenized_docs = [preprocess_text(doc, lang) for doc in documents]
    tokenized_query = preprocess_text(query, lang)
    
    # Modello BM25
    bm25 = BM25Okapi(tokenized_docs, k1=k1, b=b)
    scores = np.array(bm25.get_scores(tokenized_query))
    
    # Ranking
    top_idx = scores.argsort()[::-1][:top_k]
    
    results = [
        {"index": i, "document": documents[i], "score": float(scores[i])}
        for i in top_idx
    ]
    
    return results


if __name__ == "__main__":
    docs = [
        "Questo è il primo documento.",
        "Questo documento è il secondo documento.",
        "E questo è il terzo uno.",
        "È il documento numero quattro.",
        "Infine, questo è il quinto documento."
    ]
    
    query = "secondo documento"
    results = bm25_search(query, docs, top_k=3)
    
    for res in results:
        print(f"Index: {res['index']}, Score: {res['score']:.4f}, Document: {res['document']}")