import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai
import os

# Chemins des fichiers (relatifs)
METADATA_PATH = 'output/articles_metadata.parquet'
EMBEDDINGS_PATH = 'output/articles_embeddings.npy'
MODEL_NAME = 'all-MiniLM-L6-v2'

@st.cache_resource
def load_data():
    """
    Charge les données une seule fois et les garde en mémoire cache.
    """
    if not os.path.exists(METADATA_PATH) or not os.path.exists(EMBEDDINGS_PATH):
        st.error("❌ Fichiers de données introuvables. Vérifiez le dossier 'data'.")
        return None, None, None

    # 1. Charger les métadonnées
    df = pd.read_parquet(METADATA_PATH)
    
    # 2. Charger les embeddings
    embeddings = np.load(EMBEDDINGS_PATH)
    
    # 3. Charger le modèle AI
    model = SentenceTransformer(MODEL_NAME)
    
    return df, embeddings, model

def search_articles(query, model, embeddings, df, top_k=5):
    """
    Exécute la recherche sémantique.
    """
    # Encoder la requête
    query_vec = model.encode([query])
    
    # Calculer la similarité (Dot Product)
    scores = np.dot(embeddings, query_vec.T).flatten()
    
    # Récupérer les top K indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    # Construire les résultats
    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            'score': scores[idx],
            'title': row.get('title', 'No Title'),
            'abstract': row.get('abstract', 'No Abstract'),
            'authors': row.get('authors', 'Unknown'),
            'year': row.get('publication_year', 'N/A'),
            'doi': row.get('doi', ''),
            'affiliation': row.get('affiliation', '')
        })
    return results




os.environ["GOOGLE_API_KEY"] = "AIzaSyBJr5sRLI1NkRLFsa4wDtwzAgRQ2jExmOU"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def analyze_with_gemini(query, articles, api_key):
    """
    Envoie les titres et abstracts à Gemini pour une analyse de pertinence.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash') # ou 'gemini-1.5-flash' pour la vitesse
        
        # On prépare le contexte pour l'IA
        context = f"User Query: '{query}'\n\nHere are the top scientific articles found:\n"
        
        for i, art in enumerate(articles):
            context += f"--- Article {i+1} ---\n"
            context += f"Title: {art['title']}\n"
            context += f"Abstract: {art['abstract'][:800]}...\n" # On tronque pour économiser des tokens
        
        prompt = context + """
        \nTask:
        For each article, provide a brief analysis in JSON format with these fields:
        1. "summary": A 1-sentence simplified summary of what this paper actually does.
        2. "relevance_score": A score from 0 to 100 based on the user query.
        3. "relevance_reason": Explain WHY it is relevant or not (e.g., "Exact match for method X", or "Related but focuses on Y").
        
        Return ONLY the list of JSON objects, nothing else.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Erreur Gemini : {str(e)}"