import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_data
def encode_jobs(_model, job_descriptions):
    return _model.encode(job_descriptions)

class SemanticMatcher:
    def __init__(self):
        self.model = load_model()

    def apply_business_rules(self, row, score, filters):
        debug_reasons = []
        final_score = score
        
        # --- RÈGLE 1 : LA BARRIÈRE DES MATHS (Renforcée) ---
        # Si utilisateur < Intermédiaire, on TUE le score des métiers scientifiques
        math_heavy_jobs = ["Scientist", "Research", "Financial", "Algorithm", "RAG", "Generative", "Vision", "NLP"]
        user_math_weak = filters.get('math_level') in ["Débutant", "Notions"]
        
        if user_math_weak and any(keyword in row['titre'] for keyword in math_heavy_jobs):
            # Avant : penalty = 0.25
            penalty = 0.60 # ÉNORME MALUS (-60%)
            final_score -= penalty
            debug_reasons.append("⛔ Bloqué par le niveau Maths (-60%)")

        # --- RÈGLE 2 : LA BARRIÈRE DU CODE (Renforcée) ---
        code_heavy_jobs = ["Engineer", "Developer", "Architect", "Backend", "Fullstack"]
        user_code_weak = filters.get('code_level', 5) < 3 # Si moins de 3/5
        
        if user_code_weak and any(keyword in row['titre'] for keyword in code_heavy_jobs):
            penalty = 0.50 # GROS MALUS (-50%)
            final_score -= penalty
            debug_reasons.append("⛔ Bloqué par le niveau Code (-50%)")

        # RÈGLE 3 : Le Bonus de Domaine
        # Si le métier contient le mot du domaine préféré, petit boost
        fav_domain = filters.get('domain', '')
        if fav_domain != "Peu importe":
            # Simplification : on regarde si des mots clés du domaine sont dans le titre
            keywords_domain = {
                "Développement & Code": ["Developer", "Engineer"],
                "Analyse & Business": ["Analyst", "Consultant", "BI", "Product"],
                "Infrastructure & Cloud": ["Cloud", "Architect", "Reliability"],
                "Mathématiques & Recherche": ["Scientist", "Research"],
                "Éthique & Gouvernance": ["Ethicist", "Steward", "DPO", "Governance"]
            }
            
            target_keywords = keywords_domain.get(fav_domain, [])
            if any(k in row['titre'] for k in target_keywords):
                final_score += 0.10 # +10%
                debug_reasons.append("🚀 Bonus Domaine (+10%)")

        return final_score, debug_reasons

    def find_top_matches(self, user_text, df_jobs, filters=None, top_k=3):
        # 1. Calcul SBERT (Comme avant)
        user_embedding = self.model.encode([user_text])
        job_descriptions = df_jobs['description_semantique'].tolist()
        job_embeddings = encode_jobs(self.model, job_descriptions)
        
        semantic_scores = cosine_similarity(user_embedding, job_embeddings)[0]
        
        # 2. Application du Scoring Hybride
        hybrid_results = []
        
        for idx, score in enumerate(semantic_scores):
            row = df_jobs.iloc[idx]
            
            # Appel de la fonction de règles
            final_score, reasons = self.apply_business_rules(row, score, filters or {})
            
            hybrid_results.append({
                "titre": row['titre'],
                "original_score": float(score),
                "score": float(final_score),
                "description": row['description_semantique'],
                "competences": row['competences_cles'],
                "reasons": reasons # On garde les explications pour l'affichage
            })
            
        # 3. Tri sur le NOUVEAU score hybride
        # On trie la liste de dictionnaires par la clé 'score'
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        return hybrid_results[:top_k]