from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    # Chargement unique du modèle en mémoire cache
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class SemanticMatcher:
    def __init__(self):
        self.model = load_model()
        
        # Définition des Blocs de Compétences pour l'analyse intermédiaire
        self.competency_blocks = {
            "Data Analysis": "Nettoyage de données, visualisation, dashboard, statistiques, SQL, Excel, interprétation business",
            "Machine Learning": "Modèles prédictifs, regression, classification, scikit-learn, algorithmes, entrainement modèle",
            "Data Engineering": "Infrastructure, cloud, pipeline, ETL, big data, spark, hadoop, base de données, automatisation",
            "NLP & GenAI": "Texte, langage naturel, LLM, transformers, chatbot, hugging face, embeddings"
        }
        
        # Pré-calcul des embeddings des blocs
        self.block_embeddings = {
            name: self.model.encode([desc]) 
            for name, desc in self.competency_blocks.items()
        }

    def calculate_block_scores(self, user_text):
        """Calcul la similarité entre l'utilisateur et les grands blocs."""
        user_emb = self.model.encode([user_text])
        scores = {}
        for name, block_emb in self.block_embeddings.items():
            sim = cosine_similarity(user_emb, block_emb)[0][0]
            scores[name] = max(0, float(sim))
        return scores

    def get_job_weights(self, job_title):
        """Définit les poids (Wi) selon le titre du métier."""
        weights = {k: 1.0 for k in self.competency_blocks}
        title = job_title.lower()
        
        if "scientist" in title:
            weights["Machine Learning"] = 2.0
            weights["Data Analysis"] = 1.5
        elif "analyst" in title or "bi" in title:
            weights["Data Analysis"] = 2.5
            weights["Soft Skills"] = 1.5
            weights["Machine Learning"] = 0.5
        elif "engineer" in title or "architect" in title:
            weights["Data Engineering"] = 2.5
        elif "nlp" in title or "generative" in title:
            weights["NLP & GenAI"] = 3.0
            
        return weights

    def apply_business_rules(self, job_title, base_score, filters):
        """Applique les bonus/malus selon les niveaux maths/code."""
        final_score = base_score
        reasons = []
        
        # Règle Maths
        math_heavy = ["Scientist", "Research", "Algorithm", "Vision", "NLP"]
        # On vérifie si le niveau utilisateur est faible (Débutant/Notions)
        if filters.get('math_level') in ["Débutant", "Notions", "Beginner", "Basic"] and any(k in job_title for k in math_heavy):
            final_score -= 0.60
            reasons.append("⛔ Bloqué par le niveau Maths (-60%)")

        # Règle Code
        code_heavy = ["Engineer", "Developer", "Architect", "Backend"]
        if filters.get('code_level', 5) < 3 and any(k in job_title for k in code_heavy):
            final_score -= 0.50
            reasons.append("⛔ Bloqué par le niveau Code (-50%)")

        # Règle Domaine
        fav_domain = filters.get('domain', '')
        if fav_domain != "Peu importe" and fav_domain != "Any":
            # Correspondance simple
            if fav_domain[:4].lower() in job_title.lower():
                 final_score += 0.10
                 reasons.append("🚀 Bonus Domaine (+10%)")
             
        return final_score, reasons

    def find_top_matches(self, user_text, df_jobs, filters=None, top_k=3):
        block_scores = self.calculate_block_scores(user_text)
        hybrid_results = []
        
        for _, row in df_jobs.iterrows():
            job_title = row['titre']
            
            # 1. Score de couverture pondéré
            weights = self.get_job_weights(job_title)
            numerator = sum(weights[block] * block_scores[block] for block in self.competency_blocks)
            denominator = sum(weights.values())
            coverage_score = numerator / denominator
            
            # 2. Règles métiers
            final_score, reasons = self.apply_business_rules(job_title, coverage_score, filters or {})
            
            # 3. Construction du résultat (CORRECTION DU BUG ICI : on utilise 'skills')
            hybrid_results.append({
                "titre": job_title,
                "score": float(final_score),
                "original_score": float(coverage_score),
                "block_scores": block_scores,
                "description": row['description'],
                "skills": row['skills'], # <--- C'est ici que ça plantait (avant c'était it_skills)
                "reasons": reasons
            })
            
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_results[:top_k]