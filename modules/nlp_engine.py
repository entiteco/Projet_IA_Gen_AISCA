from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class SemanticMatcher:
    def __init__(self):
        self.model = load_model()
        
        # --- EXIGENCE STEP 1 : DÉFINITION DES BLOCS DE COMPÉTENCES ---
        # On définit ce que signifie sémantiquement chaque bloc
        self.competency_blocks = {
            "Data Analysis": "Nettoyage de données, visualisation, dashboard, statistiques, SQL, Excel, interprétation business",
            "Machine Learning": "Modèles prédictifs, regression, classification, scikit-learn, algorithmes, entrainement modèle",
            "Data Engineering": "Infrastructure, cloud, pipeline, ETL, big data, spark, hadoop, base de données, automatisation",
            "NLP & GenAI": "Texte, langage naturel, LLM, transformers, chatbot, hugging face, embeddings",
            "Soft Skills": "Communication, gestion de projet, présentation, éthique, travail d'équipe, curiosité"
        }
        
        # On pré-calcule les embeddings des définitions des blocs
        self.block_embeddings = {
            name: self.model.encode([desc]) 
            for name, desc in self.competency_blocks.items()
        }

    def calculate_block_scores(self, user_text):
        """
        EXIGENCE STEP 3 : Calculer la similarité entre l'utilisateur et CHAQUE bloc.
        Renvoie un dictionnaire : {'Data Analysis': 0.85, 'NLP': 0.40 ...}
        """
        user_emb = self.model.encode([user_text])
        scores = {}
        
        for name, block_emb in self.block_embeddings.items():
            sim = cosine_similarity(user_emb, block_emb)[0][0]
            scores[name] = max(0, float(sim)) # On garde le score positif
            
        return scores

    def get_job_weights(self, job_title):
        """
        Définit les POIDS (Wi) pour la formule du STEP 4.
        Comme nous n'avons pas les poids dans le JSON, on les déduit du titre.
        """
        # Poids par défaut (équilibré)
        weights = {k: 1.0 for k in self.competency_blocks}
        
        title = job_title.lower()
        
        if "scientist" in title:
            weights["Machine Learning"] = 2.0
            weights["Data Analysis"] = 1.5
            weights["NLP & GenAI"] = 1.2
        elif "analyst" in title or "bi" in title:
            weights["Data Analysis"] = 2.5
            weights["Soft Skills"] = 1.5
            weights["Machine Learning"] = 0.5
        elif "engineer" in title or "architect" in title:
            weights["Data Engineering"] = 2.5
            weights["Machine Learning"] = 0.8
        elif "nlp" in title or "generative" in title:
            weights["NLP & GenAI"] = 3.0
            weights["Machine Learning"] = 1.5
            
        return weights

    def apply_business_rules(self, job_title, base_score, filters):
        """
        Vos règles métiers existantes (Bonus/Malus).
        Elles s'appliquent EN PLUS du score sémantique théorique.
        """
        final_score = base_score
        reasons = []
        
        # --- RÈGLE 1 : MATHS ---
        math_heavy = ["Scientist", "Research", "Algorithm", "Vision", "NLP"]
        if filters.get('math_level') in ["Débutant", "Notions"] and any(k in job_title for k in math_heavy):
            final_score -= 0.60
            reasons.append("⛔ Bloqué par le niveau Maths (-60%)")

        # --- RÈGLE 2 : CODE ---
        code_heavy = ["Engineer", "Developer", "Architect", "Backend"]
        if filters.get('code_level', 5) < 3 and any(k in job_title for k in code_heavy):
            final_score -= 0.50
            reasons.append("⛔ Bloqué par le niveau Code (-50%)")

        # --- RÈGLE 3 : DOMAINE ---
        fav_domain = filters.get('domain', '')
        # (Simplifié pour l'exemple)
        if fav_domain != "Peu importe" and fav_domain[:4] in job_title:
             final_score += 0.10
             reasons.append("🚀 Bonus Domaine (+10%)")
             
        return final_score, reasons

    def find_top_matches(self, user_text, df_jobs, filters=None, top_k=3):
        # 1. Calculer le "Profil de Compétences" (Si pour chaque bloc)
        # C'est l'exigence "Générer un profil pour l'utilisateur"
        block_scores = self.calculate_block_scores(user_text)
        
        hybrid_results = []
        
        for _, row in df_jobs.iterrows():
            job_title = row['titre']
            
            # 2. Récupérer les poids (Wi) pour ce métier
            weights = self.get_job_weights(job_title)
            
            # 3. Appliquer la FORMULE DU STEP 4 : Somme(Wi * Si) / Somme(Wi)
            numerator = sum(weights[block] * block_scores[block] for block in self.competency_blocks)
            denominator = sum(weights.values())
            
            coverage_score = numerator / denominator
            
            # 4. Appliquer vos Règles Métiers (Bonus/Malus)
            final_score, reasons = self.apply_business_rules(job_title, coverage_score, filters or {})
            
            hybrid_results.append({
                "titre": job_title,
                "score": float(final_score),
                "original_score": float(coverage_score), # Le score purement sémantique
                "block_scores": block_scores, # On garde le détail pour les graphiques !
                "description": row['description_semantique'],
                "competences": row['competences_cles'],
                "reasons": reasons
            })
            
        # Tri
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_results[:top_k]