from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Loads the SBERT model and caches it to optimize performance.
    Model: 'paraphrase-multilingual-MiniLM-L12-v2' (Lightweight & Multilingual)
    """
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class SemanticMatcher:
    def __init__(self):
        self.model = load_model()
        
        # 1. Competency Blocks Definition
        # These archetypes represent the main axes of the IT job market.
        self.competency_blocks = {
            "Data Analysis": "Nettoyage de données, visualisation, dashboard, statistiques, SQL, Excel, interprétation business",
            "Machine Learning": "Modèles prédictifs, regression, classification, scikit-learn, algorithmes, entrainement modèle, deep learning",
            "Data Engineering": "Infrastructure, cloud, pipeline, ETL, big data, spark, hadoop, base de données, automatisation, docker, kubernetes",
            "NLP & GenAI": "Texte, langage naturel, LLM, transformers, chatbot, hugging face, embeddings, rag, generative ai",
            "Soft Skills": "Communication, gestion de projet, présentation, éthique, travail d'équipe, curiosité, leadership"
        }
        
        # 2. Pre-calculation of embeddings for blocks
        # Done once at initialization to speed up inference.
        self.block_embeddings = {
            name: self.model.encode([desc]) 
            for name, desc in self.competency_blocks.items()
        }

    def calculate_block_scores(self, user_text):
        """
        Projects the user profile onto the 5 competency axes using Cosine Similarity.
        Returns a dictionary of scores {Block: Score}.
        """
        user_emb = self.model.encode([user_text])
        scores = {}
        for name, block_emb in self.block_embeddings.items():
            sim = cosine_similarity(user_emb, block_emb)[0][0]
            scores[name] = max(0, float(sim))
        return scores

    def get_job_weights(self, job_title):
        """
        Dynamic Weighting Strategy: Assigns weights (Wi) to blocks based on Job Title keywords.
        Example: 'Data Scientist' weights ML higher than Engineering.
        """
        weights = {k: 1.0 for k in self.competency_blocks}
        title = job_title.lower()
        
        if "scientist" in title:
            weights["Machine Learning"] = 3.0
            weights["Data Analysis"] = 2.0
        elif "analyst" in title or "bi " in title or "business" in title:
            weights["Data Analysis"] = 3.0
            weights["Soft Skills"] = 2.0
            weights["Machine Learning"] = 0.5
        elif "engineer" in title or "architect" in title or "devops" in title:
            weights["Data Engineering"] = 3.0
            weights["Machine Learning"] = 1.5
        elif "nlp" in title or "generative" in title or "ai " in title:
            weights["NLP & GenAI"] = 3.5
            weights["Machine Learning"] = 2.0
        elif "manager" in title or "lead" in title or "scrum" in title:
            weights["Soft Skills"] = 3.0
            weights["Data Analysis"] = 1.5
            
        return weights

    def normalize_score_ux(self, raw_score):
        """
        UX Normalization: Transforms raw cosine scores (typically 0.3-0.8) 
        into a user-friendly percentage (0-95%) using a Sigmoid curve.
        """
        # Center x around 0.30 to penalize weak semantic matches heavily
        x = (raw_score - 0.30) * 9 
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def apply_business_rules(self, job_title, base_score, filters):
        """
        Hybrid Filtering (Guardrails): Applies penalties or bonuses based on hard logic rules.
        - Math Level check
        - Code Level check
        - Domain Matching
        """
        final_score = base_score
        reasons = []
        
        title_lower = job_title.lower()
        
        # --- Rule 1: Math Level Guardrail ---
        # Penalize theoretical roles if Math level is low
        math_heavy = [
            "scientist", "research", "algorithm", "vision", "nlp", "ai ", 
            "financial", "quant", "statistics", "actuary"
        ]
        user_math = filters.get('math_level', 3) 
        
        is_math_job = any(k in title_lower for k in math_heavy)
        
        if is_math_job:
            if user_math < 2:
                final_score *= 0.4 # Severe penalty (-60%)
                reasons.append("⛔ Niveau Maths trop faible")
            elif user_math == 2:
                final_score *= 0.7 # Moderate penalty
                reasons.append("⚠️ Niveau Maths juste")

        # --- Rule 2: Code Level Guardrail ---
        # Penalize technical engineering roles if Code level is low
        code_heavy = [
            "engineer", "developer", "architect", "backend", "full stack",
            "admin", "ops", "sre", "cloud", "big data", "programmer", 
            "integrator", "automation", "technical", "specialist"
        ]
        
        # Managers/Leads are exempted from strict coding penalties
        is_manager = "manager" in title_lower or "lead" in title_lower or "scrum" in title_lower
        
        user_code = filters.get('code_level', 3)
        is_code_job = any(k in title_lower for k in code_heavy)
        
        if is_code_job and not is_manager:
            if user_code < 2:
                final_score *= 0.4 
                reasons.append("⛔ Niveau Code trop faible")
            elif user_code == 2:
                final_score *= 0.7
                reasons.append("⚠️ Niveau Code juste")

        # --- Rule 3: Domain Bonus ---
        # Boost score if the job matches the user's target sector
        fav_domain = filters.get('domain', 'Peu importe')
        if fav_domain and fav_domain not in ["Peu importe", "Any"]:
            domain_map = {
                "Data Science": ["scientist", "analyst", "ai", "data"],
                "Dev & Cloud": ["engineer", "developer", "architect", "cloud", "web", "ops"],
                "Robotics": ["vision", "robot", "hardware", "embedded", "iot"],
                "Management": ["manager", "lead", "head", "director", "scrum", "product"]
            }
            keywords = domain_map.get(fav_domain, [])
            if any(k in title_lower for k in keywords):
                final_score *= 1.15 # +15% Boost
                reasons.append(f"🚀 Boost Domaine ({fav_domain})")

        return final_score, reasons

    def find_top_matches(self, user_text, df_jobs, filters=None, top_k=3):
        """
        Main Engine Logic: Combines Semantic Score, Skill Bonus, and Business Rules.
        """
        # 1. Semantic Analysis
        block_scores = self.calculate_block_scores(user_text)
        
        # 2. Skill Keywords Extraction
        user_skills_list = [s.lower() for s in filters.get('skills', [])]
        
        hybrid_results = []
        
        for _, row in df_jobs.iterrows():
            job_title = row['titre']
            
            # A. Weighted Semantic Score
            weights = self.get_job_weights(job_title)
            numerator = sum(weights[block] * block_scores[block] for block in self.competency_blocks)
            denominator = sum(weights.values())
            semantic_score = numerator / denominator
            
            # B. Skill Overlap Bonus
            job_skills = [s.lower() for s in row['skills']] if isinstance(row['skills'], list) else []
            skill_match_count = sum(1 for s in user_skills_list if any(s in js for js in job_skills))
            skill_bonus = min(0.20, skill_match_count * 0.04) # Max +20%
            
            # C. Combine & Normalize
            raw_combined_score = semantic_score + skill_bonus
            ux_score = self.normalize_score_ux(raw_combined_score)
            
            # D. Apply Guardrails
            final_score, reasons = self.apply_business_rules(job_title, ux_score, filters or {})
            
            # E. Score Clamping (Max 98% to avoid unrealistic perfect matches)
            final_score = min(0.98, max(0.01, final_score))
            
            hybrid_results.append({
                "titre": job_title,
                "score": float(final_score),
                "raw_score": float(raw_combined_score),
                "block_scores": block_scores,
                "description": row['description'],
                "skills": row['skills'], 
                "reasons": reasons
            })
            
        # Sort by final score descending
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_results[:top_k]