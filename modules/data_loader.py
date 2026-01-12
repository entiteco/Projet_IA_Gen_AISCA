import pandas as pd
import os
import re
import streamlit as st

def clean_skill_string(skill):
    """Nettoie les noms de compétences."""
    if not isinstance(skill, str): return ""
    s = skill.replace("#", "").replace("*", "")
    s = re.sub(r'^\d+[\.\)]\s*', '', s)
    s = re.sub(r'^[^a-zA-Z0-9\.\+]+', '', s)
    s = re.sub(r'[^a-zA-Z0-9\.\+]+$', '', s)
    return s.strip()

def load_data(filepath="data/filtered_it_jobs.json"):
    """
    Charge les données. Structure : job_title, job_description, skills.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        df = pd.read_json(filepath)
        
        # Adaptation aux colonnes du JSON fourni
        rename_map = {
            'job_title': 'titre',
            'job_description': 'description'
        }
        df = df.rename(columns=rename_map)
        
        if 'skills' not in df.columns:
            return pd.DataFrame()

        # Nettoyage
        def process_skills_list(skills_list):
            if not isinstance(skills_list, list): return []
            cleaned_list = []
            for s in skills_list:
                cleaned = clean_skill_string(s)
                if len(cleaned) > 0:
                    cleaned_list.append(cleaned)
            return cleaned_list

        df['skills'] = df['skills'].apply(process_skills_list)
        
        # Champ sémantique
        df['description_semantique'] = df.apply(
            lambda row: f"{row['titre']} . {row['description']} . Skills: {' '.join(row['skills'])}",
            axis=1
        )
        
        return df

    except Exception as e:
        print(f"Erreur data loader: {e}")
        return pd.DataFrame()