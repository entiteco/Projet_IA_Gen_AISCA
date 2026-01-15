import pandas as pd
import os
import re
import streamlit as st

def clean_skill_string(skill):
    """Cleans and standardizes individual skill names using Regex."""
    if not isinstance(skill, str): return ""
    
    # Remove common formatting artifacts
    s = skill.replace("#", "").replace("*", "")
    
    # Remove leading numbering (e.g., "1. Python" -> "Python")
    s = re.sub(r'^\d+[\.\)]\s*', '', s)
    
    # Remove non-alphanumeric chars from edges, preserving special chars like + (C++) or . (.NET)
    s = re.sub(r'^[^a-zA-Z0-9\.\+]+', '', s)
    s = re.sub(r'[^a-zA-Z0-9\.\+]+$', '', s)
    
    return s.strip()

def load_data(filepath="data/filtered_it_jobs.json"):
    """
    Loads, transforms, and prepares the dataset.
    Pipeline: JSON Ingestion -> Schema Mapping -> Skill Cleaning -> Semantic Field Creation.
    """
    # Robustness check: Ensure file exists before reading
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        # 1. Extraction
        df = pd.read_json(filepath)
        
        # 2. Transformation: Map raw JSON keys to internal schema
        rename_map = {
            'job_title': 'titre',
            'job_description': 'description'
        }
        df = df.rename(columns=rename_map)
        
        # Validation
        if 'skills' not in df.columns:
            return pd.DataFrame()

        # 3. Cleaning: Process the list of skills for each row
        def process_skills_list(skills_list):
            if not isinstance(skills_list, list): return []
            cleaned_list = []
            for s in skills_list:
                cleaned = clean_skill_string(s)
                if len(cleaned) > 0:
                    cleaned_list.append(cleaned)
            return cleaned_list

        df['skills'] = df['skills'].apply(process_skills_list)
        
        # 4. Feature Engineering: Create 'Semantic Description'
        # Concatenates Title, Description, and Skills for rich vectorization by SBERT
        df['description_semantique'] = df.apply(
            lambda row: f"{row['titre']} . {row['description']} . Skills: {' '.join(row['skills'])}",
            axis=1
        )
        
        return df

    except Exception as e:
        # Error handling for debugging
        print(f"Erreur data loader: {e}")
        return pd.DataFrame()