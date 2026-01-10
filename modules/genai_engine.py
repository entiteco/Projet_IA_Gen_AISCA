import os
import google.generativeai as genai
import streamlit as st

def get_api_key():
    """Récupère la clé API de manière sécurisée (Local ou Cloud)."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        return os.environ.get("GOOGLE_API_KEY")

def augment_short_text(short_text):
    """
    EF4.1 : Enrichissement des textes courts (< 5 mots) via GenAI.
    Transforme un mot-clé (ex: "Python") en un paragraphe contextuel riche.
    """
    # Récupération sécurisée de la clé (compatible Local et Cloud)
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key: 
        return short_text # Si pas de clé, on renvoie le texte tel quel

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest') # Modèle rapide pour cette tâche
        
        # PROMPT D'ENRICHISSEMENT
        prompt = f"""
        Tu es un expert en recrutement Tech.
        Un candidat a rempli son profil avec une description trop courte : "{short_text}".
        
        TA MISSION :
        Réécris cette description pour en faire un paragraphe professionnel de 2-3 phrases à la première personne.
        Déduis le contexte probable (Data, IA, Dev) et ajoute des mots-clés techniques pertinents
        pour améliorer le matching sémantique. 
        
        Exemple si entrée="Python" -> Sortie="Je suis un développeur passionné par Python, avec une expérience en automatisation de scripts et en analyse de données via Pandas."
        
        Reste fidèle à l'intention d'origine mais étoffe-la.
        Ne mets pas de guillemets, juste le texte.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        # En cas d'erreur (quota, internet...), on ne bloque pas l'app, on renvoie l'original
        return short_text

def generate_career_advice(user_profile_text, job_title, job_desc, filters):
    """
    EF4.2 : Génère le plan de progression personnalisé.
    """
    api_key = get_api_key()
    if not api_key: return "⚠️ Erreur : Clé API introuvable."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest') 
        
        prompt = f"""
        Agis comme un coach de carrière expert en Data & IA.
        
        PROFIL CANDIDAT :
        "{user_profile_text}"
        (Niveau Maths: {filters.get('math_level')}, Code: {filters.get('code_level')}/5)

        MÉTIER VISÉ : {job_title}
        
        TA MISSION :
        1. Synthèse : Dis pourquoi ce métier correspond au profil (1 phrase).
        2. Gap Analysis : Identifie 2 ou 3 compétences clés manquantes ou à renforcer pour ce poste spécifique.
        3. Plan d'action : Propose un plan d'apprentissage concret.
        
        Format : Court, direct, structuré avec des bullet points.
        """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Erreur IA : {str(e)}"