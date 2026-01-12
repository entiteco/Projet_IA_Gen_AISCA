import os
import google.generativeai as genai
import streamlit as st

def get_api_key():
    """Récupère la clé API."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except:
        return os.environ.get("GOOGLE_API_KEY")

def augment_short_text(short_text, language="fr"):
    """Enrichissement des textes courts."""
    api_key = get_api_key()
    if not api_key: return short_text 

    try:
        genai.configure(api_key=api_key)
        # MODIFICATION : Utilisation du modèle disponible dans votre liste
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        if language == "en":
            prompt = f"""
            Rewrite this short profile description into a professional paragraph (2-3 sentences).
            Input: "{short_text}"
            Output (English only, no quotes):
            """
        else:
            prompt = f"""
            Réécris cette courte description de profil en un paragraphe professionnel (2-3 phrases).
            Entrée : "{short_text}"
            Sortie (Français uniquement, sans guillemets) :
            """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        # En cas d'erreur, on renvoie le texte brut pour ne pas bloquer l'app
        return short_text

def translate_to_french(text):
    """Traduit le titre ou la description en Français."""
    if not text: return ""
    api_key = get_api_key()
    if not api_key: return text

    try:
        genai.configure(api_key=api_key)
        # MODIFICATION : Utilisation du modèle disponible
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""
        Translate the following job text to French. 
        Keep technical terms in English (e.g. "Data Scientist", "Framework").
        Do not add introductory text.
        Text: "{text}"
        Translation:
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return text

def generate_career_advice(user_profile_text, job_title, job_desc, filters, language="fr"):
    """Génère le coaching."""
    api_key = get_api_key()
    if not api_key: return "⚠️ Erreur Clé API."

    try:
        genai.configure(api_key=api_key)
        # MODIFICATION : Utilisation du modèle disponible
        model = genai.GenerativeModel('gemini-2.5-flash-lite') 
        
        if language == "en":
            prompt = f"""
            Act as an expert Data & AI Career Coach.
            Candidate Profile: "{user_profile_text}"
            Target Job: {job_title}
            Mission: Explain why it matches, identify gaps, propose a learning path.
            Format: Markdown, bullet points, concise. English.
            """
        else:
            prompt = f"""
            Agis comme un coach de carrière expert en Data & IA.
            Profil Candidat : "{user_profile_text}"
            Métier Visé : {job_title}
            Mission : Explique la correspondance, identifie les manques (Gap Analysis), propose un plan d'action.
            Format : Markdown, bullet points, concis. Français.
            """

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"⚠️ Erreur IA (Quota ou Modèle) : {str(e)}"