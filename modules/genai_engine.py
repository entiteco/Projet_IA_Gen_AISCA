import os
import google.generativeai as genai
import streamlit as st

def generate_career_advice(user_profile_text, job_title, job_desc, filters):
    """
    Utilise Gemini Flash pour générer une synthèse personnalisée.
    """
    
    # 1. Récupération de la clé API (depuis le .env injecté dans Docker)
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        return "⚠️ Erreur : Clé API Google introuvable. Vérifiez votre fichier .env."

    # 2. Configuration
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')

        # 3. Construction du Prompt (L'instruction donnée à l'IA)
        # On injecte les données de l'utilisateur pour que la réponse soit unique.
        prompt = f"""
        Tu es un expert en orientation professionnelle spécialisé dans la Data et l'IA.
        Ton rôle est d'expliquer à un candidat pourquoi le métier suivant lui correspond.

        --- DONNÉES DU CANDIDAT ---
        Profil déclaré : "{user_profile_text}"
        Niveau Maths : {filters.get('math_level')}
        Niveau Code : {filters.get('code_level')}/5
        Domaine préféré : {filters.get('domain')}

        --- MÉTIER RECOMMANDÉ PAR L'ALGORITHME ---
        Intitulé : {job_title}
        Description : {job_desc}

        --- TÂCHE ---
        Rédige un paragraphe court (3-4 phrases max) et encourageant.
        1. Explique pourquoi ce métier matche avec ses compétences ou ses goûts.
        2. Si son niveau en maths ou code est faible, rassure-le en disant que ce métier est accessible ou propose une piste d'amélioration.
        3. Adopte un ton professionnel, bienveillant et motivant.
        4. Ne dis pas "Selon l'algorithme", parle directement : "Je vous recommande..."
        """

        # 4. Appel à l'API
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Désolé, l'assistant IA est momentanément indisponible. (Erreur: {str(e)})"