import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import Counter

# Imports des modules personnalisés
from modules.data_loader import load_data
from modules.nlp_engine import SemanticMatcher
from modules.genai_engine import generate_career_advice, augment_short_text, translate_to_french

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AISCA", # Titre onglet simplifié
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. TABLE DE TRADUCTION DES SKILLS ---
SKILL_MAP = {
    # Hard Skills (Traduisibles)
    "Machine Learning": "Machine Learning",
    "Deep Learning": "Deep Learning",
    "Data Analysis": "Analyse de données",
    "Data Visualization": "Visualisation de données",
    "Software Development": "Développement logiciel",
    "Cloud Computing": "Cloud Computing",
    "Artificial Intelligence": "Intelligence Artificielle",
    "Computer Vision": "Vision par ordinateur",
    "Natural Language Processing": "NLP (Traitement du langage)",
    "Statistics": "Statistiques",
    "Mathematics": "Mathématiques",
    "Database Management": "Gestion BDD",
    "Software Engineering": "Génie logiciel",
    "Agile Methodologies": "Méthodes Agiles",
    "Algorithm Design": "Algorithmique",
    "System Architecture": "Architecture système",
    "Cybersecurity": "Cybersécurité",
    "Web Development": "Dév. Web",
    "Mobile Development": "Dév. Mobile",
    "Robotics": "Robotique",
    "Version Control": "Gestion de versions (Git)",
    "Big Data": "Big Data",
    "Project Management": "Gestion de projet",
    "Hadoop": "Hadoop",
    "Spark": "Spark",
    "NoSQL": "NoSQL",
    "Data Warehousing": "Entrepôt de données"
}

def translate_list(skills_list, lang):
    """Traduit une liste de skills si la langue est FR."""
    if not skills_list: return []
    if lang == 'en': return skills_list
    
    translated = []
    for s in skills_list:
        key = s.title().strip()
        trans = SKILL_MAP.get(key, s)
        translated.append(trans)
    return translated

# --- 2. CHARGEMENT DONNÉES ---
df = load_data()

top_skills = []

if not df.empty:
    all_skills_raw = [item for sublist in df['skills'] for item in sublist]
    all_skills_normalized = [s.title() for s in all_skills_raw if len(s) > 1]
    count_skills = Counter(all_skills_normalized)
    top_skills = [skill for skill, count in count_skills.most_common(150)]
    top_skills.sort()
else:
    st.error("⚠️ Erreur : Le fichier 'filtered_it_jobs.json' est introuvable ou vide.")
    st.stop()

# --- CSS ---
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] { padding-top: 10px; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stExpander"] div[role="button"] p { font-weight: bold; }
    h3 { color: #2E86C1; }
    </style>
""", unsafe_allow_html=True)

# --- LANGUE ---
if 'language' not in st.session_state:
    st.session_state.language = 'fr'

# --- TEXTES LATÉRAUX ENRICHIS ---
TRANS = {
    'fr': {
        'sidebar_title': "🤖 AISCA",
        'sidebar_info': """
        ### 📌 À propos du Projet
        **AISCA** (Artificial Intelligence Skills & Career Agent) est un outil d'orientation intelligent conçu pour mapper vos compétences réelles aux opportunités du marché Data & IA.

        ### ⚙️ Architecture Technique
        
        **1. Ingestion & Nettoyage (Data Engineering)**
        * Chargement d'un Dataset JSON brut.
        * Nettoyage des doublons et formatage des compétences via Python/Pandas.
        
        **2. Moteur Sémantique (NLP - SBERT)**
        * Utilisation du modèle `paraphrase-multilingual-MiniLM-L12-v2`.
        * Vectorisation de votre profil et des offres d'emploi dans un espace vectoriel à 384 dimensions.
        * Calcul de similarité cosinus pour détecter le **sens** au-delà des mots-clés.

        **3. Filtrage Hybride (Règles Métiers)**
        * Application de "Garde-fous" logiques.
        * Vérification des pré-requis (Niveau Maths / Programmation) pour éviter les recommandations irréalistes.

        **4. Intelligence Artificielle (GenAI)**
        * Connexion à l'API **Google Gemini**.
        * **Augmentation :** Enrichissement automatique des descriptions courtes.
        * **Coaching :** Analyse des écarts (Gap Analysis) et génération de plans d'apprentissage.
        * **Traduction :** Traduction à la volée des offres anglophones.
        """,
        'main_title': "Cartographie des Compétences & Orientation",
        'subtitle': "Trouvez le métier Data/IA aligné avec votre profil réel.",
        'p1_title': "1️⃣ Compétences Techniques",
        'domain_q': "Quel domaine vous attire le plus ?",
        'skills_q': "Quels outils maîtrisez-vous ?",
        'select_ph': "Choisissez une ou plusieurs options...",
        'p2_title': "2️⃣ Niveaux & Auto-évaluation",
        'code_lvl': "Niveau en Programmation (Python/SQL)",
        'scale_txt': "Échelle 0-5",
        'math_lvl': "Aisance Mathématiques & Stats",
        'level_txt': "Niveau perçu",
        'p3_title': "3️⃣ Votre Histoire",
        'story_desc': "L'IA analyse le **sens** de vos phrases.",
        'story_ph': "Ex: J'ai travaillé sur un projet de détection de fraude...",
        'submit_btn': "🚀 Lancer l'Analyse Complète",
        'short_text_warn': "✨ Texte court détecté : L'IA enrichit votre profil...",
        'ia_coach_title': "🤖 L'Avis de l'Assistant IA (Gemini)",
        'ia_loading': "🧠 Gemini analyse votre profil et rédige le coaching...",
        'viz_title': "📊 Cartographie Visuelle",
        'results_title': "🏆 Top 3 des Métiers Recommandés",
        'loading': "🧠 Analyse Hybride en cours...",
        'radar_title': "Radar des Compétences",
        'heat_title': "Carte de Chaleur Sémantique",
        'bar_title': "Comparatif de Pertinence (Top 3)",
        'it_view': "🛠️ Compétences Requises",
        'domains': ["Peu importe", "Développement & Code", "Analyse & Business", "Infrastructure & Cloud", "Mathématiques & Recherche", "Éthique & Gouvernance"],
        'levels': ["Débutant", "Notions", "Intermédiaire", "Avancé", "Expert"],
        'axes': ["Développement", "Mathématiques", "Business", "Cloud/Ops", "IA/Modeling"]
    },
    'en': {
        'sidebar_title': "🤖 AISCA",
        'sidebar_info': """
        ### 📌 About the Project
        **AISCA** is an intelligent career agent designed to map your real skills to Data & AI market opportunities.

        ### ⚙️ Technical Architecture
        
        **1. Ingestion & Cleaning (Data Engineering)**
        * Loading raw JSON datasets.
        * Cleaning duplicates and formatting skills via Python/Pandas.
        
        **2. Semantic Engine (NLP - SBERT)**
        * Using `paraphrase-multilingual-MiniLM-L12-v2`.
        * Vectorizing profiles and jobs into a 384-dimensional vector space.
        * Cosine similarity calculation to match **meaning** beyond keywords.

        **3. Hybrid Filtering (Business Rules)**
        * Application of logical "Guardrails".
        * Validation of prerequisites (Math / Coding levels) to prevent unrealistic recommendations.

        **4. Generative AI (GenAI)**
        * Powered by **Google Gemini API**.
        * **Augmentation:** Automatic enrichment of short user inputs.
        * **Coaching:** Gap Analysis and generation of tailored learning paths.
        * **Translation:** On-the-fly translation of English job descriptions.
        """,
        'main_title': "Skills Mapping & Career Orientation",
        'subtitle': "Find the Data/AI job truly aligned with your profile.",
        'p1_title': "1️⃣ Technical Skills",
        'domain_q': "Which domain attracts you the most?",
        'skills_q': "Which tools do you master?",
        'select_ph': "Choose one or multiple options...",
        'p2_title': "2️⃣ Self-Assessment",
        'code_lvl': "Programming Level (Python/SQL)",
        'scale_txt': "Scale 0-5",
        'math_lvl': "Mathematics & Stats Proficiency",
        'level_txt': "Perceived Level",
        'p3_title': "3️⃣ Your Story",
        'story_desc': "AI analyzes the **semantic meaning** of your sentences.",
        'story_ph': "Ex: I worked on a bank fraud detection project...",
        'submit_btn': "🚀 Start Full Analysis",
        'short_text_warn': "✨ Short text detected: AI is augmenting your profile...",
        'ia_coach_title': "🤖 AI Coach Advice (Gemini)",
        'ia_loading': "🧠 Gemini is analyzing your profile and writing advice...",
        'viz_title': "📊 Visual Mapping",
        'results_title': "🏆 Top 3 Recommended Jobs",
        'loading': "🧠 Hybrid Analysis in progress...",
        'radar_title': "Skills Radar",
        'heat_title': "Semantic Heatmap",
        'bar_title': "Relevance Comparison (Top 3)",
        'it_view': "🛠️ Required Skills",
        'domains': ["Any", "Development & Code", "Analysis & Business", "Infrastructure & Cloud", "Maths & Research", "Ethics & Governance"],
        'levels': ["Beginner", "Basic", "Intermediate", "Advanced", "Expert"],
        'axes': ["Development", "Mathematics", "Business", "Cloud/Ops", "AI/Modeling"]
    }
}

def t(key): return TRANS[st.session_state.language][key]

# --- FONCTIONS UTILITAIRES ---
def save_to_history(user_text, filters, top_result, augmented=False):
    log_dir = "logs"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file = os.path.join(log_dir, "history.csv")
    
    new_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "language": st.session_state.language,
        "user_profile_raw": user_text,
        "is_augmented": augmented,
        "domain_pref": filters['domain'],
        "skills": ", ".join(filters.get('skills', [])),
        "recommended_job": top_result['titre'],
        "match_score": round(top_result['score'], 4)
    }
    df = pd.DataFrame([new_data])
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False, sep=';', encoding='utf-8')
    else:
        df.to_csv(log_file, mode='a', header=False, index=False, sep=';', encoding='utf-8')

def get_axes_data(user_text, model):
    axes_labels = t('axes')
    axes_embeddings = model.encode(axes_labels)
    user_embedding = model.encode([user_text])
    scores = cosine_similarity(user_embedding, axes_embeddings)[0]
    return axes_labels, scores

def plot_radar_chart(axes, scores):
    r_values = list(scores) + [scores[0]]
    theta_values = axes + [axes[0]]
    fig = go.Figure(data=go.Scatterpolar(r=r_values, theta=theta_values, fill='toself', name='Profile', line_color='#2E86C1'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, title=t('radar_title'), height=350, margin=dict(t=30, b=20, l=40, r=40))
    return fig

def plot_heatmap(axes, scores):
    z = [scores]
    fig = go.Figure(data=go.Heatmap(z=z, x=axes, y=["Intensity"], colorscale='Blues', zmin=0, zmax=1, showscale=False))
    fig.update_layout(title=t('heat_title'), height=200, margin=dict(t=30, b=10, l=10, r=10), yaxis=dict(showticklabels=False))
    return fig

def plot_bar_chart(results):
    titles = [res['titre'] for res in results][::-1]
    scores = [res['score'] for res in results][::-1]
    colors = ['#2ECC71' if s > 0.6 else '#F1C40F' for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=titles, orientation='h', marker_color=colors, text=[f"{s*100:.0f}%" for s in scores], textposition='auto'))
    fig.update_layout(title=t('bar_title'), xaxis_title="Score (0-1)", margin=dict(t=30, b=20, l=10, r=10), height=250)
    return fig

# --- SIDEBAR & LANGUE ---
with st.sidebar:
    lang_choice = st.radio("Langue / Language", ["Français", "English"], horizontal=True)
    st.session_state.language = 'en' if lang_choice == "English" else 'fr'
    st.divider()
    st.title(t('sidebar_title'))
    st.markdown(t('sidebar_info'))
    st.write("---")
    st.caption("Projet IA Gen - EFREI")

# --- TITRE PRINCIPAL (SIMPLIFIÉ) ---
st.title(t('main_title'))
st.markdown(f"##### {t('subtitle')}")

# --- FORMULAIRE ---
with st.form("profiling_form"):
    
    # Listes traduites pour l'affichage (si FR)
    display_skills = translate_list(top_skills, st.session_state.language)

    # 1. COMPÉTENCES
    with st.container(border=True):
        st.subheader(t('p1_title'))
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            domaine_prefere = st.selectbox(t('domain_q'), t('domains'))
        with c2:
            skills_user = st.multiselect(t('skills_q'), display_skills, placeholder=t('select_ph'))

    st.write("") 

    # 2. NIVEAUX
    with st.container(border=True):
        st.subheader(t('p2_title'))
        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            st.markdown(f"**{t('code_lvl')}**")
            niveau_python = st.slider(t('scale_txt'), 0, 5, 2, key="slider_code")
        with c4:
            st.markdown(f"**{t('math_lvl')}**")
            niveau_maths_ui = st.select_slider(t('level_txt'), options=t('levels'), value=t('levels')[2])

    st.write("") 

    # 3. HISTOIRE
    with st.container(border=True):
        st.subheader(t('p3_title'))
        st.markdown(t('story_desc'))
        user_text_input = st.text_area("Description", height=150, placeholder=t('story_ph'), label_visibility="collapsed")

    st.write("")
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        submitted = st.form_submit_button(t('submit_btn'), use_container_width=True)

# --- TRAITEMENT ---
if submitted:
    full_text_history = user_text_input
    is_augmented = False
    
    # Mapping niveau maths
    levels_en = TRANS['en']['levels']
    levels_fr = TRANS['fr']['levels']
    if st.session_state.language == 'en':
        try:
            idx = levels_en.index(niveau_maths_ui)
            niveau_maths_engine = levels_fr[idx]
        except: niveau_maths_engine = "Intermédiaire"
    else:
        niveau_maths_engine = niveau_maths_ui

    # Augmentation IA
    if len(user_text_input.split()) < 5:
        with st.spinner(t('short_text_warn')):
            augmented_text = augment_short_text(user_text_input, language=st.session_state.language)
            with st.expander("AI Augmented Text", expanded=True):
                st.write(f"**Original:** {user_text_input}")
                st.write(f"**Augmented:** {augmented_text}")
            full_text_history = augmented_text
            is_augmented = True

    # Contexte
    contexte_skills = f"Skills: {', '.join(skills_user)}." if skills_user else ""
    contexte_pref = f"Domain: {domaine_prefere}." 
    contexte_niveau = f"Code Level {niveau_python}/5. Math Level: {niveau_maths_ui}."
    
    full_profile_text = f"{full_text_history} {contexte_skills} {contexte_pref} {contexte_niveau}"
    
    user_filters = {
        "math_level": niveau_maths_engine,
        "code_level": niveau_python, 
        "domain": domaine_prefere,
        "skills": skills_user
    }
    
    matcher = SemanticMatcher()
    with st.spinner(t('loading')):
        results = matcher.find_top_matches(full_profile_text, df, filters=user_filters, top_k=3)
    
    if results:
        save_to_history(user_text=user_text_input, filters=user_filters, top_result=results[0], augmented=is_augmented)

    # --- RESULTATS ---
    st.divider()
    st.markdown(f"### {t('ia_coach_title')}")
    if results:
        top_job = results[0]
        with st.chat_message("assistant"):
            # PHRASE DE CHARGEMENT
            with st.spinner(t('ia_loading')):
                advice = generate_career_advice(full_profile_text, top_job['titre'], top_job['description'], filters=user_filters, language=st.session_state.language)
            st.markdown(advice)

    st.divider()
    st.markdown(f"### {t('viz_title')}")
    axes_labels, axes_scores = get_axes_data(full_profile_text, matcher.model)
    c_viz1, c_viz2 = st.columns(2)
    with c_viz1: st.plotly_chart(plot_radar_chart(axes_labels, axes_scores), use_container_width=True)
    with c_viz2: 
        st.plotly_chart(plot_heatmap(axes_labels, axes_scores), use_container_width=True)
        st.plotly_chart(plot_bar_chart(results), use_container_width=True)

    st.divider()
    st.markdown(f"### {t('results_title')}")
    cols = st.columns(3, gap="medium")
    for i, res in enumerate(results):
        with cols[i]:
            with st.container(border=True):
                # TRADUCTION ET AFFICHAGE TITRE
                display_title = res['titre']
                
                # Traduction si FR
                if st.session_state.language == 'fr':
                    display_title = translate_to_french(res['titre'])
                    # On traduit TOUTE la description pour l'afficher direct
                    display_desc = translate_to_french(res['description'])
                else:
                    display_desc = res['description']
                
                st.markdown(f"#### #{i+1} {display_title}")
                
                score_val = res['score']
                color = "green" if score_val > 0.6 else "orange" if score_val > 0.4 else "red"
                st.markdown(f":{color}[**Score : {score_val*100:.1f}%**]")
                st.progress(min(max(score_val, 0.0), 1.0))
                
                if res['reasons']:
                    for r in res['reasons']: st.markdown(f":red[{r}]")
                
                # Affichage complet direct
                st.caption(f"_{display_desc}_") 
                
                with st.expander(t('it_view')):
                    skills_disp = translate_list(res.get('skills', []), st.session_state.language)
                    st.write(", ".join(skills_disp))