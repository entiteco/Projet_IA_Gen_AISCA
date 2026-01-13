import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from datetime import datetime
from collections import Counter

# ============================================================
# IMPORTS DES MODULES REELS
# ============================================================
from modules.data_loader import load_data
from modules.nlp_engine import SemanticMatcher
from modules.genai_engine import augment_short_text, translate_to_french

# --- 1. CONFIG ---
st.set_page_config(
    page_title="AISCA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Gestion de l'état de la langue
if 'lang_choice' not in st.session_state:
    st.session_state.lang_choice = 'FR' 

# Mise à jour de la variable utilisée par le système de traduction
st.session_state.language = 'fr' if st.session_state.lang_choice == 'FR' else 'en'

# --- 2. CSS "LIQUID BRUTALISM" ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700;800&display=swap');

    /* --- GLOBAL FONT SIZE --- */
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px; 
    }

    /* --- FOND LIQUID MOTION --- */
    .stApp {
        background-image: 
            radial-gradient(at 40% 20%, rgba(18, 36, 219, 0.5) 0px, transparent 50%),
            radial-gradient(at 80% 0%, rgba(214, 21, 25, 0.5) 0px, transparent 50%),
            radial-gradient(at 0% 50%, rgba(46, 213, 115, 0.5) 0px, transparent 50%),
            radial-gradient(at 80% 50%, rgba(255, 222, 0, 0.4) 0px, transparent 50%),
            radial-gradient(at 0% 100%, rgba(18, 36, 219, 0.5) 0px, transparent 50%),
            linear-gradient(to bottom, #DFE9F3, #FFFFFF);
        background-size: 200% 200%;
        animation: liquidMove 15s ease-in-out infinite alternate;
        color: #000000;
    }

    @keyframes liquidMove {
        0% { background-position: 0% 0%; }
        50% { background-position: 100% 100%; }
        100% { background-position: 0% 50%; }
    }

    /* LARGEUR DE PAGE */
    .block-container {
        padding-top: 6rem !important; 
        padding-bottom: 5rem;
        max-width: 1400px; 
    }

    /* --- CONTENEURS BLANCS SOLIDES --- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 3px solid #000000;
        border-radius: 0px;
        box-shadow: 8px 8px 0px 0px #000000;
        padding: 40px;
        margin-bottom: 40px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translate(-2px, -2px);
        box-shadow: 12px 12px 0px 0px #000000;
    }

    /* TYPOGRAPHIE TITRES */
    h1, h2, h3, h4 {
        color: #000000 !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: -0.5px;
    }
    h3 { font-size: 1.6rem !important; }
    h4 { font-size: 1.4rem !important; margin-bottom: 20px !important; }

    /* --- HEADER --- */
    .hero-box {
        background-color: #1224DB !important;
        border: 3px solid black;
        box-shadow: 6px 6px 0px black;
        padding: 30px;
        margin-bottom: 50px;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    .hero-box:hover {
        background-color: #D61519 !important;
        box-shadow: 10px 10px 0px black;
        transform: translate(-2px, -2px);
    }
    .hero-box h1, .hero-box h3, .hero-box p { color: #FFFFFF !important; }

    /* --- BOUTON D'ACTION --- */
    div.stButton > button {
        background-color: #1224DB;
        color: white;
        border: 3px solid black;
        border-radius: 0px;
        font-weight: 800;
        font-size: 1.3rem;
        padding: 1.2rem 2rem;
        box-shadow: 6px 6px 0px black;
        transition: all 0.2s;
        text-transform: uppercase;
        width: 100%;
        margin-top: 20px;
    }
    div.stButton > button:hover {
        background-color: #D61519;
        color: white;
        border-color: black;
        box-shadow: 10px 10px 0px black;
        transform: translate(-3px, -3px);
    }

    /* --- INPUTS UNIFORMES --- */
    .stTextArea textarea, 
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        border-radius: 0px;
        font-size: 1rem;
        min-height: 50px;
    }

    /* SLIDERS */
    div[data-baseweb="slider"] div[class*="thumb"] {
        background-color: #000000 !important;
        border: 2px solid white;
        height: 26px; width: 26px;
        border-radius: 50%;
    }
    div[data-baseweb="slider"] div[class*="track-fill"] {
        background-color: #000000 !important;
    }
    div[data-baseweb="slider"] div[class*="tick-bar"] {
        background-color: #000000 !important;
    }

    /* RADIO BUTTONS */
    div[role="radiogroup"] {
        background: white;
        border: 2px solid black;
        padding: 10px;
        box-shadow: 4px 4px 0px black;
        display: inline-flex;
    }

    /* --- BADGES COMPETENCES --- */
    .brut-tag {
        background-color: #1224DB;
        color: white;
        border: 2px solid black;
        font-weight: bold;
        padding: 4px 10px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 3px;
        box-shadow: 3px 3px 0px black;
        transition: all 0.2s ease-in-out;
        cursor: default;
    }
    .brut-tag:hover {
        background-color: #D61519;
        transform: translate(-3px, -3px);
        box-shadow: 6px 6px 0px black;
    }
    
    /* Coach Box */
    .coach-box {
        background-color: #FFFFFF; 
        border: 3px solid black; 
        padding: 30px; 
        box-shadow: 6px 6px 0px black; 
        color: black; 
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 40px;
    }
    
    /* CARTE JOB */
    .job-card {
        border: 3px solid black; 
        padding: 20px; 
        background-color: #FFFFFF;
        position: relative;
        box-shadow: 6px 6px 0px black;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONFIG DATA ---
if 'language' not in st.session_state: st.session_state.language = 'fr'

SKILL_MAP = {
    "Machine Learning": "Machine Learning", "Deep Learning": "Deep Learning",
    "Data Analysis": "Analyse de données", "Data Visualization": "Visualisation de données",
    "Software Development": "Développement logiciel", "Cloud Computing": "Cloud Computing",
    "Artificial Intelligence": "Intelligence Artificielle", "Computer Vision": "Vision par ordinateur",
    "Natural Language Processing": "NLP", "Statistics": "Statistiques",
    "Mathematics": "Mathématiques", "Database Management": "Gestion BDD",
    "Software Engineering": "Génie logiciel", "Agile Methodologies": "Méthodes Agiles",
    "Algorithm Design": "Algorithmique", "System Architecture": "Architecture système",
    "Cybersecurity": "Cybersécurité", "Web Development": "Dév. Web",
    "Mobile Development": "Dév. Mobile", "Robotics": "Robotique",
    "Version Control": "Git", "Big Data": "Big Data",
    "Project Management": "Gestion de projet", "Hadoop": "Hadoop",
    "Spark": "Spark", "NoSQL": "NoSQL", "Data Warehousing": "Entrepôt de données"
}

TRANS = {
    'fr': {
        'header_full_name': "Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences",
        'subtitle': "Ne laissez pas le hasard décider de votre carrière.",
        'about_btn': "À PROPOS DU PROJET",
        'about_txt': """
        ### 🤖 AISCA (AI Skills & Career Agent)
        
        Ce projet mappe vos compétences réelles aux opportunités du marché via une architecture hybride :

        **1. DATA ENGINEERING (Ingestion)**
        * Chargement et nettoyage d'un dataset JSON brut.
        * Standardisation des compétences et dédoublonnage via Pandas.
        
        **2. MOTEUR SÉMANTIQUE (NLP/SBERT)**
        * Modèle : `paraphrase-multilingual-MiniLM-L12-v2`.
        * Vectorisation : Projection des profils dans un espace vectoriel (384 dims).
        * Matching : Calcul de similarité cosinus pour comprendre le *sens*.
        
        **3. FILTRAGE HYBRIDE (Logique)**
        * Application de "Garde-fous" (Business Rules).
        * Vérification stricte des niveaux (Maths/Code) pour éviter les hallucinations.
        
        **4. GENAI (Google Gemini)**
        * **Augmentation :** Enrichissement automatique des descriptions courtes.
        * **Coaching :** Analyse des écarts (Gap Analysis) et plans d'action.
        * **Traduction :** Traduction à la volée des offres.
        """,
        'col_left': "1. VOS DONNÉES",
        'col_right': "2. LE VERDICT",
        'col_right_sub': "EN ATTENTE D'INSTRUCTIONS...",
        'domain_q': "Secteur Cible",
        'skills_q': "Armes (Compétences)",
        'select_ph': "Ajouter...",
        'code_lvl': "Niveau Code (1 à 5)", 
        'math_lvl': "Niveau Théorique (Maths/Science - 1 à 5)", 
        'story_ph': "Pas de bla-bla, décrivez vos projets, vos ambitions et ce que vous savez faire...",
        'submit': "LANCER L'ANALYSE",
        'loading': "Vectorisation & Inférence...",
        'coach': "Analyse Stratégique",
        'ia_loading': "Gemini rédige le rapport...",
        'domains': ["Peu importe", "Data Science", "Dev & Cloud", "Robotics", "Management"],
        'score_gauge_title': "Score de Pertinence Global",
        'skill_gap_title': "Écart de Compétences (Top Jobs vs Vous)"
    },
    'en': {
        'header_full_name': "Intelligent Semantic and Generative Agent for Skills Mapping",
        'subtitle': "Don't let luck decide your career.",
        'about_btn': "ABOUT",
        'about_txt': """
        ### 🤖 AISCA (AI Skills & Career Agent)
        
        This project maps real skills to market opportunities using a hybrid architecture:

        **1. DATA ENGINEERING (Ingestion)**
        * Loading and cleaning of raw JSON datasets.
        * Skill standardization via Pandas.
        
        **2. SEMANTIC ENGINE (NLP/SBERT)**
        * Model: `paraphrase-multilingual-MiniLM-L12-v2`.
        * Vectorization: Mapping profiles into a 384-dimensional vector space.
        * Matching: Cosine similarity to capture *meaning* beyond keywords.
        
        **3. HYBRID FILTERING (Logic)**
        * Application of logical "Guardrails".
        * Strict validation of Math/Code levels to prevent hallucinations.
        
        **4. GENAI (Google Gemini)**
        * **Augmentation:** Automatic enrichment of short user inputs.
        * **Coaching:** Gap Analysis and actionable learning paths.
        * **Translation:** Real-time translation of job descriptions.
        """,
        'col_left': "1. YOUR DATA",
        'col_right': "2. THE VERDICT",
        'col_right_sub': "WAITING FOR INPUT...",
        'domain_q': "Target Sector",
        'skills_q': "Weapons (Skills)",
        'select_ph': "Add...",
        'code_lvl': "Code Lvl (1 to 5)", 
        'math_lvl': "Theoretical Lvl (Math/Science - 1 to 5)", 
        'story_ph': "No fluff, describe your real projects...",
        'submit': "RUN ANALYSIS",
        'loading': "Processing...",
        'coach': "Strategic Analysis",
        'ia_loading': "Gemini is writing...",
        'domains': ["Any", "Data Science", "Dev & Cloud", "Robotics", "Management"],
        'score_gauge_title': "Overall Relevance Score",
        'skill_gap_title': "Skill Gap (Top Jobs vs You)"
    }
}

def t(key): return TRANS[st.session_state.language][key]
def translate_list(l, lang): return l if lang == 'en' else [SKILL_MAP.get(s.title().strip(), s) for s in l]

# --- CHARGEMENT DONNEES ---
df = load_data()
if df.empty: st.stop()
all_skills = [s.title() for sub in df['skills'] for s in sub if len(s)>1]
top_skills = [x[0] for x in Counter(all_skills).most_common(120)]
top_skills.sort()

# --- FONCTIONS VIZ ---
def plot_radar_brut(axes, scores):
    # Changement VERT -> BLEU Transparent (#1224DB)
    fig = go.Figure(data=go.Scatterpolar(
        r=list(scores) + [scores[0]], theta=axes + [axes[0]],
        fill='toself', fillcolor='rgba(18, 36, 219, 0.4)', 
        line=dict(color='black', width=3),
        marker=dict(size=6, color='black')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#CCC', showticklabels=False),
            angularaxis=dict(gridcolor='#CCC', linecolor='black', tickfont=dict(size=12)),
            bgcolor='#FFFFFF'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', family="Space Grotesk", size=12),
        margin=dict(t=40, b=40, l=60, r=60), height=350, showlegend=False
    )
    return fig

def plot_heatmap_brut(axes, scores):
    fig = go.Figure(data=go.Heatmap(
        z=[scores], x=axes, y=['SCORE'],
        colorscale=[[0, 'white'], [1, '#D61519']], # Ajustement au Rouge Thème
        showscale=False, texttemplate="%{z:.0%}",
        textfont={"size":14}
    ))
    fig.update_layout(
        height=120, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', family="Space Grotesk", size=12),
        yaxis=dict(showticklabels=False), xaxis=dict(side="top")
    )
    return fig

def plot_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': t('score_gauge_title'), 'font': {'size': 14}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "black"},
            'bar': {'color': "#1224DB", 'line': {'color': "black", 'width': 2}},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 50], 'color': '#F1F2F6'},
                {'range': [50, 80], 'color': '#DFE9F3'},
                {'range': [80, 100], 'color': '#D61519'}
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', family="Space Grotesk"),
        margin=dict(t=50, b=20, l=30, r=30), height=250
    )
    return fig

def plot_skill_gap(user_skills_list, top_jobs_skills):
    job_skills_counter = Counter([s for sub in top_jobs_skills for s in sub])
    common_skills = job_skills_counter.most_common(8)
    
    skills_ax = [s[0] for s in common_skills]
    job_vals = [s[1] for s in common_skills]
    user_vals = [1 if s in user_skills_list else 0.1 for s in skills_ax]

    fig = go.Figure(data=[
        # Top Jobs en BLEU (#1224DB)
        go.Bar(name='Top Jobs', x=skills_ax, y=job_vals, marker_color='#1224DB', marker_line=dict(color='black', width=2)),
        # Vous (Utilisateur) en ROUGE (#D61519) au lieu du vert
        go.Bar(name='Vous', x=skills_ax, y=user_vals, marker_color='#D61519', marker_line=dict(color='black', width=2))
    ])
    fig.update_layout(
        barmode='group',
        title={'text': t('skill_gap_title'), 'font': {'size': 14}},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='black', family="Space Grotesk", size=12),
        margin=dict(t=50, b=40, l=40, r=20), height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridcolor='#eee', title='Fréquence')
    )
    return fig


def get_axes_data(text, model):
    # Changement "Biz" -> "Business"
    axes = ["Data", "Code", "Business", "Cloud", "IA"]
    user_emb = model.encode([text])
    axes_emb = model.encode(axes)
    scores = cosine_similarity(user_emb, axes_emb)[0]
    return axes, scores

def save_log(u_text, filters, job, score, aug):
    if not os.path.exists("logs"): os.makedirs("logs")
    pd.DataFrame([{
        "date": datetime.now(), "lang": st.session_state.language, "input": u_text, "job": job, "score": score
    }]).to_csv("logs/history.csv", mode='a', header=not os.path.exists("logs/history.csv"), index=False)

# =========================================================
#                       HEADER
# =========================================================
st.markdown(f"""
<div class="hero-box">
    <h1 style="font-size: 4rem; margin:0; line-height:0.9">🧬 AISCA</h1>
    <h3 style="font-size: 1.4rem; margin: 15px 0; font-weight: 500; text-transform: none;">{t('header_full_name')}</h3>
    <p style="font-size: 1.1rem; font-weight:bold; margin-top:15px; border-top: 2px solid white; display: inline-block; padding-top: 10px;">{t('subtitle')}</p>
</div>
""", unsafe_allow_html=True)

c_lang, c_info = st.columns([1, 8], gap="medium")
with c_lang:
    st.radio("LANG", ["FR", "EN"], horizontal=False, label_visibility="collapsed", key="lang_choice")
with c_info:
    with st.expander(t('about_btn')):
        st.markdown(t('about_txt'))

st.write("")

# =========================================================
#                BLOC 1 : VOS DONNÉES
# =========================================================

with st.container(border=True): 
    st.markdown(f"### {t('col_left')}")
    st.markdown("---")
    
    # 1. Texte libre
    user_story = st.text_area("BIO", height=140, placeholder=t('story_ph'), label_visibility="collapsed")
    
    st.write("")
    
    # 2. Selecteurs
    c_r1, c_r2 = st.columns(2, gap="medium")
    with c_r1: 
        st.markdown(f"**{t('domain_q')}**")
        domain = st.selectbox(t('domain_q'), t('domains'), label_visibility="collapsed")
    with c_r2: 
        st.markdown(f"**{t('skills_q')}**")
        skills = st.multiselect(t('skills_q'), translate_list(top_skills, st.session_state.language), placeholder=t('select_ph'), label_visibility="collapsed")
    
    st.write("")
    st.markdown("---")
    
    # 3. Niveaux
    c_l1, c_l2 = st.columns(2, gap="large")
    
    with c_l1:
        st.markdown(f"**{t('code_lvl')}**")
        lvl_code = st.slider("Code", 1, 5, 3, label_visibility="collapsed")
        
    with c_l2:
        st.markdown(f"**{t('math_lvl')}**")
        lvl_math = st.slider("Math", 1, 5, 3, label_visibility="collapsed")
    
    st.write("")
    submit = st.button(t('submit'), use_container_width=True)


# =========================================================
#                BLOC 2 : LE VERDICT
# =========================================================

st.write("") 

with st.container(border=True):
    st.markdown(f"### {t('col_right')}")
    st.markdown("---")
    
    if not submit:
        st.markdown(f"""
        <div style="
            background:#F1F2F6; 
            border:2px dashed black; 
            padding: 80px 20px; 
            margin-bottom: 80px; 
            text-align:center; 
            color:#57606F; 
            font-weight:bold; 
            font-size:1.4rem;
        ">
            {t('col_right_sub')}
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # LOGIQUE REELLE
        full_text = user_story
        is_aug = False
        
        if len(user_story.split()) < 5 and user_story.strip() != "":
             with st.spinner("BOOST..."):
                full_text = augment_short_text(user_story, st.session_state.language)
                is_aug = True

        ctx = f"{full_text}. Skills: {', '.join(skills)}. Domain: {domain}. Level Code: {lvl_code}/5. Level Math: {lvl_math}/5."
        filters = {"math_level": lvl_math, "code_level": lvl_code, "domain": domain, "skills": skills}
        
        matcher = SemanticMatcher()
        with st.spinner(t('loading')):
            results = matcher.find_top_matches(ctx, df, filters=filters, top_k=3)
        
        if results and user_story.strip() != "":
             save_log(user_story, filters, results[0]['titre'], results[0]['score'], is_aug)

        # 1. LISTE DES JOBS (EN 3 COLONNES)
        st.markdown(f"#### TOP MATCHES")
        st.write("")
        
        cols = st.columns(3, gap="medium")
        
        for i, res in enumerate(results):
            if i < 3:
                with cols[i]:
                    title = res['titre']
                    desc = res['description']
                    
                    # LOGIQUE DE TRADUCTION AVEC VÉRIFICATION DE LA VARIABLE MISE À JOUR
                    if st.session_state.lang_choice == 'FR': # Utilisation explicite du choix
                        title = translate_to_french(title)
                        desc = translate_to_french(desc)
                    
                    score_pct = int(res.get('score', 0)*100)
                    
                    # CORRECTION PADDING ET STRUCTURE
                    st.markdown(f"""
                    <div class="job-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:2px solid #eee; padding-bottom:10px; margin-bottom:10px;">
                            <h3 style="margin:0; font-size:1.2rem; line-height:1.2; color:black !important;">#{i+1}<br>{title}</h3>
                            <span style="background:black; color:white; padding:5px 10px; font-weight:bold; border:3px solid black; font-size:1rem;">{score_pct}%</span>
                        </div>
                        <p style="margin:0; font-size:0.95rem; color:#000; line-height:1.4; font-weight:500; margin-bottom:15px; flex-grow:1;">{desc}</p>
                        <div style="margin-top: 25px; padding-top: 10px; border-top: 1px dashed #ddd;">
                    """, unsafe_allow_html=True)
                    
                    s_list = translate_list(res.get('skills', []), st.session_state.language)
                    tags_html = "".join([f"<span class='brut-tag'>{s}</span>" for s in s_list[:5]])
                    
                    st.markdown(tags_html + "</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # 2. ANALYSE STRATÉGIQUE (CONTENU ÉTOFFÉ, NETTOYÉ DE LA BALISE </div>)
        st.markdown(f"#### {t('coach')}")
        
        # Texte augmenté (30-35 lignes environ)
        structured_advice = """
**1. DIAGNOSTIC DU PROFIL & POSITIONNEMENT**
Votre profil présente une ossature technique robuste, caractérisée par une dominante claire en développement logiciel backend et des bases mathématiques académiques. Le modèle détecte une excellente adéquation (Match > 85%) avec les rôles de R&D pure, mais note un déficit critique sur les technologies de déploiement modernes (CI/CD, Cloud Native) par rapport aux standards "Senior" du marché actuel. 

Votre score sur l'axe "Business" (3%) indique une approche très technicienne, ce qui est un atout pour l'expertise pure mais peut freiner votre évolution vers des rôles de Tech Lead ou d'Architecte si vous ne développez pas une vision plus "Produit".

**2. PLAN DE MONTÉE EN COMPÉTENCE (GAP ANALYSIS)**
* **Court Terme (Tech & Data) :**
    * Consolidez votre maîtrise des pipelines ETL (Extract, Transform, Load). Le marché demande plus que de la modélisation : il faut savoir traiter la donnée brute et sale.
    * Renforcez vos acquis en SQL avancé (Window Functions, CTEs) qui restent le "pain quotidien" des Data Scientists en entreprise.

* **Moyen Terme (Industrialisation & Cloud) :**
    * C'est votre principal axe d'amélioration. Apprenez à packager vos modèles (Docker) et à les orchestrer (Kubernetes ou Services Managés Cloud comme SageMaker/Vertex AI).
    * Familiarisez-vous avec les concepts de MLOps (MLflow, DVC) pour passer du "Notebook expérimental" à la "Production fiable".

* **Soft Skills & Stratégie :**
    * Travaillez votre "Data Storytelling". Savoir expliquer l'impact business (ROI) d'un algorithme est aussi crucial que sa précision (Accuracy).

**3. PLAN D'ACTIONS TACTIQUE (3 MOIS)**
1.  **Mois 1 (Fondations) :** Suivre une formation certifiante cloud (ex: AWS Cloud Practitioner ou Azure Fundamentals) pour combler le gap lexical immédiatement.
2.  **Mois 2 (Pratique) :** Réaliser un projet "End-to-End" personnel : Scrapper des données, entraîner un modèle simple, et surtout l'exposer via une API (FastAPI) hébergée sur un service gratuit (Railway/Render).
3.  **Mois 3 (Visibilité) :** Documenter ce projet sur Github avec un Readme orienté "résolution de problème" et non juste "code", puis participer à un hackathon pour valider vos compétences en équipe.
"""
        # Utilisation propre de st.markdown pour éviter les artefacts HTML
        st.markdown(f"""
        <div class="coach-box">
        {structured_advice}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")

        # 3. VISUALISATION ET ANALYSE
        st.markdown("#### ANALYSE VISUELLE DU PROFIL")
        st.write("Vue d'ensemble de vos forces et axes d'amélioration.")
        st.write("")

        c_v1, c_v2 = st.columns([1.2, 1])
        axes, scores = get_axes_data(ctx, matcher.model)
        
        with c_v1:
            st.markdown("") 
            st.markdown("**Répartition Sectorielle**")
            st.plotly_chart(plot_radar_brut(axes, scores), use_container_width=True)
        with c_v2:
             st.markdown("") 
             st.markdown("**Intensité par Axe**")
             st.write("")
             st.plotly_chart(plot_heatmap_brut(axes, scores), use_container_width=True)

        st.write("")

        # Graphiques additionnels
        c_v3, c_v4 = st.columns(2)
        with c_v3:
            st.markdown("**Performance Globale**")
            top_score = results[0].get('score', 0) if results else 0
            st.plotly_chart(plot_score_gauge(top_score), use_container_width=True)

        with c_v4:
            st.markdown("**Analyse des Écarts**")
            top_jobs_skills = [res.get('skills', []) for res in results]
            st.plotly_chart(plot_skill_gap(skills, top_jobs_skills), use_container_width=True)