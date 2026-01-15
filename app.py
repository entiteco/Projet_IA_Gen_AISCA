import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import ast
from datetime import datetime
from collections import Counter
import tempfile
from fpdf import FPDF

# ============================================================
# CUSTOM MODULE IMPORTS (Backend Logic)
# ============================================================
from modules.data_loader import load_data
from modules.nlp_engine import SemanticMatcher
from modules.genai_engine import augment_short_text, translate_to_french, generate_career_advice

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AISCA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Session State Management ---
# Manages the navigation between the Landing Page and the Main App
if 'page_state' not in st.session_state:
    st.session_state.page_state = 'landing'

# Manages Language Preferences (FR/EN)
if 'lang_choice' not in st.session_state:
    st.session_state.lang_choice = 'FR' 

st.session_state.language = 'fr' if st.session_state.lang_choice == 'FR' else 'en'

# ============================================================
# 2. DESIGN SYSTEM: "LIQUID BRUTALISM" (CSS)
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;500;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px; 
    }

    /* Animated Liquid Background */
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

    .block-container {
        padding-top: 4rem !important; 
        padding-bottom: 5rem;
        max-width: 1400px; 
    }

    /* Neo-Brutalist Containers */
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

    h1, h2, h3, h4 {
        color: #000000 !important;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: -0.5px;
    }
    
    /* Hero Section Styling */
    .hero-box {
        background-color: #1224DB !important;
        border: 3px solid black;
        box-shadow: 6px 6px 0px black;
        padding: 30px;
        margin-bottom: 50px;
        text-align: center;
        transition: all 0.2s ease-in-out;
    }
    .hero-box h1, .hero-box h3, .hero-box p { color: #FFFFFF !important; }

    /* Buttons */
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

    /* Form Inputs */
    .stTextArea textarea, 
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        border-radius: 0px;
        font-size: 1rem;
        min-height: 50px;
    }

    /* Sliders */
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

    /* Skill Badges */
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

# =========================================================
# 3. ROUTING LOGIC
# =========================================================

if st.session_state.page_state == 'landing':
    # --- LANDING PAGE ---
    
    st.markdown("<div style='height: 5vh;'></div>", unsafe_allow_html=True)
    
    col_l, col_c, col_r = st.columns([1, 2, 1])
    
    with col_c:
        # Logo Display
        if os.path.exists("LOGO_AISCA.png"):
            st.image("LOGO_AISCA.png", use_container_width=True)
        else:
            st.markdown("<h1 style='font-size:6rem; margin:0; text-align:center;'>🧬 AISCA</h1>", unsafe_allow_html=True)

        # Subtitles (Styled P tags to avoid anchor links)
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 1.3rem; color:#000; margin-top:25px; margin-bottom:15px; font-weight:800; text-transform:uppercase; line-height: 1.2;">
                AGENT INTELLIGENT SÉMANTIQUE & GÉNÉRATIF
            </p>
            <p style="font-size: 1.1rem; margin-bottom: 40px; font-weight:500;">
                Cartographie des compétences . Analyse RAG . Coaching IA
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Start Button
        if st.button("INITIALISER LE SYSTÈME", use_container_width=True):
            st.session_state.page_state = 'app'
            st.rerun()

else:
    # --- MAIN APPLICATION ---

    # --- DATA CONFIGURATION ---
    # Dictionary for translating skills if necessary
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

    # UI Translations
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
            * **Augmentation:** Enrichment of short user inputs.
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

    # Helper function for translations
    def t(key): return TRANS[st.session_state.language][key]
    def translate_list(l, lang): return l if lang == 'en' else [SKILL_MAP.get(s.title().strip(), s) for s in l]

    # --- DATA LOADING ---
    df = load_data()
    if df.empty: st.stop()
    
    # Process skills for the MultiSelect input
    all_skills = [s.title() for sub in df['skills'] for s in sub if len(s)>1]
    top_skills = [x[0] for x in Counter(all_skills).most_common(120)]
    top_skills.sort()

    # --- VISUALIZATION FUNCTIONS (PLOTLY) ---
    def plot_radar_brut(axes, scores):
        """Generates a Radar Chart for competency distribution."""
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
        """Generates a linear Heatmap for competency intensity."""
        fig = go.Figure(data=go.Heatmap(
            z=[scores], x=axes, y=['SCORE'],
            colorscale=[[0, 'white'], [1, '#D61519']],
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
        """Generates a Gauge Chart for the overall match score."""
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
        """Generates a Bar Chart comparing User Skills vs Market Demand."""
        job_skills_counter = Counter([s for sub in top_jobs_skills for s in sub])
        common_skills = job_skills_counter.most_common(8)
        
        skills_ax = [s[0] for s in common_skills]
        job_vals = [s[1] for s in common_skills]
        user_vals = [1 if s in user_skills_list else 0.1 for s in skills_ax]

        fig = go.Figure(data=[
            go.Bar(name='Top Jobs', x=skills_ax, y=job_vals, marker_color='#1224DB', marker_line=dict(color='black', width=2)),
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
        """Computes embedding scores for the 5 main axes."""
        axes = ["Data", "Code", "Business", "Cloud", "IA"]
        user_emb = model.encode([text])
        axes_emb = model.encode(axes)
        scores = cosine_similarity(user_emb, axes_emb)[0]
        return axes, scores

    # --- CACHING & LOGGING LOGIC ---
    CSV_FILE = "logs/history.csv"

    def check_cache(u_text, filters):
        """Checks if the exact query has already been processed to save API costs."""
        if not os.path.exists(CSV_FILE):
            return False, None, None

        try:
            hist_df = pd.read_csv(CSV_FILE)
            if hist_df.empty: return False, None, None

            current_skills = sorted(filters.get('skills', []))
            current_domain = filters.get('domain', 'N/A')
            current_lvl_c = filters.get('code_level', 0)
            current_lvl_m = filters.get('math_level', 0)
            current_text = u_text.strip()

            for _, row in hist_df.iterrows():
                cached_skills_str = str(row['tech_skills']) if pd.notna(row['tech_skills']) else ""
                cached_skills = sorted([s.strip() for s in cached_skills_str.split(',') if s.strip()])
                
                if (str(row['user_profile_raw']).strip() == current_text and
                    str(row['domain_pref']) == current_domain and
                    cached_skills == current_skills and
                    row['level_code'] == current_lvl_c and
                    row['level_math'] == current_lvl_m):
                    
                    cached_results = []
                    try:
                        if 'full_results_json' in row and pd.notna(row['full_results_json']):
                            cached_results = ast.literal_eval(row['full_results_json'])
                    except:
                        pass
                    
                    cached_advice = row['coach_advice'] if 'coach_advice' in row else None
                    
                    if cached_results and cached_advice:
                        return True, cached_results, cached_advice
                        
        except Exception as e:
            return False, None, None

        return False, None, None

    def save_log(u_text, filters, results, advice, aug):
        """Saves the query results to CSV for history and caching."""
        top_job = results[0]['titre'] if results else "N/A"
        top_score = results[0]['score'] if results else 0
        
        results_str = str(results)

        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_profile_raw": u_text.strip(),
            "is_augmented": aug,
            "domain_pref": filters.get('domain', 'N/A'),
            "tech_skills": ", ".join(sorted(filters.get('skills', []))),
            "level_code": filters.get('code_level', 0),
            "level_math": filters.get('math_level', 0),
            
            "top_match_job": top_job,
            "match_score": top_score,
            "coach_advice": advice,          
            "full_results_json": results_str, 
            
            "language": st.session_state.language
        }
        
        df_new = pd.DataFrame([log_data])
        
        if not os.path.exists(CSV_FILE):
            df_new.to_csv(CSV_FILE, index=False)
        else:
            df_new.to_csv(CSV_FILE, mode='a', header=False, index=False)

    # =========================================================
    #            PDF GENERATION UTILITY
    # =========================================================
    def create_pdf(user_data, results, advice, figures):
        """
        Generates a professional PDF report containing:
        1. User Profile Data
        2. Top 3 Job Matches
        3. AI Strategic Coaching
        4. Visualization Snapshots (Radar, Heatmap, etc.)
        """
        class PDF(FPDF):
            def header(self):
                # Add Logo if available
                if os.path.exists("LOGO_AISCA.png"):
                    self.image("LOGO_AISCA.png", 10, 8, 33)
                self.set_font('Arial', 'B', 15)
                self.cell(80)
                self.cell(30, 10, 'AISCA - Rapport de Carrière', 0, 0, 'C')
                self.ln(20)

            def footer(self):
                self.set_y(-15)
                self.set_font('Arial', 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Généré par AISCA le {datetime.now().strftime("%d/%m/%Y")}', 0, 0, 'C')

        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Helper to handle encoding (Latin-1) for accents
        def s(text):
            return str(text).encode('latin-1', 'replace').decode('latin-1')

        # 1. PROFILE SECTION
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(18, 36, 219) # AISCA Blue
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, s("1. PROFIL ANALYSÉ"), 0, 1, 'L', 1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.ln(2)
        
        pdf.multi_cell(0, 6, s(f"Secteur : {user_data.get('domain','')}"))
        pdf.multi_cell(0, 6, s(f"Compétences : {', '.join(user_data.get('skills',[]))}"))
        pdf.multi_cell(0, 6, s(f"Niveaux : Code {user_data.get('code_level')}/5 - Maths {user_data.get('math_level')}/5"))
        pdf.ln(5)

        # 2. TOP MATCHES SECTION
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(18, 36, 219)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, s("2. TOP 3 RECOMMANDATIONS"), 0, 1, 'L', 1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

        for i, res in enumerate(results[:3]):
            title = res['titre']
            score = int(res['score'] * 100)
            desc = res['description'][:300] + "..." # Truncate description
            
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 8, s(f"#{i+1} : {title} ({score}%)"), 0, 1)
            pdf.set_font('Arial', 'I', 9)
            pdf.multi_cell(0, 5, s(desc))
            pdf.ln(3)
        
        pdf.ln(5)

        # 3. AI COACHING SECTION
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(214, 21, 25) # AISCA Red
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, s("3. ANALYSE STRATÉGIQUE (IA)"), 0, 1, 'L', 1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 10)
        # Basic Markdown cleanup
        clean_advice = advice.replace('**', '').replace('###', '').replace('* ', '- ')
        pdf.multi_cell(0, 6, s(clean_advice))
        
        # 4. VISUALIZATION SECTION
        pdf.add_page()
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(0, 0, 0)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, s("4. VISUALISATION"), 0, 1, 'L', 1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        
        try:
            # Loop through charts, convert Plotly objects to temporary PNG images
            for name, fig in figures.items():
                if fig:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                        fig.write_image(tmpfile.name, width=500, height=300)
                        
                        pdf.set_font('Arial', 'B', 10)
                        pdf.cell(0, 10, s(name), 0, 1)
                        pdf.image(tmpfile.name, w=150)
                        pdf.ln(5)
                        
                        # Cleanup temp files
                        tmpfile.close()
                        os.unlink(tmpfile.name) 
        except Exception as e:
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 10, s(f"(Graphiques non disponibles : {str(e)})"), 0, 1)

        return pdf.output(dest='S').encode('latin-1')

    # =========================================================
    #                       HEADER UI
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
    #                BLOCK 1: USER INPUTS
    # =========================================================

    with st.container(border=True): 
        st.markdown(f"### {t('col_left')}")
        st.markdown("---")
        
        # Narrative Bio
        user_story = st.text_area("BIO", height=140, placeholder=t('story_ph'), label_visibility="collapsed")
        
        st.write("")
        
        # Dropdowns & MultiSelects
        c_r1, c_r2 = st.columns(2, gap="medium")
        with c_r1: 
            st.markdown(f"**{t('domain_q')}**")
            domain = st.selectbox(t('domain_q'), t('domains'), label_visibility="collapsed")
        with c_r2: 
            st.markdown(f"**{t('skills_q')}**")
            skills = st.multiselect(t('skills_q'), translate_list(top_skills, st.session_state.language), placeholder=t('select_ph'), label_visibility="collapsed")
        
        st.write("")
        st.markdown("---")
        
        # Sliders for Proficiency
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
    #                BLOCK 2: RESULTS & ANALYSIS
    # =========================================================

    st.write("") 

    with st.container(border=True):
        st.markdown(f"### {t('col_right')}")
        st.markdown("---")
        
        if not submit:
            # Placeholder State
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
            # --- MAIN LOGIC EXECUTION ---
            filters = {"math_level": lvl_math, "code_level": lvl_code, "domain": domain, "skills": skills}
            
            # 1. Check Local Cache (Performance Optimization)
            is_cached, cached_results, cached_advice = check_cache(user_story, filters)
            
            if is_cached:
                st.success("⚡ Résultat récupéré du cache (History) !")
                results = cached_results
                advice = cached_advice
                is_aug = False 
                full_text = user_story 
                ctx = f"{user_story}. Skills: {', '.join(skills)}. Domain: {domain}. Level Code: {lvl_code}/5."
                matcher = SemanticMatcher() 
                
            else:
                # 2. Run Analysis Pipeline (API Calls + Local NLP)
                full_text = user_story
                is_aug = False
                
                # Step 2a: Augment text if input is too short
                if len(user_story.split()) < 5 and user_story.strip() != "":
                    with st.spinner("BOOST..."):
                        full_text = augment_short_text(user_story, st.session_state.language)
                        is_aug = True
        
                ctx = f"{full_text}. Skills: {', '.join(skills)}. Domain: {domain}. Level Code: {lvl_code}/5. Level Math: {lvl_math}/5."
                
                # Step 2b: Semantic Matching
                matcher = SemanticMatcher()
                with st.spinner(t('loading')):
                    results = matcher.find_top_matches(ctx, df, filters=filters, top_k=3)
                
                # Step 2c: Generative AI Coaching
                with st.spinner(t('ia_loading')):
                    advice = generate_career_advice(ctx, results[0]['titre'], results[0]['description'], filters, st.session_state.language)
                
                # Step 3: Save to Logs
                if results and user_story.strip() != "":
                    save_log(user_story, filters, results, advice, is_aug)

            # --- DISPLAY RESULTS ---
            
            # 1. Job Cards
            st.markdown(f"#### TOP MATCHES")
            st.write("")
            
            cols = st.columns(3, gap="medium")
            
            for i, res in enumerate(results):
                if i < 3:
                    with cols[i]:
                        title = res['titre']
                        desc = res['description']
                        
                        if st.session_state.lang_choice == 'FR':
                            title = translate_to_french(title)
                            desc = translate_to_french(desc)
                        
                        score_pct = int(res.get('score', 0)*100)
                        
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

            # 2. AI Coach Block
            st.markdown(f"#### {t('coach')}")
            
            st.markdown(f"""
            <div class="coach-box">
            {advice}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")

            # 3. Data Visualization
            st.markdown("#### ANALYSE VISUELLE DU PROFIL")
            st.write("Vue d'ensemble de vos forces et axes d'amélioration.")
            st.write("")

            c_v1, c_v2 = st.columns([1.2, 1])
            axes, scores = get_axes_data(ctx, matcher.model)
            
            # Create chart objects
            fig_radar = plot_radar_brut(axes, scores)
            fig_heat = plot_heatmap_brut(axes, scores)
            
            with c_v1:
                st.markdown("") 
                st.markdown("**Répartition Sectorielle**")
                st.plotly_chart(fig_radar, use_container_width=True)
            with c_v2:
                st.markdown("") 
                st.markdown("**Intensité par Axe**")
                st.write("")
                st.plotly_chart(fig_heat, use_container_width=True)

            st.write("")

            c_v3, c_v4 = st.columns(2)
            
            top_score = results[0].get('score', 0) if results else 0
            fig_gauge = plot_score_gauge(top_score)
            
            top_jobs_skills = [res.get('skills', []) for res in results]
            fig_gap = plot_skill_gap(skills, top_jobs_skills)
            
            with c_v3:
                st.markdown("**Performance Globale**")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with c_v4:
                st.markdown("**Analyse des Écarts**")
                st.plotly_chart(fig_gap, use_container_width=True)

            # --- PDF DOWNLOAD SECTION ---
            st.markdown("---")
            st.write("")
            
            # Prepare data package for PDF
            user_data_pdf = {
                'domain': domain, 'skills': skills,
                'code_level': lvl_code, 'math_level': lvl_math
            }
            figures_pdf = {
                "Répartition Radar": fig_radar,
                "Heatmap Compétences": fig_heat,
                "Jauge de Score": fig_gauge,
                "Gap Analysis": fig_gap
            }

            # Generate PDF bytes in memory
            pdf_bytes = create_pdf(user_data_pdf, results, advice, figures_pdf)

            # Render Download Button
            st.download_button(
                label="📥 TÉLÉCHARGER LE RAPPORT COMPLET (PDF)",
                data=pdf_bytes,
                file_name=f"Rapport_AISCA_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )