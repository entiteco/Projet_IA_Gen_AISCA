import streamlit as st
from modules.data_loader import load_data
from modules.nlp_engine import SemanticMatcher
from modules.genai_engine import generate_career_advice

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="AISCA - Orientation IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .stSlider [data-baseweb="slider"] { padding-top: 10px; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 AISCA")
    st.info(
        """
        **Bienvenue sur l'agent d'orientation.**
        1. Remplissez vos préférences.
        2. Évaluez votre niveau.
        3. Racontez votre parcours.
        """
    )
    st.write("---")
    st.caption("Projet IA Gen - EFREI")
    st.caption("Valentin MASSONNIERE - Kévin HEUGAS")    

# --- TITRE PRINCIPAL ---
st.title("Cartographie des Compétences & Orientation")
st.markdown("##### 🎯 Trouvez le métier Data/IA qui matche réellement avec votre profil.")
st.write("") 

# Chargement des données
df = load_data()
if df.empty:
    st.error("⚠️ Erreur : Le fichier de données 'referentiel_competences.json' est introuvable.")
    st.stop()

# --- FORMULAIRE PRINCIPAL ---
with st.form("profiling_form"):
    
    # SECTION 1 : PRÉFÉRENCES
    with st.container(border=True):
        st.subheader("1️⃣ Vos Préférences & Outils")
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            domaine_prefere = st.selectbox(
                "Quel domaine vous attire le plus ?",
                ["Peu importe", "Développement & Code", "Analyse & Business", "Infrastructure & Cloud", "Mathématiques & Recherche", "Éthique & Gouvernance"]
            )
        with c2:
            skills_tech = st.multiselect(
                "Quels outils maîtrisez-vous techniquement ?",
                ["Python", "SQL", "Excel", "PowerBI", "Tableau", "Docker", "AWS", "Azure", "GCP", "Java", "R", "Git"]
            )

    st.write("") 

    # SECTION 2 : AUTO-ÉVALUATION
    with st.container(border=True):
        st.subheader("2️⃣ Auto-évaluation")
        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            st.markdown("**Niveau en Programmation (Python/SQL)**")
            niveau_python = st.slider("Note sur 5", 1, 5, 2, key="slider_code")
        with c4:
            st.markdown("**Aisance Mathématiques & Stats**")
            niveau_maths = st.select_slider(
                "Niveau perçu",
                options=["Débutant", "Notions", "Intermédiaire", "Avancé", "Expert"],
                value="Intermédiaire"
            )

    st.write("") 

    # SECTION 3 : HISTOIRE
    with st.container(border=True):
        st.subheader("3️⃣ Votre Histoire (Le plus important)")
        user_text_input = st.text_area(
            "Décrivez vos expériences, vos projets ou vos ambitions :", 
            height=200,
            placeholder="Exemple : J'ai travaillé sur un projet de détection de fraude..."
        )

    st.write("")
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        submitted = st.form_submit_button("🚀 Lancer l'Analyse Sémantique", use_container_width=True)

# --- TRAITEMENT ET RÉSULTATS ---
if submitted:
    # 1. Préparation texte
    contexte_skills = f"Je maîtrise les outils techniques suivants : {', '.join(skills_tech)}." if skills_tech else ""
    contexte_pref = f"Mon domaine de prédilection est : {domaine_prefere}." if domaine_prefere != "Peu importe" else ""
    contexte_niveau = f"J'ai un niveau technique {niveau_python}/5 et un niveau en mathématiques '{niveau_maths}'."
    
    full_profile_text = f"{user_text_input} {contexte_skills} {contexte_pref} {contexte_niveau}"
    
    user_filters = {
        "math_level": niveau_maths,
        "code_level": niveau_python,
        "domain": domaine_prefere
    }
    
    # 2. Appel au moteur SBERT
    matcher = SemanticMatcher()
    with st.spinner('🧠 Analyse Hybride (Sémantique + Règles Métier)...'):
        results = matcher.find_top_matches(full_profile_text, df, filters=user_filters, top_k=3)
    
    # ---------------------------------------------------------
    # PARTIE 1 : L'IA PARLE (GEMINI) - AFFICHAGE EN PREMIER
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### 🤖 L'Analyse de l'Assistant IA")

    top_job = results[0]
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ *Rédaction de la synthèse personnalisée en cours...*")
        
        # Appel API Gemini
        advice = generate_career_advice(
            user_profile_text=full_profile_text,
            job_title=top_job['titre'],
            job_desc=top_job['description'],
            filters=user_filters
        )
        
        message_placeholder.markdown(advice)

    # ---------------------------------------------------------
    # PARTIE 2 : LES CARTES DÉTAILLÉES - AFFICHAGE EN SECOND
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🏆 Top 3 des recommandations")
    
    cols = st.columns(3, gap="medium")
    
    for i, res in enumerate(results):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"#### #{i+1} {res['titre']}")
                
                # Jauge
                score_val = res['score']
                if score_val > 0.6: color = "green"
                elif score_val > 0.4: color = "orange"
                else: color = "red"
                
                st.markdown(f":{color}[**Pertinence : {score_val*100:.1f}%**]")
                st.progress(min(max(score_val, 0.0), 1.0))
                
                # Bonus / Malus
                if res['reasons']:
                    st.markdown("---")
                    for reason in res['reasons']:
                        st.markdown(f":red[**{reason}**]") 
                    st.markdown("---")
                
                # Description et Compétences
                st.markdown(f"_{res['description']}_") 
                
                with st.expander("Voir compétences clés"):
                    st.write(", ".join(res['competences']))