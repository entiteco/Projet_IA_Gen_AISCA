import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Imports des modules personnalisés
from modules.data_loader import load_data
from modules.nlp_engine import SemanticMatcher
from modules.genai_engine import generate_career_advice, augment_short_text

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
    div[data-testid="stExpander"] div[role="button"] p { font-weight: bold; }
    h3 { color: #2E86C1; }
    </style>
""", unsafe_allow_html=True)

# --- FONCTIONS UTILITAIRES (DÉFINIES AVANT L'USAGE) ---

def save_to_history(user_text, filters, top_result, augmented=False):
    """
    Sauvegarde la requête utilisateur et le résultat dans un fichier CSV (Logs).
    Gère la création du dossier si inexistant.
    """
    log_dir = "logs"
    log_file = os.path.join(log_dir, "history.csv")
    
    # Création du dossier s'il n'existe pas
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Préparation de la ligne de données
    new_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_profile_raw": user_text,
        "is_augmented": augmented,
        "domain_pref": filters['domain'],
        "tech_skills": ", ".join(filters.get('skills', [])),
        "level_code": filters['code_level'],
        "level_math": filters['math_level'],
        "recommended_job": top_result['titre'],
        "match_score": round(top_result['score'], 4)
    }
    
    # Conversion en DataFrame
    df = pd.DataFrame([new_data])
    
    # Mode 'append' (ajout) : on écrit l'entête seulement si le fichier est nouveau
    if not os.path.exists(log_file):
        df.to_csv(log_file, index=False, sep=';', encoding='utf-8')
    else:
        df.to_csv(log_file, mode='a', header=False, index=False, sep=';', encoding='utf-8')

def get_axes_data(user_text, model):
    """Calcule les scores de similarité sur 5 axes."""
    axes = ["Développement & Code", "Mathématiques & Stats", "Business & Communication", "Cloud & Infrastructure", "IA & Modeling"]
    axes_embeddings = model.encode(axes)
    user_embedding = model.encode([user_text])
    scores = cosine_similarity(user_embedding, axes_embeddings)[0]
    return axes, scores

def plot_radar_chart(axes, scores):
    """Génère le graphique Radar."""
    r_values = list(scores) + [scores[0]]
    theta_values = axes + [axes[0]]
    fig = go.Figure(data=go.Scatterpolar(
        r=r_values, theta=theta_values, fill='toself', name='Votre Profil', line_color='#2E86C1'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, title="Radar des Compétences",
        margin=dict(t=30, b=20, l=40, r=40), height=350
    )
    return fig

def plot_heatmap(axes, scores):
    """Génère une Heatmap."""
    z = [scores]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=axes, y=["Intensité"], colorscale='Blues', zmin=0, zmax=1, showscale=False
    ))
    fig.update_layout(
        title="Carte de Chaleur Sémantique", height=200,
        margin=dict(t=30, b=10, l=10, r=10), yaxis=dict(showticklabels=False)
    )
    return fig

def plot_bar_chart(results):
    """Génère un Bar Chart horizontal."""
    titles = [res['titre'] for res in results][::-1]
    scores = [res['score'] for res in results][::-1]
    colors = ['#2ECC71' if s > 0.6 else '#F1C40F' for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=titles, orientation='h', marker_color=colors,
        text=[f"{s*100:.0f}%" for s in scores], textposition='auto'
    ))
    fig.update_layout(
        title="Comparatif de Pertinence (Top 3)", xaxis_title="Score (0-1)",
        margin=dict(t=30, b=20, l=10, r=10), height=250
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.title("🤖 AISCA")
    st.info(
        """
        **Bienvenue sur l'agent d'orientation.**
        
        Ce système utilise une approche hybride :
        1. **Sémantique (SBERT)** pour comprendre vos écrits.
        2. **Logique Métier** pour valider les pré-requis.
        3. **GenAI (Gemini)** pour l'analyse et l'augmentation.
        """
    )
    st.write("---")
    st.caption("Projet IA Gen - EFREI")
    st.caption("Valentin MASSONNIERE - Kévin HEUGAS")    

# --- TITRE PRINCIPAL ---
c_logo, c_title = st.columns([1, 10])
with c_logo:
    st.write(" ") 
    st.write("🎯") 
with c_title:
    st.title("Cartographie des Compétences & Orientation")

st.markdown("##### Trouvez le métier Data/IA aligné avec votre profil réel.")

# Chargement des données
df = load_data()
if df.empty:
    st.error("⚠️ Erreur Critique : Le fichier 'referentiel_competences.json' est introuvable.")
    st.stop()

# --- FORMULAIRE ---
with st.form("profiling_form"):
    
    # BLOC 1 : PRÉFÉRENCES
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
                [
                    "Python", "SQL", "R", "Java", "Scala", "C++", "Julia", "Bash/Shell", "SAS",
                    "Spark", "Hadoop", "Kafka", "Airflow", "dbt", "Databricks", "Snowflake", "BigQuery", "Redshift", "Talend",
                    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "Hugging Face", "XGBoost", "LightGBM", "NLTK/Spacy",
                    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Git", "GitHub Actions", "GitLab CI",
                    "MLflow", "Kubeflow", "Weights & Biases", "DVC",
                    "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Neo4j", "Cassandra",
                    "Excel", "PowerBI", "Tableau", "Looker", "Qlik", "Streamlit", "Plotly", "Matplotlib/Seaborn"
                ],
                placeholder="Tapez ou sélectionnez vos compétences..."
            )

    st.write("") 

    # BLOC 2 : AUTO-ÉVALUATION
    with st.container(border=True):
        st.subheader("2️⃣ Auto-évaluation")
        c3, c4 = st.columns([1, 1], gap="large")
        with c3:
            st.markdown("**Niveau en Programmation (Python/SQL)**")
            niveau_python = st.slider("Note sur 5", 1, 5, 2, key="slider_code")
            if niveau_python <= 2: st.caption("🟢 Débutant")
            elif niveau_python <= 4: st.caption("🟡 Confirmé")
            else: st.caption("🔴 Expert")
        with c4:
            st.markdown("**Aisance Mathématiques & Stats**")
            niveau_maths = st.select_slider(
                "Niveau perçu",
                options=["Débutant", "Notions", "Intermédiaire", "Avancé", "Expert"],
                value="Intermédiaire"
            )

    st.write("") 

    # BLOC 3 : HISTOIRE
    with st.container(border=True):
        st.subheader("3️⃣ Votre Histoire")
        st.markdown("L'IA analyse le **sens** de vos phrases.")
        user_text_input = st.text_area(
            "Décrivez vos expériences, vos projets ou vos ambitions :", 
            height=150,
            placeholder="Exemple : J'ai travaillé sur un projet de détection de fraude bancaire..."
        )

    st.write("")
    b1, b2, b3 = st.columns([1, 2, 1])
    with b2:
        submitted = st.form_submit_button("🚀 Lancer l'Analyse Complète", use_container_width=True)

# --- LOGIQUE DE TRAITEMENT ---
if submitted:
    
    full_text_history = user_text_input
    is_augmented = False
    
    # 1. Gestion texte court (Pre-processing IA)
    if len(user_text_input.split()) < 5:
        with st.spinner("✨ Texte court détecté : L'IA enrichit votre profil..."):
            augmented_text = augment_short_text(user_text_input)
            
            with st.expander("Voir l'enrichissement sémantique (IA)", expanded=True):
                st.write(f"**Avant :** {user_text_input}")
                st.write(f"**Après (Augmenté) :** {augmented_text}")
            
            full_text_history = augmented_text
            is_augmented = True

    # 2. Construction du contexte complet pour SBERT
    contexte_skills = f"Je maîtrise les outils : {', '.join(skills_tech)}." if skills_tech else ""
    contexte_pref = f"Domaine favori : {domaine_prefere}." if domaine_prefere != "Peu importe" else ""
    contexte_niveau = f"Niveau Code {niveau_python}/5. Niveau Maths {niveau_maths}."
    
    full_profile_text = f"{full_text_history} {contexte_skills} {contexte_pref} {contexte_niveau}"
    
    # 3. Préparation Filtres
    user_filters = {
        "math_level": niveau_maths, 
        "code_level": niveau_python, 
        "domain": domaine_prefere,
        "skills": skills_tech # Ajouté pour les logs
    }
    
    # 4. Appel Moteur Sémantique
    matcher = SemanticMatcher()
    with st.spinner('🧠 Analyse Hybride en cours (SBERT + Filtres)...'):
        results = matcher.find_top_matches(full_profile_text, df, filters=user_filters, top_k=3)
    
    # 5. Sauvegarde Data Engineering (Logs)
    if results:
        save_to_history(
            user_text=user_text_input, # On sauvegarde le texte original
            filters=user_filters,
            top_result=results[0],
            augmented=is_augmented
        )

    # --- AFFICHAGE : 1. SECTION GENAI (COACH) ---
    st.divider()
    st.markdown("### 🤖 L'Avis de l'Assistant IA (Gemini)")
    
    if results:
        top_job = results[0]
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("⏳ *Analyse de votre profil et rédaction du plan de progression...*")
            
            advice = generate_career_advice(
                user_profile_text=full_profile_text,
                job_title=top_job['titre'],
                job_desc=top_job['description'],
                filters=user_filters
            )
            message_placeholder.markdown(advice)
    
    # --- AFFICHAGE : 2. SECTION VISUALISATION ---
    st.divider()
    st.markdown("### 📊 Cartographie Visuelle")
    
    axes_labels, axes_scores = get_axes_data(full_profile_text, matcher.model)
    col_viz_left, col_viz_right = st.columns([1, 1], gap="medium")
    
    with col_viz_left:
        fig_radar = plot_radar_chart(axes_labels, axes_scores)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_viz_right:
        fig_heat = plot_heatmap(axes_labels, axes_scores)
        st.plotly_chart(fig_heat, use_container_width=True)
        
        fig_bar = plot_bar_chart(results)
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- AFFICHAGE : 3. DÉTAILS MÉTIERS ---
    st.divider()
    st.markdown("### 🏆 Top 3 des Métiers Recommandés")
    
    cols = st.columns(3, gap="medium")
    
    for i, res in enumerate(results):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"#### #{i+1} {res['titre']}")
                
                score_val = res['score']
                if score_val > 0.6: color = "green"
                elif score_val > 0.4: color = "orange"
                else: color = "red"
                
                st.markdown(f":{color}[**Pertinence : {score_val*100:.1f}%**]")
                st.progress(min(max(score_val, 0.0), 1.0))
                
                if res['reasons']:
                    st.markdown("---")
                    for reason in res['reasons']:
                        st.markdown(f":red[**{reason}**]") 
                    st.markdown("---")
                
                st.caption(f"_{res['description']}_") 
                
                with st.expander("Voir compétences requises"):
                    st.write(", ".join(res['competences']))