import json

# 1. Les Templates ROME (inchangés)
ROME_TEMPLATES = {
    "M1805": {"base": "Développement de solutions informatiques et modèles mathématiques.", "niv": "Bac+5"},
    "M1802": {"base": "Expertise infrastructure, flux de données et systèmes.", "niv": "Bac+5"},
    "M1403": {"base": "Extraction d'information décisionnelle et reporting.", "niv": "Bac+3/5"},
    "M1801": {"base": "Administration, sécurité et intégrité des bases de données.", "niv": "Bac+2/5"},
    "M1806": {"base": "Conseil en stratégie digitale et gouvernance des données.", "niv": "Bac+5"}
}

# 2. LISTE A : Métiers Génériques (Avec compétences ajoutées en 4ème position)
# Structure : (Code, Titre, Description, [Liste de compétences])
GENERIC_ROLES = [
    ("M1805", "Data Scientist", "Création de modèles prédictifs et Machine Learning.", ["Python", "Scikit-Learn", "Statistiques", "MLOps", "Pandas"]),
    ("M1802", "Data Engineer", "Création de pipelines ETL et gestion de flux de données.", ["Python", "SQL", "Spark", "Airflow", "Docker", "Cloud"]),
    ("M1403", "Data Analyst", "Analyse de données et création de dashboards décisionnels.", ["SQL", "Tableau", "PowerBI", "Excel", "Data Viz"]),
    ("M1805", "Machine Learning Engineer", "Industrialisation des modèles IA et MLOps.", ["Python", "Docker", "Kubernetes", "API Rest", "CI/CD"]),
    ("M1403", "Consultant BI", "Business Intelligence et aide à la décision pour les métiers.", ["Modélisation de données", "SQL", "PowerBI", "DAX", "Gestion de projet"]),
    ("M1805", "Développeur Python Data", "Développement Backend orienté traitement de données.", ["Python", "FastAPI", "SQLAlchemy", "Git", "Tests unitaires"]),
    ("M1802", "Cloud Data Engineer", "Ingénierie des données sur le Cloud.", ["AWS", "Azure", "GCP", "Terraform", "IAM"]),
]

# 3. LISTE B : Métiers Spécialistes (Avec compétences ajoutées en 4ème position)
SPECIALIST_ROLES = [
    ("M1805", "NLP Engineer", "Traitement du langage naturel, LLM, Chatbots et Text Mining.", ["NLP", "Transformers", "Spacy", "HuggingFace", "Python"]),
    ("M1805", "Computer Vision Engineer", "Traitement d'images, reconnaissance visuelle et Deep Learning.", ["OpenCV", "Deep Learning", "YOLO", "PyTorch", "Image Processing"]),
    ("M1805", "Generative AI Specialist", "Expertise sur les modèles génératifs (GPT, Llama), Prompt Engineering.", ["LLM", "Prompt Engineering", "LangChain", "RAG", "Python"]),
    ("M1805", "AI Research Scientist", "Recherche fondamentale en IA, lecture de papiers.", ["Mathématiques", "PyTorch", "TensorFlow", "Recherche", "Anglais"]),
    ("M1802", "Big Data Architect", "Architecture de systèmes distribués haute performance.", ["Hadoop", "Kafka", "Spark", "Architecture", "NoSQL"]),
    ("M1802", "Analytics Engineer", "Transformation des données modernes avec dbt et SQL avancé.", ["dbt", "SQL", "Data Warehousing", "Snowflake", "BigQuery"]),
    ("M1403", "Marketing Data Analyst", "Analyse ROI des campagnes, segmentation client et CRM.", ["Google Analytics", "SQL", "CRM", "A/B Testing", "Excel"]),
    ("M1403", "Financial Data Analyst", "Modélisation financière, risques et prévisions budgétaires.", ["Excel", "VBA", "SQL", "Finance", "Reporting"]),
    ("M1403", "Product Data Analyst", "Analyse de l'usage produit, A/B testing et rétention.", ["Amplitude", "Mixpanel", "SQL", "Product Management", "KPIs"]),
    ("M1403", "Data Storyteller", "Communication des chiffres clés et vulgarisation.", ["Communication", "PowerPoint", "Data Viz", "Synthèse", "Public Speaking"]),
    ("M1806", "Chief Data Officer (CDO)", "Directeur de la stratégie Data, gouvernance et conformité.", ["Stratégie", "Management", "Gouvernance", "Budget", "Leadership"]),
    ("M1801", "Data Protection Officer (DPO)", "Responsable de la conformité RGPD et protection des données.", ["RGPD", "Droit", "Conformité", "Audit", "Sécurité"]),
    ("M1801", "Data Steward", "Garant de la qualité et de la documentation des données.", ["Data Quality", "Documentation", "Master Data", "SQL", "Métier"]),
    ("M1801", "Database Administrator (DBA)", "Administration et optimisation des bases SQL.", ["Oracle", "PostgreSQL", "Administration BDD", "Backup", "Performance"]),
    ("M1801", "NoSQL Expert", "Expertise bases non-relationnelles.", ["MongoDB", "Cassandra", "ElasticSearch", "Redis", "JSON"]),
    ("M1805", "RAG Developer", "Développeur spécialisé en Retrieval-Augmented Generation.", ["Vector DB", "Pinecone", "LangChain", "Embeddings", "Python"]),
    ("M1806", "AI Ethicist", "Audit des biais algorithmiques et éthique de l'IA.", ["Éthique", "Biais", "Régulation", "Audit IA", "Philosophie"]),
    ("M1802", "Data Reliability Engineer", "Observabilité des pipelines et qualité en production.", ["Observabilité", "Datadog", "SRE", "Python", "Monitoring"])
]

def generate_data():
    full_data = []
    
    # BOUCLE 1 : Génération des variantes Junior / Confirmé / Senior
    levels = [
        ("Junior", "Débutant, focus sur l'exécution technique et l'apprentissage.", "Bac+3"),
        ("Confirmé", "Autonome, capable de gérer des projets complets.", "Bac+5"),
        ("Senior", "Expert, encadre les juniors et définit les choix techniques complexes.", "Expert")
    ]
    
    # Notez l'ajout de la variable 'skills' dans la boucle
    for code, titre, desc, skills in GENERIC_ROLES:
        for lvl_name, lvl_desc, lvl_dip in levels:
            full_titre = f"{titre} {lvl_name}"
            full_desc = f"{titre} niveau {lvl_name}. {desc} {lvl_desc} {ROME_TEMPLATES[code]['base']}"
            
            full_data.append({
                "id_metier": f"{code}_{full_titre.replace(' ', '_').upper()}",
                "code_rome": code,
                "titre": full_titre,
                "description_semantique": full_desc,
                "competences_cles": skills, # Utilisation de la vraie liste
                "niveau_attendu": lvl_dip
            })

    # BOUCLE 2 : Ajout des spécialistes
    for code, titre, desc, skills in SPECIALIST_ROLES:
        full_desc = f"{titre}. {desc} {ROME_TEMPLATES[code]['base']}"
        full_data.append({
            "id_metier": f"{code}_{titre.replace(' ', '_').upper()}",
            "code_rome": code,
            "titre": titre,
            "description_semantique": full_desc,
            "competences_cles": skills, # Utilisation de la vraie liste
            "niveau_attendu": "Bac+5"
        })

    # Sauvegarde
    output_path = "data/referentiel_competences.json"
    print(f"Génération de {len(full_data)} métiers Data & IA avec compétences détaillées...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Fichier généré avec succès dans {output_path}")

if __name__ == "__main__":
    generate_data()