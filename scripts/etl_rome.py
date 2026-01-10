import json
import os

# --- CONFIGURATION ---
# Chemin vers vos fichiers bruts (sur votre Mac/Docker)
RAW_PATH = "data/raw_rome/unix_fiche_emploi_metier_v460.json"
OUTPUT_PATH = "data/referentiel_competences.json"

# Codes ROME cibles (Data & IA)
# M1805 : Études et développement informatique (contient souvent Data Scientist)
# M1403 : Études et prospective socio-économique (Data Analyst)
# M1801 : Administration de systèmes d'information (DBA)
# M1802 : Expertise et support technique en systèmes d'information (Data Engineer parfois)
TARGET_CODES = ["M1805", "M1403", "M1801", "M1802", "M1806"]

def run_etl():
    print(f"chargement de {RAW_PATH}...")
    
    # 1. Chargement du fichier brut
    if not os.path.exists(RAW_PATH):
        print(f"ERREUR : Le fichier {RAW_PATH} est introuvable.")
        return

    with open(RAW_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 2. Transformation
    print("Transformation des données...")
    clean_data = []
    
    # NOTE: La structure du JSON ROME est généralement une liste d'objets.
    # Il faut adapter les clés ci-dessous si le format a changé dans la v460.
    # On cherche généralement : 'code_rome', 'libelle_appellation_court', 'definition_emploi'
    
    for item in raw_data:
        # On sécurise l'accès aux clés (adapter selon la structure réelle si besoin)
        code_rome = item.get('code_rome')   
        
        if code_rome in TARGET_CODES:
            # Création de la description sémantique pour SBERT
            # On combine le titre et la définition pour donner du contexte
            titre = item.get('libelle_appellation_court', 'Métier Inconnu')
            definition = item.get('definition_emploi', '')
            
            description_complete = f"{titre}. {definition}"
            
            # Objet final propre
            job_entry = {
                "id_metier": str(item.get('id_fiche_emploi', 'N/A')),
                "code_rome": code_rome,
                "titre": titre,
                "description_semantique": description_complete,
                # Pour les compétences, le ROME les stocke dans un autre fichier (unix_referentiel_competence).
                # Pour le MVP, on met une liste générique ou vide pour l'instant.
                "competences_cles": ["Voir fiche détaillée ROME"], 
                "niveau_attendu": "Bac+3/5"
            }
            clean_data.append(job_entry)

    # 3. Sauvegarde
    print(f"Sauvegarde de {len(clean_data)} métiers trouvés dans {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)
    
    print("✅ ETL terminé avec succès !")

if __name__ == "__main__":
    run_etl()