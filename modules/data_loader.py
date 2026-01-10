import json
import os
import pandas as pd

def load_data(filepath="data/referentiel_competences.json"):
    """
    Charge le fichier JSON et le convertit en DataFrame Pandas pour faciliter la manipulation.
    """
    # Vérification que le fichier existe (au cas où)
    if not os.path.exists(filepath):
        # On retourne un DataFrame vide pour éviter que l'app plante
        return pd.DataFrame()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Conversion en DataFrame Pandas (plus facile à manipuler que des listes)
    df = pd.DataFrame(data)
    return df